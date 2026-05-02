import functools
try:
    from gym.spaces import Discrete, Box
except ImportError:  # pragma: no cover - compatibility path
    from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
import random
from torch.distributions import uniform, normal
import torch
import numpy as np

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

# azione 1 e` cooperativa

def env(config):
    env = raw_env(config)
    # This wrapper is only for environments which print results to the terminal
    # env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    if (config.fraction == False):
        env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(config):
    env = parallel_env(config)
    env = parallel_to_aec(env)
    return env

class parallel_env(ParallelEnv):
    metadata = {
        'render.modes': ['human'],
        "name": "pgg_parallel_v0"
        }

    def __init__(self, config):
        '''
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spacesf

        These attributes should not be changed after initialization.

        In config we need:
        - n_agents
        - n_game_iterations
        - mult_factor (list with two numers, bounduaries or the same: [0., 5.] or [1., 1.])
        - uncertainies (list with uncertainty for every agent)
        '''

        for key, val in config.items():
            setattr(self, key, val)

        # Optional config keys for DSC-EPGG extensions.
        self.rho = float(getattr(self, "rho", 0.05))
        self.epsilon_tremble = float(getattr(self, "epsilon_tremble", 0.0))
        self.endowment = float(getattr(self, "endowment", 4.0))
        self.F = list(getattr(self, "F", getattr(self, "mult_fact", [1.0])))
        if len(self.F) == 0:
            self.F = [1.0]
        self.F = [float(f) for f in self.F]

        if hasattr(self.mult_fact, '__len__'):
            self.min_mult = torch.tensor([float(min(self.mult_fact))], dtype=torch.float32, device=device)
            self.max_mult = torch.tensor([float(max(self.mult_fact))], dtype=torch.float32, device=device)
        else:
            self.min_mult = torch.tensor([float(self.mult_fact)], dtype=torch.float32, device=device)
            self.max_mult = torch.tensor([float(self.mult_fact)], dtype=torch.float32, device=device)

        self.possible_agents = ["agent_" + str(r) for r in range(self.n_agents)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.infos = {agent: {} for agent in self.agents}

        if (self.uncertainties is not None):
            assert (self.n_agents == len(self.uncertainties))
            self.uncertainties_dict = {}
            self.min_observable_mult = {}
            self.max_observable_mult = {}
            for idx, agent in enumerate(self.agents):
                self.uncertainties_dict[agent] = float(self.uncertainties[idx])
        else:
            self.uncertainties_dict = {agent: 0.0 for agent in self.agents}
        self.n_actions = 2 # give money (1), keep money (0)
        self.obs_space_size = 2 # [f_hat, endowment]
        self.obs_dim = 2

        self.current_multiplier = torch.tensor([0.0], dtype=torch.float32, device=device)
        self._last_intended_actions = {}
        self._last_executed_actions = {}
        self._last_flips = {}

    # this cache ensures that same space object is returned for the same agent
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Keep observation as continuous vector [f_hat, endowment].
        return Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if (self.fraction == True):
            return Box(low=torch.Tensor([-0.001]), high=torch.Tensor([1.001]))
        else:
            return Discrete(self.n_actions)

    def close(self):
        pass

    def assign_coins_fixed(self):
        self.coins = {}
        self.normalized_coins = {}
        for agent in self.agents:
            self.coins[agent] = float(self.endowment) # what they have
            self.normalized_coins[agent] = 0. # kept for backwards compatibility

    def assign_coins_uniform(self):
        self.coins = {}
        self.normalized_coins = {}
        d = uniform.Uniform(self.coins_min, self.coins_max)
        for agent in self.agents:
            self.coins[agent] = d.sample()
            self.normalized_coins[agent] = (self.coins[agent] - self.coins_min)/(self.coins_max - self.coins_min)

    def observe(self):

        self.observations = {}
        current_f = float(self.current_multiplier.item())
        for agent in self.agents:
            sigma = float(self.uncertainties_dict[agent])
            # No clipping/clamping: keep raw Gaussian observation for exact likelihood modeling.
            obs_multiplier = np.random.normal(current_f, sigma)
            self.observations[agent] = torch.tensor(
                [float(obs_multiplier), float(self.coins[agent])],
                dtype=torch.float32,
                device=device,
            )

        return self.observations

    def _set_current_multiplier(self, value):
        self.current_multiplier = torch.tensor([float(value)], dtype=torch.float32, device=device)

    def _sample_initial_multiplier(self):
        self._set_current_multiplier(random.choice(self.F))

    def _update_multiplier(self):
        """Sticky regime switching: with hazard rho switch to a different multiplier."""
        if len(self.F) <= 1:
            return
        if np.random.random() < self.rho:
            current = float(self.current_multiplier.item())
            other_values = [f for f in self.F if f != current]
            if len(other_values) > 0:
                self._set_current_multiplier(random.choice(other_values))

    def _coerce_binary_action(self, action_value):
        if isinstance(action_value, torch.Tensor):
            if action_value.numel() == 0:
                value = 0.0
            else:
                value = float(action_value.detach().cpu().view(-1)[0].item())
        else:
            value = float(action_value)
        if self.fraction:
            value = float(np.clip(value, 0.0, 1.0))
            return 1 if value >= 0.5 else 0
        return int(round(value))

    def reset(self, mult_in=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.dones = {agent: False for agent in self.agents}

        if (mult_in is not None):
            self._set_current_multiplier(mult_in)
        else:
            self._sample_initial_multiplier()

        self.state = {agent: None for agent in self.agents}

        self.assign_coins_fixed()

        self.num_moves = 0

        return self.observe()

    def get_coins(self):
        return self.coins

    def step(self, actions):
        '''
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        '''
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {}

        # Store intended policy actions and apply trembling-hand flips to get executed actions.
        intended_actions = {}
        executed_actions = {}
        flips = {}
        for agent in self.agents:
            intended = self._coerce_binary_action(actions[agent])
            intended_actions[agent] = intended
            flip = (np.random.random() < self.epsilon_tremble)
            if flip:
                executed = 1 - intended
            else:
                executed = intended
            executed_actions[agent] = executed
            flips[agent] = bool(flip)

        step_multiplier = float(self.current_multiplier.item())
        common_pot = torch.tensor(
            sum(self.coins[agent] * executed_actions[agent] for agent in self.agents),
            dtype=torch.float32,
            device=device,
        )

        for agent in self.agents:
            rewards[agent] = common_pot / self.n_agents * step_multiplier + \
                (self.coins[agent] - self.coins[agent] * executed_actions[agent])

        self.num_moves += 1
        env_done = self.num_moves >= self.num_game_iterations

        # Keep fixed endowments each round; still call for consistency with existing code structure.
        self.assign_coins_fixed()

        # Update latent regime for the NEXT round if episode is still running.
        if not env_done:
            self._update_multiplier()

        # Always generate observations at every step (including done step to keep API consistent).
        observations = self.observe()

        infos = {agent: {
            "intended_action": intended_actions[agent],
            "executed_action": executed_actions[agent],
            "flipped": flips[agent],
            "true_f": step_multiplier,
        } for agent in self.agents}
        infos["intended_actions"] = intended_actions
        infos["executed_actions"] = executed_actions
        infos["flips"] = flips
        infos["true_f"] = step_multiplier

        if env_done:
            self.agents = []

        self._last_intended_actions = intended_actions
        self._last_executed_actions = executed_actions
        self._last_flips = flips

        return observations, rewards, env_done, infos
