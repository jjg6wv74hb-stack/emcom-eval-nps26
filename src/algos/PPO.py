
from src.algos.agent import Agent
from src.nets.ActorCritic import ActorCritic
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

#https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

class PPO(Agent):

    def __init__(self, params, idx=0):
        Agent.__init__(self, params, idx)

        opt_params = []

        self.policy_act = ActorCritic(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act, gmm=self.gmm_).to(device)
        self.policy_act_old = copy.deepcopy(self.policy_act).to(device)
        self.policy_act_old.load_state_dict(self.policy_act.state_dict())
        opt_params.append({'params': self.policy_act.actor.parameters(), 'lr': self.lr_actor})
        opt_params.append({'params': self.policy_act.critic.parameters(), 'lr': self.lr_critic})

        if (self.is_communicating):
            self.policy_comm = ActorCritic(params=params, input_size=self.input_comm, output_size=self.mex_size, \
                    n_hidden=self.n_hidden_comm, hidden_size=self.hidden_size_comm, gmm=self.gmm_).to(device)
            self.policy_comm_old = copy.deepcopy(self.policy_comm).to(device)
            self.policy_comm_old.load_state_dict(self.policy_comm.state_dict())
            opt_params.append({'params': self.policy_comm.actor.parameters(), 'lr': self.lr_actor_comm})
            opt_params.append({'params': self.policy_comm.critic.parameters(), 'lr': self.lr_critic_comm})

        self.optimizer = torch.optim.Adam(opt_params)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)
        self.MseLoss = torch.nn.MSELoss()

    def eval_mex(self, state_c):

        messages_logprobs = []
        with torch.no_grad():
            state_c = torch.FloatTensor(state_c).to(device)
            for mex in range(self.mex_size):
                m = torch.Tensor([mex])
                _, mex_logprob, _= self.policy_comm_old.evaluate(state_c, m)
                messages_logprobs.append(mex_logprob.item())

        return messages_logprobs

    def eval_action(self, state_a, message):

        actions_logprobs = []
        with torch.no_grad():
            state_a = torch.FloatTensor(state_a).to(device)
            for action in range(self.action_size):
                a = torch.Tensor([action])
                _, action_logprob, _= self.policy_act_old.evaluate(state_a, message, a)
                actions_logprobs.append(action_logprob.item())

        return actions_logprobs

    def update(self):
        rewards = []
        discounted_reward = 0
        #for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        if (self.policy_act.input_size == 1):
            old_states_a = torch.stack(self.buffer.states_a, dim=0).detach().to(device)
            old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
            old_logprobs_act = torch.stack(self.buffer.act_logprobs, dim=0).detach().to(device)
        else:
            old_states_a = torch.squeeze(torch.stack(self.buffer.states_a, dim=0)).detach().to(device)
            old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
            old_logprobs_act = torch.squeeze(torch.stack(self.buffer.act_logprobs, dim=0)).detach().to(device)

        if (self.is_communicating):
            if (self.policy_comm.input_size == 1):
                old_states_c = torch.stack(self.buffer.states_c, dim=0).detach().to(device)
                old_messages = torch.stack(self.buffer.messages, dim=0).detach().to(device)
                old_logprobs_comm = torch.stack(self.buffer.comm_logprobs, dim=0).detach().to(device)
            else:
                old_states_c = torch.squeeze(torch.stack(self.buffer.states_c, dim=0)).detach().to(device)
                old_messages = torch.squeeze(torch.stack(self.buffer.messages, dim=0)).detach().to(device)
                old_logprobs_comm = torch.squeeze(torch.stack(self.buffer.comm_logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):

            if (self.is_communicating):
                logprobs_comm, dist_entropy_comm, state_values_comm = self.policy_comm.evaluate(old_states_c, old_messages)
                state_values_comm = torch.squeeze(state_values_comm)
                ratios_comm = torch.exp(logprobs_comm - old_logprobs_comm.detach())
                mutinfo = torch.tensor(self.buffer.mut_info, dtype=torch.float32).to(device)
                advantages_comm = rewards - state_values_comm.detach()
                surr1c = ratios_comm*advantages_comm
                surr2c = torch.clamp(ratios_comm, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages_comm

                loss_c = (-torch.min(surr1c, surr2c) + \
                    self.c3*self.MseLoss(state_values_comm, rewards) - \
                    self.c4*dist_entropy_comm)

            #print(old_states_a[0], old_actions[0])
            logprobs_act, dist_entropy_act, state_values_act = self.policy_act.evaluate(old_states_a, old_actions)
            state_values_act = torch.squeeze(state_values_act)
            ratios_act = torch.exp(logprobs_act - old_logprobs_act.detach())
            advantages_act = rewards - state_values_act.detach()
            surr1a = ratios_act*advantages_act
            surr2a = torch.clamp(ratios_act, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages_act

            loss_a = (-torch.min(surr1a, surr2a) + \
                self.c1*self.MseLoss(state_values_act, rewards) - \
                self.c2*dist_entropy_act)
            #print("loss_a=", loss_a)

            if (self.is_communicating):
                loss_a += loss_c
                #self.saved_losses_comm.append(torch.mean(torch.Tensor([i.detach() for i in self.comm_logprobs])))
                #tmp = [torch.ones(a.data.shape) for a in self.comm_logprobs]
                #autograd.backward(self.comm_logprobs, tmp, retain_graph=True)
            self.saved_losses.append(torch.mean(loss_a.detach()))

            self.optimizer.zero_grad()
            loss_a.mean().backward()
            self.optimizer.step()
            #print(self.scheduler.get_lr())

        self.scheduler.step()
        #print(self.scheduler.get_lr())

        # Copy new weights into old policy
        self.policy_act_old.load_state_dict(self.policy_act.state_dict())
        if (self.is_communicating):
            self.policy_comm_old.load_state_dict(self.policy_comm.state_dict())

        # clear buffer
        self.buffer.clear()
        self.reset()


class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class PPOAgentV2:
    """
    Standalone PPO agent used by train_ppo.py.
    Keeps action and optional message policies separate while sharing one value head.
    """

    def __init__(
        self,
        obs_dim: int,
        action_size: int = 2,
        can_send: bool = False,
        vocab_size: int = 2,
        hidden_size: int = 64,
        value_obs_dim: int = None,
        lr: float = 3e-4,
    ):
        self.obs_dim = int(obs_dim)
        self.value_obs_dim = int(value_obs_dim) if value_obs_dim is not None else int(obs_dim)
        self.action_size = int(action_size)
        self.can_send = bool(can_send)
        self.vocab_size = int(vocab_size)

        self.action_actor = _MLP(self.obs_dim, self.action_size, hidden_size=hidden_size).to(device)
        self.value_net = _MLP(self.value_obs_dim, 1, hidden_size=hidden_size).to(device)
        if self.can_send:
            self.message_actor = _MLP(self.obs_dim, self.vocab_size, hidden_size=hidden_size).to(device)
        else:
            self.message_actor = None

        params = list(self.action_actor.parameters()) + list(self.value_net.parameters())
        if self.message_actor is not None:
            params += list(self.message_actor.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _obs_tensor(self, obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            out = obs.to(device=device, dtype=torch.float32)
        else:
            out = torch.tensor(obs, dtype=torch.float32, device=device)
        return out

    def sample_action(self, obs, value_obs=None):
        obs_t = self._obs_tensor(obs)
        logits = self.action_actor(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        if value_obs is None:
            value_t = obs_t
        else:
            value_t = self._obs_tensor(value_obs)
        value = self.value_net(value_t).squeeze(-1)
        probs = torch.softmax(logits, dim=-1)
        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(value.item()),
            float(dist.entropy().item()),
            probs.detach(),
        )

    def sample_action_batch(self, obs_batch, value_obs_batch=None):
        obs_t = self._obs_tensor(obs_batch)
        logits = self.action_actor(obs_t)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        if value_obs_batch is None:
            value_t = obs_t
        else:
            value_t = self._obs_tensor(value_obs_batch)
        values = self.value_net(value_t).squeeze(-1)
        probs = torch.softmax(logits, dim=-1)
        return (
            actions.detach().cpu().numpy().astype(np.int32),
            dist.log_prob(actions).detach().cpu().numpy().astype(np.float32),
            values.detach().cpu().numpy().astype(np.float32),
            dist.entropy().detach().cpu().numpy().astype(np.float32),
            probs.detach(),
        )

    def sample_message(self, obs):
        if not self.can_send or self.message_actor is None:
            return None, None, None, None
        obs_t = self._obs_tensor(obs)
        logits = self.message_actor(obs_t)
        dist = Categorical(logits=logits)
        msg = dist.sample()
        probs = torch.softmax(logits, dim=-1)
        return (
            int(msg.item()),
            float(dist.log_prob(msg).item()),
            float(dist.entropy().item()),
            probs.detach(),
        )

    def sample_message_batch(self, obs_batch):
        if not self.can_send or self.message_actor is None:
            return None, None, None, None
        obs_t = self._obs_tensor(obs_batch)
        logits = self.message_actor(obs_t)
        dist = Categorical(logits=logits)
        msgs = dist.sample()
        probs = torch.softmax(logits, dim=-1)
        return (
            msgs.detach().cpu().numpy().astype(np.int32),
            dist.log_prob(msgs).detach().cpu().numpy().astype(np.float32),
            dist.entropy().detach().cpu().numpy().astype(np.float32),
            probs.detach(),
        )

    def value(self, obs_batch: torch.Tensor) -> torch.Tensor:
        return self.value_net(obs_batch).squeeze(-1)

    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        value_obs_batch: torch.Tensor = None,
    ):
        logits = self.action_actor(obs_batch)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_batch)
        entropy = dist.entropy()
        values = self.value(value_obs_batch if value_obs_batch is not None else obs_batch)
        return log_probs, entropy, values

    def evaluate_messages(self, obs_batch: torch.Tensor, messages_batch: torch.Tensor):
        if not self.can_send or self.message_actor is None:
            raise RuntimeError("evaluate_messages called for non-sender agent")
        logits = self.message_actor(obs_batch)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(messages_batch)
        entropy = dist.entropy()
        return log_probs, entropy

    def action_distribution(self, obs_batch: torch.Tensor) -> torch.Tensor:
        logits = self.action_actor(obs_batch)
        return torch.softmax(logits, dim=-1)


class PPOTrainer:
    def __init__(
        self,
        agents,
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        msg_entropy_coeff: float = None,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 32,
        sign_lambda: float = 0.0,
        list_lambda: float = 0.0,
        message_entropy_target: float = None,
        normalize_value_loss: bool = True,
    ):
        """
        Entropy coefficient for the message head. If None, defaults to entropy_coeff (backward compatible).
        """
        del lr  # agents own optimizers; kept for API compatibility with plan text.
        self.agents = agents
        self.clip_ratio = float(clip_ratio)
        self.value_coeff = float(value_coeff)
        self.entropy_coeff = float(entropy_coeff)
        self.msg_entropy_coeff = (
            float(msg_entropy_coeff) if msg_entropy_coeff is not None else float(entropy_coeff)
        )
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.mini_batch_size = int(mini_batch_size)
        self.sign_lambda = float(sign_lambda)
        self.list_lambda = float(list_lambda)
        self.message_entropy_target = message_entropy_target
        self.normalize_value_loss = bool(normalize_value_loss)

    def _batched_indices(self, n_items: int):
        order = torch.randperm(n_items, device=device)
        for start in range(0, n_items, self.mini_batch_size):
            yield order[start : start + self.mini_batch_size]

    def update(self, buffer, advantages, returns):
        T = buffer.t
        metrics = {
            "loss_total": 0.0,
            "loss_policy": 0.0,
            "loss_value": 0.0,
            "entropy_action": 0.0,
            "loss_message": 0.0,
            "entropy_message": 0.0,
        }
        n_metric_updates = 0

        if T == 0:
            return metrics

        adv_np = np.asarray(advantages[:T], dtype=np.float32)
        ret_np = np.asarray(returns[:T], dtype=np.float32)

        for agent_id, agent in self.agents.items():
            idx = buffer.agent_to_idx[agent_id]

            obs = torch.tensor(buffer.observations[:T, idx], dtype=torch.float32, device=device)
            value_obs = torch.tensor(
                buffer.value_observations[:T, idx], dtype=torch.float32, device=device
            )
            actions = torch.tensor(buffer.actions[:T, idx], dtype=torch.long, device=device)
            old_log_probs = torch.tensor(buffer.log_probs[:T, idx], dtype=torch.float32, device=device)
            returns_t = torch.tensor(ret_np[:T, idx], dtype=torch.float32, device=device)
            adv_t = torch.tensor(adv_np[:T, idx], dtype=torch.float32, device=device)

            if adv_t.numel() > 1:
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

            list_bonus = torch.tensor(
                buffer.listening_bonus[:T, idx], dtype=torch.float32, device=device
            )

            for _ in range(self.ppo_epochs):
                for batch_idx in self._batched_indices(T):
                    obs_b = obs[batch_idx]
                    value_obs_b = value_obs[batch_idx]
                    actions_b = actions[batch_idx]
                    old_log_probs_b = old_log_probs[batch_idx]
                    returns_b = returns_t[batch_idx]
                    adv_b = adv_t[batch_idx]
                    list_bonus_b = list_bonus[batch_idx]

                    new_log_probs, entropy, values = agent.evaluate_actions(
                        obs_b, actions_b, value_obs_batch=value_obs_b
                    )
                    if self.normalize_value_loss:
                        ret_mean = returns_b.mean()
                        ret_std = returns_b.std(unbiased=False) + 1e-8
                        value_loss = F.mse_loss(
                            (values - ret_mean) / ret_std,
                            (returns_b - ret_mean) / ret_std,
                        )
                    else:
                        value_loss = F.mse_loss(values, returns_b)
                    entropy_bonus = entropy.mean()

                    # Default (action-only) clipped surrogate.
                    ratios = torch.exp(new_log_probs - old_log_probs_b)
                    surr1 = ratios * adv_b
                    surr2 = torch.clamp(
                        ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                    ) * adv_b
                    action_policy_loss = -torch.min(surr1, surr2).mean()
                    policy_loss = action_policy_loss

                    msg_loss_val = torch.tensor(0.0, device=device)
                    msg_entropy_val = torch.tensor(0.0, device=device)
                    msg_sign_loss = torch.tensor(0.0, device=device)
                    if (
                        agent.can_send
                        and buffer.message_actions is not None
                        and buffer.message_log_probs is not None
                    ):
                        msg_actions_all = torch.tensor(
                            buffer.message_actions[:T, idx], dtype=torch.long, device=device
                        )
                        msg_mask_all = msg_actions_all >= 0
                        msg_mask_b = msg_mask_all[batch_idx]
                        if torch.any(msg_mask_b):
                            obs_msg = obs_b[msg_mask_b]
                            msg_actions_b = msg_actions_all[batch_idx][msg_mask_b]
                            old_msg_lp_all = torch.tensor(
                                buffer.message_log_probs[:T, idx],
                                dtype=torch.float32,
                                device=device,
                            )
                            old_msg_lp_b = old_msg_lp_all[batch_idx][msg_mask_b]
                            msg_adv_b = adv_b[msg_mask_b]

                            new_msg_lp, msg_entropy = agent.evaluate_messages(
                                obs_msg, msg_actions_b
                            )

                            # Joint sender decision objective: one clipped ratio over
                            # log pi(a_t, m_t) = log pi(a_t) + log pi(m_t).
                            joint_new_lp = new_log_probs[msg_mask_b] + new_msg_lp
                            joint_old_lp = old_log_probs_b[msg_mask_b] + old_msg_lp_b
                            joint_ratio = torch.exp(joint_new_lp - joint_old_lp)
                            joint_s1 = joint_ratio * msg_adv_b
                            joint_s2 = torch.clamp(
                                joint_ratio,
                                1.0 - self.clip_ratio,
                                1.0 + self.clip_ratio,
                            ) * msg_adv_b
                            joint_policy_loss = -torch.min(joint_s1, joint_s2).mean()
                            msg_entropy_bonus = msg_entropy.mean()

                            if self.sign_lambda != 0.0:
                                if self.message_entropy_target is None:
                                    h_target = np.log(agent.vocab_size) / 2.0
                                else:
                                    h_target = float(self.message_entropy_target)
                                msg_sign_loss = ((h_target - msg_entropy) ** 2).mean()

                            # Use joint objective for sender-updated samples. If any
                            # unmasked items exist, blend with action-only objective.
                            if torch.all(msg_mask_b):
                                policy_loss = joint_policy_loss
                            else:
                                unmasked = ~msg_mask_b
                                if torch.any(unmasked):
                                    action_unmasked = -torch.min(
                                        surr1[unmasked], surr2[unmasked]
                                    ).mean()
                                    n_joint = float(torch.sum(msg_mask_b).item())
                                    n_action = float(torch.sum(unmasked).item())
                                    policy_loss = (
                                        joint_policy_loss * n_joint
                                        + action_unmasked * n_action
                                    ) / float(n_joint + n_action)
                                else:
                                    policy_loss = joint_policy_loss

                            msg_loss_val = joint_policy_loss
                            msg_entropy_val = msg_entropy_bonus

                    total_loss = (
                        policy_loss
                        + self.value_coeff * value_loss
                        - self.entropy_coeff * entropy_bonus
                    )
                    if msg_entropy_val.abs().item() > 0.0:
                        total_loss = total_loss - self.msg_entropy_coeff * msg_entropy_val
                    if self.sign_lambda != 0.0:
                        total_loss = total_loss + self.sign_lambda * msg_sign_loss

                    if self.list_lambda != 0.0:
                        # listening_bonus is negative by construction: larger magnitude means
                        # stronger policy shift due to messages.
                        total_loss = total_loss + self.list_lambda * list_bonus_b.mean()

                    agent.optimizer.zero_grad()
                    total_loss.backward()
                    # Clip actor/message and critic gradients separately so
                    # large value-network gradients do not downscale policy updates.
                    actor_params = list(agent.action_actor.parameters())
                    if agent.message_actor is not None:
                        actor_params += list(agent.message_actor.parameters())
                    critic_params = list(agent.value_net.parameters())
                    nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
                    nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm)
                    agent.optimizer.step()

                    metrics["loss_total"] += float(total_loss.detach().item())
                    metrics["loss_policy"] += float(policy_loss.detach().item())
                    metrics["loss_value"] += float(value_loss.detach().item())
                    metrics["entropy_action"] += float(entropy_bonus.detach().item())
                    metrics["loss_message"] += float(msg_loss_val.detach().item())
                    metrics["entropy_message"] += float(msg_entropy_val.detach().item())
                    n_metric_updates += 1

        if n_metric_updates > 0:
            for key in metrics:
                metrics[key] /= float(n_metric_updates)
        metrics["msg_entropy_coeff"] = float(self.msg_entropy_coeff)
        return metrics
