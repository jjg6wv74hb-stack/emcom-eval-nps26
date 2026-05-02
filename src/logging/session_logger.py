from __future__ import annotations

import os
from glob import glob
from typing import Dict

import numpy as np


class SessionLogger:
    def __init__(self, save_dir: str, condition_name: str, seed: int):
        self.save_dir = save_dir
        self.condition_name = condition_name
        self.seed = int(seed)
        self.session_count = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def _path(self, session_id: int) -> str:
        return os.path.join(
            self.save_dir,
            f"data_{self.condition_name}_{self.seed}_{session_id}.npz",
        )

    @staticmethod
    def _regime_codes(true_f: np.ndarray, n_agents: int) -> np.ndarray:
        """
        Encode game regime per round:
        0 -> competitive (f <= 1)
        1 -> mixed-motive (1 < f <= N)
        2 -> cooperative (f > N)
        """
        arr = np.asarray(true_f, dtype=np.float32)
        out = np.zeros(arr.shape, dtype=np.int8)
        out[arr > 1.0] = 1
        out[arr > float(n_agents)] = 2
        return out

    def log_session(self, buffer) -> str:
        t = buffer.t
        out_path = self._path(self.session_count)
        regime_t = self._regime_codes(buffer.true_f[:t], n_agents=buffer.n_agents)

        payload = dict(
            true_f=buffer.true_f[:t],
            f_t=buffer.true_f[:t],
            regime_t=regime_t,
            f_hats=buffer.f_hats[:t],
            intended_actions=buffer.actions[:t],
            executed_actions=buffer.executed_actions[:t],
            flips=buffer.flips[:t],
            rewards=buffer.agent_rewards[:t],
            cooperation_count=buffer.executed_actions[:t].sum(axis=1),
            welfare=buffer.agent_rewards[:t].sum(axis=1),
        )

        if getattr(buffer, "messages", None) is not None:
            payload["messages"] = buffer.messages[:t]
        else:
            payload["messages"] = np.array([], dtype=np.int32)

        np.savez(out_path, **payload)
        self.session_count += 1
        return out_path

    def consolidate(self, delete_parts: bool = False) -> str:
        pattern = os.path.join(
            self.save_dir, f"data_{self.condition_name}_{self.seed}_*.npz"
        )
        files = sorted(glob(pattern))
        if len(files) == 0:
            raise FileNotFoundError(f"no per-session files found for pattern: {pattern}")

        stacked: Dict[str, list] = {}
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                for key in data.files:
                    stacked.setdefault(key, []).append(data[key])

        out = {}
        for key, values in stacked.items():
            out[key] = np.stack(values, axis=0)

        consolidated_path = os.path.join(
            self.save_dir, f"data_{self.condition_name}_{self.seed}_consolidated.npz"
        )
        np.savez(consolidated_path, **out)

        if delete_parts:
            for path in files:
                os.remove(path)

        return consolidated_path
