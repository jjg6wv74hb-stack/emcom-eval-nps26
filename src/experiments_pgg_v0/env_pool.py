from __future__ import annotations

import multiprocessing as mp
import random
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def _make_raw_env(env_cfg: Dict[str, Any]):
    from src.environments import pgg_parallel_v0

    return pgg_parallel_v0.parallel_env(dict(env_cfg))


def _derived_seed(base_seed: int, env_idx: int) -> int:
    return int(base_seed) + 100_003 * int(env_idx)


def _initial_rng_states(seed: int) -> Tuple[object, tuple]:
    py_state = random.Random(int(seed)).getstate()
    np_state = np.random.RandomState(int(seed) % (2**32 - 1)).get_state()
    return py_state, np_state


def _seed_worker(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(int(seed))
    except Exception:
        pass


def _obs_to_numpy(raw_obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for agent_id, value in raw_obs.items():
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        out[str(agent_id)] = np.asarray(arr, dtype=np.float32).reshape(-1).copy()
    return out


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    return value


def _sanitize_step_result(step_result):
    raw_obs, rewards, done, infos = step_result
    clean_rewards = {
        str(agent_id): float(_to_python(value))
        for agent_id, value in dict(rewards).items()
    }
    return _obs_to_numpy(raw_obs), clean_rewards, bool(done), _to_python(infos)


class EnvWorkerError(RuntimeError):
    pass


@dataclass
class _WorkerProcess:
    env_idx: int
    parent_conn: Any
    process: mp.Process


class SerialParallelEnvPool:
    def __init__(self, env_cfg: Dict[str, Any], n_envs: int, base_seed: int) -> None:
        self.env_cfg = dict(env_cfg)
        self.n_envs = int(n_envs)
        self.base_seed = int(base_seed)
        self._envs = [_make_raw_env(self.env_cfg) for _ in range(self.n_envs)]
        self._rng_states = [
            _initial_rng_states(_derived_seed(self.base_seed, env_idx))
            for env_idx in range(self.n_envs)
        ]
        self._closed = False

    def _call_with_env_rng(self, env_idx: int, method_name: str, *args, **kwargs):
        if self._closed:
            raise RuntimeError("SerialParallelEnvPool is closed")
        py_prev = random.getstate()
        np_prev = np.random.get_state()
        py_state, np_state = self._rng_states[env_idx]
        random.setstate(py_state)
        np.random.set_state(np_state)
        try:
            env = self._envs[env_idx]
            result = getattr(env, method_name)(*args, **kwargs)
            self._rng_states[env_idx] = (random.getstate(), np.random.get_state())
            return result
        finally:
            random.setstate(py_prev)
            np.random.set_state(np_prev)

    def reset_all(self) -> List[Dict[str, np.ndarray]]:
        return [
            _obs_to_numpy(self._call_with_env_rng(env_idx, "reset"))
            for env_idx in range(self.n_envs)
        ]

    def step_batch(self, actions_by_env: Sequence[Dict[str, int]]):
        if len(actions_by_env) != self.n_envs:
            raise ValueError(f"expected {self.n_envs} action dicts, got {len(actions_by_env)}")
        results = [
            _sanitize_step_result(
                self._call_with_env_rng(env_idx, "step", dict(actions_by_env[env_idx]))
            )
            for env_idx in range(self.n_envs)
        ]
        return (
            [row[0] for row in results],
            [row[1] for row in results],
            [row[2] for row in results],
            [row[3] for row in results],
        )

    def close(self) -> None:
        if self._closed:
            return
        for env in self._envs:
            try:
                env.close()
            except Exception:
                pass
        self._closed = True


def _subproc_worker(conn, env_cfg: Dict[str, Any], seed: int, env_idx: int) -> None:
    env = None
    try:
        _seed_worker(seed)
        env = _make_raw_env(env_cfg)
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                conn.send({"ok": True, "obs": _obs_to_numpy(env.reset())})
                continue
            if cmd == "step":
                obs, rewards, done, infos = _sanitize_step_result(env.step(dict(payload)))
                conn.send(
                    {
                        "ok": True,
                        "obs": obs,
                        "rewards": rewards,
                        "done": done,
                        "infos": infos,
                    }
                )
                continue
            if cmd == "close":
                break
            raise ValueError(f"unknown worker command: {cmd!r}")
    except EOFError:
        pass
    except Exception as exc:
        try:
            conn.send(
                {
                    "ok": False,
                    "env_idx": int(env_idx),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


class SubprocParallelEnvPool:
    def __init__(
        self,
        env_cfg: Dict[str, Any],
        n_envs: int,
        base_seed: int,
        start_method: str = "spawn",
    ) -> None:
        self.env_cfg = dict(env_cfg)
        self.n_envs = int(n_envs)
        self.base_seed = int(base_seed)
        self.start_method = str(start_method)
        self._ctx = mp.get_context(self.start_method)
        self._workers: List[_WorkerProcess] = []
        self._closed = False

        for env_idx in range(self.n_envs):
            parent_conn, child_conn = self._ctx.Pipe()
            proc = self._ctx.Process(
                target=_subproc_worker,
                args=(
                    child_conn,
                    self.env_cfg,
                    _derived_seed(self.base_seed, env_idx),
                    env_idx,
                ),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._workers.append(
                _WorkerProcess(env_idx=env_idx, parent_conn=parent_conn, process=proc)
            )

    def _recv(self, worker: _WorkerProcess, command: str):
        try:
            payload = worker.parent_conn.recv()
        except EOFError as exc:
            raise EnvWorkerError(
                f"worker {worker.env_idx} exited during {command}"
            ) from exc
        if not payload.get("ok", False):
            raise EnvWorkerError(
                f"worker {worker.env_idx} failed during {command}: "
                f"{payload.get('error_type', 'RuntimeError')}: {payload.get('message', '')}\n"
                f"{payload.get('traceback', '').rstrip()}"
            )
        return payload

    def reset_all(self) -> List[Dict[str, np.ndarray]]:
        if self._closed:
            raise RuntimeError("SubprocParallelEnvPool is closed")
        for worker in self._workers:
            worker.parent_conn.send(("reset", None))
        return [self._recv(worker, "reset")["obs"] for worker in self._workers]

    def step_batch(self, actions_by_env: Sequence[Dict[str, int]]):
        if self._closed:
            raise RuntimeError("SubprocParallelEnvPool is closed")
        if len(actions_by_env) != self.n_envs:
            raise ValueError(f"expected {self.n_envs} action dicts, got {len(actions_by_env)}")
        for worker, actions in zip(self._workers, actions_by_env):
            worker.parent_conn.send(("step", dict(actions)))
        payloads = [self._recv(worker, "step") for worker in self._workers]
        return (
            [payload["obs"] for payload in payloads],
            [payload["rewards"] for payload in payloads],
            [payload["done"] for payload in payloads],
            [payload["infos"] for payload in payloads],
        )

    def close(self) -> None:
        if self._closed:
            return
        for worker in self._workers:
            if not worker.process.is_alive():
                continue
            try:
                worker.parent_conn.send(("close", None))
            except Exception:
                pass
        for worker in self._workers:
            try:
                worker.parent_conn.close()
            except Exception:
                pass
            worker.process.join(timeout=2.0)
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=2.0)
        self._closed = True
