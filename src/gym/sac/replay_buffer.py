"""
replay_buffer.py — FIFO ring-buffer replay buffer for SAC.

Pre-allocates contiguous numpy arrays at construction time to avoid
per-step heap allocation overhead during training.

Usage:
    buf = ReplayBuffer(capacity=8_000_000, obs_dim=125, action_dim=3)
    buf.push(obs, action, reward, next_obs, done)
    batch = buf.sample(batch_size=128, device='cuda')
    # batch is a dict of torch tensors on device

Dual-buffer variant:
    dual = DualReplayBuffer(capacity=25_000, obs_dim=125, action_dim=3)
    dual.push(obs, action, reward, next_obs, done)
    # routes to positive_buffer (reward > 0) or negative_buffer (reward <= 0)
    batch = dual.sample(batch_size=256, device='cuda')
    # 50/50 mix from each buffer, shuffled; falls back gracefully when one is sparse
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    FIFO ring-buffer replay buffer.

    Storage is pre-allocated as numpy arrays.  Oldest transitions are
    overwritten when the buffer reaches capacity.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.
    obs_dim : int
        Dimensionality of the observation vector (125 for this project).
    action_dim : int
        Number of action dimensions (3 for steer/throttle/brake).
    """

    def __init__(
        self,
        capacity: int = 8_000_000,
        obs_dim: int = 125,
        action_dim: int = 3,
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Pre-allocate storage
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward = np.zeros((capacity, 1), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._done = np.zeros((capacity, 1), dtype=np.float32)

        self._ptr = 0       # Write pointer (next slot to overwrite)
        self._size = 0      # Current number of valid transitions

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store one transition.

        Silently overwrites the oldest entry when the buffer is full.

        Parameters
        ----------
        obs      : (obs_dim,) float32
        action   : (action_dim,) float32 — expected in [-1, 1]
        reward   : scalar float
        next_obs : (obs_dim,) float32
        done     : bool (or 0/1)
        """
        idx = self._ptr
        self._obs[idx] = obs
        self._action[idx] = action
        self._reward[idx, 0] = float(reward)
        self._next_obs[idx] = next_obs
        self._done[idx, 0] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cuda") -> dict:
        """
        Sample a random mini-batch.

        Parameters
        ----------
        batch_size : int
        device : str
            PyTorch device string ('cuda', 'cpu', etc.)

        Returns
        -------
        dict with keys:
            obs, action, reward, next_obs, done
            Each is a torch.Tensor on the specified device.
        """
        assert self._size >= batch_size, (
            f"Buffer has only {self._size} transitions; "
            f"cannot sample {batch_size}."
        )
        idx = np.random.randint(0, self._size, size=batch_size)

        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr[idx], dtype=torch.float32).to(device)

        return {
            "obs":      _to_tensor(self._obs),
            "action":   _to_tensor(self._action),
            "reward":   _to_tensor(self._reward),
            "next_obs": _to_tensor(self._next_obs),
            "done":     _to_tensor(self._done),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def is_ready(self, min_size: int) -> bool:
        """Return True once the buffer contains at least min_size transitions."""
        return self._size >= min_size

    def clear(self) -> None:
        """
        Reset the buffer to empty, discarding all stored transitions.

        The pre-allocated arrays are reused; no memory is freed or reallocated.
        Call this after each training phase so the next collection phase starts
        with a clean buffer containing only freshly collected frames.
        """
        self._ptr = 0
        self._size = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save buffer contents to a compressed .npz file.

        Parameters
        ----------
        path : str
            Destination file path (typically ``<checkpoint_dir>/buffer.npz``).
            The .npz extension is added by numpy if omitted.
        """
        np.savez_compressed(
            path,
            _obs=self._obs[: self._size],
            _action=self._action[: self._size],
            _reward=self._reward[: self._size],
            _next_obs=self._next_obs[: self._size],
            _done=self._done[: self._size],
            _ptr=np.array(self._ptr, dtype=np.int64),
            _size=np.array(self._size, dtype=np.int64),
        )

    def load(self, path: str) -> None:
        """Load buffer contents from a .npz file produced by :meth:`save`.

        Restores all transition arrays and the write/size pointers.  The file
        must have been saved with the same ``obs_dim`` and ``action_dim`` as
        this buffer instance.

        Parameters
        ----------
        path : str
            Source file path.

        Raises
        ------
        ValueError
            If the stored dimensions do not match this buffer's configuration.
        """
        data = np.load(path)
        size = int(data["_size"])
        if size == 0:
            return

        stored_obs_dim = data["_obs"].shape[1]
        stored_action_dim = data["_action"].shape[1]
        if stored_obs_dim != self.obs_dim:
            raise ValueError(
                f"Buffer obs_dim mismatch: file has {stored_obs_dim}, "
                f"buffer configured for {self.obs_dim}."
            )
        if stored_action_dim != self.action_dim:
            raise ValueError(
                f"Buffer action_dim mismatch: file has {stored_action_dim}, "
                f"buffer configured for {self.action_dim}."
            )

        # Clamp to capacity in case the file was saved from a larger buffer.
        n = min(size, self.capacity)
        self._obs[:n] = data["_obs"][:n]
        self._action[:n] = data["_action"][:n]
        self._reward[:n] = data["_reward"][:n]
        self._next_obs[:n] = data["_next_obs"][:n]
        self._done[:n] = data["_done"][:n]
        self._size = n
        self._ptr = int(data["_ptr"]) % self.capacity


# ---------------------------------------------------------------------------
# DualReplayBuffer
# ---------------------------------------------------------------------------

class DualReplayBuffer:
    """
    Dual replay buffer that routes transitions by reward sign.

    Transitions with reward > 0 go to ``positive_buffer``.
    Transitions with reward <= 0 go to ``negative_buffer``.

    Sampling draws N//2 from each buffer (with graceful fallback when one
    side is under-populated) and shuffles the combined batch before
    returning it.

    Both buffers are independent FIFO ring-buffers (``ReplayBuffer``
    instances) and persist across training phases — they are never cleared.

    Parameters
    ----------
    capacity : int
        Maximum transitions per sub-buffer (positive and negative each).
        Defaults to 25,000.
    obs_dim : int
    action_dim : int
    """

    def __init__(
        self,
        capacity: int = 25_000,
        obs_dim: int = 125,
        action_dim: int = 3,
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.positive_buffer = ReplayBuffer(capacity, obs_dim, action_dim)
        self.negative_buffer = ReplayBuffer(capacity, obs_dim, action_dim)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Route transition to the appropriate sub-buffer based on reward sign."""
        if float(reward) > 0:
            self.positive_buffer.push(obs, action, reward, next_obs, done)
        else:
            self.negative_buffer.push(obs, action, reward, next_obs, done)

    def sample(self, batch_size: int, device: str = "cuda") -> dict:
        """
        Sample a balanced mini-batch.

        Attempts to draw batch_size // 2 transitions from each sub-buffer.
        If one buffer is empty, the full batch is drawn from the other.
        If one buffer has fewer than batch_size // 2 transitions, all
        available transitions from that side are used and the shortfall is
        padded from the other side.  The combined batch is shuffled before
        returning.

        Parameters
        ----------
        batch_size : int
        device : str

        Returns
        -------
        dict with keys: obs, action, reward, next_obs, done
            Each is a torch.Tensor on the specified device.
        """
        pos_size = len(self.positive_buffer)
        neg_size = len(self.negative_buffer)
        total = pos_size + neg_size

        assert total >= batch_size, (
            f"DualReplayBuffer has only {total} transitions "
            f"(pos={pos_size}, neg={neg_size}); cannot sample {batch_size}."
        )

        half = batch_size // 2

        # Determine how many to take from each side.
        if pos_size == 0:
            n_pos, n_neg = 0, batch_size
        elif neg_size == 0:
            n_pos, n_neg = batch_size, 0
        else:
            n_pos = min(half, pos_size)
            n_neg = min(half, neg_size)
            # Pad shortfall from the other side.
            shortfall = batch_size - n_pos - n_neg
            if shortfall > 0:
                if pos_size - n_pos >= shortfall:
                    n_pos += shortfall
                else:
                    n_neg += shortfall

        # Collect numpy index arrays (sample() returns tensors — we need raw
        # arrays to concatenate before converting, so we re-implement the
        # sampling here using the sub-buffer internals directly).
        def _sample_raw(buf: ReplayBuffer, n: int) -> dict:
            """Return dict of numpy arrays sampled from buf."""
            idx = np.random.randint(0, buf._size, size=n)
            return {
                "obs":      buf._obs[idx],
                "action":   buf._action[idx],
                "reward":   buf._reward[idx],
                "next_obs": buf._next_obs[idx],
                "done":     buf._done[idx],
            }

        batches = []
        if n_pos > 0:
            batches.append(_sample_raw(self.positive_buffer, n_pos))
        if n_neg > 0:
            batches.append(_sample_raw(self.negative_buffer, n_neg))

        # Concatenate and shuffle.
        combined = {
            key: np.concatenate([b[key] for b in batches], axis=0)
            for key in ("obs", "action", "reward", "next_obs", "done")
        }
        perm = np.random.permutation(len(combined["obs"]))
        for key in combined:
            combined[key] = combined[key][perm]

        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr, dtype=torch.float32).to(device)

        return {k: _to_tensor(v) for k, v in combined.items()}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of transitions across both sub-buffers."""
        return len(self.positive_buffer) + len(self.negative_buffer)

    def is_ready(self, min_size: int) -> bool:
        """Return True once the combined buffer holds at least min_size transitions."""
        return len(self) >= min_size

    def clear(self) -> None:
        """Clear both sub-buffers (not called during normal training — buffers persist)."""
        self.positive_buffer.clear()
        self.negative_buffer.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save both sub-buffers.

        Saves two files:
            <path>_pos.npz  — positive_buffer contents
            <path>_neg.npz  — negative_buffer contents

        Parameters
        ----------
        path : str
            Base path.  The suffixes ``_pos`` and ``_neg`` are appended
            before the ``.npz`` extension (or after the stem if no
            extension is present).
        """
        stem = path[:-4] if path.endswith(".npz") else path
        self.positive_buffer.save(stem + "_pos.npz")
        self.negative_buffer.save(stem + "_neg.npz")

    def load(self, path: str) -> None:
        """
        Load both sub-buffers from files produced by :meth:`save`.

        Parameters
        ----------
        path : str
            Base path passed to :meth:`save` (same stem convention).
        """
        stem = path[:-4] if path.endswith(".npz") else path
        pos_path = stem + "_pos.npz"
        neg_path = stem + "_neg.npz"
        if _file_exists(pos_path):
            self.positive_buffer.load(pos_path)
        if _file_exists(neg_path):
            self.negative_buffer.load(neg_path)


def _file_exists(path: str) -> bool:
    """Return True if path points to an existing file."""
    import os
    return os.path.isfile(path)
