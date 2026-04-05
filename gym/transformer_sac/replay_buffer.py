"""
replay_buffer.py — Window-based dual replay buffer for Transformer SAC.

Key difference from the previous episode-based design:
    OLD: store full episodes; slice windows lazily at sample time.
    NEW: pre-compute all windows at episode end; store them in pre-allocated
         circular numpy arrays.  Sample time is a single random index lookup.

Design:

    WindowReplayBuffer (single side)
    ---------------------------------
    - Pre-allocated circular numpy arrays of capacity 50,000 windows.
    - Each slot: obs_seq (75, 50), next_obs_seq (75, 50), action (3,),
                 reward (1,), done (1,).
    - FIFO circular buffer with _ptr and _size.
    - push(obs_seq, next_obs_seq, action, reward, done) -- one window per call.
    - sample(batch_size, device) -- uniform random draw, returns dict of tensors.
    - save(path) / load(path) -- save/load filled portion as npz.

    DualWindowReplayBuffer (pos + neg sides)
    -----------------------------------------
    - Two WindowReplayBuffer instances, capacity 50,000 each.
    - push_episode(obs_tokens, actions, rewards, dones):
        * Episode must have >= 76 frames (window_size + 1), else discarded.
        * Slide t from window_size-1 (=74) to T-2 (need t+1 for next_obs_seq).
        * For each t:
            obs_seq      = obs_tokens[t-74 : t+1]      shape (75, 50)
            next_obs_seq = obs_tokens[t-73 : t+2]      shape (75, 50)
            action       = actions[t]
            reward       = rewards[t]
            done         = dones[t]
            if reward[t] > 0    -> positive_buffer.push(...)
            else                -> negative_buffer.push(...)
    - sample(batch_size, device):
        * Draw batch_size // 2 from each side.
        * Graceful fallback if one side is sparse.
        * Concatenate + shuffle -> return dict of tensors.
    - __len__  = total windows across both sides.
    - is_ready(min_steps) = len(self) >= min_steps.
    - save(path) / load(path):
        * Save pos as {stem}_pos.npz, neg as {stem}_neg.npz.
    - clear() -- reset both sides.
"""

import os
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class WindowReplayBuffer:
    """
    Pre-allocated circular replay buffer storing pre-computed windows.

    Parameters
    ----------
    capacity : int
        Maximum number of windows to retain (FIFO).
    token_dim : int
        Dimensionality of each observation token (50).
    action_dim : int
        Number of action dimensions (3).
    window_size : int
        Length of each observation sequence (75).
    """

    def __init__(
        self,
        capacity: int = 50_000,
        token_dim: int = 50,
        action_dim: int = 3,
        window_size: int = 75,
    ):
        self.capacity   = capacity
        self.token_dim  = token_dim
        self.action_dim = action_dim
        self.window_size = window_size

        # Pre-allocate arrays.
        self._obs_seq      = np.zeros((capacity, window_size, token_dim), dtype=np.float32)
        self._next_obs_seq = np.zeros((capacity, window_size, token_dim), dtype=np.float32)
        self._action       = np.zeros((capacity, action_dim),             dtype=np.float32)
        self._reward       = np.zeros((capacity, 1),                      dtype=np.float32)
        self._done         = np.zeros((capacity, 1),                      dtype=np.float32)

        self._ptr  = 0   # next write position
        self._size = 0   # number of valid entries

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def push(
        self,
        obs_seq:      np.ndarray,
        next_obs_seq: np.ndarray,
        action:       np.ndarray,
        reward:       float,
        done:         float,
    ) -> None:
        """
        Store one pre-computed window.

        Parameters
        ----------
        obs_seq      : (window_size, token_dim)
        next_obs_seq : (window_size, token_dim)
        action       : (action_dim,)
        reward       : scalar
        done         : scalar (0.0 or 1.0)
        """
        self._obs_seq[self._ptr]      = obs_seq
        self._next_obs_seq[self._ptr] = next_obs_seq
        self._action[self._ptr]       = action
        self._reward[self._ptr, 0]    = reward
        self._done[self._ptr, 0]      = done

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int, device: str = "cuda") -> dict:
        """
        Sample a mini-batch by uniform random index.

        Parameters
        ----------
        batch_size : int
        device : str

        Returns
        -------
        dict with keys:
            obs_seq      : torch.Tensor (B, window_size, token_dim)
            action       : torch.Tensor (B, action_dim)
            reward       : torch.Tensor (B, 1)
            next_obs_seq : torch.Tensor (B, window_size, token_dim)
            done         : torch.Tensor (B, 1)
        """
        assert self._size >= batch_size, (
            f"WindowReplayBuffer has {self._size} windows; cannot sample {batch_size}."
        )
        idx = np.random.randint(0, self._size, size=batch_size)

        def _to_t(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr, dtype=torch.float32).to(device)

        return {
            "obs_seq":      _to_t(self._obs_seq[idx]),
            "action":       _to_t(self._action[idx]),
            "reward":       _to_t(self._reward[idx]),
            "next_obs_seq": _to_t(self._next_obs_seq[idx]),
            "done":         _to_t(self._done[idx]),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def is_ready(self, min_steps: int) -> bool:
        return self._size >= min_steps

    def clear(self) -> None:
        self._ptr  = 0
        self._size = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the filled portion of the buffer to a compressed .npz file.

        Parameters
        ----------
        path : str
            Destination file path (e.g. checkpoints/sac_transformer_v1/buffer_pos.npz).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            obs_seq      = self._obs_seq[:self._size],
            next_obs_seq = self._next_obs_seq[:self._size],
            action       = self._action[:self._size],
            reward       = self._reward[:self._size],
            done         = self._done[:self._size],
            _ptr         = np.array(self._ptr,  dtype=np.int64),
            _size        = np.array(self._size, dtype=np.int64),
            _capacity    = np.array(self.capacity, dtype=np.int64),
        )
        logger.info(
            f"WindowReplayBuffer saved: {path}  size={self._size}  ptr={self._ptr}"
        )

    def load(self, path: str) -> None:
        """
        Load buffer from a .npz file produced by save().

        The saved capacity must be <= self.capacity.  If the file was saved
        with a smaller capacity, the entries are placed at the beginning and
        _ptr / _size are set accordingly.

        Parameters
        ----------
        path : str
        """
        data  = np.load(path, allow_pickle=False)
        saved_size = int(data["_size"])
        saved_ptr  = int(data["_ptr"])

        if saved_size > self.capacity:
            logger.warning(
                f"WindowReplayBuffer.load: saved size {saved_size} > capacity {self.capacity}. "
                f"Loading only the most recent {self.capacity} windows."
            )
            # Take the most recent capacity entries (circular-buffer order).
            # The circular buffer writes oldest at _ptr and newest just before _ptr.
            # Reconstruct the full ordered array from the raw on-disk arrays,
            # then keep only the last capacity entries.
            obs_raw = data["obs_seq"]   # (saved_size, W, T)
            # saved_ptr points to the slot AFTER the last written entry.
            # Oldest = saved_ptr, newest = saved_ptr - 1 (mod saved_size).
            order = np.concatenate([
                np.arange(saved_ptr, saved_size),
                np.arange(0, saved_ptr),
            ])  # oldest-first ordering
            order = order[-self.capacity:]  # keep most recent capacity
            n = len(order)
            self._obs_seq[:n]      = data["obs_seq"][order]
            self._next_obs_seq[:n] = data["next_obs_seq"][order]
            self._action[:n]       = data["action"][order]
            self._reward[:n]       = data["reward"][order]
            self._done[:n]         = data["done"][order]
            self._size = n
            self._ptr  = n % self.capacity
        else:
            n = saved_size
            self._obs_seq[:n]      = data["obs_seq"]
            self._next_obs_seq[:n] = data["next_obs_seq"]
            self._action[:n]       = data["action"]
            self._reward[:n]       = data["reward"]
            self._done[:n]         = data["done"]
            self._size = n
            # Restore the original _ptr so that subsequent pushes overwrite
            # in FIFO order relative to the saved content.
            self._ptr  = saved_ptr if saved_size < self.capacity else saved_ptr

        logger.info(
            f"WindowReplayBuffer loaded: {path}  size={self._size}  ptr={self._ptr}"
        )


# ── DualWindowReplayBuffer ─────────────────────────────────────────────────────

class DualWindowReplayBuffer:
    """
    Dual window-based replay buffer that splits windows by the sign of reward[t].

    Windows with reward[t] > 0 go to positive_buffer.
    Windows with reward[t] <= 0 go to negative_buffer.

    Sampling draws batch_size // 2 from each buffer (with graceful fallback
    when one side is sparse) and shuffles the combined batch.

    Parameters
    ----------
    capacity : int
        Maximum windows per sub-buffer (50,000 each).
    token_dim : int
    action_dim : int
    window_size : int
    """

    def __init__(
        self,
        capacity:   int = 50_000,
        token_dim:  int = 50,
        action_dim: int = 3,
        window_size: int = 75,
    ):
        self.capacity    = capacity
        self.token_dim   = token_dim
        self.action_dim  = action_dim
        self.window_size = window_size

        self.positive_buffer = WindowReplayBuffer(capacity, token_dim, action_dim, window_size)
        self.negative_buffer = WindowReplayBuffer(capacity, token_dim, action_dim, window_size)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def push_episode(
        self,
        obs_tokens: np.ndarray,
        actions:    np.ndarray,
        rewards:    np.ndarray,
        dones:      np.ndarray,
    ) -> int:
        """
        Slide a window over the episode and push each window to the correct side.

        Episodes shorter than window_size + 1 (= 76 frames) are discarded
        entirely: there is no valid (obs_seq, next_obs_seq) pair.

        Parameters
        ----------
        obs_tokens : np.ndarray, shape (T, token_dim)
        actions    : np.ndarray, shape (T, action_dim)
        rewards    : np.ndarray, shape (T,)
        dones      : np.ndarray, shape (T,)

        Returns
        -------
        int : number of windows added (0 if episode was discarded)
        """
        T = len(obs_tokens)
        min_len = self.window_size + 1  # need at least 76 frames

        if T < min_len:
            logger.debug(
                f"DualWindowReplayBuffer.push_episode: discarding episode of length {T} "
                f"(< {min_len} frames required)."
            )
            return 0

        obs_tokens = np.asarray(obs_tokens, dtype=np.float32)
        actions    = np.asarray(actions,    dtype=np.float32)
        rewards    = np.asarray(rewards,    dtype=np.float32)
        dones      = np.asarray(dones,      dtype=np.float32)

        W = self.window_size
        n_added = 0

        # t ranges from W-1 (=74) to T-2 (inclusive); t+1 is the last index
        # needed for next_obs_seq.
        for t in range(W - 1, T - 1):
            obs_seq      = obs_tokens[t - W + 1 : t + 1]      # (W, token_dim)
            next_obs_seq = obs_tokens[t - W + 2 : t + 2]      # (W, token_dim)
            action       = actions[t]                           # (action_dim,)
            reward       = float(rewards[t])
            done         = float(dones[t])

            # Route by the single reward at timestep t.
            if float(rewards[t]) > 0.0:
                self.positive_buffer.push(obs_seq, next_obs_seq, action, reward, done)
            else:
                self.negative_buffer.push(obs_seq, next_obs_seq, action, reward, done)
            n_added += 1

        return n_added

    def sample(self, batch_size: int, device: str = "cuda") -> dict:
        """
        Sample a balanced mini-batch (50/50 pos/neg split with graceful fallback).

        Parameters
        ----------
        batch_size : int
        device : str

        Returns
        -------
        dict with keys: obs_seq, action, reward, next_obs_seq, done
        """
        pos_size = len(self.positive_buffer)
        neg_size = len(self.negative_buffer)
        total    = pos_size + neg_size

        assert total >= batch_size, (
            f"DualWindowReplayBuffer has only {total} windows "
            f"(pos={pos_size}, neg={neg_size}); cannot sample {batch_size}."
        )

        half = batch_size // 2

        if pos_size == 0:
            n_pos, n_neg = 0, batch_size
        elif neg_size == 0:
            n_pos, n_neg = batch_size, 0
        else:
            n_pos = min(half, pos_size)
            n_neg = min(half, neg_size)
            shortfall = batch_size - n_pos - n_neg
            if shortfall > 0:
                if pos_size - n_pos >= shortfall:
                    n_pos += shortfall
                else:
                    n_neg += shortfall

        batches = []
        if n_pos > 0:
            batches.append({
                k: v.cpu().numpy()
                for k, v in self.positive_buffer.sample(n_pos, device="cpu").items()
            })
        if n_neg > 0:
            batches.append({
                k: v.cpu().numpy()
                for k, v in self.negative_buffer.sample(n_neg, device="cpu").items()
            })

        combined = {
            key: np.concatenate([b[key] for b in batches], axis=0)
            for key in batches[0].keys()
        }
        perm = np.random.permutation(batch_size)
        for key in combined:
            combined[key] = combined[key][perm]

        def _to_t(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr, dtype=torch.float32).to(device)

        return {k: _to_t(v) for k, v in combined.items()}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.positive_buffer) + len(self.negative_buffer)

    def is_ready(self, min_steps: int) -> bool:
        return len(self) >= min_steps

    def clear(self) -> None:
        self.positive_buffer.clear()
        self.negative_buffer.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save both sub-buffers.

        Files: <stem>_pos.npz and <stem>_neg.npz.

        Parameters
        ----------
        path : str
            Base path (with or without .npz extension).
        """
        stem = path[:-4] if path.endswith(".npz") else path
        self.positive_buffer.save(stem + "_pos.npz")
        self.negative_buffer.save(stem + "_neg.npz")

    def load(self, path: str) -> None:
        """
        Load both sub-buffers from files produced by save().

        Parameters
        ----------
        path : str
            Base path (with or without .npz extension).
        """
        stem = path[:-4] if path.endswith(".npz") else path
        pos_path = stem + "_pos.npz"
        neg_path = stem + "_neg.npz"
        if os.path.isfile(pos_path):
            self.positive_buffer.load(pos_path)
        if os.path.isfile(neg_path):
            self.negative_buffer.load(neg_path)
