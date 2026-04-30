"""
replay_buffer.py — 6-channel memmap-backed replay buffer for stratified per-action sampling.

Architecture
------------
6 MemmapCircularBuffer instances, each backed by memory-mapped files:
    steer_pos, steer_neg       — routed by reward[0] > 0
    throttle_pos, throttle_neg — routed by reward[1] > 0
    brake_pos, brake_neg       — routed by reward[2] > 0

Each window is stored 3 times (once per channel routing decision).
All buffers store the full (3,) reward vector so every Q-head gets its
target regardless of which channel routed the window.

Sampling
--------
Stratified: batch_size // 6 windows drawn from each of the 6 buffers,
concatenated and shuffled.  Each Q-head sees ~50/50 pos/neg balance
for its own channel.

Storage
-------
All data lives on disk via np.memmap.  RAM usage is minimal — only the
accessed pages are loaded by the OS page cache.

Disk usage per buffer ~ 2 x capacity x window_size x token_dim x 4 bytes
(obs_seq + next_obs_seq dominate).

Default capacity 100K per buffer -> ~2.8 GB per buffer -> ~16.8 GB total.

Collection workflow
-------------------
push_episode(obs_tokens, actions, rewards, dones)
-> slide windows, route each to 3 of 6 buffers by per-channel reward sign
-> writes go directly to memmap (disk)

Training workflow
-----------------
sample(batch_size, device)
-> draw batch_size // 6 from each buffer
-> only sampled rows are read into RAM
-> concatenate, shuffle, move to device
"""

import os
import json
import logging
import time
import threading
import queue

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MemmapCircularBuffer — one disk-backed circular buffer
# ---------------------------------------------------------------------------

class MemmapCircularBuffer:
    """
    Single circular replay buffer backed by memory-mapped files.

    Files on disk::

        <base_dir>/<name>_obs.dat       (capacity, window_size, token_dim) float32
        <base_dir>/<name>_next_obs.dat  (capacity, window_size, token_dim) float32
        <base_dir>/<name>_action.dat    (capacity, action_dim) float32
        <base_dir>/<name>_reward.dat    (capacity, reward_dim) float32
        <base_dir>/<name>_done.dat      (capacity, 1) float32
        <base_dir>/<name>_meta.json     {ptr, size}

    Parameters
    ----------
    base_dir : str
        Directory for memmap files.
    name : str
        Buffer name (used as file prefix, e.g. ``steer_pos``).
    capacity : int
        Maximum windows (FIFO circular).
    """

    def __init__(
        self,
        base_dir: str,
        name: str,
        capacity: int = 100_000,
        token_dim: int = 50,
        action_dim: int = 3,
        window_size: int = 75,
        reward_dim: int = 3,
    ):
        self.base_dir = base_dir
        self.name = name
        self.capacity = capacity
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.window_size = window_size
        self.reward_dim = reward_dim

        os.makedirs(base_dir, exist_ok=True)

        self._meta_path = os.path.join(base_dir, f"{name}_meta.json")
        obs_path = os.path.join(base_dir, f"{name}_obs.dat")
        existing = os.path.isfile(obs_path) and os.path.isfile(self._meta_path)
        mode = "r+" if existing else "w+"

        self._obs = np.memmap(
            os.path.join(base_dir, f"{name}_obs.dat"),
            dtype=np.float32, mode=mode,
            shape=(capacity, window_size, token_dim),
        )
        self._next_obs = np.memmap(
            os.path.join(base_dir, f"{name}_next_obs.dat"),
            dtype=np.float32, mode=mode,
            shape=(capacity, window_size, token_dim),
        )
        self._action = np.memmap(
            os.path.join(base_dir, f"{name}_action.dat"),
            dtype=np.float32, mode=mode,
            shape=(capacity, action_dim),
        )
        self._reward = np.memmap(
            os.path.join(base_dir, f"{name}_reward.dat"),
            dtype=np.float32, mode=mode,
            shape=(capacity, reward_dim),
        )
        self._done = np.memmap(
            os.path.join(base_dir, f"{name}_done.dat"),
            dtype=np.float32, mode=mode,
            shape=(capacity, 1),
        )

        # Load or initialise metadata.
        if existing:
            with open(self._meta_path, "r") as f:
                meta = json.load(f)
            self._ptr = int(meta["ptr"])
            self._size = int(meta["size"])
            logger.info(
                f"MemmapCircularBuffer '{name}' resumed: "
                f"size={self._size}  ptr={self._ptr}"
            )
        else:
            self._ptr = 0
            self._size = 0
            self._save_meta()
            logger.info(
                f"MemmapCircularBuffer '{name}' created: capacity={capacity}"
            )

    # ------------------------------------------------------------------
    # Metadata persistence
    # ------------------------------------------------------------------

    def _save_meta(self) -> None:
        with open(self._meta_path, "w") as f:
            json.dump({"ptr": self._ptr, "size": self._size}, f)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def push(
        self,
        obs_seq: np.ndarray,
        next_obs_seq: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: float,
    ) -> None:
        """Store one pre-computed window at the current write pointer."""
        self._obs[self._ptr] = obs_seq
        self._next_obs[self._ptr] = next_obs_seq
        self._action[self._ptr] = action
        self._reward[self._ptr] = reward
        self._done[self._ptr, 0] = done

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------
    # Preload (memmap -> RAM)
    # ------------------------------------------------------------------

    def preload(self) -> None:
        """
        Copy the filled portion of memmap arrays into contiguous RAM arrays.
        After this call, sample() reads from RAM instead of disk.
        """
        n = self._size
        if n == 0:
            logger.info(f"MemmapCircularBuffer '{self.name}' is empty — nothing to preload.")
            return

        t0 = time.time()
        self._obs_ram      = np.array(self._obs[:n])
        self._next_obs_ram = np.array(self._next_obs[:n])
        self._action_ram   = np.array(self._action[:n])
        self._reward_ram   = np.array(self._reward[:n])
        self._done_ram     = np.array(self._done[:n])
        elapsed = time.time() - t0
        ram_gb = (self._obs_ram.nbytes + self._next_obs_ram.nbytes +
                  self._action_ram.nbytes + self._reward_ram.nbytes +
                  self._done_ram.nbytes) / 1e9
        logger.info(
            f"MemmapCircularBuffer '{self.name}' preloaded: "
            f"{n} windows  {ram_gb:.2f} GB  {elapsed:.1f}s"
        )

    @property
    def _is_preloaded(self) -> bool:
        return hasattr(self, '_obs_ram')

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int, device: str = "cuda") -> dict:
        """Uniform random sample from the filled portion."""
        assert self._size >= batch_size, (
            f"MemmapCircularBuffer '{self.name}' has {self._size} windows; "
            f"cannot sample {batch_size}."
        )
        idx = np.random.randint(0, self._size, size=batch_size)

        if self._is_preloaded:
            obs      = self._obs_ram[idx]
            next_obs = self._next_obs_ram[idx]
            action   = self._action_ram[idx]
            reward   = self._reward_ram[idx]
            done     = self._done_ram[idx]
        else:
            obs      = np.array(self._obs[idx])
            next_obs = np.array(self._next_obs[idx])
            action   = np.array(self._action[idx])
            reward   = np.array(self._reward[idx])
            done     = np.array(self._done[idx])

        def _to_t(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr, dtype=torch.float32).to(device)

        return {
            "obs_seq":      _to_t(obs),
            "next_obs_seq": _to_t(next_obs),
            "action":       _to_t(action),
            "reward":       _to_t(reward),
            "done":         _to_t(done),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush memmap pages to disk and persist ptr/size metadata."""
        self._obs.flush()
        self._next_obs.flush()
        self._action.flush()
        self._reward.flush()
        self._done.flush()
        self._save_meta()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def is_ready(self, min_steps: int) -> bool:
        return self._size >= min_steps


# ---------------------------------------------------------------------------
# SixChannelMemmapBuffer — per-channel stratified sampling
# ---------------------------------------------------------------------------

class SixChannelMemmapBuffer:
    """
    6 memmap-backed circular buffers for per-channel stratified sampling.

    Each window is stored 3 times — once per channel routing decision::

        reward[0] > 0 -> steer_pos,    else -> steer_neg
        reward[1] > 0 -> throttle_pos, else -> throttle_neg
        reward[2] > 0 -> brake_pos,    else -> brake_neg

    Sampling draws ``batch_size // 6`` from each buffer for balanced training.
    Graceful fallback when some buffers are empty or sparse.

    Parameters
    ----------
    base_dir : str
        Directory for all memmap files.
    capacity_per_buffer : int
        Windows per sub-buffer (FIFO). Default 100K.
    """

    CHANNEL_NAMES = [
        "steer_pos", "steer_neg",
        "throttle_pos", "throttle_neg",
        "brake_pos", "brake_neg",
    ]

    def __init__(
        self,
        base_dir: str,
        capacity_per_buffer: int = 100_000,
        token_dim: int = 50,
        action_dim: int = 3,
        window_size: int = 75,
        reward_dim: int = 3,
    ):
        self.base_dir = base_dir
        self.capacity = capacity_per_buffer
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.window_size = window_size
        self.reward_dim = reward_dim

        self.buffers: dict[str, MemmapCircularBuffer] = {}
        for name in self.CHANNEL_NAMES:
            self.buffers[name] = MemmapCircularBuffer(
                base_dir=base_dir,
                name=name,
                capacity=capacity_per_buffer,
                token_dim=token_dim,
                action_dim=action_dim,
                window_size=window_size,
                reward_dim=reward_dim,
            )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def push_episode(
        self,
        obs_tokens: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> int:
        """
        Slide a window over the episode and route each window to 3 channel buffers.

        Each window is stored 3 times (once per channel based on reward sign).
        Episodes shorter than ``window_size + 1`` (76 frames) are discarded.

        Parameters
        ----------
        obs_tokens : (T, token_dim)
        actions    : (T, action_dim)
        rewards    : (T, reward_dim) — vector rewards
        dones      : (T,)

        Returns
        -------
        int : number of unique windows added (each stored 3x internally)
        """
        T = len(obs_tokens)
        min_len = self.window_size + 1
        if T < min_len:
            logger.debug(
                f"SixChannelMemmapBuffer.push_episode: discarding episode of "
                f"length {T} (< {min_len} frames required)."
            )
            return 0

        obs_tokens = np.asarray(obs_tokens, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        W = self.window_size
        n_added = 0

        for t in range(W - 1, T - 1):
            obs_seq      = obs_tokens[t - W + 1 : t + 1]      # (W, token_dim)
            next_obs_seq = obs_tokens[t - W + 2 : t + 2]      # (W, token_dim)
            action       = actions[t]                           # (action_dim,)
            reward       = rewards[t]                           # (reward_dim,)
            done         = float(dones[t])

            # Route to steer channel pair.
            if float(reward[0]) > 0.0:
                self.buffers["steer_pos"].push(
                    obs_seq, next_obs_seq, action, reward, done)
            else:
                self.buffers["steer_neg"].push(
                    obs_seq, next_obs_seq, action, reward, done)

            # Route to throttle channel pair.
            if float(reward[1]) > 0.0:
                self.buffers["throttle_pos"].push(
                    obs_seq, next_obs_seq, action, reward, done)
            else:
                self.buffers["throttle_neg"].push(
                    obs_seq, next_obs_seq, action, reward, done)

            # Route to brake channel pair.
            if float(reward[2]) > 0.0:
                self.buffers["brake_pos"].push(
                    obs_seq, next_obs_seq, action, reward, done)
            else:
                self.buffers["brake_neg"].push(
                    obs_seq, next_obs_seq, action, reward, done)

            n_added += 1

        # Terminal frame: done=1 means crash/OOT → reward is [-1,-1,-1].
        # Store with dummy next_obs (zeroed out by 1-done in Bellman target).
        if T >= W + 1 and float(dones[-1]) == 1.0:
            t = T - 1
            obs_seq      = obs_tokens[t - W + 1 : t + 1]
            next_obs_seq = obs_seq.copy()              # dummy — not used when done=1
            action       = actions[t]
            reward       = rewards[t]

            self.buffers["steer_neg"].push(obs_seq, next_obs_seq, action, reward, 1.0)
            self.buffers["throttle_neg"].push(obs_seq, next_obs_seq, action, reward, 1.0)
            self.buffers["brake_neg"].push(obs_seq, next_obs_seq, action, reward, 1.0)

            n_added += 1

        return n_added

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int, device: str = "cuda") -> dict:
        """
        Stratified sample: draw ``batch_size // 6`` from each of the 6 buffers.

        Falls back gracefully when some buffers are empty — draws more from
        non-empty buffers to fill the batch.

        Returns
        -------
        dict with keys: obs_seq, action, reward, next_obs_seq, done
            Same format as DualWindowReplayBuffer for drop-in compatibility.
        """
        sizes = {name: len(buf) for name, buf in self.buffers.items()}
        total = sum(sizes.values())
        assert total >= batch_size, (
            f"SixChannelMemmapBuffer has {total} total entries "
            f"({sizes}); cannot sample {batch_size}."
        )

        non_empty = [(name, sz) for name, sz in sizes.items() if sz > 0]
        n_non_empty = len(non_empty)

        if n_non_empty == 0:
            raise RuntimeError("All 6 channel buffers are empty.")

        # Compute per-buffer draw counts: target batch_size // 6 each.
        n_target = batch_size // max(n_non_empty, 1)
        draws: dict[str, int] = {}
        remainder = batch_size

        for name, sz in non_empty:
            n = min(n_target, sz)
            draws[name] = n
            remainder -= n

        # Distribute remaining draws across non-empty buffers.
        while remainder > 0:
            distributed = False
            for name, sz in non_empty:
                if draws[name] < sz and remainder > 0:
                    draws[name] += 1
                    remainder -= 1
                    distributed = True
            if not distributed:
                break  # all buffers fully drawn

        # Sample from each buffer.
        batches = []
        for name, n in draws.items():
            if n > 0:
                batch = self.buffers[name].sample(n, device="cpu")
                batches.append({k: v.numpy() for k, v in batch.items()})

        # Concatenate.
        combined = {
            key: np.concatenate([b[key] for b in batches], axis=0)
            for key in batches[0].keys()
        }

        # Shuffle.
        actual_size = combined["obs_seq"].shape[0]
        perm = np.random.permutation(actual_size)
        for key in combined:
            combined[key] = combined[key][perm]

        def _to_t(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr, dtype=torch.float32).to(device)

        return {k: _to_t(v) for k, v in combined.items()}

    # ------------------------------------------------------------------
    # Preload (memmap -> RAM for fast sampling)
    # ------------------------------------------------------------------

    def preload(self) -> None:
        """
        Preload all 6 buffers from disk into RAM.
        After this, sampling is pure RAM reads — no disk I/O.
        """
        t0 = time.time()
        total_gb = 0.0
        for name, buf in self.buffers.items():
            buf.preload()
            if buf._is_preloaded:
                total_gb += (buf._obs_ram.nbytes + buf._next_obs_ram.nbytes +
                             buf._action_ram.nbytes + buf._reward_ram.nbytes +
                             buf._done_ram.nbytes) / 1e9
        elapsed = time.time() - t0
        logger.info(
            f"SixChannelMemmapBuffer preloaded: {total_gb:.2f} GB total  {elapsed:.1f}s"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush all 6 memmap buffers to disk and persist metadata."""
        for buf in self.buffers.values():
            buf.flush()
        logger.info(
            f"SixChannelMemmapBuffer flushed — sizes: {self.sizes()}"
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return sum(len(buf) for buf in self.buffers.values())

    def is_ready(self, min_steps: int) -> bool:
        return len(self) >= min_steps

    def sizes(self) -> dict:
        """Return size of each sub-buffer."""
        return {name: len(buf) for name, buf in self.buffers.items()}

    def clear(self) -> None:
        """Reset all buffer pointers (data on disk is overwritable)."""
        for buf in self.buffers.values():
            buf._ptr = 0
            buf._size = 0
            buf._save_meta()


# ---------------------------------------------------------------------------
# ChunkedPrefetcher — double-buffered chunk loading for low-RAM systems
# ---------------------------------------------------------------------------

class ChunkedPrefetcher:
    """
    Double-buffered chunk sampler for systems with limited RAM.

    Pipeline:
        1. Background thread loads a chunk (e.g. 50K windows) from memmap → RAM
        2. Main thread samples mini-batches from the in-RAM chunk (fast)
        3. When the chunk is exhausted, swap to the next pre-loaded chunk
        4. Background thread starts loading the next chunk immediately

    RAM usage: ~2 chunks worth of data. At chunk_size=50K:
        2 * 50K * 30KB ≈ 3 GB RAM (vs 12 GB for full preload).

    Usage::

        prefetcher = ChunkedPrefetcher(replay_buffer, chunk_size=50_000,
                                        batch_size=256, device="cuda")
        prefetcher.start()

        for step in range(n_steps):
            batch, wait_ms = prefetcher.get()
            losses = sac.update_from_batch(batch)

        prefetcher.stop()

    Parameters
    ----------
    buffer : SixChannelMemmapBuffer
    chunk_size : int
        Total windows to load per chunk (split across 6 buffers).
    batch_size : int
        Mini-batch size for gradient steps.
    device : str
        GPU device string.
    """

    def __init__(
        self,
        buffer: SixChannelMemmapBuffer,
        chunk_size: int = 50_000,
        batch_size: int = 256,
        device: str = "cuda",
    ):
        self.buffer = buffer
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.device = device

        self._chunk_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Pre-allocate two fixed chunk buffers (A/B) to avoid repeated alloc/free.
        W = buffer.window_size
        D = buffer.token_dim
        A = buffer.action_dim
        R = buffer.reward_dim
        self._pool = [
            {
                "obs_seq":      np.empty((chunk_size, W, D), dtype=np.float32),
                "next_obs_seq": np.empty((chunk_size, W, D), dtype=np.float32),
                "action":       np.empty((chunk_size, A),    dtype=np.float32),
                "reward":       np.empty((chunk_size, R),    dtype=np.float32),
                "done":         np.empty((chunk_size, 1),    dtype=np.float32),
            }
            for _ in range(2)
        ]
        self._pool_idx = 0  # which buffer the worker writes into next

        pool_mb = sum(v.nbytes for v in self._pool[0].values()) * 2 / 1e6
        logger.info(f"ChunkedPrefetcher: pre-allocated 2 chunk buffers ({pool_mb:.0f} MB total)")

        # Current chunk state.
        self._chunk: dict | None = None       # numpy arrays in RAM
        self._chunk_ptr: int = 0              # next index within chunk
        self._chunk_len: int = 0              # actual chunk size

        # Stats.
        self._wait_times: list[float] = []
        self._gpu_wait_times: list[float] = []
        self._chunk_load_times: list[float] = []
        self._chunks_loaded: int = 0

    def _load_one_chunk(self) -> tuple:
        """
        Load one chunk from the 6-channel buffer into a pre-allocated buffer.

        Draws chunk_size // 6 from each buffer (stratified),
        copies into fixed memory, shuffles in-place — zero allocations.

        Returns (chunk_dict, actual_size).
        """
        sizes = {name: len(buf) for name, buf in self.buffer.buffers.items()}
        non_empty = [(name, sz) for name, sz in sizes.items() if sz > 0]

        # Compute per-buffer draw counts.
        n_target = self.chunk_size // max(len(non_empty), 1)
        draws: dict[str, int] = {}
        remainder = self.chunk_size

        for name, sz in non_empty:
            n = min(n_target, sz)
            draws[name] = n
            remainder -= n

        while remainder > 0:
            distributed = False
            for name, sz in non_empty:
                if draws[name] < sz and remainder > 0:
                    draws[name] += 1
                    remainder -= 1
                    distributed = True
            if not distributed:
                break

        # Grab the next pre-allocated buffer from the pool.
        dest = self._pool[self._pool_idx]
        self._pool_idx = 1 - self._pool_idx

        # Copy from each sub-buffer (memmap → pre-allocated RAM).
        # Use contiguous slices (sequential disk reads) for fast memmap I/O.
        offset = 0

        for name, n in draws.items():
            if n <= 0:
                continue
            buf = self.buffer.buffers[name]
            t_buf = time.perf_counter()

            if buf._is_preloaded:
                idx = np.random.randint(0, buf._size, size=n)
                dest["obs_seq"][offset:offset+n]      = buf._obs_ram[idx]
                dest["next_obs_seq"][offset:offset+n]  = buf._next_obs_ram[idx]
                dest["action"][offset:offset+n]        = buf._action_ram[idx]
                dest["reward"][offset:offset+n]        = buf._reward_ram[idx]
                dest["done"][offset:offset+n]          = buf._done_ram[idx]
                src = "ram"
            else:
                # Contiguous slice: pick a random start, read n rows sequentially.
                if n >= buf._size:
                    sl = slice(0, buf._size)
                    dest["obs_seq"][offset:offset+n]      = buf._obs[sl]
                    dest["next_obs_seq"][offset:offset+n]  = buf._next_obs[sl]
                    dest["action"][offset:offset+n]        = buf._action[sl]
                    dest["reward"][offset:offset+n]        = buf._reward[sl]
                    dest["done"][offset:offset+n]          = buf._done[sl]
                else:
                    start = np.random.randint(0, buf._size)
                    end = start + n
                    if end <= buf._size:
                        sl = slice(start, end)
                        dest["obs_seq"][offset:offset+n]      = buf._obs[sl]
                        dest["next_obs_seq"][offset:offset+n]  = buf._next_obs[sl]
                        dest["action"][offset:offset+n]        = buf._action[sl]
                        dest["reward"][offset:offset+n]        = buf._reward[sl]
                        dest["done"][offset:offset+n]          = buf._done[sl]
                    else:
                        # Wrap around — two contiguous reads.
                        wrap = end - buf._size
                        n1 = buf._size - start
                        s1, s2 = slice(start, buf._size), slice(0, wrap)
                        dest["obs_seq"][offset:offset+n1]          = buf._obs[s1]
                        dest["obs_seq"][offset+n1:offset+n]        = buf._obs[s2]
                        dest["next_obs_seq"][offset:offset+n1]     = buf._next_obs[s1]
                        dest["next_obs_seq"][offset+n1:offset+n]   = buf._next_obs[s2]
                        dest["action"][offset:offset+n1]           = buf._action[s1]
                        dest["action"][offset+n1:offset+n]         = buf._action[s2]
                        dest["reward"][offset:offset+n1]           = buf._reward[s1]
                        dest["reward"][offset+n1:offset+n]         = buf._reward[s2]
                        dest["done"][offset:offset+n1]             = buf._done[s1]
                        dest["done"][offset+n1:offset+n]           = buf._done[s2]
                src = "disk"
            buf_ms = (time.perf_counter() - t_buf) * 1000.0
            logger.debug(
                f"    [chunk] {name}: {n} windows from {src} in {buf_ms:.0f}ms"
            )
            offset += n

        # Shuffle in-place using the filled portion.
        t_merge = time.perf_counter()
        perm = np.random.permutation(offset)
        for k in dest:
            dest[k][:offset] = dest[k][perm]
        merge_ms = (time.perf_counter() - t_merge) * 1000.0
        logger.debug(f"    [chunk] shuffle: {offset} windows in {merge_ms:.0f}ms")

        return dest, offset

    def _worker(self) -> None:
        """Background thread: continuously loads chunks into the queue."""
        chunk_id = 1  # first chunk loaded in start(), this is #2+
        while not self._stop_event.is_set():
            chunk_id += 1
            logger.debug(f"  [prefetch] chunk #{chunk_id} loading ...")
            t0 = time.perf_counter()
            try:
                chunk, actual_size = self._load_one_chunk()
            except Exception as e:
                logger.error(f"ChunkedPrefetcher worker error: {e}")
                break
            load_ms = (time.perf_counter() - t0) * 1000.0
            self._chunk_load_times.append(load_ms)
            self._chunks_loaded += 1
            logger.debug(
                f"  [prefetch] chunk #{chunk_id} ready — "
                f"{actual_size} windows  loaded in {load_ms:.0f}ms  "
                f"queue={'waiting' if self._chunk_queue.full() else 'has slot'}"
            )

            try:
                self._queue.put((chunk, actual_size, load_ms), timeout=1.0)
            except queue.Full:
                if self._stop_event.is_set():
                    break
                logger.debug(f"  [prefetch] queue full — GPU still using current chunk, waiting ...")
                try:
                    self._chunk_queue.put((chunk, actual_size, load_ms), timeout=60.0)
                except queue.Full:
                    continue
            logger.debug(f"  [prefetch] chunk #{chunk_id} queued for GPU")

    # Alias for compatibility.
    @property
    def _queue(self):
        return self._chunk_queue

    def _swap_chunk(self) -> float:
        """Block until the next chunk is ready, then swap it in. Returns wait_ms."""
        queue_ready = not self._chunk_queue.empty()
        if not queue_ready:
            logger.debug("  [prefetch] chunk exhausted — waiting for background load ...")
        t0 = time.perf_counter()
        self._chunk, self._chunk_len, load_ms = self._chunk_queue.get()
        wait_ms = (time.perf_counter() - t0) * 1000.0
        self._chunk_ptr = 0
        logger.debug(
            f"  [prefetch] swapped to new chunk — "
            f"{self._chunk_len} windows  "
            f"{'instant' if wait_ms < 1.0 else f'waited {wait_ms:.0f}ms'}  "
            f"(chunk was loaded in {load_ms:.0f}ms)"
        )
        return wait_ms

    def start(self) -> None:
        """Start the background chunk loader."""
        self._stop_event.clear()

        # Load the first chunk synchronously so get() never blocks on empty.
        logger.info(
            f"ChunkedPrefetcher: loading first chunk ({self.chunk_size} windows) ..."
        )
        t0 = time.perf_counter()
        self._chunk, self._chunk_len = self._load_one_chunk()
        first_load_ms = (time.perf_counter() - t0) * 1000.0
        self._chunk_ptr = 0
        self._chunk_load_times.append(first_load_ms)
        self._chunks_loaded += 1

        pool_mb = sum(v.nbytes for v in self._pool[0].values()) * 2 / 1e6
        logger.info(
            f"ChunkedPrefetcher started — "
            f"chunk_size={self._chunk_len}  pool_ram={pool_mb:.0f} MB (fixed)  "
            f"first_load={first_load_ms:.0f}ms  "
            f"batch_size={self.batch_size}  "
            f"batches_per_chunk={self._chunk_len // self.batch_size}  "
            f"device={self.device}"
        )

        # Start background thread to pre-load the next chunk.
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and log stats."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

        # Drain queue.
        while not self._chunk_queue.empty():
            try:
                self._chunk_queue.get_nowait()
            except queue.Empty:
                break

        # Free chunk RAM.
        self._chunk = None

        if self._wait_times:
            logger.info(
                f"ChunkedPrefetcher stopped — "
                f"chunks_loaded={self._chunks_loaded}  "
                f"avg_chunk_load={np.mean(self._chunk_load_times):.0f}ms  "
                f"avg_batch_wait={np.mean(self._wait_times):.1f}ms  "
                f"max_batch_wait={max(self._wait_times):.1f}ms  "
                f"avg_gpu={np.mean(self._gpu_wait_times):.1f}ms  "
                f"total_batches={len(self._wait_times)}"
            )

    def get(self) -> tuple[dict, float]:
        """
        Get the next mini-batch from the current in-RAM chunk.

        When the chunk is exhausted, swaps to the next pre-loaded chunk.
        Batch is moved to GPU here.

        Returns
        -------
        batch : dict — tensors on GPU
        wait_ms : float — time spent waiting (>0 only during chunk swap)
        """
        wait_ms = 0.0

        # Check if current chunk is exhausted.
        if self._chunk_ptr + self.batch_size > self._chunk_len:
            wait_ms = self._swap_chunk()

        # Slice mini-batch from RAM chunk (near-instant).
        t0 = time.perf_counter()
        lo = self._chunk_ptr
        hi = lo + self.batch_size
        self._chunk_ptr = hi

        batch = {}
        for key in self._chunk:
            arr = self._chunk[key][lo:hi]
            batch[key] = torch.as_tensor(arr, dtype=torch.float32).to(self.device)

        gpu_ms = (time.perf_counter() - t0) * 1000.0

        self._wait_times.append(wait_ms)
        self._gpu_wait_times.append(gpu_ms)

        return batch, wait_ms
