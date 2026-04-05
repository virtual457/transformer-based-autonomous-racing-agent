"""
RewardLogger — per-episode reward truth log.

Buffers one row per step in memory.  At episode end, flush() overwrites
reward_truth.jsonl with that episode's rows only.  The file always contains
the latest episode — no rotation, no append across episodes.

Usage
-----
    from rewards.reward_logger import RewardLogger

    logger = RewardLogger()

    # inside the step loop:
    logger.push(row_dict)

    # at episode end (done=True):
    logger.flush()

    # at warmup episode end (don't want to write):
    logger.reset()
"""

import json
import os
from pathlib import Path

REWARD_LOG_PATH = Path("D:/Git/virtual457-projects/AssetoCorsa/reward_log/reward_truth.jsonl")


class RewardLogger:
    """
    Buffers reward-truth rows for one episode and writes them on flush().

    flush() opens the file in 'w' mode (overwrite), so the file always
    contains exactly the latest episode's rows.
    """

    def __init__(self, path: Path = REWARD_LOG_PATH):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, row: dict) -> None:
        """Append one step row to the in-memory buffer."""
        self._buffer.append(row)

    def flush(self) -> None:
        """
        Write all buffered rows to reward_truth.jsonl using 'w' mode
        (overwrite), then clear the buffer.

        No-op if the buffer is empty.
        """
        if not self._buffer:
            return

        with open(self._path, "w", encoding="utf-8") as fh:
            for row in self._buffer:
                fh.write(json.dumps(row) + "\n")

        self._buffer.clear()

    def reset(self) -> None:
        """
        Discard the in-memory buffer without writing.

        Use this at the end of warmup or dummy episodes where the data
        should not be persisted.
        """
        self._buffer.clear()
