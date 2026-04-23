"""
One-shot: snapshot the first 50K rows of each NEGATIVE channel buffer.
Produces a truncated, drop-in-loadable backup (capacity=50000).

DOES NOT TOUCH THE LIVE BUFFER — read-only on source files.
"""
import os
import json
import shutil
import time
from datetime import datetime

SRC_DIR = r"D:/Git/virtual457-projects/AssetoCorsa/gym/transformer_sac_vectorq_v2_final_fineTune/checkpoints/buffers"
BACKUP_ROOT = r"D:/Git/virtual457-projects/AssetoCorsa/trained_models/SAC_VectorQ_V2/buffer_backups"

TOKEN_DIM   = 50
WINDOW_SIZE = 75
ACTION_DIM  = 3
REWARD_DIM  = 3
N_KEEP      = 50_000

CHANNELS = ["steer_neg", "throttle_neg", "brake_neg"]

# bytes per row per file kind (float32 = 4 bytes)
ROW_BYTES = {
    "obs.dat":      WINDOW_SIZE * TOKEN_DIM * 4,   # 15000
    "next_obs.dat": WINDOW_SIZE * TOKEN_DIM * 4,   # 15000
    "action.dat":   ACTION_DIM * 4,                # 12
    "reward.dat":   REWARD_DIM * 4,                # 12
    "done.dat":     1 * 4,                         # 4
}

CHUNK = 64 * 1024 * 1024  # 64 MB per read/write pass


def copy_first_n_bytes(src: str, dst: str, nbytes: int) -> None:
    """Copy exactly `nbytes` from the start of src -> dst, streamed."""
    remaining = nbytes
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while remaining > 0:
            buf = fi.read(min(CHUNK, remaining))
            if not buf:
                break
            fo.write(buf)
            remaining -= len(buf)


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_dir = os.path.join(BACKUP_ROOT, f"{ts}_negatives_first50k")
    os.makedirs(dst_dir, exist_ok=True)
    print(f"[backup] dest: {dst_dir}")

    total_bytes = 0
    t0 = time.time()

    for ch in CHANNELS:
        print(f"\n[{ch}]")
        for fname, row_bytes in ROW_BYTES.items():
            src = os.path.join(SRC_DIR, f"{ch}_{fname}")
            dst = os.path.join(dst_dir, f"{ch}_{fname}")
            nbytes = N_KEEP * row_bytes
            src_size = os.path.getsize(src)
            assert src_size >= nbytes, (
                f"{src} is only {src_size} bytes; need at least {nbytes}"
            )
            copy_first_n_bytes(src, dst, nbytes)
            actual = os.path.getsize(dst)
            print(f"  {fname:14s} {actual/1e6:>9.1f} MB  (kept {N_KEEP} rows)")
            total_bytes += actual

        # meta reflects the truncated snapshot (capacity is declared at load time)
        meta_path = os.path.join(dst_dir, f"{ch}_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"ptr": N_KEEP, "size": N_KEEP}, f)
        print(f"  meta.json     -> ptr={N_KEEP}, size={N_KEEP}")

    dt = time.time() - t0
    print(
        f"\n[backup] DONE — {total_bytes/1e9:.2f} GB copied in {dt:.1f}s  "
        f"({total_bytes/1e6/dt:.0f} MB/s)"
    )
    print(f"[backup] path: {dst_dir}")
    print(
        "\nTo reload later: open SixChannelMemmapBuffer with "
        "capacity_per_buffer=50000 pointed at this dir."
    )


if __name__ == "__main__":
    main()
