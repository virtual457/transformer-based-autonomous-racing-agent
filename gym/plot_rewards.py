"""
plot_rewards.py — Visualise one episode from reward_truth.jsonl.

Usage
-----
    python gym/plot_rewards.py                          # latest episode
    python gym/plot_rewards.py --episode 12             # specific episode
    python gym/plot_rewards.py --input path/to/file     # custom log path
    python gym/plot_rewards.py --show                   # open window instead of save

Output
------
    reward_log/ep_<episode>.png   — 4-panel figure saved automatically

Panels
------
    1. Raw component rewards over frames   (r_progress, r_speed, r_gap_abs, r_smoothness, r_yaw)
    2. Weighted contributions (stacked bar) and total reward
    3. Total reward + cumulative sum
    4. Car state: speed (m/s), lateral gap (m), yaw error inferred from r_yaw, steer action
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_LOG = Path("D:/Git/virtual457-projects/AssetoCorsa/reward_log/reward_truth.jsonl")


def load_episode(log_path: Path, episode: int | None) -> list[dict]:
    """Load all rows for one episode.  If episode is None, use the last episode found."""
    rows: list[dict] = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"No data in {log_path}")

    if episode is None:
        # file is overwrite-per-episode → all rows are the same episode
        episode = rows[0].get("episode", 0)

    ep_rows = [r for r in rows if r.get("episode") == episode]
    if not ep_rows:
        available = sorted({r.get("episode") for r in rows})
        raise ValueError(f"Episode {episode} not found. Available: {available}")

    ep_rows.sort(key=lambda r: r.get("step", 0))
    return ep_rows


def _col(rows, key, default=0.0):
    return np.array([r.get(key, default) for r in rows], dtype=float)


def plot(rows: list[dict], show: bool, out_dir: Path) -> Path | None:
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — run:  pip install matplotlib")
        sys.exit(1)

    episode = rows[0].get("episode", 0)
    phase   = rows[0].get("phase", 0)
    steps   = _col(rows, "step")
    T       = len(steps)

    # ── Component rewards ─────────────────────────────────────────────────────
    r_progress   = _col(rows, "r_progress")
    r_speed      = _col(rows, "r_speed")
    r_gap_abs    = _col(rows, "r_gap_abs")
    r_smoothness = _col(rows, "r_smoothness")
    r_yaw        = _col(rows, "r_yaw")
    r_crash      = _col(rows, "r_crash")

    # ── Weighted contributions ────────────────────────────────────────────────
    c_progress   = _col(rows, "c_progress")
    c_speed      = _col(rows, "c_speed")
    c_gap_abs    = _col(rows, "c_gap_abs")
    c_smoothness = _col(rows, "c_smoothness")
    c_yaw        = _col(rows, "c_yaw")

    total_reward = _col(rows, "total_reward")
    cumulative   = np.cumsum(total_reward)

    # ── Car state ─────────────────────────────────────────────────────────────
    speed_ms = _col(rows, "speed_ms")
    gap_m    = _col(rows, "gap_m")
    steer    = _col(rows, "action_steer")
    throttle = _col(rows, "action_throttle")
    brake    = _col(rows, "action_brake")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Episode {episode}  (phase {phase})  —  {T} frames  |  "
        f"total = {total_reward.sum():.2f}  |  mean/step = {total_reward.mean():.3f}",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.42)

    colors = {
        "r_progress":   "#4CAF50",
        "r_speed":      "#2196F3",
        "r_gap_abs":    "#FF9800",
        "r_smoothness": "#9C27B0",
        "r_yaw":        "#00BCD4",
        "crash":        "#F44336",
    }

    x = np.arange(T)

    # ── Panel 1: raw component rewards ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.axhline(0, color="#555", linewidth=0.6, linestyle="--")
    ax1.plot(x, r_progress,   color=colors["r_progress"],   linewidth=0.9, label="r_progress")
    ax1.plot(x, r_speed,      color=colors["r_speed"],      linewidth=0.9, label="r_speed")
    ax1.plot(x, r_gap_abs,    color=colors["r_gap_abs"],    linewidth=0.9, label="r_gap_abs")
    ax1.plot(x, r_smoothness, color=colors["r_smoothness"], linewidth=0.9, label="r_smoothness")
    ax1.plot(x, r_yaw,        color=colors["r_yaw"],        linewidth=0.9, label="r_yaw")
    if r_crash.any():
        crash_frames = np.where(r_crash > 0)[0]
        ax1.vlines(crash_frames, -1, 1, color=colors["crash"], linewidth=1.2, label="crash", alpha=0.6)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_ylabel("component [-1,1]")
    ax1.set_title("Raw component rewards")
    ax1.legend(loc="upper right", fontsize=8, ncol=3)
    ax1.set_xlim(0, T - 1)

    # ── Panel 2: weighted contributions + total ───────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.axhline(0, color="#555", linewidth=0.6, linestyle="--")

    # Stacked positive contributions
    pos = np.zeros(T)
    neg = np.zeros(T)
    for name, vals, col in [
        ("c_progress",   c_progress,   colors["r_progress"]),
        ("c_speed",      c_speed,      colors["r_speed"]),
        ("c_gap_abs",    c_gap_abs,    colors["r_gap_abs"]),
        ("c_smoothness", c_smoothness, colors["r_smoothness"]),
        ("c_yaw",        c_yaw,        colors["r_yaw"]),
    ]:
        pos_part = np.where(vals >= 0, vals, 0)
        neg_part = np.where(vals < 0,  vals, 0)
        ax2.bar(x, pos_part, bottom=pos, color=col, alpha=0.65, label=name, width=1.0)
        ax2.bar(x, neg_part, bottom=neg, color=col, alpha=0.65, width=1.0)
        pos += pos_part
        neg += neg_part

    ax2.plot(x, total_reward, color="white", linewidth=1.2, label="total", zorder=5)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_ylabel("weighted contribution")
    ax2.set_title("Weighted contributions (stacked) + total reward")
    ax2.legend(loc="upper right", fontsize=7, ncol=3)
    ax2.set_xlim(0, T - 1)

    # ── Panel 3: total reward + cumulative ────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.axhline(0, color="#555", linewidth=0.6, linestyle="--")
    ax3.plot(x, total_reward, color="#E91E63", linewidth=0.9, alpha=0.8, label="total reward")
    ax3b = ax3.twinx()
    ax3b.plot(x, cumulative, color="#FF5722", linewidth=1.2, linestyle="-.", label="cumulative")
    ax3b.set_ylabel("cumulative", color="#FF5722")
    ax3.set_ylabel("reward [-1,1]")
    ax3.set_title("Total reward per frame + cumulative sum")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax3.set_xlim(0, T - 1)

    # ── Panel 4: car state ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(x, speed_ms,    color="#03A9F4", linewidth=0.9, label="speed (m/s)")
    ax4_r = ax4.twinx()
    ax4_r.plot(x, gap_m,     color="#FF9800", linewidth=0.9, linestyle="--", label="gap (m)")
    ax4_r.plot(x, steer,     color="#F44336", linewidth=0.7, linestyle="-",  label="steer", alpha=0.7)
    ax4_r.plot(x, throttle,  color="#4CAF50", linewidth=0.7, linestyle="-",  label="throttle", alpha=0.7)
    ax4_r.plot(x, brake,     color="#9C27B0", linewidth=0.7, linestyle="-",  label="brake", alpha=0.7)
    ax4.set_ylabel("speed (m/s)", color="#03A9F4")
    ax4_r.set_ylabel("gap / actions", color="#FF9800")
    ax4.set_title("Car state: speed, gap, actions")
    ax4.set_xlabel("frame")
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_r.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7, ncol=3)
    ax4.set_xlim(0, T - 1)

    # ── Background style ──────────────────────────────────────────────────────
    fig.patch.set_facecolor("#1A1A2E")
    for ax in [ax1, ax2, ax3, ax3b, ax4, ax4_r]:
        ax.set_facecolor("#16213E")
        ax.tick_params(colors="#CCCCCC", labelsize=8)
        ax.yaxis.label.set_color("#CCCCCC")
        ax.xaxis.label.set_color("#CCCCCC")
        ax.title.set_color("#EEEEEE")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    if show:
        plt.show()
        return None
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"ep_{episode}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved → {out_path}")
        return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot reward components from reward_truth.jsonl")
    parser.add_argument("--input",   "-i", type=Path, default=DEFAULT_LOG,
                        help=f"Path to reward_truth.jsonl (default: {DEFAULT_LOG})")
    parser.add_argument("--episode", "-e", type=int,  default=None,
                        help="Episode number to plot (default: whatever is in the file)")
    parser.add_argument("--show",    "-s", action="store_true",
                        help="Open interactive window instead of saving PNG")
    parser.add_argument("--out-dir", "-o", type=Path,
                        default=Path("D:/Git/virtual457-projects/AssetoCorsa/reward_log"),
                        help="Directory to save PNG (default: reward_log/)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Log not found: {args.input}")
        sys.exit(1)

    rows = load_episode(args.input, args.episode)
    ep   = rows[0].get("episode", 0)
    ph   = rows[0].get("phase", 0)
    print(f"Loaded {len(rows)} frames — episode={ep}  phase={ph}")

    plot(rows, show=args.show, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
