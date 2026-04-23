"""
q_on_aiclone.py — Diagnostic: what does the current critic think of the AI's actions?

For every frame in the AI-clone lap, compute:

    Q_ai(s, a_ai)        — critic value of the AI-driver action at state s
    Q_pi(s, a_pi)         — critic value of the current policy's (deterministic)
                            action at the same state
    Δ_pi = Q_pi - Q_ai    — "policy thinks it is X units BETTER than the AI"

If Δ_pi > 0 at corner X, the policy is over-confident (Q says "my way is better"
while reality = crash). That points at a Q-miscalibration.
If Δ_pi < 0 at corner X but the policy STILL does its own thing, the policy
is ignoring a correct critic — i.e. it got stuck in a local minimum of
the policy-improvement loss. That points at a policy-side problem.

Results
-------
1. ``outputs/q_on_aiclone_<ts>/frame_q.csv``
   One row per window: frame index, per-channel Q_ai/Q_pi, per-channel actions.
2. ``outputs/q_on_aiclone_<ts>/summary.json``
   Aggregate means / max-abs deltas per channel.
3. ``outputs/q_on_aiclone_<ts>/q_trace.png``
   Plot of Q_ai and Q_pi vs frame index (proxy for track position), 3 subplots
   (one per channel).
4. ``outputs/q_on_aiclone_<ts>/delta_trace.png``
   Plot of Q_pi - Q_ai per channel — peaks = where policy disagrees most with AI.

Run
---
    ./AssetoCorsa/Scripts/python.exe \\
        gym/transformer_sac_vectorq_v2_final_fineTune/q_on_aiclone.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

# ── Path bootstrap ───────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "assetto_corsa_gym"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "assetto_corsa_gym", "assetto_corsa_gym"))
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vectorq_v2_final.q_on_aiclone")

# ── Constants ────────────────────────────────────────────────────────────────
TOKEN_DIM   = 50
WINDOW_SIZE = 75
ACTION_DIM  = 3

TRANSFORMER_CONFIG = {
    "token_dim": TOKEN_DIM, "action_dim": ACTION_DIM, "window_size": WINDOW_SIZE,
    "d_model": 256, "n_heads": 4, "n_layers": 4, "ffn_dim": 1024,
    "policy_hidden": [256], "q_hidden": [256],
}
SAC_STATIC = {"gamma": 0.992, "tau": 0.005, "target_entropy": -2.0}

CHECKPOINT_DIR = os.path.join(_THIS_DIR, "checkpoints")
AICLONE_NPZ    = os.path.join(_THIS_DIR, "..", "..", "AICLONE", "data", "monza_miata", "aiclone_dataset.npz")
OUTPUTS_ROOT   = os.path.join(_THIS_DIR, "outputs")

CHANNEL_NAMES = ["steer", "throttle", "brake"]


def _resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def _build_windows(tokens: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows of length ``WINDOW_SIZE``.

    Returns
    -------
    windows   : (N, WINDOW_SIZE, TOKEN_DIM)
    actions_w : (N, ACTION_DIM) — the AI action recorded at the window's last frame
    frame_idx : (N,) — the original frame index of each window's last frame
    """
    T = len(tokens)
    W = WINDOW_SIZE
    N = T - W + 1
    assert N > 0, f"AI lap too short for window {W}: got {T} frames."

    windows = np.empty((N, W, TOKEN_DIM), dtype=np.float32)
    for i in range(N):
        windows[i] = tokens[i : i + W]
    actions_w = actions[W - 1 :].astype(np.float32)
    frame_idx = np.arange(W - 1, T, dtype=np.int64)
    return windows, actions_w, frame_idx


@torch.no_grad()
def _compute_q(sac, windows: np.ndarray, actions_ai: np.ndarray,
               batch_size: int, device: str) -> dict:
    """
    For each window compute Q_ai(s, a_ai), the policy's deterministic action a_pi,
    and Q_pi(s, a_pi).  All on GPU in batches.
    """
    sac.critic_encoder.eval()
    sac.policy_encoder.eval()
    sac.policy_head.eval()
    sac.twin_q.eval()

    N = len(windows)
    # Allocations for output arrays.
    Q_ai      = np.empty((N, ACTION_DIM), dtype=np.float32)  # min(q1, q2) using a_ai
    Q_pi      = np.empty((N, ACTION_DIM), dtype=np.float32)  # min(q1, q2) using a_pi
    A_pi_det  = np.empty((N, ACTION_DIM), dtype=np.float32)  # deterministic policy output

    x_tensor = torch.as_tensor(windows, dtype=torch.float32)
    a_ai_tensor = torch.as_tensor(actions_ai, dtype=torch.float32)

    n_batches = (N + batch_size - 1) // batch_size
    t0 = time.perf_counter()
    for bi in range(n_batches):
        lo, hi = bi * batch_size, min((bi + 1) * batch_size, N)
        xb = x_tensor[lo:hi].to(device).clamp(-3.0, 3.0)
        ab = a_ai_tensor[lo:hi].to(device)

        # Critic embedding (shared by both Q calls).
        emb_c = sac.critic_encoder(xb)                           # (B, d_model)
        # Policy embedding (for deterministic action lookup).
        emb_p = sac.policy_encoder(xb)                           # (B, d_model)

        # Deterministic policy action = tanh(mean_raw) from the head.
        head_out = sac.policy_head.net(emb_p)                    # (B, 2*A)
        mean_raw, _log_std = head_out.chunk(2, dim=-1)
        a_pi = torch.tanh(mean_raw)                              # (B, A)

        # Q values per-channel.
        q1_ai, q2_ai = sac.twin_q(emb_c, ab)                     # each (B, 3)
        q_ai_min = torch.min(q1_ai, q2_ai)

        q1_pi, q2_pi = sac.twin_q(emb_c, a_pi)                   # each (B, 3)
        q_pi_min = torch.min(q1_pi, q2_pi)

        Q_ai[lo:hi]     = q_ai_min.cpu().numpy()
        Q_pi[lo:hi]     = q_pi_min.cpu().numpy()
        A_pi_det[lo:hi] = a_pi.cpu().numpy()

        if (bi + 1) % 20 == 0 or bi == n_batches - 1:
            logger.info(
                f"  batch {bi + 1}/{n_batches}  "
                f"N_done={hi}/{N}  elapsed={time.perf_counter() - t0:.1f}s"
            )
    return {"Q_ai": Q_ai, "Q_pi": Q_pi, "A_pi": A_pi_det}


def _plot_traces(frame_idx, Q_ai, Q_pi, A_ai, A_pi, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Q trace per channel ────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for i, name in enumerate(CHANNEL_NAMES):
        axes[i].plot(frame_idx, Q_ai[:, i], label=f"Q(s, a_AI)",    color="tab:blue",   linewidth=0.8)
        axes[i].plot(frame_idx, Q_pi[:, i], label=f"Q(s, a_policy)", color="tab:orange", linewidth=0.8)
        axes[i].axhline(0, color="gray", linewidth=0.4, linestyle="--")
        axes[i].set_ylabel(f"Q_{name}")
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("AI lap frame index (proxy for track position)")
    fig.suptitle("Per-channel critic value at AI-lap states: AI action vs. current policy action", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "q_trace.png"), dpi=120)
    plt.close(fig)

    # ── Δ = Q_pi - Q_ai ────────────────────────────────────────────────────
    delta = Q_pi - Q_ai
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    for i, name in enumerate(CHANNEL_NAMES):
        ax.plot(frame_idx, delta[:, i], label=f"Δ_{name}", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("AI lap frame index")
    ax.set_ylabel("Q(s, π) − Q(s, AI)")
    ax.set_title("Policy self-valuation minus AI valuation (positive = policy thinks it is better than AI)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "delta_trace.png"), dpi=120)
    plt.close(fig)

    # ── Action comparison — AI vs policy ───────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for i, name in enumerate(CHANNEL_NAMES):
        axes[i].plot(frame_idx, A_ai[:, i], label="AI",     color="tab:blue",   linewidth=0.8)
        axes[i].plot(frame_idx, A_pi[:, i], label="policy", color="tab:orange", linewidth=0.8)
        axes[i].axhline(0, color="gray", linewidth=0.4, linestyle="--")
        axes[i].set_ylabel(f"a_{name}")
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("AI lap frame index")
    fig.suptitle("Action comparison: AI driver vs current policy (deterministic)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "action_trace.png"), dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "latest.pt"))
    parser.add_argument("--aiclone-npz", default=AICLONE_NPZ)
    parser.add_argument("--batch-size",  type=int, default=256)
    parser.add_argument("--device",      default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--save-csv",    action="store_true", default=True,
                        help="Write frame_q.csv (per-window Q values).")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUTS_ROOT, f"q_on_aiclone_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=" * 64)
    logger.info(f"q_on_aiclone  device={device}  out={out_dir}")
    logger.info("=" * 64)

    # 1) Load AI clone data.
    npz_path = os.path.abspath(args.aiclone_npz)
    assert os.path.isfile(npz_path), f"Missing AI clone NPZ: {npz_path}"
    d = np.load(npz_path)
    obs_all     = d["obs"].astype(np.float32)
    actions_all = d["actions"].astype(np.float32)
    tokens = obs_all[:, :TOKEN_DIM]
    T = len(tokens)
    logger.info(f"AI clone: {T} frames  tokens.shape={tokens.shape}  actions.shape={actions_all.shape}")

    # 2) Sliding windows.
    windows, actions_w, frame_idx = _build_windows(tokens, actions_all)
    logger.info(f"Built {len(windows)} windows (W={WINDOW_SIZE})")

    # 3) Load SAC checkpoint.
    from transformer_sac_vectorq_v2_final_fineTune.sac import TransformerSAC

    sac = TransformerSAC(**TRANSFORMER_CONFIG, lr=3e-4, **SAC_STATIC, device=device)
    assert os.path.isfile(args.checkpoint), f"No checkpoint at {args.checkpoint}"
    sac.load(args.checkpoint)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # 4) Compute Q values.
    logger.info("Computing Q values ...")
    result = _compute_q(sac, windows, actions_w, batch_size=args.batch_size, device=device)
    Q_ai, Q_pi, A_pi = result["Q_ai"], result["Q_pi"], result["A_pi"]
    A_ai = actions_w  # alias for naming clarity

    # 5) Aggregate stats.
    delta = Q_pi - Q_ai                                      # (N, 3)
    summary = {
        "checkpoint":   os.path.abspath(args.checkpoint),
        "n_windows":    int(len(windows)),
        "Q_ai": {
            name: {
                "mean": float(Q_ai[:, i].mean()),
                "std":  float(Q_ai[:, i].std()),
                "min":  float(Q_ai[:, i].min()),
                "max":  float(Q_ai[:, i].max()),
                "p10":  float(np.percentile(Q_ai[:, i], 10)),
                "p50":  float(np.percentile(Q_ai[:, i], 50)),
                "p90":  float(np.percentile(Q_ai[:, i], 90)),
            } for i, name in enumerate(CHANNEL_NAMES)
        },
        "Q_pi": {
            name: {
                "mean": float(Q_pi[:, i].mean()),
                "std":  float(Q_pi[:, i].std()),
                "min":  float(Q_pi[:, i].min()),
                "max":  float(Q_pi[:, i].max()),
                "p10":  float(np.percentile(Q_pi[:, i], 10)),
                "p50":  float(np.percentile(Q_pi[:, i], 50)),
                "p90":  float(np.percentile(Q_pi[:, i], 90)),
            } for i, name in enumerate(CHANNEL_NAMES)
        },
        "delta_Q_pi_minus_ai": {
            name: {
                "mean":     float(delta[:, i].mean()),
                "max_abs":  float(np.abs(delta[:, i]).max()),
                "max_pos":  float(delta[:, i].max()),
                "max_neg":  float(delta[:, i].min()),
                "argmax_pos_frame": int(frame_idx[int(np.argmax(delta[:, i]))]),
                "argmax_neg_frame": int(frame_idx[int(np.argmin(delta[:, i]))]),
                "frac_policy_better": float((delta[:, i] > 0).mean()),
            } for i, name in enumerate(CHANNEL_NAMES)
        },
        "action_diff_abs_mean": {
            name: float(np.abs(A_pi[:, i] - A_ai[:, i]).mean())
            for i, name in enumerate(CHANNEL_NAMES)
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 6) Per-frame CSV (heavy but useful for later exploration).
    if args.save_csv:
        csv_path = os.path.join(out_dir, "frame_q.csv")
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "frame_idx",
                "Q_ai_steer", "Q_ai_throttle", "Q_ai_brake",
                "Q_pi_steer", "Q_pi_throttle", "Q_pi_brake",
                "delta_steer", "delta_throttle", "delta_brake",
                "a_ai_steer", "a_ai_throttle", "a_ai_brake",
                "a_pi_steer", "a_pi_throttle", "a_pi_brake",
            ])
            for k in range(len(windows)):
                w.writerow([
                    int(frame_idx[k]),
                    *[f"{Q_ai[k, i]:.5f}"     for i in range(3)],
                    *[f"{Q_pi[k, i]:.5f}"     for i in range(3)],
                    *[f"{delta[k, i]:.5f}"    for i in range(3)],
                    *[f"{A_ai[k, i]:.5f}"     for i in range(3)],
                    *[f"{A_pi[k, i]:.5f}"     for i in range(3)],
                ])
        logger.info(f"Wrote {csv_path}")

    # 7) Plots.
    _plot_traces(frame_idx, Q_ai, Q_pi, A_ai, A_pi, out_dir)
    logger.info(f"Wrote plots to {out_dir}")

    # 8) Pretty print.
    logger.info("─" * 64)
    logger.info("Summary (per-channel means):")
    for name in CHANNEL_NAMES:
        qa = summary["Q_ai"][name]["mean"]
        qp = summary["Q_pi"][name]["mean"]
        dl = summary["delta_Q_pi_minus_ai"][name]
        ad = summary["action_diff_abs_mean"][name]
        logger.info(
            f"  {name:<9s}  Q_ai={qa:+7.3f}  Q_pi={qp:+7.3f}  "
            f"Δ_mean={dl['mean']:+7.3f}  Δ_maxAbs={dl['max_abs']:6.3f}  "
            f"π>AI frac={dl['frac_policy_better']:.2%}  "
            f"|a_π - a_AI|={ad:.3f}"
        )
    logger.info("─" * 64)
    logger.info(f"Artefacts: {out_dir}")


if __name__ == "__main__":
    main()
