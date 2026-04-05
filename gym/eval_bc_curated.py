"""
eval_bc_curated.py — Offline curated-sample inference check for a BC model.

Loads the preprocessed .npz (no AC connection needed), bins the steering
range from actual-min to actual-max into equal-width buckets, then samples
up to --samples-per-bin observations from each bucket and runs the model.

Prints a per-bin table:
    bin_lo | bin_hi | n_available | n_sampled | true_steer_mean | pred_steer_mean | MAE | bias

All values are in SAC space [-1, 1] (same convention as the npz actions).
No invert_steer is applied — model output is compared directly to the labels.

Usage
-----
    .\\AssetoCorsa\\Scripts\\python.exe gym/eval_bc_curated.py \\
        --checkpoint checkpoints/bc_monza_human_v1/best.pt \\
        --data human_data/processed/monza_ks_mazda_miata_mlp.npz \\
        --bins 10 --samples-per-bin 100

    # Or use data/processed/ if you want to evaluate on sim-recorded data:
    .\\AssetoCorsa\\Scripts\\python.exe gym/eval_bc_curated.py \\
        --checkpoint checkpoints/bc_monza_human_v1/best.pt \\
        --data data/processed/monza_ks_mazda_miata_mlp.npz
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
GYM_DIR   = REPO_ROOT / "gym"
DISCOR    = REPO_ROOT / "assetto_corsa_gym" / "algorithm" / "discor"

for p in [str(GYM_DIR), str(DISCOR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from policies.models import load_model  # noqa: E402


# ── Sampling ──────────────────────────────────────────────────────────────────

def curated_sample(actions: np.ndarray, n_bins: int, samples_per_bin: int, seed: int):
    """
    Bin steer column (actions[:, 0]) into n_bins equal-width bins from actual
    min to actual max, then sample up to samples_per_bin indices per bin.

    Returns
    -------
    list of dicts, one per bin:
        lo, hi        — bin edges in SAC steer space [-1, 1]
        indices       — np.ndarray of sampled row indices into the npz arrays
        n_available   — total rows in this bin before sampling
    """
    rng      = np.random.default_rng(seed)
    steer    = actions[:, 0]
    lo_edge  = float(steer.min())
    hi_edge  = float(steer.max())
    edges    = np.linspace(lo_edge, hi_edge, n_bins + 1)

    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # include right edge in last bin to avoid missing the max value
        if i < n_bins - 1:
            mask = (steer >= lo) & (steer < hi)
        else:
            mask = (steer >= lo) & (steer <= hi)

        candidate_idx = np.where(mask)[0]
        n_available   = len(candidate_idx)

        if n_available == 0:
            sampled_idx = np.array([], dtype=np.int64)
        elif n_available <= samples_per_bin:
            sampled_idx = candidate_idx.copy()
        else:
            sampled_idx = rng.choice(candidate_idx, size=samples_per_bin, replace=False)
            sampled_idx.sort()

        bins.append({
            "lo":          float(lo),
            "hi":          float(hi),
            "indices":     sampled_idx,
            "n_available": n_available,
        })

    return bins, lo_edge, hi_edge


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference_on_bins(model, obs: np.ndarray, bins: list, device):
    """
    Run model.get_action() on each sampled obs in each bin.
    Returns list of dicts with per-bin results.
    """
    results = []
    for bin_info in bins:
        idx = bin_info["indices"]
        if len(idx) == 0:
            results.append({
                "lo":            bin_info["lo"],
                "hi":            bin_info["hi"],
                "n_available":   bin_info["n_available"],
                "n_sampled":     0,
                "true_steer":    [],
                "pred_steer":    [],
            })
            continue

        obs_batch = torch.FloatTensor(obs[idx]).to(device)

        with torch.no_grad():
            actions_pred = model.get_action(obs_batch, deterministic=True)
            pred = actions_pred.cpu().numpy()   # (n, 3), SAC space [-1, 1]

        results.append({
            "lo":          bin_info["lo"],
            "hi":          bin_info["hi"],
            "n_available": bin_info["n_available"],
            "n_sampled":   len(idx),
            "true_steer":  list(map(float, obs[idx, 0])),   # placeholder — see NOTE below
            "pred_steer":  list(map(float, pred[:, 0])),
            "_indices":    idx.tolist(),
        })

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def build_report(results, actions_true: np.ndarray, bins: list):
    """
    Fill in true_steer from the actions array and compute summary stats.
    """
    rows = []
    for r, b in zip(results, bins):
        idx = b["indices"]
        if len(idx) == 0:
            rows.append({
                "bin_lo":         round(r["lo"], 4),
                "bin_hi":         round(r["hi"], 4),
                "n_available":    r["n_available"],
                "n_sampled":      0,
                "true_steer_mean": None,
                "pred_steer_mean": None,
                "mae":            None,
                "bias":           None,   # pred - true (positive = model steers more right)
            })
            continue

        true_steer = actions_true[idx, 0]                # SAC [-1, 1]
        pred_steer = np.array(r["pred_steer"])

        rows.append({
            "bin_lo":          round(r["lo"],  4),
            "bin_hi":          round(r["hi"],  4),
            "n_available":     r["n_available"],
            "n_sampled":       r["n_sampled"],
            "true_steer_mean": round(float(true_steer.mean()), 4),
            "pred_steer_mean": round(float(pred_steer.mean()),  4),
            "mae":             round(float(np.abs(pred_steer - true_steer).mean()), 4),
            "bias":            round(float((pred_steer - true_steer).mean()), 4),
        })

    return rows


def print_report(rows, lo_edge, hi_edge, checkpoint, npz_path):
    header = (
        f"{'bin_lo':>8} {'bin_hi':>8}  {'avail':>7}  {'sampled':>7}  "
        f"{'true_mean':>10} {'pred_mean':>10}  {'MAE':>7}  {'bias':>7}"
    )
    sep = "-" * len(header)

    print()
    print(f"Curated BC Evaluation — {Path(checkpoint).name}")
    print(f"Data:  {Path(npz_path).name}  |  steer range [{lo_edge:.4f}, {hi_edge:.4f}]")
    print(sep)
    print(header)
    print(sep)

    total_avail = total_sampled = 0
    all_true, all_pred = [], []

    for r in rows:
        if r["n_sampled"] == 0:
            print(f"  {r['bin_lo']:>6.4f}  {r['bin_hi']:>6.4f}  {'---':>7}  {'---':>7}  "
                  f"{'---':>10} {'---':>10}  {'---':>7}  {'---':>7}  [no samples in bin]")
            continue

        total_avail   += r["n_available"]
        total_sampled += r["n_sampled"]

        print(
            f"  {r['bin_lo']:>6.4f}  {r['bin_hi']:>6.4f}"
            f"  {r['n_available']:>7}  {r['n_sampled']:>7}"
            f"  {r['true_steer_mean']:>10.4f}  {r['pred_steer_mean']:>10.4f}"
            f"  {r['mae']:>7.4f}  {r['bias']:>+7.4f}"
        )

    print(sep)
    print()

    # Overall stats
    bias_vals = [r["bias"] for r in rows if r["bias"] is not None]
    mae_vals  = [r["mae"]  for r in rows if r["mae"]  is not None]
    if bias_vals:
        print(f"Overall: {total_sampled} samples across {sum(1 for r in rows if r['n_sampled']>0)} bins")
        print(f"  Mean MAE  : {np.mean(mae_vals):.4f}")
        print(f"  Mean bias : {np.mean(bias_vals):+.4f}  (positive = model biases RIGHT vs human)")
        print()

        # Straight-collapse check
        extreme_bins = [r for r in rows if abs(r.get("true_steer_mean", 0)) > 0.2]
        if extreme_bins:
            turning_bias = np.mean([r["bias"] for r in extreme_bins])
            turning_mae  = np.mean([r["mae"]  for r in extreme_bins])
            print(f"  Turning bins only (|true_mean| > 0.2):  MAE={turning_mae:.4f}  bias={turning_bias:+.4f}")

        # Warn if model looks collapsed to straight
        middle_bin_preds = [r["pred_steer_mean"] for r in rows if r["pred_steer_mean"] is not None]
        if middle_bin_preds:
            pred_std = np.std(middle_bin_preds)
            if pred_std < 0.05:
                print()
                print("  WARNING: pred_steer_mean std across bins = "
                      f"{pred_std:.4f} < 0.05 — model may have collapsed to straight!")
            else:
                print(f"  Pred steer spread (std across bin means): {pred_std:.4f}  [healthy if > 0.1]")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offline curated-sample inference check for a BC model.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/bc_monza_human_v1/best.pt",
        help="Path to model checkpoint (.pt). Default: bc_monza_human_v1/best.pt",
    )
    parser.add_argument(
        "--data", type=str,
        default="human_data/processed/monza_ks_mazda_miata_mlp.npz",
        help=(
            "Path to preprocessed .npz (preprocess_bc.py output).\n"
            "Default: human_data/processed/monza_ks_mazda_miata_mlp.npz\n"
            "Alt:     data/processed/monza_ks_mazda_miata_mlp.npz"
        ),
    )
    parser.add_argument(
        "--bins", type=int, default=10,
        help="Number of equal-width bins across the steer range. Default: 10",
    )
    parser.add_argument(
        "--samples-per-bin", type=int, default=100,
        help="Max samples to draw from each bin. Default: 100",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for bin sampling. Default: 42",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto|cuda|cpu. Default: auto",
    )
    parser.add_argument(
        "--save-json", type=str, default=None,
        metavar="PATH",
        help="If set, save per-bin results JSON to this path.",
    )
    args = parser.parse_args()

    # Resolve paths relative to repo root when run from any directory
    checkpoint = str(Path(args.checkpoint).resolve()) if Path(args.checkpoint).exists() \
                 else str(REPO_ROOT / args.checkpoint)
    npz_path   = str(Path(args.data).resolve()) if Path(args.data).exists() \
                 else str(REPO_ROOT / args.data)

    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not Path(npz_path).exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"Loading model from {Path(checkpoint).name} on {device_str} ...")
    model = load_model(model_type="mlp", checkpoint_path=checkpoint, device=device_str)
    model.eval()
    print(f"  obs_dim={model.obs_dim}  action_dim={model.action_dim}  "
          f"hidden={model.hidden_units}")

    # ── 2. Load data ──────────────────────────────────────────────────────────
    print(f"Loading data from {Path(npz_path).name} ...")
    data    = np.load(npz_path)
    obs     = data["obs"]       # (N, 125)
    actions = data["actions"]   # (N, 3)  SAC space [-1, 1]
    print(f"  Total steps: {len(obs)}")
    print(f"  Steer range: [{actions[:, 0].min():.4f}, {actions[:, 0].max():.4f}]")
    print(f"  Steer mean:  {actions[:, 0].mean():.4f}  std: {actions[:, 0].std():.4f}")

    steer_counts = {
        "left  (< -0.10)":    int((actions[:, 0] < -0.10).sum()),
        "straight [-0.10,0.10]": int((np.abs(actions[:, 0]) <= 0.10).sum()),
        "right (>  0.10)":    int((actions[:, 0] > 0.10).sum()),
    }
    total = len(actions)
    print("  Class distribution (true labels):")
    for label, count in steer_counts.items():
        print(f"    {label}: {count:6d}  ({100*count/total:.1f}%)")
    print()

    # ── 3. Curated sampling ───────────────────────────────────────────────────
    print(f"Binning into {args.bins} equal-width bins, "
          f"up to {args.samples_per_bin} samples each ...")
    bins_info, lo_edge, hi_edge = curated_sample(
        actions, args.bins, args.samples_per_bin, args.seed
    )
    print(f"  Steer range covered: [{lo_edge:.4f}, {hi_edge:.4f}]")
    print()

    # ── 4. Run model ──────────────────────────────────────────────────────────
    print("Running model forward pass on each bin ...")
    results = run_inference_on_bins(model, obs, bins_info, device)

    # ── 5. Build and print report ─────────────────────────────────────────────
    rows = build_report(results, actions, bins_info)
    print_report(rows, lo_edge, hi_edge, checkpoint, npz_path)

    # ── 6. Optional JSON save ─────────────────────────────────────────────────
    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "checkpoint":       checkpoint,
            "npz":              npz_path,
            "n_bins":           args.bins,
            "samples_per_bin":  args.samples_per_bin,
            "steer_range":      [lo_edge, hi_edge],
            "bins":             rows,
        }
        with open(out_path, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
