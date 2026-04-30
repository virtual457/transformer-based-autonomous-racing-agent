"""
eval_bc_precision_recall.py — Offline steering precision/recall evaluation.

Evaluates a behaviour-cloning model by discretising continuous steer values
into three classes (left / straight / right) and computing per-class and
macro-averaged precision, recall, and F1. Accuracy is intentionally NOT the
headline metric because the straight-heavy class distribution makes it
trivially inflatable.

Usage (from project root):
    .\\AssetoCorsa\\Scripts\\python.exe eval\\eval_bc_precision_recall.py
"""

# ---------------------------------------------------------------------------
# Path setup — must happen before any gym imports
# ---------------------------------------------------------------------------
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent   # D:/Git/virtual457-projects/AssetoCorsa
GYM_DIR   = REPO_ROOT / "gym"
DISCOR    = REPO_ROOT / "assetto_corsa_gym" / "algorithm" / "discor"

import sys
sys.path.insert(0, str(GYM_DIR))
sys.path.insert(0, str(DISCOR))

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import json
import warnings

import numpy as np
import torch

from policies.models import load_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "bc_monza_human_v1" / "best.pt"
DATA_PATH       = REPO_ROOT / "human_data" / "processed" / "monza_ks_mazda_miata_mlp.npz"
REPORT_PATH     = REPO_ROOT / "results" / "eval_reports" / "bc_monza_human_v1.json"
CONFMAT_PATH    = REPO_ROOT / "results" / "confusion_matrices" / "bc_monza_human_v1.png"

# Ensure output directories exist
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
CONFMAT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE      = 1024
STEER_THRESHOLD = 0.10          # boundary between straight and turning
CLASS_NAMES     = ["left", "straight", "right"]
MODEL_NAME      = "bc_monza_human_v1"

# ---------------------------------------------------------------------------
# Discretisation helper
# ---------------------------------------------------------------------------

def discretize_steer(steer: np.ndarray, threshold: float = STEER_THRESHOLD) -> np.ndarray:
    """Convert continuous steer values to integer class labels.

    Classes:
        0 — left     (steer < -threshold)
        1 — straight (-threshold <= steer <= threshold)
        2 — right    (steer > threshold)

    Parameters
    ----------
    steer : np.ndarray, shape (N,)
    threshold : float

    Returns
    -------
    np.ndarray of int, shape (N,)
    """
    labels = np.ones(len(steer), dtype=np.int64)   # default: straight
    labels[steer < -threshold] = 0                  # left
    labels[steer >  threshold] = 2                  # right
    return labels


# ---------------------------------------------------------------------------
# Naive-baseline macro-F1 (always predict straight)
# ---------------------------------------------------------------------------

def naive_baseline_macro_f1(true_labels: np.ndarray) -> float:
    """Macro-F1 achieved by a model that always predicts class 1 (straight)."""
    try:
        from sklearn.metrics import f1_score
        dummy = np.ones_like(true_labels)
        return float(f1_score(true_labels, dummy, average="macro", zero_division=0))
    except ImportError:
        # Manual fallback — only "straight" gets any predictions so left/right
        # have precision=0, recall=0, F1=0.  Straight recall=1 but precision
        # equals straight_support / N.
        n_straight = int(np.sum(true_labels == 1))
        n_total    = len(true_labels)
        prec_s = n_straight / n_total if n_total > 0 else 0.0
        rec_s  = 1.0
        f1_s   = 2 * prec_s * rec_s / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0.0
        return float(f1_s / 3)   # left and right contribute 0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print(f"Evaluating: {MODEL_NAME}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    print(f"\nLoading data from:\n  {DATA_PATH}")
    data = np.load(DATA_PATH)
    obs     = data["obs"].astype(np.float32)     # (N, 125)
    actions = data["actions"].astype(np.float32)  # (N, 3)
    N = len(obs)
    print(f"  Samples : {N:,}")
    print(f"  obs     : {obs.shape}")
    print(f"  actions : {actions.shape}")

    true_steer = actions[:, 0]   # SAC space [-1, 1]

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    print(f"\nLoading model from:\n  {CHECKPOINT_PATH}")
    model = load_model(
        model_type="mlp",
        checkpoint_path=str(CHECKPOINT_PATH),
        device="auto",
    )
    model.eval()

    # Determine device from model internals
    device = next(model.parameters()).device
    print(f"  Device  : {device}")

    # ------------------------------------------------------------------
    # 3. Batch inference
    # ------------------------------------------------------------------
    print(f"\nRunning inference (batch_size={BATCH_SIZE}) ...")
    all_preds = []
    for i in range(0, N, BATCH_SIZE):
        batch = torch.FloatTensor(obs[i : i + BATCH_SIZE]).to(device)
        with torch.no_grad():
            pred = model.get_action(batch, deterministic=True)  # (n, 3)
        all_preds.append(pred.cpu().numpy())

    pred_actions = np.concatenate(all_preds, axis=0)   # (N, 3)
    pred_steer   = pred_actions[:, 0]                   # SAC space [-1, 1]
    print(f"  Predictions shape: {pred_actions.shape}")

    # ------------------------------------------------------------------
    # 4. Discretise
    # ------------------------------------------------------------------
    true_labels = discretize_steer(true_steer)
    pred_labels = discretize_steer(pred_steer)

    true_counts = {c: int(np.sum(true_labels == i)) for i, c in enumerate(CLASS_NAMES)}
    pred_counts = {c: int(np.sum(pred_labels == i)) for i, c in enumerate(CLASS_NAMES)}
    print(f"\n  True class distribution : {true_counts}")
    print(f"  Pred class distribution : {pred_counts}")

    # ------------------------------------------------------------------
    # 5. Precision / recall / F1
    # ------------------------------------------------------------------
    try:
        from sklearn.metrics import (
            precision_recall_fscore_support,
            confusion_matrix,
        )
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels,
            labels=[0, 1, 2],
            zero_division=0,
        )
        conf_mat_raw = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])

    except ImportError:
        warnings.warn(
            "scikit-learn not found — falling back to manual numpy computation.",
            ImportWarning,
        )
        precision = np.zeros(3)
        recall    = np.zeros(3)
        f1        = np.zeros(3)
        support   = np.zeros(3, dtype=int)
        conf_mat_raw = np.zeros((3, 3), dtype=int)

        for i in range(3):
            tp = int(np.sum((pred_labels == i) & (true_labels == i)))
            fp = int(np.sum((pred_labels == i) & (true_labels != i)))
            fn = int(np.sum((pred_labels != i) & (true_labels == i)))
            support[i] = tp + fn
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall[i]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = precision[i] + recall[i]
            f1[i]        = 2 * precision[i] * recall[i] / denom if denom > 0 else 0.0

        for i in range(3):
            for j in range(3):
                conf_mat_raw[i, j] = int(
                    np.sum((true_labels == i) & (pred_labels == j))
                )

    macro_f1 = float(np.mean(f1))

    # ------------------------------------------------------------------
    # 6. Derived metrics
    # ------------------------------------------------------------------
    # Naive baseline
    naive_f1 = naive_baseline_macro_f1(true_labels)

    # Degenerate flag
    straight_recall = float(recall[1])
    left_recall     = float(recall[0])
    right_recall    = float(recall[2])
    is_degenerate   = (
        straight_recall > 0.95
        and left_recall  < 0.30
        and right_recall < 0.30
    )

    # MAE — all steers
    mae_all = float(np.mean(np.abs(true_steer - pred_steer)))

    # MAE — turning only (|true_steer| > threshold)
    turning_mask = np.abs(true_steer) > STEER_THRESHOLD
    if turning_mask.sum() > 0:
        mae_turning = float(np.mean(np.abs(true_steer[turning_mask] - pred_steer[turning_mask])))
    else:
        mae_turning = float("nan")

    # Action entropy — histogram of predicted steer (20 bins)
    hist_counts, hist_edges = np.histogram(pred_steer, bins=20, range=(-1.0, 1.0))
    # Shannon entropy (nats) of the empirical distribution
    hist_prob = hist_counts / hist_counts.sum()
    nonzero   = hist_prob > 0
    action_entropy = float(-np.sum(hist_prob[nonzero] * np.log(hist_prob[nonzero])))

    # Normalised confusion matrix (row = true, col = pred; each row sums to 1)
    row_sums = conf_mat_raw.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0   # avoid division by zero for unseen classes
    conf_mat_norm = (conf_mat_raw / row_sums).tolist()

    # ------------------------------------------------------------------
    # 7. Print report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 54)
    for i, cls in enumerate(CLASS_NAMES):
        print(
            f"{cls:<12} {precision[i]:>10.4f} {recall[i]:>10.4f}"
            f" {f1[i]:>10.4f} {int(support[i]):>10,}"
        )
    print("-" * 54)
    print(f"{'Macro avg':<12} {'':>10} {'':>10} {macro_f1:>10.4f}")

    print(f"\n  Macro-F1             : {macro_f1:.4f}")
    print(f"  Naive baseline F1   : {naive_f1:.4f}  (always predict straight)")
    print(f"  Degenerate model    : {is_degenerate}")
    print(f"  MAE (all steer)     : {mae_all:.6f}")
    print(f"  MAE (turning only)  : {mae_turning:.6f}")
    print(f"  Action entropy      : {action_entropy:.4f} nats")

    print("\nConfusion matrix (normalised by true label):")
    header = f"{'':>12}" + "".join(f"{c:>12}" for c in CLASS_NAMES)
    print(header)
    for i, cls in enumerate(CLASS_NAMES):
        row = f"{cls:>12}" + "".join(f"{conf_mat_norm[i][j]:>12.4f}" for j in range(3))
        print(row)

    print("\nPredicted steer histogram (20 bins, -1 to +1):")
    bin_labels = [f"[{hist_edges[b]:.2f},{hist_edges[b+1]:.2f})" for b in range(20)]
    max_count  = max(hist_counts) if max(hist_counts) > 0 else 1
    bar_width  = 40
    for b in range(20):
        bar  = "#" * int(hist_counts[b] / max_count * bar_width)
        print(f"  {bin_labels[b]:>16}  {hist_counts[b]:>7,}  {bar}")

    # ------------------------------------------------------------------
    # 8. Save JSON report
    # ------------------------------------------------------------------
    report = {
        "model":          MODEL_NAME,
        "checkpoint":     str(CHECKPOINT_PATH),
        "data":           str(DATA_PATH),
        "n_samples":      N,
        "steer_threshold": STEER_THRESHOLD,
        "per_class": {
            cls: {
                "precision": float(precision[i]),
                "recall":    float(recall[i]),
                "f1":        float(f1[i]),
                "support":   int(support[i]),
            }
            for i, cls in enumerate(CLASS_NAMES)
        },
        "macro_f1":             macro_f1,
        "naive_baseline_macro_f1": naive_f1,
        "is_degenerate":        is_degenerate,
        "mae_all_steer":        mae_all,
        "mae_turning_steer":    mae_turning,
        "action_entropy_nats":  action_entropy,
        "confusion_matrix_norm": conf_mat_norm,
        "pred_steer_histogram": {
            "bin_edges":  hist_edges.tolist(),
            "counts":     hist_counts.tolist(),
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved to:\n  {REPORT_PATH}")

    # ------------------------------------------------------------------
    # 9. Confusion matrix plot
    # ------------------------------------------------------------------
    _save_confusion_matrix_plot(conf_mat_norm)

    print("\nDone.")


def _save_confusion_matrix_plot(conf_mat_norm: list) -> None:
    """Render and save the normalised confusion matrix as a PNG."""
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — safe on headless machines
    import matplotlib.pyplot as plt

    mat = np.array(conf_mat_norm)

    fig, ax = plt.subplots(figsize=(6, 5))

    try:
        import seaborn as sns
        sns.heatmap(
            mat,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            linewidths=0.5,
            ax=ax,
        )
    except ImportError:
        # Fallback: plain matplotlib imshow
        im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        for i in range(3):
            for j in range(3):
                ax.text(
                    j, i, f"{mat[i, j]:.3f}",
                    ha="center", va="center",
                    color="black" if mat[i, j] < 0.6 else "white",
                )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Steering Confusion Matrix — {MODEL_NAME} (best.pt)")

    plt.tight_layout()
    CONFMAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(CONFMAT_PATH), dpi=150)
    plt.close(fig)
    print(f"Confusion matrix plot saved to:\n  {CONFMAT_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
