"""
plot_q_function.py — Visualise the Q-function of the trained Vector Q v2 model.

Loads the checkpoint and 6-channel memmap buffer, samples windows,
runs them through the critic to produce per-channel Q-values, and
generates:
    1. Interactive HTML dashboard (Plotly)
    2. Static PNG images (matplotlib) for offline viewing

Run:
    .\\AssetoCorsa\\Scripts\\python.exe gym/transformer_sac_vectorq_v2_final_fineTune/plot_q_function.py
"""

import os
import sys
import logging

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("plot_q_function")

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join(_THIS_DIR, "checkpoints", "latest.pt")
BUFFER_DIR      = os.path.join(_THIS_DIR, "checkpoints", "buffers")
SAMPLES_PER_BUFFER = 2000
OUTPUT_DIR      = os.path.join(_THIS_DIR, "outputs", "q_plots")
OUTPUT_HTML     = os.path.join(_THIS_DIR, "outputs", "q_function_dashboard.html")

TOKEN_DIM   = 50
WINDOW_SIZE = 75
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

CH_NAMES  = ["Steer", "Throttle", "Brake"]
CH_COLORS_MPL = ["#F44336", "#4CAF50", "#9C27B0"]
POS_COLOR = "#4CAF50"
NEG_COLOR = "#F44336"

# matplotlib dark style
plt.rcParams.update({
    "figure.facecolor": "#1A1A2E",
    "axes.facecolor": "#16213E",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#CCC",
    "xtick.color": "#AAA",
    "ytick.color": "#AAA",
    "text.color": "#EEE",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
    "legend.facecolor": "#16213E",
    "legend.edgecolor": "#444",
})


def load_model():
    from transformer_sac_vectorq_v2_final_fineTune.sac import TransformerSAC
    logger.info(f"Loading model from {CHECKPOINT_PATH}")
    sac = TransformerSAC.from_checkpoint(CHECKPOINT_PATH, device=DEVICE)
    sac.critic_encoder.eval()
    sac.policy_encoder.eval()
    sac.policy_head.eval()
    sac.twin_q.eval()
    logger.info("Model loaded and set to eval mode.")
    return sac


def load_buffer():
    from transformer_sac_vectorq_v2_final_fineTune.replay_buffer import SixChannelMemmapBuffer
    buf_dir = os.path.abspath(BUFFER_DIR)
    logger.info(f"Loading buffer from {buf_dir}")
    buffer = SixChannelMemmapBuffer(
        base_dir=buf_dir,
        capacity_per_buffer=100_000,
        token_dim=TOKEN_DIM,
        action_dim=3,
        window_size=WINDOW_SIZE,
        reward_dim=3,
    )
    sizes = buffer.sizes()
    logger.info(f"Buffer sizes: {sizes}  total: {sum(sizes.values())}")
    return buffer


def compute_q_values(sac, obs_seq_batch):
    with torch.no_grad():
        obs = obs_seq_batch.clamp(-3.0, 3.0).to(DEVICE)
        policy_emb = sac.policy_encoder(obs)
        action, log_prob, mean_action = sac.policy_head(policy_emb)
        critic_emb = sac.critic_encoder(obs)
        q1, q2 = sac.twin_q(critic_emb, action)
        min_q = torch.min(q1, q2)
    return {
        "action":      action.cpu().numpy(),
        "q1":          q1.cpu().numpy(),
        "q2":          q2.cpu().numpy(),
        "min_q":       min_q.cpu().numpy(),
        "log_prob":    log_prob.cpu().numpy(),
        "mean_action": mean_action.cpu().numpy(),
    }


def sample_from_buffer(buffer, buffer_name, n_samples):
    buf = buffer.buffers[buffer_name]
    n = min(n_samples, len(buf))
    if n == 0:
        return None
    batch = buf.sample(n, device="cpu")
    return {k: v.numpy() for k, v in batch.items()}


def collect_data(sac, buffer):
    """Sample from all 6 buffers and compute Q-values. Returns all_data dict."""
    all_data = {}
    for name in buffer.CHANNEL_NAMES:
        batch = sample_from_buffer(buffer, name, SAMPLES_PER_BUFFER)
        if batch is None:
            logger.warning(f"  {name}: empty, skipping")
            continue
        q_vals = compute_q_values(
            sac, torch.as_tensor(batch["obs_seq"], dtype=torch.float32)
        )
        all_data[name] = {"batch": batch, "q": q_vals}
        logger.info(
            f"  {name}: sampled {len(batch['obs_seq'])}  "
            f"min_q mean=[{q_vals['min_q'][:,0].mean():.3f}, "
            f"{q_vals['min_q'][:,1].mean():.3f}, "
            f"{q_vals['min_q'][:,2].mean():.3f}]"
        )
    return all_data


# ── PNG plots (matplotlib) ────────────────────────────────────────────────────

def save_png_plots(all_data, output_dir):
    """Generate and save all plots as PNG files."""
    os.makedirs(output_dir, exist_ok=True)
    channel_pairs = [
        ("steer_pos", "steer_neg"),
        ("throttle_pos", "throttle_neg"),
        ("brake_pos", "brake_neg"),
    ]

    # 1. Q-value distribution per channel
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Q-Value Distribution per Channel", fontsize=16)
    for i, (ch_name, color) in enumerate(zip(CH_NAMES, CH_COLORS_MPL)):
        all_q = []
        for name, data in all_data.items():
            all_q.append(data["q"]["min_q"][:, i])
        if all_q:
            combined = np.concatenate(all_q)
            axes[i].hist(combined, bins=80, color=color, alpha=0.8, edgecolor="none")
            axes[i].set_title(ch_name, fontsize=13)
            axes[i].set_xlabel("Q-value")
            axes[i].set_ylabel("count")
            axes[i].axvline(x=0, color="#888", linewidth=0.8, linestyle="--")
            axes[i].grid(True)
    plt.tight_layout()
    path = os.path.join(output_dir, "1_q_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {path}")

    # 2. Q vs actual reward (scatter)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Q-Value vs Actual Reward", fontsize=16)
    for i, (ch_name, color) in enumerate(zip(CH_NAMES, CH_COLORS_MPL)):
        all_q = []
        all_r = []
        for name, data in all_data.items():
            all_q.append(data["q"]["min_q"][:, i])
            all_r.append(data["batch"]["reward"][:, i])
        if all_q:
            q = np.concatenate(all_q)
            r = np.concatenate(all_r)
            n = len(q)
            idx = np.random.choice(n, size=min(3000, n), replace=False)
            axes[i].scatter(r[idx], q[idx], c=color, s=4, alpha=0.3, edgecolors="none")
            axes[i].set_title(ch_name, fontsize=13)
            axes[i].set_xlabel("actual reward [-1, 1]")
            axes[i].set_ylabel("Q-value")
            axes[i].axhline(y=0, color="#888", linewidth=0.8, linestyle="--")
            axes[i].axvline(x=0, color="#888", linewidth=0.8, linestyle="--")
            axes[i].grid(True)
    plt.tight_layout()
    path = os.path.join(output_dir, "2_q_vs_reward.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {path}")

    # 3. Q positive vs negative buffer (box plot)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Q-Value: Positive vs Negative Buffer", fontsize=16)
    for i, (pos_name, neg_name) in enumerate(channel_pairs):
        ch_name = CH_NAMES[i]
        box_data = []
        box_labels = []
        box_colors = []
        if pos_name in all_data:
            box_data.append(all_data[pos_name]["q"]["min_q"][:, i])
            box_labels.append(f"{ch_name} +")
            box_colors.append(POS_COLOR)
        if neg_name in all_data:
            box_data.append(all_data[neg_name]["q"]["min_q"][:, i])
            box_labels.append(f"{ch_name} -")
            box_colors.append(NEG_COLOR)
        if box_data:
            bp = axes[i].boxplot(box_data, labels=box_labels, patch_artist=True,
                                 showmeans=True, meanline=True)
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            axes[i].set_title(ch_name, fontsize=13)
            axes[i].set_ylabel("Q-value")
            axes[i].axhline(y=0, color="#888", linewidth=0.8, linestyle="--")
            axes[i].grid(True)
    plt.tight_layout()
    path = os.path.join(output_dir, "3_q_pos_vs_neg.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {path}")

    # 4. Per-channel Q comparison (overlay histogram)
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Per-Channel Q Comparison (overlay)", fontsize=16)
    for i, (ch_name, color) in enumerate(zip(CH_NAMES, CH_COLORS_MPL)):
        all_q = []
        for name, data in all_data.items():
            all_q.append(data["q"]["min_q"][:, i])
        if all_q:
            combined = np.concatenate(all_q)
            ax.hist(combined, bins=80, color=color, alpha=0.45, label=ch_name, edgecolor="none")
    ax.set_xlabel("Q-value")
    ax.set_ylabel("count")
    ax.legend()
    ax.axvline(x=0, color="#888", linewidth=0.8, linestyle="--")
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(output_dir, "4_q_channel_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {path}")

    # 5. Q vs speed feature (scatter)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Q-Value vs Speed Feature (obs[0])", fontsize=16)
    for i, (ch_name, color) in enumerate(zip(CH_NAMES, CH_COLORS_MPL)):
        all_q = []
        all_speed = []
        for name, data in all_data.items():
            all_q.append(data["q"]["min_q"][:, i])
            last_token = data["batch"]["obs_seq"][:, -1, :]
            all_speed.append(last_token[:, 0])
        if all_q:
            q = np.concatenate(all_q)
            speed = np.concatenate(all_speed)
            n = len(q)
            idx = np.random.choice(n, size=min(3000, n), replace=False)
            axes[i].scatter(speed[idx], q[idx], c=color, s=4, alpha=0.3, edgecolors="none")
            axes[i].set_title(ch_name, fontsize=13)
            axes[i].set_xlabel("speed feature (obs[0])")
            axes[i].set_ylabel("Q-value")
            axes[i].axhline(y=0, color="#888", linewidth=0.8, linestyle="--")
            axes[i].grid(True)
    plt.tight_layout()
    path = os.path.join(output_dir, "5_q_vs_speed.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {path}")

    # 6. Summary stats as image
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis("off")
    fig.suptitle("Summary Statistics", fontsize=16, y=0.98)
    headers = ["Buffer", "N", "Q_s mean", "Q_t mean", "Q_b mean",
               "Q_s std", "Q_t std", "Q_b std",
               "R_s mean", "R_t mean", "R_b mean"]
    rows = []
    for name in ["steer_pos", "steer_neg", "throttle_pos", "throttle_neg",
                  "brake_pos", "brake_neg"]:
        if name not in all_data:
            continue
        d = all_data[name]
        q = d["q"]["min_q"]
        r = d["batch"]["reward"]
        rows.append([
            name, str(len(q)),
            f"{q[:,0].mean():.2f}", f"{q[:,1].mean():.2f}", f"{q[:,2].mean():.2f}",
            f"{q[:,0].std():.2f}", f"{q[:,1].std():.2f}", f"{q[:,2].std():.2f}",
            f"{r[:,0].mean():.3f}", f"{r[:,1].mean():.3f}", f"{r[:,2].mean():.3f}",
        ])
    if rows:
        colors = []
        for row in rows:
            if "pos" in row[0]:
                colors.append(["#1a3a1a"] * len(headers))
            else:
                colors.append(["#3a1a1a"] * len(headers))
        table = ax.table(cellText=rows, colLabels=headers, loc="center",
                         cellColours=colors,
                         colColours=["#2a2a4a"] * len(headers))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        for key, cell in table.get_celld().items():
            cell.set_edgecolor("#444")
            cell.set_text_props(color="#EEE")
    plt.tight_layout()
    path = os.path.join(output_dir, "6_summary_stats.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


# ── HTML dashboard (Plotly) ───────────────────────────────────────────────────

def save_html_dashboard(all_data):
    """Generate interactive HTML dashboard with Plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _BG    = "#1A1A2E"
    _PAPER = "#16213E"
    _GRID  = "#2a2a4a"

    def _base(title, height=400):
        return dict(
            title=dict(text=title, font=dict(color="#EEEEEE", size=14)),
            paper_bgcolor=_BG, plot_bgcolor=_PAPER,
            font=dict(color="#CCCCCC"),
            hovermode="closest", height=height,
            legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#444"),
        )

    channel_pairs = [
        ("steer_pos", "steer_neg"),
        ("throttle_pos", "throttle_neg"),
        ("brake_pos", "brake_neg"),
    ]

    charts = []

    # 1. Q-value distribution
    fig = make_subplots(rows=1, cols=3, subplot_titles=CH_NAMES)
    for i, (ch_name, color) in enumerate(zip(CH_NAMES, CH_COLORS_MPL)):
        all_q = []
        for name, data in all_data.items():
            all_q.append(data["q"]["min_q"][:, i])
        if all_q:
            combined = np.concatenate(all_q)
            fig.add_trace(go.Histogram(x=combined.tolist(), nbinsx=80, name=ch_name,
                                       marker_color=color, opacity=0.8), row=1, col=i+1)
            fig.update_xaxes(title_text="Q-value", row=1, col=i+1, gridcolor=_GRID)
            fig.update_yaxes(title_text="count", row=1, col=i+1, gridcolor=_GRID)
    fig.update_layout(**_base("Q-Value Distribution per Channel", 400))
    charts.append(("Q-Value Distribution", fig))

    # 2. Q vs reward
    fig = make_subplots(rows=1, cols=3, subplot_titles=CH_NAMES)
    for i, (ch_name, color) in enumerate(zip(CH_NAMES, CH_COLORS_MPL)):
        all_q, all_r = [], []
        for name, data in all_data.items():
            all_q.append(data["q"]["min_q"][:, i])
            all_r.append(data["batch"]["reward"][:, i])
        if all_q:
            q = np.concatenate(all_q)
            r = np.concatenate(all_r)
            n = len(q)
            idx = np.random.choice(n, size=min(3000, n), replace=False)
            fig.add_trace(go.Scatter(x=r[idx].tolist(), y=q[idx].tolist(), mode="markers",
                                     name=ch_name, marker=dict(color=color, size=3, opacity=0.4)),
                          row=1, col=i+1)
            fig.update_xaxes(title_text="actual reward", row=1, col=i+1, gridcolor=_GRID)
            fig.update_yaxes(title_text="Q-value", row=1, col=i+1, gridcolor=_GRID)
    fig.update_layout(**_base("Q-Value vs Actual Reward", 400))
    charts.append(("Q vs Reward", fig))

    # 3. Q pos vs neg
    fig = make_subplots(rows=1, cols=3, subplot_titles=CH_NAMES)
    for i, (pos_name, neg_name) in enumerate(channel_pairs):
        ch_name = CH_NAMES[i]
        if pos_name in all_data:
            q_pos = all_data[pos_name]["q"]["min_q"][:, i]
            fig.add_trace(go.Box(y=q_pos.tolist(), name=f"{ch_name} +",
                                  marker_color=POS_COLOR, boxmean=True), row=1, col=i+1)
        if neg_name in all_data:
            q_neg = all_data[neg_name]["q"]["min_q"][:, i]
            fig.add_trace(go.Box(y=q_neg.tolist(), name=f"{ch_name} -",
                                  marker_color=NEG_COLOR, boxmean=True), row=1, col=i+1)
        fig.update_yaxes(title_text="Q-value", row=1, col=i+1, gridcolor=_GRID)
    fig.update_layout(**_base("Q-Value: Positive vs Negative Buffer", 450))
    charts.append(("Q Pos vs Neg", fig))

    # 4. Channel overlay
    fig = go.Figure()
    for i, (ch_name, color) in enumerate(zip(CH_NAMES, CH_COLORS_MPL)):
        all_q = []
        for name, data in all_data.items():
            all_q.append(data["q"]["min_q"][:, i])
        if all_q:
            combined = np.concatenate(all_q)
            fig.add_trace(go.Histogram(x=combined.tolist(), nbinsx=80, name=ch_name,
                                       marker_color=color, opacity=0.5))
    fig.update_layout(**_base("Per-Channel Q Comparison (overlay)", 400),
                      barmode="overlay",
                      xaxis=dict(title="Q-value", gridcolor=_GRID),
                      yaxis=dict(title="count", gridcolor=_GRID))
    charts.append(("Q Channel Comparison", fig))

    # 5. Q vs speed
    fig = make_subplots(rows=1, cols=3, subplot_titles=CH_NAMES)
    for i, (ch_name, color) in enumerate(zip(CH_NAMES, CH_COLORS_MPL)):
        all_q, all_speed = [], []
        for name, data in all_data.items():
            all_q.append(data["q"]["min_q"][:, i])
            all_speed.append(data["batch"]["obs_seq"][:, -1, 0])
        if all_q:
            q = np.concatenate(all_q)
            speed = np.concatenate(all_speed)
            n = len(q)
            idx = np.random.choice(n, size=min(3000, n), replace=False)
            fig.add_trace(go.Scatter(x=speed[idx].tolist(), y=q[idx].tolist(), mode="markers",
                                     name=ch_name, marker=dict(color=color, size=3, opacity=0.4)),
                          row=1, col=i+1)
            fig.update_xaxes(title_text="speed feature (obs[0])", row=1, col=i+1, gridcolor=_GRID)
            fig.update_yaxes(title_text="Q-value", row=1, col=i+1, gridcolor=_GRID)
    fig.update_layout(**_base("Q-Value vs Speed Feature", 400))
    charts.append(("Q vs Speed", fig))

    # Build HTML
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)

    checkboxes = ""
    chart_htmls = ""
    for idx, (label, fig) in enumerate(charts):
        cid = f"chart_{idx}"
        checkboxes += (
            f'<label style="margin-right:16px;cursor:pointer;">'
            f'<input type="checkbox" id="cb_{idx}" checked '
            f'onchange="document.getElementById(\'{cid}\').style.display='
            f'this.checked?\'block\':\'none\'"> {label}</label>\n'
        )
        fig_html = fig.to_html(full_html=False, include_plotlyjs=False,
                               div_id=cid, config={"responsive": True})
        chart_htmls += f'<div style="margin-bottom:16px;">{fig_html}</div>\n'

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Q-Function Dashboard — Vector Q v2</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ background: #1A1A2E; color: #CCCCCC; font-family: sans-serif; padding: 16px; }}
  h2   {{ color: #EEEEEE; margin-bottom: 4px; }}
  .controls {{ background: #16213E; padding: 12px 16px; border-radius: 6px;
               margin-bottom: 16px; line-height: 2; }}
  input[type=checkbox] {{ margin-right: 4px; accent-color: #4CAF50; }}
  .stats {{ color: #AAAAAA; font-size: 13px; margin-bottom: 12px; }}
</style>
</head>
<body>
<h2>Q-Function Dashboard — Vector Q v2</h2>
<div class="stats">
  Checkpoint: {CHECKPOINT_PATH}<br>
  Buffer: {BUFFER_DIR}<br>
  Samples per buffer: {SAMPLES_PER_BUFFER} &nbsp;|&nbsp; Device: {DEVICE}
</div>
<div class="controls">
  <strong style="color:#EEEEEE;margin-right:12px;">Show:</strong>
  {checkboxes}
</div>
{chart_htmls}
</body>
</html>"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"HTML dashboard saved to: {OUTPUT_HTML}")


def main():
    sac = load_model()
    buffer = load_buffer()

    sizes = buffer.sizes()
    total = sum(sizes.values())
    if total == 0:
        logger.error("Buffer is empty — nothing to plot. Run training first.")
        return

    logger.info("Sampling from buffers and computing Q-values...")
    all_data = collect_data(sac, buffer)

    logger.info("\nSaving PNG plots...")
    save_png_plots(all_data, OUTPUT_DIR)

    logger.info("\nSaving HTML dashboard...")
    save_html_dashboard(all_data)

    logger.info(f"\nDone. PNGs at: {OUTPUT_DIR}")
    logger.info(f"HTML at: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
