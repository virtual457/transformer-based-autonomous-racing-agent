"""
episode_logger.py — Per-episode Vector Q reward and action plots.

Shows per-channel (steer, throttle, brake) rewards step-wise with their
weighted component breakdowns.  No cumulative reward — all step-wise.

Output
------
    outputs/episode_<N>/
        dashboard.html — all charts on one page with checkboxes
"""

from pathlib import Path

import numpy as np


_BG    = "#1A1A2E"
_PAPER = "#16213E"
_GRID  = "#2a2a4a"


def _base(title, height):
    return dict(
        title=dict(text=title, font=dict(color="#EEEEEE", size=12)),
        paper_bgcolor=_BG, plot_bgcolor=_PAPER,
        font=dict(color="#CCCCCC"),
        hovermode="x unified", height=height,
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#444"),
    )


def plot_episode(
    episode_num: int,
    actions: list,
    means: list,
    stds: list,
    reward_vecs: list,
    vec_breakdowns: list,
    components: list,
    metrics: list,
    output_dir: str,
) -> None:
    """
    Generate dashboard.html for one episode with full per-channel detail.

    Parameters
    ----------
    reward_vecs    : list of (3,) ndarray — [r_steer, r_throttle, r_brake] per step
    vec_breakdowns : list of dict — per-channel weighted component breakdown per step
                     keys: steer, throttle, brake — each a dict of component→weighted_value
    components     : list of dict — raw reward_components per step
    metrics        : list of dict — speed_ms, target_speed_ms, gap_m, yaw_error_deg
    """
    import json
    import plotly.graph_objects as go

    if not reward_vecs:
        return

    ep_dir = Path(output_dir) / f"episode_{episode_num}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    T = len(reward_vecs)
    frames = list(range(T))

    # Ensure arrays.
    reward_arr  = np.array([np.asarray(r) for r in reward_vecs], dtype=np.float32)  # (T, 3)
    actions_arr = np.array(actions, dtype=float) if actions else np.zeros((T, 3))
    means_arr   = np.array(means,   dtype=float) if means   else np.zeros((T, 3))
    stds_arr    = np.array(stds,    dtype=float) if stds    else np.zeros((T, 3))

    charts = []

    def _fig_to_json(fig):
        return fig.to_json()

    # ── Channel reward colors ────────────────────────────────────────────────
    CH_COLORS = {
        "steer":    "#F44336",
        "throttle": "#4CAF50",
        "brake":    "#9C27B0",
    }
    COMP_COLORS = {
        "gap":      "#FF9800",
        "gap_dir":  "#FFEB3B",
        "yaw":      "#00BCD4",
        "smooth":   "#607D8B",
        "speed":    "#2196F3",
        "progress": "#8BC34A",
        "crash":    "#F44336",
    }

    # ── 1. Per-channel reward total (overlay) ────────────────────────────────
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#555", dash="dash", width=0.8))
    for i, (ch_name, color) in enumerate(CH_COLORS.items()):
        fig.add_trace(go.Scatter(
            x=frames, y=reward_arr[:, i].tolist(), name=ch_name,
            line=dict(color=color, width=1.5),
            hovertemplate=f"frame=%{{x}}<br>{ch_name}=%{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(**_base("Per-Channel Reward (step-wise)", 400),
                      yaxis=dict(range=[-1.1, 1.1], gridcolor=_GRID, title="reward [-1,1]"),
                      xaxis=dict(gridcolor=_GRID, title="frame"))
    charts.append(("Channel Rewards", _fig_to_json(fig)))

    # ── 2. Steer reward + weighted components ────────────────────────────────
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#555", dash="dash", width=0.8))
    fig.add_trace(go.Scatter(
        x=frames, y=reward_arr[:, 0].tolist(), name="r_steer (total)",
        line=dict(color=CH_COLORS["steer"], width=2),
        hovertemplate="frame=%{x}<br>r_steer=%{y:.4f}<extra></extra>",
    ))
    # Component traces.
    for comp_key in ["gap", "gap_dir", "yaw", "smooth", "crash"]:
        vals = [bd.get("steer", {}).get(comp_key, 0.0) for bd in vec_breakdowns]
        if any(v != 0.0 for v in vals):
            fig.add_trace(go.Scatter(
                x=frames, y=vals, name=f"w*{comp_key}",
                line=dict(color=COMP_COLORS.get(comp_key, "#888"), width=1, dash="dot"),
                hovertemplate=f"frame=%{{x}}<br>w*{comp_key}=%{{y:.4f}}<extra></extra>",
            ))
    fig.update_layout(**_base("Steer Reward — Components", 350),
                      yaxis=dict(range=[-1.1, 1.1], gridcolor=_GRID),
                      xaxis=dict(gridcolor=_GRID, title="frame"))
    charts.append(("Steer Reward", _fig_to_json(fig)))

    # ── 3. Throttle reward + weighted components ─────────────────────────────
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#555", dash="dash", width=0.8))
    fig.add_trace(go.Scatter(
        x=frames, y=reward_arr[:, 1].tolist(), name="r_throttle (total)",
        line=dict(color=CH_COLORS["throttle"], width=2),
        hovertemplate="frame=%{x}<br>r_throttle=%{y:.4f}<extra></extra>",
    ))
    for comp_key in ["speed", "gap", "yaw", "smooth", "crash"]:
        vals = [bd.get("throttle", {}).get(comp_key, 0.0) for bd in vec_breakdowns]
        if any(v != 0.0 for v in vals):
            fig.add_trace(go.Scatter(
                x=frames, y=vals, name=f"w*{comp_key}",
                line=dict(color=COMP_COLORS.get(comp_key, "#888"), width=1, dash="dot"),
                hovertemplate=f"frame=%{{x}}<br>w*{comp_key}=%{{y:.4f}}<extra></extra>",
            ))
    fig.update_layout(**_base("Throttle Reward — Components", 350),
                      yaxis=dict(range=[-1.1, 1.1], gridcolor=_GRID),
                      xaxis=dict(gridcolor=_GRID, title="frame"))
    charts.append(("Throttle Reward", _fig_to_json(fig)))

    # ── 4. Brake reward + weighted components ────────────────────────────────
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#555", dash="dash", width=0.8))
    fig.add_trace(go.Scatter(
        x=frames, y=reward_arr[:, 2].tolist(), name="r_brake (total)",
        line=dict(color=CH_COLORS["brake"], width=2),
        hovertemplate="frame=%{x}<br>r_brake=%{y:.4f}<extra></extra>",
    ))
    for comp_key in ["speed", "gap", "yaw", "smooth", "crash"]:
        vals = [bd.get("brake", {}).get(comp_key, 0.0) for bd in vec_breakdowns]
        if any(v != 0.0 for v in vals):
            fig.add_trace(go.Scatter(
                x=frames, y=vals, name=f"w*{comp_key}",
                line=dict(color=COMP_COLORS.get(comp_key, "#888"), width=1, dash="dot"),
                hovertemplate=f"frame=%{{x}}<br>w*{comp_key}=%{{y:.4f}}<extra></extra>",
            ))
    fig.update_layout(**_base("Brake Reward — Components", 350),
                      yaxis=dict(range=[-1.1, 1.1], gridcolor=_GRID),
                      xaxis=dict(gridcolor=_GRID, title="frame"))
    charts.append(("Brake Reward", _fig_to_json(fig)))

    # ── 5. Raw reward components (from env) ──────────────────────────────────
    def _comp(key):
        return [c.get(key, 0.0) for c in components]

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#555", dash="dash", width=0.8))
    for name, color in [
        ("r_progress",   "#4CAF50"),
        ("r_speed",      "#2196F3"),
        ("r_gap_abs",    "#FF9800"),
        ("r_smoothness", "#9C27B0"),
        ("r_yaw",        "#00BCD4"),
        ("r_crash",      "#F44336"),
    ]:
        fig.add_trace(go.Scatter(
            x=frames, y=_comp(name), name=name,
            line=dict(color=color, width=1.2),
            hovertemplate=f"frame=%{{x}}<br>{name}=%{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(**_base("Raw Reward Components (env)", 400),
                      yaxis=dict(range=[-1.1, 1.1], gridcolor=_GRID),
                      xaxis=dict(gridcolor=_GRID, title="frame"))
    charts.append(("Raw Components", _fig_to_json(fig)))

    # ── 6. Action charts (one per action) ────────────────────────────────────
    action_names  = ["Steer", "Throttle", "Brake"]
    action_colors = ["#F44336", "#4CAF50", "#9C27B0"]
    for i, (name, color) in enumerate(zip(action_names, action_colors)):
        sampled = actions_arr[:, i].tolist() if actions_arr.ndim == 2 else [0.0] * T
        mean_i  = means_arr[:, i].tolist()   if means_arr.ndim == 2   else [0.0] * T
        std_i   = stds_arr[:, i]             if stds_arr.ndim == 2    else np.zeros(T)
        upper   = (means_arr[:, i] + std_i).tolist() if means_arr.ndim == 2 else [0.0] * T
        lower   = (means_arr[:, i] - std_i).tolist() if means_arr.ndim == 2 else [0.0] * T
        _rgba   = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frames + frames[::-1], y=upper + lower[::-1],
            fill="toself", fillcolor=_rgba, line=dict(color="rgba(0,0,0,0)"),
            name="±std", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=frames, y=mean_i, name="mean",
            line=dict(color=color, width=1.2, dash="dash"),
            hovertemplate=f"frame=%{{x}}<br>mean=%{{y:.4f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=frames, y=sampled, name="sampled",
            line=dict(color=color, width=1.0),
            hovertemplate=f"frame=%{{x}}<br>sampled=%{{y:.4f}}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color="#555", dash="dash", width=0.6))
        fig.update_layout(**_base(f"Action: {name}", 300),
                          yaxis=dict(range=[-1.1, 1.1], gridcolor=_GRID),
                          xaxis=dict(gridcolor=_GRID, title="frame"))
        charts.append((f"Action: {name}", _fig_to_json(fig)))

    # ── 7. Obs charts (speed + target, gap, yaw) ────────────────────────────
    if metrics:
        speed_ms        = [m.get("speed_ms",        0.0) for m in metrics]
        target_speed_ms = [m.get("target_speed_ms",  0.0) for m in metrics]
        gap_m           = [m.get("gap_m",            0.0) for m in metrics]
        yaw_error_deg   = [m.get("yaw_error_deg",    0.0) for m in metrics]

        # Speed + target speed overlay.
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frames, y=speed_ms, name="speed",
            line=dict(color="#03A9F4", width=1.2),
            hovertemplate="frame=%{x}<br>speed=%{y:.2f} m/s<extra></extra>"))
        if any(t > 0 for t in target_speed_ms):
            fig.add_trace(go.Scatter(x=frames, y=target_speed_ms, name="target",
                line=dict(color="#FF5722", width=1.2, dash="dash"),
                hovertemplate="frame=%{x}<br>target=%{y:.2f} m/s<extra></extra>"))
        fig.update_layout(**_base("Speed vs Target (m/s)", 300),
                          yaxis=dict(gridcolor=_GRID, title="m/s", rangemode="tozero"),
                          xaxis=dict(gridcolor=_GRID, title="frame"))
        charts.append(("Speed", _fig_to_json(fig)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frames, y=gap_m, name="gap_m",
            line=dict(color="#FF9800", width=1.2),
            hovertemplate="frame=%{x}<br>gap=%{y:.3f} m<extra></extra>"))
        fig.add_hline(y=0, line=dict(color="#555", dash="dash", width=0.6))
        fig.update_layout(**_base("Gap from Racing Line (m, signed)", 300),
                          yaxis=dict(gridcolor=_GRID, title="metres"),
                          xaxis=dict(gridcolor=_GRID, title="frame"))
        charts.append(("Gap", _fig_to_json(fig)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frames, y=yaw_error_deg, name="yaw_error",
            line=dict(color="#00BCD4", width=1.2),
            hovertemplate="frame=%{x}<br>yaw=%{y:.2f} deg<extra></extra>"))
        fig.update_layout(**_base("Yaw Error (deg, absolute)", 300),
                          yaxis=dict(gridcolor=_GRID, title="degrees", rangemode="tozero"),
                          xaxis=dict(gridcolor=_GRID, title="frame"))
        charts.append(("Yaw Error", _fig_to_json(fig)))

    # ── Build HTML dashboard ─────────────────────────────────────────────────
    checkboxes = ""
    chart_divs = ""
    for idx, (label, _) in enumerate(charts):
        cid = f"chart_{idx}"
        checkboxes += (
            f'<label style="margin-right:16px;cursor:pointer;">'
            f'<input type="checkbox" id="cb_{idx}" checked '
            f'onchange="document.getElementById(\'{cid}\').style.display='
            f'this.checked?\'block\':\'none\'"> {label}</label>\n'
        )
        chart_divs += f'<div id="{cid}" style="margin-bottom:12px;"></div>\n'

    scripts = ""
    for idx, (_, fig_json) in enumerate(charts):
        scripts += (
            f'var _f{idx} = JSON.parse({json.dumps(fig_json)});\n'
            f'Plotly.newPlot("chart_{idx}", _f{idx}.data, _f{idx}.layout, {{responsive: true}});\n'
        )

    # Summary stats.
    r_s = reward_arr[:, 0]; r_t = reward_arr[:, 1]; r_b = reward_arr[:, 2]
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Episode {episode_num} — Vector Q Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ background: {_BG}; color: #CCCCCC; font-family: sans-serif; padding: 16px; }}
  h2   {{ color: #EEEEEE; margin-bottom: 4px; }}
  .stats {{ color: #AAAAAA; font-size: 13px; margin-bottom: 12px; }}
  .controls {{ background: #16213E; padding: 12px 16px; border-radius: 6px;
               margin-bottom: 16px; line-height: 2; }}
  input[type=checkbox] {{ margin-right: 4px; accent-color: #4CAF50; }}
</style>
</head>
<body>
<h2>Episode {episode_num} — Vector Q Dashboard</h2>
<div class="stats">
  frames={T} &nbsp;|&nbsp;
  steer: mean={r_s.mean():.3f} &nbsp;|&nbsp;
  throttle: mean={r_t.mean():.3f} &nbsp;|&nbsp;
  brake: mean={r_b.mean():.3f} &nbsp;|&nbsp;
  total: mean={reward_arr.sum(axis=1).mean():.3f}
</div>
<div class="controls">
  <strong style="color:#EEEEEE;margin-right:12px;">Show:</strong>
  {checkboxes}
</div>
{chart_divs}
<script>
{scripts}
</script>
</body>
</html>"""

    (ep_dir / "dashboard.html").write_text(html, encoding="utf-8")
