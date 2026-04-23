"""
episode_logger.py — Per-episode reward and action plots for Transformer SAC finetuning.

Called by agent.py at the end of every episode.

Output
------
    outputs/episode_<N>/
        reward_plot.html  — total reward + composite components per frame (interactive)
        action_plot.html  — sampled action + policy mean + std per frame (interactive)

Open in any browser: zoom, hover for exact values, click legend to toggle lines.
"""

from pathlib import Path

import numpy as np


def plot_episode(
    episode_num: int,
    actions: list,
    means: list,
    stds: list,
    rewards: list,
    components: list,
    metrics: list,
    output_dir: str,
) -> None:
    """
    Generate reward_plot.html and action_plot.html for one episode.

    Parameters
    ----------
    episode_num  : episode index within the phase (1-based), folder overwritten each phase
    actions      : list of np.ndarray (action_dim,) — sampled action per frame
    means        : list of np.ndarray (action_dim,) — tanh(mean) per frame
    stds         : list of np.ndarray (action_dim,) — std per frame
    rewards      : list of float — total reward per frame
    components   : list of dict  — reward_components dict per frame
                   keys: r_progress, r_speed, r_gap_abs, r_smoothness, r_yaw, r_crash
    output_dir   : root outputs directory (phase_P_episode_N/ subdir created here)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not rewards:
        return

    ep_dir = Path(output_dir) / f"episode_{episode_num}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    T = len(rewards)
    frames = list(range(T))

    actions_arr = np.array(actions, dtype=float) if actions else np.zeros((T, 3))
    means_arr   = np.array(means,   dtype=float) if means   else np.zeros((T, 3))
    stds_arr    = np.array(stds,    dtype=float) if stds    else np.zeros((T, 3))
    rewards_arr = np.array(rewards, dtype=float)

    def _comp(key):
        return [c.get(key, 0.0) for c in components]

    r_progress   = _comp("r_progress")
    r_speed      = _comp("r_speed")
    r_gap_abs    = _comp("r_gap_abs")
    r_smoothness = _comp("r_smoothness")
    r_yaw        = _comp("r_yaw")
    r_crash      = _comp("r_crash")

    _BG    = "#1A1A2E"
    _PAPER = "#16213E"
    _GRID  = "#2a2a4a"

    def _base_layout(title):
        return dict(
            title=dict(text=title, font=dict(color="#EEEEEE", size=13)),
            paper_bgcolor=_BG,
            plot_bgcolor=_PAPER,
            font=dict(color="#CCCCCC"),
            xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID, title="frame"),
            legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#444"),
            hovermode="x unified",
        )

    # ── reward_plot.html ──────────────────────────────────────────────────────
    fig_r = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Reward components per frame", "Total reward + cumulative"),
        vertical_spacing=0.12,
    )

    component_traces = [
        ("r_progress",   r_progress,   "#4CAF50"),
        ("r_speed",      r_speed,      "#2196F3"),
        ("r_gap_abs",    r_gap_abs,    "#FF9800"),
        ("r_smoothness", r_smoothness, "#9C27B0"),
        ("r_yaw",        r_yaw,        "#00BCD4"),
        ("r_crash",      r_crash,      "#F44336"),
    ]
    for name, vals, color in component_traces:
        fig_r.add_trace(go.Scatter(
            x=frames, y=vals, name=name,
            line=dict(color=color, width=1.5),
            hovertemplate=f"frame=%{{x}}<br>{name}=%{{y:.4f}}<extra></extra>",
        ), row=1, col=1)

    fig_r.add_hline(y=0, line=dict(color="#555", dash="dash", width=0.8), row=1, col=1)

    fig_r.add_trace(go.Scatter(
        x=frames, y=rewards_arr.tolist(), name="total reward",
        line=dict(color="#E91E63", width=1.5),
        hovertemplate="frame=%{x}<br>reward=%{y:.4f}<extra></extra>",
    ), row=2, col=1)

    fig_r.add_trace(go.Scatter(
        x=frames, y=np.cumsum(rewards_arr).tolist(), name="cumulative",
        line=dict(color="#FF5722", width=1.5, dash="dot"),
        hovertemplate="frame=%{x}<br>cumulative=%{y:.2f}<extra></extra>",
        yaxis="y3",
    ), row=2, col=1)

    fig_r.add_hline(y=0, line=dict(color="#555", dash="dash", width=0.8), row=2, col=1)

    layout = _base_layout(
        f"Episode {episode_num} — Reward | frames={T}  "
        f"total={rewards_arr.sum():.2f}  mean/step={rewards_arr.mean():.3f}"
    )
    layout["yaxis"]  = dict(range=[-1.05, 1.05], gridcolor=_GRID, title="component [-1,1]")
    layout["yaxis2"] = dict(range=[-1.05, 1.05], gridcolor=_GRID, title="reward [-1,1]")
    layout["xaxis2"] = dict(gridcolor=_GRID, title="frame")
    layout["height"] = 700
    fig_r.update_layout(**layout)
    fig_r.update_annotations(font_color="#CCCCCC")

    fig_r.write_html(str(ep_dir / "reward_plot.html"), include_plotlyjs="cdn")

    # ── action_plot.html ──────────────────────────────────────────────────────
    action_names  = ["steer", "throttle", "brake"]
    action_colors = ["#F44336", "#4CAF50", "#9C27B0"]

    fig_a = make_subplots(
        rows=3, cols=1,
        subplot_titles=action_names,
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    for i, (name, color) in enumerate(zip(action_names, action_colors)):
        row = i + 1
        sampled = actions_arr[:, i].tolist() if actions_arr.ndim == 2 else [0.0] * T
        mean_i  = means_arr[:, i].tolist()   if means_arr.ndim == 2   else [0.0] * T
        std_i   = stds_arr[:, i]             if stds_arr.ndim == 2    else np.zeros(T)

        upper = (means_arr[:, i] + std_i).tolist() if means_arr.ndim == 2 else [0.0] * T
        lower = (means_arr[:, i] - std_i).tolist() if means_arr.ndim == 2 else [0.0] * T

        # ±1 std band
        _rgba = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)"
        fig_a.add_trace(go.Scatter(
            x=frames + frames[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=_rgba,
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{name} ±std",
            showlegend=True,
            hoverinfo="skip",
        ), row=row, col=1)

        fig_a.add_trace(go.Scatter(
            x=frames, y=mean_i, name=f"{name} mean",
            line=dict(color=color, width=1.2, dash="dash"),
            hovertemplate=f"frame=%{{x}}<br>mean=%{{y:.4f}}<extra></extra>",
        ), row=row, col=1)

        fig_a.add_trace(go.Scatter(
            x=frames, y=sampled, name=f"{name} sampled",
            line=dict(color=color, width=1.0),
            opacity=0.9,
            hovertemplate=f"frame=%{{x}}<br>sampled=%{{y:.4f}}<extra></extra>",
        ), row=row, col=1)

        fig_a.add_hline(y=0, line=dict(color="#555", dash="dash", width=0.6), row=row, col=1)

    layout_a = _base_layout(f"Episode {episode_num} — Actions | frames={T}")
    layout_a["height"] = 900
    for i in range(1, 4):
        key = "yaxis" if i == 1 else f"yaxis{i}"
        layout_a[key] = dict(range=[-1.1, 1.1], gridcolor=_GRID)
    layout_a[f"xaxis3"] = dict(gridcolor=_GRID, title="frame")
    fig_a.update_layout(**layout_a)
    fig_a.update_annotations(font_color="#CCCCCC")

    fig_a.write_html(str(ep_dir / "action_plot.html"), include_plotlyjs="cdn")

    # ── obs_plot.html ─────────────────────────────────────────────────────────
    if metrics:
        speed_ms      = [m.get("speed_ms",      0.0) for m in metrics]
        gap_m         = [m.get("gap_m",          0.0) for m in metrics]
        yaw_error_deg = [m.get("yaw_error_deg",  0.0) for m in metrics]

        fig_o = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Speed (m/s)", "Gap from racing line (m)", "Yaw error (deg, absolute)"),
            shared_xaxes=True,
            vertical_spacing=0.08,
        )

        fig_o.add_trace(go.Scatter(
            x=frames, y=speed_ms, name="speed_ms",
            line=dict(color="#03A9F4", width=1.2),
            hovertemplate="frame=%{x}<br>speed=%{y:.2f} m/s<extra></extra>",
        ), row=1, col=1)

        fig_o.add_trace(go.Scatter(
            x=frames, y=gap_m, name="gap_m",
            line=dict(color="#FF9800", width=1.2),
            hovertemplate="frame=%{x}<br>gap=%{y:.3f} m<extra></extra>",
        ), row=2, col=1)
        fig_o.add_hline(y=0, line=dict(color="#555", dash="dash", width=0.6), row=2, col=1)

        fig_o.add_trace(go.Scatter(
            x=frames, y=yaw_error_deg, name="yaw_error_deg",
            line=dict(color="#00BCD4", width=1.2),
            hovertemplate="frame=%{x}<br>yaw=%{y:.2f}°<extra></extra>",
        ), row=3, col=1)

        layout_o = _base_layout(f"Episode {episode_num} — Observations | frames={T}")
        layout_o["height"] = 750
        layout_o["yaxis"]  = dict(gridcolor=_GRID, title="m/s")
        layout_o["yaxis2"] = dict(gridcolor=_GRID, title="metres", rangemode="tozero")
        layout_o["yaxis3"] = dict(gridcolor=_GRID, title="degrees", rangemode="tozero")
        layout_o["xaxis3"] = dict(gridcolor=_GRID, title="frame")
        fig_o.update_layout(**layout_o)
        fig_o.update_annotations(font_color="#CCCCCC")

        fig_o.write_html(str(ep_dir / "obs_plot.html"), include_plotlyjs="cdn")

    # ── dashboard.html — all charts with checkboxes ───────────────────────────
    _write_dashboard(
        ep_dir=ep_dir,
        episode_num=episode_num,
        frames=frames,
        T=T,
        rewards_arr=rewards_arr,
        r_progress=r_progress, r_speed=r_speed, r_gap_abs=r_gap_abs,
        r_smoothness=r_smoothness, r_yaw=r_yaw, r_crash=r_crash,
        actions_arr=actions_arr, means_arr=means_arr, stds_arr=stds_arr,
        speed_ms=speed_ms if metrics else None,
        gap_m=gap_m if metrics else None,
        yaw_error_deg=yaw_error_deg if metrics else None,
        _BG=_BG, _PAPER=_PAPER, _GRID=_GRID,
    )


def _write_dashboard(
    ep_dir, episode_num, frames, T,
    rewards_arr, r_progress, r_speed, r_gap_abs, r_smoothness, r_yaw, r_crash,
    actions_arr, means_arr, stds_arr,
    speed_ms, gap_m, yaw_error_deg,
    _BG, _PAPER, _GRID,
):
    """Generate dashboard.html — all charts on one page with checkboxes."""
    import json
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def _fig_to_json(fig):
        return fig.to_json()

    def _base(title, height):
        return dict(
            title=dict(text=title, font=dict(color="#EEEEEE", size=12)),
            paper_bgcolor=_BG, plot_bgcolor=_PAPER,
            font=dict(color="#CCCCCC"),
            hovermode="x unified", height=height,
            legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#444"),
        )

    charts = []  # list of (label, fig_json)

    # ── Reward components ──────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#555", dash="dash", width=0.8))
    for name, vals, color in [
        ("r_progress",   r_progress,   "#4CAF50"),
        ("r_speed",      r_speed,      "#2196F3"),
        ("r_gap_abs",    r_gap_abs,    "#FF9800"),
        ("r_smoothness", r_smoothness, "#9C27B0"),
        ("r_yaw",        r_yaw,        "#00BCD4"),
        ("r_crash",      r_crash,      "#F44336"),
    ]:
        fig.add_trace(go.Scatter(x=frames, y=list(vals), name=name,
            line=dict(color=color, width=1.5),
            hovertemplate=f"frame=%{{x}}<br>{name}=%{{y:.4f}}<extra></extra>"))
    fig.update_layout(**_base("Reward Components", 400),
                      yaxis=dict(range=[-1.05, 1.05], gridcolor=_GRID),
                      xaxis=dict(gridcolor=_GRID, title="frame"))
    charts.append(("Reward Components", _fig_to_json(fig)))

    # ── Total reward + cumulative ──────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frames, y=rewards_arr.tolist(), name="total reward",
        line=dict(color="#E91E63", width=1.5),
        hovertemplate="frame=%{x}<br>reward=%{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=frames, y=np.cumsum(rewards_arr).tolist(), name="cumulative",
        line=dict(color="#FF5722", width=1.5, dash="dot"),
        hovertemplate="frame=%{x}<br>cumulative=%{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#555", dash="dash", width=0.8))
    fig.update_layout(**_base("Total Reward + Cumulative", 350),
                      yaxis=dict(gridcolor=_GRID),
                      xaxis=dict(gridcolor=_GRID, title="frame"))
    charts.append(("Total Reward", _fig_to_json(fig)))

    # ── Action charts (one per action) ────────────────────────────────────────
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
        fig.update_layout(**_base(name, 300),
                          yaxis=dict(range=[-1.1, 1.1], gridcolor=_GRID),
                          xaxis=dict(gridcolor=_GRID, title="frame"))
        charts.append((name, _fig_to_json(fig)))

    # ── Obs charts ────────────────────────────────────────────────────────────
    if speed_ms is not None:
        for label, vals, color, ytitle, tmpl in [
            ("Speed (m/s)",    speed_ms,      "#03A9F4", "m/s",     "speed=%{y:.2f} m/s"),
            ("Gap (m)",        gap_m,         "#FF9800", "metres",  "gap=%{y:.3f} m"),
            ("Yaw Error (°)",  yaw_error_deg, "#00BCD4", "degrees", "yaw=%{y:.2f}°"),
        ]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=frames, y=list(vals), name=label,
                line=dict(color=color, width=1.2),
                hovertemplate=f"frame=%{{x}}<br>{tmpl}<extra></extra>"))
            fig.update_layout(**_base(label, 300),
                              yaxis=dict(gridcolor=_GRID, title=ytitle, rangemode="tozero"),
                              xaxis=dict(gridcolor=_GRID, title="frame"))
            charts.append((label, _fig_to_json(fig)))

    # ── Build HTML ────────────────────────────────────────────────────────────
    chart_divs = ""
    checkboxes = ""
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

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Episode {episode_num} Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ background: {_BG}; color: #CCCCCC; font-family: sans-serif; padding: 16px; }}
  h2   {{ color: #EEEEEE; margin-bottom: 12px; }}
  .controls {{ background: #16213E; padding: 12px 16px; border-radius: 6px;
               margin-bottom: 16px; line-height: 2; }}
  input[type=checkbox] {{ margin-right: 4px; accent-color: #4CAF50; }}
</style>
</head>
<body>
<h2>Episode {episode_num} — Dashboard &nbsp;
  <small style="font-size:13px;color:#888;">frames={T} &nbsp;
  total={float(rewards_arr.sum()):.2f} &nbsp;
  mean/step={float(rewards_arr.mean()):.3f}</small>
</h2>
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
