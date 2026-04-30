"""
Offline inference demo for Vector-Q Transformer SAC.

Loads the trained checkpoint, reads a handful of observation windows from
the stratified replay buffer on disk, and runs the policy + vector-Q
critic forward. No Assetto Corsa needed, since the buffer .dat files contain
the exact (75, 50) observation tokens the policy was trained on.

Each block below is marked with `# %%` so it renders as a cell in
Jupyter / VS Code / Colab. Run top-to-bottom.
"""

# %% ---------------------------------------------------------------------
# Imports and paths

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "gym"))

from transformer_sac_vectorq_v2_final_fineTune.network import (  # noqa: E402
    LOG_STD_MAX,
    LOG_STD_MIN,
    PolicyHead,
    TransformerEncoder,
    TwinQHead,
)

VARIANT = "transformer_sac_vectorq_v2_final_fineTune"
CKPT    = os.path.join(ROOT, "gym", VARIANT, "checkpoints", "latest.pt")
BUF_DIR = os.path.join(ROOT, "gym", VARIANT, "checkpoints", "buffers")
OUT_PNG = os.path.join(os.path.dirname(__file__), "inference_demo.png")

# %% ---------------------------------------------------------------------
# Load the trained checkpoint

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg  = ckpt["config"]
print("checkpoint config:")
for k in ("token_dim", "window_size", "d_model", "n_heads",
         "n_layers", "ffn_dim", "action_dim"):
    print(f"  {k:<12} = {cfg[k]}")
print(f"  target_entropy = {cfg['target_entropy'].tolist()}")

# %% ---------------------------------------------------------------------
# Rebuild the policy and critic networks from the checkpoint

policy_encoder = TransformerEncoder(
    token_dim=cfg["token_dim"],   d_model=cfg["d_model"],
    n_heads=cfg["n_heads"],       n_layers=cfg["n_layers"],
    ffn_dim=cfg["ffn_dim"],       window_size=cfg["window_size"],
)
critic_encoder = TransformerEncoder(
    token_dim=cfg["token_dim"],   d_model=cfg["d_model"],
    n_heads=cfg["n_heads"],       n_layers=cfg["n_layers"],
    ffn_dim=cfg["ffn_dim"],       window_size=cfg["window_size"],
)
policy_head = PolicyHead(d_model=cfg["d_model"],
                         action_dim=cfg["action_dim"],
                         hidden_units=cfg["policy_hidden"])
twin_q      = TwinQHead(d_model=cfg["d_model"],
                        action_dim=cfg["action_dim"],
                        reward_dim=cfg["action_dim"],
                        hidden_units=cfg["q_hidden"])

policy_encoder.load_state_dict(ckpt["policy_encoder"])
critic_encoder.load_state_dict(ckpt["critic_encoder"])
policy_head.load_state_dict(ckpt["policy_head"])
twin_q.load_state_dict(ckpt["twin_q"])

for m in (policy_encoder, critic_encoder, policy_head, twin_q):
    m.eval()

n_params = sum(p.numel() for m in (policy_encoder, critic_encoder,
                                   policy_head, twin_q)
                         for p in m.parameters())
print(f"\nloaded {n_params/1e6:.2f}M parameters (policy + critic + heads)")

# %% ---------------------------------------------------------------------
# Load a handful of observation windows from the stratified replay buffer

def load_buffer(name, capacity=100_000, W=None, T=None, A=3, R=3):
    W = W or cfg["window_size"]
    T = T or cfg["token_dim"]
    base = os.path.join(BUF_DIR, name)
    meta = json.load(open(f"{base}_meta.json"))
    obs = np.memmap(f"{base}_obs.dat",    dtype=np.float32, mode="r",
                    shape=(capacity, W, T))
    act = np.memmap(f"{base}_action.dat", dtype=np.float32, mode="r",
                    shape=(capacity, A))
    rew = np.memmap(f"{base}_reward.dat", dtype=np.float32, mode="r",
                    shape=(capacity, R))
    return obs, act, rew, meta["size"]

# Pool windows across all six sub-buffers so we see a representative mix
# of positive and negative per-channel states.
rng = np.random.default_rng(0)
N_PER_BUF = 24
bufs = ["steer_pos", "steer_neg",
        "throttle_pos", "throttle_neg",
        "brake_pos", "brake_neg"]

sample_obs, sample_act_stored, sample_rew_stored, sample_tag = [], [], [], []
for b in bufs:
    obs_mm, act_mm, rew_mm, size = load_buffer(b)
    idx = rng.choice(size, size=N_PER_BUF, replace=False)
    sample_obs.append(np.array(obs_mm[idx]))
    sample_act_stored.append(np.array(act_mm[idx]))
    sample_rew_stored.append(np.array(rew_mm[idx]))
    sample_tag.extend([b] * N_PER_BUF)
sample_obs        = np.concatenate(sample_obs, axis=0)     # (N, 75, 50)
sample_act_stored = np.concatenate(sample_act_stored, 0)    # (N, 3)
sample_rew_stored = np.concatenate(sample_rew_stored, 0)    # (N, 3)
print(f"loaded {len(sample_obs)} observation windows from"
      f" {len(bufs)} stratified sub-buffers")

# %% ---------------------------------------------------------------------
# Forward pass: policy → action, critic → per-channel Q on that action

x = torch.from_numpy(sample_obs).float().clamp(-3.0, 3.0)

with torch.no_grad():
    pol_emb = policy_encoder(x)                    # (N, d)
    out     = policy_head.net(pol_emb)
    mean_raw, log_std = out.chunk(2, dim=-1)
    log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
    std     = log_std.exp()
    action_det = torch.tanh(mean_raw)              # deterministic policy

    crit_emb = critic_encoder(x)                   # (N, d)
    q1, q2 = twin_q(crit_emb, action_det)          # each (N, 3)
    q = torch.min(q1, q2).numpy()                  # clipped double-Q

action_det = action_det.numpy()
std        = std.numpy()
print(f"action means:  steer={action_det[:,0].mean():+.3f}"
      f"  throttle={action_det[:,1].mean():+.3f}"
      f"  brake={action_det[:,2].mean():+.3f}")
print(f"policy std:    steer={std[:,0].mean():.3f}"
      f"   throttle={std[:,1].mean():.3f}"
      f"   brake={std[:,2].mean():.3f}")
print(f"clipped-Q mean: steer={q[:,0].mean():.2f}"
      f"  throttle={q[:,1].mean():.2f}"
      f"  brake={q[:,2].mean():.2f}")

# %% ---------------------------------------------------------------------
# Visualise: predicted vs stored action + per-channel Q histograms

fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2),
                         constrained_layout=True)
names  = ["steer", "throttle", "brake"]
colors = ["tab:blue", "tab:orange", "tab:red"]

# Top row: scatter of predicted action (this model) vs action stored when
# the transition was collected. Tight diagonal clustering = policy has
# converged toward what it originally did on these states.
for i, (ax, name, c) in enumerate(zip(axes[0], names, colors)):
    ax.scatter(sample_act_stored[:, i], action_det[:, i],
               s=12, alpha=0.55, color=c)
    ax.plot([-1, 1], [-1, 1], "k--", lw=0.7)
    ax.set_xlim(-1.05, 1.05);  ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(f"stored $a_{{\\mathrm{{{name}}}}}$")
    ax.set_ylabel(f"predicted $a_{{\\mathrm{{{name}}}}}$")
    ax.set_title(f"{name}: policy vs stored")
    ax.grid(True, ls=":", alpha=0.35)

# Bottom row: per-channel Q on the action the policy just picked.
for i, (ax, name, c) in enumerate(zip(axes[1], names, colors)):
    ax.hist(q[:, i], bins=28, color=c, alpha=0.8, edgecolor="black",
            linewidth=0.4)
    ax.axvline(q[:, i].mean(), color="black", lw=1.2, ls="--",
               label=f"mean = {q[:, i].mean():.1f}")
    ax.set_xlabel(f"$Q^{{\\mathrm{{{name}}}}}(s, \\pi(s))$")
    ax.set_ylabel("count")
    ax.set_title(f"{name}: per-channel $Q$")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.35)

fig.suptitle("Vector-Q Transformer SAC: offline inference on "
             f"{len(sample_obs)} replay-buffer windows", fontsize=12)
plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"wrote {OUT_PNG}")

# %% ---------------------------------------------------------------------
# Optional sanity check: one window end-to-end

idx = 0
print("\n(single-window inspection)")
print(f"buffer tag:            {sample_tag[idx]}")
print(f"stored action:         {sample_act_stored[idx].tolist()}")
print(f"stored per-ch reward:  {sample_rew_stored[idx].tolist()}")
print(f"policy predicted:      {action_det[idx].tolist()}")
print(f"policy std per chan:   {std[idx].tolist()}")
print(f"per-channel Q at pi(s):{q[idx].tolist()}")
