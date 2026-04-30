"""
benchmark_transformer_memory.py — Measure peak VRAM during a full SAC training step.

Tests the FULL training memory cost:
    - Encoder (shared) + Policy head + Twin Q heads
    - Forward + backward pass on a batch of sequences
    - Adam optimizer step

This is what actually matters for GPU fit, not just inference.

Run:
    .\\AssetoCorsa\\Scripts\\python.exe gym/sac/benchmark_transformer_memory.py
"""

import torch
import torch.nn as nn
import itertools

# ── Fixed constants ────────────────────────────────────────────────────────────
TOKEN_DIM    = 50     # per-frame: basic obs + oot + curvature + past actions + cur action
ACTION_DIM   = 3
WINDOW_SIZE  = 75
BATCH_SIZE   = 256
MLP_HIDDEN   = 256
MAX_VRAM_GB  = 6.0    # leave 2GB headroom on 8GB card

# ── Search grid ───────────────────────────────────────────────────────────────
D_MODELS     = [64, 128, 256, 512]
N_HEADS_LIST = [4, 8]
N_LAYERS_LIST = [2, 4, 6]


# ── Architecture ──────────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    def __init__(self, token_dim, d_model, n_heads, n_layers):
        super().__init__()
        self.proj = nn.Linear(token_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        return self.encoder(self.proj(x)).mean(dim=1)  # (B, d_model)


class PolicyHead(nn.Module):
    def __init__(self, d_model, action_dim, mlp_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, action_dim * 2),
        )
    def forward(self, emb): return self.net(emb)


class QHead(nn.Module):
    def __init__(self, d_model, action_dim, mlp_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + action_dim, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )
    def forward(self, emb, action):
        return self.net(torch.cat([emb, action], dim=-1))


# ── Memory benchmark ──────────────────────────────────────────────────────────

def measure_training_vram(d_model, n_heads, n_layers, device):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # One shared encoder, one policy head, two Q heads
    encoder  = TransformerEncoder(TOKEN_DIM, d_model, n_heads, n_layers).to(device)
    policy   = PolicyHead(d_model, ACTION_DIM, MLP_HIDDEN).to(device)
    q1       = QHead(d_model, ACTION_DIM, MLP_HIDDEN).to(device)
    q2       = QHead(d_model, ACTION_DIM, MLP_HIDDEN).to(device)

    all_params = (list(encoder.parameters()) + list(policy.parameters()) +
                  list(q1.parameters()) + list(q2.parameters()))
    optimizer = torch.optim.Adam(all_params, lr=3e-4)

    # Dummy batch
    obs_seq    = torch.randn(BATCH_SIZE, WINDOW_SIZE, TOKEN_DIM, device=device)
    next_seq   = torch.randn(BATCH_SIZE, WINDOW_SIZE, TOKEN_DIM, device=device)
    action     = torch.randn(BATCH_SIZE, ACTION_DIM, device=device)
    reward     = torch.randn(BATCH_SIZE, 1, device=device)
    done       = torch.zeros(BATCH_SIZE, 1, device=device)

    try:
        optimizer.zero_grad()

        # ── Critic update ──────────────────────────────────────────────────────
        with torch.no_grad():
            next_emb    = encoder(next_seq)
            next_out    = policy(next_emb)
            next_mean, next_log_std = next_out.chunk(2, dim=-1)
            next_action = torch.tanh(next_mean)
            q1t = q1(next_emb, next_action)
            q2t = q2(next_emb, next_action)
            q_target = reward + 0.99 * (1 - done) * torch.min(q1t, q2t)

        emb    = encoder(obs_seq)
        q1_val = q1(emb, action)
        q2_val = q2(emb, action)
        q_loss = (q1_val - q_target).pow(2).mean() + (q2_val - q_target).pow(2).mean()
        q_loss.backward()

        # ── Policy update ──────────────────────────────────────────────────────
        optimizer.zero_grad()
        emb2      = encoder(obs_seq)
        out2      = policy(emb2)
        mean2, _  = out2.chunk(2, dim=-1)
        act2      = torch.tanh(mean2)
        q1v2      = q1(emb2, act2)
        q2v2      = q2(emb2, act2)
        p_loss    = -torch.min(q1v2, q2v2).mean()
        p_loss.backward()

        optimizer.step()

        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_gb    = peak_bytes / 1024**3

        total_params = sum(p.numel() for p in all_params)
        fits = peak_gb < MAX_VRAM_GB
        return peak_gb, total_params, fits, None

    except torch.cuda.OutOfMemoryError as e:
        return None, None, False, "OOM"
    finally:
        del encoder, policy, q1, q2, optimizer
        torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda")
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"VRAM budget   : {MAX_VRAM_GB} GB  (leaving 2 GB headroom)")
    print(f"Batch size    : {BATCH_SIZE}  window={WINDOW_SIZE}  token_dim={TOKEN_DIM}")
    print()

    col = f"{'d':>5} {'h':>3} {'L':>2}  {'params':>9}  {'peak_GB':>8}  {'ok':>4}"
    print(col)
    print("-" * len(col))

    results = []
    for d_model, n_heads, n_layers in itertools.product(D_MODELS, N_HEADS_LIST, N_LAYERS_LIST):
        if d_model % n_heads != 0:
            continue

        peak_gb, params, fits, err = measure_training_vram(d_model, n_heads, n_layers, device)

        if err == "OOM":
            print(f"{d_model:>5} {n_heads:>3} {n_layers:>2}  {'?':>9}  {'OOM':>8}  {'N':>4}")
        else:
            print(
                f"{d_model:>5} {n_heads:>3} {n_layers:>2}  "
                f"{params:>9,}  {peak_gb:>8.2f}  {'Y' if fits else 'N':>4}"
            )
            results.append({
                "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
                "params": params, "peak_gb": peak_gb, "fits": fits,
            })

    passing = [r for r in results if r["fits"]]
    print()
    print(f"Configs within {MAX_VRAM_GB}GB budget: {len(passing)} / {len(results)}")

    if passing:
        best = max(passing, key=lambda r: (r["n_layers"], r["d_model"]))
        print()
        print("-- Largest config that fits --")
        print(f"  d_model  = {best['d_model']}")
        print(f"  n_heads  = {best['n_heads']}")
        print(f"  n_layers = {best['n_layers']}")
        print(f"  params   = {best['params']:,}")
        print(f"  peak_GB  = {best['peak_gb']:.2f} GB")


if __name__ == "__main__":
    main()
