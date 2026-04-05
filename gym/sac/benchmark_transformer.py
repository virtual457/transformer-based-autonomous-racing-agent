"""
benchmark_transformer.py — Benchmark Transformer encoder + policy head
across different architectures and window sizes.

Architecture under test (one encoder, one actor):
    obs_window (1, window_size, obs_dim=125)
    → Linear(obs_dim, d_model)                   # input projection
    → TransformerEncoder(n_layers, n_heads, d_model)
    → mean pool → embedding (d_model,)
    → MLP policy head → (action_dim * 2,)        # mean + log_std

Run:
    .\\AssetoCorsa\\Scripts\\python.exe gym/sac/benchmark_transformer.py

Prints a table of all configs and flags which fit inside MAX_INFERENCE_MS.
"""

import time
import itertools

import torch
import torch.nn as nn

# ── Fixed constants ────────────────────────────────────────────────────────────
OBS_DIM        = 50    # per-frame token: basic obs + oot + curvature + past actions + current action
ACTION_DIM     = 3
N_WARMUP       = 20
N_RUNS         = 100
MAX_INFERENCE_MS = 25.0
POLICY_MLP_HIDDEN = 256   # policy head hidden dim (fixed)

# ── Search grid ────────────────────────────────────────────────────────────────
WINDOW_SIZES   = [75]
D_MODELS       = [64, 128, 256]
N_HEADS_LIST   = [2, 4, 8]
N_LAYERS_LIST  = [1, 2, 3, 4]


# ── Model ─────────────────────────────────────────────────────────────────────

class TransformerActorEncoder(nn.Module):
    """
    Shared Transformer encoder + Gaussian policy head.

    Forward:
        obs_window : (batch, window_size, obs_dim)  →  (batch, action_dim * 2)

    The encoder produces a mean-pooled embedding which the policy head
    maps to (mean, log_std) for action sampling.
    """

    def __init__(
        self,
        obs_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        action_dim: int,
        mlp_hidden: int = 256,
    ):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,          # Pre-LN — more stable in RL
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, action_dim * 2),   # mean + log_std
        )

    def forward(self, obs_window: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(obs_window)   # (B, W, d_model)
        x = self.encoder(x)               # (B, W, d_model)
        x = x.mean(dim=1)                 # (B, d_model)  — mean pool over window
        return self.policy_head(x)        # (B, action_dim * 2)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(model: nn.Module, window_size: int, device: torch.device) -> dict:
    model.eval()
    dummy = torch.randn(1, window_size, OBS_DIM, device=device)

    # Warmup — JIT / cuDNN autotuning
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(N_RUNS):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": sum(times) / len(times),
        "max_ms":  max(times),
        "min_ms":  min(times),
        "p95_ms":  sorted(times)[int(0.95 * len(times))],
    }


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device        : {device}")
    print(f"Budget        : {MAX_INFERENCE_MS} ms  (max over {N_RUNS} runs)")
    print(f"obs_dim       : {OBS_DIM}   action_dim: {ACTION_DIM}")
    print(f"policy_mlp    : [{POLICY_MLP_HIDDEN}]")
    print()

    col = f"{'win':>4} {'d':>5} {'h':>3} {'L':>2}  {'params':>8}  {'mean':>7} {'p95':>7} {'max':>7}  {'ok':>4}"
    print(col)
    print("-" * len(col))

    results = []

    combos = list(itertools.product(WINDOW_SIZES, D_MODELS, N_HEADS_LIST, N_LAYERS_LIST))
    for window_size, d_model, n_heads, n_layers in combos:
        if d_model % n_heads != 0:
            continue   # invalid head config

        model = TransformerActorEncoder(
            obs_dim=OBS_DIM,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            action_dim=ACTION_DIM,
            mlp_hidden=POLICY_MLP_HIDDEN,
        ).to(device)

        stats  = benchmark(model, window_size, device)
        params = param_count(model)
        fits   = stats["max_ms"] < MAX_INFERENCE_MS

        print(
            f"{window_size:>4} {d_model:>5} {n_heads:>3} {n_layers:>2}  "
            f"{params:>8,}  "
            f"{stats['mean_ms']:>7.2f} {stats['p95_ms']:>7.2f} {stats['max_ms']:>7.2f}  "
            f"{'Y' if fits else 'N':>4}"
        )

        results.append({"window_size": window_size, "d_model": d_model,
                         "n_heads": n_heads, "n_layers": n_layers,
                         "params": params, "fits": fits, **stats})

        del model

    # ── Summary ───────────────────────────────────────────────────────────────
    passing = [r for r in results if r["fits"]]
    print()
    print(f"Configs within budget : {len(passing)} / {len(results)}")

    if passing:
        # Most expressive = most layers, then largest d_model, then largest window
        best = max(passing, key=lambda r: (r["n_layers"], r["d_model"], r["window_size"]))
        print()
        print("-- Recommended (most expressive within budget) --")
        print(f"  window_size = {best['window_size']}")
        print(f"  d_model     = {best['d_model']}")
        print(f"  n_heads     = {best['n_heads']}")
        print(f"  n_layers    = {best['n_layers']}")
        print(f"  params      = {best['params']:,}")
        print(f"  mean_ms     = {best['mean_ms']:.2f}")
        print(f"  p95_ms      = {best['p95_ms']:.2f}")
        print(f"  max_ms      = {best['max_ms']:.2f}")


if __name__ == "__main__":
    main()
