"""
CompositeReward — bounded [-1, 1] weighted sum of 4 normalised components.

Architecture
------------
1. Crash short-circuit: if telem.out_of_track is True, return -1.0 immediately.
   No component is evaluated.
2. Weighted sum of 4 normalised components (weights must sum to 1.0).
3. Result is clamped to [-1, 1].  In practice the weighted sum of [-1,1]
   values with weights summing to 1.0 is already in [-1, 1], but clamping
   guards against floating-point drift.

Weights (hardcoded defaults, validated at init)
-----------------------------------------------
    r_progress   0.30
    r_speed      0.20
    r_gap_abs    0.35
    r_smoothness 0.15
    ──────────────────
    total        1.00

No speed multiplier, no REWARD_SCALE, no negate flags (negation is baked into
each component's formula in components.py).

Logging keys (backwards-compatible with reward_logger.py)
---------------------------------------------------------
    r_progress, r_speed, r_gap_abs, r_smoothness, r_crash
    c_progress, c_speed, c_gap_abs, c_smoothness, c_crash
    composite_pre_mult  — weighted sum (= total when not crashed)
    multiplier          — always 1.0 (kept for log schema compatibility)
    composite_post_mult — same as composite_pre_mult (no multiplier step)
    reward_scale        — always 1.0 (kept for log schema compatibility)
    total_reward        — final clipped value returned to the agent (-1.0 on crash)

Usage
-----
    from rewards.composite import CompositeReward

    # Default weights (0.30 / 0.20 / 0.35 / 0.15):
    reward = CompositeReward.default()

    # Custom weights (must sum to 1.0):
    reward = CompositeReward.from_weights_dict({
        "r_progress":   0.30,
        "r_speed":      0.20,
        "r_gap_abs":    0.35,
        "r_smoothness": 0.15,
    })

    result = reward.compute(telem, action, prev_action, prev_lap_dist, track_length)
    # result["total"]      → float in [-1, 1]
    # result["components"] → {"r_progress": ..., "r_speed": ..., ...}

Config-driven usage (from OmegaConf / our_env.py)
--------------------------------------------------
    reward = CompositeReward.from_weights(cfg.our_env.reward_weights)
    reward = CompositeReward.from_config(cfg.our_env.reward)
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from rewards.base import BaseRewardComponent
from telemetry.base import TelemetryFrame


# ---------------------------------------------------------------------------
# Weight validation tolerance
# ---------------------------------------------------------------------------

_WEIGHT_SUM_TOLERANCE = 1e-6


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


# ---------------------------------------------------------------------------
# ComponentConfig
# ---------------------------------------------------------------------------

@dataclass
class ComponentConfig:
    component: BaseRewardComponent
    weight: float
    # negate is kept for API compatibility with from_weights() / from_config()
    # but has NO effect on the new bounded reward path — negation is baked into
    # each component's formula.
    negate: bool = False


# ---------------------------------------------------------------------------
# CompositeReward
# ---------------------------------------------------------------------------

class CompositeReward:
    """
    Bounded [-1, 1] weighted sum of 4 normalised reward components.

    Crash overrides all: returns -1.0 immediately if telem.out_of_track.
    """

    def __init__(self, components: list):
        """
        Parameters
        ----------
        components : list of ComponentConfig
            Weights must sum to 1.0 (within _WEIGHT_SUM_TOLERANCE).
        """
        self._components = components
        weight_sum = sum(cc.weight for cc in components)
        if abs(weight_sum - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"ComponentConfig weights must sum to 1.0, got {weight_sum:.6f}. "
                f"Weights: {[cc.weight for cc in components]}"
            )

    def reset(self) -> None:
        """
        No-op in the bounded reward design (no per-episode speed accumulator).
        Kept for API compatibility with our_env.py which calls reward.reset()
        at the start of every episode.
        """

    def compute(
        self,
        telem: TelemetryFrame,
        action: np.ndarray,
        prev_action: np.ndarray,
        prev_lap_dist: float,
        track_length: float = 5793.0,
        prev_gap_m: float = 0.0,
    ) -> dict:
        """
        Compute the bounded reward for one step.

        Returns
        -------
        dict with keys:
            "total"           : float in [-1, 1]
            "components"      : dict[name -> normalised component value]
            "speed_multiplier": float — always 1.0 (backwards compat)
            "log_row"         : dict — all reward internals for RewardLogger
                Reward-side keys:
                    r_progress, r_speed, r_gap_abs, r_smoothness, r_crash
                    c_progress, c_speed, c_gap_abs, c_smoothness, c_crash
                    composite_pre_mult, multiplier,
                    composite_post_mult, reward_scale, total_reward
        """
        # ── Crash short-circuit ──────────────────────────────────────────
        if telem.out_of_track:
            parts = {cc.component.name: 0.0 for cc in self._components}
            parts["r_crash"] = 1.0
            contributions = {"c_" + cc.component.name[2:]: 0.0
                             for cc in self._components}
            contributions["c_crash"] = -1.0
            log_row = self._build_log_row(
                parts, contributions,
                composite_pre_mult=-1.0,
                total=-1.0,
            )
            return {
                "total":            -1.0,
                "components":       parts,
                "speed_multiplier": 1.0,
                "log_row":          log_row,
            }

        # ── Evaluate 4 components ────────────────────────────────────────
        parts = {}
        contributions = {}
        total = 0.0

        for cc in self._components:
            raw = cc.component.compute(
                telem, action, prev_action, prev_lap_dist, track_length,
                prev_gap_m=prev_gap_m,
            )
            weighted = cc.weight * raw      # negation is baked into raw
            parts[cc.component.name] = raw
            contrib_key = "c_" + cc.component.name[2:]   # "r_X" → "c_X"
            contributions[contrib_key] = weighted
            total += weighted

        # Guard against floating-point drift outside [-1, 1]
        total = _clamp(total, -1.0, 1.0)

        # r_crash is always 0.0 on non-crash steps — populate for log schema
        if "r_crash" not in parts:
            parts["r_crash"] = 0.0
            contributions["c_crash"] = 0.0

        log_row = self._build_log_row(
            parts, contributions,
            composite_pre_mult=total,
            total=total,
        )

        return {
            "total":            total,
            "components":       parts,
            "speed_multiplier": 1.0,
            "log_row":          log_row,
        }

    # ------------------------------------------------------------------
    # Log row builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_log_row(
        parts: dict,
        contributions: dict,
        composite_pre_mult: float,
        total: float,
    ) -> dict:
        return {
            # raw (normalised) component values
            "r_progress":   parts.get("r_progress",   0.0),
            "r_speed":      parts.get("r_speed",       0.0),
            "r_gap_abs":    parts.get("r_gap_abs",     0.0),
            "r_smoothness": parts.get("r_smoothness",  0.0),
            "r_crash":      parts.get("r_crash",       0.0),
            # signed weighted contributions
            "c_progress":   contributions.get("c_progress",   0.0),
            "c_speed":      contributions.get("c_speed",       0.0),
            "c_gap_abs":    contributions.get("c_gap_abs",     0.0),
            "c_smoothness": contributions.get("c_smoothness",  0.0),
            "c_crash":      contributions.get("c_crash",       0.0),
            # pipeline scalars (multiplier/reward_scale kept for log schema compat)
            "composite_pre_mult":    composite_pre_mult,
            "multiplier":            1.0,
            "composite_post_mult":   composite_pre_mult,  # no multiplier step
            "reward_scale":          1.0,
            "total_reward":          total,
        }

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "CompositeReward":
        """
        Build with the locked design weights:
            r_progress   0.30
            r_speed      0.20
            r_gap_abs    0.35
            r_smoothness 0.15
        """
        from rewards.components import (
            ProgressReward, TargetSpeedReward, GapReward, SmoothnessReward,
        )
        return cls([
            ComponentConfig(ProgressReward(),        weight=0.30),
            ComponentConfig(TargetSpeedReward(),     weight=0.20),
            ComponentConfig(GapReward(),             weight=0.35),
            ComponentConfig(SmoothnessReward(),      weight=0.15),
        ])

    @classmethod
    def from_weights_dict(cls, weights: Dict[str, float]) -> "CompositeReward":
        """
        Build from a plain dict mapping component name to weight.

        The sum of weights must equal 1.0.

        Example
        -------
            CompositeReward.from_weights_dict({
                "r_progress":   0.30,
                "r_speed":      0.20,
                "r_gap_abs":    0.35,
                "r_smoothness": 0.15,
            })
        """
        from rewards.components import (
            ProgressReward, SpeedReward, TargetSpeedReward, GapReward,
            SmoothnessReward,
        )
        _name_to_class = {
            "r_progress":     ProgressReward,
            "r_speed":        TargetSpeedReward,   # new default; SpeedReward kept as alias
            "r_speed_legacy": SpeedReward,          # backwards compat
            "r_gap_abs":      GapReward,
            "r_smoothness":   SmoothnessReward,
        }
        components = []
        for name, weight in weights.items():
            klass = _name_to_class.get(name)
            if klass is None:
                raise ValueError(
                    f"Unknown component name '{name}'. "
                    f"Valid names: {list(_name_to_class.keys())}"
                )
            components.append(ComponentConfig(component=klass(), weight=float(weight)))
        return cls(components)

    @classmethod
    def from_weights(cls, w) -> "CompositeReward":
        """
        Build from the legacy reward_weights OmegaConf block.

        Normalises the weights so they sum to 1.0 (backwards-compat shim for
        configs that predate the bounded design).

        Accepted keys: w1_progress, w2_speed, w3_gap_abs, w5_smoothness.
        w4_gap_delta and w6_crash are silently ignored (gap-delta removed;
        crash is a hard override).
        """
        from rewards.components import (
            ProgressReward, TargetSpeedReward, GapReward, SmoothnessReward,
        )
        raw_weights = [
            float(getattr(w, "w1_progress",  0.30)),
            float(getattr(w, "w2_speed",      0.20)),
            float(getattr(w, "w3_gap_abs",    0.35)),
            float(getattr(w, "w5_smoothness", 0.15)),
        ]
        total = sum(raw_weights)
        if abs(total) < 1e-9:
            raise ValueError("from_weights: all weights are zero.")
        normalised = [rw / total for rw in raw_weights]

        return cls([
            ComponentConfig(ProgressReward(),        weight=normalised[0]),
            ComponentConfig(TargetSpeedReward(),     weight=normalised[1]),
            ComponentConfig(GapReward(),             weight=normalised[2]),
            ComponentConfig(SmoothnessReward(),      weight=normalised[3]),
        ])

    @classmethod
    def from_config(cls, reward_list) -> "CompositeReward":
        """
        Build from the list-of-dicts OmegaConf block.

        Normalises weights to sum to 1.0.  CrashReward entries are silently
        ignored — crash is a hard override in CompositeReward.compute().

        Example YAML:
            reward:
              - type: ProgressReward
                weight: 0.30
              - type: SpeedReward
                weight: 0.20
              - type: GapReward
                weight: 0.35
              - type: SmoothnessReward
                weight: 0.15
        """
        from rewards.components import (
            ProgressReward, SpeedReward, TargetSpeedReward, GapReward,
            SmoothnessReward, CrashReward,
        )
        _cls_map = {
            "ProgressReward":      ProgressReward,
            "SpeedReward":         SpeedReward,          # backwards compat
            "TargetSpeedReward":   TargetSpeedReward,    # new
            "GapReward":           GapReward,
            "SmoothnessReward":    SmoothnessReward,
            "CrashReward":         CrashReward,          # accepted but skipped below
        }
        _skip_classes = {CrashReward}
        components = []
        for entry in reward_list:
            klass = _cls_map[entry.type]
            if klass in _skip_classes:
                continue   # crash = hard override
            components.append(ComponentConfig(
                component=klass(),
                weight=float(entry.weight),
            ))
        # Normalise weights
        total = sum(cc.weight for cc in components)
        if abs(total) < 1e-9:
            raise ValueError("from_config: no valid components with non-zero weight.")
        for cc in components:
            cc.weight = cc.weight / total
        return cls(components)

    def component_names(self) -> list:
        return [cc.component.name for cc in self._components]
