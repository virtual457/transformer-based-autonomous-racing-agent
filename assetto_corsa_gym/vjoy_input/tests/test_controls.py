"""
test_controls.py
----------------
Pure math tests for axis conversion logic.
No vJoy device required -- runs fully offline.

Confirmed axis mapping (from find_axes.py diagnostic run):
    wAxisX  -> STEERING
    wAxisY  -> IGNORED
    wAxisZ  -> COMBINED GAS/BRAKE (single axis)

Run:
    cd assetto_corsa_gym
    python -m pytest vjoy_input/tests/test_controls.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from vjoy_input.controls import (
    steer_to_axis,
    combined_to_axis,
    buttons_mask,
    encode,
    neutral,
    AXIS_MAX,
    AXIS_MID,
    BUTTON_SHIFT_UP,
    BUTTON_SHIFT_DOWN,
)


# ------------------------------------------------------------------
# steer_to_axis
# ------------------------------------------------------------------

class TestSteerToAxis:
    def test_center(self):
        assert steer_to_axis(0.0) == AXIS_MID   # 16384

    def test_full_left(self):
        assert steer_to_axis(-1.0) == 0

    def test_full_right(self):
        assert steer_to_axis(1.0) == AXIS_MAX   # 32768

    def test_half_right(self):
        val = steer_to_axis(0.5)
        assert AXIS_MID < val < AXIS_MAX

    def test_half_left(self):
        val = steer_to_axis(-0.5)
        assert 0 < val < AXIS_MID

    def test_clamp_above(self):
        assert steer_to_axis(2.0) == steer_to_axis(1.0)

    def test_clamp_below(self):
        assert steer_to_axis(-2.0) == steer_to_axis(-1.0)

    def test_range_always_valid(self):
        for v in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            result = steer_to_axis(v)
            assert 0 <= result <= AXIS_MAX, f"Out of range for steer={v}: {result}"


# ------------------------------------------------------------------
# combined_to_axis
# ------------------------------------------------------------------

class TestCombinedToAxis:
    def test_neutral_both_zero(self):
        # No throttle, no brake -> center
        assert combined_to_axis(0.0, 0.0) == AXIS_MID   # 16384

    def test_full_throttle_no_brake(self):
        # Full throttle -> axis goes LOW
        assert combined_to_axis(1.0, 0.0) == 0

    def test_full_brake_no_throttle(self):
        # Full brake -> axis goes HIGH
        assert combined_to_axis(0.0, 1.0) == AXIS_MAX   # 32768

    def test_half_throttle(self):
        val = combined_to_axis(0.5, 0.0)
        assert 0 < val < AXIS_MID

    def test_half_brake(self):
        val = combined_to_axis(0.0, 0.5)
        assert AXIS_MID < val < AXIS_MAX

    def test_clamp_throttle_above(self):
        assert combined_to_axis(2.0, 0.0) == combined_to_axis(1.0, 0.0)

    def test_clamp_brake_above(self):
        assert combined_to_axis(0.0, 2.0) == combined_to_axis(0.0, 1.0)

    def test_clamp_throttle_below(self):
        assert combined_to_axis(-0.5, 0.0) == combined_to_axis(0.0, 0.0)

    def test_clamp_brake_below(self):
        assert combined_to_axis(0.0, -0.5) == combined_to_axis(0.0, 0.0)

    def test_range_always_valid(self):
        for t in [0.0, 0.5, 1.0]:
            for b in [0.0, 0.5, 1.0]:
                result = combined_to_axis(t, b)
                assert 0 <= result <= AXIS_MAX, \
                    f"Out of range for throttle={t}, brake={b}: {result}"

    def test_throttle_monotonic_decreasing(self):
        # More throttle -> lower axis value
        vals = [combined_to_axis(t, 0.0) for t in [0.0, 0.5, 1.0]]
        assert vals == sorted(vals, reverse=True)

    def test_brake_monotonic_increasing(self):
        # More brake -> higher axis value
        vals = [combined_to_axis(0.0, b) for b in [0.0, 0.5, 1.0]]
        assert vals == sorted(vals)


# ------------------------------------------------------------------
# buttons_mask
# ------------------------------------------------------------------

class TestButtonsMask:
    def test_no_buttons(self):
        assert buttons_mask() == 0

    def test_shift_up(self):
        assert buttons_mask(shift_up=True) == BUTTON_SHIFT_UP

    def test_shift_down(self):
        assert buttons_mask(shift_down=True) == BUTTON_SHIFT_DOWN

    def test_both(self):
        both = buttons_mask(shift_up=True, shift_down=True)
        assert both == (BUTTON_SHIFT_UP | BUTTON_SHIFT_DOWN)


# ------------------------------------------------------------------
# encode (full round-trip)
# ------------------------------------------------------------------

class TestEncode:
    def test_returns_required_keys(self):
        result = encode()
        # wAxisX = steer, wAxisZ = combined gas/brake
        assert "wAxisX"   in result
        assert "wAxisZ"   in result
        assert "lButtons" in result

    def test_no_old_keys(self):
        # wAxisXRot and wAxisYRot are no longer used
        result = encode()
        assert "wAxisXRot" not in result
        assert "wAxisYRot" not in result

    def test_neutral_defaults(self):
        # Default: steer=0 (center), throttle=0, brake=0 (combined center)
        result = encode()
        assert result["wAxisX"]   == AXIS_MID   # steer straight
        assert result["wAxisZ"]   == AXIS_MID   # combined neutral
        assert result["lButtons"] == 0

    def test_full_throttle_no_brake(self):
        result = encode(steer=0.0, throttle=1.0, brake=0.0)
        assert result["wAxisZ"] == 0            # full throttle = axis LOW

    def test_full_brake_no_throttle(self):
        result = encode(steer=0.0, throttle=0.0, brake=1.0)
        assert result["wAxisZ"] == AXIS_MAX     # full brake = axis HIGH

    def test_steer_left(self):
        result = encode(steer=-1.0)
        assert result["wAxisX"] == 0

    def test_steer_right(self):
        result = encode(steer=1.0)
        assert result["wAxisX"] == AXIS_MAX

    def test_shift_up_button(self):
        result = encode(shift_up=True)
        assert result["lButtons"] == BUTTON_SHIFT_UP


# ------------------------------------------------------------------
# neutral
# ------------------------------------------------------------------

class TestNeutral:
    def test_neutral_matches_encode_defaults(self):
        # neutral() must equal encode with all-zero inputs
        assert neutral() == encode(steer=0.0, throttle=0.0, brake=0.0)

    def test_steer_is_center(self):
        assert neutral()["wAxisX"] == AXIS_MID   # wAxisX = steer

    def test_combined_is_center(self):
        # Both pedals released -> combined axis at center
        assert neutral()["wAxisZ"] == AXIS_MID   # wAxisZ = combined gas/brake


# ------------------------------------------------------------------
# Consistency: steer mapping is symmetric
# ------------------------------------------------------------------

class TestSymmetry:
    def test_steer_symmetry(self):
        left   = steer_to_axis(-0.5)
        right  = steer_to_axis(0.5)
        center = steer_to_axis(0.0)
        # center - left == right - center  (within 1 due to int truncation)
        assert abs((center - left) - (right - center)) <= 1

    def test_combined_symmetry(self):
        # Equal throttle and brake offset from center should be symmetric
        half_throttle = combined_to_axis(0.5, 0.0)
        half_brake    = combined_to_axis(0.0, 0.5)
        center        = AXIS_MID
        assert abs((center - half_throttle) - (half_brake - center)) <= 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
