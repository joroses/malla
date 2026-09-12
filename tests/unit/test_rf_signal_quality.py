"""Tests for the preset-aware RF signal-quality model.

Whether an SNR means a solid link depends on the modem preset: SF11
demodulates down to -17.5 dB, SF7 only to -7.5 dB. These tests pin the
preset resolution rules, the six demodulation limits, exact tier and
reliability boundaries, missing-value/sentinel handling, link-balance
states, and the shared color palette (see LINK_QUALITY_CONTRACT.md).
"""

import math

import pytest

from malla.config import _clear_config_cache
from malla.utils.signal_quality import (
    BALANCE_ASYMMETRIC_MARGINAL,
    BALANCE_BALANCED,
    BALANCE_MARGINAL_BOTH,
    BALANCE_UNIDIRECTIONAL,
    BALANCE_UNKNOWN,
    DEFAULT_LORA_PRESET,
    FADE_MARGIN_FAIR_DB,
    FADE_MARGIN_GOOD_DB,
    LORA_PRESET_SPREADING_FACTORS,
    QUALITY_COLORS,
    QUALITY_FAIR,
    QUALITY_GOOD,
    QUALITY_MARGINAL,
    QUALITY_UNKNOWN,
    RELIABILITY_MIDPOINT_DB,
    RELIABILITY_STEEPNESS_PER_DB,
    TRACEROUTE_UNKNOWN_SNR,
    calculate_estimated_reliability,
    calculate_fade_margin,
    classify_link_balance,
    classify_signal_quality,
    get_demodulation_snr_limit,
    get_quality_color,
    resolve_spreading_factor,
)

SF11_LIMIT = -17.5  # LongFast floor
SF7_LIMIT = -7.5


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    """Isolate the config singleton from local config.yaml and shell env."""
    monkeypatch.setenv("MALLA_CONFIG_FILE", str(tmp_path / "does-not-exist.yaml"))
    for var in ("MALLA_LORA_PRESET", "MALLA_LORA_SPREADING_FACTOR"):
        monkeypatch.delenv(var, raising=False)
    _clear_config_cache()
    yield
    _clear_config_cache()


class TestPresetResolution:
    @pytest.mark.parametrize(
        ("preset", "expected_sf"),
        sorted(LORA_PRESET_SPREADING_FACTORS.items()),
    )
    def test_every_supported_preset(self, preset, expected_sf):
        assert resolve_spreading_factor(preset=preset) == expected_sf

    @pytest.mark.parametrize(
        ("preset", "expected_sf"),
        [
            ("LongFast", 11),
            ("LONGFAST", 11),
            ("long_fast", 11),
            ("Long-Fast", 11),
            ("  LongFast  ", 11),
            ("long slow", 12),
            ("LONG_SLOW", 12),
            ("Medium-Slow", 10),
            ("SFNarrow", 7),
            ("sf_narrow", 7),
            ("SF7", 7),
            ("sf12", 12),
        ],
    )
    def test_name_normalization(self, preset, expected_sf):
        assert resolve_spreading_factor(preset=preset) == expected_sf

    @pytest.mark.parametrize(
        ("value", "expected_sf"),
        [(8, 8), (12, 12), ("sf10", 10), ("SF9", 9), ("11", 11), (" 7 ", 7)],
    )
    def test_explicit_spreading_factor_formats(self, value, expected_sf):
        assert resolve_spreading_factor(spreading_factor=value) == expected_sf

    @pytest.mark.parametrize(
        "value",
        [6, 13, 0, -3, "sf13", "sf6", "banana", "", "9.5", "1e1", None],
    )
    def test_invalid_explicit_sf_falls_back_never_clamps(self, value):
        # Invalid explicit values are fallback candidates, not clamped.
        assert resolve_spreading_factor(preset="LongSlow", spreading_factor=value) == 12

    def test_explicit_sf_beats_preset(self):
        assert resolve_spreading_factor(preset="LongFast", spreading_factor=7) == 7

    def test_preset_beats_config(self, monkeypatch):
        monkeypatch.setenv("MALLA_LORA_PRESET", "MediumFast")
        assert resolve_spreading_factor(preset="ShortFast") == 7

    def test_invalid_preset_falls_back_to_config(self, monkeypatch):
        monkeypatch.setenv("MALLA_LORA_PRESET", "MediumFast")
        assert resolve_spreading_factor(preset="NotAPreset") == 9

    def test_unrecognized_channel_name_is_only_a_hint(self, monkeypatch):
        monkeypatch.setenv("MALLA_LORA_PRESET", "LongSlow")
        assert resolve_spreading_factor(preset="my-private-channel") == 12

    def test_config_sf_beats_config_preset(self, monkeypatch):
        monkeypatch.setenv("MALLA_LORA_PRESET", "LongFast")
        monkeypatch.setenv("MALLA_LORA_SPREADING_FACTOR", "8")
        assert resolve_spreading_factor() == 8

    @pytest.mark.parametrize(
        ("sf_env", "expected_sf", "snr", "expected_quality"),
        [
            ("7", 7, -6.0, "marginal"),
            ("sf7", 7, -6.0, "marginal"),
            ("SF7", 7, -6.0, "marginal"),
            ("sf8", 8, -6.0, "fair"),
            ("sf11", 11, -6.0, "good"),
        ],
    )
    def test_config_env_sfn_format_classifies_correctly(
        self, monkeypatch, sf_env, expected_sf, snr, expected_quality
    ):
        """Documented sfN spreading factor env values must not be discarded or fallback to SF11."""
        monkeypatch.setenv("MALLA_LORA_PRESET", "LongFast")
        monkeypatch.setenv("MALLA_LORA_SPREADING_FACTOR", sf_env)
        sf = resolve_spreading_factor()
        assert sf == expected_sf
        assert classify_signal_quality(snr, sf) == expected_quality


    def test_config_env_blank_sf_falls_back_to_config_preset(self, monkeypatch):
        # Compose forwards MALLA_LORA_SPREADING_FACTOR="" when unset.
        monkeypatch.setenv("MALLA_LORA_PRESET", "ShortSlow")
        monkeypatch.setenv("MALLA_LORA_SPREADING_FACTOR", "")
        assert resolve_spreading_factor() == 8

    def test_config_invalid_sf_falls_back_to_config_preset(self, monkeypatch):
        monkeypatch.setenv("MALLA_LORA_PRESET", "MediumFast")
        monkeypatch.setenv("MALLA_LORA_SPREADING_FACTOR", "banana")
        assert resolve_spreading_factor() == 9

    def test_default_is_long_fast(self):
        assert DEFAULT_LORA_PRESET == "LongFast"
        assert resolve_spreading_factor() == 11
        assert resolve_spreading_factor(preset=None, spreading_factor=None) == 11

    @pytest.mark.parametrize("preset", [9, 12, 7])
    def test_integer_preset_argument(self, preset):
        assert resolve_spreading_factor(preset=preset) == preset

    @pytest.mark.parametrize("preset", [6, 13, 0])
    def test_integer_preset_out_of_range_falls_back(self, preset):
        assert resolve_spreading_factor(preset=preset) == 11

    def test_resolution_uses_config_at_call_time(self, monkeypatch):
        assert resolve_spreading_factor() == 11
        monkeypatch.setenv("MALLA_LORA_PRESET", "ShortFast")
        _clear_config_cache()
        assert resolve_spreading_factor() == 7


class TestDemodulationLimits:
    @pytest.mark.parametrize(
        ("sf", "limit"),
        [(7, -7.5), (8, -10.0), (9, -12.5), (10, -15.0), (11, -17.5), (12, -20.0)],
    )
    def test_all_six_limits(self, sf, limit):
        assert get_demodulation_snr_limit(sf) == pytest.approx(limit)

    @pytest.mark.parametrize("sf", [6, 13, 0, "sf99", None, "banana"])
    def test_invalid_sf_raises(self, sf):
        with pytest.raises(ValueError, match="spreading factor"):
            get_demodulation_snr_limit(sf)  # type: ignore[arg-type]


class TestFadeMargin:
    def test_margin_above_floor(self):
        assert calculate_fade_margin(-7.5, 11) == pytest.approx(10.0)
        assert calculate_fade_margin(0.0, 7) == pytest.approx(7.5)

    def test_real_zero_db_is_a_valid_measurement(self):
        assert calculate_fade_margin(0.0, 11) == pytest.approx(17.5)

    def test_missing_values_yield_none(self):
        assert calculate_fade_margin(None, 11) is None

    @pytest.mark.parametrize("snr", [float("nan"), float("inf"), 31.0, -30.5])
    def test_nonfinite_and_implausible_yield_none(self, snr):
        assert calculate_fade_margin(snr, 11) is None

    def test_traceroute_unknown_sentinel_is_rejected(self):
        assert TRACEROUTE_UNKNOWN_SNR == -32.0
        assert calculate_fade_margin(TRACEROUTE_UNKNOWN_SNR, 12) is None


class TestSignalQualityTiers:
    # SF11 floor is -17.5 dB, so margin boundaries sit at -7.5 / -13.5 dB.
    def test_exact_boundaries(self):
        assert FADE_MARGIN_GOOD_DB == 10.0
        assert FADE_MARGIN_FAIR_DB == 4.0
        assert classify_signal_quality(-7.5, 11) == QUALITY_GOOD  # margin 10.0
        assert classify_signal_quality(-7.51, 11) == QUALITY_FAIR  # margin 9.99
        assert classify_signal_quality(-13.5, 11) == QUALITY_FAIR  # margin 4.0
        assert classify_signal_quality(-13.51, 11) == QUALITY_MARGINAL  # margin 3.99

    def test_same_snr_different_tiers_per_preset(self):
        # -12 dB is marginal on SF7 (floor -7.5) but fair on SF11 (floor -17.5).
        assert classify_signal_quality(-12.0, 7) == QUALITY_MARGINAL
        assert classify_signal_quality(-12.0, 11) == QUALITY_FAIR

    def test_real_zero_db_classifies(self):
        assert classify_signal_quality(0.0, 7) == QUALITY_FAIR  # margin 7.5

    @pytest.mark.parametrize(
        "snr", [None, float("nan"), float("-inf"), TRACEROUTE_UNKNOWN_SNR]
    )
    def test_missing_or_unusable_snr_is_unknown(self, snr):
        assert classify_signal_quality(snr, 11) == QUALITY_UNKNOWN


class TestEstimatedReliability:
    def test_curve_constants(self):
        assert RELIABILITY_STEEPNESS_PER_DB == 0.6
        assert RELIABILITY_MIDPOINT_DB == 2.5

    def test_anchor_zero_margin(self):
        # 100 / (1 + e^1.5) ≈ 18.2 %
        value = calculate_estimated_reliability(-7.5, 7)  # margin exactly 0
        assert value == pytest.approx(100.0 / (1.0 + math.exp(1.5)))
        assert value == pytest.approx(18.2, abs=0.05)

    def test_anchor_plus_ten_db(self):
        # 100 / (1 + e^-4.5) ≈ 98.9 %
        value = calculate_estimated_reliability(2.5, 7)  # margin exactly 10
        assert value == pytest.approx(100.0 / (1.0 + math.exp(-4.5)))
        assert value == pytest.approx(98.9, abs=0.05)

    def test_midpoint_is_fifty_percent(self):
        snr = -7.5 + RELIABILITY_MIDPOINT_DB
        assert calculate_estimated_reliability(snr, 7) == pytest.approx(50.0)

    def test_monotonic_in_margin(self):
        previous = -1.0
        for margin in range(-12, 16, 2):
            value = calculate_estimated_reliability(-7.5 + margin, 7)
            assert value is not None
            assert value > previous
            assert 0.0 < value < 100.0
            previous = value

    @pytest.mark.parametrize("snr", [None, float("nan"), TRACEROUTE_UNKNOWN_SNR])
    def test_missing_snr_yields_null_reliability(self, snr):
        assert calculate_estimated_reliability(snr, 11) is None


class TestLinkBalance:
    def test_balanced(self):
        # Forward good (margin 10), return fair (margin 4) on SF11.
        assert (
            classify_link_balance(-7.5, -13.5, spreading_factor=11) == BALANCE_BALANCED
        )

    def test_asymmetric_marginal(self):
        # Forward fair (margin 4), return marginal (margin 2.5) on SF11.
        assert (
            classify_link_balance(-13.5, -15.0, spreading_factor=11)
            == BALANCE_ASYMMETRIC_MARGINAL
        )

    def test_marginal_both(self):
        assert (
            classify_link_balance(-15.0, -16.0, spreading_factor=11)
            == BALANCE_MARGINAL_BOTH
        )

    def test_unidirectional_when_other_direction_never_observed(self):
        assert (
            classify_link_balance(-10.0, None, spreading_factor=11)
            == BALANCE_UNIDIRECTIONAL
        )

    def test_unidirectional_with_explicit_zero_count(self):
        assert (
            classify_link_balance(
                -10.0, None, spreading_factor=11, return_observations=0
            )
            == BALANCE_UNIDIRECTIONAL
        )

    def test_observed_without_usable_snr_is_unknown(self):
        assert (
            classify_link_balance(
                -10.0, None, spreading_factor=11, return_observations=3
            )
            == BALANCE_UNKNOWN
        )

    def test_both_directions_missing_is_unknown(self):
        assert classify_link_balance(None, None, spreading_factor=11) == BALANCE_UNKNOWN
        assert (
            classify_link_balance(None, TRACEROUTE_UNKNOWN_SNR, spreading_factor=11)
            == BALANCE_UNKNOWN
        )

    def test_balance_is_preset_dependent(self):
        # Two SNRs alone cannot decide: -12 dB flips tier with the preset.
        assert (
            classify_link_balance(-12.0, -6.0, spreading_factor=7)
            == BALANCE_MARGINAL_BOTH
        )
        assert (
            classify_link_balance(-12.0, -6.0, spreading_factor=11) == BALANCE_BALANCED
        )

    def test_preclassified_qualities_override_snr_args(self):
        assert (
            classify_link_balance(
                forward_quality=QUALITY_GOOD, return_quality=QUALITY_MARGINAL
            )
            == BALANCE_ASYMMETRIC_MARGINAL
        )
        assert (
            classify_link_balance(
                forward_quality=QUALITY_FAIR, return_quality=QUALITY_FAIR
            )
            == BALANCE_BALANCED
        )

    def test_preclassified_unknown_respects_counts(self):
        assert (
            classify_link_balance(
                forward_quality=QUALITY_UNKNOWN,
                return_quality=QUALITY_GOOD,
                forward_observations=0,
            )
            == BALANCE_UNIDIRECTIONAL
        )
        assert (
            classify_link_balance(
                forward_quality=QUALITY_UNKNOWN,
                return_quality=QUALITY_GOOD,
                forward_observations=2,
            )
            == BALANCE_UNKNOWN
        )

    def test_marginal_plus_unobserved_is_unidirectional(self):
        # Not asymmetric_marginal: the other direction has no reading at all.
        assert (
            classify_link_balance(-16.0, None, spreading_factor=11)
            == BALANCE_UNIDIRECTIONAL
        )

    def test_uses_configured_preset_when_sf_not_given(self, monkeypatch):
        monkeypatch.setenv("MALLA_LORA_PRESET", "ShortFast")  # SF7 floor -7.5
        # Forward -3.5 → margin 4 (fair); return -6.5 → margin 1 (marginal).
        assert classify_link_balance(-3.5, -6.5) == BALANCE_ASYMMETRIC_MARGINAL


class TestQualityColors:
    def test_stable_palette(self):
        assert QUALITY_COLORS == {
            QUALITY_GOOD: "#28a745",
            QUALITY_FAIR: "#ffc107",
            QUALITY_MARGINAL: "#dc3545",
            QUALITY_UNKNOWN: "#6c757d",
        }

    @pytest.mark.parametrize(
        ("quality", "color"),
        [
            (QUALITY_GOOD, "#28a745"),
            (QUALITY_FAIR, "#ffc107"),
            (QUALITY_MARGINAL, "#dc3545"),
            (QUALITY_UNKNOWN, "#6c757d"),
            (None, "#6c757d"),
            ("excellent", "#6c757d"),
            ("", "#6c757d"),
        ],
    )
    def test_one_stable_color_per_tier(self, quality, color):
        assert get_quality_color(quality) == color
