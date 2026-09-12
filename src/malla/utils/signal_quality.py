"""Plausibility bounds and preset-aware quality models for LoRa signal metrics.

Corrupt gateway frames occasionally survive protobuf parsing and land in
packet_history with garbage signal values (rx_rssi like -1386841926, rx_snr
like 2.8e-36). A single such row is enough to drag an unguarded AVG(rssi)
to five-digit nonsense. packet_history deliberately stores whatever the
frame contained — it is the faithful raw record — so every read-side
aggregate over these columns must restrict itself to the plausible LoRa
range using the predicates below.

Conventions shared with Meshtastic firmware:
- rssi == 0 means "not provided" (e.g. the gateway's own uplinked packets);
  real LoRa receptions are always negative and above the sensitivity floor.
- snr is reported in quarter-dB steps within roughly [-20, 15] dB; the
  bounds here are deliberately generous so no real reception is rejected.

The second half of this module implements the preset-aware link-quality
model. Whether an SNR means a solid link depends on the modem preset: SF11
demodulates down to -17.5 dB while SF7 only reaches -7.5 dB, so raw SNR
cutoffs misjudge links. Helpers here resolve the configured preset into a
spreading factor and turn SNR readings into fade margin, quality tiers and
an "Estimated reliability" percentage. The reliability curve models signal
headroom only — it does not account for congestion or collisions, and it is
an estimate, not measured packet delivery.

Zero-SNR note (see LINK_QUALITY_CONTRACT.md): traceroute RF-hop
qualification treats snr == 0 as "not evidenced" and breaks inferred paths,
while this module (like packet receptions) treats a real 0.0 dB reading as
a usable measurement.
"""

import math
from typing import TypeGuard

from malla.config import get_config

RSSI_PLAUSIBLE_MIN = -150  # dBm; below any LoRa sensitivity floor (~-137)
RSSI_PLAUSIBLE_MAX = -1  # dBm; 0 is the "not provided" sentinel, positive is garbage
SNR_PLAUSIBLE_MIN = -30.0  # dB
SNR_PLAUSIBLE_MAX = 30.0  # dB

# Traceroute RouteDiscovery payloads encode "SNR unknown" as INT8_MIN (-128),
# which parse_traceroute_payload scales to -128/4 = -32.0 dB. That sentinel
# marks a real hop whose SNR simply wasn't recorded, so traceroute-payload
# consumers must not reclassify it as garbage.
TRACEROUTE_UNKNOWN_SNR = -32.0


def rssi_valid_sql(column: str = "rssi") -> str:
    """SQL predicate matching plausible RSSI values.

    NULL and the 0 "not provided" sentinel both fail the predicate, so this
    subsumes the older ``rssi IS NOT NULL AND rssi != 0`` guards.
    """
    return f"{column} BETWEEN {RSSI_PLAUSIBLE_MIN} AND {RSSI_PLAUSIBLE_MAX}"


def snr_valid_sql(column: str = "snr") -> str:
    """SQL predicate matching plausible SNR values (NULL fails it)."""
    return f"{column} BETWEEN {SNR_PLAUSIBLE_MIN} AND {SNR_PLAUSIBLE_MAX}"


def is_plausible_rssi(value: float | int | None) -> bool:
    """True when ``value`` is a real reception RSSI (0 sentinel excluded)."""
    return (
        value is not None
        and math.isfinite(value)
        and RSSI_PLAUSIBLE_MIN <= value <= RSSI_PLAUSIBLE_MAX
    )


def is_plausible_snr(value: float | int | None) -> bool:
    """True when ``value`` is a plausible SNR (0.0 is allowed here)."""
    return (
        value is not None
        and math.isfinite(value)
        and SNR_PLAUSIBLE_MIN <= value <= SNR_PLAUSIBLE_MAX
    )


def is_plausible_traceroute_snr(value: float | int | None) -> TypeGuard[float]:
    """True for plausible traceroute hop SNR, including the -32.0 "unknown" sentinel.

    Use this (not :func:`is_plausible_snr`) when filtering SNR values parsed
    from RouteDiscovery payloads, so hops whose SNR wasn't recorded keep
    appearing in graphs exactly as they did before the plausibility guards.
    The ``TypeGuard`` return type lets callers use a passing value as a plain
    ``float`` without re-checking for ``None``.
    """
    return value is not None and (
        value == TRACEROUTE_UNKNOWN_SNR or is_plausible_snr(value)
    )


# ---------------------------------------------------------------------------
# Preset-aware signal-quality model
# ---------------------------------------------------------------------------

DEFAULT_LORA_PRESET = "LongFast"
MIN_SPREADING_FACTOR = 7
MAX_SPREADING_FACTOR = 12

# Spreading factor per supported preset name. Standard mappings follow
# Meshtastic's radio settings (bandwidth differs per preset but the quality
# model only depends on the SF). "SFNarrow" is the regional SF7 alias used by
# the published Spanish community configuration. Channel names that match a
# preset are treated as hints only — see resolve_spreading_factor().
LORA_PRESET_SPREADING_FACTORS: dict[str, int] = {
    "shortfast": 7,
    "shortslow": 8,
    "mediumfast": 9,
    "mediumslow": 10,
    "longfast": 11,
    "longslow": 12,
    "verylongslow": 11,
    "shortturbo": 7,
    "longmoderate": 11,
    "sfnarrow": 7,
}

# Quality tiers for fade margin (dB above the demodulation floor).
QUALITY_GOOD = "good"
QUALITY_FAIR = "fair"
QUALITY_MARGINAL = "marginal"
QUALITY_UNKNOWN = "unknown"

FADE_MARGIN_GOOD_DB = 10.0  # margin >= this → good
FADE_MARGIN_FAIR_DB = 4.0  # margin >= this (but < good) → fair; below → marginal

# Logistic reliability curve constants:
#   reliability = 100 / (1 + exp(-STEEPNESS * (margin - MIDPOINT)))
# giving ~18.2% at zero fade margin and ~98.9% at +10 dB.
RELIABILITY_STEEPNESS_PER_DB = 0.6
RELIABILITY_MIDPOINT_DB = 2.5

# Link balance states.
BALANCE_BALANCED = "balanced"
BALANCE_ASYMMETRIC_MARGINAL = "asymmetric_marginal"
BALANCE_MARGINAL_BOTH = "marginal_both"
BALANCE_UNIDIRECTIONAL = "unidirectional"
BALANCE_UNKNOWN = "unknown"

# One stable color per quality tier, shared by every UI.
QUALITY_GOOD_COLOR = "#28a745"
QUALITY_FAIR_COLOR = "#ffc107"
QUALITY_MARGINAL_COLOR = "#dc3545"
QUALITY_UNKNOWN_COLOR = "#6c757d"
QUALITY_COLORS: dict[str, str] = {
    QUALITY_GOOD: QUALITY_GOOD_COLOR,
    QUALITY_FAIR: QUALITY_FAIR_COLOR,
    QUALITY_MARGINAL: QUALITY_MARGINAL_COLOR,
    QUALITY_UNKNOWN: QUALITY_UNKNOWN_COLOR,
}

_USABLE_QUALITIES = frozenset({QUALITY_GOOD, QUALITY_FAIR})


def _normalize_preset_name(name: object) -> str | None:
    """Normalize a preset/channel name for table lookup.

    Whitespace, case, underscores and hyphens are ignored ("Long_Fast",
    "LONG-FAST", " longfast " all match). Returns None for non-strings or
    blank names.
    """
    if not isinstance(name, str):
        return None
    normalized = name.strip().lower()
    for separator in (" ", "_", "-"):
        normalized = normalized.replace(separator, "")
    return normalized or None


def _normalize_spreading_factor(value: object) -> int | None:
    """Return **value** as a spreading factor in 7–12, or None.

    Accepts ints, whole floats, numeric strings and "sfN" strings. Anything
    else — including out-of-range SFs — is a fallback candidate (None), never
    clamped into a plausible setting.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        sf: int | None = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        sf = int(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("sf"):
            text = text[2:]
        try:
            sf = int(text)
        except ValueError:
            return None
    else:
        return None
    if sf is not None and MIN_SPREADING_FACTOR <= sf <= MAX_SPREADING_FACTOR:
        return sf
    return None


def _preset_name_to_spreading_factor(name: object) -> int | None:
    """Resolve a preset/channel name (or raw SF) to a spreading factor."""
    if isinstance(name, (int, float)) and not isinstance(name, bool):
        return _normalize_spreading_factor(name)
    normalized = _normalize_preset_name(name)
    if normalized is None:
        return None
    if normalized in LORA_PRESET_SPREADING_FACTORS:
        return LORA_PRESET_SPREADING_FACTORS[normalized]
    return _normalize_spreading_factor(normalized)  # "sf11"-style names


def resolve_spreading_factor(
    preset: str | int | None = None,
    spreading_factor: int | str | None = None,
) -> int:
    """Resolve the spreading factor (SF7–SF12) used by the quality model.

    Precedence (first *valid* value wins; invalid inputs are fallback
    candidates and are never clamped into plausible settings):

    1. explicit *spreading_factor* (int 7–12, "sfN", or numeric string),
    2. recognized *preset* / per-link channel name (standard Meshtastic
       presets, the SFNarrow regional alias, "sfN" names, or an int SF),
    3. configured ``lora_spreading_factor``, then configured ``lora_preset``,
    4. LongFast (SF11).

    Channel names are only hints: an unrecognized channel name falls through
    to the configured preset. Callers should expose the resolved SF so users
    can understand the estimate.
    """
    explicit = _normalize_spreading_factor(spreading_factor)
    if explicit is not None:
        return explicit

    preset_sf = _preset_name_to_spreading_factor(preset)
    if preset_sf is not None:
        return preset_sf

    cfg = get_config()
    config_sf = _normalize_spreading_factor(cfg.lora_spreading_factor)
    if config_sf is not None:
        return config_sf
    config_preset_sf = _preset_name_to_spreading_factor(cfg.lora_preset)
    if config_preset_sf is not None:
        return config_preset_sf

    return LORA_PRESET_SPREADING_FACTORS["longfast"]  # LongFast default → SF11


def get_demodulation_snr_limit(spreading_factor: int) -> float:
    """Demodulation SNR floor for *spreading_factor* (SX1262/SX1276).

    ``-7.5 - 2.5 × (SF - 7)`` dB: SF7 → -7.5 dB … SF12 → -20.0 dB.
    Raises ValueError for anything outside SF 7–12.
    """
    sf = _normalize_spreading_factor(spreading_factor)
    if sf is None:
        msg = f"spreading factor must be {MIN_SPREADING_FACTOR}-{MAX_SPREADING_FACTOR}, got {spreading_factor!r}"
        raise ValueError(msg)
    return -7.5 - 2.5 * (sf - 7)


def calculate_fade_margin(
    snr: float | int | None,
    spreading_factor: int,
) -> float | None:
    """Headroom (dB) of *snr* above the preset's demodulation floor.

    Returns None when the SNR is missing, nonfinite or implausible (which
    includes the -32.0 dB traceroute "unknown" sentinel); real 0.0 dB
    readings are valid measurements.
    """
    if snr is None or not is_plausible_snr(snr):
        return None
    return float(snr) - get_demodulation_snr_limit(spreading_factor)


def classify_signal_quality(
    snr: float | int | None,
    spreading_factor: int,
) -> str:
    """Quality tier for *snr* at *spreading_factor*.

    good: fade margin ≥ 10 dB; fair: ≥ 4 dB; marginal: below 4 dB.
    Missing/unusable SNR yields "unknown".
    """
    margin = calculate_fade_margin(snr, spreading_factor)
    if margin is None:
        return QUALITY_UNKNOWN
    if margin >= FADE_MARGIN_GOOD_DB:
        return QUALITY_GOOD
    if margin >= FADE_MARGIN_FAIR_DB:
        return QUALITY_FAIR
    return QUALITY_MARGINAL


def calculate_estimated_reliability(
    snr: float | int | None,
    spreading_factor: int,
) -> float | None:
    """Estimated reliability percentage (0–100) from fade margin.

    Logistic delivery-rate curve
    ``100 / (1 + exp(-RELIABILITY_STEEPNESS_PER_DB × (margin - RELIABILITY_MIDPOINT_DB)))``
    (~18.2% at zero margin, 98.9% at +10 dB). This models signal headroom
    only — congestion and collisions are not accounted for — and it is an
    estimate, not measured packet delivery. Missing/unusable SNR yields None.
    """
    margin = calculate_fade_margin(snr, spreading_factor)
    if margin is None:
        return None
    return 100.0 / (
        1.0
        + math.exp(-RELIABILITY_STEEPNESS_PER_DB * (margin - RELIABILITY_MIDPOINT_DB))
    )


def classify_link_balance(
    forward_snr: float | int | None = None,
    return_snr: float | int | None = None,
    *,
    spreading_factor: int | str | None = None,
    forward_quality: str | None = None,
    return_quality: str | None = None,
    forward_observations: int | None = None,
    return_observations: int | None = None,
) -> str:
    """Classify the directionality/balance of a link.

    Directional qualities may be passed directly (``forward_quality`` /
    ``return_quality``) or derived from the SNR samples plus a spreading
    factor — two SNR values alone cannot identify preset-dependent
    marginality, so when qualities are not supplied the *spreading_factor*
    (or the configured preset) selects the tier thresholds.

    States:
    - ``balanced``: both directions usable (good/fair), neither marginal.
    - ``asymmetric_marginal``: exactly one direction marginal, the other
      usable. (A marginal direction paired with a direction that has no
      usable SNR is not this state; see ``unidirectional``.)
    - ``marginal_both``: both directions marginal.
    - ``unidirectional``: only one direction has usable SNR and the other
      direction was never observed (observation count 0, or counts not
      tracked at all).
    - ``unknown``: a direction was observed (count > 0) but produced no
      usable SNR, or neither direction has usable SNR.
    """
    sf = resolve_spreading_factor(spreading_factor=spreading_factor)
    forward = (
        forward_quality
        if forward_quality is not None
        else classify_signal_quality(forward_snr, sf)
    )
    ret = (
        return_quality
        if return_quality is not None
        else classify_signal_quality(return_snr, sf)
    )

    forward_usable = forward in _USABLE_QUALITIES
    ret_usable = ret in _USABLE_QUALITIES
    forward_marginal = forward == QUALITY_MARGINAL
    ret_marginal = ret == QUALITY_MARGINAL

    if forward_usable and ret_usable:
        return BALANCE_BALANCED
    if forward_marginal and ret_marginal:
        return BALANCE_MARGINAL_BOTH
    if forward_marginal != ret_marginal and (forward_usable or ret_usable):
        return BALANCE_ASYMMETRIC_MARGINAL

    # At most one direction produced a usable/marginal reading.
    if not (forward_usable or forward_marginal or ret_usable or ret_marginal):
        return BALANCE_UNKNOWN
    other_observations = (
        return_observations
        if (forward_usable or forward_marginal)
        else forward_observations
    )
    if other_observations:
        # Observed but without usable SNR — cannot judge the balance.
        return BALANCE_UNKNOWN
    return BALANCE_UNIDIRECTIONAL


def get_quality_color(quality: str | None) -> str:
    """Stable color per quality tier (unknown color for anything else)."""
    return QUALITY_COLORS.get(quality or "", QUALITY_UNKNOWN_COLOR)
