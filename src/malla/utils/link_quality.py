"""Shared per-link quality enrichment for every RF link consumer.

The three data paths (traceroute hops, packet receptions, link analysis)
must expose the same metrics so observation volume stops being conflated
with link goodness (see LINK_QUALITY_CONTRACT.md). This module turns
aggregated directional measurements plus a channel/config hint into the
single canonical link payload every consumer serializes:

- ``channel_id`` / ``spreading_factor``: most recent channel hint and the
  resolved SF used by the model (channel names are hints only; an
  unrecognized name falls back to the configured preset).
- ``forward_quality`` / ``return_quality`` / ``quality``: directional and
  overall tiers. The overall tier describes the *worst observed*
  direction; with a single measured direction it describes that direction,
  and the missing direction stays unknown for the UI to flag.
- ``worst_snr``: plausibility-filtered minimum of the directional
  averages (never the combined average, which still contains the
  traceroute "unknown" sentinel).
- ``forward_estimated_reliability`` / ``return_estimated_reliability`` /
  ``estimated_reliability``: model estimates (0–100) for the worst
  observed direction — signal headroom only, not measured delivery.
- ``link_balance`` / ``is_bidirectional``: balance classification and
  observation coverage.
- ``observation_count`` / ``strength``: observation volume and the display
  width derived from it. ``strength`` is *pure* volume:
  ``clamp(1.5 + 2.5·log10(count), 1.5, 8.0)`` — thickness shows "how much
  data", color shows quality.

Missing measurements are ``None``/unknown, never 0.
"""

from __future__ import annotations

import math
from typing import Any

from .signal_quality import (
    calculate_estimated_reliability,
    classify_link_balance,
    classify_signal_quality,
    get_quality_color,
    is_plausible_snr,
    resolve_spreading_factor,
)

STRENGTH_MIN = 1.5
STRENGTH_MAX = 8.0
STRENGTH_LOG_SCALE = 2.5
STRENGTH_LOG_OFFSET = 1.5

ENRICHMENT_FIELDS = (
    "channel_id",
    "spreading_factor",
    "forward_quality",
    "return_quality",
    "quality",
    "worst_snr",
    "forward_estimated_reliability",
    "return_estimated_reliability",
    "estimated_reliability",
    "link_balance",
    "is_bidirectional",
    "observation_count",
    "strength",
    "forward_color",
    "return_color",
    "quality_color",
)


def observation_strength(observation_count: int | None) -> float:
    """Display width from observation volume alone (never from SNR).

    ``clamp(1.5 + 2.5·log10(count), 1.5, 8.0)``: 1 observation → 1.5,
    10 → 4.0, 100 → 6.5, clamped at 8.0. Zero/negative counts are guarded
    to the single-observation width before taking the logarithm.
    """
    count = max(int(observation_count or 0), 1)
    width = STRENGTH_LOG_OFFSET + STRENGTH_LOG_SCALE * math.log10(count)
    return round(min(STRENGTH_MAX, max(STRENGTH_MIN, width)), 1)


def enrich_link_quality(
    *,
    channel_id: str | None = None,
    forward_avg_snr: float | None = None,
    return_avg_snr: float | None = None,
    forward_observations: int = 0,
    return_observations: int = 0,
    total_observations: int | None = None,
) -> dict[str, Any]:
    """Build the canonical per-link quality payload from aggregates.

    Directional averages are the inputs the caller already exposes (i.e.
    rounded to one decimal), so every consumer computing enrichment from
    the same aggregates produces identical output. Pass observation
    counts separately from valid-SNR sample counts: an observed direction
    without usable SNR still counts as coverage.
    """
    channel_id = channel_id.strip() if isinstance(channel_id, str) else channel_id
    channel_id = channel_id or None  # blank channel names are no hint
    spreading_factor = resolve_spreading_factor(preset=channel_id)
    forward_quality = classify_signal_quality(forward_avg_snr, spreading_factor)
    return_quality = classify_signal_quality(return_avg_snr, spreading_factor)

    # worst_snr: min over the *valid* directional averages only.
    directional_averages: list[float] = []
    for average in (forward_avg_snr, return_avg_snr):
        if average is not None and is_plausible_snr(average):
            directional_averages.append(float(average))
    worst_snr = min(directional_averages) if directional_averages else None
    quality = classify_signal_quality(worst_snr, spreading_factor)

    forward_estimated_reliability = calculate_estimated_reliability(
        forward_avg_snr, spreading_factor
    )
    return_estimated_reliability = calculate_estimated_reliability(
        return_avg_snr, spreading_factor
    )
    directional_reliabilities: list[float] = [
        reliability
        for reliability in (
            forward_estimated_reliability,
            return_estimated_reliability,
        )
        if reliability is not None
    ]
    estimated_reliability = (
        min(directional_reliabilities) if directional_reliabilities else None
    )

    forward_count = int(forward_observations or 0)
    return_count = int(return_observations or 0)
    observation_count = (
        int(total_observations)
        if total_observations is not None
        else forward_count + return_count
    )

    return {
        "channel_id": channel_id,
        "spreading_factor": spreading_factor,
        "forward_quality": forward_quality,
        "return_quality": return_quality,
        "quality": quality,
        "worst_snr": round(worst_snr, 1) if worst_snr is not None else None,
        "forward_estimated_reliability": round(forward_estimated_reliability, 1)
        if forward_estimated_reliability is not None
        else None,
        "return_estimated_reliability": round(return_estimated_reliability, 1)
        if return_estimated_reliability is not None
        else None,
        "estimated_reliability": round(estimated_reliability, 1)
        if estimated_reliability is not None
        else None,
        "link_balance": classify_link_balance(
            forward_quality=forward_quality,
            return_quality=return_quality,
            forward_observations=forward_count,
            return_observations=return_count,
        ),
        "is_bidirectional": forward_count > 0 and return_count > 0,
        "observation_count": observation_count,
        "strength": observation_strength(observation_count),
        "forward_color": get_quality_color(forward_quality),
        "return_color": get_quality_color(return_quality),
        "quality_color": get_quality_color(quality),
    }
