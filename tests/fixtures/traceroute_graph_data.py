"""
Fixture data for traceroute graph testing.

The sample payload mirrors the production /api/traceroute/graph contract
(see ``TracerouteService.get_network_graph_data``): direct links carry the
full directional quality enrichment (computed here with the real
``enrich_link_quality`` so tiers, colors, reliability and observation
widths can never drift from the backend), nodes carry preset-aware
quality tiers, and indirect connections stay identifiable as inferred
paths with unknown RF quality.

Canonical link direction: ``source`` is the lower numeric node id and
``target`` the higher one; forward means source → target measured at the
target (the receiver), return means target → source measured at the
source — the same receiver rule as the map.

The direct links intentionally cover every quality/balance state at
LongFast (SF11, floor -17.5 dB: good ≥ -7.5, fair ≥ -13.5):

- Alpha↔Bravo: good in both directions (balanced).
- Bravo↔Charlie: forward marginal, return good (asymmetric).
- Charlie↔Delta: fair in both directions (balanced).
- Alpha↔Echo: only the forward direction ever observed.
- Bravo↔Fox: observed both ways, no usable SNR on either side (unknown).
- Charlie↔Echo: marginal in both directions.
- Delta↔Echo: ShortFast (SF7) asymmetric — preset-aware tiers differ from
  the same SNR values at SF11.
- Alpha↔Delta: good in both directions with a large observation volume.
- Echo↔Fox: only the return direction ever observed.
"""

import math
import time

from malla.utils.link_quality import enrich_link_quality, observation_strength
from malla.utils.signal_quality import (
    BALANCE_UNKNOWN,
    QUALITY_UNKNOWN,
    classify_signal_quality,
    get_quality_color,
    resolve_spreading_factor,
)

NODE_ALPHA = 0x11111111
NODE_BRAVO = 0x22222222
NODE_CHARLIE = 0x33333333
NODE_DELTA = 0x44444444
NODE_ECHO = 0x55555555
NODE_FOX = 0x66666666
NODE_GOLF = 0x77777777
NODE_HOTEL = 0x88888888

_LONGFAST = "LongFast"
_SHORTFAST = "ShortFast"


def _node(
    node_id,
    name,
    avg_snr,
    packet_count,
    connections,
    seen_seconds_ago,
    location=None,
    spreading_factor=11,
):
    quality = classify_signal_quality(
        avg_snr, resolve_spreading_factor(spreading_factor=spreading_factor)
    )
    info = {
        "id": node_id,
        "name": name,
        "avg_snr": avg_snr,
        "quality": quality,
        "quality_color": get_quality_color(quality),
        "connections": connections,
        "packet_count": packet_count,
        "size": min(20, max(5, math.log10(packet_count + 1) * 3)),
        "last_seen": int(time.time()) - seen_seconds_ago,
    }
    if location is not None:
        info["location"] = location
    return info


def _direct_link(
    source,
    target,
    *,
    forward_snr,
    return_snr,
    forward_count,
    return_count,
    channel_id=_LONGFAST,
    packet_count=None,
    seen_seconds_ago=600,
):
    """Build one canonical direct link with the real backend enrichment."""
    enrichment = enrich_link_quality(
        channel_id=channel_id,
        forward_avg_snr=forward_snr,
        return_avg_snr=return_snr,
        forward_observations=forward_count,
        return_observations=return_count,
    )
    samples = [snr for snr in (forward_snr, return_snr) if snr is not None]
    avg_snr = round(sum(samples) / len(samples), 1) if samples else None
    if packet_count is None:
        packet_count = forward_count + return_count
    return {
        "source": source,
        "target": target,
        "type": "direct",
        "avg_snr": avg_snr,
        "forward_avg_snr": forward_snr,
        "return_avg_snr": return_snr,
        "forward_count": forward_count,
        "return_count": return_count,
        "packet_count": packet_count,
        "last_seen": int(time.time()) - seen_seconds_ago,
        "last_packet_id": packet_count,
        **enrichment,
    }


def _indirect_link(
    source, target, *, hop_count, path_count, avg_snr, seen_seconds_ago=1500
):
    """Inferred multi-hop connection: RF quality stays unknown."""
    observation_count = hop_count * path_count
    unknown = get_quality_color(QUALITY_UNKNOWN)
    return {
        "source": source,
        "target": target,
        "type": "indirect",
        "is_inferred": True,
        "hop_count": hop_count,
        "path_count": path_count,
        "avg_snr": avg_snr,
        "channel_id": None,
        "spreading_factor": None,
        "forward_quality": QUALITY_UNKNOWN,
        "return_quality": QUALITY_UNKNOWN,
        "quality": QUALITY_UNKNOWN,
        "worst_snr": None,
        "forward_estimated_reliability": None,
        "return_estimated_reliability": None,
        "estimated_reliability": None,
        "link_balance": BALANCE_UNKNOWN,
        "is_bidirectional": None,
        "observation_count": observation_count,
        "strength": observation_strength(observation_count),
        "forward_color": unknown,
        "return_color": unknown,
        "quality_color": unknown,
        "last_seen": int(time.time()) - seen_seconds_ago,
        "last_packet_id": observation_count,
    }


def _build_sample_graph_data():
    nodes = [
        _node(
            NODE_ALPHA,
            "Test Node Alpha",
            -3.5,
            1250,
            6,
            3600,
            {"latitude": 40.7128, "longitude": -74.0060, "altitude": 10},
        ),
        _node(
            NODE_BRAVO,
            "Test Node Bravo",
            -6.2,
            2500,
            12,
            1800,
            {"latitude": 40.7589, "longitude": -73.9851, "altitude": 25},
        ),
        _node(
            NODE_CHARLIE,
            "Test Node Charlie",
            -12.8,
            800,
            4,
            900,
            {"latitude": 40.6782, "longitude": -73.9442, "altitude": 5},
        ),
        _node(
            NODE_DELTA,
            "Test Node Delta",
            -8.4,
            650,
            3,
            600,
            {"latitude": 40.7505, "longitude": -73.9934, "altitude": 15},
        ),
        _node(
            NODE_ECHO,
            "Test Node Echo",
            -15.5,
            1800,
            8,
            300,
            {"latitude": 40.7282, "longitude": -74.0776, "altitude": 30},
        ),
        _node(
            NODE_FOX,
            "Test Node Fox",
            -11.9,
            450,
            2,
            7200,
            {"latitude": 40.6892, "longitude": -74.0445, "altitude": 8},
        ),
        _node(
            NODE_GOLF,
            "Test Node Golf",
            -5.1,
            1100,
            7,
            1200,
            {"latitude": 40.7831, "longitude": -73.9712, "altitude": 20},
        ),
        # This node intentionally has no location to test mixed scenarios.
        _node(NODE_HOTEL, "Test Node Hotel", None, 2200, 9, 450),
    ]

    links = [
        # Good in both directions (balanced).
        _direct_link(
            NODE_ALPHA,
            NODE_BRAVO,
            forward_snr=-2.5,
            return_snr=-3.0,
            forward_count=12,
            return_count=9,
        ),
        # Asymmetric: forward marginal, return good.
        _direct_link(
            NODE_BRAVO,
            NODE_CHARLIE,
            forward_snr=-15.0,
            return_snr=-6.0,
            forward_count=6,
            return_count=7,
            seen_seconds_ago=800,
        ),
        # Fair in both directions (balanced).
        _direct_link(
            NODE_CHARLIE,
            NODE_DELTA,
            forward_snr=-10.0,
            return_snr=-11.0,
            forward_count=4,
            return_count=5,
            seen_seconds_ago=400,
        ),
        # One direction: only the forward direction ever observed.
        _direct_link(
            NODE_ALPHA,
            NODE_ECHO,
            forward_snr=-1.0,
            return_snr=None,
            forward_count=8,
            return_count=0,
            seen_seconds_ago=700,
        ),
        # Unknown: observed both ways, no usable SNR on either side.
        _direct_link(
            NODE_BRAVO,
            NODE_FOX,
            forward_snr=None,
            return_snr=None,
            forward_count=3,
            return_count=2,
            seen_seconds_ago=1200,
        ),
        # Marginal in both directions.
        _direct_link(
            NODE_CHARLIE,
            NODE_ECHO,
            forward_snr=-16.5,
            return_snr=-17.0,
            forward_count=5,
            return_count=4,
            seen_seconds_ago=900,
        ),
        # ShortFast (SF7): same ballpark SNRs classify differently —
        # forward marginal, return fair (preset-aware asymmetry).
        _direct_link(
            NODE_DELTA,
            NODE_ECHO,
            forward_snr=-10.0,
            return_snr=-2.0,
            forward_count=6,
            return_count=5,
            channel_id=_SHORTFAST,
            seen_seconds_ago=350,
        ),
        # High observation volume in both directions (thickness case).
        _direct_link(
            NODE_ALPHA,
            NODE_DELTA,
            forward_snr=-4.0,
            return_snr=-4.5,
            forward_count=40,
            return_count=35,
            seen_seconds_ago=200,
        ),
        # One direction, reversed: only the return direction observed.
        _direct_link(
            NODE_ECHO,
            NODE_FOX,
            forward_snr=None,
            return_snr=-8.0,
            forward_count=0,
            return_count=5,
            seen_seconds_ago=500,
        ),
        _direct_link(
            NODE_BRAVO,
            NODE_GOLF,
            forward_snr=-7.8,
            return_snr=-9.9,
            forward_count=15,
            return_count=11,
            seen_seconds_ago=300,
        ),
    ]

    indirect_connections = [
        _indirect_link(
            NODE_ALPHA,
            NODE_FOX,
            hop_count=2,
            path_count=18,
            avg_snr=-9.4,
        ),
        _indirect_link(
            NODE_CHARLIE,
            NODE_FOX,
            hop_count=3,
            path_count=7,
            avg_snr=-12.2,
            seen_seconds_ago=1800,
        ),
    ]

    return {
        "nodes": nodes,
        "links": links,
        "indirect_connections": indirect_connections,
        "stats": {
            "links_found": len(links),
            "packets_analyzed": 3250,
            "packets_with_rf_hops": 3250,
            "total_rf_hops": 288,
            "links_filtered_by_snr": 12,
            "links_filtered_due_to_snr_0": 3,
        },
        "filters": {"hours": 24, "min_snr": -200, "include_indirect": True},
    }


# Sample traceroute graph data for testing
SAMPLE_GRAPH_DATA = _build_sample_graph_data()


def get_sample_graph_data():
    """Return a copy of the sample graph data."""
    import copy

    return copy.deepcopy(SAMPLE_GRAPH_DATA)
