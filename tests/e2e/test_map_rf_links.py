"""
E2E tests for per-direction RF link quality rendering on the map.

Two complementary strategies keep the split-half rendering honest:

- The /api/locations response is intercepted with a deterministic payload so
  the split-half rendering can be verified exactly: midpoint color
  orientation, unknown-direction styling, link-type dash patterns, popup
  values (including a real 0 dB SNR, 0.0% reliability, marginal-both and
  unknown/unknown links) and repeated redraw/filter cycles — with role
  filters that genuinely remove nodes and links — without orphaned segments.
- A separately seeded Flask server serves /api/locations through the real
  backend aggregation so the payload contract (directional counts,
  once-at-the-end rounding, reliability tiers) cannot drift behind the
  intercepted fixtures.
"""

import json
import os
import socket
import sqlite3
import tempfile
import threading
import time
import urllib.request

import pytest
from playwright.sync_api import Page, Route, expect

from malla.utils.signal_quality import QUALITY_COLORS

DEFAULT_TIMEOUT = 20000  # ms

# Straight-line distances are large enough that the initial fitBounds keeps
# the markers out of spiderfy clusters, so markers stay individually
# clickable.
NODE_ALPHA = {"node_id": 100, "lat": 40.00, "lng": -105.00}
NODE_BRAVO = {"node_id": 200, "lat": 40.00, "lng": -104.90}
NODE_CHARLIE = {"node_id": 300, "lat": 40.05, "lng": -105.05}
NODE_DELTA = {"node_id": 400, "lat": 40.08, "lng": -104.95}
# The only ROUTER in the fixture: selecting ROUTER in the role filter keeps
# this node alone, while CLIENT removes it together with both of its links.
NODE_ECHO = {"node_id": 500, "lat": 40.02, "lng": -105.10}

LINK_SEGMENTS_JS = """
() => {
    const dump = (groups) => groups.flatMap(group => group.getLayers().map(layer => ({
        color: layer.options.color,
        dashArray: layer.options.dashArray || null,
        opacity: layer.options.opacity,
        weight: layer.options.weight,
        latlngs: layer.getLatLngs().map(ll => [ll.lat, ll.lng]),
    })));
    return { traceroute: dump(tracerouteLinks), packet: dump(packetLinks) };
}
"""

LINK_PATH_COUNT_JS = f"""
() => {{
    const colors = new Set({json.dumps(sorted(QUALITY_COLORS.values()))});
    return Array.from(document.querySelectorAll('.leaflet-overlay-pane path'))
        .filter(p => colors.has(p.getAttribute('stroke'))).length;
}}
"""


def _make_node(node, name, short, now, role="CLIENT"):
    return {
        "node_id": node["node_id"],
        "display_name": name,
        "short_name": short,
        "role": role,
        "hw_model": "TBEAM",
        "latitude": node["lat"],
        "longitude": node["lng"],
        "timestamp": now,
        "precision_meters": 5,
        "primary_channel": "LongFast",
    }


def _direction(snr, count, quality, reliability, rssi=None):
    return {
        "avg_snr": snr,
        "avg_rssi": rssi,
        "count": count,
        "quality": quality,
        "color": QUALITY_COLORS[quality],
        "estimated_reliability": reliability,
    }


def _make_link(from_node, to_node, link_type, forward, ret, **extra):
    now = time.time()
    link = {
        "from_node_id": from_node["node_id"],
        "to_node_id": to_node["node_id"],
        "link_type": link_type,
        "forward_avg_snr": forward["avg_snr"],
        "forward_avg_rssi": forward["avg_rssi"],
        "forward_count": forward["count"],
        "forward_quality": forward["quality"],
        "forward_color": forward["color"],
        "forward_estimated_reliability": forward["estimated_reliability"],
        "return_avg_snr": ret["avg_snr"],
        "return_avg_rssi": ret["avg_rssi"],
        "return_count": ret["count"],
        "return_quality": ret["quality"],
        "return_color": ret["color"],
        "return_estimated_reliability": ret["estimated_reliability"],
        "channel_id": "LongFast",
        "spreading_factor": 11,
        "last_seen": now - 600,
        "last_seen_str": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 600)),
    }
    link.update(extra)
    return link


def build_locations_payload():
    """Deterministic /api/locations payload covering every link state.

    - T1 traceroute Alpha→Bravo: forward good, return marginal (asymmetric)
    - T2 traceroute Bravo→Charlie: forward fair, return never observed
    - P1 packet Alpha→Charlie: both directions measured, RSSI available
    - P2 packet Alpha→Delta: forward observed without usable SNR, return good
    - P3 packet Echo→Bravo: forward average is exactly 0.0 dB — a real
      measurement, not missing data
    - P4 packet Charlie→Delta (ShortFast, SF7): both directions marginal and
      reliability rounded to 0.0% (marginal_both)
    - P5 packet Echo→Alpha: observed in both directions but without usable
      SNR on either side (unknown/unknown)
    """
    now = time.time()
    nodes = [
        _make_node(NODE_ALPHA, "Node Alpha", "ALPH", now),
        _make_node(NODE_BRAVO, "Node Bravo", "BRAV", now),
        _make_node(NODE_CHARLIE, "Node Charlie", "CHAR", now),
        _make_node(NODE_DELTA, "Node Delta", "DELT", now),
        _make_node(NODE_ECHO, "Node Echo", "ECHO", now, role="ROUTER"),
    ]

    t1 = _make_link(
        NODE_ALPHA,
        NODE_BRAVO,
        "traceroute",
        _direction(12.5, 10, "good", 100.0),
        _direction(-15.0, 4, "marginal", 50.0),
        quality="marginal",
        quality_color=QUALITY_COLORS["marginal"],
        worst_snr=-15.0,
        estimated_reliability=50.0,
        link_balance="asymmetric_marginal",
        is_bidirectional=True,
        observation_count=14,
        total_hops_seen=14,
        strength=4.0,
        avg_snr=-1.3,
    )
    t2 = _make_link(
        NODE_BRAVO,
        NODE_CHARLIE,
        "traceroute",
        _direction(-10.0, 5, "fair", 95.3),
        _direction(None, 0, "unknown", None),
        quality="fair",
        quality_color=QUALITY_COLORS["fair"],
        worst_snr=-10.0,
        estimated_reliability=95.3,
        link_balance="unidirectional",
        is_bidirectional=False,
        observation_count=5,
        total_hops_seen=5,
        strength=3.2,
        avg_snr=-10.0,
    )
    p1 = _make_link(
        NODE_ALPHA,
        NODE_CHARLIE,
        "packet",
        _direction(8.5, 3, "good", 100.0, rssi=-80.5),
        _direction(-9.0, 2, "fair", 91.9, rssi=-95.1),
        quality="fair",
        quality_color=QUALITY_COLORS["fair"],
        worst_snr=-9.0,
        estimated_reliability=91.9,
        link_balance="balanced",
        is_bidirectional=True,
        observation_count=5,
        total_hops_seen=5,
        strength=3.2,
        avg_snr=-0.3,
        avg_rssi=-87.8,
    )
    p2 = _make_link(
        NODE_ALPHA,
        NODE_DELTA,
        "packet",
        _direction(None, 4, "unknown", None, rssi=-101.5),
        _direction(6.0, 7, "good", 99.1, rssi=-88.0),
        quality="unknown",
        quality_color=QUALITY_COLORS["unknown"],
        worst_snr=6.0,
        estimated_reliability=99.1,
        link_balance="unknown",
        is_bidirectional=True,
        observation_count=11,
        total_hops_seen=11,
        strength=3.7,
        avg_snr=6.0,
        avg_rssi=-94.8,
    )
    # Forward: a genuine 0.0 dB average. At SF11 that is 20 dB of fade
    # margin — a good link — and must render as measured, never as the
    # unknown styling a falsy zero check would produce.
    p3 = _make_link(
        NODE_ECHO,
        NODE_BRAVO,
        "packet",
        _direction(0.0, 3, "good", 100.0, rssi=-84.0),
        _direction(5.5, 2, "good", 100.0, rssi=-80.0),
        quality="good",
        quality_color=QUALITY_COLORS["good"],
        worst_snr=0.0,
        estimated_reliability=100.0,
        link_balance="balanced",
        is_bidirectional=True,
        observation_count=5,
        total_hops_seen=5,
        strength=3.2,
        avg_snr=2.8,
        avg_rssi=-82.0,
    )
    # ShortFast resolves to SF7 (floor -7.5 dB): both directions sit ~13 dB
    # below the floor, so each estimates 0.0% reliability and the link is
    # marginal in both directions.
    p4 = _make_link(
        NODE_CHARLIE,
        NODE_DELTA,
        "packet",
        _direction(-21.0, 2, "marginal", 0.0, rssi=-100.0),
        _direction(-20.5, 1, "marginal", 0.0, rssi=-101.0),
        quality="marginal",
        quality_color=QUALITY_COLORS["marginal"],
        worst_snr=-21.0,
        estimated_reliability=0.0,
        link_balance="marginal_both",
        is_bidirectional=True,
        observation_count=3,
        total_hops_seen=3,
        strength=2.7,
        avg_snr=-20.8,
        avg_rssi=-100.5,
        channel_id="ShortFast",
        spreading_factor=7,
    )
    # Observed in both directions (counts > 0, RSSI present) but neither
    # side produced a usable SNR sample.
    p5 = _make_link(
        NODE_ECHO,
        NODE_ALPHA,
        "packet",
        _direction(None, 2, "unknown", None, rssi=-98.2),
        _direction(None, 1, "unknown", None, rssi=-103.7),
        quality="unknown",
        quality_color=QUALITY_COLORS["unknown"],
        worst_snr=None,
        estimated_reliability=None,
        link_balance="unknown",
        is_bidirectional=True,
        observation_count=3,
        total_hops_seen=3,
        strength=2.7,
        avg_snr=None,
        avg_rssi=None,
    )

    return {
        "locations": nodes,
        "traceroute_links": [t1, t2],
        "packet_links": [p1, p2, p3, p4, p5],
    }


@pytest.fixture()
def rf_link_map_page(page: Page, test_server_url: str):
    """Map page served by the real test app but with deterministic links."""

    def fulfill_locations(route: Route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(build_locations_payload()),
        )

    page.route("**/api/locations*", fulfill_locations)
    page.goto(f"{test_server_url}/map")
    page.wait_for_selector("#mapLoading", state="hidden", timeout=DEFAULT_TIMEOUT)
    # Wait for the first fitBounds + initial link draw.
    page.wait_for_function(
        "() => tracerouteLinks.length > 0 && tracerouteLinks[0].getLayers().length === 2",
        timeout=DEFAULT_TIMEOUT,
    )
    return page


def open_link_popup(page: Page, store: str, color: str):
    """Open the popup of a link half by its stroke color via Leaflet events.

    The event is fired with propagation enabled, exactly like a real DOM
    click dispatched through the map (so the shared feature-group popup
    bound by bindPopup receives it)."""
    page.evaluate(
        """([store, color]) => {
            if (map) { map.closePopup(); }
            const groups = store === 'packetLinks' ? packetLinks : tracerouteLinks;
            const group = groups.find(g =>
                g.getLayers().some(l => l.options.color === color));
            const half = group.getLayers().find(l => l.options.color === color);
            half.fire('click', { latlng: half.getCenter() }, true);
        }""",
        [store, color],
    )
    expect(page.locator(".leaflet-popup-content").last).to_be_visible(
        timeout=DEFAULT_TIMEOUT
    )
    return page.locator(".leaflet-popup-content").last


def open_link_popup_between(page: Page, store: str, pos_a, pos_b):
    """Open the popup of the link joining two node positions.

    Color is not always unique across links (several good halves exist), so
    identify the link by its endpoint node positions instead."""
    page.evaluate(
        """([store, a, b]) => {
            if (map) { map.closePopup(); }
            const groups = store === 'packetLinks' ? packetLinks : tracerouteLinks;
            const near = (ll, p) =>
                Math.abs(ll.lat - p[0]) < 1e-4 && Math.abs(ll.lng - p[1]) < 1e-4;
            const group = groups.find(g => {
                const points = g.getLayers().flatMap(l => l.getLatLngs());
                return points.some(ll => near(ll, a)) && points.some(ll => near(ll, b));
            });
            const half = group.getLayers()[0];
            half.fire('click', { latlng: half.getCenter() }, true);
        }""",
        [store, list(pos_a), list(pos_b)],
    )
    expect(page.locator(".leaflet-popup-content").last).to_be_visible(
        timeout=DEFAULT_TIMEOUT
    )
    return page.locator(".leaflet-popup-content").last


def popup_text(popup) -> str:
    """Non-None text content of the popup locator."""
    text = popup.text_content()
    assert text is not None, "Popup should have text content"
    return text


def link_halves(page: Page, store: str, pos_a, pos_b, color: str):
    """Segments of the link joining two node positions, filtered by color.

    Each half spans one endpoint and the midpoint, so a half belongs to the
    link when it touches either endpoint. The color filter disambiguates
    against other links that share a node."""
    segments = page.evaluate(LINK_SEGMENTS_JS)[
        "packet" if store == "packetLinks" else "traceroute"
    ]
    endpoints = (pos_a, pos_b)

    def touches_link(seg):
        return any(
            abs(ll[0] - pos[0]) < 1e-4 and abs(ll[1] - pos[1]) < 1e-4
            for ll in seg["latlngs"]
            for pos in endpoints
        )

    halves = [s for s in segments if s["color"] == color and touches_link(s)]
    assert len(halves) == 2, (
        f"Expected the two {color} halves of the link between {pos_a} and "
        f"{pos_b}, got: {segments}"
    )
    return halves


def enable_packet_links(page: Page) -> None:
    """Show packet links and wait until all five fixture links are drawn."""
    page.locator("#packetLinksCheckbox").check()
    page.wait_for_function("() => packetLinks.length === 5", timeout=DEFAULT_TIMEOUT)


class TestMapDirectionalRfLinks:
    """Per-direction link quality rendering on the map."""

    @pytest.mark.e2e
    def test_halves_colored_by_receiver_quality_at_midpoint(
        self, rf_link_map_page: Page
    ):
        """T1 (Alpha→Bravo): forward good half touches Bravo, return
        marginal half touches Alpha, joined at the midpoint."""
        segments = rf_link_map_page.evaluate(LINK_SEGMENTS_JS)["traceroute"]
        t1_halves = [
            s
            for s in segments
            if s["color"] in (QUALITY_COLORS["good"], QUALITY_COLORS["marginal"])
        ]
        assert len(t1_halves) == 2, f"Expected T1 good+marginal halves, got: {segments}"

        def mean_lng(seg):
            return sum(ll[1] for ll in seg["latlngs"]) / len(seg["latlngs"])

        forward_half = next(
            s for s in t1_halves if s["color"] == QUALITY_COLORS["good"]
        )
        return_half = next(
            s for s in t1_halves if s["color"] == QUALITY_COLORS["marginal"]
        )

        # Alpha (-105.0) is west, Bravo (-104.9) is east; midpoint ≈ -104.95.
        assert mean_lng(forward_half) > -104.95, (
            "Forward (good) half must be the eastern half nearest Bravo"
        )
        assert mean_lng(return_half) < -104.95, (
            "Return (marginal) half must be the western half nearest Alpha"
        )
        # The forward half joins the midpoint and ends at Bravo; the return
        # half starts at Alpha and ends at the midpoint.
        assert abs(forward_half["latlngs"][1][1] - NODE_BRAVO["lng"]) < 0.001
        assert abs(forward_half["latlngs"][0][1] - (-104.95)) < 0.001
        assert abs(return_half["latlngs"][0][1] - NODE_ALPHA["lng"]) < 0.001
        assert abs(return_half["latlngs"][1][1] - (-104.95)) < 0.001
        # Both halves share the backend observation-volume strength.
        assert forward_half["weight"] == 4.0
        assert return_half["weight"] == 4.0

    @pytest.mark.e2e
    def test_unknown_direction_half_dashed_and_faded(self, rf_link_map_page: Page):
        """T2 (Bravo→Charlie): the unobserved return direction renders with
        the unknown color, a dash pattern and reduced opacity — never an
        average of the measured direction."""
        segments = rf_link_map_page.evaluate(LINK_SEGMENTS_JS)["traceroute"]
        unknown = [s for s in segments if s["color"] == QUALITY_COLORS["unknown"]]
        assert len(unknown) == 1, f"Expected exactly one unknown half, got: {segments}"
        measured = next(s for s in segments if s["color"] == QUALITY_COLORS["fair"])

        assert unknown[0]["dashArray"] == "8, 6"
        assert unknown[0]["opacity"] == pytest.approx(0.6 * 0.55, abs=0.01)
        assert measured["dashArray"] is None
        assert measured["opacity"] == pytest.approx(0.6, abs=0.01)

        # The unknown half is the one nearest Bravo (from_node): forward
        # (fair) is measured at Charlie (to_node).
        unknown_endpoints = {round(ll[1], 3) for ll in unknown[0]["latlngs"]}
        assert round(NODE_BRAVO["lng"], 3) in unknown_endpoints

    @pytest.mark.e2e
    def test_packet_links_keep_distinct_dash_pattern(self, rf_link_map_page: Page):
        """Packet halves keep their dot dash even when both directions are
        measured; fully-measured traceroute halves stay solid."""
        rf_link_map_page.locator("#packetLinksCheckbox").check()
        rf_link_map_page.wait_for_function(
            "() => packetLinks.length > 0", timeout=DEFAULT_TIMEOUT
        )

        segments = rf_link_map_page.evaluate(LINK_SEGMENTS_JS)
        for seg in segments["packet"]:
            assert seg["dashArray"] == "3, 6", (
                "Every packet half keeps the distinct packet dash pattern"
            )

        traceroute_solid = [
            s for s in segments["traceroute"] if s["color"] != QUALITY_COLORS["unknown"]
        ]
        assert traceroute_solid, "Expected measured traceroute halves"
        for seg in traceroute_solid:
            assert seg["dashArray"] is None

    @pytest.mark.e2e
    def test_popup_directional_values_and_warnings(self, rf_link_map_page: Page):
        """Popup identifies the receiver per direction, shows SNR, counts,
        reliability and the asymmetric warning; traceroute popups show no
        RSSI (unavailable there)."""
        popup = open_link_popup(
            rf_link_map_page, "tracerouteLinks", QUALITY_COLORS["marginal"]
        )
        text = popup_text(popup)

        assert "Traceroute RF Hop" in text
        assert "Node Alpha → Node Bravo (received at Node Bravo)" in text
        assert "12.5 dB · 10 observations · est. 100.0%" in text
        assert "Node Bravo → Node Alpha (received at Node Alpha)" in text
        assert "-15.0 dB · 4 observations · est. 50.0%" in text
        assert "Estimated reliability: 50.0%" in text
        assert "Asymmetric — one direction marginal" in text
        assert "Worst-direction SNR: -15.0 dB" in text
        assert "Observations: 14" in text
        assert "Quality model: SF11 · LongFast" in text
        # RSSI is only rendered where the backend provides it.
        assert "dBm" not in text
        # History and line-of-sight actions are retained.
        assert "View History" in text
        assert "Line of Sight" in text

    @pytest.mark.e2e
    def test_popup_distinguishes_unobserved_from_no_snr(self, rf_link_map_page: Page):
        """'Only one direction observed' (unidirectional) and 'observed,
        SNR unavailable' render as distinct states."""
        # T2: return direction never observed.
        popup_t2 = open_link_popup(
            rf_link_map_page, "tracerouteLinks", QUALITY_COLORS["fair"]
        )
        text_t2 = popup_text(popup_t2)
        assert "No observations in this direction" in text_t2
        assert "SNR unavailable" not in text_t2
        assert "Only one direction observed" in text_t2

        # P2: forward direction observed but produced no usable SNR.
        rf_link_map_page.locator("#packetLinksCheckbox").check()
        rf_link_map_page.wait_for_function(
            "() => packetLinks.length > 0", timeout=DEFAULT_TIMEOUT
        )
        popup_p2 = open_link_popup(
            rf_link_map_page, "packetLinks", QUALITY_COLORS["unknown"]
        )
        text_p2 = popup_text(popup_p2)
        assert "Direct Packet Link" in text_p2
        assert "SNR unavailable" in text_p2
        assert "-101.5 dBm" in text_p2
        assert "No observations in this direction" not in text_p2
        assert "Observed, but SNR unavailable for balance assessment" in text_p2

    @pytest.mark.e2e
    def test_real_zero_snr_renders_as_measurement(self, rf_link_map_page: Page):
        """P3 (Echo→Bravo): a forward average of exactly 0.0 dB is a real
        measurement. Both halves render measured (packet dash, full
        opacity) and the popup shows 0.0 dB with its reliability — never
        the unknown styling a falsy zero check would produce."""
        page = rf_link_map_page
        enable_packet_links(page)

        p3_halves = link_halves(
            page,
            "packetLinks",
            (NODE_ECHO["lat"], NODE_ECHO["lng"]),
            (NODE_BRAVO["lat"], NODE_BRAVO["lng"]),
            QUALITY_COLORS["good"],
        )
        for seg in p3_halves:
            assert seg["dashArray"] == "3, 6"
            assert seg["opacity"] == pytest.approx(0.6, abs=0.01), (
                "A 0.0 dB average must not be faded like an unknown direction"
            )

        popup = open_link_popup_between(
            page,
            "packetLinks",
            (NODE_ECHO["lat"], NODE_ECHO["lng"]),
            (NODE_BRAVO["lat"], NODE_BRAVO["lng"]),
        )
        text = popup_text(popup)
        assert "Node Echo → Node Bravo (received at Node Bravo)" in text
        assert "0.0 dB · -84.0 dBm · 3 observations · est. 100.0%" in text
        assert "Node Bravo → Node Echo (received at Node Echo)" in text
        assert "5.5 dB · -80.0 dBm · 2 observations · est. 100.0%" in text
        assert "Balanced — both directions usable" in text
        assert "Worst-direction SNR: 0.0 dB" in text
        assert "Observations: 5" in text
        assert "SNR unavailable" not in text

    @pytest.mark.e2e
    def test_marginal_both_with_zero_percent_reliability(self, rf_link_map_page: Page):
        """P4 (Charlie→Delta, ShortFast): both directions marginal renders
        'Fragile', shows est. 0.0% per direction and overall, and names the
        SF7 preset that produced the estimate."""
        page = rf_link_map_page
        enable_packet_links(page)

        p4_halves = link_halves(
            page,
            "packetLinks",
            (NODE_CHARLIE["lat"], NODE_CHARLIE["lng"]),
            (NODE_DELTA["lat"], NODE_DELTA["lng"]),
            QUALITY_COLORS["marginal"],
        )
        for seg in p4_halves:
            # Both directions were measured: full opacity, packet dash.
            assert seg["dashArray"] == "3, 6"
            assert seg["opacity"] == pytest.approx(0.6, abs=0.01)

        popup = open_link_popup_between(
            page,
            "packetLinks",
            (NODE_CHARLIE["lat"], NODE_CHARLIE["lng"]),
            (NODE_DELTA["lat"], NODE_DELTA["lng"]),
        )
        text = popup_text(popup)
        assert "Fragile — both directions marginal" in text
        assert "Estimated reliability: 0.0%" in text
        assert "-21.0 dB · -100.0 dBm · 2 observations · est. 0.0%" in text
        assert "-20.5 dB · -101.0 dBm · 1 observation · est. 0.0%" in text
        assert "Quality model: SF7 · ShortFast" in text
        assert "Worst-direction SNR: -21.0 dB" in text

    @pytest.mark.e2e
    def test_both_directions_unknown_state(self, rf_link_map_page: Page):
        """P5 (Echo↔Alpha): observed in both directions without usable SNR
        on either side. Both halves render unknown and faded, both
        direction blocks say 'SNR unavailable', and the balance note
        explains that the balance cannot be judged."""
        page = rf_link_map_page
        enable_packet_links(page)

        p5_halves = link_halves(
            page,
            "packetLinks",
            (NODE_ECHO["lat"], NODE_ECHO["lng"]),
            (NODE_ALPHA["lat"], NODE_ALPHA["lng"]),
            QUALITY_COLORS["unknown"],
        )
        for seg in p5_halves:
            assert seg["dashArray"] == "3, 6"
            assert seg["opacity"] == pytest.approx(0.6 * 0.55, abs=0.01)

        popup = open_link_popup_between(
            page,
            "packetLinks",
            (NODE_ECHO["lat"], NODE_ECHO["lng"]),
            (NODE_ALPHA["lat"], NODE_ALPHA["lng"]),
        )
        text = popup_text(popup)
        # Two direction blocks plus the balance note mention the phrase.
        assert text.count("SNR unavailable") == 3
        assert "SNR unavailable · -98.2 dBm · 2 observations" in text
        assert "SNR unavailable · -103.7 dBm · 1 observation" in text
        assert "Observed, but SNR unavailable for balance assessment" in text
        assert "Estimated reliability:" not in text
        assert "est." not in text
        assert "Observations: 3" in text

    @pytest.mark.e2e
    def test_redraw_and_filters_leave_no_orphan_segments(self, rf_link_map_page: Page):
        """Toggles, filters and node selection redraw both halves together:
        after every cycle the DOM contains exactly the expected segments.
        The role filter genuinely removes nodes — Echo is the only ROUTER —
        so links touching removed nodes disappear entirely."""
        page = rf_link_map_page

        def link_paths() -> int:
            return int(page.evaluate(LINK_PATH_COUNT_JS))

        page.wait_for_timeout(500)  # let the initial draw settle
        assert link_paths() == 4, "2 traceroute links × 2 halves"

        # Toggle traceroute links off/on.
        page.locator("#tracerouteLinksCheckbox").uncheck()
        page.wait_for_function("() => tracerouteLinks.length === 0")
        assert link_paths() == 0
        page.locator("#tracerouteLinksCheckbox").check()
        page.wait_for_function("() => tracerouteLinks.length === 2")
        assert link_paths() == 4

        # Enable packet links on top (P1–P5).
        page.locator("#packetLinksCheckbox").check()
        page.wait_for_function("() => packetLinks.length === 5")
        assert link_paths() == 14, "7 links × 2 halves"

        # ROUTER keeps Echo alone: every link loses at least one endpoint,
        # so no segment may survive.
        page.locator("#roleFilter").select_option("ROUTER")
        page.locator("#locationFilterForm button[type='submit']").click()
        page.wait_for_function(
            "() => nodeData.length === 1 && nodeMarkers.length === 1"
            " && tracerouteLinks.length === 0 && packetLinks.length === 0"
        )
        assert link_paths() == 0

        # CLIENT hides Echo and with him both links that touch him (P3, P5).
        page.locator("#roleFilter").select_option("CLIENT")
        page.locator("#locationFilterForm button[type='submit']").click()
        page.wait_for_function(
            "() => nodeData.length === 4 && nodeMarkers.length === 4"
            " && tracerouteLinks.length === 2 && packetLinks.length === 3"
        )
        assert link_paths() == 10

        # All Roles restores the full network.
        page.locator("#roleFilter").select_option("")
        page.locator("#locationFilterForm button[type='submit']").click()
        page.wait_for_function(
            "() => nodeData.length === 5 && tracerouteLinks.length === 2"
            " && packetLinks.length === 5"
        )
        assert link_paths() == 14

        # Selecting a node redraws only its 1-hop links (Alpha has T1, P1,
        # P2 and P5).
        page.evaluate("() => selectNode(nodeData.find(n => n.node_id === 100))")
        expect(page.locator("#hopDepthSection")).to_be_visible(timeout=DEFAULT_TIMEOUT)
        page.wait_for_function(
            "() => tracerouteLinks.length === 1 && packetLinks.length === 3"
        )
        assert link_paths() == 8, "4 links touching the selected node × 2 halves"

        # Clearing the selection restores the full network without orphans.
        page.locator("#clearSelection").click()
        page.wait_for_function(
            "() => tracerouteLinks.length === 2 && packetLinks.length === 5"
        )
        assert link_paths() == 14

    @pytest.mark.e2e
    def test_midpoint_handles_antimeridian(self, rf_link_map_page: Page):
        """The midpoint is computed in projected space and unwraps
        longitudes so links crossing ±180° bend the short way — from either
        endpoint order, and in the actual drawn segment geometry."""
        midpoint_checks = rf_link_map_page.evaluate(
            """() => ({
                east: rfLinkMidpoint([0, 170], [0, -170]),
                west: rfLinkMidpoint([0, -170], [0, 170]),
                normal: rfLinkMidpoint([40, -105], [40, -104.9]),
            })"""
        )
        # 170 and -170 are 20° apart across the antimeridian, so the
        # midpoint lands on the seam itself from either direction.
        for direction in ("east", "west"):
            assert abs(abs(midpoint_checks[direction][1]) - 180) < 0.001
            assert midpoint_checks[direction][0] == pytest.approx(0, abs=0.001)
        # Regular midpoint stays at the linear average at equal latitude.
        assert midpoint_checks["normal"][1] == pytest.approx(-104.95, abs=0.001)

        # Draw real link layers across the seam and verify the geometry of
        # every polyline: each half must span its short 10° arc (never the
        # ~350° long way) and be replicated onto both sides of the seam.
        drawn = rf_link_map_page.evaluate(
            """() => {
                const link = {
                    from_node_id: 100, to_node_id: 200, link_type: 'traceroute',
                };
                const draw = (fromPos, toPos) =>
                    createRfLinkLayer(link, fromPos, toPos, null, false)
                        .getLayers()
                        .map(layer => layer.getLatLngs().map(ll => [ll.lat, ll.lng]));
                return {
                    east: draw([0, 170], [0, -170]),
                    west: draw([0, -170], [0, 170]),
                };
            }"""
        )

        def norm_lng(lng):
            return ((lng + 180) % 360) - 180

        for direction, segments in drawn.items():
            assert len(segments) == 4, (
                f"{direction}: expected 2 halves × 2 world copies, got: {segments}"
            )
            for seg in segments:
                span = abs(seg[0][1] - seg[1][1])
                assert span == pytest.approx(10, abs=0.01), (
                    f"{direction} segment {seg} spans {span:.1f}° "
                    "instead of the 10° short way"
                )
                for lat in (seg[0][0], seg[1][0]):
                    assert lat == pytest.approx(0, abs=0.001)
            # The four segments chain node → seam → node on each side, so
            # the normalized endpoints are exactly the two nodes plus the
            # seam midpoint, each visited twice.
            endpoints = sorted(
                round(norm_lng(lng), 3) for seg in segments for _, lng in seg
            )
            assert endpoints == pytest.approx(
                [-180, -180, -180, -180, -170, -170, 170, 170]
            ), f"{direction}: segments do not join at the seam: {segments}"

    @pytest.mark.e2e
    def test_legend_documents_quality_language(self, rf_link_map_page: Page):
        """The legend explains quality colors, observation width and the
        link-type / missing-data patterns."""
        legend = rf_link_map_page.locator(".legend-content")
        expect(legend).to_be_visible()
        for text in (
            "Good quality (half nearest receiver)",
            "Marginal quality",
            "Direction not observed (dashed, faded)",
            "Asymmetric link — color changes midway",
            "Thicker = more observations",
            "Traceroute link (solid)",
            "Packet link (dotted)",
        ):
            expect(legend).to_contain_text(text)


# ---------------------------------------------------------------------------
# Backend-produced payload: a real Flask app aggregating a seeded database
# ---------------------------------------------------------------------------

# Gateway that only receives position broadcasts; it must never appear in a
# packet link because those packets travel one hop.
BACKEND_POSITION_GATEWAY = 0x00000999

BACKEND_LINK_NODES = {
    "ping": {
        "node_id": 0x11110001,
        "name": "Backend Ping",
        "short": "BPIN",
        "lat": 40.00,
        "lng": -105.00,
    },
    "pong": {
        "node_id": 0x22220002,
        "name": "Backend Pong",
        "short": "BPON",
        "lat": 40.00,
        "lng": -104.90,
    },
    "rack": {
        "node_id": 0x33330003,
        "name": "Backend Rack",
        "short": "BRAC",
        "lat": 40.05,
        "lng": -105.05,
    },
    "sled": {
        "node_id": 0x44440004,
        "name": "Backend Sled",
        "short": "BSLE",
        "lat": 40.05,
        "lng": -104.95,
    },
    "unit": {
        "node_id": 0x55550005,
        "name": "Backend Unit",
        "short": "BUNI",
        "lat": 40.10,
        "lng": -105.00,
    },
    "victor": {
        "node_id": 0x66660006,
        "name": "Backend Victor",
        "short": "BVIC",
        "lat": 40.10,
        "lng": -104.90,
    },
}


def _backend_hex(node_id: int) -> str:
    return f"!{node_id:08x}"


def _backend_pos(key: str):
    node = BACKEND_LINK_NODES[key]
    return (node["lat"], node["lng"])


def _insert_backend_packet(
    cursor: sqlite3.Cursor,
    *,
    packet_id: int,
    from_node_id: int,
    gateway_hex: str,
    portnum: int,
    portnum_name: str,
    snr,
    rssi,
    mesh_packet_id: int,
    timestamp: float,
    channel_id: str,
    hop_start: int,
    hop_limit: int,
    raw_payload: bytes,
) -> None:
    """Insert one packet_history row with the signal values verbatim.

    The helper in DatabaseFixtures replaces falsy values with defaults
    (``snr or ...``), which would swallow exactly the 0.0 dB and NULL
    samples these tests exist to cover, so rows are written directly."""
    cursor.execute(
        """INSERT INTO packet_history
           (id, timestamp, topic, from_node_id, to_node_id, portnum,
            portnum_name, gateway_id, channel_id, rssi, snr, hop_limit,
            hop_start, payload_length, raw_payload, mesh_packet_id,
            processed_successfully)
           VALUES (?, ?, ?, ?, 4294967295, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
        (
            packet_id,
            timestamp,
            f"msh/EU_868/2/c/{channel_id}",
            from_node_id,
            portnum,
            portnum_name,
            gateway_hex,
            channel_id,
            rssi,
            snr,
            hop_limit,
            hop_start,
            len(raw_payload),
            raw_payload,
            mesh_packet_id,
        ),
    )


def _seed_backend_rf_link_database(db_path: str) -> None:
    """Create the production schema and seed direct receptions whose
    backend aggregation is fully predictable:

    - ping↔pong (LongFast/SF11): forward averages a real 0.0 dB from
      +0.25/-0.25 samples with RSSI -90/-91 (average -90.5).
    - rack↔sled (ShortFast/SF7): both directions ~13 dB below the SF7
      floor → marginal_both with reliability rounding to 0.0%.
    - unit↔victor: observed both ways, but SNR is NULL everywhere.
    """
    from meshtastic import mesh_pb2

    from malla.database.schema import ensure_startup_schema
    from tests.fixtures.database_fixtures import DatabaseFixtures

    fixtures = DatabaseFixtures()
    now = time.time()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        fixtures._create_schema(cursor)
        ensure_startup_schema(cursor)

        for node in BACKEND_LINK_NODES.values():
            cursor.execute(
                """INSERT INTO node_info
                   (node_id, hex_id, long_name, short_name, hw_model, role,
                    primary_channel, is_licensed, mac_address, first_seen,
                    last_updated)
                   VALUES (?, ?, ?, ?, 'TBEAM', 'CLIENT', 'LongFast', 0,
                           '24:6f:28:00:00:01', ?, ?)""",
                (
                    node["node_id"],
                    _backend_hex(node["node_id"]),
                    node["name"],
                    node["short"],
                    now - 86400,
                    now - 600,
                ),
            )

        # Position broadcasts (one hop, so they never form packet links).
        for i, node in enumerate(BACKEND_LINK_NODES.values()):
            position = mesh_pb2.Position()
            position.latitude_i = int(node["lat"] * 1e7)
            position.longitude_i = int(node["lng"] * 1e7)
            position.altitude = 100 + i
            position.sats_in_view = 9
            position.precision_bits = 17
            _insert_backend_packet(
                cursor,
                packet_id=130 + i,
                from_node_id=node["node_id"],
                gateway_hex=_backend_hex(BACKEND_POSITION_GATEWAY),
                portnum=3,
                portnum_name="POSITION_APP",
                snr=None,
                rssi=None,
                mesh_packet_id=8000 + i,
                timestamp=now - 7200,
                channel_id="LongFast",
                hop_start=4,
                hop_limit=3,
                raw_payload=position.SerializeToString(),
            )

        # Direct (0-hop) receptions: (id, transmitter, receiver, snr, rssi,
        # mesh_packet_id, channel). Distinct mesh_packet_ids keep every
        # reception a separate observation.
        direct_packets = [
            # ping↔pong: forward carries a real 0 dB average (0.25, -0.25).
            (101, "ping", "pong", 0.25, -90, 9001, "LongFast"),
            (102, "ping", "pong", -0.25, -91, 9002, "LongFast"),
            (103, "pong", "ping", 5.5, -82, 9003, "LongFast"),
            # rack↔sled on ShortFast: both directions marginal → est. 0.0%.
            (111, "rack", "sled", -21.2, -100, 9101, "ShortFast"),
            (112, "rack", "sled", -20.8, -105, 9102, "ShortFast"),
            (113, "sled", "rack", -20.5, -101, 9103, "ShortFast"),
            # unit↔victor: observed both ways, no usable SNR on either side.
            (121, "unit", "victor", None, -97, 9201, "LongFast"),
            (122, "unit", "victor", None, -99, 9202, "LongFast"),
            (123, "victor", "unit", None, None, 9203, "LongFast"),
        ]
        for (
            packet_id,
            transmitter,
            receiver,
            snr,
            rssi,
            mesh_packet_id,
            channel_id,
        ) in direct_packets:
            _insert_backend_packet(
                cursor,
                packet_id=packet_id,
                from_node_id=BACKEND_LINK_NODES[transmitter]["node_id"],
                gateway_hex=_backend_hex(BACKEND_LINK_NODES[receiver]["node_id"]),
                portnum=1,
                portnum_name="TEXT_MESSAGE_APP",
                snr=snr,
                rssi=rssi,
                mesh_packet_id=mesh_packet_id,
                timestamp=now - 600,
                channel_id=channel_id,
                hop_start=3,
                hop_limit=3,
                raw_payload=f"direct {packet_id}".encode(),
            )

        conn.commit()


@pytest.fixture(scope="module")
def backend_rf_link_server():
    """Real Flask app serving /api/locations from a seeded database.

    No route interception: the payload is produced by the production SQL
    aggregation and enrichment pipeline, so count and rounding contracts
    are exercised end to end."""
    import malla.config
    import malla.services.location_service as location_service
    from malla.config import AppConfig
    from malla.web_ui import create_app

    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    _seed_backend_rf_link_database(temp_db.name)

    # The packet-links cache and the DB path/config singleton are
    # process-wide. Pin them to this database and restore the previous
    # state afterwards so the session test server keeps serving its own
    # data once this module finishes.
    previous_env = os.environ.get("MALLA_DATABASE_FILE")
    previous_config = malla.config._config_singleton
    location_service._PACKET_LINKS_CACHE.clear()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    app = create_app(
        AppConfig(database_file=temp_db.name, host="127.0.0.1", port=port, debug=False)
    )
    server_thread = threading.Thread(
        target=app.run,
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "debug": False,
            "use_reloader": False,
        },
        daemon=True,
    )
    server_thread.start()
    for _ in range(100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    break
        except OSError:
            pass
        time.sleep(0.1)
    else:
        raise RuntimeError("Seeded RF-link test server failed to start")

    yield f"http://127.0.0.1:{port}"

    location_service._PACKET_LINKS_CACHE.clear()
    if previous_env is None:
        os.environ.pop("MALLA_DATABASE_FILE", None)
    else:
        os.environ["MALLA_DATABASE_FILE"] = previous_env
    malla.config._config_singleton = previous_config
    try:
        os.unlink(temp_db.name)
    except FileNotFoundError:
        pass


class TestMapBackendProducedRfLinks:
    """The real /api/locations pipeline must honor the link payload
    contract that the map's split-half rendering depends on."""

    @pytest.mark.e2e
    def test_backend_link_payload_contract(self, backend_rf_link_server):
        """Directional counts, averages and rounding survive the SQL
        aggregation: a real 0 dB reading, SF-aware 0.0% reliability and
        observed-but-no-SNR nulls all reach the client verbatim."""
        with urllib.request.urlopen(
            f"{backend_rf_link_server}/api/locations", timeout=60
        ) as response:
            data = json.loads(response.read().decode("utf-8"))

        links = {
            (link["from_node_id"], link["to_node_id"]): link
            for link in data["packet_links"]
        }
        assert len(links) == 3, f"Expected the three seeded links, got: {links}"

        ping = BACKEND_LINK_NODES["ping"]
        pong = BACKEND_LINK_NODES["pong"]
        l1 = links[(ping["node_id"], pong["node_id"])]
        # Forward: average of +0.25/-0.25 is a real 0.0 dB measurement.
        assert l1["forward_count"] == 2
        assert l1["return_count"] == 1
        assert l1["forward_avg_snr"] is not None
        assert l1["forward_avg_snr"] == pytest.approx(0.0, abs=1e-9)
        assert l1["forward_avg_rssi"] == pytest.approx(-90.5)
        assert l1["return_avg_snr"] == pytest.approx(5.5)
        assert l1["return_avg_rssi"] == pytest.approx(-82.0)
        # Combined average rounds once, at the end: mean(0.0, 5.5) → 2.8.
        assert l1["avg_snr"] == pytest.approx(2.8)
        assert l1["observation_count"] == 3
        assert l1["strength"] == pytest.approx(2.7)
        assert l1["link_balance"] == "balanced"
        assert l1["forward_quality"] == "good"
        assert l1["return_quality"] == "good"
        assert l1["forward_estimated_reliability"] == pytest.approx(100.0)
        assert l1["is_bidirectional"] is True

        rack = BACKEND_LINK_NODES["rack"]
        sled = BACKEND_LINK_NODES["sled"]
        l2 = links[(rack["node_id"], sled["node_id"])]
        assert l2["channel_id"] == "ShortFast"
        assert l2["spreading_factor"] == 7
        assert l2["forward_avg_snr"] == pytest.approx(-21.0)
        assert l2["return_avg_snr"] == pytest.approx(-20.5)
        assert l2["forward_avg_rssi"] == pytest.approx(-102.5)
        assert l2["link_balance"] == "marginal_both"
        assert l2["forward_estimated_reliability"] == 0.0
        assert l2["return_estimated_reliability"] == 0.0
        assert l2["estimated_reliability"] == 0.0
        assert l2["quality"] == "marginal"
        assert l2["worst_snr"] == pytest.approx(-21.0)

        unit = BACKEND_LINK_NODES["unit"]
        victor = BACKEND_LINK_NODES["victor"]
        l3 = links[(unit["node_id"], victor["node_id"])]
        assert l3["forward_avg_snr"] is None
        assert l3["return_avg_snr"] is None
        assert l3["forward_count"] == 2
        assert l3["return_count"] == 1
        assert l3["forward_avg_rssi"] == pytest.approx(-98.0)
        assert l3["return_avg_rssi"] is None
        assert l3["avg_snr"] is None
        assert l3["link_balance"] == "unknown"
        assert l3["estimated_reliability"] is None
        assert l3["is_bidirectional"] is True

    @pytest.mark.e2e
    def test_backend_link_popup_values(self, page: Page, backend_rf_link_server):
        """Popups render the backend-produced values: receiver-labelled
        directions with observation counts, the 0 dB measurement, 0.0%
        reliability on an SF7 marginal-both link, and the unknown/unknown
        state."""
        page.goto(f"{backend_rf_link_server}/map")
        page.wait_for_selector("#mapLoading", state="hidden", timeout=DEFAULT_TIMEOUT)
        page.locator("#packetLinksCheckbox").check()
        page.wait_for_function(
            "() => packetLinks.length === 3", timeout=DEFAULT_TIMEOUT
        )

        l1 = open_link_popup_between(
            page, "packetLinks", _backend_pos("ping"), _backend_pos("pong")
        )
        text = popup_text(l1)
        assert "Backend Ping → Backend Pong (received at Backend Pong)" in text
        assert "0.0 dB · -90.5 dBm · 2 observations · est. 100.0%" in text
        assert "Backend Pong → Backend Ping (received at Backend Ping)" in text
        assert "5.5 dB · -82.0 dBm · 1 observation · est. 100.0%" in text
        assert "Balanced — both directions usable" in text
        assert "Worst-direction SNR: 0.0 dB" in text
        assert "Observations: 3" in text
        assert "SNR unavailable" not in text

        l2 = open_link_popup_between(
            page, "packetLinks", _backend_pos("rack"), _backend_pos("sled")
        )
        text = popup_text(l2)
        assert "-21.0 dB · -102.5 dBm · 2 observations · est. 0.0%" in text
        assert "-20.5 dB · -101.0 dBm · 1 observation · est. 0.0%" in text
        assert "Fragile — both directions marginal" in text
        assert "Estimated reliability: 0.0%" in text
        assert "Quality model: SF7 · ShortFast" in text
        assert "Worst-direction SNR: -21.0 dB" in text

        l3 = open_link_popup_between(
            page, "packetLinks", _backend_pos("unit"), _backend_pos("victor")
        )
        text = popup_text(l3)
        # Two direction blocks plus the balance note mention the phrase.
        assert text.count("SNR unavailable") == 3
        assert "SNR unavailable · -98.0 dBm · 2 observations" in text
        assert "SNR unavailable · 1 observation" in text
        assert "Observed, but SNR unavailable for balance assessment" in text
        assert "Estimated reliability:" not in text
        assert "Observations: 3" in text
