"""
E2E tests for per-direction RF link quality rendering on the map.

The /api/locations response is intercepted with a deterministic payload so
the split-half rendering can be verified exactly: midpoint color
orientation, unknown-direction styling, link-type dash patterns, popup
values and repeated redraw/filter cycles without orphaned segments.
"""

import json
import time

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


def _make_node(node, name, short, now):
    return {
        "node_id": node["node_id"],
        "display_name": name,
        "short_name": short,
        "role": "CLIENT",
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
    """
    now = time.time()
    nodes = [
        _make_node(NODE_ALPHA, "Node Alpha", "ALPH", now),
        _make_node(NODE_BRAVO, "Node Bravo", "BRAV", now),
        _make_node(NODE_CHARLIE, "Node Charlie", "CHAR", now),
        _make_node(NODE_DELTA, "Node Delta", "DELT", now),
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

    return {
        "locations": nodes,
        "traceroute_links": [t1, t2],
        "packet_links": [p1, p2],
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


def popup_text(popup) -> str:
    """Non-None text content of the popup locator."""
    text = popup.text_content()
    assert text is not None, "Popup should have text content"
    return text


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
    def test_redraw_and_filters_leave_no_orphan_segments(self, rf_link_map_page: Page):
        """Toggles, filters and node selection redraw both halves together:
        after every cycle the DOM contains exactly the expected segments."""
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

        # Enable packet links on top.
        page.locator("#packetLinksCheckbox").check()
        page.wait_for_function("() => packetLinks.length === 2")
        assert link_paths() == 8, "All four links drawn as two halves each"

        # Client-side filter cycle (role filter redraws everything).
        page.locator("#roleFilter").select_option("CLIENT")
        page.locator("#locationFilterForm button[type='submit']").click()
        page.wait_for_timeout(1000)
        assert link_paths() == 8

        # Selecting a node redraws only its 1-hop links (Alpha has T1, P1, P2).
        page.evaluate("() => selectNode(nodeData.find(n => n.node_id === 100))")
        expect(page.locator("#hopDepthSection")).to_be_visible(timeout=DEFAULT_TIMEOUT)
        page.wait_for_timeout(500)
        assert link_paths() == 6, "3 links touching the selected node × 2 halves"

        # Clearing the selection restores the full network without orphans.
        page.locator("#clearSelection").click()
        page.wait_for_function(
            "() => tracerouteLinks.length === 2 && packetLinks.length === 2"
        )
        assert link_paths() == 8

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
