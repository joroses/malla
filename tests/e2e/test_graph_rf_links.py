"""
E2E tests for per-direction RF link quality rendering on the network graph.

The /api/traceroute/graph response is intercepted with the fixture payload
from tests/fixtures/traceroute_graph_data.py (built with the real backend
enrichment), so the split-half rendering can be verified exactly:

- receiver-rule half orientation (forward quality nearest the target),
- observation-only width shared by both halves,
- unknown-direction halves dashed and faded,
- hover and selected-link panels rendering identical directional details
  through the shared renderer (same language as the map popup),
- deselection restoring quality colors, widths, opacities and dash
  patterns (interaction rides on a separate outline),
- node dragging keeping both halves joined at the displayed midpoint,
- one logical edge per node pair in the D3 force simulation.
"""

import json

import pytest
from playwright.sync_api import Page, Route, expect

from malla.utils.signal_quality import QUALITY_COLORS
from tests.fixtures.traceroute_graph_data import (
    NODE_ALPHA,
    NODE_BRAVO,
    NODE_CHARLIE,
    NODE_ECHO,
    NODE_FOX,
    get_sample_graph_data,
)

DEFAULT_TIMEOUT = 20000  # ms


@pytest.fixture()
def rf_link_graph_page(page: Page, test_server_url: str):
    """Graph page served by the real test app with deterministic links."""

    def fulfill_graph(route: Route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(get_sample_graph_data()),
        )

    page.route("**/api/traceroute/graph*", fulfill_graph)
    page.goto(f"{test_server_url}/traceroute-graph")
    page.wait_for_selector("#networkGraph svg", timeout=DEFAULT_TIMEOUT)
    page.wait_for_function(
        "() => document.querySelectorAll('.links g.link').length > 0",
        timeout=DEFAULT_TIMEOUT,
    )
    return page


def link_group_js(source_id: int, target_id: int) -> str:
    return (
        "([sourceId, targetId]) => {"
        "  const group = [...document.querySelectorAll('.links g.link')]"
        "    .find(g => g.__data__.source.id === sourceId"
        "        && g.__data__.target.id === targetId);"
        "  return group || null;"
        "}"
    )


def get_link_group(page: Page, source_id: int, target_id: int):
    group = page.evaluate(link_group_js(source_id, target_id), [source_id, target_id])
    assert group is not None, f"No graph link group for {source_id:#x}–{target_id:#x}"
    return group


def halves_state(page: Page, source_id: int, target_id: int):
    """Stroke/width/opacity/dash of both halves plus the group's class."""
    return page.evaluate(
        """([sourceId, targetId]) => {
            const group = [...document.querySelectorAll('.links g.link')]
                .find(g => g.__data__.source.id === sourceId
                    && g.__data__.target.id === targetId);
            const read = line => ({
                stroke: line.getAttribute('stroke'),
                width: line.getAttribute('stroke-width'),
                opacity: parseFloat(line.getAttribute('stroke-opacity')),
                dash: line.getAttribute('stroke-dasharray'),
            });
            return {
                className: group.getAttribute('class'),
                groupOpacity: group.style.opacity,
                outlineStroke: getComputedStyle(group.querySelector('.link-outline')).stroke,
                returnHalf: read(group.querySelector('.half-return')),
                forwardHalf: read(group.querySelector('.half-forward')),
            };
        }""",
        [source_id, target_id],
    )


def hover_link(page: Page, source_id: int, target_id: int) -> str:
    """Hover one half of the link and return the hover panel text."""
    page.evaluate(
        """([sourceId, targetId]) => {
            const group = [...document.querySelectorAll('.links g.link')]
                .find(g => g.__data__.source.id === sourceId
                    && g.__data__.target.id === targetId);
            group.querySelector('.half-forward')
                .dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
        }""",
        [source_id, target_id],
    )
    return panel_text(page, "#hoverDetails")


def select_link(page: Page, source_id: int, target_id: int) -> str:
    page.evaluate(
        """([sourceId, targetId]) => {
            const group = [...document.querySelectorAll('.links g.link')]
                .find(g => g.__data__.source.id === sourceId
                    && g.__data__.target.id === targetId);
            group.dispatchEvent(new MouseEvent('click', { bubbles: true }));
        }""",
        [source_id, target_id],
    )
    expect(page.locator("#selectedDetails")).to_be_visible(timeout=DEFAULT_TIMEOUT)
    return panel_text(page, "#selectedDetailsContent")


def panel_text(page: Page, selector: str) -> str:
    text = page.locator(selector).text_content()
    assert text is not None
    return text


def fixture_link(source_id: int, target_id: int) -> dict:
    data = get_sample_graph_data()
    return next(
        link
        for link in data["links"]
        if link["source"] == source_id and link["target"] == target_id
    )


def direction_line(sender: str, receiver: str, link: dict, prefix: str) -> str:
    snr = link[f"{prefix}_avg_snr"]
    count = link[f"{prefix}_count"]
    reliability = link[f"{prefix}_estimated_reliability"]
    parts = [f"{snr:.1f} dB", f"{count} observation{'s' if count != 1 else ''}"]
    if reliability is not None:
        parts.append(f"est. {reliability:.1f}%")
    return f"{sender} → {receiver} (received at {receiver}): " + " · ".join(parts)


class TestGraphDirectionalRfLinks:
    """Per-direction link quality rendering on the network graph."""

    @pytest.mark.e2e
    def test_halves_colored_by_receiver_quality_at_midpoint(
        self, rf_link_graph_page: Page
    ):
        """Bravo↔Charlie (asymmetric): the half nearest the target carries
        the forward (marginal) quality measured at the target; the half
        nearest the source carries the return (good) quality. Both halves
        meet at the displayed midpoint and share the observation width."""
        page = rf_link_graph_page
        geometry = page.evaluate(
            """([sourceId, targetId]) => {
                const group = [...document.querySelectorAll('.links g.link')]
                    .find(g => g.__data__.source.id === sourceId
                        && g.__data__.target.id === targetId);
                const read = line => ({
                    stroke: line.getAttribute('stroke'),
                    width: line.getAttribute('stroke-width'),
                    opacity: parseFloat(line.getAttribute('stroke-opacity')),
                    dash: line.getAttribute('stroke-dasharray'),
                    x1: parseFloat(line.getAttribute('x1')),
                    y1: parseFloat(line.getAttribute('y1')),
                    x2: parseFloat(line.getAttribute('x2')),
                    y2: parseFloat(line.getAttribute('y2')),
                });
                return {
                    returnHalf: read(group.querySelector('.half-return')),
                    forwardHalf: read(group.querySelector('.half-forward')),
                };
            }""",
            [NODE_BRAVO, NODE_CHARLIE],
        )

        assert geometry["returnHalf"]["stroke"] == QUALITY_COLORS["good"], (
            "Half nearest the source must show the return (good) quality"
        )
        assert geometry["forwardHalf"]["stroke"] == QUALITY_COLORS["marginal"], (
            "Half nearest the target must show the forward (marginal) quality"
        )

        link = fixture_link(NODE_BRAVO, NODE_CHARLIE)
        for half in (geometry["returnHalf"], geometry["forwardHalf"]):
            assert float(half["width"]) == pytest.approx(link["strength"]), (
                "Both halves use the observation-volume width"
            )
            assert half["dash"] is None, "Measured halves render solid"
            assert half["opacity"] == pytest.approx(0.8)

        # The halves join at the displayed midpoint of the node positions.
        ret, fwd = geometry["returnHalf"], geometry["forwardHalf"]
        assert ret["x2"] == pytest.approx(fwd["x1"], abs=1e-6)
        assert ret["y2"] == pytest.approx(fwd["y1"], abs=1e-6)
        assert fwd["x2"] - fwd["x1"] == pytest.approx(ret["x2"] - ret["x1"], abs=1e-6)
        assert fwd["y2"] - fwd["y1"] == pytest.approx(ret["y2"] - ret["y1"], abs=1e-6)

    @pytest.mark.e2e
    def test_unknown_direction_half_dashed_and_faded(self, rf_link_graph_page: Page):
        """Alpha↔Echo (one direction): the unobserved return direction
        renders with the unknown color, a dash pattern and reduced opacity
        — never an average of the measured direction."""
        state = halves_state(rf_link_graph_page, NODE_ALPHA, NODE_ECHO)
        assert state["returnHalf"]["stroke"] == QUALITY_COLORS["unknown"]
        assert state["returnHalf"]["dash"] == "8, 6"
        assert state["returnHalf"]["opacity"] == pytest.approx(0.8 * 0.55)
        assert state["forwardHalf"]["stroke"] == QUALITY_COLORS["good"]
        assert state["forwardHalf"]["dash"] is None
        assert state["forwardHalf"]["opacity"] == pytest.approx(0.8)
        # One logical edge: exactly one group with two halves.
        assert get_link_group(rf_link_graph_page, NODE_ALPHA, NODE_ECHO)

    @pytest.mark.e2e
    def test_hover_and_selected_show_identical_directional_details(
        self, rf_link_graph_page: Page
    ):
        """Hover and selection render through the same shared details
        renderer: identical directional labels with receiver names, counts,
        reliability, quality badges and balance warnings."""
        page = rf_link_graph_page
        link = fixture_link(NODE_BRAVO, NODE_CHARLIE)

        forward_line = direction_line(
            "Test Node Bravo", "Test Node Charlie", link, "forward"
        )
        return_line = direction_line(
            "Test Node Charlie", "Test Node Bravo", link, "return"
        )

        hover_text = hover_link(page, NODE_BRAVO, NODE_CHARLIE)
        selected_text = select_link(page, NODE_BRAVO, NODE_CHARLIE)

        for text in (hover_text, selected_text):
            assert "Test Node Bravo ↔ Test Node Charlie" in text
            assert "Direct RF Link" in text
            assert forward_line in text
            assert return_line in text
            assert (
                f"Estimated reliability: {link['estimated_reliability']:.1f}%" in text
            )
            assert "Asymmetric — one direction marginal" in text
            assert f"Quality model: SF{link['spreading_factor']} · LongFast" in text
            assert f"Worst-direction SNR: {link['worst_snr']:.1f} dB" in text
            assert f"Observations: {link['observation_count']}" in text

        # Quality badges accompany each directional measurement line.
        assert "Marginal" in hover_text
        assert "Good" in hover_text

    @pytest.mark.e2e
    def test_indirect_connections_stay_identifiable_as_inferred(
        self, rf_link_graph_page: Page
    ):
        """Inferred multi-hop paths render as dashed unknown-quality lines
        and their details say the path is inferred, not a measured link."""
        page = rf_link_graph_page
        indirect = page.evaluate(
            """() => {
                const group = [...document.querySelectorAll('.indirect-links g.link')][0];
                const line = group.querySelector('.half-indirect');
                return {
                    className: group.getAttribute('class'),
                    stroke: line.getAttribute('stroke'),
                    dash: line.getAttribute('stroke-dasharray'),
                    opacity: line.getAttribute('stroke-opacity'),
                };
            }"""
        )
        assert indirect["className"] == "link indirect"
        assert indirect["stroke"] == QUALITY_COLORS["unknown"]
        assert indirect["dash"] == "5,5"
        assert float(indirect["opacity"]) == pytest.approx(0.4)

        selected_text = page.evaluate(
            """() => {
                const group = [...document.querySelectorAll('.indirect-links g.link')][0];
                group.dispatchEvent(new MouseEvent('click', { bubbles: true }));
                return document.getElementById('selectedDetailsContent').textContent;
            }"""
        )
        assert "Inferred multi-hop path — RF quality not measured" in selected_text
        assert "Multi-hop Connection (inferred)" in selected_text
        assert "received at" not in selected_text, (
            "Inferred paths must not present directional measurements"
        )

    @pytest.mark.e2e
    def test_selection_uses_outline_and_deselection_restores_styles(
        self, rf_link_graph_page: Page
    ):
        """Selecting a link highlights through the separate outline; the
        halves keep their quality colors, widths, opacities and dash
        patterns. Clearing the selection restores everything, including
        the pre-selection one-direction dash styling of other links."""
        page = rf_link_graph_page
        before = halves_state(page, NODE_BRAVO, NODE_CHARLIE)
        one_direction_before = halves_state(page, NODE_ALPHA, NODE_ECHO)

        select_link(page, NODE_BRAVO, NODE_CHARLIE)
        selected = halves_state(page, NODE_BRAVO, NODE_CHARLIE)

        assert selected["className"] == "link selected"
        assert selected["outlineStroke"] == "rgb(0, 123, 255)", (
            "Selection must ride on the blue outline"
        )
        assert selected["returnHalf"] == before["returnHalf"], (
            "Selection must not restyle the return half"
        )
        assert selected["forwardHalf"] == before["forwardHalf"], (
            "Selection must not restyle the forward half"
        )

        page.locator("#clearSelection").click()
        after = halves_state(page, NODE_BRAVO, NODE_CHARLIE)
        one_direction_after = halves_state(page, NODE_ALPHA, NODE_ECHO)

        assert after["className"] == "link"
        assert after["outlineStroke"] == "rgba(0, 0, 0, 0)", (
            "Outline must return to transparent"
        )
        assert after["returnHalf"] == before["returnHalf"]
        assert after["forwardHalf"] == before["forwardHalf"]
        assert after["groupOpacity"] == ""
        assert (
            one_direction_after["returnHalf"] == one_direction_before["returnHalf"]
        ), "Deselection must restore the unknown-direction dash and fade"

    @pytest.mark.e2e
    def test_node_dragging_keeps_both_halves_joined(self, rf_link_graph_page: Page):
        """Dragging a node moves the whole logical edge: both halves stay
        joined at the displayed midpoint of the new endpoint positions."""
        page = rf_link_graph_page
        page.wait_for_timeout(2000)  # let the geo layout release fixed positions

        group = page.locator(".node").filter(has=page.locator("text=Test Node Charlie"))
        box = group.locator("circle").bounding_box()
        assert box is not None, "Charlie's node circle should be rendered"
        start_x = box["x"] + box["width"] / 2
        start_y = box["y"] + box["height"] / 2

        page.mouse.move(start_x, start_y)
        page.mouse.down()
        page.mouse.move(start_x + 180, start_y + 120, steps=12)
        page.mouse.up()
        page.wait_for_timeout(500)

        geometry = page.evaluate(
            """([sourceId, targetId]) => {
                const group = [...document.querySelectorAll('.links g.link')]
                    .find(g => g.__data__.source.id === sourceId
                        && g.__data__.target.id === targetId);
                const read = line => ({
                    x1: parseFloat(line.getAttribute('x1')),
                    y1: parseFloat(line.getAttribute('y1')),
                    x2: parseFloat(line.getAttribute('x2')),
                    y2: parseFloat(line.getAttribute('y2')),
                });
                return {
                    source: { x: group.__data__.source.x, y: group.__data__.source.y },
                    target: { x: group.__data__.target.x, y: group.__data__.target.y },
                    returnHalf: read(group.querySelector('.half-return')),
                    forwardHalf: read(group.querySelector('.half-forward')),
                };
            }""",
            [NODE_BRAVO, NODE_CHARLIE],
        )

        source, target = geometry["source"], geometry["target"]
        ret, fwd = geometry["returnHalf"], geometry["forwardHalf"]
        mid_x = (source["x"] + target["x"]) / 2
        mid_y = (source["y"] + target["y"]) / 2

        # Both halves still span source → midpoint → target after the drag.
        assert (ret["x1"], ret["y1"]) == pytest.approx(
            (source["x"], source["y"]), abs=1e-6
        )
        assert (fwd["x2"], fwd["y2"]) == pytest.approx(
            (target["x"], target["y"]), abs=1e-6
        )
        assert (ret["x2"], ret["y2"]) == pytest.approx((mid_x, mid_y), abs=1e-6)
        assert (fwd["x1"], fwd["y1"]) == pytest.approx((mid_x, mid_y), abs=1e-6)

    @pytest.mark.e2e
    def test_one_logical_edge_per_node_pair_in_simulation(
        self, rf_link_graph_page: Page
    ):
        """The D3 force simulation keeps a single edge per node pair even
        though rendering draws two halves per link."""
        counts = rf_link_graph_page.evaluate(
            """() => ({
                payloadLinks: window.currentGraph.links.length,
                linkGroups: document.querySelectorAll('.links g.link').length,
                halves: document.querySelectorAll('.links g.link .link-half').length,
            })"""
        )
        assert counts["linkGroups"] == counts["payloadLinks"], (
            "One rendering group per logical edge"
        )
        assert counts["halves"] == 2 * counts["payloadLinks"], (
            "Each logical edge renders exactly two halves"
        )

    @pytest.mark.e2e
    def test_path_highlight_preserves_quality_colors(self, rf_link_graph_page: Page):
        """Hop-filter path highlighting (selected node + hovered node)
        uses the outline overlay; the underlying halves never change."""
        page = rf_link_graph_page
        before = halves_state(page, NODE_BRAVO, NODE_CHARLIE)

        # Select Alpha, then hover Charlie: every shortest path between
        # them runs over Bravo↔Charlie.
        page.evaluate(
            """(nodeId) => {
                const group = [...document.querySelectorAll('.node')]
                    .find(n => n.__data__.id === nodeId);
                group.dispatchEvent(new MouseEvent('click', { bubbles: true }));
            }""",
            NODE_ALPHA,
        )
        expect(page.locator("#hopFilterSection")).to_be_visible(timeout=DEFAULT_TIMEOUT)
        page.evaluate(
            """(nodeId) => {
                const group = [...document.querySelectorAll('.node')]
                    .find(n => n.__data__.id === nodeId);
                group.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
            }""",
            NODE_CHARLIE,
        )

        highlighted = halves_state(page, NODE_BRAVO, NODE_CHARLIE)
        assert highlighted["className"] == "link path-highlight", (
            "Path links are highlighted through the outline class"
        )
        assert highlighted["outlineStroke"] == "rgb(255, 193, 7)"
        assert highlighted["returnHalf"] == before["returnHalf"], (
            "Path highlighting must not restyle the return half"
        )
        assert highlighted["forwardHalf"] == before["forwardHalf"], (
            "Path highlighting must not restyle the forward half"
        )

        # Mouse out clears the outline class; halves remain untouched.
        page.evaluate(
            """(nodeId) => {
                const group = [...document.querySelectorAll('.node')]
                    .find(n => n.__data__.id === nodeId);
                group.dispatchEvent(new MouseEvent('mouseout', { bubbles: true }));
            }""",
            NODE_CHARLIE,
        )
        cleared = halves_state(page, NODE_BRAVO, NODE_CHARLIE)
        assert "path-highlight" not in cleared["className"]
        assert cleared["outlineStroke"] == "rgba(0, 0, 0, 0)"
        assert cleared["returnHalf"] == before["returnHalf"]
        assert cleared["forwardHalf"] == before["forwardHalf"]

    @pytest.mark.e2e
    def test_node_quality_badges_in_panels(self, rf_link_graph_page: Page):
        """Node panels show the backend quality tier badge instead of a
        raw-SNR color cutoff; a node without measurements shows N/A."""
        page = rf_link_graph_page
        page.evaluate(
            """(nodeId) => {
                const group = [...document.querySelectorAll('.node')]
                    .find(n => n.__data__.id === nodeId);
                group.dispatchEvent(new MouseEvent('click', { bubbles: true }));
            }""",
            NODE_ALPHA,
        )
        selected = panel_text(page, "#selectedDetailsContent")
        assert "-3.5 dB" in selected
        assert "Good" in selected

        page.evaluate(
            """(nodeId) => {
                const group = [...document.querySelectorAll('.node')]
                    .find(n => n.__data__.id === nodeId);
                group.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
            }""",
            NODE_FOX,
        )
        hover = panel_text(page, "#hoverDetails")
        assert "-11.9 dB" in hover
        assert "Fair" in hover

    @pytest.mark.e2e
    def test_legend_documents_quality_language(self, rf_link_graph_page: Page):
        """The graph legend shares the map's quality entries."""
        legend = rf_link_graph_page.locator(".legend-content")
        expect(legend).to_be_visible()
        for text in (
            "Good quality (half nearest receiver)",
            "Fair quality",
            "Marginal quality",
            "Direction not observed (dashed, faded)",
            "Asymmetric link — color changes midway",
            "Thicker = more observations",
            "Multi-hop path (inferred, dashed)",
            "Each RF link half is colored by the quality the node at that end receives",
        ):
            expect(legend).to_contain_text(text)
