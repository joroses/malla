"""
E2E tests for the map's gateway text-message reliability feature toggle.

Selecting a node badges gateways with the % of its broadcast text messages
they received (see /api/node/<id>/text-message-reliability). The Link Types
"Reliability" checkbox (on by default) enables/disables the whole feature:
badges, the "Broadcasts received" node-popup note, the legend line, and the
backend fetch itself.
"""

import json
import time

import pytest
from playwright.sync_api import Page, expect

DEFAULT_TIMEOUT = 20000  # ms

NODE_ALPHA = {"node_id": 100, "lat": 40.00, "lng": -105.00}
NODE_BRAVO = {"node_id": 200, "lat": 40.00, "lng": -104.90}
NODE_CHARLIE = {"node_id": 300, "lat": 40.05, "lng": -105.05}


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


def gateway_reliability_map_page(page: Page, test_server_url: str):
    """Map page with deterministic locations and gateway reliability."""
    now = time.time()

    def fulfill_locations(route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "locations": [
                        _make_node(NODE_ALPHA, "Node Alpha", "ALPH", now),
                        _make_node(NODE_BRAVO, "Node Bravo", "BRAV", now),
                        _make_node(NODE_CHARLIE, "Node Charlie", "CHAR", now),
                    ],
                    "traceroute_links": [],
                    "packet_links": [],
                }
            ),
        )

    def fulfill_reliability(route) -> None:
        node_id = int(route.request.url.split("/node/")[1].split("/")[0])
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "node_id": node_id,
                    "total_sent": 10,
                    "gateways": [
                        {
                            "gateway_node_id": NODE_BRAVO["node_id"],
                            "received": 8,
                            "percent": 80,
                        }
                    ],
                }
            ),
        )

    reliability_requests: list = []
    page.on(
        "request",
        lambda r: reliability_requests.append(r.url)
        if "text-message-reliability" in r.url
        else None,
    )
    page.route("**/api/locations*", fulfill_locations)
    page.route("**/text-message-reliability*", fulfill_reliability)
    page.goto(f"{test_server_url}/map")
    page.wait_for_selector("#mapLoading", state="hidden", timeout=DEFAULT_TIMEOUT)
    page.wait_for_function(
        "() => typeof allNodeData !== 'undefined' && allNodeData.length === 3",
        timeout=DEFAULT_TIMEOUT,
    )
    return reliability_requests


def select_node(page: Page, node_id: int) -> None:
    """Select a node through the same handler a marker click invokes."""
    page.evaluate(
        "(nodeId) => selectNode(allNodeData.find(n => n.node_id === nodeId))",
        node_id,
    )


def node_popup_text(page: Page, node_id: int) -> str:
    """Render the node popup through the same builder bound to markers."""
    return page.evaluate(
        "(nodeId) => createNodePopupContent("
        "allNodeData.find(n => n.node_id === nodeId)).textContent",
        node_id,
    )


class TestGatewayReliabilityToggle:
    @pytest.mark.e2e
    def test_reliability_on_by_default(self, page: Page, test_server_url):
        """Checkbox starts checked; selecting a node badges the gateway and
        the popup note shows the reception figures."""
        gateway_reliability_map_page(page, test_server_url)

        expect(page.locator("#reliabilityCheckbox")).to_be_checked()
        expect(page.locator("#gatewayReliabilityLegend")).to_be_visible()

        select_node(page, NODE_ALPHA["node_id"])
        expect(page.locator(".node-marker-reliability")).to_have_text(
            "80%", timeout=DEFAULT_TIMEOUT
        )
        assert "Broadcasts received: 8/10 (80%)" in node_popup_text(
            page, NODE_BRAVO["node_id"]
        )

    @pytest.mark.e2e
    def test_toggle_disables_feature_and_fetch(self, page: Page, test_server_url):
        """Unchecking clears badges, popup notes and the legend line, and no
        reliability fetch happens on (re-)selection; re-checking restores."""
        reliability_requests = gateway_reliability_map_page(page, test_server_url)

        select_node(page, NODE_ALPHA["node_id"])
        expect(page.locator(".node-marker-reliability")).to_have_text(
            "80%", timeout=DEFAULT_TIMEOUT
        )
        assert len(reliability_requests) >= 1

        page.locator("#reliabilityCheckbox").uncheck()
        expect(page.locator(".node-marker-reliability")).to_have_count(0)
        expect(page.locator("#gatewayReliabilityLegend")).to_be_hidden()
        assert "Broadcasts received" not in node_popup_text(page, NODE_BRAVO["node_id"])

        # Re-selecting while disabled must not fetch reliability.
        seen = len(reliability_requests)
        select_node(page, NODE_ALPHA["node_id"])
        page.wait_for_function("() => selectedNodeId === 100", timeout=DEFAULT_TIMEOUT)
        page.wait_for_timeout(1200)
        assert len(reliability_requests) == seen
        expect(page.locator(".node-marker-reliability")).to_have_count(0)

        # Re-enabling refetches for the current selection and restores badges.
        page.locator("#reliabilityCheckbox").check()
        expect(page.locator("#gatewayReliabilityLegend")).to_be_visible()
        expect(page.locator(".node-marker-reliability")).to_have_text(
            "80%", timeout=DEFAULT_TIMEOUT
        )
        assert len(reliability_requests) > seen
        assert "Broadcasts received: 8/10 (80%)" in node_popup_text(
            page, NODE_BRAVO["node_id"]
        )
