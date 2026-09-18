"""A selected node's neighborhood bypasses only the overview filters."""

import time
from datetime import UTC, datetime

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


@pytest.fixture
def selection_map(page: Page, test_server_url: str):
    now = time.time()
    activity = now - 2 * 3600

    def node(node_id, **overrides):
        return {
            "node_id": node_id,
            "display_name": f"Station {node_id}",
            "short_name": str(node_id),
            "role": "CLIENT",
            "latitude": 40 + node_id / 100000,
            "longitude": -105,
            "timestamp": activity,
            "primary_channel": "LongFast",
            "broadcast_text_count": 0,
            **overrides,
        }

    def link(source, target, timestamp=activity):
        return {
            "from_node_id": source,
            "to_node_id": target,
            "total_hops_seen": 1,
            "last_seen_str": datetime.fromtimestamp(timestamp, UTC).isoformat(),
        }

    payload = {
        "locations": [
            node(100, role="ROUTER", broadcast_text_count=20),
            node(200, timestamp=now - 4 * 3600),  # Traceroute neighbor
            node(300, role="SENSOR", timestamp=now - 1800),  # Packet neighbor
            node(400),  # Reliable gateway without an RF link
            node(500),  # Unrelated, fails overview filters
            node(600, primary_channel="LongSlow"),
            node(700, timestamp=now - 36 * 3600),
            node(800),  # Recent node, but its only link is old
            node(900),  # Two hops away through node 200
            node(1000),  # Reachable only through wrong-channel node 600
            node(1100),  # Reachable only through stale node 700
            node(1200, role="ROUTER", broadcast_text_count=20),
            node(1300, role="ROUTER", broadcast_text_count=20),
        ],
        "traceroute_links": [
            link(100, 200),
            link(200, 900),
            link(100, 600),
            link(600, 1000),
            link(100, 700),
            link(700, 1100),
            link(100, 800, now - 36 * 3600),
            link(1200, 1300),
        ],
        "packet_links": [link(100, 300)],
    }
    page.route("**/api/locations*", lambda route: route.fulfill(json=payload))
    page.route(
        "**/api/meshtastic/channels",
        lambda route: route.fulfill(json={"channels": ["LongFast", "LongSlow"]}),
    )
    page.route(
        "**/text-message-reliability*",
        lambda route: route.fulfill(
            json={
                "node_id": 100,
                "total_sent": 20,
                "gateways": [
                    {"gateway_node_id": node_id, "received": 16, "percent": 80}
                    for node_id in (400, 600, 700)
                ],
            }
        ),
    )
    page.goto(f"{test_server_url}/map")
    page.wait_for_selector("#mapLoading", state="hidden")
    expect(page.locator('#channelFilter option[value="LongFast"]')).to_be_attached()
    page.locator("#channelFilter").select_option("LongFast")
    page.locator('#locationFilterForm button[type="submit"]').click()
    return now


def apply_filters(page: Page, role="ROUTER", broadcasts=10, contacts=10):
    page.locator("#roleFilter").select_option(role)
    page.locator("#minBroadcasts").fill(str(broadcasts))
    page.locator("#minContacts").fill(str(contacts))
    page.locator('#locationFilterForm button[type="submit"]').click()


def expect_nodes(page: Page, node_ids: list[int]):
    """Check both the sidebar and the map's rendered markers."""
    page.wait_for_function(
        """expected => {
            const actual = [...document.querySelectorAll('#nodeList .node-list-item')]
                .map(item => Number(item.dataset.nodeId)).sort((a, b) => a - b);
            const markers = nodeMarkers.map(marker => Number(
                marker.getElement().querySelector('.node-marker-label').textContent
            )).sort((a, b) => a - b);
            return JSON.stringify(actual) === JSON.stringify(expected) &&
                JSON.stringify(markers) === JSON.stringify(expected);
        }""",
        arg=sorted(node_ids),
    )
    expect(page.locator("#nodeCount")).to_have_text(str(len(node_ids)))
    expect(page.locator("#statsNodes")).to_have_text(str(len(node_ids)))


def expect_links(page: Page, traceroute: int, packet: int):
    assert page.evaluate("() => [tracerouteLinks.length, packetLinks.length]") == [
        traceroute,
        packet,
    ]


@pytest.mark.parametrize(
    ("role", "broadcasts", "contacts"),
    [("ROUTER", 0, 1), ("", 10, 1), ("", 0, 10), ("ROUTER", 10, 10)],
)
def test_selection_exempts_overview_filters(
    page: Page, selection_map, role, broadcasts, contacts
):
    apply_filters(page, role, broadcasts, contacts)
    page.locator("#packetLinksCheckbox").check()
    overview_nodes = page.locator("#nodeList .node-list-item").evaluate_all(
        "items => items.map(item => Number(item.dataset.nodeId))"
    )
    if role or broadcasts:
        expect_nodes(page, [100, 1200, 1300])
    overview_links = page.evaluate("() => [tracerouteLinks.length, packetLinks.length]")
    if contacts > 1:
        expect_links(page, 0, 0)

    page.locator('#nodeList [data-node-id="100"]').click()
    expect_nodes(page, [100, 200, 300, 400])
    expect_links(page, 1, 1)
    expect(page.locator(".node-marker-reliability")).to_have_text("80%")

    # Clearing selection reapplies the role, broadcast and contact filters.
    page.locator("#clearSelection").click()
    expect_nodes(page, overview_nodes)
    expect_links(page, *overview_links)


def test_link_type_toggles_and_hop_depth(page: Page, selection_map):
    apply_filters(page)
    page.locator('#nodeList [data-node-id="100"]').click()
    expect_nodes(page, [100, 200, 400])
    expect_links(page, 1, 0)

    page.locator("#tracerouteLinksCheckbox").uncheck()
    expect_nodes(page, [100, 400])
    expect_links(page, 0, 0)

    page.locator("#packetLinksCheckbox").check()
    expect_nodes(page, [100, 300, 400])
    expect_links(page, 0, 1)

    page.locator("#reliabilityCheckbox").uncheck()
    expect_nodes(page, [100, 300])
    page.locator("#packetLinksCheckbox").uncheck()
    expect_nodes(page, [100])

    page.locator("#tracerouteLinksCheckbox").check()
    page.locator("#hopDepthSelect").select_option("2")
    expect_nodes(page, [100, 200, 900])
    expect_links(page, 2, 0)

    # All keeps ordinary filtered nodes, but exempts only related nodes.
    # Low-contact links between unrelated nodes remain filtered out.
    page.locator("#hopDepthSelect").select_option("999")
    expect_nodes(page, [100, 200, 900, 1200, 1300])
    expect_links(page, 2, 0)
    page.locator("#tracerouteLinksCheckbox").uncheck()
    expect_nodes(page, [100, 1200, 1300])
    expect_links(page, 0, 0)


def test_filter_changes_during_selection(page: Page, selection_map):
    page.locator("#packetLinksCheckbox").check()
    page.locator('#nodeList [data-node-id="100"]').click()
    expect_nodes(page, [100, 200, 300, 400])

    apply_filters(page)
    expect_nodes(page, [100, 200, 300, 400])
    expect_links(page, 1, 1)

    # A revealed neighbor can itself become the selection despite its role.
    page.locator('#nodeList [data-node-id="200"]').click()
    expect_nodes(page, [100, 200, 900])
    expect_links(page, 2, 0)
    page.locator("#clearSelection").click()
    expect_nodes(page, [100, 1200, 1300])
    expect_links(page, 0, 0)


def test_custom_date_range_and_channel_always_apply(page: Page, selection_map):
    now = selection_map
    apply_filters(page)
    page.locator("#packetLinksCheckbox").check()
    page.locator('#nodeList [data-node-id="100"]').click()
    expect_nodes(page, [100, 200, 300, 400])

    # Keep the selected node and reliable gateway, excluding an older
    # traceroute neighbor and a packet neighbor newer than the custom end.
    for selector, timestamp in (
        ("#startDateTime", now - 3 * 3600),
        ("#endDateTime", now - 3600),
    ):
        local_value = page.evaluate(
            """timestamp => {
                const date = new Date(timestamp * 1000);
                return new Date(date - date.getTimezoneOffset() * 60000)
                    .toISOString().slice(0, 16);
            }""",
            timestamp,
        )
        page.locator(selector).fill(local_value)
        page.locator(selector).dispatch_event("change")
    page.locator('#locationFilterForm button[type="submit"]').click()
    expect_nodes(page, [100, 400])
    expect_links(page, 0, 0)

    # Neither a selected node nor a raw-payload fallback may bypass channel.
    page.locator("#channelFilter").select_option("LongSlow")
    page.locator('#locationFilterForm button[type="submit"]').click()
    expect_nodes(page, [600])
    page.locator("#reliabilityCheckbox").uncheck()
    expect_nodes(page, [])


def test_old_reliability_does_not_reveal_nodes_in_new_window(page: Page, selection_map):
    apply_filters(page)
    page.locator('#nodeList [data-node-id="100"]').click()
    expect_nodes(page, [100, 200, 400])

    # Hold the new reliability response while locations finish reloading.
    pending = []
    page.route("**/text-message-reliability*", lambda route: pending.append(route))
    with page.expect_request("**/text-message-reliability?hours=6"):
        page.locator("#maxAge").select_option("6")
    expect_nodes(page, [100, 200])
    expect(page.locator(".node-marker-reliability")).to_have_count(0)

    assert len(pending) == 1
    pending[0].fulfill(json={"node_id": 100, "total_sent": 0, "gateways": []})
    expect_nodes(page, [100, 200])
