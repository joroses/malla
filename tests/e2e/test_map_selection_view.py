"""Automatic selection fitting and persistent map selection indicators."""

import time

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


def node(node_id, latitude=40, longitude=-105):
    return {
        "node_id": node_id,
        "display_name": f"Station {node_id}",
        "short_name": str(node_id),
        "role": "CLIENT",
        "latitude": latitude,
        "longitude": longitude,
        "timestamp": time.time(),
        "primary_channel": "LongFast",
    }


def open_map(page: Page, test_server_url, nodes, neighbors=()):
    # Start from a saved view so the initial-load fit cannot mask a missing
    # selection fit. Keep external basemap requests out of these tests.
    page.add_init_script("""localStorage.setItem('malla_map_view', JSON.stringify({
        lat: 40, lng: -105, zoom: 8
    }));""")
    page.route(
        "https://tiles.openfreemap.org/styles/*",
        lambda route: route.fulfill(json={"version": 8, "sources": {}, "layers": []}),
    )
    page.route(
        "**/api/meshtastic/channels",
        lambda route: route.fulfill(json={"channels": ["LongFast"]}),
    )
    page.route(
        "**/api/locations*",
        lambda route: route.fulfill(
            json={
                "locations": nodes,
                "traceroute_links": [
                    {"from_node_id": 100, "to_node_id": target, "total_hops_seen": 1}
                    for target in neighbors
                ],
                "packet_links": [],
            }
        ),
    )
    pending = []
    page.route("**/text-message-reliability*", lambda route: pending.append(route))
    page.goto(f"{test_server_url}/map")
    page.wait_for_selector("#mapLoading", state="hidden")
    expect(page.locator("#nodeCount")).to_have_text(str(len(nodes)))
    return pending


def select_node(page: Page, node_id):
    with page.expect_request(f"**/node/{node_id}/text-message-reliability*"):
        page.locator(f'#nodeList [data-node-id="{node_id}"]').click()


def reliability_response(route, node_id=100, gateways=()):
    route.fulfill(
        json={
            "node_id": node_id,
            "total_sent": 10,
            "gateways": [
                {"gateway_node_id": gateway, "received": 8, "percent": 80}
                for gateway in gateways
            ],
        }
    )


def expect_selection_in_view(page: Page, count):
    expect(page.locator("#nodeCount")).to_have_text(str(count))
    page.wait_for_function("""() => {
        if (document.querySelector('.leaflet-zoom-anim')) return false;
        const mapRect = document.querySelector('#map').getBoundingClientRect();
        const sidebar = document.querySelector('#sidebar').getBoundingClientRect();
        const bottom = sidebar.left < mapRect.right && sidebar.right > mapRect.left
            ? Math.min(mapRect.bottom, sidebar.top) : mapRect.bottom;
        const elements = document.querySelectorAll(
            '.node-marker-container, .selected-node-label'
        );
        return [...elements].every(element => {
            const rect = element.getBoundingClientRect();
            return rect.left >= mapRect.left + 10 && rect.right <= mapRect.right - 10 &&
                rect.top >= mapRect.top + 10 && rect.bottom <= bottom - 10;
        });
    }""")


def test_auto_fit_includes_distant_selected_node_and_reliable_gateway(
    page: Page, test_server_url
):
    # More than 40 neighbors triggers percentile trimming in the overview.
    # Both the selected node and the reliable gateway are geographic outliers.
    neighbors = [node(200 + i, 40 + i / 10000, -105 + i / 10000) for i in range(45)]
    nodes = [node(100, 35, -110), *neighbors, node(300, 50, -80)]
    pending = open_map(page, test_server_url, nodes, [n["node_id"] for n in neighbors])
    select_node(page, 100)
    expect_selection_in_view(page, 46)

    reliability_response(pending[0], gateways=[300])
    expect_selection_in_view(page, 47)
    expect(page.locator(".node-marker-reliability")).to_have_text("80%")
    expect(page.locator(".selected-node-label")).to_have_text("Selected: Station 100")


@pytest.mark.parametrize("select_from", ["marker", "sidebar"])
def test_reliability_arriving_during_zoom_refits_when_zoom_finishes(
    page: Page, test_server_url, select_from
):
    pending = open_map(
        page,
        test_server_url,
        [node(100), node(200, 40.06, -104.94), node(300, 45, -98)],
        [200],
    )
    # The selection changes zoom by only a few levels, making Leaflet animate.
    page.evaluate("map.setView([40.03, -104.97], 11, {animate: false})")
    if select_from == "marker":
        with page.expect_request("**/node/100/text-message-reliability*"):
            page.get_by_role("button", name="Station 100", exact=True).click()
    else:
        select_node(page, 100)
    page.wait_for_function("!!document.querySelector('.leaflet-zoom-anim')")
    reliability_response(pending[0], gateways=[300])
    expect_selection_in_view(page, 3)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_selected_marker_stays_identifiable_on_mobile(
    page: Page, test_server_url, theme, tmp_path
):
    page.set_viewport_size({"width": 390, "height": 844})
    pending = open_map(page, test_server_url, [node(100), node(200)], [200])
    page.evaluate("theme => document.documentElement.dataset.bsTheme = theme", theme)
    select_node(page, 100)
    reliability_response(pending[0])
    # Co-located nodes need a finite zoom and padding above the mobile sidebar.
    expect_selection_in_view(page, 2)
    selected = page.locator('.leaflet-marker-icon[aria-current="true"]')
    expect(selected).to_have_count(1)
    expect(selected).to_have_attribute("aria-label", "Selected: Station 100")
    assert selected.evaluate("el => Number(el.style.zIndex)") > page.locator(
        '.leaflet-marker-icon:not([aria-current="true"])'
    ).evaluate("el => Number(el.style.zIndex)")
    assert (
        page.locator(".node-marker-selected").evaluate(
            "el => getComputedStyle(el).boxShadow"
        )
        != "none"
    )
    page.screenshot(path=str(tmp_path / f"selected-node-{theme}.png"))

    # Redrawing a selection keeps its indicator; switching and clearing
    # selection transfer/remove both the ring and permanent name label.
    page.locator("#hopDepthSelect").select_option("2")
    expect(page.locator(".node-marker-selected")).to_have_count(1)
    select_node(page, 200)
    reliability_response(pending[1], node_id=200)
    expect(page.locator(".selected-node-label")).to_have_text("Selected: Station 200")
    expect(selected).to_have_attribute("aria-label", "Selected: Station 200")
    page.locator("#clearSelection").click()
    expect(page.locator(".node-marker-selected")).to_have_count(0)
    expect(page.locator(".selected-node-label")).to_have_count(0)
    expect(selected).to_have_count(0)


def test_single_selected_node_is_above_mobile_sidebar(page: Page, test_server_url):
    page.set_viewport_size({"width": 390, "height": 844})
    pending = open_map(page, test_server_url, [node(100)])
    select_node(page, 100)
    reliability_response(pending[0])
    expect_selection_in_view(page, 1)
    assert page.evaluate("map.getZoom()") == 15
