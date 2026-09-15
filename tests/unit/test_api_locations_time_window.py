"""
Unit tests for the /api/locations time-window handling.

The endpoint must aggregate link metrics (traceroute hops, direct packet
receptions) over the exact window selected on the map, while keeping the
node position lookup on a wide window so actively routing nodes whose last
GPS report is older than the selection stay visible.
"""

import time as time_module
from unittest.mock import patch

import pytest

EMPTY_NETWORK_DATA = {"nodes": [], "links": []}


@pytest.fixture
def mocked_location_services():
    """Patch the expensive service calls used by /api/locations."""
    with (
        patch(
            "src.malla.routes.api_routes.TracerouteService.get_network_graph_data",
            return_value=EMPTY_NETWORK_DATA,
        ) as graph_mock,
        patch(
            "src.malla.routes.api_routes.LocationService.get_packet_links",
            return_value=[],
        ) as packet_links_mock,
        patch(
            "src.malla.routes.api_routes.LocationService.get_node_locations",
            return_value=[],
        ) as node_locations_mock,
        patch(
            "src.malla.routes.api_routes.LocationService.get_traceroute_links",
            return_value=[],
        ) as traceroute_links_mock,
    ):
        yield {
            "graph": graph_mock,
            "packet_links": packet_links_mock,
            "node_locations": node_locations_mock,
            "traceroute_links": traceroute_links_mock,
        }


class TestApiLocationsTimeWindow:
    """Test /api/locations server-side time window resolution."""

    @pytest.mark.unit
    def test_hours_param_scopes_link_aggregation(
        self, client, mocked_location_services
    ):
        """hours=1 aggregates links over the last hour only."""
        before = time_module.time()
        response = client.get("/api/locations?hours=1")
        after = time_module.time()

        assert response.status_code == 200

        graph_filters = mocked_location_services["graph"].call_args.kwargs["filters"]
        assert graph_filters["start_time"] >= before - 3600
        # The derived end is ceiled onto the cache grid, so it may sit up to
        # one grid step ahead of wall-clock time.
        from src.malla.routes.api_routes import _LOCATIONS_NOW_GRID_SECONDS

        assert graph_filters["end_time"] <= after + _LOCATIONS_NOW_GRID_SECONDS

        packet_filters = mocked_location_services["packet_links"].call_args.args[0]
        assert packet_filters["start_time"] == graph_filters["start_time"]

        traceroute_filters = mocked_location_services[
            "traceroute_links"
        ].call_args.args[0]
        assert traceroute_filters["start_time"] == graph_filters["start_time"]

    @pytest.mark.unit
    def test_start_end_params_scope_link_aggregation(
        self, client, mocked_location_services
    ):
        """Explicit epoch start/end bound the link aggregation window."""
        end = time_module.time() - 7200
        start = end - 3600

        response = client.get(f"/api/locations?start_time={start}&end_time={end}")

        assert response.status_code == 200

        graph_filters = mocked_location_services["graph"].call_args.kwargs["filters"]
        assert graph_filters["start_time"] == start
        assert graph_filters["end_time"] == end

        packet_filters = mocked_location_services["packet_links"].call_args.args[0]
        assert packet_filters["start_time"] == start
        assert packet_filters["end_time"] == end

        traceroute_filters = mocked_location_services[
            "traceroute_links"
        ].call_args.args[0]
        assert traceroute_filters["start_time"] == start
        assert traceroute_filters["end_time"] == end

    @pytest.mark.unit
    def test_max_age_hours_alias_supported(self, client, mocked_location_services):
        """max_age_hours is accepted as an alias for hours."""
        before = time_module.time()
        response = client.get("/api/locations?max_age_hours=6")
        assert response.status_code == 200

        graph_filters = mocked_location_services["graph"].call_args.kwargs["filters"]
        assert graph_filters["start_time"] >= before - 6 * 3600 - 5
        assert graph_filters["start_time"] <= before - 6 * 3600 + 3600

    @pytest.mark.unit
    def test_default_window_is_14_days_without_time_params(
        self, client, mocked_location_services
    ):
        """No time parameters keeps the historical 14-day default window."""
        before = time_module.time()

        response = client.get("/api/locations")

        assert response.status_code == 200
        graph_filters = mocked_location_services["graph"].call_args.kwargs["filters"]
        # Start must be roughly 14 days ago (allow small execution slack)
        assert before - 14 * 24 * 3600 - 5 <= graph_filters["start_time"]
        assert graph_filters["start_time"] <= before - 14 * 24 * 3600 + 5

    @pytest.mark.unit
    def test_window_capped_at_14_days(self, client, mocked_location_services):
        """Windows larger than 14 days are clamped for performance."""
        end = time_module.time()
        start = end - 20 * 24 * 3600

        response = client.get(f"/api/locations?start_time={start}&end_time={end}")

        assert response.status_code == 200
        graph_filters = mocked_location_services["graph"].call_args.kwargs["filters"]
        assert graph_filters["start_time"] >= end - 14 * 24 * 3600 - 5
        assert graph_filters["start_time"] <= end - 14 * 24 * 3600 + 5

    @pytest.mark.unit
    def test_invalid_time_range_returns_400(self, client, mocked_location_services):
        """start_time >= end_time is rejected."""
        now = time_module.time()
        response = client.get(f"/api/locations?start_time={now}&end_time={now - 10}")
        assert response.status_code == 400

    @pytest.mark.unit
    def test_position_lookup_keeps_wide_window_despite_short_link_window(
        self, client, mocked_location_services
    ):
        """Regression guard: a 1-hour link window must not narrow the GPS lookup.

        Nodes actively routing now whose last position broadcast is hours or
        days old must remain visible at their last known good position, so
        the position lookup keeps the wide 14-day window even when link
        aggregates are computed for the last hour only.
        """
        before = time_module.time()

        response = client.get("/api/locations?hours=1")

        assert response.status_code == 200

        graph_filters = mocked_location_services["graph"].call_args.kwargs["filters"]
        position_filters = mocked_location_services["node_locations"].call_args.args[0]

        # Link aggregation is scoped to the last hour...
        assert graph_filters["start_time"] >= before - 3600 - 5

        # ...while the position lookup keeps the wide window.
        assert before - 14 * 24 * 3600 - 5 <= position_filters["start_time"]
        assert position_filters["start_time"] <= before - 13 * 24 * 3600
        assert position_filters["start_time"] < graph_filters["start_time"]

    @pytest.mark.unit
    def test_gateway_filter_applied_to_both_windows(
        self, client, mocked_location_services
    ):
        """gateway_id filters both link aggregation and position lookup."""
        response = client.get("/api/locations?hours=1&gateway_id=42")

        assert response.status_code == 200

        packet_filters = mocked_location_services["packet_links"].call_args.args[0]
        position_filters = mocked_location_services["node_locations"].call_args.args[0]
        assert packet_filters["gateway_id"] == 42
        assert position_filters["gateway_id"] == 42

    @pytest.mark.unit
    def test_invalid_gateway_id_returns_400(self, client, mocked_location_services):
        response = client.get("/api/locations?gateway_id=not-a-number")
        assert response.status_code == 400

    @pytest.mark.unit
    def test_derived_end_time_snapped_to_grid(self, client, mocked_location_services):
        """Server-derived end_time is snapped onto the cache grid.

        The map typically sends only start_time. Deriving end_time from raw
        datetime.now() (microsecond floats) gave every request a unique
        filter set, so the _NETWORK_GRAPH_CACHE/_PACKET_LINKS_CACHE TTL
        caches in the service layer never hit.
        """
        from src.malla.routes.api_routes import _LOCATIONS_NOW_GRID_SECONDS

        start = int(time_module.time()) - 24 * 3600
        response = client.get(f"/api/locations?start_time={start}")

        assert response.status_code == 200
        graph_filters = mocked_location_services["graph"].call_args.kwargs["filters"]
        # Client-supplied start is preserved exactly...
        assert graph_filters["start_time"] == start
        # ...while the derived end lands on the grid boundary.
        assert graph_filters["end_time"] % _LOCATIONS_NOW_GRID_SECONDS == 0

        packet_filters = mocked_location_services["packet_links"].call_args.args[0]
        assert packet_filters["start_time"] == start
        assert packet_filters["end_time"] == graph_filters["end_time"]

    @pytest.mark.unit
    def test_repeated_requests_resolve_identical_windows(
        self, client, mocked_location_services
    ):
        """Back-to-back requests with the same start_time resolve the same
        server-derived window (cache hit); without grid snapping every
        request produced a fresh microsecond end_time (guaranteed miss)."""
        from src.malla.routes.api_routes import _LOCATIONS_NOW_GRID_SECONDS

        start = int(time_module.time()) - 24 * 3600
        client.get(f"/api/locations?start_time={start}")
        client.get(f"/api/locations?start_time={start}")

        calls = mocked_location_services["graph"].call_args_list
        first_end = calls[0].kwargs["filters"]["end_time"]
        second_end = calls[1].kwargs["filters"]["end_time"]
        # Identical within a bucket; one grid step apart at most if a bucket
        # boundary happened to be crossed between the two requests.
        assert second_end - first_end in (0, _LOCATIONS_NOW_GRID_SECONDS)
