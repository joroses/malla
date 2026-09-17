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

from src.malla.services.locations_response_cache import LocationsResponseCache

EMPTY_NETWORK_DATA = {"nodes": [], "links": []}


@pytest.fixture
def mocked_location_services(monkeypatch):
    """Patch the expensive service calls used by /api/locations.

    Also keeps the response cache from lazily spawning its refresher
    thread inside these tests: the assertions below reason about which
    windows the service layer was invoked with, and a background default
    warm would add unrelated calls.
    """
    monkeypatch.setattr(
        LocationsResponseCache, "_ensure_background_refresh", lambda *a: None
    )
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
        """Explicit epoch start/end bound the link aggregation window.

        Bounds are snapped onto the cache grid (start floored, end ceiled),
        so the resolved window contains the requested one and both edges
        land on grid multiples.
        """
        from src.malla.routes.api_routes import _LOCATIONS_NOW_GRID_SECONDS

        end = time_module.time() - 7200
        start = end - 3600

        response = client.get(f"/api/locations?start_time={start}&end_time={end}")

        assert response.status_code == 200

        graph_filters = mocked_location_services["graph"].call_args.kwargs["filters"]
        # Snapped onto the grid, containing the requested window
        assert graph_filters["start_time"] % _LOCATIONS_NOW_GRID_SECONDS == 0
        assert graph_filters["end_time"] % _LOCATIONS_NOW_GRID_SECONDS == 0
        assert graph_filters["start_time"] <= start
        assert graph_filters["end_time"] >= end
        assert start - graph_filters["start_time"] < _LOCATIONS_NOW_GRID_SECONDS
        assert graph_filters["end_time"] - end < _LOCATIONS_NOW_GRID_SECONDS

        packet_filters = mocked_location_services["packet_links"].call_args.args[0]
        assert packet_filters["start_time"] == graph_filters["start_time"]
        assert packet_filters["end_time"] == graph_filters["end_time"]

        traceroute_filters = mocked_location_services[
            "traceroute_links"
        ].call_args.args[0]
        assert traceroute_filters["start_time"] == graph_filters["start_time"]
        assert traceroute_filters["end_time"] == graph_filters["end_time"]

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
        # Start must be roughly 14 days ago; the ceiled grid end can sit up
        # to one grid step ahead of wall-clock time, dragging start with it.
        from src.malla.routes.api_routes import _LOCATIONS_NOW_GRID_SECONDS

        assert before - 14 * 24 * 3600 - 5 <= graph_filters["start_time"]
        assert graph_filters["start_time"] <= (
            before - 14 * 24 * 3600 + _LOCATIONS_NOW_GRID_SECONDS
        )

    @pytest.mark.unit
    def test_window_capped_at_14_days(self, client, mocked_location_services):
        """Windows larger than 14 days are clamped for performance."""
        from src.malla.routes.api_routes import _LOCATIONS_NOW_GRID_SECONDS

        end = time_module.time()
        start = end - 20 * 24 * 3600

        response = client.get(f"/api/locations?start_time={start}&end_time={end}")

        assert response.status_code == 200
        graph_filters = mocked_location_services["graph"].call_args.kwargs["filters"]
        # The ceiled end can sit up to one grid step past the requested
        # end, dragging the capped start with it.
        assert graph_filters["start_time"] >= end - 14 * 24 * 3600 - 5
        assert graph_filters["start_time"] <= (
            end - 14 * 24 * 3600 + _LOCATIONS_NOW_GRID_SECONDS
        )

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
        # Client-supplied start is floored onto the grid...
        assert graph_filters["start_time"] % _LOCATIONS_NOW_GRID_SECONDS == 0
        assert graph_filters["start_time"] <= start
        assert start - graph_filters["start_time"] < _LOCATIONS_NOW_GRID_SECONDS
        # ...while the derived end lands on the grid boundary.
        assert graph_filters["end_time"] % _LOCATIONS_NOW_GRID_SECONDS == 0

        packet_filters = mocked_location_services["packet_links"].call_args.args[0]
        assert packet_filters["start_time"] == graph_filters["start_time"]
        assert packet_filters["end_time"] == graph_filters["end_time"]

    @pytest.mark.unit
    def test_visits_seconds_apart_share_one_window(
        self, client, mocked_location_services
    ):
        """Two map visits seconds apart resolve identical filter sets.

        The map materializes its preset start_time from the client clock at
        page-load time, so back-to-back visits send start_times a few
        seconds apart. Un-snapped, each visit minted a unique window and
        re-ran the full multi-day graph build (which can take minutes on
        large windows). Both starts inside one grid bucket must collapse
        onto the same window — served from the whole-response cache, so
        the second visit neither recomputes nor re-resolves different
        filters.
        """
        from src.malla.routes.api_routes import _LOCATIONS_NOW_GRID_SECONDS

        # Historic window: identical end for both requests isolates the
        # start snapping from server-now drift.
        end = int(time_module.time()) - 48 * 3600
        bucket = (end // _LOCATIONS_NOW_GRID_SECONDS) * _LOCATIONS_NOW_GRID_SECONDS
        start_a = bucket - 24 * 3600 + 5  # same grid bucket, seconds apart
        start_b = bucket - 24 * 3600 + 20

        first = client.get(f"/api/locations?start_time={start_a}&end_time={end}")
        second = client.get(f"/api/locations?start_time={start_b}&end_time={end}")

        assert first.status_code == second.status_code == 200
        assert second.data == first.data
        # One compute total: the second visit hit the response cache.
        assert mocked_location_services["graph"].call_count == 1
        assert mocked_location_services["packet_links"].call_count == 1
        filters = first.get_json()["filters_applied"]
        assert filters["start_time"] % _LOCATIONS_NOW_GRID_SECONDS == 0
        assert filters["end_time"] % _LOCATIONS_NOW_GRID_SECONDS == 0

    @pytest.mark.unit
    def test_repeated_requests_resolve_identical_windows(
        self, client, mocked_location_services
    ):
        """Back-to-back requests with the same start_time resolve the same
        server-derived window (response cache hit); without grid snapping
        every request produced a fresh microsecond end_time (guaranteed
        miss and recompute)."""
        from src.malla.routes.api_routes import _LOCATIONS_NOW_GRID_SECONDS

        start = int(time_module.time()) - 24 * 3600
        first = client.get(f"/api/locations?start_time={start}")
        second = client.get(f"/api/locations?start_time={start}")

        assert first.status_code == second.status_code == 200
        first_end = first.get_json()["filters_applied"]["end_time"]
        second_end = second.get_json()["filters_applied"]["end_time"]
        assert first_end % _LOCATIONS_NOW_GRID_SECONDS == 0
        # Identical within a bucket; one grid step apart at most if a bucket
        # boundary happened to be crossed between the two requests.
        assert second_end - first_end in (0, _LOCATIONS_NOW_GRID_SECONDS)

    @pytest.mark.unit
    def test_default_window_snapped_to_grid(self, client, mocked_location_services):
        """Parameterless requests also resolve a grid-stable window.

        The no-params path copies the wide 14-day position window into the
        link filters; building that window from raw datetime.now()
        (microsecond floats) gave every default request a unique filter
        set, so the caches never hit for the most common request of all
        (map first load, LocationCache, packet detail pages).
        """
        from src.malla.routes.api_routes import _LOCATIONS_NOW_GRID_SECONDS

        first = client.get("/api/locations")
        second = client.get("/api/locations")

        assert first.status_code == second.status_code == 200
        first_filters = first.get_json()["filters_applied"]
        second_filters = second.get_json()["filters_applied"]
        for filters in (first_filters, second_filters):
            assert filters["end_time"] % _LOCATIONS_NOW_GRID_SECONDS == 0
        # Identical within a bucket; one grid step apart at most if a bucket
        # boundary happened to be crossed between the two requests.
        assert second_filters["end_time"] - first_filters["end_time"] in (
            0,
            _LOCATIONS_NOW_GRID_SECONDS,
        )
        assert second_filters["start_time"] - first_filters["start_time"] in (
            0,
            _LOCATIONS_NOW_GRID_SECONDS,
        )

    @pytest.mark.unit
    def test_grid_step_exceeds_service_cache_ttls(self):
        """The snapping grid must cover the service TTL caches.

        The resolved window (and therefore the _NETWORK_GRAPH_CACHE /
        _PACKET_LINKS_CACHE keys) rolls with the clock every grid step. If
        the step were shorter than a cache's TTL, entries would become
        unreachable well before expiring and reloads tens of seconds apart
        would always recompute -- the failure mode this grid exists to
        prevent. The network-graph TTL is grid-sized (300 s) so the 60 s
        background refresher self-hits across cycles; the shorter packet
        cache keeps the original 2x headroom.
        """
        from src.malla.routes.api_routes import _LOCATIONS_NOW_GRID_SECONDS
        from src.malla.services.location_service import (
            _PACKET_LINKS_CACHE_TTL_SECONDS,
        )
        from src.malla.services.traceroute_service import (
            _NETWORK_GRAPH_CACHE_TTL_SECONDS,
        )

        for ttl in (_NETWORK_GRAPH_CACHE_TTL_SECONDS, _PACKET_LINKS_CACHE_TTL_SECONDS):
            assert _LOCATIONS_NOW_GRID_SECONDS >= ttl
        assert _LOCATIONS_NOW_GRID_SECONDS >= 2 * _PACKET_LINKS_CACHE_TTL_SECONDS

    @pytest.mark.unit
    def test_reloads_a_minute_apart_share_one_window(
        self, client, mocked_location_services
    ):
        """Reloads ~60 s apart resolve identical windows (cache hits).

        With a 30 s grid the derived bounds rolled every 30 s while the
        caches lived 60 s, so any two visits more than 30 s apart minted
        different start_time/end_time, different cache keys, and a full
        recompute. The grid now exceeds the TTL: the two simulated
        visits below (60 s apart, pinned mid-bucket away from a grid
        edge) must resolve identical filters — the second visit is
        served the cached response bytes without recompute.
        """
        import datetime as datetime_module
        import sys
        import types
        from datetime import datetime as real_datetime_class

        from src.malla.routes.api_routes import _LOCATIONS_NOW_GRID_SECONDS

        grid = _LOCATIONS_NOW_GRID_SECONDS
        clock = {"t": (int(time_module.time()) // grid + 1) * grid + grid // 2}

        class FrozenDatetime(datetime_module.datetime):
            @classmethod
            def now(cls, tz=None):
                return real_datetime_class.fromtimestamp(clock["t"], tz)

        frozen_datetime_module = types.ModuleType("datetime")
        frozen_datetime_module.__dict__.update(datetime_module.__dict__)
        frozen_datetime_module.datetime = FrozenDatetime

        with (
            patch.dict(sys.modules, {"datetime": frozen_datetime_module}),
            patch("time.time", side_effect=lambda: clock["t"]),
        ):
            first = client.get("/api/locations?hours=24")
            clock["t"] += 60
            second = client.get("/api/locations?hours=24")

        assert first.status_code == second.status_code == 200
        assert second.data == first.data
        # One compute total: the reload 60 s later hit the response cache.
        assert mocked_location_services["graph"].call_count == 1
        packet_calls = mocked_location_services["packet_links"].call_args_list
        assert len(packet_calls) == 1
        assert packet_calls[0].args[0]["start_time"] % grid == 0
