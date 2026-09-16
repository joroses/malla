"""Tests for the /api/locations whole-response cache and refresher.

/api/locations used to recompute its multi-second payload inside the
request whenever the service-layer TTL caches lapsed, and re-serialized
~5 MB of JSON on every hit. The response cache now pre-serializes each
resolved window once and a background refresher thread (the same
strategy as the analytics cache) keeps recently served windows warm, so
requests replay cached bytes and never sit inside a compute window.
"""

import json
import time

import pytest

import src.malla.services.locations_response_cache as locations_cache_module
from src.malla.services.locations_response_cache import (
    _LOCATIONS_NOW_GRID_SECONDS,
    LocationsResponseCache,
    LocationsWindowError,
    locations_cache_key,
    normalize_recipe,
    resolve_locations_filters,
    serialize_locations_payload,
)

pytestmark = pytest.mark.unit

EMPTY_NETWORK_DATA = {"nodes": [], "links": []}


@pytest.fixture(autouse=True)
def _isolated_cache():
    """Isolate the class-level cache and threads between tests."""
    LocationsResponseCache.clear()
    yield
    LocationsResponseCache.stop_background_refresh()
    LocationsResponseCache.clear()


@pytest.fixture
def no_auto_refresh(monkeypatch):
    """Keep request paths from spawning the refresher (pure cache tests)."""
    monkeypatch.setattr(
        LocationsResponseCache, "_ensure_background_refresh", lambda *a: None
    )


@pytest.fixture
def stub_compute(monkeypatch):
    """Replace the expensive payload compute with a counting stub."""
    calls = {"n": 0}

    def fake_compute(link_filters, position_filters):
        calls["n"] += 1
        return {
            "locations": [{"node_id": calls["n"]}],
            "traceroute_links": [],
            "packet_links": [],
            "total_count": 1,
            "filters_applied": dict(link_filters),
            "data_period_days": 14,
        }

    monkeypatch.setattr(
        locations_cache_module, "compute_locations_payload", fake_compute
    )
    return calls


class TestResolveLocationsFilters:
    def test_parameterless_uses_wide_default_window(self):
        grid_now = 12345 * _LOCATIONS_NOW_GRID_SECONDS
        link, position = resolve_locations_filters(
            normalize_recipe(None, None, None, None, None), server_now=grid_now
        )

        assert link == position
        assert link["end_time"] == grid_now
        assert link["end_time"] - link["start_time"] == 14 * 24 * 3600

    def test_hours_derives_grid_aligned_window(self):
        grid_now = 999 * _LOCATIONS_NOW_GRID_SECONDS
        link, _ = resolve_locations_filters(
            normalize_recipe(None, None, 6, None, None), server_now=grid_now
        )

        assert link["end_time"] == grid_now
        assert link["end_time"] - link["start_time"] == 6 * 3600
        assert link["start_time"] % _LOCATIONS_NOW_GRID_SECONDS == 0

    def test_explicit_bounds_snapped_onto_grid(self):
        end = 123456 * _LOCATIONS_NOW_GRID_SECONDS + 7
        start = end - 3600

        link, _ = resolve_locations_filters(
            normalize_recipe(start, end, None, None, None)
        )

        assert link["start_time"] % _LOCATIONS_NOW_GRID_SECONDS == 0
        assert link["end_time"] % _LOCATIONS_NOW_GRID_SECONDS == 0
        assert link["start_time"] <= start
        assert link["end_time"] >= end

    def test_window_capped_at_14_days(self):
        end = 50000 * _LOCATIONS_NOW_GRID_SECONDS
        start = end - 20 * 24 * 3600

        link, _ = resolve_locations_filters(
            normalize_recipe(start, end, None, None, None)
        )

        assert link["end_time"] - link["start_time"] == 14 * 24 * 3600

    def test_inverted_window_rejected(self):
        now = time.time()
        with pytest.raises(LocationsWindowError):
            resolve_locations_filters(
                normalize_recipe(now, now - 10, None, None, None)
            )

    def test_gateway_and_search_applied_to_both_windows(self):
        link, position = resolve_locations_filters(
            normalize_recipe(None, None, 1, 42, "basement")
        )

        assert link["gateway_id"] == 42 and position["gateway_id"] == 42
        assert link["search"] == "basement" and position["search"] == "basement"

    def test_zero_hours_normalizes_to_default_window(self):
        grid_now = 777 * _LOCATIONS_NOW_GRID_SECONDS
        link, _ = resolve_locations_filters(
            normalize_recipe(None, None, 0, None, None), server_now=grid_now
        )

        assert link["end_time"] - link["start_time"] == 14 * 24 * 3600


class TestServeFromCache:
    def test_second_request_within_max_stale_served_from_cache(
        self, no_auto_refresh, stub_compute
    ):
        recipe = normalize_recipe(None, None, 24, None, None)
        link, _ = resolve_locations_filters(recipe)
        key = locations_cache_key(link)

        first = LocationsResponseCache.serve(recipe, key)
        assert first is None  # cache miss
        body = LocationsResponseCache.store(
            recipe, key, locations_cache_module.compute_locations_payload(link, {})
        )

        second = LocationsResponseCache.serve(recipe, key)

        assert second is body
        assert stub_compute["n"] == 1

    def test_entry_between_ttl_and_max_stale_served_without_recompute(
        self, no_auto_refresh, stub_compute
    ):
        """A visitor arriving after the service TTLs lapse is recompute-free.

        The entry is older than the service-layer 60 s TTLs but younger
        than _MAX_STALE_SEC; it must be served as-is while the background
        refresher (not the request) brings it back up to date.
        """
        recipe = normalize_recipe(None, None, 24, None, None)
        link, _ = resolve_locations_filters(recipe)
        key = locations_cache_key(link)

        stale_body = b'{"stale": true}'
        LocationsResponseCache._CACHE[key] = (time.time() - 120, stale_body)

        assert LocationsResponseCache.serve(recipe, key) is stale_body
        assert stub_compute["n"] == 0

    def test_entry_past_max_stale_recomputed_inline(self, no_auto_refresh, stub_compute):
        """Dead-refresher fallback: beyond _MAX_STALE_SEC a request recomputes."""
        recipe = normalize_recipe(None, None, 24, None, None)
        link, _ = resolve_locations_filters(recipe)
        key = locations_cache_key(link)
        LocationsResponseCache._CACHE[key] = (
            time.time() - 2 * LocationsResponseCache._MAX_STALE_SEC,
            b'{"old": true}',
        )

        assert LocationsResponseCache.serve(recipe, key) is None

        payload = locations_cache_module.compute_locations_payload(link, {})
        body = LocationsResponseCache.store(recipe, key, payload)
        # And the refreshed entry is served afterwards without recompute.
        assert LocationsResponseCache.serve(recipe, key) is body
        assert stub_compute["n"] == 1

    def test_first_request_for_new_recipe_computes_inline(
        self, no_auto_refresh, stub_compute
    ):
        recipe = normalize_recipe(None, None, 24, "!aabbccdd", None)
        link, _ = resolve_locations_filters(recipe)
        key = locations_cache_key(link)

        assert LocationsResponseCache.serve(recipe, key) is None
        LocationsResponseCache.store(
            recipe, key, locations_cache_module.compute_locations_payload(link, {})
        )
        assert LocationsResponseCache.serve(recipe, key) is not None
        assert stub_compute["n"] == 1

    def test_pre_serialized_body_is_valid_compact_json(self):
        payload = {
            "total_count": 1,
            "nan_value": float("nan"),
            "locations": [{"lat": 1.5}],
        }
        body = serialize_locations_payload(payload)

        assert isinstance(body, bytes)
        # Compact separators: no space after the key/value colon
        assert b'"total_count":1' in body
        decoded = json.loads(body)
        assert decoded["total_count"] == 1
        # NaN sanitized to null instead of invalid JSON
        assert decoded["nan_value"] is None


class TestBackgroundRefresher:
    def test_start_warms_default_recipe_immediately(self, stub_compute):
        LocationsResponseCache.start_background_refresh()

        deadline = time.time() + 10
        while time.time() < deadline:
            if LocationsResponseCache._CACHE:
                break
            time.sleep(0.02)

        assert len(LocationsResponseCache._CACHE) == 1
        key = next(iter(LocationsResponseCache._CACHE))
        # The default recipe resolves the wide 14-day window
        assert key[1] - key[0] == 14 * 24 * 3600
        assert json.loads(LocationsResponseCache._CACHE[key][1])["data_period_days"] == 14

    def test_refresh_keeps_recent_recipes_and_evicts_idle_ones(self, stub_compute):
        now = time.time()
        recent = normalize_recipe(None, None, 24, None, None)
        idle = normalize_recipe(None, None, 168, None, None)
        LocationsResponseCache._RECIPE_ACCESS[recent] = now - 60
        LocationsResponseCache._RECIPE_ACCESS[idle] = now - 2 * (
            LocationsResponseCache._KEY_IDLE_SEC + 1
        )

        LocationsResponseCache._refresh_active_keys()

        assert recent in LocationsResponseCache._RECIPES
        assert idle not in LocationsResponseCache._RECIPES
        assert idle not in LocationsResponseCache._RECIPE_ACCESS

    def test_refresh_failure_keeps_previous_cached_value(self, monkeypatch):
        recipe = LocationsResponseCache._DEFAULT_RECIPE
        key = (1.0, 2.0, None, None)
        LocationsResponseCache._CACHE[key] = (time.time(), b'{"previous": true}')
        LocationsResponseCache._LAST_ACCESS[key] = time.time()
        LocationsResponseCache._RECIPES[recipe] = key

        def exploding_compute(*args, **kwargs):
            raise RuntimeError("database unavailable")

        monkeypatch.setattr(
            locations_cache_module, "compute_locations_payload", exploding_compute
        )
        LocationsResponseCache._refresh_active_keys()

        assert LocationsResponseCache._CACHE[key][1] == b'{"previous": true}'

    def test_relative_recipe_rolls_to_new_grid_bucket(self, monkeypatch, stub_compute):
        """Refresher re-resolves relative recipes with the current clock.

        hours-based windows derive their bounds from "now", so the minted
        key must follow the grid forward; the superseded bucket's entry
        is dropped instead of lingering until idle eviction.
        """
        grid = _LOCATIONS_NOW_GRID_SECONDS
        clock = {"now": 20000 * grid}
        monkeypatch.setattr(
            locations_cache_module, "_grid_now", lambda: clock["now"]
        )

        recipe = normalize_recipe(None, None, 24, None, None)
        LocationsResponseCache._RECIPE_ACCESS[recipe] = time.time()
        LocationsResponseCache._refresh_active_keys()

        first_key = LocationsResponseCache._RECIPES[recipe]
        assert first_key in LocationsResponseCache._CACHE

        clock["now"] += grid  # one grid bucket elapses
        LocationsResponseCache._refresh_active_keys()

        second_key = LocationsResponseCache._RECIPES[recipe]
        assert second_key != first_key
        assert second_key in LocationsResponseCache._CACHE
        # The request path resolves the *current* bucket's key and hits
        body = LocationsResponseCache.serve(recipe, second_key)
        assert body == LocationsResponseCache._CACHE[second_key][1]
        # Superseded bucket dropped eagerly
        assert first_key not in LocationsResponseCache._CACHE

    def test_explicit_recipe_key_is_static_across_refreshes(
        self, monkeypatch, stub_compute
    ):
        grid = _LOCATIONS_NOW_GRID_SECONDS
        end = 40000 * grid
        start = end - 24 * 3600
        recipe = normalize_recipe(start, end, None, None, None)
        LocationsResponseCache._RECIPE_ACCESS[recipe] = time.time()

        LocationsResponseCache._refresh_active_keys()
        first_key = LocationsResponseCache._RECIPES[recipe]
        LocationsResponseCache._refresh_active_keys()

        assert LocationsResponseCache._RECIPES[recipe] == first_key
        assert first_key in LocationsResponseCache._CACHE

    def test_serve_starts_refresher_and_stop_joins_it(self, stub_compute):
        recipe = normalize_recipe(None, None, 24, None, None)
        link, _ = resolve_locations_filters(recipe)
        LocationsResponseCache.serve(recipe, locations_cache_key(link))

        thread = LocationsResponseCache._refresher_thread
        assert thread is not None and thread.is_alive()

        LocationsResponseCache.stop_background_refresh()
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    def test_refresher_restarts_after_stop(self, stub_compute):
        """Dead threads (e.g. after fork) are replaced, not latched."""
        LocationsResponseCache.start_background_refresh()
        first = LocationsResponseCache._refresher_thread
        LocationsResponseCache.stop_background_refresh()

        LocationsResponseCache.start_background_refresh()
        second = LocationsResponseCache._refresher_thread

        assert second is not first and second.is_alive()
        LocationsResponseCache.stop_background_refresh()

    def test_repeated_start_calls_spawn_one_thread(self, stub_compute, monkeypatch):
        import threading

        started = []
        real_start = threading.Thread.start

        def counting_start(self):
            started.append(self)
            real_start(self)

        monkeypatch.setattr(threading.Thread, "start", counting_start)

        LocationsResponseCache.start_background_refresh()
        thread = LocationsResponseCache._refresher_thread
        LocationsResponseCache.start_background_refresh()

        assert len(started) == 1
        assert LocationsResponseCache._refresher_thread is thread


class TestEndpointResponseCache:
    """The endpoint replays pre-serialized bytes and skips recompute."""

    @pytest.fixture
    def mocked_location_services(self):
        from unittest.mock import patch

        with (
            patch(
                "src.malla.routes.api_routes.TracerouteService.get_network_graph_data",
                return_value=EMPTY_NETWORK_DATA,
            ) as graph_mock,
            patch(
                "src.malla.routes.api_routes.LocationService.get_packet_links",
                return_value=[],
            ) as packet_links_mock,
        ):
            yield {"graph": graph_mock, "packet_links": packet_links_mock}

    def test_second_identical_request_replays_cached_bytes(
        self, client, mocked_location_services, no_auto_refresh
    ):
        first = client.get("/api/locations?hours=24")
        assert first.status_code == 200
        assert first.content_type == "application/json"
        assert mocked_location_services["graph"].call_count == 1

        second = client.get("/api/locations?hours=24")

        assert second.status_code == 200
        assert second.data == first.data
        # Served from the response cache: no second compute
        assert mocked_location_services["graph"].call_count == 1
        assert mocked_location_services["packet_links"].call_count == 1
        assert json.loads(second.data)["data_period_days"] == 14

    def test_different_search_computes_separately(
        self, client, mocked_location_services, no_auto_refresh
    ):
        client.get("/api/locations?hours=24&search=alpha")
        client.get("/api/locations?hours=24&search=beta")

        assert mocked_location_services["graph"].call_count == 2

    def test_wsgi_post_fork_warms_both_caches(self, monkeypatch, stub_compute):
        from src.malla.wsgi import _warm_caches_post_fork

        monkeypatch.setattr(
            "src.malla.services.analytics_service.AnalyticsService"
            ".start_background_refresh",
            lambda: None,
        )
        _warm_caches_post_fork(None, None)

        assert (
            LocationsResponseCache._refresher_thread is not None
            and LocationsResponseCache._refresher_thread.is_alive()
        )
