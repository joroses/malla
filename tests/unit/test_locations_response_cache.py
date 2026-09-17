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
    is_relative_recipe,
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


class TestNormalizeRecipe:
    def test_canonicalize_equivalent_explicit_bounds(self):
        grid = _LOCATIONS_NOW_GRID_SECONDS
        start1 = 1000 * grid + 15
        end1 = start1 + 3600
        start2 = 1000 * grid + 180
        end2 = start2 + 3550

        r1 = normalize_recipe(start1, end1, None, None, None)
        r2 = normalize_recipe(start2, end2, None, None, None)

        assert r1 == r2
        assert r1[0] % grid == 0
        assert r1[1] % grid == 0

    def test_explicit_bounds_supersede_hours(self):
        grid = _LOCATIONS_NOW_GRID_SECONDS
        start = 1000 * grid
        end = start + 3600

        r_with_hours = normalize_recipe(start, end, 24, 42, "node")
        r_without_hours = normalize_recipe(start, end, None, 42, "node")

        assert r_with_hours == r_without_hours
        assert r_with_hours[2] is None

    def test_search_and_hours_sanitized(self):
        r1 = normalize_recipe(None, None, 24, None, "  router-1  ")
        assert r1[2] == 24.0
        assert r1[4] == "router-1"

        r2 = normalize_recipe(None, None, None, None, "   ")
        assert r2[4] is None

    def test_inverted_bounds_kept_for_validation(self):
        r = normalize_recipe(1000, 500, None, None, None)
        assert r[0] == 1000
        assert r[1] == 500
        with pytest.raises(LocationsWindowError):
            resolve_locations_filters(r)

    def test_is_relative_recipe(self):
        # Default recipe (all None) is relative
        assert is_relative_recipe(normalize_recipe(None, None, None, None, None))
        # Hours without explicit bounds is relative
        assert is_relative_recipe(normalize_recipe(None, None, 24, None, None))
        # Start only or end only is relative
        assert is_relative_recipe((1000.0, None, None, None, None))
        assert is_relative_recipe((None, 2000.0, None, None, None))
        # Explicit start and end is NOT relative
        assert not is_relative_recipe((1000.0, 2000.0, None, None, None))


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

    def test_request_after_rollover_before_refresh_serves_bounded_stale(
        self, monkeypatch, stub_compute, no_auto_refresh
    ):
        grid = _LOCATIONS_NOW_GRID_SECONDS
        t0 = 20000 * grid
        clock = {"now": t0}
        monkeypatch.setattr(
            locations_cache_module, "_grid_now", lambda: clock["now"]
        )

        recipe = normalize_recipe(None, None, 24, None, None)
        link0, pos0 = resolve_locations_filters(recipe, server_now=t0)
        key0 = locations_cache_key(link0)

        # 1. Warm initial entry at t0
        LocationsResponseCache.store(
            recipe, key0, locations_cache_module.compute_locations_payload(link0, pos0)
        )
        assert stub_compute["n"] == 1
        assert key0 in LocationsResponseCache._CACHE
        initial_body = LocationsResponseCache._CACHE[key0][1]

        # 2. Advance clock past the 300s grid boundary
        clock["now"] = t0 + grid
        link1, _ = resolve_locations_filters(recipe, server_now=clock["now"])
        key1 = locations_cache_key(link1)
        assert key1 != key0
        assert key1 not in LocationsResponseCache._CACHE

        # 3. Visitor arrives AFTER rollover but BEFORE refresher runs:
        # Must return the cached body from key0 without triggering inline compute
        body = LocationsResponseCache.serve(recipe, key1)
        assert body is initial_body
        assert stub_compute["n"] == 1  # No synchronous recompute!

        # 4. Returned body preserves the actual filters applied when it was computed
        payload = json.loads(body)
        assert payload["filters_applied"]["end_time"] == t0

        # 5. Background refresher subsequently executes:
        # Mints key1 and cleans up key0
        LocationsResponseCache._refresh_active_keys()
        assert stub_compute["n"] == 3  # default recipe (1) + key1 (1)
        assert key1 in LocationsResponseCache._CACHE
        assert key0 not in LocationsResponseCache._CACHE
        assert LocationsResponseCache._RECIPES[recipe] == key1

    def test_request_after_rollover_past_max_stale_recomputes_inline(
        self, monkeypatch, stub_compute, no_auto_refresh
    ):
        grid = _LOCATIONS_NOW_GRID_SECONDS
        t0 = 20000 * grid
        clock = {"now": t0}
        monkeypatch.setattr(
            locations_cache_module, "_grid_now", lambda: clock["now"]
        )

        recipe = normalize_recipe(None, None, 24, None, None)
        link0, pos0 = resolve_locations_filters(recipe, server_now=t0)
        key0 = locations_cache_key(link0)

        LocationsResponseCache.store(
            recipe, key0, locations_cache_module.compute_locations_payload(link0, pos0)
        )

        # Advance time past _MAX_STALE_SEC
        stale_time = time.time() + 2 * LocationsResponseCache._MAX_STALE_SEC
        monkeypatch.setattr(time, "time", lambda: stale_time)
        clock["now"] = t0 + 2 * grid
        link1, _ = resolve_locations_filters(recipe, server_now=clock["now"])
        key1 = locations_cache_key(link1)

        # Expired stale response must NOT be served; returns None for inline recompute
        assert LocationsResponseCache.serve(recipe, key1) is None

    def test_first_time_store_registers_recipe_for_refresh(
        self, stub_compute, no_auto_refresh
    ):
        recipe = normalize_recipe(None, None, 12, None, None)
        link, pos = resolve_locations_filters(recipe)
        key = locations_cache_key(link)

        # Cold request miss
        assert LocationsResponseCache.serve(recipe, key) is None
        assert recipe not in LocationsResponseCache._RECIPE_ACCESS

        # Successful inline store must record request activity
        LocationsResponseCache.store(
            recipe, key, locations_cache_module.compute_locations_payload(link, pos)
        )
        assert recipe in LocationsResponseCache._RECIPE_ACCESS
        assert LocationsResponseCache._RECIPE_ACCESS[recipe] > 0

        # Background refresher should now recompute this recipe without manual seeding
        stub_compute["n"] = 0
        LocationsResponseCache._refresh_active_keys()
        # Default recipe (1) + recipe (1) = 2
        assert stub_compute["n"] == 2


class TestMintOrdering:
    """A late mint for an older window must never replace a newer one."""

    def test_late_older_mint_cannot_replace_or_delete_newer_response(
        self, no_auto_refresh, stub_compute
    ):
        """Refresh started pre-rollover, request minted the new window first.

        The refresh resolves key0 and computes slowly; the grid rolls, a
        request resolves key1, computes and publishes it; only then does
        the refresh finish. Its older publication must not repoint the
        recipe, delete the newer response, or become the next request's
        stale-while-revalidate body.
        """
        grid = _LOCATIONS_NOW_GRID_SECONDS
        t0 = 20000 * grid
        recipe = normalize_recipe(None, None, 24, None, None)
        link0, _ = resolve_locations_filters(recipe, server_now=t0)
        key0 = locations_cache_key(link0)
        link1, _ = resolve_locations_filters(recipe, server_now=t0 + grid)
        key1 = locations_cache_key(link1)
        assert key1[1] > key0[1]

        # Post-rollover request computes and publishes the newer window
        newer_body = LocationsResponseCache.store(
            recipe, key1, {"locations": [{"node_id": "new"}]}
        )
        assert key1 in LocationsResponseCache._CACHE

        # The pre-rollover refresh finishes late with the older window
        older_body = LocationsResponseCache._mint(
            recipe, key0, {"locations": [{"node_id": "old"}]}
        )

        # The late mint's caller still gets its computed response...
        assert json.loads(older_body)["locations"][0]["node_id"] == "old"
        # ...but nothing was published: the recipe keeps pointing at the
        # newer key, whose entry survives.
        assert LocationsResponseCache._RECIPES[recipe] == key1
        assert LocationsResponseCache._CACHE[key1][1] is newer_body
        assert LocationsResponseCache.serve(recipe, key1) is newer_body
        assert key0 not in LocationsResponseCache._CACHE

    def test_late_older_mint_keeps_shared_key_of_static_recipe(
        self, no_auto_refresh, stub_compute, monkeypatch
    ):
        """A grouped refresh may still store a key a static recipe maps to.

        The refresher groups recipes by resolved key: when the late older
        mint also serves an explicit (static) recipe that still maps to
        that key, the entry must be stored and kept — only the relative
        recipe's newer mapping is protected.
        """
        grid = _LOCATIONS_NOW_GRID_SECONDS
        t0 = 30000 * grid
        clock = {"now": t0}
        monkeypatch.setattr(
            locations_cache_module, "_grid_now", lambda: clock["now"]
        )

        relative = normalize_recipe(None, None, 6, None, None)
        static = normalize_recipe(t0 - 6 * 3600, t0, None, None, None)
        link0, pos0 = resolve_locations_filters(relative, server_now=t0)
        key0 = locations_cache_key(link0)
        static_link, _ = resolve_locations_filters(static)
        assert locations_cache_key(static_link) == key0

        # Both recipes currently resolve to key0; the relative one is
        # already published onto the post-rollover key1 by a request.
        link1, _ = resolve_locations_filters(relative, server_now=t0 + grid)
        key1 = locations_cache_key(link1)
        LocationsResponseCache._RECIPES[relative] = key1
        LocationsResponseCache._CACHE[key1] = (time.time(), b'{"newer": true}')
        LocationsResponseCache._RECIPES[static] = key0

        LocationsResponseCache._mint([relative, static], key0, {"v": "old"})

        assert LocationsResponseCache._RECIPES[relative] == key1
        assert LocationsResponseCache._RECIPES[static] == key0
        # key0 stays stored and fresh: the static recipe still needs it
        assert key0 in LocationsResponseCache._CACHE
        assert key1 in LocationsResponseCache._CACHE

    def test_mint_accepts_various_recipe_iterables_and_single_recipe(
        self, no_auto_refresh
    ):
        """_mint handles single recipes and various iterable types properly."""
        r1 = normalize_recipe(None, None, 1, None, None)
        r2 = normalize_recipe(None, None, 2, None, None)
        link1, _ = resolve_locations_filters(r1)
        key1 = locations_cache_key(link1)

        # Single recipe
        LocationsResponseCache._mint(r1, key1, {"v": 1})
        assert LocationsResponseCache._RECIPES[r1] == key1

        # Tuple of recipes
        link2, _ = resolve_locations_filters(r2)
        key2 = locations_cache_key(link2)
        LocationsResponseCache._mint((r1, r2), key2, {"v": 2})
        assert LocationsResponseCache._RECIPES[r1] == key2
        assert LocationsResponseCache._RECIPES[r2] == key2

        # Set of recipes
        LocationsResponseCache._mint({r1}, key1, {"v": 3})
        assert LocationsResponseCache._RECIPES[r1] == key1

        # Generator of recipes
        LocationsResponseCache._mint((r for r in [r1, r2]), key2, {"v": 4})
        assert LocationsResponseCache._RECIPES[r1] == key2
        assert LocationsResponseCache._RECIPES[r2] == key2


class TestInFlightDeduplication:
    """Cache-miss requests join running computes instead of duplicating them."""

    def test_request_reuses_inflight_compute(self, no_auto_refresh, stub_compute):
        """A request arriving mid-refresh gets that compute's body."""
        import threading

        recipe = normalize_recipe(None, None, 24, None, None)
        link, pos = resolve_locations_filters(recipe)
        key = locations_cache_key(link)

        # Simulate the refresher mid-compute for this exact window
        record = LocationsResponseCache._try_begin_compute(key)
        assert record is not None

        result = {}

        def request_thread():
            result["body"] = LocationsResponseCache.get_or_compute(
                recipe, key, link, pos
            )

        waiter = threading.Thread(target=request_thread)
        waiter.start()
        try:
            time.sleep(0.05)  # let the request block on the in-flight record
            payload = locations_cache_module.compute_locations_payload(link, pos)
            body = LocationsResponseCache.store(recipe, key, payload)
            LocationsResponseCache._end_compute(key, record, body)
        finally:
            waiter.join(timeout=5)

        assert not waiter.is_alive()
        assert result["body"] is body
        # Exactly one compute: the waiting request did not recompute
        assert stub_compute["n"] == 1

    def test_request_computes_inline_after_inflight_wait_timeout(
        self, no_auto_refresh, stub_compute, monkeypatch
    ):
        """A stuck in-flight owner must not block the request forever."""
        monkeypatch.setattr(LocationsResponseCache, "_INFLIGHT_WAIT_SEC", 0.05)

        recipe = normalize_recipe(None, None, 24, None, None)
        link, pos = resolve_locations_filters(recipe)
        key = locations_cache_key(link)
        assert LocationsResponseCache._try_begin_compute(key) is not None

        body = LocationsResponseCache.get_or_compute(recipe, key, link, pos)

        assert stub_compute["n"] == 1  # fell back to computing inline
        assert LocationsResponseCache.serve(recipe, key) is body

    def test_request_computes_inline_after_owner_failure(
        self, no_auto_refresh, monkeypatch
    ):
        """An owner whose compute raised must not take waiters down with it."""
        import threading

        calls = {"n": 0}

        def flaky_compute(link_filters, position_filters):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("database unavailable")
            return {"locations": [], "traceroute_links": [], "packet_links": []}

        monkeypatch.setattr(
            locations_cache_module, "compute_locations_payload", flaky_compute
        )

        recipe = normalize_recipe(None, None, 24, None, None)
        link, pos = resolve_locations_filters(recipe)
        key = locations_cache_key(link)

        owner_failure = {}

        def owner_thread():
            try:
                LocationsResponseCache.get_or_compute(recipe, key, link, pos)
            except RuntimeError as err:
                owner_failure["err"] = err

        owner = threading.Thread(target=owner_thread)
        owner.start()
        owner.join(timeout=5)
        assert not owner.is_alive()  # owner surfaced the failure
        assert isinstance(owner_failure.get("err"), RuntimeError)

        # The next request must still get a computed response
        body = LocationsResponseCache.get_or_compute(recipe, key, link, pos)
        assert calls["n"] == 2
        assert LocationsResponseCache.serve(recipe, key) is body

    def test_refresh_skips_window_computed_by_request_thread(self, stub_compute):
        """The refresher does not duplicate a request thread's compute."""
        link, _ = resolve_locations_filters(LocationsResponseCache._DEFAULT_RECIPE)
        key = locations_cache_key(link)
        record = LocationsResponseCache._try_begin_compute(key)
        assert record is not None

        LocationsResponseCache._refresh_active_keys()

        # Default-recipe compute skipped: a request owns the window
        assert stub_compute["n"] == 0
        assert key not in LocationsResponseCache._CACHE

        LocationsResponseCache._end_compute(key, record, b"{}")

    def test_waiter_records_recipe_access(self, no_auto_refresh, stub_compute):
        """A request served from an in-flight compute stays refresh-warm."""
        import threading

        recipe = normalize_recipe(None, None, 24, None, None)
        link, pos = resolve_locations_filters(recipe)
        key = locations_cache_key(link)
        record = LocationsResponseCache._try_begin_compute(key)

        result = {}

        def request_thread():
            result["body"] = LocationsResponseCache.get_or_compute(
                recipe, key, link, pos
            )

        waiter = threading.Thread(target=request_thread)
        waiter.start()
        time.sleep(0.05)
        LocationsResponseCache._end_compute(key, record, b'{"served": 1}')
        waiter.join(timeout=5)

        assert result["body"] == b'{"served": 1}'
        assert LocationsResponseCache._RECIPE_ACCESS[recipe] > 0


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

    def test_refresh_deduplicates_equivalent_recipes_to_single_compute(
        self, monkeypatch, stub_compute
    ):
        grid = _LOCATIONS_NOW_GRID_SECONDS
        clock = {"now": 30000 * grid}
        monkeypatch.setattr(
            locations_cache_module, "_grid_now", lambda: clock["now"]
        )

        now_ts = time.time()
        start = clock["now"] - 24 * 3600
        end = clock["now"]

        # 5 recipes (relative, explicit, un-snapped raw tuple) that resolve to the same 24h window
        recipes = [
            normalize_recipe(None, None, 24, None, None),
            normalize_recipe(None, None, 24.0, None, None),
            (start, end, None, None, None),
            (start + 10, end - 10, None, None, None),
            (start, end, 24, None, None),
        ]

        for r in recipes:
            LocationsResponseCache._RECIPE_ACCESS[r] = now_ts

        LocationsResponseCache._refresh_active_keys()

        # 1 compute for default recipe + 1 compute for the shared 24h window = 2 total computes (not 6)
        assert stub_compute["n"] == 2

        key_24h = (start, end, None, None)
        assert key_24h in LocationsResponseCache._CACHE
        for r in recipes:
            assert LocationsResponseCache._RECIPES[r] == key_24h

    def test_refresh_grouped_recipes_cleans_superseded_key(
        self, monkeypatch, stub_compute
    ):
        grid = _LOCATIONS_NOW_GRID_SECONDS
        clock = {"now": 40000 * grid}
        monkeypatch.setattr(
            locations_cache_module, "_grid_now", lambda: clock["now"]
        )

        r1 = normalize_recipe(None, None, 6, None, None)
        r2 = (None, None, 6.0, None, None)

        now_ts = time.time()
        LocationsResponseCache._RECIPE_ACCESS[r1] = now_ts
        LocationsResponseCache._RECIPE_ACCESS[r2] = now_ts

        LocationsResponseCache._refresh_active_keys()
        first_key = LocationsResponseCache._RECIPES[r1]
        assert first_key in LocationsResponseCache._CACHE

        clock["now"] += grid
        LocationsResponseCache._refresh_active_keys()

        second_key = LocationsResponseCache._RECIPES[r1]
        assert second_key != first_key
        assert second_key in LocationsResponseCache._CACHE
        assert LocationsResponseCache._RECIPES[r2] == second_key

        assert first_key not in LocationsResponseCache._CACHE
        assert first_key not in LocationsResponseCache._LAST_ACCESS

    def test_background_refresh_does_not_extend_recipe_access_lifetime(
        self, stub_compute
    ):
        recipe = normalize_recipe(None, None, 12, None, None)
        link, pos = resolve_locations_filters(recipe)
        key = locations_cache_key(link)

        LocationsResponseCache.store(
            recipe, key, locations_cache_module.compute_locations_payload(link, pos)
        )
        stored_at = LocationsResponseCache._RECIPE_ACCESS[recipe]

        # Ensure clock time moves forward
        time.sleep(0.01)
        LocationsResponseCache._refresh_active_keys()

        # Background refresh must NOT overwrite _RECIPE_ACCESS with current time
        assert LocationsResponseCache._RECIPE_ACCESS[recipe] == stored_at

    def test_orphaned_recipes_evicted_from_recipes_dict(self, stub_compute):
        # A recipe present in _RECIPES but absent from _RECIPE_ACCESS
        orphan = normalize_recipe(None, None, 48, None, None)
        dummy_key = (1.0, 2.0, None, None)
        LocationsResponseCache._RECIPES[orphan] = dummy_key

        LocationsResponseCache._refresh_active_keys()

        assert orphan not in LocationsResponseCache._RECIPES


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
