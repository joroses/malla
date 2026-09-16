"""
Whole-response cache with a background refresher for /api/locations.

The /api/locations payload (node positions plus traceroute/packet link
aggregates over the selected window) takes on the order of 15 s to
compute and serializes to multiple megabytes. Two costs follow:

1. Recompute latency: whenever the service-layer TTL caches lapse, a
   visitor paid the full compute inside the request.
2. Serialization cost: every hit re-ran jsonify over ~5 MB of data.

Both are solved the same way the analytics dashboard solves them
(``services/analytics_service.py``): a daemon refresher thread recomputes
the *whole response* for recently served request shapes once per minute,
pre-serializes it to JSON bytes, and requests replay those bytes for up
to ``_MAX_STALE_SEC``. Only a never-seen filter combination (or a dead
refresher) computes inline.

Request shapes are stored as "recipes" of raw request parameters, not as
resolved windows: relative recipes (``hours=…``, or the parameterless
default) resolve against the current clock, so the refresher re-resolves
them every cycle and the minted entry always matches the window the next
request will ask for — even as the 300 s grid rolls the resolved bounds
forward. Explicit start/end recipes resolve statically. Stale entries
from superseded grid buckets are dropped when the recipe rolls forward,
and recipes not served for ``_KEY_IDLE_SEC`` stop being refreshed
altogether, so refresher work and memory track actual usage.
"""

import json
import logging
import math
import threading
import time
from collections.abc import Iterable
from typing import Any

from ..utils.serialization_utils import sanitize_floats
from .location_service import LocationService
from .traceroute_service import TracerouteService

logger = logging.getLogger(__name__)

# /api/locations resolves every link window onto this grid. Server-derived
# bounds (the end from "now", the start for hours-only requests) are snapped
# here, and client-supplied start_time/end_time are floored/ceiled onto the
# same grid: raw datetime.now() floats carry microsecond precision and the
# map's per-visit start_time (Math.floor(now) - N*3600, recomputed at every
# page load) changes by the second, so un-snapped bounds gave every request
# a unique filter set and turned the service-layer TTL caches into
# guaranteed misses. Snapping widens a window by less than one grid step
# per side.
#
# The step must also comfortably exceed the service-layer caches' 60 s TTL.
# Snapped bounds roll with the clock every grid step, and an entry minted
# under one bucket's bounds becomes unreachable the moment the next bucket
# starts, so a 30 s grid capped the effective hit window below the TTL:
# reloads more than 30 s apart always minted fresh keys and recomputed in
# full. A 300 s step (5x the TTL) lets a revisit inside the TTL resolve the
# same bounds with probability 1 - gap/300 (~80 % for a 60 s gap), keeps the
# per-side window widening under five minutes (0.35 % of the default 24 h
# preset, 8 % of the smallest 1 h preset), and divides every hour preset
# and the 14-day cap exactly, so derived starts stay grid-aligned.
_LOCATIONS_NOW_GRID_SECONDS = 300

# Wide position-lookup window (performance cap only): nodes whose last GPS
# fix is older than the selected link window must stay visible at their
# last known good position instead of disappearing.
_LOCATIONS_MAX_WINDOW_SECONDS = 14 * 24 * 3600

# Raw request parameters identifying one response shape. None means the
# parameter was absent; hours falsy (0) normalizes to None (matches the
# endpoint's historical truthiness handling). Relative recipes re-resolve
# against the current clock on every refresh; explicit start/end recipes
# resolve statically.
Recipe = tuple[float | None, float | None, float | None, int | None, str | None]

# Resolved link window (grid-snapped) plus the scalar filters — the cache
# identity of one exact response.
Key = tuple[float, float, int | None, str | None]


class LocationsWindowError(ValueError):
    """Raised when the requested time window is empty or inverted."""


def _grid_now() -> float:
    """Current wall clock ceiled onto the cache grid.

    ``datetime`` is imported lazily so tests can freeze the clock by
    swapping ``sys.modules["datetime"]`` after this module was imported.
    """
    from datetime import datetime

    # Ceil (not floor) so the derived end can never precede a client
    # start_time picked seconds ago within the current grid bucket.
    return (
        math.ceil(datetime.now().timestamp() / _LOCATIONS_NOW_GRID_SECONDS)
        * _LOCATIONS_NOW_GRID_SECONDS
    )


def normalize_recipe(
    start_arg: float | None,
    end_arg: float | None,
    hours_arg: float | None,
    gateway_id: int | None,
    search: str | None,
) -> Recipe:
    """Canonicalize raw request parameters into a recipe.

    When explicit start and end times are provided and form a valid window,
    they are snapped onto the cache grid immediately so equivalent request
    shapes share a single recipe identity instead of multiplying refresh work.
    """
    clean_hours = float(hours_arg) if hours_arg else None
    clean_search = search.strip() if search else None

    if start_arg is not None and end_arg is not None:
        if start_arg < end_arg:
            snapped_start = (
                math.floor(start_arg / _LOCATIONS_NOW_GRID_SECONDS)
                * _LOCATIONS_NOW_GRID_SECONDS
            )
            snapped_end = (
                math.ceil(end_arg / _LOCATIONS_NOW_GRID_SECONDS)
                * _LOCATIONS_NOW_GRID_SECONDS
            )
            if snapped_end - snapped_start > _LOCATIONS_MAX_WINDOW_SECONDS:
                snapped_start = snapped_end - _LOCATIONS_MAX_WINDOW_SECONDS
            return (snapped_start, snapped_end, None, gateway_id, clean_search or None)
        # Inverted or empty window: keep raw values so resolve_locations_filters can raise LocationsWindowError
        return (start_arg, end_arg, clean_hours, gateway_id, clean_search or None)

    return (start_arg, end_arg, clean_hours, gateway_id, clean_search or None)


def is_relative_recipe(recipe: Recipe) -> bool:
    """Return True if the recipe derives its window bounds from current time."""
    start_arg, end_arg, _, _, _ = recipe
    return start_arg is None or end_arg is None


def resolve_locations_filters(
    recipe: Recipe, server_now: float | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a recipe into (link_filters, position_filters).

    Pure function shared by the request path and the background refresher
    (which re-resolves relative recipes with the current clock). Raises
    :class:`LocationsWindowError` for an explicitly inverted window.

    Explicit start/end win; otherwise hours/max_age_hours is applied
    relative to now; with no time parameter at all the endpoint keeps its
    historical 14-day default. Every bound is snapped onto the cache grid
    (start down, end up; derived bounds are already grid multiples) so two
    map visits seconds apart resolve identical filters and hit the caches.
    """
    start_arg, end_arg, hours_arg, gateway_id, search = recipe
    if server_now is None:
        server_now = _grid_now()

    # Wide position-lookup window (performance cap only). This window is
    # intentionally NOT narrowed by the client's time selection so nodes
    # with stale GPS fixes remain visible while active. Both bounds are
    # snapped onto the cache grid — this dict also becomes the link window
    # for parameterless requests.
    position_filters: dict[str, Any] = {
        "start_time": server_now - _LOCATIONS_MAX_WINDOW_SECONDS,
        "end_time": server_now,
    }

    link_filters: dict[str, Any] = {}
    if start_arg is not None or end_arg is not None or hours_arg:
        if start_arg is None:
            lookback = hours_arg * 3600 if hours_arg else _LOCATIONS_MAX_WINDOW_SECONDS
            start_arg = (end_arg if end_arg is not None else server_now) - lookback
        if end_arg is None:
            end_arg = server_now
        if start_arg >= end_arg:
            raise LocationsWindowError("start_time must be before end_time")
        start_arg = (
            math.floor(start_arg / _LOCATIONS_NOW_GRID_SECONDS)
            * _LOCATIONS_NOW_GRID_SECONDS
        )
        end_arg = (
            math.ceil(end_arg / _LOCATIONS_NOW_GRID_SECONDS)
            * _LOCATIONS_NOW_GRID_SECONDS
        )
        # Cap the aggregation window for performance
        if end_arg - start_arg > _LOCATIONS_MAX_WINDOW_SECONDS:
            start_arg = end_arg - _LOCATIONS_MAX_WINDOW_SECONDS
        link_filters["start_time"] = start_arg
        link_filters["end_time"] = end_arg
    else:
        link_filters.update(position_filters)

    if gateway_id is not None:
        link_filters["gateway_id"] = gateway_id
        position_filters["gateway_id"] = gateway_id

    if search:
        link_filters["search"] = search
        position_filters["search"] = search

    return link_filters, position_filters


def locations_cache_key(link_filters: dict[str, Any]) -> Key:
    """Cache identity of one exact /api/locations response."""
    return (
        link_filters["start_time"],
        link_filters["end_time"],
        link_filters.get("gateway_id"),
        link_filters.get("search"),
    )


def compute_locations_payload(
    link_filters: dict[str, Any], position_filters: dict[str, Any]
) -> dict[str, Any]:
    """Compute the full /api/locations payload from the database (no caching).

    Expensive operations run exactly once and their results are passed
    down: the network graph feeds both the position enrichment and the
    traceroute link conversion, and the packet links feed both the
    position enrichment and the response body.
    """
    compute_start = time.time()

    network_filters: dict[str, Any] = {}
    if link_filters.get("start_time"):
        network_filters["start_time"] = link_filters["start_time"]
    if link_filters.get("end_time"):
        network_filters["end_time"] = link_filters["end_time"]
    if link_filters.get("gateway_id"):
        network_filters["gateway_id"] = link_filters["gateway_id"]

    # The explicit start/end filters take precedence inside the service;
    # hours is kept consistent with the resolved window for cache keys.
    time_diff = link_filters["end_time"] - link_filters["start_time"]
    hours = max(1, min(168, int(time_diff / 3600)))  # Between 1 and 168 hours

    # 1. Network topology data (used by both node locations and traceroute links)
    network_data = TracerouteService.get_network_graph_data(
        hours=hours,
        include_indirect=False,
        filters=network_filters,
    )

    # 2. Packet links (used by node locations and returned in the response)
    packet_links = LocationService.get_packet_links(link_filters)

    # 3. Enhanced location data, passing pre-computed data. Position
    #    lookups use the wide window; node activity timestamps come from
    #    the window-scoped network/packet data computed above.
    locations = LocationService.get_node_locations(
        position_filters, network_data=network_data, packet_links=packet_links
    )

    # 4. Traceroute links, passing pre-computed network data
    traceroute_links = LocationService.get_traceroute_links(
        link_filters, network_data=network_data
    )

    logger.info(
        "Computed /api/locations payload in %.3fs (filters: %s)",
        time.time() - compute_start,
        link_filters,
    )

    return {
        "locations": locations,
        "traceroute_links": traceroute_links,
        "packet_links": packet_links,
        "total_count": len(locations) if isinstance(locations, list) else 0,
        "filters_applied": link_filters,
        "data_period_days": 14,
    }


def serialize_locations_payload(payload: dict[str, Any]) -> bytes:
    """Pre-serialize the payload to JSON bytes once, at compute time.

    Every cache hit then replays these bytes verbatim instead of
    re-running jsonify over multi-megabyte data. NaN/Inf values are
    sanitized exactly like ``safe_jsonify`` does.
    """
    try:
        sanitized = sanitize_floats(payload)
    except Exception as err:
        logger.debug(f"JSON sanitation failed, serializing original data: {err}")
        sanitized = payload
    return json.dumps(sanitized, separators=(",", ":")).encode("utf-8")


class _InFlightCompute:
    """Shared per-key compute record for request/refresher deduplication.

    The owner thread (whichever of the background refresher or a request
    first started the compute) sets :attr:`body` and the event when done;
    waiters read the body directly instead of running a second
    multi-second compute for the same window.
    """

    __slots__ = ("event", "body")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.body: bytes | None = None


class LocationsResponseCache:
    """In-process, pre-serialized whole-response cache for /api/locations.

    Requests are served whenever an entry for the resolved key exists and
    is younger than ``_MAX_STALE_SEC``; the background refresher keeps
    entries of recently served recipes warm so requests never sit inside
    a compute window. Each worker process maintains its own cache (the
    same per-worker duplication tradeoff the analytics refresher makes).
    """

    # key → (minted_at, pre-serialized JSON body)
    _CACHE: dict[Key, tuple[float, bytes]] = {}
    # key → last time it was served or minted (idle eviction bookkeeping)
    _LAST_ACCESS: dict[Key, float] = {}
    # recipe → key currently minted for it (superseded-key cleanup)
    _RECIPES: dict[Recipe, Key] = {}
    # recipe → last time a request used it (refresher activity tracking)
    _RECIPE_ACCESS: dict[Recipe, float] = {}

    # key → in-flight compute record; a cache-miss request joins the
    # running compute (usually the refresher's) instead of duplicating it
    _INFLIGHT: dict[Key, _InFlightCompute] = {}
    _INFLIGHT_LOCK = threading.Lock()
    # How long a request waits on an in-flight compute before falling
    # back to computing inline itself (bounds the damage of a stuck owner)
    _INFLIGHT_WAIT_SEC: float = 20.0

    _REFRESH_INTERVAL_SEC: float = 60.0
    _MAX_STALE_SEC: float = 300.0

    # Recipes are refreshed only while they are being served; idle ones
    # are evicted so refresher work and cache memory track actual usage.
    _KEY_IDLE_SEC: float = 900.0

    _DEFAULT_RECIPE: Recipe = (None, None, None, None, None)

    _refresher_thread: threading.Thread | None = None
    _refresher_lock = threading.Lock()
    _refresher_stop = threading.Event()

    # ------------------------------------------------------------------
    # Request path
    # ------------------------------------------------------------------

    @classmethod
    def serve(cls, recipe: Recipe, key: Key) -> bytes | None:
        """Return the pre-serialized body for *key* if fresh enough.

        Also records the access so the background refresher keeps this
        recipe warm. ``None`` means the caller must compute inline (never
        seen recipe, or the refresher has demonstrably died).
        """
        # Fork safety: after gunicorn preload forks workers, the inherited
        # thread object is dead but the flag still says "started" — the
        # aliveness check recovers by starting a fresh thread.
        cls._ensure_background_refresh()

        now_ts = time.time()
        cached = cls._CACHE.get(key)
        if cached and (now_ts - cached[0] < cls._MAX_STALE_SEC):
            cls._LAST_ACCESS[key] = now_ts
            cls._RECIPE_ACCESS[recipe] = now_ts
            cls._RECIPES.setdefault(recipe, key)
            return cached[1]

        # Stale-while-revalidate for relative recipes across grid rollovers:
        # Immediately after a grid rollover, the newly rolled key is not yet warm,
        # but the previous response minted for this recipe is still valid and fresh.
        # Serve it to avoid blocking the visitor on an expensive inline recomputation
        # while recording request activity so the refresher warms the new window.
        if is_relative_recipe(recipe):
            previous_key = cls._RECIPES.get(recipe)
            if previous_key is not None and previous_key != key:
                cached_prev = cls._CACHE.get(previous_key)
                if cached_prev and (now_ts - cached_prev[0] < cls._MAX_STALE_SEC):
                    cls._LAST_ACCESS[previous_key] = now_ts
                    cls._RECIPE_ACCESS[recipe] = now_ts
                    return cached_prev[1]

        return None

    @classmethod
    def get_or_compute(
        cls,
        recipe: Recipe,
        key: Key,
        link_filters: dict[str, Any],
        position_filters: dict[str, Any],
    ) -> bytes:
        """Serve *key* from cache, or compute it exactly once per process.

        A cache miss first checks the per-key in-flight registry: when the
        background refresher (or another request thread) is already
        computing this exact window, the caller waits for and reuses that
        result instead of running a second multi-second computation in
        parallel. Only when no compute is running does the caller's
        thread become the owner — and the refresher, for its part, skips
        keys owned by requests.
        """
        body = cls.serve(recipe, key)
        if body is not None:
            return body

        record = cls._try_begin_compute(key)
        if record is None:
            with cls._INFLIGHT_LOCK:
                existing = cls._INFLIGHT.get(key)
            if (
                existing is not None
                and existing.event.wait(timeout=cls._INFLIGHT_WAIT_SEC)
                and existing.body is not None
            ):
                cls._RECIPE_ACCESS[recipe] = time.time()
                return existing.body
            # The owner failed or outlived the wait bound: serve whatever
            # it minted, then retry ownership once before computing
            # uncoordinated (the pre-dedup behavior).
            body = cls.serve(recipe, key)
            if body is not None:
                return body
            record = cls._try_begin_compute(key)
            if record is None:
                payload = compute_locations_payload(link_filters, position_filters)
                return cls.store(recipe, key, payload)

        compute_start = time.time()
        logger.info("Computing /api/locations response (cache miss): recipe=%s", recipe)
        body = None
        try:
            payload = compute_locations_payload(link_filters, position_filters)
            body = cls.store(recipe, key, payload)
            return body
        finally:
            cls._end_compute(key, record, body)
            logger.info(
                "/api/locations computed in %.3fs", time.time() - compute_start
            )

    @classmethod
    def _try_begin_compute(cls, key: Key) -> _InFlightCompute | None:
        """Register as the sole compute owner of *key*; None if taken."""
        with cls._INFLIGHT_LOCK:
            if key in cls._INFLIGHT:
                return None
            record = _InFlightCompute()
            cls._INFLIGHT[key] = record
            return record

    @classmethod
    def _end_compute(
        cls, key: Key, record: _InFlightCompute, body: bytes | None
    ) -> None:
        """Publish *body* to waiters and release ownership of *key*."""
        record.body = body
        with cls._INFLIGHT_LOCK:
            if cls._INFLIGHT.get(key) is record:
                cls._INFLIGHT.pop(key, None)
        record.event.set()

    @classmethod
    def store(cls, recipe: Recipe, key: Key, payload: dict[str, Any]) -> bytes:
        """Serialize and mint *payload* under *key*; return the body bytes.

        Also records request activity for *recipe* so first-time requests
        register for background refreshing and subsequent idle eviction.
        """
        body = cls._mint(recipe, key, payload)
        cls._RECIPE_ACCESS[recipe] = time.time()
        return body

    @classmethod
    def clear(cls) -> None:
        """Drop all cached state (tests, graceful shutdown)."""
        cls._CACHE.clear()
        cls._LAST_ACCESS.clear()
        cls._RECIPES.clear()
        cls._RECIPE_ACCESS.clear()
        with cls._INFLIGHT_LOCK:
            records = list(cls._INFLIGHT.values())
            cls._INFLIGHT.clear()
        # Wake waiters on abandoned records so they fall back to computing
        # instead of blocking for the full wait bound.
        for record in records:
            record.event.set()

    # ------------------------------------------------------------------
    # Background refresher
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_background_refresh(cls) -> None:
        """Start the refresher thread exactly once per process.

        Idempotent and fork-safe: a stale (dead) thread handle — e.g.
        inherited from a gunicorn preload master — is replaced with a
        fresh thread and a fresh stop event (the old event's internal
        lock may be stranded by a fork and must not be reused).
        """
        thread = cls._refresher_thread
        if thread is not None and thread.is_alive():
            return
        with cls._refresher_lock:
            thread = cls._refresher_thread
            if thread is not None and thread.is_alive():
                return
            cls._refresher_stop = threading.Event()
            cls._refresher_thread = threading.Thread(
                target=cls._refresh_worker,
                name="locations-cache-refresher",
                daemon=True,
            )
            cls._refresher_thread.start()

    @classmethod
    def start_background_refresh(cls) -> None:
        """Warm the default recipe now and keep served recipes fresh.

        The first refresher cycle runs immediately, so calling this at
        process startup means the parameterless map payload is ready
        before the first visitor arrives.
        """
        cls._ensure_background_refresh()

    @classmethod
    def stop_background_refresh(cls) -> None:
        """Stop the refresher thread (tests, graceful shutdown)."""
        cls._refresher_stop.set()
        thread = cls._refresher_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    @classmethod
    def _refresh_worker(cls) -> None:
        """Warm immediately, then recompute active recipes once per interval."""
        while True:
            try:
                cls._refresh_active_keys()
            except Exception:
                logger.exception("Locations cache refresh cycle failed")
            if cls._refresher_stop.wait(cls._REFRESH_INTERVAL_SEC):
                return

    @classmethod
    def _refresh_active_keys(cls) -> None:
        """Recompute recently served recipes; evict idle ones.

        The default (parameterless) recipe is always refreshed so the
        next visitor finds it warm even after a long idle period.
        Recomputation is serial (the same tradeoff the analytics
        refresher makes): each cycle refreshes every active recipe, so
        per-key staleness is bounded by the interval plus the sum of
        compute times — comfortably below ``_MAX_STALE_SEC`` for any
        realistic number of active recipes, and SQLite readers do not
        contend with each other or with request threads.
        """
        now_ts = time.time()

        recipes = {cls._DEFAULT_RECIPE}
        for recipe, last_hit in list(cls._RECIPE_ACCESS.items()):
            if now_ts - last_hit <= cls._KEY_IDLE_SEC:
                recipes.add(recipe)
            else:
                # Idle request shape: stop refreshing and forget it.
                cls._RECIPE_ACCESS.pop(recipe, None)
                cls._RECIPES.pop(recipe, None)

        # Evict orphaned recipes that lack activity tracking (except default recipe)
        for recipe in list(cls._RECIPES):
            if recipe != cls._DEFAULT_RECIPE and recipe not in cls._RECIPE_ACCESS:
                cls._RECIPES.pop(recipe, None)

        # Evict entries nobody served or minted recently (superseded grid
        # buckets of a still-active recipe are cleaned eagerly in
        # _mint; this catches everything else).
        for key in list(cls._CACHE):
            last_hit = cls._LAST_ACCESS.get(key)
            if last_hit is None or now_ts - last_hit > cls._KEY_IDLE_SEC:
                cls._CACHE.pop(key, None)
                cls._LAST_ACCESS.pop(key, None)

        # Group active recipes by resolved cache key to deduplicate refresh work.
        # Distinct recipes (e.g. relative vs explicit windows, or equivalent
        # custom bounds) resolving to the same window compute once per cycle.
        grouped_by_key: dict[
            Key,
            tuple[dict[str, Any], dict[str, Any], list[Recipe]],
        ] = {}

        for recipe in recipes:
            try:
                link_filters, position_filters = resolve_locations_filters(recipe)
            except Exception:
                logger.exception(
                    "Failed to resolve filters for recipe %s during refresh", recipe
                )
                continue
            key = locations_cache_key(link_filters)
            if key not in grouped_by_key:
                grouped_by_key[key] = (link_filters, position_filters, [recipe])
            else:
                grouped_by_key[key][2].append(recipe)

        # Default recipe group first so the most common payload stays freshest
        # even when slower windows are also active.
        sorted_groups = sorted(
            grouped_by_key.items(),
            key=lambda item: cls._DEFAULT_RECIPE not in item[1][2],
        )

        for key, (link_filters, position_filters, group_recipes) in sorted_groups:
            record = cls._try_begin_compute(key)
            if record is None:
                # A request thread is computing this exact window right now
                # and will mint it; a second compute here would only
                # duplicate that work.
                continue
            try:
                payload = compute_locations_payload(link_filters, position_filters)
            except Exception:
                logger.exception(
                    "Locations cache refresh failed for key %s (recipes: %s)",
                    key,
                    group_recipes,
                )
                cls._end_compute(key, record, None)
                continue
            body = cls._mint(group_recipes, key, payload)
            cls._end_compute(key, record, body)

    @classmethod
    def _mint(
        cls,
        recipes: Recipe | Iterable[Recipe],
        key: Key,
        payload: dict[str, Any],
    ) -> bytes:
        """Serialize *payload*, store it under *key*, and return the body.

        Publication is order-checked per recipe. Relative recipes roll to
        a new grid bucket as the clock advances, so a compute that started
        before a rollover resolves an older window than the one already
        published for its recipe (e.g. a request that resolved the new
        window and finished first). Unconditionally repointing would roll
        the recipe back and delete that newer response — the next request
        would then be served the older window — so mints for an older
        window never replace a newer publication. Superseded keys are
        still dropped eagerly (unless another recipe maps to them), and a
        mint nothing maps to is not stored at all.
        """
        body = serialize_locations_payload(payload)

        recipe_list: list[Recipe]
        if isinstance(recipes, list):
            recipe_list = recipes
        elif isinstance(recipes, set):
            recipe_list = list(recipes)
        elif isinstance(recipes, tuple) and len(recipes) > 0 and isinstance(recipes[0], tuple):
            recipe_list = list(recipes)
        else:
            recipe_list = [recipes]  # type: ignore[list-item]

        # Decision pass (reads only): which recipes may repoint to *key*?
        # For a given recipe only relative resolution changes the key, and
        # it only ever rolls forward, so a smaller end_time is strictly a
        # pre-rollover (older) window.
        repoint: list[Recipe] = []
        previous_keys: set[Key] = set()
        for recipe in recipe_list:
            previous_key = cls._RECIPES.get(recipe)
            if (
                previous_key is not None
                and previous_key != key
                and key[1] < previous_key[1]
            ):
                continue  # stale window: keep the newer publication
            if previous_key is not None and previous_key != key:
                previous_keys.add(previous_key)
            repoint.append(recipe)

        if repoint:
            now_ts = time.time()
            # Store the entry before repointing so a concurrent serve()
            # never observes a recipe mapped to a missing cache entry.
            cls._CACHE[key] = (now_ts, body)
            cls._LAST_ACCESS[key] = now_ts
            for recipe in repoint:
                cls._RECIPES[recipe] = key

        # Drop superseded keys of repointed recipes unless another recipe
        # still resolves to them.
        for previous_key in previous_keys:
            if previous_key not in cls._RECIPES.values():
                cls._CACHE.pop(previous_key, None)
                cls._LAST_ACCESS.pop(previous_key, None)

        return body
