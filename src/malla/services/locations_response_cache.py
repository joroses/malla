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
    """Canonicalize raw request parameters into a recipe."""
    return (start_arg, end_arg, hours_arg or None, gateway_id, search or None)


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
        return None

    @classmethod
    def store(cls, recipe: Recipe, key: Key, payload: dict[str, Any]) -> bytes:
        """Serialize and mint *payload* under *key*; return the body bytes."""
        return cls._mint(recipe, key, payload)

    @classmethod
    def clear(cls) -> None:
        """Drop all cached state (tests, graceful shutdown)."""
        cls._CACHE.clear()
        cls._LAST_ACCESS.clear()
        cls._RECIPES.clear()
        cls._RECIPE_ACCESS.clear()

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

        # Evict entries nobody served or minted recently (superseded grid
        # buckets of a still-active recipe are cleaned eagerly in
        # _mint; this catches everything else).
        for key in list(cls._CACHE):
            last_hit = cls._LAST_ACCESS.get(key)
            if last_hit is None or now_ts - last_hit > cls._KEY_IDLE_SEC:
                cls._CACHE.pop(key, None)
                cls._LAST_ACCESS.pop(key, None)

        # Default recipe first so the most common payload stays freshest
        # even when slower windows are also active.
        for recipe in sorted(recipes, key=lambda r: r != cls._DEFAULT_RECIPE):
            try:
                link_filters, position_filters = resolve_locations_filters(recipe)
                payload = compute_locations_payload(link_filters, position_filters)
            except Exception:
                logger.exception(
                    "Locations cache refresh failed for recipe %s", recipe
                )
                continue
            cls._mint(recipe, locations_cache_key(link_filters), payload)

    @classmethod
    def _mint(cls, recipe: Recipe, key: Key, payload: dict[str, Any]) -> bytes:
        """Serialize *payload*, store it under *key*, and return the body.

        Also cleans up the recipe's superseded key: a relative recipe that
        rolled to a new grid bucket leaves its previous key unresolvable,
        so it is dropped immediately instead of holding megabytes until
        idle eviction — unless another recipe still maps to it.
        """
        body = serialize_locations_payload(payload)
        now_ts = time.time()

        previous_key = cls._RECIPES.get(recipe)
        cls._CACHE[key] = (now_ts, body)
        cls._LAST_ACCESS[key] = now_ts
        cls._RECIPES[recipe] = key

        if (
            previous_key is not None
            and previous_key != key
            and previous_key not in cls._RECIPES.values()
        ):
            cls._CACHE.pop(previous_key, None)
            cls._LAST_ACCESS.pop(previous_key, None)

        return body
