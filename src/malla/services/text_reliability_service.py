"""Service layer for broadcast text-message gateway reliability.

Thin caching wrapper around
:func:`malla.database.text_reliability_repository.get_text_message_reliability`.
The map requests one payload per node selection, so a short TTL cache
(like the other map services) keeps repeated selections and refetches
after time-window changes cheap.
"""

import logging
import time
from typing import Any

from ..database.connection import get_db_connection
from ..database.text_reliability_repository import get_text_message_reliability

logger = logging.getLogger(__name__)

_CACHE: dict[tuple[int, float | None, float | None], tuple[float, dict[str, Any]]] = {}
_CACHE_TTL_SECONDS = 60
_CACHE_MAX_ENTRIES = 256


def _prune_cache(now: float) -> None:
    expired_keys = [
        key
        for key, (cached_at, _) in _CACHE.items()
        if now - cached_at > _CACHE_TTL_SECONDS
    ]
    for key in expired_keys:
        _CACHE.pop(key, None)

    overflow = len(_CACHE) - _CACHE_MAX_ENTRIES
    if overflow > 0:
        oldest_keys = sorted(_CACHE.items(), key=lambda item: item[1][0])[:overflow]
        for key, _ in oldest_keys:
            _CACHE.pop(key, None)


class TextReliabilityService:
    """Gateway broadcast text-message reception reliability with caching."""

    @staticmethod
    def clear_cache() -> None:
        """Clear cached reliability payloads (tests, resets)."""
        _CACHE.clear()

    @staticmethod
    def get_text_message_reliability(
        node_id: int,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reliability payload for ``node_id`` over the filtered window.

        Args:
            node_id: Numeric node ID of the transmitting node.
            filters: Filters carrying ``start_time``/``end_time`` (epoch
                seconds); other keys are ignored. Callers pass the exact
                link window resolved for /api/locations so the percentages
                obey the map's selected time range.

        Returns:
            Payload from the repository (``node_id``, ``total_sent``,
            ``gateways`` with per-gateway ``received`` and ``percent``).
        """
        filters = filters or {}
        start_time = filters.get("start_time")
        end_time = filters.get("end_time")
        key = (node_id, start_time, end_time)

        now = time.time()
        _prune_cache(now)
        cached = _CACHE.get(key)
        if cached and now - cached[0] < _CACHE_TTL_SECONDS:
            logger.debug(
                "Returning cached text-message reliability for node %s", node_id
            )
            return _copy_payload(cached[1])

        conn = get_db_connection()
        try:
            result = get_text_message_reliability(
                conn.cursor(), node_id, start_time, end_time
            )
        finally:
            conn.close()

        _CACHE[key] = (time.time(), result)
        return _copy_payload(result)


def _copy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Deep enough copy that callers can never mutate a cached payload."""
    return {
        **payload,
        "gateways": [dict(gateway) for gateway in payload.get("gateways", [])],
    }
