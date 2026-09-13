"""Derived materialization schema orchestrator for read-path aggregation.

These tables are owned by the live MQTT capture path and the explicit
backfill command; they are intentionally NOT part of the shared web startup
schema so web instances never create them in databases they only read
(notably the canonical import database, whose single writer lives elsewhere).

This module orchestrates the domain schemas:
- ``observation_schema``: collapses packet_history receptions into
  ``packet_observations`` per (tx_id, gateway_id) with counters and signal metrics.
- ``position_schema``: stores validated fixes in ``node_positions`` decoded once.
- ``traceroute_schema``: stores routes and hops in ``traceroute_routes`` and ``traceroute_hops``.
- ``materialization_state``: tracks backfill progress so the single-pass history walk is
  resumable and never re-applies increments (which would double-count).
"""

import sqlite3

from .observation_schema import (
    OBSERVATION_INDEX_SPECS,
    OBSERVATIONS_TABLE_SQL,
    ensure_observation_schema,
)
from .position_schema import (
    POSITION_INDEX_SPECS,
    POSITIONS_TABLE_SQL,
    ensure_position_schema,
)
from .traceroute_schema import ensure_traceroute_schema

MATERIALIZATION_STATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS materialization_state (
        name TEXT PRIMARY KEY,
        last_packet_id INTEGER NOT NULL DEFAULT 0,
        updated_at REAL NOT NULL
    ) WITHOUT ROWID
"""

__all__ = [
    "MATERIALIZATION_STATE_TABLE_SQL",
    "OBSERVATION_INDEX_SPECS",
    "OBSERVATIONS_TABLE_SQL",
    "POSITION_INDEX_SPECS",
    "POSITIONS_TABLE_SQL",
    "ensure_materialization_schema",
    "ensure_observation_schema",
    "ensure_position_schema",
    "ensure_traceroute_schema",
]


def ensure_materialization_schema(cursor: sqlite3.Cursor) -> None:
    """Create all derived domain tables, indexes, triggers, and the watermark state."""

    ensure_observation_schema(cursor)
    ensure_position_schema(cursor)
    ensure_traceroute_schema(cursor)
    cursor.execute(MATERIALIZATION_STATE_TABLE_SQL)
