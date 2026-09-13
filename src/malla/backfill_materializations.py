"""Explicit, resumable backfill of the materialized transmission tables.

Walks ``packet_history`` in id order and feeds every row through the same
:func:`malla.database.materializations.materialize_packet` writer the live
capture path uses, so backfilled and live data are indistinguishable.

Derived rows are keyed by packet identity with ON CONFLICT DO NOTHING, so
the walk is naturally idempotent. Progress is tracked in ``materialization_state``
and each batch commits its rows plus the watermark atomically. Concurrent capture
inserts commit with higher rowids and are materialized by the capture hook itself.

Re-running after completion is a no-op. ``--reset`` drops the derived tables
and the watermark for a clean rebuild.

This command is for the live MQTT database; it must not be pointed at the
canonical import database.
"""

import argparse
import json
import sqlite3
import sys
import time
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from .database.materialization_schema import ensure_materialization_schema
from .database.materializations import (
    BACKFILL_STATE_NAME,
    clear_materialization_cache,
    materialization_tx_scope,
    materialize_packet,
)
from .database.traceroute_schema import TRACEROUTE_PREDICATE
from .database.traceroutes import (
    PARSER_VERSION,
    inspect_traceroutes,
)


def _get_backfill_columns(cursor: sqlite3.Cursor) -> str:
    cursor.execute("PRAGMA table_info(packet_history)")
    cols = {r[1] for r in cursor.fetchall()}
    relay_col = "relay_node" if "relay_node" in cols else "NULL AS relay_node"
    return (
        "id, timestamp, mesh_packet_id, from_node_id, to_node_id, portnum, "
        "portnum_name, gateway_id, channel_id, rssi, snr, hop_limit, hop_start, "
        f"payload_length, raw_payload, processed_successfully, {relay_col}"
    )


BACKFILL_COLUMNS = (
    "id, timestamp, mesh_packet_id, from_node_id, to_node_id, portnum, "
    "portnum_name, gateway_id, channel_id, rssi, snr, hop_limit, hop_start, "
    "payload_length, raw_payload, processed_successfully"
)


BACKFILL_STATE_UPSERT_SQL = """
    INSERT INTO materialization_state (name, last_packet_id, updated_at)
    VALUES (?, ?, ?)
    ON CONFLICT (name) DO UPDATE SET
        last_packet_id = excluded.last_packet_id,
        updated_at = excluded.updated_at
"""


def _read_watermark(cursor: sqlite3.Cursor) -> int:
    row = cursor.execute(
        "SELECT last_packet_id FROM materialization_state WHERE name = ?",
        (BACKFILL_STATE_NAME,),
    ).fetchone()
    return row[0] if row else 0


def reset_materializations(conn: sqlite3.Connection) -> None:
    """Drop derived rows and the watermark so a fresh walk cannot double-count."""

    clear_materialization_cache()
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM packet_observations")
        cursor.execute("DELETE FROM node_positions")
        cursor.execute("DELETE FROM traceroute_hops")
        cursor.execute("DELETE FROM traceroute_routes")
        cursor.execute(
            "DELETE FROM materialization_state WHERE name = ?", (BACKFILL_STATE_NAME,)
        )


def backfill_materializations(
    conn: sqlite3.Connection,
    batch_size: int = 2000,
    progress: Callable[[int], None] | None = None,
) -> dict:
    """Materialize unprocessed and pending packet_history rows, batch by batch.

    An id boundary (``upper_id``) keeps continuous capture from extending the
    run indefinitely. Returns progress details; ``complete`` is True when all
    pending repairs and history up to ``upper_id`` are fully materialized.
    """

    clear_materialization_cache()

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()
        ensure_materialization_schema(cursor)
        upper_id = cursor.execute(
            "SELECT COALESCE(MAX(id), 0) FROM packet_history"
        ).fetchone()[0]
        last_id = _read_watermark(cursor)
        if last_id > upper_id:
            cursor.execute(
                BACKFILL_STATE_UPSERT_SQL, (BACKFILL_STATE_NAME, upper_id, time.time())
            )
            last_id = upper_id

        # Clean up any orphaned derived records whose raw packet was deleted
        cursor.execute(
            """
            DELETE FROM traceroute_hops WHERE NOT EXISTS (
                SELECT 1 FROM traceroute_routes WHERE packet_id = traceroute_hops.packet_id
            )
            """
        )
        cursor.execute(
            f"""
            DELETE FROM traceroute_routes WHERE NOT EXISTS (
                SELECT 1 FROM packet_history
                WHERE id = traceroute_routes.packet_id AND ({TRACEROUTE_PREDICATE})
            )
            """
        )
        cursor.execute(
            """
            DELETE FROM packet_observations WHERE NOT EXISTS (
                SELECT 1 FROM packet_history WHERE id = packet_observations.packet_id
            )
            """
        )
        cursor.execute(
            """
            DELETE FROM node_positions WHERE NOT EXISTS (
                SELECT 1 FROM packet_history WHERE id = node_positions.packet_id
            )
            """
        )

    initial_watermark = last_id
    processed = 0

    # Phase 1: Repair any mutated / pending / missing packets with id <= watermark
    while True:
        with materialization_tx_scope(), conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.cursor()
            cols = _get_backfill_columns(cursor)
            rows = cursor.execute(
                f"""
                SELECT {cols}
                FROM packet_history
                WHERE id <= ? AND (
                    NOT EXISTS (
                        SELECT 1 FROM packet_observations WHERE packet_id = packet_history.id
                    )
                    OR (
                        ({TRACEROUTE_PREDICATE})
                        AND NOT EXISTS (
                            SELECT 1 FROM traceroute_routes
                            WHERE packet_id = packet_history.id
                              AND parser_version = ?
                              AND parse_status != 'pending'
                        )
                    )
                )
                ORDER BY id
                LIMIT ?
                """,
                (last_id, PARSER_VERSION, batch_size),
            ).fetchall()
            if not rows:
                break
            for row in rows:
                materialize_packet(cursor, dict(row))
        processed += len(rows)
        if progress is not None:
            progress(processed)

    # Phase 2: Sequential watermark advance for rows with id > watermark
    while True:
        with materialization_tx_scope(), conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.cursor()
            cols = _get_backfill_columns(cursor)
            rows = cursor.execute(
                f"""
                SELECT {cols}
                FROM packet_history
                WHERE id > ? AND id <= ?
                ORDER BY id
                LIMIT ?
                """,
                (last_id, upper_id, batch_size),
            ).fetchall()
            if not rows:
                break
            for row in rows:
                materialize_packet(cursor, dict(row))
            last_id = rows[-1]["id"]
            cursor.execute(
                BACKFILL_STATE_UPSERT_SQL, (BACKFILL_STATE_NAME, last_id, time.time())
            )
        processed += len(rows)
        if progress is not None:
            progress(processed)

    with conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()
        result = inspect_materializations(cursor)
    return {
        **result,
        "upper_bound_id": upper_id,
        "processed_this_run": processed,
        "previous_watermark": initial_watermark,
    }


def inspect_materializations(cursor: sqlite3.Cursor) -> dict:
    """Audit derived tables and pending history explicitly."""

    upper_id = cursor.execute(
        "SELECT COALESCE(MAX(id), 0) FROM packet_history"
    ).fetchone()[0]
    watermark = _read_watermark(cursor)
    traceroute_audit = inspect_traceroutes(cursor)

    pending_packets = cursor.execute(
        f"""
        SELECT COUNT(*) FROM packet_history
        WHERE id <= ? AND (
            id > ?
            OR NOT EXISTS (
                SELECT 1 FROM packet_observations WHERE packet_id = packet_history.id
            )
            OR (
                ({TRACEROUTE_PREDICATE})
                AND NOT EXISTS (
                    SELECT 1 FROM traceroute_routes
                    WHERE packet_id = packet_history.id
                      AND parser_version = ?
                      AND parse_status != 'pending'
                )
            )
        )
        """,
        (upper_id, watermark, PARSER_VERSION),
    ).fetchone()[0]

    complete = (
        pending_packets == 0
        and traceroute_audit.get("complete", False)
        and watermark >= upper_id
    )

    return {
        "packet_history_max_id": upper_id,
        "watermark": watermark,
        "pending_packets": pending_packets,
        "packet_observations_rows": cursor.execute(
            "SELECT COUNT(*) FROM packet_observations"
        ).fetchone()[0],
        "node_positions_rows": cursor.execute(
            "SELECT COUNT(*) FROM node_positions"
        ).fetchone()[0],
        "traceroute_routes_rows": cursor.execute(
            "SELECT COUNT(*) FROM traceroute_routes"
        ).fetchone()[0],
        "traceroute_hops_rows": cursor.execute(
            "SELECT COUNT(*) FROM traceroute_hops"
        ).fetchone()[0],
        "traceroutes": traceroute_audit,
        "complete": complete,
    }


def _inspect_readonly(conn: sqlite3.Connection) -> dict:
    """Check-mode variant that never creates missing schema."""

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if not {
        "packet_observations",
        "node_positions",
        "traceroute_routes",
        "traceroute_hops",
        "materialization_state",
    } <= (tables):
        return {
            "packet_history_max_id": conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM packet_history"
            ).fetchone()[0],
            "schema_absent": True,
            "complete": False,
        }
    with conn:
        conn.execute("BEGIN")
        return inspect_materializations(conn.cursor())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", required=True, type=Path, help="Existing database path"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2000, help="Packets per transaction"
    )
    parser.add_argument(
        "--check",
        "--dry-run",
        action="store_true",
        help="Report counts without modifying the database",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete derived rows and the watermark before running, for a clean"
            " rebuild (required after changing decode/validity rules); slower"
            " than resuming"
        ),
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    path = args.database.expanduser().resolve()
    print(f"Database: {path}", flush=True)
    try:
        # mode=rw refuses to create a database for a misspelled path. Neither
        # configuration defaults nor capture startup participate in this command.
        mode = "ro" if args.check else "rw"
        with closing(
            sqlite3.connect(f"{path.as_uri()}?mode={mode}", uri=True, timeout=30.0)
        ) as conn:
            conn.row_factory = sqlite3.Row
            if args.check:
                result = _inspect_readonly(conn)
            else:
                if args.reset:
                    with conn:
                        conn.execute("BEGIN IMMEDIATE")
                        ensure_materialization_schema(conn.cursor())
                    reset_materializations(conn)
                result = backfill_materializations(
                    conn,
                    batch_size=args.batch_size,
                    progress=lambda count: print(
                        f"Materialized {count} packets", flush=True
                    ),
                )
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result.get("complete", True) else 1
    except KeyboardInterrupt:
        print(
            "Interrupted. Committed batches are safe; run the same command to resume.",
            file=sys.stderr,
        )
        return 130
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"Materialization backfill failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
