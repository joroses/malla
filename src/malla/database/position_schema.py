"""Schema definitions for derived node position fixes.

``node_positions`` stores position fixes decoded once at write time.
Only fixes passing :func:`malla.utils.geo_utils.is_valid_position` are
kept, which makes "latest valid fix per node" an indexed lookup instead of
a rank-and-decode walk over raw payloads.
"""

import sqlite3

POSITIONS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS node_positions (
        node_id INTEGER NOT NULL,
        tx_id INTEGER,
        gateway_id TEXT NOT NULL DEFAULT '',
        timestamp REAL NOT NULL,
        packet_id INTEGER PRIMARY KEY REFERENCES packet_history(id) ON DELETE CASCADE,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        altitude INTEGER,
        precision_bits INTEGER,
        sats_in_view INTEGER
    )
"""

POSITION_INDEX_SPECS: tuple[str, ...] = (
    """
    CREATE INDEX IF NOT EXISTS idx_np_node_time
    ON node_positions(node_id, timestamp DESC, packet_id DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_np_gateway_time
    ON node_positions(gateway_id, timestamp DESC, packet_id DESC)
    """,
)


def ensure_position_schema(cursor: sqlite3.Cursor) -> None:
    """Create the node_positions table, triggers, and its indexes."""
    cursor.execute("PRAGMA table_info(node_positions)")
    cols = cursor.fetchall()
    if cols:
        pk_cols = [c[1] for c in cols if c[5] > 0]
        if pk_cols != ["packet_id"]:
            cursor.execute("DROP TABLE node_positions")
    cursor.execute(POSITIONS_TABLE_SQL)

    # Recreate legacy indexes missing the packet_id tie-breaker
    cursor.execute("PRAGMA index_info(idx_np_node_time)")
    node_idx_cols = [c[2] for c in cursor.fetchall()]
    if node_idx_cols and node_idx_cols != ["node_id", "timestamp", "packet_id"]:
        cursor.execute("DROP INDEX IF EXISTS idx_np_node_time")

    cursor.execute("PRAGMA index_info(idx_np_gateway_time)")
    gw_idx_cols = [c[2] for c in cursor.fetchall()]
    if gw_idx_cols and gw_idx_cols != ["gateway_id", "timestamp", "packet_id"]:
        cursor.execute("DROP INDEX IF EXISTS idx_np_gateway_time")

    for spec in POSITION_INDEX_SPECS:
        cursor.execute(spec)

    has_packet_history = bool(
        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'packet_history'"
        ).fetchone()
    )
    if has_packet_history:
        # Explicit child deletion also handles raw import connections with foreign
        # keys disabled, and INSERT OR REPLACE reusing a packet row ID.
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS position_packet_insert
            AFTER INSERT ON packet_history
            BEGIN
                DELETE FROM node_positions WHERE packet_id = NEW.id;
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS position_packet_delete
            AFTER DELETE ON packet_history
            BEGIN
                DELETE FROM node_positions WHERE packet_id = OLD.id;
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS position_packet_update
            AFTER UPDATE ON packet_history
            BEGIN
                DELETE FROM node_positions WHERE packet_id = OLD.id;
                DELETE FROM node_positions WHERE packet_id = NEW.id;
            END
        """)

