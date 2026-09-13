"""Schema definitions for derived packet observation projections.

``packet_observations`` projects raw receptions from packet_history into
a lean, indexed relational table without multi-megabyte Protobuf BLOBs.
Each reception is stored with its own packet_id and timestamp, preserving
100% filter fidelity (time, RSSI, channel, hop count) before aggregation,
while sharing a resolved tx_id for fast transmission grouping.
"""

import sqlite3

OBSERVATIONS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS packet_observations (
        packet_id INTEGER PRIMARY KEY REFERENCES packet_history(id) ON DELETE CASCADE,
        tx_id INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        from_node_id INTEGER,
        to_node_id INTEGER,
        portnum INTEGER,
        portnum_name TEXT,
        gateway_id TEXT NOT NULL DEFAULT '',
        channel_id TEXT,
        mesh_packet_id INTEGER,
        rssi INTEGER,
        snr REAL,
        hop_start INTEGER,
        hop_limit INTEGER,
        is_direct INTEGER NOT NULL DEFAULT 0,
        payload_length INTEGER,
        processed_successfully INTEGER,
        relay_node INTEGER
    )
"""

OBSERVATION_INDEX_SPECS: tuple[str, ...] = (
    """
    CREATE INDEX IF NOT EXISTS idx_pobs_time
    ON packet_observations(timestamp DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_pobs_tx_id
    ON packet_observations(tx_id, timestamp DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_pobs_tx_lookup
    ON packet_observations(from_node_id, mesh_packet_id, to_node_id, portnum, timestamp DESC)
    WHERE mesh_packet_id IS NOT NULL AND from_node_id IS NOT NULL
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_pobs_direct_time
    ON packet_observations(from_node_id, gateway_id, timestamp DESC)
    WHERE is_direct = 1
        AND from_node_id IS NOT NULL
        AND gateway_id != ''
    """,
    """
    -- Covering index for the map's direct-RF-link aggregation: partial (only
    -- direct receptions), time-leading (window range scans never touch the
    -- wider table), and covering for every column get_packet_links reads
    -- (packet_id rides along implicitly as the rowid).  Self-receptions
    -- (gateway hearing its own node id) never form links, so they are kept
    -- out of the index entirely; get_packet_links repeats this predicate
    -- verbatim to stay eligible for the partial index.
    CREATE INDEX IF NOT EXISTS idx_pobs_link_time
    ON packet_observations(
        timestamp DESC, from_node_id, gateway_id, tx_id, channel_id, snr, rssi
    )
    WHERE is_direct = 1
        AND from_node_id IS NOT NULL
        AND gateway_id != ''
        AND lower(gateway_id) != printf('!%08x', from_node_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_pobs_gateway_time
    ON packet_observations(gateway_id, timestamp DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_pobs_from_time
    ON packet_observations(from_node_id, timestamp DESC)
    WHERE from_node_id IS NOT NULL
    """,
    """
    -- Covering index for the node-list 24h aggregation (get_nodes): one
    -- ordered scan per from_node_id yields the packet count, MAX(timestamp)
    -- and the direct-only RSSI/SNR averages straight from the index, with
    -- no row lookups and no GROUP BY sort.  get_nodes repeats the partial
    -- index predicate verbatim to stay eligible for it.
    CREATE INDEX IF NOT EXISTS idx_pobs_node_stats
    ON packet_observations(from_node_id, timestamp DESC, is_direct, rssi, snr)
    WHERE from_node_id IS NOT NULL
    """,
)


def ensure_observation_schema(cursor: sqlite3.Cursor) -> None:
    """Create the packet_observations table, triggers, and its indexes."""
    cursor.execute("PRAGMA table_info(packet_observations)")
    rows = cursor.fetchall()
    if rows:
        cols = {row[1] for row in rows}
        pk_cols = [row[1] for row in rows if row[5] > 0]
        # Drop legacy collapsed schema if present (had (tx_id, gateway_id) PK or lacked packet_id)
        if "packet_id" not in cols or pk_cols != ["packet_id"]:
            cursor.execute("DROP TABLE packet_observations")

    cursor.execute(OBSERVATIONS_TABLE_SQL)

    cursor.execute("PRAGMA table_info(packet_observations)")
    current_cols = {row[1] for row in cursor.fetchall()}
    if "relay_node" not in current_cols:
        cursor.execute("ALTER TABLE packet_observations ADD COLUMN relay_node INTEGER")

    # Recreate direct time index if it used legacy direct_last_seen predicate
    row = cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_pobs_direct_time'"
    ).fetchone()
    if row and row[0] and "direct_last_seen" in row[0]:
        cursor.execute("DROP INDEX IF EXISTS idx_pobs_direct_time")

    # A legacy idx_pobs_link_time must not survive: neither the collapsed
    # pre-reception summary layout (no tx_id) nor the earlier direct-only
    # predicate (no self-reception exclusion) matches the modern spec, which
    # is (re)created below.
    row = cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_pobs_link_time'"
    ).fetchone()
    if row and row[0] and (
        "tx_id" not in row[0] or "lower(gateway_id)" not in row[0]
    ):
        cursor.execute("DROP INDEX IF EXISTS idx_pobs_link_time")

    for spec in OBSERVATION_INDEX_SPECS:
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
            CREATE TRIGGER IF NOT EXISTS observation_packet_insert
            AFTER INSERT ON packet_history
            BEGIN
                DELETE FROM packet_observations WHERE packet_id = NEW.id;
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS observation_packet_delete
            AFTER DELETE ON packet_history
            BEGIN
                DELETE FROM packet_observations WHERE packet_id = OLD.id;
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS observation_packet_update
            AFTER UPDATE ON packet_history
            BEGIN
                DELETE FROM packet_observations WHERE packet_id = OLD.id;
                DELETE FROM packet_observations WHERE packet_id = NEW.id;
            END
        """)

