"""In-memory hop databases for exercising the SQL graph aggregation path.

Service-level graph tests feed hop dicts straight into
``get_traceroute_graph_aggregates`` via a real SQLite database so the
SQL bucketing (directions, sentinels, recency argmax) is covered end to
end instead of being mocked away.
"""

import sqlite3

from malla.database.traceroutes import PARSER_VERSION


class NonClosingConnection:
    """Pseudo-connection whose ``close`` is a no-op for repeated queries."""

    def __init__(self, connection):
        self.connection = connection

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def close(self):
        pass


_SCHEMA = """
CREATE TABLE packet_history (
    id INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    portnum INTEGER,
    portnum_name TEXT,
    mesh_packet_id INTEGER,
    from_node_id INTEGER,
    to_node_id INTEGER,
    gateway_id TEXT,
    channel_id TEXT,
    hop_start INTEGER,
    hop_limit INTEGER,
    rssi REAL,
    snr REAL,
    payload_length INTEGER,
    raw_payload BLOB,
    processed_successfully INTEGER DEFAULT 1
);
CREATE TABLE traceroute_routes (
    packet_id INTEGER PRIMARY KEY,
    timestamp REAL,
    from_node_id INTEGER,
    to_node_id INTEGER,
    mesh_packet_id INTEGER,
    channel_id TEXT,
    route_nodes_json TEXT,
    snr_towards_json TEXT,
    route_back_json TEXT,
    snr_back_json TEXT,
    parse_status TEXT,
    parser_version INTEGER
);
CREATE TABLE traceroute_hops (
    packet_id INTEGER NOT NULL,
    direction TEXT NOT NULL,
    hop_index INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    from_node_id INTEGER NOT NULL,
    to_node_id INTEGER NOT NULL,
    snr REAL,
    PRIMARY KEY (packet_id, direction, hop_index)
);
"""


def build_graph_hops_database(hops):
    """Materialize ``_hop``-style dicts into an in-memory hop database.

    One ``packet_history``/``traceroute_routes`` row is created per distinct
    ``packet_id`` (timestamp, endpoints and channel taken from its first
    hop); every dict becomes one ``traceroute_hops`` row verbatim. Returns
    a :class:`NonClosingConnection` suitable for patching
    ``get_db_connection`` in the read repository.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)

    seen_packets = set()
    for hop in hops:
        packet_id = hop["packet_id"]
        if packet_id not in seen_packets:
            seen_packets.add(packet_id)
            conn.execute(
                """
                INSERT INTO packet_history (
                    id, timestamp, portnum, portnum_name, mesh_packet_id,
                    from_node_id, to_node_id, gateway_id, channel_id,
                    hop_start, hop_limit, rssi, snr, payload_length, raw_payload
                ) VALUES (?, ?, 70, 'TRACEROUTE_APP', ?, ?, ?, '!00000001', ?,
                          5, 3, -80, -10, 0, X'')
                """,
                (
                    packet_id,
                    hop["timestamp"],
                    1000 + packet_id,
                    hop["from_node_id"],
                    hop["to_node_id"],
                    hop.get("channel_id"),
                ),
            )
            conn.execute(
                """
                INSERT INTO traceroute_routes (
                    packet_id, timestamp, from_node_id, to_node_id,
                    mesh_packet_id, channel_id, route_nodes_json,
                    snr_towards_json, route_back_json, snr_back_json,
                    parse_status, parser_version
                ) VALUES (?, ?, ?, ?, ?, ?, '[]', '[]', '[]', '[]', 'parsed', ?)
                """,
                (
                    packet_id,
                    hop["timestamp"],
                    hop["from_node_id"],
                    hop["to_node_id"],
                    1000 + packet_id,
                    hop.get("channel_id"),
                    PARSER_VERSION,
                ),
            )
        conn.execute(
            """
            INSERT INTO traceroute_hops (
                packet_id, direction, hop_index, timestamp,
                from_node_id, to_node_id, snr
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                packet_id,
                hop.get("direction", "forward"),
                hop["hop_index"],
                hop["timestamp"],
                hop["from_node_id"],
                hop["to_node_id"],
                hop["snr"],
            ),
        )
    conn.commit()
    return NonClosingConnection(conn)
