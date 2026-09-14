"""Tests for the materialized 24h node-stats reader in NodeRepository.get_nodes.

get_nodes computes per-node 24h packet counts, last-seen times and direct
RSSI/SNR averages, plus per-gateway reception counts. When the
``packet_observations`` projection is populated the aggregates read the lean
materialized rows instead of scanning raw packet_history; every scenario
therefore runs against both reader paths and must produce identical results.
"""

import sqlite3
import time
from contextlib import closing
from unittest.mock import patch

import pytest

from malla.database.materialization_schema import ensure_materialization_schema
from malla.database.materializations import (
    clear_materialization_cache,
    materialization_tx_scope,
    materialize_packet,
)
from malla.database.repositories import NodeRepository

pytestmark = pytest.mark.unit

GW_HEX = "!aabbccdd"
GW_NODE_ID = int("aabbccdd", 16)
TX_NODE_ID = int("11111111", 16)
IDLE_NODE_ID = int("22222222", 16)


@pytest.fixture(autouse=True)
def _clear_materialization_cache():
    clear_materialization_cache()
    yield
    clear_materialization_cache()


@pytest.fixture(params=["legacy", "materialized"])
def database(tmp_path, request):
    path = tmp_path / f"node-stats-{request.param}.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE packet_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                mesh_packet_id INTEGER,
                from_node_id INTEGER,
                to_node_id INTEGER,
                portnum INTEGER,
                portnum_name TEXT,
                gateway_id TEXT,
                channel_id TEXT,
                rssi INTEGER,
                snr REAL,
                hop_limit INTEGER,
                hop_start INTEGER,
                payload_length INTEGER,
                raw_payload BLOB,
                processed_successfully INTEGER DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE node_info (
                node_id INTEGER PRIMARY KEY,
                long_name TEXT,
                short_name TEXT,
                hw_model TEXT,
                role TEXT,
                primary_channel TEXT,
                last_updated REAL
            )
        """)
        if request.param == "materialized":
            ensure_materialization_schema(conn.cursor())
        conn.commit()
    return path


def _connection(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _insert_packet(conn, **fields):
    values = {
        "timestamp": time.time(),
        "mesh_packet_id": None,
        "from_node_id": TX_NODE_ID,
        "to_node_id": None,
        "portnum": 1,
        "portnum_name": "TEXT_MESSAGE_APP",
        "gateway_id": GW_HEX,
        "channel_id": None,
        "rssi": None,
        "snr": None,
        "hop_limit": None,
        "hop_start": None,
        "payload_length": 10,
        "raw_payload": b"",
        "processed_successfully": 1,
    }
    values.update(fields)
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    cur = conn.execute(
        f"INSERT INTO packet_history ({columns}) VALUES ({placeholders})",
        tuple(values.values()),
    )
    has_projection = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'packet_observations'"
    ).fetchone()
    if has_projection:
        # Same writer the live capture path uses: raw + derived rows commit
        # atomically.
        packet = dict(
            conn.execute(
                "SELECT * FROM packet_history WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
        )
        with materialization_tx_scope(), conn:
            materialize_packet(conn.cursor(), packet)
    else:
        conn.commit()


def _seed_nodes(conn):
    now = time.time()
    conn.execute(
        "INSERT INTO node_info VALUES (?, ?, ?, ?, ?, ?, ?)",
        (TX_NODE_ID, "Tx Node", "TXN", "TBEAM", "CLIENT", None, now - 50000),
    )
    conn.execute(
        "INSERT INTO node_info VALUES (?, ?, ?, ?, ?, ?, ?)",
        (GW_NODE_ID, "Gateway Node", "GWN", "RAK4631", "ROUTER", None, now - 40000),
    )
    conn.execute(
        "INSERT INTO node_info VALUES (?, ?, ?, ?, ?, ?, ?)",
        (IDLE_NODE_ID, "Idle Node", "IDL", "TBEAM", "CLIENT", None, now - 30000),
    )
    conn.commit()


class TestGetNodes24hStats:
    def test_stats_match_across_reader_paths(self, database):
        now = time.time()
        with closing(_connection(database)) as conn:
            _seed_nodes(conn)
            # Two direct receptions with plausible signal metrics.
            _insert_packet(conn, timestamp=now - 100, rssi=-80, snr=9.5,
                           hop_start=3, hop_limit=3)
            _insert_packet(conn, timestamp=now - 200, rssi=-70, snr=7.5,
                           hop_start=0, hop_limit=0)
            # Relayed reception: counted, but its signal belongs to the relay.
            _insert_packet(conn, timestamp=now - 300, rssi=-60, snr=5.0,
                           hop_start=3, hop_limit=1)
            # Direct reception with implausible signal values: counted,
            # excluded from averages.
            _insert_packet(conn, timestamp=now - 400, rssi=20, snr=99.0,
                           hop_start=3, hop_limit=3)
            # Outside the 24h window: excluded entirely.
            _insert_packet(conn, timestamp=now - 90000, rssi=-50, snr=6.0,
                           hop_start=3, hop_limit=3)

        with patch(
            "malla.database.repositories.get_db_connection",
            side_effect=lambda: _connection(database),
        ):
            # Default order (last_packet_time) is the path the nodes page
            # exercises on every load.
            result = NodeRepository.get_nodes(limit=10)

        nodes = {n["node_id"]: n for n in result["nodes"]}
        tx = nodes[TX_NODE_ID]
        assert tx["packet_count_24h"] == 4
        assert tx["avg_rssi"] == pytest.approx(-75.0)
        assert tx["avg_snr"] == pytest.approx(8.5)
        assert tx["last_packet_time"] == pytest.approx(now - 100, abs=5)

        gw = nodes[GW_NODE_ID]
        assert gw["gateway_packet_count_24h"] == 4

        idle = nodes[IDLE_NODE_ID]
        assert idle["packet_count_24h"] == 0
        assert idle["gateway_packet_count_24h"] == 0
        assert idle["avg_rssi"] is None
        assert idle["avg_snr"] is None

    def test_active_only_filter_uses_stats_on_both_paths(self, database):
        now = time.time()
        with closing(_connection(database)) as conn:
            _seed_nodes(conn)
            _insert_packet(conn, timestamp=now - 100, rssi=-80, snr=9.5,
                           hop_start=3, hop_limit=3)

        with patch(
            "malla.database.repositories.get_db_connection",
            side_effect=lambda: _connection(database),
        ):
            result = NodeRepository.get_nodes(
                limit=10, order_by="node_id", filters={"active_only": True}
            )

        assert [n["node_id"] for n in result["nodes"]] == [TX_NODE_ID]
