"""Tests for grouped packet reads served from packet_observations.

get_packets(group_packets=True) aggregates duplicate receptions of one
transmission. When the ``packet_observations`` projection exists, groups,
totals and pagination come from SQL aggregation over
tx_id (COUNT(DISTINCT tx_id) plus GROUP BY), so every transmission in the
window is visible no matter how many duplicate receptions it has. Every
scenario runs against both reader paths and must produce equivalent
results.
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
from malla.database.repositories import PacketRepository

pytestmark = pytest.mark.unit

GW1 = "!11110001"
GW2 = "!11110002"
GW3 = "!11110003"
TX1 = int("11111111", 16)
TX2 = int("22222222", 16)
TX3 = int("33333333", 16)
TX4 = int("44444444", 16)

T0 = time.time() - 3600


@pytest.fixture(autouse=True)
def _reset_caches():
    clear_materialization_cache()
    yield
    clear_materialization_cache()


@pytest.fixture(params=["legacy", "materialized"])
def database(tmp_path, request):
    path = tmp_path / f"packet-grouping-{request.param}.db"
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
                relay_node INTEGER,
                processed_successfully INTEGER DEFAULT 1,
                via_mqtt INTEGER,
                want_ack INTEGER,
                priority INTEGER,
                delayed INTEGER,
                channel_index INTEGER,
                rx_time INTEGER,
                pki_encrypted INTEGER,
                next_hop INTEGER,
                tx_after INTEGER
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
        "timestamp": T0,
        "mesh_packet_id": None,
        "from_node_id": TX1,
        "to_node_id": None,
        "portnum": 1,
        "portnum_name": "TEXT_MESSAGE_APP",
        "gateway_id": GW1,
        "channel_id": None,
        "rssi": None,
        "snr": None,
        "hop_limit": None,
        "hop_start": None,
        "payload_length": 20,
        "raw_payload": None,
        "relay_node": None,
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


def _seed_transmissions(conn):
    # Transmission A: 4 receptions across 3 gateways. The last reception
    # carries the rssi=0 "not provided" sentinel and a garbage SNR, which
    # must stay out of the signal ranges but still count as a reception.
    _insert_packet(
        conn,
        timestamp=T0,
        mesh_packet_id=5000,
        gateway_id=GW1,
        channel_id="testChannel",
        rssi=-80,
        snr=8.5,
        hop_start=3,
        hop_limit=3,
        relay_node=0x17,
        raw_payload=b"hello world",
    )
    _insert_packet(
        conn,
        timestamp=T0 + 5,
        mesh_packet_id=5000,
        gateway_id=GW2,
        channel_id="testChannel",
        rssi=-70,
        snr=10.0,
        hop_start=3,
        hop_limit=2,
        relay_node=0x17,
        raw_payload=b"hello world",
    )
    _insert_packet(
        conn,
        timestamp=T0 + 10,
        mesh_packet_id=5000,
        gateway_id=GW1,
        channel_id="testChannel",
        rssi=-75,
        snr=9.0,
        hop_start=3,
        hop_limit=1,
        raw_payload=b"hello world",
    )
    _insert_packet(
        conn,
        timestamp=T0 + 30,
        mesh_packet_id=5000,
        gateway_id=GW3,
        channel_id="testChannel",
        rssi=0,
        snr=99.0,
        hop_start=3,
        hop_limit=3,
        raw_payload=b"hello world",
    )
    # Transmission B: 2 receptions on 2 gateways.
    _insert_packet(
        conn,
        timestamp=T0 + 100,
        mesh_packet_id=5001,
        from_node_id=TX2,
        portnum=3,
        portnum_name="POSITION_APP",
        payload_length=14,
        gateway_id=GW1,
        rssi=-60,
        snr=7.0,
        hop_start=3,
        hop_limit=3,
    )
    _insert_packet(
        conn,
        timestamp=T0 + 105,
        mesh_packet_id=5001,
        from_node_id=TX2,
        portnum=3,
        portnum_name="POSITION_APP",
        payload_length=14,
        gateway_id=GW2,
        rssi=-55,
        snr=7.5,
        hop_start=3,
        hop_limit=2,
    )
    # Transmission C: a single reception without hop or channel data.
    _insert_packet(
        conn,
        timestamp=T0 + 200,
        mesh_packet_id=5002,
        from_node_id=TX3,
        gateway_id=GW2,
        rssi=-60,
        snr=7.0,
    )
    # Unidentifiable transmission (no mesh packet id): never grouped.
    _insert_packet(conn, timestamp=T0 + 300, from_node_id=TX4)
    # Identifiable but outside the default 7-day grouped window.
    _insert_packet(
        conn,
        timestamp=time.time() - (8 * 24 * 3600),
        mesh_packet_id=5003,
        gateway_id=GW1,
        rssi=-70,
        snr=8.0,
    )
    _insert_packet(
        conn,
        timestamp=time.time() - (8 * 24 * 3600) + 5,
        mesh_packet_id=5003,
        gateway_id=GW2,
        rssi=-70,
        snr=8.0,
    )


def _get_packets(path, **kwargs):
    kwargs.setdefault("group_packets", True)
    with patch(
        "malla.database.repositories.get_db_connection",
        side_effect=lambda: _connection(path),
    ):
        return PacketRepository.get_packets(**kwargs)


def _has_projection(path) -> bool:
    with closing(_connection(path)) as conn:
        return bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master"
                " WHERE type = 'table' AND name = 'packet_observations'"
            ).fetchone()
        )


class TestGroupedPacketsFromObservations:
    def test_first_page_reports_every_transmission(self, database):
        with closing(_connection(database)) as conn:
            _seed_transmissions(conn)

        result = _get_packets(database, limit=2, offset=0)

        # Three transmissions in the window: the total is the real group
        # count, not an estimate extrapolated from the page.
        assert result["total_count"] == 3
        assert result["has_more"] is True
        assert result["is_grouped"] is True

        # Newest transmissions first: C (t0+200), B (t0+100).
        assert [p["mesh_packet_id"] for p in result["packets"]] == [5002, 5001]
        assert result["packets"][0]["reception_count"] == 1
        assert result["packets"][1]["reception_count"] == 2
        assert result["packets"][1]["gateway_count"] == 2

    def test_second_page_aggregates_the_multi_gateway_transmission(self, database):
        with closing(_connection(database)) as conn:
            _seed_transmissions(conn)

        result = _get_packets(database, limit=2, offset=2)

        assert result["total_count"] == 3
        assert result["has_more"] is False

        packet = result["packets"][0]
        assert packet["mesh_packet_id"] == 5000
        assert packet["from_node_id"] == TX1
        assert packet["portnum_name"] == "TEXT_MESSAGE_APP"
        assert packet["channel_id"] == "testChannel"
        assert packet["reception_count"] == 4
        assert packet["gateway_count"] == 3
        assert set(packet["gateway_list"].split(",")) == {GW1, GW2, GW3}
        # The representative row is the earliest reception.
        assert packet["id"] == 1
        assert packet["timestamp"] == T0
        # Sentinel/garbage signal values are excluded from the ranges.
        assert packet["min_rssi"] == -80
        assert packet["max_rssi"] == -70
        assert packet["min_snr"] == 8.5
        assert packet["max_snr"] == 10.0
        assert packet["rssi_range"] == "-80.0 to -70.0 dBm"
        assert packet["snr_range"] == "8.50 to 10.00 dB"
        assert packet["hop_range"] == "0-2"
        assert packet["avg_payload_length"] == pytest.approx(20.0)
        assert packet["relay_node_grouped"] == "17*2"
        assert packet["text_content"] == "hello world"
        assert packet["is_grouped"] is True

    def test_gateway_count_sorting(self, database):
        with closing(_connection(database)) as conn:
            _seed_transmissions(conn)

        desc = _get_packets(
            database, limit=10, order_by="gateway_id", order_dir="desc"
        )
        assert [p["gateway_count"] for p in desc["packets"]] == [3, 2, 1]

        asc = _get_packets(database, limit=10, order_by="gateway_id", order_dir="asc")
        assert [p["gateway_count"] for p in asc["packets"]] == [1, 2, 3]

    def test_gateway_filter_narrows_receptions(self, database):
        with closing(_connection(database)) as conn:
            _seed_transmissions(conn)

        result = _get_packets(database, limit=10, filters={"gateway_id": GW3})

        # Only transmission A was heard by GW3, and only that reception
        # counts towards the group.
        assert result["total_count"] == 1
        packet = result["packets"][0]
        assert packet["mesh_packet_id"] == 5000
        assert packet["reception_count"] == 1
        assert packet["gateway_count"] == 1
        assert packet["gateway_list"] == GW3

    def test_start_time_filter(self, database):
        with closing(_connection(database)) as conn:
            _seed_transmissions(conn)

        result = _get_packets(database, limit=10, filters={"start_time": T0 + 150})

        assert result["total_count"] == 1
        assert result["packets"][0]["mesh_packet_id"] == 5002


class TestPartialProjection:
    def test_partial_coverage_serves_from_projection(self, database):
        if not _has_projection(database):
            pytest.skip("projection tests only apply to the materialized reader")

        with closing(_connection(database)) as conn:
            _seed_transmissions(conn)
            # Simulate an interrupted backfill: the projection holds rows
            # but only covers part of the history. The grouped read must
            # still serve from it (full SQL grouping and pagination) rather
            # than falling back to the fetch-capped in-memory path.
            conn.execute("DELETE FROM packet_observations WHERE mesh_packet_id != 5000")
            conn.commit()

        result = _get_packets(database, limit=2, offset=0)
        assert result["total_count"] == 1
        assert [p["mesh_packet_id"] for p in result["packets"]] == [5000]
