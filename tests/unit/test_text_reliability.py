"""Tests for broadcast text-message gateway reliability.

``get_text_message_reliability`` answers: of the broadcast
TEXT_MESSAGE_APP transmissions a node originated in a time window, what
fraction did each MQTT gateway hear (any hop depth, deduplicated per
transmission)? Every scenario runs against both reader paths — the raw
``packet_history`` scan and the materialized ``packet_observations``
projection written by the real capture-path materializer — and must
produce identical results.
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
from malla.database.text_reliability_repository import get_text_message_reliability
from malla.services.text_reliability_service import TextReliabilityService

pytestmark = pytest.mark.unit

TX_NODE_ID = int("11111111", 16)
OTHER_NODE_ID = int("22222222", 16)
GW_B_ID = int("aaaabbbb", 16)
GW_B_HEX = "!aaaabbbb"
GW_C_ID = int("ccccdddd", 16)
GW_C_HEX = "!ccccdddd"
BROADCAST = 4294967295


@pytest.fixture(autouse=True)
def _clear_caches():
    clear_materialization_cache()
    TextReliabilityService.clear_cache()
    yield
    clear_materialization_cache()
    TextReliabilityService.clear_cache()


@pytest.fixture(params=["legacy", "materialized"])
def database(tmp_path, request):
    path = tmp_path / f"text-reliability-{request.param}.db"
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
        "to_node_id": BROADCAST,
        "portnum": 1,
        "portnum_name": "TEXT_MESSAGE_APP",
        "gateway_id": GW_B_HEX,
        "channel_id": None,
        "rssi": None,
        "snr": None,
        "hop_limit": 3,
        "hop_start": 3,
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


def _reliability(path, node_id=TX_NODE_ID, **window):
    with closing(_connection(path)) as conn:
        return get_text_message_reliability(conn.cursor(), node_id, **window)


class TestTextMessageReliability:
    def test_percentages_and_total_sent(self, database):
        """4 broadcasts sent; gateway B hears 2 (50%), gateway C hears 4 (100%)."""
        with closing(_connection(database)) as conn:
            for tx in range(4):
                _insert_packet(conn, mesh_packet_id=1000 + tx, gateway_id=GW_C_HEX)
                if tx < 2:
                    _insert_packet(conn, mesh_packet_id=1000 + tx, gateway_id=GW_B_HEX)

        result = _reliability(database)

        assert result["node_id"] == TX_NODE_ID
        assert result["total_sent"] == 4
        by_gateway = {g["gateway_node_id"]: g for g in result["gateways"]}
        assert by_gateway[GW_B_ID]["received"] == 2
        assert by_gateway[GW_B_ID]["percent"] == 50
        assert by_gateway[GW_C_ID]["received"] == 4
        assert by_gateway[GW_C_ID]["percent"] == 100
        # Sorted by percentage descending
        assert result["gateways"][0]["gateway_node_id"] == GW_C_ID

    def test_duplicate_receptions_count_once(self, database):
        """Repeated receptions of one transmission by the same gateway (duplicate
        MQTT publishes, multi-hop copies) are one received message."""
        with closing(_connection(database)) as conn:
            _insert_packet(conn, mesh_packet_id=2000, gateway_id=GW_B_HEX)
            _insert_packet(conn, mesh_packet_id=2000, gateway_id=GW_B_HEX)
            _insert_packet(conn, mesh_packet_id=2000, gateway_id=GW_C_HEX)

        result = _reliability(database)

        assert result["total_sent"] == 1
        by_gateway = {g["gateway_node_id"]: g for g in result["gateways"]}
        assert by_gateway[GW_B_ID]["received"] == 1
        assert by_gateway[GW_B_ID]["percent"] == 100
        assert by_gateway[GW_C_ID]["received"] == 1

    def test_direct_messages_excluded(self, database):
        """Only broadcast (to=0xFFFFFFFF) text messages count."""
        with closing(_connection(database)) as conn:
            _insert_packet(conn, mesh_packet_id=3000, gateway_id=GW_B_HEX)
            _insert_packet(
                conn, mesh_packet_id=3001, gateway_id=GW_B_HEX, to_node_id=OTHER_NODE_ID
            )

        result = _reliability(database)

        assert result["total_sent"] == 1
        assert result["gateways"][0]["received"] == 1

    def test_non_text_portnums_excluded(self, database):
        """Position/telemetry packets are not text messages."""
        with closing(_connection(database)) as conn:
            _insert_packet(conn, mesh_packet_id=4000, gateway_id=GW_B_HEX, portnum=3)
            _insert_packet(conn, mesh_packet_id=4001, gateway_id=GW_B_HEX, portnum=67)

        result = _reliability(database)

        assert result["total_sent"] == 0
        assert result["gateways"] == []

    def test_self_reception_excluded(self, database):
        """A gateway sharing the sender's node id only heard its own loopback."""
        self_gw = f"!{TX_NODE_ID:08x}"
        with closing(_connection(database)) as conn:
            _insert_packet(conn, mesh_packet_id=5000, gateway_id=self_gw)
            _insert_packet(conn, mesh_packet_id=5001, gateway_id=self_gw)

        result = _reliability(database)

        assert result["total_sent"] == 0
        assert result["gateways"] == []

    def test_other_senders_excluded(self, database):
        """Another node's broadcasts never enter this node's statistics."""
        with closing(_connection(database)) as conn:
            _insert_packet(conn, mesh_packet_id=6000, gateway_id=GW_B_HEX)
            _insert_packet(
                conn, mesh_packet_id=6001, gateway_id=GW_B_HEX, from_node_id=OTHER_NODE_ID
            )

        result = _reliability(database)

        assert result["total_sent"] == 1

    def test_any_hop_depth_counts(self, database):
        """Relayed receptions (hop_limit decremented by re-transmissions) count."""
        with closing(_connection(database)) as conn:
            _insert_packet(conn, mesh_packet_id=7000, gateway_id=GW_B_HEX, hop_start=3, hop_limit=3)
            _insert_packet(conn, mesh_packet_id=7000, gateway_id=GW_C_HEX, hop_start=3, hop_limit=1)

        result = _reliability(database)

        assert result["total_sent"] == 1
        assert {g["gateway_node_id"] for g in result["gateways"]} == {GW_B_ID, GW_C_ID}

    def test_time_window_filters_transmissions(self, database):
        """start_time/end_time bound the counted transmissions."""
        now = time.time()
        with closing(_connection(database)) as conn:
            _insert_packet(
                conn, mesh_packet_id=8000, gateway_id=GW_B_HEX, timestamp=now - 10 * 3600
            )
            _insert_packet(
                conn, mesh_packet_id=8001, gateway_id=GW_B_HEX, timestamp=now - 3600
            )
            _insert_packet(
                conn, mesh_packet_id=8002, gateway_id=GW_B_HEX, timestamp=now - 2 * 3600
            )

        full = _reliability(database)
        recent = _reliability(database, start_time=now - 3 * 3600, end_time=now)
        oldest = _reliability(database, start_time=now - 12 * 3600, end_time=now - 5 * 3600)

        assert full["total_sent"] == 3
        assert recent["total_sent"] == 2
        assert oldest["total_sent"] == 1

    def test_denominator_spans_all_gateways(self, database):
        """A transmission only one gateway heard still counts as sent."""
        with closing(_connection(database)) as conn:
            _insert_packet(conn, mesh_packet_id=9000, gateway_id=GW_C_HEX)
            _insert_packet(conn, mesh_packet_id=9001, gateway_id=GW_B_HEX)
            _insert_packet(conn, mesh_packet_id=9002, gateway_id=GW_B_HEX)

        result = _reliability(database)

        assert result["total_sent"] == 3
        by_gateway = {g["gateway_node_id"]: g for g in result["gateways"]}
        # B received 2 of 3
        assert by_gateway[GW_B_ID]["percent"] == 67
        # C received 1 of 3
        assert by_gateway[GW_C_ID]["percent"] == 33

    def test_empty_node_returns_zeroed_payload(self, database):
        result = _reliability(database, node_id=OTHER_NODE_ID)

        assert result == {
            "node_id": OTHER_NODE_ID,
            "total_sent": 0,
            "gateways": [],
        }


class TestTextMessageReliabilityEndpoint:
    """Test /api/node/<id>/text-message-reliability parameter handling."""

    @staticmethod
    def _payload(*_args, **_kwargs):
        return {"node_id": 1, "total_sent": 0, "gateways": []}

    def test_hours_param_resolves_window(self, client):
        with patch(
            "src.malla.routes.api_routes.TextReliabilityService.get_text_message_reliability",
            side_effect=self._payload,
        ) as service_mock:
            response = client.get("/api/node/42/text-message-reliability?hours=24")

        assert response.status_code == 200
        node_id, filters = service_mock.call_args.args
        assert node_id == 42
        assert filters["end_time"] - filters["start_time"] == pytest.approx(
            24 * 3600, abs=3600
        )

    def test_start_end_params_pass_through_grid_snapped(self, client):
        end = int(time.time()) - 7200
        start = end - 3600
        with patch(
            "src.malla.routes.api_routes.TextReliabilityService.get_text_message_reliability",
            side_effect=self._payload,
        ) as service_mock:
            response = client.get(
                f"/api/node/42/text-message-reliability?start_time={start}&end_time={end}"
            )

        assert response.status_code == 200
        _node_id, filters = service_mock.call_args.args
        assert filters["start_time"] <= start
        assert filters["end_time"] >= end

    def test_hex_node_id_accepted(self, client):
        with patch(
            "src.malla.routes.api_routes.TextReliabilityService.get_text_message_reliability",
            side_effect=self._payload,
        ) as service_mock:
            response = client.get("/api/node/!0000002a/text-message-reliability")

        assert response.status_code == 200
        assert service_mock.call_args.args[0] == 42

    def test_invalid_node_id_returns_400(self, client):
        response = client.get("/api/node/not-a-node/text-message-reliability")
        assert response.status_code == 400

    def test_inverted_window_returns_400(self, client):
        now = time.time()
        response = client.get(
            f"/api/node/42/text-message-reliability?start_time={now}&end_time={now - 10}"
        )
        assert response.status_code == 400

    def test_response_shape(self, client):
        payload = {
            "node_id": 42,
            "total_sent": 100,
            "gateways": [
                {
                    "gateway_node_id": 7,
                    "gateway_id": "!00000007",
                    "received": 50,
                    "percent": 50,
                }
            ],
        }
        with patch(
            "src.malla.routes.api_routes.TextReliabilityService.get_text_message_reliability",
            return_value=payload,
        ):
            response = client.get("/api/node/42/text-message-reliability?hours=24")

        assert response.status_code == 200
        assert response.get_json() == payload
