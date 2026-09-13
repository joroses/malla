"""Tests for the materialized transmission/position writers and backfill.

The derived tables collapse raw receptions once, at write time: duplicate
receptions of one mesh packet at one gateway must merge into counters and
plausible-value signal sums, position fixes must be decoded exactly once and
only stored when valid, and the backfill walk must never revisit a packet row
(transmission upserts are increments, so a revisit would double-count).
"""

import json
import sqlite3
from types import SimpleNamespace

import pytest
from meshtastic import mesh_pb2

from malla.backfill_materializations import (
    backfill_materializations,
    inspect_materializations,
    reset_materializations,
)
from malla.database.materialization_schema import ensure_materialization_schema
from malla.database.materializations import (
    clear_materialization_cache,
    decode_position,
    materialization_tx_scope,
    materialize_packet,
    observation_tx_id,
    transmission_tx_id,
    write_observation,
    write_position,
    write_transmission,
)
from malla.database.position_schema import ensure_position_schema

pytestmark = pytest.mark.unit

VALID_LAT = 52.37
VALID_LON = 4.89


def test_backwards_compatibility_aliases():
    assert transmission_tx_id is observation_tx_id
    assert write_transmission is write_observation


@pytest.fixture
def conn():
    clear_materialization_cache()
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    cursor.execute(
        """
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
            processed_successfully BOOLEAN DEFAULT TRUE
        )
        """
    )
    ensure_materialization_schema(cursor)
    connection.commit()
    yield connection
    connection.close()
    clear_materialization_cache()


def _packet_fields(**overrides):
    fields = {
        "timestamp": 1000.0,
        "mesh_packet_id": None,
        "from_node_id": 111,
        "to_node_id": None,
        "portnum": 1,
        "portnum_name": "TEXT_MESSAGE_APP",
        "gateway_id": "!aabbccdd",
        "channel_id": None,
        "rssi": None,
        "snr": None,
        "hop_limit": None,
        "hop_start": None,
        "payload_length": 10,
        "raw_payload": b"",
        "processed_successfully": 1,
    }
    fields.update(overrides)
    return fields


def _insert_raw(cursor, **overrides):
    """Insert a packet_history row without materializing it (backfill input)."""

    fields = _packet_fields(**overrides)
    columns = ", ".join(fields)
    placeholders = ", ".join("?" for _ in fields)
    cursor.execute(
        f"INSERT INTO packet_history ({columns}) VALUES ({placeholders})",
        tuple(fields.values()),
    )
    return dict(
        cursor.execute(
            "SELECT * FROM packet_history WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
    )


def _capture(cursor, **overrides):
    """Insert a row and materialize it, as the capture hook does."""

    row = _insert_raw(cursor, **overrides)
    materialize_packet(cursor, row)
    return row


def _observation(cursor, tx_id_or_mesh_id, gateway_id="!aabbccdd"):
    return cursor.execute(
        "SELECT * FROM packet_observations WHERE (tx_id = ? OR mesh_packet_id = ?) AND gateway_id = ? ORDER BY timestamp DESC LIMIT 1",
        (tx_id_or_mesh_id, tx_id_or_mesh_id, gateway_id),
    ).fetchone()


# Backwards-compatibility alias for test helpers
_transmission = _observation


def _position_payload(
    lat=VALID_LAT, lon=VALID_LON, altitude=42, precision_bits=17, sats_in_view=8
):
    return mesh_pb2.Position(
        latitude_i=int(lat * 1e7),
        longitude_i=int(lon * 1e7),
        altitude=altitude,
        precision_bits=precision_bits,
        sats_in_view=sats_in_view,
    ).SerializeToString()


def _traceroute_payload(route=(100, 200), snr=(10,)):
    return mesh_pb2.RouteDiscovery(
        route=route, snr_towards=snr, route_back=(), snr_back=()
    ).SerializeToString()


class TestWriteObservation:
    def test_first_reception_creates_row(self, conn):
        cursor = conn.cursor()
        row = _capture(cursor, mesh_packet_id=500, snr=7.5, rssi=-90)
        obs = _observation(cursor, 500)
        assert obs["packet_id"] == row["id"]
        assert obs["tx_id"] == row["id"]
        assert obs["from_node_id"] == 111
        assert obs["snr"] == 7.5
        assert obs["rssi"] == -90
        assert obs["timestamp"] == row["timestamp"]
        assert obs["is_direct"] == 0

    def test_duplicate_receptions_store_separate_rows_sharing_tx_id(self, conn):
        cursor = conn.cursor()
        r1 = _capture(cursor, mesh_packet_id=500, snr=5.0, rssi=-80, timestamp=1000.0)
        r2 = _capture(cursor, mesh_packet_id=500, snr=7.5, rssi=-90, timestamp=1001.0)
        rows = cursor.execute(
            "SELECT * FROM packet_observations WHERE mesh_packet_id = 500 ORDER BY timestamp ASC"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["packet_id"] == r1["id"]
        assert rows[1]["packet_id"] == r2["id"]
        assert rows[0]["tx_id"] == rows[1]["tx_id"] == r1["id"]
        assert rows[0]["snr"] == 5.0
        assert rows[1]["snr"] == 7.5
        assert rows[0]["rssi"] == -80
        assert rows[1]["rssi"] == -90

    def test_reception_filters_preserved_with_timestamps_1000_and_2000(self, conn):
        """Reproduce and verify the filtering behavior on multi-reception transmissions.

        With receptions at timestamps 1000 and 2000:
        - timestamp >= 1500 returns only the 2000 reception (no overcounting).
        - timestamp <= 1500 returns only the 1000 reception (group is not lost).
        - is_direct = 1 isolates the direct reception without relay contamination.
        - rssi threshold filters accurately at the reception level.
        """
        cursor = conn.cursor()
        r1 = _capture(
            cursor,
            mesh_packet_id=500,
            hop_start=3,
            hop_limit=3,
            snr=2.0,
            rssi=-110,
            timestamp=1000.0,
        )
        r2 = _capture(
            cursor,
            mesh_packet_id=500,
            hop_start=3,
            hop_limit=2,
            snr=10.0,
            rssi=-80,
            timestamp=2000.0,
        )

        # Both receptions share the same tx_id
        rows = cursor.execute(
            "SELECT * FROM packet_observations WHERE mesh_packet_id = 500"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["tx_id"] == rows[1]["tx_id"]

        # Filter timestamp >= 1500: returns only the 2000.0 reception (no overcounting)
        after_1500 = cursor.execute(
            "SELECT * FROM packet_observations WHERE timestamp >= 1500.0"
        ).fetchall()
        assert len(after_1500) == 1
        assert after_1500[0]["packet_id"] == r2["id"]
        assert after_1500[0]["timestamp"] == 2000.0
        assert after_1500[0]["rssi"] == -80

        # Filter timestamp <= 1500: returns only the 1000.0 reception (group is NOT lost)
        before_1500 = cursor.execute(
            "SELECT * FROM packet_observations WHERE timestamp <= 1500.0"
        ).fetchall()
        assert len(before_1500) == 1
        assert before_1500[0]["packet_id"] == r1["id"]
        assert before_1500[0]["timestamp"] == 1000.0
        assert before_1500[0]["rssi"] == -110

        # Filter is_direct = 1: returns only the direct reception (snr=2.0, not contaminated by 10.0)
        direct_rows = cursor.execute(
            "SELECT * FROM packet_observations WHERE is_direct = 1"
        ).fetchall()
        assert len(direct_rows) == 1
        assert direct_rows[0]["packet_id"] == r1["id"]
        assert direct_rows[0]["snr"] == 2.0

        # Filter rssi >= -90: returns only r2 (r1 at -110 is excluded)
        strong_rssi = cursor.execute(
            "SELECT * FROM packet_observations WHERE rssi >= -90"
        ).fetchall()
        assert len(strong_rssi) == 1
        assert strong_rssi[0]["packet_id"] == r2["id"]

    def test_separate_gateways_are_separate_rows(self, conn):
        cursor = conn.cursor()
        r1 = _capture(cursor, mesh_packet_id=500, gateway_id="!aabbccdd")
        _capture(cursor, mesh_packet_id=500, gateway_id="!11223344")
        rows = cursor.execute(
            "SELECT * FROM packet_observations WHERE tx_id = ?",
            (r1["id"],),
        ).fetchall()
        assert len(rows) == 2
        assert {r["gateway_id"] for r in rows} == {"!aabbccdd", "!11223344"}

    def test_unknown_gateway_stored_as_empty_sentinel(self, conn):
        cursor = conn.cursor()
        _capture(cursor, mesh_packet_id=500, gateway_id=None)
        assert _observation(cursor, 500, gateway_id="") is not None

    def test_missing_mesh_packet_id_falls_back_to_negative_row_id(self, conn):
        cursor = conn.cursor()
        first = _capture(cursor, mesh_packet_id=None)
        second = _capture(cursor, mesh_packet_id=0)
        _capture(cursor, mesh_packet_id=700)
        assert _observation(cursor, -first["id"]) is not None
        assert _observation(cursor, -second["id"]) is not None
        assert first["id"] != second["id"]
        assert _observation(cursor, 700) is not None
        assert observation_tx_id({"id": 5, "mesh_packet_id": None}) == -5
        assert observation_tx_id({"id": 5, "mesh_packet_id": 0}) == -5
        assert (
            observation_tx_id({"id": 5, "mesh_packet_id": 700, "from_node_id": 111})
            == 5
        )
        assert (
            observation_tx_id({"id": 5, "mesh_packet_id": 700, "from_node_id": None})
            == -5
        )

    def test_two_senders_same_mesh_id_create_separate_observations(self, conn):
        cursor = conn.cursor()
        _capture(cursor, mesh_packet_id=500, from_node_id=111, gateway_id="!gw1")
        _capture(cursor, mesh_packet_id=500, from_node_id=222, gateway_id="!gw1")

        obs1 = cursor.execute(
            "SELECT * FROM packet_observations WHERE from_node_id = 111 AND gateway_id = '!gw1'"
        ).fetchone()
        obs2 = cursor.execute(
            "SELECT * FROM packet_observations WHERE from_node_id = 222 AND gateway_id = '!gw1'"
        ).fetchone()

        assert obs1 is not None
        assert obs2 is not None
        assert obs1["tx_id"] != obs2["tx_id"]
        assert obs1["from_node_id"] == 111
        assert obs2["from_node_id"] == 222
        assert obs1["mesh_packet_id"] == 500
        assert obs2["mesh_packet_id"] == 500

        total = cursor.execute(
            "SELECT COUNT(*) FROM packet_observations WHERE gateway_id = '!gw1'"
        ).fetchone()[0]
        assert total == 2

    def test_id_reuse_across_time_window_creates_new_transmission(self, conn):
        cursor = conn.cursor()
        _capture(cursor, mesh_packet_id=500, from_node_id=111, timestamp=1000.0)
        _capture(
            cursor,
            mesh_packet_id=500,
            from_node_id=111,
            timestamp=1000.0 + 7200.0,
        )

        obs = cursor.execute(
            "SELECT * FROM packet_observations WHERE from_node_id = 111 ORDER BY timestamp ASC"
        ).fetchall()
        assert len(obs) == 2
        assert obs[0]["tx_id"] != obs[1]["tx_id"]
        assert obs[0]["timestamp"] == 1000.0
        assert obs[1]["timestamp"] == 8200.0

    def test_grouping_contract_destination_and_port_separate_transmissions(self, conn):
        cursor = conn.cursor()
        _capture(
            cursor, mesh_packet_id=500, from_node_id=111, to_node_id=222, portnum=1
        )
        _capture(
            cursor, mesh_packet_id=500, from_node_id=111, to_node_id=333, portnum=1
        )
        _capture(
            cursor, mesh_packet_id=500, from_node_id=111, to_node_id=222, portnum=3
        )

        obs = cursor.execute(
            "SELECT * FROM packet_observations WHERE from_node_id = 111"
        ).fetchall()
        assert len(obs) == 3
        tx_ids = {r["tx_id"] for r in obs}
        assert len(tx_ids) == 3

    def test_duplicate_receptions_within_window_share_tx_id(self, conn):
        cursor = conn.cursor()
        r1 = _capture(
            cursor,
            mesh_packet_id=500,
            from_node_id=111,
            to_node_id=222,
            portnum=1,
            gateway_id="!gw1",
            timestamp=1000.0,
        )
        _capture(
            cursor,
            mesh_packet_id=500,
            from_node_id=111,
            to_node_id=222,
            portnum=1,
            gateway_id="!gw2",
            timestamp=1001.0,
        )
        _capture(
            cursor,
            mesh_packet_id=500,
            from_node_id=111,
            to_node_id=222,
            portnum=1,
            gateway_id="!gw1",
            timestamp=1002.0,
        )

        rows = cursor.execute(
            "SELECT * FROM packet_observations WHERE from_node_id = 111 ORDER BY timestamp ASC"
        ).fetchall()
        assert len(rows) == 3
        assert all(r["tx_id"] == r1["id"] for r in rows)

    def test_missing_sender_or_missing_mesh_id_never_collides(self, conn):
        cursor = conn.cursor()
        _capture(cursor, mesh_packet_id=500, from_node_id=None)
        _capture(cursor, mesh_packet_id=500, from_node_id=None)
        _capture(cursor, mesh_packet_id=None, from_node_id=111)
        _capture(cursor, mesh_packet_id=0, from_node_id=111)

        obs = cursor.execute("SELECT * FROM packet_observations").fetchall()
        assert len(obs) == 4
        tx_ids = [r["tx_id"] for r in obs]
        assert all(tx < 0 for tx in tx_ids)
        assert len(set(tx_ids)) == 4

    def test_direct_and_relayed_reception_hops_isolated(self, conn):
        cursor = conn.cursor()
        r_direct = _capture(
            cursor,
            mesh_packet_id=500,
            hop_start=3,
            hop_limit=3,
            snr=2.0,
            rssi=-70,
            timestamp=1000.0,
        )
        r_relay = _capture(
            cursor,
            mesh_packet_id=500,
            hop_start=3,
            hop_limit=2,
            snr=10.0,
            rssi=-50,
            timestamp=1005.0,
        )

        rows = cursor.execute(
            "SELECT * FROM packet_observations WHERE mesh_packet_id = 500 ORDER BY timestamp ASC"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["packet_id"] == r_direct["id"]
        assert rows[0]["is_direct"] == 1
        assert rows[0]["snr"] == 2.0
        assert rows[0]["rssi"] == -70

        assert rows[1]["packet_id"] == r_relay["id"]
        assert rows[1]["is_direct"] == 0
        assert rows[1]["snr"] == 10.0
        assert rows[1]["rssi"] == -50

    def test_direct_link_mean_of_transmission_means_parity(self, conn):
        cursor = conn.cursor()
        # Transmission 1 (mesh_packet_id 101): Direct SNR 2.0, plus later Relayed SNR 10.0
        _capture(
            cursor,
            mesh_packet_id=101,
            from_node_id=111,
            gateway_id="!gw1",
            hop_start=3,
            hop_limit=3,
            snr=2.0,
            timestamp=1000.0,
        )
        _capture(
            cursor,
            mesh_packet_id=101,
            from_node_id=111,
            gateway_id="!gw1",
            hop_start=3,
            hop_limit=2,
            snr=10.0,
            timestamp=1002.0,
        )

        # Transmission 2 (mesh_packet_id 102): Two direct receptions with SNR -6.0 and -4.0 (mean = -5.0)
        _capture(
            cursor,
            mesh_packet_id=102,
            from_node_id=111,
            gateway_id="!gw1",
            hop_start=3,
            hop_limit=3,
            snr=-6.0,
            timestamp=1004.0,
        )
        _capture(
            cursor,
            mesh_packet_id=102,
            from_node_id=111,
            gateway_id="!gw1",
            hop_start=3,
            hop_limit=3,
            snr=-4.0,
            timestamp=1005.0,
        )

        # Expected mean of transmission means:
        # Tx 1 direct mean = 2.0
        # Tx 2 direct mean = (-6.0 + -4.0) / 2 = -5.0
        # Combined link SNR = (2.0 + (-5.0)) / 2 = -1.5 dB
        row = cursor.execute(
            """
            WITH transmissions AS (
                SELECT
                    from_node_id,
                    gateway_id,
                    tx_id,
                    COUNT(*) AS reception_count,
                    AVG(snr) AS tx_snr
                FROM packet_observations
                WHERE is_direct = 1
                GROUP BY from_node_id, gateway_id, tx_id
            )
            SELECT
                from_node_id,
                gateway_id,
                COUNT(*) AS packet_count,
                SUM(reception_count) AS reception_count,
                AVG(tx_snr) AS avg_snr
            FROM transmissions
            GROUP BY from_node_id, gateway_id
            """
        ).fetchone()

        assert row["packet_count"] == 2
        assert row["reception_count"] == 3  # 1 direct for Tx1 + 2 direct for Tx2
        assert row["avg_snr"] == pytest.approx(-1.5)

    def test_schema_migration_drops_legacy_collapsed_table(self):
        # Create an isolated connection with older table definition
        mem_conn = sqlite3.connect(":memory:")
        mem_conn.row_factory = sqlite3.Row
        cur = mem_conn.cursor()
        cur.execute(
            """
            CREATE TABLE packet_observations (
                tx_id INTEGER NOT NULL,
                gateway_id TEXT NOT NULL,
                from_node_id INTEGER,
                reception_count INTEGER NOT NULL DEFAULT 1,
                last_seen REAL NOT NULL,
                PRIMARY KEY (tx_id, gateway_id)
            ) WITHOUT ROWID
            """
        )

        from malla.database.observation_schema import ensure_observation_schema

        ensure_observation_schema(cur)

        cur.execute("PRAGMA table_info(packet_observations)")
        cols = {r[1] for r in cur.fetchall()}
        assert "packet_id" in cols
        assert "is_direct" in cols
        assert "reception_count" not in cols

        cur.execute("PRAGMA table_info(packet_observations)")
        pk_cols = [r[1] for r in cur.fetchall() if r[5] > 0]
        assert pk_cols == ["packet_id"]
        mem_conn.close()

    def test_requires_active_transaction(self, conn):
        cursor = conn.cursor()
        conn.commit()
        with pytest.raises(ValueError, match="active transaction"):
            write_observation(cursor, _packet_fields(id=1, mesh_packet_id=500))


class TestTxCacheTransactionSafety:
    def test_rolled_back_tx_identity_never_leaks_to_committed_writes(self, conn):
        cursor = conn.cursor()

        with pytest.raises(RuntimeError, match="capture failure"):
            with materialization_tx_scope(), conn:
                _capture(cursor, mesh_packet_id=500, from_node_id=111, timestamp=1000.0)
                raise RuntimeError("capture failure")

        with materialization_tx_scope(), conn:
            unrelated = _capture(
                cursor, mesh_packet_id=600, from_node_id=111, timestamp=1001.0
            )
        with materialization_tx_scope(), conn:
            retried = _capture(
                cursor, mesh_packet_id=500, from_node_id=111, timestamp=1002.0
            )

        merged = cursor.execute(
            """
            SELECT tx_id
            FROM packet_observations
            GROUP BY tx_id
            HAVING COUNT(DISTINCT mesh_packet_id) > 1
            """
        ).fetchall()
        assert merged == []
        assert (
            cursor.execute(
                "SELECT tx_id FROM packet_observations WHERE packet_id = ?",
                (unrelated["id"],),
            ).fetchone()["tx_id"]
            == unrelated["id"]
        )
        assert (
            cursor.execute(
                "SELECT tx_id FROM packet_observations WHERE packet_id = ?",
                (retried["id"],),
            ).fetchone()["tx_id"]
            == retried["id"]
        )

    def test_committed_tx_identity_correlates_receptions_across_transactions(
        self, conn
    ):
        cursor = conn.cursor()
        with materialization_tx_scope(), conn:
            first = _capture(
                cursor,
                mesh_packet_id=500,
                from_node_id=111,
                gateway_id="!gw1",
                timestamp=1000.0,
            )
        with materialization_tx_scope(), conn:
            _capture(
                cursor,
                mesh_packet_id=500,
                from_node_id=111,
                gateway_id="!gw2",
                timestamp=1001.0,
            )

        rows = cursor.execute(
            "SELECT tx_id FROM packet_observations ORDER BY timestamp ASC"
        ).fetchall()
        assert len(rows) == 2
        assert all(r["tx_id"] == first["id"] for r in rows)


class TestWritePosition:
    def test_valid_position_stored_decoded(self, conn):
        cursor = conn.cursor()
        _capture(
            cursor,
            mesh_packet_id=900,
            portnum=3,
            portnum_name="POSITION_APP",
            raw_payload=_position_payload(),
        )
        position = cursor.execute("SELECT * FROM node_positions").fetchone()
        assert position["node_id"] == 111
        assert position["tx_id"] == 900
        assert position["latitude"] == pytest.approx(VALID_LAT)
        assert position["longitude"] == pytest.approx(VALID_LON)
        assert position["altitude"] == 42
        assert position["precision_bits"] == 17
        assert position["sats_in_view"] == 8

    def test_null_island_fix_skipped(self, conn):
        cursor = conn.cursor()
        _capture(
            cursor,
            portnum=3,
            raw_payload=_position_payload(lat=0.0001, lon=0.0001),
        )
        assert cursor.execute("SELECT COUNT(*) FROM node_positions").fetchone()[0] == 0

    def test_undecodable_payload_skipped(self, conn):
        assert decode_position(b"\x0a\xff\xff\xff\xff") is None
        assert decode_position(b"") is None
        assert decode_position(None) is None

    def test_non_position_portnum_skipped(self, conn):
        cursor = conn.cursor()
        _capture(cursor, portnum=1, raw_payload=_position_payload())
        assert cursor.execute("SELECT COUNT(*) FROM node_positions").fetchone()[0] == 0

    def test_position_without_from_node_skipped(self, conn):
        cursor = conn.cursor()
        _capture(
            cursor,
            portnum=3,
            from_node_id=None,
            raw_payload=_position_payload(),
        )
        assert cursor.execute("SELECT COUNT(*) FROM node_positions").fetchone()[0] == 0

    def test_reprocessing_same_packet_is_idempotent(self, conn):
        cursor = conn.cursor()
        row = _capture(
            cursor,
            mesh_packet_id=900,
            portnum=3,
            timestamp=1000.0,
            raw_payload=_position_payload(),
        )
        write_position(cursor, row)
        rows = cursor.execute("SELECT * FROM node_positions").fetchall()
        assert len(rows) == 1
        assert rows[0]["packet_id"] == row["id"]

    def test_two_senders_same_mesh_id_both_store_positions(self, conn):
        cursor = conn.cursor()
        _capture(
            cursor,
            mesh_packet_id=500,
            from_node_id=111,
            portnum=3,
            gateway_id="!gw1",
            raw_payload=_position_payload(lat=52.0, lon=4.0),
        )
        _capture(
            cursor,
            mesh_packet_id=500,
            from_node_id=222,
            portnum=3,
            gateway_id="!gw1",
            raw_payload=_position_payload(lat=53.0, lon=5.0),
        )
        rows = cursor.execute(
            "SELECT * FROM node_positions ORDER BY node_id"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["node_id"] == 111
        assert rows[0]["latitude"] == pytest.approx(52.0)
        assert rows[1]["node_id"] == 222
        assert rows[1]["latitude"] == pytest.approx(53.0)

    def test_multiple_fixes_same_node_both_stored(self, conn):
        cursor = conn.cursor()
        first = _capture(
            cursor,
            mesh_packet_id=500,
            from_node_id=111,
            portnum=3,
            timestamp=1000.0,
            raw_payload=_position_payload(lat=52.0, lon=4.0),
        )
        second = _capture(
            cursor,
            mesh_packet_id=500,
            from_node_id=111,
            portnum=3,
            timestamp=8200.0,
            raw_payload=_position_payload(lat=52.5, lon=4.5),
        )
        rows = cursor.execute(
            "SELECT * FROM node_positions WHERE node_id = 111 ORDER BY timestamp ASC"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["packet_id"] == first["id"]
        assert rows[1]["packet_id"] == second["id"]
        assert rows[0]["latitude"] == pytest.approx(52.0)
        assert rows[1]["latitude"] == pytest.approx(52.5)

    def test_same_transmission_per_gateway(self, conn):
        cursor = conn.cursor()
        _capture(
            cursor,
            mesh_packet_id=900,
            portnum=3,
            gateway_id="!aabbccdd",
            raw_payload=_position_payload(),
        )
        _capture(
            cursor,
            mesh_packet_id=900,
            portnum=3,
            gateway_id="!11223344",
            raw_payload=_position_payload(),
        )
        assert cursor.execute("SELECT COUNT(*) FROM node_positions").fetchone()[0] == 2

    def test_repeating_position_preserves_historical_windows_and_deterministic_order(
        self, conn
    ):
        cursor = conn.cursor()
        # Initial position reception at t=1000
        first = _capture(
            cursor,
            mesh_packet_id=500,
            from_node_id=111,
            portnum=3,
            timestamp=1000.0,
            raw_payload=_position_payload(lat=52.0, lon=4.0),
        )
        # Identical payload repeated at t=2000
        second = _capture(
            cursor,
            mesh_packet_id=500,
            from_node_id=111,
            portnum=3,
            timestamp=2000.0,
            raw_payload=_position_payload(lat=52.0, lon=4.0),
        )

        # Both receptions must be stored, keyed by packet_id
        rows = cursor.execute(
            "SELECT * FROM node_positions WHERE node_id = 111 ORDER BY timestamp ASC, packet_id ASC"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["packet_id"] == first["id"]
        assert rows[0]["timestamp"] == 1000.0
        assert rows[1]["packet_id"] == second["id"]
        assert rows[1]["timestamp"] == 2000.0

        # Window starting at 1500 returns the second reception (not empty)
        after_1500 = cursor.execute(
            "SELECT * FROM node_positions WHERE node_id = 111 AND timestamp >= 1500"
        ).fetchall()
        assert len(after_1500) == 1
        assert after_1500[0]["packet_id"] == second["id"]

        # Window ending at 1500 returns the first reception (evidence not overwritten)
        before_1500 = cursor.execute(
            "SELECT * FROM node_positions WHERE node_id = 111 AND timestamp <= 1500"
        ).fetchall()
        assert len(before_1500) == 1
        assert before_1500[0]["packet_id"] == first["id"]

        # Deterministic latest-fix query selects the newest timestamp
        latest = cursor.execute(
            "SELECT * FROM node_positions WHERE node_id = 111 ORDER BY timestamp DESC, packet_id DESC LIMIT 1"
        ).fetchone()
        assert latest["packet_id"] == second["id"]

        # Tied timestamp reception at t=2000 from another gateway
        third = _capture(
            cursor,
            mesh_packet_id=500,
            from_node_id=111,
            portnum=3,
            gateway_id="!anothergw",
            timestamp=2000.0,
            raw_payload=_position_payload(lat=52.0, lon=4.0),
        )
        # packet_id DESC breaks the tie deterministically
        latest_tied = cursor.execute(
            "SELECT * FROM node_positions WHERE node_id = 111 ORDER BY timestamp DESC, packet_id DESC LIMIT 1"
        ).fetchone()
        assert latest_tied["packet_id"] == third["id"]

    def test_position_schema_index_upgrade_adds_packet_id_tie_breaker(self, conn):
        cursor = conn.cursor()
        # Drop current indexes and create legacy 2-column indexes
        cursor.execute("DROP INDEX IF EXISTS idx_np_node_time")
        cursor.execute("DROP INDEX IF EXISTS idx_np_gateway_time")
        cursor.execute(
            "CREATE INDEX idx_np_node_time ON node_positions(node_id, timestamp DESC)"
        )
        cursor.execute(
            "CREATE INDEX idx_np_gateway_time ON node_positions(gateway_id, timestamp DESC)"
        )

        # PRAGMA index_info should reflect legacy 2 columns
        cursor.execute("PRAGMA index_info(idx_np_node_time)")
        assert [c[2] for c in cursor.fetchall()] == ["node_id", "timestamp"]

        # ensure_position_schema must detect and upgrade the indexes
        ensure_position_schema(cursor)

        cursor.execute("PRAGMA index_info(idx_np_node_time)")
        assert [c[2] for c in cursor.fetchall()] == [
            "node_id",
            "timestamp",
            "packet_id",
        ]
        cursor.execute("PRAGMA index_info(idx_np_gateway_time)")
        assert [c[2] for c in cursor.fetchall()] == [
            "gateway_id",
            "timestamp",
            "packet_id",
        ]


class TestWriteTraceroute:
    def test_traceroute_materialized_via_materialize_packet(self, conn):
        cursor = conn.cursor()
        ensure_materialization_schema(cursor)
        row = _capture(
            cursor,
            mesh_packet_id=777,
            from_node_id=100,
            to_node_id=200,
            portnum=70,
            portnum_name="TRACEROUTE_APP",
            raw_payload=_traceroute_payload(route=[110, 100, 110], snr=[4, 8, 12, 16]),
        )
        route = cursor.execute(
            "SELECT * FROM traceroute_routes WHERE packet_id = ?", (row["id"],)
        ).fetchone()
        assert route is not None
        assert route["parse_status"] == "parsed"
        assert json.loads(route["route_nodes_json"]) == [110, 100, 110]
        hops = cursor.execute(
            "SELECT * FROM traceroute_hops WHERE packet_id = ?", (row["id"],)
        ).fetchall()
        assert len(hops) > 0

    def test_non_traceroute_portnum_skipped(self, conn):
        cursor = conn.cursor()
        ensure_materialization_schema(cursor)
        row = _capture(
            cursor,
            mesh_packet_id=888,
            portnum=1,
            portnum_name="TEXT_MESSAGE_APP",
            raw_payload=_traceroute_payload(),
        )
        route = cursor.execute(
            "SELECT * FROM traceroute_routes WHERE packet_id = ?", (row["id"],)
        ).fetchone()
        # Even if SQLite trigger inserted a pending stub on raw insert,
        # materialize_packet shouldn't have parsed it for portnum=1
        assert route is None or route["parse_status"] == "pending"


class TestBackfill:
    def test_processes_all_rows_and_sets_watermark(self, conn):
        cursor = conn.cursor()
        _insert_raw(cursor, mesh_packet_id=500)
        _insert_raw(cursor, mesh_packet_id=501)
        _insert_raw(cursor, mesh_packet_id=None)
        conn.commit()
        result = backfill_materializations(conn, batch_size=2)
        assert result["processed_this_run"] == 3
        assert result["complete"] is True
        assert result["watermark"] == result["packet_history_max_id"]
        assert result["packet_observations_rows"] == 3
        conn.commit()

    def test_resume_processes_only_new_rows_without_double_counting(self, conn):
        cursor = conn.cursor()
        _insert_raw(cursor, mesh_packet_id=500)
        _insert_raw(cursor, mesh_packet_id=500)
        conn.commit()
        backfill_materializations(conn, batch_size=1)
        count = cursor.execute(
            "SELECT COUNT(*) FROM packet_observations WHERE mesh_packet_id = 500"
        ).fetchone()[0]
        assert count == 2

        _insert_raw(cursor, mesh_packet_id=500)
        conn.commit()
        result = backfill_materializations(conn, batch_size=10)
        assert result["processed_this_run"] == 1
        assert result["previous_watermark"] > 0
        count = cursor.execute(
            "SELECT COUNT(*) FROM packet_observations WHERE mesh_packet_id = 500"
        ).fetchone()[0]
        assert count == 3
        conn.commit()

    def test_rerun_after_completion_is_noop(self, conn):
        cursor = conn.cursor()
        _insert_raw(cursor, mesh_packet_id=500)
        _insert_raw(cursor, mesh_packet_id=501)
        conn.commit()
        backfill_materializations(conn)
        result = backfill_materializations(conn)
        assert result["processed_this_run"] == 0
        assert result["complete"] is True
        conn.commit()

    def test_reset_clears_derived_rows_and_watermark(self, conn):
        cursor = conn.cursor()
        _insert_raw(cursor, mesh_packet_id=500, portnum=3, raw_payload=b"x")
        conn.commit()
        backfill_materializations(conn)
        reset_materializations(conn)
        assert inspect_materializations(conn.cursor())["pending_packets"] == 1
        result = backfill_materializations(conn)
        assert result["processed_this_run"] == 1
        conn.commit()

    def test_positions_materialized_from_backfill(self, conn):
        cursor = conn.cursor()
        _insert_raw(
            cursor,
            mesh_packet_id=900,
            portnum=3,
            raw_payload=_position_payload(),
        )
        conn.commit()
        backfill_materializations(conn)
        position = cursor.execute("SELECT * FROM node_positions").fetchone()
        assert position["tx_id"] == 900
        conn.commit()

    def test_traceroutes_materialized_from_backfill(self, conn):
        cursor = conn.cursor()
        _insert_raw(
            cursor,
            mesh_packet_id=777,
            from_node_id=100,
            to_node_id=200,
            portnum=70,
            portnum_name="TRACEROUTE_APP",
            raw_payload=_traceroute_payload(route=[110, 100, 110], snr=[4, 8, 12, 16]),
        )
        conn.commit()
        result = backfill_materializations(conn)
        assert result["traceroute_routes_rows"] == 1
        assert result["traceroute_hops_rows"] > 0
        route = cursor.execute("SELECT * FROM traceroute_routes").fetchone()
        assert route["mesh_packet_id"] == 777
        assert route["parse_status"] == "parsed"
        conn.commit()


class TestCaptureHook:
    def test_log_packet_writes_materializations(self, tmp_path, monkeypatch):
        from malla import mqtt_capture

        database = tmp_path / "capture.db"
        monkeypatch.setattr(mqtt_capture, "DATABASE_FILE", str(database))
        monkeypatch.setattr(
            mqtt_capture, "seed_query_planner_stats_async", lambda *_: None
        )
        mqtt_capture.init_database()

        mesh_packet = SimpleNamespace(
            **{"from": 123, "to": 0},
            id=900,
            decoded=SimpleNamespace(portnum=3, payload=_position_payload()),
            rx_rssi=-90,
            rx_snr=8.0,
            hop_limit=3,
            hop_start=3,
        )
        service_envelope = SimpleNamespace(gateway_id="!aabbccdd", channel_id="test")

        mqtt_capture.log_packet_to_database(
            topic="/test/topic",
            service_envelope=service_envelope,
            mesh_packet=mesh_packet,
        )

        with sqlite3.connect(database) as conn:
            conn.row_factory = sqlite3.Row
            obs = conn.execute(
                "SELECT * FROM packet_observations WHERE mesh_packet_id = 900"
            ).fetchone()
            assert obs is not None
            assert obs["is_direct"] == 1
            assert obs["gateway_id"] == "!aabbccdd"
            position = conn.execute(
                "SELECT * FROM node_positions WHERE node_id = 123"
            ).fetchone()
            assert position is not None
            assert position["latitude"] == pytest.approx(VALID_LAT)

        # Also test traceroute packet in capture hook
        traceroute_packet = SimpleNamespace(
            **{"from": 100, "to": 200},
            id=901,
            decoded=SimpleNamespace(portnum=70, payload=_traceroute_payload()),
            rx_rssi=-85,
            rx_snr=9.0,
            hop_limit=3,
            hop_start=3,
        )
        mqtt_capture.log_packet_to_database(
            topic="/test/traceroute",
            service_envelope=service_envelope,
            mesh_packet=traceroute_packet,
        )

        with sqlite3.connect(database) as conn:
            conn.row_factory = sqlite3.Row
            route = conn.execute(
                "SELECT * FROM traceroute_routes WHERE mesh_packet_id = 901"
            ).fetchone()
            assert route is not None
            assert route["parse_status"] == "parsed"
            hops = conn.execute(
                "SELECT * FROM traceroute_hops WHERE packet_id = ?",
                (route["packet_id"],),
            ).fetchall()
            assert len(hops) > 0


class TestCascadeDeletion:
    def test_delete_single_packet_cascades_derived_tables(self, conn):
        cursor = conn.cursor()
        obs_row = _capture(cursor, mesh_packet_id=101, snr=6.0)
        pos_row = _capture(
            cursor, mesh_packet_id=102, portnum=3, raw_payload=_position_payload()
        )
        tr_row = _capture(
            cursor,
            mesh_packet_id=103,
            portnum=70,
            portnum_name="TRACEROUTE_APP",
            raw_payload=_traceroute_payload(),
        )
        conn.commit()

        # Delete the observation packet
        cursor.execute("DELETE FROM packet_history WHERE id = ?", (obs_row["id"],))
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM packet_observations WHERE packet_id = ?",
                (obs_row["id"],),
            ).fetchone()[0]
            == 0
        )

        # Delete the position packet
        cursor.execute("DELETE FROM packet_history WHERE id = ?", (pos_row["id"],))
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM node_positions WHERE packet_id = ?",
                (pos_row["id"],),
            ).fetchone()[0]
            == 0
        )

        # Delete the traceroute packet
        cursor.execute("DELETE FROM packet_history WHERE id = ?", (tr_row["id"],))
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM traceroute_routes WHERE packet_id = ?",
                (tr_row["id"],),
            ).fetchone()[0]
            == 0
        )
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM traceroute_hops WHERE packet_id = ?",
                (tr_row["id"],),
            ).fetchone()[0]
            == 0
        )

    def test_delete_all_raw_packets_leaves_derived_tables_empty(self, conn):
        cursor = conn.cursor()
        _capture(cursor, mesh_packet_id=201)
        _capture(cursor, mesh_packet_id=202, portnum=3, raw_payload=_position_payload())
        _capture(
            cursor,
            mesh_packet_id=203,
            from_node_id=100,
            to_node_id=200,
            portnum=70,
            portnum_name="TRACEROUTE_APP",
            raw_payload=_traceroute_payload(route=[110, 100, 110], snr=[4, 8, 12, 16]),
        )
        conn.commit()

        assert (
            cursor.execute("SELECT COUNT(*) FROM packet_observations").fetchone()[0] > 0
        )
        assert cursor.execute("SELECT COUNT(*) FROM node_positions").fetchone()[0] > 0
        assert (
            cursor.execute("SELECT COUNT(*) FROM traceroute_routes").fetchone()[0] > 0
        )
        assert cursor.execute("SELECT COUNT(*) FROM traceroute_hops").fetchone()[0] > 0

        # Reproduction step: deleting all raw packets
        cursor.execute("DELETE FROM packet_history")
        conn.commit()

        assert (
            cursor.execute("SELECT COUNT(*) FROM packet_observations").fetchone()[0]
            == 0
        )
        assert cursor.execute("SELECT COUNT(*) FROM node_positions").fetchone()[0] == 0
        assert (
            cursor.execute("SELECT COUNT(*) FROM traceroute_routes").fetchone()[0] == 0
        )
        assert cursor.execute("SELECT COUNT(*) FROM traceroute_hops").fetchone()[0] == 0

    def test_cascade_works_even_when_foreign_keys_disabled(self, conn):
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = OFF")
        row = _capture(
            cursor, mesh_packet_id=301, portnum=3, raw_payload=_position_payload()
        )
        conn.commit()

        cursor.execute("DELETE FROM packet_history WHERE id = ?", (row["id"],))
        conn.commit()

        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM packet_observations WHERE packet_id = ?",
                (row["id"],),
            ).fetchone()[0]
            == 0
        )
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM node_positions WHERE packet_id = ?",
                (row["id"],),
            ).fetchone()[0]
            == 0
        )

    @pytest.mark.parametrize("foreign_keys", [True, False])
    def test_raw_insert_or_replace_removes_stale_observations_and_positions(
        self, conn, foreign_keys
    ):
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA foreign_keys = {'ON' if foreign_keys else 'OFF'}")
        row = _capture(
            cursor,
            mesh_packet_id=302,
            from_node_id=111,
            portnum=3,
            portnum_name="POSITION_APP",
            raw_payload=_position_payload(),
        )
        conn.commit()

        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM packet_observations WHERE packet_id = ?",
                (row["id"],),
            ).fetchone()[0]
            == 1
        )
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM node_positions WHERE packet_id = ?",
                (row["id"],),
            ).fetchone()[0]
            == 1
        )

        cursor.execute(
            """
            INSERT OR REPLACE INTO packet_history
                (id, timestamp, from_node_id, to_node_id, portnum, portnum_name)
            VALUES (?, 2000, 222, 333, 1, 'TEXT_MESSAGE_APP')
            """,
            (row["id"],),
        )
        conn.commit()

        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM packet_observations WHERE packet_id = ?",
                (row["id"],),
            ).fetchone()[0]
            == 0
        )
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM node_positions WHERE packet_id = ?",
                (row["id"],),
            ).fetchone()[0]
            == 0
        )


class TestMutationAndRepair:
    def test_updating_already_backfilled_traceroute_repaired_by_unified_backfill(
        self, conn
    ):
        cursor = conn.cursor()
        _insert_raw(cursor, mesh_packet_id=400)
        tr_row = _insert_raw(
            cursor,
            mesh_packet_id=401,
            from_node_id=10,
            to_node_id=20,
            portnum=70,
            portnum_name="TRACEROUTE_APP",
            raw_payload=_traceroute_payload(route=[10, 15, 20], snr=[5, 10]),
        )
        conn.commit()

        initial_res = backfill_materializations(conn)
        assert initial_res["complete"] is True
        assert initial_res["pending_packets"] == 0

        # Mutate the already-backfilled traceroute in packet_history
        new_payload = _traceroute_payload(route=[10, 12, 14, 20], snr=[3, 6, 9])
        cursor.execute(
            "UPDATE packet_history SET raw_payload = ? WHERE id = ?",
            (new_payload, tr_row["id"]),
        )
        conn.commit()

        # The update trigger should mark traceroute pending and clear observation
        audit = inspect_materializations(conn.cursor())
        assert audit["traceroutes"]["pending"] == 1
        assert audit["pending_packets"] == 1
        assert audit["complete"] is False

        # Unified backfill should detect and repair the pending traceroute below the watermark
        repair_res = backfill_materializations(conn)
        assert repair_res["processed_this_run"] == 1
        assert repair_res["complete"] is True
        assert repair_res["pending_packets"] == 0

        route = cursor.execute(
            "SELECT * FROM traceroute_routes WHERE packet_id = ?", (tr_row["id"],)
        ).fetchone()
        assert route["parse_status"] == "parsed"
        assert json.loads(route["route_nodes_json"]) == [10, 12, 14, 20]

    def test_updating_already_backfilled_observation_repaired_by_backfill(self, conn):
        cursor = conn.cursor()
        row = _insert_raw(cursor, mesh_packet_id=500, gateway_id="!orig_gw", snr=5.0)
        conn.commit()

        backfill_materializations(conn)
        assert inspect_materializations(conn.cursor())["complete"] is True

        # Mutate raw packet
        cursor.execute(
            "UPDATE packet_history SET gateway_id = '!new_gw', snr = 12.5 WHERE id = ?",
            (row["id"],),
        )
        conn.commit()

        audit = inspect_materializations(conn.cursor())
        assert audit["pending_packets"] == 1
        assert audit["complete"] is False

        # Run backfill to repair
        repair_res = backfill_materializations(conn)
        assert repair_res["processed_this_run"] == 1
        assert repair_res["complete"] is True

        obs = cursor.execute(
            "SELECT * FROM packet_observations WHERE packet_id = ?", (row["id"],)
        ).fetchone()
        assert obs["gateway_id"] == "!new_gw"
        assert obs["snr"] == 12.5

    def test_updating_already_backfilled_position_repaired_by_backfill(self, conn):
        cursor = conn.cursor()
        row = _insert_raw(
            cursor,
            mesh_packet_id=600,
            portnum=3,
            raw_payload=_position_payload(lat=52.0, lon=4.0),
        )
        conn.commit()

        backfill_materializations(conn)
        pos = cursor.execute(
            "SELECT * FROM node_positions WHERE packet_id = ?", (row["id"],)
        ).fetchone()
        assert pos["latitude"] == pytest.approx(52.0)

        # Mutate with new coordinates
        new_payload = _position_payload(lat=53.5, lon=5.5)
        cursor.execute(
            "UPDATE packet_history SET raw_payload = ? WHERE id = ?",
            (new_payload, row["id"]),
        )
        conn.commit()

        audit = inspect_materializations(conn.cursor())
        assert audit["pending_packets"] == 1
        repair_res = backfill_materializations(conn)
        assert repair_res["processed_this_run"] == 1

        pos_updated = cursor.execute(
            "SELECT * FROM node_positions WHERE packet_id = ?", (row["id"],)
        ).fetchone()
        assert pos_updated["latitude"] == pytest.approx(53.5)

        # Mutate position to invalid null-island fix: backfill should remove it
        null_island_payload = _position_payload(lat=0.0, lon=0.0)
        cursor.execute(
            "UPDATE packet_history SET raw_payload = ? WHERE id = ?",
            (null_island_payload, row["id"]),
        )
        conn.commit()

        backfill_materializations(conn)
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM node_positions WHERE packet_id = ?", (row["id"],)
            ).fetchone()[0]
            == 0
        )


class TestExtremaAndRepresentativesRecalculation:
    def test_deleting_reception_recalculates_signal_extrema(self, conn):
        cursor = conn.cursor()
        r1 = _capture(
            cursor, mesh_packet_id=701, from_node_id=999, gateway_id="!gw1", snr=4.0
        )
        _capture(
            cursor, mesh_packet_id=702, from_node_id=999, gateway_id="!gw1", snr=14.0
        )
        _capture(
            cursor, mesh_packet_id=703, from_node_id=999, gateway_id="!gw1", snr=9.0
        )
        conn.commit()

        # Query dynamic extrema over derived observations
        stats = cursor.execute(
            """
            SELECT MIN(snr), MAX(snr), COUNT(*)
            FROM packet_observations
            WHERE from_node_id = 999 AND gateway_id = '!gw1'
            """
        ).fetchone()
        assert stats[0] == 4.0
        assert stats[1] == 14.0
        assert stats[2] == 3

        # Delete the minimum SNR reception (r1)
        cursor.execute("DELETE FROM packet_history WHERE id = ?", (r1["id"],))
        conn.commit()

        stats_after_min_del = cursor.execute(
            """
            SELECT MIN(snr), MAX(snr), COUNT(*)
            FROM packet_observations
            WHERE from_node_id = 999 AND gateway_id = '!gw1'
            """
        ).fetchone()
        assert stats_after_min_del[0] == 9.0  # repaired new minimum!
        assert stats_after_min_del[1] == 14.0
        assert stats_after_min_del[2] == 2

    def test_deleting_reception_recalculates_deterministic_latest_position(self, conn):
        cursor = conn.cursor()
        _capture(
            cursor,
            mesh_packet_id=801,
            from_node_id=888,
            timestamp=1000.0,
            portnum=3,
            raw_payload=_position_payload(lat=51.0, lon=3.0),
        )
        r2 = _capture(
            cursor,
            mesh_packet_id=802,
            from_node_id=888,
            timestamp=2000.0,
            portnum=3,
            raw_payload=_position_payload(lat=52.0, lon=4.0),
        )
        conn.commit()

        latest = cursor.execute(
            """
            SELECT latitude, longitude FROM node_positions
            WHERE node_id = 888
            ORDER BY timestamp DESC, packet_id DESC
            LIMIT 1
            """
        ).fetchone()
        assert latest["latitude"] == pytest.approx(52.0)

        # Deleting the contributing newest reception repairs the latest position
        cursor.execute("DELETE FROM packet_history WHERE id = ?", (r2["id"],))
        conn.commit()

        latest_after_del = cursor.execute(
            """
            SELECT latitude, longitude FROM node_positions
            WHERE node_id = 888
            ORDER BY timestamp DESC, packet_id DESC
            LIMIT 1
            """
        ).fetchone()
        assert latest_after_del["latitude"] == pytest.approx(51.0)


class TestRetentionCleanup:
    def test_cleanup_old_data_removes_old_observations_and_positions(
        self, tmp_path, monkeypatch
    ):
        from malla import mqtt_capture

        database = tmp_path / "retention_test.db"
        monkeypatch.setattr(mqtt_capture, "DATABASE_FILE", str(database))
        monkeypatch.setattr(mqtt_capture, "DATA_RETENTION_HOURS", 24)
        monkeypatch.setattr(
            mqtt_capture, "seed_query_planner_stats_async", lambda *_: None
        )
        mqtt_capture.init_database()

        current_time = 1000000.0
        monkeypatch.setattr(mqtt_capture.time, "time", lambda: current_time)

        old_timestamp = current_time - (48 * 3600)  # 48h old (older than 24h)
        new_timestamp = current_time - (1 * 3600)  # 1h old (active)

        with sqlite3.connect(database) as conn:
            cursor = conn.cursor()
            # Old packet
            cursor.execute(
                """
                INSERT INTO packet_history (topic, timestamp, from_node_id, portnum, raw_payload)
                VALUES ('test/topic', ?, 111, 3, ?)
                """,
                (old_timestamp, _position_payload(lat=50.0, lon=4.0)),
            )
            old_id = cursor.lastrowid
            materialize_packet(
                cursor,
                {
                    "id": old_id,
                    "timestamp": old_timestamp,
                    "from_node_id": 111,
                    "portnum": 3,
                    "raw_payload": _position_payload(lat=50.0, lon=4.0),
                },
            )

            # New packet
            cursor.execute(
                """
                INSERT INTO packet_history (topic, timestamp, from_node_id, portnum, raw_payload)
                VALUES ('test/topic', ?, 222, 3, ?)
                """,
                (new_timestamp, _position_payload(lat=51.0, lon=5.0)),
            )
            new_id = cursor.lastrowid
            materialize_packet(
                cursor,
                {
                    "id": new_id,
                    "timestamp": new_timestamp,
                    "from_node_id": 222,
                    "portnum": 3,
                    "raw_payload": _position_payload(lat=51.0, lon=5.0),
                },
            )
            conn.commit()

        # Run retention cleanup
        mqtt_capture.cleanup_old_data()

        with sqlite3.connect(database) as conn:
            cursor = conn.cursor()
            # Old records must be deleted
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM packet_history WHERE id = ?", (old_id,)
                ).fetchone()[0]
                == 0
            )
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM packet_observations WHERE packet_id = ?",
                    (old_id,),
                ).fetchone()[0]
                == 0
            )
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM node_positions WHERE packet_id = ?", (old_id,)
                ).fetchone()[0]
                == 0
            )

            # New records must be preserved
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM packet_history WHERE id = ?", (new_id,)
                ).fetchone()[0]
                == 1
            )
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM packet_observations WHERE packet_id = ?",
                    (new_id,),
                ).fetchone()[0]
                == 1
            )
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM node_positions WHERE packet_id = ?", (new_id,)
                ).fetchone()[0]
                == 1
            )
