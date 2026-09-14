"""Tests pinning the packet_observations migration of the analytics queries.

Every packet query in analytics_service.py reads the lean
``packet_observations`` projection when it is populated and falls back to
``packet_history`` otherwise. Because row identity is 1:1 (one observation
per reception) the two reader paths must produce identical analytics
payloads; each scenario therefore seeds both variants of the same data and
compares the full ``_compute_analytics_data`` output.
"""

import sqlite3
import time
from contextlib import closing
from unittest.mock import patch

import pytest

import malla.services.analytics_service as analytics_module
from malla.database.materialization_schema import ensure_materialization_schema
from malla.database.materializations import (
    clear_materialization_cache,
    materialization_tx_scope,
    materialize_packet,
)
from malla.services.analytics_service import AnalyticsService

pytestmark = pytest.mark.unit

GW_HEX = "!aabbccdd"
HOUR = 3600
DAY = 86400
GARBAGE_RSSI = -1386841926
GARBAGE_SNR = 1e9

NODE_A = int("11111111", 16)
NODE_B = int("22222222", 16)
NODE_C = int("33333333", 16)
GW_NODE_ID = int("aabbccdd", 16)


@pytest.fixture(autouse=True)
def _clear_caches():
    clear_materialization_cache()
    analytics_module._source_decision = None
    yield
    clear_materialization_cache()
    analytics_module._source_decision = None


def _make_database(path, materialized):
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
        if materialized:
            ensure_materialization_schema(conn.cursor())
        conn.commit()


def _connection(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _insert_packet(conn, **fields):
    values = {
        "timestamp": time.time(),
        "mesh_packet_id": None,
        "from_node_id": NODE_A,
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


def _seed(conn, now):
    """Seed one controlled dataset (12 rows; 10 inside the 24h window).

    Timestamps derive from a single caller-provided *now* so two databases
    seeded in the same test hold bit-identical values.
    """
    for node_id, name, short in (
        (NODE_A, "Node Alpha", "ALPHA"),
        (NODE_B, "Node Bravo", "BRAVO"),
        (NODE_C, "Node Charlie", "CHARL"),
        (GW_NODE_ID, "Gateway Node", "GWN"),
    ):
        conn.execute(
            "INSERT INTO node_info VALUES (?, ?, ?, ?, ?, ?, ?)",
            (node_id, name, short, "TBEAM", "CLIENT", None, now - 50000),
        )
    conn.commit()

    # Distinct 24h packet counts per node (5 / 4 / 1) keep the top-nodes
    # ordering deterministic for the payload comparison.
    _insert_packet(conn, timestamp=now - 1 * HOUR, from_node_id=NODE_A,
                   gateway_id=GW_HEX, rssi=-65, snr=12.0,
                   hop_start=3, hop_limit=3, payload_length=32)
    _insert_packet(conn, timestamp=now - 2 * HOUR, from_node_id=NODE_A,
                   gateway_id=GW_HEX, rssi=-85, snr=4.0,
                   hop_start=3, hop_limit=2, portnum=3,
                   portnum_name="POSITION_APP", payload_length=16)
    _insert_packet(conn, timestamp=now - 3 * HOUR, from_node_id=NODE_A,
                   gateway_id=None, rssi=-95, snr=-2.0,
                   hop_start=3, hop_limit=1, portnum=4,
                   portnum_name="NODEINFO_APP", payload_length=8)
    _insert_packet(conn, timestamp=now - 4 * HOUR, from_node_id=NODE_A,
                   gateway_id=GW_HEX, rssi=0, snr=0.0,
                   hop_start=3, hop_limit=3, portnum=64,
                   portnum_name="TELEMETRY_APP", payload_length=0)
    _insert_packet(conn, timestamp=now - 5 * HOUR, from_node_id=NODE_A,
                   gateway_id=GW_HEX, rssi=GARBAGE_RSSI, snr=GARBAGE_SNR,
                   hop_start=3, hop_limit=3, portnum=70,
                   portnum_name="ROUTING_APP", payload_length=4,
                   processed_successfully=0)

    # Outside 24h but inside 7d: feeds nodes_seen_7d only. One more far
    # outside the 7d window must vanish entirely.
    _insert_packet(conn, timestamp=now - 30 * HOUR, from_node_id=NODE_B,
                   gateway_id=GW_HEX, rssi=-75, snr=8.5,
                   hop_start=3, hop_limit=3)
    _insert_packet(conn, timestamp=now - 8 * DAY, from_node_id=NODE_B,
                   gateway_id=GW_HEX, rssi=-75, snr=8.5,
                   hop_start=3, hop_limit=3)
    _insert_packet(conn, timestamp=now - 6 * HOUR, from_node_id=NODE_B,
                   gateway_id=GW_HEX, rssi=-75, snr=8.5,
                   hop_start=3, hop_limit=3)

    _insert_packet(conn, timestamp=now - 7 * HOUR, from_node_id=NODE_C,
                   gateway_id=GW_HEX, rssi=-75, snr=8.5,
                   hop_start=3, hop_limit=3)
    # NULL portnum / unknown hops / NULL gateway corner cases.
    _insert_packet(conn, timestamp=now - 8 * HOUR, from_node_id=NODE_C,
                   gateway_id=None, portnum=None, portnum_name=None,
                   payload_length=None)
    _insert_packet(conn, timestamp=now - 9 * HOUR, from_node_id=NODE_C,
                   gateway_id=GW_HEX, rssi=-75, snr=8.5,
                   hop_start=5, hop_limit=2)
    _insert_packet(conn, timestamp=now - 10 * HOUR, from_node_id=NODE_C,
                   gateway_id=GW_HEX, rssi=-75, snr=8.5,
                   hop_start=173, hop_limit=0)


def _payload(path, materialized, now=None, **filters):
    """Create *path*, seed it and compute the full analytics payload on it.

    Returns ``(payload, selected_source)``. The source decision cache is
    reset before deciding so each database selects (and reports) its own
    reader path instead of reusing a decision latched by a previous
    database in the same test.
    """
    if now is None:
        now = time.time()
    _make_database(path, materialized=materialized)
    with closing(_connection(path)) as conn:
        _seed(conn, now)
        analytics_module._source_decision = None
        source = analytics_module._packet_source(conn.cursor())
        analytics_module._source_decision = None

    with (
        patch(
            "malla.database.connection.get_db_connection",
            side_effect=lambda: _connection(path),
        ),
        patch(
            "malla.database.repositories.get_db_connection",
            side_effect=lambda: _connection(path),
        ),
    ):
        return AnalyticsService._compute_analytics_data(**filters), source


def test_analytics_payload_identical_across_reader_paths(tmp_path):
    now = time.time()
    legacy, legacy_source = _payload(
        tmp_path / "analytics-legacy.db", materialized=False, now=now
    )
    materialized, materialized_source = _payload(
        tmp_path / "analytics-materialized.db", materialized=True, now=now
    )

    # Each database must have exercised its intended reader path, otherwise
    # the equality below would be a raw-table comparison with itself.
    assert legacy_source == "packet_history"
    assert materialized_source == "packet_observations"
    assert legacy == materialized
    # Sanity: the fixture actually exercised every panel (not two empty
    # payloads trivially equal).
    assert legacy["packet_statistics"]["total_packets"] == 10
    assert legacy["packet_statistics"]["failed_packets"] == 1
    assert legacy["node_statistics"]["nodes_seen_7d"] == 3
    assert legacy["node_statistics"]["active_nodes"] == 3
    gateways = {row["gateway_id"]: row for row in legacy["gateway_distribution"]}
    # NULL gateway (legacy) and '' gateway (projection) both group as Unknown.
    assert gateways["Unknown"]["total_packets"] == 2
    hops = {row["hops"]: row["count"] for row in legacy["hop_distribution"]}
    assert hops == {0: 5, 1: 1, 2: 1, 3: 1}  # unknown + corrupt hops excluded
    assert len(legacy["top_nodes"]) == 3
    assert legacy["top_nodes"][0]["node_id"] == NODE_A


def test_materialized_database_really_used(tmp_path):
    """Guard the equivalence test itself: the projection path must be taken."""
    path = tmp_path / "analytics-probe.db"
    _make_database(path, materialized=True)

    with closing(_connection(path)) as conn:
        # Observation row present but packet_history empty: identical counts
        # prove the queries read packet_observations, not the raw table.
        conn.execute("""
            INSERT INTO packet_observations (
                packet_id, tx_id, timestamp, from_node_id, gateway_id,
                portnum_name, rssi, snr, hop_start, hop_limit,
                processed_successfully
            ) VALUES (1, 1, ?, ?, ?, 'TEXT_MESSAGE_APP', -80, 5.0, 3, 3, 1)
        """, (time.time(), NODE_A, GW_HEX))
        conn.commit()

    with patch(
        "malla.database.connection.get_db_connection",
        side_effect=lambda: _connection(path),
    ):
        stats = AnalyticsService._get_packet_statistics({}, time.time() - HOUR)

    assert stats["total_packets"] == 1


def test_partially_backfilled_database_serves_from_projection(tmp_path):
    """A partial projection serves analytics reads with no coverage check.

    Pins the intended trade-off: rows the projection lacks are
    pre-projection history, and analytics must not fall back to the slow
    raw scans (nor hide the projection's rows) just because coverage is
    incomplete — the single writer materializes from now on and the gap
    ages out of retention.
    """
    path = tmp_path / "analytics-partial.db"
    _make_database(path, materialized=True)

    with closing(_connection(path)) as conn:
        for i in range(20):
            _insert_packet(conn, timestamp=time.time() - i, from_node_id=NODE_A)
        # Drop the projection back to a near-empty partial state.
        conn.execute("DELETE FROM packet_observations WHERE packet_id > 4")
        conn.commit()

        source = analytics_module._packet_source(conn.cursor())

    assert source == "packet_observations"


def test_source_decision_is_cached_between_checks(tmp_path):
    """The source decision is not recomputed on every query."""
    path = tmp_path / "analytics-decision.db"
    _make_database(path, materialized=True)
    with closing(_connection(path)) as conn:
        _insert_packet(conn, timestamp=time.time(), from_node_id=NODE_A)

        assert analytics_module._packet_source(conn.cursor()) == "packet_observations"
        # Backfill regresses (rows vanish): the cached decision still
        # serves observations until the next periodic recheck.
        conn.execute("DELETE FROM packet_observations")
        conn.commit()
        assert analytics_module._packet_source(conn.cursor()) == "packet_observations"

        # Expire the cache: the recheck now falls back.
        analytics_module._source_decision = (0.0, "packet_observations")
        assert analytics_module._packet_source(conn.cursor()) == "packet_history"


def test_filtered_payload_identical_across_reader_paths(tmp_path):
    now = time.time()
    legacy, legacy_source = _payload(
        tmp_path / "analytics-filtered-legacy.db",
        materialized=False,
        now=now,
        gateway_id=GW_HEX,
        from_node=NODE_A,
        hop_count=0,
    )
    materialized, materialized_source = _payload(
        tmp_path / "analytics-filtered-materialized.db",
        materialized=True,
        now=now,
        gateway_id=GW_HEX,
        from_node=NODE_A,
        hop_count=0,
    )

    assert legacy_source == "packet_history"
    assert materialized_source == "packet_observations"
    assert legacy == materialized
    # Gateway filter drops the no-gateway row, the node filter keeps only
    # NODE_A's rows, hop_count=0 keeps only direct receptions.
    assert legacy["packet_statistics"]["total_packets"] == 3
