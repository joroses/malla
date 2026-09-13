"""Live capture must preserve MeshPacket fields into packet_observations.

Pins the preservation guarantee of MATERIALIZATION_ARCHITECTURE.md §3.1:
every field documented as preserved (e.g. ``relay_node`` for grouping and
repeater analysis) must reach the materialized observation row, not just
the raw ``packet_history`` record.
"""

import sqlite3
from contextlib import closing

import pytest
from meshtastic import mesh_pb2, mqtt_pb2, portnums_pb2

from malla import mqtt_capture

pytestmark = pytest.mark.unit


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "history.db"
    monkeypatch.setattr(mqtt_capture, "DATABASE_FILE", str(path))
    monkeypatch.setattr(
        mqtt_capture, "seed_query_planner_stats_async", lambda *_: False
    )
    mqtt_capture.init_database()
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn


def capture_text_message(relay_node=None):
    packet = mesh_pb2.MeshPacket(id=42, to=0, hop_start=3, hop_limit=3)
    setattr(packet, "from", 100)
    if relay_node is not None:
        packet.relay_node = relay_node
    packet.decoded.portnum = portnums_pb2.PortNum.TEXT_MESSAGE_APP
    packet.decoded.payload = b"hello"
    envelope = mqtt_pb2.ServiceEnvelope(gateway_id="!12345678", channel_id="LongFast")
    mqtt_capture.log_packet_to_database("msh/test/e/LongFast", envelope, packet)


def test_relay_node_preserved_in_observation(database):
    capture_text_message(relay_node=171)

    history = database.execute(
        "SELECT relay_node FROM packet_history"
    ).fetchone()
    assert history["relay_node"] == 171

    observation = database.execute(
        "SELECT relay_node FROM packet_observations"
    ).fetchone()
    assert observation is not None
    assert observation["relay_node"] == 171


def test_unset_relay_node_stays_zero_in_observation(database):
    capture_text_message()

    observation = database.execute(
        "SELECT relay_node FROM packet_observations"
    ).fetchone()
    assert observation is not None
    assert observation["relay_node"] == 0
