"""Tests for the MQTT-ingest low-information packet filters.

Pins the database-growth guards in ``mqtt_capture``:

- undecryptable traffic (``UNKNOWN_APP`` / ``PRIVATE_APP``) is dropped
  before persistence — after a decryption attempt — when
  ``capture_drop_undecryptable`` is enabled;
- ``MAP_REPORT_APP`` packets are throttled to one stored row per node per
  configured interval;
- successfully decoded packets of informative types are always stored.
"""

from types import SimpleNamespace

import pytest
from meshtastic import mesh_pb2, mqtt_pb2, portnums_pb2

from src.malla import mqtt_capture

pytestmark = pytest.mark.unit


def build_message(portnum: int, payload: bytes = b"x", encrypted: bool = False) -> SimpleNamespace:
    """Create a minimal MQTT message with a decoded packet of **portnum**."""
    mesh_packet = mesh_pb2.MeshPacket()
    setattr(mesh_packet, "from", 0x7F6E5D4C)
    mesh_packet.to = 0
    mesh_packet.decoded.portnum = portnum
    mesh_packet.decoded.payload = payload
    if encrypted:
        mesh_packet.encrypted = b"\x01\x02\x03\x04"

    service_envelope = mqtt_pb2.ServiceEnvelope()
    service_envelope.channel_id = "LongFast"
    service_envelope.gateway_id = "!a2e96b40"
    service_envelope.packet.CopyFrom(mesh_packet)

    return SimpleNamespace(
        topic="msh/TW/2/e/LongFast/!a2e96b40",
        payload=service_envelope.SerializeToString(),
    )


@pytest.fixture(autouse=True)
def _clean_filter_state():
    mqtt_capture.reset_ingest_filter_state()
    yield
    mqtt_capture.reset_ingest_filter_state()


@pytest.fixture
def stored(monkeypatch):
    """Record calls to log_packet_to_database instead of hitting the DB."""
    calls: list[tuple] = []
    monkeypatch.setattr(mqtt_capture, "log_packet_to_database", lambda *a: calls.append(a))
    return calls


class TestUndecryptableFilter:
    def test_unknown_app_is_dropped(self, stored):
        msg = build_message(portnums_pb2.PortNum.UNKNOWN_APP, encrypted=True)
        mqtt_capture.on_message(None, None, msg)
        assert stored == []
        assert mqtt_capture._dropped_undecryptable_count == 1

    def test_private_app_is_dropped(self, stored):
        msg = build_message(portnums_pb2.PortNum.PRIVATE_APP, encrypted=True)
        mqtt_capture.on_message(None, None, msg)
        assert stored == []
        assert mqtt_capture._dropped_undecryptable_count == 1

    def test_flag_disabled_stores_unknown_app(self, stored, monkeypatch):
        monkeypatch.setattr(mqtt_capture, "DROP_UNDECRYPTABLE_PACKETS", False)
        msg = build_message(portnums_pb2.PortNum.UNKNOWN_APP, encrypted=True)
        mqtt_capture.on_message(None, None, msg)
        assert len(stored) == 1

    def test_text_message_is_stored(self, stored):
        msg = build_message(portnums_pb2.PortNum.TEXT_MESSAGE_APP, b"hello")
        mqtt_capture.on_message(None, None, msg)
        assert len(stored) == 1
        assert mqtt_capture._dropped_undecryptable_count == 0

    def test_empty_malformed_envelope_is_dropped(self, stored):
        # An empty ServiceEnvelope parses to a default MeshPacket whose
        # portnum is UNKNOWN_APP (protobuf default 0); all UNKNOWN_APP
        # traffic is dropped, including empty/malformed envelopes.
        mqtt_capture.on_message(
            None, None, SimpleNamespace(topic="msh/TW/2/e/LongFast/!abc", payload=b"")
        )
        assert stored == []
        assert mqtt_capture._dropped_undecryptable_count == 1


class TestMapReportThrottle:
    def test_duplicate_within_interval_is_dropped(self, stored, monkeypatch):
        monkeypatch.setattr(mqtt_capture, "MAP_REPORT_MIN_INTERVAL_SECONDS", 3600)
        mqtt_capture.on_message(None, None, build_message(portnums_pb2.PortNum.MAP_REPORT_APP))
        mqtt_capture.on_message(None, None, build_message(portnums_pb2.PortNum.MAP_REPORT_APP))
        assert len(stored) == 1
        assert mqtt_capture._dropped_map_report_count == 1

    def test_interval_zero_stores_all(self, stored, monkeypatch):
        monkeypatch.setattr(mqtt_capture, "MAP_REPORT_MIN_INTERVAL_SECONDS", 0)
        mqtt_capture.on_message(None, None, build_message(portnums_pb2.PortNum.MAP_REPORT_APP))
        mqtt_capture.on_message(None, None, build_message(portnums_pb2.PortNum.MAP_REPORT_APP))
        assert len(stored) == 2

    def test_other_node_not_throttled(self, stored, monkeypatch):
        monkeypatch.setattr(mqtt_capture, "MAP_REPORT_MIN_INTERVAL_SECONDS", 3600)
        first = build_message(portnums_pb2.PortNum.MAP_REPORT_APP)
        second = build_message(portnums_pb2.PortNum.MAP_REPORT_APP)
        mqtt_capture.on_message(None, None, first)
        # Different sender: must be stored despite the throttle window.
        envelope = mqtt_pb2.ServiceEnvelope()
        envelope.ParseFromString(second.payload)
        setattr(envelope.packet, "from", 0x11223344)
        second.payload = envelope.SerializeToString()
        mqtt_capture.on_message(None, None, second)
        assert len(stored) == 2


class TestShouldStorePacketDirect:
    def test_none_packet_is_stored(self):
        assert mqtt_capture._should_store_packet(None) is True

    def test_packet_without_decoded_field_is_stored(self):
        assert mqtt_capture._should_store_packet(SimpleNamespace()) is True

    def test_reset_clears_state(self, monkeypatch):
        monkeypatch.setattr(mqtt_capture, "MAP_REPORT_MIN_INTERVAL_SECONDS", 3600)
        mqtt_capture.on_message(None, None, build_message(portnums_pb2.PortNum.MAP_REPORT_APP))
        assert mqtt_capture._last_map_report_ts
        assert mqtt_capture._dropped_map_report_count == 0
        mqtt_capture.reset_ingest_filter_state()
        assert not mqtt_capture._last_map_report_ts
