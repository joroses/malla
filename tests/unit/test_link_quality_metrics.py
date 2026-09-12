"""Unified link-quality metrics tests across every RF link consumer.

Feature 3 extends the Feature 2 directional tests: all consumers (network
graph, map traceroute links, map packet links, link-analysis endpoint)
must expose the same quality/reliability/balance metrics from the shared
enrichment helper, with strength redefined as pure observation volume
(thickness = "how much data", color = quality). The key preset
regression is -7.5 dB: good on LongFast (SF11, +10 dB margin) but
marginal on SFNarrow (SF7, zero margin).
"""

import sqlite3
import time
from contextlib import closing
from unittest.mock import patch

import pytest
from flask import Flask

from malla.config import _clear_config_cache
from malla.database.traceroute_read_repository import (
    get_traceroute_hops_for_graph,
    get_traceroute_link,
)
from malla.routes.api_routes import register_api_routes
from malla.services.location_service import _PACKET_LINKS_CACHE, LocationService
from malla.services.traceroute_service import _NETWORK_GRAPH_CACHE, TracerouteService
from malla.utils.link_quality import (
    ENRICHMENT_FIELDS,
    enrich_link_quality,
    observation_strength,
)
from malla.utils.signal_quality import (
    QUALITY_COLORS,
    QUALITY_FAIR,
    QUALITY_GOOD,
    QUALITY_MARGINAL,
    QUALITY_UNKNOWN,
)

pytestmark = pytest.mark.unit

SF11_RELIABILITY_PLUS_10_DB = 98.9  # 100 / (1 + e^-4.5)
SF7_RELIABILITY_ZERO_MARGIN = 18.2  # 100 / (1 + e^1.5)


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    """Isolate the config singleton from local config.yaml and shell env."""
    monkeypatch.setenv("MALLA_CONFIG_FILE", str(tmp_path / "does-not-exist.yaml"))
    for var in ("MALLA_LORA_PRESET", "MALLA_LORA_SPREADING_FACTOR"):
        monkeypatch.delenv(var, raising=False)
    _clear_config_cache()
    yield
    _clear_config_cache()


class _NonClosingConnection:
    def __init__(self, connection):
        self.connection = connection

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def close(self):
        pass


def _hop(
    packet_id,
    hop_index,
    from_node,
    to_node,
    snr,
    timestamp,
    direction="forward",
    channel_id=None,
):
    return {
        "packet_id": packet_id,
        "direction": direction,
        "hop_index": hop_index,
        "timestamp": timestamp,
        "from_node_id": from_node,
        "to_node_id": to_node,
        "snr": snr,
        "channel_id": channel_id,
    }


class TestObservationStrength:
    """strength = clamp(1.5 + 2.5·log10(count), 1.5, 8.0), SNR-independent."""

    @pytest.mark.parametrize(
        ("count", "expected"),
        [(1, 1.5), (10, 4.0), (100, 6.5)],
    )
    def test_expected_widths(self, count, expected):
        assert observation_strength(count) == expected

    @pytest.mark.parametrize("count", [0, -5, None])
    def test_zero_counts_guarded_before_logarithm(self, count):
        assert observation_strength(count) == 1.5

    def test_clamped_at_eight(self):
        assert observation_strength(10_000) == 8.0



class TestEnrichLinkQuality:
    """The shared helper computes the canonical per-link payload."""

    def test_key_preset_regression_minus_7_5_db(self):
        # Same SNR: +10 dB margin/good on LongFast; zero margin/marginal on
        # SFNarrow.
        longfast = enrich_link_quality(
            channel_id="LongFast", forward_avg_snr=-7.5, forward_observations=1
        )
        narrow = enrich_link_quality(
            channel_id="SFNarrow", forward_avg_snr=-7.5, forward_observations=1
        )
        assert longfast["spreading_factor"] == 11
        assert longfast["forward_quality"] == QUALITY_GOOD
        assert longfast["quality"] == QUALITY_GOOD
        assert longfast["forward_estimated_reliability"] == pytest.approx(
            SF11_RELIABILITY_PLUS_10_DB, abs=0.05
        )
        assert narrow["spreading_factor"] == 7
        assert narrow["forward_quality"] == QUALITY_MARGINAL
        assert narrow["quality"] == QUALITY_MARGINAL
        assert narrow["forward_estimated_reliability"] == pytest.approx(
            SF7_RELIABILITY_ZERO_MARGIN, abs=0.05
        )

    def test_overall_quality_describes_worst_observed_direction(self):
        result = enrich_link_quality(
            forward_avg_snr=-5.0,
            return_avg_snr=-25.0,
            forward_observations=2,
            return_observations=2,
        )
        assert result["forward_quality"] == QUALITY_GOOD
        assert result["return_quality"] == QUALITY_MARGINAL
        assert result["quality"] == QUALITY_MARGINAL
        assert result["worst_snr"] == -25.0

    def test_single_direction_overall_describes_observed_direction(self):
        result = enrich_link_quality(
            forward_avg_snr=-7.5, forward_observations=4, return_observations=0
        )
        assert result["quality"] == QUALITY_GOOD
        assert result["return_quality"] == QUALITY_UNKNOWN
        assert result["return_estimated_reliability"] is None
        assert result["is_bidirectional"] is False
        assert result["link_balance"] == "unidirectional"

    def test_no_valid_measurements_are_unknown_not_zero(self):
        result = enrich_link_quality(forward_observations=3, return_observations=1)
        assert result["quality"] == QUALITY_UNKNOWN
        assert result["worst_snr"] is None
        assert result["estimated_reliability"] is None
        assert result["link_balance"] == "unknown"

    def test_worst_snr_filters_implausible_averages(self):
        result = enrich_link_quality(
            forward_avg_snr=1e9, return_avg_snr=-9.0, forward_observations=1
        )
        assert result["worst_snr"] == -9.0
        assert result["forward_quality"] == QUALITY_UNKNOWN

    def test_estimated_reliability_is_worst_direction(self):
        result = enrich_link_quality(
            forward_avg_snr=-5.0,
            return_avg_snr=-15.0,
            forward_observations=1,
            return_observations=1,
        )
        assert result["estimated_reliability"] == min(
            result["forward_estimated_reliability"],
            result["return_estimated_reliability"],
        )

    def test_bidirectional_and_balance_from_observation_counts(self):
        result = enrich_link_quality(
            forward_avg_snr=-5.0,
            return_avg_snr=-6.0,
            forward_observations=3,
            return_observations=2,
        )
        assert result["is_bidirectional"] is True
        assert result["link_balance"] == "balanced"
        assert result["observation_count"] == 5
        assert result["strength"] == observation_strength(5)

    def test_unrecognized_channel_falls_back_to_configured_preset(
        self, monkeypatch
    ):
        monkeypatch.setenv("MALLA_LORA_PRESET", "ShortFast")  # SF7 floor -7.5
        result = enrich_link_quality(
            channel_id="my-private-channel", forward_avg_snr=-7.5
        )
        assert result["spreading_factor"] == 7
        assert result["quality"] == QUALITY_MARGINAL

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_channel_is_no_hint(self, blank):
        result = enrich_link_quality(channel_id=blank, forward_avg_snr=-7.5)
        assert result["channel_id"] is None
        assert result["spreading_factor"] == 11  # default LongFast

    def test_colors_come_from_the_shared_palette(self):
        result = enrich_link_quality(
            channel_id="LongFast",
            forward_avg_snr=-7.5,
            return_avg_snr=-13.5,
            forward_observations=1,
            return_observations=1,
        )
        assert result["forward_color"] == QUALITY_COLORS[QUALITY_GOOD]
        assert result["return_color"] == QUALITY_COLORS[QUALITY_FAIR]
        assert result["quality_color"] == QUALITY_COLORS[QUALITY_FAIR]


class TestGraphLinkMetrics:
    """TracerouteService.get_network_graph_data enriches every direct link."""

    def _graph(self, hops, *, include_indirect=False):
        now = time.time()
        _NETWORK_GRAPH_CACHE.clear()
        try:
            with (
                patch(
                    "malla.services.traceroute_service.get_traceroute_hops_for_graph",
                    return_value=hops,
                ),
                patch(
                    "malla.services.traceroute_service.get_bulk_node_names",
                    return_value={},
                ),
                patch(
                    "malla.services.traceroute_service.LocationRepository.get_node_locations",
                    return_value=[],
                ),
            ):
                return TracerouteService.get_network_graph_data(
                    hours=24,
                    min_snr=-200.0,
                    include_indirect=include_indirect,
                    filters={"start_time": now - 60, "end_time": now + 60},
                )
        finally:
            _NETWORK_GRAPH_CACHE.clear()

    def test_new_link_fields_with_configured_preset_default(self):
        now = time.time()
        graph = self._graph([_hop(1, 0, 100, 200, -7.5, now)])
        link = graph["links"][0]

        assert link["channel_id"] is None
        assert link["spreading_factor"] == 11  # default LongFast
        assert link["forward_quality"] == QUALITY_GOOD
        assert link["quality"] == QUALITY_GOOD
        assert link["worst_snr"] == -7.5
        assert link["forward_estimated_reliability"] == pytest.approx(
            SF11_RELIABILITY_PLUS_10_DB, abs=0.05
        )
        assert link["return_quality"] == QUALITY_UNKNOWN
        assert link["return_estimated_reliability"] is None
        assert link["estimated_reliability"] == pytest.approx(
            SF11_RELIABILITY_PLUS_10_DB, abs=0.05
        )
        assert link["link_balance"] == "unidirectional"
        assert link["is_bidirectional"] is False
        assert link["observation_count"] == 1
        assert link["packet_count"] == 1  # observation alias kept
        assert link["strength"] == 1.5
        assert link["forward_color"] == QUALITY_COLORS[QUALITY_GOOD]
        assert link["quality_color"] == QUALITY_COLORS[QUALITY_GOOD]

    def test_per_link_preset_resolution_same_snr_different_tiers(self):
        now = time.time()
        graph = self._graph(
            [
                _hop(1, 0, 100, 200, -7.5, now, channel_id="LongFast"),
                _hop(2, 0, 300, 400, -7.5, now, channel_id="SFNarrow"),
            ]
        )
        by_key = {(link["source"], link["target"]): link for link in graph["links"]}

        longfast = by_key[(100, 200)]
        narrow = by_key[(300, 400)]
        assert longfast["spreading_factor"] == 11
        assert longfast["quality"] == QUALITY_GOOD
        assert longfast["quality_color"] == QUALITY_COLORS[QUALITY_GOOD]
        assert narrow["spreading_factor"] == 7
        assert narrow["quality"] == QUALITY_MARGINAL
        assert narrow["quality_color"] == QUALITY_COLORS[QUALITY_MARGINAL]

    def test_most_recent_channel_wins_for_link_preset(self):
        now = time.time()
        graph = self._graph(
            [
                _hop(1, 0, 100, 200, -7.5, now - 100, channel_id="LongFast"),
                _hop(2, 0, 100, 200, -7.5, now, channel_id="SFNarrow"),
            ]
        )
        assert graph["links"][0]["channel_id"] == "SFNarrow"
        assert graph["links"][0]["spreading_factor"] == 7

    def test_channel_tie_broken_by_packet_id(self):
        now = time.time()
        graph = self._graph(
            [
                _hop(9, 0, 100, 200, -7.5, now, channel_id="SFNarrow"),
                _hop(5, 0, 100, 200, -7.5, now, channel_id="LongFast"),
            ]
        )
        # Same timestamp: the higher packet id is the more recent hint.
        assert graph["links"][0]["channel_id"] == "SFNarrow"
        assert graph["links"][0]["spreading_factor"] == 7

    def test_width_is_observation_volume_not_snr(self):
        now = time.time()
        graph = self._graph(
            [
                _hop(1, 0, 100, 200, -5.0, now),
                _hop(2, 0, 100, 200, -5.0, now),
                _hop(3, 0, 300, 400, -25.0, now),
                _hop(4, 0, 300, 400, -25.0, now),
            ]
        )
        by_key = {(link["source"], link["target"]): link for link in graph["links"]}

        strong, weak = by_key[(100, 200)], by_key[(300, 400)]
        assert strong["strength"] == weak["strength"] == observation_strength(2)
        assert strong["quality"] == QUALITY_GOOD
        assert weak["quality"] == QUALITY_MARGINAL
        assert strong["quality_color"] != weak["quality_color"]

    def test_unknown_only_observation_reports_unknown_metrics(self):
        now = time.time()
        graph = self._graph([_hop(1, 0, 100, 200, -32.0, now)])
        link = graph["links"][0]

        assert link["observation_count"] == 1
        assert link["worst_snr"] is None
        assert link["quality"] == QUALITY_UNKNOWN
        assert link["estimated_reliability"] is None
        assert link["strength"] == 1.5

    def test_indirect_connections_stay_unknown_and_distinct(self):
        now = time.time()
        graph = self._graph(
            [
                _hop(1, 0, 100, 200, -5.0, now),
                _hop(1, 1, 200, 300, -5.0, now),
            ],
            include_indirect=True,
        )
        assert len(graph["indirect_connections"]) == 1
        conn = graph["indirect_connections"][0]

        assert conn["is_inferred"] is True
        assert conn["quality"] == QUALITY_UNKNOWN
        assert conn["quality_color"] == QUALITY_COLORS[QUALITY_UNKNOWN]
        assert conn["worst_snr"] is None
        assert conn["estimated_reliability"] is None
        assert conn["link_balance"] == "unknown"
        assert conn["is_bidirectional"] is None
        assert conn["observation_count"] == 2  # hops × paths
        assert conn["strength"] == observation_strength(2)
        # The path SNR average stays available as a legacy field but is not
        # presented as a measured direct-link quality.
        assert conn["avg_snr"] == -5.0

    def test_indirect_mixed_path_lengths_count_actual_hops(self):
        now = time.time()
        graph = self._graph(
            [
                # Packet 1: three-hop path 100 → 201 → 202 → 300
                _hop(1, 0, 100, 201, -5.0, now),
                _hop(1, 1, 201, 202, -5.0, now),
                _hop(1, 2, 202, 300, -5.0, now),
                # Packet 2: two-hop path 100 → 203 → 300
                _hop(2, 0, 100, 203, -6.0, now),
                _hop(2, 1, 203, 300, -6.0, now),
            ],
            include_indirect=True,
        )
        assert len(graph["indirect_connections"]) == 1
        conn = graph["indirect_connections"][0]

        # Actual observations: 3 + 2 = 5 hops, not first_hop_count ×
        # path_count = 6.
        assert conn["path_count"] == 2
        assert conn["observation_count"] == 5
        assert conn["strength"] == observation_strength(5)
        # Reported hop count is the average hops per path (2.5 → 2 under
        # Python's banker's rounding).
        assert conn["hop_count"] == 2

    def test_rf_configuration_is_part_of_cache_identity(self, monkeypatch):
        now = time.time()
        hops = [_hop(1, 0, 100, 200, -7.5, now)]

        def build():
            with (
                patch(
                    "malla.services.traceroute_service.get_traceroute_hops_for_graph",
                    return_value=hops,
                ),
                patch(
                    "malla.services.traceroute_service.get_bulk_node_names",
                    return_value={},
                ),
                patch(
                    "malla.services.traceroute_service.LocationRepository.get_node_locations",
                    return_value=[],
                ),
            ):
                return TracerouteService.get_network_graph_data(
                    hours=24,
                    min_snr=-200.0,
                    include_indirect=False,
                    filters={"start_time": now - 60, "end_time": now + 60},
                )

        try:
            default = build()
            assert default["links"][0]["quality"] == QUALITY_GOOD

            # Change the preset and rebuild WITHOUT clearing the graph cache:
            # the new config must land in a different cache entry.
            monkeypatch.setenv("MALLA_LORA_PRESET", "SFNarrow")
            _clear_config_cache()
            reconfigured = build()
        finally:
            _NETWORK_GRAPH_CACHE.clear()

        assert reconfigured["links"][0]["quality"] == QUALITY_MARGINAL
        assert reconfigured["links"][0]["spreading_factor"] == 7


class TestTracerouteLinksPassThrough:
    """Map traceroute links reuse the graph's enriched fields verbatim."""

    def test_enriched_fields_pass_through_without_recalculation(self):
        network_data = {
            "links": [
                {
                    "source": 10,
                    "target": 20,
                    "packet_count": 5,
                    "last_seen": 1000.0,
                    "avg_snr": -10.0,
                    "forward_avg_snr": -8.0,
                    "return_avg_snr": -12.0,
                    "forward_count": 3,
                    "return_count": 2,
                    "last_packet_id": 42,
                    "channel_id": "SFNarrow",
                    "spreading_factor": 7,
                    "forward_quality": "fair",
                    "return_quality": "marginal",
                    "quality": "marginal",
                    "worst_snr": -12.0,
                    "forward_estimated_reliability": 91.8,
                    "return_estimated_reliability": 4.7,
                    "estimated_reliability": 4.7,
                    "link_balance": "marginal_both",
                    "is_bidirectional": True,
                    "observation_count": 5,
                    "strength": 3.2,
                    "forward_color": "#ffc107",
                    "return_color": "#dc3545",
                    "quality_color": "#dc3545",
                }
            ]
        }

        links = LocationService.get_traceroute_links(network_data=network_data)
        link = links[0]

        for field in ENRICHMENT_FIELDS:
            assert link[field] == network_data["links"][0][field]
        # Observation aliases survive for existing consumers.
        assert link["total_hops_seen"] == 5


class TestPacketLinksMetrics:
    """Packet-based RF links expose the same metrics with channel resolution."""

    def _database(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE packet_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                from_node_id INTEGER,
                gateway_id TEXT,
                channel_id TEXT,
                hop_start INTEGER,
                hop_limit INTEGER,
                rssi REAL,
                snr REAL
            )
            """
        )
        return conn

    @staticmethod
    def _insert(conn, timestamp, from_node, gateway_hex, rssi, snr, channel=None):
        conn.execute(
            """
            INSERT INTO packet_history (
                timestamp, from_node_id, gateway_id, channel_id,
                hop_start, hop_limit, rssi, snr
            ) VALUES (?, ?, ?, ?, 3, 3, ?, ?)
            """,
            (timestamp, from_node, gateway_hex, channel, rssi, snr),
        )

    def _packet_links(self, conn):
        _PACKET_LINKS_CACHE.clear()
        try:
            with patch(
                "malla.database.connection.get_db_connection",
                return_value=_NonClosingConnection(conn),
            ):
                return LocationService.get_packet_links({})
        finally:
            _PACKET_LINKS_CACHE.clear()

    def test_enrichment_matches_shared_helper(self):
        conn = self._database()
        # Canonical forward 100 -> 200 (gateway 0xc8): avg -10 over 3 packets.
        self._insert(conn, 1000.0, 100, "!000000c8", -60.0, -8.0, "SFNarrow")
        self._insert(conn, 1001.0, 100, "!000000c8", -60.0, -10.0, "SFNarrow")
        self._insert(conn, 1002.0, 100, "!000000c8", -60.0, -12.0, "SFNarrow")
        # Canonical return 200 -> 100 (gateway 0x64): avg -6 over 2 packets,
        # with the newest observation overall carrying the channel hint.
        self._insert(conn, 1003.0, 200, "!00000064", -90.0, -5.0, "LongFast")
        self._insert(conn, 1004.0, 200, "!00000064", -90.0, -7.0, "SFNarrow")
        conn.commit()

        links = self._packet_links(conn)
        link = {(link["from_node_id"], link["to_node_id"]): link for link in links}[
            (100, 200)
        ]

        expected = enrich_link_quality(
            channel_id="SFNarrow",
            forward_avg_snr=-10.0,
            return_avg_snr=-6.0,
            forward_observations=3,
            return_observations=2,
        )
        for field in ENRICHMENT_FIELDS:
            assert link[field] == expected[field], field
        assert link["total_hops_seen"] == link["observation_count"] == 5
        assert link["last_packet_id"] == 5

    def test_gateway_alias_rows_resolve_link_channel_by_recency(self):
        conn = self._database()
        # Two gateway aliases for the same direction, different channels and
        # last-seen timestamps: the newer observation provides the hint.
        self._insert(conn, 1000.0, 100, "!000000c8", -60.0, -10.0, "LongFast")
        self._insert(conn, 2000.0, 100, "!000000C8", -60.0, -10.0, "SFNarrow")
        conn.commit()

        link = self._packet_links(conn)[0]
        assert link["channel_id"] == "SFNarrow"
        assert link["forward_count"] == 2

    def test_width_scales_with_observations_not_snr(self):
        conn = self._database()
        for i in range(100):
            self._insert(conn, 1000.0 + i, 100, "!000000c8", -60.0, -5.0, "LongFast")
        for i in range(100):
            self._insert(
                conn, 1000.0 + i, 300, "!00000190", -100.0, -25.0, "LongFast"
            )
        conn.commit()

        by_key = {
            (link["from_node_id"], link["to_node_id"]): link
            for link in self._packet_links(conn)
        }
        strong, weak = by_key[(100, 200)], by_key[(300, 400)]
        assert strong["strength"] == weak["strength"] == 6.5
        assert strong["quality"] == QUALITY_GOOD
        assert weak["quality"] == QUALITY_MARGINAL

    def test_rf_configuration_is_part_of_cache_identity(self, monkeypatch):
        conn = self._database()
        self._insert(conn, 1000.0, 100, "!000000c8", -60.0, -7.5, None)
        conn.commit()

        def links():
            # Deliberately do NOT clear _PACKET_LINKS_CACHE between calls.
            with patch(
                "malla.database.connection.get_db_connection",
                return_value=_NonClosingConnection(conn),
            ):
                return LocationService.get_packet_links({})

        try:
            default = links()
            assert default[0]["quality"] == QUALITY_GOOD

            monkeypatch.setenv("MALLA_LORA_PRESET", "SFNarrow")
            _clear_config_cache()
            reconfigured = links()
        finally:
            _PACKET_LINKS_CACHE.clear()

        assert reconfigured[0]["quality"] == QUALITY_MARGINAL
        assert reconfigured[0]["spreading_factor"] == 7


class TestHopQueryChannelSelection:
    """get_traceroute_hops_for_graph always joins packet_history for channel."""

    @pytest.fixture
    def database(self, tmp_path):
        from meshtastic import mesh_pb2

        from malla.database.traceroute_schema import ensure_traceroute_schema
        from malla.database.traceroutes import write_traceroute

        path = tmp_path / "reader.db"
        with closing(sqlite3.connect(path)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute(
                """
                CREATE TABLE packet_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                )
                """
            )
            ensure_traceroute_schema(conn.cursor())

            def insert(packet_id, timestamp, mesh_id, gateway, channel):
                raw = mesh_pb2.RouteDiscovery(
                    route=(901,), snr_towards=[-40, -40]
                ).SerializeToString()
                cursor = conn.execute(
                    """
                    INSERT INTO packet_history (
                        id, timestamp, portnum, portnum_name, mesh_packet_id,
                        from_node_id, to_node_id, gateway_id, channel_id,
                        hop_start, hop_limit, rssi, snr, payload_length, raw_payload
                    ) VALUES (?, ?, 70, 'TRACEROUTE_APP', ?, 100, 200, ?, ?,
                              5, 3, -80, -10, ?, ?)
                    """,
                    (packet_id, timestamp, mesh_id, gateway, channel, len(raw), raw),
                )
                packet = dict(
                    conn.execute(
                        "SELECT * FROM packet_history WHERE id = ?",
                        (cursor.lastrowid,),
                    ).fetchone()
                )
                write_traceroute(conn.cursor(), packet)

            insert(1, 10.0, 101, "!00000001", "LongFast")
            insert(2, 11.0, 102, "!00000002", "SFNarrow")
            conn.commit()
        return path

    def _connection(self, path):
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn

    def test_hops_carry_channel_id_and_preserve_filters(self, database):
        with patch(
            "malla.database.traceroute_read_repository.get_db_connection",
            side_effect=lambda: self._connection(database),
        ):
            hops = get_traceroute_hops_for_graph(
                filters={"start_time": 0.0, "end_time": 20.0}
            )
            gateway_hops = get_traceroute_hops_for_graph(
                filters={"gateway_id": "!00000002"}
            )

        assert {h["channel_id"] for h in hops} == {"LongFast", "SFNarrow"}
        assert [h["from_node_id"] for h in hops] == [100, 901, 100, 901]
        assert [h["channel_id"] for h in gateway_hops] == ["SFNarrow", "SFNarrow"]


class TestLinkAnalysisQualityFields:
    """/api/traceroute/link/<n1>/<n2> reports the shared fields per URL order."""

    def _client(self, link_result, names):
        app = Flask(__name__)
        register_api_routes(app)
        return (
            app.test_client(),
            patch(
                "malla.routes.api_routes.get_traceroute_link",
                return_value=link_result,
            ),
            patch(
                "malla.routes.api_routes.NodeRepository.get_bulk_node_names",
                return_value=names,
            ),
        )

    def test_reports_quality_and_per_direction_reliability(self):
        link_result = {
            "packets": [],
            "total_count": 5,
            "total_attempts": 5,
            "forward_count": 3,
            "reverse_count": 2,
            "avg_snr": -10.0,
            "forward_avg_snr": -10.0,
            "reverse_avg_snr": -6.0,
            "channel_id": "SFNarrow",
        }
        client, query, names = self._client(link_result, {100: "Node A", 200: "Node B"})
        with query, names, client:
            response = client.get("/api/traceroute/link/100/200")

        assert response.status_code == 200
        data = response.get_json()
        expected = enrich_link_quality(
            channel_id="SFNarrow",
            forward_avg_snr=-10.0,
            return_avg_snr=-6.0,
            forward_observations=3,
            return_observations=2,
        )
        for field in ENRICHMENT_FIELDS:
            assert data[field] == expected[field], field
        assert data["forward_estimated_reliability"] is not None
        assert data["return_estimated_reliability"] is not None

    def test_key_preset_regression_through_endpoint(self):
        link_result = {
            "packets": [],
            "total_count": 1,
            "total_attempts": 1,
            "forward_count": 1,
            "reverse_count": 0,
            "avg_snr": -7.5,
            "forward_avg_snr": -7.5,
            "channel_id": "LongFast",
        }
        client, query, names = self._client(link_result, {100: "Node A", 200: "Node B"})
        with query, names, client:
            response = client.get("/api/traceroute/link/100/200")

        data = response.get_json()
        assert data["quality"] == QUALITY_GOOD
        assert data["spreading_factor"] == 11

        narrow = dict(link_result, channel_id="SFNarrow")
        client, query, names = self._client(narrow, {100: "Node A", 200: "Node B"})
        with query, names, client:
            response = client.get("/api/traceroute/link/100/200")

        data = response.get_json()
        assert data["quality"] == QUALITY_MARGINAL
        assert data["spreading_factor"] == 7

    def test_reversed_url_order_flips_directional_quality(self):
        link_result = {
            "packets": [],
            "total_count": 3,
            "total_attempts": 3,
            "forward_count": 2,
            "reverse_count": 1,
            "avg_snr": -10.0,
            "forward_avg_snr": -10.0,
            "reverse_avg_snr": -2.0,
            "channel_id": "LongFast",
        }
        client, query, names = self._client(link_result, {100: "Node A", 200: "Node B"})
        with query, names, client:
            response = client.get("/api/traceroute/link/200/100")

        data = response.get_json()
        # forward follows the URL order (200 → 100): the repository's forward
        # average is the 200→100 direction, i.e. the fair one here.
        assert data["forward_quality"] == QUALITY_FAIR
        assert data["return_quality"] == QUALITY_GOOD
        assert data["worst_snr"] == -10.0

    def test_averages_rounded_before_classification_for_parity(self):
        # -13.549 would classify as marginal unrounded (margin 3.951) but
        # must be rounded to -13.5 first (margin exactly 4.0 → fair), the
        # same value the graph and packet links classify.
        link_result = {
            "packets": [],
            "total_count": 2,
            "total_attempts": 2,
            "forward_count": 2,
            "reverse_count": 0,
            "avg_snr": -13.549,
            "forward_avg_snr": -13.549,
            "channel_id": "LongFast",
        }
        client, query, names = self._client(link_result, {100: "Node A", 200: "Node B"})
        with query, names, client:
            response = client.get("/api/traceroute/link/100/200")

        data = response.get_json()
        assert data["forward_avg_snr"] == -13.5
        assert data["forward_quality"] == QUALITY_FAIR

    def test_repository_stats_select_latest_channel_deterministically(self):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            CREATE TABLE packet_history (
                id INTEGER PRIMARY KEY, gateway_id TEXT, channel_id TEXT,
                hop_start INTEGER, hop_limit INTEGER, rssi REAL, snr REAL,
                payload_length INTEGER, processed_successfully INTEGER
            );
            CREATE TABLE traceroute_routes (
                packet_id INTEGER PRIMARY KEY, timestamp REAL, from_node_id INTEGER,
                to_node_id INTEGER, mesh_packet_id INTEGER, route_nodes_json TEXT,
                snr_towards_json TEXT, route_back_json TEXT, snr_back_json TEXT,
                parse_status TEXT, parser_version INTEGER
            );
            CREATE TABLE traceroute_hops (
                packet_id INTEGER, direction TEXT, hop_index INTEGER, timestamp REAL,
                from_node_id INTEGER, to_node_id INTEGER, snr REAL
            );
            """
        )
        now = time.time()
        rows = [
            (1, now + 1, 100, 200, -10.0, "LongFast"),
            (2, now + 2, 100, 200, -12.0, "LongFast"),
            # Newest observation carries a different channel hint.
            (3, now + 3, 200, 100, -6.0, "SFNarrow"),
        ]
        for packet_id, ts, from_node, to_node, snr, channel in rows:
            connection.execute(
                "INSERT INTO packet_history VALUES (?, '!0000012c', ?, 5, 4, -80, 1, 10, 1)",
                (packet_id, channel),
            )
            connection.execute(
                "INSERT INTO traceroute_routes VALUES "
                "(?, ?, 100, 200, ?, '[]', '[]', '[]', '[]', 'parsed', 1)",
                (packet_id, ts, packet_id),
            )
            connection.execute(
                "INSERT INTO traceroute_hops VALUES (?, 'forward', 0, ?, ?, ?, ?)",
                (packet_id, ts, from_node, to_node, snr),
            )
        connection.commit()

        with patch(
            "malla.database.traceroute_read_repository.get_db_connection",
            return_value=_NonClosingConnection(connection),
        ):
            result = get_traceroute_link(
                100, 200, start_time=now, end_time=now + 30, limit=10, offset=0
            )

        assert result["channel_id"] == "SFNarrow"
        expected = enrich_link_quality(
            channel_id="SFNarrow",
            forward_avg_snr=-11.0,
            return_avg_snr=-6.0,
            forward_observations=2,
            return_observations=1,
        )
        assert expected["forward_quality"] == QUALITY_MARGINAL
        assert expected["return_quality"] == QUALITY_MARGINAL


class TestConsumerParity:
    """Identical aggregates produce identical metrics in every consumer."""

    def test_graph_packet_links_and_endpoint_agree(self):
        now = time.time()
        channel = "SFNarrow"
        forward_avg, return_avg = -10.0, -6.0
        forward_count, return_count = 3, 2

        expected = enrich_link_quality(
            channel_id=channel,
            forward_avg_snr=forward_avg,
            return_avg_snr=return_avg,
            forward_observations=forward_count,
            return_observations=return_count,
        )

        # Graph aggregation: forward 100->200 over three hops, return over two.
        hops = [
            _hop(1, 0, 100, 200, -8.0, now - 5, channel_id=channel),
            _hop(2, 0, 100, 200, -10.0, now - 4, channel_id=channel),
            _hop(3, 0, 100, 200, -12.0, now - 3, channel_id=channel),
            _hop(4, 0, 200, 100, -5.0, now - 2, channel_id=channel),
            _hop(5, 0, 200, 100, -7.0, now, channel_id=channel),
        ]
        _NETWORK_GRAPH_CACHE.clear()
        try:
            with (
                patch(
                    "malla.services.traceroute_service.get_traceroute_hops_for_graph",
                    return_value=hops,
                ),
                patch(
                    "malla.services.traceroute_service.get_bulk_node_names",
                    return_value={},
                ),
                patch(
                    "malla.services.traceroute_service.LocationRepository.get_node_locations",
                    return_value=[],
                ),
            ):
                graph = TracerouteService.get_network_graph_data(
                    hours=24,
                    min_snr=-200.0,
                    include_indirect=False,
                    filters={"start_time": now - 60, "end_time": now + 60},
                )
        finally:
            _NETWORK_GRAPH_CACHE.clear()
        graph_link = graph["links"][0]

        # Packet links with the same aggregates.
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE packet_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL, from_node_id INTEGER, gateway_id TEXT,
                channel_id TEXT, hop_start INTEGER, hop_limit INTEGER,
                rssi REAL, snr REAL
            )
            """
        )
        packet_rows = [
            (1000.0, 100, "!000000c8", channel, -60.0, -8.0),
            (1001.0, 100, "!000000c8", channel, -60.0, -10.0),
            (1002.0, 100, "!000000c8", channel, -60.0, -12.0),
            (1003.0, 200, "!00000064", channel, -90.0, -5.0),
            (1004.0, 200, "!00000064", channel, -90.0, -7.0),
        ]
        conn.executemany(
            """
            INSERT INTO packet_history (
                timestamp, from_node_id, gateway_id, channel_id,
                hop_start, hop_limit, rssi, snr
            ) VALUES (?, ?, ?, ?, 3, 3, ?, ?)
            """,
            packet_rows,
        )
        conn.commit()
        _PACKET_LINKS_CACHE.clear()
        try:
            with patch(
                "malla.database.connection.get_db_connection",
                return_value=_NonClosingConnection(conn),
            ):
                packet_link = {
                    (link["from_node_id"], link["to_node_id"]): link
                    for link in LocationService.get_packet_links({})
                }[(100, 200)]
        finally:
            _PACKET_LINKS_CACHE.clear()

        # Link-analysis endpoint with the same aggregates.
        link_result = {
            "packets": [],
            "total_count": 5,
            "total_attempts": 5,
            "forward_count": forward_count,
            "reverse_count": return_count,
            "avg_snr": -9.6,
            "forward_avg_snr": forward_avg,
            "reverse_avg_snr": return_avg,
            "channel_id": channel,
        }
        app = Flask(__name__)
        register_api_routes(app)
        with (
            patch(
                "malla.routes.api_routes.get_traceroute_link",
                return_value=link_result,
            ),
            patch(
                "malla.routes.api_routes.NodeRepository.get_bulk_node_names",
                return_value={},
            ),
            app.test_client() as client,
        ):
            endpoint_data = client.get("/api/traceroute/link/100/200").get_json()

        for field in ENRICHMENT_FIELDS:
            assert graph_link[field] == expected[field], f"graph mismatch: {field}"
            assert packet_link[field] == expected[field], f"packet mismatch: {field}"
            assert endpoint_data[field] == expected[field], f"endpoint mismatch: {field}"

    def test_reproduction_zero_snr_reverse_hop_parity(self):
        now = time.time()
        # Older hop: forward 100->200, -7.5 dB, LongFast
        # Newer hop: reverse 200->100, 0 dB, SFNarrow
        hops = [
            _hop(1, 0, 100, 200, -7.5, now - 10, channel_id="LongFast"),
            _hop(2, 0, 200, 100, 0.0, now, channel_id="SFNarrow", direction="return"),
        ]
        _NETWORK_GRAPH_CACHE.clear()
        try:
            with (
                patch(
                    "malla.services.traceroute_service.get_traceroute_hops_for_graph",
                    return_value=hops,
                ),
                patch(
                    "malla.services.traceroute_service.get_bulk_node_names",
                    return_value={100: "Node A", 200: "Node B"},
                ),
                patch(
                    "malla.services.traceroute_service.LocationRepository.get_node_locations",
                    return_value=[],
                ),
            ):
                graph = TracerouteService.get_network_graph_data(
                    hours=24,
                    min_snr=-200.0,
                    include_indirect=False,
                    filters={"start_time": now - 60, "end_time": now + 60},
                )
        finally:
            _NETWORK_GRAPH_CACHE.clear()
        graph_link = graph["links"][0]

        # In graph: 0 dB reverse hop is excluded by topology rules.
        assert graph_link["channel_id"] == "LongFast"
        assert graph_link["spreading_factor"] == 11
        assert graph_link["quality"] == QUALITY_GOOD
        assert graph_link["estimated_reliability"] == SF11_RELIABILITY_PLUS_10_DB
        assert graph_link["is_bidirectional"] is False
        assert graph_link["link_balance"] == "unidirectional"

        # Now for the endpoint with the repository returning the eligible observations:
        link_result = {
            "packets": [],
            "total_count": 2,
            "total_attempts": 2,
            "forward_count": 1,
            "reverse_count": 1,
            "forward_observations": 1,
            "reverse_observations": 0,
            "avg_snr": -7.5,
            "forward_avg_snr": -7.5,
            "reverse_avg_snr": None,
            "channel_id": "LongFast",
        }
        app = Flask(__name__)
        register_api_routes(app)
        with (
            patch(
                "malla.routes.api_routes.get_traceroute_link",
                return_value=link_result,
            ),
            patch(
                "malla.routes.api_routes.NodeRepository.get_bulk_node_names",
                return_value={100: "Node A", 200: "Node B"},
            ),
            app.test_client() as client,
        ):
            endpoint_data = client.get("/api/traceroute/link/100/200").get_json()

        # Endpoint produces identical quality, reliability, and coverage metrics to the graph
        assert endpoint_data["channel_id"] == graph_link["channel_id"] == "LongFast"
        assert endpoint_data["spreading_factor"] == graph_link["spreading_factor"] == 11
        assert endpoint_data["quality"] == graph_link["quality"] == QUALITY_GOOD
        assert (
            endpoint_data["estimated_reliability"]
            == graph_link["estimated_reliability"]
            == SF11_RELIABILITY_PLUS_10_DB
        )
        assert endpoint_data["is_bidirectional"] == graph_link["is_bidirectional"] is False
        assert endpoint_data["link_balance"] == graph_link["link_balance"] == "unidirectional"

        # Legacy counts remain preserved in endpoint response
        assert endpoint_data["total_attempts"] == 2
        assert endpoint_data["forward_count"] == 1
        assert endpoint_data["return_count"] == 1
        assert endpoint_data["forward_observations"] == 1
        assert endpoint_data["return_observations"] == 0

