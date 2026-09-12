"""Directional (forward/return) RF measurement tests across the link pipeline.

Covers the graph aggregation in TracerouteService, the LocationService link
mappers, the materialized link-statistics query and the HTTP endpoint —
including unequal sample counts, reversed URL order, return paths containing
canonical forward hops, one-direction links, unknown-only measurements,
pagination, and weighted vs. mean-of-means averaging.
"""

import sqlite3
import time
from unittest.mock import patch

import pytest
from flask import Flask

from malla.database.traceroute_read_repository import get_traceroute_link
from malla.routes.api_routes import register_api_routes
from malla.services.location_service import _PACKET_LINKS_CACHE, LocationService
from malla.services.traceroute_service import _NETWORK_GRAPH_CACHE, TracerouteService

pytestmark = pytest.mark.unit


class _NonClosingConnection:
    def __init__(self, connection):
        self.connection = connection

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def close(self):
        pass


def _hop(packet_id, hop_index, from_node, to_node, snr, timestamp, direction="forward"):
    return {
        "packet_id": packet_id,
        "direction": direction,
        "hop_index": hop_index,
        "timestamp": timestamp,
        "from_node_id": from_node,
        "to_node_id": to_node,
        "snr": snr,
    }


class TestNetworkGraphDirectionalAggregation:
    """TracerouteService.get_network_graph_data buckets hops per direction."""

    def _graph(self, hops):
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
                    include_indirect=False,
                    filters={"start_time": now - 60, "end_time": now + 60},
                )
        finally:
            _NETWORK_GRAPH_CACHE.clear()

    def test_direction_assigned_by_hop_endpoints_with_unequal_counts(self):
        now = time.time()
        graph = self._graph(
            [
                # Forward path hops: 100 -> 200 (canonical forward direction)
                _hop(1, 0, 100, 200, -10.0, now),
                _hop(2, 0, 100, 200, -20.0, now),
                # Return path hop: 200 -> 100 (canonical return direction)
                _hop(3, 0, 200, 100, -5.0, now, direction="return"),
                # A stored *return* path containing a canonical forward hop:
                # direction must follow the hop endpoints, not the path.
                _hop(4, 0, 100, 200, -8.0, now, direction="return"),
            ]
        )

        assert len(graph["links"]) == 1
        link = graph["links"][0]
        assert (link["source"], link["target"]) == (100, 200)
        # Weighted combined average is preserved (all four samples)
        assert link["avg_snr"] == round((-10.0 + -20.0 + -5.0 + -8.0) / 4, 1)
        # Directional averages keep the unequal sample counts separate
        assert link["forward_avg_snr"] == round((-10.0 + -20.0 + -8.0) / 3, 1)
        assert link["return_avg_snr"] == -5.0
        assert link["forward_count"] == 3
        assert link["return_count"] == 1
        assert link["packet_count"] == 4

    def test_one_direction_link_reports_missing_direction(self):
        now = time.time()
        graph = self._graph([_hop(5, 0, 300, 400, -7.0, now)])

        link = graph["links"][0]
        assert (link["source"], link["target"]) == (300, 400)
        assert link["forward_avg_snr"] == -7.0
        assert link["forward_count"] == 1
        assert link["return_avg_snr"] is None
        assert link["return_count"] == 0

    def test_unknown_only_measurement_counts_observation_without_snr(self):
        now = time.time()
        graph = self._graph([_hop(6, 0, 500, 600, -32.0, now)])

        link = graph["links"][0]
        assert link["forward_count"] == 1  # observed...
        assert link["forward_avg_snr"] is None  # ...but SNR unrecorded
        assert link["return_count"] == 0
        # Combined behavior unchanged: the sentinel is still averaged in.
        assert link["avg_snr"] == -32.0

    def test_zero_snr_hop_excluded_from_every_bucket(self):
        now = time.time()
        graph = self._graph(
            [
                _hop(7, 0, 100, 200, 0.0, now),
                _hop(8, 0, 100, 200, -4.0, now),
            ]
        )

        link = graph["links"][0]
        assert link["forward_count"] == 1
        assert link["forward_avg_snr"] == -4.0
        assert link["return_count"] == 0
        assert graph["stats"]["links_filtered_due_to_snr_0"] == 1


class TestTracerouteLinksMapping:
    """LocationService.get_traceroute_links passes directional fields through."""

    def test_passes_directional_fields_and_derives_bidirectional(self):
        network_data = {
            "links": [
                {
                    "source": 10,
                    "target": 20,
                    "packet_count": 4,
                    "last_seen": 1000.0,
                    "avg_snr": 1.0,
                    "forward_avg_snr": 5.0,
                    "return_avg_snr": -3.0,
                    "forward_count": 2,
                    "return_count": 2,
                    "last_packet_id": 42,
                },
                {
                    "source": 30,
                    "target": 40,
                    "packet_count": 3,
                    "last_seen": 1000.0,
                    "avg_snr": -7.0,
                    "forward_avg_snr": -7.0,
                    "return_avg_snr": None,
                    "forward_count": 3,
                    "return_count": 0,
                    "last_packet_id": 43,
                },
            ]
        }

        links = LocationService.get_traceroute_links(network_data=network_data)

        by_key = {(link["from_node_id"], link["to_node_id"]): link for link in links}
        bidirectional = by_key[(10, 20)]
        assert bidirectional["is_bidirectional"] is True
        assert bidirectional["avg_snr"] == 1.0
        assert bidirectional["forward_avg_snr"] == 5.0
        assert bidirectional["return_avg_snr"] == -3.0
        assert bidirectional["forward_count"] == 2
        assert bidirectional["return_count"] == 2

        one_way = by_key[(30, 40)]
        assert one_way["is_bidirectional"] is False
        assert one_way["forward_avg_snr"] == -7.0
        assert one_way["return_avg_snr"] is None
        assert one_way["forward_count"] == 3
        assert one_way["return_count"] == 0

    def test_legacy_payload_without_counts_stays_bidirectional(self):
        network_data = {
            "links": [
                {
                    "source": 10,
                    "target": 20,
                    "packet_count": 4,
                    "last_seen": 1000.0,
                    "avg_snr": 1.0,
                    "last_packet_id": 42,
                }
            ]
        }

        links = LocationService.get_traceroute_links(network_data=network_data)

        assert links[0]["is_bidirectional"] is True
        assert links[0]["forward_avg_snr"] is None
        assert links[0]["forward_count"] is None


class TestPacketLinksDirectionalMerge:
    """LocationService.get_packet_links keeps per-direction measurements."""

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
                hop_start INTEGER,
                hop_limit INTEGER,
                rssi REAL,
                snr REAL
            )
            """
        )
        return conn

    @staticmethod
    def _insert(conn, timestamp, from_node, gateway_hex, rssi, snr):
        conn.execute(
            """
            INSERT INTO packet_history (
                timestamp, from_node_id, gateway_id, hop_start, hop_limit, rssi, snr
            ) VALUES (?, ?, ?, 3, 3, ?, ?)
            """,
            (timestamp, from_node, gateway_hex, rssi, snr),
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

    def test_bidirectional_merge_keeps_directions_and_mean_of_means(self):
        conn = self._database()
        # Canonical forward 100 -> 200 (gateway 0xc8): three receptions
        self._insert(conn, 1000.0, 100, "!000000c8", -60.0, 10.0)
        self._insert(conn, 1001.0, 100, "!000000c8", -80.0, 4.0)
        self._insert(conn, 1002.0, 100, "!000000c8", -70.0, 0.0)
        # Canonical return 200 -> 100 (gateway 0x64): one reception
        self._insert(conn, 1003.0, 200, "!00000064", -100.0, -2.0)
        conn.commit()

        links = self._packet_links(conn)
        by_key = {(link["from_node_id"], link["to_node_id"]): link for link in links}

        main = by_key[(100, 200)]
        assert main["forward_count"] == 3
        assert main["return_count"] == 1
        assert main["forward_avg_snr"] == 4.7  # (10 + 4 + 0) / 3
        assert main["return_avg_snr"] == -2.0
        assert main["forward_avg_rssi"] == -70.0
        assert main["return_avg_rssi"] == -100.0
        # Combined values are a mean of the directional means, rounded after
        # the merge — not a weighted average over the four receptions.
        assert main["avg_snr"] == 1.3  # ((10+4+0)/3 + -2.0) / 2
        assert main["avg_rssi"] == -85.0
        assert main["is_bidirectional"] is True
        assert main["total_hops_seen"] == 4

    def test_one_direction_link_preserves_real_zero_db_average(self):
        conn = self._database()
        # 300 -> 100 only: a 0.0 dB average must survive the null-safe merge.
        self._insert(conn, 1000.0, 300, "!00000064", None, 0.0)
        self._insert(conn, 1001.0, 300, "!00000064", None, 0.0)
        conn.commit()

        links = self._packet_links(conn)
        by_key = {(link["from_node_id"], link["to_node_id"]): link for link in links}

        one_way = by_key[(100, 300)]
        assert one_way["forward_count"] == 0
        assert one_way["forward_avg_snr"] is None
        assert one_way["return_count"] == 2
        assert one_way["return_avg_snr"] == 0.0
        assert one_way["avg_snr"] == 0.0  # truthiness must not drop a real 0.0
        assert one_way["avg_rssi"] is None
        assert one_way["is_bidirectional"] is False

    @pytest.mark.parametrize("from_node", [100, 300])
    def test_gateway_aliases_preserve_one_direction_observations(self, from_node):
        conn = self._database()
        try:
            self._insert(conn, 1000.0, from_node, "!000000c8", -60.0, 10.0)
            self._insert(conn, 1001.0, from_node, "!000000C8", -100.0, -10.0)
            conn.commit()

            links = self._packet_links(conn)
        finally:
            conn.close()

        assert len(links) == 1
        link = links[0]
        direction = "forward" if from_node < 200 else "return"
        other_direction = "return" if from_node < 200 else "forward"
        assert link[f"{direction}_count"] == 2
        assert link[f"{direction}_avg_snr"] == 0.0
        assert link[f"{direction}_avg_rssi"] == -80.0
        assert link[f"{other_direction}_count"] == 0
        assert link[f"{other_direction}_avg_snr"] is None
        assert link["total_hops_seen"] == 2
        assert link["avg_snr"] == 0.0
        assert link["avg_rssi"] == -80.0
        assert link["is_bidirectional"] is False

    def test_gateway_aliases_weight_each_metric_by_its_valid_samples(self):
        conn = self._database()
        try:
            # Six forward observations, but only three usable SNR readings
            # and two usable RSSI readings, spread across gateway aliases.
            self._insert(conn, 1000.0, 160, "!000000c8", -60.0, 10.0)
            self._insert(conn, 1001.0, 160, "!000000c8", None, 4.0)
            self._insert(conn, 1002.0, 160, "!000000c8", 0.0, None)
            self._insert(conn, 1003.0, 160, "!000000C8", -100.0, -2.0)
            self._insert(conn, 1004.0, 160, "!c8", -1e9, 1e9)
            self._insert(conn, 1005.0, 160, "000000c8", None, None)
            # Reverse observations have their own gateway aliases and means.
            self._insert(conn, 1006.0, 200, "!000000a0", -110.0, 1.0)
            self._insert(conn, 1007.0, 200, "!000000A0", None, 3.0)
            conn.commit()

            links = self._packet_links(conn)
        finally:
            conn.close()

        assert len(links) == 1
        link = links[0]
        assert link["forward_count"] == 6
        assert link["return_count"] == 2
        assert link["total_hops_seen"] == 8
        assert link["forward_avg_snr"] == 4.0  # (10 + 4 - 2) / 3
        assert link["forward_avg_rssi"] == -80.0  # (-60 - 100) / 2
        assert link["return_avg_snr"] == 2.0
        assert link["return_avg_rssi"] == -110.0
        # Preserve the combined mean of directional means, not a mean of
        # gateway-alias means or a weighted average over both directions.
        assert link["avg_snr"] == 3.0
        assert link["avg_rssi"] == -95.0
        assert link["is_bidirectional"] is True


class TestTracerouteLinkRepositoryDirectionalStats:
    """get_traceroute_link computes directional averages over the full window."""

    def _database(self):
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
            # forward 100 -> 200
            (1, now + 1, 100, 200, -10.0),
            (2, now + 2, 100, 200, -20.0),
            (3, now + 3, 100, 200, -30.0),
            # reverse 200 -> 100, including an unknown-SNR-only direction pair
            (4, now + 4, 200, 100, -5.0),
            (5, now + 5, 200, 100, -32.0),
        ]
        for packet_id, ts, from_node, to_node, snr in rows:
            connection.execute(
                "INSERT INTO packet_history VALUES (?, ?, '', 5, 4, -80, 1, 10, 1)",
                (packet_id, "!0000012c"),
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
        return connection, now

    def test_directional_statistics_cover_window_despite_pagination(self):
        connection, now = self._database()
        with patch(
            "malla.database.traceroute_read_repository.get_db_connection",
            return_value=_NonClosingConnection(connection),
        ):
            result = get_traceroute_link(
                100, 200, start_time=now, end_time=now + 30, limit=2, offset=0
            )

        assert result["total_count"] == 5
        assert len(result["packets"]) == 2  # details stay paginated
        assert result["forward_count"] == 3
        assert result["forward_avg_snr"] == pytest.approx(-20.0)
        assert result["reverse_count"] == 2
        # The -32.0 sentinel is an observation without SNR: excluded from
        # the directional average, kept as a count.
        assert result["reverse_avg_snr"] == pytest.approx(-5.0)
        # Combined average keeps its historical validity window (sentinel in).
        assert result["avg_snr"] == pytest.approx(
            (-10.0 - 20.0 - 30.0 - 5.0 - 32.0) / 5
        )

    def test_reversed_node_order_flips_directional_fields(self):
        connection, now = self._database()
        with patch(
            "malla.database.traceroute_read_repository.get_db_connection",
            return_value=_NonClosingConnection(connection),
        ):
            result = get_traceroute_link(
                200, 100, start_time=now, end_time=now + 30, limit=10, offset=0
            )

        # forward is now 200 -> 100 (the URL's node order)
        assert result["forward_count"] == 2
        assert result["forward_avg_snr"] == pytest.approx(-5.0)
        assert result["reverse_count"] == 3
        assert result["reverse_avg_snr"] == pytest.approx(-20.0)

    @pytest.mark.parametrize("node_order", [(100, 200), (200, 100)])
    def test_directional_averages_exclude_zero_snr_like_graph(self, node_order):
        connection, now = self._database()
        try:
            connection.execute(
                "UPDATE traceroute_hops SET snr = 0 WHERE packet_id IN (2, 3, 4)"
            )
            connection.commit()
            hops = [
                dict(row) for row in connection.execute("SELECT * FROM traceroute_hops")
            ]
            with patch(
                "malla.database.traceroute_read_repository.get_db_connection",
                return_value=_NonClosingConnection(connection),
            ):
                result = get_traceroute_link(
                    *node_order, start_time=now, end_time=now + 30, limit=10, offset=0
                )
        finally:
            connection.close()

        graph_link = TestNetworkGraphDirectionalAggregation()._graph(hops)["links"][0]
        assert graph_link["forward_avg_snr"] == -10.0
        assert graph_link["return_avg_snr"] is None
        forward_direction, reverse_direction = (
            ("forward", "return") if node_order[0] == 100 else ("return", "forward")
        )
        assert result["forward_avg_snr"] == graph_link[f"{forward_direction}_avg_snr"]
        assert result["reverse_avg_snr"] == graph_link[f"{reverse_direction}_avg_snr"]
        # Raw observations and legacy combined statistics remain available.
        assert result["total_attempts"] == 5
        assert result["forward_count"] == (3 if node_order[0] == 100 else 2)
        assert result["reverse_count"] == (2 if node_order[0] == 100 else 3)
        assert result["avg_snr"] == pytest.approx((-10.0 - 32.0) / 5)
        assert sum(packet["target_hop_snr"] == 0 for packet in result["packets"]) == 3


class TestTracerouteLinkEndpointDirectionalFields:
    """/api/traceroute/link/<n1>/<n2> reports directional fields per URL order."""

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

    def test_reports_directional_fields_relative_to_url_order(self):
        link_result = {
            "packets": [],
            "total_count": 4,
            "total_attempts": 4,
            "forward_count": 3,
            "reverse_count": 1,
            "avg_snr": -15.0,
            "forward_avg_snr": -20.0,
            "reverse_avg_snr": -5.0,
        }
        client, query, names = self._client(link_result, {100: "Node A", 200: "Node B"})
        with query, names, client:
            response = client.get("/api/traceroute/link/200/100")

        assert response.status_code == 200
        data = response.get_json()
        assert data["from_node_id"] == 200
        assert data["to_node_id"] == 100
        # forward follows the URL node order (200 → 100), not the numeric ids
        assert data["forward_avg_snr"] == -20.0
        assert data["return_avg_snr"] == -5.0
        assert data["forward_count"] == 3
        assert data["return_count"] == 1
        # Compatibility: direction_counts keeps display-name labels
        assert data["direction_counts"] == {"Node B → Node A": 3, "Node A → Node B": 1}
        assert data["avg_snr"] == -15.0

    def test_legacy_result_without_directional_fields_defaults_cleanly(self):
        link_result = {
            "packets": [],
            "total_count": 2,
            "total_attempts": 2,
            "forward_count": 2,
            "reverse_count": 0,
            "avg_snr": -12.0,
        }
        client, query, names = self._client(link_result, {100: "Node A", 200: "Node B"})
        with query, names, client:
            response = client.get("/api/traceroute/link/100/200")

        data = response.get_json()
        assert response.status_code == 200
        assert data["forward_avg_snr"] is None
        assert data["return_avg_snr"] is None
        assert data["forward_count"] == 2
        assert data["return_count"] == 0
