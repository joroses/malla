"""
Unit tests for LocationService node activity consolidation and location enrichment.
"""

from unittest.mock import patch

import pytest

from malla.services.location_service import LocationService


class TestLocationServiceNodeLocations:
    """Test LocationService.get_node_locations consolidation of timestamps."""

    @pytest.mark.unit
    def test_node_active_timestamp_consolidated_from_traceroute(self):
        """When traceroute activity is newer than position, node.timestamp uses traceroute time."""
        pos_time = 1000.0
        tr_time = 5000.0

        mock_raw_locations = [
            {
                "node_id": 12345,
                "hex_id": "!00003039",
                "display_name": "Test Node",
                "long_name": "Test Node Long",
                "short_name": "TN",
                "hw_model": "T-Beam",
                "role": "ROUTER",
                "latitude": 40.0,
                "longitude": -95.0,
                "altitude": 100,
                "timestamp": pos_time,
                "precision_bits": 16,
                "precision_meters": 364.0,
                "sats_in_view": 8,
            }
        ]

        mock_network_data = {
            "nodes": [
                {
                    "id": 12345,
                    "name": "Test Node",
                    "packet_count": 5,
                    "avg_snr": 8.5,
                    "last_seen": tr_time,
                }
            ],
            "links": [],
        }

        with patch("malla.database.repositories.LocationRepository.get_node_locations", return_value=mock_raw_locations):
            results = LocationService.get_node_locations(
                filters={},
                network_data=mock_network_data,
                packet_links=[],
            )

        assert len(results) == 1
        node = results[0]
        # timestamp must be updated to the active (traceroute) timestamp
        assert node["timestamp"] == tr_time
        # position_timestamp must preserve original position packet timestamp
        assert node["position_timestamp"] == pos_time
        # trimmed payload fields must stay absent
        assert "position_timestamp_str" not in node
        assert node["last_seen_network"] == tr_time

    @pytest.mark.unit
    def test_node_active_timestamp_consolidated_from_packet_link(self):
        """When packet link activity is newer than position and traceroute, node.timestamp uses packet time."""
        pos_time = 1000.0
        pkt_time = 8000.0

        mock_raw_locations = [
            {
                "node_id": 12345,
                "hex_id": "!00003039",
                "display_name": "Test Node",
                "long_name": "Test Node Long",
                "short_name": "TN",
                "hw_model": "T-Beam",
                "role": "ROUTER",
                "latitude": 40.0,
                "longitude": -95.0,
                "altitude": 100,
                "timestamp": pos_time,
            }
        ]

        mock_packet_links = [
            {
                "from_node_id": 12345,
                "to_node_id": 99999,
                "last_seen": pkt_time,
                "total_hops_seen": 2,
            }
        ]

        with patch("malla.database.repositories.LocationRepository.get_node_locations", return_value=mock_raw_locations):
            results = LocationService.get_node_locations(
                filters={},
                network_data={"nodes": [], "links": []},
                packet_links=mock_packet_links,
            )

        assert len(results) == 1
        node = results[0]
        assert node["timestamp"] == pkt_time
        assert node["position_timestamp"] == pos_time
        assert node["last_seen_packet"] == pkt_time

    @pytest.mark.unit
    def test_node_active_timestamp_defaults_to_position_when_newest(self):
        """When position is newest, node.timestamp remains the position timestamp."""
        pos_time = 10000.0
        tr_time = 5000.0

        mock_raw_locations = [
            {
                "node_id": 12345,
                "hex_id": "!00003039",
                "display_name": "Test Node",
                "long_name": "Test Node Long",
                "short_name": "TN",
                "hw_model": "T-Beam",
                "role": "ROUTER",
                "latitude": 40.0,
                "longitude": -95.0,
                "altitude": 100,
                "timestamp": pos_time,
            }
        ]

        mock_network_data = {
            "nodes": [
                {
                    "id": 12345,
                    "name": "Test Node",
                    "packet_count": 1,
                    "last_seen": tr_time,
                }
            ],
            "links": [],
        }

        with patch("malla.database.repositories.LocationRepository.get_node_locations", return_value=mock_raw_locations):
            results = LocationService.get_node_locations(
                filters={},
                network_data=mock_network_data,
                packet_links=[],
            )

        assert len(results) == 1
        node = results[0]
        assert node["timestamp"] == pos_time
        assert node["position_timestamp"] == pos_time


class TestLocationServiceCacheKey:
    """Test self-computed LocationService.get_node_locations caching behavior."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        LocationService.clear_cache()
        yield
        LocationService.clear_cache()

    def _make_mock_node(self, node_id: int, timestamp: float = 1000.0):
        return {
            "node_id": node_id,
            "hex_id": f"!{node_id:08x}",
            "display_name": f"Node {node_id}",
            "long_name": f"Node Long {node_id}",
            "short_name": f"N{node_id}",
            "hw_model": "T-Beam",
            "role": "ROUTER",
            "latitude": 40.0,
            "longitude": -95.0,
            "altitude": 100,
            "timestamp": timestamp,
            "precision_bits": 16,
            "precision_meters": 364.0,
            "sats_in_view": 8,
        }

    @pytest.mark.unit
    def test_distinct_node_ids_do_not_collide(self):
        """Filters with different node_ids must not return cached results of each other."""
        node1 = self._make_mock_node(1, 1000.0)
        node2 = self._make_mock_node(2, 2000.0)

        def fake_get_node_locations(filters=None):
            filters = filters or {}
            nids = filters.get("node_ids")
            if nids == [1]:
                return [dict(node1)]
            if nids == [2]:
                return [dict(node2)]
            return [dict(node1), dict(node2)]

        with (
            patch(
                "malla.database.repositories.LocationRepository.get_node_locations",
                side_effect=fake_get_node_locations,
            ) as repo_mock,
            patch(
                "malla.services.traceroute_service.TracerouteService.get_network_graph_data",
                return_value={"nodes": [], "links": []},
            ),
            patch(
                "malla.services.location_service.LocationService.get_packet_links",
                return_value=[],
            ),
        ):
            # First call for node 1
            res1 = LocationService.get_node_locations({"node_ids": [1]})
            assert len(res1) == 1
            assert res1[0]["node_id"] == 1
            assert repo_mock.call_count == 1

            # Second call for node 2 must NOT hit node 1 cache entry
            res2 = LocationService.get_node_locations({"node_ids": [2]})
            assert len(res2) == 1
            assert res2[0]["node_id"] == 2
            assert repo_mock.call_count == 2

            # Repeat call for node 1 should hit cache
            res1_again = LocationService.get_node_locations({"node_ids": [1]})
            assert len(res1_again) == 1
            assert res1_again[0]["node_id"] == 1
            assert repo_mock.call_count == 2

    @pytest.mark.unit
    def test_equivalent_node_ids_hit_cache(self):
        """Different representations of the same node set must hit the cache."""
        node12 = [
            self._make_mock_node(1, 1000.0),
            self._make_mock_node(2, 2000.0),
        ]

        with (
            patch(
                "malla.database.repositories.LocationRepository.get_node_locations",
                return_value=[dict(n) for n in node12],
            ) as repo_mock,
            patch(
                "malla.services.traceroute_service.TracerouteService.get_network_graph_data",
                return_value={"nodes": [], "links": []},
            ),
            patch(
                "malla.services.location_service.LocationService.get_packet_links",
                return_value=[],
            ),
        ):
            LocationService.get_node_locations({"node_ids": [1, 2]})
            assert repo_mock.call_count == 1

            # Reversed list should hit cache
            LocationService.get_node_locations({"node_ids": [2, 1]})
            assert repo_mock.call_count == 1

            # Hex string IDs should hit cache
            LocationService.get_node_locations({"node_ids": ["!00000001", "!00000002"]})
            assert repo_mock.call_count == 1

    @pytest.mark.unit
    def test_distinct_age_filters_do_not_collide(self):
        """Different min_age_hours or max_age_hours must produce distinct cache entries."""
        import time

        now_ts = time.time()
        nodes = [
            self._make_mock_node(1, now_ts - 7200),
            self._make_mock_node(2, now_ts - 72000),
        ]

        with (
            patch(
                "malla.database.repositories.LocationRepository.get_node_locations",
                return_value=[dict(n) for n in nodes],
            ) as repo_mock,
            patch(
                "malla.services.traceroute_service.TracerouteService.get_network_graph_data",
                return_value={"nodes": [], "links": []},
            ),
            patch(
                "malla.services.location_service.LocationService.get_packet_links",
                return_value=[],
            ),
        ):
            LocationService.get_node_locations({"min_age_hours": 1})
            assert repo_mock.call_count == 1

            LocationService.get_node_locations({"min_age_hours": 24})
            assert repo_mock.call_count == 2

            LocationService.get_node_locations({"max_age_hours": 6})
            assert repo_mock.call_count == 3

            LocationService.get_node_locations({"max_age_hours": 12})
            assert repo_mock.call_count == 4

    @pytest.mark.unit
    def test_canonicalize_node_ids(self):
        from malla.services.location_service import _canonicalize_node_ids

        assert _canonicalize_node_ids(None) is None
        assert _canonicalize_node_ids([]) is None
        assert _canonicalize_node_ids([42]) == (42,)
        assert _canonicalize_node_ids(["42"]) == (42,)
        assert _canonicalize_node_ids(["!0000002a"]) == (42,)
        assert _canonicalize_node_ids([2, 1, 2]) == (1, 2)
        assert _canonicalize_node_ids(42) == (42,)
        assert _canonicalize_node_ids("!0000002a") == (42,)

    @pytest.mark.unit
    def test_string_node_ids_normalized_before_query_and_cache(self):
        """node_ids="12" must query node 12, not the characters '1' and '2'.

        The cache key always canonicalized the string to (12,), but the
        repository used to receive the raw value and iterate it per
        character — poisoning the entry that node_ids=[12] then served.
        """
        node12 = self._make_mock_node(12)
        captured = {}

        def fake_repo(filters=None):
            filters = filters or {}
            captured.update(filters)
            nids = filters.get("node_ids")
            if isinstance(nids, str):
                # Emulate the repository's per-character iteration
                return [dict(self._make_mock_node(int(ch))) for ch in nids]
            return [dict(node12)] if nids == [12] else []

        with (
            patch(
                "malla.database.repositories.LocationRepository.get_node_locations",
                side_effect=fake_repo,
            ) as repo_mock,
            patch(
                "malla.services.traceroute_service.TracerouteService.get_network_graph_data",
                return_value={"nodes": [], "links": []},
            ),
            patch(
                "malla.services.location_service.LocationService.get_packet_links",
                return_value=[],
            ),
        ):
            res_str = LocationService.get_node_locations({"node_ids": "12"})
            assert [loc["node_id"] for loc in res_str] == [12]
            assert captured["node_ids"] == [12]

            # The [12] request shares the cache entry: same result, and
            # the repository is not re-queried
            res_list = LocationService.get_node_locations({"node_ids": [12]})
            assert [loc["node_id"] for loc in res_list] == [12]
            assert repo_mock.call_count == 1

    @pytest.mark.unit
    def test_gateway_id_spellings_share_normalized_query_and_cache(self):
        """gateway_id=42 and "!0000002a" must query and cache identically.

        Both spellings canonicalize to node 42 in the cache key; the
        repository must therefore also see the same (canonical '!hex')
        value instead of one spelling matching rows the other misses.
        """
        node42 = self._make_mock_node(42)
        captured = []
        packet_filters_seen = []

        def fake_repo(filters=None):
            captured.append(dict(filters or {}))
            return [dict(node42)]

        with (
            patch(
                "malla.database.repositories.LocationRepository.get_node_locations",
                side_effect=fake_repo,
            ) as repo_mock,
            patch(
                "malla.services.traceroute_service.TracerouteService.get_network_graph_data",
                return_value={"nodes": [], "links": []},
            ),
            patch(
                "malla.services.location_service.LocationService.get_packet_links",
                side_effect=lambda filters=None: packet_filters_seen.append(
                    dict(filters or {})
                )
                or [],
            ),
        ):
            LocationService.get_node_locations({"gateway_id": 42})
            LocationService.get_node_locations({"gateway_id": "!0000002a"})

        # The repository saw the canonical TEXT form both times...
        assert len(captured) == 1  # second spelling hit the cache entry
        assert captured[0]["gateway_id"] == "!0000002a"
        # ...and so did the internally derived packet-link filters
        assert packet_filters_seen[0]["gateway_id"] == "!0000002a"
        assert repo_mock.call_count == 1
