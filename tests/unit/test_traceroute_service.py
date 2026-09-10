"""
Unit tests for TracerouteService class.

Tests the business logic and service methods for traceroute analysis.
"""

from datetime import datetime
from unittest.mock import patch

from src.malla.services.traceroute_service import TracerouteService


class TestTracerouteServiceLongestLinks:
    """Test TracerouteService longest links analysis functionality."""

    @patch("src.malla.services.traceroute_service.get_bulk_node_names")
    @patch("src.malla.services.traceroute_service.LocationRepository.get_nodes_location_history")
    @patch("src.malla.services.traceroute_service.get_traceroute_hops_for_longest_links")
    def test_longest_links_analysis_basic(
        self, mock_get_hops, mock_get_locs, mock_get_names
    ):
        """Test basic longest links analysis functionality."""
        now_ts = datetime.now().timestamp()
        mock_hop = {
            "packet_id": 1,
            "direction": "forward",
            "hop_index": 0,
            "timestamp": now_ts,
            "from_node_id": 100,
            "to_node_id": 200,
            "snr": -5.0,
        }
        mock_get_hops.return_value = [mock_hop]
        mock_get_locs.return_value = {
            100: [
                {
                    "from_node_id": 100,
                    "latitude": 40.0,
                    "longitude": -3.0,
                    "altitude": 100,
                    "timestamp": now_ts,
                }
            ],
            200: [
                {
                    "from_node_id": 200,
                    "latitude": 40.045,
                    "longitude": -3.0,
                    "altitude": 100,
                    "timestamp": now_ts,
                }
            ],
        }
        mock_get_names.return_value = {100: "Node100", 200: "Node200"}

        # Call the method
        result = TracerouteService.get_longest_links_analysis(
            min_distance_km=1.0, min_snr=-10.0, max_results=10
        )

        # Verify structure
        assert "summary" in result
        assert "direct_links" in result
        assert "indirect_links" in result

        # Verify summary
        summary = result["summary"]
        assert summary["total_links"] == 1
        assert summary["direct_links"] == 1
        assert summary["longest_direct"] is not None
        assert summary["longest_path"] is None

        # Verify direct links
        assert len(result["direct_links"]) == 1
        direct_link = result["direct_links"][0]
        assert direct_link["from_node_id"] == 100
        assert direct_link["to_node_id"] == 200
        assert direct_link["distance_km"] > 4.0
        assert direct_link["avg_snr"] == -5.0
        assert direct_link["traceroute_count"] == 1

    @patch("src.malla.services.traceroute_service.get_traceroute_hops_for_longest_links")
    def test_longest_links_analysis_empty_data(self, mock_get_hops):
        """Test analysis with no traceroute data."""
        mock_get_hops.return_value = []

        # Call the method
        result = TracerouteService.get_longest_links_analysis()

        # Should return empty results with proper structure
        assert result["summary"]["total_links"] == 0
        assert result["summary"]["direct_links"] == 0
        assert result["summary"]["longest_direct"] is None
        assert result["summary"]["longest_path"] is None
        assert len(result["direct_links"]) == 0
        assert len(result["indirect_links"]) == 0

    @patch("src.malla.services.traceroute_service.get_bulk_node_names")
    @patch("src.malla.services.traceroute_service.get_node_traceroute_statistics")
    def test_node_traceroute_stats(self, mock_get_stats, mock_get_names):
        """Test node traceroute stats delegates to SQL statistics."""
        mock_get_stats.return_value = {
            "node_id": 12345,
            "as_source": {"total": 10, "successful": 8, "success_rate": 80.0},
            "as_destination": {"total": 5, "successful": 4, "success_rate": 80.0},
            "as_intermediate_hop": {"participation_count": 3},
            "total_involvement": 18,
        }
        mock_get_names.return_value = {12345: "TestNode"}

        stats = TracerouteService.get_node_traceroute_stats(12345)
        assert stats["node_id"] == 12345
        assert stats["node_name"] == "TestNode"
        assert stats["total_involvement"] == 18
