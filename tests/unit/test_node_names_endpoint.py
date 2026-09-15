"""
Tests for lightweight /api/nodes/names endpoint and NodeRepository.get_node_names.
"""

import sqlite3

import pytest

from malla.config import AppConfig, _override_config
from malla.database.repositories import NodeRepository


class TestNodeNamesRepository:
    """Unit tests for NodeRepository.get_node_names."""

    def test_get_node_names_empty_db(self, tmp_path):
        """Returns empty list when node_info table does not exist."""
        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.close()

        cfg = AppConfig(database_file=db_path)
        _override_config(cfg)

        nodes = NodeRepository.get_node_names()
        assert nodes == []

    def test_get_node_names_success(self, tmp_path):
        """Returns lightweight node identity objects with expected fields."""
        db_path = str(tmp_path / "test_nodes.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE node_info (
                node_id INTEGER PRIMARY KEY,
                long_name TEXT,
                short_name TEXT,
                hw_model TEXT,
                role TEXT,
                primary_channel TEXT,
                last_updated REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO node_info (node_id, long_name, short_name, hw_model, role, primary_channel, last_updated)
            VALUES
                (12345678, 'Node Alpha', 'ALPH', 'T-Beam', 'ROUTER', 'LongFast', 1700000001),
                (87654321, 'Node Beta', 'BETA', 'Heltec', 'CLIENT', 'LongFast', 1700000002)
            """
        )
        conn.commit()
        conn.close()

        cfg = AppConfig(database_file=db_path)
        _override_config(cfg)

        nodes = NodeRepository.get_node_names(limit=10)
        assert len(nodes) == 2

        # Most recently updated node first
        assert nodes[0]["node_id"] == 87654321
        assert nodes[0]["long_name"] == "Node Beta"
        assert nodes[0]["short_name"] == "BETA"
        assert nodes[0]["hw_model"] == "Heltec"
        assert nodes[0]["hex_id"] == f"!{87654321:08x}"

        assert nodes[1]["node_id"] == 12345678
        assert nodes[1]["hex_id"] == f"!{12345678:08x}"

    def test_get_node_names_limit(self, tmp_path):
        """Respects the limit argument."""
        db_path = str(tmp_path / "test_limit.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE node_info (
                node_id INTEGER PRIMARY KEY,
                long_name TEXT,
                short_name TEXT,
                hw_model TEXT,
                last_updated REAL
            )
            """
        )
        for i in range(10):
            conn.execute(
                "INSERT INTO node_info VALUES (?, ?, ?, ?, ?)",
                (i + 1, f"Node {i}", f"N{i}", "T-Beam", 1700000000 + i),
            )
        conn.commit()
        conn.close()

        cfg = AppConfig(database_file=db_path)
        _override_config(cfg)

        nodes = NodeRepository.get_node_names(limit=3)
        assert len(nodes) == 3


class TestNodeNamesEndpoint:
    """Integration tests for /api/nodes/names endpoint."""

    @pytest.mark.integration
    @pytest.mark.api
    def test_api_nodes_names_basic(self, client):
        """Test /api/nodes/names returns 200 with expected structure."""
        response = client.get("/api/nodes/names")
        assert response.status_code == 200

        data = response.get_json()
        assert "nodes" in data
        assert "total_count" in data
        assert isinstance(data["nodes"], list)
        assert data["total_count"] == len(data["nodes"])

        if len(data["nodes"]) > 0:
            node = data["nodes"][0]
            for key in ["node_id", "hex_id", "long_name", "short_name", "hw_model"]:
                assert key in node
            # Ensure expensive aggregates are NOT in the payload
            assert "avg_rssi" not in node
            assert "avg_snr" not in node
            assert "packet_count_24h" not in node
