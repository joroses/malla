"""Integration tests for /api/nodes/data pagination, filtering, and sorting."""

import pytest


@pytest.mark.integration
@pytest.mark.api
class TestNodesDataPagination:
    """Test pagination, filtering, and sorting on /api/nodes/data."""

    def test_nodes_data_default_pagination(self, client):
        """Test default pagination behavior on /api/nodes/data."""
        response = client.get("/api/nodes/data")
        assert response.status_code == 200

        body = response.get_json()
        assert "data" in body
        assert "total_count" in body
        assert "page" in body
        assert "limit" in body
        assert "total_pages" in body

        assert body["page"] == 1
        assert body["limit"] == 100
        assert isinstance(body["data"], list)
        assert body["total_count"] >= len(body["data"])

        expected_total_pages = (
            (body["total_count"] + body["limit"] - 1) // body["limit"]
            if body["total_count"] > 0
            else 0
        )
        assert body["total_pages"] == expected_total_pages

        if body["data"]:
            node = body["data"][0]
            for key in (
                "node_id",
                "hex_id",
                "node_name",
                "hw_model",
                "role",
                "last_packet_str",
                "last_packet_time",
                "packet_count_24h",
                "status",
            ):
                assert key in node

    def test_nodes_data_custom_pagination_pages(self, client):
        """Test page traversal with small limit."""
        resp_p1 = client.get("/api/nodes/data", query_string={"page": 1, "limit": 2})
        assert resp_p1.status_code == 200
        p1_json = resp_p1.get_json()
        assert p1_json["page"] == 1
        assert p1_json["limit"] == 2

        if p1_json["total_count"] > 2:
            resp_p2 = client.get("/api/nodes/data", query_string={"page": 2, "limit": 2})
            assert resp_p2.status_code == 200
            p2_json = resp_p2.get_json()
            assert p2_json["page"] == 2

            p1_ids = {n["node_id"] for n in p1_json["data"]}
            p2_ids = {n["node_id"] for n in p2_json["data"]}
            # Pages must not overlap
            assert p1_ids.isdisjoint(p2_ids)

    def test_nodes_data_named_only_filter(self, client):
        """Test that named_only filter returns only named nodes and accurate total_count."""
        resp = client.get(
            "/api/nodes/data", query_string={"named_only": "true", "limit": 100}
        )
        assert resp.status_code == 200
        body = resp.get_json()

        for node in body["data"]:
            assert node.get("long_name") or node.get("short_name")

    def test_nodes_data_active_only_filter(self, client):
        """Test active_only filter parameter."""
        resp = client.get(
            "/api/nodes/data", query_string={"active_only": "true", "limit": 100}
        )
        assert resp.status_code == 200
        body = resp.get_json()

        for node in body["data"]:
            assert node["packet_count_24h"] > 0
            assert node["status"] == "Active"

        assert body["total_count"] == len(body["data"]) or body["total_count"] >= len(
            body["data"]
        )

    def test_nodes_data_sorting(self, client):
        """Test sorting by various columns."""
        # Sort by node_id ASC
        resp_asc = client.get(
            "/api/nodes/data",
            query_string={"sort_by": "node_id", "sort_order": "asc", "limit": 10},
        )
        assert resp_asc.status_code == 200
        asc_ids = [n["node_id"] for n in resp_asc.get_json()["data"]]
        assert asc_ids == sorted(asc_ids)

        # Sort by node_id DESC
        resp_desc = client.get(
            "/api/nodes/data",
            query_string={"sort_by": "node_id", "sort_order": "desc", "limit": 10},
        )
        assert resp_desc.status_code == 200
        desc_ids = [n["node_id"] for n in resp_desc.get_json()["data"]]
        assert desc_ids == sorted(desc_ids, reverse=True)

    def test_nodes_data_search_filter(self, client):
        """Test searching nodes."""
        all_nodes = client.get("/api/nodes/data", query_string={"limit": 5}).get_json()[
            "data"
        ]
        if not all_nodes:
            pytest.skip("No nodes available for search test")

        target = all_nodes[0]
        # Search by hex_id (without prefix !)
        clean_hex = target["hex_id"].lstrip("!")
        resp = client.get(
            "/api/nodes/data", query_string={"search": clean_hex, "limit": 10}
        )
        assert resp.status_code == 200
        data = resp.get_json()["data"]
        assert any(n["node_id"] == target["node_id"] for n in data)
