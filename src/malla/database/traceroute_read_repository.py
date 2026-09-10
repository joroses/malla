"""Queries for materialized traceroutes used by web readers."""

from __future__ import annotations

import json
import time
from typing import Any

from ..utils.signal_quality import (
    SNR_PLAUSIBLE_MAX,
    SNR_PLAUSIBLE_MIN,
    TRACEROUTE_UNKNOWN_SNR,
    rssi_valid_sql,
    snr_valid_sql,
)
from .connection import get_db_connection
from .traceroutes import PARSER_VERSION


def _where_clause(
    filters: dict[str, Any],
    search: str | None,
    *,
    include_success_filter: bool = True,
) -> tuple[str, list[Any]]:
    conditions = ["r.parser_version = ?", "r.parse_status != 'pending'"]
    params: list[Any] = [PARSER_VERSION]
    field_filters = (
        ("start_time", "r.timestamp >= ?"),
        ("end_time", "r.timestamp <= ?"),
        ("from_node", "r.from_node_id = ?"),
        ("to_node", "r.to_node_id = ?"),
        ("gateway_id", "p.gateway_id = ?"),
        ("primary_channel", "p.channel_id = ?"),
    )
    for key, sql in field_filters:
        value = filters.get(key)
        if value is not None and value != "":
            conditions.append(sql)
            params.append(value)
    if include_success_filter and (
        filters.get("processed_successfully_only") or filters.get("success_only")
    ):
        conditions.append("r.parse_status IN ('parsed', 'valid_empty')")
    if filters.get("return_path_only"):
        conditions.append("r.route_back_json != '[]'")
    route_node = filters.get("route_node")
    if route_node is not None and route_node != "":
        try:
            route_node = int(route_node)
        except (ValueError, TypeError):
            try:
                if isinstance(route_node, str) and route_node.startswith(("0x", "0X")):
                    route_node = int(route_node, 16)
            except (ValueError, TypeError):
                pass
        conditions.append(
            "(r.from_node_id = ? OR r.to_node_id = ? "
            "OR EXISTS (SELECT 1 FROM traceroute_hops h WHERE h.packet_id = r.packet_id "
            "AND (h.from_node_id = ? OR h.to_node_id = ?)) "
            "OR EXISTS (SELECT 1 FROM json_each(r.route_nodes_json) WHERE value = ?) "
            "OR EXISTS (SELECT 1 FROM json_each(r.route_back_json) WHERE value = ?))"
        )
        params.extend([route_node] * 6)
    if search:
        conditions.append(
            "(p.gateway_id LIKE ? OR CAST(r.from_node_id AS TEXT) LIKE ? "
            "OR CAST(r.to_node_id AS TEXT) LIKE ?)"
        )
        params.extend([f"%{search}%"] * 3)
    return " AND ".join(conditions), params


_SORT_EXPRESSIONS = {
    "timestamp": "timestamp",
    "from_node_id": "from_node_id",
    "to_node_id": "to_node_id",
    "gateway_id": "gateway_count",
    "gateway_count": "gateway_count",
    "rssi": "min_rssi",
    "snr": "min_snr",
    "hop_count": "min_hops",
    "payload_length": "avg_payload_length",
}


def get_traceroute_packets(
    *,
    limit: int = 100,
    offset: int = 0,
    filters: dict[str, Any] | None = None,
    order_by: str = "timestamp",
    order_dir: str = "desc",
    search: str | None = None,
    group_packets: bool = False,
) -> dict[str, Any]:
    """Return exact, post-filter pagination from saved route rows."""
    filters = dict(filters or {})
    if group_packets and "start_time" not in filters and "end_time" not in filters:
        filters["start_time"] = time.time() - 7 * 24 * 3600
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        where, params = _where_clause(filters, search)
        direction = "ASC" if order_dir.lower() == "asc" else "DESC"
        sort_expression = _SORT_EXPRESSIONS.get(order_by, "timestamp")

        if group_packets:
            group_conditions = ["r.mesh_packet_id IS NOT NULL", "r.mesh_packet_id != 0"]
            grouped_where = f"{where} AND {' AND '.join(group_conditions)}"
            group_key = "mesh_packet_id, from_node_id, to_node_id"
            total_count = cursor.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT 1
                    FROM traceroute_routes r
                    JOIN packet_history p ON p.id = r.packet_id
                    WHERE {grouped_where}
                    GROUP BY r.mesh_packet_id, r.from_node_id, r.to_node_id
                )
                """,
                params,
            ).fetchone()[0]
            rows = cursor.execute(
                f"""
                WITH filtered AS (
                    SELECT
                        p.id, r.timestamp, r.from_node_id, r.to_node_id,
                        p.gateway_id, p.channel_id, p.hop_start, p.hop_limit,
                        p.rssi, p.snr, p.payload_length, p.processed_successfully,
                        r.mesh_packet_id, r.route_nodes_json, r.snr_towards_json,
                        r.route_back_json, r.snr_back_json, r.parse_status,
                        ROW_NUMBER() OVER (
                            PARTITION BY r.mesh_packet_id, r.from_node_id, r.to_node_id
                            ORDER BY p.payload_length DESC, r.timestamp DESC, p.id DESC
                        ) AS representative_rank
                    FROM traceroute_routes r
                    JOIN packet_history p ON p.id = r.packet_id
                    WHERE {grouped_where}
                ), grouped AS (
                    SELECT
                        MAX(CASE WHEN representative_rank = 1 THEN id END) AS id,
                        MAX(timestamp) AS timestamp,
                        from_node_id, to_node_id, mesh_packet_id,
                        COUNT(DISTINCT gateway_id) AS gateway_count,
                        GROUP_CONCAT(DISTINCT gateway_id) AS gateway_list,
                        COUNT(*) AS reception_count,
                        MAX(processed_successfully) AS processed_successfully,
                        MAX(CASE WHEN representative_rank = 1 THEN channel_id END) AS channel_id,
                        MIN(CASE WHEN {rssi_valid_sql()} THEN rssi END) AS min_rssi,
                        MAX(CASE WHEN {rssi_valid_sql()} THEN rssi END) AS max_rssi,
                        MIN(CASE WHEN {snr_valid_sql()} THEN snr END) AS min_snr,
                        MAX(CASE WHEN {snr_valid_sql()} THEN snr END) AS max_snr,
                        MIN(hop_start - hop_limit) AS min_hops,
                        MAX(hop_start - hop_limit) AS max_hops,
                        AVG(payload_length) AS avg_payload_length,
                        MAX(CASE WHEN representative_rank = 1 THEN route_nodes_json END) AS route_nodes_json,
                        MAX(CASE WHEN representative_rank = 1 THEN snr_towards_json END) AS snr_towards_json,
                        MAX(CASE WHEN representative_rank = 1 THEN route_back_json END) AS route_back_json,
                        MAX(CASE WHEN representative_rank = 1 THEN snr_back_json END) AS snr_back_json,
                        MAX(CASE WHEN representative_rank = 1 THEN parse_status END) AS parse_status
                    FROM filtered
                    GROUP BY {group_key}
                )
                SELECT *, datetime(timestamp, 'unixepoch') AS timestamp_str
                FROM grouped
                ORDER BY {sort_expression} {direction}, id {direction}
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        else:
            total_count = cursor.execute(
                f"""
                SELECT COUNT(*)
                FROM traceroute_routes r
                JOIN packet_history p ON p.id = r.packet_id
                WHERE {where}
                """,
                params,
            ).fetchone()[0]
            ungrouped_sort = {
                "gateway_count": "p.gateway_id",
                "min_rssi": "p.rssi",
                "min_snr": "p.snr",
                "min_hops": "(p.hop_start - p.hop_limit)",
                "avg_payload_length": "p.payload_length",
            }.get(sort_expression, f"r.{sort_expression}")
            rows = cursor.execute(
                f"""
                SELECT
                    p.id, r.timestamp, r.from_node_id, r.to_node_id,
                    p.gateway_id, p.channel_id, p.hop_start, p.hop_limit,
                    p.rssi, p.snr, p.payload_length, p.processed_successfully,
                    r.mesh_packet_id, r.route_nodes_json, r.snr_towards_json,
                    r.route_back_json, r.snr_back_json, r.parse_status,
                    r.route_nodes_json AS route,
                    (p.hop_start - p.hop_limit) AS hop_count,
                    datetime(r.timestamp, 'unixepoch') AS timestamp_str
                FROM traceroute_routes r
                JOIN packet_history p ON p.id = r.packet_id
                WHERE {where}
                ORDER BY {ungrouped_sort} {direction}, p.id {direction}
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()

        packets = [dict(row) for row in rows]
        for packet in packets:
            packet["route"] = packet["route_nodes_json"]
            if group_packets:
                packet["is_grouped"] = True
                packet["rssi_range"] = _range(packet["min_rssi"], packet["max_rssi"], "dBm", 1)
                packet["snr_range"] = _range(packet["min_snr"], packet["max_snr"], "dB", 2)
                packet["hop_range"] = _range(packet["min_hops"], packet["max_hops"], "", 0)
                packet["rssi"] = packet["rssi_range"]
                packet["snr"] = packet["snr_range"]
                packet["hop_count"] = packet["min_hops"]
        return {
            "packets": packets,
            "total_count": total_count,
        }
    finally:
        conn.close()


def get_traceroute_link(
    node1_id: int,
    node2_id: int,
    *,
    start_time: float,
    end_time: float,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    """Return aggregate link statistics and one page of matching traceroutes.

    The hop table identifies matches without decoding or scanning traceroute
    payloads. Statistics cover every matching hop in the requested window, while
    the detail query returns each matching packet once.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        match_sql = """
            h.timestamp >= ? AND h.timestamp <= ?
            AND ((h.from_node_id = ? AND h.to_node_id = ?)
                 OR (h.from_node_id = ? AND h.to_node_id = ?))
        """
        match_params = [
            start_time,
            end_time,
            node1_id,
            node2_id,
            node2_id,
            node1_id,
        ]
        stats = cursor.execute(
            f"""
            SELECT
                COUNT(DISTINCT h.packet_id) AS total_attempts,
                SUM(CASE WHEN h.from_node_id = ? AND h.to_node_id = ?
                         THEN 1 ELSE 0 END) AS forward_count,
                SUM(CASE WHEN h.from_node_id = ? AND h.to_node_id = ?
                         THEN 1 ELSE 0 END) AS reverse_count,
                AVG(CASE WHEN h.snr = {TRACEROUTE_UNKNOWN_SNR}
                              OR h.snr BETWEEN {SNR_PLAUSIBLE_MIN} AND {SNR_PLAUSIBLE_MAX}
                         THEN h.snr END) AS avg_snr
            FROM traceroute_hops h
            JOIN traceroute_routes r ON r.packet_id = h.packet_id
            WHERE r.parser_version = ? AND r.parse_status = 'parsed'
              AND {match_sql}
            """,
            [
                node1_id,
                node2_id,
                node2_id,
                node1_id,
                PARSER_VERSION,
                *match_params,
            ],
        ).fetchone()

        total_count = int(stats["total_attempts"] or 0)
        rows = cursor.execute(
            f"""
            WITH matching AS (
                SELECT
                    h.packet_id,
                    h.from_node_id AS target_from_node_id,
                    h.to_node_id AS target_to_node_id,
                    h.snr AS target_hop_snr,
                    ROW_NUMBER() OVER (
                        PARTITION BY h.packet_id
                        ORDER BY CASE h.direction WHEN 'forward' THEN 0 ELSE 1 END,
                                 h.hop_index
                    ) AS target_rank
                FROM traceroute_hops h
                JOIN traceroute_routes r ON r.packet_id = h.packet_id
                WHERE r.parser_version = ? AND r.parse_status = 'parsed'
                  AND {match_sql}
            )
            SELECT
                p.id, r.timestamp, r.from_node_id, r.to_node_id,
                p.gateway_id, p.channel_id, p.hop_start, p.hop_limit,
                p.rssi, p.snr, p.payload_length, p.processed_successfully,
                r.mesh_packet_id, r.route_nodes_json, r.snr_towards_json,
                r.route_back_json, r.snr_back_json, r.parse_status,
                m.target_from_node_id, m.target_to_node_id, m.target_hop_snr,
                datetime(r.timestamp, 'unixepoch') AS timestamp_str
            FROM matching m
            JOIN traceroute_routes r ON r.packet_id = m.packet_id
            JOIN packet_history p ON p.id = m.packet_id
            WHERE m.target_rank = 1
            ORDER BY r.timestamp DESC, p.id DESC
            LIMIT ? OFFSET ?
            """,
            [PARSER_VERSION, *match_params, limit, offset],
        ).fetchall()

        return {
            "packets": [dict(row) for row in rows],
            "total_count": total_count,
            "total_attempts": total_count,
            "forward_count": int(stats["forward_count"] or 0),
            "reverse_count": int(stats["reverse_count"] or 0),
            "avg_snr": stats["avg_snr"],
        }
    finally:
        conn.close()


def _range(minimum: Any, maximum: Any, unit: str, decimals: int) -> str | None:
    if minimum is None or maximum is None:
        return None
    suffix = f" {unit}" if unit else ""
    if decimals:
        low = f"{minimum:.{decimals}f}"
        high = f"{maximum:.{decimals}f}"
    else:
        low, high = str(minimum), str(maximum)
    return f"{low}{suffix}" if minimum == maximum else f"{low}-{high}{suffix}"


def route_data_from_row(packet: dict[str, Any]) -> dict[str, list[Any]] | None:
    """Decode the four saved JSON arrays without touching the packet payload."""
    keys = ("route_nodes", "snr_towards", "route_back", "snr_back")
    try:
        return {key: json.loads(packet[f"{key}_json"]) for key in keys}
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
