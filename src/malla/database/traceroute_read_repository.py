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


def _routes_index_hint(filters: dict[str, Any]) -> str:
    """Pin a traceroute_routes index matched to the active filters.

    The planner estimates parser_version = ? as highly selective, but that
    value matches every materialized row, so it prefers
    idx_traceroute_routes_version and applies time/node constraints only via
    per-row table fetches. Forcing the (node, timestamp) indexes keeps cost
    proportional to the filter window and lets the unfiltered default view
    stream the newest routes through the timestamp index instead of sorting
    the whole table.

    A selective gateway equality on packet_history is the exception: driving
    from idx_packet_history_gateway_time_desc fetches only the matching
    receptions, which is orders of magnitude cheaper than any window scan of
    traceroute_routes. No index is pinned there so the planner stays free.
    """
    if filters.get("gateway_id"):
        return ""
    if filters.get("from_node"):
        return " INDEXED BY idx_traceroute_routes_source_time"
    if filters.get("to_node"):
        return " INDEXED BY idx_traceroute_routes_target_time"
    return " INDEXED BY idx_traceroute_routes_time"


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
        routes_index = _routes_index_hint(filters)

        if group_packets:
            group_conditions = ["r.mesh_packet_id IS NOT NULL", "r.mesh_packet_id != 0"]
            grouped_where = f"{where} AND {' AND '.join(group_conditions)}"
            group_key = "mesh_packet_id, from_node_id, to_node_id"
            total_count = cursor.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT 1
                    FROM traceroute_routes r{routes_index}
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
                    FROM traceroute_routes r{routes_index}
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
                FROM traceroute_routes r{routes_index}
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
                FROM traceroute_routes r{routes_index}
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
                packet["rssi_range"] = _range(
                    packet["min_rssi"], packet["max_rssi"], "dBm", 1
                )
                packet["snr_range"] = _range(
                    packet["min_snr"], packet["max_snr"], "dB", 2
                )
                packet["hop_range"] = _range(
                    packet["min_hops"], packet["max_hops"], "", 0
                )
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
        # Directional averages and quality aggregates follow graph eligibility:
        # zero-SNR hops and implausible SNR contribute no SNR and do not count
        # as evidenced topology observations. The -32.0 "unknown SNR" sentinel
        # counts as an observation (volume) but contributes no numeric SNR.
        # Raw observation counts and the combined average keep their historical
        # definitions. The recency rank picks the most recent channel from
        # eligible hops deterministically (timestamp, then packet id).
        stats = cursor.execute(
            f"""
            WITH matching AS (
                SELECT
                    h.packet_id,
                    h.from_node_id,
                    h.to_node_id,
                    h.snr,
                    p.channel_id,
                    r.timestamp AS route_timestamp,
                    p.id AS packet_row_id,
                    CASE WHEN (h.snr = {TRACEROUTE_UNKNOWN_SNR}
                               OR (h.snr BETWEEN {SNR_PLAUSIBLE_MIN} AND {SNR_PLAUSIBLE_MAX} AND h.snr != 0))
                              AND h.from_node_id != 4294967295 AND h.to_node_id != 4294967295
                         THEN 1 ELSE 0 END AS is_eligible,
                    ROW_NUMBER() OVER (
                        PARTITION BY CASE WHEN (h.snr = {TRACEROUTE_UNKNOWN_SNR}
                                               OR (h.snr BETWEEN {SNR_PLAUSIBLE_MIN} AND {SNR_PLAUSIBLE_MAX} AND h.snr != 0))
                                              AND h.from_node_id != 4294967295 AND h.to_node_id != 4294967295
                                         THEN 1 ELSE 0 END
                        ORDER BY r.timestamp DESC, p.id DESC
                    ) AS eligible_recency_rank
                FROM traceroute_hops h
                JOIN traceroute_routes r ON r.packet_id = h.packet_id
                JOIN packet_history p ON p.id = h.packet_id
                WHERE r.parser_version = ? AND r.parse_status = 'parsed'
                  AND {match_sql}
            )
            SELECT
                COUNT(DISTINCT packet_id) AS total_attempts,
                SUM(CASE WHEN from_node_id = ? AND to_node_id = ?
                         THEN 1 ELSE 0 END) AS forward_count,
                SUM(CASE WHEN from_node_id = ? AND to_node_id = ?
                         THEN 1 ELSE 0 END) AS reverse_count,
                SUM(CASE WHEN from_node_id = ? AND to_node_id = ? AND is_eligible = 1
                         THEN 1 ELSE 0 END) AS forward_observations,
                SUM(CASE WHEN from_node_id = ? AND to_node_id = ? AND is_eligible = 1
                         THEN 1 ELSE 0 END) AS reverse_observations,
                AVG(CASE WHEN snr BETWEEN {SNR_PLAUSIBLE_MIN} AND {SNR_PLAUSIBLE_MAX}
                              AND snr != 0
                         THEN snr END) AS avg_snr,
                AVG(CASE WHEN from_node_id = ? AND to_node_id = ?
                         AND snr BETWEEN {SNR_PLAUSIBLE_MIN} AND {SNR_PLAUSIBLE_MAX}
                         AND snr != 0
                         THEN snr END) AS forward_avg_snr,
                AVG(CASE WHEN from_node_id = ? AND to_node_id = ?
                         AND snr BETWEEN {SNR_PLAUSIBLE_MIN} AND {SNR_PLAUSIBLE_MAX}
                         AND snr != 0
                         THEN snr END) AS reverse_avg_snr,
                MAX(CASE WHEN is_eligible = 1 AND eligible_recency_rank = 1 THEN channel_id END) AS channel_id
            FROM matching
            """,
            [
                PARSER_VERSION,
                *match_params,
                node1_id,
                node2_id,
                node2_id,
                node1_id,
                node1_id,
                node2_id,
                node2_id,
                node1_id,
                node1_id,
                node2_id,
                node2_id,
                node1_id,
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
            "forward_observations": int(stats["forward_observations"] or 0),
            "reverse_observations": int(stats["reverse_observations"] or 0),
            "avg_snr": stats["avg_snr"],
            "forward_avg_snr": stats["forward_avg_snr"],
            "reverse_avg_snr": stats["reverse_avg_snr"],
            "channel_id": stats["channel_id"],
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


def _graph_hop_window(
    filters: dict[str, Any] | None,
) -> tuple[str, list[Any], str]:
    """Shared WHERE/FROM construction for the graph hop scans.

    Returns the where clause, its parameters, and the FROM clause. Hops join
    only traceroute_routes, which carries the denormalized reception
    channel_id; packet_history (its heavy raw rows) is joined solely for a
    gateway filter. Without that filter the routes join uses CROSS JOIN to pin
    hops as the outer loop: the planner otherwise drives from
    idx_traceroute_routes_version (parser_version=?), which matches every
    route, making cost independent of the time window. A selective gateway
    equality overrides that pin with plain JOINs so the planner can drive from
    idx_packet_history_gateway_time_desc and fetch only the matching
    receptions' hops instead of scanning every hop in the window.
    """
    filters = dict(filters or {})
    conditions = [
        "r.parser_version = ?",
        "r.parse_status = 'parsed'",
        "h.from_node_id != 4294967295",
        "h.to_node_id != 4294967295",
    ]
    params: list[Any] = [PARSER_VERSION]
    from_clause = (
        "traceroute_hops h CROSS JOIN traceroute_routes r ON r.packet_id = h.packet_id"
    )

    if filters.get("start_time") is not None:
        conditions.append("h.timestamp >= ?")
        params.append(filters["start_time"])
    if filters.get("end_time") is not None:
        conditions.append("h.timestamp <= ?")
        params.append(filters["end_time"])
    if filters.get("gateway_id"):
        conditions.append("p.gateway_id = ?")
        params.append(filters["gateway_id"])
        from_clause = (
            "traceroute_hops h JOIN traceroute_routes r ON r.packet_id = h.packet_id "
            "JOIN packet_history p ON p.id = h.packet_id"
        )
    if filters.get("primary_channel"):
        conditions.append("r.channel_id = ?")
        params.append(filters["primary_channel"])

    return " AND ".join(conditions), params, from_clause


def get_traceroute_hops_for_graph(
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the complete hop sequences of parsed traceroutes for graph building.

    Hops are returned in path order with their SNR untouched so callers can
    preserve path structure: dropping individual rows here (invalid SNR, weak
    links) would splice the remaining hops of one route into a shorter,
    connected-looking path. Quality filtering and continuity validation are the
    caller's responsibility.

    Each hop carries the MQTT channel_id its reception arrived on, read from
    the route's denormalized copy; consumers resolve the modem preset
    (spreading factor) from that channel hint, falling back to the configured
    preset.
    """
    where_clause, params, from_clause = _graph_hop_window(filters)
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        query = f"""
            SELECT
                h.packet_id,
                h.direction,
                h.hop_index,
                h.timestamp,
                h.from_node_id,
                h.to_node_id,
                h.snr,
                r.channel_id
            FROM {from_clause}
            WHERE {where_clause}
            ORDER BY h.timestamp DESC, h.packet_id, h.direction, h.hop_index
        """
        rows = cursor.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_traceroute_graph_aggregates(
    filters: dict[str, Any] | None = None,
    *,
    min_snr: float = -200.0,
) -> dict[str, Any]:
    """Aggregate direct-link graph data in SQL for ``include_indirect=False``.

    Buckets the window's hops per canonical node pair entirely inside
    SQLite, so a multi-day window on a busy broker costs a few thousand
    aggregate rows instead of millions of hop dicts in the worker's heap
    (the Python per-hop loop this replaces was the memory/timeout killer
    on large windows).

    Eligibility mirrors the historical Python filter exactly:

    * implausible SNR, and hops below ``min_snr`` when it is set (any value
      other than the -200 "unset" sentinel), count as ``links_filtered_by_snr``
    * plausible zero-SNR hops count as ``links_filtered_due_to_snr_0``
    * the -32.0 "unknown" sentinel is an observation that contributes no
      SNR to any average (``*_snr_count`` stays smaller than the matching
      observation count)
    * recency hints (``last_seen``/``last_packet_id``/``last_channel``)
      follow the (timestamp, packet_id) argmax among eligible hops

    Node aggregates are derived from the returned link rows by the caller:
    every eligible hop touches both endpoints of exactly one canonical
    pair and its SNR belongs to the transmitting (``from``) node, so pair
    sums fully determine per-node counts, averages, and recency.

    Stats counters cover every hop in the window (eligible or not),
    matching the previous whole-window Python statistics.
    """
    where_clause, where_params, from_clause = _graph_hop_window(filters)

    plausible_sql = (
        f"COALESCE(h.snr = {TRACEROUTE_UNKNOWN_SNR}"
        f" OR (h.snr BETWEEN {SNR_PLAUSIBLE_MIN} AND {SNR_PLAUSIBLE_MAX}), 0)"
    )
    # min_snr == -200.0 is the service's "no floor" sentinel.
    floor_sql = "1" if min_snr == -200.0 else "snr >= ?"
    floor_params: list[Any] = [] if min_snr == -200.0 else [float(min_snr)]

    window_cte = f"""
        window_hops AS (
            SELECT
                h.packet_id,
                h.direction,
                h.timestamp,
                h.from_node_id,
                h.to_node_id,
                h.snr,
                r.channel_id,
                {plausible_sql} AS snr_plausible
            FROM {from_clause}
            WHERE {where_clause}
        )
    """

    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        stats_row = cursor.execute(
            f"""
            WITH {window_cte}
            SELECT
                COUNT(*) AS total_rf_hops,
                COUNT(DISTINCT packet_id) AS packets_analyzed,
                COUNT(DISTINCT packet_id || ':' || direction)
                    AS packets_with_rf_hops,
                SUM(
                    CASE WHEN NOT snr_plausible OR NOT ({floor_sql}) THEN 1 ELSE 0 END
                ) AS links_filtered_by_snr,
                SUM(
                    CASE WHEN snr_plausible AND ({floor_sql}) AND snr = 0
                         THEN 1 ELSE 0 END
                ) AS links_filtered_due_to_snr_0
            FROM window_hops
            """,
            [*where_params, *floor_params, *floor_params],
        ).fetchone()

        link_rows = cursor.execute(
            f"""
            WITH {window_cte},
            eligible AS (
                SELECT
                    v.*,
                    CASE WHEN v.from_node_id <= v.to_node_id
                         THEN v.from_node_id ELSE v.to_node_id END AS link_source,
                    CASE WHEN v.from_node_id <= v.to_node_id
                         THEN v.to_node_id ELSE v.from_node_id END AS link_target,
                    CASE WHEN v.from_node_id <= v.to_node_id THEN 1 ELSE 0 END
                        AS is_forward,
                    (v.snr != {TRACEROUTE_UNKNOWN_SNR}) AS snr_known
                FROM window_hops v
                WHERE v.snr_plausible = 1 AND ({floor_sql}) AND v.snr != 0
            ),
            pair_ranked AS (
                SELECT
                    e.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.link_source, e.link_target
                        ORDER BY e.timestamp DESC, e.packet_id DESC
                    ) AS recency_rank
                FROM eligible e
            )
            SELECT
                link_source,
                link_target,
                SUM(is_forward) AS forward_observations,
                COALESCE(
                    SUM(CASE WHEN is_forward = 1 AND snr_known THEN snr END), 0.0
                ) AS forward_snr_sum,
                SUM(
                    CASE WHEN is_forward = 1 AND snr_known THEN 1 ELSE 0 END
                ) AS forward_snr_count,
                SUM(CASE WHEN is_forward = 0 THEN 1 ELSE 0 END) AS return_observations,
                COALESCE(
                    SUM(CASE WHEN is_forward = 0 AND snr_known THEN snr END), 0.0
                ) AS return_snr_sum,
                SUM(
                    CASE WHEN is_forward = 0 AND snr_known THEN 1 ELSE 0 END
                ) AS return_snr_count,
                COUNT(*) AS packet_count,
                MAX(CASE WHEN recency_rank = 1 THEN timestamp END) AS last_seen,
                MAX(CASE WHEN recency_rank = 1 THEN packet_id END) AS last_packet_id,
                MAX(CASE WHEN recency_rank = 1 THEN channel_id END) AS last_channel
            FROM pair_ranked
            GROUP BY link_source, link_target
            ORDER BY link_source, link_target
            """,
            [*where_params, *floor_params],
        ).fetchall()

        return {
            "links": [dict(row) for row in link_rows],
            "stats": {
                "packets_analyzed": int(stats_row["packets_analyzed"] or 0),
                "packets_with_rf_hops": int(stats_row["packets_with_rf_hops"] or 0),
                "total_rf_hops": int(stats_row["total_rf_hops"] or 0),
                "links_filtered_by_snr": int(stats_row["links_filtered_by_snr"] or 0),
                "links_filtered_due_to_snr_0": int(
                    stats_row["links_filtered_due_to_snr_0"] or 0
                ),
            },
        }
    finally:
        conn.close()


def get_traceroute_hops_for_longest_links(
    start_time: float,
    end_time: float,
) -> list[dict[str, Any]]:
    """Return all RF hops from traceroute_hops in the specified time window."""
    return get_traceroute_hops_for_graph(
        filters={"start_time": start_time, "end_time": end_time}
    )


def get_route_patterns_data(
    start_time: float,
    end_time: float,
    limit: int = 50,
) -> dict[str, Any]:
    """Aggregate route patterns directly in SQL from traceroute_routes.

    The routes table is accessed via idx_traceroute_routes_time so the scan
    covers the requested window only; the planner would otherwise drive from
    idx_traceroute_routes_version (parser_version=?), which matches every row.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        total_analyzed = cursor.execute(
            """
            SELECT COUNT(*)
            FROM traceroute_routes INDEXED BY idx_traceroute_routes_time
            WHERE parser_version = ? AND parse_status = 'parsed'
              AND timestamp >= ? AND timestamp <= ?
            """,
            (PARSER_VERSION, start_time, end_time),
        ).fetchone()[0]

        query = """
            WITH ranked_patterns AS (
                SELECT
                    p.id AS packet_id,
                    r.timestamp,
                    r.from_node_id,
                    r.to_node_id,
                    r.route_nodes_json,
                    ROW_NUMBER() OVER (
                        PARTITION BY r.from_node_id, r.to_node_id, r.route_nodes_json
                        ORDER BY r.timestamp DESC, p.id DESC
                    ) AS rank,
                    COUNT(*) OVER (
                        PARTITION BY r.from_node_id, r.to_node_id, r.route_nodes_json
                    ) AS pattern_count
                FROM traceroute_routes r INDEXED BY idx_traceroute_routes_time
                JOIN packet_history p ON p.id = r.packet_id
                WHERE r.parser_version = ? AND r.parse_status = 'parsed'
                  AND r.route_nodes_json != '[]'
                  AND r.timestamp >= ? AND r.timestamp <= ?
            )
            SELECT
                packet_id,
                timestamp,
                from_node_id,
                to_node_id,
                route_nodes_json,
                pattern_count,
                rank
            FROM ranked_patterns
            WHERE rank <= 3
            ORDER BY pattern_count DESC, from_node_id, to_node_id
        """
        rows = cursor.execute(query, (PARSER_VERSION, start_time, end_time)).fetchall()

        route_patterns: dict[
            tuple[tuple[int, int], tuple[int, ...]], dict[str, Any]
        ] = {}

        for row in rows:
            try:
                route_nodes = tuple(json.loads(row["route_nodes_json"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not route_nodes:
                continue

            endpoints = tuple(sorted([row["from_node_id"], row["to_node_id"]]))
            pattern_key = (endpoints, route_nodes)

            if pattern_key not in route_patterns:
                route_patterns[pattern_key] = {
                    "count": 0,
                    "endpoints": endpoints,
                    "examples": [],
                }

            if row["rank"] == 1:
                route_patterns[pattern_key]["count"] += row["pattern_count"]

            if len(route_patterns[pattern_key]["examples"]) < 3:
                route_patterns[pattern_key]["examples"].append(
                    {
                        "packet_id": row["packet_id"],
                        "timestamp": row["timestamp"],
                        "from_node": row["from_node_id"],
                        "to_node": row["to_node_id"],
                    }
                )

        sorted_patterns = sorted(
            route_patterns.items(), key=lambda x: x[1]["count"], reverse=True
        )[:limit]

        return {
            "sorted_patterns": sorted_patterns,
            "total_patterns": len(route_patterns),
            "analyzed_traceroutes": total_analyzed,
        }
    finally:
        conn.close()


def get_node_traceroute_statistics(
    node_id: int,
    start_time: float | None = None,
    end_time: float | None = None,
) -> dict[str, Any]:
    """Compute traceroute source, destination, and intermediate participation in SQL."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        time_conditions = []
        time_params: list[Any] = []
        if start_time is not None:
            time_conditions.append("timestamp >= ?")
            time_params.append(start_time)
        if end_time is not None:
            time_conditions.append("timestamp <= ?")
            time_params.append(end_time)

        time_sql = f" AND {' AND '.join(time_conditions)}" if time_conditions else ""

        # Query 1: As source and as destination
        endpoint_query = f"""
            SELECT
                COUNT(CASE WHEN from_node_id = ? THEN 1 END) AS source_total,
                COUNT(CASE WHEN from_node_id = ? AND parse_status IN ('parsed', 'valid_empty') THEN 1 END) AS source_successful,
                COUNT(CASE WHEN to_node_id = ? THEN 1 END) AS dest_total,
                COUNT(CASE WHEN to_node_id = ? AND parse_status IN ('parsed', 'valid_empty') THEN 1 END) AS dest_successful
            FROM traceroute_routes
            WHERE parser_version = ?
              AND (from_node_id = ? OR to_node_id = ?)
              {time_sql}
        """
        endpoint_params = [
            node_id,
            node_id,
            node_id,
            node_id,
            PARSER_VERSION,
            node_id,
            node_id,
            *time_params,
        ]
        row = cursor.execute(endpoint_query, endpoint_params).fetchone()
        source_total = int(row["source_total"] or 0)
        source_successful = int(row["source_successful"] or 0)
        dest_total = int(row["dest_total"] or 0)
        dest_successful = int(row["dest_successful"] or 0)

        # Query 2: Intermediate participation. INDEXED BY keeps the scan on
        # the timestamp index (window-bounded) instead of the version index,
        # which matches every row and decouples cost from the time window.
        intermediate_query = f"""
            SELECT COUNT(*)
            FROM traceroute_routes INDEXED BY idx_traceroute_routes_time
            WHERE parser_version = ?
              AND parse_status IN ('parsed', 'valid_empty')
              {time_sql}
              AND EXISTS (
                  SELECT 1 FROM json_each(route_nodes_json) WHERE value = ?
              )
        """
        intermediate_params = [
            PARSER_VERSION,
            *time_params,
            node_id,
        ]
        participation_count = cursor.execute(
            intermediate_query, intermediate_params
        ).fetchone()[0]

        return {
            "node_id": node_id,
            "as_source": {
                "total": source_total,
                "successful": source_successful,
            },
            "as_destination": {
                "total": dest_total,
                "successful": dest_successful,
            },
            "as_intermediate_hop": {"participation_count": participation_count},
            "total_involvement": source_total + dest_total + participation_count,
        }
    finally:
        conn.close()
