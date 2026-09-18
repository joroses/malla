"""Reader for per-gateway broadcast text-message reception reliability.

Selecting a node on the map asks: of the broadcast TEXT_MESSAGE_APP
transmissions the node originated inside the selected time window, what
fraction did each MQTT gateway hear? The answer feeds the per-gateway
percentage badges rendered on the map while a node is selected.

A transmission is identified by its resolved ``tx_id`` (the mesh packet
id shared by every reception of one flood), so duplicate MQTT publishes
of the same reception and multi-hop re-transmissions heard by a gateway
count once. Receptions count at any hop depth: a message relayed to a
gateway still reached it.

Reads the materialized ``packet_observations`` projection when populated
(written at capture time / by the backfill tool) and falls back to a raw
``packet_history`` scan otherwise; both paths produce identical results.
"""

import logging
import sqlite3
from typing import Any

from .repositories import derived_table_populated

logger = logging.getLogger(__name__)

# PortNum.TEXT_MESSAGE_APP; the numeric column is authoritative because
# portnum_name is not populated on every database (e.g. the import DB).
TEXT_MESSAGE_PORTNUM = 1

# 0xFFFFFFFF — the mesh-wide broadcast destination.
BROADCAST_NODE_ID = 4294967295


def get_text_message_reliability(
    cursor: sqlite3.Cursor,
    node_id: int,
    start_time: float | None = None,
    end_time: float | None = None,
) -> dict[str, Any]:
    """Per-gateway reception statistics for a node's broadcast text messages.

    Args:
        cursor: Database cursor on a connection with Row factory.
        node_id: Numeric node ID of the transmitting node.
        start_time: Optional window lower bound (epoch seconds, inclusive).
        end_time: Optional window upper bound (epoch seconds, inclusive).

    Returns:
        Dict with ``node_id``, ``total_sent`` (distinct broadcast text
        transmissions observed by any gateway) and ``gateways`` — one
        entry per receiving gateway with the numeric gateway node id,
        distinct transmissions received and the rounded percentage, sorted
        by percentage descending. ``percent`` is 0 for every gateway when
        nothing was sent.
    """
    where_clauses = [
        "from_node_id = ?",
        "portnum = ?",
        "to_node_id = ?",
        "gateway_id != ''",
        # A gateway that shares the sender's node id only heard its own
        # transmission loop back; it is not an independent reception.
        "lower(gateway_id) != printf('!%08x', from_node_id)",
    ]
    params: list[Any] = [node_id, TEXT_MESSAGE_PORTNUM, BROADCAST_NODE_ID]

    if start_time is not None:
        where_clauses.append("timestamp >= ?")
        params.append(start_time)
    if end_time is not None:
        where_clauses.append("timestamp <= ?")
        params.append(end_time)
    filter_sql = " AND ".join(where_clauses)

    if derived_table_populated(cursor, "packet_observations"):
        logger.debug(
            "text-message reliability: reading materialized packet_observations"
        )
        source_sql = f"""
            SELECT tx_id, gateway_id
            FROM packet_observations
            WHERE {filter_sql}
        """
    else:
        # Legacy fallback over raw receptions. Check if the
        # mesh_packet_id column is present to safely support in-memory
        # test databases or older schemas.
        cursor.execute("PRAGMA table_info(packet_history)")
        ph_columns = {r[1] for r in cursor.fetchall()}
        tx_id_expr = (
            "COALESCE(NULLIF(mesh_packet_id, 0), -id)"
            if "mesh_packet_id" in ph_columns
            else "-id"
        )
        source_sql = f"""
            SELECT {tx_id_expr} AS tx_id, gateway_id
            FROM packet_history
            WHERE gateway_id IS NOT NULL
              AND {filter_sql}
        """

    # One row per distinct (transmission, gateway) pair; the denominator
    # counts the union of transmissions across every gateway, so a
    # message only one gateway heard still counts as sent.
    query = f"""
        WITH tx AS (
            SELECT DISTINCT tx_id, gateway_id
            FROM ({source_sql})
        )
        SELECT
            gateway_id,
            COUNT(*) AS received,
            (SELECT COUNT(DISTINCT tx_id) FROM tx) AS total_sent
        FROM tx
        GROUP BY gateway_id
        ORDER BY received DESC
    """
    cursor.execute(query, params)
    rows = cursor.fetchall()

    gateways: list[dict[str, Any]] = []
    total_sent = 0
    for row in rows:
        total_sent = max(total_sent, row["total_sent"] or 0)
        gw_id_raw = row["gateway_id"]
        try:
            gateway_node_id = int(gw_id_raw.lstrip("!"), 16)
        except ValueError:
            logger.warning(
                "Skipping gateway id that could not be parsed: %s", gw_id_raw
            )
            continue
        gateways.append(
            {
                "gateway_node_id": gateway_node_id,
                "gateway_id": gw_id_raw,
                "received": row["received"],
            }
        )

    for gateway in gateways:
        gateway["percent"] = (
            round(gateway["received"] * 100 / total_sent) if total_sent else 0
        )
    gateways.sort(key=lambda g: (-g["percent"], -g["received"]))

    return {
        "node_id": node_id,
        "total_sent": total_sent,
        "gateways": gateways,
    }
