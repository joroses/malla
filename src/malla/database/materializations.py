"""Shared capture/backfill writers for the materialized packet tables.

The derived tables project raw receptions once, at write time, into lean
indexed tables instead of re-scanning heavy Protobuf payloads in packet_history:

- ``packet_observations``: one row per raw reception keyed by packet_id,
  with pre-computed transmission identity (tx_id) and 0-hop direct flag
  (is_direct). Preserves exact timestamps, RSSI, SNR, hop count, and channel
  so all query filters apply with 100% fidelity before grouping.
- ``node_positions``: position fixes decoded once and validated with
  :func:`malla.utils.geo_utils.is_valid_position`; invalid fixes (null
  island) are never stored, which makes the read-side fallback-to-previous-
  valid-fix behaviour automatic.
- ``traceroute_routes`` / ``traceroute_hops``: traceroute discovery packets
  decoded into structured paths and radio hops with directional SNR.

Observation identity is ``tx_id``: the mesh packet id correlated within an ID
reuse window, or ``-packet_history.id`` when the sender assigned none. The
negative fallback can never collide with the uint32 mesh id space.

Writers run inside the caller's transaction (like
:func:`malla.database.traceroutes.write_traceroute`): raw and derived data
stay atomic and storage errors are never swallowed. Writers that bypass
these helpers leave gaps, not corruption; re-derive history with
``python -m malla.backfill_materializations``.

Transmission identities resolved inside the caller's transaction are staged
in a pending overlay and published to the shared cache only once that
transaction commits (:func:`materialization_tx_scope`); a rollback discards
them, so a failed write can never leak its packet ids into unrelated
transmissions.
"""

import sqlite3
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager

from google.protobuf.message import DecodeError
from meshtastic import mesh_pb2

from ..utils.geo_utils import is_valid_position
from .traceroute_schema import TRACEROUTE_PORT
from .traceroutes import write_traceroute

POSITION_PORT = 3

# Default threshold for Meshtastic packet ID reuse (1 hour). Duplicate
# receptions of an active transmission arrive within seconds to minutes;
# identical packet IDs after this window are treated as distinct transmissions.
ID_REUSE_WINDOW_SECONDS: float = 3600.0

# Single watermark row for the combined history walk.
BACKFILL_STATE_NAME = "packet_materializations"

# LRU cache for active transmission identities during live capture / backfill:
# (from_node_id, mesh_packet_id, to_node_id, portnum) -> (tx_id, last_seen)
_TxKey = tuple[int, int, int | None, int | None]
_TxEntry = tuple[int, float]

_TX_CACHE_MAX_SIZE = 2000
_tx_cache: OrderedDict[_TxKey, _TxEntry] = OrderedDict()

# Identities resolved during the caller's open transaction. Published to
# _tx_cache only on commit; discarded on rollback so a failed write cannot
# hand its packet_history ids to receptions of unrelated transmissions.
_tx_pending: dict[_TxKey, _TxEntry] | None = None
_tx_pending_conn: sqlite3.Connection | None = None


def _discard_stale_pending(conn: sqlite3.Connection) -> None:
    """Drop staged identities when a previous transaction went unpublished."""

    global _tx_pending
    if _tx_pending is not None and _tx_pending_conn is not conn:
        _tx_pending = None


def _stage_pending(conn: sqlite3.Connection, key: _TxKey, entry: _TxEntry) -> None:
    global _tx_pending, _tx_pending_conn
    if _tx_pending is None:
        _tx_pending = {}
        _tx_pending_conn = conn
    _tx_pending[key] = entry


def clear_materialization_cache() -> None:
    """Clear in-memory transmission correlation caches."""
    global _tx_pending, _tx_pending_conn
    _tx_cache.clear()
    _tx_pending = None
    _tx_pending_conn = None


def commit_materialization_cache() -> None:
    """Publish identities staged during the transaction that just committed."""
    global _tx_pending, _tx_pending_conn
    if _tx_pending:
        for key, entry in _tx_pending.items():
            _tx_cache[key] = entry
            _tx_cache.move_to_end(key)
        while len(_tx_cache) > _TX_CACHE_MAX_SIZE:
            _tx_cache.popitem(last=False)
    _tx_pending = None
    _tx_pending_conn = None


def rollback_materialization_cache() -> None:
    """Discard identities staged during the transaction that rolled back."""
    global _tx_pending, _tx_pending_conn
    _tx_pending = None
    _tx_pending_conn = None


@contextmanager
def materialization_tx_scope() -> Iterator[None]:
    """Publish staged transmission identities only after commit.

    Wrap exactly one caller transaction, immediately outside its
    ``with conn:`` block: the SQL commit or rollback happens first, then
    staged identities are published on commit and discarded on any error.
    Identities staged before the scope opens (e.g. by a writer whose
    transaction never committed) are discarded as untrusted.
    """
    rollback_materialization_cache()
    try:
        yield
    except BaseException:
        rollback_materialization_cache()
        raise
    else:
        commit_materialization_cache()


OBSERVATION_INSERT_SQL = """
    INSERT INTO packet_observations (
        packet_id, tx_id, timestamp, from_node_id, to_node_id, portnum, portnum_name,
        gateway_id, channel_id, mesh_packet_id, rssi, snr, hop_start, hop_limit,
        is_direct, payload_length, processed_successfully, relay_node
    ) VALUES (
        ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?
    )
    ON CONFLICT (packet_id) DO UPDATE SET
        tx_id = excluded.tx_id,
        timestamp = excluded.timestamp,
        from_node_id = excluded.from_node_id,
        to_node_id = excluded.to_node_id,
        portnum = excluded.portnum,
        portnum_name = excluded.portnum_name,
        gateway_id = excluded.gateway_id,
        channel_id = excluded.channel_id,
        mesh_packet_id = excluded.mesh_packet_id,
        rssi = excluded.rssi,
        snr = excluded.snr,
        hop_start = excluded.hop_start,
        hop_limit = excluded.hop_limit,
        is_direct = excluded.is_direct,
        payload_length = excluded.payload_length,
        processed_successfully = excluded.processed_successfully,
        relay_node = excluded.relay_node
"""

# Backwards-compatibility aliases
OBSERVATION_UPSERT_SQL = OBSERVATION_INSERT_SQL
TRANSMISSION_UPSERT_SQL = OBSERVATION_INSERT_SQL

POSITION_INSERT_SQL = """
    INSERT INTO node_positions (
        node_id, tx_id, gateway_id, timestamp, packet_id,
        latitude, longitude, altitude, precision_bits, sats_in_view
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT (packet_id) DO UPDATE SET
        node_id = excluded.node_id,
        tx_id = excluded.tx_id,
        gateway_id = excluded.gateway_id,
        timestamp = excluded.timestamp,
        latitude = excluded.latitude,
        longitude = excluded.longitude,
        altitude = excluded.altitude,
        precision_bits = excluded.precision_bits,
        sats_in_view = excluded.sats_in_view
"""


def resolve_observation_tx_id(
    cursor: sqlite3.Cursor,
    packet: dict,
    id_reuse_window: float = ID_REUSE_WINDOW_SECONDS,
) -> int:
    """Resolve a transmission identity (tx_id) for logical grouping.

    - Positive mesh_packet_id with known from_node_id: correlated with active
      transmissions matching (from_node_id, mesh_packet_id, to_node_id, portnum)
      within id_reuse_window.
    - Missing mesh_packet_id (None or <= 0) or missing from_node_id: falls back
      to -packet["id"] so unidentifiable receptions never collide.
    """
    mesh_packet_id = packet.get("mesh_packet_id")
    from_node_id = packet.get("from_node_id")
    packet_id = int(packet["id"])

    if mesh_packet_id is None or int(mesh_packet_id) <= 0 or from_node_id is None:
        return -packet_id

    mesh_packet_id = int(mesh_packet_id)
    from_node_id = int(from_node_id)
    to_node_id = (
        int(packet["to_node_id"]) if packet.get("to_node_id") is not None else None
    )
    portnum = int(packet["portnum"]) if packet.get("portnum") is not None else None
    timestamp = float(packet.get("timestamp") or 0.0)

    cache_key = (from_node_id, mesh_packet_id, to_node_id, portnum)
    _discard_stale_pending(cursor.connection)
    cached: _TxEntry | None = (
        _tx_pending.get(cache_key) if _tx_pending is not None else None
    )
    if cached is None:
        cached = _tx_cache.get(cache_key)
    if cached is not None:
        cached_tx_id, cached_last_seen = cached
        if abs(timestamp - cached_last_seen) <= id_reuse_window:
            _stage_pending(
                cursor.connection,
                cache_key,
                (cached_tx_id, max(cached_last_seen, timestamp)),
            )
            return cached_tx_id

    # Check database index for an active transmission
    min_time = timestamp - id_reuse_window
    max_time = timestamp + id_reuse_window

    cursor.execute(
        """
        SELECT tx_id, timestamp
        FROM packet_observations
        WHERE from_node_id = ?
          AND mesh_packet_id = ?
          AND to_node_id IS ?
          AND portnum IS ?
          AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (from_node_id, mesh_packet_id, to_node_id, portnum, min_time, max_time),
    )
    row = cursor.fetchone()
    if row is not None:
        tx_id = int(row[0])
        db_timestamp = float(row[1])
        _stage_pending(
            cursor.connection, cache_key, (tx_id, max(db_timestamp, timestamp))
        )
        return tx_id

    # First reception of this transmission: packet_id becomes tx_id
    _stage_pending(cursor.connection, cache_key, (packet_id, timestamp))
    return packet_id


def observation_tx_id(
    packet: dict,
    cursor: sqlite3.Cursor | None = None,
    id_reuse_window: float = ID_REUSE_WINDOW_SECONDS,
) -> int:
    """Resolve a transmission identity (tx_id) for logical grouping.

    When *cursor* is provided, resolves against active database observations.
    When *cursor* is omitted (e.g. unit tests without DB context), falls back
    to in-memory correlation or packet-based identifiers.
    """
    if cursor is not None:
        return resolve_observation_tx_id(cursor, packet, id_reuse_window)

    mesh_packet_id = packet.get("mesh_packet_id")
    from_node_id = packet.get("from_node_id")
    packet_id = int(packet.get("id") or 0)

    if mesh_packet_id is None or int(mesh_packet_id) <= 0 or from_node_id is None:
        return -packet_id if packet_id else -1

    cache_key = (
        int(from_node_id),
        int(mesh_packet_id),
        int(packet["to_node_id"]) if packet.get("to_node_id") is not None else None,
        int(packet["portnum"]) if packet.get("portnum") is not None else None,
    )
    timestamp = float(packet.get("timestamp") or 0.0)
    cached = _tx_cache.get(cache_key)
    if cached is not None and abs(timestamp - cached[1]) <= id_reuse_window:
        return cached[0]

    return packet_id if packet_id else int(mesh_packet_id)


# Backwards-compatibility alias
transmission_tx_id = observation_tx_id


def _require_transaction(cursor: sqlite3.Cursor) -> None:
    if not cursor.connection.in_transaction:
        raise ValueError("Materialization writes require an active transaction")


def write_observation(cursor: sqlite3.Cursor, packet: dict) -> None:
    """Store one lean projected observation row for an inserted packet.

    Each reception from packet_history is projected as its own row keyed by
    packet_id, preserving individual timestamps, signal metrics, channel, and
    hop characteristics with 100% filter fidelity. Receptions of the same
    transmission (within the ID reuse window) share a resolved tx_id so read
    paths can group them dynamically with exact SQL WHERE filters.
    """
    _require_transaction(cursor)
    tx_id = resolve_observation_tx_id(cursor, packet)
    hop_start = packet.get("hop_start")
    hop_limit = packet.get("hop_limit")
    is_direct = (
        1
        if (hop_start is not None and hop_limit is not None and hop_start == hop_limit)
        else 0
    )

    cursor.execute(
        OBSERVATION_INSERT_SQL,
        (
            packet["id"],
            tx_id,
            packet["timestamp"],
            packet.get("from_node_id"),
            packet.get("to_node_id"),
            packet.get("portnum"),
            packet.get("portnum_name"),
            packet.get("gateway_id") or "",
            packet.get("channel_id"),
            packet.get("mesh_packet_id"),
            packet.get("rssi"),
            packet.get("snr"),
            hop_start,
            hop_limit,
            is_direct,
            packet.get("payload_length"),
            packet.get("processed_successfully"),
            packet.get("relay_node"),
        ),
    )


# Backwards-compatibility alias
write_transmission = write_observation


def decode_position(raw_payload: bytes | None) -> dict | None:
    """Decode a POSITION_APP payload; ``None`` unless it is a valid fix."""

    if not raw_payload:
        return None
    position = mesh_pb2.Position()
    try:
        position.ParseFromString(bytes(raw_payload))
    except DecodeError:
        return None
    latitude = position.latitude_i / 1e7 if position.latitude_i else None
    longitude = position.longitude_i / 1e7 if position.longitude_i else None
    if not is_valid_position(latitude, longitude):
        return None
    return {
        "latitude": latitude,
        "longitude": longitude,
        "altitude": position.altitude if position.altitude else None,
        "precision_bits": getattr(position, "precision_bits", None),
        "sats_in_view": getattr(position, "sats_in_view", None),
    }


def write_position(cursor: sqlite3.Cursor, packet: dict) -> None:
    """Store a position fix for one already-inserted POSITION_APP packet.

    Keyed by raw reception identity (``packet_id``): each valid position fix
    decoded from packet_history is stored once per reception. Preserves all
    historical reception windows without deduplicating subsequent fixes or
    overwriting older evidence. The latest matching fix is selected
    deterministically by ``(timestamp DESC, packet_id DESC)``.
    """

    _require_transaction(cursor)
    if packet.get("portnum") != POSITION_PORT:
        cursor.execute(
            "DELETE FROM node_positions WHERE packet_id = ?", (packet["id"],)
        )
        return
    from_node_id = packet.get("from_node_id")
    if from_node_id is None:
        cursor.execute(
            "DELETE FROM node_positions WHERE packet_id = ?", (packet["id"],)
        )
        return
    decoded = decode_position(packet.get("raw_payload"))
    if decoded is None:
        cursor.execute(
            "DELETE FROM node_positions WHERE packet_id = ?", (packet["id"],)
        )
        return
    tx_id = packet.get("mesh_packet_id") or packet["id"]
    cursor.execute(
        POSITION_INSERT_SQL,
        (
            from_node_id,
            tx_id,
            packet.get("gateway_id") or "",
            packet["timestamp"],
            packet["id"],
            decoded["latitude"],
            decoded["longitude"],
            decoded["altitude"],
            decoded["precision_bits"],
            decoded["sats_in_view"],
        ),
    )


def materialize_packet(cursor: sqlite3.Cursor, packet: dict) -> None:
    """Write every derived row for one already-inserted packet_history row.

    *packet* uses packet_history column names and must contain at least
    ``id`` and ``timestamp``. Neither commits nor swallows storage errors,
    so raw and derived data stay atomic within the caller's transaction.
    """

    write_observation(cursor, packet)
    write_position(cursor, packet)
    if (
        packet.get("portnum") == TRACEROUTE_PORT
        or packet.get("portnum_name") == "TRACEROUTE_APP"
    ):
        write_traceroute(cursor, packet)
