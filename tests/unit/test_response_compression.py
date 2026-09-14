"""Tests for HTTP response compression of large HTML/JSON payloads."""

import os
import tempfile

from malla.config import AppConfig, _clear_config_cache
from malla.web_ui import create_app
from tests.fixtures.database_fixtures import DatabaseFixtures


def _client():
    """Build a Flask test client backed by a temporary fixture database."""

    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    cfg = AppConfig()
    cfg.database_file = temp_db.name
    DatabaseFixtures().create_test_database(temp_db.name)
    app = create_app(cfg)
    app.config["TESTING"] = True
    return app.test_client(), temp_db.name


def test_large_responses_compressed_when_client_offers_gzip():
    """Pages and JSON APIs are compressed when the client offers gzip."""

    client, db_path = _client()
    try:
        html_response = client.get(
            "/nodes", headers={"Accept-Encoding": "gzip"}
        )
        assert html_response.status_code == 200
        assert html_response.headers["Content-Encoding"] == "gzip"

        json_response = client.get("/api/nodes", headers={"Accept-Encoding": "gzip"})
        assert json_response.status_code == 200
        assert json_response.headers["Content-Encoding"] == "gzip"

        # The compressed body must be smaller than the identity payload.
        import gzip as gzip_module

        compressed = json_response.get_data()
        decompressed = gzip_module.decompress(compressed)
        assert len(compressed) == int(json_response.headers["Content-Length"])
        assert len(decompressed) > len(compressed)
    finally:
        _clear_config_cache()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass


def test_responses_uncompressed_without_accept_encoding():
    """No compression is applied when the client does not offer it."""

    client, db_path = _client()
    try:
        response = client.get("/api/nodes")
        assert response.status_code == 200
        assert "Content-Encoding" not in response.headers
    finally:
        _clear_config_cache()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass


def test_small_payloads_not_compressed():
    """Responses below the compression threshold pass through untouched."""

    client, db_path = _client()
    try:
        response = client.get("/health", headers={"Accept-Encoding": "gzip"})
        assert response.status_code == 200
        assert "Content-Encoding" not in response.headers
    finally:
        _clear_config_cache()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass
