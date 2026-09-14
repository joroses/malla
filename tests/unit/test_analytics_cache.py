"""Tests for the analytics cache refresh strategy.

/api/analytics used to recompute its multi-second payload inside the
request whenever the 60s TTL lapsed — every visitor arriving more than a
minute after the previous one paid the whole compute. The service now
serves cached entries for up to _MAX_STALE_SEC while a background refresher
thread keeps them warm, so requests never sit inside a compute window.
"""

import threading
import time

import pytest

import malla.services.analytics_service as analytics_module
from malla.services.analytics_service import AnalyticsService

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_cache():
    """Isolate the class-level cache and threads between tests."""
    AnalyticsService._CACHE.clear()
    AnalyticsService._LAST_ACCESS.clear()
    analytics_module._source_decision = None
    yield
    AnalyticsService.stop_background_refresh()
    AnalyticsService._CACHE.clear()
    AnalyticsService._LAST_ACCESS.clear()
    analytics_module._source_decision = None


@pytest.fixture
def no_auto_refresh(monkeypatch):
    """Keep request paths from spawning the refresher (pure cache tests)."""
    monkeypatch.setattr(AnalyticsService, "_ensure_background_refresh", lambda *a: None)


class TestServeFromCache:
    def test_second_request_within_max_stale_served_from_cache(
        self, monkeypatch, no_auto_refresh
    ):
        marker = {"computed": 0}

        def fake_compute(*args, **kwargs):
            marker["computed"] += 1
            return {"packet_statistics": {"total_packets": marker["computed"]}}

        monkeypatch.setattr(AnalyticsService, "_compute_analytics_data", fake_compute)

        first = AnalyticsService.get_analytics_data()
        second = AnalyticsService.get_analytics_data()

        assert marker["computed"] == 1
        assert first is second

    def test_entry_past_old_ttl_still_served_without_recompute(
        self, monkeypatch, no_auto_refresh
    ):
        """The reported bug: a visitor >60s after the last one recompute-free.

        An entry older than the old 60s TTL (but younger than
        _MAX_STALE_SEC) must be served as-is; the background refresher, not
        the request, brings it back up to date.
        """

        def refuse_to_compute(*args, **kwargs):
            raise AssertionError("stale cache entry must be served, not recomputed")

        monkeypatch.setattr(
            AnalyticsService, "_compute_analytics_data", refuse_to_compute
        )

        stale = {"packet_statistics": {"total_packets": 42}}
        key = AnalyticsService._DEFAULT_CACHE_KEY
        AnalyticsService._CACHE[key] = (time.time() - 120, stale)

        assert AnalyticsService.get_analytics_data() is stale

    def test_entry_past_max_stale_recomputed_inline(
        self, monkeypatch, no_auto_refresh
    ):
        """Dead-refresher fallback: beyond _MAX_STALE_SEC a request recomputes."""
        calls = {"n": 0}

        def fake_compute(*args, **kwargs):
            calls["n"] += 1
            return {"packet_statistics": {"total_packets": calls["n"]}}

        monkeypatch.setattr(AnalyticsService, "_compute_analytics_data", fake_compute)

        AnalyticsService._CACHE[AnalyticsService._DEFAULT_CACHE_KEY] = (
            time.time() - 2 * AnalyticsService._MAX_STALE_SEC,
            {"old": True},
        )

        result = AnalyticsService.get_analytics_data()

        assert calls["n"] == 1
        assert result == {"packet_statistics": {"total_packets": 1}}
        # And the refreshed entry is served afterwards without recompute.
        assert AnalyticsService.get_analytics_data() is result
        assert calls["n"] == 1

    def test_first_request_for_new_filter_key_computes_inline(
        self, monkeypatch, no_auto_refresh
    ):
        calls = {"n": 0}

        def fake_compute(gateway_id=None, from_node=None, hop_count=None):
            calls["n"] += 1
            return {"key": (gateway_id, from_node, hop_count)}

        monkeypatch.setattr(AnalyticsService, "_compute_analytics_data", fake_compute)

        first = AnalyticsService.get_analytics_data(
            gateway_id="!aabbccdd", hop_count=2
        )
        second = AnalyticsService.get_analytics_data(
            gateway_id="!aabbccdd", hop_count=2
        )

        assert first == {"key": ("!aabbccdd", None, 2)}
        assert first is second
        assert calls["n"] == 1


class TestBackgroundRefresher:
    def test_start_warms_default_key_immediately(self, temp_database, monkeypatch):
        """First refresher cycle (the startup warm) populates the default key."""
        monkeypatch.setenv("MALLA_DATABASE_FILE", temp_database)
        AnalyticsService._CACHE.clear()

        AnalyticsService.start_background_refresh()

        deadline = time.time() + 10
        while time.time() < deadline:
            if AnalyticsService._DEFAULT_CACHE_KEY in AnalyticsService._CACHE:
                break
            time.sleep(0.02)
        assert AnalyticsService._DEFAULT_CACHE_KEY in AnalyticsService._CACHE
        assert AnalyticsService.get_analytics_data()["packet_statistics"][
            "total_packets"
        ] >= 0

    def test_refresh_keeps_recent_keys_and_evicts_idle_ones(self, monkeypatch):
        monkeypatch.setattr(
            AnalyticsService,
            "_compute_analytics_data",
            lambda *args, **kwargs: {"fresh": True},
        )
        now = time.time()
        recent_key = ("!gw", None, None)
        idle_key = ("!old", None, None)
        AnalyticsService._CACHE[recent_key] = (now - 120, {"fresh": False})
        AnalyticsService._LAST_ACCESS[recent_key] = now - 60
        AnalyticsService._CACHE[idle_key] = (now - 120, {"fresh": False})
        AnalyticsService._LAST_ACCESS[idle_key] = now - 2 * (
            AnalyticsService._KEY_IDLE_SEC + 1
        )

        AnalyticsService._refresh_active_keys()

        # Default key warmed, recent key refreshed, idle key evicted.
        assert AnalyticsService._CACHE[AnalyticsService._DEFAULT_CACHE_KEY][1] == {
            "fresh": True
        }
        assert AnalyticsService._CACHE[recent_key][1] == {"fresh": True}
        assert AnalyticsService._CACHE[recent_key][0] == pytest.approx(
            time.time(), abs=5
        )
        assert idle_key not in AnalyticsService._CACHE
        assert idle_key not in AnalyticsService._LAST_ACCESS

    def test_refresh_failure_keeps_previous_cached_value(self, monkeypatch):
        key = AnalyticsService._DEFAULT_CACHE_KEY
        AnalyticsService._CACHE[key] = (time.time(), {"previous": True})

        def exploding_compute(*args, **kwargs):
            raise RuntimeError("database unavailable")

        monkeypatch.setattr(
            AnalyticsService, "_compute_analytics_data", exploding_compute
        )
        AnalyticsService._refresh_active_keys()

        assert AnalyticsService._CACHE[key][1] == {"previous": True}

    def test_get_analytics_data_starts_refresher_and_stop_joins_it(
        self, temp_database, monkeypatch
    ):
        monkeypatch.setenv("MALLA_DATABASE_FILE", temp_database)
        AnalyticsService.get_analytics_data()

        thread = AnalyticsService._refresher_thread
        assert thread is not None and thread.is_alive()

        AnalyticsService.stop_background_refresh()
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    def test_refresher_restarts_after_stop(self, temp_database, monkeypatch):
        """Dead threads (e.g. after fork) are replaced, not latched."""
        monkeypatch.setenv("MALLA_DATABASE_FILE", temp_database)
        AnalyticsService.get_analytics_data()
        first = AnalyticsService._refresher_thread
        AnalyticsService.stop_background_refresh()

        AnalyticsService.start_background_refresh()
        second = AnalyticsService._refresher_thread

        assert second is not first and second.is_alive()
        AnalyticsService.stop_background_refresh()

    def test_repeated_start_calls_spawn_one_thread(self, monkeypatch):
        started = []
        real_start = threading.Thread.start

        def counting_start(self):
            started.append(self)
            real_start(self)

        monkeypatch.setattr(threading.Thread, "start", counting_start)
        monkeypatch.setattr(
            AnalyticsService,
            "_compute_analytics_data",
            lambda *args, **kwargs: {},
        )

        AnalyticsService.start_background_refresh()
        thread = AnalyticsService._refresher_thread
        AnalyticsService.start_background_refresh()

        assert len(started) == 1
        assert AnalyticsService._refresher_thread is thread
