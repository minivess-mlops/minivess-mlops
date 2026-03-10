"""Tests for dashboard service adapters (T-DASH.1.1).

Validates the adapter base class and concrete adapters (MLflow, Prefect, BentoML).
All tests use mocked HTTP — no real services required.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

from minivess.dashboard.adapters.base import ServiceAdapter
from minivess.dashboard.adapters.bentoml_adapter import BentoMLAdapter
from minivess.dashboard.adapters.mlflow_adapter import MLflowAdapter
from minivess.dashboard.adapters.prefect_adapter import PrefectAdapter


class _DummyAdapter(ServiceAdapter):
    """Concrete adapter for testing the ABC."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("Dummy", "http://fake:9999", **kwargs)
        self._responses: dict[str, dict[str, Any]] = {}
        self._should_fail = False

    def set_response(self, endpoint: str, data: dict[str, Any]) -> None:
        self._responses[endpoint] = data

    def set_fail(self, fail: bool = True) -> None:
        self._should_fail = fail

    def _fetch(self, endpoint: str) -> dict[str, Any]:
        if self._should_fail:
            msg = "Connection refused"
            raise ConnectionError(msg)
        return self._responses.get(endpoint, {})


class TestAdapterBase:
    """Test the ServiceAdapter ABC contract."""

    def test_adapter_has_query_method(self) -> None:
        adapter = _DummyAdapter()
        assert hasattr(adapter, "query")
        assert callable(adapter.query)

    def test_adapter_has_status_method(self) -> None:
        adapter = _DummyAdapter()
        assert hasattr(adapter, "status")
        status = adapter.status()
        assert hasattr(status, "healthy")

    def test_adapter_cache_ttl_respected(self) -> None:
        """Second call within TTL returns cached result."""
        adapter = _DummyAdapter(cache_ttl_s=60)
        adapter.set_response("test", {"value": 42})

        result1 = adapter.query("test")
        assert result1 == {"value": 42}

        # Change response — but cache should still return old value
        adapter.set_response("test", {"value": 99})
        result2 = adapter.query("test")
        assert result2 == {"value": 42}  # cached

    def test_adapter_cache_expires(self) -> None:
        """After TTL, fresh data is fetched."""
        adapter = _DummyAdapter(cache_ttl_s=0.01)
        adapter.set_response("test", {"value": 42})
        adapter.query("test")

        time.sleep(0.02)
        adapter.set_response("test", {"value": 99})
        result = adapter.query("test")
        assert result == {"value": 99}

    def test_adapter_returns_empty_on_unavailable(self) -> None:
        """Connection error returns {} not exception."""
        adapter = _DummyAdapter()
        adapter.set_fail()
        result = adapter.query("anything")
        assert result == {}

    def test_adapter_status_reports_healthy_after_success(self) -> None:
        adapter = _DummyAdapter()
        adapter.set_response("test", {"ok": True})
        adapter.query("test")
        assert adapter.status().healthy is True

    def test_adapter_status_reports_unhealthy_after_failure(self) -> None:
        adapter = _DummyAdapter()
        adapter.set_fail()
        adapter.query("test")
        assert adapter.status().healthy is False


class TestMLflowAdapter:
    """Test MLflow adapter."""

    def test_mlflow_adapter_url_from_env(self) -> None:
        with patch.dict(
            "os.environ", {"MLFLOW_TRACKING_URI": "http://test-mlflow:5555"}
        ):
            adapter = MLflowAdapter()
            assert adapter.base_url == "http://test-mlflow:5555"

    def test_mlflow_adapter_has_get_experiments(self) -> None:
        adapter = MLflowAdapter()
        assert callable(adapter.get_experiments)

    def test_mlflow_adapter_has_get_runs(self) -> None:
        adapter = MLflowAdapter()
        assert callable(adapter.get_runs)


class TestPrefectAdapter:
    """Test Prefect adapter."""

    def test_prefect_adapter_url_from_env(self) -> None:
        with patch.dict(
            "os.environ", {"PREFECT_API_URL": "http://test-prefect:4200/api"}
        ):
            adapter = PrefectAdapter()
            assert adapter.base_url == "http://test-prefect:4200/api"

    def test_prefect_adapter_has_get_flow_runs(self) -> None:
        adapter = PrefectAdapter()
        assert callable(adapter.get_flow_runs)


class TestBentoMLAdapter:
    """Test BentoML adapter."""

    def test_bentoml_adapter_url_from_env(self) -> None:
        with patch.dict("os.environ", {"BENTO_SERVER_URL": "http://test-bento:3333"}):
            adapter = BentoMLAdapter()
            assert adapter.base_url == "http://test-bento:3333"

    def test_bentoml_adapter_has_get_health(self) -> None:
        adapter = BentoMLAdapter()
        assert callable(adapter.get_health)
