# Health endpoint tests

import pytest
from httpx import AsyncClient


class TestHealth:
    """Health and readiness endpoint tests."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client: AsyncClient):
        """Test /health endpoint returns 200."""
        response = await client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert data["service"] == "llm-inference-runtime"
        assert "uptime_seconds" in data
        assert "checks" in data

    @pytest.mark.asyncio
    async def test_ready_endpoint(self, client: AsyncClient):
        """Test /ready endpoint."""
        response = await client.get("/ready")
        # May be 200 or 503 depending on model load state
        assert response.status_code in (200, 503)

        data = response.json()
        assert "ready" in data
        assert data["service"] == "llm-inference-runtime"
        assert "checks" in data

    @pytest.mark.asyncio
    async def test_version_endpoint(self, client: AsyncClient):
        """Test /version endpoint."""
        response = await client.get("/version")
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "llm-inference-runtime"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client: AsyncClient):
        """Test /metrics endpoint."""
        response = await client.get("/metrics")
        # May be 503 if metrics disabled
        assert response.status_code in (200, 503)

        if response.status_code == 200:
            content = response.text
            assert "llm_" in content  # Our custom metrics prefix


class TestRoot:
    """Root endpoint tests."""

    @pytest.mark.asyncio
    async def test_root_endpoint(self, client: AsyncClient):
        """Test root endpoint returns service info."""
        response = await client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "llm-inference-runtime"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"