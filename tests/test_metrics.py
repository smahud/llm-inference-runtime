# Metrics tests

import pytest
from httpx import AsyncClient
from unittest.mock import patch, MagicMock


class TestMetrics:
    """Metrics endpoint tests."""

    @pytest.mark.asyncio
    async def test_metrics_disabled(self, client: AsyncClient):
        """Test metrics endpoint when disabled returns 503."""
        with patch("src.main.get_metrics") as mock_metrics:
            mock_metrics.return_value = None
            response = await client.get("/metrics")
            assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_metrics_enabled(self, client: AsyncClient):
        """Test metrics endpoint when enabled returns Prometheus format."""
        mock_collector = MagicMock()
        mock_collector.generate_metrics.return_value = b"llm_requests_total 1.0\n"

        with patch("src.main.get_metrics") as mock_metrics:
            mock_metrics.return_value = mock_collector
            response = await client.get("/metrics")
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
            assert b"llm_requests_total" in response.content