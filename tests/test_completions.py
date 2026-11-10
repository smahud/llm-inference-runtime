# Chat completion tests

import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch, MagicMock


class TestChatCompletions:
    """Chat completions endpoint tests."""

    @pytest.mark.asyncio
    async def test_chat_completions_missing_model(self, client: AsyncClient):
        """Test chat completions without model returns 422."""
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_completions_invalid_role(self, client: AsyncClient):
        """Test chat completions with invalid role returns 422."""
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "invalid", "content": "Hello"}]
            }
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_completions_not_ready(self, client: AsyncClient):
        """Test chat completions when engine not ready returns 503."""
        with patch("src.main.get_engine") as mock_engine:
            mock_engine.return_value.is_ready = False
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}]
                }
            )
            assert response.status_code == 503
            data = response.json()
            assert data["error"]["type"] == "service_unavailable"

    @pytest.mark.asyncio
    async def test_chat_completions_stream_not_ready(self, client: AsyncClient):
        """Test streaming chat completions when engine not ready returns 503."""
        with patch("src.main.get_engine") as mock_engine:
            mock_engine.return_value.is_ready = False
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True
                }
            )
            assert response.status_code == 503


class TestCompletions:
    """Legacy completions endpoint tests."""

    @pytest.mark.asyncio
    async def test_completions_missing_prompt(self, client: AsyncClient):
        """Test completions without prompt returns 422."""
        response = await client.post(
            "/v1/completions",
            json={"model": "test-model"}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_completions_not_ready(self, client: AsyncClient):
        """Test completions when engine not ready returns 503."""
        with patch("src.main.get_engine") as mock_engine:
            mock_engine.return_value.is_ready = False
            response = await client.post(
                "/v1/completions",
                json={"model": "test-model", "prompt": "Hello"}
            )
            assert response.status_code == 503


class TestEmbeddings:
    """Embeddings endpoint tests."""

    @pytest.mark.asyncio
    async def test_embeddings_missing_input(self, client: AsyncClient):
        """Test embeddings without input returns 422."""
        response = await client.post(
            "/v1/embeddings",
            json={"model": "test-model"}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_embeddings_not_ready(self, client: AsyncClient):
        """Test embeddings when engine not ready returns 503."""
        with patch("src.main.get_engine") as mock_engine:
            mock_engine.return_value.is_ready = False
            response = await client.post(
                "/v1/embeddings",
                json={"model": "test-model", "input": "Hello"}
            )
            assert response.status_code == 503


class TestModels:
    """Models endpoint tests."""

    @pytest.mark.asyncio
    async def test_list_models_not_ready(self, client: AsyncClient):
        """Test list models when engine not ready returns 503."""
        with patch("src.main.get_engine") as mock_engine:
            mock_engine.return_value.is_ready = False
            response = await client.get("/v1/models")
            assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_get_model_not_ready(self, client: AsyncClient):
        """Test get model when engine not ready returns 503."""
        with patch("src.main.get_engine") as mock_engine:
            mock_engine.return_value.is_ready = False
            response = await client.get("/v1/models/test-model")
            assert response.status_code == 503