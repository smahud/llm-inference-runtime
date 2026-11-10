# Streaming tests

import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch, MagicMock


class TestStreaming:
    """Streaming endpoint tests."""

    @pytest.mark.asyncio
    async def test_chat_stream_format(self, client: AsyncClient):
        """Test chat streaming response format."""
        mock_chunk = MagicMock()
        mock_chunk.model_dump_json.return_value = '{"id": "test", "object": "chat.completion.chunk", "choices": []}'

        async def mock_stream():
            yield mock_chunk
            yield mock_chunk

        with patch("src.main.get_engine") as mock_engine:
            mock_engine.return_value.is_ready = True
            mock_engine.return_value.chat_completion = AsyncMock(return_value=mock_stream())

            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True
                }
            )
            # Should be streaming response
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_completion_stream_format(self, client: AsyncClient):
        """Test completion streaming response format."""
        mock_chunk = MagicMock()
        mock_chunk.model_dump_json.return_value = '{"id": "test", "object": "text_completion.chunk", "choices": []}'

        async def mock_stream():
            yield mock_chunk
            yield mock_chunk

        with patch("src.main.get_engine") as mock_engine:
            mock_engine.return_value.is_ready = True
            mock_engine.return_value.completion = AsyncMock(return_value=mock_stream())

            response = await client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "Hello",
                    "stream": True
                }
            )
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")