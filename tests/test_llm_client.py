"""
Tests for LLM client.

Tests:
- Multi-provider support
- Retry logic
- Provider fallback
- Token tracking
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time

from src.llm import LLMClient, LLMResponse, LLMProvider


class TestLLMClient:
    """Test LLM client functionality."""

    def test_client_initialization(self):
        """Test client initializes correctly."""
        with patch("src.llm.client.OPENAI_AVAILABLE", True):
            with patch("src.llm.client.OpenAI"):
                client = LLMClient(provider="openai", model="gpt-4")

                assert client.provider == LLMProvider.OPENAI
                assert client.model == "gpt-4"
                assert client.temperature == 0.1
                assert client.max_retries == 3

    def test_default_models(self):
        """Test default models for each provider."""
        with patch("src.llm.client.OPENAI_AVAILABLE", True):
            with patch("src.llm.client.OpenAI"):
                client = LLMClient(provider="openai")
                assert client.model == "gpt-4"

        with patch("src.llm.client.ANTHROPIC_AVAILABLE", True):
            with patch("src.llm.client.Anthropic"):
                client = LLMClient(provider="anthropic")
                assert client.model == "claude-3-5-sonnet-20241022"

    def test_build_prompt_with_context(self):
        """Test prompt building with context."""
        with patch("src.llm.client.OPENAI_AVAILABLE", True):
            with patch("src.llm.client.OpenAI"):
                client = LLMClient(provider="openai")

                prompt = client._build_prompt(
                    question="What is beam bending?",
                    context=["Beam bending is...", "The stress distribution..."],
                )

                assert "What is beam bending?" in prompt
                assert "[Context 1]" in prompt
                assert "[Context 2]" in prompt
                assert "Beam bending is..." in prompt

    def test_build_prompt_without_context(self):
        """Test prompt building without context."""
        with patch("src.llm.client.OPENAI_AVAILABLE", True):
            with patch("src.llm.client.OpenAI"):
                client = LLMClient(provider="openai")

                prompt = client._build_prompt(
                    question="What is beam bending?",
                    context=None,
                )

                assert prompt == "What is beam bending?"

    @patch("src.llm.client.OPENAI_AVAILABLE", True)
    @patch("src.llm.client.OpenAI")
    def test_openai_generation(self, mock_openai):
        """Test OpenAI generation."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test answer"))]
        mock_response.usage = Mock(
            total_tokens=100,
            prompt_tokens=50,
            completion_tokens=50,
        )

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LLMClient(provider="openai", model="gpt-4")

        response = client._generate_openai(
            model="gpt-4",
            prompt="Test prompt",
            system_prompt="Test system",
        )

        assert response.success is True
        assert response.content == "Test answer"
        assert response.tokens_used == 100
        assert response.provider == "openai"

    @patch("src.llm.client.ANTHROPIC_AVAILABLE", True)
    @patch("src.llm.client.Anthropic")
    def test_anthropic_generation(self, mock_anthropic):
        """Test Anthropic generation."""
        # Mock Anthropic response
        mock_response = Mock()
        mock_response.content = [Mock(text="Test answer")]
        mock_response.usage = Mock(
            input_tokens=50,
            output_tokens=50,
        )

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        client = LLMClient(provider="anthropic", model="claude-3-5-sonnet-20241022")

        response = client._generate_anthropic(
            model="claude-3-5-sonnet-20241022",
            prompt="Test prompt",
            system_prompt="Test system",
        )

        assert response.success is True
        assert response.content == "Test answer"
        assert response.tokens_used == 100
        assert response.provider == "anthropic"

    @patch("src.llm.client.OPENAI_AVAILABLE", True)
    @patch("src.llm.client.OpenAI")
    def test_retry_logic(self, mock_openai):
        """Test retry logic with exponential backoff."""
        mock_client = Mock()

        # Fail twice, succeed on third attempt
        mock_client.chat.completions.create.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            Mock(
                choices=[Mock(message=Mock(content="Success"))],
                usage=Mock(total_tokens=100, prompt_tokens=50, completion_tokens=50),
            ),
        ]

        mock_openai.return_value = mock_client

        client = LLMClient(provider="openai", max_retries=3)

        with patch("time.sleep"):  # Skip actual sleep
            response = client.generate(prompt="Test", context=None)

        assert response.success is True
        assert response.content == "Success"

    @patch("src.llm.client.OPENAI_AVAILABLE", True)
    @patch("src.llm.client.OpenAI")
    def test_all_retries_fail(self, mock_openai):
        """Test when all retries fail."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        client = LLMClient(provider="openai", max_retries=2)

        with patch("time.sleep"):
            response = client.generate(prompt="Test", context=None)

        assert response.success is False
        # Error can be "Failed after N retries" or "All providers failed"
        assert response.error is not None
        assert "failed" in response.error.lower()

    @patch("src.llm.client.OPENAI_AVAILABLE", True)
    @patch("src.llm.client.ANTHROPIC_AVAILABLE", True)
    @patch("src.llm.client.OpenAI")
    @patch("src.llm.client.Anthropic")
    def test_provider_fallback(self, mock_anthropic, mock_openai):
        """Test fallback to secondary provider."""
        # OpenAI fails
        mock_openai_client = Mock()
        mock_openai_client.chat.completions.create.side_effect = Exception("OpenAI Error")
        mock_openai.return_value = mock_openai_client

        # Anthropic succeeds
        mock_anthropic_response = Mock()
        mock_anthropic_response.content = [Mock(text="Fallback answer")]
        mock_anthropic_response.usage = Mock(input_tokens=50, output_tokens=50)

        mock_anthropic_client = Mock()
        mock_anthropic_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_anthropic_client

        client = LLMClient(
            provider="openai",
            max_retries=1,
            fallback_providers=["anthropic"],
        )

        with patch("time.sleep"):
            response = client.generate(prompt="Test", context=None)

        assert response.success is True
        assert response.content == "Fallback answer"
        assert response.provider == "anthropic"

    def test_system_prompt(self):
        """Test default system prompt."""
        with patch("src.llm.client.OPENAI_AVAILABLE", True):
            with patch("src.llm.client.OpenAI"):
                client = LLMClient(provider="openai")

                system_prompt = client._get_default_system_prompt()

                assert "aerospace engineering" in system_prompt
                assert "LaTeX equations" in system_prompt
                assert "context" in system_prompt

    def test_temperature_control(self):
        """Test temperature setting."""
        with patch("src.llm.client.OPENAI_AVAILABLE", True):
            with patch("src.llm.client.OpenAI"):
                # Low temperature for technical accuracy
                client = LLMClient(provider="openai", temperature=0.1)
                assert client.temperature == 0.1

                # Higher temperature for creative tasks
                client = LLMClient(provider="openai", temperature=0.7)
                assert client.temperature == 0.7


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_response_structure(self):
        """Test response contains all fields."""
        response = LLMResponse(
            content="Test answer",
            provider="openai",
            model="gpt-4",
            tokens_used=100,
            prompt_tokens=50,
            completion_tokens=50,
            latency_ms=1500.0,
            success=True,
        )

        assert response.content == "Test answer"
        assert response.provider == "openai"
        assert response.model == "gpt-4"
        assert response.tokens_used == 100
        assert response.latency_ms == 1500.0
        assert response.success is True
        assert response.error is None

    def test_failed_response(self):
        """Test response for failed generation."""
        response = LLMResponse(
            content="",
            provider="openai",
            model="gpt-4",
            success=False,
            error="API Error",
        )

        assert response.success is False
        assert response.error == "API Error"
        assert response.content == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
