"""
Multi-provider LLM client for answer generation.

Supports:
- OpenAI (GPT-4, GPT-3.5-turbo)
- Anthropic (Claude 3 Opus, Sonnet, Haiku)
- Ollama (local models like Llama, Mistral)

Features:
- Retry logic with exponential backoff
- Provider fallback on failure
- Token usage tracking
- Temperature control for technical accuracy
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
import time

from loguru import logger

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not installed. Install with: pip install openai")

try:
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic not installed. Install with: pip install anthropic")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not installed. Install with: pip install ollama")


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass
class LLMResponse:
    """
    Response from LLM generation.

    Attributes:
        content: Generated answer text
        provider: Which provider was used
        model: Which model was used
        tokens_used: Total tokens consumed
        prompt_tokens: Tokens in prompt
        completion_tokens: Tokens in completion
        latency_ms: Generation time in milliseconds
        success: Whether generation succeeded
        error: Error message if failed
    """
    content: str
    provider: str
    model: str
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


class LLMClient:
    """
    Multi-provider LLM client with retry and fallback logic.

    Usage:
        client = LLMClient(provider="openai", model="gpt-4")
        response = client.generate(
            prompt="Answer this question: What is beam bending?",
            context=["Beam bending is...", "The stress distribution..."]
        )
        print(response.content)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        max_retries: int = 3,
        timeout: int = 60,
        api_key: Optional[str] = None,
        fallback_providers: Optional[List[str]] = None,
    ):
        """
        Initialize LLM client.

        Args:
            provider: Primary provider ("openai", "anthropic", "ollama")
            model: Model name (defaults to best for provider)
            temperature: Sampling temperature (0.0-1.0). Use 0.1 for technical accuracy
            max_tokens: Maximum tokens in completion
            max_retries: Number of retry attempts on failure
            timeout: Request timeout in seconds
            api_key: API key (defaults to environment variable)
            fallback_providers: List of backup providers to try on failure
        """
        self.provider = LLMProvider(provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.fallback_providers = fallback_providers or []

        # Set default models
        if model is None:
            model = self._get_default_model(self.provider)
        self.model = model

        # Initialize provider clients
        self.clients = {}
        self._init_provider(self.provider, api_key)

        # Initialize fallback providers
        for fallback in self.fallback_providers:
            try:
                self._init_provider(LLMProvider(fallback), api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize fallback {fallback}: {e}")

        logger.info(
            f"LLM Client initialized: {self.provider.value}/{self.model} "
            f"(temperature={self.temperature}, fallbacks={self.fallback_providers})"
        )

    def _get_default_model(self, provider: LLMProvider) -> str:
        """Get default model for provider."""
        defaults = {
            LLMProvider.OPENAI: "gpt-4",
            LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
            LLMProvider.OLLAMA: "llama3.1:8b",
        }
        return defaults[provider]

    def _init_provider(self, provider: LLMProvider, api_key: Optional[str] = None):
        """Initialize a provider client."""
        if provider == LLMProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not installed. Install with: pip install openai")
            self.clients[provider] = OpenAI(api_key=api_key)

        elif provider == LLMProvider.ANTHROPIC:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic not installed. Install with: pip install anthropic")
            self.clients[provider] = Anthropic(api_key=api_key)

        elif provider == LLMProvider.OLLAMA:
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama not installed. Install with: pip install ollama")
            # Ollama client doesn't need API key
            self.clients[provider] = None

    def generate(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate answer using LLM.

        Args:
            prompt: Question or instruction
            context: List of context chunks to include
            system_prompt: System prompt (defaults to technical QA prompt)

        Returns:
            LLMResponse with generated content
        """
        # Build full prompt with context
        full_prompt = self._build_prompt(prompt, context)

        # Use default system prompt if not provided
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        # Try primary provider
        response = self._generate_with_retry(
            self.provider,
            self.model,
            full_prompt,
            system_prompt,
        )

        if response.success:
            return response

        # Try fallback providers
        for fallback in self.fallback_providers:
            logger.warning(f"Primary provider failed, trying fallback: {fallback}")

            try:
                fallback_provider = LLMProvider(fallback)
                fallback_model = self._get_default_model(fallback_provider)

                response = self._generate_with_retry(
                    fallback_provider,
                    fallback_model,
                    full_prompt,
                    system_prompt,
                )

                if response.success:
                    return response

            except Exception as e:
                logger.error(f"Fallback {fallback} failed: {e}")

        # All providers failed
        return LLMResponse(
            content="",
            provider=self.provider.value,
            model=self.model,
            success=False,
            error="All providers failed",
        )

    def _generate_with_retry(
        self,
        provider: LLMProvider,
        model: str,
        prompt: str,
        system_prompt: str,
    ) -> LLMResponse:
        """Generate with exponential backoff retry."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                if provider == LLMProvider.OPENAI:
                    response = self._generate_openai(model, prompt, system_prompt)
                elif provider == LLMProvider.ANTHROPIC:
                    response = self._generate_anthropic(model, prompt, system_prompt)
                elif provider == LLMProvider.OLLAMA:
                    response = self._generate_ollama(model, prompt, system_prompt)
                else:
                    raise ValueError(f"Unknown provider: {provider}")

                latency_ms = (time.time() - start_time) * 1000
                response.latency_ms = latency_ms

                logger.info(
                    f"✅ LLM generation successful: {provider.value}/{model} "
                    f"({latency_ms:.0f}ms, {response.tokens_used} tokens)"
                )

                return response

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )

                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)

        # All retries failed
        return LLMResponse(
            content="",
            provider=provider.value,
            model=model,
            success=False,
            error=f"Failed after {self.max_retries} retries: {last_error}",
        )

    def _generate_openai(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
    ) -> LLMResponse:
        """Generate using OpenAI."""
        client = self.clients[LLMProvider.OPENAI]

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            provider="openai",
            model=model,
            tokens_used=response.usage.total_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            success=True,
        )

    def _generate_anthropic(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
    ) -> LLMResponse:
        """Generate using Anthropic."""
        client = self.clients[LLMProvider.ANTHROPIC]

        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

        # Extract text from response
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            provider="anthropic",
            model=model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            success=True,
        )

    def _generate_ollama(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
    ) -> LLMResponse:
        """Generate using Ollama."""
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )

        return LLMResponse(
            content=response["message"]["content"],
            provider="ollama",
            model=model,
            tokens_used=response.get("eval_count", 0) + response.get("prompt_eval_count", 0),
            prompt_tokens=response.get("prompt_eval_count", 0),
            completion_tokens=response.get("eval_count", 0),
            success=True,
        )

    def _build_prompt(
        self,
        question: str,
        context: Optional[List[str]] = None,
    ) -> str:
        """Build full prompt with context."""
        if not context:
            return question

        # Format context chunks
        context_text = "\n\n".join([
            f"[Context {i+1}]\n{chunk}"
            for i, chunk in enumerate(context)
        ])

        return f"""Context:
{context_text}

Question:
{question}

Please answer the question based on the provided context. Be precise and technical."""

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for technical QA."""
        return """You are a technical assistant specializing in aerospace engineering and structural mechanics.

Your task is to answer questions based on the provided context from technical documents. Follow these guidelines:

1. **Be Precise**: Use exact values, formulas, and terminology from the context
2. **Preserve Equations**: Maintain LaTeX equations exactly as written (e.g., $I = \\frac{bh^3}{12}$)
3. **Cite Sources**: Reference specific context chunks when making claims
4. **Stay Grounded**: Only answer based on the provided context - don't add external knowledge
5. **Be Technical**: Use proper engineering terminology and notation
6. **Structure Clearly**: Use clear paragraphs and bullet points for complex answers

If the context doesn't contain enough information to answer fully, acknowledge this limitation."""


    def generate_stream(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Generate answer with streaming support.

        Args:
            prompt: Question or instruction
            context: List of context chunks to include
            system_prompt: System prompt (defaults to technical QA prompt)

        Yields:
            str: Token chunks as they are generated

        Usage:
            for chunk in client.generate_stream(prompt="...", context=[...]):
                print(chunk, end="", flush=True)
        """
        # Build full prompt with context
        full_prompt = self._build_prompt(prompt, context)

        # Use default system prompt if not provided
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        # Try primary provider
        try:
            yield from self._generate_stream_with_provider(
                self.provider,
                self.model,
                full_prompt,
                system_prompt,
            )
            return

        except Exception as e:
            logger.warning(f"Primary provider streaming failed: {e}")

        # Try fallback providers
        for fallback in self.fallback_providers:
            logger.warning(f"Trying fallback streaming: {fallback}")

            try:
                fallback_provider = LLMProvider(fallback)
                fallback_model = self._get_default_model(fallback_provider)

                yield from self._generate_stream_with_provider(
                    fallback_provider,
                    fallback_model,
                    full_prompt,
                    system_prompt,
                )
                return

            except Exception as e:
                logger.error(f"Fallback {fallback} streaming failed: {e}")

        # All providers failed
        raise RuntimeError("All providers failed for streaming generation")

    def _generate_stream_with_provider(
        self,
        provider: LLMProvider,
        model: str,
        prompt: str,
        system_prompt: str,
    ):
        """Generate streaming response from a specific provider."""
        if provider == LLMProvider.OPENAI:
            yield from self._generate_stream_openai(model, prompt, system_prompt)
        elif provider == LLMProvider.ANTHROPIC:
            yield from self._generate_stream_anthropic(model, prompt, system_prompt)
        elif provider == LLMProvider.OLLAMA:
            yield from self._generate_stream_ollama(model, prompt, system_prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _generate_stream_openai(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
    ):
        """Stream response from OpenAI."""
        client = self.clients[LLMProvider.OPENAI]

        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _generate_stream_anthropic(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
    ):
        """Stream response from Anthropic."""
        client = self.clients[LLMProvider.ANTHROPIC]

        with client.messages.stream(
            model=model,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def _generate_stream_ollama(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
    ):
        """Stream response from Ollama."""
        stream = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            stream=True,
        )

        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]


def get_llm_client(
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """
    Convenience function to get LLM client.

    Args:
        provider: Provider name
        model: Model name
        **kwargs: Additional arguments for LLMClient

    Returns:
        Initialized LLMClient
    """
    return LLMClient(provider=provider, model=model, **kwargs)


if __name__ == "__main__":
    logger.add("logs/llm_client.log", rotation="10 MB")

    print("\n" + "=" * 70)
    print("LLM CLIENT - Multi-Provider Support")
    print("=" * 70)
    print("\nSupported Providers:")
    print("  • OpenAI (GPT-4, GPT-3.5-turbo)")
    print("  • Anthropic (Claude 3 Opus, Sonnet, Haiku)")
    print("  • Ollama (Llama, Mistral, etc.)")
    print("\nFeatures:")
    print("  • Retry logic with exponential backoff")
    print("  • Provider fallback on failure")
    print("  • Token usage tracking")
    print("  • Low temperature (0.1) for technical accuracy")
    print("\nUsage:")
    print("  client = LLMClient(provider='openai', model='gpt-4')")
    print("  response = client.generate(prompt='...', context=[...])")
    print("  print(response.content)")
    print("=" * 70 + "\n")
