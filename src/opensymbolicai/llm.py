"""LLM provider abstractions for OpenSymbolicAI."""

import hashlib
import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage information from an LLM response."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """Response from an LLM generation request."""

    text: str = Field(..., description="Generated text content")
    usage: TokenUsage = Field(default_factory=TokenUsage)
    provider: str = Field(default="", description="Provider name")
    model: str = Field(default="", description="Model name")


class Provider(str, Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    FIREWORKS = "fireworks"
    GROQ = "groq"


class GenerationParams(BaseModel):
    """Parameters for LLM text generation."""

    temperature: float = Field(default=0, description="Sampling temperature (0.0-2.0)")
    top_p: float | None = Field(
        default=None, description="Nucleus sampling probability (0.0-1.0)"
    )
    top_k: int | None = Field(default=None, description="Top-k sampling parameter")
    max_tokens: int | None = Field(
        default=None, description="Maximum tokens to generate"
    )
    stop: list[str] | None = Field(default=None, description="Stop sequences")
    frequency_penalty: float | None = Field(
        default=None, description="Frequency penalty (-2.0-2.0)"
    )
    presence_penalty: float | None = Field(
        default=None, description="Presence penalty (-2.0-2.0)"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )

    def to_api_dict(self) -> dict[str, Any]:
        """Convert to dict for API calls, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class LLMConfig(BaseModel):
    """Unified configuration for LLM provider, model, and parameters.

    Combines provider selection, model specification, and generation parameters
    into a single configuration object.

    Example:
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            params=GenerationParams(temperature=0.7, max_tokens=1024),
        )
        llm = create_llm(config)
        response = llm.generate("Hello, world!")
    """

    provider: Provider | str = Field(..., description="LLM provider name")
    model: str = Field(..., description="Model name/ID to use")
    api_key: str | None = Field(
        default=None,
        description="API key (falls back to environment variable if not set)",
    )
    base_url: str | None = Field(
        default=None,
        description="Custom API base URL (uses provider default if not set)",
    )
    params: GenerationParams = Field(
        default_factory=GenerationParams,
        description="Generation parameters",
    )

    @property
    def provider_name(self) -> str:
        """Get the provider name as a string."""
        if isinstance(self.provider, Provider):
            return self.provider.value
        return self.provider.lower()


# =============================================================================
# Cache Abstractions
# =============================================================================


class CacheEntry(BaseModel):
    """A cached LLM response entry."""

    response: LLMResponse
    config_hash: str = Field(..., description="Hash of the config used")


class LLMCache(ABC):
    """Abstract base class for LLM response caching."""

    @staticmethod
    def compute_cache_key(config: LLMConfig, prompt: str) -> str:
        """Compute a cache key from config and prompt.

        Args:
            config: The LLM configuration.
            prompt: The input prompt.

        Returns:
            A SHA-256 hash string to use as the cache key.
        """
        key_dict: dict[str, Any] = {
            "provider": config.provider_name,
            "model": config.model,
            "prompt": prompt,
            "params": config.params.to_api_dict(),
        }
        key_data = json.dumps(key_dict, sort_keys=True)
        return hashlib.sha256(key_data.encode("utf-8")).hexdigest()

    @abstractmethod
    def get(self, key: str) -> CacheEntry | None:
        """Retrieve a cached entry by key."""

    @abstractmethod
    def set(self, key: str, entry: CacheEntry) -> None:
        """Store an entry in the cache."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete an entry from the cache."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from the cache."""


class InMemoryCache(LLMCache):
    """In-memory LLM response cache with optional size limit."""

    def __init__(self, max_size: int = 0) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._insertion_order: list[str] = []
        self._max_size = max_size

    def get(self, key: str) -> CacheEntry | None:
        return self._cache.get(key)

    def set(self, key: str, entry: CacheEntry) -> None:
        if key not in self._cache:
            self._insertion_order.append(key)
            if self._max_size > 0 and len(self._cache) >= self._max_size:
                oldest_key = self._insertion_order.pop(0)
                self._cache.pop(oldest_key, None)
        self._cache[key] = entry

    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            if key in self._insertion_order:
                self._insertion_order.remove(key)
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()
        self._insertion_order.clear()

    def size(self) -> int:
        return len(self._cache)


# =============================================================================
# LLM Provider Base
# =============================================================================


class LLM(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig, cache: LLMCache | None = None) -> None:
        self.config = config
        self.cache = cache

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a text completion, with optional caching.

        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters that override config params.

        Returns:
            LLMResponse with the generated text.
        """
        if self.cache is None:
            return self._generate_impl(prompt, **kwargs)

        cache_key = LLMCache.compute_cache_key(self.config, prompt)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached.response

        response = self._generate_impl(prompt, **kwargs)
        self.cache.set(cache_key, CacheEntry(response=response, config_hash=cache_key))
        return response

    @abstractmethod
    def _generate_impl(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Implementation of text generation."""


# =============================================================================
# Provider Implementations
# =============================================================================


class OllamaLLM(LLM):
    """Ollama local LLM provider."""

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, config: LLMConfig, cache: LLMCache | None = None) -> None:
        super().__init__(config, cache)
        self.base_url = (config.base_url or self.DEFAULT_BASE_URL).rstrip("/")

    def _generate_impl(self, prompt: str, **kwargs: Any) -> LLMResponse:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            **self.config.params.to_api_dict(),
            **kwargs,
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "opensymbolicai/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request) as response:
                result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return LLMResponse(
                    text=str(result.get("response", "")),
                    usage=TokenUsage(
                        input_tokens=result.get("prompt_eval_count", 0),
                        output_tokens=result.get("eval_count", 0),
                    ),
                    provider="ollama",
                    model=self.config.model,
                )
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama API request failed: {e} - {body} (url={url}, model={self.config.model})"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama API request failed: {e}") from e


class OpenAILLM(LLM):
    """OpenAI LLM provider."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, config: LLMConfig, cache: LLMCache | None = None) -> None:
        import os

        super().__init__(config, cache)
        self.base_url = (config.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required (set api_key or OPENAI_API_KEY)")
        self.api_key = api_key

    def _generate_impl(self, prompt: str, **kwargs: Any) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            **self.config.params.to_api_dict(),
            **kwargs,
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "opensymbolicai/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request) as response:
                result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                usage_data = result.get("usage", {})
                return LLMResponse(
                    text=str(result["choices"][0]["message"]["content"]),
                    usage=TokenUsage(
                        input_tokens=usage_data.get("prompt_tokens", 0),
                        output_tokens=usage_data.get("completion_tokens", 0),
                    ),
                    provider="openai",
                    model=self.config.model,
                )
        except urllib.error.URLError as e:
            raise RuntimeError(f"OpenAI API request failed: {e}") from e
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected OpenAI response format: {e}") from e


class AnthropicLLM(LLM):
    """Anthropic Claude LLM provider."""

    DEFAULT_BASE_URL = "https://api.anthropic.com"

    def __init__(self, config: LLMConfig, cache: LLMCache | None = None) -> None:
        import os

        super().__init__(config, cache)
        self.base_url = (config.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required (set api_key or ANTHROPIC_API_KEY)"
            )
        self.api_key = api_key

    def _generate_impl(self, prompt: str, **kwargs: Any) -> LLMResponse:
        url = f"{self.base_url}/v1/messages"
        params = self.config.params.to_api_dict()
        max_tokens = params.pop("max_tokens", 1024)

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            **params,
            **kwargs,
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "User-Agent": "opensymbolicai/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request) as response:
                result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                usage_data = result.get("usage", {})
                return LLMResponse(
                    text=str(result["content"][0]["text"]),
                    usage=TokenUsage(
                        input_tokens=usage_data.get("input_tokens", 0),
                        output_tokens=usage_data.get("output_tokens", 0),
                    ),
                    provider="anthropic",
                    model=self.config.model,
                )
        except urllib.error.URLError as e:
            raise RuntimeError(f"Anthropic API request failed: {e}") from e
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Anthropic response format: {e}") from e


class FireworksLLM(LLM):
    """Fireworks AI LLM provider."""

    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"

    def __init__(self, config: LLMConfig, cache: LLMCache | None = None) -> None:
        import os

        super().__init__(config, cache)
        self.base_url = (config.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        api_key = config.api_key or os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError(
                "Fireworks API key required (set api_key or FIREWORKS_API_KEY)"
            )
        self.api_key = api_key

    def _generate_impl(self, prompt: str, **kwargs: Any) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            **self.config.params.to_api_dict(),
            **kwargs,
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "opensymbolicai/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request) as response:
                result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                usage_data = result.get("usage", {})
                return LLMResponse(
                    text=str(result["choices"][0]["message"]["content"]),
                    usage=TokenUsage(
                        input_tokens=usage_data.get("prompt_tokens", 0),
                        output_tokens=usage_data.get("completion_tokens", 0),
                    ),
                    provider="fireworks",
                    model=self.config.model,
                )
        except urllib.error.URLError as e:
            raise RuntimeError(f"Fireworks API request failed: {e}") from e
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Fireworks response format: {e}") from e


class GroqLLM(LLM):
    """Groq LLM provider."""

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, config: LLMConfig, cache: LLMCache | None = None) -> None:
        import os

        super().__init__(config, cache)
        self.base_url = (config.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        api_key = config.api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key required (set api_key or GROQ_API_KEY)")
        self.api_key = api_key

    def _generate_impl(self, prompt: str, **kwargs: Any) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            **self.config.params.to_api_dict(),
            **kwargs,
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "opensymbolicai/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request) as response:
                result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                usage_data = result.get("usage", {})
                return LLMResponse(
                    text=str(result["choices"][0]["message"]["content"]),
                    usage=TokenUsage(
                        input_tokens=usage_data.get("prompt_tokens", 0),
                        output_tokens=usage_data.get("completion_tokens", 0),
                    ),
                    provider="groq",
                    model=self.config.model,
                )
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Groq API request failed: {e} - {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Groq API request failed: {e}") from e
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Groq response format: {e}") from e


# =============================================================================
# Factory
# =============================================================================

_PROVIDER_MAP: dict[str, type[LLM]] = {
    "ollama": OllamaLLM,
    "openai": OpenAILLM,
    "anthropic": AnthropicLLM,
    "fireworks": FireworksLLM,
    "groq": GroqLLM,
}

ProviderName = Literal["ollama", "openai", "anthropic", "fireworks", "groq"]


def create_llm(config: LLMConfig, cache: LLMCache | None = None) -> LLM:
    """Create an LLM instance from configuration.

    Args:
        config: The unified LLM configuration.
        cache: Optional cache for response caching.

    Returns:
        An LLM instance for the specified provider.

    Raises:
        ValueError: If the provider is not supported.

    Example:
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-sonnet-20240229",
            params=GenerationParams(temperature=0.7),
        )
        llm = create_llm(config)
        response = llm.generate("Write a haiku about coding")
    """
    provider_name = config.provider_name
    llm_class = _PROVIDER_MAP.get(provider_name)
    if llm_class is None:
        available = ", ".join(_PROVIDER_MAP.keys())
        raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")
    return llm_class(config, cache)


def list_providers() -> list[str]:
    """List available provider names."""
    return list(_PROVIDER_MAP.keys())
