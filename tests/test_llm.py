"""Unit tests for LLM module."""

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from opensymbolicai.llm import (
    AnthropicLLM,
    CacheEntry,
    GenerationParams,
    InMemoryCache,
    LLMCache,
    LLMConfig,
    LLMResponse,
    OllamaLLM,
    OpenAILLM,
    Provider,
    create_llm,
)


class TestGenerationParams:
    """Test parameter serialization logic."""

    def test_excludes_none_values_from_api_dict(self):
        """Only non-None params should be sent to APIs."""
        params = GenerationParams(temperature=0.7, max_tokens=100)
        result = params.to_api_dict()

        assert result == {"temperature": 0.7, "max_tokens": 100}
        assert "top_p" not in result
        assert "seed" not in result


class TestLLMConfig:
    """Test config normalization."""

    def test_normalizes_provider_enum_to_lowercase_string(self):
        config = LLMConfig(provider=Provider.OPENAI, model="gpt-4")
        assert config.provider_name == "openai"

    def test_normalizes_mixed_case_string_provider(self):
        config = LLMConfig(provider="OpenAI", model="gpt-4")
        assert config.provider_name == "openai"


class TestInMemoryCache:
    """Test cache eviction and ordering."""

    def test_evicts_oldest_entry_when_max_size_exceeded(self):
        cache = InMemoryCache(max_size=2)

        cache.set("first", CacheEntry(response=LLMResponse(text="1"), config_hash="h"))
        cache.set("second", CacheEntry(response=LLMResponse(text="2"), config_hash="h"))
        cache.set("third", CacheEntry(response=LLMResponse(text="3"), config_hash="h"))

        assert cache.get("first") is None  # Evicted
        assert cache.get("second") is not None
        assert cache.get("third") is not None
        assert cache.size() == 2

    def test_updating_existing_key_does_not_count_as_new_entry(self):
        cache = InMemoryCache(max_size=2)

        cache.set("k1", CacheEntry(response=LLMResponse(text="v1"), config_hash="h"))
        cache.set("k2", CacheEntry(response=LLMResponse(text="v2"), config_hash="h"))
        cache.set(
            "k1", CacheEntry(response=LLMResponse(text="v1-updated"), config_hash="h")
        )

        # k2 should still exist - updating k1 shouldn't trigger eviction
        assert cache.get("k2") is not None
        assert cache.get("k1").response.text == "v1-updated"
        assert cache.size() == 2

    def test_delete_returns_false_for_missing_key(self):
        cache = InMemoryCache()
        assert cache.delete("nonexistent") is False

    def test_delete_removes_from_insertion_order(self):
        """Ensures deleted keys don't get evicted again later."""
        cache = InMemoryCache(max_size=2)

        cache.set("a", CacheEntry(response=LLMResponse(text="a"), config_hash="h"))
        cache.set("b", CacheEntry(response=LLMResponse(text="b"), config_hash="h"))
        cache.delete("a")
        cache.set("c", CacheEntry(response=LLMResponse(text="c"), config_hash="h"))

        # Should have b and c, not trigger eviction of already-deleted a
        assert cache.size() == 2
        assert cache.get("b") is not None
        assert cache.get("c") is not None


class TestCacheKeyComputation:
    """Test cache key uniqueness."""

    def test_same_input_produces_same_key(self):
        config = LLMConfig(provider="openai", model="gpt-4")
        key1 = LLMCache.compute_cache_key(config, "hello")
        key2 = LLMCache.compute_cache_key(config, "hello")
        assert key1 == key2

    def test_different_prompts_produce_different_keys(self):
        config = LLMConfig(provider="openai", model="gpt-4")
        key1 = LLMCache.compute_cache_key(config, "hello")
        key2 = LLMCache.compute_cache_key(config, "goodbye")
        assert key1 != key2

    def test_different_models_produce_different_keys(self):
        config1 = LLMConfig(provider="openai", model="gpt-4")
        config2 = LLMConfig(provider="openai", model="gpt-3.5-turbo")
        key1 = LLMCache.compute_cache_key(config1, "hello")
        key2 = LLMCache.compute_cache_key(config2, "hello")
        assert key1 != key2

    def test_different_params_produce_different_keys(self):
        """Temperature differences must produce different cache keys."""
        config1 = LLMConfig(
            provider="openai",
            model="gpt-4",
            params=GenerationParams(temperature=0),
        )
        config2 = LLMConfig(
            provider="openai",
            model="gpt-4",
            params=GenerationParams(temperature=1),
        )
        key1 = LLMCache.compute_cache_key(config1, "hello")
        key2 = LLMCache.compute_cache_key(config2, "hello")
        assert key1 != key2


class TestCreateLLMFactory:
    """Test factory error handling and provider selection."""

    def test_raises_for_unknown_provider(self):
        config = LLMConfig(provider="nonexistent", model="model")
        with pytest.raises(ValueError, match="Unknown provider 'nonexistent'"):
            create_llm(config)

    def test_error_message_lists_available_providers(self):
        config = LLMConfig(provider="bad", model="model")
        with pytest.raises(ValueError, match="ollama.*openai.*anthropic"):
            create_llm(config)

    def test_passes_cache_to_llm_instance(self):
        config = LLMConfig(provider="ollama", model="llama2")
        cache = InMemoryCache()
        llm = create_llm(config, cache)
        assert llm.cache is cache


class TestAPIKeyValidation:
    """Test API key requirement and fallback logic."""

    def test_openai_raises_without_key(self):
        config = LLMConfig(provider="openai", model="gpt-4")
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="OpenAI API key required"),
        ):
            OpenAILLM(config)

    def test_openai_uses_env_var_as_fallback(self):
        config = LLMConfig(provider="openai", model="gpt-4")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            llm = OpenAILLM(config)
            assert llm.api_key == "env-key"

    def test_config_api_key_takes_precedence_over_env(self):
        config = LLMConfig(provider="openai", model="gpt-4", api_key="config-key")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            llm = OpenAILLM(config)
            assert llm.api_key == "config-key"

    def test_anthropic_raises_without_key(self):
        config = LLMConfig(provider="anthropic", model="claude")
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="Anthropic API key required"),
        ):
            AnthropicLLM(config)


class TestLLMCachingBehavior:
    """Test cache integration with LLM generate."""

    def test_returns_cached_response_without_api_call(self):
        config = LLMConfig(provider="ollama", model="llama2")
        cache = InMemoryCache()
        llm = OllamaLLM(config, cache)

        # Pre-populate cache
        cache_key = LLMCache.compute_cache_key(config, "hello")
        cached_response = LLMResponse(text="cached", provider="ollama", model="llama2")
        cache.set(
            cache_key, CacheEntry(response=cached_response, config_hash=cache_key)
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            result = llm.generate("hello")
            mock_urlopen.assert_not_called()

        assert result.text == "cached"

    def test_stores_response_in_cache_on_miss(self):
        config = LLMConfig(provider="ollama", model="llama2")
        cache = InMemoryCache()
        llm = OllamaLLM(config, cache)

        mock_response = _make_mock_response({"response": "generated"})

        with patch("urllib.request.urlopen", return_value=mock_response):
            llm.generate("hello")

        assert cache.size() == 1
        cached = cache.get(LLMCache.compute_cache_key(config, "hello"))
        assert cached.response.text == "generated"


class TestOllamaLLM:
    """Test Ollama-specific behavior."""

    def test_parses_response_correctly(self):
        config = LLMConfig(provider="ollama", model="llama2")
        llm = OllamaLLM(config)

        mock_response = _make_mock_response(
            {
                "response": "Hello!",
                "prompt_eval_count": 10,
                "eval_count": 5,
            }
        )

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = llm.generate("Hi")

        assert result.text == "Hello!"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_sets_stream_false_in_request(self):
        config = LLMConfig(provider="ollama", model="llama2")
        llm = OllamaLLM(config)

        mock_response = _make_mock_response({"response": "x"})

        with patch(
            "urllib.request.urlopen", return_value=mock_response
        ) as mock_urlopen:
            llm.generate("test")

        payload = json.loads(mock_urlopen.call_args[0][0].data)
        assert payload["stream"] is False

    def test_wraps_url_error_with_context(self):
        config = LLMConfig(provider="ollama", model="llama2")
        llm = OllamaLLM(config)

        with (
            patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.URLError("Connection refused"),
            ),
            pytest.raises(RuntimeError, match="Ollama API request failed"),
        ):
            llm.generate("test")


class TestOpenAILLM:
    """Test OpenAI-specific behavior."""

    def test_sends_bearer_token_header(self):
        config = LLMConfig(provider="openai", model="gpt-4", api_key="sk-test123")
        llm = OpenAILLM(config)

        mock_response = _make_mock_response(
            {
                "choices": [{"message": {"content": "response"}}],
                "usage": {},
            }
        )

        with patch(
            "urllib.request.urlopen", return_value=mock_response
        ) as mock_urlopen:
            llm.generate("test")

        request = mock_urlopen.call_args[0][0]
        assert request.headers["Authorization"] == "Bearer sk-test123"

    def test_raises_on_malformed_response(self):
        config = LLMConfig(provider="openai", model="gpt-4", api_key="key")
        llm = OpenAILLM(config)

        mock_response = _make_mock_response({"choices": []})  # Empty choices

        with (
            patch("urllib.request.urlopen", return_value=mock_response),
            pytest.raises(RuntimeError, match="Unexpected OpenAI response format"),
        ):
            llm.generate("test")


class TestAnthropicLLM:
    """Test Anthropic-specific behavior."""

    def test_uses_x_api_key_header(self):
        config = LLMConfig(
            provider="anthropic", model="claude-3", api_key="sk-ant-test"
        )
        llm = AnthropicLLM(config)

        mock_response = _make_mock_response(
            {
                "content": [{"text": "response"}],
                "usage": {},
            }
        )

        with patch(
            "urllib.request.urlopen", return_value=mock_response
        ) as mock_urlopen:
            llm.generate("test")

        request = mock_urlopen.call_args[0][0]
        assert request.headers["X-api-key"] == "sk-ant-test"
        assert request.headers["Anthropic-version"] == "2023-06-01"

    def test_defaults_max_tokens_to_1024(self):
        config = LLMConfig(provider="anthropic", model="claude-3", api_key="key")
        llm = AnthropicLLM(config)

        mock_response = _make_mock_response(
            {
                "content": [{"text": "x"}],
                "usage": {},
            }
        )

        with patch(
            "urllib.request.urlopen", return_value=mock_response
        ) as mock_urlopen:
            llm.generate("test")

        payload = json.loads(mock_urlopen.call_args[0][0].data)
        assert payload["max_tokens"] == 1024

    def test_respects_custom_max_tokens(self):
        config = LLMConfig(
            provider="anthropic",
            model="claude-3",
            api_key="key",
            params=GenerationParams(max_tokens=500),
        )
        llm = AnthropicLLM(config)

        mock_response = _make_mock_response(
            {
                "content": [{"text": "x"}],
                "usage": {},
            }
        )

        with patch(
            "urllib.request.urlopen", return_value=mock_response
        ) as mock_urlopen:
            llm.generate("test")

        payload = json.loads(mock_urlopen.call_args[0][0].data)
        assert payload["max_tokens"] == 500


def _make_mock_response(data: dict) -> MagicMock:
    """Create a mock urllib response object."""
    mock = MagicMock()
    mock.read.return_value = json.dumps(data).encode()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock
