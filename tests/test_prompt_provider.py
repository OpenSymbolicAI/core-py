"""Tests for PromptProvider: selecting primitives and decompositions for prompts."""

from typing import Any

from opensymbolicai.blueprints.plan_execute import PlanExecute
from opensymbolicai.core import decomposition, primitive
from opensymbolicai.llm import LLM, LLMConfig, LLMResponse, TokenUsage
from opensymbolicai.models import (
    DecompositionInfo,
    ParameterInfo,
    PlanExecuteConfig,
    PrimitiveInfo,
    PromptProvider,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockLLM(LLM):
    """Mock LLM that returns predefined responses."""

    def __init__(self, responses: list[str] | None = None):
        config = LLMConfig(provider="mock", model="mock-model")
        super().__init__(config, cache=None)
        self.responses = responses or []
        self.call_count = 0
        self.prompts: list[str] = []

    def _generate_impl(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.prompts.append(prompt)
        response_text = (
            self.responses[self.call_count]
            if self.call_count < len(self.responses)
            else "result = 0"
        )
        self.call_count += 1
        return LLMResponse(
            text=response_text,
            provider="mock",
            model="mock-model",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )


class FullAgent(PlanExecute):
    """Agent with several primitives and decompositions for testing selection."""

    @primitive(read_only=True)
    def search(self, query: str) -> list[str]:
        """Search the knowledge base."""
        return [f"result for {query}"]

    @primitive(read_only=True)
    def get_article(self, id: str) -> str:
        """Fetch an article by ID."""
        return f"article-{id}"

    @primitive(read_only=False)
    def save_note(self, text: str) -> str:
        """Save a note (mutating)."""
        return f"saved: {text}"

    @primitive(read_only=False, deterministic=False)
    def delete_article(self, id: str) -> bool:
        """Delete an article (mutating, non-deterministic)."""
        return True

    @decomposition(intent="Find articles about Python")
    def _example_read(self) -> str:
        results = self.search(query="Python")
        article = self.get_article(id=results[0])
        return article

    @decomposition(intent="Save a summary note")
    def _example_write(self) -> str:
        results = self.search(query="AI")
        note = self.save_note(text=results[0])
        return note


# ---------------------------------------------------------------------------
# PromptProvider subclasses for testing
# ---------------------------------------------------------------------------


class ReadOnlyProvider(PromptProvider):
    """Only exposes read-only primitives and their decompositions."""

    def select_primitives(self, available: list[PrimitiveInfo]) -> list[str]:
        return [p.name for p in available if p.read_only]

    def select_decompositions(self, available: list[DecompositionInfo]) -> list[str]:
        return [d.name for d in available if "read" in d.intent.lower()]


class NameFilterProvider(PromptProvider):
    """Filters by explicit name set."""

    primitive_names: set[str]
    decomp_names: set[str] = set()

    def select_primitives(self, available: list[PrimitiveInfo]) -> list[str]:
        return [p.name for p in available if p.name in self.primitive_names]

    def select_decompositions(self, available: list[DecompositionInfo]) -> list[str]:
        return [d.name for d in available if d.name in self.decomp_names]


class AllProvider(PromptProvider):
    """Default — returns everything (tests the base class)."""


class EmptyProvider(PromptProvider):
    """Returns nothing."""

    def select_primitives(self, available: list[PrimitiveInfo]) -> list[str]:
        return []

    def select_decompositions(self, available: list[DecompositionInfo]) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# Tests: PromptProvider base behaviour
# ---------------------------------------------------------------------------


class TestPromptProviderDefaults:
    def test_default_returns_all_primitives(self):
        provider = PromptProvider()
        infos = [
            PrimitiveInfo(name="a", docstring="", read_only=False, deterministic=True),
            PrimitiveInfo(name="b", docstring="", read_only=True, deterministic=True),
        ]
        assert provider.select_primitives(infos) == ["a", "b"]

    def test_default_returns_all_decompositions(self):
        provider = PromptProvider()
        infos = [
            DecompositionInfo(name="x", intent="do x"),
            DecompositionInfo(name="y", intent="do y"),
        ]
        assert provider.select_decompositions(infos) == ["x", "y"]


# ---------------------------------------------------------------------------
# Tests: PrimitiveInfo / DecompositionInfo metadata
# ---------------------------------------------------------------------------


class TestInfoMetadata:
    def _make_agent(self) -> FullAgent:
        return FullAgent(llm=MockLLM())

    def test_primitive_info_fields(self):
        agent = self._make_agent()
        infos = [agent._build_primitive_info(n, m) for n, m in agent._get_primitive_methods()]
        by_name = {i.name: i for i in infos}

        search = by_name["search"]
        assert search.read_only is True
        assert search.deterministic is True
        assert search.docstring == "Search the knowledge base."
        assert search.return_type == "list[str]"
        assert len(search.parameters) == 1
        assert search.parameters[0].name == "query"
        assert search.parameters[0].type == "str"
        assert search.parameters[0].default is None

        delete = by_name["delete_article"]
        assert delete.read_only is False
        assert delete.deterministic is False

    def test_decomposition_info_fields(self):
        agent = self._make_agent()
        decomps = agent._get_decomposition_methods()
        infos = [
            agent._build_decomposition_info(n, m, i, e)
            for n, m, i, e in decomps
        ]
        by_name = {i.name: i for i in infos}

        read = by_name["_example_read"]
        assert read.intent == "Find articles about Python"
        assert read.return_type == "str"
        assert read.source != ""

        write = by_name["_example_write"]
        assert write.intent == "Save a summary note"

    def test_parameter_info_with_default(self):
        """Primitives with default values should populate ParameterInfo.default."""

        class AgentWithDefaults(PlanExecute):
            @primitive(read_only=True)
            def greet(self, name: str = "world") -> str:
                """Say hello."""
                return f"hello {name}"

        agent = AgentWithDefaults(llm=MockLLM())
        info = agent._build_primitive_info("greet", agent.greet)
        assert info.parameters[0].default == "'world'"


# ---------------------------------------------------------------------------
# Tests: Prompt filtering via _get_prompt_primitives / _get_prompt_decompositions
# ---------------------------------------------------------------------------


class TestPromptFiltering:
    def test_no_provider_returns_all(self):
        agent = FullAgent(llm=MockLLM())
        prims = agent._get_prompt_primitives()
        decomps = agent._get_prompt_decompositions()
        assert {n for n, _ in prims} == {"search", "get_article", "save_note", "delete_article"}
        assert {n for n, _, _, _ in decomps} == {"_example_read", "_example_write"}

    def test_read_only_provider_filters_primitives(self):
        config = PlanExecuteConfig(prompt_provider=ReadOnlyProvider())
        agent = FullAgent(llm=MockLLM(), config=config)
        prims = agent._get_prompt_primitives()
        assert {n for n, _ in prims} == {"search", "get_article"}

    def test_read_only_provider_filters_decompositions(self):
        config = PlanExecuteConfig(prompt_provider=ReadOnlyProvider())
        agent = FullAgent(llm=MockLLM(), config=config)
        decomps = agent._get_prompt_decompositions()
        names = {n for n, _, _, _ in decomps}
        # Neither intent contains "read", so both are filtered out
        assert names == set()

    def test_name_filter_provider(self):
        config = PlanExecuteConfig(
            prompt_provider=NameFilterProvider(
                primitive_names={"search", "save_note"},
                decomp_names={"_example_write"},
            )
        )
        agent = FullAgent(llm=MockLLM(), config=config)
        prims = agent._get_prompt_primitives()
        decomps = agent._get_prompt_decompositions()
        assert {n for n, _ in prims} == {"search", "save_note"}
        assert {n for n, _, _, _ in decomps} == {"_example_write"}

    def test_empty_provider_returns_nothing(self):
        config = PlanExecuteConfig(prompt_provider=EmptyProvider())
        agent = FullAgent(llm=MockLLM(), config=config)
        assert agent._get_prompt_primitives() == []
        assert agent._get_prompt_decompositions() == []

    def test_all_provider_returns_everything(self):
        config = PlanExecuteConfig(prompt_provider=AllProvider())
        agent = FullAgent(llm=MockLLM(), config=config)
        prims = agent._get_prompt_primitives()
        assert {n for n, _ in prims} == {"search", "get_article", "save_note", "delete_article"}


# ---------------------------------------------------------------------------
# Tests: Prompt content reflects provider selection
# ---------------------------------------------------------------------------


class TestPromptContent:
    def test_filtered_primitives_appear_in_prompt(self):
        config = PlanExecuteConfig(
            prompt_provider=NameFilterProvider(primitive_names={"search"})
        )
        agent = FullAgent(llm=MockLLM(), config=config)
        prompt = agent.build_plan_prompt("find something")

        assert "search(" in prompt
        assert "get_article(" not in prompt
        assert "save_note(" not in prompt
        assert "delete_article(" not in prompt

    def test_filtered_decompositions_appear_in_prompt(self):
        config = PlanExecuteConfig(
            prompt_provider=NameFilterProvider(
                primitive_names={"search", "get_article", "save_note", "delete_article"},
                decomp_names={"_example_write"},
            )
        )
        agent = FullAgent(llm=MockLLM(), config=config)
        prompt = agent.build_plan_prompt("do something")

        assert "Save a summary note" in prompt
        assert "Find articles about Python" not in prompt

    def test_no_provider_includes_everything_in_prompt(self):
        agent = FullAgent(llm=MockLLM())
        prompt = agent.build_plan_prompt("do something")

        assert "search(" in prompt
        assert "get_article(" in prompt
        assert "save_note(" in prompt
        assert "delete_article(" in prompt
        assert "Find articles about Python" in prompt
        assert "Save a summary note" in prompt


# ---------------------------------------------------------------------------
# Tests: Execution still works with all primitives (not just prompt-visible)
# ---------------------------------------------------------------------------


class TestExecutionUnaffected:
    def test_filtered_primitives_still_callable(self):
        """Even when the provider hides primitives from the prompt,
        they remain available for execution (e.g. if a plan was
        generated externally or from cache)."""
        config = PlanExecuteConfig(
            prompt_provider=NameFilterProvider(primitive_names={"search"})
        )
        agent = FullAgent(llm=MockLLM(), config=config)

        # Directly call a hidden primitive — should still work
        result = agent.save_note(text="hello")
        assert result == "saved: hello"

    def test_execution_uses_all_primitives(self):
        """Plans can call any primitive regardless of prompt provider filtering."""
        llm = MockLLM(responses=[
            '```python\nresult = get_article(id="42")\n```'
        ])
        config = PlanExecuteConfig(
            prompt_provider=NameFilterProvider(primitive_names={"search"})
        )
        agent = FullAgent(llm=llm, config=config)

        # get_article is hidden from the prompt but still executable
        result = agent.run(task="get article 42")
        assert result.success
        assert result.result == "article-42"


# ---------------------------------------------------------------------------
# Tests: ParameterInfo model
# ---------------------------------------------------------------------------


class TestParameterInfoModel:
    def test_required_parameter(self):
        p = ParameterInfo(name="query", type="str")
        assert p.name == "query"
        assert p.type == "str"
        assert p.default is None

    def test_optional_parameter(self):
        p = ParameterInfo(name="limit", type="int", default="10")
        assert p.default == "10"


# ---------------------------------------------------------------------------
# Tests: PrimitiveInfo model
# ---------------------------------------------------------------------------


class TestPrimitiveInfoModel:
    def test_defaults(self):
        info = PrimitiveInfo(name="foo")
        assert info.docstring == ""
        assert info.read_only is False
        assert info.deterministic is True
        assert info.parameters == []
        assert info.return_type == "Any"

    def test_full_construction(self):
        info = PrimitiveInfo(
            name="search",
            docstring="Search things.",
            read_only=True,
            deterministic=False,
            parameters=[ParameterInfo(name="q", type="str")],
            return_type="list[str]",
        )
        assert info.name == "search"
        assert info.parameters[0].name == "q"


# ---------------------------------------------------------------------------
# Tests: DecompositionInfo model
# ---------------------------------------------------------------------------


class TestDecompositionInfoModel:
    def test_defaults(self):
        info = DecompositionInfo(name="bar")
        assert info.intent == ""
        assert info.expanded_intent == ""
        assert info.parameters == []
        assert info.return_type == "Any"
        assert info.source == ""

    def test_full_construction(self):
        info = DecompositionInfo(
            name="_ex",
            intent="Do something",
            expanded_intent="Step 1, Step 2",
            parameters=[ParameterInfo(name="x", type="int")],
            return_type="str",
            source="result = foo(x=x)",
        )
        assert info.intent == "Do something"
        assert info.source == "result = foo(x=x)"
