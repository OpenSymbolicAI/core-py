"""Tests for determinism and prompt-splitting APIs."""

from __future__ import annotations

import pytest

from opensymbolicai import (
    PROMPT_CONTEXT_BEGIN,
    PROMPT_CONTEXT_END,
    PROMPT_DEFINITIONS_BEGIN,
    PROMPT_DEFINITIONS_END,
    PROMPT_INSTRUCTIONS_BEGIN,
    PROMPT_INSTRUCTIONS_END,
    DesignExecute,
    GoalSeeking,
    PlanExecute,
    PromptSections,
    extract_context,
    split_prompt,
)
from opensymbolicai.core import MethodType, primitive

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_PROMPT = f"""You are TestAgent, an AI agent that generates Python code plans.

A test agent for unit tests.

{PROMPT_DEFINITIONS_BEGIN}

## Available Primitive Methods

```python
add(a: float, b: float) -> float
    \"\"\"Add two numbers.\"\"\"
```

## Example Decompositions

No examples available.

{PROMPT_DEFINITIONS_END}

{PROMPT_CONTEXT_BEGIN}

## Task

Generate Python code to accomplish this task: What is 2 + 3?

{PROMPT_CONTEXT_END}

{PROMPT_INSTRUCTIONS_BEGIN}

## Rules

1. Output ONLY Python assignment statements

## Output

```python

{PROMPT_INSTRUCTIONS_END}
"""


class SimplePlanExecute(PlanExecute):
    @primitive(read_only=True)
    def add(self, a: float, b: float) -> float:
        return a + b

    @primitive(read_only=True)
    def multiply(self, a: float, b: float) -> float:
        return a * b


class SimpleDesignExecute(DesignExecute):
    @primitive(read_only=True)
    def lookup(self, key: str) -> str:
        return key.upper()

    @primitive(read_only=True, deterministic=False)
    def resolve(self, name: str) -> str:
        return name


class SimpleGoalSeeking(GoalSeeking):
    @primitive(read_only=True)
    def evaluate_fn(self, x: float) -> float:
        return x * x


# ── @primitive deterministic flag ─────────────────────────────────────────────


class TestPrimitiveDeterministicFlag:
    def test_default_is_true(self) -> None:
        @primitive(read_only=True)
        def my_func(x: int) -> int:
            return x

        assert my_func.__primitive_deterministic__ is True

    def test_explicit_false(self) -> None:
        @primitive(deterministic=False)
        def my_func(x: int) -> int:
            return x

        assert my_func.__primitive_deterministic__ is False

    def test_explicit_true(self) -> None:
        @primitive(deterministic=True)
        def my_func(x: int) -> int:
            return x

        assert my_func.__primitive_deterministic__ is True

    def test_method_type_still_set(self) -> None:
        @primitive(deterministic=False)
        def my_func(x: int) -> int:
            return x

        assert my_func.__method_type__ == MethodType.PRIMITIVE

    def test_read_only_still_works(self) -> None:
        @primitive(read_only=True, deterministic=False)
        def my_func(x: int) -> int:
            return x

        assert my_func.__primitive_read_only__ is True
        assert my_func.__primitive_deterministic__ is False


# ── blueprint_type property ──────────────────────────────────────────────────


class TestBlueprintType:
    def test_plan_execute(self) -> None:
        agent = SimplePlanExecute.__new__(SimplePlanExecute)
        assert agent.blueprint_type == "PlanExecute"

    def test_design_execute(self) -> None:
        agent = SimpleDesignExecute.__new__(SimpleDesignExecute)
        assert agent.blueprint_type == "DesignExecute"

    def test_goal_seeking(self) -> None:
        agent = SimpleGoalSeeking.__new__(SimpleGoalSeeking)
        assert agent.blueprint_type == "GoalSeeking"


# ── compute_signature_hash ───────────────────────────────────────────────────


class TestSignatureHash:
    def _make_agent(self, cls: type) -> PlanExecute:
        agent = cls.__new__(cls)
        return agent

    def test_returns_16_char_hex(self) -> None:
        agent = self._make_agent(SimplePlanExecute)
        h = agent.compute_signature_hash()
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_stable_across_calls(self) -> None:
        agent = self._make_agent(SimplePlanExecute)
        h1 = agent.compute_signature_hash()
        h2 = agent.compute_signature_hash()
        assert h1 == h2

    def test_different_agents_different_hash(self) -> None:
        a1 = self._make_agent(SimplePlanExecute)
        a2 = self._make_agent(SimpleDesignExecute)
        assert a1.compute_signature_hash() != a2.compute_signature_hash()

    def test_hash_changes_when_primitives_differ(self) -> None:
        class AgentV1(PlanExecute):
            @primitive(read_only=True)
            def add(self, a: float, b: float) -> float:
                return a + b

        class AgentV2(PlanExecute):
            @primitive(read_only=True)
            def add(self, a: float, b: float) -> float:
                return a + b

            @primitive(read_only=True)
            def subtract(self, a: float, b: float) -> float:
                return a - b

        h1 = self._make_agent(AgentV1).compute_signature_hash()
        h2 = self._make_agent(AgentV2).compute_signature_hash()
        assert h1 != h2


# ── _get_primitive_determinism_map ────────────────────────────────────────────


class TestDeterminismMap:
    def test_all_deterministic(self) -> None:
        agent = SimplePlanExecute.__new__(SimplePlanExecute)
        det_map = agent._get_primitive_determinism_map()
        assert det_map == {"add": True, "multiply": True}

    def test_mixed_determinism(self) -> None:
        agent = SimpleDesignExecute.__new__(SimpleDesignExecute)
        det_map = agent._get_primitive_determinism_map()
        assert det_map == {"lookup": True, "resolve": False}


# ── split_prompt ─────────────────────────────────────────────────────────────


class TestSplitPrompt:
    def test_returns_prompt_sections(self) -> None:
        result = split_prompt(SAMPLE_PROMPT)
        assert isinstance(result, PromptSections)

    def test_preamble(self) -> None:
        result = split_prompt(SAMPLE_PROMPT)
        assert "You are TestAgent" in result.preamble
        assert "A test agent for unit tests." in result.preamble

    def test_definitions(self) -> None:
        result = split_prompt(SAMPLE_PROMPT)
        assert "Available Primitive Methods" in result.definitions
        assert "add(a: float, b: float)" in result.definitions

    def test_context(self) -> None:
        result = split_prompt(SAMPLE_PROMPT)
        assert "What is 2 + 3?" in result.context
        assert "## Task" in result.context

    def test_instructions(self) -> None:
        result = split_prompt(SAMPLE_PROMPT)
        assert "## Rules" in result.instructions
        assert "Output ONLY Python assignment statements" in result.instructions

    def test_no_markers_returns_full_text_as_preamble(self) -> None:
        result = split_prompt("Just a plain prompt with no markers.")
        assert result.preamble == "Just a plain prompt with no markers."
        assert result.definitions == ""
        assert result.context == ""
        assert result.instructions == ""


# ── extract_context ──────────────────────────────────────────────────────────


class TestExtractContext:
    def test_extracts_context(self) -> None:
        ctx = extract_context(SAMPLE_PROMPT)
        assert "What is 2 + 3?" in ctx
        assert "## Task" in ctx

    def test_no_definitions_or_instructions(self) -> None:
        ctx = extract_context(SAMPLE_PROMPT)
        assert "Available Primitive Methods" not in ctx
        assert "## Rules" not in ctx
        assert PROMPT_DEFINITIONS_BEGIN not in ctx
        assert PROMPT_INSTRUCTIONS_BEGIN not in ctx

    def test_raises_on_missing_markers(self) -> None:
        with pytest.raises(ValueError, match="CONTEXT markers"):
            extract_context("No markers here.")
