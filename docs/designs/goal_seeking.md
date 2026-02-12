# GoalSeeking Blueprint Design

> **Status**: Draft
> **Author**: Design Session
> **Date**: 2024

## Overview

The **GoalSeeking** pattern is an iterative agent that pursues a goal through repeated plan-execute-evaluate cycles until the goal is satisfied or a termination condition is met.

This pattern is ideal for:
- Multi-hop RAG / Research agents
- Iterative problem solving
- Goal-directed exploration
- Adaptive task completion

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         GoalSeeking                              │
│                                                                  │
│   ┌──────────┐    ┌──────────────────┐                           │
│   │  GOAL    │───▶│  PLAN EVALUATOR  │  (once, before loop)      │
│   └──────────┘    └────────┬─────────┘                           │
│                            │ evaluator code                      │
│                            ▼                                     │
│                   ┌─────────────────┐                            │
│                   │  PLAN ITERATION │  (each iteration)          │
│                   └────────┬────────┘                            │
│                            │          uses context insights      │
│                            ▼          (not raw results)          │
│                   ┌──────────────┐                               │
│                   │   EXECUTE    │                               │
│                   └──────┬───────┘                               │
│                          │ raw ExecutionResult                   │
│                          ▼                                       │
│                   ┌──────────────┐                               │
│                   │  INTROSPECT  │  update_context()             │
│                   └──────┬───────┘  raw result → insights       │
│                          │ structured insights in context        │
│                          ▼                                       │
│                   ┌──────────────┐                               │
│                   │   EVALUATE   │  checks context, not result   │
│                   └──────┬───────┘                               │
│                          │                                       │
│                    ┌─────┴─────┐                                 │
│                    │Goal Met?  │                                  │
│              ┌─────│   NO      │                                 │
│              │     └─────┬─────┘                                 │
│              │           │YES                                    │
│              ▼           ▼                                       │
│         ┌─────────┐ ┌───────────┐                                │
│         │ REPLAN  │ │  RESULT   │                                │
│         └────┬────┘ └───────────┘                                │
│              │                                                   │
│              └──────────▶ (back to PLAN ITERATION)               │
└──────────────────────────────────────────────────────────────────┘
```

The evaluator is resolved once before the loop starts:

```
1. @evaluator method exists?  →  use it directly (static, user-defined)
2. otherwise                  →  plan_evaluator(goal) generates code (dynamic)
```

## Comparison with PlanAndExecute

| Feature | PlanAndExecute | GoalSeeking |
|---------|----------------|-------------|
| **Execution Model** | Single plan → execute → done | Iterative until goal met |
| **Goal Awareness** | Implicit in task | Explicit goal with evaluation |
| **Context** | Single turn (or multi-turn history) | Accumulated findings across iterations |
| **Termination** | Plan completes | Goal achieved OR max iterations |
| **Evaluation** | None (just execution success) | `@evaluator` (static) or LLM-generated code (dynamic) |
| **Replanning** | Only on validation failure | After each iteration based on results |

---

## Introspection Boundary

**Core principle: raw execution results never leak into planning prompts or evaluation.**

The whole point of SymbolicAI is that data and instructions don't mix in uncontrolled ways. The LLM generates plans (code), and execution is deterministic. If we dump raw `ExecutionResult` into the next planning prompt, we break that boundary — the LLM sees unstructured data that could be arbitrarily large, contain noise, or even be adversarial.

Instead, raw results are **introspected** — by code or by LLM — into structured insights that live in `GoalContext`. Only those insights are used by the planner and evaluator.

```
execute(plan) → raw ExecutionResult
       │
       ▼
  update_context()          ← THE INTROSPECTION BOUNDARY
  raw result → structured insights written into context fields
       │
       ▼
  context.findings          ← planner sees this
  context.sources_checked   ← evaluator checks this
  context.coverage_summary  ← NOT raw ExecutionResult
```

**What this means in practice:**

- `build_goal_prompt()` reads from **context fields** (insights), never from raw `ExecutionResult`
- The evaluator receives `(goal, context)` — no `latest_result` parameter
- `update_context()` is the bridge: it receives the raw `ExecutionResult` and writes structured insights into context
- Raw results are preserved in `Iteration.execution_result` for **traceability only** — they are never passed to the LLM

---

## Data Models

### GoalStatus

```python
class GoalStatus(str, Enum):
    """Status of goal pursuit."""
    PURSUING = "pursuing"          # Still working toward goal
    ACHIEVED = "achieved"          # Goal satisfied
    FAILED = "failed"              # Cannot achieve goal
    MAX_ITERATIONS = "max_iterations"  # Hit iteration limit
```

### GoalEvaluation

Result of evaluating progress toward goal after each iteration.

```python
class GoalEvaluation(BaseModel):
    """Minimal result of evaluating progress toward goal.

    Subclass to add domain-specific fields (findings, confidence, etc.)
    """

    goal_achieved: bool
    """Whether the goal has been achieved."""


# Example extension with confidence scoring:
class ScoredEvaluation(GoalEvaluation):
    confidence: float = Field(ge=0.0, le=1.0)
```

### Iteration

A single iteration of the goal-seeking loop.

```python
class Iteration(BaseModel):
    """A single iteration of the goal-seeking loop."""

    iteration_number: int
    """1-based iteration number."""

    plan_result: PlanResult
    """The plan generated for this iteration."""

    execution_result: ExecutionResult
    """Result of executing the plan."""

    evaluation: GoalEvaluation
    """Evaluation of progress toward goal."""
```

### GoalContext

The knowledge layer between execution and planning. Subclass to add domain-specific
fields for structured insights. The planner and evaluator read from context — never
from raw execution results.

```python
class GoalContext(BaseModel):
    """Accumulated structured knowledge across iterations.

    This is the introspection boundary. Subclass to add domain-specific
    insight fields that update_context() populates from raw execution results.
    The planner and evaluator only see these fields — never raw results.
    """

    goal: str
    """The original goal."""

    iterations: list[Iteration] = []
    """All completed iterations."""

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)

    @property
    def last_evaluation(self) -> GoalEvaluation | None:
        if self.iterations:
            return self.iterations[-1].evaluation
        return None


# Example extension for research agents:
class ResearchContext(GoalContext):
    findings: list[str] = []

    def add_finding(self, finding: str) -> None:
        if finding not in self.findings:
            self.findings.append(finding)
```

### GoalSeekingConfig

```python
class GoalSeekingConfig(BaseModel):
    """Configuration for GoalSeeking agents."""

    max_iterations: int = 10
    """Maximum iterations before stopping."""
```

### GoalSeekingResult

Final result from a complete goal-seeking run.

```python
class GoalSeekingResult(BaseModel):
    """Minimal result from a complete goal-seeking run.

    Subclass to add domain-specific result fields.
    """

    goal: str
    """The original goal."""

    status: GoalStatus
    """Final status."""

    final_answer: Any | None = None
    """The final result/answer."""

    iterations: list[Iteration] = []
    """All iterations performed."""

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)

    @property
    def succeeded(self) -> bool:
        return self.status == GoalStatus.ACHIEVED
```

---

## GoalSeeking Class

### Constructor

```python
class GoalSeeking(Planner):
    """Agent that iteratively pursues a goal through plan-execute-evaluate cycles."""

    def __init__(
        self,
        llm: LLM | LLMConfig,
        name: str = "",
        description: str = "",
        config: GoalSeekingConfig | None = None,
    ) -> None:
        """Initialize the GoalSeeking agent.

        Args:
            llm: LLM for planning.
            name: Agent name for prompts.
            description: Agent description for prompts.
            config: Configuration options.
        """
        ...
```

### Core Methods

#### Prompt Building

```python
def build_goal_prompt(self, goal: str, context: GoalContext) -> str:
    """Build the prompt for planning the next iteration.

    Includes:
    - The original goal
    - Available primitives (from introspection)
    - Decomposition examples
    - Structured insights from context (NOT raw execution results)

    The planner sees context fields like context.findings,
    context.sources_checked — never raw ExecutionResult data.

    Returns:
        Complete prompt for plan generation.
    """
    ...
```

#### Planning

```python
def plan_iteration(self, goal: str, context: GoalContext) -> PlanResult:
    """Generate a plan for the next iteration.

    Uses accumulated context to inform planning.
    Supports retries on validation failure.

    Args:
        goal: The goal being pursued.
        context: Accumulated context from previous iterations.

    Returns:
        PlanResult with the generated plan.
    """
    ...
```

#### Evaluation (two-tier: static or dynamic)

Evaluation is resolved once before the loop starts. Two tiers, checked in order:

**Tier 1: Static (`@evaluator` decorator)**

If the user defines an `@evaluator` method, it is used directly. The runtime discovers it via introspection (same pattern as `@primitive` and `@decomposition`). At most one `@evaluator` per agent.

The evaluator receives `(goal, context)` — no raw result. By the time the evaluator runs, `update_context()` has already introspected the raw result into context fields.

```python
@evaluator
def check_goal(self, goal: str, context: GoalContext) -> GoalEvaluation:
    return GoalEvaluation(goal_achieved=len(context.findings) >= 5)
```

**Tier 2: Dynamic (`plan_evaluator()` — LLM-generated code)**

If no `@evaluator` is found, the LLM generates evaluator code from the goal. This code is validated and executed using the same sandboxed pipeline as iteration plans.

```python
def plan_evaluator(self, goal: str) -> str:
    """Generate evaluator code from the goal using the LLM.

    Called once before the loop starts. The generated code:
    - Has access to: goal, context, self (primitives)
    - Must assign `result = GoalEvaluation(goal_achieved=...)`
    - Goes through the same validation as plans (no imports, no dangerous builtins)
    - Checks context fields (insights), NOT raw ExecutionResult

    Returns:
        Python code string for evaluation.
    """
    prompt = self.build_evaluator_prompt(goal)
    response = self._llm.generate(prompt)
    code = self._extract_code_block(response.text)
    self.validate_plan(code)
    return code

def build_evaluator_prompt(self, goal: str) -> str:
    """Build the prompt for evaluator code generation.

    Includes:
    - The goal
    - Available primitives (from introspection)
    - GoalContext structure and its insight fields
    - Available variables: goal, context, self

    Override for custom prompt construction.
    """
    ...

def run_evaluator(
    self,
    evaluator_code: str,
    goal: str,
    context: GoalContext,
) -> GoalEvaluation:
    """Execute the generated evaluator code in a sandboxed namespace.

    The code runs with the same execution engine as plans.
    The namespace includes: goal, context, self, GoalEvaluation.

    Returns:
        GoalEvaluation from the `result` variable in the executed code.
    """
    ...
```

**Example: what the LLM generates for "find 100 cars and compare their cost"**

```python
# Generated evaluator code — checks context insights, not raw results
count = self.count_collected(context)
has_comparison = context.has_comparison_report
result = GoalEvaluation(goal_achieved=count >= 100 and has_comparison)
```

The generated code can also call the LLM:

```python
# Generated evaluator code — uses LLM to assess context
prompt = f"Goal: {goal}\nFindings: {context.findings}\nSources checked: {context.sources_checked}\nIs the goal fully achieved? YES or NO."
response = self._llm.generate(prompt)
result = GoalEvaluation(goal_achieved="YES" in response.text.upper())
```

#### Termination Logic

```python
def should_continue(
    self,
    context: GoalContext,
    evaluation: GoalEvaluation,
) -> tuple[bool, GoalStatus]:
    """Determine if the loop should continue.

    Checks:
    1. Goal achieved → stop (ACHIEVED)
    2. Max iterations exceeded → stop (MAX_ITERATIONS)
    3. Otherwise → continue (PURSUING)

    Override for custom termination logic.

    Returns:
        Tuple of (should_continue, status_if_stopping)
    """
    ...
```

### Main Orchestration

#### Simple Execution

```python
def seek(self, goal: str) -> GoalSeekingResult:
    """Pursue a goal through iterative plan-execute-evaluate cycles.

    Args:
        goal: The goal to achieve (natural language).

    Returns:
        GoalSeekingResult with final answer and iteration history.
    """
    context = self.create_context(goal)

    # Resolve evaluator: static (@evaluator) or dynamic (LLM-generated)
    static_evaluator = self._get_evaluator_method()
    evaluator_code: str | None = None
    if not static_evaluator:
        evaluator_code = self.plan_evaluator(goal)

    while True:
        # 1. Plan next iteration (reads context insights, not raw results)
        plan_result = self.plan_iteration(goal, context)

        # 2. Execute the plan
        exec_result = self.execute(plan_result.plan)

        # 3. Introspect: derive structured insights from raw result
        #    This is the boundary — raw data goes in, structured knowledge comes out
        self.update_context(context, exec_result)

        # 4. Evaluate progress (checks context insights, not raw result)
        if static_evaluator:
            evaluation = static_evaluator(goal, context)
        else:
            evaluation = self.run_evaluator(evaluator_code, goal, context)

        # 5. Record iteration (raw result preserved for traceability)
        iteration = Iteration(
            iteration_number=context.iteration_count + 1,
            plan_result=plan_result,
            execution_result=exec_result,
            evaluation=evaluation,
        )
        context.iterations.append(iteration)

        # 6. Check termination
        should_continue, status = self.should_continue(context, evaluation)

        if not should_continue:
            return GoalSeekingResult(
                goal=goal,
                status=status,
                final_answer=self._extract_final_answer(context),
                iterations=context.iterations,
            )
```

### Hooks for Customization

```python
def on_iteration_start(self, iteration_number: int, context: GoalContext) -> None:
    """Hook called at the start of each iteration.

    Override for logging, metrics, or custom setup.
    """
    pass

def on_iteration_complete(self, iteration: Iteration, context: GoalContext) -> None:
    """Hook called after each iteration completes.

    Override for logging, metrics, or triggering side effects.
    """
    pass

def on_goal_achieved(self, result: GoalSeekingResult) -> None:
    """Hook called when goal is achieved.

    Override for notifications, cleanup, or celebration.
    """
    pass

def update_context(self, context: GoalContext, execution_result: ExecutionResult) -> None:
    """THE INTROSPECTION BOUNDARY. Called after each execution.

    This is where raw ExecutionResult is introspected into structured
    insights on the context. The planner and evaluator only see what
    this method writes into context — never the raw result.

    Override to extract domain-specific insights from the execution result.

    Example:
        def update_context(self, context: ResearchContext, execution_result: ExecutionResult) -> None:
            # Introspect: extract structured insights from raw result
            last_step = execution_result.trace.last_step
            if last_step and isinstance(last_step.result_value, list):
                for finding in last_step.result_value:
                    context.add_finding(finding)
            context.sources_checked += 1
    """
    pass

def create_context(self, goal: str) -> GoalContext:
    """Factory hook to create the initial context.

    Override to return a custom context subclass.

    Example:
        def create_context(self, goal: str) -> ResearchContext:
            return ResearchContext(goal=goal)
    """
    return GoalContext(goal=goal)
```

### Answer Extraction

```python
def _extract_final_answer(self, context: GoalContext) -> Any:
    """Extract the final answer from context.

    Default: returns last execution result value.
    Override for custom answer extraction logic.
    """
    if context.iterations:
        return context.iterations[-1].execution_result.trace.last_step.result_value
    return None
```

---

## Example Implementations

### Research Agent

```python
# Custom context for research
class ResearchContext(GoalContext):
    findings: list[str] = []


class ResearchAgent(GoalSeeking):
    """Multi-hop research agent that answers complex questions."""

    def __init__(self, llm: LLM, vector_store: VectorStore, min_facts: int = 5):
        super().__init__(
            llm=llm,
            name="ResearchAgent",
            description="Answers complex questions through iterative research",
            config=GoalSeekingConfig(max_iterations=5),
        )
        self.vector_store = vector_store
        self.min_facts = min_facts

    def create_context(self, goal: str) -> ResearchContext:
        return ResearchContext(goal=goal)

    def update_context(self, context: ResearchContext, execution_result: ExecutionResult) -> None:
        # INTROSPECT: extract structured insights from raw result
        last_step = execution_result.trace.last_step
        if last_step and isinstance(last_step.result_value, list):
            context.findings.extend(last_step.result_value)

    @primitive(read_only=True)
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search the knowledge base for relevant documents."""
        return self.vector_store.search(query, top_k=top_k)

    @primitive(read_only=True)
    def read_document(self, doc_id: str) -> str:
        """Read the full content of a document."""
        return self.vector_store.get_document(doc_id)

    @primitive(read_only=True)
    def extract_facts(self, text: str, topic: str) -> list[str]:
        """Extract facts about a topic from text."""
        ...

    @primitive(read_only=True)
    def synthesize_answer(self, facts: list[str], question: str) -> str:
        """Synthesize facts into a coherent answer."""
        ...

    @primitive(read_only=True)
    def score_answer(self, answer: str, question: str) -> float:
        """Score how well an answer addresses the question (0-1)."""
        ...

    @evaluator
    def check_research_goal(self, goal: str, context: ResearchContext) -> GoalEvaluation:
        """Evaluate based on introspected context — not raw results."""
        if len(context.findings) < self.min_facts:
            return GoalEvaluation(goal_achieved=False)

        answer = self.synthesize_answer(context.findings, goal)
        score = self.score_answer(answer, goal)
        return GoalEvaluation(goal_achieved=score >= 0.8)

    @decomposition(intent="Find information about a topic")
    def _example_research(self) -> str:
        results = self.search("topic query")
        doc = self.read_document(results[0].id)
        facts = self.extract_facts(doc, "topic")
        answer = self.synthesize_answer(facts, "What is the topic?")
        return answer


# Usage
agent = ResearchAgent(llm=my_llm, vector_store=my_store)
result = agent.seek("What were the economic impacts of the 2008 financial crisis?")

print(f"Answer: {result.final_answer}")
print(f"Iterations: {result.iteration_count}")
```

### Web Research Agent (dynamic evaluator)

No `@evaluator` defined — the LLM generates evaluator code from the goal.

```python
class WebResearchAgent(GoalSeeking):
    """Agent that researches topics using web search and page reading.

    No @evaluator — uses dynamic evaluator generation.
    The LLM generates evaluation code based on the goal.
    """

    @primitive(read_only=True)
    def web_search(self, query: str) -> list[SearchResult]:
        """Search the web for relevant pages."""
        ...

    @primitive(read_only=True)
    def fetch_page(self, url: str) -> str:
        """Fetch and extract text from a web page."""
        ...

    @primitive(read_only=True)
    def summarize(self, text: str, focus: str) -> str:
        """Summarize text with focus on specific aspects."""
        ...

    @primitive(read_only=True)
    def compare_sources(self, summaries: list[str]) -> str:
        """Compare and synthesize multiple source summaries."""
        ...


# Usage — evaluator is generated dynamically from the goal:
agent = WebResearchAgent(llm=my_llm)

# Goal: "find 5 sources about climate change and compare them"
# LLM generates evaluator code that checks context insights:
#   source_count = context.sources_checked
#   has_comparison = context.has_comparison
#   result = GoalEvaluation(goal_achieved=source_count >= 5 and has_comparison)
result = agent.seek("find 5 sources about climate change and compare them")
```

---

## Evaluator Patterns

### Tier 1: Static (`@evaluator`)

Use when the evaluation logic is known at class definition time. The evaluator checks **context fields** (insights), not raw results.

#### Rule-Based

```python
class DataCollectionAgent(GoalSeeking):
    """Collects N data points matching criteria."""

    def __init__(self, llm: LLM, required_count: int = 10):
        super().__init__(llm=llm, name="DataCollector")
        self.required_count = required_count

    def update_context(self, context, execution_result) -> None:
        # Introspect: count items from raw result into context
        last_step = execution_result.trace.last_step
        if last_step and isinstance(last_step.result_value, list):
            context.collected_items.extend(last_step.result_value)

    @evaluator
    def check_count(self, goal, context) -> GoalEvaluation:
        return GoalEvaluation(goal_achieved=len(context.collected_items) >= self.required_count)
```

#### Primitive-Based

```python
class QualityResearchAgent(GoalSeeking):

    @primitive(read_only=True)
    def score_answer_quality(self, answer: str, question: str) -> float:
        """Score answer quality from 0-1."""
        return self.quality_api.score(answer, question)

    @evaluator
    def check_quality(self, goal, context) -> GoalEvaluation:
        if not context.latest_answer:
            return GoalEvaluation(goal_achieved=False)
        score = self.score_answer_quality(context.latest_answer, goal)
        return GoalEvaluation(goal_achieved=score >= 0.8)
```

### Tier 2: Dynamic (LLM-generated)

Use when the evaluation criteria come from the goal itself. No `@evaluator` needed — the LLM generates the code.

The agent just defines primitives. The evaluator is generated dynamically based on what the user asks.

```python
class FlexibleAgent(GoalSeeking):
    """Agent where evaluation criteria are derived from the goal."""

    @primitive(read_only=True)
    def search(self, query: str) -> list[str]:
        """Search for items."""
        ...

    @primitive(read_only=True)
    def count_results(self, context: GoalContext) -> int:
        """Count accumulated results."""
        ...

    @primitive(read_only=True)
    def has_report(self, context: GoalContext) -> bool:
        """Check if a report has been generated."""
        ...

    # No @evaluator — LLM generates evaluator code from the goal.
    # "find 100 cars" → count_results(context) >= 100
    # "find 50 hotels" → count_results(context) >= 50
    # Same agent, different goals, different evaluators.
```

#### What the LLM generates

For `agent.seek("find 100 cars and compare their cost")`:

```python
# Generated evaluator code
count = self.count_results(context)
has_report = self.has_report(context)
result = GoalEvaluation(goal_achieved=count >= 100 and has_report)
```

For `agent.seek("find any 3 luxury hotels in Paris")`:

```python
# Generated evaluator code
count = self.count_results(context)
result = GoalEvaluation(goal_achieved=count >= 3)
```

Same agent class, different generated evaluators — criteria extracted from the goal.

---

## Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Introspection Boundary** | `update_context()` converts raw results to structured insights | Data and instructions don't mix — planner/evaluator never see raw `ExecutionResult` |
| **Context as Knowledge Layer** | `GoalContext` holds insights, not raw data | Planner and evaluator read structured fields, raw results preserved in `Iteration` for traceability only |
| **Two-tier Evaluation** | `@evaluator` (static) or LLM-generated code (dynamic) | Static for known criteria, dynamic for goal-derived criteria |
| **Evaluator Checks Context** | Evaluator receives `(goal, context)` — no raw result | Introspection has already happened; evaluator works with structured knowledge |
| **Evaluator as Code** | Generated evaluator is Python code, same as plans | Reuses validation, sandboxing, and execution engine |
| **Generated Once** | `plan_evaluator()` runs once before the loop | Evaluation criteria don't change across iterations |
| **Minimal GoalEvaluation** | Just `goal_achieved: bool` | Subclass to add confidence, findings, etc. |
| **Minimal Config** | Just `max_iterations` | Subclass to add thresholds, early stopping, etc. |
| **Hooks** | Lifecycle callbacks | Allow logging, metrics, custom logic |
| **Inherits PlanExecute** | Reuse validation/execution | Same safe execution guarantees |

---

## File Structure

```
src/opensymbolicai/
├── blueprints/
│   ├── __init__.py
│   ├── planner.py          # Base Planner ABC
│   ├── plan_execute.py     # Existing PlanExecute
│   └── goal_seeking.py     # New GoalSeeking (NEW)
└── models.py               # Add GoalEvaluation, GoalContext, etc.
```

---

## Future Extensions

These can be added later by subclassing:

- **Confidence scoring**: Subclass `GoalEvaluation` to add `confidence: float`
- **Checkpointing**: Add `seek_stepwise()` for long-running tasks
- **Early stopping**: Subclass config to add `no_progress_threshold`
- **Human-in-the-loop**: Add `PAUSED` / `AWAITING_INPUT` statuses

---

## Next Steps

1. [ ] Review design
2. [ ] Implement goal-seeking models
3. [ ] Implement `GoalSeeking` class
4. [ ] Add tests
5. [ ] Add example agent
