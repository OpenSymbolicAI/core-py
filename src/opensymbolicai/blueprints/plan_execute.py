"""Plan-and-execute blueprint: generates and executes Python plans using LLMs."""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
import time
from collections.abc import Callable, Iterator
from typing import Any

from opensymbolicai.blueprints.planner import Planner
from opensymbolicai.checkpoint import (
    CheckpointStatus,
    CheckpointStore,
    ExecutionCheckpoint,
    PendingMutation,
    PlanContext,
    SerializerRegistry,
    default_serializer_registry,
)
from opensymbolicai.core import MethodType
from opensymbolicai.llm import LLM, LLMConfig, create_llm
from opensymbolicai.models import (
    MUTATION_REJECTED_PREFIX,
    NONE_TYPE_NAME,
    NULL_JSON,
    PLAN_COMPILE_SOURCE,
    ArgumentValue,
    ConversationTurn,
    ExecutionMetrics,
    ExecutionResult,
    ExecutionStep,
    ExecutionTrace,
    LLMInteraction,
    MutationDetection,
    MutationHookContext,
    OrchestrationResult,
    PlanAnalysis,
    PlanAttempt,
    PlanExecuteConfig,
    PlanGeneration,
    PlanResult,
    PrimitiveCall,
    TokenUsage,
    empty_builtins,
)
from opensymbolicai.observability.events import (
    EventType,
    ExecutionSummary,
    RunCompleteSummary,
    RunErrorSummary,
)
from opensymbolicai.observability.tracer import Tracer

# Default safe builtins allowed in execution
DEFAULT_ALLOWED_BUILTINS: dict[str, Any] = {
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "pow": pow,
    "divmod": divmod,
    "all": all,
    "any": any,
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "frozenset": frozenset,
    "repr": repr,
    "True": True,
    "False": False,
    "None": None,
}


class PlanExecute(Planner):
    """Agent that generates and executes Python plans using LLMs.

    PlanExecute orchestrators use an LLM to generate Python code that calls
    primitive methods, then execute that code step-by-step with full tracing.

    Subclasses should:
    1. Define primitive methods using the @primitive decorator
    2. Optionally define decomposition examples using @decomposition

    Example:
        class Calculator(PlanExecute):
            @primitive(read_only=True)
            def add(self, a: float, b: float) -> float:
                return a + b

            @decomposition(intent="What is 2 + 3?")
            def _example_add(self) -> float:
                result = self.add(2, 3)
                return result
    """

    DANGEROUS_BUILTINS: set[str] = {
        "exec",
        "eval",
        "compile",
        "open",
        "__import__",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "type",
        "isinstance",
        "issubclass",
        "callable",
        "breakpoint",
        "input",
        "memoryview",
        "object",
    }

    def __init__(
        self,
        llm: LLM | LLMConfig,
        name: str = "",
        description: str = "",
        config: PlanExecuteConfig | None = None,
    ) -> None:
        """Initialize the PlanExecute agent.

        Args:
            llm: LLM instance or config for plan generation.
            name: Agent name for prompts.
            description: Agent description for prompts.
            config: Extended configuration options.
        """
        if isinstance(llm, LLMConfig):
            self._llm = create_llm(llm)
        else:
            self._llm = llm

        self.name = name or self.__class__.__name__
        self.description = description or self.__class__.__doc__ or ""
        self.config = config or PlanExecuteConfig()
        self.allowed_builtins = (
            self.config.allowed_builtins
            if self.config.allowed_builtins is not None
            else DEFAULT_ALLOWED_BUILTINS.copy()
        )

        # Multi-turn state
        self._history: list[ConversationTurn] = []
        self._persisted_namespace: dict[str, Any] = {}

        # Observability
        self._tracer: Tracer | None = None
        if self.config.observability and self.config.observability.enabled:
            self._tracer = Tracer(self.config.observability, type(self).__name__)

    @property
    def history(self) -> list[ConversationTurn]:
        """Get the conversation history (multi-turn mode only)."""
        return self._history

    @property
    def persisted_variables(self) -> dict[str, Any]:
        """Get the persisted variables from previous turns (multi-turn mode only)."""
        return self._persisted_namespace.copy()

    # -------------------------------------------------------------------------
    # Introspection: Extract primitives and decompositions
    # -------------------------------------------------------------------------

    def _get_primitive_methods(self) -> list[tuple[str, Callable[..., Any]]]:
        """Get all methods decorated with @primitive."""
        primitives = []
        for name in dir(self):
            if name.startswith("_"):
                continue
            method = getattr(self, name, None)
            if (
                callable(method)
                and hasattr(method, "__method_type__")
                and method.__method_type__ == MethodType.PRIMITIVE
            ):
                primitives.append((name, method))
        return primitives

    def _get_decomposition_methods(
        self,
    ) -> list[tuple[str, Callable[..., Any], str, str]]:
        """Get all methods decorated with @decomposition.

        Returns:
            List of (name, method, intent, expanded_intent) tuples.
        """
        decompositions = []
        for name in dir(self):
            method = getattr(self, name, None)
            if (
                callable(method)
                and hasattr(method, "__method_type__")
                and method.__method_type__ == MethodType.DECOMPOSITION
            ):
                intent = getattr(method, "__decomposition_intent__", "")
                expanded = getattr(method, "__decomposition_expanded_intent__", "")
                decompositions.append((name, method, intent, expanded))
        return decompositions

    def _get_primitive_names(self) -> set[str]:
        """Get the names of all primitive methods."""
        return {name for name, _ in self._get_primitive_methods()}

    def _get_primitive_read_only_map(self) -> dict[str, bool]:
        """Get a mapping of primitive names to their read_only status."""
        return {
            name: getattr(method, "__primitive_read_only__", False)
            for name, method in self._get_primitive_methods()
        }

    def _format_primitive_signature(self, name: str, method: Callable[..., Any]) -> str:
        """Format a primitive method's signature and docstring for the prompt."""
        sig = inspect.signature(method)
        params = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            annotation = (
                param.annotation.__name__
                if hasattr(param.annotation, "__name__")
                else str(param.annotation)
            )
            if param.annotation == inspect.Parameter.empty:
                annotation = "Any"
            if param.default != inspect.Parameter.empty:
                params.append(f"{param_name}: {annotation} = {param.default!r}")
            else:
                params.append(f"{param_name}: {annotation}")

        return_annotation = sig.return_annotation
        if return_annotation == inspect.Signature.empty:
            return_type = "Any"
        elif hasattr(return_annotation, "__name__"):
            return_type = return_annotation.__name__
        else:
            return_type = str(return_annotation)

        signature_str = f"{name}({', '.join(params)}) -> {return_type}"
        docstring = inspect.getdoc(method) or ""
        return f'{signature_str}\n    """{docstring}"""'

    def _get_decomposition_source(self, method: Callable[..., Any]) -> str:
        """Extract the source code body of a decomposition method."""
        try:
            source = inspect.getsource(method)
            # Dedent to remove class-level indentation before parsing
            source = textwrap.dedent(source)
            # Parse and extract just the function body (skip decorator and def line)
            tree = ast.parse(source)
            if tree.body and isinstance(tree.body[0], ast.FunctionDef):
                func_def = tree.body[0]
                # Get the body statements as source
                body_lines = []
                for stmt in func_def.body:
                    # Skip docstrings
                    if isinstance(stmt, ast.Expr) and isinstance(
                        stmt.value, ast.Constant
                    ):
                        continue
                    body_lines.append(ast.unparse(stmt))
                # Strip self. prefixes — plans use bare function calls, not method calls
                return "\n".join(body_lines).replace("self.", "")
        except (OSError, TypeError):
            pass
        return ""

    # -------------------------------------------------------------------------
    # Prompt Building
    # -------------------------------------------------------------------------

    def _format_type_definitions(
        self, primitives: list[tuple[str, Callable[..., Any]]]
    ) -> str:
        """Build a ## Type Definitions prompt section for Pydantic BaseModel types.

        Scans parameter and return type annotations of all primitives,
        recursively unwraps generic wrappers (list[X], X | None, Union, etc.),
        collects any BaseModel subclasses found, and returns a formatted section
        listing each model's fields and types.  Returns "" when no models are
        referenced.
        """
        from typing import get_args, get_type_hints

        from pydantic import BaseModel

        found: set[type[BaseModel]] = set()

        def _extract_models(annotation: Any) -> None:
            if (
                isinstance(annotation, type)
                and issubclass(annotation, BaseModel)
                and annotation is not BaseModel
            ):
                found.add(annotation)
                return
            for arg in get_args(annotation):
                _extract_models(arg)

        for _name, method in primitives:
            try:
                hints = get_type_hints(method)
            except Exception:
                # Fallback: use inspect.signature annotations directly
                sig = inspect.signature(method)
                hints = {}
                for pname, param in sig.parameters.items():
                    if pname != "self" and param.annotation != inspect.Parameter.empty:
                        hints[pname] = param.annotation
                if sig.return_annotation != inspect.Signature.empty:
                    hints["return"] = sig.return_annotation

            for hint in hints.values():
                _extract_models(hint)

        if not found:
            return ""

        models = sorted(found, key=lambda cls: cls.__name__)
        lines: list[str] = []
        for model in models:
            fields: list[str] = []
            for field_name, field_info in model.model_fields.items():
                ann = field_info.annotation
                type_str = (
                    ann.__name__
                    if ann is not None and hasattr(ann, "__name__")
                    else str(ann)
                )
                fields.append(f"{field_name}: {type_str}")
            lines.append(f"{model.__name__}({', '.join(fields)})")

        return "\n## Type Definitions\n\n" + "\n".join(lines) + "\n"

    def _format_history_for_prompt(self) -> str:
        """Format conversation history for inclusion in the prompt."""
        if not self._history:
            return ""

        sections = []
        for i, turn in enumerate(self._history, 1):
            section = f"### Turn {i}\n"
            section += f"Task: {turn.task}\n"
            section += f"Plan:\n```python\n{turn.plan}\n```\n"
            if turn.success:
                section += f"Result: {turn.result!r}"
            else:
                section += f"Error: {turn.error}"
            sections.append(section)

        return "\n\n".join(sections)

    def build_plan_prompt(self, task: str, feedback: str | None = None) -> str:
        """Build the prompt for the LLM to generate a plan.

        Override this method in subclasses to customize the prompt sent to the LLM.

        Args:
            task: The task description to plan for.
            feedback: Optional error feedback from a previous failed plan attempt.

        Returns:
            The complete prompt string to send to the LLM.
        """
        primitives = self._get_primitive_methods()
        decompositions = self._get_decomposition_methods()

        # Build primitive documentation
        primitive_docs = [
            self._format_primitive_signature(name, method)
            for name, method in primitives
        ]

        # Build decomposition examples
        examples = []
        for _name, method, intent, expanded in decompositions:
            source = self._get_decomposition_source(method)
            if source:
                example = f"Intent: {intent}"
                if expanded:
                    example += f"\nApproach: {expanded}"
                example += f"\nPython:\n{source}"
                examples.append(example)

        # Build type definitions section for Pydantic models
        type_defs_section = self._format_type_definitions(primitives)

        # Build conversation history section if in multi-turn mode
        history_section = ""
        if self.config.multi_turn and self._history:
            history_section = f"""
## Conversation History

Previous turns in this conversation. You can reference variables from previous turns.

{self._format_history_for_prompt()}

"""

        # Build feedback section if retrying after a failed plan
        feedback_section = ""
        if feedback:
            feedback_section = f"""
## Previous Attempt Failed

Your previous plan was invalid. Please fix the following error and regenerate:

{feedback}

"""

        prompt = f"""You are {self.name}, an AI agent that generates Python code plans.

{self.description}

## Available Primitive Methods

You can ONLY call these methods:

```python
{chr(10).join(primitive_docs)}
```
{type_defs_section}
## Example Decompositions

Here are examples of how to compose primitives:

{chr(10).join(f"### Example {i + 1}{chr(10)}{ex}" for i, ex in enumerate(examples)) if examples else "No examples available."}
{history_section}{feedback_section}## Task

Generate Python code to accomplish this task: {task}

## Rules

1. Output ONLY Python assignment statements
2. Each statement must assign a result to a variable
3. You can ONLY call the primitive methods listed above
4. Do NOT use imports, loops, conditionals, or function definitions
5. Do NOT use any dangerous operations (exec, eval, open, etc.)
6. Call primitives directly (e.g. `add(a=1, b=2)`), do NOT use `self.`
7. The last assigned variable will be the final result

## Output

```python
"""
        return prompt

    def _extract_code_block(self, response_text: str) -> str:
        """Extract code from markdown code blocks in an LLM response.

        Args:
            response_text: The raw text response from the LLM.

        Returns:
            The extracted code string.
        """
        text = response_text.strip()
        code = text
        if "```python" in text:
            # Extract code between ```python and ```
            start = text.find("```python") + len("```python")
            end = text.find("```", start)
            if end > start:
                code = text[start:end]
        elif "```" in text:
            # Generic code block
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                code = text[start:end]

        # Normalize indentation - some models (especially smaller local ones)
        # may return code with unexpected leading indentation
        code = textwrap.dedent(code).strip()

        # Strip trailing markdown code block delimiters that weren't properly matched
        # This handles cases where the LLM adds a closing ``` without proper opening
        if code.endswith("```"):
            code = code[:-3].rstrip()

        return code

    def on_code_extracted(self, raw_response: str, extracted_code: str) -> str:
        """Hook called after code is extracted from the LLM response.

        Override this method in subclasses to observe and/or modify the
        extracted code before it is used for planning.

        Args:
            raw_response: The original raw text response from the LLM.
            extracted_code: The code extracted by _extract_code_block.

        Returns:
            The final code to use (can be modified from extracted_code).
        """
        return extracted_code

    # -------------------------------------------------------------------------
    # Plan Generation
    # -------------------------------------------------------------------------

    def plan(self, task: str, feedback: str | None = None) -> PlanResult:
        """Generate a plan (Python statements) for a task.

        Args:
            task: The task description to plan for.
            feedback: Optional error feedback from a previous failed plan attempt.

        Returns:
            PlanResult containing the generated plan and metrics.
        """
        plan_span: str | None = None
        if self._tracer:
            plan_span = self._tracer.start_span(
                EventType.PLAN_START, {"task": task, "feedback": feedback}
            )

        prompt = self.build_plan_prompt(task, feedback=feedback)

        if self._tracer and self._tracer.config.capture_llm_prompts:
            self._tracer.emit(EventType.PLAN_LLM_REQUEST, {"prompt": prompt})

        start_time = time.perf_counter()
        response = self._llm.generate(prompt)
        elapsed = time.perf_counter() - start_time

        # Extract code from response (handle markdown code blocks)
        raw_response = response.text
        extracted_code = self._extract_code_block(raw_response)
        plan_text = self.on_code_extracted(raw_response, extracted_code)

        # Build full LLM interaction details
        llm_interaction = LLMInteraction(
            prompt=prompt,
            response=raw_response,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            time_seconds=elapsed,
            provider=response.provider,
            model=response.model,
        )
        plan_generation = PlanGeneration(
            llm_interaction=llm_interaction,
            extracted_code=extracted_code,
        )

        if self._tracer and self._tracer.config.capture_llm_responses:
            self._tracer.emit(
                EventType.PLAN_LLM_RESPONSE,
                self._tracer.filter.llm_interaction(llm_interaction.model_dump()),
            )

        plan_result = PlanResult(
            plan=plan_text,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            time_seconds=elapsed,
            provider=response.provider,
            model=response.model,
            plan_generation=plan_generation,
        )

        if self._tracer and plan_span:
            self._tracer.end_span(
                plan_span,
                EventType.PLAN_COMPLETE,
                self._tracer.filter.plan_result(plan_result.model_dump()),
            )

        return plan_result

    # -------------------------------------------------------------------------
    # Plan Analysis and Validation
    # -------------------------------------------------------------------------

    def analyze_plan(self, plan: str) -> PlanAnalysis:
        """Analyze a plan to extract primitive calls.

        Args:
            plan: The Python statements to analyze.

        Returns:
            PlanAnalysis containing all primitive calls found.

        Raises:
            ValueError: If the plan has invalid syntax.
        """
        try:
            tree = ast.parse(plan)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}") from e

        primitive_names = self._get_primitive_names()
        read_only_map = self._get_primitive_read_only_map()
        calls: list[PrimitiveCall] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            method_name: str | None = None

            # Direct call: method_name(...)
            if isinstance(node.func, ast.Name) and node.func.id in primitive_names:
                method_name = node.func.id

            if method_name is not None:
                args: dict[str, Any] = {}
                for kw in node.keywords:
                    if kw.arg is not None:
                        try:
                            args[kw.arg] = ast.literal_eval(kw.value)
                        except (ValueError, TypeError):
                            args[kw.arg] = ast.unparse(kw.value)

                for i, arg in enumerate(node.args):
                    try:
                        args[f"arg{i}"] = ast.literal_eval(arg)
                    except (ValueError, TypeError):
                        args[f"arg{i}"] = ast.unparse(arg)

                calls.append(
                    PrimitiveCall(
                        method_name=method_name,
                        read_only=read_only_map.get(method_name, False),
                        args=args,
                    )
                )

        return PlanAnalysis(calls=calls)

    def validate_plan(self, plan: str) -> None:
        """Validate that a plan only uses allowed operations.

        Args:
            plan: The Python statements to validate.

        Raises:
            ValueError: If the plan contains disallowed operations.
        """
        try:
            tree = ast.parse(plan)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}") from e

        primitive_names = self._get_primitive_names()

        disallowed_statements: tuple[type, ...] = (
            ast.If,
            ast.For,
            ast.While,
            ast.Try,
            ast.With,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Import,
            ast.ImportFrom,
            ast.Global,
            ast.Nonlocal,
            ast.Raise,
            ast.Assert,
            ast.Delete,
        )
        if hasattr(ast, "Match"):
            disallowed_statements = (*disallowed_statements, ast.Match)

        for node in ast.walk(tree):
            if isinstance(node, disallowed_statements):
                node_type = type(node).__name__
                raise ValueError(f"{node_type} statements are not allowed in plans")

        # Every top-level statement must be an assignment
        for stmt in tree.body:
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                stmt_type = type(stmt).__name__
                raise ValueError(
                    f"Every statement must be an assignment. Found: {stmt_type}"
                )

        self._validate_ast_nodes(tree, primitive_names)

    def _validate_ast_nodes(self, tree: ast.Module, primitive_names: set[str]) -> None:
        """Validate AST nodes for dangerous ops, self. prefix, and private attrs.

        Shared between PlanExecute and DesignExecute validation.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise ValueError(
                    f"Accessing private/dunder attributes not allowed: {node.attr}"
                )

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.DANGEROUS_BUILTINS:
                        raise ValueError(f"Calling '{func_name}' is not allowed")
                    allowed_names = primitive_names | set(self.allowed_builtins.keys())
                    if func_name not in allowed_names:
                        raise ValueError(
                            f"Function '{func_name}' is not allowed. "
                            f"Only primitive methods and allowed builtins can be called."
                        )

                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                ):
                    raise ValueError(
                        "Do not use 'self.' prefix — call primitives directly "
                        f"(e.g. `{node.func.attr}(...)` instead of "
                        f"`self.{node.func.attr}(...)`)."
                    )

    # -------------------------------------------------------------------------
    # Plan Generation Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _plan_generation_from_result(plan_result: PlanResult) -> PlanGeneration:
        """Get a PlanGeneration from a PlanResult, creating a fallback if needed."""
        if plan_result.plan_generation is not None:
            return plan_result.plan_generation
        return PlanGeneration(
            llm_interaction=LLMInteraction(
                prompt="",
                response="",
                input_tokens=plan_result.usage.input_tokens,
                output_tokens=plan_result.usage.output_tokens,
                time_seconds=plan_result.time_seconds,
                provider=plan_result.provider,
                model=plan_result.model,
            ),
            extracted_code=plan_result.plan,
        )

    # -------------------------------------------------------------------------
    # Mutation Detection (for checkpoint-based execution)
    # -------------------------------------------------------------------------

    @staticmethod
    def _detect_mutation(
        stmt: ast.stmt, read_only_map: dict[str, bool]
    ) -> MutationDetection:
        """Detect whether an AST statement is a non-read-only primitive call."""
        result = MutationDetection()

        if isinstance(stmt, ast.Assign):
            if stmt.targets and isinstance(stmt.targets[0], ast.Name):
                result.variable_name = stmt.targets[0].id
            if isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Name):
                    result.method_name = call.func.id
                elif isinstance(call.func, ast.Attribute):
                    result.method_name = call.func.attr

                if result.method_name and not read_only_map.get(result.method_name, True):
                    result.is_mutation = True
                    for idx, arg in enumerate(call.args):
                        try:
                            result.args[f"arg{idx}"] = ast.literal_eval(arg)
                        except (ValueError, TypeError):
                            result.args[f"arg{idx}"] = ast.unparse(arg)
                    for kw in call.keywords:
                        if kw.arg:
                            try:
                                result.args[kw.arg] = ast.literal_eval(kw.value)
                            except (ValueError, TypeError):
                                result.args[kw.arg] = ast.unparse(kw.value)

        return result

    # -------------------------------------------------------------------------
    # Namespace Helpers
    # -------------------------------------------------------------------------

    def _build_reserved_names(self) -> set[str]:
        """Names that should be excluded from namespace snapshots."""
        return (
            set(self.allowed_builtins.keys())
            | {name for name, _ in self._get_primitive_methods()}
        )

    def _build_namespace(self) -> dict[str, Any]:
        """Build the execution namespace with primitives, builtins, and persisted vars.

        Note: adds *raw* primitive methods.  ``DesignExecute`` intentionally
        does NOT call this because it wraps primitives in traced wrappers.
        """
        namespace: dict[str, Any] = {}
        for name, method in self._get_primitive_methods():
            namespace[name] = method
        namespace.update(self.allowed_builtins)
        if self.config.multi_turn:
            namespace.update(self._persisted_namespace)
        return namespace

    def _persist_user_variables(
        self, namespace: dict[str, Any], reserved_names: set[str]
    ) -> None:
        """Persist user-defined variables for multi-turn mode."""
        if self.config.multi_turn:
            for key, value in namespace.items():
                if key not in reserved_names:
                    self._persisted_namespace[key] = value

    # -------------------------------------------------------------------------
    # Step-by-Step Execution
    # -------------------------------------------------------------------------

    def _snapshot_namespace(
        self, namespace: dict[str, Any], reserved_names: set[str]
    ) -> dict[str, Any]:
        """Create a JSON-serializable snapshot of user-defined variables in the namespace."""
        snapshot: dict[str, Any] = {}
        for key, value in namespace.items():
            if key in reserved_names:
                continue
            try:
                # Try to JSON serialize to ensure it's capturable
                json.dumps(value, default=str)
                snapshot[key] = value
            except (TypeError, ValueError):
                snapshot[key] = f"<{type(value).__name__}>"
        return snapshot

    def _resolve_arg_value(
        self, node: ast.expr, namespace: dict[str, Any]
    ) -> ArgumentValue:
        """Resolve an argument AST node to an ArgumentValue with expression and resolved value."""
        expression = ast.unparse(node)
        variable_reference: str | None = None
        resolved_value: Any = None

        # Check if it's a simple variable reference
        if isinstance(node, ast.Name):
            variable_reference = node.id
            resolved_value = namespace.get(node.id)
        else:
            # Try to evaluate the expression in the namespace
            try:
                resolved_value = ast.literal_eval(node)
            except (ValueError, TypeError):
                # Try to eval in the namespace for more complex expressions
                try:
                    resolved_value = eval(  # noqa: S307
                        expression, empty_builtins(), namespace
                    )
                except Exception:
                    resolved_value = None

        return ArgumentValue(
            expression=expression,
            resolved_value=resolved_value,
            variable_reference=variable_reference,
        )

    def _execute_statement(
        self,
        stmt: ast.stmt,
        namespace: dict[str, Any],
        step_number: int,
        read_only_map: dict[str, bool] | None = None,
        reserved_names: set[str] | None = None,
    ) -> ExecutionStep:
        """Execute a single statement and return the step result."""
        statement_str = ast.unparse(stmt)
        start_time = time.perf_counter()

        if reserved_names is None:
            reserved_names = set()

        # Capture namespace before execution
        namespace_before = self._snapshot_namespace(namespace, reserved_names)

        # Determine variable name and extract call info
        variable_name = ""
        primitive_called = None
        args: dict[str, ArgumentValue] = {}

        if isinstance(stmt, ast.Assign):
            if stmt.targets and isinstance(stmt.targets[0], ast.Name):
                variable_name = stmt.targets[0].id
            # Check if value is a Call
            if isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Name):
                    primitive_called = call.func.id
                elif isinstance(call.func, ast.Attribute):
                    primitive_called = call.func.attr

                # Extract positional args
                for i, arg in enumerate(call.args):
                    args[f"arg{i}"] = self._resolve_arg_value(arg, namespace)

                # Extract keyword args
                for kw in call.keywords:
                    if kw.arg is not None:
                        args[kw.arg] = self._resolve_arg_value(kw.value, namespace)

        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name):
                variable_name = stmt.target.id
            # Check if value is a Call (annotated assignments can also have calls)
            if stmt.value is not None and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Name):
                    primitive_called = call.func.id
                elif isinstance(call.func, ast.Attribute):
                    primitive_called = call.func.attr

                # Extract positional args
                for i, arg in enumerate(call.args):
                    args[f"arg{i}"] = self._resolve_arg_value(arg, namespace)

                # Extract keyword args
                for kw in call.keywords:
                    if kw.arg is not None:
                        args[kw.arg] = self._resolve_arg_value(kw.value, namespace)

        # Check mutation hook BEFORE execution for non-read-only primitives
        is_mutation = (
            primitive_called is not None
            and read_only_map is not None
            and not read_only_map.get(primitive_called, True)
        )

        if is_mutation and self.config.on_mutation is not None:
            assert primitive_called is not None  # Guaranteed by is_mutation check
            # Convert ArgumentValue dict to plain dict for hook context
            plain_args = {k: v.resolved_value for k, v in args.items()}
            hook_context = MutationHookContext(
                method_name=primitive_called,
                args=plain_args,
                result=None,  # Not yet executed
                step=None,  # Not yet created
            )
            rejection_reason = self.config.on_mutation(hook_context)

            if rejection_reason is not None:
                # Hook rejected the mutation - fail without executing
                elapsed = time.perf_counter() - start_time
                return ExecutionStep(
                    step_number=step_number,
                    statement=statement_str,
                    variable_name=variable_name,
                    primitive_called=primitive_called,
                    args=args,
                    namespace_before=namespace_before,
                    namespace_after=namespace_before,  # No change since rejected
                    result_type="",
                    result_value=None,
                    result_json=NULL_JSON,
                    time_seconds=elapsed,
                    success=False,
                    error=f"{MUTATION_REJECTED_PREFIX}: {rejection_reason}",
                )

        try:
            # Execute the single statement
            exec(  # noqa: S102
                compile(ast.Module(body=[stmt], type_ignores=[]), PLAN_COMPILE_SOURCE, "exec"),
                empty_builtins(),
                namespace,
            )
            elapsed = time.perf_counter() - start_time

            # Capture namespace after execution
            namespace_after = self._snapshot_namespace(namespace, reserved_names)

            # Get the result value
            result_value = namespace.get(variable_name) if variable_name else None

            result_json = (
                NULL_JSON
                if self.config.skip_result_serialization
                else json.dumps(result_value, default=str)
            )

            step = ExecutionStep(
                step_number=step_number,
                statement=statement_str,
                variable_name=variable_name,
                primitive_called=primitive_called,
                args=args,
                namespace_before=namespace_before,
                namespace_after=namespace_after,
                result_type=type(result_value).__name__
                if result_value is not None
                else NONE_TYPE_NAME,
                result_value=result_value,
                result_json=result_json,
                time_seconds=elapsed,
                success=True,
            )

            return step

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            # Capture namespace after failed execution (may have partial changes)
            namespace_after = self._snapshot_namespace(namespace, reserved_names)
            return ExecutionStep(
                step_number=step_number,
                statement=statement_str,
                variable_name=variable_name,
                primitive_called=primitive_called,
                args=args,
                namespace_before=namespace_before,
                namespace_after=namespace_after,
                result_type="",
                result_value=None,
                result_json=NULL_JSON,
                time_seconds=elapsed,
                success=False,
                error=str(e),
            )

    def execute(self, plan: str) -> ExecutionResult:
        """Execute a plan step-by-step with full tracing.

        Args:
            plan: The Python statements to execute.

        Returns:
            ExecutionResult containing the final value and execution trace.

        Raises:
            ValueError: If the plan contains disallowed operations.
        """
        self.validate_plan(plan)

        exec_span: str | None = None
        if self._tracer:
            exec_span = self._tracer.start_span(
                EventType.EXECUTION_START,
                {"plan": plan} if self._tracer.config.capture_plan_source else {},
            )

        tree = ast.parse(plan)
        steps: list[ExecutionStep] = []
        total_start = time.perf_counter()

        namespace = self._build_namespace()
        read_only_map = self._get_primitive_read_only_map()
        reserved_names = self._build_reserved_names()

        # Execute statement by statement
        for i, stmt in enumerate(tree.body, 1):
            step = self._execute_statement(
                stmt, namespace, i, read_only_map, reserved_names
            )
            steps.append(step)

            if self._tracer and self._tracer.config.capture_execution_steps:
                self._tracer.emit(
                    EventType.EXECUTION_STEP,
                    self._tracer.filter.execution_step(step.model_dump()),
                )

            # Stop on first error
            if not step.success:
                break

        self._persist_user_variables(namespace, reserved_names)

        total_elapsed = time.perf_counter() - total_start

        trace = ExecutionTrace(
            steps=steps,
            total_time_seconds=total_elapsed,
        )

        # Determine final result
        if steps and steps[-1].success:
            last = steps[-1]
            result = ExecutionResult(
                value_type=last.result_type,
                value_name=last.variable_name,
                value_json=last.result_json,
                trace=trace,
            )
        else:
            result = ExecutionResult(
                value_type=NONE_TYPE_NAME,
                value_name="",
                value_json=NULL_JSON,
                trace=trace,
            )

        if self._tracer and exec_span:
            self._tracer.end_span(
                exec_span,
                EventType.EXECUTION_COMPLETE,
                ExecutionSummary(
                    value_type=result.value_type,
                    value_name=result.value_name,
                    step_count=trace.step_count,
                    all_succeeded=trace.all_succeeded,
                    total_time_seconds=total_elapsed,
                ).model_dump(),
            )

        return result

    # -------------------------------------------------------------------------
    # Checkpoint-based Execution (Distributed/Interruptible)
    # -------------------------------------------------------------------------

    def execute_stepwise(
        self,
        task: str,
        plan_context: PlanContext | None = None,
        serializer: SerializerRegistry | None = None,
        checkpoint_id: str | None = None,
    ) -> Iterator[ExecutionCheckpoint]:
        """Execute a plan step-by-step, yielding checkpoints for persistence.

        This method enables distributed execution by yielding a checkpoint after
        each step. The checkpoint can be persisted to a database and resumed on
        any worker.

        When `require_mutation_approval` is True (default), execution pauses
        before mutations and yields a checkpoint with status `awaiting_approval`.
        Call `resume_from_checkpoint()` with `approve_mutation=True` to continue.

        Args:
            task: The task description to accomplish.
            plan_context: Optional pre-computed plan context (from a previous
                planning phase). If None, will generate a new plan.
            serializer: Custom serializer registry. Defaults to the global registry.
            checkpoint_id: Optional ID for the checkpoint. Auto-generated if None.

        Yields:
            ExecutionCheckpoint after each step, allowing for persistence and
            distributed resume.

        Example:
            # Simple usage - collect all checkpoints
            for checkpoint in agent.execute_stepwise("do something"):
                store.save(checkpoint)
                if checkpoint.status == CheckpointStatus.AWAITING_APPROVAL:
                    # Handle approval externally
                    break

            # Resume after approval
            checkpoint = store.load(checkpoint_id)
            for checkpoint in agent.resume_from_checkpoint(checkpoint, approve_mutation=True):
                store.save(checkpoint)
        """
        serializer = serializer or default_serializer_registry

        # Generate plan if not provided
        if plan_context is None:
            plan_result = self.plan(task)
            plan_context = PlanContext(
                plan_attempts=[
                    PlanAttempt(
                        attempt_number=1,
                        plan_generation=self._plan_generation_from_result(plan_result),
                        success=True,
                    )
                ],
                final_plan=plan_result.plan,
                generation_time_seconds=plan_result.time_seconds,
            )

        plan = plan_context.final_plan

        # Validate plan
        try:
            self.validate_plan(plan)
        except ValueError as e:
            checkpoint = ExecutionCheckpoint(
                task=task,
                plan=plan,
                plan_context=plan_context,
                current_step=0,
                total_steps=0,
                status=CheckpointStatus.FAILED,
                error=str(e),
                worker_id=self.config.worker_id,
            )
            if checkpoint_id is not None:
                checkpoint.checkpoint_id = checkpoint_id
            yield checkpoint
            return

        tree = ast.parse(plan)
        total_steps = len(tree.body)

        namespace = self._build_namespace()
        read_only_map = self._get_primitive_read_only_map()
        reserved_names = self._build_reserved_names()

        steps: list[ExecutionStep] = []

        # Create initial checkpoint
        checkpoint = ExecutionCheckpoint(
            task=task,
            plan=plan,
            plan_context=plan_context,
            current_step=0,
            total_steps=total_steps,
            status=CheckpointStatus.RUNNING,
            namespace_snapshot=serializer.serialize_namespace(
                namespace, reserved_names
            ),
            completed_steps=[],
            worker_id=self.config.worker_id,
        )
        if checkpoint_id is not None:
            checkpoint.checkpoint_id = checkpoint_id

        # Execute statement by statement
        for i, stmt in enumerate(tree.body):
            step_number = i + 1
            statement_str = ast.unparse(stmt)

            mutation = self._detect_mutation(stmt, read_only_map)

            # If mutation requires approval, pause and yield
            if mutation.is_mutation and self.config.require_mutation_approval:
                checkpoint.current_step = i
                checkpoint.status = CheckpointStatus.AWAITING_APPROVAL
                checkpoint.pending_mutation = PendingMutation(
                    method_name=mutation.method_name or "",
                    args=mutation.args,
                    statement=statement_str,
                    step_number=step_number,
                    variable_name=mutation.variable_name,
                )
                checkpoint.namespace_snapshot = serializer.serialize_namespace(
                    namespace, reserved_names
                )
                checkpoint.completed_steps = steps.copy()
                checkpoint.touch(self.config.worker_id)
                yield checkpoint
                return  # Caller must resume with approve_mutation=True

            # Execute the step
            step = self._execute_statement(
                stmt, namespace, step_number, read_only_map, reserved_names
            )
            steps.append(step)

            # Update checkpoint
            checkpoint.current_step = step_number
            checkpoint.completed_steps = steps.copy()
            checkpoint.namespace_snapshot = serializer.serialize_namespace(
                namespace, reserved_names
            )

            if not step.success:
                checkpoint.status = CheckpointStatus.FAILED
                checkpoint.error = step.error
                checkpoint.touch(self.config.worker_id)
                yield checkpoint
                return

            # Yield progress checkpoint
            checkpoint.status = CheckpointStatus.RUNNING
            checkpoint.touch(self.config.worker_id)
            yield checkpoint

        # Execution completed successfully
        checkpoint.status = CheckpointStatus.COMPLETED
        if steps:
            last_step = steps[-1]
            checkpoint.result_variable = last_step.variable_name
            result_value = namespace.get(last_step.variable_name)
            checkpoint.result_value = serializer.serialize(result_value)

        self._persist_user_variables(namespace, reserved_names)

        checkpoint.touch(self.config.worker_id)
        yield checkpoint

    def resume_from_checkpoint(
        self,
        checkpoint: ExecutionCheckpoint,
        approve_mutation: bool = False,
        serializer: SerializerRegistry | None = None,
    ) -> Iterator[ExecutionCheckpoint]:
        """Resume execution from a persisted checkpoint.

        This method reconstructs execution state from a checkpoint and continues
        execution from where it left off.

        Args:
            checkpoint: The checkpoint to resume from.
            approve_mutation: If True and checkpoint is awaiting approval,
                execute the pending mutation and continue.
            serializer: Custom serializer registry. Must match the one used
                when creating the checkpoint.

        Yields:
            ExecutionCheckpoint after each step.

        Raises:
            ValueError: If checkpoint is in a terminal state or approval is
                required but not provided.

        Example:
            # Load and resume
            checkpoint = store.load(checkpoint_id)
            if checkpoint.status == CheckpointStatus.AWAITING_APPROVAL:
                # Get user approval somehow
                if user_approved:
                    for cp in agent.resume_from_checkpoint(checkpoint, approve_mutation=True):
                        store.save(cp)
        """
        serializer = serializer or default_serializer_registry

        if checkpoint.is_terminal:
            raise ValueError(
                f"Cannot resume from terminal checkpoint (status: {checkpoint.status})"
            )

        if (
            checkpoint.status == CheckpointStatus.AWAITING_APPROVAL
            and not approve_mutation
        ):
            raise ValueError(
                "Checkpoint is awaiting mutation approval. "
                "Set approve_mutation=True to continue."
            )

        # Parse the plan
        tree = ast.parse(checkpoint.plan)
        total_steps = len(tree.body)

        namespace = self._build_namespace()

        # Restore serialized variables (skip undeserializable values)
        import contextlib

        for var_name, serialized_val in checkpoint.namespace_snapshot.items():
            with contextlib.suppress(ValueError):
                namespace[var_name] = serializer.deserialize(serialized_val)

        read_only_map = self._get_primitive_read_only_map()
        reserved_names = self._build_reserved_names()

        # Copy completed steps
        steps = list(checkpoint.completed_steps)

        # Determine starting point
        start_step = checkpoint.current_step

        # If resuming from awaiting_approval, execute the pending mutation first
        if (
            checkpoint.status == CheckpointStatus.AWAITING_APPROVAL
            and approve_mutation
            and checkpoint.pending_mutation
        ):
            stmt = tree.body[start_step]
            step_number = start_step + 1

            step = self._execute_statement(
                stmt, namespace, step_number, read_only_map, reserved_names
            )
            steps.append(step)

            checkpoint.completed_steps = steps.copy()
            checkpoint.pending_mutation = None

            if not step.success:
                checkpoint.status = CheckpointStatus.FAILED
                checkpoint.error = step.error
                checkpoint.current_step = step_number
                checkpoint.namespace_snapshot = serializer.serialize_namespace(
                    namespace, reserved_names
                )
                checkpoint.touch(self.config.worker_id)
                yield checkpoint
                return

            checkpoint.current_step = step_number
            checkpoint.namespace_snapshot = serializer.serialize_namespace(
                namespace, reserved_names
            )
            checkpoint.status = CheckpointStatus.RUNNING
            checkpoint.touch(self.config.worker_id)
            yield checkpoint

            start_step += 1

        # Continue with remaining steps
        for i in range(start_step, total_steps):
            stmt = tree.body[i]
            step_number = i + 1
            statement_str = ast.unparse(stmt)

            mutation = self._detect_mutation(stmt, read_only_map)

            # Pause for mutation approval if required
            if mutation.is_mutation and self.config.require_mutation_approval:
                checkpoint.current_step = i
                checkpoint.status = CheckpointStatus.AWAITING_APPROVAL
                checkpoint.pending_mutation = PendingMutation(
                    method_name=mutation.method_name or "",
                    args=mutation.args,
                    statement=statement_str,
                    step_number=step_number,
                    variable_name=mutation.variable_name,
                )
                checkpoint.namespace_snapshot = serializer.serialize_namespace(
                    namespace, reserved_names
                )
                checkpoint.completed_steps = steps.copy()
                checkpoint.touch(self.config.worker_id)
                yield checkpoint
                return

            # Execute step
            step = self._execute_statement(
                stmt, namespace, step_number, read_only_map, reserved_names
            )
            steps.append(step)

            checkpoint.current_step = step_number
            checkpoint.completed_steps = steps.copy()
            checkpoint.namespace_snapshot = serializer.serialize_namespace(
                namespace, reserved_names
            )

            if not step.success:
                checkpoint.status = CheckpointStatus.FAILED
                checkpoint.error = step.error
                checkpoint.touch(self.config.worker_id)
                yield checkpoint
                return

            checkpoint.status = CheckpointStatus.RUNNING
            checkpoint.touch(self.config.worker_id)
            yield checkpoint

        # Completed
        checkpoint.status = CheckpointStatus.COMPLETED
        if steps:
            last_step = steps[-1]
            checkpoint.result_variable = last_step.variable_name
            result_value = namespace.get(last_step.variable_name)
            checkpoint.result_value = serializer.serialize(result_value)

        self._persist_user_variables(namespace, reserved_names)

        checkpoint.touch(self.config.worker_id)
        yield checkpoint

    def run_with_checkpoints(
        self,
        task: str,
        store: CheckpointStore,
        serializer: SerializerRegistry | None = None,
        auto_approve: bool = False,
    ) -> ExecutionCheckpoint:
        """Run execution with automatic checkpoint persistence.

        Convenience method that handles checkpoint saving automatically.

        Args:
            task: The task to execute.
            store: Checkpoint store for persistence.
            serializer: Custom serializer registry.
            auto_approve: If True, automatically approve all mutations.

        Returns:
            The final checkpoint (completed or failed).
        """
        checkpoint: ExecutionCheckpoint | None = None

        for checkpoint in self.execute_stepwise(task, serializer=serializer):
            store.save(checkpoint)

            if checkpoint.status == CheckpointStatus.AWAITING_APPROVAL:
                if auto_approve:
                    # Resume with approval - consume all checkpoints from resume
                    current_cp = checkpoint
                    while current_cp.status == CheckpointStatus.AWAITING_APPROVAL:
                        for next_cp in self.resume_from_checkpoint(
                            current_cp, approve_mutation=True, serializer=serializer
                        ):
                            store.save(next_cp)
                            current_cp = next_cp
                    checkpoint = current_cp
                else:
                    # Return checkpoint for external approval
                    return checkpoint

        if checkpoint is None:
            raise RuntimeError("No checkpoint was generated")

        return checkpoint

    # -------------------------------------------------------------------------
    # Main Orchestration
    # -------------------------------------------------------------------------

    def _emit_run_end(
        self, run_span: str, result: OrchestrationResult
    ) -> None:
        """Emit the RUN_COMPLETE or RUN_ERROR event for a run span."""
        if not self._tracer:
            return
        if result.success:
            self._tracer.end_span(
                run_span,
                EventType.RUN_COMPLETE,
                RunCompleteSummary(
                    result_type=type(result.result).__name__
                    if result.result is not None
                    else NONE_TYPE_NAME,
                    steps_executed=result.metrics.steps_executed
                    if result.metrics
                    else 0,
                ).model_dump(),
            )
        else:
            self._tracer.end_span(
                run_span,
                EventType.RUN_ERROR,
                RunErrorSummary(
                    error=result.error or "Unknown error",
                ).model_dump(),
            )

    def run(self, task: str) -> OrchestrationResult:
        """Run the complete plan-and-execute cycle.

        Args:
            task: The task description to accomplish.

        Returns:
            OrchestrationResult containing the outcome and metrics.
        """
        run_span: str | None = None
        if self._tracer:
            self._tracer.new_trace()
            run_span = self._tracer.start_span(
                EventType.RUN_START,
                {"task": task, "config": self.config.model_dump(exclude={"observability"})},
            )

        try:
            return self._run_inner(task, run_span)
        finally:
            if self._tracer:
                self._tracer.flush()

    def _run_inner(
        self, task: str, run_span: str | None
    ) -> OrchestrationResult:
        """Core run logic, separated so ``run()`` can wrap with try/finally."""
        plan_result = None
        feedback: str | None = None
        max_attempts = 1 + self.config.max_plan_retries
        plan_attempts: list[PlanAttempt] = []

        for attempt in range(max_attempts):
            try:
                # Generate plan (with feedback if retrying)
                plan_result = self.plan(task, feedback=feedback)

                # Record the plan attempt
                plan_attempt = PlanAttempt(
                    attempt_number=attempt + 1,
                    plan_generation=self._plan_generation_from_result(plan_result),
                    feedback=feedback,
                    validation_error=None,
                    success=True,
                )

                # Validate plan before execution (to catch validation errors for retry)
                try:
                    self.validate_plan(plan_result.plan)
                except ValueError as validation_error:
                    plan_attempt.validation_error = str(validation_error)
                    plan_attempt.success = False
                    plan_attempts.append(plan_attempt)
                    if self._tracer:
                        self._tracer.emit(
                            EventType.PLAN_VALIDATION_ERROR,
                            {"error": str(validation_error), "attempt": attempt + 1},
                        )
                    raise

                plan_attempts.append(plan_attempt)

                # Execute plan
                exec_start = time.perf_counter()
                exec_result = self.execute(plan_result.plan)
                exec_time = time.perf_counter() - exec_start

                metrics = ExecutionMetrics(
                    plan_tokens=plan_result.usage,
                    plan_time_seconds=plan_result.time_seconds,
                    execute_time_seconds=exec_time,
                    steps_executed=exec_result.trace.step_count,
                    provider=plan_result.provider,
                    model=plan_result.model,
                )

                # Check if execution succeeded
                if exec_result.trace.all_succeeded:
                    result_value = exec_result.get_value()

                    # Track history in multi-turn mode
                    if self.config.multi_turn:
                        self._history.append(
                            ConversationTurn(
                                task=task,
                                plan=plan_result.plan,
                                result=result_value,
                                success=True,
                            )
                        )

                    orch_result = OrchestrationResult(
                        success=True,
                        result=result_value,
                        metrics=metrics,
                        plan=plan_result.plan,
                        trace=exec_result.trace,
                        plan_attempts=plan_attempts,
                        task=task,
                    )
                    if run_span:
                        self._emit_run_end(run_span, orch_result)
                    return orch_result
                else:
                    # Get error from failed step
                    failed = (
                        exec_result.trace.failed_steps[0]
                        if exec_result.trace.failed_steps
                        else None
                    )
                    error_msg = failed.error if failed else "Unknown execution error"

                    # Track history in multi-turn mode (even for failures)
                    if self.config.multi_turn:
                        self._history.append(
                            ConversationTurn(
                                task=task,
                                plan=plan_result.plan,
                                success=False,
                                error=error_msg,
                            )
                        )

                    orch_result = OrchestrationResult(
                        success=False,
                        error=error_msg,
                        metrics=metrics,
                        plan=plan_result.plan,
                        trace=exec_result.trace,
                        plan_attempts=plan_attempts,
                        task=task,
                    )
                    if run_span:
                        self._emit_run_end(run_span, orch_result)
                    return orch_result

            except ValueError as e:
                # Validation error - retry if attempts remaining
                error_msg = str(e)
                if attempt < max_attempts - 1:
                    feedback = error_msg
                    continue

                # No more retries - return failure
                if self.config.multi_turn and plan_result is not None:
                    self._history.append(
                        ConversationTurn(
                            task=task,
                            plan=plan_result.plan,
                            success=False,
                            error=error_msg,
                        )
                    )

                orch_result = OrchestrationResult(
                    success=False,
                    error=error_msg,
                    plan=plan_result.plan if plan_result else None,
                    plan_attempts=plan_attempts,
                    task=task,
                )
                if run_span:
                    self._emit_run_end(run_span, orch_result)
                return orch_result

            except Exception as e:
                error_msg = str(e)

                # Track history in multi-turn mode (even for validation/other failures)
                if self.config.multi_turn and plan_result is not None:
                    self._history.append(
                        ConversationTurn(
                            task=task,
                            plan=plan_result.plan,
                            success=False,
                            error=error_msg,
                        )
                    )

                orch_result = OrchestrationResult(
                    success=False,
                    error=error_msg,
                    plan=plan_result.plan if plan_result else None,
                    plan_attempts=plan_attempts,
                    task=task,
                )
                if run_span:
                    self._emit_run_end(run_span, orch_result)
                return orch_result

        # Should not reach here, but satisfy type checker
        return OrchestrationResult(
            success=False,
            error="Unexpected: exhausted all plan retry attempts",
            plan=plan_result.plan if plan_result else None,
            plan_attempts=plan_attempts,
            task=task,
        )
