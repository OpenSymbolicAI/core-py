"""DesignExecute blueprint: extends PlanExecute with control flow support."""

from __future__ import annotations

import ast
import inspect
import json
import time
from collections.abc import Iterator
from typing import Any

from opensymbolicai.blueprints.plan_execute import PlanExecute
from opensymbolicai.checkpoint import (
    ExecutionCheckpoint,
    PlanContext,
    SerializerRegistry,
)
from opensymbolicai.llm import LLM, LLMConfig
from opensymbolicai.models import (
    MUTATION_REJECTED_PREFIX,
    NONE_TYPE_NAME,
    NULL_JSON,
    PLAN_COMPILE_SOURCE,
    PROMPT_CONTEXT_BEGIN,
    PROMPT_CONTEXT_END,
    PROMPT_DEFINITIONS_BEGIN,
    PROMPT_DEFINITIONS_END,
    PROMPT_INSTRUCTIONS_BEGIN,
    PROMPT_INSTRUCTIONS_END,
    ArgumentValue,
    DesignExecuteConfig,
    ExecutionResult,
    ExecutionStep,
    ExecutionTrace,
    MutationHookContext,
    empty_builtins,
)
from opensymbolicai.observability.events import EventType, ExecutionSummary

# Sentinel message used by loop guard injection
_LOOP_LIMIT_EXCEEDED_MSG = "__opensymbolicai_loop_limit_exceeded__"


class _LoopGuardTransformer(ast.NodeTransformer):
    """AST transformer that injects iteration counters into loops.

    Transforms::

        for x in items:
            body

    Into::

        __loop_guard_1__ = 0
        for x in items:
            __loop_guard_1__ += 1
            if __loop_guard_1__ > __loop_limit__:
                __loop_guard_raise__()
            body

    Similarly for while loops.
    """

    def __init__(self) -> None:
        self._counter_id = 0

    def _next_counter_name(self) -> str:
        self._counter_id += 1
        return f"__loop_guard_{self._counter_id}__"

    def _make_guard_stmts(self, counter_name: str) -> list[ast.stmt]:
        """Create the increment + check statements to prepend to loop body."""
        # counter_name += 1
        increment = ast.AugAssign(
            target=ast.Name(id=counter_name, ctx=ast.Store()),
            op=ast.Add(),
            value=ast.Constant(value=1),
        )

        # if counter_name > __loop_limit__:
        #     __loop_guard_raise__()
        check = ast.If(
            test=ast.Compare(
                left=ast.Name(id=counter_name, ctx=ast.Load()),
                ops=[ast.Gt()],
                comparators=[ast.Name(id="__loop_limit__", ctx=ast.Load())],
            ),
            body=[
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id="__loop_guard_raise__", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    )
                )
            ],
            orelse=[],
        )

        return [increment, check]

    def _make_init_stmt(self, counter_name: str) -> ast.stmt:
        """Create: counter_name = 0"""
        return ast.Assign(
            targets=[ast.Name(id=counter_name, ctx=ast.Store())],
            value=ast.Constant(value=0),
        )

    def visit_For(self, node: ast.For) -> list[ast.stmt]:
        # Visit children first (handles nested loops)
        self.generic_visit(node)

        counter_name = self._next_counter_name()
        guard_stmts = self._make_guard_stmts(counter_name)

        # Prepend guard to loop body
        node.body = guard_stmts + node.body

        # Return init statement + modified loop
        return [self._make_init_stmt(counter_name), node]

    def visit_While(self, node: ast.While) -> list[ast.stmt]:
        self.generic_visit(node)

        counter_name = self._next_counter_name()
        guard_stmts = self._make_guard_stmts(counter_name)

        node.body = guard_stmts + node.body

        return [self._make_init_stmt(counter_name), node]


class DesignExecute(PlanExecute):
    """Agent that generates and executes Python plans with control flow.

    DesignExecute extends PlanExecute to allow for, while, if/elif/else,
    try/except, and raise in LLM-generated plans while maintaining safety
    (loop limits, blocked dangerous ops) and full traceability (every
    primitive call is recorded).

    Plans may include:
    - Assignment statements: ``result = add(1, 2)``
    - For loops: ``for item in items:``
    - While loops: ``while condition:``
    - Conditionals: ``if/elif/else``
    - Try/except blocks: ``try: ... except ValueError: ...``
    - Raise statements: ``raise ValueError("message")``
    - break/continue (configurable)

    Plans may NOT include:
    - Function/class definitions
    - Import statements
    - With blocks
    - exec/eval/open or other dangerous builtins
    - Private/dunder attribute access

    Tracing works by wrapping all primitives with instrumentation wrappers
    that record each call (name, args, result, timing) into a flat trace.

    Example::

        class DataProcessor(DesignExecute):
            @primitive(read_only=True)
            def process(self, item: str) -> str:
                return item.upper()

            @primitive(read_only=True)
            def summarize(self, items: list[str]) -> str:
                return ", ".join(items)
    """

    # Exception types available in plans for raise/try-except
    ALLOWED_EXCEPTIONS: dict[str, type] = {
        "ValueError": ValueError,
        "RuntimeError": RuntimeError,
        "TypeError": TypeError,
        "KeyError": KeyError,
    }

    def __init__(
        self,
        llm: LLM | LLMConfig,
        name: str = "",
        description: str = "",
        config: DesignExecuteConfig | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            name=name,
            description=description,
            config=config or DesignExecuteConfig(),
        )
        # Make exception types available for validation and execution
        self.allowed_builtins.update(self.ALLOWED_EXCEPTIONS)

    @property
    def design_config(self) -> DesignExecuteConfig:
        """Get the DesignExecute-specific config."""
        assert isinstance(self.config, DesignExecuteConfig)
        return self.config

    @property
    def blueprint_type(self) -> str:
        """The blueprint type: 'PlanExecute', 'DesignExecute', or 'GoalSeeking'."""
        return "DesignExecute"

    # -------------------------------------------------------------------------
    # Prompt Building (Override)
    # -------------------------------------------------------------------------

    def build_plan_prompt(self, task: str, feedback: str | None = None) -> str:
        """Build prompt that allows control flow.

        Fully overrides the parent prompt to replace the rules section
        with DesignExecute-specific rules permitting loops and conditionals.
        """
        primitives = self._get_prompt_primitives()
        decompositions = self._get_prompt_decompositions()

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

        max_iters = self.design_config.max_loop_iterations

        prompt = f"""You are {self.name}, an AI agent that generates Python code plans.

{self.description}

{PROMPT_DEFINITIONS_BEGIN}

## Available Primitive Methods

You can ONLY call these methods:

```python
{chr(10).join(primitive_docs)}
```
{type_defs_section}
## Example Decompositions

Here are examples of how to compose primitives:

{chr(10).join(f"### Example {i + 1}{chr(10)}{ex}" for i, ex in enumerate(examples)) if examples else "No examples available."}

{PROMPT_DEFINITIONS_END}

{PROMPT_CONTEXT_BEGIN}
{history_section}{feedback_section}## Task

Generate Python code to accomplish this task: {task}

{PROMPT_CONTEXT_END}

{PROMPT_INSTRUCTIONS_BEGIN}

## Rules

1. You can use assignment statements, for loops, while loops, if/elif/else, try/except, and raise
2. You can ONLY call the primitive methods listed above
3. Do NOT use imports, function definitions, class definitions, or with statements
4. Do NOT use any dangerous operations (exec, eval, open, etc.)
5. While loops MUST have a clear termination condition (max {max_iters} iterations)
6. Assign the final result to a variable named `result`
7. Call primitives directly (e.g. `lookup_price(item=item)`), do NOT use `self.`
8. Use loops when you need to process collections or repeat operations
9. Use conditionals when the task requires branching logic
10. Use raise ValueError("message") to signal errors (e.g. missing required input)
11. Available exception types: ValueError, RuntimeError, TypeError, KeyError

## Output

```python

{PROMPT_INSTRUCTIONS_END}
"""
        return prompt

    # -------------------------------------------------------------------------
    # Plan Validation (Override)
    # -------------------------------------------------------------------------

    def validate_plan(self, plan: str) -> None:
        """Validate plan allowing control flow but blocking dangerous ops.

        Compared to PlanExecute, this allows For, While, If, AugAssign,
        and Expr (bare function calls) at the top level.
        """
        try:
            tree = ast.parse(plan)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}") from e

        primitive_names = self._get_primitive_names()

        # These are still disallowed even in DesignExecute
        disallowed_statements: tuple[type, ...] = (
            ast.With,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Import,
            ast.ImportFrom,
            ast.Global,
            ast.Nonlocal,
            ast.Assert,
            ast.Delete,
        )
        if hasattr(ast, "Match"):
            disallowed_statements = (*disallowed_statements, ast.Match)

        # Optionally disallow break/continue
        if not self.design_config.allow_break_continue:
            disallowed_statements = (*disallowed_statements, ast.Break, ast.Continue)

        for node in ast.walk(tree):
            if isinstance(node, disallowed_statements):
                node_type = type(node).__name__
                raise ValueError(f"{node_type} statements are not allowed in plans")

        # Validate that top-level statements are allowed types
        allowed_top_level = (
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.For,
            ast.While,
            ast.If,
            ast.Expr,
            ast.Try,
            ast.Raise,
        )
        for stmt in tree.body:
            if not isinstance(stmt, allowed_top_level):
                stmt_type = type(stmt).__name__
                raise ValueError(
                    f"Statement type '{stmt_type}' is not allowed at top level. "
                    f"Allowed: assignments, for, while, if, try/except, raise, expressions."
                )

        self._validate_ast_nodes(tree, primitive_names)

    # -------------------------------------------------------------------------
    # AST Transformation: Loop Guard Injection
    # -------------------------------------------------------------------------

    def _inject_loop_guards(self, tree: ast.Module) -> ast.Module:
        """Transform AST to inject loop iteration counters.

        For every ``for`` and ``while`` loop, injects a counter variable
        and a limit check that calls ``__loop_guard_raise__()`` if exceeded.
        """
        transformer = _LoopGuardTransformer()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        return new_tree  # type: ignore[no-any-return]

    # -------------------------------------------------------------------------
    # Primitive Instrumentation
    # -------------------------------------------------------------------------

    def _create_traced_primitive(
        self,
        name: str,
        method: Any,
        trace_list: list[ExecutionStep],
        step_counter: list[int],
        read_only_map: dict[str, bool],
        namespace: dict[str, Any],
        reserved_names: set[str],
    ) -> Any:
        """Create a wrapper around a primitive that records each call."""
        sig = inspect.signature(method)
        param_names = [p for p in sig.parameters if p != "self"]

        def traced_wrapper(*args: Any, **kwargs: Any) -> Any:
            step_number = step_counter[0]
            step_counter[0] += 1

            # Check total call limit
            if step_number > self.design_config.max_total_primitive_calls:
                raise RuntimeError(
                    f"Exceeded maximum total primitive calls "
                    f"({self.design_config.max_total_primitive_calls}). "
                    f"Plan may contain too many iterations."
                )

            # Build args dict for tracing
            traced_args: dict[str, ArgumentValue] = {}
            for i, arg_val in enumerate(args):
                pname = param_names[i] if i < len(param_names) else f"arg{i}"
                traced_args[pname] = ArgumentValue(
                    expression=repr(arg_val),
                    resolved_value=arg_val,
                )
            for kw_name, kw_val in kwargs.items():
                traced_args[kw_name] = ArgumentValue(
                    expression=repr(kw_val),
                    resolved_value=kw_val,
                )

            # Namespace snapshot before
            namespace_before = self._snapshot_namespace(namespace, reserved_names)

            # Check mutation hook
            is_mutation = not read_only_map.get(name, True)
            if is_mutation and self.config.on_mutation is not None:
                plain_args = {k: v.resolved_value for k, v in traced_args.items()}
                hook_context = MutationHookContext(
                    method_name=name,
                    args=plain_args,
                    result=None,
                    step=None,
                )
                rejection_reason = self.config.on_mutation(hook_context)
                if rejection_reason is not None:
                    step = ExecutionStep(
                        step_number=step_number,
                        statement=f"{name}(...)",
                        primitive_called=name,
                        args=traced_args,
                        namespace_before=namespace_before,
                        namespace_after=namespace_before,
                        time_seconds=0.0,
                        success=False,
                        error=f"{MUTATION_REJECTED_PREFIX}: {rejection_reason}",
                    )
                    trace_list.append(step)
                    raise RuntimeError(f"{MUTATION_REJECTED_PREFIX}: {rejection_reason}")

            # Execute the actual primitive
            start_time = time.perf_counter()
            try:
                result = method(*args, **kwargs)
                elapsed = time.perf_counter() - start_time

                namespace_after = self._snapshot_namespace(namespace, reserved_names)

                result_json = (
                    NULL_JSON
                    if self.config.skip_result_serialization
                    else json.dumps(result, default=str)
                )

                arg_strs = [
                    f"{k}={v.expression}" for k, v in traced_args.items()
                ]
                step = ExecutionStep(
                    step_number=step_number,
                    statement=f"{name}({', '.join(arg_strs)})",
                    primitive_called=name,
                    args=traced_args,
                    namespace_before=namespace_before,
                    namespace_after=namespace_after,
                    result_type=(
                        type(result).__name__ if result is not None else NONE_TYPE_NAME
                    ),
                    result_value=result,
                    result_json=result_json,
                    time_seconds=elapsed,
                    success=True,
                )
                trace_list.append(step)

                if self._tracer and self._tracer.config.capture_execution_steps:
                    self._tracer.emit(
                        EventType.EXECUTION_STEP,
                        self._tracer.filter.execution_step(step.model_dump()),
                    )

                return result

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                namespace_after = self._snapshot_namespace(namespace, reserved_names)
                step = ExecutionStep(
                    step_number=step_number,
                    statement=f"{name}(...)",
                    primitive_called=name,
                    args=traced_args,
                    namespace_before=namespace_before,
                    namespace_after=namespace_after,
                    time_seconds=elapsed,
                    success=False,
                    error=str(e),
                )
                trace_list.append(step)
                raise

        return traced_wrapper

    # -------------------------------------------------------------------------
    # Execution (Override)
    # -------------------------------------------------------------------------

    def execute(self, plan: str) -> ExecutionResult:
        """Execute a plan with control flow support and full tracing.

        Unlike PlanExecute which executes statement-by-statement, this
        instruments all primitives with tracing wrappers and executes
        the entire plan as a single block. Loop guards are injected via
        AST transformation to prevent infinite loops.
        """
        self.validate_plan(plan)

        exec_span: str | None = None
        if self._tracer:
            exec_span = self._tracer.start_span(
                EventType.EXECUTION_START,
                {"plan": plan} if self._tracer.config.capture_plan_source else {},
            )

        tree = ast.parse(plan)

        # Inject loop guards for safety
        tree = self._inject_loop_guards(tree)

        # Build execution namespace — primitives are added below as traced
        # wrappers, NOT as raw methods, so the LLM cannot bypass tracing.
        namespace: dict[str, Any] = {}
        namespace.update(self.allowed_builtins)
        if self.config.multi_turn:
            namespace.update(self._persisted_namespace)

        # Setup tracing infrastructure
        trace_steps: list[ExecutionStep] = []
        step_counter = [1]  # mutable counter
        read_only_map = self._get_primitive_read_only_map()

        # Reserved names include internal loop guard variables
        reserved_names = (
            {"__loop_limit__", "__loop_guard_raise__"}
            | self._build_reserved_names()
        )

        # Instrument primitives with tracing wrappers
        for prim_name, method in self._get_primitive_methods():
            traced = self._create_traced_primitive(
                prim_name,
                method,
                trace_steps,
                step_counter,
                read_only_map,
                namespace,
                reserved_names,
            )
            namespace[prim_name] = traced

        # Add loop guard infrastructure to namespace
        def _loop_guard_raise() -> None:
            raise RuntimeError(_LOOP_LIMIT_EXCEEDED_MSG)

        namespace["__loop_guard_raise__"] = _loop_guard_raise
        namespace["__loop_limit__"] = self.design_config.max_loop_iterations

        total_start = time.perf_counter()

        try:
            exec(  # noqa: S102
                compile(tree, PLAN_COMPILE_SOURCE, "exec"),
                empty_builtins(),
                namespace,
            )
        except RuntimeError as e:
            err_msg = str(e)
            if MUTATION_REJECTED_PREFIX not in err_msg:
                if _LOOP_LIMIT_EXCEEDED_MSG in err_msg:
                    trace_steps.append(
                        ExecutionStep(
                            step_number=step_counter[0],
                            statement="<loop limit exceeded>",
                            success=False,
                            error=f"Loop exceeded maximum iterations "
                            f"({self.design_config.max_loop_iterations})",
                        )
                    )
                else:
                    trace_steps.append(
                        ExecutionStep(
                            step_number=step_counter[0],
                            statement="<runtime error>",
                            success=False,
                            error=err_msg,
                        )
                    )
        except Exception as e:
            trace_steps.append(
                ExecutionStep(
                    step_number=step_counter[0],
                    statement="<execution error>",
                    success=False,
                    error=str(e),
                )
            )

        total_elapsed = time.perf_counter() - total_start

        self._persist_user_variables(namespace, reserved_names)

        trace = ExecutionTrace(
            steps=trace_steps,
            total_time_seconds=total_elapsed,
        )

        # Determine final result
        result_value = None
        result_type = NONE_TYPE_NAME
        result_name = ""
        result_json = NULL_JSON

        # Priority 1: explicit 'result' variable
        if "result" in namespace and "result" not in reserved_names:
            result_value = namespace["result"]
            result_name = "result"
        else:
            # Priority 2: last user-assigned variable in namespace
            for key in reversed(list(namespace.keys())):
                if key not in reserved_names and not key.startswith("__loop_guard_"):
                    result_value = namespace[key]
                    result_name = key
                    break

        if result_value is not None:
            result_type = type(result_value).__name__
            if not self.config.skip_result_serialization:
                try:
                    result_json = json.dumps(result_value, default=str)
                except (TypeError, ValueError):
                    result_json = NULL_JSON

        result = ExecutionResult(
            value_type=result_type,
            value_name=result_name,
            value_json=result_json,
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
    # Checkpoint-based Execution (Not Supported)
    # -------------------------------------------------------------------------

    def execute_stepwise(
        self,
        task: str,
        plan_context: PlanContext | None = None,
        serializer: SerializerRegistry | None = None,
        checkpoint_id: str | None = None,
    ) -> Iterator[ExecutionCheckpoint]:
        """Not supported for DesignExecute.

        Stepwise checkpoint execution is incompatible with block-based
        control flow execution. Use ``execute()`` or ``run()`` instead.
        """
        raise NotImplementedError(
            "DesignExecute does not support stepwise execution with control flow. "
            "Use execute() or run() instead."
        )
