"""Utilities for splitting and extracting sections from demarcated LLM prompts."""

from __future__ import annotations

from pydantic import BaseModel

from opensymbolicai.models import (
    PROMPT_CONTEXT_BEGIN,
    PROMPT_CONTEXT_END,
    PROMPT_DEFINITIONS_BEGIN,
    PROMPT_DEFINITIONS_END,
    PROMPT_INSTRUCTIONS_BEGIN,
    PROMPT_INSTRUCTIONS_END,
)


class PromptSections(BaseModel):
    """The three demarcated sections of an OpenSymbolicAI prompt."""

    preamble: str
    """Text before the DEFINITIONS marker (agent identity/description)."""

    definitions: str
    """Static per-agent content: primitive signatures, type defs, examples."""

    context: str
    """Dynamic per-call content: task, history, feedback."""

    instructions: str
    """Static per-blueprint content: rules and output format."""


def _extract_section(text: str, begin: str, end: str) -> str:
    """Extract text between a begin/end marker pair."""
    start_idx = text.find(begin)
    end_idx = text.find(end)
    if start_idx == -1 or end_idx == -1:
        return ""
    return text[start_idx + len(begin) : end_idx].strip()


def split_prompt(full_prompt: str) -> PromptSections:
    """Split a full 3-section prompt into its component parts.

    Sections with missing markers return empty strings. If no markers are
    present at all, the entire prompt is returned as the preamble.

    Args:
        full_prompt: A prompt, optionally containing DEFINITIONS, CONTEXT,
            and INSTRUCTIONS sections delimited by
            ``###<<SECTION:BEGIN>>###`` / ``###<<SECTION:END>>###`` markers.

    Returns:
        A PromptSections instance with the four extracted parts.
    """
    first_marker = full_prompt.find(PROMPT_DEFINITIONS_BEGIN)
    preamble = full_prompt[:first_marker].strip() if first_marker != -1 else full_prompt.strip()

    return PromptSections(
        preamble=preamble,
        definitions=_extract_section(
            full_prompt, PROMPT_DEFINITIONS_BEGIN, PROMPT_DEFINITIONS_END
        ),
        context=_extract_section(
            full_prompt, PROMPT_CONTEXT_BEGIN, PROMPT_CONTEXT_END
        ),
        instructions=_extract_section(
            full_prompt, PROMPT_INSTRUCTIONS_BEGIN, PROMPT_INSTRUCTIONS_END
        ),
    )


def extract_context(full_prompt: str) -> str:
    """Extract only the CONTEXT section from a full prompt.

    This is the dynamic per-call content, independent of the static
    DEFINITIONS and INSTRUCTIONS sections.

    Args:
        full_prompt: A prompt containing CONTEXT markers.

    Returns:
        The text between ``###<<CONTEXT:BEGIN>>###`` and
        ``###<<CONTEXT:END>>###``.

    Raises:
        ValueError: If the prompt is missing CONTEXT markers.
    """
    begin = full_prompt.find(PROMPT_CONTEXT_BEGIN)
    end = full_prompt.find(PROMPT_CONTEXT_END)
    if begin == -1 or end == -1:
        raise ValueError("Prompt missing CONTEXT markers")
    return full_prompt[begin + len(PROMPT_CONTEXT_BEGIN) : end].strip()
