"""Async structured-output adapter over VEA's sync LLM managers.

``OpenRouterManager`` / ``GeminiGenaiManager`` expose one blocking call,
``LLM_request(prompt_contents, schema=...)``, which returns either text or a
parsed pydantic model. Callers that want ``await llm.generate_structured(
messages, Schema) -> (parsed, usage)`` wrap the manager in ``StructuredLLM``.

The manager does not report token counts, so ``Usage`` comes back zeroed;
call sites should treat a zero as "unknown" rather than "free".
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List


@dataclass
class Message:
    """One chat turn. ``role`` is ``system`` / ``user`` / ``assistant``."""

    role: str
    content: str


@dataclass
class Usage:
    """Token accounting. Zero means the backend did not report it."""

    input_tokens: int = 0
    output_tokens: int = 0


class StructuredLLM:
    """Adapt a sync VEA LLM manager to an async structured-output call."""

    def __init__(self, manager: Any, context: str = "structured") -> None:
        self.manager = manager
        self.context = context

    async def generate_structured(
        self,
        messages: List[Message],
        schema: Any,
        retry_delay: int = 30,
        max_retries: int = 2,
    ) -> tuple[Any, Usage]:
        prompts = [m.content for m in messages]
        parsed = await asyncio.to_thread(
            self.manager.LLM_request, prompts, schema, retry_delay, max_retries, self.context
        )
        return parsed, Usage()
