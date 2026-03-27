"""
OpenRouter LLM Manager — drop-in replacement for GeminiGenaiManager.

Uses the OpenAI SDK pointed at OpenRouter's API. Supports:
- Text generation (LLM_request)
- Structured output via JSON schema
- Function calling / tool use (for agent session)
- Multimodal input (images as base64, video frames extracted)
- Multi-turn conversation history

All methods match GeminiGenaiManager's interface so calling code
doesn't need to change.
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import time
import traceback
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Type

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OpenRouterManager:
    """Drop-in replacement for GeminiGenaiManager using OpenRouter API."""

    def __init__(
        self,
        model: str = "google/gemini-2.5-flash",
        api_key: str = "",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        # Compatibility: agent_session.py accesses self.gemini.genai_client
        # We provide a shim that routes generate_content calls through OpenAI
        self.genai_client = _GenaiClientShim(self)

    # ── LLM_request (Pattern A: 10 of 11 integration points) ──────────

    def LLM_request(
        self,
        prompt_contents: list,
        schema: BaseModel = None,
        retry_delay: int = 60,
        max_retries: int = 3,
        context: Optional[str] = None,
    ):
        """
        Call the LLM with prompt and optional structured output schema.
        Interface matches GeminiGenaiManager.LLM_request exactly.
        """
        messages = self._build_messages(prompt_contents)
        response_format = self._resolve_schema(schema)

        # Log request summary
        part_summaries = []
        for msg in messages:
            if isinstance(msg["content"], str):
                part_summaries.append(f"text({len(msg['content'])} chars)")
            elif isinstance(msg["content"], list):
                for part in msg["content"]:
                    if part.get("type") == "text":
                        part_summaries.append(f"text({len(part['text'])} chars)")
                    elif part.get("type") == "image_url":
                        part_summaries.append("image")
        logger.info(
            f"[OPENROUTER] LLM_request: model={self.model} context={context} "
            f"parts=[{', '.join(part_summaries)}] "
            f"schema={schema.__name__ if hasattr(schema, '__name__') else schema}"
        )

        for attempt in range(max_retries):
            try:
                t0 = time.time()

                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                }
                if response_format:
                    kwargs["response_format"] = response_format

                response = self.client.chat.completions.create(**kwargs)
                elapsed = time.time() - t0

                text = response.choices[0].message.content or ""

                if not text.strip():
                    raise ValueError("Response text is None, likely an error occurred.")

                # Log metrics
                usage = response.usage
                token_info = ""
                if usage:
                    token_info = (
                        f" input_tokens={usage.prompt_tokens}"
                        f" output_tokens={usage.completion_tokens}"
                    )
                logger.info(
                    f"[OPENROUTER] Response: context={context} {elapsed:.1f}s{token_info} "
                    f"response_len={len(text)} chars"
                )

                # If schema requested, parse JSON response into Pydantic model
                if schema is not None and hasattr(schema, "model_validate"):
                    try:
                        parsed = schema.model_validate_json(text)
                        return parsed
                    except Exception as e:
                        # Try cleaning the JSON (remove markdown code fences)
                        cleaned = text.strip()
                        if cleaned.startswith("```"):
                            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                        if cleaned.endswith("```"):
                            cleaned = cleaned.rsplit("```", 1)[0]
                        cleaned = cleaned.strip()
                        parsed = schema.model_validate_json(cleaned)
                        return parsed

                return text

            except Exception as e:
                print(f"[ERROR] OpenRouter call failed: {e} (Attempt {attempt + 1}/{max_retries})")
                traceback.print_exc()
                if attempt + 1 < max_retries:
                    print(f"[INFO] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError("OpenRouter failed after all retries.") from e

    # ── Message building ──────────────────────────────────────────────

    def _build_messages(self, prompt_contents: list) -> List[Dict]:
        """Convert prompt_contents list to OpenAI messages format."""
        parts = []
        for item in prompt_contents:
            if isinstance(item, Path):
                # File input — convert to base64 image or extract video frames
                if not item.exists():
                    raise FileNotFoundError(f"File not found: {item}")
                mime_type, _ = mimetypes.guess_type(str(item))
                if mime_type and mime_type.startswith("image/"):
                    data = base64.b64encode(item.read_bytes()).decode("utf-8")
                    parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{data}",
                        },
                    })
                elif mime_type and mime_type.startswith("video/"):
                    # Extract keyframes from video and send as images
                    frames = self._extract_video_frames(item, max_frames=8)
                    for frame_data, frame_mime in frames:
                        parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{frame_mime};base64,{frame_data}",
                            },
                        })
                else:
                    # Unknown file type — send as text description
                    parts.append({
                        "type": "text",
                        "text": f"[File: {item.name}, type: {mime_type}, size: {item.stat().st_size} bytes]",
                    })
            elif isinstance(item, str) and (item.startswith("gs://") or item.startswith("https://")):
                # GCS/HTTP URI — mention as context (can't send directly)
                parts.append({"type": "text", "text": f"[Reference: {item}]"})
            else:
                parts.append({"type": "text", "text": str(item)})

        # If all parts are text, flatten to a single string
        if all(p.get("type") == "text" for p in parts):
            combined = "\n\n".join(p["text"] for p in parts)
            return [{"role": "user", "content": combined}]

        return [{"role": "user", "content": parts}]

    def _extract_video_frames(
        self, video_path: Path, max_frames: int = 8
    ) -> List[Tuple[str, str]]:
        """Extract keyframes from video as base64 JPEG images."""
        try:
            import cv2
            import numpy as np

            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return []

            indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
            frames = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Resize to max 512px width for token efficiency
                    h, w = frame.shape[:2]
                    if w > 512:
                        scale = 512 / w
                        frame = cv2.resize(frame, (512, int(h * scale)))
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                    frames.append((b64, "image/jpeg"))

            cap.release()
            return frames
        except Exception as e:
            logger.warning(f"[OPENROUTER] Failed to extract video frames: {e}")
            return []

    # ── Schema resolution ─────────────────────────────────────────────

    def _resolve_schema(self, schema: Any) -> Optional[Dict]:
        """Convert Pydantic model / Enum / type to OpenAI response_format."""
        if schema is None:
            return None

        if hasattr(schema, "model_json_schema"):
            json_schema = schema.model_json_schema()
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": getattr(schema, "__name__", "output"),
                    "schema": json_schema,
                    "strict": False,
                },
            }

        if isinstance(schema, type) and issubclass(schema, Enum):
            return {"type": "json_object"}

        if isinstance(schema, dict):
            return {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema, "strict": False},
            }

        # Fallback: request JSON output
        return {"type": "json_object"}


# ═══════════════════════════════════════════════════════════════════════
# Shim: make OpenRouter look like google.genai for agent_session.py
# ═══════════════════════════════════════════════════════════════════════

class _GenaiClientShim:
    """
    Wraps OpenRouterManager to expose a .models.generate_content() interface
    matching google.genai.Client so agent_session.py works unchanged.
    """

    def __init__(self, manager: OpenRouterManager):
        self._mgr = manager
        self.models = self

    def generate_content(self, model: str, contents: list, config=None):
        """
        Convert Google genai generate_content call to OpenAI chat completion.

        Args:
            model: Model name (ignored, uses manager's model)
            contents: List of Content objects (google.genai.types.Content)
            config: GenerateContentConfig with system_instruction, tools, safety_settings
        """
        messages = []

        # Extract system instruction
        if config and hasattr(config, "system_instruction") and config.system_instruction:
            messages.append({"role": "system", "content": str(config.system_instruction)})

        # Convert Content objects to OpenAI messages
        for content in contents:
            role = _map_role(getattr(content, "role", "user"))
            parts = getattr(content, "parts", [])

            # Handle function call responses (tool role)
            if role == "tool":
                for part in parts:
                    fn_resp = getattr(part, "function_response", None)
                    if fn_resp:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": _make_tool_call_id(fn_resp.name),
                            "content": json.dumps(fn_resp.response) if isinstance(fn_resp.response, dict) else str(fn_resp.response),
                        })
                continue

            # Handle model responses with function calls
            if role == "assistant":
                text_parts = []
                tool_calls = []
                for part in parts:
                    if getattr(part, "function_call", None):
                        fc = part.function_call
                        tool_calls.append({
                            "id": _make_tool_call_id(fc.name),
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(dict(fc.args)) if fc.args else "{}",
                            },
                        })
                    elif getattr(part, "text", None):
                        text_parts.append(part.text)

                msg: Dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    msg["content"] = "\n".join(text_parts)
                else:
                    msg["content"] = None
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                messages.append(msg)
                continue

            # Regular user/model text messages
            text_parts = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
            if text_parts:
                messages.append({"role": role, "content": "\n".join(text_parts)})

        # Convert tool declarations
        tools = None
        if config and hasattr(config, "tools") and config.tools:
            tools = _convert_tool_declarations(config.tools)

        # Make the API call
        kwargs: Dict[str, Any] = {
            "model": self._mgr.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        t0 = time.time()
        response = self._mgr.client.chat.completions.create(**kwargs)
        elapsed = time.time() - t0

        usage = response.usage
        token_info = f" input={usage.prompt_tokens} output={usage.completion_tokens}" if usage else ""
        logger.info(f"[OPENROUTER] generate_content: {elapsed:.1f}s{token_info}")

        # Convert OpenAI response back to Google genai format
        return _convert_response(response)


# ═══════════════════════════════════════════════════════════════════════
# Conversion utilities
# ═══════════════════════════════════════════════════════════════════════

def _map_role(role: str) -> str:
    """Map Google roles to OpenAI roles."""
    return {"user": "user", "model": "assistant", "tool": "tool"}.get(role, role)


_tool_call_counter = 0

def _make_tool_call_id(name: str) -> str:
    """Generate a deterministic tool_call_id from function name."""
    global _tool_call_counter
    _tool_call_counter += 1
    return f"call_{name}_{_tool_call_counter}"


def _convert_tool_declarations(tools_list: list) -> List[Dict]:
    """Convert Google Tool(function_declarations=[...]) to OpenAI tools format."""
    result = []
    for tool in tools_list:
        declarations = getattr(tool, "function_declarations", [])
        for fd in declarations:
            # Convert Schema protobuf to dict
            params = fd.parameters
            if hasattr(params, "to_json_dict"):
                params = _lowercase_schema_types(params.to_json_dict())
            elif not isinstance(params, dict):
                params = {"type": "object", "properties": {}}
            result.append({
                "type": "function",
                "function": {
                    "name": fd.name,
                    "description": fd.description or "",
                    "parameters": params,
                },
            })
    return result


def _lowercase_schema_types(schema: dict) -> dict:
    """Convert Google Schema uppercase types (STRING, OBJECT) to JSON Schema lowercase."""
    type_map = {
        "STRING": "string", "OBJECT": "object", "ARRAY": "array",
        "NUMBER": "number", "INTEGER": "integer", "BOOLEAN": "boolean",
    }
    result = {}
    for k, v in schema.items():
        if k == "type" and isinstance(v, str):
            result[k] = type_map.get(v, v.lower())
        elif isinstance(v, dict):
            result[k] = _lowercase_schema_types(v)
        elif isinstance(v, list):
            result[k] = [_lowercase_schema_types(i) if isinstance(i, dict) else i for i in v]
        else:
            result[k] = v
    return result


def _convert_response(openai_response) -> SimpleNamespace:
    """
    Convert OpenAI ChatCompletion response to a google.genai-like response
    that agent_session.py can consume.

    Returns an object with:
        .candidates[0].content.role = "model"
        .candidates[0].content.parts = [Part(...), ...]
    """
    choice = openai_response.choices[0]
    message = choice.message

    parts = []

    # Text content
    if message.content:
        parts.append(SimpleNamespace(
            text=message.content,
            function_call=None,
            function_response=None,
            inline_data=None,
            file_data=None,
        ))

    # Tool calls → function_call parts
    if message.tool_calls:
        for tc in message.tool_calls:
            args = {}
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {"raw": tc.function.arguments}

            parts.append(SimpleNamespace(
                text=None,
                function_call=SimpleNamespace(
                    name=tc.function.name,
                    args=args,
                ),
                function_response=None,
                inline_data=None,
                file_data=None,
            ))

    content = SimpleNamespace(
        role="model",
        parts=parts,
    )
    candidate = SimpleNamespace(content=content)

    return SimpleNamespace(
        candidates=[candidate],
        text=message.content,
        parsed=None,
        usage_metadata=SimpleNamespace(
            prompt_token_count=openai_response.usage.prompt_tokens if openai_response.usage else 0,
            candidates_token_count=openai_response.usage.completion_tokens if openai_response.usage else 0,
        ) if openai_response.usage else None,
    )
