from google import genai
from google.genai.types import (
    Part,
    Blob,
    FileData,
    GenerateContentConfig,
    HttpOptions,
)
from google.genai.types import SafetySetting, HarmCategory, HarmBlockThreshold

from enum import Enum
from pathlib import Path
from pydantic import BaseModel
import logging
import mimetypes
import json
import os
import time
import traceback
from typing import Any, Dict, Optional, Tuple, Type
from lib.utils.metrics_collector import metrics_collector


logger = logging.getLogger(__name__)


class GeminiGenaiManager:
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        location: Optional[str] = None,
        project: Optional[str] = None,
        http_timeout_s: int = 60,
    ):
        self.model = model
        # Resolve project/location from env at construction time so tests and
        # callers can override without editing the source.
        resolved_project = (
            project
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT")
        )
        resolved_location = (
            location
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or os.environ.get("GCP_LOCATION")
            or "us-central1"
        )
        if not resolved_project:
            raise RuntimeError(
                "GeminiGenaiManager requires GOOGLE_CLOUD_PROJECT (or an explicit "
                "project= arg). Set it in config.json or the environment."
            )
        self.project = resolved_project
        self.location = resolved_location
        self.genai_client = genai.Client(
            vertexai=True,
            location=resolved_location,
            project=resolved_project,
            http_options=HttpOptions(timeout=http_timeout_s * 1000),  # ms
        )

    def _convert_to_part(self, item) -> Part:
        """Convert Patok h, GCS URI, or string into Gemini-compatible Part"""
        if isinstance(item, Path):
            if not item.exists():
                raise FileNotFoundError(f"File not found: {item}")
            mime_type, _ = mimetypes.guess_type(item)
            if mime_type is None:
                raise ValueError(f"Could not guess MIME type for: {item}")
            return Part(inline_data=Blob(data=item.read_bytes(), mime_type=mime_type))
        elif isinstance(item, str) and (item.startswith("gs://") or item.startswith("https://")):
            return Part(file_data=FileData(file_uri=item))
        else:
            return Part(text=str(item))

    def _resolve_schema(self, schema: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Build a JSON schema dict and MIME type from various schema inputs."""
        if schema is None:
            return None, None

        # Pydantic BaseModel (class or instance) expose model_json_schema directly
        if hasattr(schema, "model_json_schema"):
            return schema.model_json_schema(), "application/json"

        # Enum subclasses can be converted into a simple enum JSON schema
        if isinstance(schema, type) and issubclass(schema, Enum):
            enum_values = [member.value for member in schema]
            if not enum_values:
                raise ValueError("Enum schema must define at least one value.")

            sample = enum_values[0]
            if isinstance(sample, str):
                json_type = "string"
            elif isinstance(sample, bool):
                json_type = "boolean"
            elif isinstance(sample, int):
                json_type = "integer"
            elif isinstance(sample, float):
                json_type = "number"
            else:
                raise TypeError(
                    "Unsupported enum value type for structured output: "
                    f"{type(sample).__name__}"
                )

            return {"type": json_type, "enum": enum_values}, "application/json"

        # Built-in JSON primitives
        builtin_schema_map: Dict[Type[Any], Dict[str, str]] = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
        }
        if isinstance(schema, type) and schema in builtin_schema_map:
            return builtin_schema_map[schema], "application/json"

        if isinstance(schema, dict):
            return schema, "application/json"

        # Fallback: no structured schema available, so return plain text
        return None, None

    def LLM_request(self, prompt_contents: list, schema: BaseModel = None, retry_delay=60, max_retries=3, context: Optional[str] = None):
        """
        Call Gemini with prompt and optional structured output schema.

        Args:
            prompt_contents: List of content items (strings, Paths, GCS URIs)
            schema: Optional Pydantic model for structured output
            retry_delay: Seconds to wait between retries
            max_retries: Maximum number of retry attempts
            context: Optional context string for metrics tracking (e.g., "evidence_retrieval")
        """
        parts = [self._convert_to_part(p) for p in prompt_contents]

        response_schema, response_mime_type = self._resolve_schema(schema)

        safety_settings = [
                    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
                    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
                    SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
                    SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
                    SafetySetting(category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=HarmBlockThreshold.BLOCK_NONE),
                ]
        config = GenerateContentConfig(
            response_schema=response_schema,
            response_mime_type=response_mime_type,
            safety_settings=safety_settings,
        )
        # Log request summary
        part_summaries = []
        for p in parts:
            if p.text:
                part_summaries.append(f"text({len(p.text)} chars)")
            elif p.inline_data:
                part_summaries.append(f"file({p.inline_data.mime_type}, {len(p.inline_data.data)} bytes)")
            elif p.file_data:
                part_summaries.append(f"uri({p.file_data.file_uri})")
        logger.info(
            f"[GEMINI] LLM_request: model={self.model} context={context} "
            f"parts=[{', '.join(part_summaries)}] schema={schema.__name__ if hasattr(schema, '__name__') else schema}"
        )

        for attempt in range(max_retries):
            try:
                t0 = time.time()
                response = self.genai_client.models.generate_content(
                    model=self.model,
                    contents=[{"role": "user", "parts": parts}],
                    config=config,
                )
                elapsed = time.time() - t0

                if response.text == None:
                    raise ValueError("Response text is None, likely an error occurred.")

                # Log token usage metrics if context provided
                usage = getattr(response, 'usage_metadata', None)
                if context and usage:
                    metrics_collector.log_tokens(context, usage)

                token_info = ""
                if usage:
                    token_info = (
                        f" input_tokens={getattr(usage, 'prompt_token_count', '?')}"
                        f" output_tokens={getattr(usage, 'candidates_token_count', '?')}"
                    )
                logger.info(
                    f"[GEMINI] Response: context={context} {elapsed:.1f}s{token_info} "
                    f"response_len={len(response.text or '')} chars"
                )

                if response_mime_type == "application/json":
                    if response.parsed is None:
                        raise ValueError("Response parsed is None, likely an error occurred.")
                    return response.parsed  # Already parsed
                return response.text

            except Exception as e:
                print(f"[ERROR] Gemini call failed: {e} (Attempt {attempt + 1}/{max_retries})")
                traceback.print_exc()
                if attempt + 1 < max_retries:
                    print(f"[INFO] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError("Gemini failed after all retries.") from e
