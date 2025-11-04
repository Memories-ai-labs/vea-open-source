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
import mimetypes
import json
import os
import time
import traceback
from typing import Any, Dict, Optional, Tuple, Type
from src.config import API_KEYS_PATH
from lib.utils.metrics_collector import metrics_collector


class GeminiGenaiManager:
    def __init__(self, model="gemini-2.5-flash", location="us-central1", project="research-459618"):
        self.load_api_keys()
        self.model = model
        self.genai_client = genai.Client(
            vertexai=True,
            location=location,
            project=project
        )

    def load_api_keys(self):
        """Load API keys from JSON into env vars"""
        if os.path.exists(API_KEYS_PATH):
            with open(API_KEYS_PATH, "r") as file:
                api_keys = json.load(file)
                for key, value in api_keys.items():
                    os.environ[key] = value
                    print(f"Loaded API key: {key}")
        else:
            print("Warning: API key file not found.")

    def _convert_to_part(self, item) -> Part:
        """Convert Path, GCS URI, or string into Gemini-compatible Part"""
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
        for attempt in range(max_retries):
            try:
                response = self.genai_client.models.generate_content(
                    model=self.model,
                    contents=[{"role": "user", "parts": parts}],
                    config=config,
                )

                if response.text == None:
                    raise ValueError("Response text is None, likely an error occurred.")

                # Log token usage metrics if context provided
                if context and hasattr(response, 'usage_metadata'):
                    metrics_collector.log_tokens(context, response.usage_metadata)

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
