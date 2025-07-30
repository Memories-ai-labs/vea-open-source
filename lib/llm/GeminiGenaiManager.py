from google import genai
from google.genai.types import (
    Part,
    Blob,
    FileData,
    GenerateContentConfig,
    HttpOptions,
)
from google.genai.types import SafetySetting, HarmCategory, HarmBlockThreshold

from pathlib import Path
from pydantic import BaseModel
import mimetypes
import json
import os
import time
import traceback
from src.config import API_KEYS_PATH


class GeminiGenaiManager:
    def __init__(self, model="gemini-2.5-flash", location="us-central1", project="alex-oix"):
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

    def LLM_request(self, prompt_contents: list, schema: BaseModel = None, retry_delay=60, max_retries=3):
        """
        Call Gemini with prompt and optional structured output schema.
        """
        parts = [self._convert_to_part(p) for p in prompt_contents]

        response_schema = None
        response_mime_type = None
        if schema is not None:
            response_schema = schema.model_json_schema()
            response_mime_type = "application/json"

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

                if response_mime_type == "application/json":
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
