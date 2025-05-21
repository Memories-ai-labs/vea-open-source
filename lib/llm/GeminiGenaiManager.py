from google import genai
from tqdm import tqdm
import os
from pathlib import Path
import json
import time 
import traceback
from google.genai import types
import mimetypes
from src.config import API_KEYS_PATH

class GeminiGenaiManager:
    def __init__(self, model = "gemini-2.0-flash"):
        self.load_api_keys()
        self.gemini_api_key = os.getenv("GENAI_API_KEY")
        self.genai_client = genai.Client(api_key=self.gemini_api_key)
        self.model = model

    def load_api_keys(self):
        """Loads API keys from a JSON file and sets them as environment variables."""
        if os.path.exists(API_KEYS_PATH):
            with open(API_KEYS_PATH, "r") as file:
                api_keys = json.load(file)
                for key, value in api_keys.items():
                    os.environ[key] = value
                    print(f"Loaded API key: {key}")  # not printing actual values for security
        else:
            print("Warning: config.json not found. API keys not loaded.")
            
    def _convert_path_to_blob_part(self, file_path: Path) -> types.Part:
        """
        Converts a file at the given path into a Gemini-compatible Part containing an inline Blob.

        Args:
            file_path (Path): Path to the media file.

        Returns:
            types.Part: A Gemini Part containing the file blob with correct mime_type.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If MIME type cannot be determined or file is too large.
        """
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size
    
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            raise ValueError(f"Could not determine MIME type for: {file_path}")

        file_bytes = file_path.read_bytes()

        return types.Part(
            inline_data=types.Blob(data=file_bytes, mime_type=mime_type)
        )

    def LLM_request(self, prompt_contents, config=None, retry_delay=60, max_retries=3):
        """
        Sends prompt_contents to Gemini. Converts Path objects to inline file blobs.
        Handles JSON parse errors, blank responses, rate limits (429), and retries.
        """
        parts = []
        for item in prompt_contents:
            if isinstance(item, Path):
                parts.append(self._convert_path_to_blob_part(item))
            else:
                parts.append(types.Part(text=item))

        attempt = 0
        while attempt < max_retries:
            try:
                response = self.genai_client.models.generate_content(
                    model=self.model,
                    contents=types.Content(parts=parts),
                    config=config or {}
                )

                # Handle JSON config separately
                if config and config.get("response_mime_type") == "application/json":
                    if not response.text or not response.text.strip():
                        raise ValueError("Blank JSON response from Gemini.")
                    return json.loads(response.text)

                # Handle plain text
                if not response.text or not response.text.strip():
                    print("[INFO] Detected blank response.")
                    raise ValueError("Blank response from Gemini.")

                return response.text

            except Exception as e:
                attempt += 1
                error_str = str(e)
                print(f"[ERROR] Gemini call failed: {error_str} (Attempt {attempt}/{max_retries})")
                print(response)
                if isinstance(e, json.JSONDecodeError):
                    print("[INFO] Detected JSON decode error. Retrying in 10 seconds...")
                    time.sleep(10)
                elif "429" in error_str:
                    print("[INFO] Detected rate limit error (429). Retrying in 60 seconds...")
                    time.sleep(60)
                else:
                    print(prompt_contents)
                    traceback.print_exc()
                    print(f"[INFO] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

        raise RuntimeError(f"[FAILURE] Gemini failed after {max_retries} attempts.")
