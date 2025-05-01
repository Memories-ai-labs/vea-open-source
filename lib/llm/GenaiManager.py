from google import genai
from tqdm import tqdm
import os
import shutil
from pathlib import Path
import json
import time 
from src.config import API_KEYS_PATH
import traceback

class GeminiMovieRecap:
    def __init__(self):
        self.gemini_api_key = os.getenv("GENAI_API_KEY")
        self.genai_client = genai.Client(api_key=self.gemini_api_key)
        self.model = "gemini-2.0-flash"

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
            

    def robust_gemini_call(self, prompt_contents, config=None, retry_delay=60, max_retries=3):
        attempt = 0
        while attempt < max_retries:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt_contents,
                    config=config or {}
                )
                if config and config.get("response_mime_type") == "application/json":
                    return json.loads(response.text)
                return response.text

            except Exception as e:
                attempt += 1
                error_str = str(e)
                print(f"[ERROR] Gemini call failed: {error_str} (Attempt {attempt}/{max_retries})")
                
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
