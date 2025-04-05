import time
from google import genai
from google.genai.types import File
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Optional, Dict, Any
import logging
import json
import certifi
import os

os.environ['SSL_CERT_FILE'] = certifi.where()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.api_key = "AIzaSyDj7qXlQxeTH1JTFRBr2NKFBqjAFNEvK4g"
        self.model = "gemini-1.5-flash"
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
            
            # Default parameters
        self.default_params = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def generate_response(
        self,
        contents: List[Any],
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response using Gemini client with retry mechanism.
        
        Args:
            prompt: The input prompt for Gemini
            params: Optional parameters to override defaults
            
        Returns:
            The generated response text
            
        Raises:
            Exception: If generation fails after retries
        """
        try:
            generation_config = self.default_params.copy()
            if params:
                generation_config.update(params)

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generation_config
            )

            # Check for empty response
            if not response.candidates:
                return "No response candidates received"

            candidate = response.candidates[0]
            if not candidate.content.parts:
                return "Empty response content"

            result = candidate.content.parts[0].text

            logger.info("Successfully generated response from Gemini")
            return result

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Log the full error details for debugging
            logger.debug(f"Full error details: {json.dumps(str(e), indent=2)}")
            raise

    async def get_model_status(self) -> bool:
        """
        Check if the model is available and responding.
        """
        try:
            test_prompt = "Test connection"
            response = await self.generate_response(test_prompt)
            return bool(response)
        except Exception as e:
            logger.error(f"Error checking model status: {str(e)}")
            return False

    def update_default_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update the default parameters for generation.
        """
        self.default_params.update(new_params)
        logger.info(f"Updated default parameters: {json.dumps(new_params, indent=2)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def upload_file(self, file_path: str) -> File:
        """
        Upload a file to Gemini.
        
        Args:
            file_path: The path to the file to upload
        """
        try:
            uploaded_file = self.client.files.upload(file=file_path)
            while uploaded_file.state.name == "PROCESSING":
                print(f"processing video... {uploaded_file.name}")
                time.sleep(5)
                uploaded_file = self.client.files.get(name=uploaded_file.name)
            uploaded_file.display_name = os.path.basename(file_path)
            print(f"processing video {uploaded_file.display_name} { uploaded_file.state.name}")
            return uploaded_file
        except Exception as e:
            print(f"[WARN] Upload failed for {file_path}: {str(e)}")
            raise ValueError(e)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def delete_all_files(self, file_paths: List[File]):
        all_files = self.client.files.list()
        all_files_names = [f.name for f in all_files]
        for f in file_paths:
            if f.name in all_files_names:
                self.client.files.delete(name=f.name)
