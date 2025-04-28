import asyncio
import random
from io import IOBase
from pathlib import Path
from typing import AsyncIterable, Iterable, Literal, overload
from google.genai.types import ContentListUnion, ContentListUnionDict, HttpOptions

from google import genai
import vertexai
from google.api_core.exceptions import (
    InternalServerError,
    ResourceExhausted,
    ServiceUnavailable,
)
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry,
    stop_after_attempt,
    wait_exponential,
)
from vertexai.generative_models import (
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    SafetySetting,
)
from vertexai.generative_models._generative_models import ContentsType
import logging
import certifi
import os

os.environ["SSL_CERT_FILE"] = certifi.where()

logger = logging.getLogger(__name__)

OldSafetySettings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

SafetySettings = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
]


def build_gcs_part(uri: str, mime_type: str):
    return Part.from_uri(uri=uri, mime_type=mime_type)


def gemini_response_to_str(response: GenerationResponse):
    if response.candidates and response.candidates[0].content.parts:
        generated_text = response.candidates[0].content.parts[0].text
    else:
        generated_text = f"Please ask another question. Reason: {response.prompt_feedback.block_reason}"
    return generated_text


def gemini_retry_condition(retry_state):
    if isinstance(retry_state.outcome.exception(), asyncio.TimeoutError):
        return True
    elif isinstance(retry_state.outcome.exception(), ResourceExhausted):
        return True
    elif isinstance(retry_state.outcome.exception(), InternalServerError):
        return True
    elif isinstance(retry_state.outcome.exception(), ServiceUnavailable):
        return True
    return False


ModelRegions = {
    "gemini-2.0-flash": [
        "us-east5",
        "us-south1",
        "us-central1",
        "us-west4",
        "us-east1",
        "us-east4",
        "us-west1",
    ]
}


class Gemini:
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        location: str | None = None,
    ):
        self.logger = logger
        self.model_name = model_name

        if model_name not in ModelRegions.keys():
            raise ValueError(f"Model name {model_name} is not supported")

        if location is not None:
            if location not in ModelRegions[model_name]:
                raise ValueError(
                    f"Location {location} is not supported for model {model_name}"
                )
            else:
                self.current_region = location
        else:
            self.available_regions = ModelRegions.get(model_name, ["us-central1"])
            self.current_region = random.choice(self.available_regions)

        self.logger.info(
            f"Model name : {self.model_name}, Available regions: {self.available_regions}"
        )

        vertexai.init(
            location=self.current_region,
            api_transport="grpc",
            project="gen-lang-client-0057517563",
        )

        self.client = genai.Client(
            vertexai=True,
            project="gen-lang-client-0057517563",
            location=self.current_region,
        )

    def build_gcs_part(self, uri: str, mime_type: str):
        return build_gcs_part(uri, mime_type)

    def random_change_region(self):
        exclude_current_region = [
            r for r in self.available_regions if r != self.current_region
        ]
        region = random.choice(exclude_current_region)
        vertexai.init(location=region, api_transport="grpc")
        self.logger.info(f"Changed region to {region}")

    @overload
    def generate(
        self,
        contents: ContentListUnion | ContentListUnionDict,
    ) -> Iterable[GenerationResponse]: ...

    @overload
    def generate(
        self,
        contents: ContentListUnion | ContentListUnionDict,
    ) -> GenerationResponse: ...

    def generate(
        self,
        contents: ContentListUnion | ContentListUnionDict,
        generation_config: GenerationConfig,
    ) -> Iterable[GenerationResponse] | GenerationResponse:
        for attempt in Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=4, max=30),
            retry=gemini_retry_condition,
        ):
            with attempt:
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=generation_config,
                    )
                    return response

                except ResourceExhausted as e:
                    self.logger.error(f"Resource exhausted: {e}")
                    self.random_change_region()
                    raise
                except ServiceUnavailable as e:
                    self.logger.error(f"Service unavailable: {e}")
                    self.random_change_region()
                    raise
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred: {e}")
                    raise
        raise Exception("Failed to generate content")

    @overload
    async def generate_async(
        self, prompt: ContentsType
    ) -> AsyncIterable[GenerationResponse]: ...

    @overload
    async def generate_async(self, prompt: ContentsType) -> GenerationResponse: ...

    async def generate_async(self, prompt: ContentsType, generation_config: GenerationConfig):
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=4, max=30),
            retry=gemini_retry_condition,
        ):
            with attempt:
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=generation_config,
                    )
                    return response

                except asyncio.TimeoutError:
                    self.logger.error("LLM request timed out")
                    raise
                except ResourceExhausted as e:
                    self.logger.error(f"Resource exhausted: {e}")
                    self.random_change_region()
                    raise
                except ServiceUnavailable as e:
                    self.logger.error(f"Service unavailable: {e}")
                    self.random_change_region()
                    raise
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred: {e}")
                    raise

        raise Exception(
            f"Failed to generate content for prompt {prompt}. All retry attempts failed."
        )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def upload_file(
        self,
        path: str | Path | IOBase,
        mime_type: str,
    ):
        image_ref = genai.upload_file(path, mime_type=mime_type)
        return image_ref

    def count_tokens(self, content: ContentsType) -> int:
        return self.client.models.count_tokens(content).total_tokens


if __name__ == "__main__":

    def main():
        gemini = Gemini()
        prompt = "What is the capital of France?"
        print(gemini.generate(prompt, stream=False))

    async def async_main():
        gemini = Gemini()
        prompt = "What is the capital of France?"
        print(await gemini.generate_async(prompt, stream=False))
        gemini.random_change_region()
        await asyncio.sleep(5)
        print(await gemini.generate_async(prompt, stream=False))

    asyncio.run(async_main())
    # aiplatform.init()
    # print(aiplatform.Model.list())
