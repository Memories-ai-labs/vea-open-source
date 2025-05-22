# src/pipelines/movieRecapEditing/tasks/classify_response_type.py

import asyncio
from google import genai
from src.pipelines.flexibleResponse.schema import ResponseForm

class ClassifyResponseType:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, user_prompt: str, raw_gemini_response: str):
        prompt = (
            "You are a classifier. You are given a user's prompt and an assistant's response "
            "to a question about a movie or TV show.\n\n"
            "Your job is to classify the **type of response** the assistant produced based on the following rules:\n"
            "- If the assistant clearly and directly answers the user's question using only text, choose `text_only`.\n"
            "- If the assistant answers the question and provides one or more specific video timestamps, scene descriptions, or references to clips from the media, choose `text_and_evidence`.\n"
            "Here is the user's prompt:\n"
            f"---\n{user_prompt.strip()}\n---\n\n"
            "Here is the assistant's full response:\n"
            f"---\n{raw_gemini_response.strip()}\n---\n\n"
            "Now return only the correct enum value:\n"
            "- text_only\n"
            "- text_and_evidence\n"
        )

        response = await asyncio.to_thread(
            self.llm.LLM_request,
            [prompt],
            config={
                "response_mime_type": "text/x.enum",
                "response_schema": ResponseForm
            }
        )

        print(f"[DEBUG] Classified response type: {response}")
        return response
