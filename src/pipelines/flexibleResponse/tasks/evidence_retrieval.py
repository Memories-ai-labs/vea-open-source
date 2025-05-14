# src/pipelines/flexibleResponse/tasks/evidence_retrieval.py

import asyncio
import json
from src.pipelines.flexibleResponse.schema import EvidenceClip

class EvidenceRetrieval:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, initial_response: str, indexing_data: dict, file_descriptions: dict):
        scenes_json = indexing_data.get("scenes.json", {})

        prompt = (
            "You are given an assistant's long-form text response to a user prompt about a movie.\n"
            "Here is the response:\n"
            f"---\n{initial_response.strip()}\n---\n\n"
            "You also have access to scene metadata (scenes.json) that includes timestamped scene descriptions.\n"
            "Select relevant scenes that could serve as **visual evidence** for the points made in the response.\n\n"
            "For each clip you select, include:\n"
            "- `start`: timestamp (HH:MM:SS)\n"
            "- `end`: timestamp (HH:MM:SS)\n"
            "- `description`: short description of the visual content\n"
            "- `reason`: explanation of why this scene supports the assistant's response\n\n"
            "Respond using structured JSON format only. Do not include any extra text or explanation.\n"
            "Here is the scene metadata:\n"
            f"{json.dumps(scenes_json, ensure_ascii=False)}"
        )

        return await asyncio.to_thread(
            self.llm.LLM_request,
            [prompt],
            {
                "response_mime_type": "application/json",
                "response_schema": list[EvidenceClip]
            }
        )
