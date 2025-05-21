from src.pipelines.longFormFlexibleResponse.schema import NarratedClip
import asyncio
import json

class GenerateVideoClipPlan:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, narration_script: str, user_prompt: str, indexing_data: dict, file_descriptions: dict):

        # Format descriptions of each file
        context_description = "\n".join(
            f"- {fname}: {desc}" for fname, desc in file_descriptions.items()
        )

        # Flatten the contents for inline use
        content_dump = "\n\n".join(
            f"===== {fname} =====\n{indexing_data[fname]}" for fname in indexing_data
        )
        
        prompt = (
            "You are assisting in building a video using clips from a movie to support a narration script.\n\n"
            f"User prompt:\n---\n{user_prompt.strip()}\n---\n\n"
            f"Narration script:\n---\n{narration_script.strip()}\n---\n\n"
            "The following indexing files have been provided:\n"
            f"{context_description}\n\n"
            "Below are the contents of the files:\n"
            f"{content_dump}\n\n"
            "Use the scene metadata provided below to match narration sentences to appropriate clips.\n"
            "For each clip, include:\n"
            "- `id`: unique identifier for the clip, just a sequentially incrementing integer is fine\n"
            "- `start`: timestamp (HH:MM:SS)\n"
            "- `end`: timestamp (HH:MM:SS)\n"
            "- `narration`: the line from the script this clip visually supports\n\n"
            "Unless explicitly asked for by the user's prompt, do not include timestamps in the narration sentence.\n"
            "Respond in JSON list format sorted in the order clips should appear in the final video.\n"
        )

        return await asyncio.to_thread(
            self.llm.LLM_request,
            [prompt],
            {
                "response_mime_type": "application/json",
                "response_schema": list[NarratedClip]
            }
        )
