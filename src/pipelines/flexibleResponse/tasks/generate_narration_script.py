
import asyncio
import json
class GenerateNarrationScript:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, initial_response: str, user_prompt: str, indexing_data: dict, file_descriptions: dict):
        context_description = "\n".join(
            f"- {fname}: {desc}" for fname, desc in file_descriptions.items()
        )

        content_dump = json.dumps(indexing_data, ensure_ascii=False, indent=4)

        prompt = (
            "You are refining a narration script for a video response to a user question about a movie or a collection of media files.\n"
            "you should tailor the narration to the user's prompt, and abide by any specific requests made in the user's prompt.\n"
            f"User prompt:\n---\n{user_prompt.strip()}\n---\n\n"
            f"Initial assistant response:\n---\n{initial_response.strip()}\n---\n\n"
            "Below are supporting indexing documents (possibly from multiple media files):\n"
            f"{context_description}\n\n{content_dump}\n\n"
            "Please return a well-structured, spoken narration script appropriate for a text-to-speech voiceover. "
            "If the user prompt asks a question, you should clearly address the question. your language should be engaging and fluent.\n"
            "- Avoid unusual characters or formatting\n"
            "- Ensure a natural storytelling flow\n"
            "- If your narration references specific clips or scenes, specify both the media file name and the timestamps.\n"
            "- Respect the length implied by the user prompt (e.g. if they asked for a short summary, keep it tight)\n"
            "- Output only the narration text\n"
        )

        return await asyncio.to_thread(self.llm.LLM_request, [prompt])