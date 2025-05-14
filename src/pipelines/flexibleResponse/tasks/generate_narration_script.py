import asyncio

class GenerateNarrationScript:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, initial_response: str, user_prompt: str, indexing_data: dict, file_descriptions: dict):
        context_description = "\n".join(
            f"- {fname}: {desc}" for fname, desc in file_descriptions.items()
        )

        content_dump = "\n\n".join(
            f"===== {fname} =====\n{indexing_data[fname]}" for fname in indexing_data
        )

        prompt = (
            "You are refining a narration script for a video response to a user question about a movie.\n"
            f"User prompt:\n---\n{user_prompt.strip()}\n---\n\n"
            f"Initial assistant response:\n---\n{initial_response.strip()}\n---\n\n"
            "Below are supporting indexing documents:\n"
            f"{context_description}\n\n{content_dump}\n\n"
            "Please return a well-structured, spoken narration script appropriate for a text-to-speech voiceover.\n"
            "- Avoid unusual characters or formatting\n"
            "- Ensure a natural storytelling flow\n"
            "- Ensure you clearly refer to the timestamp of the clips if you are referencing them\n"
            "- Respect the length implied by the user prompt (e.g. if they asked for a short summary, keep it tight)\n"
            "- Output only the narration text\n"
        )

        return await asyncio.to_thread(self.llm.LLM_request, [prompt])
