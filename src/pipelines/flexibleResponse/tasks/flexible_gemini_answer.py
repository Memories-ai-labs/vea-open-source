import asyncio

class FlexibleGeminiAnswer:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, user_prompt: str, indexing_data: dict, file_descriptions: dict):
        # Format descriptions of each file
        context_description = "\n".join(
            f"- {fname}: {desc}" for fname, desc in file_descriptions.items()
        )
        # Flatten the contents for inline use
        content_dump = "\n\n".join(
            f"===== {fname} =====\n{indexing_data[fname]}" for fname in indexing_data
        )

        # Construct the Gemini prompt
        prompt = (
            "You are an intelligent assistant helping a user analyze a long-form narrative media like a movie or TV episode.\n\n"
            "The following indexing files have been provided:\n"
            f"{context_description}\n\n"
            "Below are the contents of the files:\n"
            f"{content_dump}\n\n"
            "The user has asked the following question or prompt:\n"
            f"---\n{user_prompt}\n---\n\n"
            "Your job is to generate the best possible response. You may respond in one of three ways:\n\n"
            "1. **Text Response Only**: If the question can be clearly answered using only text, do so concisely.\n\n"
            "2. **Text + Video Clips as Evidence**: If your answer would be stronger with supporting visual evidence, include a list of scene timestamps from `scenes.json`.\n"
            "   - For each scene used, include:\n"
            "     - start and end timestamps\n"
            "     - the scene description\n"
            "     - a short explanation of why it supports your answer\n\n"
            "3. **Video Response**: If the user explicitly asked for a video response:\n"
            "   - First, write a short narration script or video script suitable for voiceover.\n"
            "   - Then, identify scenes from `scenes.json` whose visual content best matches the script.\n"
            "   - Provide a list of these scene timestamps with descriptions and how they match the lines of the script.\n\n"
            "Respond in detailed plain text. DO NOT format your answer as structured JSON.\n"
            "**Another agent will handle formatting and structuring later.**"
        )
        response = await asyncio.to_thread(self.llm.LLM_request, [prompt])

        print(response)

        return response