import asyncio
import json 
class FlexibleGeminiAnswer:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, user_prompt: str, indexing_data: dict, file_descriptions: dict):
        # Compose file descriptions for context
        context_description = "\n".join(
            f"- {fname}: {desc}" for fname, desc in file_descriptions.items()
        )
        # Flatten contents, so each file is marked by its key ("media_name::file_type")
        content_dump = json.dumps(indexing_data, ensure_ascii=False, indent=4)

        # Construct the Gemini prompt
        prompt = (
            "You are an intelligent assistant helping a user analyze a set of video files (such as short clips or movie segments) with detailed indexing files.\n\n"
            "You have been provided the following index files, which cover one or more media files:\n"
            f"{context_description}\n\n"
            "Below are the contents of those index files (each section starts with the source file and type):\n"
            f"{content_dump}\n\n"
            "The user has asked:\n"
            f"---\n{user_prompt}\n---\n\n"
            "Your job is to generate the best possible response. You may answer in three ways:\n"
            "1. **Text Response Only**: If the question can be answered clearly using only text, do so concisely.\n"
            "2. **Text + Video Clips as Evidence**: If your answer would be improved with supporting video, include a list of scene timestamps from available files. **For each scene you reference, always specify which video file it comes from by name (e.g., travel1.mp4, videoA.mov, etc).** Include:\n"
            "    - The file name (media file)\n"
            "    - The start and end timestamps\n"
            "    - The scene description\n"
            "    - A short explanation for why you chose it\n"
            "3. **Video Response**: If the user explicitly asked for a video response:\n"
            "   - First, write a short narration script or video script suitable for voiceover.\n"
            "   - Then, identify scenes from  whose visual content best matches the script.\n"
            "   - Provide a list of these scene timestamps and file names with descriptions and how they match the lines of the script.\n\n"
            "If you reference any scenes, timestamps, or visuals, **always say which media file each one comes from**. This is essential when the user uploads multiple videos.\n"
            "\n"
            "Respond in detailed plain text (do NOT use JSON). Another agent will format your answer for structuring and downstream processing."
        )

        response = await asyncio.to_thread(self.llm.LLM_request, [prompt])
        # print(response)
        return response
