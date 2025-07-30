import asyncio
from pathlib import Path

class NaiveGeminiComprehension:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, segments: list[dict]):
        descriptions = []

        for i, seg in enumerate(segments):
            path = seg["path"]
            print(f"[INFO] Gemini naive baseline - segment {i+1}: {path.name}")

            prompt = (
                "you are an AI tasked with understanding video content, such as movies, tv shows, interviews, sports games etc. "
                "Describe the contents of this segment in detail. include key events, characters, and any important dialogue or actions." 
                "then, provide a list of scenes in the segment and describe each scene briefly including timestamps"
                "Return plain text only."
            )
            desc = await asyncio.to_thread(self.llm.LLM_request, [path, prompt])
            descriptions.append(desc.strip())

        # Merge into one combined summary
        final_prompt = (
            "You will be given multiple segment-level descriptions of a video as well as a list of scenes for each segment. "
            "Your task is to combine them into one coherent story summary, then create a list of all scenes in the video with their timestamps. "
        )
        combined = await asyncio.to_thread(self.llm.LLM_request, ["\n\n".join(descriptions), final_prompt])
        return combined
