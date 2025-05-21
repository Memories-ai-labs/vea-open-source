import asyncio
from typing import List
from pathlib import Path

class ContentTranscription:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, segment_path: Path) -> str:
        """
        Ask Gemini to transcribe a video segment.
        The transcription should include:
        - Verbatim speech
        - Visible text from signs, screens, or blackboards
        - Timestamps for when content appears
        - Plain text format only
        """

        prompt = (
            "You are given a video segment. Please transcribe the following:\n\n"
            "- Verbatim spoken words by people in the video\n"
            "- Any readable text visible in the video (e.g. signs, posters, presentation slides, blackboards)\n\n"
            "For each piece of speech or visible text, include a timestamp of when it occurred in HH:MM:SS format.\n"
            "If a sentence spans multiple timestamps, start with the earliest one.\n"
            "If possible, interleave spoken and visual text in chronological order.\n\n"
            "Return plain text only in this format:\n\n"
            "HH:MM:SS - [transcribed sentence or visible text]\n"
            "HH:MM:SS - [another sentence]\n"
            "...\n\n"
            "Avoid summaries or formatting beyond this structure, but you may include the person's name or role if it is known.\n"
        )

        response_text = await asyncio.to_thread(
            self.llm.LLM_request,
            [segment_path, prompt],
        )

        return response_text.strip()
