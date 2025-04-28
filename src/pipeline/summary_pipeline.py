import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from src.config import CREDENTIAL_PATH, BUCKET_NAME, SUMMARY_FPS, SUMMARY_CRF
from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from lib.utils.media import convert_video
from lib.llm.gemini import Gemini
from src.pipeline.utils import generate_response_for_video
from src.pipeline.schema import ClipSummary

logger = logging.getLogger(__name__)


clip_prompt = """
You are a professional screenwriting assistant tasked with analyzing a complete movie, segment by segment.

Please carefully review the content of the following movie clip and complete the following tasks:
- Summarize the main plot points of this clip in no more than 200 words.
- List the key characters introduced or involved in this clip (include name and a brief description).
- Identify critical events that occur in this clip (one sentence per event).
- Highlight any foreshadowing, major conflicts, plot twists, or emotional shifts, if present.

Please output in the following structured JSON format:
{
  "segment_summary": "...",
  "characters": [{"name": "...", "description": "..."}, ...],
  "key_events": ["...", "...", "..."],
  "notes": ["...(foreshadowing/conflict/plot twist/emotion shift)", ...]
}
"""

integration_prompt = """
You are a senior screenwriter tasked with creating a complete movie outline based on segmented summaries.

Please carefully read the following extracted summaries from all movie clips, and compile a full movie outline including the following sections:

- Story Setting (briefly introduce the world, time, and background)
- Main Plot Summary (chronologically narrate the story, dividing into beginning, development, climax, and ending)
- Key Characters (describe their personality, relationships, and any character arcs)
- Major Conflicts and Climaxes (summarize key dramatic moments)
- Themes and Emotional Arcs (summarize the underlying themes and emotional progression)

【Extracted Clip Summaries】:
<<<
{all_segment_summaries_json}
>>>

Output format requirements:
1. Story Setting (~100 words)
2. Main Plot Summary (500–800 words, clearly structured: beginning, development, climax, ending)
3. Key Characters (50–100 words per character)
4. Major Conflicts and Climaxes (~150 words)
5. Themes and Emotional Arcs (~100 words)

Notes:
- Maintain overall narrative coherence and logical consistency.
- Ensure that any foreshadowing, plot twists, or callbacks are properly reflected in the summary.
- Use a professional, concise, and clear writing style.
"""


class SummaryPipeline:
    def __init__(
        self, initial_prompt: str = "", interval_seconds: int = 1200, debug_dir=None
    ):
        self.initial_prompt = initial_prompt
        self.interval_seconds = interval_seconds
        self.debug_dir = debug_dir

        self._setup_working_directories()
        self._setup_storage_client()
        self.llm = Gemini()

    def _setup_working_directories(self):
        self.temp_dir = self.debug_dir or tempfile.mkdtemp()

    def _setup_storage_client(self):
        self.oss_client = GoogleCloudStorage(
            credentials=credentials_from_file(CREDENTIAL_PATH)
        )

    async def _download_video_if_needed(self, gcs_path: str, video_path: str) -> str:
        if gcs_path:
            local_video_path = os.path.join(self.temp_dir, os.path.basename(gcs_path))
            await self.oss_client.download_to_file_with_progress(
                BUCKET_NAME, gcs_path, local_video_path
            )
            return local_video_path
        elif video_path:
            return str(Path(video_path).resolve())
        else:
            raise ValueError("Either gcs_path or video_path must be provided.")

    async def _get_clips(self, video_path: str):
        clip_paths = await convert_video(
            video_path, self.temp_dir, self.interval_seconds, SUMMARY_FPS, SUMMARY_CRF
        )
        return clip_paths

    async def _generate_gist(self, video_paths: list[str]):
        clips = []
        for video_path in video_paths:
            response = await generate_response_for_video(
                self.llm,
                clip_prompt,
                video_path,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ClipSummary,
                },
            )
            clips.append(response.text)
        res = await self.llm.generate_async(
            integration_prompt.format(all_segment_summaries_json="\n".join(clips)),
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": ClipSummary,
            },
        )
        return res

    async def run(self, gcs_path: str = None) -> str:
        logger.info("Starting Movie Recap Pipeline")
        try:
            # Step 1: Prepare input video
            # local_video = await self._download_video_if_needed(gcs_path, video_path)
            local_video = gcs_path

            # Step 2: Split into fragments
            clip_paths = await self._get_clips(local_video)

            # Step 4: Generate story gist
            plot_summary = await self._generate_gist(clip_paths)
            return plot_summary
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise e
        finally:
            await self._cleanup()

    async def _cleanup(self):
        if not self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        logger.info("Cleaned up temporary files.")



if __name__ == "__main__":
    import asyncio
    video_path = "E:/OpenInterX-Code-Source/mavi-edit-service/test/爱在日落黄昏时.mkv" ## -- replace it with your movie path
    output_path = "E:/OpenInterX-Code-Source/mavi-edit-service/test/output" ## -- replace it with your debug dir path
    s = SummaryPipeline(debug_dir=output_path)
    summary = asyncio.run(s.run(video_path))
    print(summary)
