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
from src.pipelines.utils import generate_response_for_video
from src.pipelines.movieRecap.schema import ClipSummary

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
    2. Main Plot Summary (~800 words, clearly structured: beginning, development, climax, ending)
    3. Key Characters (~50 words per character)
    4. Major Conflicts and Climaxes (~150 words)
    5. Themes and Emotional Arcs (~100 words)

    Notes:
    - Maintain overall narrative coherence and logical consistency.
    - Ensure that any foreshadowing, plot twists, or callbacks are properly reflected in the summary.
    - Use a professional, concise, and clear writing style.
    """

class SummaryPipeline:
    """
    SummaryPipeline is an asynchronous video summarization pipeline that processes a video into short clips,
    generates a brief summary for each clip using an LLM (Gemini), and compiles the segment summaries into
    a cohesive plot summary.
    """

    def __init__(self, initial_prompt: str = "", interval_seconds: int = 900, debug_dir=None):
        self.initial_prompt = initial_prompt
        self.interval_seconds = interval_seconds
        self.debug_dir = debug_dir

        logger.info("Initializing SummaryPipeline...")
        self._setup_working_directories()
        self._setup_storage_client()
        self.llm = Gemini()
        logger.info("SummaryPipeline initialized successfully.")

    def _setup_working_directories(self):
        self.temp_dir = self.debug_dir or tempfile.mkdtemp()
        logger.info(f"Temporary working directory set up at: {self.temp_dir}")

    def _setup_storage_client(self):
        self.oss_client = GoogleCloudStorage(
            credentials=credentials_from_file(CREDENTIAL_PATH)
        )
        logger.info("GCP OSS client initialized.")

    async def _download_video_if_needed(self, gcs_path: str, video_path: str) -> str:
        logger.info(f"Preparing to download video from GCS: {gcs_path}")
        if gcs_path:
            local_video_path = os.path.join(self.temp_dir, os.path.basename(gcs_path))
            await self.oss_client.download_to_file_with_progress(
                BUCKET_NAME, gcs_path, local_video_path
            )
            logger.info(f"Video downloaded to local path: {local_video_path}")
            return local_video_path
        elif video_path:
            resolved_path = str(Path(video_path).resolve())
            logger.info(f"Using provided local video path: {resolved_path}")
            return resolved_path
        else:
            logger.error("Neither GCS path nor local video path provided.")
            raise ValueError("Either gcs_path or video_path must be provided.")

    async def _get_clips(self, video_path: str):
        logger.info(f"Splitting video into clips: {video_path}")
        clip_paths = await convert_video(
            video_path, self.temp_dir, self.interval_seconds, SUMMARY_FPS, SUMMARY_CRF
        )
        logger.info(f"{len(clip_paths)} clips generated.")
        return clip_paths

    async def _generate_gist(self, video_paths: list[str]):
        logger.info("Generating summaries for each clip...")
        clips = []
        for i, video_path in enumerate(video_paths):
            logger.info(f"Processing clip {i + 1}/{len(video_paths)}: {video_path}")
            response = await generate_response_for_video(
                self.llm,
                clip_prompt,
                video_path,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ClipSummary,
                },
            )
            logger.info(f"Summary for clip {i + 1} generated.")
            clips.append(response.text)

        logger.info("Generating integrated full movie summary...")
        res = await self.llm.generate_async(
            integration_prompt.format(all_segment_summaries_json="\n".join(clips)),
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": ClipSummary,
            },
        )
        logger.info("Integrated summary successfully generated.")
        return res.text

    async def run(self, gcs_path: str = None) -> str:
        logger.info("Starting full summarization pipeline...")
        try:
            # Step 1: Download video
            local_video = await self._download_video_if_needed(
                gcs_path, os.path.join(self.temp_dir, os.path.basename(gcs_path))
            )

            # Step 2: Split video into clips
            clip_paths = await self._get_clips(local_video)

            # Step 3: Generate story gist
            plot_summary = await self._generate_gist(clip_paths)

            logger.info("Summarization pipeline completed successfully.")
            return plot_summary
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            await self._cleanup()

    async def _cleanup(self):
        if not self.debug_dir and self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Temporary working directory cleaned up.")

if __name__ == "__main__":
    import asyncio

    video_path = "E:/OpenInterX-Code-Source/vea-playground/test/爱在日落黄昏时.mkv"  # Example path
    output_path = "E:/OpenInterX-Code-Source/vea-playground/test/output"

    s = SummaryPipeline(debug_dir=output_path)
    summary = asyncio.run(s.run(video_path))
    print(summary)
