import asyncio
import os
import shutil
import logging
from src.llm import LLMService
from src.utils.media import split_video, trim_video_clip
from src.utils.crawler.youtube_crawler import download_youtube_video
from src.utils.parse import timestamp_to_seconds
from src.agents.movieRecap import MovieRecap
from src.utils.textToSpeech import movie_recap_generate_narration_for_clips
from src.utils.videoEditor import create_final_video
import tempfile
from src.utils.oss.gcp_oss import GoogleCloudStorage
from src.utils.oss.auth import credentials_from_file
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class VideoAgentPipeline:
    def __init__(
        self,
        movie_name: str = "",
        initial_prompt: str = "",
        long_interval_seconds: int = 1200,
        short_interval_seconds: int = 300,
        bgm_path: str = None,
        debug_dir: str = None,
    ):
        self.movie_name = movie_name
        self.initial_prompt = initial_prompt
        self.long_interval_seconds = long_interval_seconds
        self.short_interval_seconds = short_interval_seconds
        self.bgm_path = bgm_path
        # Initialize Gemini client
        self.genai_client = LLMService()
        self.movie_recap_agent = MovieRecap(self.genai_client)
        self.bucket = "openinterx-vea"
        
        # Fix the credential path to point to the correct location
        BASE_DIR = Path(__file__).resolve().parent.parent  # Go up one level from src to backend
        CREDENTIAL_PATH = BASE_DIR / "config" / "gen-lang-client-0057517563-0319d78ed5fe.json"
            
        self.oss_client = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))
        self.files_tracker = []

        self.setup_work_directory(debug_dir=debug_dir)

    def setup_work_directory(self, debug_dir: str = None):
        """Creates the work directory and video fragments folder."""
        if debug_dir:
            self.work_dir = debug_dir
        else:
            self.work_dir = tempfile.mkdtemp()
        self.long_fragments_dir = os.path.join(self.work_dir, "long_video_fragments")
        self.short_fragments_dir = os.path.join(self.work_dir, "short_video_fragments")
        self.textual_repr_dir = os.path.join(self.work_dir, "textual_repr")
        self.chosen_clips_dir = os.path.join(self.work_dir, "chosen_clips_videos")
        self.ttv_dir = os.path.join(self.work_dir, "text_to_voice")
        self.final_video_path = os.path.join(self.work_dir, "final_recap.mp4")

    async def preprocess_videos(self, video_path: str):
        """Splits videos into long and short fragments."""
        long_fragments = []
        short_fragments = []
        long_parts = split_video(
            video_path,
            interval_time=self.long_interval_seconds,
            output_path=self.long_fragments_dir,
        )
        long_fragments.extend(long_parts)
        # 5-minute split
        short_parts = split_video(
            video_path,
            interval_time=self.short_interval_seconds,
            output_path=self.short_fragments_dir,
        )
        short_fragments.extend(short_parts)

        logger.info(f"Long fragments created: {len(long_fragments)}")
        logger.info(f"Short fragments created: {len(short_fragments)}")

    async def retrieve_clips(self, video_path, chosen_clips):
        """
        Extracts specific video clips from the full movie based on chosen_clips info.

        Args:
            chosen_clips (list): List of dicts with 'id', 'start_timestamp', 'end_timestamp', etc.
        """
        # Extract each chosen clip
        for clip in chosen_clips:
            try:
                clip_id = clip["id"]
                start = timestamp_to_seconds(clip["start_timestamp"])
                end = timestamp_to_seconds(clip["end_timestamp"])
                logger.info(
                    f"Trimming clip ID {clip_id} from {clip['start_timestamp']} to {clip['end_timestamp']}"
                )
                output_clip_path = os.path.join(self.chosen_clips_dir, f"{clip_id}.mp4")
                trim_video_clip(video_path, start, end, output_clip_path)

            except Exception as e:
                logger.error(f"Failed to trim clip ID {clip.get('id', '?')}: {e}")

    async def run(self, youtube_url: str = None, video_path: str = None):
        """Runs the full movie recap pipeline."""
        logger.info("Starting Movie Recap Pipeline...")
        
        try:
            if youtube_url is not None:
                # Download YouTube video
                logger.info(f"Downloading video from YouTube: {youtube_url}")
                video_path = os.path.join(self.work_dir, "source_video.mp4")
                await download_youtube_video(youtube_url, video_path)            
            # Continue with existing pipeline...
            await self.preprocess_videos(video_path)
            
            # Step 2: Upload long fragments
            logger.info("Uploading long video fragments...")
            long_uploaded = [
                await self.genai_client.upload_file(
                    os.path.join(self.long_fragments_dir, fpath)
                )
                for fpath in os.listdir(self.long_fragments_dir)
            ]
            self.files_tracker.extend(long_uploaded)

            # Step 3: Upload short fragments
            logger.info("Uploading short video fragments...")
            short_uploaded = [
                await self.genai_client.upload_file(
                    os.path.join(self.short_fragments_dir, fpath)
                )
                for fpath in os.listdir(self.short_fragments_dir)
            ]
            self.files_tracker.extend(short_uploaded)

            # Step 4: Generate textual gist & character descriptions using long fragments
            logger.info("Generating gist and character descriptions...")
            plot_and_characters_text = (
                await self.movie_recap_agent.generate_gist_and_characters(
                    long_uploaded,
                )
            )

            # Step 5: Generate scene descriptions using short fragments
            scenes = await self.movie_recap_agent.fixed_interval_transcribe(
                short_uploaded,
                plot_and_characters_text,
            )

            # Step 6: Choose clips based on scenes
            chosen_clips = await self.movie_recap_agent.choose_clip_for_recap(
                plot_and_characters_text,
                scenes,
            )

            # Step 7: Retrieve actual clip videos
            await self.retrieve_clips(video_path, chosen_clips)

            # Step 8: Generate text-to-voice narration
            await movie_recap_generate_narration_for_clips(chosen_clips, self.ttv_dir)

            # create final video
            await create_final_video(
                clips=chosen_clips,
                work_dir=self.work_dir,
                chosen_videos_dir=self.chosen_clips_dir,
                narration_dir=self.ttv_dir,
                background_music_path=self.bgm_path,
                output_path=self.final_video_path,
            )
            self.oss_client.upload_from_file(self.final_video_path, self.bucket, self.movie_name + "/final_recap.mp4")
            logger.info("Movie Recap Pipeline Completed.")

            await self.genai_client.delete_all_files(self.files_tracker)
            shutil.rmtree(self.work_dir)
            logger.info("All files deleted.")
            return self.oss_client.get_public_download_url(self.bucket, self.movie_name + "/final_recap.mp4", expired_in_hour=24)

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            if not self.work_dir:
                shutil.rmtree(self.work_dir)
            await self.genai_client.delete_all_files(self.files_tracker)
            raise e


if __name__ == "__main__":
    try:
        video_path = "E:/OpenInterX Code Source/vea-playground/test_data/wusha2.mp4"
        youtube_url = "https://www.youtube.com/watch?v=D5hmCf_HIJg"
        debug_dir = "E:/OpenInterX Code Source/vea-playground/test_data/debug"
        va = VideoAgentPipeline(
            movie_name="test_video",
            bgm_path="E:/OpenInterX Code Source/vea-playground/test_data/Else - Paris.mp3",
            long_interval_seconds=20 * 60,
            short_interval_seconds=5 * 60,
            debug_dir=debug_dir,
        )
        asyncio.run(va.run(youtube_url=youtube_url, video_path=video_path))
    except Exception as e:
        logger.error(f"Work directory: {va.work_dir}")
        if not debug_dir or debug_dir == "":
            shutil.rmtree(va.work_dir)
        asyncio.run(va.genai_client.delete_all_files(va.files_tracker))
        raise e
