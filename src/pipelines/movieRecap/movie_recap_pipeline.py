import os
import shutil
from pathlib import Path
import asyncio

from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from src.pipelines.movieRecap.tasks.rough_comprehension import RoughComprehension
from src.pipelines.movieRecap.tasks.scene_by_scene_comprehension import SceneBySceneComprehension
from src.pipelines.movieRecap.tasks.refine_plot_summary import RefinePlotSummary
from src.pipelines.movieRecap.tasks.choose_clip_for_recap import ChooseClipForRecap
from src.pipelines.movieRecap.tasks.generate_narration import GenerateNarrationForClips
from src.pipelines.movieRecap.tasks.music_selection import MusicSelection
from src.pipelines.movieRecap.tasks.edit_recap_video import EditMovieRecapVideo
from lib.utils.media import convert_video

class MovieRecapPipeline:
    def __init__(self, movie_dir, movie_name, output_dir, movie_language="English", output_language="English", SF=False):
        self.movie_dir = movie_dir
        self.work_dir = os.path.join(movie_dir, "workdir")
        self.movie_name = movie_name
        self.movie_path = os.path.join(movie_dir, movie_name)
        self.output_dir = output_dir
        self.long_segments_dir = os.path.join(self.work_dir, "long_segments")
        self.short_segments_dir = os.path.join(self.work_dir, "short_segments")
        self.text_to_voice_dir = os.path.join(self.work_dir, "text_to_voice")
        self.textual_representation_dir = os.path.join(self.work_dir, "textual_representation")
        self.SF = SF
        self.movie_language = movie_language
        self.output_language = output_language

        if SF and os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
            
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.long_segments_dir, exist_ok=True)
        os.makedirs(self.short_segments_dir, exist_ok=True)
        os.makedirs(self.textual_representation_dir, exist_ok=True)
        os.makedirs(self.text_to_voice_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.llm = GeminiGenaiManager()

    async def run(self):
        self.long_segments = await convert_video(self.movie_path, self.long_segments_dir, interval_seconds=15*60, fps=1, crf=28)
        self.short_segments = await convert_video(self.movie_path, self.short_segments_dir, interval_seconds=5*60, fps=1, crf=28)

        self.rough_comprehension = RoughComprehension(
            self.textual_representation_dir,
            self.movie_language,
            self.llm
        )
        combined_summary_draft, characters = await self.rough_comprehension(self.long_segments)

        self.scene_by_scene_comprehension = SceneBySceneComprehension(
            self.textual_representation_dir,
            self.movie_language,
            self.llm
        )
        scenes = await self.scene_by_scene_comprehension(self.short_segments, self.long_segments, combined_summary_draft, characters)

        self.refine_plot_summary = RefinePlotSummary(
            self.textual_representation_dir,
            self.movie_language,
            self.llm
        )
        plot_json, plot = await self.refine_plot_summary(combined_summary_draft, scenes)

        self.choose_clip_for_recap = ChooseClipForRecap(
            self.textual_representation_dir,
            self.movie_language,
            self.output_language,
            self.llm
        )
        chosen_clips = await self.choose_clip_for_recap(plot_json, scenes)

        self.generate_narration = GenerateNarrationForClips(
            self.text_to_voice_dir, self.output_language
        )
        await self.generate_narration(chosen_clips)

        self.music_selection = MusicSelection(
            self.llm
        )
        chosen_music_path = await self.music_selection(plot)

        self.videoedit = EditMovieRecapVideo(
            os.path.join(self.output_dir, "recap.mp4"),
            0.5
        )
        await self.videoedit(chosen_clips, self.movie_path, self.text_to_voice_dir, chosen_music_path)
        


if __name__ == "__main__":

    movie_folder = "data/space_between"
    movie_name = "spacebetween.mp4"
    output_dir = "data/space_between"
    movie_language = "English"
    output_language = "English"
    SF = False

    pipeline = MovieRecapPipeline(movie_folder, movie_name, output_dir, movie_language, output_language, SF)
    summary = asyncio.run(pipeline.run())
    print(summary)
