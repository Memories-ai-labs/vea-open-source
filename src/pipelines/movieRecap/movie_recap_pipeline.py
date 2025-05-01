import os
import shutil
from pathlib import Path

class MovieRecapPipeline:
    def __init__(self, movie_dir, movie_name, movie_language="English", output_language="English", SF=False):
        self.movie_dir = movie_dir
        self.work_dir = os.path.join(movie_dir, "workdir")
        self.movie_path = os.path.join(movie_dir, movie_name)
        self.long_segments_dir = os.path.join(self.work_dir, "long_segments")
        self.short_segments_dir = os.path.join(self.work_dir, "short_segments")
        self.text_to_voice_dir = os.path.join(self.work_dir, "text_to_voice")
        self.textual_representation_dir = os.path.join(self.work_dir, "textual_representation")
        self.SF = SF
        self.movie_language = movie_language
        self.output_language = output_language

        if SF and os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)




if __name__ == "__main__":
    movie_folder = "data/space_between"
    movie_name = "spacebetween.mp4" ## -- replace it with your movie name
    output_path = "data/output" ## -- replace it with your debug dir path
    pipeline = MovieRecapPipeline()
    summary = pipeline.run()
    print(summary)
