import json
import logging
from src.schema import Scene, ChosenClip
from src.utils.parse import timestamp_to_seconds, seconds_to_timestamp

# Set up logging
logger = logging.getLogger(__name__)

class MovieRecap:
    def __init__(self, llm):
        self.llm = llm

    async def generate_gist_and_characters(self, video_files):
        """
        Generates a textual synopsis, character descriptions, and relationships by sequentially prompting Gemini
        with sorted 20-minute video segments.
        """
        initial_prompt = (
            "here is the first segment of a movie. "
            "in the first section, based on the segment, describe the plot in detail. "
            "your plot summary should be compelling enough as a story."
            "Then in the next section, create a list of the characters, including name, appearances, and role/job. "
            "If a characters name is not known, replace it with their occupation or relationship to another character"
            "Then in the final section, create a list of character relationships."
            "return the answer in plain text with minimal formatting"
        )
        continuation_prompt = (
            "Provided is the next consecutive segment of the movie, and the textual description "
            "of the movie thus far. The textual description includes sections for the plot and character list. "
            "Using the movie segment and information of the movie provided, "
            "describe the plot in detail for this segment. also refine character descriptions and relationships. "
            "your plot description should be detailed and engaging, as if telling a story."
        )
        consolidation_prompt = (
            "provided is a draft of a textual description of a movie. the movie was split into segments, "
            " and the plot and character list were generated per segment. consolidate the segmented drafts "
            "into one story. keep the plot recap detailed, include the whole story arc form start to end."
            "the first section is the plot recap and the second section is a list of characters along with their descriptions"
        )
        segment_num = 1
        # Sort video files by display name
        video_files.sort(key=lambda f: f.display_name)

        # Initial prompt for the first 20-minute segment
        draft = ""

        # Process first segment
        logger.info(f"Processing first video segment: {video_files[0].display_name}")
        response_text = await self.llm.generate_response(
            contents=[video_files[0], initial_prompt]
        )

        draft = (
            f"Segment {segment_num}:\n" + response_text
        )  # Store the initial response
        # Process subsequent segments
        for video_file in video_files[1:]:
            segment_num += 1
            logger.info(f"Refining with video segment: {video_file.display_name}")
            response_text = await self.llm.generate_response(
                contents=[video_file, draft + "\n\n" + continuation_prompt]
            )
            draft += f"\n\n\n\n\nSegment {segment_num}:\n" + response_text

        response_text = await self.llm.generate_response(
            contents=[draft + "\n\n" + consolidation_prompt]
        )

        logger.info("Gist and character descriptions generated successfully.")
        return response_text

    async def fixed_interval_transcribe(self, video_files, plot_and_characters_text):
        try:
            video_files.sort(key=lambda f: f.display_name)
            scenes = []
            scene_id = 1

            for video_file in video_files:
                file_parts = video_file.display_name.split(".")[0].rsplit("_", 2)
                start_seconds = int(file_parts[1])

                logger.info(
                    f"Transcribing scene-by-scene for segment: {video_file.display_name}"
                )

                prompt = f"""{plot_and_characters_text}

                        You are provided with a segment of a movie, including a textual recap of its plot and a list of characters. Your task is to describe the scene in detail at 30-second intervals throughout the entire segment. For each interval, specify the characters involved and their actions, using the character names as presented in the plot recap. Ensure that the entire segment is covered. Ensuring the response does not exceed 800 tokens.

                        Format each scene description as a JSON object with the following keys:
                        - "start_timestamp": string, formatted as "HH:MM:SS"
                        - "end_timestamp": string, formatted as "HH:MM:SS"
                        - "description": string

                        Please output a JSON array containing these objects, each representing a 30-second scene description. 
                        """
                response = await self.llm.generate_response(
                    contents=[video_file, prompt],
                    params={
                        "response_mime_type": "application/json",
                        "response_schema": list[Scene],
                    },
                )

                scene_data = json.loads(response)

                for scene in scene_data:
                    scene_start_sec = timestamp_to_seconds(scene["start_timestamp"])
                    scene_end_sec = timestamp_to_seconds(scene["end_timestamp"])
                    scene["start_timestamp"] = seconds_to_timestamp(
                        start_seconds + scene_start_sec
                    )
                    scene["end_timestamp"] = seconds_to_timestamp(
                        start_seconds + scene_end_sec
                    )
                    scene["id"] = scene_id
                    scene_id += 1
                scenes.extend(scene_data)
            logger.info("Scenes transcribed successfully.")
            return scenes
        except Exception as e:
            logger.warning(f"Fixed interval transcribe failed: {e}")
            logger.warning(f"{response}")
            raise e

    async def choose_clip_for_recap(self, gist, scenes):
        prompt = (
            f"{gist}\n\n\n"
            f"{json.dumps(scenes)}\n\n\n"
            "provided is a textual summary of a movie's plot, including the story and the characters. "
            "the section of text following is a json of 30 minute clips the movies with metadata and scene descriptions. "
            "for each sentence in the plot summary, select a clip from the json that best represents it. "
            "both the summary and scenes are sorted in chronological order, so make sure chosen clips are also "
            "in ascending order, this is more important than scene description matching."
        )

        response = await self.llm.generate_response(
            contents=[prompt],
            params={
                "response_mime_type": "application/json",
                "response_schema": list[ChosenClip],
            },
        )

        chosen_clips = json.loads(response)

        logger.info("clips chosen successfully.")
        return chosen_clips
