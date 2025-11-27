import asyncio
import json
from src.pipelines.flexibleResponse.schema import EvidenceClips
from src.pipelines.common.schema import convert_timestamp_to_string


class EvidenceRetrieval:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(
        self,
        initial_response: str,
        indexing_data: list,  # This is now the list of media_files dicts
        file_descriptions: dict  # manifest
    ):
        # Build a dict mapping file_name to cloud_storage_path
        file_name_to_gcs = {
            entry['name']: entry.get('cloud_storage_path', None)
            for entry in indexing_data
        }

        # Prepare file descriptions (for LLM context)
        context_description = "\n".join(
            f"- {fname}: {desc}" for fname, desc in file_descriptions.items()
        )

        # Dump all content for context
        content_dump = json.dumps(indexing_data, ensure_ascii=False, indent=4)

        prompt = (
            "You are given an assistant's long-form text response to a user's question about one or more videos (movie, TV, or multiple clips).\n"
            "Here is the response:\n"
            f"---\n{initial_response.strip()}\n---\n\n"
            "You have been provided the following indexing files for one or more media files:\n"
            f"{context_description}\n\n"
            "Below are the contents of the files (each section is for a particular file and index type):\n"
            f"{content_dump}\n\n"
            "Your task is to select relevant scenes that could serve as **visual evidence** for points made in the response. you may choose up to 5 clips as evidence, so choose the most important ones. \n"
            "For each evidence clip you select, always include:\n"
            "- `file_name`: the name of the source video file for this clip (e.g., mymovie.mp4, travel1.mov, etc)\n"
            "- `start`, `end`: timestamps as objects with separate numeric fields:\n"
            "  {\"hours\": <0-23>, \"minutes\": <0-59>, \"seconds\": <0-59>, \"milliseconds\": <0-999>}\n"
            "  Examples:\n"
            "  - 2 minutes 5 seconds = {\"hours\": 0, \"minutes\": 2, \"seconds\": 5, \"milliseconds\": 0}\n"
            "  - 1 hour 15 minutes = {\"hours\": 1, \"minutes\": 15, \"seconds\": 0, \"milliseconds\": 0}\n"
            "  - 45 seconds = {\"hours\": 0, \"minutes\": 0, \"seconds\": 45, \"milliseconds\": 0}\n"
            "- `description`: a short description of the visual content\n"
            "- `reason`: an explanation of why this scene supports the assistant's response\n\n"
            "Do not include any extra text or explanation. Only output a valid JSON list."
        )

        # Call LLM as before
        clips = await asyncio.to_thread(
            self.llm.LLM_request,
            [prompt],
            EvidenceClips,
            60,  # retry_delay
            3,   # max_retries
            "evidence_retrieval"  # context
        )

        # Post-process: Convert timestamps and attach cloud_storage_path
        for clip in clips:
            # Convert structured timestamps to strings
            if "start" in clip:
                clip["start"] = convert_timestamp_to_string(clip["start"])
            if "end" in clip:
                clip["end"] = convert_timestamp_to_string(clip["end"])
            # Attach cloud storage path
            fname = clip.get("file_name")
            clip["cloud_storage_path"] = file_name_to_gcs.get(fname)

        return clips
