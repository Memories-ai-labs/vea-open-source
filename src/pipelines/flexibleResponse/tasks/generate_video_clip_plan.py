from src.pipelines.flexibleResponse.schema import NarratedClip
import asyncio
import json

class GenerateVideoClipPlan:
    def __init__(self, llm):
        self.llm = llm

    def _build_file_to_cloud_path(self, indexing_data):
        return {
            entry["name"]: entry["cloud_storage_path"]
            for entry in indexing_data
            if "name" in entry and "cloud_storage_path" in entry
        }

    async def __call__(self, narration_script: str, user_prompt: str, indexing_data: list, file_descriptions: dict):
        # Format descriptions of each file (include media file name as part of key)
        context_description = "\n".join(
            f"- {fname}: {desc}" for fname, desc in file_descriptions.items()
        )
        content_dump = json.dumps(indexing_data, ensure_ascii=False, indent=4)
        prompt = (
            "You are assisting in building a video using clips from a movie or a collection of video files to support a narration script.\n\n"
            "For each clip, set a `priority` field (choose only one):\n"
            "- `narration`: Use text-to-speech narration as primary audio. This is best when narration is more important than original audio, "
            "for example, summarizing story scenes or when visual context is enough and the narration is primary.\n"
            "- `clip_audio`: Use the original audio from the video clip as primary. Choose this if the meaning depends on hearing specific words, interviews, quotes, or presentations from people in the clip (e.g., documentary quotes, news soundbites, speeches, interview moments, presentations).\n"
            "- `clip_video`: Use the entire video segment, uncut, including all original audio, even if it is longer than the narration. Use this for clips where it's crucial not to miss any action, such as sports plays or complex sequences where timing is critical.\n\n"
            "For each clip, choose one `priority` and set the field accordingly. Only set `clip_audio` or `clip_video` if absolutely necessary; prefer `narration` otherwise.\n\n"
            f"User prompt:\n---\n{user_prompt.strip()}\n---\n\n"
            f"Narration script:\n---\n{narration_script.strip()}\n---\n\n"
            "The following indexing files have been provided:\n"
            f"{context_description}\n\n"
            "Below are the contents of the files (media indexing JSON, including metadata and scene breakdowns):\n"
            f"{content_dump}\n\n"
            "For each output clip, include these fields:\n"
            "- `id`: unique integer\n"
            "- `file_name`: video file name\n"
            "- `start`, `end`: timestamps\n"
            "- `narration`: the narration line for the clip\n"
            "- `priority`: one of `narration`, `clip_audio`, or `clip_video`\n"
            "IMPORTANT: Do not include timestamps in the narration sentence, as this ruins the text to speech narration.\n"
            "IMPORTANT: Do not include file names with the extension in the narration sentence, as this ruins the text to speech narration.\n"
            "Respond with a JSON list, sorted in the order clips should appear in the video.\n"
        )

        # Gemini pass 1: Get initial clips with narration
        clips = await asyncio.to_thread(
            self.llm.LLM_request,
            [prompt],
            {
                "response_mime_type": "application/json",
                "response_schema": list[NarratedClip]
            }
        )

        # Gemini pass 2: Clean up narration for TTS
        clean_prompt = (
            "You are a narration script editor for text-to-speech. "
            "You are given a JSON list of video clips, each with narration. "
            "For each narration sentence, STRICTLY ensure:\n"
            "- No timestamps\n"
            "- No file names or video file extensions\n"
            "- No odd/unpronounceable characters\n"
            "- The narration is smooth and natural to read aloud\n"
            "Your task is to rewrite ONLY the narration fields as needed to meet these requirements, "
            "but do not change the meaning or the alignment with the associated video clips."
            "Edit the `narration` field as follows:\n"
            "- If `priority` is `narration`, ensure the narration is smooth, natural, and has no timestamps, file names, or odd characters.\n"
            "- If `priority` is `clip_audio`, you may shorten the narration and use a very short summary of original audio's content to avoid overlap with the original audio.\n"
            "Return the updated JSON with all fields preserved except the edited narration."
        )

        cleaned_clips = await asyncio.to_thread(
            self.llm.LLM_request,
            [json.dumps(clips, ensure_ascii=False, indent=2), clean_prompt],
            {
                "response_mime_type": "application/json",
                "response_schema": list[NarratedClip]
            }
        )

        file_to_cloud_path = self._build_file_to_cloud_path(indexing_data)
        for clip in cleaned_clips:
            fname = clip.get("file_name")
            clip["cloud_storage_path"] = file_to_cloud_path.get(fname, "")
            if not clip["cloud_storage_path"]:
                print(f"[WARN] No cloud_storage_path found for file: {fname}")

        return cleaned_clips
