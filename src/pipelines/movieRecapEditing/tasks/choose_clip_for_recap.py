import asyncio
import json
from src.pipelines.movieRecapEditing.schema import ChosenClip  # update path if needed

class ChooseClipForRecap:
    def __init__(self, llm, out_lang="English"):
        self.llm = llm
        self.out_lang = out_lang

    async def __call__(self, plot_json, scenes):
        segment_nums = sorted(set(entry["segment_num"] for entry in plot_json))
        all_chosen_clips = []

        for seg_num in segment_nums:
            print(f"[INFO] Choosing clips for segment {seg_num}...")

            segment_plot = [s for s in plot_json if s["segment_num"] == seg_num]
            segment_scenes = [s for s in scenes if s["segment_num"] == seg_num]

            select_prompt = (
                f"{json.dumps(segment_plot, ensure_ascii=False)}\n\n\n"
                f"{json.dumps(segment_scenes, ensure_ascii=False)}\n\n\n"
                "You are given a plot summary and a set of scene descriptions for the same segment of a long-form narrative media (e.g. movie, TV show, or documentary).\n"
                "Select one visual scene clip for each plot sentence to use in a recap video:\n"
                "- Each clip should be the best match for the corresponding sentence.\n"
                "- No clip should be used more than once.\n"
                "- Ignore clips that show only credits, logos, or irrelevant visuals.\n"
                "- Use only one clip per sentence.\n"
                f"Output in English except for character names, which should remain in the original language."
            )

            chosen_clips = await asyncio.to_thread(
                self.llm.LLM_request,
                [select_prompt],
                {
                    "response_mime_type": "application/json",
                    "response_schema": list[ChosenClip]
                }
            )

            refine_prompt = (
                f"{json.dumps(chosen_clips, ensure_ascii=False)}\n\n\n"
                "Above is a list of selected clips for a segment of a recap.\n"
                "Please revise this list to ensure that:\n"
                "- Each recap sentence appears only once (no duplicates).\n"
                "- If a clip is reused, merge the summary sentences if possible, or discard extras.\n"
                "- Remove any special characters that would be challenging for text-to-voice.\n"
                "- Discard any clips not relevant to the story (e.g., logos, credits).\n\n"
                f"Translate each recap sentence from English to {self.out_lang}, keeping character names in the original language. Output only the cleaned JSON list."
            )

            refined_clips = await asyncio.to_thread(
                self.llm.LLM_request,
                [refine_prompt],
                {
                    "response_mime_type": "application/json",
                    "response_schema": list[ChosenClip]
                }
            )

            seen_sentences = set()
            seen_clip_ids = set()
            filtered_clips = []

            for clip in refined_clips:
                sentence = clip.get("corresponding_summary_sentence", "").strip()
                clip_id = clip.get("id", "").strip()
                if sentence and clip_id and sentence not in seen_sentences and clip_id not in seen_clip_ids:
                    seen_sentences.add(sentence)
                    seen_clip_ids.add(clip_id)
                    filtered_clips.append(clip)

            print(f"[INFO] Segment {seg_num}: Filtered from {len(refined_clips)} to {len(filtered_clips)} clips.")
            all_chosen_clips.extend(filtered_clips)

        all_chosen_clips.sort(key=lambda c: int(c.get("id", 0)))
        print("[INFO] All segment clips chosen and cleaned successfully.")
        return all_chosen_clips
