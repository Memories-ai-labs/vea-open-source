import os
import json
import asyncio

from src.pipelines.movieRecap.schema import ChosenClip


class ChooseClipForRecap:
    def __init__(self, textual_repr_dir, lang, out_lang, llm):
        self.textual_repr_dir = textual_repr_dir
        self.lang = lang
        self.out_lang = out_lang
        self.llm = llm

    async def __call__(self, plot_json, scenes):
        chosen_clips_path = os.path.join(self.textual_repr_dir, "chosen_clips.json")
        os.makedirs(self.textual_repr_dir, exist_ok=True)

        if os.path.exists(chosen_clips_path):
            print("[INFO] Loading existing chosen clips...")
            with open(chosen_clips_path, "r", encoding="utf-8") as f:
                return json.load(f)

        segment_nums = sorted(set(entry["segment_num"] for entry in plot_json))
        all_chosen_clips = []

        for seg_num in segment_nums:
            print(f"[INFO] Choosing clips for segment {seg_num}...")

            segment_plot = [s for s in plot_json if s["segment_num"] == seg_num]
            segment_scenes = [s for s in scenes if s["segment_num"] == seg_num]

            select_prompt = (
                f"{json.dumps(segment_plot, ensure_ascii=False)}\n\n\n"
                f"{json.dumps(segment_scenes, ensure_ascii=False)}\n\n\n"
                "You are given a plot summary and a set of scene descriptions for the same movie segment.\n"
                "Select one visual scene clip for each plot sentence to use in a video recap:\n"
                "- Each clip should be the best match for the corresponding sentence.\n"
                "- No clip should be used more than once.\n"
                "- Ignore clips that show only credits, logos, or irrelevant visuals.\n"
                "- Use only one clip per sentence.\n"
                f"The movie is in {self.lang}. Output in English except for character names, which should remain in the original language."
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
                "Above is a list of selected clips for a segment of a movie recap.\n"
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

        with open(chosen_clips_path, "w", encoding="utf-8") as f:
            json.dump(all_chosen_clips, f, indent=4, ensure_ascii=False)

        print("[INFO] All segment clips chosen and cleaned successfully.")
        return all_chosen_clips
