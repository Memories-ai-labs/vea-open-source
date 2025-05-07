import os
import json
import asyncio
from collections import defaultdict

from src.pipelines.movieRecap.schema import RecapSentence


class RefinePlotSummary:
    def __init__(self, textual_repr_dir: str, lang: str, llm):
        self.textual_repr_dir = textual_repr_dir
        self.lang = lang
        self.llm = llm

    async def __call__(self, summary_draft, scenes):
        print("[INFO] Refining plot summary segment-by-segment...")

        plot_json_path = os.path.join(self.textual_repr_dir, "plot.json")
        plot_txt_path = os.path.join(self.textual_repr_dir, "plot.txt")

        if os.path.exists(plot_json_path) and os.path.exists(plot_txt_path):
            print("[INFO] Loading existing refined plot and JSON...")
            with open(plot_txt_path, "r", encoding="utf-8") as f:
                full_refined_plot = f.read()
            with open(plot_json_path, "r", encoding="utf-8") as f:
                plot_json_final = json.load(f)
            return plot_json_final, full_refined_plot

        # Step 1: Convert to structured JSON
        plot_json = await self._convert_summary_to_json(summary_draft)

        # Step 2: Group by segment number
        grouped_sentences = defaultdict(list)
        grouped_scenes = defaultdict(list)

        for entry in plot_json:
            grouped_sentences[entry["segment_num"]].append(entry["sentence_text"])
        for scene in scenes:
            grouped_scenes[scene["segment_num"]].append(scene)

        # Step 3: Refine each segment
        refined_sentences = []
        for segment_num in sorted(grouped_sentences.keys()):
            plot_block = "\n".join(grouped_sentences[segment_num])
            scene_block = "\n".join(
                f"{scene['start_timestamp']}: {scene['scene_description']}"
                for scene in grouped_scenes.get(segment_num, [])
            )

            refinement_prompt = (
                f"You are given two descriptions for segment {segment_num} of a movie:\n\n"
                "1. A plot summary written from a long segment.\n"
                "2. A detailed scene-by-scene breakdown with timestamps.\n\n"
                "Your task is to revise and improve the plot summary for this segment.\n"
                "- Fix any logical inconsistencies.\n"
                "- Improve the storytelling flow and transitions.\n"
                "- Resolve unclear pronouns or character name mismatches.\n"
                "- You may rewrite or add new sentences if needed to make the plot clearer.\n"
                "- Treat the scene-by-scene breakdown as the more accurate source for factual events.\n"
                "- However, the refined summary should be much shorter than the scene-by-scene breakdown and closer in length to the summary draft.\n"
                "- Remove irrelevant or non-story content like studio names or credits.\n"
                f"Make sure each sentence ends with (segment: {segment_num}).\n\n"
                f"The movie is in {self.lang}. Output only the improved plot summary for segment {segment_num} as plain text."
            )

            segment_refined = await asyncio.to_thread(
                self.llm.LLM_request,
                [plot_block, scene_block, refinement_prompt]
            )
            refined_sentences.append(segment_refined.strip())

        full_refined_plot = "\n\n".join(refined_sentences)
        with open(plot_txt_path, "w", encoding="utf-8") as f:
            f.write(full_refined_plot)

        # Step 4: Convert final version back to JSON
        plot_json_final = await self._convert_summary_to_json(full_refined_plot)

        with open(plot_json_path, "w", encoding="utf-8") as f:
            json.dump(plot_json_final, f, ensure_ascii=False, indent=4)

        print("[INFO] Refined plot summary and JSON saved successfully.")
        return plot_json_final, full_refined_plot

    async def _convert_summary_to_json(self, text):
        """
        Converts a text-based summary into a list of RecapSentence-style dicts
        by calling the Gemini LLM with a structured JSON conversion prompt.
        """
        convert_prompt = (
            "Convert the plot summary into structured JSON format. Each sentence should be stored as an entry with its corresponding 'segment_num'.\n\n"
            "Remove the (segment: N) marker from the sentence text and instead store the segment number as a separate field.\n"
            f"The movie is in {self.lang}. Output in English, preserving original-language character names."
        )
        return await asyncio.to_thread(
            self.llm.LLM_request,
            [text, convert_prompt],
            {
                "response_mime_type": "application/json",
                "response_schema": list[RecapSentence]
            }
        )
