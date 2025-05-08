import asyncio
from collections import defaultdict

from src.pipelines.longForm.schema import RecapSentence

class RefinePlotSummary:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, summary_draft, scenes):
        print("[INFO] Refining plot summary segment-by-segment...")

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
                f"You are given two descriptions for segment {segment_num} of a long-form storytelling video, "
                "such as a movie, TV show, or documentary:\n\n"
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
                "Output only the improved plot summary for segment as plain text."
            )

            segment_refined = await asyncio.to_thread(
                self.llm.LLM_request,
                [plot_block, scene_block, refinement_prompt]
            )
            refined_sentences.append(segment_refined.strip())

        full_refined_plot = "\n\n".join(refined_sentences)

        # Step 4: Convert final version back to JSON
        plot_json_final = await self._convert_summary_to_json(full_refined_plot)

        print("[INFO] Refined plot summary and JSON generated successfully.")
        return plot_json_final, full_refined_plot

    async def _convert_summary_to_json(self, text):
        convert_prompt = (
            "Convert the plot summary into structured JSON format. Each sentence should be stored as an entry with its corresponding 'segment_num'.\n\n"
            "Remove the (segment: N) marker from the sentence text and instead store the segment number as a separate field.\n"
            "Output should be in JSON format following the expected structure."
        )
        return await asyncio.to_thread(
            self.llm.LLM_request,
            [text, convert_prompt],
            {
                "response_mime_type": "application/json",
                "response_schema": list[RecapSentence]
            }
        )
