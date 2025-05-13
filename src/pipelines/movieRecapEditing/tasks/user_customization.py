import asyncio
from collections import defaultdict
from src.pipelines.movieRecapEditing.schema import RecapSentence

class UserCustomization:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(
        self,
        plot_json,
        scenes,
        characters,
        artistic_elements,
        user_context=None,
        user_instruction=None,
    ):
        if user_context or user_instruction:
            print("[INFO] Customizing according to user preferences...")
            # Group by segment number
            grouped_sentences = defaultdict(list)
            grouped_scenes = defaultdict(list)

            for entry in plot_json:
                grouped_sentences[entry["segment_num"]].append(entry["sentence_text"])
            for scene in scenes:
                grouped_scenes[scene["segment_num"]].append(scene)

            segments_output = []
            for segment_num in sorted(grouped_sentences.keys()):
                plot_block = "\n".join(grouped_sentences[segment_num])
                scene_block = "\n".join(
                    f"{s['start_timestamp']}: {s['scene_description']}"
                    for s in grouped_scenes.get(segment_num, [])
                )
                artistic_block = "\n".join(
                    f"{a['start_timestamp']}: {a['artistic_description']}"
                    for a in artistic_elements if a.get("segment_num") == segment_num
                )

                if user_context or user_instruction:
                    # Run customization
                    custom_prompt = (
                        "You are an assistant enhancing the summary of a long-form narrative media (such as a movie, TV show, or documentary).\n\n"
                        "**User Context** provides helpful background information (e.g., time period, genre, known themes).\n"
                        "**User Instruction** specifies how to enhance the summary (e.g., emphasize visual style or symbolic meaning).\n"
                        "Example (not actual instructions): Context - 'Set in 1960s America'; Instruction - 'Highlight racial injustice themes.'\n\n"
                        f"User Context:\n{user_context or 'N/A'}\n\n"
                        f"User Instruction:\n{user_instruction or 'N/A'}\n\n"
                        f"Original Plot Summary for Segment {segment_num}:\n{plot_block}\n\n"
                        f"Scene Breakdown:\n{scene_block}\n\n"
                        f"Artistic Notes:\n{artistic_block}\n\n"
                        f"Characters:\n{characters}\n\n"
                        f"Revise and enrich the plot summary for **segment {segment_num}** according to the instructions above.\n"
                        "- Focus on enhancing only this segment.\n"
                        f"- Each sentence must end with (segment: {segment_num}).\n"
                        "- Do not invent or hallucinate details.\n"
                        "- Return only plain text with no headings or metadata."
                    )
                    segment_text = await asyncio.to_thread(self.llm.LLM_request, [custom_prompt])
                else:
                    # No customization: just rejoin and tag each sentence
                    segment_text = "\n".join(
                        s.strip() if s.strip().endswith(f"(segment: {segment_num})") else f"{s.strip()} (segment: {segment_num})"
                        for s in grouped_sentences[segment_num]
                    )

                segments_output.append(segment_text.strip())

            full_text = "\n\n".join(segments_output)

            # Convert to JSON using segment markers
            final_json = await self._convert_summary_to_json(full_text)

            print("[INFO] Customization/translation complete.")
            return final_json
        else:
            print("[INFO] No user context or instruction provided. Returning original plot JSON.")
            return plot_json

    async def _convert_summary_to_json(self, text):
        convert_prompt = (
            "Convert the plot summary into structured JSON format. "
            "Each sentence ends with (segment: N). Remove the marker from the sentence and use N as the 'segment_num' field.\n\n"
            "Return an array of objects with keys: 'sentence_text' and 'segment_num'.\n"
            "The final output should match the RecapSentence schema."
        )
        return await asyncio.to_thread(
            self.llm.LLM_request,
            [text, convert_prompt],
            {
                "response_mime_type": "application/json",
                "response_schema": list[RecapSentence]
            }
        )
