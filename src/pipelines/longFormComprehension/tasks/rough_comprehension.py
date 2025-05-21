import asyncio

class RoughComprehension:
    def __init__(self, llm):
        """
        Args:
            llm: A manager for LLM calls (synchronous).
        """
        self.llm = llm

    async def __call__(self, long_segments):
        long_segments.sort()
        segment_by_segment_draft = ""
        segment_num = 1
        compact_context = ""

        for segment in long_segments:
            print(f"[INFO] Processing segment {segment_num}: {segment}")
            # print("compact context: ", compact_context)

            if segment_num == 1:
                prompt = (
                    "This is the first segment of a long-form storytelling video, such as a movie, TV show, or documentary. "
                    "Your task is to create a comprehensive summary to help someone understand the plot without watching it.\n\n"
                    "First, write a detailed plot summary based solely on this segment. Use the full content of the video, including the ending even if it's incomplete.\n"
                    "Next, list all identified characters, including their names, physical appearance, and roles. If a name is unknown, refer to them by role or relationship.\n"
                    "Finally, provide a list of relationships between characters based on this segment.\n\n"
                    "Return plain text only. No preamble or extra commentary."
                )
                response_text = await asyncio.to_thread(self.llm.LLM_request, [segment, prompt])
            else:
                prompt = (
                    "You are provided with the current plot summary and character list from earlier segments, and a new video segment starting now.\n\n"
                    "Write a detailed plot summary based only on this new segment. Use the entire video, including the end even if it cuts off abruptly.\n"
                    "Do not repeat previous plot points. Focus only on what is new and clearly visible in this segment.\n"
                    "Paraphrase dialogue and update the character list with any new details or characters.\n"
                    "If the segment only contains end credits or filler, omit that from the description.\n\n"
                    "Note: The previously described events may not be in chronological order, as the long-form storytelling video may include flashbacks or non-linear storytelling.\n"
                    "Return plain text only. No preamble or extra commentary."
                )
                response_text = await asyncio.to_thread(self.llm.LLM_request, [segment, compact_context, prompt])

            # print(response_text)
            
            segment_by_segment_draft += f"\n\n\n\n\nSegment {segment_num}:\n" + response_text

            summary_prompt = (
                "Below is a draft of the plot across multiple video segments. Your task is to compress and consolidate it into a compact, logically coherent summary.\n\n"
                "Keep all essential details needed to follow the story. If multiple segments are included, summarize them as one continuous paragraph.\n"
                "Also, extract and list all known characters from the draft.\n"
                "Note: Events in the draft may not be in chronological order, as the long-form storytelling video may include flashbacks, dreams, or shifts in time.\n\n"
                "Do not add formatting or headings. Return plain text only."
            )
            compact_context = await asyncio.to_thread(self.llm.LLM_request, [segment_by_segment_draft, summary_prompt])
            segment_num += 1

        plot_prompt = (
            "The text below is a draft of partial plot summaries from different video segments. Your task is to combine them into a complete, logically coherent plot summary.\n\n"
            "You may interpret and connect fragments to ensure the overall narrative makes sense. Remove duplicates, redundant phrasing, or irrelevant dialogue.\n"
            "Ensure character names are consistent, and fix any illogical or broken sentences. Maintain natural storytelling flow.\n"
            "Note: The plot may contain flashbacks or scenes presented out of chronological order. Use context to reconstruct a coherent storyline, but keep the segment order intact.\n\n"
            "At the end of each sentence, append (segment: N) to indicate which segment it came from. Do not reorder segments; maintain original segment order.\n"
            "Do not include end credits or non-narrative elements. Return the plot as plain text only, with no headings or commentary."
        )
        combined_summary_draft = await asyncio.to_thread(self.llm.LLM_request, [segment_by_segment_draft, plot_prompt])

        character_prompt = (
            "Using the following plot and character mentions, create a clean and complete character list.\n\n"
            "For each character, include their name, physical description, role or job, and any clearly established relationships with other characters.\n"
            "Group related characters if appropriate. Return plain text only, no formatting or headings."
        )
        characters = await asyncio.to_thread(self.llm.LLM_request, [combined_summary_draft, character_prompt])

        print("[INFO] Rough comprehension of long-form video complete.")
        return combined_summary_draft, characters
