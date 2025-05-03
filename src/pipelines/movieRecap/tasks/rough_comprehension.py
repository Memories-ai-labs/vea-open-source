import os
import asyncio

class RoughComprehension:
    def __init__(self, textual_repr_dir: str, lang: str, llm):
        """
        Args:
            textual_repr_dir (str): Directory where outputs will be saved.
            lang (str): The original language of the movie.
            llm: A manager for LLM calls (synchronous).
        """
        self.textual_repr_dir = textual_repr_dir
        self.lang = lang
        self.llm = llm

        os.makedirs(self.textual_repr_dir, exist_ok=True)
        self.character_file_path = os.path.join(self.textual_repr_dir, "characters.txt")
        self.segment_by_segment_draft_path = os.path.join(self.textual_repr_dir, "segment_by_segment_draft.txt")
        self.combined_summary_draft_path = os.path.join(self.textual_repr_dir, "combined_summary_draft.txt")

    async def __call__(self, long_segments):
        if os.path.exists(self.combined_summary_draft_path):
            print("[INFO] Loading existing rough_comprehension file...")
            with open(self.combined_summary_draft_path, "r", encoding="utf-8") as f:
                combined_summary_draft = f.read()
            with open(self.character_file_path, "r", encoding="utf-8") as f:
                characters = f.read()
            return combined_summary_draft, characters

        long_segments.sort()
        segment_by_segment_draft = ""
        segment_num = 1
        compact_context = ""

        for segment in long_segments:
            print(f"[INFO] Processing segment {segment_num}: {segment}")
            print("compact context: ", compact_context)

            if segment_num == 1:
                prompt = (
                    "This is the first segment of a movie. Your task is to create a comprehensive summary to help someone understand the plot without watching it.\n\n"
                    "First, write a detailed plot summary based solely on this segment. Use the full content of the video, including the ending even if it's incomplete.\n"
                    "Next, list all identified characters, including their names, physical appearance, and roles. If a name is unknown, refer to them by role or relationship.\n"
                    "Finally, provide a list of relationships between characters based on this segment.\n\n"
                    "Return plain text only. No preamble or extra commentary.\n"
                    f"The movie is in {self.lang}. Output in English, but preserve all original-language character names."
                )
                response_text = await asyncio.to_thread(self.llm.LLM_request, [segment, prompt])
            else:
                prompt = (
                    "You are provided with the current plot summary and character list from earlier segments, and a new video segment starting now.\n\n"
                    "Write a detailed plot summary based only on this new segment. Use the entire video, including the end even if it cuts off abruptly.\n"
                    "Do not repeat previous plot points. Focus only on what is new and clearly visible in this segment.\n"
                    "Paraphrase dialogue and update the character list with any new details or characters.\n"
                    "If the segment only contains end credits or filler, omit that from the description.\n\n"
                    "Note: The previously described events may not be in chronological order, as the movie may include flashbacks or non-linear storytelling.\n"
                    "Return plain text only. No preamble or extra commentary.\n"
                    f"The movie is in {self.lang}. Output in English, preserving original-language character names."
                )
                response_text = await asyncio.to_thread(self.llm.LLM_request, [segment, compact_context, prompt])

            segment_by_segment_draft += f"\n\n\n\n\nSegment {segment_num}:\n" + response_text

            with open(self.segment_by_segment_draft_path, "w", encoding="utf-8") as f:
                f.write(segment_by_segment_draft)

            summary_prompt = (
                "Below is a draft of the plot across multiple movie segments. Your task is to compress and consolidate it into a compact, logically coherent summary.\n\n"
                "Keep all essential details needed to follow the story. If multiple segments are included, summarize them as one continuous paragraph.\n"
                "Also, extract and list all known characters from the draft.\n"
                "Note: Events in the draft may not be in chronological order, as the movie may include flashbacks, dreams, or shifts in time.\n\n"
                "Do not add formatting or headings. Return plain text only.\n"
                f"The movie is in {self.lang}. Output in English, preserving original-language character names."
            )
            compact_context = await asyncio.to_thread(self.llm.LLM_request, [segment_by_segment_draft, summary_prompt])
            segment_num += 1

        plot_prompt = (
            "The text below is a draft of partial plot summaries from different movie segments. Your task is to combine them into a complete, logically coherent plot summary.\n\n"
            "You may interpret and connect fragments to ensure the overall narrative makes sense. Remove duplicates, redundant phrasing, or irrelevant dialogue.\n"
            "Ensure character names are consistent, and fix any illogical or broken sentences. Maintain natural storytelling flow.\n"
            "Note: The plot may contain flashbacks or scenes presented out of chronological order. Use context to reconstruct a coherent storyline, but keep the segment order intact.\n\n"
            "At the end of each sentence, append (segment: N) to indicate which segment it came from. Do not reorder segments; maintain original segment order.\n"
            "Do not include end credits or non-narrative elements. Return the plot as plain text only, with no headings or commentary.\n"
            f"The movie is in {self.lang}. Output in English, preserving original-language character names."
        )
        combined_summary_draft = await asyncio.to_thread(self.llm.LLM_request, [segment_by_segment_draft, plot_prompt])

        with open(self.combined_summary_draft_path, "w", encoding="utf-8") as f:
            f.write(combined_summary_draft)

        character_prompt = (
            "Using the following plot and character mentions, create a clean and complete character list.\n\n"
            "For each character, include their name, physical description, role or job, and any clearly established relationships with other characters.\n"
            "Group related characters if appropriate. Return plain text only, no formatting or headings.\n"
            f"The movie is in {self.lang}. Output in English, preserving original-language character names."
        )
        characters = await asyncio.to_thread(self.llm.LLM_request, [combined_summary_draft, character_prompt])

        with open(self.character_file_path, "w", encoding="utf-8") as f:
            f.write(characters)

        print("[INFO] Rough movie comprehension complete.")
        return combined_summary_draft, characters
