import asyncio
from pathlib import Path

class RoughComprehension:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, long_segments: list[dict]):
        long_segments = sorted(long_segments, key=lambda seg: seg["start"])
        segment_by_segment_draft = ""
        segment_num = 1
        compact_context = ""

        for seg in long_segments:
            # path = Path(seg["gcs_uri"])
            path = Path(seg["path"])

            print(f"[INFO] Processing segment {segment_num}: {path.name}")

            if segment_num == 1:
                prompt = (
                    "This is the first segment of a long-form video. All videos, regardless of genre, style, or purpose, contain a story—sometimes profound, sometimes simple or surface-level. "
                    "Your task is to create a comprehensive summary to help someone understand the story, sequence of events, or key message without watching it.\n\n"
                    "First, write a detailed summary based solely on this segment. Use the full content of the video, including the ending even if it's incomplete.\n"
                    "Next, list all identified people or characters, including names, appearance, and roles. If a name is unknown, refer to them by role, relationship, or distinguishing features.\n"
                    "Finally, provide a list of relationships or interactions observed in this segment.\n\n"
                    "Return plain text only. No preamble or extra commentary."
                )
                response_text = await asyncio.to_thread(self.llm.LLM_request, [path, prompt])
            else:
                prompt = (
                    "You are provided with the current story summary and character list from earlier segments, and a new video segment starting now.\n\n"
                    "Write a detailed summary based only on this new segment. Use the entire video, including the end even if it cuts off abruptly.\n"
                    "Do not repeat previous story points. Focus only on what is new and clearly visible in this segment.\n"
                    "Paraphrase dialogue, update the people/character list with any new details, roles, or individuals, and note significant interactions or actions.\n"
                    "If the segment only contains credits, filler, or unrelated footage, omit that from the description.\n\n"
                    "Note: The previously described events may not be in chronological order, as long-form videos may include flashbacks, time jumps, or non-linear presentation.\n"
                    "Return plain text only. No preamble or extra commentary."
                )
                response_text = await asyncio.to_thread(self.llm.LLM_request, [path, compact_context, prompt])

            segment_by_segment_draft += f"\n\n\n\n\nSegment {segment_num}:\n{response_text}"
            compact_context = await asyncio.to_thread(
                self.llm.LLM_request,
                [segment_by_segment_draft, (
                    "Below is a draft of the story or sequence of events across multiple video segments. Your task is to compress and consolidate it into a compact, logically coherent summary.\n\n"
                    "Keep all essential details needed to follow what happens. If multiple segments are included, summarize them as one continuous paragraph.\n"
                    "Also, extract and list all people or characters mentioned in the draft.\n"
                    "Note: Events may not be in chronological order, as long-form videos may include flashbacks, time jumps, or dream sequences.\n\n"
                    "Do not add formatting or headings. Return plain text only."
                )]
            )
            segment_num += 1

        combined_summary_draft = await asyncio.to_thread(self.llm.LLM_request, [
            segment_by_segment_draft,
            (
                "The text below is a draft of partial summaries from different video segments. Your task is to combine them into a complete, logically coherent summary of the video’s story, message, or progression.\n\n"
                "You may interpret and connect fragments to ensure the overall narrative makes sense. Remove duplicates, redundant phrasing, or irrelevant dialogue.\n"
                "Ensure names and references are consistent, and fix any illogical or broken sentences. Maintain natural storytelling flow.\n"
                "Note: The video may include flashbacks, scenes out of order, or non-narrative content. Use context to reconstruct a coherent description, but keep the segment order intact.\n\n"
                "At the end of each sentence, append (segment: N) to indicate which segment it came from. Do not reorder segments; maintain original segment order.\n"
                "Do not include end credits or non-relevant footage. Return as plain text only, with no headings or commentary."
            )
        ])

        characters = await asyncio.to_thread(self.llm.LLM_request, [
            combined_summary_draft,
            (
                "Using the following story summary and people/character mentions, create a clean and complete list of all individuals observed in the video.\n\n"
                "For each, include their name (if known), description, role or job (if clear), and any observed relationships or interactions with others.\n"
                "Group related individuals if appropriate. Return plain text only, no formatting or headings."
            )
        ])

        print("[INFO] Rough comprehension of long-form video complete.")
        return combined_summary_draft, characters
