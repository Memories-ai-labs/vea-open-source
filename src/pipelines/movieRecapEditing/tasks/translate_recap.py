import asyncio
from src.pipelines.movieRecapEditing.schema import ChosenClip
import json
class TranslateRecap:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, chosen_clips: list, target_language: str):
        if target_language.lower() == "english":
            return chosen_clips

        print(f"[INFO] Translating recap sentences to {target_language}...")
        translated = []

        batch_size = 20
        for i in range(0, len(chosen_clips), batch_size):
            batch = chosen_clips[i:i+batch_size]

            prompt = (
                f"{json.dumps(batch, ensure_ascii=False)}\n\n"
                "Above is a list of selected clips for a recap video.\n"
                f"Please translate only the `corresponding_summary_sentence` fields to {target_language}, "
                "preserving character names in their original language.\n"
                "Return valid JSON in the same format."
            )

            batch_translated = await asyncio.to_thread(
                self.llm.LLM_request,
                [prompt],
                {
                    "response_mime_type": "application/json",
                    "response_schema": list[ChosenClip]
                }
            )

            translated.extend(batch_translated)

        return translated

