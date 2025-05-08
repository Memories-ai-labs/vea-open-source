import os
import asyncio
import edge_tts

ENGLISH_VOICE = "en-US-GuyNeural"
CHINESE_VOICE = "zh-CN-XiaoxiaoNeural"


class GenerateNarrationForClips:
    def __init__(self, voice_output_dir, language = "English"):
        self.voice_output_dir = voice_output_dir
        self.language = language.lower()
        os.makedirs(self.voice_output_dir, exist_ok=True)

        if self.language == "chinese":
            self.voice = CHINESE_VOICE
        else:
            self.voice = ENGLISH_VOICE

    async def __call__(self, chosen_clips):
        tasks = []
        for clip in chosen_clips:
            clip_id = clip["id"]
            sentence = clip["corresponding_summary_sentence"]
            output_path = os.path.join(self.voice_output_dir, f"{clip_id}.mp3")

            if os.path.exists(output_path):
                print(f"[INFO] Skipping voice generation for clip {clip_id}, already exists.")
                continue

            print(f"[INFO] Generating voice for clip {clip_id}: {sentence}")
            tasks.append(self._generate_voice(sentence, output_path))

        await asyncio.gather(*tasks)
        print("[INFO] All voice clips generated.")

    async def _generate_voice(self, text, output_path):
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_path)
            print(f"[INFO] Narration generated: {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to generate voice over for {output_path}: {e}")
