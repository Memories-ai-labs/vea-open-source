import os
import asyncio
from elevenlabs.client import ElevenLabs

class GenerateNarrationForClips:
    def __init__(self, voice_output_dir, language="English"):
        self.voice_output_dir = voice_output_dir
        self.language = language.lower()
        os.makedirs(self.voice_output_dir, exist_ok=True)

        self.elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    async def __call__(self, narrated_clips):
        for clip in narrated_clips:
            clip_id = clip['id']
            sentence = clip["corresponding_summary_sentence"]
            output_path = os.path.join(self.voice_output_dir, f"{clip_id}.mp3")

            if os.path.exists(output_path):
                print(f"[INFO] Skipping voice generation for clip {clip_id}, already exists.")
                continue

            print(f"[INFO] Generating voice for clip {clip_id}: {sentence}")
            self._generate_voice_sync(sentence, output_path)

        print("[INFO] All voice clips generated.")

    def _generate_voice_sync(self, text, output_path):
        try:
            audio = self.elevenlabs.text_to_speech.convert(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_flash_v2_5",
            )
            with open(output_path, "wb") as f:
                for chunk in audio:
                    if chunk:
                        f.write(chunk)
            print(f"[INFO] Narration generated: {output_path}")
            return output_path
        except Exception as e:
            print(f"[ERROR] Failed to generate voice for {output_path}: {e}")
