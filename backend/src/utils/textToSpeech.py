import edge_tts
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

ENGLISH_VOICES = {
    "male1": "en-US-GuyNeural",
    "male2": "en-GB-RyanNeural",
    "female1": "en-US-JennyNeural",
    "female2": "en-GB-SoniaNeural"
}


async def generate_voice(text: str, output_path: str, voice_type: str = "male1"):
    voice = ENGLISH_VOICES.get(voice_type, ENGLISH_VOICES["male1"])
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        logger.info(f"[INFO] Narration generated: {output_path}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to generate voice over: {e}")


async def movie_recap_generate_narration_for_clips(chosen_clips, voice_output_dir, voice_type="female1"):
    """
    Generates English TTS audio files for the `corresponding_summary_sentence` of each clip.
    """
    os.makedirs(voice_output_dir, exist_ok=True)
    for clip in chosen_clips:
        clip_id = clip["id"]
        sentence = clip["corresponding_summary_sentence"]
        output_path = os.path.join(voice_output_dir, f"{clip_id}.mp3")

        logger.info(f"[INFO] Generating voice for clip {clip_id}: {sentence}")
        await generate_voice(sentence, output_path, voice_type=voice_type)
