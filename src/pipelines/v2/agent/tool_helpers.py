"""Standalone helper functions for narration (TTS), STT, and music (Soundstripe)."""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def tts_sync(text: str, output_path: str, api_key: str) -> None:
    """Blocking ElevenLabs TTS call (audio only, no timestamps)."""
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=api_key)
    audio = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_flash_v2_5",
    )
    with open(output_path, "wb") as f:
        for chunk in audio:
            if chunk:
                f.write(chunk)


def tts_with_timestamps(text: str, output_path: str, api_key: str) -> List[Dict]:
    """ElevenLabs TTS that also returns word-level timestamps.

    Calls convert_with_timestamps which returns:
      - audio_base_64: the rendered audio
      - alignment.characters: per-character list
      - alignment.character_start_times_seconds: parallel start times
      - alignment.character_end_times_seconds: parallel end times

    We write the audio to output_path and walk through the characters to derive
    word boundaries. Whitespace separates words; punctuation is attached to the
    preceding word for sentence detection.

    Returns a list of word dicts:
      [{"text": "Hello,", "start": 0.18, "end": 0.42, "is_sentence_end": False}, ...]
    """
    import base64
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=api_key)
    result = client.text_to_speech.convert_with_timestamps(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        text=text,
        model_id="eleven_flash_v2_5",
    )
    # Decode and write the audio
    audio_bytes = base64.b64decode(result.audio_base_64)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    # Extract word boundaries from the character alignment
    align = result.alignment or result.normalized_alignment
    if not align:
        logger.warning("[TTS] No alignment data returned from convert_with_timestamps")
        return []
    chars = align.characters
    starts = align.character_start_times_seconds
    ends = align.character_end_times_seconds
    if not (len(chars) == len(starts) == len(ends)):
        logger.warning(
            f"[TTS] Alignment array length mismatch: chars={len(chars)} "
            f"starts={len(starts)} ends={len(ends)}"
        )
        return []

    words: List[Dict] = []
    cur_text = ""
    cur_start: float = 0.0
    cur_end: float = 0.0
    SENTENCE_END_CHARS = {".", "?", "!"}

    def _flush():
        nonlocal cur_text
        if cur_text.strip():
            words.append({
                "text": cur_text,
                "start": round(cur_start, 3),
                "end": round(cur_end, 3),
                "is_sentence_end": bool(cur_text and cur_text.rstrip()[-1:] in SENTENCE_END_CHARS),
            })
        cur_text = ""

    for ch, s, e in zip(chars, starts, ends):
        if ch.isspace():
            _flush()
            continue
        if not cur_text:
            cur_start = float(s)
        cur_text += ch
        cur_end = float(e)
    _flush()

    return words


def stt_word_timestamps(audio_path: str, api_key: str) -> List[Dict]:
    """Transcribe audio with word-level timestamps via ElevenLabs Scribe.

    Returns a list of word dicts: [{"text": "Hello", "start": 0.12, "end": 0.45, "type": "word"}, ...]
    """
    from io import BytesIO
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=api_key)
    with open(audio_path, "rb") as f:
        audio_data = BytesIO(f.read())
    result = client.speech_to_text.convert(
        file=audio_data,
        model_id="scribe_v1",
        tag_audio_events=True,
        diarize=True,
    )
    words = []
    for w in getattr(result, "words", []) or []:
        if isinstance(w, dict):
            text = w.get("text", "")
            start = w.get("start", 0)
            end = w.get("end", 0)
            wtype = w.get("type", "word")
        else:
            text = getattr(w, "text", "")
            start = getattr(w, "start", 0)
            end = getattr(w, "end", 0)
            wtype = getattr(w, "type", "word")
        if text.strip():
            words.append({
                "text": text.strip(),
                "start": float(start),
                "end": float(end),
                "type": str(wtype),
            })
    return words


def generate_music_track(
    api_key: str,
    prompt: str,
    output_path: str,
    duration_ms: int = 120_000,
) -> dict:
    """Generate a music track using ElevenLabs Eleven Music API.

    Args:
        api_key: ElevenLabs API key.
        prompt: Natural language description of desired music
            (mood, genre, tempo, instruments, energy).
        output_path: Where to save the MP3.
        duration_ms: Desired track length in milliseconds (3000–300000).

    Returns:
        dict with ``success`` bool, ``path``, and ``error`` on failure.
    """
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=api_key)
    # Clamp duration to API limits
    duration_ms = max(3_000, min(300_000, duration_ms))
    logger.info(f"[MUSIC] Generating track via ElevenLabs: {prompt[:80]}... ({duration_ms}ms)")

    try:
        audio_iter = client.music.compose(
            prompt=prompt,
            music_length_ms=duration_ms,
            force_instrumental=True,
            output_format="mp3_44100_128",
        )
        with open(output_path, "wb") as f:
            for chunk in audio_iter:
                f.write(chunk)
        logger.info(f"[MUSIC] Track saved to {output_path}")
        return {"success": True, "path": output_path}
    except Exception as e:
        err_msg = str(e)
        # Surface prompt-rejection errors clearly
        if "bad_prompt" in err_msg.lower():
            logger.warning(f"[MUSIC] ElevenLabs rejected prompt: {err_msg}")
            return {
                "success": False,
                "error": (
                    f"ElevenLabs rejected the music prompt (likely contains copyrighted "
                    f"references — artist names, song titles, or copyrighted lyrics are "
                    f"not allowed). Try rephrasing without naming specific artists or songs. "
                    f"Detail: {err_msg[:200]}"
                ),
            }
        logger.error(f"[MUSIC] ElevenLabs music generation failed: {err_msg}")
        return {"success": False, "error": f"Music generation failed: {err_msg[:300]}"}
