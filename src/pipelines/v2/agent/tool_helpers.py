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
    duration_seconds: int = 120,
) -> dict:
    """Generate a music track using Google Lyria 3 Pro via OpenRouter.

    Uses the OpenAI-compatible API pointed at OpenRouter, which proxies
    to Google's Lyria 3 Pro model ($0.08/song).

    Args:
        api_key: OpenRouter API key.
        prompt: Natural language description of desired music
            (mood, genre, tempo, instruments, energy).
        output_path: Where to save the audio file.
        duration_seconds: Desired track length hint (embedded in prompt).
            Lyria 3 Pro generates up to ~3 minutes.

    Returns:
        dict with ``success`` bool, ``path``, and ``error`` on failure.
    """
    import base64
    import requests

    # Embed duration hint into the prompt so the model targets the right length
    duration_seconds = max(10, min(180, duration_seconds))
    full_prompt = (
        f"{prompt}\n\n"
        f"The track should be approximately {duration_seconds} seconds long. "
        f"Instrumental only, no vocals."
    )

    logger.info(f"[MUSIC] Generating track via Lyria 3 Pro (OpenRouter): {prompt[:80]}...")

    try:
        # OpenRouter requires stream=true for audio output models.
        # We collect the streamed SSE chunks, extract audio data from
        # the final assembled response.
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "google/lyria-3-pro-preview",
                "messages": [{"role": "user", "content": full_prompt}],
                "modalities": ["audio", "text"],
                "stream": True,
            },
            timeout=180,
            stream=True,
        )

        if resp.status_code != 200:
            detail = resp.text[:400]
            logger.error(f"[MUSIC] OpenRouter API error (HTTP {resp.status_code}): {detail}")
            return {
                "success": False,
                "error": f"Music generation API error (HTTP {resp.status_code}): {detail[:200]}",
            }

        # Parse SSE stream — collect audio chunks and text
        audio_b64_parts: list = []
        text_content = ""
        import json as _json

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]  # strip "data: "
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = _json.loads(payload)
            except _json.JSONDecodeError:
                continue

            delta = (chunk.get("choices") or [{}])[0].get("delta", {})

            # Audio data in delta
            audio_obj = delta.get("audio")
            if audio_obj:
                if isinstance(audio_obj, dict) and audio_obj.get("data"):
                    audio_b64_parts.append(audio_obj["data"])
                elif isinstance(audio_obj, str):
                    audio_b64_parts.append(audio_obj)

            # Text content in delta
            if delta.get("content"):
                content = delta["content"]
                if isinstance(content, str):
                    text_content += content
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "audio" and part.get("data"):
                                audio_b64_parts.append(part["data"])
                            elif part.get("type") == "text":
                                text_content += part.get("text", "")

        if not audio_b64_parts:
            logger.error(
                f"[MUSIC] No audio data found in streamed response. "
                f"Text received: {text_content[:200]}"
            )
            return {
                "success": False,
                "error": (
                    "Music generation completed but no audio data was returned. "
                    f"Model response: {text_content[:200] if text_content else '(empty)'}"
                ),
            }

        # Decode and concatenate audio
        audio_bytes = b"".join(base64.b64decode(chunk) for chunk in audio_b64_parts)

        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        file_size_kb = len(audio_bytes) / 1024
        logger.info(f"[MUSIC] Track saved to {output_path} ({file_size_kb:.0f} KB)")
        if text_content:
            logger.info(f"[MUSIC] Lyrics/structure: {text_content[:200]}")

        return {"success": True, "path": output_path}

    except requests.Timeout:
        logger.error("[MUSIC] OpenRouter request timed out (180s)")
        return {"success": False, "error": "Music generation timed out. Try a shorter duration."}
    except Exception as e:
        err_msg = str(e)
        logger.error(f"[MUSIC] Music generation failed: {err_msg}")
        return {"success": False, "error": f"Music generation failed: {err_msg[:300]}"}
