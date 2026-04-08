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


class SoundstripeAPIError(Exception):
    """Raised when the Soundstripe API returns an unexpected status."""
    pass


def fetch_soundstripe_tracks(api_key: str, page_count: int = 3) -> list:
    """Fetch tracks from Soundstripe API with audio_files sideloaded.

    Raises SoundstripeAPIError on auth/quota failures so the agent gets a
    distinguishable error rather than an empty list.
    """
    import requests
    headers = {
        "Authorization": f"Token {api_key}",
        "Accept": "application/vnd.api+json",
    }
    all_tracks = []
    # Map audio_file id -> mp3 URL from sideloaded includes
    audio_urls: dict = {}
    for page in range(1, page_count + 1):
        try:
            resp = requests.get(
                "https://api.soundstripe.com/v1/songs",
                headers=headers,
                params={
                    "page[size]": 50,
                    "page[number]": page,
                    "include": "audio_files",
                },
                timeout=15,
            )
            if resp.status_code != 200:
                # Surface auth/quota errors instead of silently returning empty
                detail = resp.text[:300]
                if resp.status_code in (401, 403):
                    raise SoundstripeAPIError(
                        f"Soundstripe authentication failed (HTTP {resp.status_code}): {detail}. "
                        f"The SOUNDSTRIPE_KEY may have expired or is invalid."
                    )
                if resp.status_code == 429:
                    raise SoundstripeAPIError(
                        f"Soundstripe rate limit exceeded (HTTP 429): {detail}"
                    )
                raise SoundstripeAPIError(
                    f"Soundstripe API error (HTTP {resp.status_code}): {detail}"
                )
            body = resp.json()
            # Collect audio_file URLs from the `included` sideload
            for inc in body.get("included", []):
                if inc.get("type") == "audio_files":
                    versions = inc.get("attributes", {}).get("versions", {})
                    mp3_url = versions.get("mp3")
                    if mp3_url:
                        audio_urls[inc["id"]] = mp3_url
            all_tracks.extend(body.get("data", []))
        except SoundstripeAPIError:
            raise  # bubble up — agent needs to know
        except Exception as e:
            logger.warning(f"[MUSIC] Soundstripe fetch error: {e}")
            # Network/parse errors stop pagination but don't fail the whole call
            # if we already have some tracks
            if not all_tracks:
                raise SoundstripeAPIError(f"Soundstripe network/parse error: {e}")
            break

    # Attach the resolved mp3 URL to each track for easy download later
    for track in all_tracks:
        audio_rels = track.get("relationships", {}).get("audio_files", {}).get("data", [])
        for af in audio_rels:
            url = audio_urls.get(af.get("id"))
            if url:
                track["_mp3_url"] = url
                break

    return all_tracks


def download_soundstripe_track(track: dict, output_path: str) -> bool:
    """Download the mp3 for the selected track."""
    import requests
    url = track.get("_mp3_url")
    if not url:
        logger.warning(f"[MUSIC] No download URL found for track {track.get('id')}")
        return False

    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"[MUSIC] Download failed: {e}")
        return False
