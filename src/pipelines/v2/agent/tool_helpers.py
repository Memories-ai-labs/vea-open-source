"""Standalone helper functions for narration (TTS), STT, and music (Soundstripe)."""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def tts_sync(text: str, output_path: str, api_key: str) -> None:
    """Blocking ElevenLabs TTS call."""
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


def stt_word_timestamps(audio_path: str, api_key: str) -> List[Dict]:
    """Transcribe audio with word-level timestamps via ElevenLabs Scribe.

    Returns a list of word dicts: [{"text": "Hello", "start": 0.12, "end": 0.45}, ...]
    """
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=api_key)
    with open(audio_path, "rb") as f:
        result = client.speech_to_text.convert(
            file=f,
            model_id="scribe_v1",
            timestamps_granularity="word",
        )
    words = []
    for w in getattr(result, "words", []) or []:
        text = getattr(w, "text", "") if not isinstance(w, dict) else w.get("text", "")
        start = getattr(w, "start", 0) if not isinstance(w, dict) else w.get("start", 0)
        end = getattr(w, "end", 0) if not isinstance(w, dict) else w.get("end", 0)
        if text.strip():
            words.append({"text": text.strip(), "start": float(start), "end": float(end)})
    return words


def fetch_soundstripe_tracks(api_key: str, page_count: int = 3) -> list:
    """Fetch tracks from Soundstripe API with audio_files sideloaded."""
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
                break
            body = resp.json()
            # Collect audio_file URLs from the `included` sideload
            for inc in body.get("included", []):
                if inc.get("type") == "audio_files":
                    versions = inc.get("attributes", {}).get("versions", {})
                    mp3_url = versions.get("mp3")
                    if mp3_url:
                        audio_urls[inc["id"]] = mp3_url
            all_tracks.extend(body.get("data", []))
        except Exception as e:
            logger.warning(f"[MUSIC] Soundstripe fetch error: {e}")
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
