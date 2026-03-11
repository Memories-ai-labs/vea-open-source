"""
V2 Music Pipeline — on-demand, triggered after a plan is ready.

Workflow:
  1. Use Gemini to determine mood/genre from storyboard + user prompt
  2. Fetch candidate tracks from Soundstripe (or skip if key unavailable)
  3. Gemini selects the best track
  4. Download selected track to workspace music/track.mp3
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from src.pipelines.v2.planning.planning_prompts import MUSIC_MOOD_SYSTEM, MUSIC_MOOD_USER
from src.pipelines.v2.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

SOUNDSTRIPE_BASE_URL = "https://api.soundstripe.com/v1/songs"


class MusicPipeline:
    """
    Selects and downloads background music for a workspace.

    Usage:
        pipeline = MusicPipeline(gemini, workspace)
        music_path = await pipeline.run(user_prompt="...", mood="upbeat")
    """

    def __init__(self, gemini: GeminiGenaiManager, workspace: WorkspaceManager):
        self.gemini = gemini
        self.workspace = workspace

    async def run(
        self,
        user_prompt: str,
        mood: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Optional[str]:
        """
        Select and download music. Returns path to track.mp3, or None if skipped.
        """
        soundstripe_key = os.environ.get("SOUNDSTRIPE_KEY")
        if not soundstripe_key:
            logger.warning("[MUSIC] SOUNDSTRIPE_KEY not set — skipping music selection")
            return None

        storyboard = self.workspace.load_storyboard()

        # Step 1: Determine mood
        resolved_mood = mood or await self._determine_mood(storyboard, user_prompt, prompt)
        logger.info(f"[MUSIC] Resolved mood: {resolved_mood}")

        # Step 2: Fetch candidates from Soundstripe
        tracks = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _fetch_tracks(soundstripe_key, resolved_mood)
        )
        if not tracks:
            logger.warning("[MUSIC] No tracks found — skipping")
            return None

        # Step 3: LLM picks best track
        best_track = await self._select_track(tracks, storyboard, user_prompt, resolved_mood)
        if not best_track:
            logger.warning("[MUSIC] No track selected")
            return None

        # Step 4: Download track
        music_path = self.workspace.get_music_path()
        music_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _download_track(best_track, str(music_path))
        )
        if downloaded:
            logger.info(f"[MUSIC] Track downloaded to {music_path}")
            return str(music_path)
        return None

    async def _determine_mood(self, storyboard, user_prompt: str, extra_prompt: Optional[str]) -> str:
        """Use Gemini to determine appropriate music mood from storyboard + prompt."""
        if not storyboard:
            return "neutral background"

        shots_summary = "; ".join(
            f"{s.purpose}" for s in (storyboard.shots[:5] if storyboard else [])
        )
        combined_prompt = user_prompt
        if extra_prompt:
            combined_prompt += f" {extra_prompt}"

        user_content = MUSIC_MOOD_USER.format(
            user_prompt=combined_prompt,
            theme=storyboard.theme if storyboard else "",
            narrative_arc=storyboard.narrative_arc if storyboard else "",
            shots_summary=shots_summary,
        )
        prompt_contents = [MUSIC_MOOD_SYSTEM, user_content]

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.gemini.LLM_request(prompt_contents, schema=None),
        )
        return (result if isinstance(result, str) else str(result)).strip()[:200]

    async def _select_track(
        self, tracks: list, storyboard, user_prompt: str, mood: str
    ) -> Optional[dict]:
        """Use simple score matching; Gemini selection is a future enhancement."""
        # For now: sort by mood match (simple keyword heuristic), return top track
        mood_words = set(mood.lower().split())
        scored = []
        for t in tracks:
            attrs = t.get("attributes", {})
            track_mood = (attrs.get("mood") or "").lower()
            track_genre = (attrs.get("genre") or "").lower()
            overlap = len(mood_words & set(track_mood.split() + track_genre.split()))
            scored.append((overlap, t))
        scored.sort(key=lambda x: -x[0])
        return scored[0][1] if scored else None


def _fetch_tracks(api_key: str, mood: str, page_count: int = 3) -> list:
    """Fetch tracks from Soundstripe API."""
    import requests
    headers = {
        "Authorization": f"Token {api_key}",
        "Accept": "application/vnd.api+json",
    }
    all_tracks = []
    for page in range(1, page_count + 1):
        try:
            resp = requests.get(
                SOUNDSTRIPE_BASE_URL,
                headers=headers,
                params={"page[size]": 50, "page[number]": page},
                timeout=15,
            )
            if resp.status_code != 200:
                break
            all_tracks.extend(resp.json().get("data", []))
        except Exception as e:
            logger.warning(f"[MUSIC] Soundstripe fetch error: {e}")
            break
    return all_tracks


def _download_track(track: dict, output_path: str) -> bool:
    """Download the mp3 for the selected track."""
    import requests
    attrs = track.get("attributes", {})
    # Soundstripe track has a versions array with download URLs
    versions = attrs.get("versions", [])
    url = None
    for v in versions:
        if isinstance(v, dict):
            url = v.get("audio_file_url") or v.get("download_url")
            if url:
                break
    if not url:
        # Try direct mp3_url
        url = attrs.get("mp3_url") or attrs.get("download_url")
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
