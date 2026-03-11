"""
V2 Narration Pipeline — on-demand, triggered after a plan is ready.

Workflow:
  1. Generate narration script from storyboard shots using Gemini
  2. Concatenate all shot narrations into one full script (with pause markers)
  3. Call ElevenLabs TTS to produce a single narration.mp3
  4. Save script + audio to workspace
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional

from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from src.pipelines.v2.planning.planning_prompts import NARRATION_SCRIPT_SYSTEM, NARRATION_SCRIPT_USER
from src.pipelines.v2.schemas import Shot, Storyboard
from src.pipelines.v2.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

# ElevenLabs voice settings
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
ELEVENLABS_MODEL_ID = "eleven_flash_v2_5"


class NarrationPipeline:
    """
    Generates narration script + audio for a workspace that has a storyboard.

    Usage:
        pipeline = NarrationPipeline(gemini, workspace)
        audio_path = await pipeline.run(user_prompt="...", override_script=None)
    """

    def __init__(self, gemini: GeminiGenaiManager, workspace: WorkspaceManager):
        self.gemini = gemini
        self.workspace = workspace

    async def run(
        self,
        user_prompt: str,
        override_script: Optional[str] = None,
    ) -> str:
        """
        Generate narration audio and return path to narration.mp3.

        Args:
            user_prompt: The original user editing intent (used in script prompt).
            override_script: If provided, skip LLM script generation and use this text.

        Returns:
            Path to narration.mp3 in workspace.
        """
        storyboard = self.workspace.load_storyboard()
        if not storyboard:
            raise ValueError("No storyboard found — run /v2/plan first")

        # Step 1: Generate or use override script
        if override_script:
            script = override_script
        else:
            script = await self._generate_script(storyboard, user_prompt)

        # Save script to workspace
        script_path = self.workspace.get_narration_script_path()
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script, encoding="utf-8")
        logger.info(f"[NARRATION] Script saved to {script_path}")

        # Step 2: Generate TTS audio
        audio_path = self.workspace.get_narration_path()
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        await self._generate_audio(script, str(audio_path))

        logger.info(f"[NARRATION] Audio saved to {audio_path}")
        return str(audio_path)

    async def _generate_script(self, storyboard: Storyboard, user_prompt: str) -> str:
        """Call Gemini to write narration for each shot."""
        shots_detail = _format_shots_for_narration(storyboard.shots)

        try:
            gist = self.workspace.load_session().gist
        except Exception:
            gist = ""

        user_content = NARRATION_SCRIPT_USER.format(
            user_prompt=user_prompt,
            shots_detail=shots_detail,
            gist=gist,
        )
        prompt_contents = [NARRATION_SCRIPT_SYSTEM, user_content]

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.gemini.LLM_request(prompt_contents, schema=None),
        )
        return result if isinstance(result, str) else str(result)

    async def _generate_audio(self, script: str, output_path: str) -> None:
        """Call ElevenLabs TTS to convert script to audio."""
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: _tts_sync(script, output_path, api_key))


def _tts_sync(text: str, output_path: str, api_key: str) -> None:
    """Blocking ElevenLabs TTS call."""
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=api_key)
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=ELEVENLABS_VOICE_ID,
        model_id=ELEVENLABS_MODEL_ID,
    )
    with open(output_path, "wb") as f:
        for chunk in audio:
            if chunk:
                f.write(chunk)


def _format_shots_for_narration(shots: List[Shot]) -> str:
    lines = []
    for i, shot in enumerate(shots, 1):
        clip_desc = ""
        if shot.retrieved_clip:
            clip_desc = f" | clip: {shot.retrieved_clip.description[:80]}"
        lines.append(
            f"Shot {i} (id={shot.id}, duration={shot.duration_seconds:.1f}s, "
            f"purpose={shot.purpose!r}{clip_desc})\n"
            f"  existing narration: {shot.narration or '(none)'}"
        )
    return "\n\n".join(lines)
