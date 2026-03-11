"""
FCPXML Agent — LLM-driven enhancement + validation loop.

Workflow:
  1. scaffold()       → always-valid baseline FCPXML
  2. autofix()        → correct common LLM mistakes before any LLM call
  3. compile()        → validate scaffold (should always pass)
  4. LLM enhance      → add transitions, narration track, music track
  5. autofix + compile → validate enhanced XML
  6. Loop (up to MAX_CORRECT_TRIES): LLM correct → autofix + compile
  7. Fallback         → use scaffold if all correction attempts fail
  8. Write final FCPXML to workspace
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional

from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from src.pipelines.v2.fcpxml.fcpxml_compiler import ValidationResult, autofix, compile_fcpxml
from src.pipelines.v2.fcpxml.fcpxml_scaffold import build_scaffold
from src.pipelines.v2.planning.planning_prompts import (
    GENERATE_FCPXML_CORRECT_USER,
    GENERATE_FCPXML_ENHANCE_USER,
    GENERATE_FCPXML_SYSTEM,
)
from src.pipelines.v2.schemas import RetrievedClip, Storyboard
from src.pipelines.v2.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

MAX_CORRECT_TRIES = 3


class FcpxmlAgent:
    """
    Orchestrates scaffold → LLM enhancement → validation → correction loop.

    Usage:
        agent = FcpxmlAgent(gemini, workspace, storyboard, clips_by_id)
        fcpxml_path = await agent.run(
            narration_path="...", narration_duration=45.0,
            music_path="...", music_duration=120.0,
        )
    """

    def __init__(
        self,
        gemini: GeminiGenaiManager,
        workspace: WorkspaceManager,
        storyboard: Storyboard,
        clips_by_id: Dict[str, RetrievedClip],
        frame_rate: float = 24.0,
        width: int = 1920,
        height: int = 1080,
    ):
        self.gemini = gemini
        self.workspace = workspace
        self.storyboard = storyboard
        self.clips_by_id = clips_by_id
        self.frame_rate = frame_rate
        self.width = width
        self.height = height

    async def run(
        self,
        narration_path: Optional[str] = None,
        narration_duration: float = 0.0,
        music_path: Optional[str] = None,
        music_duration: float = 0.0,
        music_gain_db: float = -12.0,
        version: int = 1,
    ) -> str:
        """
        Generate, enhance, validate and save the FCPXML.

        Returns the path to the final .fcpxml file.
        """
        output_path = str(self.workspace.get_fcpxml_path(version))

        # ------------------------------------------------------------------
        # Step 1: Build scaffold (guaranteed valid)
        # ------------------------------------------------------------------
        scaffold_path = output_path.replace(".fcpxml", "_scaffold.fcpxml")
        build_scaffold(
            self.storyboard,
            self.clips_by_id,
            output_path=scaffold_path,
            frame_rate=self.frame_rate,
            width=self.width,
            height=self.height,
            project_name=f"{self.workspace.project_name} v{version}",
            narration_path=narration_path,
            narration_duration=narration_duration,
            music_path=music_path,
            music_duration=music_duration,
            music_gain_db=music_gain_db,
        )
        scaffold_xml = Path(scaffold_path).read_text(encoding="utf-8")

        # Validate scaffold (sanity check — should always pass)
        scaffold_result = compile_fcpxml(scaffold_xml)
        if not scaffold_result:
            logger.error(f"[FCPXML AGENT] Scaffold failed validation — using as-is:\n{scaffold_result.error_summary()}")
            # Write scaffold as final output (better than nothing)
            Path(output_path).write_text(scaffold_xml, encoding="utf-8")
            return output_path

        # ------------------------------------------------------------------
        # Step 2: LLM Enhancement
        # ------------------------------------------------------------------
        enhanced_xml = await self._llm_enhance(
            scaffold_xml, narration_path, music_path
        )
        enhanced_xml = autofix(enhanced_xml)
        enhanced_result = compile_fcpxml(enhanced_xml)

        if enhanced_result:
            logger.info("[FCPXML AGENT] LLM enhancement valid — proceeding")
            current_xml = enhanced_xml
        else:
            logger.warning(
                f"[FCPXML AGENT] LLM enhancement invalid ({len(enhanced_result.errors)} errors) "
                f"— entering correction loop"
            )
            current_xml = enhanced_xml
            current_result = enhanced_result

            # ------------------------------------------------------------------
            # Step 3: Correction loop
            # ------------------------------------------------------------------
            corrected = False
            for attempt in range(1, MAX_CORRECT_TRIES + 1):
                logger.info(f"[FCPXML AGENT] Correction attempt {attempt}/{MAX_CORRECT_TRIES}")
                corrected_xml = await self._llm_correct(current_xml, current_result)
                corrected_xml = autofix(corrected_xml)
                corrected_result = compile_fcpxml(corrected_xml)

                if corrected_result:
                    logger.info(f"[FCPXML AGENT] Correction attempt {attempt} succeeded")
                    current_xml = corrected_xml
                    corrected = True
                    break
                else:
                    logger.warning(
                        f"[FCPXML AGENT] Correction attempt {attempt} still has "
                        f"{len(corrected_result.errors)} errors"
                    )
                    current_xml = corrected_xml
                    current_result = corrected_result

            if not corrected:
                logger.warning("[FCPXML AGENT] All correction attempts failed — falling back to scaffold")
                current_xml = scaffold_xml

        # ------------------------------------------------------------------
        # Step 4: Write final output
        # ------------------------------------------------------------------
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(current_xml, encoding="utf-8")
        logger.info(f"[FCPXML AGENT] Final FCPXML written to {output_path}")
        return output_path

    # -------------------------------------------------------------------------
    # LLM calls
    # -------------------------------------------------------------------------

    async def _llm_enhance(
        self,
        scaffold_xml: str,
        narration_path: Optional[str],
        music_path: Optional[str],
    ) -> str:
        """Call Gemini to enhance the scaffold with transitions and audio tracks."""
        user_content = GENERATE_FCPXML_ENHANCE_USER.format(
            narration_path=narration_path or "(none)",
            music_path=music_path or "(none)",
            current_xml=scaffold_xml,
        )
        prompt_contents = [GENERATE_FCPXML_SYSTEM, user_content]

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.gemini.LLM_request(prompt_contents, schema=None),
            )
            text = result if isinstance(result, str) else str(result)
            return text
        except Exception as e:
            logger.error(f"[FCPXML AGENT] Enhancement LLM call failed: {e}")
            return scaffold_xml  # Return scaffold unchanged if LLM fails

    async def _llm_correct(self, xml_text: str, result: ValidationResult) -> str:
        """Call Gemini to fix validation errors."""
        user_content = GENERATE_FCPXML_CORRECT_USER.format(
            errors=result.error_summary(),
            current_xml=xml_text,
        )
        prompt_contents = [GENERATE_FCPXML_SYSTEM, user_content]

        try:
            loop = asyncio.get_event_loop()
            corrected = await loop.run_in_executor(
                None,
                lambda: self.gemini.LLM_request(prompt_contents, schema=None),
            )
            text = corrected if isinstance(corrected, str) else str(corrected)
            return text
        except Exception as e:
            logger.error(f"[FCPXML AGENT] Correction LLM call failed: {e}")
            return xml_text  # Return unchanged if LLM fails


# ---------------------------------------------------------------------------
# Convenience function (used by the API endpoint)
# ---------------------------------------------------------------------------

async def generate_fcpxml(
    gemini: GeminiGenaiManager,
    workspace: WorkspaceManager,
    *,
    frame_rate: float = 24.0,
    width: int = 1920,
    height: int = 1080,
    narration_path: Optional[str] = None,
    narration_duration: float = 0.0,
    music_path: Optional[str] = None,
    music_duration: float = 0.0,
) -> str:
    """
    Load storyboard and clips from workspace, run the FCPXML agent,
    update session status, and return the path to the generated .fcpxml.
    """
    storyboard = workspace.load_storyboard()
    if not storyboard:
        raise ValueError("No storyboard found — run /v2/plan first")

    clips = workspace.load_clips()

    # Resolve which clip to assign to each shot (best-scoring clip for the shot query)
    clips_by_id: Dict[str, RetrievedClip] = {}
    for shot in storyboard.shots:
        if shot.retrieved_clip:
            clips_by_id[shot.id] = shot.retrieved_clip
        else:
            # Find best clip matching this shot's search_query from accumulated pool
            best = _find_best_clip(shot.search_query, clips)
            if best:
                clips_by_id[shot.id] = best

    # Pick next version number
    version = _next_version(workspace)

    agent = FcpxmlAgent(
        gemini=gemini,
        workspace=workspace,
        storyboard=storyboard,
        clips_by_id=clips_by_id,
        frame_rate=frame_rate,
        width=width,
        height=height,
    )

    fcpxml_path = await agent.run(
        narration_path=narration_path,
        narration_duration=narration_duration,
        music_path=music_path,
        music_duration=music_duration,
        version=version,
    )

    workspace.update_status("fcpxml_ready")
    return fcpxml_path


def _find_best_clip(query: str, clips: list) -> Optional[RetrievedClip]:
    """Find the highest-scored clip whose shot_query overlaps with the given query."""
    if not clips:
        return None
    query_words = set(query.lower().split())
    best_score = -1.0
    best_clip = None
    for clip in clips:
        # Simple word overlap heuristic for matching
        clip_words = set(clip.shot_query.lower().split())
        overlap = len(query_words & clip_words)
        adjusted = clip.score + overlap * 0.05
        if adjusted > best_score:
            best_score = adjusted
            best_clip = clip
    return best_clip


def _next_version(workspace: WorkspaceManager) -> int:
    """Find the next available version number for fcpxml/edit_v{n}.fcpxml."""
    fcpxml_dir = workspace.root / "fcpxml"
    v = 1
    while (fcpxml_dir / f"edit_v{v}.fcpxml").exists():
        v += 1
    return v
