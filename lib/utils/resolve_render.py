"""DaVinci Resolve render workflow — FCPXML import and automated rendering."""
from __future__ import annotations
import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Callable, Coroutine, Literal, Optional

from lib.utils.resolve_setup import ensure_pythonpath, check_resolve_status

logger = logging.getLogger(__name__)

RENDER_PRESETS = {
    "preview": {
        "format": "MP4",
        "codec": "H264_NVENC",  # fallback to H264 if NVENC unavailable
        "codec_fallback": "H264",
        "description": "H.264 preview render (480p)",
        "width": 854,
        "height": 480,
    },
    "final": {
        "format": "QuickTime",
        "codec": "ProRes422",
        "codec_fallback": "ProRes422",
        "description": "ProRes 422 final render (native resolution)",
        "width": None,  # use timeline native resolution
        "height": None,
    },
}


class ResolveRenderer:
    """
    Wraps DaVinci Resolve Python scripting API for automated FCPXML rendering.

    Resolve must already be running as a -nogui daemon before using this class.
    Requires DaVinci Resolve Studio (not the free version).

    Usage:
        renderer = ResolveRenderer()
        output = await renderer.render(
            fcpxml_path="path/to/edit.fcpxml",
            media_dir="path/to/media/",
            output_path="path/to/output.mp4",
            quality="preview",
            progress_callback=lambda pct: print(f"{pct}%"),
        )
    """

    def __init__(self):
        ensure_pythonpath()
        self._resolve = self._connect()

    def _connect(self):
        try:
            import DaVinciResolveScript as dvr  # type: ignore
            resolve = dvr.scriptapp("Resolve")
            if resolve is None:
                raise RuntimeError(
                    "Resolve returned None. Is DaVinci Resolve Studio running with -nogui?"
                )
            return resolve
        except ImportError:
            raise RuntimeError(
                "DaVinciResolveScript not importable. Set RESOLVE_SCRIPT_API and "
                "RESOLVE_SCRIPT_LIB environment variables and ensure Resolve is running."
            )

    def is_available(self) -> bool:
        status = check_resolve_status()
        return status["running"] and status["studio"]

    async def render(
        self,
        fcpxml_path: str,
        media_dir: str,
        output_path: str,
        quality: Literal["preview", "final"] = "preview",
        progress_callback: Optional[Callable[[float], Coroutine]] = None,
    ) -> str:
        """
        Import FCPXML into Resolve, render, and return output path.

        Args:
            fcpxml_path: Path to .fcpxml file
            media_dir: Directory containing source media (for relinking)
            output_path: Desired output file path (without extension)
            quality: "preview" (H.264) or "final" (ProRes 422)
            progress_callback: Async callable(pct: float) called during render

        Returns:
            Actual output file path (Resolve constructs this from settings)
        """
        preset = RENDER_PRESETS[quality]
        project_name = f"vea_{uuid.uuid4().hex[:8]}"
        abs_output = Path(output_path).resolve()
        abs_output.parent.mkdir(parents=True, exist_ok=True)
        output_dir = str(abs_output.parent)
        output_name = abs_output.stem

        logger.info(f"[RESOLVE] Starting render: {quality} quality, project={project_name}")
        logger.info(f"[RESOLVE] Output dir: {output_dir}, name: {output_name}")
        logger.info(f"[RESOLVE] FCPXML: {fcpxml_path}, Media: {media_dir}")

        pm = self._resolve.GetProjectManager()

        # Close any currently open project so we can create a new one
        current = pm.GetCurrentProject()
        if current is not None:
            logger.info(f"[RESOLVE] Closing current project: {current.GetName()}")
            pm.CloseProject(current)

        project = pm.CreateProject(project_name)
        if project is None:
            raise RuntimeError(f"[RESOLVE] Failed to create project: {project_name}")

        try:
            media_pool = project.GetMediaPool()

            # Import FCPXML as timeline
            logger.info(f"[RESOLVE] Importing FCPXML: {fcpxml_path}")
            timeline = media_pool.ImportTimelineFromFile(
                str(Path(fcpxml_path).resolve()),
                {
                    "timelineName": "VEA Edit",
                    "importSourceClips": True,
                    "sourceClipsPath": str(Path(media_dir).resolve()),
                },
            )
            if timeline is None:
                raise RuntimeError(
                    f"[RESOLVE] Failed to import FCPXML. Check that the file is valid "
                    f"FCPXML 1.10 and media paths are reachable: {fcpxml_path}"
                )

            project.SetCurrentTimeline(timeline)
            logger.info(f"[RESOLVE] Timeline imported: {timeline.GetName()}, clips: {timeline.GetTrackCount('video')}")

            # Configure render settings
            render_settings = {
                "SelectAllFrames": True,
                "TargetDir": output_dir,
                "CustomName": output_name,
                "UniqueFilenameStyle": 0,
                "ExportVideo": True,
                "ExportAudio": True,
            }
            # Set resolution for preview renders (480p for speed)
            if preset.get("width") and preset.get("height"):
                render_settings["FormatWidth"] = preset["width"]
                render_settings["FormatHeight"] = preset["height"]
                logger.info(f"[RESOLVE] Preview resolution: {preset['width']}x{preset['height']}")
            project.SetRenderSettings(render_settings)

            # Try preferred codec, fall back if not available
            codec = preset["codec"]
            if not project.SetCurrentRenderFormatAndCodec(preset["format"], codec):
                codec = preset["codec_fallback"]
                logger.warning(f"[RESOLVE] Codec {preset['codec']} unavailable, using {codec}")
                project.SetCurrentRenderFormatAndCodec(preset["format"], codec)

            job_id = project.AddRenderJob()
            logger.info(f"[RESOLVE] Render job added: {job_id}")

            project.StartRendering(job_id)

            # Poll for completion
            while project.IsRenderingInProgress():
                try:
                    status = project.GetRenderJobStatus(job_id)
                    if status:
                        pct = float(status.get("CompletionPercentage", 0))
                        logger.info(f"[RESOLVE] Render progress: {pct:.0f}%")
                        if progress_callback:
                            await progress_callback(pct)
                except Exception as e:
                    logger.debug(f"[RESOLVE] Status poll error (non-fatal): {e}")
                await asyncio.sleep(2)

            # Check final status
            final_status = project.GetRenderJobStatus(job_id)
            job_status = "Unknown"
            if final_status:
                job_status = final_status.get("JobStatus", "Unknown")
            logger.info(f"[RESOLVE] Render complete. Status: {job_status}")

            if job_status == "Failed":
                raise RuntimeError(f"[RESOLVE] Render failed: {final_status}")

            # Construct output path (Resolve doesn't return it from API)
            ext_map = {"QuickTime": ".mov", "MP4": ".mp4"}
            ext = ext_map.get(preset["format"], ".mp4")
            actual_output = str(Path(output_dir) / f"{output_name}{ext}")
            logger.info(f"[RESOLVE] Output: {actual_output}")
            return actual_output

        finally:
            # Always clean up the temporary project
            try:
                project.DeleteAllRenderJobs()
                pm.DeleteProject(project_name)
                logger.info(f"[RESOLVE] Cleaned up project: {project_name}")
            except Exception as e:
                logger.warning(f"[RESOLVE] Cleanup failed: {e}")
