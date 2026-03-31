"""
AI Video Generation Pipeline — generate video clips from text prompts.

Uses Google Veo via Vertex AI (predictLongRunning) to generate short video clips.
Falls back to FAL.ai if configured (supports Kling, Runway, etc.).

The generated video is saved to the workspace and can be used as a clip in the edit.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Vertex AI Veo defaults
DEFAULT_MODEL = "veo-3.1-generate-001"
DEFAULT_LOCATION = "us-central1"
DEFAULT_DURATION = 8
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_RESOLUTION = "1080p"
POLL_INTERVAL = 10  # seconds
MAX_POLL_TIME = 600  # 10 minutes


def _get_gcp_access_token() -> Optional[str]:
    """Get a GCP access token from application default credentials."""
    try:
        import google.auth
        import google.auth.transport.requests
        credentials, project = google.auth.default()
        credentials.refresh(google.auth.transport.requests.Request())
        return credentials.token
    except Exception as e:
        logger.warning(f"[VIDEOGEN] Failed to get GCP access token: {e}")
        return None


def _get_gcp_project() -> Optional[str]:
    """Get the GCP project ID from environment or default credentials."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project:
        return project
    try:
        import google.auth
        _, project = google.auth.default()
        return project
    except Exception:
        return None


def generate_video_veo(
    prompt: str,
    output_path: str,
    duration: int = DEFAULT_DURATION,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    resolution: str = DEFAULT_RESOLUTION,
    generate_audio: bool = True,
    model: str = DEFAULT_MODEL,
    location: str = DEFAULT_LOCATION,
    progress_callback=None,
) -> dict:
    """
    Generate a video using Google Veo via Vertex AI.

    Args:
        prompt: Text description of the video to generate.
        output_path: Where to save the generated MP4.
        duration: Video duration in seconds (4, 6, or 8).
        aspect_ratio: "16:9" or "9:16".
        resolution: "720p", "1080p", or "4k".
        generate_audio: Whether to generate audio with the video.
        model: Veo model ID.
        location: GCP region.
        progress_callback: Optional callable(step, message) for progress updates.

    Returns:
        Dict with status, file_path, duration, and details.
    """
    token = _get_gcp_access_token()
    if not token:
        return {"error": "GCP authentication failed. Run 'gcloud auth application-default login'."}

    project = _get_gcp_project()
    if not project:
        return {"error": "GOOGLE_CLOUD_PROJECT not set and could not detect project."}

    # Submit generation request
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project}/locations/{location}/"
        f"publishers/google/models/{model}:predictLongRunning"
    )

    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "durationSeconds": duration,
            "aspectRatio": aspect_ratio,
            "resolution": resolution,
            "generateAudio": generate_audio,
            "sampleCount": 1,
            "personGeneration": "allow_adult",
        },
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    if progress_callback:
        progress_callback("submitting", f"Submitting to Veo ({model})...")

    logger.info(f"[VIDEOGEN] Submitting to Veo: model={model} duration={duration}s prompt={prompt[:80]}...")

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
    except Exception as e:
        return {"error": f"Failed to submit Veo request: {e}"}

    if resp.status_code != 200:
        error_detail = resp.text[:500]
        logger.error(f"[VIDEOGEN] Veo submit failed ({resp.status_code}): {error_detail}")
        return {"error": f"Veo API error ({resp.status_code}): {error_detail}"}

    operation = resp.json()
    operation_name = operation.get("name")
    if not operation_name:
        return {"error": f"Unexpected Veo response: {json.dumps(operation)[:300]}"}

    logger.info(f"[VIDEOGEN] Operation started: {operation_name}")

    # Poll for completion
    poll_url = f"https://{location}-aiplatform.googleapis.com/v1/{operation_name}"
    start_time = time.time()

    if progress_callback:
        progress_callback("generating", "Veo is generating video...")

    while time.time() - start_time < MAX_POLL_TIME:
        time.sleep(POLL_INTERVAL)

        # Refresh token if needed (long polls)
        elapsed = time.time() - start_time
        if elapsed > 300:
            token = _get_gcp_access_token()
            headers["Authorization"] = f"Bearer {token}"

        try:
            status_resp = requests.get(poll_url, headers=headers, timeout=15)
        except Exception as e:
            logger.warning(f"[VIDEOGEN] Poll failed: {e}")
            continue

        if status_resp.status_code != 200:
            logger.warning(f"[VIDEOGEN] Poll status {status_resp.status_code}")
            continue

        status = status_resp.json()

        if progress_callback:
            pct = min(90, int(elapsed / MAX_POLL_TIME * 90))
            progress_callback("generating", f"Generating video... ({int(elapsed)}s elapsed)")

        if status.get("done"):
            logger.info(f"[VIDEOGEN] Operation complete after {elapsed:.0f}s")

            # Check for error
            if "error" in status:
                error_msg = status["error"].get("message", str(status["error"]))
                return {"error": f"Veo generation failed: {error_msg}"}

            # Extract video from response
            result = status.get("response", status.get("result", {}))
            predictions = result.get("predictions", [])

            if not predictions:
                return {"error": "Veo returned no predictions"}

            # Save video
            video_data = predictions[0]
            if "bytesBase64Encoded" in video_data:
                video_bytes = base64.b64decode(video_data["bytesBase64Encoded"])
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(video_bytes)
                logger.info(f"[VIDEOGEN] Video saved: {output_path} ({len(video_bytes)} bytes)")
            elif "gcsUri" in video_data:
                # Download from GCS
                gcs_uri = video_data["gcsUri"]
                logger.info(f"[VIDEOGEN] Downloading from GCS: {gcs_uri}")
                _download_from_gcs(gcs_uri, output_path, token)
            else:
                return {"error": f"Unexpected prediction format: {list(video_data.keys())}"}

            if progress_callback:
                progress_callback("complete", "Video generated!")

            return {
                "status": "complete",
                "file_path": output_path,
                "duration": duration,
                "model": model,
                "generation_time_seconds": round(elapsed, 1),
            }

    return {"error": f"Veo generation timed out after {MAX_POLL_TIME}s"}


def _download_from_gcs(gcs_uri: str, output_path: str, token: str):
    """Download a file from a gs:// URI using the JSON API."""
    # gs://bucket/path/to/file.mp4 -> bucket, path/to/file.mp4
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = parts[0]
    obj_path = parts[1] if len(parts) > 1 else ""

    url = (
        f"https://storage.googleapis.com/storage/v1/b/{bucket}"
        f"/o/{requests.utils.quote(obj_path, safe='')}?alt=media"
    )
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=120)
    resp.raise_for_status()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


async def generate_video_async(
    prompt: str,
    output_path: str,
    duration: int = DEFAULT_DURATION,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    resolution: str = DEFAULT_RESOLUTION,
    generate_audio: bool = True,
    model: str = DEFAULT_MODEL,
    progress_callback=None,
) -> dict:
    """Async wrapper around generate_video_veo."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: generate_video_veo(
            prompt=prompt,
            output_path=output_path,
            duration=duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            generate_audio=generate_audio,
            model=model,
            progress_callback=progress_callback,
        ),
    )
