"""
Dynamic Cropping Module

Converts videos from landscape (16:9) to portrait (9:16) by intelligently
tracking the most salient region using ViNet neural network saliency detection.

Core Pipeline:
1. Shot Detection - Find scene cuts using PySceneDetect
2. Saliency Detection - Run ViNet model to find important regions per shot
3. Center Calculation - Aggregate saliency to a single focus point per shot
4. Crop Rendering - Re-render video cropped around focus points
5. Audio Re-attachment - Attach original audio to cropped video
"""

import asyncio
import math
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from moviepy import VideoFileClip
from tqdm import tqdm

from lib.utils.metrics_collector import metrics_collector


# =============================================================================
# Helper Functions
# =============================================================================

def detect_content_region(frames: Sequence[np.ndarray], threshold: int = 10) -> Tuple[int, int, int, int]:
    """
    Detect content bounds to ignore letterboxing (black bars).
    Returns (y1, y2, x1, x2) of the content region.
    """
    h, w = frames[0].shape[:2]
    gray_samples = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    mean_img = np.mean(np.stack(gray_samples, axis=0), axis=0)

    row_mean = mean_img.mean(axis=1)
    col_mean = mean_img.mean(axis=0)

    y_nonblack = np.where(row_mean > threshold)[0]
    x_nonblack = np.where(col_mean > threshold)[0]

    if len(y_nonblack) == 0 or len(x_nonblack) == 0:
        return 0, h, 0, w

    return int(y_nonblack[0]), int(y_nonblack[-1] + 1), int(x_nonblack[0]), int(x_nonblack[-1] + 1)


def find_shot_center(saliency_stack: np.ndarray, threshold: float = 0.6) -> Tuple[int, int]:
    """
    Aggregate saliency across a shot and return the centroid of the strongest region.
    """
    if saliency_stack.ndim == 3:
        aggregated = saliency_stack.sum(axis=0, dtype=np.float32)
    else:
        aggregated = saliency_stack.astype(np.float32)

    if not np.any(aggregated):
        h, w = aggregated.shape
        return w // 2, h // 2

    # Normalize and threshold
    normalized = (aggregated / (aggregated.max() + 1e-6) * 255).astype(np.uint8)
    _, mask = cv2.threshold(normalized, int(threshold * normalized.max()), 255, cv2.THRESH_BINARY)

    # Find contours and get centroid of largest
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = aggregated.shape

    if not contours:
        return w // 2, h // 2

    largest = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest)

    if moments["m00"] == 0:
        return w // 2, h // 2

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


# =============================================================================
# ViNet Saliency Backend
# =============================================================================

class ViNetSaliencyBackend:
    """Loads and runs the ViNet saliency model."""

    def __init__(
        self,
        repo_dir: Path,
        checkpoint_path: Path,
        variant: str = "S",
        clip_len: int = 32,
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.variant = variant.upper()

        if self.variant not in {"S", "A"}:
            raise ValueError("ViNet variant must be 'S' or 'A'.")

        # IMPORTANT: Force CPU to avoid CUDA/FFmpeg subprocess conflicts
        # CUDA context can corrupt MoviePy's FFmpeg reader state causing SIGSEGV
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide all GPUs
        torch.set_default_device("cpu")

        self.device = torch.device("cpu")
        self.clip_len = clip_len
        self.input_size = (384, 224) if self.variant == "S" else (224, 224)

        self.model = self._build_model()
        print(f"[VINET] Loaded ViNet-{self.variant} on CPU (CUDA disabled)")

    def _make_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            len_snippet=self.clip_len,
            decoder_groups=32,
            decoder_upsample=1,
            use_skip=1,
            neck_name="neck2",
            use_channel_shuffle=True,
            use_action_classification=0,
            dataset="mvva",
            split="no_split",
            batch_size=1,
            no_workers=0,
        )

    def _import_model_class(self) -> type:
        if self.variant == "S":
            subdir, module_name, class_name = "ViNet_S", "ViNet_S_model", "VideoSaliencyModel"
        else:
            subdir, module_name, class_name = "ViNet_A", "ViNet_A_model", "ViNet_A"

        sys.path.insert(0, str(self.repo_dir / subdir))
        module = __import__(module_name)
        return getattr(module, class_name)

    def _build_model(self) -> nn.Module:
        ModelClass = self._import_model_class()
        model = ModelClass(self._make_args())

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=False)

        # Set to inference mode
        model.train(False)
        return model.to(self.device)

    @torch.no_grad()
    def predict(self, clip_tensor: torch.Tensor) -> torch.Tensor:
        """Run saliency prediction on a batch of frames."""
        x = clip_tensor.to(self.device)
        y = self.model(x)

        # Normalize output shape to (batch, frames, 1, H, W)
        if y.ndim == 5 and y.shape[2] in (1, 3):
            pass
        elif y.ndim == 5 and y.shape[1] == 1:
            y = y.transpose(1, 2)
        elif y.ndim == 4:
            y = y.unsqueeze(2)
        elif y.ndim == 3:
            batch, height, width = y.shape
            y = y.view(batch, 1, 1, height, width).expand(batch, self.clip_len, 1, height, width)
        else:
            raise RuntimeError(f"Unexpected saliency output shape: {tuple(y.shape)}")

        return y.clamp(0, 1).cpu()


# =============================================================================
# Shot Detection
# =============================================================================

class ShotDetector:
    """Detects shot boundaries using PySceneDetect."""

    @staticmethod
    def detect(clip: VideoFileClip, fps: float, total_frames: int) -> List[Tuple[int, int]]:
        try:
            from scenedetect import SceneManager
            from scenedetect.detectors import AdaptiveDetector, ContentDetector
            from scenedetect.frame_timecode import FrameTimecode
        except ImportError:
            raise RuntimeError("PySceneDetect required. Install via 'pip install scenedetect[opencv]'")

        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=12, luma_only=False))
        manager.add_detector(AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=12))

        base_timecode = FrameTimecode(timecode=0, fps=fps)
        manager._base_timecode = base_timecode
        manager._start_pos = base_timecode
        manager._last_pos = base_timecode
        if manager._stats_manager is not None:
            manager._stats_manager._base_timecode = base_timecode

        progress = tqdm(total=total_frames, desc="Shot detection", unit="frame", leave=False)
        try:
            for idx, frame in enumerate(clip.iter_frames(fps=fps, dtype="uint8")):
                if idx >= total_frames:
                    break
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if manager._frame_size is None:
                    manager._frame_size = (frame_bgr.shape[1], frame_bgr.shape[0])
                manager._process_frame(idx, frame_bgr, callback=None)
                manager._last_pos = base_timecode + idx
                progress.update(1)
        finally:
            progress.close()

        manager._post_process(max(total_frames - 1, 0))
        scenes = manager.get_scene_list()

        if not scenes:
            return []
        return [(start.get_frames(), end.get_frames()) for start, end in scenes]


# =============================================================================
# Dynamic Cropping
# =============================================================================

class DynamicCropping:
    """
    Intelligent video cropping using ViNet saliency detection.
    Converts landscape video to portrait by tracking the most important region.
    """

    VINET_REPO_RELATIVE = Path("vinet_v2")
    VINET_CHECKPOINT_RELATIVE = VINET_REPO_RELATIVE / "final_models" / "ViNet_S" / "vinet_s_mvva_randomsplit.pt"

    def __init__(self, llm, workdir: Optional[str] = None, debug: bool = False):
        self.llm = llm  # Retained for interface compatibility
        self.workdir = Path(workdir or tempfile.mkdtemp())
        self.debug = debug

        # Locate ViNet assets
        self.project_root = Path(__file__).resolve().parents[3]
        self.repo_dir = self.project_root / self.VINET_REPO_RELATIVE
        self.checkpoint_path = self.project_root / self.VINET_CHECKPOINT_RELATIVE

        if not self.repo_dir.exists() or not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "ViNet assets not found. Run 'python -m lib.utils.vinet_setup' to download."
            )

        self.clip_len = 32
        self.stride = 32

        self.backend = ViNetSaliencyBackend(
            self.repo_dir,
            self.checkpoint_path,
            variant="S",
            clip_len=self.clip_len,
        )

    def _estimate_frames(self, clip: VideoFileClip, fps: float) -> int:
        if clip.duration is None or fps <= 0:
            return 0
        return max(1, int(math.ceil(clip.duration * fps)))

    def _get_content_bounds(self, clip: VideoFileClip, total_frames: int) -> Tuple[int, int, int, int]:
        sample_count = min(10, total_frames)
        times = np.linspace(0, clip.duration, num=sample_count, endpoint=False)
        frames = [clip.get_frame(float(t)) for t in times]
        return detect_content_region(frames)

    def _preprocess_frames(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """Convert frames to tensor for ViNet input."""
        width, height = self.backend.input_size
        tensors = []
        for frame in frames:
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensors.append(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)))
        stacked = np.stack(tensors, axis=1)[None]
        return torch.from_numpy(stacked)

    def _compute_saliency(self, frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        """Run saliency detection on a sequence of frames."""
        if not frames:
            return None

        chunks = []
        total = len(frames)
        start = 0

        while start < total:
            end = min(start + self.clip_len, total)
            batch = list(frames[start:end])
            valid = end - start

            if valid <= 0:
                break

            # Pad batch if needed
            if valid < self.clip_len:
                batch.extend([batch[-1]] * (self.clip_len - valid))

            tensor = self._preprocess_frames(batch)
            saliency = self.backend.predict(tensor).squeeze(0).squeeze(1).numpy()
            chunks.append(saliency[:valid].astype(np.float16))
            start += self.stride

        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)

    def _compute_shot_centers(
        self,
        clip: VideoFileClip,
        fps: float,
        shots: Sequence[Tuple[int, int]],
        bounds: Tuple[int, int, int, int],
    ) -> Tuple[List[Tuple[float, float]], List[dict]]:
        """Compute the focus center for each shot using saliency detection."""
        total_frames = self._estimate_frames(clip, fps)
        frame_centers = [(0.5, 0.5)] * total_frames
        shot_metadata = []

        y1, y2, x1, x2 = bounds
        input_width, input_height = self.backend.input_size

        print("[INFO] Computing saliency for shots...")
        for shot_start, shot_end in shots:
            if shot_end <= shot_start:
                continue

            start_time = shot_start / fps
            end_time = shot_end / fps

            # Clamp to video duration with safety margin
            if clip.duration is not None:
                margin = 0.1
                start_time = min(start_time, max(0.0, clip.duration - margin))
                end_time = min(end_time, clip.duration - margin)

            if end_time <= start_time:
                continue

            # Extract and process frames for this shot
            shot_frames = []
            try:
                shot_clip = clip.subclipped(start_time, end_time)
                frame_limit = shot_end - shot_start

                for idx, frame in enumerate(shot_clip.iter_frames(fps=fps, dtype="uint8")):
                    if idx >= frame_limit:
                        break
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # Apply content mask (ignore letterboxing)
                    masked = np.zeros_like(frame_bgr)
                    masked[y1:y2, x1:x2] = frame_bgr[y1:y2, x1:x2]
                    resized = cv2.resize(masked, (input_width, input_height))
                    shot_frames.append(resized)

                shot_clip.close()
            except Exception as e:
                print(f"[WARN] Failed to process shot {start_time:.1f}s-{end_time:.1f}s: {e}")
                center_norm = (0.5, 0.5)
                for idx in range(shot_start, min(shot_end, total_frames)):
                    frame_centers[idx] = center_norm
                shot_metadata.append({
                    "start_frame": shot_start,
                    "end_frame": shot_end,
                    "center_norm": {"x": 0.5, "y": 0.5},
                })
                continue

            # Compute saliency and find center
            saliency = self._compute_saliency(shot_frames)

            if saliency is None or saliency.ndim != 3:
                center_norm = (0.5, 0.5)
            else:
                cx, cy = find_shot_center(saliency)
                center_norm = (cx / saliency.shape[2], cy / saliency.shape[1])

            # Apply center to all frames in this shot
            for idx in range(shot_start, min(shot_end, total_frames)):
                frame_centers[idx] = center_norm

            shot_metadata.append({
                "start_frame": shot_start,
                "end_frame": shot_end,
                "center_norm": {"x": float(center_norm[0]), "y": float(center_norm[1])},
            })

        return frame_centers, shot_metadata

    def _render_cropped(
        self,
        clip: VideoFileClip,
        fps: float,
        frame_centers: Sequence[Tuple[float, float]],
        shot_metadata: Sequence[dict],
        bounds: Tuple[int, int, int, int],
        output_width: int,
        output_height: int,
    ) -> Tuple[Path, dict]:
        """Render the cropped video to a file."""
        total_frames = self._estimate_frames(clip, fps)
        y1, y2, x1, x2 = bounds

        # Calculate scaling
        frame_height, frame_width = clip.get_frame(0).shape[:2]
        scale = max(output_width / frame_width, output_height / frame_height)
        scaled_width = max(output_width, int(round(frame_width * scale)))
        scaled_height = max(output_height, int(round(frame_height * scale)))

        # Create output video
        output_path = self.workdir / f"cropped_{uuid.uuid4().hex}.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (output_width, output_height),
        )

        progress = tqdm(total=total_frames, desc="Rendering crop", unit="frame", leave=False)
        last_center = (0.5, 0.5)

        try:
            for idx, frame in enumerate(clip.iter_frames(fps=fps, dtype="uint8")):
                if idx >= total_frames:
                    break

                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Apply content mask
                masked = np.zeros_like(frame_bgr)
                masked[y1:y2, x1:x2] = frame_bgr[y1:y2, x1:x2]

                # Get center for this frame
                center = frame_centers[idx] if idx < len(frame_centers) else last_center
                last_center = center

                # Scale and crop
                resized = cv2.resize(masked, (scaled_width, scaled_height))
                cropped = self._crop_frame(resized, center, output_width, output_height)

                writer.write(cropped)
                progress.update(1)
        finally:
            progress.close()
            writer.release()

        crop_metadata = {
            "method": "dynamic_saliency_vinet",
            "fps": float(fps),
            "output_size": [output_width, output_height],
            "source_size": [frame_width, frame_height],
            "content_bounds": {"y1": y1, "y2": y2, "x1": x1, "x2": x2},
            "scale": float(scale),
            "shots": list(shot_metadata),
        }

        return output_path, crop_metadata

    def _crop_frame(
        self,
        frame: np.ndarray,
        center_norm: Tuple[float, float],
        width: int,
        height: int,
    ) -> np.ndarray:
        """Crop a single frame around the given normalized center."""
        h, w = frame.shape[:2]
        cx = int(center_norm[0] * w)
        cy = int(center_norm[1] * h)

        x1 = max(0, min(cx - width // 2, w - width))
        y1 = max(0, min(cy - height // 2, h - height))

        crop = frame[y1:y1 + height, x1:x1 + width]

        # Pad if needed
        if crop.shape[0] != height or crop.shape[1] != width:
            pad_bottom = height - crop.shape[0]
            pad_right = width - crop.shape[1]
            crop = cv2.copyMakeBorder(crop, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT)

        return crop

    def _process_single_clip(
        self,
        clip: VideoFileClip,
        output_width: int,
        output_height: int,
    ) -> Tuple[Path, Optional[dict]]:
        """Process a single clip: detect shots, compute saliency, render cropped."""
        fps = clip.fps or 30.0
        total_frames = self._estimate_frames(clip, fps)

        if total_frames == 0:
            raise RuntimeError("Cannot process clip with zero frames")

        # Detect content bounds (ignore letterboxing)
        bounds = self._get_content_bounds(clip, total_frames)

        # Detect shots
        try:
            shots = ShotDetector.detect(clip, fps, total_frames)
        except Exception as e:
            print(f"[WARN] Shot detection failed: {e}")
            shots = []

        # Ensure we have at least one shot covering the whole clip
        if not shots:
            shots = [(0, total_frames)]

        # Normalize shots to valid ranges
        normalized_shots = []
        cursor = 0
        for start, end in shots:
            start = max(cursor, min(start, total_frames - 1))
            end = max(start + 1, min(end, total_frames))
            normalized_shots.append((start, end))
            cursor = end
            if cursor >= total_frames:
                break

        if not normalized_shots:
            normalized_shots = [(0, total_frames)]
        elif normalized_shots[-1][1] < total_frames:
            normalized_shots.append((normalized_shots[-1][1], total_frames))

        # Compute saliency centers for each shot
        frame_centers, shot_metadata = self._compute_shot_centers(
            clip, fps, normalized_shots, bounds
        )

        # Render cropped video
        output_path, crop_metadata = self._render_cropped(
            clip, fps, frame_centers, shot_metadata, bounds, output_width, output_height
        )

        return output_path, crop_metadata

    async def __call__(
        self,
        desired_width: int,
        desired_height: int,
        clips: Sequence[VideoFileClip],
    ) -> List[VideoFileClip]:
        """
        Process multiple clips with dynamic cropping.

        Args:
            desired_width: Target output width
            desired_height: Target output height
            clips: Input video clips (with audio attached)

        Returns:
            List of cropped VideoFileClips with audio re-attached
        """
        with metrics_collector.track_step("dynamic_cropping"):
            print(f"[INFO] Starting DynamicCropping for {len(clips)} clips")
            results = []

            progress = tqdm(total=len(clips), desc="Dynamic cropping", unit="clip")

            for idx, clip in enumerate(clips):
                print(f"\n[INFO] Processing clip {idx + 1}/{len(clips)}")

                # Process synchronously to avoid thread-safety issues with MoviePy's FFmpeg readers
                output_path, crop_metadata = self._process_single_clip(clip, desired_width, desired_height)

                # Load the cropped video
                cropped_video = VideoFileClip(str(output_path))

                # Re-attach audio using the correct pattern:
                # Attach first, then trim the combined clip (never subclip audio directly)
                if clip.audio is not None:
                    cropped_video = cropped_video.with_audio(clip.audio)
                    if cropped_video.duration > clip.audio.duration:
                        cropped_video = cropped_video.subclipped(0, clip.audio.duration)

                # Transfer metadata
                source_metadata = getattr(clip, "_vea_metadata", None)
                if source_metadata is not None:
                    if crop_metadata:
                        source_metadata["crop"] = crop_metadata
                    source_metadata.setdefault("timeline", {})["duration"] = float(cropped_video.duration or 0)
                    setattr(cropped_video, "_vea_metadata", source_metadata)

                results.append(cropped_video)
                progress.update(1)

            progress.close()
            print("[INFO] DynamicCropping complete")
            return results

    # =========================================================================
    # Test Harness
    # =========================================================================

    @classmethod
    def test_crop_single_video(
        cls,
        video_path: str,
        output_dir: str = "/home/alex/code/vea2/vea-playground/test_outputs/dynamic_crop_outputs",
        output_width: int = 1080,
        output_height: int = 1920,
        max_duration: float = 30.0,
        start_time: float = 0.0,
        save_saliency_preview: bool = True,
    ) -> dict:
        """
        Test dynamic cropping on a single video file.

        Saves outputs to persistent location for debugging:
        - cropped_<name>.mp4: The cropped output video
        - saliency_<name>.json: Shot and saliency metadata
        - preview_<name>.mp4: Optional saliency visualization

        Usage:
            from src.pipelines.common.dynamic_cropping import DynamicCropping
            result = DynamicCropping.test_crop_single_video(
                "/path/to/test_video.mp4",
                output_width=1080,
                output_height=1920,
                max_duration=30.0,  # Only process first 30 seconds
            )
            print(result)

        Args:
            video_path: Path to source video
            output_dir: Where to save outputs
            output_width: Target width (default 1080 for portrait)
            output_height: Target height (default 1920 for portrait)
            max_duration: Maximum duration to process in seconds (default 30s)
            start_time: Start time in seconds (default 0)
            save_saliency_preview: Whether to save saliency visualization

        Returns:
            dict with paths to output files and metadata
        """
        import asyncio
        import json
        import shutil
        from pathlib import Path

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = video_path.stem

        print(f"[TEST] Loading video: {video_path}")
        full_clip = VideoFileClip(str(video_path))
        print(f"[TEST] Full video duration: {full_clip.duration:.2f}s, size: {full_clip.size}, fps: {full_clip.fps}")

        # Trim to max_duration to keep test fast
        end_time = min(start_time + max_duration, full_clip.duration - 0.1)
        if end_time <= start_time:
            raise ValueError(f"Invalid time range: {start_time} to {end_time}")

        clip = full_clip.subclipped(start_time, end_time)
        print(f"[TEST] Processing segment: {start_time:.1f}s to {end_time:.1f}s ({clip.duration:.2f}s)")

        # Create instance
        workdir = output_dir / f"workdir_{stem}"
        workdir.mkdir(exist_ok=True)

        dc = cls(llm=None, workdir=str(workdir), debug=True)

        # Process single clip
        print(f"[TEST] Processing with target size {output_width}x{output_height}")
        cropped_path, crop_metadata = dc._process_single_clip(clip, output_width, output_height)

        # Copy cropped video to output dir
        final_cropped_path = output_dir / f"cropped_{stem}.mp4"
        shutil.copy2(cropped_path, final_cropped_path)

        # Save metadata
        metadata_path = output_dir / f"saliency_{stem}.json"
        with open(metadata_path, "w") as f:
            json.dump(crop_metadata, f, indent=2)

        # Load cropped video and re-attach audio (the key pattern we're testing)
        cropped_clip = VideoFileClip(str(final_cropped_path))
        if clip.audio is not None:
            print(f"[TEST] Re-attaching audio using correct pattern...")
            cropped_clip = cropped_clip.with_audio(clip.audio)
            if cropped_clip.duration > clip.audio.duration:
                cropped_clip = cropped_clip.subclipped(0, clip.audio.duration)

            # Write final video with audio
            final_with_audio = output_dir / f"cropped_with_audio_{stem}.mp4"
            print(f"[TEST] Writing video with audio to: {final_with_audio}")
            try:
                cropped_clip.write_videofile(
                    str(final_with_audio),
                    preset="ultrafast",
                    fps=24,
                    audio_codec="aac",
                )
                print(f"[TEST] SUCCESS: Video with audio written successfully!")
            except Exception as e:
                print(f"[TEST] FAILED: Audio error - {e}")
                final_with_audio = None
            finally:
                cropped_clip.close()
        else:
            final_with_audio = None
            print(f"[TEST] No audio in source video")

        clip.close()
        full_clip.close()

        result = {
            "input_video": str(video_path),
            "cropped_video": str(final_cropped_path),
            "cropped_with_audio": str(final_with_audio) if final_with_audio else None,
            "metadata": str(metadata_path),
            "crop_info": crop_metadata,
        }

        print(f"\n[TEST] Results saved to: {output_dir}")
        print(f"  - Cropped video: {final_cropped_path.name}")
        if final_with_audio:
            print(f"  - With audio: {final_with_audio.name}")
        print(f"  - Metadata: {metadata_path.name}")
        print(f"  - Shots detected: {len(crop_metadata.get('shots', []))}")

        return result
