"""
  Standalone debug script for testing dynamic cropping with chunk-based processing.
  Hardcode your parameters below and run directly:
      python src/pipelines/common/debug_crop.py
  """

import asyncio
import gc
import hashlib
import json
import math
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from moviepy import VideoFileClip
from tqdm import tqdm

# ============================================================================
# HARDCODED PARAMETERS - EDIT THESE
# ============================================================================

# Input video path (local file)
INPUT_VIDEO_PATH = ".cache/gcs_videos/baywatch_h265.mkv"

# Output dimensions (9:16 for vertical video)
OUTPUT_WIDTH = 720
OUTPUT_HEIGHT = 1280

# Debug mode (set True to generate debug preview video)
DEBUG_MODE = True

# Output file path
OUTPUT_PATH = ".cache/cropped_output.mp4"

# Chunk duration in seconds (smaller = more checkpoints, but more overhead)
CHUNK_DURATION_SECONDS = 30.0

# Cache directory for intermediate results
CACHE_DIR = ".cache/crop_cache"

# Use CUDA for ViNet if available
USE_CUDA = torch.cuda.is_available()

# Resume from previous run if cache exists
RESUME_FROM_CACHE = True

# ============================================================================


# ---------------------------
# Helper: black bar detection
# ---------------------------
def detect_content_region(frames: Sequence[np.ndarray], tol: int = 10) -> Tuple[int, int, int, int]:
    """Detect coarse content bounds to ignore letterboxing."""
    h, w = frames[0].shape[:2]
    gray_samples = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

    mean_img = np.mean(np.stack(gray_samples, axis=0), axis=0)
    row_mean = mean_img.mean(axis=1)
    col_mean = mean_img.mean(axis=0)

    y_nonblack = np.where(row_mean > tol)[0]
    x_nonblack = np.where(col_mean > tol)[0]

    if len(y_nonblack) == 0 or len(x_nonblack) == 0:
        return 0, h, 0, w

    y1, y2 = int(y_nonblack[0]), int(y_nonblack[-1] + 1)
    x1, x2 = int(x_nonblack[0]), int(x_nonblack[-1] + 1)
    return y1, y2, x1, x2


# ---------------------------
# Helper: per-shot center from aggregated saliency
# ---------------------------
def find_shot_center(sal_stack: np.ndarray, thr: float = 0.6) -> Tuple[int, int]:
    """Aggregate saliency across a shot and return the strongest blob centroid."""
    if sal_stack.ndim == 3:
        agg = sal_stack.sum(axis=0, dtype=np.float32)
    else:
        agg = sal_stack.astype(np.float32, copy=False)
    if not np.any(agg):
        h, w = agg.shape
        return w // 2, h // 2

    agg_norm = (agg / (agg.max() + 1e-6) * 255).astype(np.uint8)
    _, mask = cv2.threshold(agg_norm, int(thr * agg_norm.max()), 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = agg.shape
    if not cnts:
        return w // 2, h // 2

    contour = max(cnts, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return w // 2, h // 2

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


# ---------------------------
# ViNet backend
# ---------------------------
class ViNetSaliencyBackend:
    """Loads the ViNet saliency model from the local repository."""

    def __init__(
        self,
        repo_dir: Path,
        checkpoint_path: Path,
        variant: str = "S",
        device: Optional[str] = None,
        clip_len: int = 32,
        fp16: bool = True,
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.variant = variant.upper()
        if self.variant not in {"S", "A"}:
            raise ValueError("ViNet variant must be 'S' or 'A'.")

        if device is None:
            device = "cuda" if USE_CUDA else "cpu"
        self.device = torch.device(device)
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        self.clip_len = int(clip_len)
        self.input_size = (384, 224) if self.variant == "S" else (224, 224)

        self.model = self._build_model().eval().to(self.device)
        print(f"[VINET] Loaded ViNet-{self.variant} on {self.device} (fp16={self.fp16}).")

    def _make_args(self):
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

    def _import_model_class(self) -> type[nn.Module]:
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

          # Convert model to FP16 if using half precision
          if self.fp16:
              model = model.half()

          return model

    @torch.no_grad()
    def predict_saliency(self, clip: torch.Tensor) -> torch.Tensor:
        x = clip.to(self.device)
        if self.fp16:
            x = x.half()
        y = self.model(x)
        if y.dtype != torch.float32:
            y = y.float()
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


# ---------------------------
# Shot detection with CUDA support
# ---------------------------
class ShotDetector:
    @staticmethod
    def detect_stream(
        clip: VideoFileClip,
        fps: float,
        total_frames: int,
        use_cuda: bool = False,
    ) -> List[Tuple[int, int]]:
        try:
            from scenedetect import SceneManager
            from scenedetect.detectors import AdaptiveDetector, ContentDetector
            from scenedetect.frame_timecode import FrameTimecode
        except ImportError as exc:
            raise RuntimeError(
                "PySceneDetect is required for dynamic cropping. Install via 'pip install scenedetect[opencv]'."
            ) from exc

        # Note: PySceneDetect doesn't directly support CUDA, but we can use OpenCV's CUDA
        # backend if available. For now, we'll note this in the output.
        if use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("[SHOT_DETECT] OpenCV CUDA is available but PySceneDetect uses CPU")

        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=12, luma_only=False))
        manager.add_detector(AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=12))

        base_timecode = FrameTimecode(timecode=0, fps=fps)
        manager._base_timecode = base_timecode
        manager._start_pos = base_timecode
        manager._last_pos = base_timecode
        if manager._stats_manager is not None:
            manager._stats_manager._base_timecode = base_timecode

        progress_bar = tqdm(
            total=max(total_frames, 1),
            desc="Shot detection",
            unit="frame",
            leave=False,
        )
        try:
            for idx, frame in enumerate(clip.iter_frames(fps=fps, dtype="uint8")):
                if total_frames and idx >= total_frames:
                    break
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if manager._frame_size is None:
                    manager._frame_size = (frame_bgr.shape[1], frame_bgr.shape[0])
                manager._process_frame(idx, frame_bgr, callback=None)
                manager._last_pos = base_timecode + idx
                progress_bar.update(1)
        finally:
            progress_bar.close()

        manager._post_process(max(total_frames - 1, 0))
        scenes = manager.get_scene_list()
        if not scenes:
            return []
        return [(start.get_frames(), end.get_frames()) for start, end in scenes]


# ---------------------------
# Main cropping processor
# ---------------------------
class ChunkedDynamicCropper:
    """Chunk-based dynamic cropping with caching."""

    VINET_REPO_RELATIVE = Path("vinet_v2")
    VINET_CHECKPOINT_RELATIVE = VINET_REPO_RELATIVE / "final_models" / "ViNet_S" / "vinet_s_mvva_randomsplit.pt"

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        output_width: int,
        output_height: int,
        chunk_duration: float,
        cache_dir: Path,
        debug: bool = False,
        resume: bool = True,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_width = output_width
        self.output_height = output_height
        self.chunk_duration = chunk_duration
        self.cache_dir = Path(cache_dir)
        self.debug = debug
        self.resume = resume

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Find ViNet assets
        self.project_root = Path(__file__).resolve().parents[3]
        self.repo_dir = self.project_root / self.VINET_REPO_RELATIVE
        self.checkpoint_path = self.project_root / self.VINET_CHECKPOINT_RELATIVE

        if not self.repo_dir.exists() or not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "ViNet assets not found. Run 'python -m lib.utils.vinet_setup' to clone the repo and download checkpoints."
            )

        self.clip_len = 32
        self.stride = 32
        self.backend = ViNetSaliencyBackend(
            self.repo_dir,
            self.checkpoint_path,
            variant="S",
            clip_len=self.clip_len,
            fp16=USE_CUDA,
        )

        # Create deterministic session ID based on input video + output dimensions
        # This ensures the same video with same settings reuses the same cache
        cache_key = f"{self.input_path.absolute()}_{self.output_width}x{self.output_height}"
        self.session_id = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
        self.session_cache = self.cache_dir / self.session_id
        self.session_cache.mkdir(parents=True, exist_ok=True)

        print(f"[CACHE] Session ID: {self.session_id} (deterministic based on input + dimensions)")
        if self.resume and (self.session_cache / "phase1_analysis.pkl").exists():
            print(f"[CACHE] Found existing cache, will resume from checkpoint")

    def _get_video_info(self, clip: VideoFileClip):
        """Extract video info."""
        fps = clip.fps or 30.0
        duration = clip.duration
        total_frames = int(math.ceil(duration * fps))
        frame_height, frame_width = clip.get_frame(0).shape[:2]
        return {
            "fps": fps,
            "duration": duration,
            "total_frames": total_frames,
            "width": frame_width,
            "height": frame_height,
        }

    def _content_bounds(self, clip: VideoFileClip, total_frames: int) -> Tuple[int, int, int, int]:
        """Detect content bounds (letterbox removal)."""
        sample_count = min(10, total_frames)
        times = np.linspace(0, clip.duration, num=sample_count, endpoint=False)
        frames = [clip.get_frame(float(t)) for t in times]
        return detect_content_region(frames)

    def _preprocess_batch(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """Preprocess frames for ViNet."""
        width, height = self.backend.input_size
        tensors = []
        for frame in frames:
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensors.append(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)))
        stacked = np.stack(tensors, axis=1)[None]
        return torch.from_numpy(stacked)

    def _collect_shot_saliency(self, frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        """Collect saliency for a shot."""
        if not frames:
            return None

        saliency_chunks: List[np.ndarray] = []
        total = len(frames)
        start = 0
        stride = max(1, self.stride)

        while start < total:
            end = min(start + self.clip_len, total)
            batch = list(frames[start:end])
            valid = end - start
            if valid <= 0:
                break

            if valid < self.clip_len:
                batch.extend([batch[-1]] * (self.clip_len - valid))

            tensor = self._preprocess_batch(batch)
            saliency = self.backend.predict_saliency(tensor).squeeze(0).squeeze(1).numpy()
            saliency_chunks.append(saliency[:valid].astype(np.float16, copy=False))
            start += stride

        if not saliency_chunks:
            return None

        return np.concatenate(saliency_chunks, axis=0)

    def phase1_analyze_video(self, clip: VideoFileClip) -> dict:
        """Phase 1: Analyze entire video for shots and frame centers."""
        cache_file = self.session_cache / "phase1_analysis.pkl"

        if self.resume and cache_file.exists():
            print("[PHASE1] Loading cached analysis...")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        print("[PHASE1] Analyzing video...")

        # Get video info
        info = self._get_video_info(clip)
        fps = info["fps"]
        total_frames = info["total_frames"]

        # Detect content bounds
        print("[PHASE1] Detecting content bounds...")
        bounds = self._content_bounds(clip, total_frames)
        y1, y2, x1, x2 = bounds

        # Detect shots
        print("[PHASE1] Detecting shots...")
        try:
            shots = ShotDetector.detect_stream(clip, fps, total_frames, use_cuda=USE_CUDA)
        except Exception as exc:
            print(f"[WARN] Shot detection failed ({exc}); falling back to single-shot processing.")
            shots = []

        if not shots:
            shots = [(0, total_frames)]

        # Normalize shots
        normalized_shots: List[Tuple[int, int]] = []
        cursor = 0
        for start, end in shots:
            start = max(cursor, min(start, max(total_frames - 1, 0)))
            end = max(start + 1, min(end, total_frames))
            normalized_shots.append((start, end))
            cursor = end
            if cursor >= total_frames:
                break
        if not normalized_shots:
            normalized_shots = [(0, total_frames)]
        elif normalized_shots[-1][1] < total_frames:
            normalized_shots.append((normalized_shots[-1][1], total_frames))

        # Calculate frame centers for all shots
        print("[PHASE1] Computing saliency-based centers for all shots...")
        frame_centers = [(0.5, 0.5)] * total_frames
        shot_metadata = []

        input_width, input_height = self.backend.input_size

        for start, end in tqdm(normalized_shots, desc="Analyzing shots", unit="shot"):
            if end <= start:
                continue

            start_time = start / fps
            end_time = end / fps

            if clip.duration is not None:
                # Add larger epsilon for end-of-video to avoid MoviePy EOF issues
                epsilon = 0.1  # 100ms buffer from end
                start_time = min(start_time, max(0.0, clip.duration - epsilon))
                end_time = min(end_time, clip.duration - epsilon)

            if end_time <= start_time:
                continue

            # Skip shots that are too short (less than 1 frame)
            if (end_time - start_time) * fps < 1.0:
                print(f"[WARN] Skipping very short shot at {start_time:.2f}s-{end_time:.2f}s")
                continue

            # Extract shot frames with error handling
            shot_frames: List[np.ndarray] = []
            frame_limit = max(0, end - start)
            shot_clip = None

            try:
                if hasattr(clip, "subclipped"):
                    shot_clip = clip.subclipped(start_time, end_time)
                else:
                    shot_clip = clip.subclip(start_time, end_time)

                for frame_idx, frame in enumerate(shot_clip.iter_frames(fps=fps, dtype="uint8")):
                    if frame_limit and frame_idx >= frame_limit:
                        break
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    masked = np.zeros_like(frame_bgr)
                    masked[y1:y2, x1:x2] = frame_bgr[y1:y2, x1:x2]
                    resized = cv2.resize(masked, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
                    shot_frames.append(resized)
            except (IOError, OSError) as exc:
                print(f"[WARN] Failed to process shot at {start_time:.2f}s-{end_time:.2f}s: {exc}")
                # Use default center for failed shot
                center_norm = (0.5, 0.5)
                for frame_idx in range(start, end):
                    if 0 <= frame_idx < total_frames:
                        frame_centers[frame_idx] = center_norm
                shot_metadata.append({
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "center_norm": {"x": float(center_norm[0]), "y": float(center_norm[1])},
                })
                continue
            finally:
                if shot_clip is not None:
                    getattr(shot_clip, "close", lambda: None)()

            # Compute saliency
            saliency_stack = self._collect_shot_saliency(shot_frames)

            if saliency_stack is None or saliency_stack.ndim != 3 or not saliency_stack.size:
                center_norm = (0.5, 0.5)
            else:
                width = saliency_stack.shape[2]
                height = saliency_stack.shape[1]
                cx, cy = find_shot_center(saliency_stack)
                center_norm = (cx / width, cy / height)

            # Assign center to all frames in shot
            for frame_idx in range(start, end):
                if 0 <= frame_idx < total_frames:
                    frame_centers[frame_idx] = center_norm

            shot_metadata.append({
                "start_frame": int(start),
                "end_frame": int(end),
                "center_norm": {"x": float(center_norm[0]), "y": float(center_norm[1])},
            })

            # Cleanup
            shot_frames.clear()
            saliency_stack = None
            gc.collect()

        analysis = {
            "info": info,
            "bounds": bounds,
            "shots": normalized_shots,
            "shot_metadata": shot_metadata,
            "frame_centers": frame_centers,
        }

        # Save to cache
        with open(cache_file, "wb") as f:
            pickle.dump(analysis, f)

        print(f"[PHASE1] Analysis complete. Found {len(normalized_shots)} shots.")
        return analysis

    def _crop_frame(
        self,
        frame: np.ndarray,
        center_norm: Tuple[float, float],
        bounds: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Crop a single frame based on center."""
        y1, y2, x1, x2 = bounds
        h, w = frame.shape[:2]

        # Apply content bounds mask
        masked = np.zeros_like(frame)
        masked[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        # Scale to fit output
        scale = max(self.output_width / w, self.output_height / h)
        scaled_width = max(self.output_width, int(round(w * scale)))
        scaled_height = max(self.output_height, int(round(h * scale)))
        resized = cv2.resize(masked, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

        # Crop based on center
        center_x = int(center_norm[0] * scaled_width)
        center_y = int(center_norm[1] * scaled_height)
        x1_crop = max(0, min(center_x - self.output_width // 2, scaled_width - self.output_width))
        y1_crop = max(0, min(center_y - self.output_height // 2, scaled_height - self.output_height))

        crop = resized[y1_crop:y1_crop + self.output_height, x1_crop:x1_crop + self.output_width]

        if crop.shape[0] != self.output_height or crop.shape[1] != self.output_width:
            pad_bottom = self.output_height - crop.shape[0]
            pad_right = self.output_width - crop.shape[1]
            crop = cv2.copyMakeBorder(crop, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        return crop

    def phase2_render_chunks(self, clip: VideoFileClip, analysis: dict) -> List[Path]:
        """Phase 2: Render video in chunks with caching."""
        print("[PHASE2] Rendering video in chunks...")

        fps = analysis["info"]["fps"]
        duration = analysis["info"]["duration"]
        bounds = analysis["bounds"]
        frame_centers = analysis["frame_centers"]

        # Calculate chunks
        num_chunks = max(1, int(math.ceil(duration / self.chunk_duration)))
        chunk_paths = []

        for chunk_idx in range(num_chunks):
            chunk_start_time = chunk_idx * self.chunk_duration
            chunk_end_time = min((chunk_idx + 1) * self.chunk_duration, duration)

            chunk_cache_file = self.session_cache / f"chunk_{chunk_idx:04d}.mp4"

            if self.resume and chunk_cache_file.exists():
                print(f"[PHASE2] Chunk {chunk_idx + 1}/{num_chunks} already cached, skipping...")
                chunk_paths.append(chunk_cache_file)
                continue

            print(f"[PHASE2] Rendering chunk {chunk_idx + 1}/{num_chunks} ({chunk_start_time:.1f}s - {chunk_end_time:.1f}s)...")

            # Extract chunk
            if hasattr(clip, "subclipped"):
                chunk_clip = clip.subclipped(chunk_start_time, chunk_end_time)
            else:
                chunk_clip = clip.subclip(chunk_start_time, chunk_end_time)

            # Determine frame range
            start_frame = int(chunk_start_time * fps)
            end_frame = int(chunk_end_time * fps)

            # Create video writer
            writer = cv2.VideoWriter(
                str(chunk_cache_file),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (self.output_width, self.output_height),
            )

            # Process frames
            frame_idx = start_frame
            last_center = (0.5, 0.5)

            for frame in tqdm(
                chunk_clip.iter_frames(fps=fps, dtype="uint8"),
                desc=f"Chunk {chunk_idx + 1}/{num_chunks}",
                unit="frame",
                leave=False,
            ):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Get center for this frame
                if frame_idx < len(frame_centers):
                    center = frame_centers[frame_idx]
                    if center is None:
                        center = last_center
                else:
                    center = last_center
                last_center = center

                # Crop frame
                cropped = self._crop_frame(frame_bgr, center, bounds)
                writer.write(cropped)

                frame_idx += 1

            writer.release()
            getattr(chunk_clip, "close", lambda: None)()

            chunk_paths.append(chunk_cache_file)
            gc.collect()

        print(f"[PHASE2] Rendered {len(chunk_paths)} chunks.")
        return chunk_paths

    def phase3_combine_chunks(self, chunk_paths: List[Path]) -> None:
        """Phase 3: Combine chunks and add audio using ffmpeg."""
        print("[PHASE3] Combining chunks with ffmpeg...")

        # Create concat file
        concat_file = self.session_cache / "concat_list.txt"
        with open(concat_file, "w") as f:
            for chunk_path in chunk_paths:
                f.write(f"file '{chunk_path.absolute()}'\n")

        # First, concatenate video chunks (no audio)
        video_only_path = self.session_cache / "video_only.mp4"
        cmd_concat = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            str(video_only_path),
        ]

        print(f"[PHASE3] Concatenating video chunks...")
        result = subprocess.run(cmd_concat, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[ERROR] ffmpeg concatenation failed: {result.stderr}")
            raise RuntimeError("Failed to concatenate chunks with ffmpeg")

        # Now add audio from the original video
        print(f"[PHASE3] Adding audio from original video...")
        cmd_audio = [
            "ffmpeg",
            "-y",
            "-i", str(video_only_path),
            "-i", str(self.input_path),
            "-c:v", "copy",  # Copy video stream (already encoded)
            "-map", "0:v:0",  # Video from concatenated file
            "-map", "1:a:0?",  # Audio from original (? = optional, won't fail if no audio)
            "-c:a", "aac",  # Encode audio as AAC
            "-b:a", "192k",  # Audio bitrate
            "-shortest",  # Match shortest stream (in case audio is longer)
            str(self.output_path),
        ]

        result = subprocess.run(cmd_audio, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[WARN] Failed to add audio: {result.stderr}")
            print(f"[WARN] Falling back to video-only output")
            # Copy video-only file as output
            shutil.copy2(video_only_path, self.output_path)
        else:
            print(f"[PHASE3] Successfully added audio to output")

        # Cleanup intermediate file
        video_only_path.unlink(missing_ok=True)

        print(f"[PHASE3] Final output saved to {self.output_path}")

    def process(self):
        """Main processing pipeline."""
        print(f"[CROP] Session ID: {self.session_id}")
        print(f"[CROP] Cache dir: {self.session_cache}")
        print(f"[CROP] Input: {self.input_path}")
        print(f"[CROP] Output: {self.output_path}")
        print(f"[CROP] CUDA available: {USE_CUDA}")

        # Load video
        print("[CROP] Loading video...")
        clip = VideoFileClip(str(self.input_path))
        print(f"[CROP] Video loaded: {clip.duration:.2f}s @ {clip.fps}fps, {clip.size}")

        try:
            # Phase 1: Analyze
            analysis = self.phase1_analyze_video(clip)

            # Phase 2: Render chunks
            chunk_paths = self.phase2_render_chunks(clip, analysis)

            # Phase 3: Combine
            self.phase3_combine_chunks(chunk_paths)

            print(f"\n[SUCCESS] ✓ Cropped video saved to: {self.output_path}")

        finally:
            clip.close()

        # Cleanup cache if requested
        if not self.resume:
            print(f"[CLEANUP] Removing cache: {self.session_cache}")
            shutil.rmtree(self.session_cache, ignore_errors=True)


async def main():
    """Run chunked dynamic cropping."""
    print("=" * 80)
    print("CHUNKED DYNAMIC CROPPING DEBUG SCRIPT")
    print("=" * 80)

    cropper = ChunkedDynamicCropper(
        input_path=INPUT_VIDEO_PATH,
        output_path=OUTPUT_PATH,
        output_width=OUTPUT_WIDTH,
        output_height=OUTPUT_HEIGHT,
        chunk_duration=CHUNK_DURATION_SECONDS,
        cache_dir=CACHE_DIR,
        debug=DEBUG_MODE,
        resume=RESUME_FROM_CACHE,
    )

    cropper.process()


if __name__ == "__main__":
    asyncio.run(main())