import asyncio
import gc
import math
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from moviepy import VideoFileClip
from tqdm import tqdm


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


class DebugCropHelper:
    """Utility to mimic the 9:16 crop preview used in the standalone script."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def crop(self, frame: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        h, w = frame.shape[:2]
        crop_h = int(h * 0.8)
        crop_w = int(crop_h * 9 / 16)
        cx, cy = center
        x1 = max(0, min(w - crop_w, cx - crop_w // 2))
        y1 = max(0, min(h - crop_h, cy - crop_h // 2))
        return frame[y1 : y1 + crop_h, x1 : x1 + crop_w]


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
            device = "cpu"
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
# Shot detection
# ---------------------------
class ShotDetector:
    @staticmethod
    def detect_stream(
        clip: VideoFileClip,
        fps: float,
        total_frames: int,
    ) -> List[Tuple[int, int]]:
        try:
            from scenedetect import SceneManager
            from scenedetect.detectors import AdaptiveDetector, ContentDetector
            from scenedetect.frame_timecode import FrameTimecode
        except ImportError as exc:
            raise RuntimeError(
                "PySceneDetect is required for dynamic cropping. Install via 'pip install scenedetect[opencv]'."
            ) from exc

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
# Dynamic cropping implementation
# ---------------------------
class DynamicCropping:
    """9:16 intelligent cropping using ViNet saliency + shot detection."""

    VINET_REPO_RELATIVE = Path("vinet_v2")
    VINET_CHECKPOINT_RELATIVE = VINET_REPO_RELATIVE / "final_models" / "ViNet_S" / "vinet_s_mvva_randomsplit.pt"

    def __init__(self, llm, workdir: Optional[str] = None, debug: bool = False):
        self.llm = llm  # retained for interface compatibility
        self.workdir = Path(workdir or tempfile.mkdtemp())
        self.debug = debug
        self.overlay_font = cv2.FONT_HERSHEY_SIMPLEX

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
            fp16=False,
        )
        self._temp_outputs: set[Path] = set()
        self._clip_finalizers: List[weakref.finalize] = []
        self._debug_outputs: List[Path] = []
        self._current_debug = None

    def cleanup(self) -> None:
        for finalizer in self._clip_finalizers:
            try:
                finalizer()
            except Exception:
                pass
        self._clip_finalizers.clear()

        for temp_path in list(self._temp_outputs):
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._temp_outputs.discard(temp_path)
        if self.workdir.exists() and not any(self.workdir.iterdir()):
            shutil.rmtree(self.workdir, ignore_errors=True)

    def _register_clip_cleanup(self, clip: VideoFileClip, temp_path: Path) -> None:
        self._temp_outputs.add(temp_path)
        self_ref = weakref.ref(self)

        def _cleanup(self_ref: weakref.ReferenceType, path: Path) -> None:
            self_obj = self_ref()
            if self_obj is None:
                return
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            self_obj._temp_outputs.discard(path)
            if self_obj.workdir.exists() and not any(self_obj.workdir.iterdir()):
                shutil.rmtree(self_obj.workdir, ignore_errors=True)

        finalizer = weakref.finalize(clip, _cleanup, self_ref, temp_path)
        self._clip_finalizers.append(finalizer)

    def _estimate_total_frames(self, clip: VideoFileClip, fps: float) -> int:
        if clip.duration is None or fps <= 0:
            return 0
        return max(1, int(math.ceil(clip.duration * fps)))

    def _content_bounds(self, clip: VideoFileClip, total_frames: int) -> Tuple[int, int, int, int]:
        sample_count = min(10, total_frames)
        times = np.linspace(0, clip.duration, num=sample_count, endpoint=False)
        frames = [clip.get_frame(float(t)) for t in times]
        return detect_content_region(frames)

    def _preprocess_batch(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        width, height = self.backend.input_size
        tensors = []
        for frame in frames:
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensors.append(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)))
        stacked = np.stack(tensors, axis=1)[None]
        return torch.from_numpy(stacked)

    def _start_debug_preview(
        self,
        clip: VideoFileClip,
        shots: Sequence[Tuple[int, int]],
        fps: float,
    ) -> None:
        if not self.debug:
            return

        input_width, input_height = self.backend.input_size
        debug_dir = self.workdir / "dynamic_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        tmp_path = debug_dir / f"dynamic_preview_{uuid.uuid4().hex}.noaudio.mp4"
        final_path = tmp_path.with_suffix(".mp4")
        writer = cv2.VideoWriter(
            str(tmp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (input_width * 2, input_height * 2),
        )

        cut_positions = {start for start, _ in shots[1:]} if len(shots) > 1 else set()
        source_path = getattr(clip, "filename", None)

        self._current_debug = {
            "writer": writer,
            "tmp_path": tmp_path,
            "final_path": final_path,
            "cut_positions": cut_positions,
            "cropper": DebugCropHelper(input_width, input_height),
            "input_size": (input_width, input_height),
            "source_path": Path(source_path).resolve() if source_path else None,
        }
        print(f"[DEBUG] Dynamic cropping preview temp file: {tmp_path}")

    def _normalize_debug_saliency(self, saliency: np.ndarray) -> np.ndarray:
        mmin = float(saliency.min())
        mmax = float(saliency.max())
        if mmax - mmin < 1e-6:
            return np.zeros_like(saliency, dtype=np.float32)
        return ((saliency - mmin) / (mmax - mmin)).astype(np.float32)

    def _debug_crop_frame(self, frame: np.ndarray, center: Tuple[int, int], cropper: DebugCropHelper) -> np.ndarray:
        input_height, input_width = frame.shape[0], frame.shape[1]
        cropped = cropper.crop(frame, center)
        target_h = input_height
        target_w = int(round(target_h * 9 / 16))
        target_w = max(1, min(target_w, input_width))
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            cropped = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else:
            cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        pad_left = max(0, (input_width - target_w) // 2)
        pad_right = max(0, input_width - target_w - pad_left)
        if pad_left > 0 or pad_right > 0:
            cropped = cv2.copyMakeBorder(cropped, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return cropped

    def _write_debug_preview(
        self,
        shot_frames: Sequence[np.ndarray],
        saliency_stack: Optional[np.ndarray],
        center_px: Tuple[int, int],
        shot_start: int,
    ) -> None:
        if not self.debug or not self._current_debug:
            return

        debug_ctx = self._current_debug
        writer: cv2.VideoWriter = debug_ctx["writer"]
        input_width, input_height = debug_ctx["input_size"]
        cropper: DebugCropHelper = debug_ctx["cropper"]
        cut_positions = debug_ctx["cut_positions"]

        if saliency_stack is None or saliency_stack.ndim != 3:
            saliency_stack = np.zeros((len(shot_frames), input_height, input_width), dtype=np.float32)

        for idx, frame in enumerate(shot_frames):
            if idx < saliency_stack.shape[0]:
                saliency_map = saliency_stack[idx]
            else:
                saliency_map = np.zeros((input_height, input_width), dtype=np.float32)

            frame_resized = cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
            frame_vis = frame_resized.copy()
            cv2.drawMarker(frame_vis, center_px, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

            heatmap = self._normalize_debug_saliency(saliency_map)
            heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            if (shot_start + idx) in cut_positions:
                cv2.line(heatmap_color, (0, 0), (input_width, 0), (0, 0, 255), thickness=6)

            cropped = self._debug_crop_frame(frame_resized, center_px, cropper)
            blank = np.zeros_like(cropped)
            top = np.hstack([frame_vis, heatmap_color])
            bottom = np.hstack([cropped, blank])
            grid = np.vstack([top, bottom])
            writer.write(grid)

    def _finalize_debug_preview(self) -> None:
        if not self.debug or not self._current_debug:
            return

        debug_ctx = self._current_debug
        writer: cv2.VideoWriter = debug_ctx["writer"]
        writer.release()

        tmp_path: Path = debug_ctx["tmp_path"]
        final_path: Path = debug_ctx["final_path"]
        source_path: Optional[Path] = debug_ctx["source_path"]

        output_path = tmp_path
        if source_path and source_path.exists():
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(tmp_path),
                "-i",
                str(source_path),
                "-c",
                "copy",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                str(final_path),
            ]
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0 and final_path.exists():
                tmp_path.unlink(missing_ok=True)
                output_path = final_path
            else:
                if final_path.exists():
                    final_path.unlink()
        else:
            if final_path.exists():
                final_path.unlink()
            try:
                tmp_path.rename(final_path)
                output_path = final_path
            except OSError:
                shutil.copy(tmp_path, final_path)
                output_path = final_path

        print(f"[DEBUG] Saved dynamic cropping preview to {output_path}")
        self._debug_outputs.append(Path(output_path))
        self._current_debug = None

    def _collect_shot_saliency(self, frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
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

    def _compute_frame_centers(
        self,
        clip: VideoFileClip,
        fps: float,
        shots: Sequence[Tuple[int, int]],
        bounds: Tuple[int, int, int, int],
    ) -> Tuple[List[Tuple[float, float]], List[dict]]:
        total_frames = self._estimate_total_frames(clip, fps)
        frame_centers: List[Tuple[float, float]] = [(0.5, 0.5)] * total_frames
        shot_metadata: List[dict] = []

        if hasattr(clip, "reader") and hasattr(clip.reader, "initialize"):
            try:
                clip.reader.initialize()
            except Exception:
                pass

        y1, y2, x1, x2 = bounds
        input_width, input_height = self.backend.input_size
        print("[INFO] Aggregating saliency for dynamic cropping shots...")
        for start, end in shots:
            if end <= start:
                continue
            start_time = start / fps
            end_time = end / fps
            if clip.duration is not None:
                epsilon = 1e-4
                start_time = min(start_time, max(0.0, clip.duration - epsilon))
                end_time = min(end_time, clip.duration)
            if end_time <= start_time:
                continue

            if hasattr(clip, "subclipped"):
                shot_clip = clip.subclipped(start_time, end_time)
            else:
                shot_clip = clip.subclip(start_time, end_time)

            shot_frames: List[np.ndarray] = []
            frame_limit = max(0, end - start)
            try:
                for frame_idx, frame in enumerate(shot_clip.iter_frames(fps=fps, dtype="uint8")):
                    if frame_limit and frame_idx >= frame_limit:
                        break
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    masked = np.zeros_like(frame_bgr)
                    masked[y1:y2, x1:x2] = frame_bgr[y1:y2, x1:x2]
                    resized = cv2.resize(masked, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
                    shot_frames.append(resized)
            finally:
                getattr(shot_clip, "close", lambda: None)()

            saliency_stack = self._collect_shot_saliency(shot_frames)

            if saliency_stack is None or saliency_stack.ndim != 3 or not saliency_stack.size:
                center_norm = (0.5, 0.5)
                center_px = (input_width // 2, input_height // 2)
            else:
                width = saliency_stack.shape[2]
                height = saliency_stack.shape[1]
                cx, cy = find_shot_center(saliency_stack)
                center_px = (cx, cy)
                center_norm = (cx / width, cy / height)

            if self.debug:
                self._write_debug_preview(shot_frames, saliency_stack, center_px, start)

            shot_frames.clear()
            saliency_stack = None

            for frame_idx in range(start, end):
                if 0 <= frame_idx < total_frames:
                    frame_centers[frame_idx] = center_norm

            shot_metadata.append(
                {
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "center_norm": {
                        "x": float(center_norm[0]),
                        "y": float(center_norm[1]),
                    },
                }
            )

        return frame_centers, shot_metadata

    def _render_clip(
        self,
        clip: VideoFileClip,
        fps: float,
        frame_centers: Sequence[Tuple[float, float]],
        shot_metadata: Sequence[dict],
        bounds: Tuple[int, int, int, int],
        desired_width: int,
        desired_height: int,
    ) -> Tuple[Path, dict]:
        total_frames = self._estimate_total_frames(clip, fps)
        y1, y2, x1, x2 = bounds

        frame_height, frame_width = clip.get_frame(0).shape[:2]
        scale = max(desired_width / frame_width, desired_height / frame_height)
        scaled_width = max(desired_width, int(round(frame_width * scale)))
        scaled_height = max(desired_height, int(round(frame_height * scale)))

        temp_output = self.workdir / f"dynamic_crop_{uuid.uuid4().hex}.mp4"
        writer = cv2.VideoWriter(
            str(temp_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (desired_width, desired_height),
        )
        frames_written = 0

        progress = tqdm(
            total=max(total_frames, 1),
            desc="Cropping frames",
            unit="frame",
            leave=False,
        )
        try:
            if hasattr(clip, "reader") and hasattr(clip.reader, "initialize"):
                try:
                    clip.reader.initialize()
                except Exception:
                    pass

            last_center = (0.5, 0.5)
            for idx, frame in enumerate(clip.iter_frames(fps=fps, dtype="uint8")):
                if idx >= total_frames:
                    break
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                masked = np.zeros_like(frame_bgr)
                masked[y1:y2, x1:x2] = frame_bgr[y1:y2, x1:x2]

                center = frame_centers[idx] if idx < len(frame_centers) else last_center
                if center is None:
                    center = last_center
                last_center = center

                resized = cv2.resize(masked, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
                cropped = self._crop_frame(resized, center, desired_width, desired_height, False)
                writer.write(cropped)
                frames_written += 1
                progress.update(1)
        finally:
            progress.close()
            writer.release()
        if frames_written == 0:
            fallback_writer = cv2.VideoWriter(
                str(temp_output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(fps, 1.0),
                (desired_width, desired_height),
            )
            try:
                sample_frame = clip.get_frame(0)
                sample_bgr = cv2.cvtColor(sample_frame, cv2.COLOR_RGB2BGR)
                masked = np.zeros_like(sample_bgr)
                masked[y1:y2, x1:x2] = sample_bgr[y1:y2, x1:x2]
                fallback_center = frame_centers[0] if frame_centers else (0.5, 0.5)
                resized = cv2.resize(masked, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
                fallback_cropped = self._crop_frame(resized, fallback_center, desired_width, desired_height, False)
            except Exception:
                fallback_cropped = np.zeros((desired_height, desired_width, 3), dtype=np.uint8)
            fallback_writer.write(fallback_cropped)
            fallback_writer.release()
        crop_metadata = {
            "method": "dynamic_saliency_vinet",
            "fps": float(fps),
            "output_size": [int(desired_width), int(desired_height)],
            "source_size": [int(frame_width), int(frame_height)],
            "content_bounds": {
                "y1": int(y1),
                "y2": int(y2),
                "x1": int(x1),
                "x2": int(x2),
            },
            "scale": float(scale),
            "shots": list(shot_metadata),
        }

        return temp_output, crop_metadata

    def _crop_frame(
        self,
        frame: np.ndarray,
        center_norm: Tuple[float, float],
        desired_width: int,
        desired_height: int,
        debug_cut: bool,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        center_x = int(center_norm[0] * w)
        center_y = int(center_norm[1] * h)
        x1 = max(0, min(center_x - desired_width // 2, w - desired_width))
        y1 = max(0, min(center_y - desired_height // 2, h - desired_height))
        crop = frame[y1:y1 + desired_height, x1:x1 + desired_width]
        if crop.shape[0] != desired_height or crop.shape[1] != desired_width:
            pad_bottom = desired_height - crop.shape[0]
            pad_right = desired_width - crop.shape[1]
            crop = cv2.copyMakeBorder(crop, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if self.debug:
            crop = self._apply_debug_overlay(crop, (desired_width // 2, desired_height // 2), debug_cut)
        return crop

    def _apply_debug_overlay(self, frame: np.ndarray, center: Tuple[int, int], cut: bool) -> np.ndarray:
        overlay = frame.copy()
        cv2.drawMarker(overlay, center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        if cut:
            cv2.line(overlay, (0, 0), (overlay.shape[1], 0), (0, 0, 255), thickness=4)
        return overlay

    def _process_clip(self, clip: VideoFileClip, desired_width: int, desired_height: int) -> Tuple[Path, dict | None]:
        fps = clip.fps or 30.0
        total_frames = self._estimate_total_frames(clip, fps)
        if total_frames == 0:
            raise RuntimeError("Failed to estimate frame count for dynamic cropping.")

        y1, y2, x1, x2 = self._content_bounds(clip, total_frames)
        bounds = (y1, y2, x1, x2)

        try:
            shots = ShotDetector.detect_stream(clip, fps, total_frames)
        except Exception as exc:
            print(f"[WARN] Shot detection failed ({exc}); falling back to single-shot processing.")
            shots = []

        if not shots:
            shots = [(0, total_frames)]

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

        if self.debug:
            self._start_debug_preview(clip, normalized_shots, fps)

        try:
            frame_centers, shot_metadata = self._compute_frame_centers(
                clip, fps, normalized_shots, bounds
            )
            temp_output, crop_metadata = self._render_clip(
                clip,
                fps,
                frame_centers,
                shot_metadata,
                bounds,
                desired_width,
                desired_height,
            )
            return temp_output, crop_metadata
        finally:
            if self.debug:
                self._finalize_debug_preview()

    async def __call__(self, desired_width: int, desired_height: int, clips: Sequence[VideoFileClip]):
        print(f"[INFO] Starting DynamicCropping for {len(clips)} clips using ViNet saliency.")
        final_clips = []

        clip_progress = tqdm(total=len(clips), desc="DynamicCropping clips", unit="clip")

        for idx, clip in enumerate(clips):
            print(f"\n[INFO] Processing clip {idx}")
            temp_result = await asyncio.to_thread(self._process_clip, clip, desired_width, desired_height)

            if isinstance(temp_result, tuple):
                temp_path, crop_metadata = temp_result
            else:
                temp_path, crop_metadata = temp_result, None

            video = VideoFileClip(str(temp_path))
            self._register_clip_cleanup(video, temp_path)
            if clip.audio:
                safe_duration = min(video.duration, clip.audio.duration)
                video = video.with_audio(clip.audio.subclipped(0, safe_duration))

            source_metadata = getattr(clip, "_vea_metadata", None)
            if source_metadata is not None:
                if crop_metadata is not None:
                    source_metadata["crop"] = crop_metadata
                source_metadata.setdefault("timeline", {})["duration"] = float(video.duration or 0.0)
                applied_start = source_metadata.setdefault("timings", {}).get("applied_start", 0.0)
                source_metadata["timings"]["applied_end"] = applied_start + float(video.duration or 0.0)
                setattr(video, "_vea_metadata", source_metadata)

            final_clips.append(video)
            gc.collect()
            clip_progress.update(1)

        clip_progress.close()

        if self.debug and self._debug_outputs:
            print("[DEBUG] Dynamic cropping previews saved:")
            for path in self._debug_outputs:
                print(f"    {path}")

        print("[INFO] DynamicCropping complete.")
        return final_clips