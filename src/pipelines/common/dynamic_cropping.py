import os
import cv2
import uuid
import numpy as np
from PIL import Image
from pathlib import Path
from moviepy import VideoFileClip
import tempfile
import asyncio
import mediapipe as mp
from src.pipelines.common.schema import CropModeResponse, GeneralCropCenterResponse
from scipy.signal import medfilt, savgol_filter
import gc

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class DynamicCropping:
    def __init__(self, llm, workdir=None, debug=False):
        self.llm = llm
        self.workdir = Path(workdir or tempfile.mkdtemp())
        self.debug = debug
        self.overlay_font = cv2.FONT_HERSHEY_SIMPLEX
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _sample_frames(self, clip, num=8):
        duration = clip.duration
        times = np.linspace(0, duration, num=num + 2)[1:-1]
        sampled = []
        for i, t in enumerate(times):
            frame = clip.get_frame(t)
            path = self.workdir / f"sample_{i}.jpg"
            Image.fromarray(frame).save(path)
            sampled.append({"frame_id": f"frame_{i}", "t": t, "path": path})
        return sampled

    async def _decide_crop_mode(self, frames):
        prompt = (
            "You are analyzing a sequence of video frames to decide the best cropping strategy.\n"
            "There are two modes:\n"
            "- `character_tracking`: Use this when there is a clear face or person that appears in at least some of the frames.\n"
            "- `general`: Use this when the frame contains no prominent subject (landscapes, buildings, crowds, or vehicles).\n\n"
            "Prefer `character_tracking` if a prominent face or body is present in most frames. Prefer `general` if there’s no primary visual subject.\n"
            "Return a JSON object with a single key `mode`, with value either 'general' or 'character_tracking'."
        )
        try:
            images = [Path(f["path"]) for f in frames]
            resp = await asyncio.to_thread(
                self.llm.LLM_request,
                images + [prompt],
                CropModeResponse
            )
            print(f"[DEBUG] Gemini crop mode decision: {resp}")
            return resp["mode"]
        except Exception as e:
            print(f"[ERROR] Failed to decide crop mode: {e}")
            return "general"

    async def _get_general_crop_center(self, frames, w, h):
        centers = []
        for frame in frames:
            try:
                prompt = (
                    "You are a professional video editor. Determine the crop center for this frame.\n"
                    "Return a JSON object with `crop_center_x` and `crop_center_y` between 0 and 1."
                )
                img_path = Path(frame["path"])
                result = await asyncio.to_thread(
                    self.llm.LLM_request,
                    [img_path, prompt],
                    GeneralCropCenterResponse
                )
                print(f"[DEBUG] Gemini general crop center result: {result}")
                centers.append((result["crop_center_x"], result["crop_center_y"]))
            except Exception as e:
                print(f"[ERROR] Failed to get crop center for frame {frame['frame_id']}: {e}")
        if not centers:
            return 0.5, 0.5
        avg_x = float(np.mean([c[0] for c in centers]))
        avg_y = float(np.mean([c[1] for c in centers]))
        print(f"[DEBUG] Averaged crop center: ({avg_x}, {avg_y})")
        return avg_x, avg_y

    def _extract_bbox(self, landmarks, indices, image_shape):
        h, w, _ = image_shape
        xs = [landmarks[i].x * w for i in indices if landmarks[i].visibility > 0.5]
        ys = [landmarks[i].y * h for i in indices if landmarks[i].visibility > 0.5]
        if not xs or not ys:
            return None
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    def _add_mode_watermark(self, frame, mode):
        overlay = frame.copy()
        text = f"Mode: {mode}"
        for y in range(0, frame.shape[0], 100):
            for x in range(0, frame.shape[1], 300):
                cv2.putText(overlay, text, (x, y + 30), self.overlay_font, 1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    def _smooth_centers(self, centers):
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        xs_med = medfilt(xs, kernel_size=35)
        ys_med = medfilt(ys, kernel_size=35)
        window = min(len(xs_med) // 2 * 2 + 1, 51)
        poly = 2 if window >= 5 else 1
        xs_smooth = savgol_filter(xs_med, window_length=window, polyorder=poly)
        ys_smooth = savgol_filter(ys_med, window_length=window, polyorder=poly)
        return list(zip(xs_smooth, ys_smooth))

    def _track_subject_streaming(self, clip, desired_width, desired_height, mode, out_path):
        fps = clip.fps
        w, h = clip.size
        scale = max(desired_width / w, desired_height / h)
        sw, sh = int(w * scale), int(h * scale)

        last_valid = (0.5, 0.5)
        centers = []

        # First pass: extract focus centers
        for t in np.arange(0, clip.duration, 1 / fps):
            frame = clip.get_frame(t)
            frame = cv2.resize(frame, (sw, sh))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.pose.process(frame_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                face_bbox = self._extract_bbox(landmarks, [0, 1, 2, 3, 4], frame.shape)
                body_bbox = self._extract_bbox(landmarks, [11, 12, 23, 24], frame.shape)
                target = face_bbox or body_bbox
                if target:
                    x1, y1, x2, y2 = target
                    cx = (x1 + x2) / 2 / sw
                    cy = (y1 + y2) / 2 / sh
                    last_valid = (cx, cy)
            centers.append(last_valid)

        smoothed = self._smooth_centers(centers)

        # Second pass: write cropped frames
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (desired_width, desired_height))
        for i, t in enumerate(np.arange(0, clip.duration, 1 / fps)):
            frame = clip.get_frame(t)
            frame = cv2.resize(frame, (sw, sh))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self.debug:
                frame = self._add_mode_watermark(frame, mode)

            cx, cy = smoothed[i]
            center_x = int(cx * sw)
            center_y = int(cy * sh)
            x1 = max(0, min(center_x - desired_width // 2, sw - desired_width))
            y1 = max(0, min(center_y - desired_height // 2, sh - desired_height))
            crop = frame[y1:y1 + desired_height, x1:x1 + desired_width]
            out.write(crop)

        out.release()

    async def __call__(self, desired_width, desired_height, clips):
        print(f"[INFO] Starting DynamicCropping for {len(clips)} clips.")
        final_clips = []

        for idx, clip in enumerate(clips):
            print(f"\n[INFO] Processing clip {idx}")
            sampled = self._sample_frames(clip)
            mode = await self._decide_crop_mode(sampled)
            print(f"[INFO] Chosen mode: {mode}")

            temp_path = str(self.workdir / f"cropped_{uuid.uuid4().hex}.mp4")

            if mode == "character_tracking":
                self._track_subject_streaming(clip, desired_width, desired_height, mode, temp_path)
            else:
                cx, cy = await self._get_general_crop_center(sampled, *clip.size)
                w, h = clip.size
                scale = max(desired_width / w, desired_height / h)
                sw, sh = int(w * scale), int(h * scale)
                fps = clip.fps

                out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (desired_width, desired_height))
                for t in np.arange(0, clip.duration, 1 / fps):
                    frame = cv2.resize(clip.get_frame(t), (sw, sh))
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if self.debug:
                        frame = self._add_mode_watermark(frame, mode)

                    center_x = int(cx * sw)
                    center_y = int(cy * sh)
                    x1 = max(0, min(center_x - desired_width // 2, sw - desired_width))
                    y1 = max(0, min(center_y - desired_height // 2, sh - desired_height))
                    crop = frame[y1:y1 + desired_height, x1:x1 + desired_width]
                    out.write(crop)
                out.release()

            video = VideoFileClip(temp_path)
            if clip.audio:
                safe_duration = min(video.duration, clip.audio.duration)
                video = video.with_audio(clip.audio.subclipped(0, safe_duration))
            final_clips.append(video)

            gc.collect()

        print("[INFO] DynamicCropping complete.")
        return final_clips
