import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from moviepy import VideoFileClip, ImageSequenceClip
import tempfile
import asyncio
from ultralytics import YOLO
from src.pipelines.common.schema import CropModeResponse, GeneralCropCenterResponse
import uuid


class DynamicCropping:
    def __init__(self, llm, workdir=None):
        self.llm = llm
        self.workdir = Path(workdir or tempfile.mkdtemp())
        self.yolo_model = YOLO("yolov8n.pt")

    def _sample_frames(self, clip, num=5):
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
            "- `general`: Use this when there is no clear main character or there are many people (like crowds or background scenes).\n"
            "- `character_tracking`: Use this when there is a clear main person or subject that appears across the frames.\n\n"
            "Return a JSON object with a single key `mode`, with value either 'general' or 'character_tracking'."
        )
        try:
            images = [Path(f["path"]) for f in frames]
            resp = await asyncio.to_thread(
                self.llm.LLM_request,
                images + [prompt],
                {
                    "response_mime_type": "application/json",
                    "response_schema": CropModeResponse,
                },
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
                    {
                        "response_mime_type": "application/json",
                        "response_schema": GeneralCropCenterResponse,
                    },
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

    def _smooth_centers(self, centers, alpha=0.8):
        smoothed = [centers[0]]
        for i in range(1, len(centers)):
            prev = smoothed[-1]
            new = (
                alpha * prev[0] + (1 - alpha) * centers[i][0],
                alpha * prev[1] + (1 - alpha) * centers[i][1],
            )
            smoothed.append(new)
        return smoothed
    
    def _detect_and_fix_outliers(self, centers, jerk_threshold=0.25, window=1):
        centers = np.array(centers)
        diffs = np.linalg.norm(np.diff(centers, axis=0), axis=1)

        outliers = []
        for i, d in enumerate(diffs):
            if d > jerk_threshold:
                outliers.append(i + 1)  # Mark the frame after the jump

        for i in outliers:
            start = max(0, i - window)
            end = min(len(centers) - 1, i + window)
            if start < i < end:
                prev = centers[start]
                next = centers[end]
                interp = (prev + next) / 2
                centers[i] = interp
                print(f"[DEBUG] Replaced outlier at index {i} with interpolated center: {interp.tolist()}")

        return centers.tolist()


    def _track_character(self, clip, desired_width, desired_height):
        fps = clip.fps
        original_w, original_h = clip.size
        scale = max(desired_width / original_w, desired_height / original_h)
        scaled_w, scaled_h = int(original_w * scale), int(original_h * scale)

        frames = []
        bboxes_per_frame = []

        print(f"[INFO] Running YOLO tracking with ByteTrack...")

        for t in np.arange(0, clip.duration, 1 / fps):
            frame = clip.get_frame(t)
            frame = cv2.resize(frame, (scaled_w, scaled_h))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            result = self.yolo_model.track(
                frame_rgb,
                tracker="bytetrack.yaml",
                persist=True,
                imgsz=640,
                conf=0.3,
                verbose=False,
            )[0]

            detections = []
            if result is not None and result.boxes is not None:
                for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                    x1, y1, x2, y2 = box.tolist()
                    area = (x2 - x1) * (y2 - y1)
                    cx = (x1 + x2) / 2 / scaled_w
                    cy = (y1 + y2) / 2 / scaled_h
                    label = self.yolo_model.names[int(cls)]
                    if label in {"face", "person"}:
                        detections.append((label, area, cx, cy))

            frames.append(frame)
            bboxes_per_frame.append(detections)

        # --- First pass: select target per frame ---
        targets = []
        last_target = None
        for i, detections in enumerate(bboxes_per_frame):
            if not detections:
                targets.append(None)
                continue

            detections.sort(key=lambda x: x[1], reverse=True)  # Sort by area
            best = detections[0]
            if len(detections) > 1:
                second = detections[1]
                if abs(best[1] - second[1]) / best[1] < 0.1 and last_target:
                    best = last_target
                else:
                    last_target = best
            else:
                last_target = best
            targets.append(best)

        # --- Second pass: enforce 30-frame stability ---
        stable_targets = []
        i = 0
        while i < len(targets):
            current = targets[i]
            j = i + 1
            while j < len(targets) and targets[j] == current:
                j += 1
            segment_len = j - i
            if segment_len < 16:
                prev = stable_targets[i - 1] if i > 0 else None
                next = targets[j] if j < len(targets) else None
                replacement = prev if prev and (not next or prev[1] > next[1]) else next
                stable_targets.extend([replacement] * segment_len)
            else:
                stable_targets.extend([current] * segment_len)
            i = j

        # --- Apply cropping based on smoothed center points ---
        centers = [(t[2], t[3]) if t else (0.5, 0.5) for t in stable_targets]

        if not centers:
            centers = [(0.5, 0.5)] * len(frames)

        smoothed_centers = self._smooth_centers(centers)
        smoothed_centers = self._detect_and_fix_outliers(smoothed_centers)

        crop_frames = []
        for frame, (cx, cy) in zip(frames, smoothed_centers):
            crop_x = int(max(0, min(cx * scaled_w - desired_width / 2, scaled_w - desired_width)))
            crop_y = int(max(0, min(cy * scaled_h - desired_height / 2, scaled_h - desired_height)))
            cropped = frame[crop_y:crop_y + desired_height, crop_x:crop_x + desired_width]
            crop_frames.append(cropped)

        return crop_frames, fps

    async def __call__(self, desired_width, desired_height, clips):
        print(f"[INFO] Starting advanced DynamicCropping for {len(clips)} clips.")
        final_clips = []

        for idx, clip in enumerate(clips):
            print(f"\n[INFO] Processing clip {idx}")
            sampled = self._sample_frames(clip)
            mode = await self._decide_crop_mode(sampled)
            print(f"[INFO] Clip {idx} crop mode: {mode}")

            if mode == "general":
                center_x, center_y = await self._get_general_crop_center(sampled, *clip.size)
                w, h = clip.size
                scale = max(desired_width / w, desired_height / h)
                clip_resized = clip.resized(scale)
                sw, sh = clip_resized.size

                crop_x = int(max(0, min(center_x * sw - desired_width / 2, sw - desired_width)))
                crop_y = int(max(0, min(center_y * sh - desired_height / 2, sh - desired_height)))

                final = clip_resized.cropped(x1=crop_x, y1=crop_y, width=desired_width, height=desired_height)
                final_clips.append(final)

            elif mode == "character_tracking":
                # Save crop_frames to a temporary video file using OpenCV
                crop_frames, fps = self._track_character(clip, desired_width, desired_height)

                temp_video_path = str(self.workdir / f"tracked_{uuid.uuid4().hex}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(temp_video_path, fourcc, fps, (desired_width, desired_height))

                for frame in crop_frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()

                # Load compressed clip with VideoFileClip (disk-backed)
                img_clip = VideoFileClip(temp_video_path)

                if clip.audio:
                    safe_duration = min(clip.audio.duration, img_clip.duration)
                    audio = clip.audio.subclipped(0, safe_duration)
                    img_clip = img_clip.with_audio(audio)

                final_clips.append(img_clip)


            else:
                print(f"[WARN] Unknown crop mode for clip {idx}, skipping.")
                continue

        print(f"[INFO] DynamicCropping complete.")
        return final_clips
