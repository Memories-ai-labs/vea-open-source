import os
import numpy as np
import tempfile
import asyncio
from pathlib import Path
from PIL import Image
from moviepy import VideoFileClip, concatenate_videoclips
from scipy.interpolate import interp1d
from src.pipelines.common.schema import CroppingResponse


class DynamicCropping:
    def __init__(self, llm, workdir=None, batch_size=5, crop_threshold=0.1):
        self.llm = llm
        self.workdir = workdir or tempfile.mkdtemp()
        self.batch_size = batch_size
        self.crop_threshold = crop_threshold

    def _aspect_ratio(self, w, h):
        return float(w) / float(h)

    def _is_aspect_ratio_similar(self, w1, h1, w2, h2, threshold=0.06):
        r1 = self._aspect_ratio(w1, h1)
        r2 = self._aspect_ratio(w2, h2)
        similar = abs(r1 - r2) / r2 < threshold
        print(f"[DEBUG] Checking aspect similarity: {w1}x{h1} vs {w2}x{h2} -> {similar}")
        return similar

    def _extract_frames(self, clip_obj, clip_idx, every_n_seconds=2):
        frames = []
        duration = clip_obj.duration
        times = [min(t, duration - 0.01) for t in np.arange(0, duration, every_n_seconds)]
        for j, t in enumerate(times):
            frame_id = f"{clip_idx}_{j}"
            out_img_path = os.path.join(self.workdir, f"clip{clip_idx}_frame{j}.jpg")
            frame = clip_obj.get_frame(t)
            img = Image.fromarray(frame)
            img.save(out_img_path)
            frames.append({"frame_id": frame_id, "t": t, "img_path": out_img_path})
        return frames

    def _segment_crop_keyframes(self, times, cx, cy, threshold, input_duration):
        # Segments logic, but last segment always ends at input_duration
        segments = []
        n = len(times)
        seg_start = 0
        for i in range(1, n):
            dx = abs(cx[i] - cx[seg_start])
            dy = abs(cy[i] - cy[seg_start])
            if dx > threshold or dy > threshold:
                avg_cx = float(np.mean(cx[seg_start:i]))
                avg_cy = float(np.mean(cy[seg_start:i]))
                segments.append({
                    "start": float(times[seg_start]),
                    "end": float(times[i]),
                    "crop_x": avg_cx,
                    "crop_y": avg_cy
                })
                seg_start = i
        # Add last segment: make sure it ends exactly at the input clip duration
        avg_cx = float(np.mean(cx[seg_start:]))
        avg_cy = float(np.mean(cy[seg_start:]))
        segments.append({
            "start": float(times[seg_start]),
            "end": float(input_duration),  # <-- make sure!
            "crop_x": avg_cx,
            "crop_y": avg_cy
        })
        return segments

    async def __call__(
        self,
        desired_width: int,
        desired_height: int,
        processed_clips: list,
        target_aspect_threshold=0.06
    ):
        print(f"[INFO] Starting DynamicCropping for {len(processed_clips)} clips. Target size: {desired_width}x{desired_height}")
        all_frames = []
        frames_by_clip = {}

        # 1. Extract frames for all clips, keep mapping
        for i, clip_obj in enumerate(processed_clips):
            w, h = clip_obj.size
            if self._is_aspect_ratio_similar(w, h, desired_width, desired_height, threshold=target_aspect_threshold):
                continue
            frames = self._extract_frames(clip_obj, i)
            all_frames.extend([dict(frame, clip_idx=i) for frame in frames])
            frames_by_clip.setdefault(i, []).extend(frames)

        # 2. Batch Gemini predictions (batches of N)
        crops_for_frames = {}
        for i in range(0, len(all_frames), self.batch_size):
            batch = all_frames[i:i+self.batch_size]
            images = [Path(frame['img_path']) for frame in batch]
            mapping = "\n".join([f"Image {j+1}: frame_id '{frame['frame_id']}'" for j, frame in enumerate(batch)])
            prompt = (
                f"You are an expert video editor tasked with cropping video frames to a target aspect ratio of {desired_width}:{desired_height}.\n"
                "You will be given a sequence of images. Each image is mapped to a frame_id as follows:\n"
                f"{mapping}\n"
                "For each image, return your crop center (x and y, between 0 and 1) for the most important content for the full scene.\n"
                "Output a JSON list of objects, each with: frame_id (string), crop_center_x (float), crop_center_y (float). Use the provided frame_id for each result.\n"
                "Entries must be in the same order as input images."
            )
            crops = await asyncio.to_thread(
                self.llm.LLM_request,
                [*images, prompt],
                {
                    "response_mime_type": "application/json",
                    "response_schema": list[CroppingResponse]
                }
            )
            for crop in crops:
                crops_for_frames[crop["frame_id"]] = crop

        # 3. For each clip, segment and crop (MoviePy only)
        new_processed_clips = []
        for clip_idx, clip_obj in enumerate(processed_clips):
            w, h = clip_obj.size
            if self._is_aspect_ratio_similar(w, h, desired_width, desired_height, threshold=target_aspect_threshold):
                new_processed_clips.append(clip_obj.resized((desired_width, desired_height)))
                continue

            frames = frames_by_clip.get(clip_idx, [])
            if not frames:
                new_processed_clips.append(clip_obj.resized((desired_width, desired_height)))
                continue

            ts = np.array([f["t"] for f in frames])
            cx = np.array([crops_for_frames[f["frame_id"]]["crop_center_x"] for f in frames])
            cy = np.array([crops_for_frames[f["frame_id"]]["crop_center_y"] for f in frames])

            # Scaling logic: fit clip so desired crop fits inside original
            scale_w = desired_width / w
            scale_h = desired_height / h
            scale = max(scale_w, scale_h)
            scaled_w, scaled_h = int(w * scale), int(h * scale)
            print(f"[DEBUG] Clip {clip_idx}: Scaling to {scaled_w}x{scaled_h}")

            upscaled_clip = clip_obj.resized((scaled_w, scaled_h))

            # --- Updated segment logic ---
            segments = self._segment_crop_keyframes(ts, cx, cy, self.crop_threshold, upscaled_clip.duration)
            seg_clips = []
            for seg in segments:
                crop_px = int(max(0, min(seg["crop_x"] * scaled_w - desired_width / 2, scaled_w - desired_width)))
                crop_py = int(max(0, min(seg["crop_y"] * scaled_h - desired_height / 2, scaled_h - desired_height)))
                start_time = max(0, seg["start"])
                end_time = min(seg["end"], upscaled_clip.duration)
                seg_video = (
                    upscaled_clip
                    .subclipped(start_time, end_time)
                    .cropped(x1=crop_px, y1=crop_py, width=desired_width, height=desired_height)
                )
                # Re-attach the correct audio segment, if available
                if upscaled_clip.audio is not None:
                    audio_subclip = upscaled_clip.audio.subclipped(start_time, end_time)
                    seg_video = seg_video.with_audio(audio_subclip)
                seg_clips.append(seg_video)
            final_clip = concatenate_videoclips(seg_clips)

            new_processed_clips.append(final_clip)


        print(f"[INFO] DynamicCropping complete. Returning {len(new_processed_clips)} clips.")
        return new_processed_clips

