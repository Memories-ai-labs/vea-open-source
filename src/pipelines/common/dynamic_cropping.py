import os
import numpy as np
import tempfile
import asyncio
from pathlib import Path
from PIL import Image
from src.pipelines.common.schema import CroppingResponse

class DynamicCropping:
    def __init__(self, llm, workdir=None, batch_size=5):
        self.llm = llm
        self.workdir = workdir or tempfile.mkdtemp()
        self.batch_size = batch_size

    def _aspect_ratio(self, w, h):
        return float(w) / float(h)

    def _is_aspect_ratio_similar(self, w1, h1, w2, h2, threshold=0.06):
        r1 = self._aspect_ratio(w1, h1)
        r2 = self._aspect_ratio(w2, h2)
        return abs(r1 - r2) / r2 < threshold

    def _extract_middle_frame_from_moviepy_clip(self, clip_obj, out_img_path):
        """
        Extract a frame from the middle of a MoviePy VideoClip and save as JPEG.
        """
        mid_t = 0.5 * (clip_obj.start + clip_obj.end) if clip_obj.end else 0.5 * clip_obj.duration
        frame = clip_obj.get_frame(mid_t)
        img = Image.fromarray(frame)
        img.save(out_img_path)
        return out_img_path
    
    def _extract_blended_frames_from_moviepy_clip(self, clip_obj, out_img_path, num_samples=3):
        """
        Evenly samples `num_samples` frames, blends them by averaging, and saves as JPEG.
        """
        duration = clip_obj.end - clip_obj.start if clip_obj.end else clip_obj.duration
        ts = np.linspace(clip_obj.start, clip_obj.start + duration, num=num_samples+2)[1:-1]  # Exclude very start/end
        frames = [clip_obj.get_frame(t) for t in ts]
        blended = np.mean(frames, axis=0).astype(np.uint8)
        img = Image.fromarray(blended)
        img.save(out_img_path)
        return out_img_path


    async def __call__(
        self, 
        desired_width: int, 
        desired_height: int, 
        processed_clips: list,  # [(clip: VideoFileClip, meta_dict: dict)]
        target_aspect_threshold=0.06
    ):
        """
        - processed_clips: list of tuples (moviepy clip object, original clip dict)
        - desired_width/height: target export resolution
        """
        img_paths = []
        original_idxs = []
        crop_instructions = [None] * len(processed_clips)

        # Prepare for cropping
        for i, clip_obj in enumerate(processed_clips):
            w, h = clip_obj.size
            
            if self._is_aspect_ratio_similar(w, h, desired_width, desired_height, threshold=target_aspect_threshold):
                continue

            # Save mid frame as image for LLM
            out_img_path = os.path.join(self.workdir, f"{i}_mid.jpg")
            self._extract_middle_frame_from_moviepy_clip(clip_obj, out_img_path)
            # self._extract_blended_frames_from_moviepy_clip(clip_obj, out_img_path)
            img_paths.append(Path(out_img_path))
            original_idxs.append(i)

        # Batch Gemini calls
        for i in range(0, len(img_paths), self.batch_size):
            batch = img_paths[i:i+self.batch_size]
            idxs = original_idxs[i:i+self.batch_size]

            prompt = (
                f"You are an expert video editor tasked with cropping video clips to a target aspect ratio of {desired_width}:{desired_height}.\n"
                f"For each of the following images (center frames from their clips), pick the best crop for the new aspect ratio that frames the most important content in the frame\n"
                "Return the crop center for each image as `crop_center_x` and `crop_center_y`, both as floats between 0 and 1 (0.5 means centered). "
                "Choose the center that keeps the most important content, such as faces or subjects, in frame. "
                "Output your crop decisions in a list of structured outputs, where entries are in the same order as the images\n"
        
            )
            
            # prompt = (
            #     f"You are an expert video editor tasked with cropping video clips to a target aspect ratio of {desired_width}:{desired_height}.\n"
            #     f"Each image is a blend of three evenly sampled frames from the video clip (averaged together) to summarize the important content across the clip.\n"
            #     "For each of the following images, pick the best crop for the new aspect ratio that frames the most important content for the entire clip—not just a single moment.\n"
            #     "Return the crop center for each image as `crop_center_x` and `crop_center_y`, both as floats between 0 and 1 (0.5 means centered). "
            #     "Choose the center that keeps the most important content, such as faces or subjects, in frame. "
            #     "Output your crop decisions in a list of structured outputs, where entries are in the same order as the images.\n"
            # )

            crops = await asyncio.to_thread(
                self.llm.LLM_request,
                [*batch, prompt],
                {
                    "response_mime_type": "application/json",
                    "response_schema": list[CroppingResponse]
                }
            )
            print(crops)
            for j, crop in enumerate(crops):
                crop_instructions[idxs[j]] = crop

        for i, clip in enumerate(processed_clips):
            if crop_instructions[i] is None:
                processed_clips[i] = clip.resized((desired_width, desired_height))
            else:
                w, h = clip.size
                scale_w = desired_width / w
                scale_h = desired_height / h
                scale = max(scale_w, scale_h)
                scaled_w, scaled_h = int(w * scale), int(h * scale)
                scaled_clip = clip.resized((scaled_w, scaled_h))
                cx, cy = crop_instructions[i]["crop_center_x"], crop_instructions[i]["crop_center_y"]
                crop_x = int(cx * scaled_w - desired_width / 2)
                crop_y = int(cy * scaled_h - desired_height / 2)
                crop_x = max(0, min(crop_x, scaled_w - desired_width))
                crop_y = max(0, min(crop_y, scaled_h - desired_height))

                processed_clips[i] = scaled_clip.cropped(
                    x1=crop_x,
                    y1=crop_y,
                    x2=crop_x + desired_width,
                    y2=crop_y + desired_height,
                )
        
        return processed_clips

