import tempfile
import os
from pathlib import Path
from lib.utils.media import preprocess_long_video
from src.pipelines.geminiNaiveComprehensionPipeline.tasks.naive_comprehension import NaiveGeminiComprehension
from lib.oss.gcp_oss import GoogleCloudStorage
from src.config import BUCKET_NAME, CREDENTIAL_PATH
from lib.oss.auth import credentials_from_file

class GeminiNaiveComprehensionPipeline:
    def __init__(self, llm):
        self.llm = llm
        self.gcs_client = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))

    async def run_for_file(self, local_media_path):
        print(f"[INFO] Starting naive comprehension pipeline for: {local_media_path}")

        # Segment video
        segments_dir = tempfile.mkdtemp()
        print(f"[INFO] Segmenting video into 30-minute chunks at 1 fps...")
        segments = await preprocess_long_video(
            local_media_path, segments_dir, interval_seconds=50*60, fps=1, crf=30
        )
        print(f"[INFO] Generated {len(segments)} segments.")

        # Upload segments to GCS and track uploaded paths
        uploaded_gcs_uris = []
        print("[INFO] Uploading segments to GCS...")
        for seg in segments:
            local_path = seg["path"]
            gcs_path = f"temp_naive_segments/{os.path.basename(local_path)}"
            print(f"[UPLOAD] {local_path} -> gs://{BUCKET_NAME}/{gcs_path}")
            self.gcs_client.upload_files(BUCKET_NAME, local_path, gcs_path)
            seg["gcs_uri"] = f"gs://{BUCKET_NAME}/{gcs_path}"
            uploaded_gcs_uris.append(gcs_path)

        # Run Gemini comprehension using GCS URIs
        print("[INFO] Running Gemini comprehension on uploaded segments...")
        ngc = NaiveGeminiComprehension(self.llm)
        combined_summary = await ngc(segments)

        # Clean up GCS
        print("[INFO] Cleaning up uploaded segments from GCS...")
        for gcs_path in uploaded_gcs_uris:
            print(f"[DELETE] gs://{BUCKET_NAME}/{gcs_path}")
            self.gcs_client.delete_blob(BUCKET_NAME, gcs_path)

        print("[SUCCESS] Naive comprehension complete.")
        return {
            "combined_summary.txt": combined_summary
        }
