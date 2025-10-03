import asyncio
import os
import tempfile
from pathlib import Path
import json
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from lib.oss.auth import credentials_from_file
from lib.oss.gcp_oss import GoogleCloudStorage
from src.config import CREDENTIAL_PATH, BUCKET_NAME, VIDEO_EXTS
from src.pipelines.videoComprehension.comprehensionPipeline import ComprehensionPipeline
from src.pipelines.geminiNaiveComprehensionPipeline.pipeline import GeminiNaiveComprehensionPipeline


async def download_from_gcs(gcs_client, gcs_path):
    filename = os.path.basename(gcs_path)
    local_path = os.path.join(tempfile.mkdtemp(), filename)
    gcs_client.download_files(BUCKET_NAME, gcs_path, local_path)
    return local_path

async def run_benchmark_for_blob(blob_path, model_name):
    print(f"\n=== [Model: {model_name}] Processing: {blob_path} ===")

    # Prepare GCS client and output folder
    gcs_client = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))
    media_name = Path(blob_path).stem
    output_folder = f"benchmark_outputs/{media_name}/{model_name}"
    os.makedirs(output_folder, exist_ok=True)

    # Instantiate LLM
    llm = GeminiGenaiManager(model=model_name) if model_name != "ours" else None

    # If using our pipeline
    if model_name == "ours":
        full_pipeline = ComprehensionPipeline(
            blob_path=blob_path,
            start_fresh=True,
            debug_output_dir=output_folder
        )
        await full_pipeline.run()
    else:
        # Always run naive baseline
        local_file = await download_from_gcs(gcs_client, blob_path)
        naive_pipeline = GeminiNaiveComprehensionPipeline(llm)
        naive_output = await naive_pipeline.run_for_file(local_file)

        # Save naive output
        naive_output_path = os.path.join(output_folder, "naive_combined_summary.txt")
        with open(naive_output_path, "w", encoding="utf-8") as f:
            f.write(naive_output["combined_summary.txt"])

    print(f"[DONE] Saved all outputs to {output_folder}")

async def main():
    # List of blob paths to process
    blob_paths = [
        "movie_library/spacebetween.mp4",
        "movie_library/174.曾经 Once (2007) 中英双字.mkv",
        "movie_library/John Wick (2014) 1080p BluRay H264 DolbyD 5.1 + nickarad.mp4",
        "movie_library/V字仇杀队.mkv",
        "movie_library/七宗罪.mkv",
        "movie_library/肖申克的救赎.mp4",
        "movie_library/千与千寻.mkv"

    ]

    # Gemini model variants to benchmark
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "ours"
    ]

    for blob_path in blob_paths:
        for model_name in models_to_try:
            try:
                await run_benchmark_for_blob(blob_path, model_name)
            except Exception as e:
                print(f"[ERROR] Failed to process {blob_path} with model {model_name}: {e}")
                continue

if __name__ == "__main__":
    asyncio.run(main())
