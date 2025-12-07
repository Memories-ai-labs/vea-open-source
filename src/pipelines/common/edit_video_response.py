import sys
import json
import asyncio
import os

from lib.oss.storage_factory import get_storage_client
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from lib.utils.metrics_collector import metrics_collector
from src.pipelines.common.timeline_constructor import TimelineConstructor

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Expected path to edit config JSON")
        sys.exit(1)

    input_path = sys.argv[1]
    metrics_output_path = sys.argv[2] if len(sys.argv) > 2 else None

    with open(input_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    gcs_client = get_storage_client()
    llm = GeminiGenaiManager(model="gemini-1.5-flash")

    constructor = TimelineConstructor(
        gcs_client=gcs_client,
        bucket_name=config.get("bucket_name"),
        llm=llm
    )

    output_dir, video_path = asyncio.run(constructor.run(
        project_name=config.get("project_name"),
        clips=config["clips"],
        narration_dir=config["narration_dir"],
        background_music_path=config.get("background_music_path"),
        original_audio=config.get("original_audio", True),
        narration_enabled=config.get("narration_enabled", True),
        aspect_ratio=config.get("aspect_ratio", 16/9),
        subtitles=config.get("subtitles", True),
        snap_to_beat=config.get("snap_to_beat", False),
    ))

    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Video path: {video_path}")

    # Write result file for parent process to read
    result_file = input_path.replace(".json", "_result.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "output_dir": str(output_dir),
            "video_path": str(video_path),
        }, f)

    # Write metrics to file if path provided
    if metrics_output_path:
        os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
        metrics_collector.write_report(metrics_output_path)
        print(f"[INFO] Subprocess metrics written to {metrics_output_path}")
