import sys
import json
import asyncio

from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from src.config import CREDENTIAL_PATH
from src.pipelines.common.timeline_constructor import TimelineConstructor

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Expected path to edit config JSON")
        sys.exit(1)

    input_path = sys.argv[1]
    with open(input_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    gcs_client = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))
    llm = GeminiGenaiManager(model="gemini-1.5-flash")

    constructor = TimelineConstructor(
        output_path=config.get("output_path", "video_response.mp4"),
        gcs_client=gcs_client,
        bucket_name=config.get("bucket_name"),
        workdir=config.get("workdir"),
        llm=llm
    )

    asyncio.run(constructor.run(
        clips=config["clips"],
        narration_dir=config["narration_dir"],
        background_music_path=config.get("background_music_path"),
        original_audio=config.get("original_audio", True),
        narration_enabled=config.get("narration_enabled", True),
        aspect_ratio=config.get("aspect_ratio", 16/9),
        subtitles=config.get("subtitles", True),
        snap_to_beat=config.get("snap_to_beat", False),
        multi_round_mode=config.get("multi_round_mode", True)
    ))
