# src/config.py

import os
import json
from pathlib import Path

# --- Static Constants ---
VIDEO_EXTS = [
    ".mp4", ".mov", ".mkv", ".avi", ".webm",
    ".mpg", ".mpeg", ".m4v"
]

# --- Configuration Files ---
CONFIG_PATH = Path("config.json")
LEGACY_API_KEYS_PATH = Path("config/apiKeys.json")
LEGACY_CREDENTIAL_PATH = Path("config/gcp_credentials.json")


def _load_config() -> dict:
    """Load configuration from config.json or fall back to defaults."""
    if CONFIG_PATH.exists():
        print(f"[CONFIG] Loading from {CONFIG_PATH}")
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    # Fall back to legacy/defaults
    print(f"[CONFIG] {CONFIG_PATH} not found, using defaults")
    return {
        "paths": {
            "videos_dir": "data/videos",
            "indexing_dir": "data/indexing",
            "outputs_dir": "data/outputs"
        },
        "api_keys": {},
        "optional_features": {
            "enable_music": True,
            "enable_dynamic_cropping": True,
            "enable_subtitles": True,
            "enable_fcpxml_export": True
        },
        "video_processing": {
            "default_fps": 24,
            "summary_fps": 1,
            "summary_crf": 40
        }
    }


def _setup_api_keys(config_data: dict):
    """Load API keys into environment variables."""
    api_keys = config_data.get("api_keys", {})

    # Load from config.json api_keys section
    for key, value in api_keys.items():
        if not key.startswith("_"):  # Skip comment fields
            os.environ[key] = str(value)

    # Also try legacy apiKeys.json for backward compatibility
    if LEGACY_API_KEYS_PATH.exists():
        with open(LEGACY_API_KEYS_PATH, "r") as f:
            legacy_keys = json.load(f)
            for key, value in legacy_keys.items():
                if key not in os.environ:  # Don't override config.json
                    os.environ[key] = str(value)


# --- Load Configuration ---
_config = _load_config()
_setup_api_keys(_config)

# --- Path Settings ---
VIDEOS_DIR = Path(_config.get("paths", {}).get("videos_dir", "data/videos"))
INDEXING_DIR = Path(_config.get("paths", {}).get("indexing_dir", "data/indexing"))
OUTPUTS_DIR = Path(_config.get("paths", {}).get("outputs_dir", "data/outputs"))
WORKSPACES_DIR = Path(_config.get("paths", {}).get("workspaces_dir", "data/workspaces"))
V2_API_PREFIX = "/video-edit/v2"

# --- Optional Features ---
_features = _config.get("optional_features", {})

# --- Video Processing Settings ---
_video_settings = _config.get("video_processing", {})
DEFAULT_FPS = _video_settings.get("default_fps", 24)
SUMMARY_FPS = _video_settings.get("summary_fps", 1)
SUMMARY_CRF = _video_settings.get("summary_crf", 40)

ENV = os.getenv("ENV", "development")


def is_local_mode() -> bool:
    """Check if running in local storage mode. Always True for open source version."""
    return True


def get_storage_mode() -> str:
    """Get current storage mode. Always 'local' for open source version."""
    return "local"


def ensure_local_directories():
    """Create local storage directories."""
    dirs = [VIDEOS_DIR, INDEXING_DIR, OUTPUTS_DIR, WORKSPACES_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[CONFIG] Local directories ready: {VIDEOS_DIR}, {INDEXING_DIR}, {OUTPUTS_DIR}, {WORKSPACES_DIR}")
