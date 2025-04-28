# src/config.py

import os
from pathlib import Path

# --- Static Constants ---
API_PREFIX = "/v1"

# --- Dynamic Constants (from environment) ---
ENV = os.getenv("ENV", "development")

SUMMARY_FPS = 1
SUMMARY_CRF = 40
BUCKET_NAME = "openinterx-vea"
MOVIE_LIBRARY= "movie_library"
CREDENTIAL_PATH = Path("config/gen-lang-client-0057517563-0319d78ed5fe.json")
