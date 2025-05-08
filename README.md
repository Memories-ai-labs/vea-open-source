## Install environment using uv

```bash
 uv sync
 ```
- also ensure to have ffmpeg installed in the system

### Environment setup
- you need API keys from Google Genai and Soundstripe, these shoule be placed in config/apiKeys.json following apiKeysExample.json
- you also need a GCP credentials json from google cloud platform, placed in config/gcp_credentials.json

### Docker
build the container
```bash
sudo docker build -t vea-recap .
```
run the container
```bash
sudo docker run -p 8000:8000 vea-recap
```

### Running server without docker
```bash
source .venv/bin/activate
python -m src.app
```

### Tools
There are some useful tools for video processing, llm interaction:

- [media.py](./lib/utils/media.py) provides 4 functions: `get_video_info`, `get_video_duration`, `convert_video` and `trim_video`. `convert_video` is especially useful if you want to resample, cut or reduce the quality of the video since llm does not require 4K quality of the video input.

- [GeminiGenaiManager.py](lib/llm/GeminiGenaiManager.py) provides a class for calling gemini to generate output based on your pompt.

