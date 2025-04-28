## Install environment using uv

```bash
 uv pip sync pyproject.toml
 ```

### test summary pipline
```bash
python -m src.pipeline.summary_pipeline
```

### Pipeline
Pipeline is the core functions for this repo, each pipeline should perform a core task in the video-editing workflow.

For exammple, [summary_pipeline](./src/pipeline/summary_pipeline.py) implemnts the pipeline that generate a gist sumamry of the input video and output the summary in json format for future usage.


### Tools
There are some useful tools for vide processing, llm interaction:

- [media.py](./lib/utils/media.py) provides 4 functions: `get_video_info`, `get_video_duration`, `convert_video` and `trim_video`. `convert_video` is especially useful if you want to resample, cut or reduce the quality of the video since llm does not require 4K quality of the video input.

- [gemini.py](./lib/llm/gemini.py) provides a `Gemini` class for calling gemini to generate output based on your pompt.

- [utils.py](./src/pipeline/utils.py) provides a `generate_response_for_video` function that prompt gemini with video efficiently without uploading a video.