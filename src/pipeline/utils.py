from lib.llm.gemini import Gemini
from google.genai import types
from vertexai.generative_models import GenerationConfig


async def generate_response_for_video(llm: Gemini, prompt: str, video_file: str, config: GenerationConfig):
    video_bytes = open(video_file, "rb").read()
    response = await llm.generate_async(
        prompt=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type="video/mp4")
                ),
                types.Part(text=prompt),
            ],
            role="user"
        ),
        generation_config=config,
    )
    return response


if __name__ == "__main__":
    import asyncio
    video_path = "E:/OpenInterX Code Source/mavi-edit-service/test/Everything Everywhere All At Once _ Official Trailer HD _ A24.mp4"
    output_path = "E:/OpenInterX Code Source/mavi-edit-service/test/output.mp4"
    llm = Gemini()
    res = asyncio.run(generate_response_for_video(llm, "describe the video", output_path))
    print(res.text)