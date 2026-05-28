"""Tests for narration transcript generation logic in agent/tools.py."""
import re
from types import SimpleNamespace
import pytest


# ---------------------------------------------------------------------------
# Inline the sentence-splitting + pro-rating logic from _generate_narration
# so we can test it in isolation without needing TTS or async infrastructure.
# ---------------------------------------------------------------------------

def _build_transcript(script: str, duration: float) -> list:
    """
    Reproduce the transcript-building logic from AgentToolExecutor._generate_narration.

    Splits script into sentences, pro-rates duration by word count, and returns
    a list of dicts with text, start, end, word_count.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', script.strip()) if s.strip()]
    word_counts = [len(s.split()) for s in sentences]
    total_words = sum(word_counts) or 1
    transcript = []
    cursor = 0.0
    for sent, wc in zip(sentences, word_counts):
        sent_dur = duration * (wc / total_words)
        transcript.append({
            "text": sent,
            "start": round(cursor, 2),
            "end": round(cursor + sent_dur, 2),
            "word_count": wc,
        })
        cursor += sent_dur
    return transcript


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_transcript_single_sentence():
    transcript = _build_transcript("Hello world.", 4.0)
    assert len(transcript) == 1
    assert transcript[0]["text"] == "Hello world."
    assert transcript[0]["start"] == 0.0
    assert transcript[0]["end"] == 4.0
    assert transcript[0]["word_count"] == 2


def test_transcript_multiple_sentences():
    script = "First sentence here. Second one. Third sentence is longer than others."
    transcript = _build_transcript(script, 10.0)
    assert len(transcript) == 3

    # Starts should be monotonically increasing
    for i in range(1, len(transcript)):
        assert transcript[i]["start"] >= transcript[i - 1]["start"]

    # Each segment's end should equal next segment's start (continuous)
    for i in range(len(transcript) - 1):
        assert abs(transcript[i]["end"] - transcript[i + 1]["start"]) < 0.02

    # First starts at 0, last ends at ~duration
    assert transcript[0]["start"] == 0.0
    assert abs(transcript[-1]["end"] - 10.0) < 0.05


def test_transcript_prorates_by_word_count():
    # "Short." has 1 word, "This sentence has five words." has 5 words
    # So the longer sentence should get 5/6 of the duration
    script = "Short. This sentence has five words."
    transcript = _build_transcript(script, 6.0)
    assert len(transcript) == 2

    short_dur = transcript[0]["end"] - transcript[0]["start"]
    long_dur = transcript[1]["end"] - transcript[1]["start"]
    assert abs(short_dur - 1.0) < 0.05, f"1-word sentence should get ~1s, got {short_dur}"
    assert abs(long_dur - 5.0) < 0.05, f"5-word sentence should get ~5s, got {long_dur}"


def test_transcript_word_counts_correct():
    script = "One. Two words. Three little words."
    transcript = _build_transcript(script, 12.0)
    assert transcript[0]["word_count"] == 1
    assert transcript[1]["word_count"] == 2
    assert transcript[2]["word_count"] == 3


def test_transcript_empty_script():
    transcript = _build_transcript("", 5.0)
    assert transcript == []


def test_transcript_exclamation_and_question_marks():
    script = "Wow! Is this working? Yes it is."
    transcript = _build_transcript(script, 9.0)
    assert len(transcript) == 3
    assert transcript[0]["text"] == "Wow!"
    assert transcript[1]["text"] == "Is this working?"
    assert transcript[2]["text"] == "Yes it is."


def test_transcript_total_duration_matches():
    script = "The quick brown fox jumps. Over the lazy dog. End."
    duration = 15.0
    transcript = _build_transcript(script, duration)
    total = sum(t["end"] - t["start"] for t in transcript)
    assert abs(total - duration) < 0.1, f"Total transcript duration {total}s should match {duration}s"


# ---------------------------------------------------------------------------
# Audio-stream probe — regression for silent stock footage.
#
# Prior bug: refine_clip_timestamps fed silent footage to `ffmpeg -vn -c:a aac`,
# which fails with "Output file does not contain any stream". The error was
# then misattributed to ElevenLabs in the user-facing warning. The fix probes
# for an audio stream first; these tests pin that probe behavior.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _ffmpeg_or_skip():
    import shutil
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe not on PATH")


def test_source_has_audio_stream_true_for_audio_video(tmp_path, _ffmpeg_or_skip):
    """A 1-second clip with both a colored video stream and a sine audio stream
    should be reported as having audio."""
    import subprocess
    from src.pipelines.v2.agent.tools import _source_has_audio_stream
    av = tmp_path / "av.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=black:s=160x120:d=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(av)],
        check=True,
    )
    assert _source_has_audio_stream(av) is True


def test_source_has_audio_stream_false_for_silent_video(tmp_path, _ffmpeg_or_skip):
    """A video-only clip (no audio stream at all) — the silent-stock-footage
    case — must be reported as having no audio so STT is skipped cleanly."""
    import subprocess
    from src.pipelines.v2.agent.tools import _source_has_audio_stream
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=black:s=160x120:d=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent)],
        check=True,
    )
    assert _source_has_audio_stream(silent) is False


def test_source_has_audio_stream_safe_on_missing_file(tmp_path):
    """Probe failure must not raise — caller treats False as 'skip STT'."""
    from src.pipelines.v2.agent.tools import _source_has_audio_stream
    assert _source_has_audio_stream(tmp_path / "nope.mp4") is False


@pytest.mark.asyncio
async def test_ask_memories_queries_each_video_with_current_mavi_signature():
    """MaviAgent.ask accepts one video_id; VEA must cover multi-source projects."""
    from src.pipelines.v2.agent.tools import ToolExecutor

    class StrictMaviAgent:
        def __init__(self):
            self.calls = []

        async def ask(self, question, video_id=None, user_id=None):
            self.calls.append(video_id)
            return SimpleNamespace(
                answer=f"answer for {video_id}",
                reranked_videos=1,
                reranked_video_ts=2,
                reranked_audio_ts=3,
                reranked_keyframes=4,
            )

    mavi = StrictMaviAgent()
    executor = ToolExecutor.__new__(ToolExecutor)
    executor.mavi_agent = mavi
    executor.video_nos = ["v1", "v2"]

    result = await ToolExecutor._ask_memories(executor, {"question": "What happens?"})

    assert mavi.calls == ["v1", "v2"]
    assert "answer for v1" in result["answer"]
    assert "answer for v2" in result["answer"]
    assert result["reference_count"] == 20
