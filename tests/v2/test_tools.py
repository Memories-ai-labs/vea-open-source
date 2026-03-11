"""Tests for narration transcript generation logic in agent/tools.py."""
import re
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
