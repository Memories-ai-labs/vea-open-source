"""Tests for fcpxml_scaffold.py — scaffold generation from Storyboard."""
import os
import pytest
import tempfile
from pathlib import Path
from xml.etree.ElementTree import fromstring

from src.pipelines.v2.fcpxml.fcpxml_scaffold import build_scaffold
from src.pipelines.v2.fcpxml.fcpxml_compiler import compile_fcpxml
from src.pipelines.v2.schemas import RetrievedClip, Shot, Storyboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_clip(video_no="v1", start=0.0, end=10.0, score=0.8) -> RetrievedClip:
    return RetrievedClip(
        video_no=video_no,
        video_name="source.mp4",
        source_path="/tmp/source.mp4",
        start_seconds=start,
        end_seconds=end,
        score=score,
        description="test clip",
        shot_query="test",
    )


def make_shot(shot_id="s1", clip: RetrievedClip = None, duration=10.0) -> Shot:
    return Shot(
        id=shot_id,
        purpose="Test shot",
        search_query="test",
        retrieved_clip=clip,
        duration_seconds=duration,
    )


def make_storyboard(shots=None) -> Storyboard:
    return Storyboard(
        iteration=1,
        target_duration_seconds=60.0,
        theme="test theme",
        narrative_arc="test arc",
        shots=shots or [],
    )


@pytest.fixture
def tmp_output(tmp_path):
    return str(tmp_path / "test.fcpxml")


@pytest.fixture
def source_video(tmp_path):
    """Create a dummy video file so _file_uri works."""
    p = tmp_path / "source.mp4"
    p.write_bytes(b"fake video")
    return str(p)


# ---------------------------------------------------------------------------
# Basic scaffold generation
# ---------------------------------------------------------------------------

def test_scaffold_creates_file(tmp_output, source_video):
    clip = make_clip()
    clip = clip.model_copy(update={"source_path": source_video})
    shots = [make_shot("s1", clip)]
    sb = make_storyboard(shots)

    path = build_scaffold(sb, {"s1": clip}, output_path=tmp_output)
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


def test_scaffold_valid_fcpxml(tmp_output, source_video):
    clip = make_clip()
    clip = clip.model_copy(update={"source_path": source_video})
    shots = [make_shot("s1", clip), make_shot("s2", clip)]
    sb = make_storyboard(shots)

    path = build_scaffold(sb, {"s1": clip, "s2": clip}, output_path=tmp_output)
    xml_text = Path(path).read_text()

    result = compile_fcpxml(xml_text, run_dtd=False)
    assert result.valid, f"Scaffold should be valid: {result.errors}"


def test_scaffold_correct_version(tmp_output, source_video):
    clip = make_clip()
    clip = clip.model_copy(update={"source_path": source_video})
    sb = make_storyboard([make_shot("s1", clip)])

    path = build_scaffold(sb, {"s1": clip}, output_path=tmp_output)
    xml = Path(path).read_text()
    root = fromstring(xml.split("<!DOCTYPE fcpxml>")[-1].strip())
    assert root.get("version") == "1.10"


def test_scaffold_contains_asset_clip(tmp_output, source_video):
    clip = make_clip()
    clip = clip.model_copy(update={"source_path": source_video})
    sb = make_storyboard([make_shot("s1", clip)])

    path = build_scaffold(sb, {"s1": clip}, output_path=tmp_output)
    xml = Path(path).read_text()
    assert "asset-clip" in xml


def test_scaffold_skips_shots_without_clip(tmp_output, source_video):
    clip = make_clip()
    clip = clip.model_copy(update={"source_path": source_video})
    # s2 has no clip
    shots = [make_shot("s1", clip), make_shot("s2", None)]
    sb = make_storyboard(shots)

    path = build_scaffold(sb, {"s1": clip}, output_path=tmp_output)
    xml = Path(path).read_text()

    result = compile_fcpxml(xml, run_dtd=False)
    assert result.valid


def test_scaffold_empty_storyboard(tmp_output):
    sb = make_storyboard([])
    path = build_scaffold(sb, {}, output_path=tmp_output)
    xml = Path(path).read_text()
    result = compile_fcpxml(xml, run_dtd=False)
    assert result.valid


def test_scaffold_multiple_shots_sequential(tmp_output, source_video):
    """Verify offset increases for each shot."""
    clip1 = make_clip(start=0, end=10)
    clip1 = clip1.model_copy(update={"source_path": source_video})
    clip2 = make_clip(start=20, end=30)
    clip2 = clip2.model_copy(update={"source_path": source_video})

    shots = [make_shot("s1", clip1), make_shot("s2", clip2)]
    sb = make_storyboard(shots)
    path = build_scaffold(sb, {"s1": clip1, "s2": clip2}, output_path=tmp_output)

    xml = Path(path).read_text()
    root = fromstring(xml.split("<!DOCTYPE fcpxml>")[-1].strip())
    spine_clips = root.findall(".//spine/asset-clip")
    assert len(spine_clips) == 2

    # Second clip should have a non-zero offset
    offsets = [c.get("offset") for c in spine_clips]
    # First clip offset is 0 (may render as "0s" or "0/1s" — both valid)
    from fractions import Fraction
    assert Fraction(offsets[0].rstrip("s")) == 0
    assert Fraction(offsets[1].rstrip("s")) > 0


def test_scaffold_with_narration(tmp_output, source_video, tmp_path):
    narration = tmp_path / "narration.mp3"
    narration.write_bytes(b"fake audio")

    clip = make_clip()
    clip = clip.model_copy(update={"source_path": source_video})
    sb = make_storyboard([make_shot("s1", clip)])

    path = build_scaffold(
        sb,
        {"s1": clip},
        output_path=tmp_output,
        narration_path=str(narration),
        narration_duration=10.0,
    )
    xml = Path(path).read_text()
    result = compile_fcpxml(xml, run_dtd=False)
    assert result.valid
    assert 'lane="-1"' in xml
    assert 'role="dialogue"' in xml


def test_scaffold_with_music(tmp_output, source_video, tmp_path):
    music = tmp_path / "music.mp3"
    music.write_bytes(b"fake music")

    clip = make_clip()
    clip = clip.model_copy(update={"source_path": source_video})
    sb = make_storyboard([make_shot("s1", clip)])

    path = build_scaffold(
        sb,
        {"s1": clip},
        output_path=tmp_output,
        music_path=str(music),
        music_duration=60.0,
        music_gain_db=-12.0,
    )
    xml = Path(path).read_text()
    result = compile_fcpxml(xml, run_dtd=False)
    assert result.valid
    assert 'lane="-2"' in xml
    assert 'role="music"' in xml
    assert "-12.0dB" in xml
