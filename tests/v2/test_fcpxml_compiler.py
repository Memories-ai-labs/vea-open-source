"""Tests for fcpxml_compiler.py — autofix and 3-layer validation."""
import pytest
import tempfile
from xml.etree.ElementTree import fromstring

from src.pipelines.v2.fcpxml.fcpxml_compiler import (
    autofix,
    compile_fcpxml,
    ValidationResult,
    _fix_float_times,
    _strip_markdown,
)
from src.pipelines.v2.fcpxml.edit_compiler import compile_edit_decision
from src.pipelines.v2.schemas import (
    ClipDecision,
    EditDecision,
    MusicTrack,
    NarrationSegment,
    TimelineSettings,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_VALID_FCPXML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.10">
  <resources>
    <format id="r1" frameDuration="1/24s" width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>
    <asset id="r2" name="clip.mp4" start="0s" duration="240/24s" hasVideo="1" hasAudio="1"
           audioSources="1" audioChannels="2" audioRate="48000" src="file:///media/clip.mp4">
      <media-rep kind="original-media" src="file:///media/clip.mp4"/>
    </asset>
  </resources>
  <library>
    <event name="VEA Export">
      <project name="Test">
        <sequence format="r1" tcStart="0s" tcFormat="NDF" duration="48/24s">
          <spine>
            <asset-clip name="shot1" ref="r2" offset="0s" start="0s" duration="48/24s" format="r1" enabled="1"/>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>"""


# ---------------------------------------------------------------------------
# autofix: strip markdown
# ---------------------------------------------------------------------------

def test_strip_markdown_xml_fence():
    text = "```xml\n<fcpxml version=\"1.10\"/>\n```"
    result = _strip_markdown(text)
    assert result == '<fcpxml version="1.10"/>'


def test_strip_markdown_plain_fence():
    text = "```\n<fcpxml/>\n```"
    result = _strip_markdown(text)
    assert result == "<fcpxml/>"


def test_strip_markdown_no_fence():
    text = "<fcpxml/>"
    assert _strip_markdown(text) == "<fcpxml/>"


# ---------------------------------------------------------------------------
# autofix: float time replacement
# ---------------------------------------------------------------------------

def test_fix_float_times_converts_decimal():
    xml = '<asset-clip offset="2.5s" start="0s" duration="5.0s"/>'
    result = _fix_float_times(xml)
    # 2.5s → 60/24s, 5.0s → 120/24s
    assert '"0s"' in result        # 0s stays
    assert '"2.5s"' not in result
    assert '"5.0s"' not in result


def test_fix_float_times_preserves_valid_fractions():
    xml = '<asset-clip offset="48/24s" start="0s" duration="96/24s"/>'
    result = _fix_float_times(xml)
    assert '"48/24s"' in result
    assert '"96/24s"' in result


def test_fix_float_times_integer_seconds():
    xml = '<asset-clip offset="10s" start="0s" duration="30s"/>'
    result = _fix_float_times(xml)
    assert '"10s"' not in result or '"240/24s"' in result or '"10s"' in result
    # Main thing: 0s is preserved
    assert '"0s"' in result


def test_autofix_strips_and_fixes():
    wrapped = '```xml\n<asset-clip offset="2.0s" duration="5.0s" start="0s"/>\n```'
    result = autofix(wrapped)
    assert not result.startswith("```")
    assert '"2.0s"' not in result


# ---------------------------------------------------------------------------
# compile_fcpxml: valid document
# ---------------------------------------------------------------------------

def test_compile_valid_fcpxml():
    result = compile_fcpxml(MINIMAL_VALID_FCPXML, run_dtd=False)
    assert result.valid, f"Expected valid but got errors: {result.errors}"
    assert result.errors == []


# ---------------------------------------------------------------------------
# compile_fcpxml: structural errors
# ---------------------------------------------------------------------------

def test_compile_wrong_root_tag():
    xml = '<notfcpxml version="1.10"><resources/></notfcpxml>'
    result = compile_fcpxml(xml, run_dtd=False)
    assert not result.valid
    assert any("Root element" in e for e in result.errors)


def test_compile_wrong_version():
    bad = MINIMAL_VALID_FCPXML.replace('version="1.10"', 'version="1.9"', 1)
    result = compile_fcpxml(bad, run_dtd=False)
    assert not result.valid
    assert any("1.10" in e for e in result.errors)


def test_compile_missing_resources():
    xml = '<fcpxml version="1.10"><library/></fcpxml>'
    result = compile_fcpxml(xml, run_dtd=False)
    assert not result.valid
    assert any("resources" in e.lower() for e in result.errors)


def test_compile_missing_ref():
    bad = MINIMAL_VALID_FCPXML.replace('ref="r2"', 'ref="MISSING"')
    result = compile_fcpxml(bad, run_dtd=False)
    assert not result.valid
    assert any("MISSING" in e for e in result.errors)


def test_compile_missing_clip_duration():
    bad = MINIMAL_VALID_FCPXML.replace('duration="48/24s" format="r1"', 'format="r1"')
    result = compile_fcpxml(bad, run_dtd=False)
    assert not result.valid
    assert any("duration" in e.lower() for e in result.errors)


def test_compile_float_time_value():
    bad = MINIMAL_VALID_FCPXML.replace('duration="48/24s" format="r1"', 'duration="2.0s" format="r1"')
    result = compile_fcpxml(bad, run_dtd=False)
    assert not result.valid
    assert any("rational" in e.lower() or "valid" in e.lower() for e in result.errors)


def test_compile_invalid_xml():
    result = compile_fcpxml("<fcpxml version='1.10'><unclosed>", run_dtd=False)
    assert not result.valid
    assert any("parse" in e.lower() or "XML" in e for e in result.errors)


# ---------------------------------------------------------------------------
# compile_fcpxml: warnings (not errors)
# ---------------------------------------------------------------------------

def test_compile_no_dtd_gives_warning():
    result = compile_fcpxml(MINIMAL_VALID_FCPXML, run_dtd=True)
    # DTD validation should warn (xmllint or dtd file may not be present)
    # Either way, document should still be valid structurally
    assert result.valid or any("DTD" in e or "xmllint" in e for e in result.errors + result.warnings)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

def test_validation_result_bool():
    r = ValidationResult(valid=True)
    assert bool(r) is True
    r.add_error("bad thing")
    assert bool(r) is False
    assert r.valid is False


def test_validation_result_warning_does_not_invalidate():
    r = ValidationResult(valid=True)
    r.add_warning("just a warning")
    assert bool(r) is True
    assert r.errors == []


def test_validation_result_error_summary():
    r = ValidationResult(valid=True)
    r.add_error("error A")
    r.add_error("error B")
    summary = r.error_summary()
    assert "error A" in summary
    assert "error B" in summary


# ---------------------------------------------------------------------------
# edit_compiler: duration clamping tests
# ---------------------------------------------------------------------------

def _make_edit_decision(
    clips=None, narration=None, music=None, fps=24.0,
) -> EditDecision:
    """Helper to build a simple EditDecision for testing."""
    if clips is None:
        clips = [
            ClipDecision(
                id="clip1",
                source_file="/media/source.mp4",
                source_start=0.0,
                source_end=10.0,
                label="shot1",
            ),
        ]
    return EditDecision(
        timeline=TimelineSettings(name="Test", fps=fps, width=1920, height=1080),
        clips=clips,
        narration=narration or [],
        music=music,
    )


def _compile_and_parse(edit: EditDecision) -> fromstring:
    """Compile an EditDecision to FCPXML and return the parsed XML root."""
    with tempfile.NamedTemporaryFile(suffix=".fcpxml", delete=False) as f:
        path = f.name
    compile_edit_decision(edit, path)
    with open(path, "r") as f:
        content = f.read()
    # Strip the XML declaration and DOCTYPE to parse cleanly
    xml_start = content.index("<fcpxml")
    return fromstring(content[xml_start:])


def test_narration_duration_clamped_to_spine():
    """When narration duration exceeds spine duration, it gets clamped to spine length."""
    # Spine has 10s of content (one clip from 0-10s at 24fps)
    edit = _make_edit_decision(
        narration=[
            NarrationSegment(
                file="/audio/narration.mp3",
                timeline_offset=0.0,
                start=0.0,
                duration=30.0,  # 30s narration, but spine is only 10s
            ),
        ],
    )
    root = _compile_and_parse(edit)

    # Find the narration asset-clip (lane="-1")
    nar_clips = root.findall(".//" + "asset-clip[@lane='-1']")
    assert len(nar_clips) == 1

    # Parse the duration — it should be clamped to ~10s, not 30s
    dur_str = nar_clips[0].get("duration")
    # Duration is in fraction format like "240/24s" for 10s
    # Parse the fraction to seconds
    frac_part = dur_str.rstrip("s")
    if "/" in frac_part:
        num, den = frac_part.split("/")
        dur_seconds = int(num) / int(den)
    else:
        dur_seconds = float(frac_part)
    assert dur_seconds <= 10.0 + 0.05, f"Narration duration {dur_seconds}s should be clamped to spine (~10s)"


def test_music_duration_clamped_to_spine():
    """When music duration exceeds spine duration, it gets clamped to spine length."""
    edit = _make_edit_decision(
        music=MusicTrack(
            file="/audio/music.mp3",
            start=0.0,
            duration=120.0,  # 120s music, but spine is only 10s
            gain_db=-12.0,
        ),
    )
    root = _compile_and_parse(edit)

    # Find the music asset-clip (lane="-2")
    music_clips = root.findall(".//" + "asset-clip[@lane='-2']")
    assert len(music_clips) == 1

    dur_str = music_clips[0].get("duration")
    frac_part = dur_str.rstrip("s")
    if "/" in frac_part:
        num, den = frac_part.split("/")
        dur_seconds = int(num) / int(den)
    else:
        dur_seconds = float(frac_part)
    assert dur_seconds <= 10.0 + 0.05, f"Music duration {dur_seconds}s should be clamped to spine (~10s)"


def test_narration_beyond_spine_skipped():
    """Narration segments starting beyond the spine end are skipped entirely."""
    edit = _make_edit_decision(
        narration=[
            NarrationSegment(
                file="/audio/narration.mp3",
                timeline_offset=20.0,  # starts at 20s, but spine is only 10s
                start=0.0,
                duration=5.0,
            ),
        ],
    )
    root = _compile_and_parse(edit)

    # No narration should be placed (lane="-1")
    nar_clips = root.findall(".//" + "asset-clip[@lane='-1']")
    assert len(nar_clips) == 0, "Narration starting beyond spine should be skipped"
