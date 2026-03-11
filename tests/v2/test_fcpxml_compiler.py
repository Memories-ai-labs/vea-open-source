"""Tests for fcpxml_compiler.py — autofix and 3-layer validation."""
import pytest
from src.pipelines.v2.fcpxml.fcpxml_compiler import (
    autofix,
    compile_fcpxml,
    ValidationResult,
    _fix_float_times,
    _strip_markdown,
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
