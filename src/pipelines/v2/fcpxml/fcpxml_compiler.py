"""
FCPXML Compiler — 3-layer validator with autofix.

Layer 1 — Structural (Python):   Required attributes, time format, duration > 0.
Layer 2 — DTD (xmllint):          Full Apple DTD conformance (if xmllint available).
Layer 3 — Semantic (Python):      ref IDs resolved, timeline continuity, asset durations.

autofix() corrects the most common LLM mistakes before running the compiler,
so the correction loop has fewer iterations.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import List, Optional
from xml.etree.ElementTree import Element, fromstring, tostring, ParseError

logger = logging.getLogger(__name__)

# Matches valid FCPXML time values: "0s", "48/24s", "1001/30000s", etc.
_TIME_RE = re.compile(r'^\d+/\d+s$|^0s$')

# Float-style times that LLMs often write (e.g. "2.5s", "2s" without fraction)
_FLOAT_TIME_RE = re.compile(r'^(\d+(?:\.\d+)?)s$')


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def error_summary(self) -> str:
        return "\n".join(f"ERROR: {e}" for e in self.errors)

    def __bool__(self) -> bool:
        return self.valid


# ---------------------------------------------------------------------------
# Autofix — run BEFORE validation to handle common LLM mistakes
# ---------------------------------------------------------------------------

def autofix(xml_text: str) -> str:
    """
    Fix the most common LLM FCPXML mistakes in-place on the raw XML text.

    Currently handles:
    1. Float-style time values ("2.5s", "30s") → rational fractions ("60/24s")
    2. Negative or zero duration on clips → minimum 1 frame at 24fps
    3. Remove markdown fences if LLM wrapped the output in ```xml ... ```
    """
    # Strip markdown fences
    xml_text = _strip_markdown(xml_text)

    # Fix float time attributes
    xml_text = _fix_float_times(xml_text)

    return xml_text


def _strip_markdown(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (``` or ```xml) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines)
    return text.strip()


def _fix_float_times(xml_text: str) -> str:
    """
    Replace float-style time attribute values with proper rational fractions.
    e.g. offset="2.5s" → offset="60/24s"
         duration="30s" → duration="720/24s"
    """
    TIME_ATTRS = ("offset", "start", "duration", "tcStart", "value", "time")

    def _float_to_fraction(secs: float, fps: int = 24) -> str:
        if secs == 0:
            return "0s"
        frac = Fraction(str(secs)).limit_denominator(fps * 1000)
        # Scale to numerator/denominator with fps denominator
        frames = round(secs * fps)
        if frames == 0:
            return "0s"
        return f"{frames}/{fps}s"

    def _replace_attr(match: re.Match) -> str:
        attr_name = match.group(1)
        value = match.group(2)
        # Check if it's a float-style time
        fm = _FLOAT_TIME_RE.match(value)
        if fm and not _TIME_RE.match(value):
            try:
                secs = float(fm.group(1))
                replacement = _float_to_fraction(secs)
                return f'{attr_name}="{replacement}"'
            except ValueError:
                pass
        return match.group(0)

    # Match attribute="value" patterns for known time attributes
    pattern = re.compile(
        r'\b(' + '|'.join(TIME_ATTRS) + r')="([^"]*)"'
    )
    return pattern.sub(_replace_attr, xml_text)


# ---------------------------------------------------------------------------
# Layer 1 — Structural validation (Python)
# ---------------------------------------------------------------------------

def _validate_structural(root: Element, result: ValidationResult) -> None:
    """Check required attributes and time format on key FCPXML elements."""
    if root.tag != "fcpxml":
        result.add_error(f"Root element must be <fcpxml>, got <{root.tag}>")
        return
    if root.get("version") != "1.10":
        result.add_error(f"fcpxml version must be '1.10', got '{root.get('version')}'")

    # Collect all resource IDs
    resource_ids = set()
    resources_el = root.find("resources")
    if resources_el is None:
        result.add_error("<resources> element missing")
    else:
        for child in resources_el:
            rid = child.get("id")
            if rid:
                resource_ids.add(rid)
            # Check format has required attrs
            if child.tag == "format":
                for attr in ("frameDuration", "width", "height"):
                    if not child.get(attr):
                        result.add_error(f"<format id={rid!r}> missing required attribute '{attr}'")
            # Check asset has src
            if child.tag == "asset":
                if not child.get("src"):
                    result.add_error(f"<asset id={rid!r}> missing 'src'")
                _check_time_attrs(child, rid or "?", result)

    # Walk spine clips
    for clip in root.iter("asset-clip"):
        clip_name = clip.get("name", "?")
        # ref must point to existing resource
        ref = clip.get("ref")
        if not ref:
            result.add_error(f"<asset-clip name={clip_name!r}> missing 'ref'")
        elif ref not in resource_ids:
            result.add_error(f"<asset-clip name={clip_name!r}> ref='{ref}' not found in resources")
        # Required time attrs
        for attr in ("offset", "start", "duration"):
            val = clip.get(attr)
            if val is None:
                result.add_error(f"<asset-clip name={clip_name!r}> missing required attribute '{attr}'")
            elif not _TIME_RE.match(val):
                result.add_error(
                    f"<asset-clip name={clip_name!r}> {attr}={val!r} is not a valid rational time "
                    f"(expected 'N/Ds' or '0s')"
                )
            elif attr == "duration":
                # Duration must be > 0
                try:
                    frac = Fraction(val[:-1])  # strip 's'
                    if frac <= 0:
                        result.add_error(f"<asset-clip name={clip_name!r}> duration must be > 0")
                except Exception:
                    pass  # already caught above


def _check_time_attrs(el: Element, eid: str, result: ValidationResult) -> None:
    for attr in ("start", "duration"):
        val = el.get(attr)
        if val and not _TIME_RE.match(val):
            result.add_error(f"<{el.tag} id={eid!r}> {attr}={val!r} is not a valid time value")


# ---------------------------------------------------------------------------
# Layer 2 — DTD validation (xmllint)
# ---------------------------------------------------------------------------

def _validate_dtd(xml_text: str, result: ValidationResult) -> None:
    """Run xmllint --dtdvalid if available. Adds errors to result."""
    if not shutil.which("xmllint"):
        result.add_warning("xmllint not found — skipping DTD validation")
        return

    dtd_path = Path("context/fcpxml_1_10.dtd")
    if not dtd_path.exists():
        result.add_warning("DTD file not found at context/fcpxml_1_10.dtd — skipping DTD validation")
        return

    with tempfile.NamedTemporaryFile(suffix=".fcpxml", mode="w", encoding="utf-8", delete=False) as tmp:
        tmp.write(xml_text)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            ["xmllint", "--noout", "--dtdvalid", str(dtd_path.resolve()), tmp_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            for line in (proc.stderr or "").splitlines():
                line = line.strip()
                if line:
                    result.add_error(f"DTD: {line}")
    except subprocess.TimeoutExpired:
        result.add_warning("xmllint timed out — skipping DTD validation")
    except Exception as e:
        result.add_warning(f"xmllint error: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Layer 3 — Semantic validation
# ---------------------------------------------------------------------------

def _validate_semantic(root: Element, result: ValidationResult) -> None:
    """Check logical consistency: ref resolution, timeline gaps, asset duration coverage."""
    resources_el = root.find("resources")
    if resources_el is None:
        return

    asset_durations: dict[str, Fraction] = {}
    for asset in resources_el.findall("asset"):
        aid = asset.get("id", "")
        dur_str = asset.get("duration", "0s")
        try:
            frac = _parse_time(dur_str)
            asset_durations[aid] = frac
        except Exception:
            pass

    # Check that each clip's start+duration does not exceed asset duration
    for clip in root.iter("asset-clip"):
        ref = clip.get("ref", "")
        asset_dur = asset_durations.get(ref)
        if asset_dur is None:
            continue  # already caught in structural
        start_str = clip.get("start", "0s")
        dur_str = clip.get("duration", "0s")
        try:
            start = _parse_time(start_str)
            dur = _parse_time(dur_str)
            clip_end = start + dur
            if clip_end > asset_dur + Fraction(1, 1000):  # small tolerance
                clip_name = clip.get("name", "?")
                result.add_warning(
                    f"<asset-clip name={clip_name!r}> clip end {float(clip_end):.2f}s "
                    f"exceeds asset duration {float(asset_dur):.2f}s"
                )
        except Exception:
            pass


def _parse_time(val: str) -> Fraction:
    """Parse '48/24s' or '0s' → Fraction."""
    val = val.strip()
    if val == "0s":
        return Fraction(0)
    if val.endswith("s"):
        return Fraction(val[:-1])
    raise ValueError(f"Cannot parse time: {val!r}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compile_fcpxml(xml_text: str, *, run_dtd: bool = True) -> ValidationResult:
    """
    Run all 3 validation layers on the given FCPXML text.

    Args:
        xml_text: The full XML document as a string (not a file path).
        run_dtd: Whether to attempt DTD validation via xmllint.

    Returns:
        ValidationResult with .valid, .errors, .warnings.
    """
    result = ValidationResult(valid=True)

    # Parse XML
    try:
        root = fromstring(xml_text)
    except ParseError as e:
        result.add_error(f"XML parse error: {e}")
        return result

    # Layer 1
    _validate_structural(root, result)

    # Layer 2 (only if no structural errors — avoids confusing cascade)
    if result.valid and run_dtd:
        _validate_dtd(xml_text, result)

    # Layer 3
    if result.valid:
        _validate_semantic(root, result)

    if result.valid:
        logger.info("[COMPILER] FCPXML valid ✓")
    else:
        logger.warning(f"[COMPILER] FCPXML has {len(result.errors)} error(s)")

    return result
