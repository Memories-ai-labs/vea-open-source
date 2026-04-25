"""Per-workspace structured logging.

Every live agent session activates a workspace-scoped log scope. While the
scope is active, all Python ``logging`` output (anything tagged ``[AGENT]``,
``[RENDER]``, ``[MEMORIES]``, ``[COMPILER]`` etc.) is mirrored as JSON lines
into ``{workspace}/logs/backend.jsonl``. Separate helpers append to
``llm.jsonl`` (one record per LLM call) and ``renders/<name>-<ts>.log``
(ffmpeg / Resolve subprocess stderr).

Contextvars carry the active workspace through asyncio ``create_task`` child
tasks — a background render spawned from inside a session still writes to
the right project's log files.

Cleared on ``clear/planning``. Nothing here is retained across sessions,
apart from ``manifest.json`` which is re-written at each session start.
"""
from __future__ import annotations

import json
import logging
import platform
import re
import subprocess
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Which workspace the current task is writing logs for. ``None`` means the
# caller isn't inside a scoped session — log records are dropped rather than
# leaking to an unrelated project's logs/ dir.
_active_workspace: ContextVar[Optional[Path]] = ContextVar(
    "vea_active_workspace", default=None
)

# Field names or substrings that almost always hold secrets. We redact both
# dict keys matching this pattern and the typical "Bearer xxx" shape inside
# free-form strings. "token" is anchored against neighboring letters so the
# legitimate ``tokens`` field (prompt/output counts) isn't redacted — that
# would hide useful debugging info.
_SECRET_KEY_RE = re.compile(
    r"(?i)("
    r"api[_-]?key|"
    r"secret|"
    r"authorization|"
    r"password|"
    r"access[_-]token|auth[_-]token|refresh[_-]token|"
    r"(?<![a-z])token(?![a-z])"
    r")"
)
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}")


def redact(obj: Any, _depth: int = 0) -> Any:
    """Return a deep copy of ``obj`` with secret-looking values masked.

    Strings are scanned for bearer tokens; dict keys matching ``_SECRET_KEY_RE``
    have their values replaced wholesale with ``"***"``. Depth is capped to
    guard against self-referential structures from e.g. SDK response objects.
    """
    if _depth > 8:
        return "<depth-limit>"
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if _SECRET_KEY_RE.search(str(k)):
                out[str(k)] = "***"
            else:
                out[str(k)] = redact(v, _depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact(v, _depth + 1) for v in obj]
    if isinstance(obj, str):
        return _BEARER_RE.sub(r"\1***", obj)
    return obj


class _JSONLFormatter(logging.Formatter):
    """Serialize a LogRecord as one JSON line.

    Anything passed as ``extra={"vea_foo": ...}`` surfaces as a top-level
    ``foo`` field — an escape hatch for structured context without adding
    more log statements.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k.startswith("vea_"):
                payload[k[4:]] = v
        return json.dumps(payload, default=str)


class _WorkspaceHandler(logging.Handler):
    """logging.Handler that routes records by the active workspace contextvar.

    One handler lives on the root logger; it writes each record to whichever
    project is currently in scope, or drops it when nothing is active. This
    keeps the setup cheap (no per-session handler churn) while still scoping
    output correctly across concurrent sessions.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.setFormatter(_JSONLFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        ws = _active_workspace.get()
        if ws is None:
            return
        try:
            line = self.format(record)
            path = ws / "logs" / "backend.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(line + "\n")
        except Exception:
            # A logging failure must never propagate — a broken disk or a
            # racing clear-planning wipe shouldn't crash the agent loop.
            pass


_installed = False


def install() -> None:
    """Attach the workspace handler to the root logger once per process."""
    global _installed
    if _installed:
        return
    handler = _WorkspaceHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    # Ensure the root level is at least INFO so our handler gets records even
    # when no other handler forces verbosity. We don't lower existing handlers.
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    _installed = True


class WorkspaceLogScope:
    """Context manager that binds ``workspace_root`` for the current task.

    Use this around any coroutine whose logs should land in a specific
    project's bundle — notably ``AgentSession.handle_user_message`` and any
    long-lived background render task. Child tasks spawned via
    ``asyncio.create_task`` inherit the scope automatically.
    """

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = Path(workspace_root)
        self._token = None

    def __enter__(self) -> "WorkspaceLogScope":
        install()
        self._token = _active_workspace.set(self.workspace_root)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _active_workspace.reset(self._token)
            self._token = None


def activate(workspace_root: Path) -> None:
    """Permanent activation for the current task (no explicit exit).

    Prefer ``WorkspaceLogScope`` for request-handler flows; ``activate`` is for
    process-wide single-project CLIs where there's nothing to restore on exit.
    """
    install()
    _active_workspace.set(Path(workspace_root))


def append_llm_event(workspace_root: Path, event: Dict[str, Any]) -> None:
    """Append one JSONL record to ``logs/llm.jsonl``, redacted.

    Used at every LLM call site so we capture prompt, response, tokens, and
    duration without needing a proxy. Missing ``logs/`` directory is created.
    """
    path = Path(workspace_root) / "logs" / "llm.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **redact(event),
        }
        with open(path, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception as e:
        logger.warning(f"[LOG] failed to append llm event: {e}")


def open_render_log(workspace_root: Path, renderer: str) -> Path:
    """Return a fresh path under ``logs/renders/`` for a single render pass.

    Caller is responsible for writing the subprocess stderr to the returned
    path. Filename includes the renderer name and a timestamp so repeated
    renders don't overwrite prior logs within the same session.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", renderer)
    path = Path(workspace_root) / "logs" / "renders" / f"{safe}-{ts}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(
    workspace_root: Path,
    *,
    project_name: str,
    models: Dict[str, str],
    mode: str,
) -> None:
    """Snapshot the runtime env to ``logs/manifest.json``.

    Called once per AgentSession init. Overwrites any prior file so the
    manifest always reflects the *current* run, not the first one that ever
    touched this workspace.
    """
    path = Path(workspace_root) / "logs" / "manifest.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "project_name": project_name,
            "mode": mode,
            "git_sha": _git_sha(),
            "ffmpeg_version": _ffmpeg_version(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "models": models,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
    except Exception as e:
        logger.warning(f"[LOG] failed to write manifest: {e}")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return (out.stdout or "").strip() or "unknown"
    except Exception:
        return "unknown"


def _ffmpeg_version() -> str:
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=3,
        )
        first = (out.stdout or "").splitlines()[0:1]
        return first[0].strip() if first else "unknown"
    except Exception:
        return "unknown"
