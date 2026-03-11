"""ScratchpadManager — persistent context pads that survive the sliding window."""

import logging
from pathlib import Path
from typing import Dict, Literal

logger = logging.getLogger(__name__)

PAD_NAMES = ("comprehension", "creative_direction", "planning", "fcpxml")
MAX_PAD_SIZE = 6000  # chars per pad


class ScratchpadManager:
    """
    Manages 4 named scratchpads with workspace persistence.

    Scratchpads are always included in the Gemini system instruction.
    They are the model's ONLY durable memory — anything not written
    to a scratchpad will eventually fall out of the conversation window.
    """

    def __init__(self, workspace_root: Path):
        self._dir = workspace_root / "scratchpads"
        self._dir.mkdir(parents=True, exist_ok=True)
        self.pads: Dict[str, str] = {name: "" for name in PAD_NAMES}
        self._load()

    # ── Public API ────────────────────────────────────────────────────────

    def update(
        self,
        name: str,
        operation: Literal["replace", "append", "prepend"],
        content: str,
    ) -> dict:
        """Apply an operation to a scratchpad and persist."""
        if name not in PAD_NAMES:
            return {"error": f"Unknown scratchpad: {name}. Valid: {PAD_NAMES}"}

        if operation == "replace":
            self.pads[name] = content
        elif operation == "append":
            self.pads[name] += content
        elif operation == "prepend":
            self.pads[name] = content + self.pads[name]
        else:
            return {"error": f"Unknown operation: {operation}. Valid: replace, append, prepend"}

        # Enforce size cap
        if len(self.pads[name]) > MAX_PAD_SIZE:
            logger.warning(
                f"[SCRATCHPAD] {name} exceeds {MAX_PAD_SIZE} chars "
                f"({len(self.pads[name])}). Truncating from the start."
            )
            self.pads[name] = self.pads[name][-MAX_PAD_SIZE:]

        self._save(name)
        return {
            "status": "updated",
            "name": name,
            "operation": operation,
            "length": len(self.pads[name]),
        }

    def read(self, name: str) -> str:
        if name not in PAD_NAMES:
            return f"(unknown scratchpad: {name})"
        return self.pads[name]

    def render_all(self) -> str:
        """Render all scratchpads for inclusion in the system instruction."""
        blocks = []
        for name in PAD_NAMES:
            content = self.pads[name] or "(empty)"
            blocks.append(
                f"=== SCRATCHPAD: {name} ===\n"
                f"{content}\n"
                f"=== END {name} ==="
            )
        return "\n\n".join(blocks)

    def seed_comprehension(self, gist: str) -> None:
        """Pre-populate comprehension from indexing gist if pad is empty."""
        if not self.pads["comprehension"] and gist:
            self.pads["comprehension"] = gist
            self._save("comprehension")

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        for name in PAD_NAMES:
            path = self._dir / f"{name}.md"
            if path.exists():
                self.pads[name] = path.read_text()

    def _save(self, name: str) -> None:
        path = self._dir / f"{name}.md"
        path.write_text(self.pads[name])
