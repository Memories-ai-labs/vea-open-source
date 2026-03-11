"""Tool declarations and implementations for the agentic editing session."""

import logging
from typing import Any, Dict, List, Optional

from google.genai.types import FunctionDeclaration, Tool

from src.pipelines.v2.agent.scratchpad import ScratchpadManager

logger = logging.getLogger(__name__)

# ── Gemini function declarations ──────────────────────────────────────────────

TOOL_DECLARATIONS = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="ask_memories",
            description=(
                "Ask a natural-language question about the indexed video footage. "
                "Memories.ai has watched every frame and can answer questions about "
                "content, people, dialogue, visuals, timing, and structure. "
                "Returns a text answer."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask about the videos.",
                    },
                },
                "required": ["question"],
            },
        ),
        FunctionDeclaration(
            name="search_footage",
            description=(
                "Search for specific video clips matching a query. "
                "Returns clips with video_name, start/end timestamps, "
                "relevance score, and description."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing the clip you want.",
                    },
                    "target_duration_seconds": {
                        "type": "number",
                        "description": "Desired clip length in seconds. Default 5.",
                    },
                },
                "required": ["query"],
            },
        ),
        FunctionDeclaration(
            name="update_scratchpad",
            description=(
                "Modify one of your 4 persistent scratchpads. These survive the "
                "sliding window — they are your ONLY durable memory. "
                "Names: comprehension, creative_direction, planning, fcpxml. "
                "Operations: replace (overwrite), append (add to end), prepend (add to start)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": ["comprehension", "creative_direction", "planning", "fcpxml"],
                        "description": "Which scratchpad to update.",
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["replace", "append", "prepend"],
                        "description": "How to apply the content.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text to write.",
                    },
                },
                "required": ["name", "operation", "content"],
            },
        ),
        FunctionDeclaration(
            name="generate_fcpxml",
            description=(
                "Generate a Final Cut Pro XML timeline from a complete edit decision. "
                "Call this when the edit plan is finalized and clips are assigned. "
                "Provide the edit as a JSON string. The system compiles it deterministically "
                "to valid FCPXML 1.10."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "edit_decision_json": {
                        "type": "string",
                        "description": (
                            "A JSON string representing the full edit decision. Schema: "
                            '{"timeline": {"name": str, "fps": number, "width": int, "height": int}, '
                            '"clips": [{"id": str, "source_file": str, "source_start": number, '
                            '"source_end": number, "label": str, "gain_db": number, '
                            '"speed": {"rate": number}, '
                            '"transition_after": {"type": "cross-dissolve"|"fade-in"|"fade-out", "duration_seconds": number}}], '
                            '"narration": [{"file": str, "timeline_offset": number, "start": number, '
                            '"duration": number, "gain_db": number}], '
                            '"music": {"file": str, "start": number, "duration": number, "gain_db": number}, '
                            '"titles": [{"text": str, "timeline_offset": number, "duration": number, "font_size": int}]}. '
                            "Only clips is required. timeline defaults to 24fps 1920x1080."
                        ),
                    },
                },
                "required": ["edit_decision_json"],
            },
        ),
        FunctionDeclaration(
            name="message_user",
            description=(
                "Send a visible message to the user in the chat interface. "
                "Use this to share findings, propose plans, ask questions, or report progress."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to show to the user.",
                    },
                },
                "required": ["message"],
            },
        ),
    ]
)


# ── Tool executor ─────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Executes tool calls from Gemini and returns results.

    Each tool call emits events over the event_callback so the frontend
    can show live status.
    """

    def __init__(
        self,
        memories_manager,
        gemini_manager,
        workspace,
        scratchpads: ScratchpadManager,
        video_nos: List[str],
        event_callback=None,
    ):
        self.memories = memories_manager
        self.gemini = gemini_manager
        self.workspace = workspace
        self.scratchpads = scratchpads
        self.video_nos = video_nos
        self._emit = event_callback or (lambda *a, **kw: None)

    async def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return the result dict."""
        try:
            if tool_name == "ask_memories":
                return await self._ask_memories(args)
            elif tool_name == "search_footage":
                return await self._search_footage(args)
            elif tool_name == "update_scratchpad":
                return self._update_scratchpad(args)
            elif tool_name == "generate_fcpxml":
                return await self._generate_fcpxml(args)
            elif tool_name == "message_user":
                return self._message_user(args)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"[AGENT] Tool {tool_name} failed: {e}", exc_info=True)
            return {"error": str(e)}

    # ── Individual tool implementations ───────────────────────────────────

    async def _ask_memories(self, args: Dict) -> Dict:
        question = args.get("question", "")
        logger.info(f"[AGENT] ask_memories: {question[:100]}")

        response = await self.memories.chat(
            video_nos=self.video_nos,
            prompt=question,
        )

        result = {
            "answer": response.text,
            "reference_count": len(response.ref_items),
        }

        # Include timestamped references if available
        if response.references:
            result["references"] = [
                {
                    "video_name": ref.video_name,
                    "timestamp": ref.timestamp,
                    "description": ref.description,
                }
                for ref in response.references[:10]  # cap at 10
            ]

        return result

    async def _search_footage(self, args: Dict) -> Dict:
        query = args.get("query", "")
        target_duration = args.get("target_duration_seconds", 5.0)
        logger.info(f"[AGENT] search_footage: {query[:100]}")

        raw_results = await self.memories.search(
            query,
            search_type="BY_CLIP",
            video_nos=self.video_nos,
        )

        # raw_results is either a list of dicts or a dict with an "items" key
        items = []
        if isinstance(raw_results, list):
            items = raw_results
        elif isinstance(raw_results, dict):
            items = raw_results.get("items", [])

        clips = []
        for item in items[:15]:  # cap results
            if not isinstance(item, dict):
                continue
            video_no = item.get("videoNo", "")
            start = float(item.get("startTime", 0))
            end_raw = item.get("endTime")
            end = float(end_raw) if end_raw else start + target_duration
            # Use target_duration if the returned clip is too short
            if end - start < target_duration:
                end = start + target_duration
            score = float(item.get("score", 0))

            clips.append({
                "video_no": video_no,
                "video_name": item.get("videoName", video_no),
                "start_seconds": start,
                "end_seconds": end,
                "score": score,
            })

        return {"clips": clips, "count": len(clips), "query": query}

    def _update_scratchpad(self, args: Dict) -> Dict:
        name = args.get("name", "")
        operation = args.get("operation", "replace")
        content = args.get("content", "")
        logger.info(f"[AGENT] update_scratchpad: {name} ({operation}, {len(content)} chars)")
        return self.scratchpads.update(name, operation, content)

    async def _generate_fcpxml(self, args: Dict) -> Dict:
        import json
        from src.pipelines.v2.schemas import EditDecision
        from src.pipelines.v2.fcpxml.edit_compiler import compile_edit_decision

        # Parse the JSON string from the tool call
        raw_json = args.get("edit_decision_json", "{}")
        try:
            edit_data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}"}

        logger.info(f"[AGENT] generate_fcpxml: {len(edit_data.get('clips', []))} clips")

        # Validate and parse the EditDecision
        try:
            edit = EditDecision.model_validate(edit_data)
        except Exception as e:
            return {"error": f"Invalid EditDecision: {e}"}

        # Resolve source_path for clips that only have source_file
        for clip in edit.clips:
            if not clip.source_path and clip.source_file:
                # Try to find the source file in the workspace footage
                for video in (self.workspace.session.videos if hasattr(self.workspace, 'session') else []):
                    if video.video_name == clip.source_file or clip.source_file in video.source_path:
                        clip.source_path = video.source_path
                        break

        # Save the EditDecision JSON for dashboard / debugging
        json_path = self.workspace.root / "fcpxml" / "edit_decision.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(edit.model_dump(), f, indent=2)

        # Compile to FCPXML
        fcpxml_path = str(self.workspace.get_fcpxml_path(version=1))
        try:
            output = compile_edit_decision(edit, fcpxml_path)
        except Exception as e:
            logger.error(f"[AGENT] FCPXML compilation failed: {e}", exc_info=True)
            return {
                "error": f"FCPXML compilation failed: {e}",
                "edit_decision_saved": str(json_path),
            }

        return {
            "status": "compiled",
            "fcpxml_path": output,
            "edit_decision_path": str(json_path),
            "clip_count": len(edit.clips),
            "narration_count": len(edit.narration),
            "has_music": edit.music is not None,
            "title_count": len(edit.titles),
        }

    def _message_user(self, args: Dict) -> Dict:
        message = args.get("message", "")
        # The event callback handles sending this to the frontend
        return {"delivered": True, "length": len(message)}
