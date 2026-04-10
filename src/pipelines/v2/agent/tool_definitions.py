"""Gemini function-calling declarations for the agentic editing session."""

from google.genai.types import FunctionDeclaration, Tool

# Schema string for generate_fcpxml — kept as a module-level constant so it's easy
# to update when new fields are added to the schemas in src/pipelines/v2/schemas.py.
_EDIT_DECISION_SCHEMA = (
    "A JSON string representing the full edit decision. Schema:\n"
    "{\n"
    '  "timeline": {"name": str, "fps": number, "width": int, "height": int},\n'
    '  "clips": [{  // REQUIRED: at least one clip; each clip requires id, source_file, source_start, source_end\n'
    '    "id": str,\n'
    '    "source_file": str,           // filename in footage/ directory\n'
    '    "source_start": number,       // in-point in seconds (must be within file duration)\n'
    '    "source_end": number,         // out-point in seconds (must be within file duration)\n'
    '    "label": str,\n'
    '    "description": str,\n'
    '    "gain_db": number,            // literal dB adjustment for source clips (see system prompt)\n'
    '    "measured_loudness_lufs": number,  // READ-ONLY: set by the renderer after measurement\n'
    '    "speed": {"rate": number},\n'
    '    "transform": {"scale_x": number, "scale_y": number, "position_x": number, "position_y": number, "rotation": number},\n'
    '    "transform_mode": "fit" | "custom" | "saliency",\n'
    '    "shot_transforms": [ShotCropResult, ...],  // set by Dynamic Crop tool\n'
    '    "source_width": int,\n'
    '    "source_height": int,\n'
    '    "transition_after": {"type": "cross-dissolve" | "fade-in" | "fade-out", "duration_seconds": number},\n'
    '    "track": int,                 // 1 = V1 spine (default, sequential), 2+ = overlay tracks\n'
    '    "timeline_offset": number,    // absolute timeline position in seconds (REQUIRED for track 2+)\n'
    '  }],\n'
    '  "narration": [{\n'
    '    "file": str,                  // usually "narration.mp3"\n'
    '    "timeline_offset": number,    // where on the timeline the segment plays\n'
    '    "start": number,              // in-point in the audio file (MUST be a word start from the words array)\n'
    '    "duration": number,           // (start + duration) MUST equal a word end from the words array\n'
    '    "gain_db": number,            // offset from -16 LUFS target (default 0)\n'
    '    "measured_loudness_lufs": number,  // READ-ONLY\n'
    '  }],\n'
    '  "music": {\n'
    '    "file": str,                  // usually "track.mp3"\n'
    '    "start": number,              // in-point in the music file\n'
    '    "duration": number,           // 0 = use full timeline length\n'
    '    "gain_db": number,            // offset from -18 LUFS target (default 0 — do NOT write -18!)\n'
    '    "measured_loudness_lufs": number,  // READ-ONLY\n'
    '  },\n'
    '  "titles": [{\n'
    '    "text": str,\n'
    '    "timeline_offset": number,\n'
    '    "duration": number,\n'
    '    "font_size": int,\n'
    '    "lane": int,                  // positive = above spine (default 1)\n'
    '    "style": "title" | "subtitle",  // title = centered graphic, subtitle = bottom caption\n'
    '    "position": "center" | "bottom" | "top",\n'
    '  }]\n'
    "}\n"
    "Only `clips` is required at the top level, and each clip requires id/source_file/"
    "source_start/source_end. Timeline defaults to 24fps 1920x1080. See the system prompt "
    "for audio levels, narration splits, and source-timestamp rules."
)


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
                "Search for specific video clips matching a query. Returns clips with "
                "video_name, start/end timestamps (within each file's actual duration), "
                "relevance score, description, and dialogue transcript for that time range. "
                "Read the transcripts to verify whether a clip contains the moment you "
                "need before committing to refine it."
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
                "Operations: replace (overwrite), append (add to end), prepend (add to start). "
                "See the system prompt 'Your scratchpads' section for formatting guidance."
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
                        "description": (
                            "The text to write. Prefer bullet-point format (`- item`) with "
                            "`**bold**` emphasis and `## Heading` sections. Short prose is OK "
                            "for nuanced explanations."
                        ),
                    },
                },
                "required": ["name", "operation", "content"],
            },
        ),
        FunctionDeclaration(
            name="generate_fcpxml",
            description=(
                "Compile a complete edit decision JSON into a Final Cut Pro XML timeline. "
                "Call this whenever the plan is ready or you want to update an existing edit. "
                "The system compiles deterministically to valid FCPXML 1.10, validates clip "
                "source ranges against actual file durations, runs beat sync if music is "
                "present, and auto-renders both a 480p draft and a native-resolution final."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "edit_decision_json": {
                        "type": "string",
                        "description": _EDIT_DECISION_SCHEMA,
                    },
                },
                "required": ["edit_decision_json"],
            },
        ),
        FunctionDeclaration(
            name="refine_clip_timestamps",
            description=(
                "Refine the in/out points of a clip to find precise cut points within a broader "
                "search range. Extracts and downsamples the video segment, transcribes audio via "
                "ElevenLabs STT, and sends both to Gemini which watches the video and returns "
                "optimized start/end timestamps. Works for dialogue (finds sentence boundaries), "
                "visual b-roll (peak moments), and audio-driven clips (beats, applause). "
                "Only call this on clips you've committed to using."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source_file": {
                        "type": "string",
                        "description": "Filename of the source video (must be in the footage directory).",
                    },
                    "source_start": {
                        "type": "number",
                        "description": "Current in-point in seconds (start of the broad segment).",
                    },
                    "source_end": {
                        "type": "number",
                        "description": "Current out-point in seconds (end of the broad segment).",
                    },
                    "target_duration": {
                        "type": "number",
                        "description": "Desired clip duration in seconds. The refined clip should be close to this length.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "What makes a good clip here — e.g. 'Find the moment where the speaker announces the product' "
                            "or 'Best visual of the crowd reacting' or 'The clearest explanation of the feature'."
                        ),
                    },
                    "clip_description": {
                        "type": "string",
                        "description": "Brief description of what this clip is supposed to show in the edit.",
                    },
                },
                "required": ["source_file", "source_start", "source_end", "target_duration", "prompt"],
            },
        ),
        FunctionDeclaration(
            name="generate_narration",
            description=(
                "Generate narration voiceover audio from a script. Produces narration.mp3 in the "
                "workspace. Returns: file path, duration, a per-sentence `transcript` array, and "
                "a per-word `words` array — both with REAL timestamps from ElevenLabs character "
                "alignment. When splitting narration around dialogue clips, segment start/end "
                "MUST equal word boundaries from the `words` array or speech will be cut mid-word. "
                "Only call this when the user has explicitly requested narration."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": (
                            "The full narration script to convert to speech. Write it as natural "
                            "spoken text — no stage directions, no shot labels. Pace at ~140 words/minute. "
                            "Use '...' for pauses between sections."
                        ),
                    },
                },
                "required": ["script"],
            },
        ),
        FunctionDeclaration(
            name="select_music",
            description=(
                "Generate a background music track using AI (Google Lyria 3). Produces an "
                "original instrumental track matching your description. Returns the file "
                "path and duration. The prompt should describe mood, energy, genre, "
                "instruments, and tempo — be specific. Generates up to ~3 minutes. "
                "Only call this when the user has explicitly requested music."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Describe the ideal music — mood, energy, genre, instruments, tempo. "
                            "Be specific: 'upbeat electronic with synths, 120bpm, energetic but not aggressive' "
                            "is better than just 'upbeat'. Do NOT name artists or songs."
                        ),
                    },
                    "duration_seconds": {
                        "type": "number",
                        "description": (
                            "Desired track length in seconds. Default 120. Max 300 (5 minutes). "
                            "Should roughly match the timeline duration."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        ),
        FunctionDeclaration(
            name="generate_subtitles",
            description=(
                "Generate subtitles for the current edit by transcribing the original audio "
                "from each clip. Uses ElevenLabs speech-to-text to get word-level timestamps, "
                "groups words into subtitle lines, and adds them to the edit decision as text "
                "overlays. Must be called AFTER clips are finalized in the edit decision. "
                "Only call this when the user has explicitly requested subtitles."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "max_words_per_line": {
                        "type": "integer",
                        "description": "Maximum words per subtitle line. Default 8.",
                    },
                    "font_size": {
                        "type": "integer",
                        "description": "Font size for subtitle text. Default 48.",
                    },
                },
                "required": [],
            },
        ),
        FunctionDeclaration(
            name="message_user",
            description=(
                "Send a visible message to the user in the chat interface. Use this to share "
                "findings, propose plans, ask questions, or report progress DURING work. "
                "This does NOT end your turn — you can keep calling tools after. "
                "When you're completely done with the user's request, call finish_turn instead."
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
        FunctionDeclaration(
            name="finish_turn",
            description=(
                "Signal that you've finished the work for the user's current request and are "
                "ready to wait for the next message. This explicitly ENDS YOUR TURN. Call at the "
                "end of your work, after all tools have been called, audio issues in the timeline "
                "view are resolved, and you've delivered the result. Pass an optional "
                "`final_message` to summarize what you did. Do NOT call this if work is pending."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "final_message": {
                        "type": "string",
                        "description": (
                            "Optional final summary message to show the user. If provided, this "
                            "is sent as a regular agent message before the turn ends."
                        ),
                    },
                },
                "required": [],
            },
        ),
    ]
)
