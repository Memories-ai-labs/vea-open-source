"""Gemini function-calling declarations for the agentic editing session."""

from google.genai.types import FunctionDeclaration, Tool

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
                "relevance score, description, and dialogue transcript "
                "for that time range. Read the transcripts to verify "
                "whether a clip contains the moment you need before "
                "committing to refine it."
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
                            '"source_end": number, "label": str, "description": str, "gain_db": number, '
                            '"measured_loudness_lufs": number (read-only, set by renderer), '
                            '"speed": {"rate": number}, '
                            '"transition_after": {"type": "cross-dissolve"|"fade-in"|"fade-out", "duration_seconds": number}, '
                            '"track": number (default 1, use 2+ for overlay tracks), '
                            '"timeline_offset": number (absolute position in seconds, required for track 2+)}], '
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
            name="refine_clip_timestamps",
            description=(
                "Refine the in/out points of a clip to find precise cut points within a broader search range. "
                "Extracts and downsamples the video segment, transcribes audio via ElevenLabs STT, "
                "sends both to Gemini which watches the video and returns optimized start/end timestamps. "
                "Works for dialogue (finds sentence boundaries), visual b-roll (peak moments), and "
                "audio-driven clips (beats, applause). IMPORTANT: Only call this on clips you've "
                "committed to using — do NOT refine immediately after searching. First read the "
                "search transcripts, select your final clips, update the planning scratchpad, "
                "then refine the selected clips."
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
                "Generate narration voiceover audio from a script. Takes the narration script text "
                "and produces a single narration.mp3 file in the workspace. Call this ONLY after "
                "an edit plan exists (so you know clip durations). The user must have requested "
                "narration — do NOT call this unprompted. Returns the file path, duration, and a "
                "transcript with per-sentence timestamps ({text, start, end}). Use the transcript "
                "to align clips to the narration — adjust clip order and durations so visuals "
                "match what's being narrated at each moment."
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
                "Search for and download a background music track. Fetches candidate tracks from "
                "the music library, uses an LLM to pick the best match based on your prompt, and "
                "downloads it. Returns the file path, track name, and duration. Use this when the "
                "user wants background music in their edit."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Describe the ideal music — mood, energy, genre, instruments, tempo. "
                            "Be specific: 'upbeat electronic with synths, 120bpm, energetic but not aggressive' "
                            "is better than just 'upbeat'."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        ),
        FunctionDeclaration(
            name="verify_preview",
            description=(
                "Watch the latest rendered preview video and provide a professional critique. "
                "Sends the render to a vision model that analyzes pacing, transitions, audio mix, "
                "visual composition, and overall flow. Returns detailed feedback with specific "
                "timestamps and actionable suggestions. Use this after rendering to quality-check "
                "the edit before delivering to the user, or when the user asks you to review."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": (
                            "What to focus the review on — e.g. 'pacing and transitions', "
                            "'does the narration sync with visuals', 'overall quality check', "
                            "'are the clip selections compelling'. Leave empty for a general review."
                        ),
                    },
                },
                "required": [],
            },
        ),
        FunctionDeclaration(
            name="generate_subtitles",
            description=(
                "Generate subtitles for the current edit by transcribing the original audio "
                "from each clip. Uses ElevenLabs speech-to-text to get word-level timestamps, "
                "groups words into subtitle lines, and adds them to the edit decision as text "
                "overlays. Must be called AFTER clips are finalized in the edit decision. "
                "The user must explicitly request subtitles."
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
            name="generate_content",
            description=(
                "Generate a short AI video clip from a text prompt using Veo (Google's video "
                "generation model). Returns the file path to the generated MP4 which can then "
                "be used as a clip in the edit decision. Generation takes 1-5 minutes. "
                "The user MUST explicitly request AI-generated content — do NOT call this unprompted. "
                "Generated clips are saved to the workspace and can be referenced by source_file "
                "in the edit decision."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Detailed description of the video to generate. Be specific about: "
                            "visual content, camera movement, lighting, mood, action. "
                            "Example: 'A slow aerial drone shot over a misty mountain range at sunrise, "
                            "golden light breaking through clouds, cinematic 4K quality.'"
                        ),
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Video duration in seconds. Options: 4, 6, or 8. Default 8.",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "description": "Aspect ratio: '16:9' (landscape) or '9:16' (portrait). Default '16:9'.",
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "Short name for the generated clip (used as filename). "
                            "Example: 'mountain_sunrise'. Default 'generated'."
                        ),
                    },
                },
                "required": ["prompt"],
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
