"""Regression tests for the lvmm-core-backed agent tools."""
from types import SimpleNamespace

from src.pipelines.v2.agent.tools import ToolExecutor


class _RecordingMaviAgent:
    def __init__(self):
        self.calls = []

    async def ask(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return SimpleNamespace(answer="answer", reranked_hits=3, search_hits=["hit"])


class _FakeDatabase:
    def __init__(self, rows_by_table):
        self.rows_by_table = rows_by_table
        self.calls = []

    async def query(self, table, filters):
        self.calls.append((table, filters))
        key = (table, filters.get("video_id"))
        return list(self.rows_by_table.get(key, self.rows_by_table.get(table, [])))


class _FakeSearcher:
    def __init__(self, hits):
        self.hits = hits
        self.requests = []

    async def search(self, request):
        self.requests.append(request)
        return SimpleNamespace(hits=self.hits)


def _executor(tmp_path, *, video_nos, mavi_agent=None, database=None):
    return ToolExecutor(
        searcher=None,
        mavi_agent=mavi_agent or _RecordingMaviAgent(),
        lvmm_ctx=SimpleNamespace(database=database or _FakeDatabase({})),
        gemini_manager=None,
        workspace=SimpleNamespace(root=tmp_path),
        scratchpads=None,
        video_nos=video_nos,
    )


async def test_ask_memories_scopes_mavi_to_all_workspace_videos(tmp_path):
    mavi_agent = _RecordingMaviAgent()
    executor = _executor(tmp_path, video_nos=["clip_a", "clip_b"], mavi_agent=mavi_agent)

    result = await executor._ask_memories({"question": "What happens?"})

    assert result["answer"] == "answer"
    assert mavi_agent.calls == [
        ("What happens?", {"video_ids": ["clip_a", "clip_b"]})
    ]


async def test_transcript_segments_fall_back_to_video_transcripts_text(tmp_path):
    database = _FakeDatabase({
        "audio_transcripts": [],
        "video_transcripts": [
            {"text": "visual caption", "start_time": 1.25, "end_time": 3.5},
        ],
    })
    executor = _executor(tmp_path, video_nos=["clip_a"], database=database)

    segments = await executor._get_transcript_segments()

    assert database.calls == [
        ("audio_transcripts", {"video_id": "clip_a"}),
        ("video_transcripts", {"video_id": "clip_a"}),
    ]
    assert segments == [
        {"video_no": "clip_a", "text": "visual caption", "start": 1.25, "end": 3.5}
    ]


async def test_search_footage_uses_transcripts_from_matching_video(tmp_path):
    searcher = _FakeSearcher([
        SimpleNamespace(
            video_id="clip_b",
            start_time=10.0,
            end_time=11.0,
            score=0.9,
            text="match",
        )
    ])
    database = _FakeDatabase({
        ("audio_transcripts", "clip_a"): [],
        ("video_transcripts", "clip_a"): [
            {"text": "wrong video", "start_time": 10.0, "end_time": 14.0},
        ],
        ("audio_transcripts", "clip_b"): [],
        ("video_transcripts", "clip_b"): [
            {"text": "right video", "start_time": 10.0, "end_time": 14.0},
        ],
    })
    executor = _executor(tmp_path, video_nos=["clip_a", "clip_b"], database=database)
    executor.searcher = searcher

    result = await executor._search_footage({
        "query": "show the matching moment",
        "target_duration_seconds": 2.0,
    })

    assert result["clips"][0]["video_no"] == "clip_b"
    assert result["clips"][0]["end_seconds"] == 14.3
    assert result["clips"][0]["transcript"] == [
        {"text": "right video", "start": 10.0, "end": 14.0}
    ]
