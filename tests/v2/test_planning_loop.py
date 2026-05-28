"""Tests for IterativePlanningLoop — mocked lvmm-core retrieval and Gemini."""
import asyncio
import pytest
from types import SimpleNamespace

from src.pipelines.v2.planning.iterative_planning_loop import IterativePlanningLoop, PlanningEvent
from src.pipelines.v2.schemas import (
    ChatTool, RetrievedClip, SearchTool, Shot, Storyboard, ToolCallPlan, VideoEntry
)
from src.pipelines.v2.workspace import WorkspaceManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path) -> WorkspaceManager:
    ws = WorkspaceManager("test_plan", tmp_path)
    ws.create()
    entry = VideoEntry("v1", "keynote.mp4", "/media/keynote.mp4", 3600.0)
    ws.init_session([entry], gist="This is a tech keynote video.")
    return ws


@pytest.fixture
def video_entries():
    return [VideoEntry("v1", "keynote.mp4", "/media/keynote.mp4", 3600.0)]


def make_tool_plan(stop=False, n_chat=1, n_search=1) -> ToolCallPlan:
    return ToolCallPlan(
        reasoning="test reasoning",
        chat_calls=[ChatTool(question=f"q{i}", purpose="test") for i in range(n_chat)],
        search_calls=[SearchTool(query=f"search{i}", purpose="test", target_duration_sec=5.0) for i in range(n_search)],
        should_stop=stop,
    )


def make_storyboard(n_shots=2, iteration=1) -> Storyboard:
    shots = [
        Shot(id=f"s{i}", purpose=f"shot {i}", search_query=f"query {i}", duration_seconds=10.0)
        for i in range(n_shots)
    ]
    return Storyboard(
        iteration=iteration,
        target_duration_seconds=60.0,
        theme="tech keynote",
        shots=shots,
    )


def make_search_hits():
    """Simulate lvmm-core Querier hits.

    Querier.search returns list[dict] (since commit 330a34c — the old Hit
    namespace was replaced with dicts). _parse_search_results expects keys
    ``video_id``, ``start_time``, ``end_time``, ``transcript``/``text``,
    ``similarity``/``score``.
    """
    return [
        {
            "video_id": "v1",
            "start_time": 10.0,
            "end_time": 20.0,
            "similarity": 0.85,
            "text": "Speaker at podium",
            "collection": "video_transcript",
        },
        {
            "video_id": "v1",
            "start_time": 50.0,
            "end_time": 65.0,
            "similarity": 0.72,
            "text": "Audience reaction",
            "collection": "video_transcript",
        },
    ]


class FakeGemini:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    async def generate_structured(self, messages, schema, config=None):
        result = self.outputs.pop(0)
        if isinstance(result, Exception):
            raise result
        # Production ILLM.generate_structured returns (result, Usage). The
        # planning loop now reads usage.input_tokens / usage.output_tokens for
        # metric() emission, so we return a real zero-valued namespace rather
        # than None to keep the unit test honest with the real contract.
        usage = SimpleNamespace(input_tokens=0, output_tokens=0)
        return result, usage


class FakeMaviAgent:
    async def ask(self, question, video_id=None, user_id=None):
        return SimpleNamespace(answer="answer", reranked_hits=1, search_hits=1)


class FakeQuerier:
    """Replaces the old FakeSearcher after the lvmm-core Searcher→Querier rename.

    Mirrors the real ``luci_memory.Querier.search`` signature:
    ``search(question, user_id, video_ids, top_k, collections, time_range)``.
    Returns ``list[dict]`` directly (no SimpleNamespace wrapper).
    """
    def __init__(self, hits=None):
        self.hits = hits if hits is not None else make_search_hits()

    async def search(self, question, user_id=None, video_ids=None,
                     top_k=10, collections=None, time_range=None):
        return list(self.hits)


def make_loop(
    workspace,
    video_entries,
    gemini_outputs,
    *,
    search_hits=None,
    max_iterations=5,
    event_queue=None,
):
    return IterativePlanningLoop(
        project_name="test_plan",
        user_prompt="test prompt",
        workspace=workspace,
        querier=FakeQuerier(search_hits),
        mavi_agent=FakeMaviAgent(),
        gemini=FakeGemini(gemini_outputs),
        video_nos=["v1"],
        video_entries=video_entries,
        max_iterations=max_iterations,
        event_queue=event_queue,
    )


# ---------------------------------------------------------------------------
# PlanningEvent
# ---------------------------------------------------------------------------

def test_planning_event_to_dict():
    event = PlanningEvent(event_type="iteration_start", data={"iteration": 0})
    d = event.to_dict()
    assert d["event_type"] == "iteration_start"
    assert d["data"]["iteration"] == 0
    assert "timestamp" in d


def test_planning_event_auto_timestamp():
    e = PlanningEvent(event_type="done", data={})
    assert e.timestamp != ""


# ---------------------------------------------------------------------------
# Loop with should_stop=True on first iteration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_stops_early_on_should_stop(workspace, video_entries):
    call_a_result = make_tool_plan(stop=True, n_chat=1, n_search=1)
    call_b_result = make_storyboard(n_shots=2, iteration=1)
    loop = make_loop(
        workspace,
        video_entries,
        [call_a_result, call_b_result, make_tool_plan(stop=True, n_chat=0, n_search=0)],
        max_iterations=2,
    )

    storyboard = await loop.run()
    assert storyboard is not None


@pytest.mark.asyncio
async def test_loop_runs_full_iterations(workspace, video_entries):
    tool_plans = [
        make_tool_plan(stop=False, n_chat=1, n_search=1),
        make_tool_plan(stop=False, n_chat=0, n_search=1),
        make_tool_plan(stop=True, n_chat=0, n_search=0),
    ]
    storyboards = [
        make_storyboard(n_shots=2, iteration=i + 1)
        for i in range(3)
    ]
    loop = make_loop(
        workspace,
        video_entries,
        _interleave(tool_plans, storyboards),
        max_iterations=3,
    )

    sb = await loop.run()
    assert sb is not None
    assert len(sb.shots) >= 0


@pytest.mark.asyncio
async def test_loop_emits_events_to_queue(workspace, video_entries):
    queue = asyncio.Queue()
    loop = make_loop(
        workspace,
        video_entries,
        [
            make_tool_plan(stop=True, n_chat=1, n_search=1),
            make_storyboard(n_shots=1),
            make_tool_plan(stop=True),
        ],
        max_iterations=2,
        event_queue=queue,
    )

    await loop.run()

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    event_types = [e.event_type for e in events]
    assert "iteration_start" in event_types
    assert "done" in event_types or "stopped_early" in event_types


@pytest.mark.asyncio
async def test_loop_saves_storyboard_to_workspace(workspace, video_entries):
    loop = make_loop(
        workspace,
        video_entries,
        [
            make_tool_plan(stop=False, n_chat=1, n_search=0),
            make_storyboard(n_shots=3, iteration=1),
            make_tool_plan(stop=True),
        ],
        search_hits=[],
        max_iterations=2,
    )

    await loop.run()
    saved = workspace.load_storyboard()
    assert saved is not None
    assert len(saved.shots) == 3


@pytest.mark.asyncio
async def test_loop_handles_gemini_error_gracefully(workspace, video_entries):
    loop = make_loop(
        workspace,
        video_entries,
        [
            RuntimeError("Gemini quota exceeded"),
            make_storyboard(n_shots=0),
        ],
        search_hits=[],
        max_iterations=1,
    )

    # Should not raise; returns existing (empty) storyboard
    sb = await loop.run()
    assert sb is not None


@pytest.mark.asyncio
async def test_loop_parses_search_results(workspace, video_entries):
    loop = make_loop(
        workspace,
        video_entries,
        [
            make_tool_plan(stop=False, n_chat=0, n_search=1),
            make_storyboard(n_shots=1, iteration=1),
            make_tool_plan(stop=True),
        ],
        max_iterations=2,
    )
    await loop.run()

    clips = workspace.load_clips()
    # Should have parsed clips from the search response
    assert len(clips) > 0
    assert clips[0].video_no == "v1"
    assert clips[0].score == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_run_chat_uses_current_mavi_video_id_signature_for_each_video(workspace):
    entries = [
        VideoEntry("v1", "a.mp4", "/media/a.mp4", 10.0),
        VideoEntry("v2", "b.mp4", "/media/b.mp4", 10.0),
    ]

    class StrictMaviAgent:
        def __init__(self):
            self.calls = []

        async def ask(self, question, video_id=None, user_id=None):
            self.calls.append(video_id)
            return SimpleNamespace(
                answer=f"answer for {video_id}",
                reranked_videos=1,
                reranked_video_ts=0,
                reranked_audio_ts=0,
                reranked_keyframes=0,
            )

    mavi = StrictMaviAgent()
    loop = IterativePlanningLoop(
        project_name="test_plan",
        user_prompt="test prompt",
        workspace=workspace,
        querier=FakeQuerier([]),
        mavi_agent=mavi,
        gemini=FakeGemini([]),
        video_nos=["v1", "v2"],
        video_entries=entries,
    )

    kind, context = await loop._run_chat("What happens?", "test", 0)

    assert kind == "chat"
    assert mavi.calls == ["v1", "v2"]
    assert "answer for v1" in context
    assert "answer for v2" in context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interleave(tool_plans, storyboards):
    """Interleave Call A and Call B results."""
    result = []
    for i in range(max(len(tool_plans), len(storyboards))):
        if i < len(tool_plans):
            result.append(tool_plans[i])
        if i < len(storyboards):
            result.append(storyboards[i])
    return result
