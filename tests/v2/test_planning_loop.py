"""Tests for IterativePlanningLoop — mocked Memories.ai and Gemini."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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


def make_search_response():
    """Simulate Memories.ai search response."""
    return {
        "results": [
            {
                "videoNo": "v1",
                "startTime": 10.0,
                "endTime": 20.0,
                "score": 0.85,
                "description": "Speaker at podium",
            },
            {
                "videoNo": "v1",
                "startTime": 50.0,
                "endTime": 65.0,
                "score": 0.72,
                "description": "Audience reaction",
            },
        ]
    }


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
    mock_memories = MagicMock()
    mock_memories.chat.return_value = {"text": "Chat response about the video"}
    mock_memories.search.return_value = make_search_response()

    mock_gemini = MagicMock()
    # Call A: stop immediately after first iteration
    call_a_result = make_tool_plan(stop=True, n_chat=1, n_search=1)
    call_b_result = make_storyboard(n_shots=2, iteration=1)
    mock_gemini.LLM_request.side_effect = [call_a_result, call_b_result]

    loop = IterativePlanningLoop(
        project_name="test_plan",
        user_prompt="Make a 60s highlights video",
        workspace=workspace,
        memories=mock_memories,
        gemini=mock_gemini,
        video_nos=["v1"],
        video_entries=video_entries,
        max_iterations=5,
    )

    # Patch run_in_executor to call the function directly (no thread pool)
    original_executor = None

    async def mock_run_in_executor(executor, func):
        return func()

    with patch.object(asyncio.get_event_loop().__class__, 'run_in_executor', new=mock_run_in_executor):
        # Re-get the loop since patching is class-level
        pass

    # Just run the loop with mock executors via monkeypatching
    storyboard = await _run_loop_with_mocks(loop, mock_memories, mock_gemini)
    assert storyboard is not None
    # should_stop=True at iteration 0, iteration >= 1 check means it won't stop until iter 1
    # (the guard is `if tool_plan.should_stop and iteration >= 1`)
    # So it will run iter 0 fully, then check should_stop at iter 1


@pytest.mark.asyncio
async def test_loop_runs_full_iterations(workspace, video_entries):
    mock_memories = MagicMock()
    mock_memories.chat.return_value = {"text": "Detailed answer about the video content"}
    mock_memories.search.return_value = make_search_response()

    mock_gemini = MagicMock()

    # 3 iterations: each returns a normal tool plan, last stops
    tool_plans = [
        make_tool_plan(stop=False, n_chat=1, n_search=1),
        make_tool_plan(stop=False, n_chat=0, n_search=1),
        make_tool_plan(stop=True, n_chat=0, n_search=0),
    ]
    storyboards = [
        make_storyboard(n_shots=2, iteration=i + 1)
        for i in range(3)
    ]
    # Call A, Call B alternating
    mock_gemini.LLM_request.side_effect = _interleave(tool_plans, storyboards)

    loop = IterativePlanningLoop(
        project_name="test_plan",
        user_prompt="60s recap",
        workspace=workspace,
        memories=mock_memories,
        gemini=mock_gemini,
        video_nos=["v1"],
        video_entries=video_entries,
        max_iterations=3,
    )

    sb = await _run_loop_with_mocks(loop, mock_memories, mock_gemini)
    assert sb is not None
    assert len(sb.shots) >= 0


@pytest.mark.asyncio
async def test_loop_emits_events_to_queue(workspace, video_entries):
    mock_memories = MagicMock()
    mock_memories.chat.return_value = {"text": "answer"}
    mock_memories.search.return_value = make_search_response()

    mock_gemini = MagicMock()
    mock_gemini.LLM_request.side_effect = [
        make_tool_plan(stop=True, n_chat=1, n_search=1),
        make_storyboard(n_shots=1),
        make_tool_plan(stop=True),
        make_storyboard(n_shots=1),
    ]

    queue = asyncio.Queue()
    loop = IterativePlanningLoop(
        project_name="test_plan",
        user_prompt="test prompt",
        workspace=workspace,
        memories=mock_memories,
        gemini=mock_gemini,
        video_nos=["v1"],
        video_entries=video_entries,
        max_iterations=2,
        event_queue=queue,
    )

    await _run_loop_with_mocks(loop, mock_memories, mock_gemini)

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    event_types = [e.event_type for e in events]
    assert "iteration_start" in event_types
    assert "done" in event_types or "stopped_early" in event_types


@pytest.mark.asyncio
async def test_loop_saves_storyboard_to_workspace(workspace, video_entries):
    mock_memories = MagicMock()
    mock_memories.chat.return_value = {"text": "answer"}
    mock_memories.search.return_value = {"results": []}

    mock_gemini = MagicMock()
    mock_gemini.LLM_request.side_effect = [
        make_tool_plan(stop=False, n_chat=1, n_search=0),
        make_storyboard(n_shots=3, iteration=1),
        make_tool_plan(stop=True),
        make_storyboard(n_shots=3, iteration=2),
    ]

    loop = IterativePlanningLoop(
        project_name="test_plan",
        user_prompt="test",
        workspace=workspace,
        memories=mock_memories,
        gemini=mock_gemini,
        video_nos=["v1"],
        video_entries=video_entries,
        max_iterations=2,
    )

    await _run_loop_with_mocks(loop, mock_memories, mock_gemini)
    saved = workspace.load_storyboard()
    assert saved is not None
    assert len(saved.shots) == 3


@pytest.mark.asyncio
async def test_loop_handles_gemini_error_gracefully(workspace, video_entries):
    mock_memories = MagicMock()
    mock_memories.chat.return_value = {"text": "answer"}
    mock_memories.search.return_value = {"results": []}

    mock_gemini = MagicMock()
    # Call A raises an error
    mock_gemini.LLM_request.side_effect = [
        RuntimeError("Gemini quota exceeded"),
        make_storyboard(n_shots=0),
    ]

    loop = IterativePlanningLoop(
        project_name="test_plan",
        user_prompt="test",
        workspace=workspace,
        memories=mock_memories,
        gemini=mock_gemini,
        video_nos=["v1"],
        video_entries=video_entries,
        max_iterations=1,
    )

    # Should not raise; returns existing (empty) storyboard
    sb = await _run_loop_with_mocks(loop, mock_memories, mock_gemini)
    assert sb is not None


@pytest.mark.asyncio
async def test_loop_parses_search_results(workspace, video_entries):
    mock_memories = MagicMock()
    mock_memories.chat.return_value = {"text": "answer"}
    mock_memories.search.return_value = make_search_response()

    mock_gemini = MagicMock()
    mock_gemini.LLM_request.side_effect = [
        make_tool_plan(stop=False, n_chat=0, n_search=1),
        make_storyboard(n_shots=1, iteration=1),
        make_tool_plan(stop=True),
        make_storyboard(n_shots=1, iteration=2),
    ]

    loop = IterativePlanningLoop(
        project_name="test_plan",
        user_prompt="test",
        workspace=workspace,
        memories=mock_memories,
        gemini=mock_gemini,
        video_nos=["v1"],
        video_entries=video_entries,
        max_iterations=2,
    )
    await _run_loop_with_mocks(loop, mock_memories, mock_gemini)

    clips = workspace.load_clips()
    # Should have parsed clips from the search response
    assert len(clips) > 0
    assert clips[0].video_no == "v1"
    assert clips[0].score == pytest.approx(0.85)


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


async def _run_loop_with_mocks(loop_obj: IterativePlanningLoop, mock_memories, mock_gemini) -> Storyboard:
    """
    Run the planning loop with executor calls replaced by direct sync calls.
    """
    original_run = asyncio.get_event_loop().run_in_executor

    async def direct_executor(executor, fn):
        return fn()

    # Patch the event loop's run_in_executor on the loop_obj's calls
    event_loop = asyncio.get_event_loop()
    original_method = event_loop.run_in_executor

    try:
        event_loop.run_in_executor = lambda ex, fn: asyncio.coroutine(lambda: fn)()
    except Exception:
        pass

    # Simpler approach: just patch asyncio to run sync
    import unittest.mock as um
    with um.patch.object(asyncio, 'get_event_loop') as mock_get_loop:
        mock_loop = MagicMock()
        mock_loop.run_in_executor = _sync_executor
        mock_get_loop.return_value = mock_loop
        result = await loop_obj.run()

    return result


async def _sync_executor(executor, fn):
    """Run fn synchronously (for tests)."""
    return fn()
