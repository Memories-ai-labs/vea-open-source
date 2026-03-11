"""Tests for WorkspaceManager — file I/O and session lifecycle."""
import json
import pytest
from pathlib import Path

from src.pipelines.v2.workspace import WorkspaceManager
from src.pipelines.v2.schemas import (
    SessionData, VideoEntry, PlanningState, Storyboard, Shot, RetrievedClip, ToolCallPlan, ChatTool, SearchTool
)


@pytest.fixture
def workspace(tmp_path) -> WorkspaceManager:
    ws = WorkspaceManager("test_project", tmp_path)
    ws.create()
    return ws


@pytest.fixture
def video_entry() -> VideoEntry:
    return VideoEntry(
        video_no="vno123",
        video_name="keynote.mp4",
        source_path="/media/keynote.mp4",
        duration_seconds=3600.0,
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_create_makes_directories(tmp_path):
    ws = WorkspaceManager("myproject", tmp_path)
    ws.create()
    for subdir in ["footage", "iterations", "narration", "music", "fcpxml", "renders", "logs"]:
        assert (tmp_path / "myproject" / subdir).is_dir()


def test_exists_false_before_create(tmp_path):
    ws = WorkspaceManager("new_project", tmp_path)
    assert not ws.exists()


def test_exists_true_after_init_session(workspace, video_entry):
    workspace.init_session([video_entry], gist="Test gist")
    assert workspace.exists()


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def test_init_and_load_session(workspace, video_entry):
    session = workspace.init_session([video_entry], gist="My gist")
    assert session.project_name == "test_project"
    assert session.gist == "My gist"
    assert session.status == "indexed"
    assert len(session.videos) == 1
    assert session.videos[0].video_no == "vno123"


def test_save_and_load_session(workspace, video_entry):
    session = workspace.init_session([video_entry])
    session.gist = "Updated gist"
    workspace.save_session(session)

    loaded = workspace.load_session()
    assert loaded.gist == "Updated gist"


def test_update_status(workspace, video_entry):
    workspace.init_session([video_entry])
    workspace.update_status("planning")
    assert workspace.load_session().status == "planning"


def test_session_roundtrip_preserves_videos(workspace):
    entries = [
        VideoEntry("v1", "a.mp4", "/a.mp4", 100.0),
        VideoEntry("v2", "b.mp4", "/b.mp4", 200.0),
    ]
    workspace.init_session(entries, gist="multi video")
    loaded = workspace.load_session()
    assert len(loaded.videos) == 2
    assert loaded.videos[0].video_no == "v1"
    assert loaded.videos[1].duration_seconds == 200.0


# ---------------------------------------------------------------------------
# Storyboard
# ---------------------------------------------------------------------------

def test_save_and_load_storyboard(workspace):
    clip = RetrievedClip(
        video_no="v1", video_name="a.mp4", source_path="/a.mp4",
        start_seconds=10.0, end_seconds=20.0, score=0.9,
    )
    shot = Shot(id="s1", purpose="intro", search_query="opening shot", retrieved_clip=clip)
    sb = Storyboard(iteration=2, theme="tech", shots=[shot])
    workspace.save_storyboard(sb)

    loaded = workspace.load_storyboard()
    assert loaded is not None
    assert loaded.iteration == 2
    assert loaded.theme == "tech"
    assert len(loaded.shots) == 1
    assert loaded.shots[0].id == "s1"
    assert loaded.shots[0].retrieved_clip.score == 0.9


def test_load_storyboard_returns_none_when_missing(workspace):
    assert workspace.load_storyboard() is None


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------

def test_save_and_load_clips(workspace):
    clips = [
        RetrievedClip(video_no="v1", video_name="a.mp4", source_path="/a.mp4",
                      start_seconds=0, end_seconds=10, score=0.8, shot_query="q"),
        RetrievedClip(video_no="v1", video_name="a.mp4", source_path="/a.mp4",
                      start_seconds=20, end_seconds=30, score=0.6, shot_query="q2"),
    ]
    workspace.save_clips(clips)
    loaded = workspace.load_clips()
    assert len(loaded) == 2
    assert loaded[0].score == 0.8
    assert loaded[1].start_seconds == 20


def test_load_clips_empty_when_missing(workspace):
    assert workspace.load_clips() == []


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

def test_append_and_load_context(workspace):
    workspace.append_context("First block\n")
    workspace.append_context("Second block\n")
    ctx = workspace.load_context()
    assert "First block" in ctx
    assert "Second block" in ctx


def test_load_context_empty_when_missing(workspace):
    assert workspace.load_context() == ""


# ---------------------------------------------------------------------------
# Iteration snapshots
# ---------------------------------------------------------------------------

def test_save_iteration_snapshot(workspace):
    plan = ToolCallPlan(
        reasoning="test",
        chat_calls=[ChatTool(question="What?", purpose="understand")],
        search_calls=[SearchTool(query="speaker at podium", purpose="get clip")],
    )
    sb = Storyboard(iteration=1, shots=[])
    workspace.save_iteration_snapshot(1, plan, sb)

    idir = workspace.root / "iterations"
    assert (idir / "iter_1_tool_plan.json").exists()
    assert (idir / "iter_1_storyboard.json").exists()

    plan_data = json.loads((idir / "iter_1_tool_plan.json").read_text())
    assert plan_data["reasoning"] == "test"
    assert len(plan_data["chat_calls"]) == 1


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

def test_fcpxml_path(workspace):
    p = workspace.get_fcpxml_path(1)
    assert p.name == "edit_v1.fcpxml"
    assert p.parent.name == "fcpxml"


def test_final_fcpxml_path(workspace):
    p = workspace.get_final_fcpxml_path()
    assert p.name == "edit_final.fcpxml"


def test_render_path(workspace):
    p = workspace.get_render_path("preview")
    assert p.name == "preview.mp4"
    assert p.parent.name == "renders"


def test_narration_path(workspace):
    p = workspace.get_narration_path()
    assert p.name == "narration.mp3"


def test_music_path(workspace):
    p = workspace.get_music_path()
    assert p.name == "track.mp3"
