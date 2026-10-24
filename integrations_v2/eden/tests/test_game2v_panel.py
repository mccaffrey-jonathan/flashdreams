# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Game2V web panel and mid-session game switch."""

from __future__ import annotations

import asyncio
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from cam2v import Cam2VConditioning
from PIL import Image

from flashdreams.runtime_v2.serving.extensions import (
    serve_extension_manifest,
    unregister_http_extension,
)
from flashdreams.runtime_v2.session_desc import SessionDesc

from eden.apps.game2v.game_timeline import GameTimeline
from eden.apps.game2v.panel import handle_game2v, register_game2v_panel
from eden.apps.game2v.session import (
    Game2VModelState,
    Game2VSessionConfig,
    _apply_pending_game_switch,
    _enter_start_gate,
    _publish_browser_metrics,
    _switch_game,
    browser_metrics_snapshot,
    request_game_switch,
    request_home,
)

pytestmark = pytest.mark.ci_cpu


def _conditioning(prompt: str, cover: Path) -> Cam2VConditioning:
    return Cam2VConditioning(
        prompt=prompt,
        first_frame_path=cover,
        base_intrinsics=torch.tensor([16.0, 16.0, 32.0, 32.0]),
        world_scale=1.0,
    )


def _config(slug: str, cover: Path) -> Game2VSessionConfig:
    return Game2VSessionConfig(
        conditioning=_conditioning("old", cover),
        total_blocks=8,
        device=torch.device("cpu"),
        model_video_width=64,
        model_video_height=64,
        first_frame_dtype=torch.float32,
        first_frame_interpolation="linear",
        warmup_blocks=0,
        game_slug=slug,
    )


def _state(slug: str = "tracer", *, cover: Path | None = None) -> Game2VModelState:
    cover = cover or Path("/tmp/old.png")
    state = Game2VModelState.__new__(Game2VModelState)
    state.config = _config(slug, cover)
    state.session_desc = SessionDesc(video_width=64, video_height=64, frames_per_second_for_step=16)
    state.pending_game_slug = None
    state.pending_game_restart = False
    state.pending_home = False
    state.start_requested = False
    state.started = False
    state.game_timeline = GameTimeline(slug)
    state.game_capture = None
    state.pending_game_chunk = None
    state.pending_game_frames = []
    state.dialog_held = False
    state.cache = object()
    state.anchor_frame = object()
    state.prompt_events = [("x", "t", "line")]
    state.static_prompt = "old scene"
    state.active_prompt = "old scene"
    state.active_event_lines = ("line",)
    state.encoded_prompt = "old scene"
    state.ar_index = 4
    state.blocks_generated = 4
    state.frames_generated = 48
    state.game_round_id = 3
    state.last_game_chunk = object()
    state.last_generate_wall_s = 0.4
    state.control_stream = SimpleNamespace(reset=lambda: None)
    state.postprocess_stream = None
    state.event_buffer = None
    state.input_timeline = SimpleNamespace(reset=lambda start_s=0.0: None)
    state.keyboard_track = SimpleNamespace(reset=lambda: None)
    state.pose_integrator = SimpleNamespace(reset=lambda: None)
    state.steady_started_at = 1.0
    state.steady_frames_generated = 12
    state.steady_held_s = 0.0
    state._recent_model_frame_rate_tracker = SimpleNamespace(reset=lambda: None)
    state.comparison_pending_generated_frames = None
    return state


@pytest.fixture
def isolated_extension():
    unregister_http_extension("game2v")
    yield
    unregister_http_extension("game2v")


def _app():
    app = web.Application()
    app.router.add_get("/extensions.json", serve_extension_manifest)
    app.router.add_route(
        "*",
        "/ext/{name}/{tail:.*}",
        lambda request: handle_game2v(request, request.match_info["tail"]),
    )
    return app


async def _with_client(coro):
    async with TestClient(TestServer(_app())) as client:
        await coro(client)


def test_extensions_manifest_lists_panel_script(isolated_extension):
    session = SimpleNamespace(model_loop=SimpleNamespace(state=_state()))
    register_game2v_panel(session)

    async def body(client):
        response = await client.get("/extensions.json")
        assert response.status == 200
        body = await response.json()
        assert body == {"scripts": ["/ext/game2v/panel.js"]}

    asyncio.run(_with_client(body))


def test_panel_lists_registered_games_and_rewrites_play_page(isolated_extension):
    session = SimpleNamespace(model_loop=SimpleNamespace(state=_state("ember-den")))
    register_game2v_panel(session)

    async def body(client):
        listed = await client.get("/ext/game2v/games")
        payload = await listed.json()
        assert payload["current"] == "ember-den"
        slugs = {game["slug"] for game in payload["games"]}
        assert "ember-den" in slugs
        assert "jade-duel" in slugs
        page = await client.get("/ext/game2v/play/ember-den")
        html = await page.text()
        assert page.status == 200
        # Standalone games own their page: it loads the game's own entry module.
        assert "/ember-den/main.js" in html
        assert "/shared/host/play-page.js" not in html
        subway = next(game for game in payload["games"] if game["slug"] == "ember-den")
        assert subway["cover"] == "files/ember-den/cover.png"
        assert slugs == {"ember-den", "iron-colossus", "jade-duel", "warehouse-sokoban"}
        assert "/game/game/" not in html
        assert '<base href="/game/ember-den/">' in html
        assert "/shared/recorder/" not in html
        script = await client.get("/ext/game2v/panel.js")
        assert script.status == 200
        assert "text/javascript" in script.content_type
        host = await client.get("/ext/game2v/host/play-page.js")
        assert host.status == 200
        assert "createRuntime" in await host.text()

    asyncio.run(_with_client(body))


def test_panel_switch_sets_pending_slug(isolated_extension):
    state = _state("ember-den")
    session = SimpleNamespace(model_loop=SimpleNamespace(state=state))
    register_game2v_panel(session)

    async def body(client):
        bad = await client.post("/ext/game2v/switch", json={"slug": "not-a-game"})
        assert bad.status == 400
        ok = await client.post("/ext/game2v/switch", json={"slug": "jade-duel"})
        assert ok.status == 200
        payload = await ok.json()
        assert payload["current"] == "jade-duel"
        assert state.pending_game_slug == "jade-duel"
        same = await client.post("/ext/game2v/switch", json={"slug": "ember-den"})
        assert same.status == 200
        assert state.pending_game_slug is None
        # "Play again" rewinds even though the slug has not changed.
        again = await client.post("/ext/game2v/switch", json={"slug": "ember-den", "restart": True})
        assert again.status == 200
        assert state.pending_game_slug == "ember-den"
        assert state.pending_game_restart is True

    asyncio.run(_with_client(body))


def test_panel_home_rearms_the_gate(isolated_extension):
    state = _state("tracer")
    state.config = replace(state.config, wait_for_start=True)
    state.start_requested = True
    state.started = True
    session = SimpleNamespace(model_loop=SimpleNamespace(state=state))
    register_game2v_panel(session)

    async def body(client):
        home = await client.post("/ext/game2v/home", json={})
        assert home.status == 200
        assert await home.json() == {"started": False}
        assert state.pending_home is True
        # A reload right after going home belongs on the home screen, so the
        # listing reports "not started" without waiting for the model thread.
        listed = await client.get("/ext/game2v/games")
        assert (await listed.json())["started"] is False

    asyncio.run(_with_client(body))


def test_panel_metrics_reports_completed_chunk_and_clears_on_switch(isolated_extension):
    state = _state("tracer")
    state.browser_metrics_lock = threading.Lock()
    state.browser_metrics_epoch = 0
    state.browser_metrics = None
    session = SimpleNamespace(model_loop=SimpleNamespace(state=state))
    register_game2v_panel(session)

    async def body(client):
        empty = await client.get("/ext/game2v/metrics")
        assert await empty.json() == {"model": None}

        _publish_browser_metrics(state, {"chunk_fps": 12.5, "model_step_wall_s": 2.0}, epoch=0)
        snapshot = browser_metrics_snapshot(state)
        assert snapshot["slug"] == "tracer"
        snapshot["chunk_fps"] = 0  # A caller cannot mutate the published result.
        response = await client.get("/ext/game2v/metrics")
        model = (await response.json())["model"]
        assert model["chunk_fps"] == 12.5
        assert model["model_step_ms"] == 2000.0
        assert model["age_s"] >= 0

        request_game_switch(state, "freeze")
        _publish_browser_metrics(state, {"chunk_fps": 9.0, "model_step_wall_s": 1.0}, epoch=0)
        cleared = await client.get("/ext/game2v/metrics")
        assert await cleared.json() == {"model": None}

    asyncio.run(_with_client(body))


def test_switch_game_rewinds_rollout_and_changes_slug(tmp_path, monkeypatch):
    cover = tmp_path / "cover.png"
    Image.new("RGB", (32, 32), (10, 20, 30)).save(cover)
    # Avoid importing game14b (DINOv3 / CUDA) just to resolve a cover+prompt.
    fake = SimpleNamespace(
        resolve_game14b_conditioning=lambda values: _conditioning(
            f"scene for {values['game']}", cover
        )
    )
    monkeypatch.setitem(sys.modules, "eden.apps.game2v.game14b", fake)
    monkeypatch.setattr("eden.apps.game2v.session.is_distributed_initialized", lambda: False)
    state = _state("tracer", cover=cover)
    closed = []
    state.game_capture = SimpleNamespace(close=lambda: closed.append(True))
    request_game_switch(state, "freeze")
    assert state.pending_game_slug == "freeze"
    _apply_pending_game_switch(state)
    assert state.config.game_slug == "freeze"
    assert state.config.conditioning.prompt == "scene for freeze"
    assert state.config.conditioning.first_frame_path == cover
    assert state.game_timeline.game_slug == "freeze"
    assert state.cache is None
    assert state.anchor_frame is None
    assert state.ar_index == 0
    assert state.blocks_generated == 0
    assert state.pending_game_slug is None
    assert closed == [True]
    assert state.static_prompt == ""
    assert state.prompt_events == []
    request_game_switch(state, "freeze")
    _apply_pending_game_switch(state)
    assert state.config.game_slug == "freeze"
    # The same slug with a restart rewinds instead of being a no-op, so "play
    # again" begins from the first frame rather than continuing the round.
    state.cache = object()
    state.ar_index = 6
    request_game_switch(state, "freeze", restart=True)
    assert state.pending_game_restart is True
    _apply_pending_game_switch(state)
    assert state.cache is None
    assert state.ar_index == 0
    assert state.pending_game_restart is False
    _switch_game(state, "tracer")
    assert state.config.game_slug == "tracer"
    assert state.config.conditioning.prompt == "scene for tracer"


def _gate_state(monkeypatch, *, started):
    """A started session whose gate is armed, with the rollout state it rewinds."""
    monkeypatch.setattr("eden.apps.game2v.session.is_distributed_initialized", lambda: False)
    state = _state("tracer")
    state.config = replace(state.config, wait_for_start=True)
    state.started = started
    state.start_requested = started
    state.pending_game_chunk = None
    state.pending_game_frames = __import__("collections").deque()
    state.postprocess_stream = None
    state.control_stream = SimpleNamespace(reset=lambda: None)
    return state


def test_start_gate_rearms_when_home_is_requested(monkeypatch):
    # Going home is decided on rank zero and broadcast, so every rank enters
    # the wait on the same step rather than on its own local flag.
    state = _gate_state(monkeypatch, started=True)
    assert _enter_start_gate(state) is False
    request_home(state)
    assert _enter_start_gate(state) is True
    # Entering clears the request and rewinds, so the wait is idle and the
    # following step does not re-enter before a game has been picked.
    assert state.pending_home is False
    assert state.started is False
    assert state.start_requested is False
    assert state.cache is None
    assert state.ar_index == 0
    assert _enter_start_gate(state) is True


def test_start_gate_follows_the_broadcast_not_the_local_flag(monkeypatch):
    # A rank whose own flags say "keep going" must still stop when the
    # broadcast says so, or it would generate while its peers wait.
    state = _gate_state(monkeypatch, started=True)
    monkeypatch.setattr("eden.apps.game2v.session.is_distributed_initialized", lambda: True)
    monkeypatch.setattr("eden.apps.game2v.session.broadcast_step_gate", lambda decision: True)
    assert state.pending_home is False
    assert _enter_start_gate(state) is True


def test_start_gate_is_off_without_wait_for_start(monkeypatch):
    # Without the flag the gate never engages, and costs no collective.
    state = _gate_state(monkeypatch, started=False)
    state.config = replace(state.config, wait_for_start=False)
    monkeypatch.setattr("eden.apps.game2v.session.is_distributed_initialized", lambda: True)
    monkeypatch.setattr(
        "eden.apps.game2v.session.broadcast_step_gate",
        lambda decision: pytest.fail("the gate must not broadcast when it is off"),
    )
    assert _enter_start_gate(state) is False
