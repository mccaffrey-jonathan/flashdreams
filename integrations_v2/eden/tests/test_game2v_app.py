# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Game2V application flags: the games root, --game, --autopilot, recording outputs."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from eden.apps.game2v.adapter import create_app_v2_5b_taehv
from eden.apps.game2v.game_renderer import (
    GamesRootError,
    default_autopilot_script,
    games_root,
    registered_game_slugs,
)
from eden.apps.game2v.game_timeline import GameTimeline
from eden.apps.game2v.session import (
    Game2VSession,
    _record_proxy_frames,
    _write_chunk_log,
)

pytestmark = pytest.mark.ci_cpu


def test_games_root_explains_how_to_get_the_games(monkeypatch, tmp_path):
    monkeypatch.delenv("EDEN_GAMES_ROOT", raising=False)
    with pytest.raises(
        GamesRootError, match=r"EDEN_GAMES_ROOT is not set.*EdenLive.*npm ci"
    ):
        games_root()
    # Tolerant callers see no games rather than an exception.
    assert registered_game_slugs() == ()
    monkeypatch.setenv("EDEN_GAMES_ROOT", str(tmp_path))
    with pytest.raises(GamesRootError, match="has no registry.json"):
        games_root()


def test_games_root_reads_the_environment(fake_games_root):
    assert games_root() == fake_games_root.resolve()
    assert registered_game_slugs()[0] == "ember-den"
    assert (
        default_autopilot_script()
        == fake_games_root.resolve() / "_shared" / "autopilot.js"
    )


def test_default_game_is_the_first_registered(fake_games_root):
    app = create_app_v2_5b_taehv()
    app.init(["--device", "cpu"])
    assert app._game_slug == "ember-den"
    assert app._autopilot is None
    app.init(["--device", "cpu", "--game", "jade-duel"])
    assert app._game_slug == "jade-duel"


def test_unknown_game_lists_the_registry(fake_games_root):
    with pytest.raises(
        ValueError, match=r"'freeze' is not in .*ember-den, iron-colossus"
    ):
        create_app_v2_5b_taehv().init(["--device", "cpu", "--game", "freeze"])


def test_live_mode_needs_the_games(monkeypatch):
    monkeypatch.delenv("EDEN_GAMES_ROOT", raising=False)
    with pytest.raises(GamesRootError):
        create_app_v2_5b_taehv().init(["--device", "cpu"])


def test_help_works_without_the_games(monkeypatch, capsys):
    monkeypatch.delenv("EDEN_GAMES_ROOT", raising=False)
    with pytest.raises(SystemExit):
        create_app_v2_5b_taehv().init(["--help"])
    assert "--proxy-video" in capsys.readouterr().out


def test_autopilot_defaults_to_the_shared_script(fake_games_root, tmp_path):
    app = create_app_v2_5b_taehv()
    app.init(["--device", "cpu", "--autopilot"])
    assert app._autopilot == fake_games_root.resolve() / "_shared" / "autopilot.js"
    custom = tmp_path / "pilot.js"
    custom.write_text("// mine")
    app.init(["--device", "cpu", "--autopilot", str(custom)])
    assert app._autopilot == custom
    with pytest.raises(FileNotFoundError, match="autopilot"):
        app.init(["--device", "cpu", "--autopilot", str(tmp_path / "missing.js")])


@pytest.mark.parametrize("flag", [["--max-frames", "9"], ["--prompts-json", "p.json"]])
def test_replay_flags_need_a_proxy_video(fake_games_root, flag):
    with pytest.raises(ValueError, match="requires --proxy-video"):
        create_app_v2_5b_taehv().init(["--device", "cpu", *flag])


def _session(app) -> Game2VSession:
    """Create a session against the stub pipeline already set on ``app``."""
    return cast(Game2VSession, app.create_session(app.session_desc()))


def _stub_pipeline():
    return SimpleNamespace(decoder=SimpleNamespace(spatial_compression_ratio=16))


@pytest.mark.parametrize("side_by_side, width", [(False, 1280), (True, 2560)])
def test_side_by_side_doubles_the_presented_width(fake_games_root, side_by_side, width):
    app = create_app_v2_5b_taehv()
    flags = (
        ["--device", "cpu", "--side-by-side"] if side_by_side else ["--device", "cpu"]
    )
    app.init(flags)
    app._pipeline = _stub_pipeline()
    session = _session(app)
    assert (session.session_desc.video_width, session.session_desc.video_height) == (
        width,
        704,
    )
    # The model and the capture keep the model's own size.
    assert session._config.model_video_width == 1280
    assert session._config.side_by_side is side_by_side


@pytest.mark.parametrize(
    "mode, panel", [("mp4", False), ("native-window", False), ("eden-webrtc", True)]
)
def test_only_the_integration_browser_window_mounts_the_panel(
    fake_games_root, mode, panel
):
    app = create_app_v2_5b_taehv()
    assert app.resolve_client_window_mode(mode) == mode
    app.init(["--device", "cpu"])
    app._pipeline = _stub_pipeline()
    assert _session(app)._config.register_panel is panel


def test_recording_outputs_reach_the_session_config(fake_games_root, tmp_path):
    app = create_app_v2_5b_taehv()
    app.init(
        [
            "--device",
            "cpu",
            "--autopilot",
            "--record-proxy",
            str(tmp_path / "proxy.mp4"),
            "--chunk-log",
            str(tmp_path / "chunks.jsonl"),
        ]
    )
    app._pipeline = _stub_pipeline()
    config = _session(app)._config
    assert config.record_proxy == tmp_path / "proxy.mp4"
    assert config.chunk_log == tmp_path / "chunks.jsonl"
    assert config.autopilot == fake_games_root.resolve() / "_shared" / "autopilot.js"
    assert config.proxy_video is None


def _chunk_state(tmp_path, *, record=False):
    timeline = GameTimeline("ember-den")
    from dataclasses import replace

    chunk = timeline.begin_chunk(9, 24)
    chunk = replace(
        chunk,
        frames=tuple(
            replace(
                frame,
                ui_frame=np.full((4, 6, 3), index, np.uint8),
                condition_frame=np.zeros((4, 6, 3), np.uint8)
                if frame.capture_condition
                else None,
                ui_state={"state": "RUNNING", "frame": index},
            )
            for index, frame in enumerate(chunk.frames)
        ),
    )
    return SimpleNamespace(
        config=SimpleNamespace(
            chunk_log=tmp_path / "logs" / "chunks.jsonl",
            record_proxy=tmp_path / "proxy.mp4" if record else None,
            dynamic_prompt=False,
        ),
        session_desc=SimpleNamespace(frames_per_second_for_step=24),
        chunk_log=None,
        proxy_recorder=None,
        last_game_chunk=chunk,
        frames_generated=9,
        frames_written=9,
        encoded_prompt="",
        opening_prompt="A cave.",
        active_event_lines=("A spider rears up.",),
    )


def test_chunk_log_writes_one_json_line_per_chunk(tmp_path):
    state = _chunk_state(tmp_path)
    for step in range(2):
        _write_chunk_log(
            state,
            step_index=step,
            ar_index=step,
            frame_count=9,
            timings_s={"generate": 0.25},
            chunk_fps=36.0,
        )
    state.chunk_log.close()
    lines = [
        json.loads(line)
        for line in (tmp_path / "logs" / "chunks.jsonl").read_text().splitlines()
    ]
    assert len(lines) == 2
    first = lines[0]
    assert first["chunk"] == 0 and first["frames"] == 9 and first["first_frame"] == 0
    assert first["condition_frames"] == [1, 5, 9]
    assert first["prompt"] == "A cave."
    assert first["events"] == ["A spider rears up."]
    assert first["timings_ms"] == {"generate": 250.0}
    assert first["game_state"] == {"state": "RUNNING", "frame": 8}


def test_record_proxy_queues_every_frame_of_the_chunk(tmp_path, monkeypatch):
    written = []

    class _Recorder:
        def __init__(self, path, *, width, height, fps):
            assert (width, height, fps) == (6, 4, 24)

        def write(self, frames):
            written.append(len(frames))

    monkeypatch.setattr("eden.apps.game2v.recording.ProxyRecorder", _Recorder)
    state = _chunk_state(tmp_path, record=True)
    _record_proxy_frames(state, state.last_game_chunk)
    _record_proxy_frames(state, state.last_game_chunk)
    assert written == [9, 9]


def test_rtx5090_preset_keeps_the_5b_app_and_compiles_dino(fake_games_root):
    from eden.apps.game2v.adapter import (
        GAME2V_EDEN_WORLD_V2_5B_TAEHV_DEFAULTS,
        create_app_v2_5b_taehv_rtx5090,
    )

    app = create_app_v2_5b_taehv_rtx5090()
    defaults = app.defaults
    assert (defaults.pixel_width, defaults.pixel_height, defaults.fps) == (
        1280,
        704,
        24,
    )
    assert defaults.control_stream_factory is (
        GAME2V_EDEN_WORLD_V2_5B_TAEHV_DEFAULTS.control_stream_factory
    )
    app.init(["--device", "cpu"])
    transformer = app.pipeline_config.diffusion_model.transformer
    assert transformer.compile_network and transformer.use_cuda_graph
    assert transformer.network.linear_backend == "rowwise_fp8"
    assert transformer.network.self_attention_backend == "sage_fp8"
    app._pipeline = _stub_pipeline()
    assert _session(app)._config.compile_control_encoder
    app.init(["--device", "cpu", "--no-compile"])
    assert not app.pipeline_config.diffusion_model.transformer.compile_network
    assert not _session(app)._config.compile_control_encoder


def test_every_game2v_entry_point_names_an_adapter_factory():
    import importlib
    from pathlib import Path

    import tomllib

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    apps = pyproject["project"]["entry-points"]["flashdreams.applications_v2"]
    game2v = {
        slug: target for slug, target in apps.items() if slug.startswith("game2v-")
    }
    assert {
        "game2v-eden-world-v2-5b-taehv",
        "game2v-eden-world-v2-5b-taehv-rtx-5090",
        "game2v-eden-world-v2-14b-taehv",
    } <= set(game2v)
    for target in game2v.values():
        module, _, name = target.partition(":")
        assert callable(getattr(importlib.import_module(module), name)), target
