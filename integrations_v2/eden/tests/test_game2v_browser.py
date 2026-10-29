# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``eden-webrtc`` browser window: routes, HUD input, browser messages."""

from __future__ import annotations

import argparse
import dataclasses
import json
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

pytest.importorskip("aiortc")

from eden.apps.game2v import extensions  # noqa: E402
from eden.apps.game2v.browser import EDEN_WEBRTC_MODE, EdenWebRTCMode  # noqa: E402
from eden.apps.game2v.browser_server import (  # noqa: E402
    EdenWebRTCClientWindow,
    EdenWebRTCServer,
    browser_page,
    parse_client_action,
)
from eden.apps.game2v.game_hud import GameUIActionEvent  # noqa: E402
from eden.apps.game2v.game_timeline import GameTimeline  # noqa: E402
from eden.apps.game2v.game_ui import (  # noqa: E402
    BrowserFeed,
    Game2VFrame,
    Game2VUILoop,
)
from eden.apps.game2v.session import _feed_browser, _game_input_payloads  # noqa: E402

from flashdreams.runtime_v2.session_desc import SessionDesc  # noqa: E402
from flashdreams.runtime_v2.user_input_events import UserInputEvents  # noqa: E402
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout  # noqa: E402

pytestmark = pytest.mark.ci_cpu


def test_page_is_flashdreams_page_plus_eden_script():
    page = browser_page()
    assert page.index('<script src="/app.js"></script>') < page.index(
        '<script src="/eden.js"></script>'
    )


@pytest.fixture
def server(fake_games_root):
    hud = fake_games_root / "ember-den" / "hud"
    hud.mkdir()
    (hud / "overlay.js").write_text(
        "export function mount(root) { return {update() {}}; }"
    )
    (hud / "overlay.css").write_text(".x{}")
    three = fake_games_root / "node_modules" / "three" / "build"
    three.mkdir(parents=True)
    (three / "three.module.js").write_text("export const REVISION = '161';")
    (fake_games_root / "host").mkdir()
    (fake_games_root / "host" / "standalone.js").write_text("// host")
    instance = EdenWebRTCServer(host="127.0.0.1", port=0)
    yield instance
    instance.close()
    extensions.unregister_http_extension("game2v")


def _get(server, path):
    with urllib.request.urlopen(
        f"{server.url.rstrip('/')}{path}", timeout=10
    ) as response:
        return (
            response.status,
            response.headers.get("content-type", ""),
            response.read(),
        )


def _status(server, path):
    try:
        return _get(server, path)[0]
    except urllib.error.HTTPError as error:
        return error.code


def test_server_serves_the_page_script_hud_and_game_tree(server):
    status, content_type, body = _get(server, "/")
    assert status == 200 and "text/html" in content_type
    assert b'src="/eden.js"' in body
    status, content_type, body = _get(server, "/eden.js")
    assert status == 200 and "javascript" in content_type
    assert b"game-overlay" in body
    assert _get(server, "/app.js")[0] == 200
    status, content_type, body = _get(server, "/games/ember-den/hud/overlay.js")
    assert "javascript" in content_type and b"mount" in body
    assert "text/css" in _get(server, "/games/ember-den/hud/overlay.css")[1]
    assert b"REVISION" in _get(server, "/vendor/three.module.js")[2]
    assert _get(server, "/host/standalone.js")[0] == 200
    assert _get(server, "/game/ember-den/index.html")[0] == 200


@pytest.mark.parametrize(
    "path",
    [
        "/games/ember-den/hud/index.html",  # only the overlay files
        "/games/..%2Fregistry.json/hud/overlay.js",
        "/game/not-registered/index.html",
        "/game/ember-den/..%2F..%2Fregistry.json",
        "/vendor/evil.js",
        "/host/..%2Fregistry.json",
    ],
)
def test_server_refuses_paths_outside_the_game_files(server, path):
    assert _status(server, path) == 404


def test_extension_routes_list_and_dispatch_the_panel(server):
    async def handler(request, tail):
        from aiohttp import web

        return web.json_response({"tail": tail})

    extensions.register_http_extension("game2v", handler, client_script="panel.js")
    manifest = json.loads(_get(server, "/extensions.json")[2])
    assert manifest == {"scripts": ["/ext/game2v/panel.js"]}
    assert json.loads(_get(server, "/ext/game2v/games")[2]) == {"tail": "games"}


def test_hud_clicks_arrive_as_game_actions(server):
    received = []
    server.register_input_callback(received.append)
    server.open(SessionDesc())
    server._buffer_browser_message(
        json.dumps({"type": "client_action", "channel": "game_ui", "action": "restart"})
    )
    (event,) = received
    assert isinstance(event, GameUIActionEvent)
    assert (event.action, event.channel) == ("restart", "game_ui")
    # Core still handles everything else.
    server._buffer_browser_message(
        json.dumps({"type": "keyboard", "key": "a", "pressed": True})
    )
    assert received[-1].get_type_name() == "keyboard"
    with pytest.raises(ValueError):
        server._buffer_browser_message(json.dumps({"type": "nonsense"}))
    assert _game_input_payloads(UserInputEvents([event])) == [
        {"action": "restart", "phase": "down"},
        {"action": "restart", "phase": "up"},
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"action": ""},
        {"action": 3},
        {"action": "x" * 65},
        {"action": "a", "channel": 1},
    ],
)
def test_client_action_validation(payload):
    with pytest.raises(ValueError):
        parse_client_action(payload, np.uint64(0))


def test_a_reconnecting_client_gets_the_current_prompt_and_hud(server):
    sent = []
    server._control_channel = SimpleNamespace(readyState="open", send=sent.append)
    server.send_client_messages(
        (
            {"type": "prompt", "static": "A cave.", "events": []},
            {"type": "game_ui", "slug": "ember-den", "frame_id": 3, "state": {}},
            {"type": "other"},
        )
    )
    server._sent_cursor_options = None  # what a new peer or channel resets
    sent.clear()
    server._send_cursor_options()
    types = [json.loads(text)["type"] for text in sent]
    assert types == ["cursor_options", "prompt", "game_ui"]
    sent.clear()
    server._send_cursor_options()  # not a new client: nothing replayed
    assert sent == []


def test_feed_carries_hud_and_prompt_to_the_presented_frame():
    timeline = GameTimeline("ember-den")
    chunk = timeline.begin_chunk(9, 24)
    paired = [
        dataclasses.replace(frame, ui_state={"n": i})
        for i, frame in enumerate(chunk.frames)
    ]
    feed = BrowserFeed()
    queued = []
    ui = SimpleNamespace(_invoke_async=lambda operation: queued.append(operation))
    state: Any = SimpleNamespace(
        browser_feed=ui,
        published_frames=12,
        static_prompt="A cave.",
        opening_prompt="",
        active_event_lines=("A spider rears up.",),
    )
    _feed_browser(state, paired, generation=2)
    (operation,) = queued
    operation(feed)
    first = feed.take(2, 13)
    assert first[0] == {
        "type": "prompt",
        "static": "A cave.",
        "events": ["A spider rears up."],
    }
    assert first[1]["type"] == "game_ui" and first[1]["state"] == {"n": 0}
    assert first[1]["frame_id"] == paired[0].game_frame_id
    assert [message["state"] for message in feed.take(2, 21)] == [
        {"n": i} for i in range(1, 9)
    ]


def test_ui_loop_attaches_the_presented_frames_messages(monkeypatch):
    feed = BrowserFeed()
    feed.add(0, {1: [{"type": "prompt"}, {"type": "game_ui", "frame_id": 1}]})
    loop: Any = Game2VUILoop()
    loop.register_session_loop_objects(
        state=feed,
        frequency=0,
        shutdown_event=__import__("threading").Event(),
        failure_queue=__import__("queue").Queue(),
    )
    loop.output_layout = VideoTensorLayout.tchw
    loop.presented_model_frames = lambda: (torch.zeros(3, 2, 2),)
    monkeypatch.setattr(
        Game2VUILoop, "presented_model_frame_count", property(lambda self: 1)
    )
    (result,) = loop.step(0, UserInputEvents([]))
    assert isinstance(result, Game2VFrame)
    assert [message["type"] for message in result.client_messages] == [
        "prompt",
        "game_ui",
    ]


def test_webrtc_resolves_to_the_browser_window_when_installed(
    fake_games_root, monkeypatch
):
    from eden.apps.game2v.adapter import create_app_v2_5b_taehv

    monkeypatch.setattr(
        "eden.apps.game2v.browser.eden_webrtc_mode_installed", lambda: True
    )
    app = create_app_v2_5b_taehv()
    assert app.resolve_client_window_mode("webrtc") == EDEN_WEBRTC_MODE
    assert app.resolve_client_window_mode("mp4") == "mp4"
    app.resolve_client_window_mode("webrtc")
    app.init(["--device", "cpu"])
    assert app._side_by_side is True
    monkeypatch.setattr(
        "eden.apps.game2v.browser.eden_webrtc_mode_installed", lambda: False
    )
    assert app.resolve_client_window_mode("webrtc") == "webrtc"


def test_mode_creates_the_browser_window_on_the_requested_interface():
    mode = EdenWebRTCMode()
    assert mode.name == "eden-webrtc"
    window = cast(
        EdenWebRTCClientWindow,
        mode.create(argparse.Namespace(host="127.0.0.1", port=0)),
    )
    try:
        assert isinstance(window.server, EdenWebRTCServer)
        assert window.server.host == "127.0.0.1"
        assert mode.starting(window) == f"Open {window.server.url} in a browser."
    finally:
        window.close()
