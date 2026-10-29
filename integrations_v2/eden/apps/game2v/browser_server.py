# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Ported for FlashDreams v0.2.0 from the EdenLive fork's in-core edits to
# flashdreams/runtime_v2/serving/webrtc_server.py: game routes, the /ext
# extension mount, client-message forwarding and HUD button input now live in
# this integration-owned subclass, so core's server is unchanged.

"""Game2V browser window: generated video, game render, HUD overlay and game panel.

Run through the ``eden-webrtc`` client-window mode (:mod:`eden.apps.game2v.browser`).
The page is FlashDreams' own (``/`` and ``/app.js``) plus ``/eden.js``. The UI
loop sends the side-by-side canvas (game render | generated video) as the one
video track, and ``eden.js`` crops it into two panels, so the two are paired
frame for frame. HUD state and the chunk prompt arrive with each frame as
:class:`~eden.apps.game2v.game_ui.Game2VFrame` messages and go out on the
control channel.

The extra routes are unauthenticated, as the fork's were: bind to localhost (the
default ``--host``) or a private network.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import socket
import threading
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aiohttp import web
from numpy import uint64

from flashdreams.runtime_v2.serving import webrtc_server
from flashdreams.runtime_v2.serving.webrtc_server import WebRTCServer
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.webrtc_client_window import WebRTCClientWindow

from .extensions import dispatch_extension, serve_extension_manifest
from .game_hud import GAME_UI_MESSAGE_TYPE, GameUIActionEvent
from .game_renderer import GamesRootError, games_root
from .game_ui import Game2VFrame
from .session import PROMPT_MESSAGE_TYPE

_STATIC = Path(__file__).resolve().parent / "static"
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_OVERLAY_FILES = {"overlay.js": "text/javascript", "overlay.css": "text/css"}
_REPLAYED_TYPES = (PROMPT_MESSAGE_TYPE, GAME_UI_MESSAGE_TYPE)
"""Message types a reconnecting client is sent again: they describe the current frame."""
_MAX_ACTION_LENGTH = 64


def browser_page() -> str:
    """FlashDreams' own page with the Game2V script loaded after ``/app.js``."""
    marker = '<script src="/app.js"></script>'
    page = webrtc_server._BROWSER_PAGE
    if marker not in page:
        raise RuntimeError("FlashDreams' WebRTC page no longer loads /app.js.")
    return page.replace(marker, f'{marker}\n    <script src="/eden.js"></script>', 1)


def parse_client_action(
    payload: Mapping[str, Any], timestamp: uint64
) -> GameUIActionEvent:
    """Validate a ``{"type": "client_action", "action", "channel"}`` browser message."""
    action = payload.get("action")
    channel = payload.get("channel", "")
    if not isinstance(action, str) or not action or len(action) > _MAX_ACTION_LENGTH:
        raise ValueError("Client action requires a short non-empty action name.")
    if not isinstance(channel, str) or len(channel) > _MAX_ACTION_LENGTH:
        raise ValueError("Client action channel must be a short string.")
    return GameUIActionEvent(timestamp=timestamp, action=action, channel=channel)


def _games_root_or_404() -> Path:
    try:
        return games_root()
    except GamesRootError as error:
        raise web.HTTPNotFound(reason="EDEN_GAMES_ROOT is not set.") from error


def _file_response(path: Path, *, root: Path) -> web.Response:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise web.HTTPNotFound()
    if resolved.suffix in {".js", ".mjs"}:
        content_type = "text/javascript"
    elif resolved.suffix == ".css":
        content_type = "text/css"
    elif resolved.suffix == ".html":
        content_type = "text/html"
    else:
        content_type = (
            mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        )
    return web.Response(body=resolved.read_bytes(), content_type=content_type)


class EdenWebRTCServer(WebRTCServer):
    """:class:`WebRTCServer` with the Game2V page, game routes and client messages."""

    def __init__(self, **kwargs: Any) -> None:
        self._latest_messages: dict[str, str] = {}
        self._latest_lock = threading.Lock()
        super().__init__(**kwargs)

    async def _start_server(self) -> None:
        """Core's routes plus the Game2V ones (core builds its router here, so copied)."""
        self._offer_lock = asyncio.Lock()
        app = web.Application()
        app.router.add_get("/", self._serve_browser)
        app.router.add_get("/app.js", self._serve_browser_script)
        app.router.add_get("/eden.js", self._serve_eden_script)
        app.router.add_get("/healthz", self._health)
        app.router.add_post("/api/webrtc/offer", self._offer)
        app.router.add_get("/games/{slug}/hud/{filename}", self._serve_game_overlay)
        app.router.add_get("/extensions.json", serve_extension_manifest)
        app.router.add_route("*", "/ext/{name}/{tail:.*}", dispatch_extension)
        # The game panel's live preview iframe loads the game tree from here.
        app.router.add_get("/vendor/{filename}", self._serve_game_vendor)
        app.router.add_get("/host/{filename}", self._serve_game_host)
        app.router.add_get("/assets/{tail:.*}", self._serve_game_assets)
        app.router.add_get("/game/{slug}/{tail:.*}", self._serve_registered_game)
        runner = web.AppRunner(app)
        await runner.setup()
        address_family = socket.AF_INET6 if ":" in self._host else socket.AF_INET
        server_socket = socket.socket(address_family, socket.SOCK_STREAM)
        try:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self._host, self._port))
            server_socket.setblocking(False)
            server_socket.listen(128)
            self._port = int(server_socket.getsockname()[1])
            site = web.SockSite(runner, server_socket)
            await site.start()
        except Exception:
            server_socket.close()
            await runner.cleanup()
            raise
        self._runner = runner

    async def _serve_browser(self, _: web.Request) -> web.Response:
        return web.Response(
            text=browser_page(),
            content_type="text/html",
            headers={"cache-control": "no-store"},
        )

    async def _serve_eden_script(self, _: web.Request) -> web.Response:
        return web.Response(
            body=(_STATIC / "eden.js").read_bytes(),
            content_type="text/javascript",
            headers={"cache-control": "no-store"},
        )

    async def _serve_game_overlay(self, request: web.Request) -> web.Response:
        """Serve ``<slug>/hud/overlay.{js,css}`` from the games checkout."""
        slug = request.match_info["slug"]
        filename = request.match_info["filename"]
        if filename not in _OVERLAY_FILES or not _SLUG.fullmatch(slug):
            raise web.HTTPNotFound()
        root = _games_root_or_404()
        return _file_response(root / slug / "hud" / filename, root=root)

    async def _serve_game_vendor(self, request: web.Request) -> web.Response:
        """Serve the shared three.js builds the games import as ``/vendor/``."""
        filename = request.match_info["filename"]
        root = _games_root_or_404()
        if filename == "three.module.js":
            path = Path(
                os.environ.get(
                    "EDEN_THREE_MODULE",
                    root / "node_modules/three/build/three.module.js",
                )
            )
            if not path.is_file():
                raise web.HTTPNotFound()
            return web.Response(body=path.read_bytes(), content_type="text/javascript")
        if filename in ("three-demo.module.js", "three.core.js"):
            name = (
                "three.module.js"
                if filename == "three-demo.module.js"
                else "three.core.js"
            )
            return _file_response(
                root / "node_modules/three-demo/build" / name, root=root
            )
        raise web.HTTPNotFound()

    async def _serve_game_host(self, request: web.Request) -> web.Response:
        filename = request.match_info["filename"]
        if not re.fullmatch(r"[A-Za-z0-9._-]+\.js", filename):
            raise web.HTTPNotFound()
        root = _games_root_or_404()
        return _file_response(root / "host" / filename, root=root)

    async def _serve_game_assets(self, request: web.Request) -> web.Response:
        root = _games_root_or_404()
        return _file_response(root / "assets" / request.match_info["tail"], root=root)

    async def _serve_registered_game(self, request: web.Request) -> web.Response:
        """Serve ``<slug>/...`` of a registered game (or ``host``/``runtime``)."""
        slug = request.match_info["slug"]
        if not _SLUG.fullmatch(slug):
            raise web.HTTPNotFound()
        root = _games_root_or_404()
        registry = json.loads((root / "registry.json").read_text(encoding="utf-8"))
        slugs = {entry.get("slug") for entry in registry.get("games", [])}
        if slug not in slugs and slug not in {"host", "runtime"}:
            raise web.HTTPNotFound()
        return _file_response(root / slug / request.match_info["tail"], root=root)

    def write(self, result: StepResult) -> None:
        """Queue the frame, then the messages that describe it.

        A client therefore never hears about a frame it has not been offered.
        """
        super().write(result)
        if isinstance(result, Game2VFrame) and result.client_messages:
            self.send_client_messages(result.client_messages)

    def send_client_messages(self, messages: tuple[Mapping[str, Any], ...]) -> None:
        """Send JSON payloads on the control channel, keeping the latest per type."""
        encoded = tuple(json.dumps(message) for message in messages)
        with self._latest_lock:
            for message, text in zip(messages, encoded, strict=True):
                if message.get("type") in _REPLAYED_TYPES:
                    self._latest_messages[str(message["type"])] = text
        loop = self._loop
        if loop is not None and self._control_channel is not None:
            loop.call_soon_threadsafe(self._send_encoded, encoded)

    def _send_encoded(self, encoded: tuple[str, ...]) -> None:
        """Send on the WebRTC loop thread, the only one that touches the channel."""
        channel = self._control_channel
        if channel is None or channel.readyState != "open":
            return
        for text in encoded:
            channel.send(text)

    def _send_cursor_options(self) -> None:
        """Also replay the current prompt and HUD to a client that just connected."""
        fresh_client = self._sent_cursor_options is None
        super()._send_cursor_options()
        if fresh_client and self._sent_cursor_options is not None:
            with self._latest_lock:
                latest = tuple(self._latest_messages.values())
            self._send_encoded(latest)

    def _buffer_browser_message(self, raw_message: object) -> None:
        """Accept the HUD's ``client_action`` messages; everything else is core's."""
        if isinstance(raw_message, str):
            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and payload.get("type") == "client_action":
                timestamp = self._timestamp_us()
                if timestamp is not None:
                    self._append_event(parse_client_action(payload, timestamp))
                return
        super()._buffer_browser_message(raw_message)


class EdenWebRTCClientWindow(WebRTCClientWindow):
    """:class:`WebRTCClientWindow` serving through :class:`EdenWebRTCServer`.

    Core's constructor builds its own server, so this one repeats its few lines
    with the subclass in its place.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        startup_timeout_seconds: float = 10.0,
    ) -> None:
        self._input_events: deque[UserInputEvent] = deque()
        self._hide_cursor = False
        self._lock_cursor_to_window = False
        self._input_lock = threading.Lock()
        self._session_event_offset_us = uint64(0)
        self.server = EdenWebRTCServer(
            host=host, port=port, startup_timeout_seconds=startup_timeout_seconds
        )

        def handle_input(event: UserInputEvent) -> None:
            with self._input_lock:
                self._input_events.append(event)

        self.server.register_input_callback(handle_input)


__all__ = [
    "EdenWebRTCClientWindow",
    "EdenWebRTCServer",
    "browser_page",
    "parse_client_action",
]
