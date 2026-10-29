# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``eden-webrtc`` page end to end in headless Chromium, with a stub model.

Manual: needs Playwright's Chromium (``python -m playwright install chromium``)
and aiortc. No games checkout, weights or GPU: the proxy is a replayed synthetic
video and the model a stub, so this checks the transport -- the side-by-side
track cropped into two panels, the prompt panel, the HUD overlay and HUD clicks
reaching the capture worker's key queue.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from cam2v import Cam2VConditioning
from PIL import Image

pytestmark = pytest.mark.manual

pytest.importorskip("aiortc")
pytest.importorskip("playwright.sync_api")
av = pytest.importorskip("av")

HEIGHT, WIDTH = 72, 128
FRAMES = 24 * 60
"""A minute of video, so the session outlives the browser's ICE gathering."""


class _Pipeline:
    def initialize_cache(self, text, image, *, release_oneshot_encoders=True):
        return object()

    def get_num_output_frames(self, autoregressive_index):
        return 9 if autoregressive_index == 0 else 12

    def generate(self, autoregressive_index, cache, input):
        count = self.get_num_output_frames(autoregressive_index)
        return torch.full((count, 3, HEIGHT, WIDTH), 0.5)

    def finalize(self, autoregressive_index, cache):
        return {}


def test_browser_shows_game_render_prompt_and_hud(tmp_path, fake_games_root):
    from eden.apps.game2v.browser_server import EdenWebRTCClientWindow
    from eden.apps.game2v.session import (
        Game2VSession,
        Game2VSessionConfig,
        _drain_live_game_events,
    )
    from playwright.sync_api import sync_playwright

    from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
    from flashdreams.runtime_v2.session_runner import run_session

    hud = fake_games_root / "ember-den" / "hud"
    hud.mkdir()
    (hud / "overlay.css").write_text("button{position:absolute;left:4px;top:4px}")
    (hud / "overlay.js").write_text(
        "export function mount(root) {"
        " root.innerHTML = '<button data-game-action=\"restart\">Play again</button>';"
        " return {update() {}}; }"
    )
    video = tmp_path / "proxy.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        for index in range(FRAMES):
            pixels = np.full((HEIGHT, WIDTH, 3), (index * 5) % 256, np.uint8)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(pixels, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    cover = tmp_path / "cover.png"
    Image.new("RGB", (WIDTH, HEIGHT)).save(cover)
    config = Game2VSessionConfig(
        conditioning=Cam2VConditioning(
            prompt="A browser test scene.",
            first_frame_path=cover,
            base_intrinsics=torch.tensor([64.0, 36.0, 64.0, 36.0]),
            world_scale=1.0,
        ),
        total_blocks=(FRAMES - 3) // 12,
        device=torch.device("cpu"),
        model_video_width=WIDTH,
        model_video_height=HEIGHT,
        first_frame_dtype=torch.float32,
        first_frame_interpolation="linear",
        warmup_blocks=0,
        game_slug="ember-den",
        proxy_video=video,
        proxy_frames=FRAMES - 3,
        side_by_side=True,
        register_panel=True,
        control_stream_factory=lambda config, desc: SimpleNamespace(
            submit=lambda index, pending: None,
            take=lambda index: torch.zeros(1),
            reset=lambda: None,
            close=lambda: None,
        ),
    )
    session_desc = SessionDesc(
        presentation_mode=PresentationMode.CONTINUOUS,
        frames_per_second_for_ui=60,
        frames_per_second_for_step=24,
        video_width=2 * WIDTH,
        video_height=HEIGHT,
    )
    session = Game2VSession(
        pipeline=_Pipeline(), config=config, session_desc=session_desc
    )
    window = EdenWebRTCClientWindow(host="127.0.0.1", port=0)
    runner = threading.Thread(
        target=run_session, args=(session, window), kwargs={"timeout_seconds": 120}
    )
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            # Opened before the session exists, as a browser opened during model
            # loading is: the page must still pick up the session's extensions.
            page.goto(window.server.url)
            page.wait_for_timeout(1000)
            runner.start()
            page.wait_for_function(
                "document.getElementById('prompt').textContent.includes('browser test scene')",
                timeout=30000,
            )
            page.wait_for_function(
                "['video', 'game-video'].every(id => document.getElementById(id).readyState >= 2)",
                timeout=30000,
            )
            assert (
                page.evaluate("document.getElementById('video').videoWidth")
                == 2 * WIDTH
            )
            page.wait_for_function(
                "document.getElementById('game-overlay').shadowRoot?.querySelector('button')",
                timeout=30000,
            )
            # The game panel extension mounted into the layout.
            page.wait_for_selector(".game2v-host", timeout=10000)
            page.evaluate(
                "document.getElementById('game-overlay').shadowRoot.querySelector('button').click()"
            )
            state = session.model_loop.state
            payloads = []
            for _ in range(100):
                payloads += _drain_live_game_events(state)
                if payloads:
                    break
                page.wait_for_timeout(50)
            assert {"action": "restart", "phase": "down"} in payloads
            # Keys pressed in the page take the same path to the game.
            page.click("#video")
            page.keyboard.press("ArrowLeft")
            payloads = []
            for _ in range(100):
                payloads += _drain_live_game_events(state)
                if {"action": "arrowleft", "phase": "up"} in payloads:
                    break
                page.wait_for_timeout(50)
            assert payloads[-2:] == [
                {"action": "arrowleft", "phase": "down"},
                {"action": "arrowleft", "phase": "up"},
            ]
            assert not errors, errors
            browser.close()
    finally:
        session._shutdown_event.set()
        if runner.is_alive():
            runner.join(timeout=30)
