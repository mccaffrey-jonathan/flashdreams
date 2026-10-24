# SPDX-License-Identifier: Apache-2.0
"""One manually stepped Chromium game, captured on its own thread.

A single page renders every model frame. Its frames serve both consumers of
the same render: they stream to the client's small game window, and the
latent-aligned subset conditions the model. The worker renders chunk N while
the model generates chunk N-1, pacing that render across the last generate
so live keys can land between frames.
"""
from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import tempfile
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np
from loguru import logger
from PIL import Image

from .game_timeline import GameChunkPacket, GameFramePacket

# Frames are encoded at the model's own frame size so no resampling happens
# between the game, the condition encoder and the game video sent to the client.
DEFAULT_CAPTURE_WIDTH = 832
DEFAULT_CAPTURE_HEIGHT = 480

# GPU containers ship NVIDIA's EGL libraries but register them only when
# NVIDIA_DRIVER_CAPABILITIES includes "graphics". Without the vendor JSON, GLVND
# reports Mesa alone, Mesa has no DRI node, and Chromium silently falls back to
# SwiftShader, where the toDataURL read-back costs ~4x more than on hardware.
# Registering the library ourselves is enough; a missing or unusable library
# makes the loader degrade rather than fail, so this is safe without a GPU.
_NVIDIA_EGL_VENDOR = {
    'file_format_version': '1.0.0',
    'ICD': {'library_path': 'libEGL_nvidia.so.0'},
}
_MESA_EGL_VENDOR = Path('/usr/share/glvnd/egl_vendor.d/50_mesa.json')
_GPU_LAUNCH_ARGS = ('--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=gl-egl')

_WEBGL_RENDERER_QUERY = '''() => {
  const c = document.createElement('canvas');
  const gl = c.getContext('webgl2') || c.getContext('webgl');
  if (!gl) return 'no webgl context';
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)
             : gl.getParameter(gl.RENDERER);
}'''
_SOFTWARE_RENDERER_MARKS = ('SwiftShader', 'llvmpipe', 'softpipe')

STANDALONE_HOST = 'standalone'
"""Registry ``host`` of a game that owns its loop, such as a GamePrototypeTurbo export.

Such a page is served unmodified except for ``host/standalone.js``, injected
ahead of every module so it can take over the page's clock.
"""


def standalone_page(html: str, entry: Mapping[str, Any]) -> str:
    """Inject the virtual-clock host into a standalone game's ``index.html``."""
    config = {
        'restartKeys': list(entry.get('restart_keys') or ['r']),
        'eventPrompts': dict(entry.get('event_prompts') or {}),
    }
    # `</` cannot appear inside the inline script, whatever a prompt contains.
    payload = json.dumps(config).replace('</', '<\\/')
    script = (
        f'<script>window.gameStandalone={payload}</script>'
        '<script src="/host/standalone.js"></script>'
    )
    head = re.search(r'<head[^>]*>', html, flags=re.IGNORECASE)
    if head is None:
        raise ValueError('Standalone game index.html has no <head>.')
    return html[: head.end()] + script + html[head.end():]


def registered_game_slugs(*, games_root: Path | None = None) -> tuple[str, ...]:
    """Return every slug ``registry.json`` declares, in registry order.

    This is the source of truth for ``--game`` choices, so a newly ported game
    only needs a registry entry. A missing registry yields ``()``.
    """
    root = games_root or Path(
        os.environ.get(
            'EDEN_GAMES_ROOT', Path(__file__).resolve().parents[4] / 'games'
        )
    ).resolve()
    registry = root / 'registry.json'
    if not registry.is_file():
        return ()
    entries = json.loads(registry.read_text())['games']
    return tuple(entry['slug'] for entry in entries if isinstance(entry.get('slug'), str))


def game_static_prompt(slug: str, *, games_root: Path | None = None) -> str:
    """Return the ``prompt`` a game's registry entry declares.

    The static prompt describes the scene and never changes during a rollout.
    Dynamic event lines are appended to it per chunk; this text is only the
    base. An unregistered slug or an entry without the field yields ``''``.
    """
    root = games_root or Path(
        os.environ.get(
            'EDEN_GAMES_ROOT', Path(__file__).resolve().parents[4] / 'games'
        )
    ).resolve()
    registry = root / 'registry.json'
    if not registry.is_file():
        return ''
    entries = json.loads(registry.read_text())['games']
    for entry in entries:
        if entry.get('slug') == slug:
            prompt = entry.get('prompt', '')
            return prompt.strip() if isinstance(prompt, str) else ''
    return ''


class GameRenderer:
    def __init__(
        self,
        slug: str,
        *,
        width: int = DEFAULT_CAPTURE_WIDTH,
        height: int = DEFAULT_CAPTURE_HEIGHT,
        games_root: Path | None = None,
    ):
        from playwright.sync_api import sync_playwright

        if width <= 0 or height <= 0:
            raise ValueError('Game capture size must be positive')
        self.root = Path(games_root or os.environ.get('EDEN_GAMES_ROOT', Path(__file__).resolve().parents[4] / 'games')).resolve()
        entries = json.loads((self.root / 'registry.json').read_text())['games']
        entry = next((entry for entry in entries if entry['slug'] == slug), None)
        if entry is None:
            raise ValueError(f'Unregistered game: {slug}')
        self.slug = slug
        self.entry = entry
        self.width = width
        self.height = height
        self.last_evaluate_s = 0.0
        self.last_decode_s = 0.0
        self.webgl_renderer = ''
        self._vendor_dir: tempfile.TemporaryDirectory | None = None
        self._playwright = sync_playwright().start()
        self._browser = None
        try:
            self._browser = self._playwright.chromium.launch(
                headless=True, **self._gpu_launch_options()
            )
            self.page = self._browser.new_page(viewport={'width': width, 'height': height}, device_scale_factor=1)
            self.page.route('**/*', self._route)
            self.page.goto(f'http://eden.local/{slug}/index.html')
            # Interval polling: Playwright's default polls on requestAnimationFrame,
            # which host/standalone.js takes over and never runs on its own.
            self.page.wait_for_function('Boolean(window.gameBackend)', timeout=30000, polling=50)
            self._log_webgl_renderer()
        except BaseException:
            self.close()
            raise

    def _gpu_launch_options(self) -> dict:
        """Register NVIDIA's EGL library so Chromium can use a present GPU.

        Set ``EDEN_GAME_GPU=0`` to keep Chromium on its software renderer.
        """
        if os.environ.get('EDEN_GAME_GPU', '1') != '1':
            return {}
        self._vendor_dir = tempfile.TemporaryDirectory(prefix='eden-egl-vendor-')
        vendor = Path(self._vendor_dir.name) / '10_nvidia.json'
        vendor.write_text(json.dumps(_NVIDIA_EGL_VENDOR))
        # This variable replaces the vendor list rather than extending it, so
        # Mesa has to be named explicitly to stay available as the fallback.
        filenames = str(vendor)
        if _MESA_EGL_VENDOR.is_file():
            filenames = f'{vendor}:{_MESA_EGL_VENDOR}'
        return {
            'args': list(_GPU_LAUNCH_ARGS),
            'env': {**os.environ, '__EGL_VENDOR_LIBRARY_FILENAMES': filenames},
        }

    def _log_webgl_renderer(self) -> None:
        """Record which renderer Chromium chose, so a silent fallback is visible."""
        try:
            self.webgl_renderer = self.page.evaluate(_WEBGL_RENDERER_QUERY)
        except Exception:  # a renderer string is never worth failing a session
            logger.opt(exception=True).debug('Could not read the WebGL renderer.')
            return
        software = any(mark in self.webgl_renderer for mark in _SOFTWARE_RENDERER_MARKS)
        logger.info(
            'Game2V capture WebGL renderer: {}{}',
            self.webgl_renderer,
            ' (software rasterization; capture encode is ~4x slower than on a GPU)'
            if software
            else '',
        )

    def _route(self, route):
        url = urlsplit(route.request.url)
        if url.hostname != 'eden.local':
            route.abort()
            return
        rel = unquote(url.path).lstrip('/')
        # A benchmark autopilot is served from outside the games tree, and only
        # when this session explicitly opted into one.
        if rel.startswith('bench/'):
            autopilot = os.environ.get('EDEN_GAME_AUTOPILOT', '')
            script = Path(autopilot).resolve() if autopilot else None
            if script is None or not script.is_file() or script.name != rel[len('bench/'):]:
                route.fulfill(status=404, body=f'Missing game resource: {rel}')
                return
            route.fulfill(body=script.read_bytes(), content_type='text/javascript')
            return
        if rel.startswith('assets/'):
            rel = f'{self.slug}/{rel}'
        if rel == 'vendor/three.module.js':
            path = Path(os.environ.get('EDEN_THREE_MODULE', self.root / 'node_modules/three/build/three.module.js')).resolve()
        elif rel in ('vendor/three-demo.module.js', 'vendor/three.core.js'):
            name = 'three.module.js' if rel.endswith('three-demo.module.js') else 'three.core.js'
            path = self.root / 'node_modules/three-demo/build' / name
        elif rel == 'vendor/rapier.es.js':
            path = Path(os.environ.get('EDEN_RAPIER_MODULE', self.root / 'node_modules/@dimforge/rapier3d-compat/rapier.es.js')).resolve()
        else:
            path = (self.root / rel).resolve()
            if not path.is_relative_to(self.root):
                route.fulfill(status=403, body='Forbidden')
                return
        if not path.is_file():
            route.fulfill(status=404, body=f'Missing game resource: {rel}')
            return
        body = path.read_bytes()
        if rel == f'{self.slug}/index.html' and self.entry.get('host') == STANDALONE_HOST:
            html = standalone_page(body.decode(), self.entry)
            # Same opt-in autopilot as the built-in pages; it must load after
            # host/standalone.js so it sees the virtual clock.
            autopilot = os.environ.get('EDEN_GAME_AUTOPILOT', '')
            if autopilot:
                marker = '<script src="/host/standalone.js"></script>'
                html = html.replace(marker, marker + f'<script src="/bench/{Path(autopilot).name}"></script>', 1)
            body = html.encode()
        elif rel == f'{self.slug}/index.html':
            html = body.decode().replace('<script type="module" src="/shared/host/play-page.js"></script>',
                f'<script>window.gameSlug={json.dumps(self.slug)}</script><script type="module" src="/host/backend.js"></script>')
            # EDEN_GAME_AUTOPILOT names a script injected after the backend, so a
            # headless benchmark can drive the game. A real session leaves it unset
            # and the page is unchanged.
            autopilot = os.environ.get('EDEN_GAME_AUTOPILOT', '')
            if autopilot:
                html = html.replace(
                    '<script type="module" src="/host/backend.js"></script>',
                    '<script type="module" src="/host/backend.js"></script>'
                    f'<script src="/bench/{Path(autopilot).name}"></script>',
                )
            html = html.replace('</head>', '''<style>
body{display:block!important}#view-big{position:absolute!important;width:100vw!important;height:100vh!important;visibility:hidden}
#app canvas{visibility:hidden}#right,#view-small{display:contents!important}
#desc-panel,#record-bar{display:none!important}
#small-stage{position:absolute!important;inset:0!important;width:100vw!important;height:100vh!important;transform:none!important}
#small-canvas{width:100%!important;height:100%!important}
*{animation:none!important;transition:none!important}
</style></head>''')
            body = html.encode()
        route.fulfill(body=body, content_type='text/javascript' if path.suffix in ('.js', '.mjs') else mimetypes.guess_type(path)[0] or 'application/octet-stream')

    def probe(self, *, drain_events=None) -> dict:
        """Take dismissing input and redraw without advancing the simulation.

        Used while a blocking card holds the rollout: the page still receives
        keys and overlay clicks, so the card can be closed, but no simulation
        tick is stepped and nothing is captured for the model.
        """
        payloads = drain_events() if drain_events is not None else []
        return self.page.evaluate(
            '(events) => window.gameBackend.probe(events)', payloads
        )

    def render_chunk(
        self,
        chunk: GameChunkPacket,
        *,
        duration_s: float = 0.0,
        drain_events=None,
    ) -> GameChunkPacket:
        """Simulate and capture one chunk, optionally stretched to ``duration_s``.

        Each model frame is its own Playwright round-trip so ``drain_events`` can
        inject keys that arrived since the previous frame. Sleeps after each
        frame so the chunk's wall time tracks the last generate; a slow frame
        simply skips the next sleep rather than drifting the later deadlines.
        """
        started = time.perf_counter()
        evaluate_s = 0.0
        decode_s = 0.0
        frames: list[GameFramePacket] = []
        frame_count = len(chunk.frames)
        for index, packet in enumerate(chunk.frames):
            payloads = drain_events() if drain_events is not None else []
            eval_started = time.perf_counter()
            captured = self.page.evaluate(
                '(args) => window.gameBackend.advanceToFrame(args)',
                {
                    'events': payloads,
                    'targetTick': packet.simulation_tick,
                },
            )
            evaluate_s += time.perf_counter() - eval_started
            decode_started = time.perf_counter()
            render = _decode_data_url(captured['ui'])
            frames.append(
                replace(
                    packet,
                    ui_frame=render,
                    ui_state=captured.get('state'),
                    condition_frame=render if packet.capture_condition else None,
                )
            )
            decode_s += time.perf_counter() - decode_started
            _sleep_until_frame_deadline(
                started_at=started,
                frame_index=index,
                frame_count=frame_count,
                duration_s=duration_s,
            )
        self.last_evaluate_s = evaluate_s
        self.last_decode_s = decode_s
        return replace(chunk, frames=tuple(frames))

    def close(self):
        # Each stage runs even if an earlier one fails, so a browser that died
        # badly cannot leak the driver process or the vendor directory.
        try:
            if self._browser is not None:
                self._browser.close()
                self._browser = None
        finally:
            try:
                self._playwright.stop()
            finally:
                vendor_dir = self._vendor_dir
                self._vendor_dir = None
                if vendor_dir is not None:
                    vendor_dir.cleanup()


class GameCaptureWorker:
    """Capture game chunks on one dedicated thread.

    Playwright's synchronous API is bound to the thread that started its driver,
    so the browser is launched, evaluated and closed only from this worker.
    ``submit()`` is the non-blocking primitive the pipelined session uses to
    render chunk N while the model generates chunk N-1; ``capture()`` waits.
    ``duration_s`` stretches that render across the last generate so the
    in-flight sim is the window that eats keys.
    """

    def __init__(
        self,
        slug: str,
        *,
        width: int = DEFAULT_CAPTURE_WIDTH,
        height: int = DEFAULT_CAPTURE_HEIGHT,
        games_root: Path | None = None,
    ) -> None:
        self.slug = slug
        self.games_root = games_root
        self.width = width
        self.height = height
        self.last_evaluate_s = 0.0
        self.last_decode_s = 0.0
        self.webgl_renderer = ''
        self._renderer: GameRenderer | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='game2v-render',
        )

    def submit(
        self,
        chunk: GameChunkPacket,
        *,
        duration_s: float = 0.0,
        drain_events=None,
    ) -> Future:
        """Queue one chunk capture and return immediately."""
        return self._executor.submit(
            self._render_chunk, chunk, duration_s, drain_events
        )

    def capture(
        self,
        chunk: GameChunkPacket,
        *,
        duration_s: float = 0.0,
        drain_events=None,
    ) -> GameChunkPacket:
        """Capture one chunk and wait until its frames are decoded."""
        return self.submit(
            chunk, duration_s=duration_s, drain_events=drain_events
        ).result()

    def _render_chunk(
        self,
        chunk: GameChunkPacket,
        duration_s: float = 0.0,
        drain_events=None,
    ) -> GameChunkPacket:
        if self._renderer is None:
            self._renderer = GameRenderer(
                self.slug, width=self.width, height=self.height,
                # Only when set, so stand-ins built for the original signature still work.
                **({'games_root': self.games_root} if self.games_root is not None else {}),
            )
            self.webgl_renderer = getattr(self._renderer, 'webgl_renderer', '')
        rendered = self._renderer.render_chunk(
            chunk, duration_s=duration_s, drain_events=drain_events
        )
        self.last_evaluate_s = self._renderer.last_evaluate_s
        self.last_decode_s = self._renderer.last_decode_s
        return rendered

    def probe(self, *, drain_events=None) -> dict:
        """Poll the live page without advancing its simulation.

        Used while a client card holds the rollout: the page receives whatever
        input has arrived and redraws, so the card can be dismissed, but no
        simulation tick is stepped and nothing is captured for the model. Runs
        on the capture thread, which owns the browser.
        """
        return self._executor.submit(
            self._probe, drain_events
        ).result()

    def _probe(self, drain_events) -> dict:
        if self._renderer is None:
            return {}
        return self._renderer.probe(drain_events=drain_events)

    def close(self) -> None:
        """Close the browser on its owning thread, then stop the worker."""
        try:
            self._executor.submit(self._close_renderer).result()
        except RuntimeError:
            # Already shut down; the renderer went with it.
            pass
        finally:
            self._executor.shutdown(wait=True)

    def _close_renderer(self) -> None:
        renderer = self._renderer
        self._renderer = None
        if renderer is not None:
            renderer.close()


def _chunk_capture_request(chunk: GameChunkPacket) -> dict:
    """Tick list for one chunk. Live keys are injected per frame, not here."""
    return {
        'events': [],
        'frames': [
            {'targetTick': packet.simulation_tick}
            for packet in chunk.frames
        ],
    }


def _sleep_until_frame_deadline(
    *,
    started_at: float,
    frame_index: int,
    frame_count: int,
    duration_s: float,
    now=time.perf_counter,
    sleep=time.sleep,
) -> None:
    """Sleep so frame ``frame_index`` finishes at its share of ``duration_s``."""
    if duration_s <= 0 or frame_count <= 0:
        return
    remaining = started_at + duration_s * (frame_index + 1) / frame_count - now()
    if remaining > 0:
        sleep(remaining)


def _expand_captured_frames(
    chunk: GameChunkPacket,
    captured: list[dict],
) -> tuple[GameFramePacket, ...]:
    """Attach one decoded render to every frame packet in the chunk.

    Each frame carries the render as ``ui_frame`` plus the optional HUD snapshot
    from ``frame['state']``. Latent-aligned frames reference that same array as
    ``condition_frame``, so the model and the client see one render rather than
    two independently rasterized ones.
    """
    if len(captured) != len(chunk.frames):
        raise RuntimeError(
            f'Game backend returned {len(captured)} frames for a '
            f'{len(chunk.frames)}-frame chunk.'
        )
    frames: list[GameFramePacket] = []
    for packet, frame in zip(chunk.frames, captured):
        render = _decode_data_url(frame['ui'])
        frames.append(
            replace(
                packet,
                ui_frame=render,
                ui_state=frame.get('state'),
                condition_frame=render if packet.capture_condition else None,
            )
        )
    return tuple(frames)


def _decode_data_url(data_url: str) -> np.ndarray:
    payload = data_url.split(',', 1)[1]
    return np.array(Image.open(io.BytesIO(base64.b64decode(payload))).convert('RGB'))
