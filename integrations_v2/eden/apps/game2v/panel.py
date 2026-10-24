# SPDX-License-Identifier: Apache-2.0
"""Rank-zero HTTP panel for listing, previewing, importing and switching games.

Mounted under ``/ext/game2v/``. The aiohttp router is frozen by the time a
session starts, so registration goes through
:func:`flashdreams.runtime_v2.serving.extensions.register_http_extension`.
The browser loads ``panel.js`` from ``/extensions.json``.
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from aiohttp import web
from loguru import logger

from flashdreams.runtime_v2.serving.extensions import (
    register_http_extension,
    unregister_http_extension,
)

from .game_renderer import STANDALONE_HOST, registered_game_slugs
from .import_game import GameImportError, _SLUG, games_root, import_game
from .session import browser_metrics_snapshot, request_game_switch, request_home

EXTENSION_NAME = "game2v"
_STATIC = Path(__file__).resolve().parent / "static"
_HOST = Path(__file__).resolve().parents[4] / "games" / "host"

_bound_session: Any | None = None


def _session_state(session: Any) -> Any:
    return session.model_loop.state


def _root() -> Path:
    return games_root()


def _registry() -> dict[str, Any]:
    path = _root() / "registry.json"
    if not path.is_file():
        return {"games": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _public_games(current: str) -> dict[str, Any]:
    games = []
    for entry in _registry().get("games", []):
        slug = entry.get("slug")
        if not isinstance(slug, str):
            continue
        source = entry.get("source") if isinstance(entry.get("source"), dict) else {}
        games.append(
            {
                "slug": slug,
                "title": entry.get("title") or slug,
                "description": entry.get("description") or "",
                "host": entry.get("host") or "ported",
                "controls": list(entry.get("controls") or ()),
                "cover": f"files/{slug}/{entry['cover']}" if entry.get("cover") else None,
                "source": source.get("skill") or "",
                "restart": list(entry.get("restart_keys") or ()),
                "current": slug == current,
            }
        )
    return {"current": current, "games": games}


def _rewrite_play_page(slug: str, html: str, entry: dict[str, Any]) -> str:
    """Serve a live-preview page. Capture still uses the Playwright rewrite."""
    if entry.get("host") == STANDALONE_HOST:
        # The standalone game drives its own requestAnimationFrame loop.
        # standalone.js replaces that clock for frame-by-frame capture, so it
        # must only be injected by game_renderer.standalone_page().
        return html
    return html.replace(
        '<script type="module" src="/shared/host/play-page.js"></script>',
        f'<script>window.gameSlug={json.dumps(slug)}</script>'
        '<script type="module" src="/ext/game2v/host/play-page.js"></script>',
    )


def _content_type(path: Path) -> str:
    if path.suffix in {".js", ".mjs"}:
        return "text/javascript"
    if path.suffix == ".html":
        return "text/html"
    if path.suffix == ".css":
        return "text/css"
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _safe_file(root: Path, rel: str) -> Path | None:
    path = (root / rel).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    return path


async def handle_game2v(request: web.Request, tail: str) -> web.StreamResponse:
    """Dispatch one ``/ext/game2v/<tail>`` request."""
    session = _bound_session
    if tail in ("panel.js", "static/panel.js"):
        return web.Response(
            body=(_STATIC / "panel.js").read_bytes(),
            content_type="text/javascript",
        )
    if tail.startswith("host/"):
        path = _safe_file(_HOST, tail[len("host/") :])
        if path is None:
            raise web.HTTPNotFound()
        return web.Response(body=path.read_bytes(), content_type=_content_type(path))
    if tail == "games" and request.method == "GET":
        current = _session_state(session).config.game_slug if session is not None else ""
        body = _public_games(current)
        body["started"] = _started(session)
        return web.json_response(body)
    if tail == "metrics" and request.method == "GET":
        snapshot = browser_metrics_snapshot(_session_state(session)) if session is not None else None
        if snapshot is None:
            return web.json_response({"model": None})
        sampled_at = float(snapshot.pop("sampled_at"))
        snapshot["age_s"] = max(0.0, time.monotonic() - sampled_at)
        return web.json_response({"model": snapshot})
    if tail == "switch" and request.method == "POST":
        return await _handle_switch(request, session)
    if tail == "home" and request.method == "POST":
        return _handle_home(session)
    if tail == "import" and request.method == "POST":
        return await _handle_import(request, session)
    if tail.startswith("play/"):
        return _handle_play(tail[len("play/") :])
    if tail.startswith("files/"):
        return _handle_files(tail[len("files/") :])
    raise web.HTTPNotFound()


def _started(session: Any) -> bool:
    """Whether rendering has been released (always, without ``--wait-for-start``).

    A pending return to the home screen already reads as "not started", before
    the model thread has consumed it: the page asks this on load to decide
    which view to open, and a reload right after going home belongs on the
    home screen rather than on the session it just left.
    """
    if session is None:
        return False
    state = _session_state(session)
    if not getattr(state.config, "wait_for_start", False):
        return True
    return state.start_requested and not getattr(state, "pending_home", False)


async def _handle_switch(request: web.Request, session: Any) -> web.Response:
    if session is None:
        raise web.HTTPConflict(reason="No Game2V session.")
    try:
        payload = await request.json()
    except (json.JSONDecodeError, web.HTTPException) as error:
        raise web.HTTPBadRequest(reason="Expected a JSON object.") from error
    slug = payload.get("slug") if isinstance(payload, dict) else None
    if not isinstance(slug, str) or slug not in registered_game_slugs():
        raise web.HTTPBadRequest(reason=f"Unknown game: {slug!r}")
    restart = bool(payload.get("restart")) if isinstance(payload, dict) else False
    request_game_switch(_session_state(session), slug, restart=restart)
    return web.json_response(_public_games(slug))


def _handle_home(session: Any) -> web.Response:
    """Stop rendering and wait for the next game to be picked."""
    if session is None:
        raise web.HTTPConflict(reason="No Game2V session.")
    request_home(_session_state(session))
    return web.json_response({"started": False})


async def _handle_import(request: web.Request, session: Any) -> web.Response:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, web.HTTPException) as error:
        raise web.HTTPBadRequest(reason="Expected a JSON object.") from error
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(reason="Expected a JSON object.")
    project = payload.get("project")
    slug = payload.get("slug")
    if not isinstance(project, str) or not project.strip():
        raise web.HTTPBadRequest(reason="project path is required.")
    if not isinstance(slug, str) or not _SLUG.fullmatch(slug):
        raise web.HTTPBadRequest(reason="slug must be lowercase words joined by '-'.")
    cover = payload.get("cover")
    cover_path = Path(cover) if isinstance(cover, str) and cover else None

    def _run() -> dict[str, Any]:
        # Skip Playwright cover capture in the web path; a missing cover.png
        # still lets the game register, and a later import can replace it.
        return import_game(
            Path(project).expanduser(),
            slug,
            cover=cover_path,
            allow_incomplete=bool(payload.get("allow_incomplete")),
            replace=bool(payload.get("replace")),
            capture=lambda *_args, **_kwargs: None,
        )

    try:
        entry = await asyncio.to_thread(_run)
    except GameImportError as error:
        raise web.HTTPBadRequest(reason=str(error)) from error
    except FileNotFoundError as error:
        raise web.HTTPBadRequest(reason=str(error)) from error
    except Exception as error:
        logger.opt(exception=True).error("Game2V import failed.")
        raise web.HTTPInternalServerError(reason=str(error)) from error
    current = _session_state(session).config.game_slug if session is not None else slug
    body = _public_games(current)
    body["imported"] = entry["slug"]
    return web.json_response(body)


def _handle_play(slug: str) -> web.Response:
    slug = unquote(slug).strip("/")
    if not _SLUG.fullmatch(slug):
        raise web.HTTPNotFound()
    entry = next(
        (game for game in _registry().get("games", []) if game.get("slug") == slug),
        None,
    )
    if entry is None:
        raise web.HTTPNotFound()
    index = _root() / slug / "index.html"
    if not index.is_file():
        raise web.HTTPNotFound()
    html = _rewrite_play_page(slug, index.read_text(encoding="utf-8"), entry)
    # Preview is served from /ext/game2v/play/<slug>, so origin-relative
    # /<slug>/... would 404. Map those onto /game/<slug>/... which the
    # WebRTC server already exposes. /vendor and /host stay.
    html = html.replace(f"/{slug}/", f"/game/{slug}/")
    # Must run after the slug rewrite above, which would otherwise double the
    # /game/ prefix in the mapped target.
    html = _map_game_assets(slug, html)
    # A game is embedded in a responsive 16:9 preview iframe.  The original
    # game pages are also used by the full-size recorder and several of them
    # keep a fixed 1280px ``#view-big`` plus a side rail.  Leaving that shell
    # in the iframe makes the renderer's drawing buffer use the iframe size
    # while the canvas CSS box keeps the old fixed size, which stretches the
    # proxy image.  Give the preview its own viewport contract: the game
    # surface fills the iframe, the side rail is removed, and both CSS and
    # the runtime see the same dimensions.  This is injected before the game
    # module runs, so its initial ``layout(innerWidth, innerHeight)`` is
    # already correct.
    preview_css = """<style data-eden-proxy-preview>
html, body { width: 100% !important; height: 100% !important; margin: 0 !important; overflow: hidden !important; }
body { display: block !important; position: relative !important; min-width: 0 !important; }
#view-big { position: absolute !important; inset: 0 !important; width: 100% !important; height: 100% !important; min-width: 0 !important; min-height: 0 !important; overflow: hidden !important; background: #000 !important; }
/* Proxy frames are produced at 16:9. Keep the game viewport at that ratio;
   the iframe may be wider or taller, so the unused area becomes letterbox
   space instead of stretching the scene. */
#app { position: absolute !important; inset: auto !important; left: 50% !important; top: 50% !important; width: min(100%, 177.7777778vh) !important; height: min(100%, 56.25vw) !important; aspect-ratio: 16 / 9 !important; transform: translate(-50%, -50%) !important; min-width: 0 !important; min-height: 0 !important; overflow: hidden !important; background: #000 !important; }
/* The canvas is sized by the aspect-lock script below.  Do not let a game's
   original ``width:100%;height:100%`` rule stretch a fixed drawing buffer. */
#app canvas { display: block !important; position: absolute !important; left: 50% !important; top: 50% !important; max-width: none !important; max-height: none !important; transform: translate(-50%, -50%) !important; object-fit: fill !important; }
#right { display: none !important; }
#view-small, #desc-panel, #record-bar { display: none !important; }
</style>"""
    # Runtime games resize their drawing buffer at different times: some do it
    # during module startup, while others do it after a ResizeObserver fires.
    # Keep the drawing buffer untouched and derive only its CSS box from the
    # current container.  This gives every game the same contain/letterbox
    # contract, including standalone exports and games that write inline CSS.
    preview_script = """<script data-eden-proxy-aspect-lock>
(function () {
  function install() {
    var host = document.querySelector('#app') || document.body;
    var canvas = host && (host.querySelector('canvas') || document.querySelector('canvas'));
    if (!host || !canvas) return false;
    var apply = function () {
      var bufferWidth = Number(canvas.width);
      var bufferHeight = Number(canvas.height);
      var box = host.getBoundingClientRect();
      if (!(bufferWidth > 0 && bufferHeight > 0 && box.width > 0 && box.height > 0)) return;
      var scale = Math.min(box.width / bufferWidth, box.height / bufferHeight);
      var width = Math.max(1, Math.round(bufferWidth * scale * 100) / 100);
      var height = Math.max(1, Math.round(bufferHeight * scale * 100) / 100);
      canvas.style.setProperty('width', width + 'px', 'important');
      canvas.style.setProperty('height', height + 'px', 'important');
    };
    var resize = typeof ResizeObserver === 'function' ? new ResizeObserver(apply) : null;
    if (resize) resize.observe(host);
    new MutationObserver(apply).observe(canvas, {attributes: true, attributeFilter: ['width', 'height']});
    window.addEventListener('resize', apply, {passive: true});
    apply();
    requestAnimationFrame(apply);
    return true;
  }
  function start() {
    if (install()) return;
    var observer = new MutationObserver(function () { if (install()) observer.disconnect(); });
    observer.observe(document.documentElement, {childList: true, subtree: true});
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, {once: true});
  else start();
})();
</script>"""
    # Relative refs (./style.css) must resolve inside the game, not /ext/.
    html = html.replace("<head>", f'<head>{preview_css}<base href="/game/{slug}/">', 1)
    # Run after the game's own module has had a chance to create #app canvas;
    # DOMContentLoaded/MutationObserver in the script handles both ordering
    # variants without changing the game source.
    html = html.replace("</body>", f"{preview_script}</body>", 1)
    # The authoring recorder only exists on the skill's dev server.
    html = html.replace('<link rel="stylesheet" href="/shared/recorder/recorder.css">', "")
    return web.Response(text=html, content_type="text/html")


_IMPORTMAP = re.compile(r'<script type="importmap">(.*?)</script>', re.DOTALL)


def _map_game_assets(slug: str, html: str) -> str:
    """Point module imports of ``/assets/`` at this game's own asset tree.

    Every game ships its own ``games/<slug>/assets/`` and imports it as
    ``/assets/...``; capture resolves that per slug in Playwright routing, so
    the live preview does the same through the page's import map.
    """
    mapping = {
        "/assets/": f"/game/{slug}/assets/",
        "/game/_shared/": "/ext/game2v/files/_shared/",
    }
    match = _IMPORTMAP.search(html)
    if match is None:
        tag = f'<script type="importmap">{json.dumps({"imports": mapping})}</script>'
        head = html.find("</head>")
        return html[:head] + tag + html[head:] if head >= 0 else tag + html
    try:
        importmap = json.loads(match.group(1))
    except json.JSONDecodeError:
        return html
    importmap.setdefault("imports", {}).update(mapping)
    tag = f'<script type="importmap">{json.dumps(importmap)}</script>'
    return html[: match.start()] + tag + html[match.end() :]


def _handle_files(rel: str) -> web.Response:
    rel = unquote(rel).lstrip("/")
    root = _root()
    if rel.startswith("host/"):
        path = _safe_file(_HOST, rel[len("host/") :])
    else:
        path = _safe_file(root, rel)
    if path is None:
        raise web.HTTPNotFound()
    return web.Response(body=path.read_bytes(), content_type=_content_type(path))


def register_game2v_panel(session: Any) -> None:
    """Serve the panel for ``session``. Rank zero only."""
    global _bound_session
    _bound_session = session
    register_http_extension(
        EXTENSION_NAME,
        handle_game2v,
        client_script="panel.js",
    )


def unregister_game2v_panel() -> None:
    """Drop the panel when the session closes."""
    global _bound_session
    _bound_session = None
    unregister_http_extension(EXTENSION_NAME)
