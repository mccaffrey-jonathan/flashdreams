# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# Modified by Vast AI for EdenLive.
# SPDX-License-Identifier: Apache-2.0
# Moved for FlashDreams v0.2.0 from the EdenLive fork's
# flashdreams/runtime_v2/serving/extensions.py into the integration, unchanged,
# so core keeps no application routes.

"""Application-owned HTTP routes mounted on the WebRTC browser server.

An application registers a named extension and the server serves it under
``/ext/<name>/...``. Lookup happens per request, so an extension registered
after the server has started is still reachable: the aiohttp router is frozen
by then, but this table is not. The browser client loads every registered
``client_script`` from ``/extensions.json``, which keeps application UI out of
the shared page.

The routes carry no authentication of their own. Anything an extension exposes
is reachable by whoever can reach the server port.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aiohttp import web

ExtensionHandler = Callable[[web.Request, str], Awaitable[web.StreamResponse]]
"""Coroutine receiving the request and the path below ``/ext/<name>/``."""

_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass(frozen=True, slots=True)
class HttpExtension:
    """One registered extension."""

    handler: ExtensionHandler
    """Serves every request below ``/ext/<name>/``."""

    client_script: str | None = None
    """Path below ``/ext/<name>/`` the browser loads as a module, if any."""


_lock = threading.Lock()
_extensions: dict[str, HttpExtension] = {}


def register_http_extension(
    name: str,
    handler: ExtensionHandler,
    *,
    client_script: str | None = None,
) -> None:
    """Serve ``handler`` under ``/ext/<name>/``, replacing an earlier registration."""
    if not _NAME.fullmatch(name):
        raise ValueError(f"Extension name must match {_NAME.pattern}: {name!r}")
    if client_script is not None and (
        not client_script or client_script.startswith("/") or ".." in client_script
    ):
        raise ValueError(
            f"Extension client script must be a relative path: {client_script!r}"
        )
    with _lock:
        _extensions[name] = HttpExtension(handler=handler, client_script=client_script)


def unregister_http_extension(name: str) -> None:
    """Stop serving ``name``. Unknown names are ignored."""
    with _lock:
        _extensions.pop(name, None)


def _lookup(name: str) -> HttpExtension | None:
    with _lock:
        return _extensions.get(name)


async def dispatch_extension(request: web.Request) -> web.StreamResponse:
    """Route ``/ext/<name>/<tail>`` to the extension registered as ``name``."""
    extension = _lookup(request.match_info["name"])
    if extension is None:
        raise web.HTTPNotFound(reason="No such extension.")
    return await extension.handler(request, request.match_info["tail"])


async def serve_extension_manifest(_: web.Request) -> web.Response:
    """List the browser scripts of every registered extension."""
    with _lock:
        scripts = [
            f"/ext/{name}/{extension.client_script}"
            for name, extension in sorted(_extensions.items())
            if extension.client_script is not None
        ]
    return web.json_response({"scripts": scripts})


__all__ = [
    "ExtensionHandler",
    "HttpExtension",
    "dispatch_extension",
    "register_http_extension",
    "serve_extension_manifest",
    "unregister_http_extension",
]
