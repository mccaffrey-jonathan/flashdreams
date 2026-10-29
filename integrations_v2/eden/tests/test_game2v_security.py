# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Game2V HTTP surface guards: paths, origins, no HTTP import."""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from eden.apps.game2v.panel import handle_game2v, safe_relative_path

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    "rel",
    [
        "../secret",
        "sub/../../secret",
        "..\\secret",
        "\\\\host\\share\\file",
        "//host/share/file",
        "/etc/passwd",
        "C:/Windows/win.ini",
        "c:secret",
        "a\x00b",
        "",
    ],
)
def test_unsafe_relative_paths_are_rejected(rel: str) -> None:
    assert safe_relative_path(rel) is None


@pytest.mark.parametrize(
    "rel", ["cover.png", "assets/model.glb", "runs/step1/a/index.html"]
)
def test_plain_relative_paths_pass(rel: str) -> None:
    assert safe_relative_path(rel) is not None


def _run(coro_fn) -> None:
    app = web.Application()
    app.router.add_route(
        "*",
        "/ext/{name}/{tail:.*}",
        lambda request: handle_game2v(request, request.match_info["tail"]),
    )

    async def main() -> None:
        async with TestClient(TestServer(app)) as client:
            await coro_fn(client)

    asyncio.run(main())


def test_state_changes_need_same_origin_json() -> None:
    async def body(client: TestClient) -> None:
        # A "simple" cross-site POST (no preflight) is refused before any work.
        simple = await client.post(
            "/ext/game2v/home", data="{}", headers={"Content-Type": "text/plain"}
        )
        assert simple.status == 415
        foreign = await client.post(
            "/ext/game2v/switch",
            json={"slug": "ember-den"},
            headers={"Origin": "https://evil.example"},
        )
        assert foreign.status == 403

    _run(body)


def test_http_import_is_not_exposed() -> None:
    async def body(client: TestClient) -> None:
        response = await client.post(
            "/ext/game2v/import",
            json={"project": "C:/", "slug": "x", "cover": "C:/secret"},
        )
        assert response.status == 404

    _run(body)
