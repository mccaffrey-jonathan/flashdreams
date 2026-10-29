# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the Game2V app tests.

The EdenLive games are not part of FlashDreams (``EDEN_GAMES_ROOT`` points at an
EdenLive checkout), so CPU tests run against a small fake registry and tests
that need the real games skip without one.
"""

from __future__ import annotations

import json
import os

import pytest
from PIL import Image

FAKE_GAME_SLUGS = ("ember-den", "iron-colossus", "jade-duel", "warehouse-sokoban")
"""Slugs of the four registered EdenLive games, mirrored by :func:`fake_games_root`."""


@pytest.fixture
def real_games_root():
    """Return the real EdenLive games directory, or skip without ``EDEN_GAMES_ROOT``."""
    if not os.environ.get("EDEN_GAMES_ROOT"):
        pytest.skip("needs the EdenLive games: set EDEN_GAMES_ROOT to realtime/games")
    from eden.apps.game2v.game_renderer import games_root

    return games_root()


@pytest.fixture
def fake_games_root(tmp_path, monkeypatch):
    """Point ``EDEN_GAMES_ROOT`` at a registry of four stub games."""
    root = tmp_path / "games"
    for slug in FAKE_GAME_SLUGS:
        (root / slug).mkdir(parents=True)
        Image.new("RGB", (32, 18), (40, 50, 60)).save(root / slug / "cover.png")
        (root / slug / "index.html").write_text(
            "<html><head></head><body></body></html>"
        )
    (root / "_shared").mkdir()
    (root / "_shared" / "autopilot.js").write_text("// stub autopilot\n")
    registry = {
        "games": [
            {
                "slug": slug,
                "title": slug,
                "prompt": f"A detailed scene for {slug}, long enough to be a prompt.",
                "cover": "cover.png",
                "host": "standalone",
                "restart_keys": ["enter"],
            }
            for slug in FAKE_GAME_SLUGS
        ]
    }
    (root / "registry.json").write_text(json.dumps(registry))
    monkeypatch.setenv("EDEN_GAMES_ROOT", str(root))
    return root
