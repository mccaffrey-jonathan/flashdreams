# SPDX-License-Identifier: Apache-2.0
"""Generic Game2V HUD envelope.

The envelope forwards ``state`` verbatim; the only field the runtime reads is
the blocking-card convention shared by the vendored HUD transports, which
decides when a rollout pauses.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

GAME_UI_MESSAGE_TYPE = "game_ui"
GAME_UI_ACTION_CHANNEL = "game_ui"

RUNNING_PLAY_STATE = "RUNNING"
"""Value of ``state.state`` while a game is playable rather than showing a card."""

OPEN_PLAY_STATES = ("", RUNNING_PLAY_STATE, "READY")
"""Play states that are not a blocking card.

``READY`` is a game waiting on its starting line for the player to begin. That
is deliberately not a pause: the rollout keeps generating so the model frame
rate can settle before play starts, which is the whole reason a game offers the
state. Only an outcome -- or an explicit overlay flag -- holds generation.
"""

DIALOG_FLAG_FIELDS = ("helpOpen", "paused")
"""Snapshot flags that raise a modal card while the play state stays RUNNING.

``paused`` is an in-game P/Esc pause. The game keeps ``state`` at RUNNING, so
the flag is what holds generation; ``probe()`` routes the resume key to the
game's ``handleUiAction`` because no simulation tick runs while held.
"""


def is_dialog_open(state: Mapping[str, Any] | None) -> bool:
    """Return whether one HUD snapshot puts a blocking card on the client.

    Every vendored transport reports an outcome as ``state`` leaving the
    :data:`OPEN_PLAY_STATES` (win, loss, or any other terminal label) and
    reports a paused overlay through a boolean flag such as ``helpOpen``. A game
    that publishes neither never pauses the rollout.
    """
    if not isinstance(state, Mapping):
        return False
    play_state = state.get("state")
    if isinstance(play_state, str) and play_state not in OPEN_PLAY_STATES:
        return True
    return any(bool(state.get(field)) for field in DIALOG_FLAG_FIELDS)


def envelope(
    slug: str,
    frame_id: int,
    state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Wrap one game snapshot for the client overlay.

    Args:
        slug: Registered game identifier; the client loads ``hud/`` from it.
        frame_id: Presented game frame id, so the overlay can drop stale mail.
        state: JSON-serializable snapshot produced by the game's transport.
    """
    return {
        "type": GAME_UI_MESSAGE_TYPE,
        "slug": slug,
        "frame_id": int(frame_id),
        "state": state,
    }
