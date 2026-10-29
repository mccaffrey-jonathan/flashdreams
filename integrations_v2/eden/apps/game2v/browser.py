# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``eden-webrtc`` client-window mode: Game2V's browser experience.

Registered under the ``flashdreams.client_windows_v2`` entry-point group, and
picked by :meth:`Game2VApplication.resolve_client_window_mode` for ``--mode
webrtc``. This module stays import-light because the runtime discovers every
mode on each launch; the server lives in :mod:`eden.apps.game2v.browser_server`.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, cast

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.client_window_factory import ClientWindowMode

if TYPE_CHECKING:
    from .browser_server import EdenWebRTCClientWindow

EDEN_WEBRTC_MODE = "eden-webrtc"


class EdenWebRTCMode(ClientWindowMode):
    """Stream Game2V to a browser with the game render, HUD and game panel.

    Takes the built-in ``webrtc`` mode's ``--host`` (default ``127.0.0.1``) and
    ``--port``; every mode's arguments share one parser, so they are not
    declared twice.
    """

    name = EDEN_WEBRTC_MODE

    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        from .browser_server import EdenWebRTCClientWindow

        return EdenWebRTCClientWindow(host=parsed_args.host, port=parsed_args.port)

    def starting(self, client_window: IClientWindow) -> str | None:
        server = cast("EdenWebRTCClientWindow", client_window).server
        return f"Open {server.url} in a browser."


def eden_webrtc_mode_installed() -> bool:
    """Whether the ``eden-webrtc`` entry point is registered in this environment."""
    from flashdreams.runtime_v2.client_window_factory import client_window_modes

    return any(mode.name == EDEN_WEBRTC_MODE for mode in client_window_modes())


__all__ = ["EDEN_WEBRTC_MODE", "EdenWebRTCMode", "eden_webrtc_mode_installed"]
