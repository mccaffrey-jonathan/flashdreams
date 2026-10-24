# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proxy-DINO conditioned AR 14B model for the Eden Game2V app.

The game's rendered frames condition generation through a frozen DINOv3
encoder whose projected features are channel-concatenated into the Wan 2.1
14B I2V DiT, matching the AR teacher-forced training recipe.
"""

from .dino import (
    ProxyDinoControlEncoder,
    ProxyDinoEncoder,
    ProxyDinoEncoderConfig,
)
from .encoder import (
    Game14BCtrlEncoder,
    Game14BCtrlEncoderConfig,
    Game14BCtrlInput,
)
from .network import (
    GAME14B_CONTROL_DIM,
    GAME14B_I2V_CHANNELS,
    Game14BDiTNetwork,
    Game14BDiTNetworkConfig,
)
from .pipeline import (
    Game14BInferencePipeline,
    Game14BInferencePipelineConfig,
)
from .transformer import (
    Game14BTransformer,
    Game14BTransformerConfig,
    I2VProxyDinoCtrl,
)

__all__ = [
    "GAME14B_CONTROL_DIM",
    "GAME14B_I2V_CHANNELS",
    "Game14BCtrlEncoder",
    "Game14BCtrlEncoderConfig",
    "Game14BCtrlInput",
    "Game14BDiTNetwork",
    "Game14BDiTNetworkConfig",
    "Game14BInferencePipeline",
    "Game14BInferencePipelineConfig",
    "Game14BTransformer",
    "Game14BTransformerConfig",
    "I2VProxyDinoCtrl",
    "ProxyDinoControlEncoder",
    "ProxyDinoEncoder",
    "ProxyDinoEncoderConfig",
]
