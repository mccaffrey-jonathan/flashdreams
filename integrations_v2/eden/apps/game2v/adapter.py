"""Eden game-to-video application entry point."""

import dataclasses
import torch
from flashdreams.api_v2.application import IApplication
from flashdreams.infra.config import derive_config
from eden.apps.game2v.application import Game2VApplication
from eden.apps.game2v.defaults import Cam2VApplicationDefaults
from eden.apps.game2v.game14b import (
    generate_game14b_step,
    make_game14b_control_stream,
    resolve_game14b_conditioning,
)
from eden.apps.game2v.game5b import (
    generate_game5b_step,
    make_game5b_control_stream,
    resolve_game5b_conditioning,
)
from eden.game14b_config import (
    PIPELINE_GAME2V_EDEN_WORLD_V2_14B,
    PIPELINE_GAME2V_EDEN_WORLD_V2_14B_TAEHV,
)
from eden.game5b_config import (
    GAME2V_5B_FPS,
    GAME2V_5B_PIXEL_HEIGHT,
    GAME2V_5B_PIXEL_WIDTH,
    PIPELINE_GAME2V_EDEN_WORLD_V2_5B,
    PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV,
)

# Only the two TAEHV applications are shipped. The non-TAEHV pipeline configs
# and defaults below are not registered as applications; they are the bases the
# TAEHV variants derive from, so they stay.

# Proxy-DINO conditioned AR 14B: the game's render conditions generation
# through DINOv3 instead of camera poses, so the first frame, prompt, step
# input and condition encoder all come from this model's own wiring.
GAME2V_EDEN_WORLD_V2_14B_DEFAULTS = dataclasses.replace(
    Cam2VApplicationDefaults(
        pipeline_config=derive_config(PIPELINE_GAME2V_EDEN_WORLD_V2_14B),
        input_resolver=resolve_game14b_conditioning,
        total_blocks=(
            (PIPELINE_GAME2V_EDEN_WORLD_V2_14B.diffusion_model.max_latent_frames + 2)
            // 3
            if PIPELINE_GAME2V_EDEN_WORLD_V2_14B.diffusion_model.max_latent_frames
            is not None
            else 20
        ),
        pixel_width=832,
        pixel_height=480,
        first_frame_dtype=torch.float32,
        first_frame_interpolation="linear",
        fps=16,
        log_model_timing=True,
        install_hint="Install the Eden integration: pip install flashdreams-eden.",
        # The first frame comes from the per-game cover, so no example data.
        input_defaults={"example_data": False, "example_idx": 0},
        generate_step=generate_game14b_step,
        control_stream_factory=make_game14b_control_stream,
    ),
    fixed_camera=True,
)


GAME2V_EDEN_WORLD_V2_14B_TAEHV_DEFAULTS = dataclasses.replace(
    GAME2V_EDEN_WORLD_V2_14B_DEFAULTS,
    pipeline_config=derive_config(PIPELINE_GAME2V_EDEN_WORLD_V2_14B_TAEHV),
)


def create_app_v2_14b_taehv() -> IApplication:
    """The 14B shell with the LightTAE decoder instead of the Wan VAE."""
    return Game2VApplication(defaults=GAME2V_EDEN_WORLD_V2_14B_TAEHV_DEFAULTS)


# Proxy-DINO conditioned 5B. Same game-conditioned shell as the 14B,
# but Wan 2.2 TI2V-5B rather than Wan 2.1 14B I2V, which moves three of the
# defaults: the frame size is 704x1280 (the only size whose token count shards
# across 8 ranks under the 16x VAE -- see eden.game5b_config), the frame rate is
# 24 rather than 16, and the control stream resamples onto a /16 grid.
GAME2V_EDEN_WORLD_V2_5B_DEFAULTS = dataclasses.replace(
    GAME2V_EDEN_WORLD_V2_14B_DEFAULTS,
    pipeline_config=derive_config(PIPELINE_GAME2V_EDEN_WORLD_V2_5B),
    input_resolver=resolve_game5b_conditioning,
    pixel_width=GAME2V_5B_PIXEL_WIDTH,
    pixel_height=GAME2V_5B_PIXEL_HEIGHT,
    fps=GAME2V_5B_FPS,
    total_blocks=(
        (PIPELINE_GAME2V_EDEN_WORLD_V2_5B.diffusion_model.max_latent_frames + 2) // 3
        if PIPELINE_GAME2V_EDEN_WORLD_V2_5B.diffusion_model.max_latent_frames
        is not None
        else 20
    ),
    generate_step=generate_game5b_step,
    control_stream_factory=make_game5b_control_stream,
)


GAME2V_EDEN_WORLD_V2_5B_TAEHV_DEFAULTS = dataclasses.replace(
    GAME2V_EDEN_WORLD_V2_5B_DEFAULTS,
    pipeline_config=derive_config(PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV),
)


def create_app_v2_5b_taehv() -> IApplication:
    """The 5B shell with the LightTAE Wan 2.2 decoder instead of the Wan VAE."""
    return Game2VApplication(defaults=GAME2V_EDEN_WORLD_V2_5B_TAEHV_DEFAULTS)
