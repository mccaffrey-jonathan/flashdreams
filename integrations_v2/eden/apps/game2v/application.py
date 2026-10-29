# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

# recorded-proxy replay, proxy/side-by-side recording and a per-chunk log.

"""Reusable camera-to-video application on the FlashDreams v2 API."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.infra.config import derive_config
from flashdreams.infra.postprocess import (
    VideoPostprocessChainConfig,
    VideoPostprocessStream,
    VideoSpec,
)
from flashdreams.plugins.registry import discover_postprocess_presets
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .defaults import Cam2VApplicationDefaults, Cam2VConditioning
from .game_renderer import (
    GamesRootError,
    default_autopilot_script,
    games_root,
    registered_game_slugs,
)
from .postprocess_comparison import comparison_output_spec
from .recording import load_chunk_prompts, recording_frame_count
from .session import Game2VSession, Game2VSessionConfig


class Game2VApplication(IApplication):
    """Reusable interactive camera-to-video application.

    The shared class owns command-line parsing, pipeline lifetime, session
    validation, and model-generation-loop construction. A concrete model
    integration contributes a runner config and an input resolver through
    :class:`Cam2VApplicationDefaults`.
    """

    def __init__(self, *, defaults: Cam2VApplicationDefaults) -> None:
        self.defaults = defaults
        self._pipeline_config = defaults.pipeline_config
        self._device = defaults.device
        self._total_blocks = defaults.total_blocks
        self._warmup_blocks = defaults.warmup_blocks
        self._use_ui = True
        self._game_slug = defaults.game_slug
        self._wait_for_start = False
        self._input_values: dict[str, Any] | None = None
        self._pipeline: Any | None = None
        self._postprocess = VideoPostprocessChainConfig()
        self._postprocess_comparison_ui = False
        # ``None`` until ``--compile`` / ``--no-compile`` says otherwise, which
        # leaves every component on its own config default.
        self._compile: bool | None = None
        self._client_window_mode: str | None = None
        self._autopilot: Path | None = None
        self._proxy_video: Path | None = None
        self._proxy_frames: int | None = None
        self._prompt_schedule: tuple[str, ...] | None = None
        self._record_proxy: Path | None = None
        self._chunk_log: Path | None = None
        self._side_by_side = False

    def resolve_client_window_mode(self, mode: str) -> str:
        """Serve ``webrtc`` through Game2V's own browser window when it is installed.

        The ``eden-webrtc`` mode adds the game render, the HUD overlay, the
        prompt panels and the game panel to FlashDreams' page. Every other mode
        is kept as asked.
        """
        from .browser import EDEN_WEBRTC_MODE, eden_webrtc_mode_installed

        if mode == "webrtc" and eden_webrtc_mode_installed():
            mode = EDEN_WEBRTC_MODE
        self._client_window_mode = mode
        return mode

    @property
    def pipeline_config(self) -> Any:
        """Return the model configuration after command-line overrides."""
        return self._pipeline_config

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse shared camera-to-video inputs without loading the model."""
        parser = argparse.ArgumentParser(
            prog="flashdreams-run-v2 CAM2V_SLUG --",
            description="Generate video from a first frame and keyboard camera input.",
        )
        # Validated against registry.json after parsing, so --help and a replay
        # work without EDEN_GAMES_ROOT.
        try:
            game_help = (
                "One of: " + ", ".join(registered_game_slugs(root=games_root())) + "."
            )
        except GamesRootError as error:
            game_help = str(error)
        parser.add_argument(
            "--game",
            default=self.defaults.game_slug or None,
            help=(
                "EdenLive game rendered as the proxy (default: the first in "
                f"$EDEN_GAMES_ROOT/registry.json). {game_help}"
            ),
        )
        parser.add_argument(
            "--autopilot",
            nargs="?",
            const="",
            default=None,
            metavar="SCRIPT",
            help=(
                "Play the game headless with a rule-based autopilot that reads "
                "<game>/demo.json, so recordings show gameplay. Default script: "
                "$EDEN_GAMES_ROOT/_shared/autopilot.js."
            ),
        )
        replay = parser.add_argument_group(
            "replay",
            "Replace the live game with a recorded proxy video, read exactly as "
            "LightX2V's EdenLive RecordingReader reads it.",
        )
        replay.add_argument(
            "--proxy-video",
            type=Path,
            default=None,
            help=(
                "Recorded proxy video. Requires --first-frame and --prompt (or "
                "--prompts-json). The rollout covers the whole video: "
                "4*((floor(duration*fps)-1)//4)+1 frames."
            ),
        )
        replay.add_argument(
            "--max-frames",
            type=int,
            default=None,
            help="Cap the replay at N frames before rounding down to 4n+1.",
        )
        replay.add_argument(
            "--prompts-json",
            type=Path,
            default=None,
            help=(
                "EdenLive prompts.json. Each chunk's prompt is staticPrompt plus "
                "the dynamicPrompts overlapping its frames; the text encoder "
                "stays resident when the prompt changes. Default: the static "
                "--prompt for every chunk."
            ),
        )
        record = parser.add_argument_group("recording")
        record.add_argument(
            "--record-proxy",
            type=Path,
            default=None,
            help="Write every proxy frame the model consumed to this MP4.",
        )
        record.add_argument(
            "--side-by-side",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Present proxy | generated frames, twice the model width.",
        )
        record.add_argument(
            "--chunk-log",
            type=Path,
            default=None,
            help="Write one JSON line per generated chunk (frames, prompt, timings).",
        )
        parser.add_argument(
            "--wait-for-start",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Render nothing until the web workbench picks a game "
                "(POST /ext/game2v/switch). Default: %(default)s."
            ),
        )
        parser.add_argument(
            "--prompt-event-chunks",
            type=int,
            default=1,
            help=(
                "Generated chunks an event line stays in the prompt, counting "
                "the chunk right after the one that simulated it. The client "
                "panel shows the same text for the same span. Default: %(default)s."
            ),
        )
        parser.add_argument(
            "--dynamic-prompt",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Re-encode the prompt for every chunk from the static scene "
                "description plus the event lines still inside their lifetime, "
                "instead of once at startup. The text encoder stays resident on "
                "rank zero. Only the proxy-DINO game presets (5B and 14B) "
                "support it; other presets reject the flag. Default: %(default)s."
            ),
        )
        input_defaults = self.defaults.input_defaults
        parser.add_argument("--prompt", default=input_defaults.get("prompt", ""))
        parser.add_argument(
            "--prompt-path",
            type=Path,
            default=input_defaults.get("prompt_path"),
        )
        parser.add_argument(
            "--image-path",
            "--first-frame",
            dest="image_path",
            type=Path,
            default=input_defaults.get("image_path"),
            help="First frame; defaults to the game's cover.",
        )
        parser.add_argument(
            "--pose-path",
            type=Path,
            default=input_defaults.get("pose_path"),
        )
        parser.add_argument(
            "--intrinsic-path",
            type=Path,
            default=input_defaults.get("intrinsic_path"),
        )
        parser.add_argument(
            "--world-scale",
            type=float,
            default=input_defaults.get("world_scale"),
        )
        parser.add_argument(
            "--example-data",
            action=argparse.BooleanOptionalAction,
            default=bool(input_defaults.get("example_data", False)),
        )
        parser.add_argument(
            "--example-idx",
            type=int,
            default=int(input_defaults.get("example_idx", 0)),
        )
        parser.add_argument(
            "--device",
            default=self.defaults.device,
            help="Device used by the shared model. Default: %(default)s.",
        )
        parser.add_argument(
            "--total-blocks",
            type=int,
            default=None,
            help=(
                "Autoregressive chunks generated per rollout. Default: "
                f"{self.defaults.total_blocks}, or the whole --proxy-video."
            ),
        )
        parser.add_argument(
            "--warmup-blocks",
            type=int,
            default=self.defaults.warmup_blocks,
            help="Leading chunks excluded from steady-state FPS.",
        )
        parser.add_argument(
            "--ui",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Render the shared camera controls and timing overlay.",
        )
        parser.add_argument(
            "--compile",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=(
                "Compile the model and enable CUDA-graph replay. With "
                "--no-compile every torch.compile and CUDA graph is turned "
                "off — the transformer, the VAE, the decoder and the "
                "condition encoder — so the first chunk arrives without the "
                "compiler's warmup."
            ),
        )
        postprocess_presets = discover_postprocess_presets()
        parser.add_argument(
            "--postprocess-preset",
            choices=("", *postprocess_presets),
            default="",
            help=(
                "Optional generated-video post-processing preset. A configured "
                "preset starts enabled and can be toggled in the UI."
            ),
        )
        parser.add_argument(
            "--postprocess-chunk-size",
            type=int,
            choices=(8, 16),
            default=8,
            help=(
                "Steady post-processing window. Eden uses 8 so its 9/12-frame "
                "model chunks always produce output. Default: %(default)s."
            ),
        )
        parser.add_argument(
            "--postprocess-compile",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Compile post-processors for steady-state performance. Disable "
                "to skip compiler autotuning during development."
            ),
        )
        parser.add_argument(
            "--postprocess-comparison-ui",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Show original and post-processed frames side by side. Requires "
                "--postprocess-preset and --ui."
            ),
        )
        parser.add_argument("--seed", type=int, default=None)
        self._configure_argument_parser(parser)
        args = parser.parse_args(list(commandline_args))

        self._pipeline_config = self.defaults.pipeline_config
        if args.proxy_video is not None:
            self._resolve_replay(args)
        else:
            self._resolve_game(args)
        if args.total_blocks is None:
            args.total_blocks = self.defaults.total_blocks
        self._validate_arguments(args)
        self._apply_parsed_arguments(args)
        if args.compile is not None:
            self._pipeline_config = _apply_compile_flag(
                self._pipeline_config, args.compile
            )
            self._compile = args.compile
        if args.seed is not None:
            self._pipeline_config = derive_config(
                self._pipeline_config,
                diffusion_model={"seed": args.seed},
            )
        self._device = _resolve_runtime_device(args.device)
        self._total_blocks = args.total_blocks
        self._warmup_blocks = args.warmup_blocks
        self._use_ui = args.ui and _is_rank_zero()
        self._game_slug = args.game
        self._prompt_event_chunks = args.prompt_event_chunks
        # A --prompts-json schedule that changes text needs the per-chunk encoder.
        self._dynamic_prompt = args.dynamic_prompt or bool(
            self._prompt_schedule and len(set(self._prompt_schedule)) > 1
        )
        self._wait_for_start = args.wait_for_start
        self._record_proxy = args.record_proxy
        self._chunk_log = args.chunk_log
        # The browser window crops its game panel out of the side-by-side canvas.
        self._side_by_side = (
            args.side_by_side or self._client_window_mode == "eden-webrtc"
        )
        self._postprocess = _resolve_postprocess_config(
            preset=args.postprocess_preset,
            chunk_size=args.postprocess_chunk_size,
            compile_network=args.postprocess_compile,
        )
        self._postprocess_comparison_ui = args.postprocess_comparison_ui
        self._input_values = {
            # Frame-conditioned presets default their first frame to the
            # selected game's cover, so the resolver has to see the slug.
            "game": args.game,
            "prompt": args.prompt,
            "prompt_path": args.prompt_path,
            "image_path": args.image_path,
            "pose_path": args.pose_path,
            "intrinsic_path": args.intrinsic_path,
            "world_scale": args.world_scale,
            "example_data": args.example_data,
            "example_idx": args.example_idx,
            "total_blocks": args.total_blocks,
            "proxy_video": args.proxy_video,
        }

    def _resolve_game(self, args: argparse.Namespace) -> None:
        """Check ``--game`` against the registry and locate the autopilot script."""
        for flag in ("max_frames", "prompts_json"):
            if getattr(args, flag) is not None:
                raise ValueError(f"--{flag.replace('_', '-')} requires --proxy-video.")
        root = games_root()
        slugs = registered_game_slugs(root=root)
        if not slugs:
            raise GamesRootError(f"{root / 'registry.json'} registers no games.")
        args.game = args.game or slugs[0]
        if args.game not in slugs:
            raise ValueError(
                f"--game {args.game!r} is not in {root / 'registry.json'}; "
                f"choose one of: {', '.join(slugs)}."
            )
        self._proxy_video = self._proxy_frames = self._prompt_schedule = None
        self._autopilot = None
        if args.autopilot is not None:
            script = (
                Path(args.autopilot) if args.autopilot else default_autopilot_script()
            )
            if not script.is_file():
                raise FileNotFoundError(f"--autopilot script is missing: {script}")
            self._autopilot = script

    def _resolve_replay(self, args: argparse.Namespace) -> None:
        """Size the rollout to the recording and bound the noise to its length.

        The whole video's initial noise is drawn in one piece
        (``max_latent_frames``), which is what reproduces the reference's RNG
        stream for a seed.
        """
        for flag in ("dynamic_prompt", "wait_for_start"):
            if getattr(args, flag):
                raise ValueError(
                    f"--{flag.replace('_', '-')} does not apply to --proxy-video."
                )
        if args.autopilot is not None:
            raise ValueError("--autopilot does not apply to --proxy-video.")
        if not args.proxy_video.is_file():
            raise FileNotFoundError(f"--proxy-video is missing: {args.proxy_video}")
        if args.max_frames is not None and args.max_frames < 5:
            raise ValueError("--max-frames must be >= 5.")
        num_frames = recording_frame_count(
            args.proxy_video, fps=self.defaults.fps, max_frames=args.max_frames
        )
        len_t = int(self._pipeline_config.diffusion_model.transformer.len_t)
        latent_frames = (num_frames - 1) // 4 + 1
        if latent_frames % len_t:
            whole = latent_frames // len_t * len_t
            raise ValueError(
                f"{args.proxy_video} gives {num_frames} frames = {latent_frames} "
                f"latent frames, which is not a whole number of {len_t}-latent "
                f"chunks; pass --max-frames {4 * whole - 3}."
            )
        self._pipeline_config = derive_config(
            self._pipeline_config, diffusion_model={"max_latent_frames": latent_frames}
        )
        if args.total_blocks is None:
            args.total_blocks = latent_frames // len_t
        self._prompt_schedule = None
        if args.prompts_json is not None:
            self._prompt_schedule = load_chunk_prompts(
                args.prompts_json,
                num_frames,
                fps=self.defaults.fps,
                chunk_size=len_t,
                static_prompt=args.prompt or None,
            )
            args.prompt = args.prompt or self._prompt_schedule[0]
        self._proxy_video = args.proxy_video
        self._proxy_frames = num_frames
        self._autopilot = None
        args.game = ""

    def session_desc(self) -> SessionDesc:
        """Return the model's default output shape and interactive rates."""
        return SessionDesc(
            output_layout=self.defaults.output_layout,
            backpressure_mode=self.defaults.backpressure_mode,
            presentation_mode=self.defaults.presentation_mode,
            frames_per_second_for_ui=self.defaults.ui_fps,
            frames_per_second_for_step=self.defaults.fps,
            video_width=self.defaults.pixel_width,
            video_height=self.defaults.pixel_height,
        )

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create an isolated rollout after lazily loading the shared pipeline."""
        input_values = self._input_values
        if input_values is None:
            raise RuntimeError(
                f"{type(self).__name__}.init() must run before create_session()."
            )
        self._validate_layout(session_desc)
        if (
            self._proxy_video is not None
            and session_desc.frames_per_second_for_step != self.defaults.fps
        ):
            raise ValueError(
                f"--proxy-video is resampled to the model's {self.defaults.fps} fps; "
                f"got --fps {session_desc.frames_per_second_for_step}."
            )
        input_spec = VideoSpec(
            height=session_desc.video_height,
            width=session_desc.video_width,
            fps=session_desc.frames_per_second_for_step,
        )
        resolved_values = {
            **input_values,
            "pixel_height": input_spec.height,
            "pixel_width": input_spec.width,
            "fps": session_desc.frames_per_second_for_step,
        }
        conditioning = self.defaults.input_resolver(resolved_values)
        if not isinstance(conditioning, Cam2VConditioning):
            raise TypeError(
                "Cam2VApplicationDefaults.input_resolver must return Cam2VConditioning."
            )

        pipeline = self._pipeline
        if pipeline is None:
            pipeline = self._pipeline_config.setup().to(self._device).eval()
            self._pipeline = pipeline
        self._validate_frame_size(session_desc, pipeline)
        output_spec = (
            self._postprocess.output_spec(input_spec)
            if self._postprocess.is_enabled()
            else input_spec
        )
        if self._postprocess_comparison_ui:
            output_spec = comparison_output_spec(
                input_spec=input_spec,
                postprocessed_spec=output_spec,
            )
        # The runtime opens the UI and output sink from this presentation contract.
        session_desc = replace(
            session_desc,
            video_width=output_spec.width * (2 if self._side_by_side else 1),
            video_height=output_spec.height,
        )
        postprocess_stream = None
        if self._postprocess.is_enabled():
            postprocess_stream = VideoPostprocessStream(
                postprocess=self._postprocess,
                output_layout=session_desc.output_layout.value,
                fps=session_desc.frames_per_second_for_step,
                per_view=False,
                world_size=_distributed_world_size(),
            )
            postprocess_stream.prepare(input_spec)
        return Game2VSession(
            pipeline=pipeline,
            postprocess_stream=postprocess_stream,
            session_desc=session_desc,
            config=Game2VSessionConfig(
                conditioning=conditioning,
                total_blocks=self._total_blocks,
                device=torch.device(self._device),
                model_video_width=input_spec.width,
                model_video_height=input_spec.height,
                first_frame_dtype=self.defaults.first_frame_dtype,
                first_frame_interpolation=self.defaults.first_frame_interpolation,
                generate_step=self.defaults.generate_step,
                pose_integrator_factory=self.defaults.pose_integrator_factory,
                warmup_blocks=self._warmup_blocks,
                log_model_timing=self.defaults.log_model_timing,
                install_hint=self.defaults.install_hint,
                postprocess_enabled=self._postprocess.is_enabled(),
                postprocess_comparison=self._postprocess_comparison_ui,
                fixed_camera=self.defaults.fixed_camera,
                game_slug=self._game_slug,
                prompt_event_chunks=self._prompt_event_chunks,
                dynamic_prompt=self._dynamic_prompt,
                wait_for_start=self._wait_for_start,
                control_stream_factory=self.defaults.control_stream_factory,
                compile_control_encoder=self._compile is not False,
                autopilot=self._autopilot,
                proxy_video=self._proxy_video,
                proxy_frames=self._proxy_frames,
                prompt_schedule=self._prompt_schedule,
                record_proxy=self._record_proxy,
                chunk_log=self._chunk_log,
                side_by_side=self._side_by_side,
                # The /ext/game2v panel needs an integration-owned browser
                # window; the built-in modes have nowhere to mount it.
                register_panel=self._client_window_mode == "eden-webrtc",
            ),
            use_ui=self._use_ui,
        )

    def close(self) -> None:
        """Release the application-owned pipeline after all sessions stop."""
        pipeline = self._pipeline
        self._pipeline = None
        self._input_values = None
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()

    def _configure_argument_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add integration-specific application arguments to ``parser``."""

    def _apply_parsed_arguments(self, args: argparse.Namespace) -> None:
        """Retain integration-specific arguments after shared validation."""

    def _validate_arguments(self, args: argparse.Namespace) -> None:
        """Reject invalid rollout and timing settings."""
        if args.postprocess_preset:
            raise ValueError(
                "Game2V pairs every generated frame with its proxy frame, so it "
                "cannot post-process (pass no --postprocess-preset)."
            )
        if args.total_blocks <= 0:
            raise ValueError("--total-blocks must be > 0.")
        model = self._pipeline_config.diffusion_model
        if model.max_latent_frames is not None:
            len_t = model.transformer.len_t
            max_blocks = (model.max_latent_frames + len_t - 1) // len_t
            if args.total_blocks > max_blocks:
                raise ValueError(
                    f"--total-blocks must be <= {max_blocks} for "
                    f"max_latent_frames={model.max_latent_frames} (len_t={len_t}); "
                    f"got {args.total_blocks}."
                )
        if args.warmup_blocks < 0:
            raise ValueError("--warmup-blocks must be >= 0.")
        if args.prompt_event_chunks <= 0:
            raise ValueError("--prompt-event-chunks must be > 0.")
        if args.dynamic_prompt and self.defaults.control_stream_factory is None:
            raise ValueError(
                "--dynamic-prompt is only supported by the proxy-DINO game "
                "presets (5B and 14B); this preset has no per-chunk text "
                "encoder."
            )
        if args.world_scale is not None and args.world_scale < 0:
            raise ValueError("--world-scale must be >= 0 when set.")
        if args.postprocess_comparison_ui and not args.ui:
            raise ValueError("--postprocess-comparison-ui requires --ui.")
        if args.postprocess_comparison_ui and not args.postprocess_preset:
            raise ValueError(
                "--postprocess-comparison-ui requires --postprocess-preset."
            )

    def _validate_layout(self, session_desc: SessionDesc) -> None:
        """Reject output layouts that differ from the model's declared layout."""
        if session_desc.output_layout is not self.defaults.output_layout:
            raise ValueError(
                "This camera-to-video model only produces "
                f"{self.defaults.output_layout.value} output, got "
                f"{session_desc.output_layout.value}."
            )
        if self._use_ui and session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError("The Cam2V SlangPy UI overlay requires tchw output.")

    def _validate_frame_size(self, session_desc: SessionDesc, pipeline: Any) -> None:
        """Reject frame dimensions that cannot map to integral latents."""
        decoder = getattr(pipeline, "decoder", None)
        ratio = getattr(decoder, "spatial_compression_ratio", None)
        if not isinstance(ratio, int) or ratio <= 0:
            raise TypeError(
                "Cam2V requires a decoder with a positive integer "
                "spatial_compression_ratio."
            )
        if session_desc.video_width % ratio or session_desc.video_height % ratio:
            raise ValueError(
                f"Frame dimensions must be multiples of {ratio}, got "
                f"{session_desc.video_width}x{session_desc.video_height}."
            )


def _apply_compile_flag(pipeline_config: Any, enabled: bool) -> Any:
    """Turn every compile and CUDA-graph switch in ``pipeline_config`` on or off.

    ``--compile`` used to reach only the transformer's ``compile_network``, so
    ``--no-compile`` left the VAE and decoder graphs, their ``torch.compile``
    and the proxy-DINO encoder compiling anyway. Each flag is patched only when
    the component actually has it, since the camera-control presets and the
    game presets do not share one config shape.
    """
    changes: dict[str, Any] = {}
    transformer = getattr(
        getattr(pipeline_config, "diffusion_model", None), "transformer", None
    )
    if transformer is not None:
        transformer_changes = {
            name: enabled
            for name in ("compile_network", "use_cuda_graph")
            if hasattr(transformer, name)
        }
        if transformer_changes:
            changes["diffusion_model"] = {"transformer": transformer_changes}
    for component in ("encoder", "decoder"):
        module = getattr(pipeline_config, component, None)
        if module is None:
            continue
        module_changes: dict[str, Any] = {
            name: enabled
            for name in ("use_compile", "use_cuda_graph")
            if hasattr(module, name)
        }
        # The proxy-DINO encoder is not part of the pipeline config; the VAE
        # it wraps for the first-frame condition is, one level down.
        inner = getattr(module, "encoder", None)
        if inner is not None:
            inner_changes = {
                name: enabled
                for name in ("use_compile", "use_cuda_graph")
                if hasattr(inner, name)
            }
            if inner_changes:
                module_changes["encoder"] = inner_changes
        if module_changes:
            changes[component] = module_changes
    if not changes:
        return pipeline_config
    return derive_config(pipeline_config, **changes)


def _resolve_postprocess_config(
    *,
    preset: str,
    chunk_size: int,
    compile_network: bool,
) -> VideoPostprocessChainConfig:
    """Resolve a preset and apply Cam2V's low-latency processor options."""
    selected = VideoPostprocessChainConfig(preset=preset).resolved_processors()
    resolved = []
    for processor in selected:
        changes: dict[str, object] = {}
        if hasattr(processor, "chunk_size"):
            changes["chunk_size"] = chunk_size
        if hasattr(processor, "compile_network"):
            changes["compile_network"] = compile_network
        if not compile_network and hasattr(processor, "use_cuda_graph"):
            changes["use_cuda_graph"] = False
        resolved.append(replace(processor, **changes) if changes else processor)
    return VideoPostprocessChainConfig(processors=tuple(resolved))


def _is_rank_zero() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def _resolve_runtime_device(device: str) -> str:
    """Pin each torchrun rank to its local GPU when the caller left device generic."""
    if device != "cuda":
        return device
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return device
    return f"cuda:{int(local_rank)}"


def _distributed_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


__all__ = ["Game2VApplication"]
