# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UMT5 text encoder config that accepts a local EdenLive weight directory.

:class:`UMT5TextEncoderConfig` types ``model_id_or_local_path`` as a ``Literal``
of the public Wan repo ids. The EdenLive release ships its own UMT5 copy under
``<model_root>/umt5/{text_encoder,tokenizer}``, which the encoder already loads
(``maybe_download_hf_repo_on_rank0`` is a no-op for local directories); only the
annotation needs widening.
"""

from __future__ import annotations

from dataclasses import dataclass

from flashdreams.infra.encoder.text.umt5 import UMT5TextEncoderConfig


@dataclass(kw_only=True)
class EdenUMT5TextEncoderConfig(UMT5TextEncoderConfig):
    """:class:`UMT5TextEncoderConfig` with a free-form local path."""

    model_id_or_local_path: str  # type: ignore[assignment]
    """Local directory holding ``text_encoder/`` and ``tokenizer/``."""


__all__ = ["EdenUMT5TextEncoderConfig"]
