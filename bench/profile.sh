#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Launcher for profile_mem.py (runs inside WSL). Mirrors bench.sh env.
# Optional: pass ALLOC_CONF env to test PYTORCH_CUDA_ALLOC_CONF (e.g. expandable_segments:True).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export LIBRARY_PATH="/usr/lib/wsl/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
[ -n "${ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" && echo "[profile.sh] PYTORCH_CUDA_ALLOC_CONF=$ALLOC_CONF"
# HF_TOKEN is read from the environment. If unset, optionally point HF_TOKEN_FILE
# at a file holding the token (e.g. export HF_TOKEN_FILE="$HOME/.hf_token").
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HF_TOKEN_FILE:-}" ] && [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
fi
cd "$HOME/flashdreams"
VENV_NVCC=$(ls .venv/lib/python3*/site-packages/nvidia/cuda_nvcc/bin/nvcc 2>/dev/null | head -1)
if [ -n "$VENV_NVCC" ]; then export CUDA_HOME="$(dirname "$(dirname "$VENV_NVCC")")"; export PATH="$CUDA_HOME/bin:$PATH";
elif [ -d /usr/local/cuda ]; then export CUDA_HOME=/usr/local/cuda; export PATH="$CUDA_HOME/bin:$PATH"; fi
tr -d '\r' < /mnt/e/flashdreams/bench/profile_mem.py > /tmp/profile_mem.py
exec uv run --no-sync --package flashdreams-omnidreams python /tmp/profile_mem.py "$@"
