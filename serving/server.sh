#!/usr/bin/env bash
set -euo pipefail

# InfiniteTalk Serving (Ulysses=8 + TeaCache) "ultimate" server command.
#
# Usage:
#   bash serving/server.sh
#
# Overrides (optional):
#   W=/nas/shared/models/InfiniteTalk/weights PORT=8000 HOST=0.0.0.0 bash serving/server.sh
#   NPROC=8 ULYSSES_SIZE=8 RING_SIZE=1 TEACACHE_THRESH=0.2 bash serving/server.sh

export TRITON_PTXAS_PATH="/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas"
export TRITON_CUOBJDUMP_PATH="/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/cuobjdump"
export TRITON_NVDISASM_PATH="/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/nvdisasm"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

W="${W:-/nas/shared/models/InfiniteTalk/weights}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

NPROC="${NPROC:-8}"
ULYSSES_SIZE="${ULYSSES_SIZE:-8}"
RING_SIZE="${RING_SIZE:-1}"

TEACACHE_THRESH="${TEACACHE_THRESH:-0.2}"

# Default generation params (client request can override size/motion_frame/sample_shift/etc.)
SIZE_DEFAULT="${SIZE_DEFAULT:-infinitetalk-480}"
MOTION_FRAME_DEFAULT="${MOTION_FRAME_DEFAULT:-9}"
TEXT_GUIDE_DEFAULT="${TEXT_GUIDE_DEFAULT:-5}"
AUDIO_GUIDE_DEFAULT="${AUDIO_GUIDE_DEFAULT:-4}"

export PYTHONUNBUFFERED=1

torchrun --nproc_per_node="${NPROC}" -m infinitetalk.serving \
  --ckpt_dir "${W}/Wan2.1-I2V-14B-480P" \
  --wav2vec_dir "${W}/chinese-wav2vec2-base" \
  --infinitetalk_dir "${W}/InfiniteTalk/single/infinitetalk.safetensors" \
  --dit_fsdp --t5_fsdp \
  --ulysses_size "${ULYSSES_SIZE}" --ring_size "${RING_SIZE}" \
  --size "${SIZE_DEFAULT}" \
  --motion_frame "${MOTION_FRAME_DEFAULT}" \
  --sample_text_guide_scale "${TEXT_GUIDE_DEFAULT}" \
  --sample_audio_guide_scale "${AUDIO_GUIDE_DEFAULT}" \
  --use_teacache --teacache_thresh "${TEACACHE_THRESH}" \
  --host "${HOST}" --port "${PORT}"
