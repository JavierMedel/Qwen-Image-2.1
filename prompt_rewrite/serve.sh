#!/bin/bash
# Launch a vLLM OpenAI-compatible server for prompt rewriting.
#
#   bash serve.sh --ckpt /path/to/ckpt
#
# Environment variables (all optional):
#   PORT      Server port            (default 8100)
#   GPUS      CUDA_VISIBLE_DEVICES   (default 0..TP-1)
#   TP        Tensor parallel size   (default = visible GPU count, capped to {1,2,4,8})
#   QUANT     "" | fp8               (default ""; only set fp8 if memory-constrained)
#   MAX_LEN   Max sequence length    (default 24576)
#   MEM_UTIL  GPU memory utilization (default 0.90)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT="${1:?Usage: bash serve.sh /path/to/ckpt [--port PORT]}"
shift
PORT="${PORT:-8100}"
MAX_LEN="${MAX_LEN:-24576}"
MEM_UTIL="${MEM_UTIL:-0.90}"
QUANT="${QUANT:-}"

if [ -n "${GPUS:-}" ]; then NGPU=$(awk -F, '{print NF}' <<<"$GPUS")
else NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NGPU:-0}" -ge 1 ] || NGPU=1; fi
_tp(){ for t in 8 4 2 1; do [ "$t" -le "$1" ] && { echo "$t"; return; }; done; echo 1; }
TP="${TP:-$(_tp "$NGPU")}"
GPUS="${GPUS:-$(seq -s, 0 $((TP-1)))}"
QARG=""; [ -n "$QUANT" ] && QARG="--quantization $QUANT"

echo "[serve] ckpt=$CKPT"
echo "[serve] GPUS=$GPUS TP=$TP dtype=${QUANT:-bf16} port=$PORT max_len=$MAX_LEN"

exec env CUDA_VISIBLE_DEVICES="$GPUS" python -m vllm.entrypoints.openai.api_server \
    --model "$CKPT" --port "$PORT" \
    --dtype bfloat16 $QARG \
    --tensor-parallel-size "$TP" --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$MEM_UTIL" \
    --reasoning-parser qwen3 --disable-custom-all-reduce --enforce-eager --no-async-scheduling
