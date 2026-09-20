#!/bin/bash
# 用 vLLM 起 qwen21_t2i_pe_9b 的 OpenAI 兼容推理服务。
#
#   bash serve.sh
#
# 环境变量(都可选):
#   PORT      服务端口            默认 8100
#   GPUS      CUDA_VISIBLE_DEVICES 默认 0..TP-1
#   TP        张量并行            默认 = 可见 GPU 数(取 {1,2,4,8})
#   QUANT     "" | fp8            默认 ""(bf16);只有显存不够时才设 fp8
#   MAX_LEN   最大序列长度        默认 24576
#   MEM_UTIL  显存占用率          默认 0.90
#   CKPT      ckpt 目录           默认 ./qwen21_t2i_pe_9b
#   PY        带 vLLM 的 python   默认 mgt_pe(vLLM 0.19.1 / transformers 5.4)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT="${CKPT:-$HERE/qwen21_t2i_pe_9b}"
PORT="${PORT:-8100}"
PY="${PY:-/cpfs02/users/zhangzk/envs/mgt_pe/bin/python}"
MAX_LEN="${MAX_LEN:-24576}"
MEM_UTIL="${MEM_UTIL:-0.90}"
QUANT="${QUANT:-}"

if [ -n "${GPUS:-}" ]; then NGPU=$(awk -F, '{print NF}' <<<"$GPUS")
else NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NGPU:-0}" -ge 1 ] || NGPU=1; fi
_tp(){ for t in 8 4 2 1; do [ "$t" -le "$1" ] && { echo "$t"; return; }; done; echo 1; }
TP="${TP:-$(_tp "$NGPU")}"
GPUS="${GPUS:-$(seq -s, 0 $((TP-1)))}"
QARG=""; [ -n "$QUANT" ] && QARG="--quantization $QUANT"

# qwen3_5 的 linear-attention 层需要 eager;NCCL 走 net0 以兼容多机
export GLOO_SOCKET_IFNAME=net0 TP_SOCKET_IFNAME=net0 VLLM_USE_CUSTOM_ALLREDUCE=0

echo "[serve] model=qwen21_t2i_pe_9b  ckpt=$CKPT"
echo "[serve] GPUS=$GPUS TP=$TP dtype=${QUANT:-bf16} port=$PORT max_len=$MAX_LEN"

exec env CUDA_VISIBLE_DEVICES="$GPUS" "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$CKPT" --served-model-name qwen21_t2i_pe_9b --port "$PORT" \
    --dtype bfloat16 $QARG \
    --tensor-parallel-size "$TP" --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$MEM_UTIL" \
    --reasoning-parser qwen3 --disable-custom-all-reduce --enforce-eager --no-async-scheduling
