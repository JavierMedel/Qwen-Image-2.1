#!/usr/bin/env python3
"""Edit Prompt Enhancer -- vLLM offline-batch impl.

Same task, base model (Qwen3.5-VL 9B) and I/O as run_transformers.py, but
batched through vLLM's LLM.chat() API. Recommended when processing >= a few
dozen samples.

Usage:
    python run_vllm.py \\
        --ckpt <local_dir_or_hf_id> \\
        --input data/example.jsonl \\
        --output out.jsonl \\
        --tp 1
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path

# Some environments preset GLOO/NCCL socket interface envs to names that don't
# exist on this host (e.g. bond1). vLLM's gloo backend then crashes on init.
# Clear them if the named interface is missing so vLLM auto-selects a live one.
for _v in ("GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME"):
    _iface = os.environ.get(_v)
    if _iface and not os.path.isdir(f"/sys/class/net/{_iface}"):
        os.environ.pop(_v, None)

from PIL import Image  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from pe_output import build_record, report_parse_failures, split_thinking  # noqa: E402


SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_prompt.txt"


def image_to_data_uri(path: Path, max_pixels: int) -> str:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if max_pixels and w * h > max_pixels:
        s = (max_pixels / float(w * h)) ** 0.5
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_messages(system_prompt: str, user_prompt: str, image_uris: list[str]) -> list[dict]:
    user_content: list[dict] = [{"type": "image_url", "image_url": {"url": u}} for u in image_uris]
    user_content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=24000)
    ap.add_argument("--limit-mm-per-prompt-image", type=int, default=10)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--swap-space", type=int, default=8, help="CPU swap for KV cache (GiB).")
    ap.add_argument("--max-num-seqs", type=int, default=32, help="Concurrent decode slots.")
    ap.add_argument("--image-max-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--max-new-tokens", type=int, default=24000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading vLLM engine from {args.ckpt} (tp={args.tp}) ...", flush=True)
    llm = LLM(
        model=args.ckpt,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": args.limit_mm_per_prompt_image},
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=False,
        seed=args.seed,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    in_path = Path(args.input).resolve()
    base_dir = in_path.parent
    with open(in_path, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        cases = cases[: args.limit]

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    conversations = []
    for case in cases:
        image_uris = []
        for p in case["input_images"]:
            img_path = Path(p) if Path(p).is_absolute() else base_dir / p
            image_uris.append(image_to_data_uri(img_path, args.image_max_pixels))
        conversations.append(build_messages(system_prompt, case["prompt"], image_uris))

    # chat_template_kwargs.enable_thinking is passed at the top-level call, not
    # via extra_body -- vLLM forwards it to processor.apply_chat_template.
    outputs = llm.chat(
        conversations,
        sampling_params=sampling,
        chat_template_kwargs={"enable_thinking": True},
    )

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with open(out_path, "w", encoding="utf-8") as fout:
        for case, out in zip(cases, outputs):
            thinking, answer = split_thinking(out.outputs[0].text)
            record = build_record(case, thinking, answer)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
    print(f"Wrote {len(records)} records to {out_path}")
    report_parse_failures(records)


if __name__ == "__main__":
    main()
