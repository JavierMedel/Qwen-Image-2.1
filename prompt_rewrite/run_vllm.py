#!/usr/bin/env python3
"""Unified prompt rewriter — vLLM offline-batch impl.

Handles both text-to-image and image editing prompt rewriting. Mode is
detected automatically from input (images present → edit, else → t2i).

Usage:
    # T2I batch
    python run_vllm.py --ckpt /path/to/t2i_ckpt --input t2i_prompts.jsonl --output out.jsonl

    # Edit batch
    python run_vllm.py --ckpt /path/to/edit_ckpt --input data/example.jsonl --output out.jsonl

    # Single T2I prompt
    python run_vllm.py --ckpt /path/to/t2i_ckpt --prompt "a corgi in the rain"
"""

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

for _v in ("GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME"):
    _iface = os.environ.get(_v)
    if _iface and not os.path.isdir(f"/sys/class/net/{_iface}"):
        os.environ.pop(_v, None)

from PIL import Image  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from pe_output import build_record, report_parse_failures, split_thinking  # noqa: E402

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def image_to_data_uri(path: Path, max_pixels: int) -> str:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if max_pixels and w * h > max_pixels:
        s = (max_pixels / float(w * h)) ** 0.5
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def detect_mode(cases: list[dict]) -> str:
    has_images = any(case.get("input_images") for case in cases)
    return "edit" if has_images else "t2i"


def get_system_prompt(mode: str, override: str | None = None) -> str:
    if override:
        return Path(override).read_text(encoding="utf-8").strip()
    name = "system_prompt_edit.txt" if mode == "edit" else "system_prompt_t2i.txt"
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def build_messages(system_prompt: str, user_prompt: str, image_uris: list[str]) -> list[dict]:
    user_content: list[dict] = [{"type": "image_url", "image_url": {"url": u}} for u in image_uris]
    user_content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def main():
    ap = argparse.ArgumentParser(description="Unified prompt rewriter (vLLM batch)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", help="Single prompt (no --input needed).")
    ap.add_argument("--images", nargs="*", default=[], help="Images for single-prompt mode.")
    ap.add_argument("--input", help="JSONL batch input.")
    ap.add_argument("--output", help="Output JSONL (required with --input).")
    ap.add_argument("--system-prompt", help="Override system prompt file.")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=24000)
    ap.add_argument("--limit-mm-per-prompt-image", type=int, default=10)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--swap-space", type=int, default=8)
    ap.add_argument("--max-num-seqs", type=int, default=32)
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

    if not args.prompt and not args.input:
        ap.error("Provide either --prompt or --input.")
    if args.input and not args.output:
        ap.error("--output is required with --input.")

    if args.prompt:
        cases = [{"id": "cli", "prompt": args.prompt}]
        if args.images:
            cases[0]["input_images"] = args.images
        base_dir = Path.cwd()
    else:
        in_path = Path(args.input).resolve()
        base_dir = in_path.parent
        with open(in_path, encoding="utf-8") as f:
            cases = [json.loads(line) for line in f if line.strip()]
        if args.limit:
            cases = cases[: args.limit]

    mode = detect_mode(cases)
    system_prompt = get_system_prompt(mode, args.system_prompt)

    # T2I model uses presence_penalty=1.5 by default
    if mode == "t2i" and args.presence_penalty == 0.0:
        args.presence_penalty = 1.5

    print(f"Mode: {mode} ({len(cases)} case(s))", flush=True)
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

    conversations = []
    for case in cases:
        image_uris = []
        for p in case.get("input_images", []):
            img_path = Path(p) if Path(p).is_absolute() else base_dir / p
            image_uris.append(image_to_data_uri(img_path, args.image_max_pixels))
        conversations.append(build_messages(system_prompt, case["prompt"], image_uris))

    outputs = llm.chat(
        conversations,
        sampling_params=sampling,
        chat_template_kwargs={"enable_thinking": True},
    )

    records = []
    for case, out in zip(cases, outputs):
        thinking, answer = split_thinking(out.outputs[0].text)
        records.append(build_record(case, thinking, answer))

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fout:
            for r in records:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} records to {out_path}")
        report_parse_failures(records)
    else:
        r = records[0]
        print(json.dumps({
            "thinking": r["thinking"],
            "rewritten_prompt": r["rewritten_prompt"],
            "wh_ratio": r["wh_ratio"],
            **({"ratio_follow": r["ratio_follow"]} if r.get("ratio_follow") else {}),
        }, ensure_ascii=False, indent=2))
        if not r["parse_ok"]:
            print("WARNING: answer did not parse as expected JSON.", file=sys.stderr)


if __name__ == "__main__":
    main()
