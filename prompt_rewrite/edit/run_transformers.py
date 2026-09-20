#!/usr/bin/env python3
"""Edit Prompt Enhancer -- HuggingFace transformers reference impl.

Rewrites a vague image-editing instruction (plus the input image[s]) into a
precise, actionable prompt suitable for a downstream image-editing model.
Base model: Qwen3.5-VL 9B. Uses the chat template's <think> path, then splits
thinking from the answer.

Usage:
    python run_transformers.py \\
        --ckpt <local_dir_or_hf_id> \\
        --input data/example.jsonl \\
        --output out.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from pe_output import build_record, report_parse_failures, split_thinking


SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_prompt.txt"


def load_image(path: str, base_dir: Path, max_pixels: int) -> Image.Image:
    """Open image (resolving relative paths against the input jsonl's dir) and
    downscale so W*H <= max_pixels, preserving aspect. Matches training's
    IMAGE_MAX_PIXELS = 1024*1024."""
    img_path = Path(path)
    if not img_path.is_absolute():
        img_path = base_dir / img_path
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    if max_pixels and w * h > max_pixels:
        s = (max_pixels / float(w * h)) ** 0.5
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    return im


@torch.inference_mode()
def rewrite(
    model,
    processor,
    system_prompt: str,
    user_prompt: str,
    images: list[Image.Image],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> tuple[str, str]:
    user_content: list[dict[str, Any]] = [{"type": "image", "image": im} for im in images]
    user_content.append({"type": "text", "text": user_prompt})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=True,
    ).to(model.device)

    # Qwen3.5-VL uses mm_token_type_ids to mark image spans; some processor versions
    # don't populate it from apply_chat_template, so create it if missing.
    if "mm_token_type_ids" not in inputs and hasattr(processor, "create_mm_token_type_ids"):
        inputs["mm_token_type_ids"] = processor.create_mm_token_type_ids(inputs["input_ids"])

    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return split_thinking(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Local HF dir or Hub id (Qwen3.5-VL 9B edit-PE ckpt).")
    ap.add_argument("--input", required=True, help="JSONL: {id, prompt, input_images, task_type?}.")
    ap.add_argument("--output", required=True, help="Output JSONL.")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image-max-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--max-new-tokens", type=int, default=24000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Process only first N lines (0 = all).")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"Loading model from {args.ckpt} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.ckpt)
    # low_cpu_mem_usage streams weights straight to the target dtype without
    # a full-precision CPU copy; .to(device) then moves the single copy to GPU.
    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, dtype=dtype, low_cpu_mem_usage=True
    ).to(args.device).eval()

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    in_path = Path(args.input).resolve()
    base_dir = in_path.parent
    with open(in_path, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        cases = cases[: args.limit]

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with open(out_path, "w", encoding="utf-8") as fout:
        for case in tqdm(cases, desc="edit-PE"):
            images = [load_image(p, base_dir, args.image_max_pixels)
                      for p in case["input_images"]]
            thinking, answer = rewrite(
                model, processor, system_prompt, case["prompt"], images,
                args.max_new_tokens, args.temperature, args.top_p, args.top_k, args.seed,
            )
            record = build_record(case, thinking, answer)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            records.append(record)
    print(f"Wrote {len(records)} records to {out_path}")
    report_parse_failures(records)


if __name__ == "__main__":
    main()
