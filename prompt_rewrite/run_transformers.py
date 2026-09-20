#!/usr/bin/env python3
"""Unified prompt rewriter — HuggingFace transformers reference impl.

Handles both text-to-image (no input images) and image editing (with input
images) prompt rewriting. The mode is determined automatically:
  - If the input has images → edit mode (uses system_prompt_edit.txt)
  - If the input has no images → t2i mode (uses system_prompt_t2i.txt)

Usage:
    # Single T2I prompt
    python run_transformers.py --ckpt /path/to/t2i_ckpt --prompt "a corgi in rain"

    # Batch edit from JSONL
    python run_transformers.py --ckpt /path/to/edit_ckpt --input data/example.jsonl --output out.jsonl

    # Single edit with image
    python run_transformers.py --ckpt /path/to/edit_ckpt --prompt "make the sky sunset" --images photo.png
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from pe_output import build_record, parse_answer, report_parse_failures, split_thinking

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_image(path: str, base_dir: Path, max_pixels: int) -> Image.Image:
    img_path = Path(path)
    if not img_path.is_absolute():
        img_path = base_dir / img_path
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    if max_pixels and w * h > max_pixels:
        s = (max_pixels / float(w * h)) ** 0.5
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    return im


def detect_mode(cases: list[dict]) -> str:
    has_images = any(case.get("input_images") for case in cases)
    return "edit" if has_images else "t2i"


def get_system_prompt(mode: str, override: str | None = None) -> str:
    if override:
        return Path(override).read_text(encoding="utf-8").strip()
    name = "system_prompt_edit.txt" if mode == "edit" else "system_prompt_t2i.txt"
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


@torch.inference_mode()
def rewrite_one(
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
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return split_thinking(text)


def main():
    ap = argparse.ArgumentParser(description="Unified prompt rewriter (T2I + Edit)")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint path or HF id.")
    # Input: either --prompt (single) or --input (JSONL batch)
    ap.add_argument("--prompt", help="Single prompt to rewrite.")
    ap.add_argument("--images", nargs="*", default=[], help="Input image(s) for single-prompt mode.")
    ap.add_argument("--input", help="JSONL file: {id, prompt, input_images?, task_type?}.")
    ap.add_argument("--output", help="Output JSONL file (required with --input).")
    # System prompt override
    ap.add_argument("--system-prompt", help="Override system prompt file path.")
    # Model config
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image-max-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--max-new-tokens", type=int, default=24000)
    # Sampling
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Process only first N lines (0 = all).")
    args = ap.parse_args()

    if not args.prompt and not args.input:
        ap.error("Provide either --prompt or --input.")
    if args.input and not args.output:
        ap.error("--output is required with --input.")

    # Build cases
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
    print(f"Mode: {mode} ({len(cases)} case(s))", flush=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"Loading model from {args.ckpt} ...", flush=True)

    if mode == "edit":
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.ckpt)
        model = AutoModelForImageTextToText.from_pretrained(
            args.ckpt, dtype=dtype, low_cpu_mem_usage=True
        ).to(args.device).eval()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        class _Proc:
            """Minimal wrapper so T2I uses the same rewrite_one interface."""
            def __init__(self, ckpt):
                self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
            def apply_chat_template(self, messages, **kwargs):
                # Flatten content lists to plain strings for the text-only model
                flat = []
                for m in messages:
                    content = m["content"]
                    if isinstance(content, list):
                        content = "".join(c["text"] for c in content if c.get("type") == "text")
                    flat.append({"role": m["role"], "content": content})
                text = self.tokenizer.apply_chat_template(
                    flat, tokenize=False, add_generation_prompt=True, enable_thinking=True
                )
                return self.tokenizer(text, return_tensors="pt")
            def create_mm_token_type_ids(self, input_ids):
                return None

        processor = _Proc(args.ckpt)
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt, dtype=dtype, low_cpu_mem_usage=True
        ).to(args.device).eval()

    records = []
    for case in tqdm(cases, desc="rewriting"):
        images = []
        if case.get("input_images"):
            images = [load_image(p, base_dir, args.image_max_pixels) for p in case["input_images"]]
        thinking, answer = rewrite_one(
            model, processor, system_prompt, case["prompt"], images,
            args.max_new_tokens, args.temperature, args.top_p, args.top_k, args.seed,
        )
        record = build_record(case, thinking, answer)
        records.append(record)

    # Output
    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fout:
            for r in records:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} records to {out_path}")
        report_parse_failures(records)
    else:
        # Single prompt mode: print to stdout
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
