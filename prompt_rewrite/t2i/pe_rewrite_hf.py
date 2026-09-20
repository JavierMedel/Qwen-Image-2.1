#!/usr/bin/env python3
"""Rewrite one image prompt with qwen21_t2i_pe_9b using plain transformers (no vLLM).

Loads the model locally and generates once. Prints the same JSON as pe_rewrite.py:
  {"thinking": "...", "rewritten_prompt": "<long English prompt>", "wh_ratio": "16:9"}

Usage:
  python pe_rewrite_hf.py "a corgi playing guitar in the rain"

Env:
  PE_MODEL_DIR  model dir (weights + system_prompt.txt)  (default ./qwen21_t2i_pe_9b)
  CUDA_VISIBLE_DEVICES  which GPU(s) to use
"""
import argparse
import json
import os
import sys

import json_repair
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(HERE, "qwen21_t2i_pe_9b")

# arena inference sampling (matches serve.sh / config.sh)
TEMPERATURE, TOP_P, TOP_K, PRESENCE_PENALTY = 1.0, 0.95, 20, 1.5
MAX_NEW_TOKENS = 16256


class PresencePenalty(LogitsProcessor):
    """vLLM-style presence penalty: subtract `penalty` from any token already generated
    (transformers has no native presence_penalty; repetition_penalty has different math)."""
    def __init__(self, penalty, prompt_len):
        self.penalty = penalty
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            gen = input_ids[b, self.prompt_len:]
            if gen.numel():
                scores[b, gen.unique()] -= self.penalty
        return scores


def parse_rewrite(content):
    obj = json_repair.repair_json(content, return_objects=True)
    if isinstance(obj, list):
        obj = obj[0] if obj else {}
    if isinstance(obj, dict):
        rp = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")
        if rp:
            return {"rewritten_prompt": rp, "wh_ratio": obj.get("wh_ratio", "")}
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", help="the user's image request")
    ap.add_argument("--model-dir", default=os.environ.get("PE_MODEL_DIR", DEFAULT_MODEL_DIR))
    a = ap.parse_args()

    sys_prompt = open(os.path.join(a.model_dir, "system_prompt.txt"), encoding="utf-8").read().strip()
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    text = tok.apply_chat_template(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": a.prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
            logits_processor=LogitsProcessorList([PresencePenalty(PRESENCE_PENALTY, prompt_len)]),
            pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0, prompt_len:], skip_special_tokens=True)

    # chat template opened <think>; split the closing tag into reasoning + answer
    thinking, _, content = gen.partition("</think>")
    parsed = parse_rewrite(content if content else gen)
    if parsed is None:
        print("rewrite failed: no rewritten_prompt in model output", file=sys.stderr)
        sys.exit(1)

    print(json.dumps({"thinking": thinking.strip(),
                      "rewritten_prompt": parsed["rewritten_prompt"],
                      "wh_ratio": parsed["wh_ratio"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
