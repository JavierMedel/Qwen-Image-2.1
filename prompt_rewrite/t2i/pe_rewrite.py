#!/usr/bin/env python3
"""
Given a user request, prints a JSON object:
  {"thinking": "...", "rewritten_prompt": "<prompt>", "wh_ratio": "<ratio>"}

Usage:
  python pe_rewrite.py "a corgi playing guitar in the rain"

Env:
  PE_URL        server base url   (default http://localhost:8100/v1)
  PE_MODEL      served model name (default qwen21_t2i_pe_9b)
  PE_MODEL_DIR  model dir holding system_prompt.txt (default ./qwen21_t2i_pe_9b)
"""
import argparse
import json
import os
import sys

import json_repair
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(HERE, "qwen21_t2i_pe_9b")

# arena inference sampling (see README)
SAMPLING = dict(temperature=1.0, top_p=0.95, presence_penalty=1.5)
EXTRA = {"top_k": 20, "min_p": 0.0, "chat_template_kwargs": {"enable_thinking": True}}


def rewrite(user_prompt, *, url, model, sys_prompt, timeout=600):
    """Return (thinking, {"rewritten_prompt", "wh_ratio"}); the latter is None on failure."""
    client = OpenAI(base_url=url, api_key="dummy")
    think, content = [], []
    for chunk in client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user_prompt}],
            stream=True, timeout=timeout, extra_body=EXTRA, **SAMPLING):
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        rc = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
        if rc:
            think.append(rc)
        if getattr(d, "content", None):
            content.append(d.content)

    obj = json_repair.repair_json("".join(content), return_objects=True)
    if isinstance(obj, list):
        obj = obj[0] if obj else {}
    parsed = None
    if isinstance(obj, dict):
        rp = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")
        if rp:
            parsed = {"rewritten_prompt": rp, "wh_ratio": obj.get("wh_ratio", "")}
    return "".join(think).strip(), parsed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", help="the user's image request")
    ap.add_argument("--url", default=os.environ.get("PE_URL", "http://localhost:8100/v1"))
    ap.add_argument("--model", default=os.environ.get("PE_MODEL", "qwen21_t2i_pe_9b"))
    ap.add_argument("--model-dir", default=os.environ.get("PE_MODEL_DIR", DEFAULT_MODEL_DIR),
                    help="model dir; system_prompt.txt is read from here")
    a = ap.parse_args()

    sys_prompt = open(os.path.join(a.model_dir, "system_prompt.txt"), encoding="utf-8").read().strip()
    thinking, parsed = rewrite(a.prompt, url=a.url, model=a.model, sys_prompt=sys_prompt)
    if parsed is None:
        print("rewrite failed: no rewritten_prompt in model output", file=sys.stderr)
        sys.exit(1)

    print(json.dumps({"thinking": thinking,
                      "rewritten_prompt": parsed["rewritten_prompt"],
                      "wh_ratio": parsed["wh_ratio"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
