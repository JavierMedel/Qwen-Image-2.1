#!/usr/bin/env python3
"""Shared answer parsing for the prompt rewrite scripts.

Both T2I and edit rewriters output a JSON object after a <think> block.
This module handles parsing for both modes:

  T2I:  {"rewritten_prompt": "...", "wh_ratio": "16:9"}
  Edit: {"rewritten_prompt": "...", "wh_ratio": "16:9", "ratio_follow": ""}
"""

from __future__ import annotations

import json
import re
from typing import Any


def split_thinking(text: str) -> tuple[str, str]:
    """Split a decoded generation into (thinking, answer).

    The chat template pre-fills ``<think>\\n`` before generation, so the decoded
    text normally starts *inside* the thinking block and closes it with
    ``</think>``.
    """
    if "</think>" in text:
        think, _, answer = text.partition("</think>")
        if "<think>" in think:
            think = think.partition("<think>")[2]
        return think.strip(), answer.strip()
    if "<think>" in text:
        return text.partition("<think>")[2].strip(), ""
    return "", text.strip()


def _iter_json_candidates(answer: str):
    """Yield candidate JSON substrings, last (most likely) first.

    Scans for balanced top-level ``{...}`` spans, ignoring braces inside JSON
    string literals.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escaped = False
    for i, ch in enumerate(answer):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append(answer[start : i + 1])
    for span in reversed(spans):
        yield span
    m = re.search(r"\{.*\}", answer, flags=re.DOTALL)
    if m and m.group(0) not in spans:
        yield m.group(0)


def parse_answer(answer: str) -> dict[str, Any]:
    """Parse the rewriter's answer into structured fields.

    Returns a dict with ``rewritten_prompt``, ``wh_ratio``, ``ratio_follow``,
    and ``parse_ok``. On failure, ``rewritten_prompt`` holds the raw answer text.
    """
    answer = (answer or "").strip()
    for candidate in _iter_json_candidates(answer):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        rewritten = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")
        if not isinstance(rewritten, str) or not rewritten.strip():
            continue
        return {
            "rewritten_prompt": rewritten.strip(),
            "wh_ratio": str(obj.get("wh_ratio") or "").strip(),
            "ratio_follow": str(obj.get("ratio_follow") or "").strip(),
            "parse_ok": True,
        }
    return {
        "rewritten_prompt": answer,
        "wh_ratio": "",
        "ratio_follow": "",
        "parse_ok": False,
    }


def build_record(case: dict[str, Any], thinking: str, answer: str) -> dict[str, Any]:
    """Build one output record from an input case and the model's generation."""
    parsed = parse_answer(answer)
    record = {
        "id": case.get("id", ""),
        "raw_prompt": case.get("prompt", ""),
        "thinking": thinking,
        "rewritten_prompt": parsed["rewritten_prompt"],
        "wh_ratio": parsed["wh_ratio"],
        "ratio_follow": parsed["ratio_follow"],
        "parse_ok": parsed["parse_ok"],
    }
    if "input_images" in case:
        record["input_images"] = case["input_images"]
    if "task_type" in case:
        record["task_type"] = case["task_type"]
    return record


def report_parse_failures(records: list[dict[str, Any]]) -> int:
    """Print a one-line parse summary. Returns the number of failures."""
    bad = [r for r in records if not r.get("parse_ok")]
    n = len(records)
    if bad:
        ids = ", ".join(str(r.get("id", "?")) for r in bad[:10])
        more = f", ... (+{len(bad) - 10})" if len(bad) > 10 else ""
        print(f"WARNING: {len(bad)}/{n} answers did not parse. ids: {ids}{more}", flush=True)
    else:
        print(f"Parsed {n}/{n} answers cleanly.", flush=True)
    return len(bad)
