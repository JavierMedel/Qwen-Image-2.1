#!/usr/bin/env python3
"""Shared answer parsing and output-record construction for the edit-PE scripts.

The expert's answer section is a JSON object with three fields that the system
prompt (`prompts/system_prompt.txt`) declares mandatory:

    {"rewritten_prompt": "...", "wh_ratio": "16:9", "ratio_follow": ""}

All three matter downstream: `rewritten_prompt` is the prompt you render, and
`wh_ratio` / `ratio_follow` decide the output canvas that the downstream editor
should render on.
`wh_ratio` and `ratio_follow` are mutually exclusive -- exactly one carries a
value; the other is "".

Both reference runners import from here so their records stay identical.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Stable output field order. `parse_ok` is last so that appending future fields
# never moves it.
OUTPUT_FIELDS = (
    "id",
    "raw_prompt",
    "input_images",
    "task_type",
    "thinking",
    "positive_prompt",
    "negative_prompt",
    "wh_ratio",
    "ratio_follow",
    "parse_ok",
)


def split_thinking(text: str) -> tuple[str, str]:
    """Split a decoded generation into (thinking, answer).

    The chat template pre-fills ``<think>\\n`` before generation, so the decoded
    text normally starts *inside* the thinking block and closes it with
    ``</think>``. The ``<think>``-present branch covers backends that echo the
    prefill (vLLM's ``llm.chat`` does not, but a custom template may).
    """
    if "</think>" in text:
        think, _, answer = text.partition("</think>")
        if "<think>" in think:
            think = think.partition("<think>")[2]
        return think.strip(), answer.strip()
    if "<think>" in text:
        # Unterminated thinking block -- the generation hit the token budget.
        return text.partition("<think>")[2].strip(), ""
    return "", text.strip()


def _iter_json_candidates(answer: str):
    """Yield candidate JSON substrings of ``answer``, most likely first.

    A single greedy ``\\{.*\\}`` span is not enough: any brace in prose after the
    object (or a second object) stretches the match past the real end and the
    parse fails silently. This scans for balanced top-level ``{...}`` spans
    instead, ignoring braces inside JSON string literals, and yields the last
    one first -- the answer object is emitted at the end of the answer section.
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
    # Last resort: an object whose closing brace was never generated (truncation)
    # or one nested in a way the scanner did not close. Greedy span, as before.
    m = re.search(r"\{.*\}", answer, flags=re.DOTALL)
    if m and m.group(0) not in spans:
        yield m.group(0)


def parse_answer(answer: str) -> dict[str, Any]:
    """Parse the expert's answer section into the three declared fields.

    Returns ``{"positive_prompt", "wh_ratio", "ratio_follow", "parse_ok"}``.

    On any parse failure ``positive_prompt`` falls back to the raw answer text so
    the model's output is never lost -- but ``parse_ok`` is then ``False``, which
    is the only way to tell a fallback from a clean parse once the record is on
    disk. Audit it: ``jq -s 'map(select(.parse_ok|not))|length' out.jsonl``.
    """
    answer = (answer or "").strip()
    for candidate in _iter_json_candidates(answer):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        # Some training runs mis-typed the key as `rewrited_prompt`; accept both.
        rewritten = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")
        if not isinstance(rewritten, str) or not rewritten.strip():
            continue
        return {
            "positive_prompt": rewritten.strip(),
            "wh_ratio": str(obj.get("wh_ratio") or "").strip(),
            "ratio_follow": str(obj.get("ratio_follow") or "").strip(),
            "parse_ok": True,
        }
    return {
        "positive_prompt": answer,
        "wh_ratio": "",
        "ratio_follow": "",
        "parse_ok": False,
    }


def build_record(case: dict[str, Any], thinking: str, answer: str) -> dict[str, Any]:
    """Build one output record with fields in ``OUTPUT_FIELDS`` order."""
    parsed = parse_answer(answer)
    record = {
        "id": case["id"],
        "raw_prompt": case["prompt"],
        "input_images": case["input_images"],
        "task_type": case.get("task_type", ""),
        "thinking": thinking,
        "positive_prompt": parsed["positive_prompt"],
        "negative_prompt": "",
        "wh_ratio": parsed["wh_ratio"],
        "ratio_follow": parsed["ratio_follow"],
        "parse_ok": parsed["parse_ok"],
    }
    return {k: record[k] for k in OUTPUT_FIELDS}


def report_parse_failures(records: list[dict[str, Any]]) -> int:
    """Print a one-line parse summary. Returns the number of failures.

    A batch that silently falls back on 5% of its rows looks exactly like a
    clean batch in the output file, so say it out loud at the end of the run.
    """
    bad = [r for r in records if not r.get("parse_ok")]
    n = len(records)
    if bad:
        ids = ", ".join(str(r["id"]) for r in bad[:10])
        more = f", ... (+{len(bad) - 10})" if len(bad) > 10 else ""
        print(
            f"WARNING: {len(bad)}/{n} answers did not parse as the expected JSON "
            f"object; positive_prompt fell back to the raw answer text and "
            f"wh_ratio/ratio_follow are empty for them. ids: {ids}{more}",
            flush=True,
        )
    else:
        print(f"Parsed {n}/{n} answers cleanly (parse_ok=true).", flush=True)
    return len(bad)
