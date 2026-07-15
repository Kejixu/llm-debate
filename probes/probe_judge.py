"""Probe 2: can a judge CLI reliably emit schema-shaped JSON?

Tries a miniature judge task (two canned answers, asks for verdict JSON).
Tests gemini first if installed (different model family = judge independence),
then claude as the fallback candidate.
"""

import json
import shutil

from common import run_probe, summarize

JUDGE_PROMPT = """You are evaluating two anonymized answers to the question: "What is 2+2?"

Participant A said: "4"
Participant B said: "The answer is 4."

Respond with ONLY a JSON object, no markdown fences, matching exactly:
{"verdict": "consensus" | "no_consensus", "convergence_score": <int 0-100>,
 "best_answer": "<text>"}"""

EXPECTED_KEYS = {"verdict", "convergence_score", "best_answer"}


def check_payload(name: str, stdout: str, extract_result_field: bool) -> None:
    text = stdout
    if extract_result_field:
        try:
            text = json.loads(stdout).get("result", "")
        except json.JSONDecodeError:
            print(f"  [{name}] envelope itself not JSON")
            return
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"  [{name}] inner payload NOT valid JSON ({exc}); got: {text[:200]!r}")
        return
    missing = sorted(EXPECTED_KEYS - payload.keys()) or "none"
    print(f"  [{name}] payload keys: {sorted(payload.keys())} | missing: {missing}")
    print(f"  [{name}] payload: {payload}")


def probe_gemini() -> None:
    if not shutil.which("gemini"):
        print("[judge_gemini] gemini CLI not installed - skipping (claude fallback below)")
        return
    result = run_probe("judge_gemini", ["gemini", "-p", JUDGE_PROMPT])
    summarize("judge_gemini", result)
    if result["exit_code"] == 0:
        check_payload("judge_gemini", result["stdout"], extract_result_field=False)


def probe_claude_judge() -> None:
    result = run_probe(
        "judge_claude",
        ["claude", "-p", JUDGE_PROMPT, "--output-format", "json"],
    )
    summarize("judge_claude", result)
    if result["exit_code"] == 0:
        check_payload("judge_claude", result["stdout"], extract_result_field=True)


if __name__ == "__main__":
    probe_gemini()
    probe_claude_judge()
