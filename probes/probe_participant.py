"""Probe 1: what do the participant CLIs actually emit in structured-output mode?

Verifies the eng-review 1A assumption against reality:
- claude -p --output-format json  -> single JSON envelope? which fields?
- codex exec --json               -> envelope or JSONL event stream?
- codex exec --output-last-message-> plain answer in a file?
"""

import json
from pathlib import Path

from common import SCRATCH, run_probe, summarize

PROMPT = "Reply with exactly the text PROBE_OK and nothing else."


def probe_claude() -> None:
    result = run_probe(
        "claude_participant",
        ["claude", "-p", PROMPT, "--output-format", "json"],
    )
    summarize("claude_participant", result)
    if result["exit_code"] == 0:
        try:
            envelope = json.loads(result["stdout"])
            print("  ENVELOPE: single JSON object with keys:")
            for key in sorted(envelope):
                print(f"    {key} = {str(envelope[key])[:80]!r}")
        except json.JSONDecodeError as exc:
            print(f"  NOT a single JSON object ({exc}); first 200 chars:")
            print(f"    {result['stdout'][:200]!r}")


def probe_codex_json() -> None:
    result = run_probe(
        "codex_participant_json",
        ["codex", "exec", "--json", "-s", "read-only", PROMPT],
    )
    summarize("codex_participant_json", result)
    if result["exit_code"] == 0:
        lines = [ln for ln in result["stdout"].splitlines() if ln.strip()]
        parsed, bad = [], 0
        for ln in lines:
            try:
                parsed.append(json.loads(ln))
            except json.JSONDecodeError:
                bad += 1
        print(f"  {len(lines)} stdout lines: {len(parsed)} valid JSON, {bad} non-JSON")
        types = [str(p.get("type") or p.get("msg", {}).get("type", "?")) for p in parsed]
        print(f"  event types seen: {sorted(set(types))}")


def probe_codex_last_message() -> None:
    out_file = SCRATCH / "codex_participant_last" / "last_message.txt"
    result = run_probe(
        "codex_participant_last",
        [
            "codex",
            "exec",
            "-s",
            "read-only",
            "--output-last-message",
            str(out_file),
            PROMPT,
        ],
    )
    summarize("codex_participant_last", result)
    if out_file.exists():
        print(f"  last-message file: {out_file.read_text()[:120]!r}")
    else:
        print("  last-message file NOT created")


if __name__ == "__main__":
    Path(SCRATCH).mkdir(exist_ok=True)
    probe_claude()
    probe_codex_json()
    probe_codex_last_message()
