"""Probe 3: how do the CLIs fail?

Verifies what call_with_retry must handle:
- bad flag -> which exit code, what stderr shape
- timeout  -> does subprocess timeout-kill work cleanly mid-generation
"""

from common import run_probe, summarize


def probe_bad_flag() -> None:
    for name, argv in [
        ("fail_claude_badflag", ["claude", "-p", "hi", "--no-such-flag"]),
        ("fail_codex_badflag", ["codex", "exec", "--no-such-flag", "hi"]),
    ]:
        result = run_probe(name, argv, timeout_s=30)
        summarize(name, result)
        print(f"  stderr head: {result['stderr'][:150]!r}")


def probe_timeout_kill() -> None:
    result = run_probe(
        "fail_claude_timeout",
        [
            "claude",
            "-p",
            "Write a detailed 2000-word essay about the history of state machines.",
            "--output-format",
            "json",
        ],
        timeout_s=5,
    )
    summarize("fail_claude_timeout", result)
    print(f"  timed_out={result['timed_out']} (expected True; proves clean kill mid-call)")


if __name__ == "__main__":
    probe_bad_flag()
    probe_timeout_kill()
