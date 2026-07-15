"""The llm-debate command: wire adapters + judge + storage, run one debate.

    llm-debate "Is a monolith or microservices better for a 2-person startup?" \
        --cap 10 --condition natural --max-minutes 30
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from llm_debate.adapters import ClaudeParticipant, CodexParticipant, Participant
from llm_debate.judge import CLIJudge
from llm_debate.machine import DebateRunner
from llm_debate.runlog import RunStorage, render_transcript
from llm_debate.types import RunConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-debate",
        description="Blind two-participant LLM debate over CLIs, with an invisible judge.",
    )
    parser.add_argument("prompt", help="the task both participants receive")
    parser.add_argument("--cap", type=int, default=10, help="max exchange rounds (default 10)")
    parser.add_argument(
        "--condition",
        choices=["natural", "steered"],
        default="natural",
        help="natural = no steering; steered = anti-sycophancy role prompt",
    )
    parser.add_argument(
        "--max-minutes", type=float, default=30.0, help="run-level wall-clock budget"
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="per-CLI-call timeout in seconds"
    )
    parser.add_argument(
        "--judge",
        choices=["claude", "codex"],
        default="claude",
        help="which CLI plays judge (different family than both debaters when available)",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="where run folders are written"
    )
    parser.add_argument(
        "--model-a",
        default=None,
        help="model for participant A / claude (e.g. claude-opus-4-8); default: CLI default",
    )
    parser.add_argument(
        "--model-b",
        default=None,
        help="model for participant B / codex (e.g. gpt-5.5); default: ~/.codex/config.toml",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="auto-export the run to Langfuse on completion (needs LANGFUSE_* env keys)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="model for the judge's CLI (e.g. claude-opus-4-8); default: CLI default",
    )
    return parser


def require_binaries(*names: str) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        joined = ", ".join(missing)
        sys.exit(f"error: required CLI not found on PATH: {joined}. Install and log in first.")


def cli_version(binary: str) -> str | None:
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or out.stderr.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def total_cost_usd(events_path: Path) -> float:
    total = 0.0
    if not events_path.exists():
        return total
    for line in events_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        cost = (event.get("usage") or {}).get("cost_usd")
        if isinstance(cost, int | float):
            total += float(cost)
    return total


async def amain(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_binaries("claude", "codex")

    config = RunConfig(
        cap=args.cap,
        max_minutes=args.max_minutes,
        condition=args.condition,
        per_call_timeout_s=args.timeout,
    )
    storage = RunStorage(args.runs_dir)

    participant_a = ClaudeParticipant(
        storage.scratch("participant_a"), timeout_s=args.timeout, model=args.model_a
    )
    participant_b = CodexParticipant(
        storage.scratch("participant_b"), timeout_s=args.timeout, model=args.model_b
    )
    judge_backend: Participant = (
        ClaudeParticipant(storage.scratch("judge"), timeout_s=args.timeout, model=args.judge_model)
        if args.judge == "claude"
        else CodexParticipant(
            storage.scratch("judge"), timeout_s=args.timeout, model=args.judge_model
        )
    )
    judge = CLIJudge(
        judge_backend,
        max_retries=config.max_retries,
        base_delay_s=config.retry_base_delay_s,
        sink=storage.append,
    )

    storage.write_config(
        {
            "task": args.prompt,
            "config": asdict(config),
            "participants": {"A": participant_a.name, "B": participant_b.name},
            "judge": judge.name,
            "cli_versions": {
                "claude": cli_version("claude"),
                "codex": cli_version("codex"),
            },
        }
    )

    print(f"run dir: {storage.dir}")
    print(f"debating (cap {config.cap}, budget {config.max_minutes:g} min) ...")
    runner = DebateRunner(participant_a, participant_b, judge, config, sink=storage.append)
    result = await runner.run(args.prompt)

    storage.write_status(result)
    storage.write_transcript(render_transcript(args.prompt, result))

    if args.export:
        if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
            from langfuse import Langfuse

            from llm_debate.export_langfuse import export_and_record

            client = Langfuse()
            record = export_and_record(storage.dir, client)
            client.flush()
            print(f"exported to Langfuse: {record['url'] or record['trace_id']}")
        else:
            print("(--export skipped: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set)")

    cost = total_cost_usd(storage.events_path)
    print(f"\nstatus: {result.status} after {result.rounds_completed} exchange round(s)")
    if result.records:
        final = result.records[-1].ruling
        print(f"convergence: {final.convergence_score}/100")
    if cost:
        print(f"logged cost (where CLIs expose it): ${cost:.2f}")
    if result.error:
        print(f"error: {result.error}")
    print(f"transcript: {storage.dir / 'transcript.md'}\n")
    print(result.best_answer or "(no answer — run failed before the first ruling)")
    return 0 if result.status in ("consensus", "cap_reached") else 1


def main() -> None:
    sys.exit(asyncio.run(amain()))
