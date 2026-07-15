"""Export a debate run's events.jsonl into Langfuse traces.

The event log is already trace-shaped (event sourcing pays off here):
  run                      -> one trace (id seeded by run dir name: re-export
                              updates the same trace, never duplicates)
  round                    -> a span grouping that round's calls
  participant_call         -> a generation (input=exact prompt, output=raw answer, usage+cost)
  judge_call               -> a generation under the same round (corrective re-asks marked)
  judge_eval               -> an evaluator observation + convergence/verdict scores
  participant_call_failed  -> an ERROR-level span
  terminal                 -> trace output + status metadata

Timestamps: Langfuse's OTel spans are stamped at export time; the ORIGINAL
event timestamps ride along in each observation's metadata (`ts`). Ordering
is preserved, durations are not — events.jsonl remains the source of truth.

Usage:
    LANGFUSE_PUBLIC_KEY=pk-... LANGFUSE_SECRET_KEY=sk-... \
        uv run llm-debate-export [runs/<id>]   # default: latest run
"""

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langfuse import Langfuse, propagate_attributes


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    config = json.loads((run_dir / "config.json").read_text())
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return config, events


def usage_details(usage: dict | None) -> dict[str, int]:
    """Map our Usage fields to Langfuse's token-count dict; absent stays absent."""
    if not usage:
        return {}
    mapping = {
        "input": usage.get("input_tokens"),
        "output": usage.get("output_tokens"),
        "cache_read_input_tokens": usage.get("cached_input_tokens"),
        "reasoning_tokens": usage.get("reasoning_output_tokens"),
    }
    details = {key: value for key, value in mapping.items() if isinstance(value, int)}
    if "input" in details and "output" in details:
        details["total"] = details["input"] + details["output"]
    return details


def cost_details(usage: dict | None) -> dict[str, float]:
    cost = (usage or {}).get("cost_usd")
    return {"total": float(cost)} if isinstance(cost, int | float) else {}


def final_answer(events: list[dict]) -> str | None:
    rulings = [event["ruling"] for event in events if event["type"] == "judge_eval"]
    return rulings[-1]["best_answer"] if rulings else None


def task_session_id(task: str) -> str:
    """Stable slug so all runs of the SAME prompt group into one Langfuse session."""
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")
    return f"task:{slug[:80]}"


def real_model(event: dict) -> str | None:
    """Best-available model id for a call, in preference order:
    the CLI's own envelope (claude emits modelUsage — provenance pays off),
    then a pinned model in the adapter name (claude-cli:claude-opus-4-8),
    then None (adapter name alone is NOT a model; it stays in metadata)."""
    raw = event.get("raw_stdout", "")
    if raw.startswith("{"):
        try:
            models = list(json.loads(raw).get("modelUsage", {}))
            if models:
                return ",".join(models)
        except json.JSONDecodeError:
            pass
    name = event.get("participant") or event.get("judge") or ""
    if ":" in name:
        return name.split(":", 1)[1].removeprefix("judge-")
    return None


def chat_input(prompt: str, role: str | None) -> list[dict]:
    """ChatML shape — Langfuse renders message lists far better than raw strings."""
    messages = []
    if role:
        messages.append({"role": "system", "content": role})
    messages.append({"role": "user", "content": prompt})
    return messages


def export_run(run_dir: Path, client: Any) -> str:
    """Walk one run's events into a Langfuse trace. Returns the trace id."""
    config, events = load_run(run_dir)
    task = config.get("task", run_dir.name)
    trace_id = client.create_trace_id(seed=run_dir.name)
    condition = (config.get("config") or {}).get("condition", "natural")
    status = next((e["status"] for e in events if e["type"] == "terminal"), "incomplete")

    # Best practices: low-cardinality trace name ("debate", never the task);
    # session groups every run of the SAME prompt; tags make condition/status
    # filterable in the UI.
    with (
        propagate_attributes(
            trace_name="debate",
            session_id=task_session_id(task),
            tags=[f"condition:{condition}", f"status:{status}"],
        ),
        client.start_as_current_observation(
            trace_context={"trace_id": trace_id},
            name="debate",
            as_type="agent",
            input=task,
            metadata={"run_dir": run_dir.name, **config},
        ) as root,
    ):
        round_span: Any = None
        round_number: int | None = None

        def close_round() -> None:
            nonlocal round_span, round_number
            if round_span is not None:
                round_span.end()
                round_span, round_number = None, None

        def ensure_round(number: int) -> Any:
            nonlocal round_span, round_number
            if round_span is None or round_number != number:
                close_round()
                round_span = root.start_observation(name=f"round {number}", as_type="span")
                round_number = number
            return round_span

        for event in events:
            kind = event["type"]
            if kind == "participant_call":
                parent = ensure_round(event["round"])
                parent.start_observation(
                    name=f"{event['label']}: {event['participant']}",
                    as_type="generation",
                    input=chat_input(event["prompt"], event.get("role")),
                    output=event["answer_raw"],
                    model=real_model(event),
                    usage_details=usage_details(event.get("usage")),
                    cost_details=cost_details(event.get("usage")),
                    metadata={
                        "ts": event.get("ts"),
                        "adapter": event["participant"],
                        "cli_session_id": event.get("session_id"),
                    },
                ).end()
            elif kind == "judge_call":
                parent = round_span or root
                name = "judge (corrective re-ask)" if event.get("corrective") else "judge"
                parent.start_observation(
                    name=name,
                    as_type="generation",
                    input=chat_input(event["prompt"], None),
                    output=event["answer_raw"],
                    model=real_model(event),
                    usage_details=usage_details(event.get("usage")),
                    cost_details=cost_details(event.get("usage")),
                    metadata={
                        "ts": event.get("ts"),
                        "adapter": event.get("judge"),
                        "cli_session_id": event.get("session_id"),
                    },
                ).end()
            elif kind == "judge_eval":
                parent = ensure_round(event["round"])
                ruling = event["ruling"]
                evaluation = parent.start_observation(
                    name="ruling",
                    as_type="evaluator",
                    input=event.get("answers_sanitized"),
                    output=ruling,
                    metadata={"ts": event.get("ts")},
                )
                evaluation.score(
                    name="convergence_score",
                    value=ruling["convergence_score"],
                    comment=f"round {event['round']}",
                )
                evaluation.score(
                    name="verdict",
                    value=ruling["verdict"],
                    data_type="CATEGORICAL",
                    comment=f"round {event['round']}",
                )
                evaluation.end()
                close_round()
            elif kind == "participant_call_failed":
                parent = round_span or root
                parent.start_observation(
                    name=f"{event['label']}: {event['participant']} FAILED",
                    as_type="span",
                    level="ERROR",
                    status_message=event.get("error"),
                    metadata={"ts": event.get("ts")},
                ).end()
            elif kind == "terminal":
                close_round()
                is_error = event["status"] == "error"
                root.update(
                    output=final_answer(events),
                    level="ERROR" if is_error else "DEFAULT",
                    status_message=event.get("error") or event["status"],
                    metadata={"status": event["status"], "rounds": event["rounds_completed"]},
                )
        close_round()
        rulings = [e["ruling"] for e in events if e["type"] == "judge_eval"]
        if rulings:
            root.score_trace(name="convergence_final", value=rulings[-1]["convergence_score"])
            root.score_trace(
                name="verdict_final", value=rulings[-1]["verdict"], data_type="CATEGORICAL"
            )
        root.score_trace(name="status", value=status, data_type="CATEGORICAL")
        client.set_current_trace_io(input=task, output=final_answer(events))
    return trace_id


def export_and_record(run_dir: Path, client: Any) -> dict:
    """Export a run AND persist trace.json — the single entry point every
    caller (CLI --export, UI auto-export, re-export endpoint) goes through,
    so the trace link is always on disk where the UI can read it."""
    trace_id = export_run(run_dir, client)
    try:
        url = client.get_trace_url(trace_id=trace_id)
    except Exception:
        url = None
    record = {
        "trace_id": trace_id,
        "url": url,
        "exported_at": datetime.now(UTC).isoformat(),
    }
    (run_dir / "trace.json").write_text(json.dumps(record, indent=2))
    return record


def latest_run(runs_base: Path) -> Path:
    candidates = sorted(d for d in runs_base.iterdir() if (d / "events.jsonl").exists())
    if not candidates:
        sys.exit(f"error: no runs with events.jsonl under {runs_base}/")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-debate-export", description="Export a debate run to Langfuse."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=None,
        help="run directory (default: latest under runs/)",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    missing = [k for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY") if not os.environ.get(k)]
    if missing:
        sys.exit(
            f"error: {', '.join(missing)} not set. Create a project at cloud.langfuse.com, "
            "copy the API keys from Settings, then:\n"
            "  export LANGFUSE_PUBLIC_KEY=pk-lf-...\n"
            "  export LANGFUSE_SECRET_KEY=sk-lf-...\n"
            "  export LANGFUSE_HOST=https://cloud.langfuse.com   # or https://us.cloud.langfuse.com"
        )

    run_dir = args.run_dir or latest_run(args.runs_dir)
    client = Langfuse()
    record = export_and_record(run_dir, client)
    client.flush()
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    print(f"exported {run_dir.name} -> trace {record['trace_id']}")
    print(
        record["url"]
        or f"view it in your Langfuse project on {host} (search trace id {record['trace_id']})"
    )
