"""Run storage: the lab notebook. Local files only, appended as things happen.

runs/<timestamp-id>/
  config.json    the full lab setup, written at INIT (review 7A)
  events.jsonl   one JSON line per event, appended + flushed live (2A, C4)
  status.json    terminal status marker: completed-vs-aborted is explicit
  transcript.md  human-readable rendering, derived FROM the records at the end
  scratch/       empty per-caller cwds (agent CLIs read instruction files from cwd)
"""

import json
import secrets
import time
from datetime import datetime
from pathlib import Path

from llm_debate.types import DebateResult


class RunStorage:
    def __init__(self, runs_base: Path) -> None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        self.dir = runs_base / run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.dir / "events.jsonl"

    def scratch(self, name: str) -> Path:
        """An isolated, empty cwd for one CLI caller."""
        path = self.dir / "scratch" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_config(self, payload: dict) -> None:
        (self.dir / "config.json").write_text(json.dumps(payload, indent=2))

    def append(self, event: dict) -> None:
        """Append one event line and flush — a crash keeps everything before it."""
        stamped = {"ts": time.time(), **event}
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(stamped) + "\n")
            handle.flush()

    def write_status(self, result: DebateResult) -> None:
        (self.dir / "status.json").write_text(
            json.dumps(
                {
                    "status": result.status,
                    "rounds_completed": result.rounds_completed,
                    "error": result.error,
                },
                indent=2,
            )
        )

    def write_transcript(self, text: str) -> None:
        (self.dir / "transcript.md").write_text(text)


def render_transcript(task: str, result: DebateResult) -> str:
    """Human-readable view, derived from the per-round records.

    Shows the SANITIZED answers (what actually circulated in the debate);
    the raw truth stays in events.jsonl.
    """
    lines = [
        "# llm-debate transcript",
        "",
        f"**Task:** {task}",
        "",
        f"**Outcome:** {result.status} after {result.rounds_completed} exchange round(s)",
    ]
    if result.error:
        lines.append(f"**Error:** {result.error}")
    for record in result.records:
        ruling = record.ruling
        lines += [
            "",
            f"## Round {record.round}",
            "",
            f"**Participant A:** {record.answers_sanitized['A']}",
            "",
            f"**Participant B:** {record.answers_sanitized['B']}",
            "",
            f"**Judge:** {ruling.verdict} (convergence {ruling.convergence_score}/100, "
            f"best answer quality: {ruling.best_answer_quality})",
        ]
        if ruling.agreement_reasons:
            lines.append(f"- Agreement: {'; '.join(ruling.agreement_reasons)}")
        if ruling.cruxes:
            lines.append(f"- Cruxes: {'; '.join(ruling.cruxes)}")
    lines += ["", "## Final answer", "", result.best_answer or "_none (run failed early)_", ""]
    return "\n".join(lines)
