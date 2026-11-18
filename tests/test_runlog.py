"""Run storage: config at INIT, live-appended events, crash durability, transcript."""

import json
from pathlib import Path

from llm_debate.machine import DebateRunner
from llm_debate.mocks import MockJudge, MockParticipant, ruling
from llm_debate.runlog import RunStorage, render_transcript
from llm_debate.types import RunConfig

TASK = "Best first database for a side project?"
FAST = RunConfig(max_retries=0, retry_base_delay_s=0.0)


def read_events(storage: RunStorage) -> list[dict]:
    return [json.loads(line) for line in storage.events_path.read_text().splitlines()]


async def test_full_run_writes_complete_lab_notebook(tmp_path: Path) -> None:
    storage = RunStorage(tmp_path)
    storage.write_config({"task": TASK, "config": {"cap": 10}})
    judge = MockJudge([ruling("no_consensus"), ruling("consensus", best_answer="sqlite")])
    a = MockParticipant("a", ["postgres"])
    b = MockParticipant("b", ["sqlite"])
    runner = DebateRunner(a, b, judge, FAST, sink=storage.append)
    result = await runner.run(TASK)
    storage.write_status(result)
    storage.write_transcript(render_transcript(TASK, result))

    assert json.loads((storage.dir / "config.json").read_text())["task"] == TASK
    events = read_events(storage)
    types = [event["type"] for event in events]
    # 2 opening calls, ruling, 2 exchange calls, ruling, terminal
    assert types.count("participant_call") == 4
    assert types.count("judge_eval") == 2
    assert types[-1] == "terminal"
    assert all("ts" in event for event in events)
    # per-call provenance (C4): the exact prompt and raw output are on disk
    first_call = next(e for e in events if e["type"] == "participant_call")
    assert first_call["prompt"] == TASK
    assert first_call["raw_stdout"] == "{}"
    status = json.loads((storage.dir / "status.json").read_text())
    assert status["status"] == "consensus"
    transcript = (storage.dir / "transcript.md").read_text()
    assert "## Round 1" in transcript and "sqlite" in transcript


async def test_crash_midrun_preserves_completed_rounds(tmp_path: Path) -> None:
    """2A: the most interesting runs are the ones that fail — keep their data."""
    storage = RunStorage(tmp_path)
    judge = MockJudge([ruling("no_consensus")])
    flaky_b = MockParticipant("b", ["sqlite"], error_after=1)  # dies on the exchange call
    runner = DebateRunner(
        MockParticipant("a", ["postgres"]), flaky_b, judge, FAST, sink=storage.append
    )
    result = await runner.run(TASK)
    assert result.status == "error"
    events = read_events(storage)
    types = [event["type"] for event in events]
    assert "judge_eval" in types  # round 0 survived on disk
    assert "participant_call_failed" in types  # the failure itself is recorded
    assert types[-1] == "terminal"
    assert json.loads(storage.events_path.read_text().splitlines()[0])  # every line valid JSON


def test_scratch_dirs_are_isolated_and_empty(tmp_path: Path) -> None:
    storage = RunStorage(tmp_path)
    scratch = storage.scratch("participant_a")
    assert scratch.is_dir() and not any(scratch.iterdir())
    assert scratch != storage.scratch("participant_b")


def test_transcript_renders_failed_run() -> None:
    from llm_debate.types import DebateResult

    text = render_transcript(TASK, DebateResult("error", None, 0, [], error="boom"))
    assert "boom" in text and "_none (run failed early)_" in text
