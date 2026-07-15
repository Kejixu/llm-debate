"""Exporter mapping: events.jsonl -> Langfuse trace, verified against a fake client."""

import json
from pathlib import Path

from llm_debate.export_langfuse import (
    chat_input,
    cost_details,
    export_run,
    final_answer,
    real_model,
    task_session_id,
    usage_details,
)

# --- pure helpers -------------------------------------------------------------


def test_usage_details_drops_absent_fields() -> None:
    assert usage_details({"input_tokens": 100, "output_tokens": None, "cost_usd": 0.2}) == {
        "input": 100
    }
    assert usage_details(None) == {}


def test_cost_details_absent_vs_zero() -> None:
    assert cost_details({"cost_usd": 0.19}) == {"total": 0.19}
    assert cost_details({"cost_usd": None}) == {}  # codex: absence stays absent


# --- fake client --------------------------------------------------------------


class FakeObservation:
    def __init__(self, log: list, kind: str, **kwargs) -> None:
        self.log = log
        self.kwargs = {"as_type": kind, **kwargs}
        log.append(("start", self.kwargs))

    def start_observation(self, *, name: str, as_type: str = "span", **kwargs):
        return FakeObservation(self.log, as_type, name=name, **kwargs)

    def score(self, **kwargs) -> None:
        self.log.append(("score", kwargs))

    def update(self, **kwargs) -> None:
        self.log.append(("update", kwargs))

    def score_trace(self, **kwargs) -> None:
        self.log.append(("score_trace", kwargs))

    def end(self, **kwargs) -> None:
        self.log.append(("end", self.kwargs.get("name")))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.log.append(("end", self.kwargs.get("name")))


class FakeLangfuse:
    def __init__(self) -> None:
        self.log: list = []

    def create_trace_id(self, *, seed: str) -> str:
        return f"trace-{seed}"

    def start_as_current_observation(self, **kwargs):
        return FakeObservation(self.log, kwargs.pop("as_type", "span"), **kwargs)

    def set_current_trace_io(self, **kwargs) -> None:
        self.log.append(("trace_io", kwargs))


def write_run(tmp_path: Path, events: list[dict]) -> Path:
    run = tmp_path / "20260713-000000-abc123"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"task": "pick a database"}))
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    return run


RULING = {
    "verdict": "consensus",
    "convergence_score": 90,
    "best_answer": "sqlite",
    "best_answer_quality": "acceptable",
    "agreement_reasons": [],
    "cruxes": [],
}


def sample_events() -> list[dict]:
    return [
        {
            "ts": 1.0,
            "type": "participant_call",
            "round": 0,
            "label": "A",
            "participant": "claude-cli",
            "prompt": "pick a database",
            "role": None,
            "answer_raw": "postgres",
            "usage": {"input_tokens": 10, "cost_usd": 0.2},
            "session_id": "s1",
            "raw_stdout": "{}",
        },
        {
            "ts": 2.0,
            "type": "participant_call",
            "round": 0,
            "label": "B",
            "participant": "codex-cli",
            "prompt": "pick a database",
            "role": None,
            "answer_raw": "sqlite",
            "usage": {"input_tokens": 12, "cost_usd": None},
            "session_id": "s2",
            "raw_stdout": "{}",
        },
        {
            "ts": 3.0,
            "type": "judge_call",
            "judge": "judge-claude-cli",
            "corrective": False,
            "prompt": "judge this",
            "answer_raw": "{...}",
            "usage": {},
            "session_id": "s3",
            "raw_stdout": "{}",
        },
        {
            "ts": 4.0,
            "type": "judge_eval",
            "round": 0,
            "answers_raw": {"A": "postgres", "B": "sqlite"},
            "answers_sanitized": {"A": "postgres", "B": "sqlite"},
            "ruling": RULING,
        },
        {
            "ts": 5.0,
            "type": "terminal",
            "status": "consensus",
            "rounds_completed": 0,
            "error": None,
        },
    ]


def test_export_builds_expected_trace_shape(tmp_path: Path) -> None:
    run = write_run(tmp_path, sample_events())
    client = FakeLangfuse()
    trace_id = export_run(run, client)

    assert trace_id == f"trace-{run.name}"  # deterministic: re-export updates, not duplicates
    starts = [entry[1] for entry in client.log if entry[0] == "start"]
    generations = [s for s in starts if s["as_type"] == "generation"]
    assert len(generations) == 3  # A, B, judge
    a_gen = next(s for s in generations if s.get("name") == "A: claude-cli")
    assert a_gen["input"] == [{"role": "user", "content": "pick a database"}]  # chatml
    assert a_gen["cost_details"] == {"total": 0.2}
    assert a_gen["metadata"]["adapter"] == "claude-cli"
    b_gen = next(s for s in generations if s.get("name") == "B: codex-cli")
    assert b_gen["cost_details"] == {}  # absence preserved

    scores = [entry[1] for entry in client.log if entry[0] == "score"]
    assert {s["name"] for s in scores} == {"convergence_score", "verdict"}
    trace_scores = [entry[1] for entry in client.log if entry[0] == "score_trace"]
    assert {s["name"] for s in trace_scores} == {"convergence_final", "verdict_final", "status"}

    trace_io = next(entry[1] for entry in client.log if entry[0] == "trace_io")
    assert trace_io["output"] == "sqlite"


def test_export_error_run_marks_error_level(tmp_path: Path) -> None:
    events = [
        *sample_events()[:2],
        {
            "ts": 3.0,
            "type": "participant_call_failed",
            "round": 1,
            "label": "B",
            "participant": "codex-cli",
            "error": "timed out",
        },
        {
            "ts": 4.0,
            "type": "terminal",
            "status": "error",
            "rounds_completed": 1,
            "error": "timed out",
        },
    ]
    run = write_run(tmp_path, events)
    client = FakeLangfuse()
    export_run(run, client)

    starts = [entry[1] for entry in client.log if entry[0] == "start"]
    failed = next(s for s in starts if "FAILED" in str(s.get("name")))
    assert failed["level"] == "ERROR"
    updates = [entry[1] for entry in client.log if entry[0] == "update"]
    assert any(u.get("level") == "ERROR" for u in updates)
    trace_io = next(entry[1] for entry in client.log if entry[0] == "trace_io")
    assert trace_io["output"] is None  # no ruling ever happened


def test_final_answer_none_without_rulings() -> None:
    assert final_answer([{"type": "terminal", "status": "error"}]) is None


def test_usage_details_adds_total() -> None:
    assert usage_details({"input_tokens": 10, "output_tokens": 5}) == {
        "input": 10,
        "output": 5,
        "total": 15,
    }


def test_task_session_id_is_stable_slug() -> None:
    a = task_session_id("Is TDD worth it, for solo devs?")
    assert a == task_session_id("Is TDD worth it, for solo devs?")
    assert a == "task:is-tdd-worth-it-for-solo-devs"


def test_real_model_prefers_envelope_then_pinned_name() -> None:
    envelope = json.dumps({"modelUsage": {"claude-fable-5": {}}})
    assert real_model({"raw_stdout": envelope, "participant": "claude-cli"}) == "claude-fable-5"
    assert real_model({"raw_stdout": "not json", "participant": "codex-cli:gpt-5.5"}) == "gpt-5.5"
    assert real_model({"raw_stdout": "{}", "participant": "codex-cli"}) is None
    assert (
        real_model({"raw_stdout": "{}", "judge": "judge-claude-cli:claude-opus-4-8"})
        == "claude-opus-4-8"
    )


def test_chat_input_shapes() -> None:
    assert chat_input("q", None) == [{"role": "user", "content": "q"}]
    assert chat_input("q", "be rigorous") == [
        {"role": "system", "content": "be rigorous"},
        {"role": "user", "content": "q"},
    ]


def test_export_and_record_writes_trace_json(tmp_path: Path) -> None:
    from llm_debate.export_langfuse import export_and_record

    run = write_run(tmp_path, sample_events())
    client = FakeLangfuse()
    client.get_trace_url = lambda *, trace_id: f"https://cloud.example/t/{trace_id}"  # type: ignore
    record = export_and_record(run, client)

    on_disk = json.loads((run / "trace.json").read_text())
    assert on_disk == record
    assert record["trace_id"] == f"trace-{run.name}"
    assert record["url"] == f"https://cloud.example/t/trace-{run.name}"
    assert record["exported_at"]  # ISO timestamp present


def test_export_and_record_tolerates_missing_url(tmp_path: Path) -> None:
    from llm_debate.export_langfuse import export_and_record

    run = write_run(tmp_path, sample_events())
    record = export_and_record(run, FakeLangfuse())  # FakeLangfuse has no get_trace_url
    assert record["url"] is None
    assert json.loads((run / "trace.json").read_text())["trace_id"] == record["trace_id"]
