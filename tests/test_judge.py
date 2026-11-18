"""CLIJudge: strict payload validation + exactly one corrective re-ask (6A)."""

import json

import pytest

from llm_debate.adapters import parse_claude_envelope
from llm_debate.judge import CLIJudge, parse_ruling
from llm_debate.mocks import MockParticipant
from llm_debate.types import CLICallError, ParseError

GOOD = json.dumps(
    {
        "verdict": "consensus",
        "convergence_score": 88,
        "best_answer": "monolith",
        "best_answer_quality": "acceptable",
        "agreement_reasons": ["both prefer simplicity"],
        "cruxes": [],
    }
)


# --- parse_ruling ------------------------------------------------------------


def test_parses_valid_payload() -> None:
    ruling = parse_ruling(GOOD)
    assert ruling.verdict == "consensus"
    assert ruling.convergence_score == 88
    assert ruling.agreement_reasons == ("both prefer simplicity",)


def test_parses_real_probe_payload() -> None:
    """Chained ground truth: real claude envelope -> inner payload -> ruling."""
    envelope_stdout = json.loads(
        (
            __import__("pathlib").Path(__file__).parent.parent
            / "probes"
            / "fixtures"
            / "judge_claude.json"
        ).read_text()
    )["stdout"]
    inner = parse_claude_envelope(envelope_stdout).answer
    # the probe schema lacked quality/reasons fields; patch minimally to v1 schema
    payload = json.loads(inner)
    payload.update({"best_answer_quality": "acceptable", "agreement_reasons": [], "cruxes": []})
    ruling = parse_ruling(json.dumps(payload))
    assert ruling.verdict == "consensus"
    assert ruling.best_answer == "4"


def test_strips_markdown_fences() -> None:
    assert parse_ruling(f"```json\n{GOOD}\n```").verdict == "consensus"


@pytest.mark.parametrize(
    "mutation",
    [
        {"verdict": "maybe"},
        {"convergence_score": "88"},
        {"convergence_score": 150},
        {"convergence_score": True},
        {"best_answer": ""},
        {"best_answer_quality": "great"},
        {"agreement_reasons": "not a list"},
    ],
)
def test_invalid_fields_raise(mutation: dict) -> None:
    payload = {**json.loads(GOOD), **mutation}
    with pytest.raises(ParseError):
        parse_ruling(json.dumps(payload))


def test_prose_raises() -> None:
    with pytest.raises(ParseError):
        parse_ruling("They basically agree, I'd say 85/100.")


# --- CLIJudge corrective re-ask ------------------------------------------------


async def test_corrective_reask_recovers() -> None:
    backend = MockParticipant("claude-cli", ["not json at all", GOOD])
    judge = CLIJudge(backend, max_retries=0, base_delay_s=0.0)
    ruling = await judge.evaluate("judge this")
    assert ruling.verdict == "consensus"
    assert len(backend.prompts_seen) == 2
    corrective_prompt = backend.prompts_seen[1][0]
    assert "could not be used" in corrective_prompt  # the parse error was fed back
    assert "not valid JSON" in corrective_prompt


async def test_second_bad_payload_fails_loud() -> None:
    backend = MockParticipant("claude-cli", ["garbage", "still garbage"])
    judge = CLIJudge(backend, max_retries=0, base_delay_s=0.0)
    with pytest.raises(CLICallError):  # ParseError is a CLICallError -> run ends in ERROR
        await judge.evaluate("judge this")
    assert len(backend.prompts_seen) == 2  # exactly ONE corrective re-ask, never a guess


async def test_judge_sink_records_calls() -> None:
    events: list[dict] = []
    backend = MockParticipant("claude-cli", [GOOD])
    judge = CLIJudge(backend, max_retries=0, base_delay_s=0.0, sink=events.append)
    await judge.evaluate("judge this")
    assert len(events) == 1
    assert events[0]["type"] == "judge_call"
    assert events[0]["corrective"] is False
    assert events[0]["prompt"] == "judge this"
