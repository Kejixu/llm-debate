"""The statechart under mocks: every terminal, the exit predicate, concurrency.

Zero CLI calls — MockParticipant/MockJudge implement the same Protocols as
the real backends, so these tests prove the orchestrator, deterministically.
"""

import asyncio
from collections.abc import Callable

import pytest

from llm_debate.machine import DebateRunner
from llm_debate.mocks import MockJudge, MockParticipant, ruling
from llm_debate.types import CLICallError, Condition, RunConfig

TASK = "Is a monolith or microservices better for a two-person startup?"

FAST = RunConfig(max_retries=0, retry_base_delay_s=0.0)


def make_runner(
    judge: MockJudge,
    config: RunConfig = FAST,
    a: MockParticipant | None = None,
    b: MockParticipant | None = None,
) -> tuple[DebateRunner, MockParticipant, MockParticipant]:
    a = a or MockParticipant("a", ["monolith", "monolith, still"])
    b = b or MockParticipant("b", ["microservices", "ok, monolith"])
    return DebateRunner(a, b, judge, config), a, b


# --- terminals ---------------------------------------------------------------


async def test_consensus_at_round_zero() -> None:
    judge = MockJudge([ruling("consensus", best_answer="monolith", score=95)])
    runner, a, _ = make_runner(judge)
    result = await runner.run(TASK)
    assert result.status == "consensus"
    assert result.best_answer == "monolith"
    assert result.rounds_completed == 0
    assert len(result.records) == 1
    assert len(a.prompts_seen) == 1  # only the blind opening


async def test_consensus_after_exchanges() -> None:
    judge = MockJudge([ruling("no_consensus"), ruling("no_consensus"), ruling("consensus")])
    runner, a, _ = make_runner(judge)
    result = await runner.run(TASK)
    assert result.status == "consensus"
    assert result.rounds_completed == 2
    assert [r.round for r in result.records] == [0, 1, 2]  # round stamped on every record
    assert len(a.prompts_seen) == 3  # opening + 2 exchanges


async def test_consensus_verdict_with_weak_quality_continues() -> None:
    """THE AND-predicate: agreement alone must never end a debate (Premise 1)."""
    judge = MockJudge(
        [ruling("consensus", quality="weak"), ruling("consensus", quality="acceptable")]
    )
    runner, _, _ = make_runner(judge)
    result = await runner.run(TASK)
    assert result.rounds_completed == 1  # weak-quality consensus forced one more round
    assert result.status == "consensus"


async def test_cap_reached_when_judge_never_rules_consensus() -> None:
    judge = MockJudge([ruling("no_consensus", best_answer="monolith (judge pick)")])
    runner, a, _ = make_runner(judge, config=RunConfig(cap=3, max_retries=0))
    result = await runner.run(TASK)
    assert result.status == "cap_reached"
    assert result.rounds_completed == 3
    assert result.best_answer == "monolith (judge pick)"  # judge-selected, with no-consensus note
    assert len(a.prompts_seen) == 4  # opening + 3 exchanges


class ClockAdvancingJudge(MockJudge):
    """Advances a fake clock after ruling — simulates a slow debate."""

    def __init__(self, rulings: list, advance: "Callable[[], None]") -> None:
        super().__init__(rulings)
        self._advance = advance

    async def evaluate(self, prompt: str):
        result = await super().evaluate(prompt)
        self._advance()
        return result


async def test_budget_exceeded_is_distinct_from_cap() -> None:
    """C1: a timeout is an interrupted measurement, never a scientific result."""
    fake_now = 0.0

    def clock() -> float:
        return fake_now

    def advance() -> None:
        nonlocal fake_now
        fake_now = 120.0  # past the 60s deadline before the machine rules

    judge = ClockAdvancingJudge([ruling("no_consensus")], advance)
    a = MockParticipant("a", ["monolith"])
    b = MockParticipant("b", ["microservices"])
    runner = DebateRunner(a, b, judge, RunConfig(max_minutes=1.0, max_retries=0), clock=clock)
    result = await runner.run(TASK)
    assert result.status == "budget_exceeded"
    assert result.rounds_completed == 0
    assert len(result.records) == 1  # the completed round survived (2A)


async def test_context_exceeded_on_oversized_exchange() -> None:
    judge = MockJudge([ruling("no_consensus")])
    a = MockParticipant("a", ["long " * 50])
    b = MockParticipant("b", ["also long " * 50])
    runner = DebateRunner(a, b, judge, RunConfig(max_prompt_chars=300, max_retries=0))
    result = await runner.run(TASK)
    assert result.status == "context_exceeded"
    assert len(result.records) == 1  # round 0 data preserved, no summarization (C3)


# --- error paths --------------------------------------------------------------


async def test_participant_failure_after_retries_is_error_with_partial_data() -> None:
    judge = MockJudge([ruling("no_consensus")])
    flaky_b = MockParticipant("b", ["microservices"], error_after=1)  # fails on exchange
    runner, _, _ = make_runner(judge, b=flaky_b)
    result = await runner.run(TASK)
    assert result.status == "error"
    assert result.error is not None and "scripted failure" in result.error
    assert len(result.records) == 1  # round 0 survived the crash (2A)


async def test_judge_failure_is_error() -> None:
    judge = MockJudge([], error=CLICallError("judge JSON bad twice"))
    runner, _, _ = make_runner(judge)
    result = await runner.run(TASK)
    assert result.status == "error"
    assert result.records == []
    assert result.best_answer is None


async def test_retry_recovers_transient_participant_failure() -> None:
    judge = MockJudge([ruling("consensus")])
    # error_after=0: first call fails, retry succeeds (index 1 > 0 still fails...
    # so allow exactly one failure by scripting error on call 0 only via barrierless flake)
    flaky = FlakyOnce("a", ["monolith"])
    runner = DebateRunner(
        flaky,
        MockParticipant("b", ["microservices"]),
        judge,
        RunConfig(max_retries=1, retry_base_delay_s=0.0),
    )
    result = await runner.run(TASK)
    assert result.status == "consensus"
    assert flaky.calls == 2  # failed once, retried, succeeded


class FlakyOnce(MockParticipant):
    """Fails exactly the first call, then behaves."""

    def __init__(self, name: str, answers: list[str]) -> None:
        super().__init__(name, answers)
        self.calls = 0

    async def answer(self, prompt: str, role: str | None = None):
        self.calls += 1
        if self.calls == 1:
            raise CLICallError("transient")
        return await super().answer(prompt, role)


# --- invariants at the runner level -------------------------------------------


async def test_debater_calls_run_concurrently() -> None:
    """A 2-party barrier deadlocks unless both debaters are in flight at once (9A)."""
    barrier = asyncio.Barrier(2)
    a = MockParticipant("a", ["monolith"], barrier=barrier)
    b = MockParticipant("b", ["microservices"], barrier=barrier)
    judge = MockJudge([ruling("consensus")])
    runner = DebateRunner(a, b, judge, FAST)
    result = await asyncio.wait_for(runner.run(TASK), timeout=2.0)
    assert result.status == "consensus"


async def test_debaters_never_see_round_judge_or_identities() -> None:
    judge = MockJudge([ruling("no_consensus"), ruling("consensus")])
    a = MockParticipant("a", ["As Claude, I say monolith."])
    b = MockParticipant("b", ["ChatGPT here: microservices."])
    runner = DebateRunner(a, b, judge, FAST)
    await runner.run(TASK)
    for participant in (a, b):
        for prompt, _role in participant.prompts_seen:
            lowered = prompt.lower()
            for forbidden in ("round", "judge", "claude", "chatgpt", "codex"):
                assert forbidden not in lowered, f"{forbidden!r} leaked to a debater"


async def test_only_sanitized_text_circulates_raw_lives_in_log() -> None:
    judge = MockJudge([ruling("no_consensus"), ruling("consensus")])
    a = MockParticipant("a", ["As Claude, I say monolith."])
    b = MockParticipant("b", ["microservices"])
    runner = DebateRunner(a, b, judge, FAST)
    result = await runner.run(TASK)
    exchange_prompt_b = b.prompts_seen[1][0]
    assert "[model-name-redacted]" in exchange_prompt_b  # opponent view sanitized
    assert "Claude" not in exchange_prompt_b
    # the raw truth is preserved in the record for research (4A)
    assert result.records[0].answers_raw["A"] == "As Claude, I say monolith."


async def test_judge_prompt_carries_round_and_sanitized_answers() -> None:
    judge = MockJudge([ruling("no_consensus"), ruling("consensus")])
    a = MockParticipant("a", ["As Claude: monolith"])
    runner, _, _ = make_runner(judge, a=a)
    await runner.run(TASK)
    assert "round 0" in judge.prompts_seen[0]
    assert "round 1" in judge.prompts_seen[1]
    assert "Claude" not in judge.prompts_seen[0]  # judge is blind to identities too


async def test_steered_condition_passes_role_to_debaters_only() -> None:
    judge = MockJudge([ruling("consensus")])
    a = MockParticipant("a", ["monolith"])
    b = MockParticipant("b", ["microservices"])
    runner = DebateRunner(a, b, judge, RunConfig(condition="steered", max_retries=0))
    await runner.run(TASK)
    assert a.prompts_seen[0][1] is not None  # role delivered
    assert b.prompts_seen[0][1] is not None


@pytest.mark.parametrize("condition", ["natural", "steered"])
async def test_natural_and_steered_prompts_identical(condition: Condition) -> None:
    judge = MockJudge([ruling("no_consensus"), ruling("consensus")])
    a = MockParticipant("a", ["monolith"])
    b = MockParticipant("b", ["microservices"])
    config = RunConfig(condition=condition, max_retries=0)
    runner = DebateRunner(a, b, judge, config)
    await runner.run(TASK)
    # collect prompts; compare across the parametrized runs via a module-level stash
    PROMPTS_BY_CONDITION[condition] = [p for p, _ in a.prompts_seen]
    if len(PROMPTS_BY_CONDITION) == 2:
        assert PROMPTS_BY_CONDITION["natural"] == PROMPTS_BY_CONDITION["steered"]


PROMPTS_BY_CONDITION: dict[str, list[str]] = {}


async def test_sink_narrates_states_and_inflight_activity() -> None:
    """The live-UI contract: transitions + started events tell the whole story."""
    events: list[dict] = []
    judge = MockJudge([ruling("no_consensus"), ruling("consensus")])
    a = MockParticipant("a", ["monolith"])
    b = MockParticipant("b", ["microservices"])
    runner = DebateRunner(a, b, judge, FAST, sink=events.append)
    await runner.run(TASK)

    states = [(e["from"], e["to"]) for e in events if e["type"] == "state"]
    assert states == [
        ("init", "independent_answer"),
        ("independent_answer", "judge_eval"),
        ("judge_eval", "debate_exchange"),
        ("debate_exchange", "judge_eval"),
        ("judge_eval", "consensus"),
    ]
    types = [e["type"] for e in events]
    # thinking is visible: every completed call was preceded by a started event
    assert types.count("call_started") == types.count("participant_call") == 4
    assert types.count("judge_started") == types.count("judge_eval") == 2
    assert types.index("call_started") < types.index("participant_call")
    # live activity streams to the sink, labeled per seat, DURING the call
    activity = [e for e in events if e["type"] == "activity"]
    assert {e["label"] for e in activity} == {"A", "B"}
    assert all(e["detail"] == "thinking…" for e in activity)
    assert types.index("activity") < types.index("participant_call")
