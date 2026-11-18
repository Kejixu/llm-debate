"""Mock adapters — first-class citizens of v1 (design decision, not test helpers).

They implement the same Participant/Judge Protocols as the CLI backends, so
the ENTIRE statechart (transitions, round counter, exit predicate, ERROR
paths, concurrency) is exercised deterministically with zero CLI spend.
"""

import asyncio
from collections.abc import Callable

from llm_debate.types import CallResult, CLICallError, JudgeRuling, Quality, Usage, Verdict


class MockParticipant:
    """Scripted debater: returns answers in order (last one repeats).

    error_after=k -> every call with index >= k raises CLICallError (retry/ERROR paths).
    barrier      -> awaited before answering; a 2-party barrier DEADLOCKS unless
                    both debaters are in flight simultaneously (concurrency proof).
    """

    def __init__(
        self,
        name: str,
        answers: list[str],
        *,
        error_after: int | None = None,
        barrier: asyncio.Barrier | None = None,
    ) -> None:
        self.name = name
        self.answers = answers
        self.error_after = error_after
        self.barrier = barrier
        self.prompts_seen: list[tuple[str, str | None]] = []
        self.on_activity: Callable[[str], None] | None = None  # same tap as CLI backends

    async def answer(self, prompt: str, role: str | None = None) -> CallResult:
        call_index = len(self.prompts_seen)
        self.prompts_seen.append((prompt, role))
        if self.on_activity is not None:
            self.on_activity("thinking…")
        if self.barrier is not None:
            await self.barrier.wait()
        if self.error_after is not None and call_index >= self.error_after:
            raise CLICallError(f"{self.name} scripted failure at call {call_index}")
        answer = self.answers[min(call_index, len(self.answers) - 1)]
        return CallResult(answer=answer, usage=Usage(), session_id=None, raw_stdout="{}")


class MockJudge:
    """Scripted judge: returns rulings in order (last one repeats)."""

    name = "mock-judge"

    def __init__(self, rulings: list[JudgeRuling], *, error: CLICallError | None = None) -> None:
        self.rulings = rulings
        self.error = error
        self.prompts_seen: list[str] = []

    async def evaluate(self, prompt: str) -> JudgeRuling:
        call_index = len(self.prompts_seen)
        self.prompts_seen.append(prompt)
        if self.error is not None:
            raise self.error
        return self.rulings[min(call_index, len(self.rulings) - 1)]


def ruling(
    verdict: Verdict = "no_consensus",
    *,
    score: int = 40,
    best_answer: str = "the best answer so far",
    quality: Quality = "acceptable",
) -> JudgeRuling:
    """Compact JudgeRuling factory for tests."""
    return JudgeRuling(
        verdict=verdict,
        convergence_score=score,
        best_answer=best_answer,
        best_answer_quality=quality,
        agreement_reasons=("they broadly align",),
        cruxes=() if verdict == "consensus" else ("they still disagree on X",),
    )
