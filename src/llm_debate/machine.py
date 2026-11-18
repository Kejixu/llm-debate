"""The debate statechart and its async runner.

    ┌──────┐ start ┌────────────────────┐ answers_in ┌────────────┐
    │ INIT ├──────►│ INDEPENDENT_ANSWER ├───────────►│ JUDGE_EVAL │
    └──────┘       └────────────────────┘            └─────┬──────┘
                                                           │ rule (guards, in priority order)
        ┌─────────────────────┬────────────────────┬───────┴────────┬──────────────────┐
        ▼ consensus AND       ▼ round == cap       ▼ over time      ▼ (else)           │
    ┌───────────┐ quality ┌─────────────┐      ┌────────────────┐  ┌─────────────────┐ │
    │ CONSENSUS │         │ CAP_REACHED │      │ BUDGET_EXCEEDED│  │ DEBATE_EXCHANGE │ │
    └───────────┘         └─────────────┘      └────────────────┘  └───────┬─────────┘ │
      (result)              (result)             (interrupted)             │answers_in │
                                                                           └───────────┘
    plus: {INDEPENDENT_ANSWER, DEBATE_EXCHANGE} --overflow--> CONTEXT_EXCEEDED (interrupted)
          {any active state} ------fail------> ERROR

Design rules encoded here:
- The MACHINE is synchronous, pure bookkeeping: which state, which round, may
  this transition fire. All I/O lives in DebateRunner, which awaits CLI calls
  and reports events. (Parallelize the waiting, not the state — review 9A.)
- Guard DECLARATION ORDER is priority: a scientific result (consensus, cap)
  always beats an interruption (budget) — review C1.
- Debaters never see the round number or the judge; the judge sees everything.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, cast

from statemachine import State, StateMachine

from llm_debate.adapters import Judge, Participant
from llm_debate.messages import BuiltMessage, MessageBuilder
from llm_debate.retry import call_with_retry
from llm_debate.sanitize import sanitize
from llm_debate.types import (
    CallResult,
    CLICallError,
    DebateResult,
    JudgeRuling,
    RoundRecord,
    RunConfig,
)


class DebateMachine(StateMachine):
    """Pure state bookkeeping for one debate run."""

    init = State(initial=True)
    independent_answer = State()
    judge_eval = State()
    debate_exchange = State()
    consensus = State(final=True)
    cap_reached = State(final=True)
    budget_exceeded = State(final=True)
    context_exceeded = State(final=True)
    error = State(final=True)

    start = init.to(independent_answer)
    answers_in = independent_answer.to(judge_eval) | debate_exchange.to(judge_eval)
    # Guard order = priority: results first, interruption second, continue last.
    rule = (
        judge_eval.to(consensus, cond="ruling_is_consensus")
        | judge_eval.to(cap_reached, cond="at_cap")
        | judge_eval.to(budget_exceeded, cond="over_budget")
        | judge_eval.to(debate_exchange)
    )
    overflow = independent_answer.to(context_exceeded) | debate_exchange.to(context_exceeded)
    fail = (
        init.to(error)
        | independent_answer.to(error)
        | judge_eval.to(error)
        | debate_exchange.to(error)
    )

    def __init__(
        self,
        *,
        cap: int,
        max_minutes: float,
        clock: Callable[[], float] = time.monotonic,
        notify: Callable[[dict], None] | None = None,
    ) -> None:
        self.cap = cap
        self._clock = clock
        self.deadline_s = clock() + max_minutes * 60.0
        self.round = 0
        self.last_ruling: JudgeRuling | None = None
        self._notify = notify
        super().__init__()

    def on_transition(self, event: str, source: State, target: State) -> None:
        """Generic hook: every transition becomes an observable event —
        the live UI's statechart lights up from these, and events.jsonl
        gains the full state history (replayable animation later)."""
        if self._notify is not None:
            self._notify(
                {
                    "type": "state",
                    "trigger": event,
                    "from": source.id,
                    "to": target.id,
                    "round": self.round,
                }
            )

    # --- guards (the exit predicate lives here) ------------------------------

    def ruling_is_consensus(self) -> bool:
        """Premise 1: agreement alone is not enough — quality must pass too."""
        ruling = self.last_ruling
        assert ruling is not None, "rule() fired before a ruling was recorded"
        return ruling.verdict == "consensus" and ruling.best_answer_quality == "acceptable"

    def at_cap(self) -> bool:
        return self.round >= self.cap

    def over_budget(self) -> bool:
        return self._clock() > self.deadline_s

    # --- actions -------------------------------------------------------------

    def on_enter_debate_exchange(self) -> None:
        self.round += 1

    # --- typed views over the active configuration (a SET of states, because
    # statecharts may run parallel regions; this flat machine always has one)

    @property
    def _active_state(self) -> State:
        (state,) = tuple(self.configuration)
        return state

    @property
    def state_id(self) -> str:
        return self._active_state.id

    @property
    def finished(self) -> bool:
        return self._active_state.final


class _Overflow(Exception):
    """Internal signal: a constructed message exceeded the context budget."""


class DebateRunner:
    """Drives one debate: awaits the CLI calls, reports events to the machine."""

    def __init__(
        self,
        participant_a: Participant,
        participant_b: Participant,
        judge: Judge,
        config: RunConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sink: Callable[[dict], None] | None = None,
    ) -> None:
        self.a = participant_a
        self.b = participant_b
        self.judge = judge
        self.config = config
        self.builder = MessageBuilder(condition=config.condition)
        self._clock = clock
        self._sink = sink

    def _emit(self, event: dict) -> None:
        if self._sink is not None:
            self._sink(event)

    async def run(self, task: str) -> DebateResult:
        machine = DebateMachine(
            cap=self.config.cap,
            max_minutes=self.config.max_minutes,
            clock=self._clock,
            notify=self._emit,
        )
        records: list[RoundRecord] = []
        error: str | None = None
        try:
            await self._debate(machine, task, records)
        except _Overflow:
            machine.overflow()
        except CLICallError as exc:
            machine.fail()
            error = str(exc)
        self._emit(
            {
                "type": "terminal",
                "status": machine.state_id,
                "rounds_completed": machine.round,
                "error": error,
            }
        )
        return self._result(machine, records, error=error)

    # --- the loop -------------------------------------------------------------

    async def _debate(self, machine: DebateMachine, task: str, records: list[RoundRecord]) -> None:
        machine.start()
        opening = self.builder.opening(task)
        raw_a, raw_b = await self._ask_both(machine, opening, opening)
        machine.answers_in()

        while True:
            sanitized_a, sanitized_b = sanitize(raw_a.answer), sanitize(raw_b.answer)
            self._tap_activity(machine, getattr(self.judge, "backend", None), "J")
            self._emit({"type": "judge_started", "round": machine.round})
            ruling = await self.judge.evaluate(
                self.builder.judge(
                    task, sanitized_a, sanitized_b, round_number=machine.round, cap=machine.cap
                )
            )
            machine.last_ruling = ruling
            record = RoundRecord(
                round=machine.round,
                answers_raw={"A": raw_a.answer, "B": raw_b.answer},
                answers_sanitized={"A": sanitized_a, "B": sanitized_b},
                ruling=ruling,
            )
            records.append(record)
            self._emit(
                {
                    "type": "judge_eval",
                    "round": record.round,
                    "answers_raw": record.answers_raw,
                    "answers_sanitized": record.answers_sanitized,
                    "ruling": asdict(ruling),
                }
            )
            machine.rule()
            if machine.finished:
                return

            # DEBATE_EXCHANGE: ONLY sanitized text circulates — even a debater's
            # own history (its raw words may self-identify, and the invariant is
            # "no identity strings in ANY constructed message"). Raw lives in the
            # event log alone (4A).
            seen_a = [r.answers_sanitized["A"] for r in records]
            seen_b = [r.answers_sanitized["B"] for r in records]
            message_a = self.builder.exchange(task, seen_a, seen_b)
            message_b = self.builder.exchange(task, seen_b, seen_a)
            if max(len(message_a.prompt), len(message_b.prompt)) > self.config.max_prompt_chars:
                raise _Overflow
            raw_a, raw_b = await self._ask_both(machine, message_a, message_b)
            machine.answers_in()

    async def _ask_both(
        self, machine: DebateMachine, message_a: BuiltMessage, message_b: BuiltMessage
    ) -> tuple[CallResult, CallResult]:
        """Both debaters answer CONCURRENTLY (independent within a round — 9A)."""
        results = await asyncio.gather(
            self._ask(machine, self.a, "A", message_a),
            self._ask(machine, self.b, "B", message_b),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        answer_a, answer_b = results
        assert isinstance(answer_a, CallResult) and isinstance(answer_b, CallResult)
        return answer_a, answer_b

    def _tap_activity(self, machine: DebateMachine, target: object, label: str) -> None:
        """Point a participant's live-activity tap (if it has one) at the sink.
        Duck-typed: mocks without the attribute just don't stream."""

        def emit(detail: str) -> None:
            self._emit(
                {"type": "activity", "round": machine.round, "label": label, "detail": detail}
            )

        if hasattr(target, "on_activity"):
            cast(Any, target).on_activity = emit

    async def _ask(
        self, machine: DebateMachine, participant: Participant, label: str, message: BuiltMessage
    ) -> CallResult:
        self._tap_activity(machine, participant, label)
        self._emit(
            {
                "type": "call_started",
                "round": machine.round,
                "label": label,
                "participant": participant.name,
            }
        )
        try:
            result = await call_with_retry(
                lambda: participant.answer(message.prompt, message.role),
                max_retries=self.config.max_retries,
                base_delay_s=self.config.retry_base_delay_s,
            )
        except CLICallError as exc:
            self._emit(
                {
                    "type": "participant_call_failed",
                    "round": machine.round,
                    "label": label,
                    "participant": participant.name,
                    "error": str(exc),
                }
            )
            raise
        # Exact per-call provenance (C4): the verbatim prompt sent and the raw
        # output received — the run folder stays self-contained forever.
        self._emit(
            {
                "type": "participant_call",
                "round": machine.round,
                "label": label,
                "participant": participant.name,
                "prompt": message.prompt,
                "role": message.role,
                "answer_raw": result.answer,
                "usage": asdict(result.usage),
                "session_id": result.session_id,
                "raw_stdout": result.raw_stdout,
            }
        )
        return result

    def _result(
        self, machine: DebateMachine, records: list[RoundRecord], error: str | None = None
    ) -> DebateResult:
        return DebateResult(
            status=machine.state_id,
            best_answer=records[-1].ruling.best_answer if records else None,
            rounds_completed=machine.round,
            records=records,
            error=error,
        )
