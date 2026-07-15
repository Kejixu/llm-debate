"""Core value types shared across llm-debate components."""

from dataclasses import dataclass, field
from typing import Literal

Condition = Literal["natural", "steered"]
Verdict = Literal["consensus", "no_consensus"]
Quality = Literal["acceptable", "weak"]


@dataclass(frozen=True)
class Usage:
    """Token/cost accounting parsed from a CLI's structured output.

    Every field is optional: each CLI exposes a different subset (claude
    exposes cost_usd, codex does not), and absence must stay distinguishable
    from zero.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class CallResult:
    """One completed CLI call: the answer plus everything provenance needs.

    raw_stdout keeps the exact envelope/event stream received so the event
    log can store what was ACTUALLY emitted (review decision C4) — never
    reconstruct from parsed fields.
    """

    answer: str
    usage: Usage
    session_id: str | None
    raw_stdout: str


class CLICallError(Exception):
    """A CLI call failed in a way call_with_retry may retry.

    Covers: non-zero exit, timeout kill, and unparseable/invalid output
    (probe findings: claude exits 1, codex exits 2, timeouts kill cleanly).
    """

    def __init__(
        self,
        message: str,
        *,
        argv: list[str] | None = None,
        exit_code: int | None = None,
        stderr: str = "",
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.argv = argv
        self.exit_code = exit_code
        self.stderr = stderr
        self.timed_out = timed_out


class ParseError(CLICallError):
    """The CLI exited 0 but its output did not match the expected shape."""


@dataclass(frozen=True)
class JudgeRuling:
    """The judge's structured verdict after one JUDGE_EVAL (v1 fields)."""

    verdict: Verdict
    convergence_score: int
    best_answer: str
    best_answer_quality: Quality
    agreement_reasons: tuple[str, ...] = ()
    cruxes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoundRecord:
    """One per-round record: what circulated, what was really said, the ruling.

    answers_raw is research data (leakage is data — review 4A); ONLY
    answers_sanitized ever reached the opponent and the judge.
    """

    round: int
    answers_raw: dict[str, str]
    answers_sanitized: dict[str, str]
    ruling: JudgeRuling


@dataclass(frozen=True)
class DebateResult:
    """Terminal outcome of a run: the machine's final state + the evidence."""

    status: str  # terminal state id: consensus | cap_reached | budget_exceeded | ...
    best_answer: str | None
    rounds_completed: int
    records: list[RoundRecord] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class RunConfig:
    """Every knob that shapes a run — serialized to runs/<id>/config.json (7A)."""

    cap: int = 10
    max_minutes: float = 30.0
    condition: Condition = "natural"
    per_call_timeout_s: float = 600.0
    max_retries: int = 2
    retry_base_delay_s: float = 2.0
    max_prompt_chars: int = 400_000  # ~100k tokens; overflow -> CONTEXT_EXCEEDED (C3)
