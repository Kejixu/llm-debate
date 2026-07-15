"""CLI-backed judge: strict JSON parsing with ONE corrective re-ask (review 6A).

No lenient free-text parser, ever: a silently mis-parsed verdict corrupts the
event log invisibly, which is worse than a dead run. On a malformed payload
the judge is re-asked once with the exact parse error; a second failure
propagates as CLICallError and the run terminates in ERROR.
"""

import json
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from llm_debate.adapters import Participant
from llm_debate.retry import call_with_retry
from llm_debate.types import JudgeRuling, ParseError


def parse_ruling(text: str) -> JudgeRuling:
    """Validate the judge's JSON payload into a JudgeRuling — strict on content.

    Markdown fences are stripped (transport noise); every FIELD is validated
    exactly, and any deviation raises ParseError with a message specific
    enough to feed back in the corrective re-ask.
    """
    cleaned = _strip_fences(text.strip())
    try:
        data: Any = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ParseError(f"reply is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ParseError("reply must be a JSON object")

    verdict = data.get("verdict")
    if verdict not in ("consensus", "no_consensus"):
        raise ParseError(f"verdict must be 'consensus' or 'no_consensus', got {verdict!r}")
    score = data.get("convergence_score")
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
        raise ParseError(f"convergence_score must be an integer 0-100, got {score!r}")
    best_answer = data.get("best_answer")
    if not isinstance(best_answer, str) or not best_answer.strip():
        raise ParseError("best_answer must be a non-empty string")
    quality = data.get("best_answer_quality")
    if quality not in ("acceptable", "weak"):
        raise ParseError(f"best_answer_quality must be 'acceptable' or 'weak', got {quality!r}")

    return JudgeRuling(
        verdict=verdict,
        convergence_score=score,
        best_answer=best_answer,
        best_answer_quality=quality,
        agreement_reasons=_str_tuple(data.get("agreement_reasons", []), "agreement_reasons"),
        cruxes=_str_tuple(data.get("cruxes", []), "cruxes"),
    )


def _strip_fences(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _str_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ParseError(f"{field} must be a list of strings")
    return tuple(value)


class CLIJudge:
    """A Participant backend + the strict-JSON judging contract."""

    def __init__(
        self,
        backend: Participant,
        *,
        max_retries: int = 2,
        base_delay_s: float = 2.0,
        sink: Callable[[dict], None] | None = None,
    ) -> None:
        self.backend = backend
        self.name = f"judge-{backend.name}"
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.sink = sink

    async def evaluate(self, prompt: str) -> JudgeRuling:
        answer = await self._call(prompt, corrective=False)
        try:
            return parse_ruling(answer)
        except ParseError as exc:
            corrective_prompt = (
                f"{prompt}\n\n"
                f"Your previous reply could not be used: {exc}\n"
                "Respond again with ONLY a valid JSON object matching the schema above — "
                "no prose, no markdown fences."
            )
            answer = await self._call(corrective_prompt, corrective=True)
            return parse_ruling(answer)  # second failure propagates -> ERROR

    async def _call(self, prompt: str, *, corrective: bool) -> str:
        result = await call_with_retry(
            lambda: self.backend.answer(prompt),
            max_retries=self.max_retries,
            base_delay_s=self.base_delay_s,
        )
        if self.sink is not None:
            self.sink(
                {
                    "type": "judge_call",
                    "judge": self.name,
                    "corrective": corrective,
                    "prompt": prompt,
                    "answer_raw": result.answer,
                    "raw_stdout": result.raw_stdout,
                    "usage": asdict(result.usage),
                    "session_id": result.session_id,
                }
            )
        return result.answer
