"""MessageBuilder — the sole owner of ALL prompt assembly (review decision 5A).

Invariants this module enforces by construction (and tests enforce by assertion):
  1. BLINDNESS   — debater messages never name a model, vendor, or CLI.
  2. INVISIBLE JUDGE — debater messages never mention a judge, evaluation,
     scoring, or consensus-seeking; debaters also never see the round number.
  3. ONE VARIABLE — steered mode differs from natural by the role text ONLY;
     the constructed prompt is byte-identical across conditions.

The judge prompt is the only place round numbers and both labeled answers
appear — the judge knows everything; the debaters know almost nothing.
"""

from dataclasses import dataclass

from llm_debate.types import Condition as Condition  # re-export: builder's public vocab

DEFAULT_STEERED_ROLE = (
    "You are a rigorous, independent thinker. Defend your view with evidence and do not "
    "abandon a position merely because someone disagrees. If you change your mind, say "
    "explicitly what convinced you; flag clearly where you genuinely agree versus where "
    "you would only be capitulating."
)

JUDGE_SCHEMA = """{
  "verdict": "consensus" | "no_consensus",
  "convergence_score": <integer 0-100>,
  "best_answer": "<the strongest complete answer currently on the table>",
  "best_answer_quality": "acceptable" | "weak",
  "agreement_reasons": ["<where and why the participants align>"],
  "cruxes": ["<each specific remaining disagreement and why it persists>"]
}"""


@dataclass(frozen=True)
class BuiltMessage:
    """What an adapter needs for one debater call: the prompt + optional role."""

    prompt: str
    role: str | None


class MessageBuilder:
    def __init__(
        self, condition: Condition = "natural", steered_role: str = DEFAULT_STEERED_ROLE
    ) -> None:
        self.condition = condition
        self._role = steered_role if condition == "steered" else None

    # --- debater messages (blind, judge-free, round-free) -------------------

    def opening(self, task: str) -> BuiltMessage:
        """Round 0: the blind opening — the task, nothing else."""
        return BuiltMessage(prompt=task, role=self._role)

    def exchange(
        self, task: str, own_answers: list[str], opponent_answers: list[str]
    ) -> BuiltMessage:
        """An exchange round: task + own history + anonymized opponent history."""
        parts = [
            task,
            "",
            "Your previous answer(s) to this question:",
            _numbered(own_answers),
            "",
            "Another participant answered the same question:",
            _numbered(opponent_answers),
            "",
            "You may revise or defend your previous answer. State where you agree or "
            "disagree with the other participant and why, then give your complete, "
            "current answer to the original question.",
        ]
        return BuiltMessage(prompt="\n".join(parts), role=self._role)

    # --- judge message (sees everything; never seen by debaters) ------------

    def judge(self, task: str, answer_a: str, answer_b: str, round_number: int, cap: int) -> str:
        parts = [
            "You are the judge of a debate between two anonymized participants. "
            f"This is round {round_number} of at most {cap}.",
            "",
            f"The question they were asked:\n{task}",
            "",
            f"Participant A's current answer:\n{answer_a}",
            "",
            f"Participant B's current answer:\n{answer_b}",
            "",
            "Judge whether the participants have reached SUBSTANTIVE agreement — not mere "
            "politeness or capitulation — and whether the best answer on the table is "
            "actually good. Agreement alone is not consensus if the agreed answer is weak.",
            "",
            "Respond with ONLY a JSON object, no markdown fences, matching exactly:",
            JUDGE_SCHEMA,
        ]
        return "\n".join(parts)


def _numbered(items: list[str]) -> str:
    return "\n".join(f"[{i}] {item}" for i, item in enumerate(items, start=1))
