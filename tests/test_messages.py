"""MessageBuilder invariants: blindness, invisible judge, one-variable steering.

These are the project's load-bearing scientific controls, enforced as code.
"""

import pytest

from llm_debate.messages import JUDGE_SCHEMA, BuiltMessage, Condition, MessageBuilder

TASK = "What are the tradeoffs of microservices versus a monolith for a two-person startup?"
OWN = ["Monolith first: less operational overhead."]
OPPONENT = ["Microservices scale better from day one."]

# Words that would leak identity (blindness) or reveal the observer (invisible judge).
FORBIDDEN_IN_DEBATER_MESSAGES = [
    "claude",
    "codex",
    "anthropic",
    "openai",
    "gpt",
    "gemini",
    "judge",
    "referee",
    "evaluat",  # evaluate / evaluation / evaluator
    "score",
    "consensus",
    "round",
]


def all_debater_messages(condition: Condition) -> list[BuiltMessage]:
    builder = MessageBuilder(condition=condition)
    return [builder.opening(TASK), builder.exchange(TASK, OWN, OPPONENT)]


@pytest.mark.parametrize("condition", ["natural", "steered"])
def test_blindness_and_invisible_judge(condition: Condition) -> None:
    for message in all_debater_messages(condition):
        text = message.prompt.lower() + " " + (message.role or "").lower()
        for word in FORBIDDEN_IN_DEBATER_MESSAGES:
            assert word not in text, f"{word!r} leaked into a debater message"


def test_steered_equals_natural_plus_role_only() -> None:
    natural = all_debater_messages("natural")
    steered = all_debater_messages("steered")
    for n, s in zip(natural, steered, strict=True):
        assert n.prompt == s.prompt  # byte-identical prompt: one-variable experiment
        assert n.role is None
        assert isinstance(s.role, str) and s.role


def test_opening_is_task_verbatim() -> None:
    assert MessageBuilder().opening(TASK).prompt == TASK


def test_exchange_contains_history_but_only_anonymized() -> None:
    prompt = MessageBuilder().exchange(TASK, OWN, OPPONENT).prompt
    assert TASK in prompt
    assert OWN[0] in prompt
    assert OPPONENT[0] in prompt
    assert "Another participant" in prompt  # anonymized label, nothing more


def test_judge_message_sees_everything() -> None:
    prompt = MessageBuilder().judge(TASK, "answer a", "answer b", round_number=3, cap=10)
    assert "round 3" in prompt and "10" in prompt  # judge always knows the round
    assert "Participant A" in prompt and "Participant B" in prompt
    assert "answer a" in prompt and "answer b" in prompt
    assert JUDGE_SCHEMA in prompt  # strict schema, verbatim
    assert "capitulation" in prompt  # consensus != correctness gate is in the ask
