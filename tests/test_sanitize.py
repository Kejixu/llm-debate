"""Sanitizer: identity leaks are redacted from circulation, detectable for research."""

from llm_debate.sanitize import REDACTED, leaked, sanitize


def test_strips_model_and_vendor_names() -> None:
    text = "As Claude (an Anthropic model), I disagree with ChatGPT and Codex."
    cleaned = sanitize(text)
    for name in ("Claude", "Anthropic", "ChatGPT", "Codex"):
        assert name not in cleaned
    assert cleaned.count(REDACTED) == 4


def test_preserves_substance() -> None:
    text = "As Claude, I recommend a monolith because deploys stay simple."
    assert "monolith because deploys stay simple" in sanitize(text)


def test_leak_detection_flags_change() -> None:
    assert leaked("I am Claude.")
    assert not leaked("I recommend a monolith.")
    clean = "No identity here."
    assert sanitize(clean) == clean  # no false-positive rewrites


def test_gpt_redaction_never_eats_the_next_word() -> None:
    assert sanitize("GPT is better.") == f"{REDACTED} is better."
    assert sanitize("GPT models hallucinate") == f"{REDACTED} models hallucinate"
    assert sanitize("I like GPT\nNext line") == f"I like {REDACTED}\nNext line"
    assert sanitize("We GPTs sometimes disagree") == f"We {REDACTED} sometimes disagree"


def test_gpt_versions_are_redacted_whole() -> None:
    for name in ("gpt-5.5", "GPT-4o", "gpt4", "GPT 4", "GPT\u00a04o", "gpt-oss"):
        assert sanitize(f"{name} said so") == f"{REDACTED} said so", name
