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
