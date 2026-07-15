"""Output sanitization: strip obvious identity leaks before an answer circulates.

Review decision 4A: the RAW answer is preserved in the event log (leakage is
research data); only the sanitized version ever reaches the opponent or the
judge. This is best-effort blinding — stylistic leakage survives and is an
accepted, logged limitation.
"""

import re

_IDENTITY_PATTERN = re.compile(
    r"\b("
    r"claude(?:\s+code)?|anthropic|"
    r"codex|chatgpt|openai|gpt[-\s]?[0-9a-z.]*|"
    r"gemini|google\s+deepmind"
    r")\b",
    re.IGNORECASE,
)

REDACTED = "[model-name-redacted]"


def sanitize(text: str) -> str:
    """Replace model/vendor self-references with a neutral marker."""
    return _IDENTITY_PATTERN.sub(REDACTED, text)


def leaked(text: str) -> bool:
    """True if the text contains an identity leak (for logging/analysis)."""
    return _IDENTITY_PATTERN.search(text) is not None
