"""Shared test fixtures: real CLI output captured by probes/, committed as ground truth."""

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent / "probes" / "fixtures"


def fixture_stdout(name: str) -> str:
    """The raw stdout a probe captured from a real CLI call."""
    record = json.loads((FIXTURES / f"{name}.json").read_text())
    return record["stdout"]


@pytest.fixture
def claude_stdout() -> str:
    return fixture_stdout("claude_participant")


@pytest.fixture
def codex_stdout() -> str:
    return fixture_stdout("codex_participant_json")


@pytest.fixture
def claude_stream_stdout() -> str:
    return fixture_stdout("claude_participant_stream")
