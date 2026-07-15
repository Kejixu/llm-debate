"""Envelope/stream parsers, asserted against REAL captured CLI output (probe fixtures)."""

import json

import pytest

from llm_debate.adapters import parse_claude_envelope, parse_codex_stream
from llm_debate.types import CLICallError, ParseError

# --- claude: single JSON envelope -------------------------------------------


def test_claude_fixture_parses(claude_stdout: str) -> None:
    result = parse_claude_envelope(claude_stdout)
    assert result.answer == "PROBE_OK"
    assert result.session_id == "716106f5-c157-4a70-af8a-f7bb25eee304"
    assert result.usage.cost_usd == pytest.approx(0.19317)
    assert result.usage.input_tokens == 9549
    assert result.raw_stdout == claude_stdout  # exact provenance preserved (C4)


def test_claude_not_json_raises() -> None:
    with pytest.raises(ParseError):
        parse_claude_envelope("I am not JSON")


def test_claude_non_object_raises() -> None:
    with pytest.raises(ParseError):
        parse_claude_envelope('["a", "list"]')


def test_claude_missing_result_raises() -> None:
    with pytest.raises(ParseError):
        parse_claude_envelope('{"type": "result", "is_error": false}')


def test_claude_is_error_raises() -> None:
    with pytest.raises(CLICallError):
        parse_claude_envelope('{"is_error": true, "result": "rate limited"}')


# --- claude: stream-json JSONL (live-observation mode) -----------------------


def test_claude_stream_fixture_parses(claude_stream_stdout: str) -> None:
    from llm_debate.adapters import parse_claude_stream

    result = parse_claude_stream(claude_stream_stdout)
    assert result.answer == "PROBE_OK"
    assert result.session_id == "a7dee742-3fb6-47df-830d-3af874fb03f6"
    assert result.usage.cost_usd == pytest.approx(0.313652)
    assert result.raw_stdout == claude_stream_stdout  # exact provenance (C4)


def test_claude_stream_no_result_event_raises() -> None:
    from llm_debate.adapters import parse_claude_stream

    with pytest.raises(ParseError):
        parse_claude_stream('{"type": "system", "subtype": "init"}\nnot json\n')


def test_claude_stream_is_error_raises() -> None:
    from llm_debate.adapters import parse_claude_stream

    with pytest.raises(CLICallError):
        parse_claude_stream('{"type": "result", "is_error": true, "result": "rate limited"}')


# --- activity extractors: stream line -> live note ---------------------------


def test_claude_activity_from_fixture_lines(claude_stream_stdout: str) -> None:
    from llm_debate.adapters import claude_activity

    notes = [claude_activity(ln) for ln in claude_stream_stdout.splitlines()]
    assert "session started" in notes
    assert "thinking… ~115 tokens" in notes  # the live "what is it doing" signal
    assert "writing answer…" in notes
    assert "finalizing" in notes
    assert claude_activity("not json") is None


def test_codex_activity_from_fixture_lines(codex_stdout: str) -> None:
    from llm_debate.adapters import codex_activity

    notes = [codex_activity(ln) for ln in codex_stdout.splitlines()]
    assert "session started" in notes
    assert "thinking…" in notes
    assert "writing answer…" in notes


def test_codex_activity_reasoning_snippet() -> None:
    from llm_debate.adapters import codex_activity

    item = {"type": "reasoning", "text": "**Weighing options**\nmore"}
    line = json.dumps({"type": "item.completed", "item": item})
    assert codex_activity(line) == "reasoning: Weighing options"


# --- codex: JSONL event stream ----------------------------------------------


def test_codex_fixture_parses(codex_stdout: str) -> None:
    result = parse_codex_stream(codex_stdout)
    assert result.answer == "PROBE_OK"
    assert result.session_id == "019f3b37-309c-7f30-9b6b-bfae8a26d027"
    assert result.usage.input_tokens == 12695
    assert result.usage.cached_input_tokens == 9600
    assert result.usage.reasoning_output_tokens == 58
    assert result.usage.cost_usd is None  # codex exposes no cost — absence, not zero


def test_codex_last_agent_message_wins() -> None:
    lines = [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "draft"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "final"}},
    ]
    stdout = "\n".join(json.dumps(line) for line in lines)
    assert parse_codex_stream(stdout).answer == "final"


def test_codex_tolerates_transport_noise() -> None:
    stdout = "\n".join(
        [
            "some banner text",
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}}),
        ]
    )
    assert parse_codex_stream(stdout).answer == "hi"


def test_codex_no_agent_message_raises() -> None:
    stdout = json.dumps({"type": "turn.completed", "usage": {}})
    with pytest.raises(ParseError):
        parse_codex_stream(stdout)


def test_codex_turn_failed_raises() -> None:
    with pytest.raises(CLICallError):
        parse_codex_stream(json.dumps({"type": "turn.failed", "error": "boom"}))


# --- argv builders: model pinning -------------------------------------------


def test_claude_argv_includes_model() -> None:
    from pathlib import Path

    from llm_debate.adapters import ClaudeParticipant

    p = ClaudeParticipant(Path("/tmp/x"), model="claude-opus-4-8")
    argv = p.build_argv("hi")
    assert argv[argv.index("--model") + 1] == "claude-opus-4-8"
    assert p.name == "claude-cli:claude-opus-4-8"


def test_codex_argv_includes_model_before_prompt() -> None:
    from pathlib import Path

    from llm_debate.adapters import CodexParticipant

    p = CodexParticipant(Path("/tmp/x"), model="gpt-5.5")
    argv = p.build_argv("hi")
    assert argv[argv.index("-m") + 1] == "gpt-5.5"
    assert argv[-1] == "hi"  # prompt stays last


# --- env hygiene: host-injected vars never reach participant CLIs ------------


async def test_run_cli_strips_host_injected_env(tmp_path, monkeypatch) -> None:
    from llm_debate.adapters import run_cli

    monkeypatch.setenv("NODE_OPTIONS", "--require=/tmp/gone.cjs")
    monkeypatch.setenv("HOME_MARKER_OK", "yes")
    stdout = await run_cli(["/usr/bin/env"], cwd=tmp_path / "scratch", timeout_s=10)
    env_seen = dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line)
    assert "NODE_OPTIONS" not in env_seen  # the cmux failure mode, prevented
    assert env_seen.get("HOME_MARKER_OK") == "yes"  # normal env passes through


async def test_run_cli_on_line_sees_stdout_live_and_full_text_survives(tmp_path) -> None:
    from llm_debate.adapters import run_cli

    seen: list[str] = []
    stdout = await run_cli(
        ["/bin/sh", "-c", "echo one; echo two"],
        cwd=tmp_path / "scratch",
        timeout_s=10,
        on_line=seen.append,
    )
    assert seen == ["one", "two"]  # the live tap
    assert stdout == "one\ntwo\n"  # parsers still get the exact full text


async def test_run_cli_on_line_exception_never_kills_the_call(tmp_path) -> None:
    from llm_debate.adapters import run_cli

    def boom(_: str) -> None:
        raise RuntimeError("display bug")

    stdout = await run_cli(
        ["/bin/sh", "-c", "echo ok"], cwd=tmp_path / "scratch", timeout_s=10, on_line=boom
    )
    assert stdout == "ok\n"


# --- timeout hygiene: the WHOLE process tree dies, not just the wrapper -------
# The npm `codex` command is a Node shim that spawns the native binary as a
# grandchild; killing only the direct child orphans a live, money-burning CLI.


async def test_run_cli_timeout_kills_grandchildren(tmp_path) -> None:
    import asyncio
    import os

    import pytest

    from llm_debate.adapters import run_cli
    from llm_debate.types import CLICallError

    scratch = tmp_path / "scratch"
    # wrapper (sh) spawns a grandchild sleep and records its pid — the shape
    # of the codex npm shim -> native binary chain
    script = "sleep 300 & echo $! > grandchild.pid; wait"
    with pytest.raises(CLICallError) as exc:
        await run_cli(["/bin/sh", "-c", script], cwd=scratch, timeout_s=0.5)
    assert exc.value.timed_out
    pid = int((scratch / "grandchild.pid").read_text().strip())
    await asyncio.sleep(0.2)  # give the kill a beat to land
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # grandchild must be gone too


async def test_run_cli_cancellation_kills_process_tree(tmp_path) -> None:
    """Ctrl-C / server shutdown cancels the awaiting task; the detached
    process group must die with it (codex cross-model review, 2026-07-16)."""
    import asyncio
    import os

    import pytest

    from llm_debate.adapters import run_cli

    scratch = tmp_path / "scratch"
    script = "sleep 300 & echo $! > grandchild.pid; wait"
    task = asyncio.create_task(
        run_cli(["/bin/sh", "-c", script], cwd=scratch, timeout_s=60)
    )
    pid_file = scratch / "grandchild.pid"
    for _ in range(200):  # wait until the grandchild exists, then cancel mid-flight
        if pid_file.exists() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    pid = int(pid_file.read_text().strip())
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
