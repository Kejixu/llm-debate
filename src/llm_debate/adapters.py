"""Participant adapters: the port (Protocol) and its CLI backends.

Layering (each testable alone):
  argv builder (pure)  ->  subprocess runner (thin I/O)  ->  parser (pure)

The parsers are exercised directly against probes/fixtures/*.json — captured
real CLI output — so tests assert against reality, not assumptions.

Probe-verified shapes (2026-07-07):
  claude -p --output-format json  -> ONE JSON envelope
     {result, is_error, session_id, total_cost_usd, usage{...}, ...}
  codex exec --json               -> JSONL EVENT STREAM
     thread.started{thread_id} / item.completed{item.type=agent_message, .text}
     / turn.completed{usage{...}}

Isolation rule: every call runs in its own EMPTY scratch cwd — agent CLIs
load instruction files (CLAUDE.md / AGENTS.md) from wherever they run, and a
debater must never inherit the orchestrator repo's context.
"""

import asyncio
import json
import os
import signal
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

from llm_debate.types import CallResult, CLICallError, JudgeRuling, ParseError, Usage

# 120s proved too short in the field: an open-ended prompt + web search ran
# both CLIs past it (2026-07-16 run 131937) and the whole round died on retry
DEFAULT_TIMEOUT_S = 600.0


class Participant(Protocol):
    """The port: anything that can answer a constructed message."""

    name: str

    async def answer(self, prompt: str, role: str | None = None) -> CallResult: ...


class Judge(Protocol):
    """The port for the (invisible) judge: prompt in, structured ruling out."""

    name: str

    async def evaluate(self, prompt: str) -> JudgeRuling: ...


# --- subprocess runner (the only side-effectful layer) ---------------------

# Env vars that host terminals/agents inject into THEIR processes and that
# would contaminate or break the participant CLIs. Discovered the hard way:
# cmux injects NODE_OPTIONS=--require=<tempfile>; when the tempfile is
# cleaned up, every Node CLI (codex) dies on startup with MODULE_NOT_FOUND.
_STRIPPED_ENV_VARS = ("NODE_OPTIONS", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")


def sanitized_env() -> dict[str, str]:
    """The orchestrator's env minus host-injected vars — same isolation
    principle as the empty scratch cwd, applied to the environment."""
    return {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV_VARS}


# Both CLIs emit JSONL where a single line can carry a whole answer; the
# asyncio default readline limit (64 KiB) would truncate real output.
_STREAM_LIMIT = 16 * 1024 * 1024


async def run_cli(
    argv: list[str],
    *,
    cwd: Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    on_line: Callable[[str], None] | None = None,
) -> str:
    """Run a CLI to completion in an isolated cwd; return stdout text.

    stdout is read line-by-line; `on_line` (if given) sees each line as it
    arrives — this is the live-observation tap. The full text is still
    returned for the strict parsers, so streaming never changes parsing.

    Raises CLICallError on timeout (process killed) or non-zero exit —
    probe-verified: claude exits 1, codex exits 2, with useful stderr.
    """
    cwd.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
        cwd=cwd,
        env=sanitized_env(),
        start_new_session=True,  # own process group, so timeout can kill the TREE
        limit=_STREAM_LIMIT,
    )
    out_chunks: list[str] = []

    async def read_stdout() -> None:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            text = line.decode(errors="replace")
            out_chunks.append(text)
            if on_line is not None:
                with suppress(Exception):  # a display bug must never kill a call
                    on_line(text.rstrip("\n"))

    async def pump() -> bytes:
        assert proc.stderr is not None
        _, stderr = await asyncio.gather(read_stdout(), proc.stderr.read())
        await proc.wait()
        return stderr

    try:
        stderr = await asyncio.wait_for(pump(), timeout=timeout_s)
    except (TimeoutError, asyncio.CancelledError) as exc:
        # npm shims (codex) spawn the real binary as a grandchild; killing only
        # the direct child orphans a live CLI that keeps pipes open and burns
        # money. Cancellation (Ctrl-C, server shutdown) must reap the detached
        # group too — start_new_session means nobody else will.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        await proc.wait()
        if isinstance(exc, asyncio.CancelledError):
            raise  # cleanup done — let cancellation propagate
        raise CLICallError(
            f"timed out after {timeout_s}s: {argv[0]}", argv=argv, timed_out=True
        ) from None
    if proc.returncode != 0:
        raise CLICallError(
            f"{argv[0]} exited {proc.returncode}",
            argv=argv,
            exit_code=proc.returncode,
            stderr=stderr.decode(errors="replace"),
        )
    return "".join(out_chunks)


# --- pure parsers -----------------------------------------------------------


def _claude_call_result(data: dict, raw_stdout: str) -> CallResult:
    """Shared claude payload -> CallResult: the stream-json `result` event
    carries the same fields as the old single envelope (probe 2026-07-16)."""
    if data.get("is_error"):
        raise CLICallError(f"claude reported is_error: {str(data.get('result'))[:200]}")
    answer = data.get("result")
    if not isinstance(answer, str):
        raise ParseError("claude envelope has no string 'result' field")
    usage_raw = data.get("usage") or {}
    usage = Usage(
        input_tokens=usage_raw.get("input_tokens"),
        output_tokens=usage_raw.get("output_tokens"),
        cached_input_tokens=usage_raw.get("cache_read_input_tokens"),
        cost_usd=data.get("total_cost_usd"),
    )
    return CallResult(
        answer=answer,
        usage=usage,
        session_id=data.get("session_id"),
        raw_stdout=raw_stdout,
    )


def parse_claude_envelope(stdout: str) -> CallResult:
    """Parse claude -p's single JSON envelope into a CallResult."""
    try:
        data: Any = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ParseError(f"claude stdout is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ParseError("claude envelope is not a JSON object")
    return _claude_call_result(data, stdout)


def parse_claude_stream(stdout: str) -> CallResult:
    """Parse claude -p --output-format stream-json (JSONL event stream).

    Tolerates non-JSON noise; strict about the payload: the last
    type=="result" event is the envelope, no result event -> ParseError.
    """
    result_event: dict | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event: Any = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result_event = event
    if result_event is None:
        raise ParseError("claude stream contained no result event")
    return _claude_call_result(result_event, stdout)


# --- activity extractors (pure): one stream line -> short human note ---------
# These feed the live Observe view ONLY (the scientist's X-ray). They never
# touch constructed messages, so blindness invariants are untouched.


def claude_activity(line: str) -> str | None:
    """Map one claude stream-json line to a live-activity note, or None."""
    try:
        event: Any = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    etype = event.get("type")
    if etype == "system" and event.get("subtype") == "init":
        return "session started"
    if etype == "system" and event.get("subtype") == "thinking_tokens":
        return f"thinking… ~{event.get('estimated_tokens', '?')} tokens"
    if etype == "assistant":
        for block in (event.get("message") or {}).get("content") or []:
            btype = block.get("type") if isinstance(block, dict) else None
            if btype == "tool_use":
                return f"tool: {block.get('name', '?')}"
            if btype == "text" and block.get("text"):
                return "writing answer…"
    if etype == "result":
        return "finalizing"
    return None


def codex_activity(line: str) -> str | None:
    """Map one codex --json line to a live-activity note, or None."""
    try:
        event: Any = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    etype = event.get("type")
    if etype == "thread.started":
        return "session started"
    if etype == "turn.started":
        return "thinking…"
    if etype in ("item.started", "item.updated", "item.completed"):
        item = event.get("item") or {}
        itype = item.get("type")
        if itype == "reasoning":
            lines = (item.get("text") or "").strip().splitlines()
            head = lines[0].strip("* ") if lines else ""
            return f"reasoning: {head[:80]}" if head else "reasoning…"
        if itype == "command_execution":
            return f"running: {str(item.get('command', ''))[:60]}"
        if itype == "web_search":
            return f"web search: {str(item.get('query', ''))[:60]}"
        if itype == "agent_message":
            return "writing answer…"
    return None


def parse_codex_stream(stdout: str) -> CallResult:
    """Parse codex exec --json's JSONL event stream into a CallResult.

    Tolerates non-JSON lines (transport noise) but is strict about the
    payload: no agent_message event -> ParseError, never a guessed answer.
    """
    answer: str | None = None
    usage = Usage()
    session_id: str | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event: Any = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "thread.started":
            session_id = event.get("thread_id")
        elif event_type == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                answer = item["text"]  # last agent_message wins
        elif event_type == "turn.completed":
            usage_raw = event.get("usage") or {}
            usage = Usage(
                input_tokens=usage_raw.get("input_tokens"),
                output_tokens=usage_raw.get("output_tokens"),
                cached_input_tokens=usage_raw.get("cached_input_tokens"),
                reasoning_output_tokens=usage_raw.get("reasoning_output_tokens"),
            )
        elif event_type in ("turn.failed", "error"):
            raise CLICallError(f"codex stream reported failure: {json.dumps(event)[:200]}")
    if answer is None:
        raise ParseError("codex stream contained no agent_message event")
    return CallResult(answer=answer, usage=usage, session_id=session_id, raw_stdout=stdout)


# --- CLI backends -----------------------------------------------------------


def _activity_tap(
    extractor: Callable[[str], str | None],
    on_activity: Callable[[str], None] | None,
) -> Callable[[str], None] | None:
    """Compose a per-line callback from an extractor + activity sink."""
    if on_activity is None:
        return None

    def tap(line: str) -> None:
        detail = extractor(line)
        if detail:
            on_activity(detail)

    return tap


class ClaudeParticipant:
    """claude -p in stream-json mode (probe 2026-07-16): incremental events
    for live observation, final `result` event as the envelope. Role steering
    via system prompt."""

    def __init__(
        self,
        scratch_dir: Path,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        model: str | None = None,
    ) -> None:
        self.scratch_dir = scratch_dir
        self.timeout_s = timeout_s
        self.model = model
        self.name = f"claude-cli:{model}" if model else "claude-cli"
        self.on_activity: Callable[[str], None] | None = None

    def build_argv(self, prompt: str, role: str | None = None) -> list[str]:
        # --verbose is required by the CLI for stream-json in -p mode
        argv = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose"]
        if self.model is not None:
            argv += ["--model", self.model]
        if role is not None:
            argv += ["--append-system-prompt", role]
        return argv

    async def answer(self, prompt: str, role: str | None = None) -> CallResult:
        stdout = await run_cli(
            self.build_argv(prompt, role),
            cwd=self.scratch_dir,
            timeout_s=self.timeout_s,
            on_line=_activity_tap(claude_activity, self.on_activity),
        )
        return parse_claude_stream(stdout)


class CodexParticipant:
    """codex exec --json. No system-prompt flag: role is prepended to the
    prompt with a delimiter (still 'natural + role ONLY' — one variable)."""

    def __init__(
        self,
        scratch_dir: Path,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        model: str | None = None,
    ) -> None:
        self.scratch_dir = scratch_dir
        self.timeout_s = timeout_s
        self.model = model
        self.name = f"codex-cli:{model}" if model else "codex-cli"
        self.on_activity: Callable[[str], None] | None = None

    def build_argv(self, prompt: str, role: str | None = None) -> list[str]:
        full_prompt = prompt if role is None else f"{role}\n\n---\n\n{prompt}"
        argv = ["codex", "exec", "--json", "-s", "read-only", "--skip-git-repo-check"]
        if self.model is not None:
            argv += ["-m", self.model]
        return [*argv, full_prompt]

    async def answer(self, prompt: str, role: str | None = None) -> CallResult:
        stdout = await run_cli(
            self.build_argv(prompt, role),
            cwd=self.scratch_dir,
            timeout_s=self.timeout_s,
            on_line=_activity_tap(codex_activity, self.on_activity),
        )
        return parse_codex_stream(stdout)
