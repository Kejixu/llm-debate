"""Shared probe plumbing: run a CLI call, capture everything, save a fixture.

Probes run each CLI from an EMPTY scratch dir on purpose: agent CLIs load
instruction files (CLAUDE.md / AGENTS.md) from their cwd, and debaters must
never inherit this repo's context. Same isolation the real adapter will use.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures"
SCRATCH = Path(__file__).parent / "scratch"


def run_probe(name: str, argv: list[str], timeout_s: float = 180.0) -> dict[str, Any]:
    FIXTURES.mkdir(exist_ok=True)
    cwd = SCRATCH / name
    cwd.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
        )
        result: dict[str, Any] = {
            "argv": argv,
            "exit_code": proc.returncode,
            "duration_s": round(time.monotonic() - started, 2),
            "timed_out": False,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        result = {
            "argv": argv,
            "exit_code": None,
            "duration_s": round(time.monotonic() - started, 2),
            "timed_out": True,
            "stdout": _as_text(exc.stdout),
            "stderr": _as_text(exc.stderr),
        }
    except FileNotFoundError:
        result = {
            "argv": argv,
            "exit_code": None,
            "duration_s": 0.0,
            "timed_out": False,
            "stdout": "",
            "stderr": f"binary not found: {argv[0]}",
        }
    (FIXTURES / f"{name}.json").write_text(json.dumps(result, indent=2))
    return result


def _as_text(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return data


def summarize(name: str, result: dict[str, Any]) -> None:
    status = "TIMEOUT" if result["timed_out"] else f"exit={result['exit_code']}"
    print(f"[{name}] {status} in {result['duration_s']}s")
    print(f"  stdout: {len(result['stdout'])} bytes | stderr: {len(result['stderr'])} bytes")
