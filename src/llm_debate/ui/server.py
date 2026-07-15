"""The cockpit server: POST a prompt, watch the debate stream live over SSE.

    uv run llm-debate-ui        # then open http://127.0.0.1:8710

Design: the runner's sink is TEED — every event goes to RunStorage (disk,
source of truth, exactly as the CLI does) AND to an in-memory broadcast for
live subscribers. Finished runs replay from events.jsonl, so the same UI
renders live debates and historical ones identically (event sourcing again).
If LANGFUSE_* keys are in the env, completed runs auto-export.
"""

import asyncio
import json
import os
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from llm_debate.adapters import ClaudeParticipant, CodexParticipant, Participant
from llm_debate.judge import CLIJudge
from llm_debate.machine import DebateRunner
from llm_debate.runlog import RunStorage, render_transcript
from llm_debate.types import RunConfig

STATIC = Path(__file__).parent / "static"


def _make_langfuse_client():
    from langfuse import Langfuse

    return Langfuse()


def _export_and_record(run_dir: Path, client):
    from llm_debate.export_langfuse import export_and_record

    return export_and_record(run_dir, client)


class LaunchRequest(BaseModel):
    prompt: str = Field(min_length=1)
    cap: int = 10
    condition: Literal["natural", "steered"] = "natural"
    max_minutes: float = 30.0
    timeout_s: float = 600.0
    model_a: str | None = None
    model_b: str | None = None
    judge: Literal["claude", "codex"] = "claude"
    judge_model: str | None = None


class ActiveRun:
    """In-memory fan-out for one live run: history for late joiners + queues."""

    def __init__(self) -> None:
        self.history: list[dict] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.done = False

    def publish(self, event: dict) -> None:
        self.history.append(event)
        for queue in list(self.subscribers):
            queue.put_nowait(event)


def create_app(runs_dir: Path = Path("runs")) -> FastAPI:
    app = FastAPI(title="llm-debate cockpit")
    active: dict[str, ActiveRun] = {}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text()

    @app.get("/static/{name}")
    def static_file(name: str) -> PlainTextResponse:
        path = STATIC / name
        if not path.is_file() or path.parent != STATIC:
            raise HTTPException(404, "not found")
        media = {"css": "text/css", "js": "text/javascript"}.get(path.suffix.lstrip("."))
        return PlainTextResponse(path.read_text(), media_type=media or "text/plain")

    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        entries = []
        if runs_dir.exists():
            for run in sorted(runs_dir.iterdir(), reverse=True):
                if not (run / "config.json").exists():
                    continue
                config = json.loads((run / "config.json").read_text())
                status = "running" if run.name in active and not active[run.name].done else None
                status_file = run / "status.json"
                if status is None:
                    status = (
                        json.loads(status_file.read_text())["status"]
                        if status_file.exists()
                        else "incomplete"
                    )
                trace_url = None
                trace_file = run / "trace.json"
                if trace_file.exists():
                    trace_url = json.loads(trace_file.read_text()).get("url")
                created_at = ""
                stamp = run.name[:15]
                with suppress(ValueError):
                    created_at = datetime.strptime(stamp, "%Y%m%d-%H%M%S").isoformat()
                entries.append(
                    {
                        "id": run.name,
                        "task": config.get("task", ""),
                        "status": status,
                        "participants": config.get("participants", {}),
                        "condition": (config.get("config") or {}).get("condition", "natural"),
                        "judge": config.get("judge", ""),
                        "trace_url": trace_url,
                        "created_at": created_at,
                    }
                )
        return entries

    @app.post("/api/runs")
    async def launch(request: LaunchRequest) -> dict:
        storage = RunStorage(runs_dir)
        run = ActiveRun()
        active[storage.dir.name] = run

        def sink(event: dict) -> None:
            storage.append(event)
            run.publish(event)

        config = RunConfig(
            cap=request.cap,
            condition=request.condition,
            max_minutes=request.max_minutes,
            per_call_timeout_s=request.timeout_s,
        )
        participant_a = ClaudeParticipant(
            storage.scratch("participant_a"),
            timeout_s=config.per_call_timeout_s,
            model=request.model_a,
        )
        participant_b = CodexParticipant(
            storage.scratch("participant_b"),
            timeout_s=config.per_call_timeout_s,
            model=request.model_b,
        )
        judge_backend: Participant = (
            ClaudeParticipant(
                storage.scratch("judge"),
                timeout_s=config.per_call_timeout_s,
                model=request.judge_model,
            )
            if request.judge == "claude"
            else CodexParticipant(
                storage.scratch("judge"),
                timeout_s=config.per_call_timeout_s,
                model=request.judge_model,
            )
        )
        judge = CLIJudge(
            judge_backend,
            max_retries=config.max_retries,
            base_delay_s=config.retry_base_delay_s,
            sink=sink,
        )
        storage.write_config(
            {
                "task": request.prompt,
                "config": asdict(config),
                "participants": {"A": participant_a.name, "B": participant_b.name},
                "judge": judge.name,
            }
        )
        runner = DebateRunner(participant_a, participant_b, judge, config, sink=sink)

        async def run_to_completion() -> None:
            try:
                result = await runner.run(request.prompt)
                storage.write_status(result)
                storage.write_transcript(render_transcript(request.prompt, result))
                if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
                    try:
                        client = _make_langfuse_client()
                        record = _export_and_record(storage.dir, client)
                        client.flush()
                        run.publish(
                            {"type": "exported", "run": storage.dir.name, "url": record["url"]}
                        )
                    except Exception as exc:  # export is best-effort, never kills a run
                        run.publish({"type": "export_failed", "error": str(exc)})
            finally:
                run.done = True
                run.publish({"type": "stream_end"})

        asyncio.get_running_loop().create_task(run_to_completion())
        return {"run_id": storage.dir.name}

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str) -> StreamingResponse:
        live = active.get(run_id)
        run_dir = runs_dir / run_id
        if live is None and not (run_dir / "events.jsonl").exists():
            raise HTTPException(404, "unknown run")

        async def stream():
            if live is not None and not live.done:
                queue: asyncio.Queue = asyncio.Queue()
                live.subscribers.add(queue)
                try:
                    for event in live.history:  # replay for late joiners
                        yield _sse(event)
                    while True:
                        event = await queue.get()
                        yield _sse(event)
                        if event["type"] == "stream_end":
                            return
                finally:
                    live.subscribers.discard(queue)
            else:  # finished run: replay events.jsonl — same wire format
                for line in (run_dir / "events.jsonl").read_text().splitlines():
                    if line.strip():
                        yield _sse(json.loads(line))
                yield _sse({"type": "stream_end"})

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/runs/{run_id}/transcript", response_class=PlainTextResponse)
    def transcript(run_id: str) -> str:
        path = runs_dir / run_id / "transcript.md"
        if not path.exists():
            raise HTTPException(404, "no transcript (run still going or failed early)")
        return path.read_text()

    @app.post("/api/runs/{run_id}/export")
    def reexport(run_id: str) -> dict:
        run_dir = runs_dir / run_id
        if not (run_dir / "events.jsonl").exists():
            raise HTTPException(404, "unknown run")
        if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
            raise HTTPException(400, "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")
        client = _make_langfuse_client()
        try:
            record = _export_and_record(run_dir, client)
        except Exception as exc:
            raise HTTPException(502, f"export failed: {exc}") from exc
        finally:
            client.flush()
        return record

    return app


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


app = create_app()


def main() -> None:
    print("llm-debate cockpit: http://127.0.0.1:8710")
    uvicorn.run(app, host="127.0.0.1", port=8710, log_level="warning")
