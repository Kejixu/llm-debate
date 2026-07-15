"""Cockpit server: run listing, historical replay over SSE, static page."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from llm_debate.ui.server import create_app


def make_finished_run(runs_dir: Path) -> str:
    run = runs_dir / "20260714-000000-abc123"
    run.mkdir(parents=True)
    (run / "config.json").write_text(
        json.dumps(
            {
                "task": "pick a db",
                "config": {"condition": "natural"},
                "participants": {"A": "claude-cli", "B": "codex-cli:gpt-5.5"},
                "judge": "judge-claude-cli",
            }
        )
    )
    (run / "status.json").write_text(json.dumps({"status": "consensus"}))
    (run / "trace.json").write_text(
        json.dumps({"trace_id": "t1", "url": "https://cloud.example/t/t1"})
    )
    events = [
        {
            "ts": 1,
            "type": "state",
            "trigger": "start",
            "from": "init",
            "to": "independent_answer",
            "round": 0,
        },
        {"ts": 2, "type": "terminal", "status": "consensus", "rounds_completed": 0, "error": None},
    ]
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    return run.name


def test_index_and_run_listing(tmp_path: Path) -> None:
    run_id = make_finished_run(tmp_path)
    client = TestClient(create_app(tmp_path))
    assert "llm-debate" in client.get("/").text
    (entry,) = client.get("/api/runs").json()
    assert entry["id"] == run_id
    assert entry["status"] == "consensus"
    assert entry["participants"] == {"A": "claude-cli", "B": "codex-cli:gpt-5.5"}
    assert entry["condition"] == "natural"
    assert entry["judge"] == "judge-claude-cli"
    assert entry["trace_url"] == "https://cloud.example/t/t1"
    assert entry["created_at"] == "2026-07-14T00:00:00"


def test_finished_run_replays_over_sse(tmp_path: Path) -> None:
    run_id = make_finished_run(tmp_path)
    client = TestClient(create_app(tmp_path))
    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
    assert [p["type"] for p in payloads] == ["state", "terminal", "stream_end"]
    assert payloads[0]["to"] == "independent_answer"


def test_unknown_run_404s(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/runs/nope/events").status_code == 404
    assert client.get("/api/runs/nope/transcript").status_code == 404


def test_reexport_unknown_run_404s(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    assert client.post("/api/runs/nope/export").status_code == 404


def test_reexport_without_keys_400s(tmp_path: Path, monkeypatch) -> None:
    run_id = make_finished_run(tmp_path)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    client = TestClient(create_app(tmp_path))
    response = client.post(f"/api/runs/{run_id}/export")
    assert response.status_code == 400
    assert "LANGFUSE_PUBLIC_KEY" in response.json()["detail"]


def test_reexport_happy_path_uses_exporter(tmp_path: Path, monkeypatch) -> None:
    run_id = make_finished_run(tmp_path)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

    import llm_debate.ui.server as server_mod

    def fake_export(run_dir, client):
        return {
            "trace_id": "t-new",
            "url": "https://cloud.example/t/t-new",
            "exported_at": "2026-07-15T00:00:00+00:00",
        }

    class FakeClient:
        def flush(self) -> None: ...

    monkeypatch.setattr(server_mod, "_make_langfuse_client", lambda: FakeClient())
    monkeypatch.setattr(server_mod, "_export_and_record", fake_export)
    client = TestClient(create_app(tmp_path))
    body = client.post(f"/api/runs/{run_id}/export").json()
    assert body == {
        "trace_id": "t-new",
        "url": "https://cloud.example/t/t-new",
        "exported_at": "2026-07-15T00:00:00+00:00",
    }


def test_reexport_failure_returns_502(tmp_path: Path, monkeypatch) -> None:
    run_id = make_finished_run(tmp_path)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

    import llm_debate.ui.server as server_mod

    flushed = []

    class FakeClient:
        def flush(self) -> None:
            flushed.append(True)

    def boom(run_dir, client):
        raise RuntimeError("network down")

    monkeypatch.setattr(server_mod, "_make_langfuse_client", lambda: FakeClient())
    monkeypatch.setattr(server_mod, "_export_and_record", boom)
    client = TestClient(create_app(tmp_path))
    response = client.post(f"/api/runs/{run_id}/export")
    assert response.status_code == 502
    assert "network down" in response.json()["detail"]
    assert flushed  # flush guaranteed even on failure


def test_static_assets_served(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
