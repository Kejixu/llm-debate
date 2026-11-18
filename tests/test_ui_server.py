"""Cockpit server: run listing, historical replay over SSE, static page."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from llm_debate.ui.server import create_app

LOCAL = "http://127.0.0.1:8710"


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
    client = TestClient(create_app(tmp_path), base_url=LOCAL)
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
    client = TestClient(create_app(tmp_path), base_url=LOCAL)
    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
    assert [p["type"] for p in payloads] == ["state", "terminal", "stream_end"]
    assert payloads[0]["to"] == "independent_answer"


def test_unknown_run_404s(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path), base_url=LOCAL)
    assert client.get("/api/runs/nope/events").status_code == 404
    assert client.get("/api/runs/nope/transcript").status_code == 404


def test_reexport_unknown_run_404s(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path), base_url=LOCAL)
    assert client.post("/api/runs/nope/export").status_code == 404


def test_reexport_without_keys_400s(tmp_path: Path, monkeypatch) -> None:
    run_id = make_finished_run(tmp_path)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    client = TestClient(create_app(tmp_path), base_url=LOCAL)
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
    client = TestClient(create_app(tmp_path), base_url=LOCAL)
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

    class FakeClient:
        pass

    def boom(run_dir, client):
        raise RuntimeError("network down")

    monkeypatch.setattr(server_mod, "_make_langfuse_client", lambda: FakeClient())
    monkeypatch.setattr(server_mod, "_export_and_record", boom)
    client = TestClient(create_app(tmp_path), base_url=LOCAL)
    response = client.post(f"/api/runs/{run_id}/export")
    assert response.status_code == 502
    assert "network down" in response.json()["detail"]


def test_static_assets_served(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path), base_url=LOCAL)
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


# --- local-only guard: DNS rebinding + cross-site POSTs are refused ----------


def test_foreign_host_header_is_refused(tmp_path: Path) -> None:
    make_finished_run(tmp_path)
    client = TestClient(create_app(tmp_path), base_url="http://evil.example:8710")
    assert client.get("/api/runs").status_code == 403
    assert client.post("/api/runs", json={"prompt": "spend money"}).status_code == 403


def test_localhost_variants_are_allowed(tmp_path: Path) -> None:
    from llm_debate.ui.server import _hostname

    for base in ("http://127.0.0.1:8710", "http://localhost:8710", "http://localhost"):
        assert TestClient(create_app(tmp_path), base_url=base).get("/api/runs").status_code == 200
    assert _hostname("[::1]:8710") == "[::1]"  # TestClient can't speak IPv6 hosts


def test_cross_origin_post_is_refused(tmp_path: Path) -> None:
    run_id = make_finished_run(tmp_path)
    client = TestClient(create_app(tmp_path), base_url=LOCAL)
    response = client.post(f"/api/runs/{run_id}/export", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403
    other_port = client.post(
        f"/api/runs/{run_id}/export", headers={"Origin": "http://127.0.0.1:3000"}
    )
    assert other_port.status_code == 403  # another localhost app is another origin
    malformed = client.post(f"/api/runs/{run_id}/export", headers={"Origin": "http://[::1"})
    assert malformed.status_code == 403
    same_origin = client.post(
        "/api/runs", json={"prompt": ""}, headers={"Origin": "http://127.0.0.1:8710"}
    )
    assert same_origin.status_code == 422  # passed the guard, failed validation


def test_replay_skips_a_torn_last_line(tmp_path: Path) -> None:
    run_id = make_finished_run(tmp_path)
    events_path = tmp_path / run_id / "events.jsonl"
    events_path.write_text(events_path.read_text() + '\n{"ts": 3, "type": "partic')
    client = TestClient(create_app(tmp_path), base_url=LOCAL)
    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        body = "".join(response.iter_text())
    events = [json.loads(chunk[6:]) for chunk in body.split("\n\n") if chunk.startswith("data: ")]
    assert [e["type"] for e in events] == ["state", "terminal", "stream_end"]


def test_replay_surfaces_mid_log_corruption(tmp_path: Path) -> None:
    run_id = make_finished_run(tmp_path)
    events_path = tmp_path / run_id / "events.jsonl"
    good = events_path.read_text().splitlines()
    events_path.write_text("\n".join([good[0], '{"broken', good[1]]))
    client = TestClient(create_app(tmp_path), base_url=LOCAL)
    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        body = "".join(response.iter_text())
    events = [json.loads(chunk[6:]) for chunk in body.split("\n\n") if chunk.startswith("data: ")]
    assert [e["type"] for e in events] == ["log_corrupt", "stream_end"]
