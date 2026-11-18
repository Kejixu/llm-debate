"""CLI helpers: version probes get the same env hygiene as real calls."""

from pathlib import Path


def test_cli_version_strips_host_injected_env(tmp_path: Path, monkeypatch) -> None:
    from llm_debate.cli import cli_version

    fake = tmp_path / "fake-cli"
    fake.write_text('#!/bin/sh\necho "v1 NODE_OPTIONS=[$NODE_OPTIONS]"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("NODE_OPTIONS", "--require=/gone/tempfile.js")
    assert cli_version(str(fake)) == "v1 NODE_OPTIONS=[]"
