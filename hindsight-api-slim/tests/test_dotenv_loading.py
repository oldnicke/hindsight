"""Tests for side-effect-free library imports and explicit CLI dotenv loading."""

import os
import subprocess
import sys
from pathlib import Path


def test_importing_config_does_not_modify_environment(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("HOST_APP_SECRET=from-dotenv\n")
    child_dir = tmp_path / "child"
    child_dir.mkdir()

    env = os.environ.copy()
    env["HOST_APP_SECRET"] = "from-application"
    api_source = str(Path(__file__).parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (api_source, env.get("PYTHONPATH"))))

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import hindsight_api.config; print(os.environ['HOST_APP_SECRET'])",
        ],
        cwd=child_dir,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "from-application"


def test_cli_dotenv_loading_only_fills_missing_values(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("HOST_APP_SECRET=from-dotenv\nHINDSIGHT_TEST_MISSING=loaded\n")
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    monkeypatch.chdir(child_dir)
    monkeypatch.setenv("HOST_APP_SECRET", "from-application")
    monkeypatch.delenv("HINDSIGHT_TEST_MISSING", raising=False)

    from hindsight_api.config import load_dotenv_for_cli

    load_dotenv_for_cli()

    assert os.environ["HOST_APP_SECRET"] == "from-application"
    assert os.environ["HINDSIGHT_TEST_MISSING"] == "loaded"
