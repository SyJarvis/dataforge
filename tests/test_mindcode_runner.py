import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_mindcode.py"
FAKE = r"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

root = Path(os.environ["MINDCODE_CONFIG_DIR"])
root.mkdir(parents=True, exist_ok=True)
with (root / "calls.jsonl").open("a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\n")
if sys.argv[1] == "chat":
    session_id = sys.argv[sys.argv.index("--session") + 1]
    workspace = sys.argv[sys.argv.index("--workspace") + 1]
    assert Path.cwd() == Path(workspace)
    for directory in ("server-sessions", "sessions"):
        (root / directory).mkdir(exist_ok=True)
    snapshot = {"sessionId": session_id, "lastResult": {
        "exitCode": int(os.environ.get("FAKE_EXIT", "0")), "runId": "current-run"},
        "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "agent "},
                                               {"type": "text", "text": "final"}],
             "final": True, "runId": "current-run"},
            {"role": "assistant", "text": "other run", "final": True, "runId": "other-run"},
        ]}
    if os.environ.get("FAKE_AGENT_ERROR"):
        snapshot["lastResult"]["error"] = {"message": os.environ["FAKE_AGENT_ERROR"]}
    (root / "server-sessions" / f"{session_id}.json").write_text(json.dumps(snapshot))
    conversation_id = hashlib.sha256(session_id.encode()).hexdigest()
    (root / "sessions" / f"{conversation_id}.json").write_text(json.dumps({
        "session_id": session_id, "rollout": [{"role": "tool", "content": "full result"}]
    }))
    print("assistant progress\nagent final", flush=True)
    print("diagnostic", file=sys.stderr, flush=True)
    if os.environ.get("FAKE_SLEEP"):
        time.sleep(30)
    sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
elif sys.argv[1:3] == ["task", "cancel"]:
    print("cancel requested")
    sys.exit(int(os.environ.get("FAKE_CANCEL_EXIT", "0")))
elif sys.argv[1:3] == ["task", "status"]:
    if os.environ.get("FAKE_STALE_RESULT"):
        print(json.dumps({"activeRunId": "active", "task": {"status": "RUNNING"},
                          "lastResult": {"outcome": "cancelled", "exitCode": 130}}))
    elif os.environ.get("FAKE_CANCEL_EXIT"):
        print(json.dumps({"activeRunId": "active", "task": {"status": "RUNNING"}}))
    else:
        print(json.dumps({"activeRunId": None, "task": {"status": "CANCELLED"},
                          "lastResult": {"outcome": "cancelled", "exitCode": 130}}))
"""


@pytest.fixture
def runner(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "examples").mkdir()
    shutil.copyfile(SCRIPT, project / "scripts/run_mindcode.py")
    (project / "examples/repositories.json").write_text(
        json.dumps(
            [
                "https://example.com/first.git",
                "https://example.com/second.git",
            ]
        )
    )
    (project / "examples/question_bank.json").write_text(
        json.dumps(
            [
                {
                    "id": "clone_project",
                    "title": "Clone",
                    "prompt": "Clone {repo_url} to {workspace} at {revision}",
                },
                {"id": "analyze_project", "title": "Analyze", "prompt": "Find the entry point."},
                {"id": "run_project", "title": "Run", "prompt": "Run the smallest example."},
            ]
        )
    )
    spec = importlib.util.spec_from_file_location(
        "mindcode_test_runner", project / "scripts/run_mindcode.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake = tmp_path / "fake mindcode"
    fake.write_text(f"#!{sys.executable}\n" + FAKE)
    fake.chmod(0o755)
    monkeypatch.setenv("MINDCODE_CONFIG_DIR", str(tmp_path / "mindcode-config"))
    monkeypatch.chdir(tmp_path)
    return module, fake, tmp_path / "run"


def read_record(output):
    return json.loads((output / "run.json").read_text())


def test_dry_run_formats_defaults_from_script_project_and_does_not_launch(runner, capsys):
    module, fake, output = runner
    assert module.main(["--mindcode", str(fake), "--output-dir", str(output), "--dry-run"]) == 0
    record = read_record(output)
    assert record["status"] == "dry_run"
    assert record["task_ids"] == ["clone_project", "analyze_project"]
    prompt = (output / "prompt.txt").read_text()
    assert f"Clone https://example.com/first.git to {output}/workspace/repo at HEAD" in prompt
    assert prompt.index("Clone") < prompt.index("Find the entry point.")
    assert prompt in capsys.readouterr().out
    assert (output / "workspace").is_dir()
    assert not (output / "workspace/repo").exists()
    assert not (output.parent / "mindcode-config/calls.jsonl").exists()
    assert (output / "stdout.log").read_text() == ""


def test_fake_agent_receives_ordered_tasks_and_persists_only_own_session(runner):
    module, fake, output = runner
    config = output.parent / "mindcode-config"
    (config / "server-sessions").mkdir(parents=True)
    (config / "server-sessions/other.json").write_text('{"unrelated":"private"}')
    code = module.main(
        [
            "--mindcode",
            str(fake),
            "--output-dir",
            str(output),
            "--repo-index",
            "1",
            "--task-id",
            "run_project",
            "--task-id",
            "analyze_project",
            "--revision",
            "v1.2.3",
            "--max-steps",
            "8",
            "--total-timeout",
            "20",
            "--provider",
            "test",
            "--model",
            "fake",
        ]
    )
    assert code == 0
    record = read_record(output)
    assert record["status"] == "agent_finished"
    assert record["task_verified"] is False
    assert record["exit_code"] == 0
    assert record["duration_seconds"] >= 0
    assert record["task_ids"] == ["run_project", "analyze_project"]
    command = record["command"]
    assert command[command.index("--provider") + 1] == "test"
    assert command[command.index("--model") + 1] == "fake"
    assert command[command.index("--approve-mode") + 1] == "full_accept"
    prompt = (output / "prompt.txt").read_text()
    assert "https://example.com/second.git" in prompt
    assert prompt.index("Run the smallest") < prompt.index("Find the entry")
    assert "若目标仓库尚未准备好" in prompt
    assert (output / "stdout.log").read_text() == "assistant progress\nagent final\n"
    assert (output / "stderr.log").read_text() == "diagnostic\n"
    assert json.loads((output / "session.json").read_text())["sessionId"] == record["session_id"]
    conversation = json.loads((output / "conversation.json").read_text())
    assert conversation["session_id"] == record["session_id"]
    assert record["agent_result"]["runId"] == "current-run"
    assert (output / "final.txt").read_text() == "agent final\n"
    assert not (output / "other.json").exists()
    assert len((config / "calls.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize(
    ("code", "status"), [(1, "failed"), (3, "partial_or_needs_input"), (124, "timed_out")]
)
def test_nonzero_agent_exit_never_reports_completion(runner, monkeypatch, code, status):
    module, fake, output = runner
    monkeypatch.setenv("FAKE_EXIT", str(code))
    assert module.main(["--mindcode", str(fake), "--output-dir", str(output)]) == code
    assert read_record(output)["status"] == status
    assert (output / "stdout.log").read_text()


def test_missing_binary_records_launch_failure(runner):
    module, _, output = runner
    assert (
        module.main(
            [
                "--mindcode",
                str(output.parent / "absent"),
                "--output-dir",
                str(output),
            ]
        )
        == 127
    )
    assert read_record(output)["status"] == "launch_error"
    assert "No such file" in (output / "stderr.log").read_text()


def test_agent_error_from_session_is_saved_and_shown(runner, monkeypatch, capsys):
    module, fake, output = runner
    monkeypatch.setenv("FAKE_EXIT", "1")
    monkeypatch.setenv("FAKE_AGENT_ERROR", "Model unavailable: 404")
    assert module.main(["--mindcode", str(fake), "--output-dir", str(output)]) == 1
    assert read_record(output)["agent_result"]["error"]["message"] == "Model unavailable: 404"
    assert json.loads(capsys.readouterr().out)["error"] == "Model unavailable: 404"


def test_final_text_supports_legacy_text_and_requires_matching_run(runner):
    module, _, _ = runner
    message = {"role": "assistant", "text": "legacy text", "final": True, "runId": "run"}
    assert module._final_text([message], "run") == "legacy text"
    assert module._final_text([message], "different-run") is None
    assert module._final_text([message], None) is None


def test_nonempty_output_is_never_overwritten(runner):
    module, fake, output = runner
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("original")
    assert module.main(["--mindcode", str(fake), "--output-dir", str(output)]) == 2
    assert sentinel.read_text() == "original"
    assert not (output / "prompt.txt").exists()


@pytest.mark.parametrize(
    "arguments", [["--repo-index", "-1"], ["--repo-index", "2"], ["--task-id", "missing"]]
)
def test_invalid_task_selection_fails_before_creating_output(runner, arguments):
    module, _, output = runner
    assert module.main(["--output-dir", str(output), *arguments]) == 2
    assert not output.exists()


@pytest.mark.parametrize("cancel_state", ["confirmed", "failed", "stale"])
def test_outer_timeout_cancels_server_task_before_reaping_cli(runner, monkeypatch, cancel_state):
    module, fake, output = runner
    monkeypatch.setattr(module, "CLI_SHUTDOWN_GRACE_SECONDS", 0)
    monkeypatch.setenv("FAKE_SLEEP", "1")
    if cancel_state == "failed":
        monkeypatch.setenv("FAKE_CANCEL_EXIT", "1")
    elif cancel_state == "stale":
        monkeypatch.setenv("FAKE_STALE_RESULT", "1")
    code = module.main(
        [
            "--mindcode",
            str(fake),
            "--output-dir",
            str(output),
            "--total-timeout",
            "1",
        ]
    )
    assert code == 124
    record = read_record(output)
    assert record["status"] == "timed_out"
    assert record["exit_code"] < 0
    assert record["cancellation"]["requested"] is (cancel_state != "failed")
    assert record["cancellation"]["confirmed"] is (cancel_state == "confirmed")
    lines = (output.parent / "mindcode-config/calls.jsonl").read_text().splitlines()
    calls = [json.loads(line) for line in lines]
    assert [call[:2] for call in calls[1:]] == [["task", "cancel"], ["task", "status"]]
    assert all(call[-1] == record["session_id"] for call in calls[1:])
