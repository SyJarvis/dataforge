#!/usr/bin/env python3
"""Assemble repository tasks and run the user's installed mindcode CLI once."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_SHUTDOWN_GRACE_SECONDS = 30


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _prompt(repo_index, task_ids, revision, workspace):
    repositories = json.loads((PROJECT_ROOT / "examples/repositories.json").read_text())
    questions = json.loads((PROJECT_ROOT / "examples/question_bank.json").read_text())
    if not isinstance(repositories, list) or not all(
        isinstance(url, str) and url.strip() for url in repositories
    ):
        raise ValueError("repositories.json must contain a list of nonempty repository URLs")
    if not 0 <= repo_index < len(repositories):
        raise ValueError(f"repo-index must be between 0 and {len(repositories) - 1}")
    if not isinstance(questions, list) or not all(
        isinstance(task, dict)
        and isinstance(task.get("id"), str)
        and isinstance(task.get("prompt"), str)
        for task in questions
    ):
        raise ValueError("question_bank.json must contain task objects with id and prompt strings")
    by_id = {task["id"]: task for task in questions}
    if len(by_id) != len(questions):
        raise ValueError("question_bank.json contains duplicate task IDs")
    unknown = [task_id for task_id in task_ids if task_id not in by_id]
    if unknown:
        raise ValueError(f"unknown task IDs: {', '.join(unknown)}")
    repo_url = repositories[repo_index]
    target = workspace / "repo"
    parts = [
        f"仓库 URL：{repo_url}\n目标仓库路径：{target}\n指定版本：{revision}",
        f"本次工作目录：{workspace}。项目克隆、依赖环境和生成文件均放在此目录中，"
        "不要修改其他目录。无需提交或推送代码。",
        "若目标仓库尚未准备好，先将上述仓库克隆到目标仓库路径并检出指定版本，"
        "再依次完成以下任务。最终说明实际执行结果、代码版本和任何阻塞点。",
    ]
    for index, task_id in enumerate(task_ids, 1):
        task = by_id[task_id]
        try:
            text = task["prompt"].format(
                repo_url=repo_url,
                workspace=str(target),
                revision=revision,
            )
        except (KeyError, ValueError, IndexError) as error:
            raise ValueError(f"cannot format task {task_id}: {error}") from error
        parts.append(f"{index}. {task.get('title', task_id)}（{task_id}）\n{text}")
    return repo_url, target, "\n\n".join(parts) + "\n"


def _cancel_task(binary, session_id, output):
    """Cancel the server task; killing the attached chat CLI alone cannot do this."""
    record = {"requested": False, "confirmed": False}
    for action in ("cancel", "status"):
        command = [binary, "task", action, session_id]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as error:
            record[f"{action}_error"] = str(error)
            continue
        (output / f"{action}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (output / f"{action}.stderr.log").write_text(result.stderr, encoding="utf-8")
        record[f"{action}_exit_code"] = result.returncode
        if action == "cancel":
            record["requested"] = result.returncode == 0
        elif result.returncode == 0:
            try:
                status = json.loads(result.stdout)
                task = status.get("task") or {}
                last_result = status.get("lastResult") or {}
                record["task_status"] = task.get("status")
                record["active_run_id"] = status.get("activeRunId")
                record["confirmed"] = not status.get("activeRunId") and (
                    task.get("status") == "CANCELLED"
                    or (
                        not task
                        and (
                            last_result.get("outcome") == "cancelled"
                            or last_result.get("exitCode") == 130
                        )
                    )
                )
            except (ValueError, AttributeError) as error:
                record["status_error"] = str(error)
    return record


def _copy_session(session_id, output):
    root = Path(os.environ.get("MINDCODE_CONFIG_DIR", "~/.mindcode")).expanduser()
    conversation_id = hashlib.sha256(session_id.encode()).hexdigest()
    artifacts = {}
    for source, name in (
        (root / "server-sessions" / f"{session_id}.json", "session.json"),
        (root / "sessions" / f"{conversation_id}.json", "conversation.json"),
    ):
        try:
            shutil.copyfile(source, output / name)
            artifacts[name] = {"copied": True, "source": str(source)}
        except OSError as error:
            artifacts[name] = {"copied": False, "error": str(error)}
    return artifacts


def _final_text(messages, run_id):
    for message in reversed(messages):
        if (
            run_id
            and message.get("runId") == run_id
            and message.get("final") is True
            and message.get("role") == "assistant"
        ):
            content = message.get("content") or []
            text = "".join(
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
            if not text and isinstance(message.get("text"), str):
                text = message["text"]
            if text:
                return text
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-index", type=int, default=0)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--mindcode", default="~/base/bin/mindcode")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=_positive_int, default=40)
    parser.add_argument("--step-timeout", type=_positive_int, default=120)
    parser.add_argument("--total-timeout", type=_positive_int, default=600)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    session_id = uuid.uuid4().hex
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + session_id[:8]
    output = (args.output_dir or PROJECT_ROOT / "runs/mindcode" / run_name).expanduser().resolve()
    workspace = output / "workspace"
    task_ids = args.task_ids or ["clone_project", "analyze_project"]
    try:
        repo_url, target, prompt = _prompt(args.repo_index, task_ids, args.revision, workspace)
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"output directory must be empty: {output}")
        workspace.mkdir(parents=True)
    except (OSError, ValueError) as error:
        print(f"Cannot prepare run: {error}", file=sys.stderr)
        return 2
    binary = os.path.expanduser(args.mindcode)
    binary = shutil.which(binary) or str(Path(binary).absolute())
    command = [
        binary,
        "chat",
        "--message",
        prompt,
        "--workspace",
        str(workspace),
        "--cwd",
        str(workspace),
        "--session",
        session_id,
        "--approve-mode",
        "full_accept",
        "--no-stream",
        "--max-steps",
        str(args.max_steps),
        "--step-timeout",
        str(args.step_timeout),
        "--total-timeout",
        str(args.total_timeout),
    ]
    for flag, value in (("--provider", args.provider), ("--model", args.model)):
        if value is not None:
            command.extend([flag, value])
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    record = {
        "session_id": session_id,
        "command": command,
        "task_ids": task_ids,
        "repository": {"index": args.repo_index, "url": repo_url, "revision": args.revision},
        "workspace": str(workspace),
        "repository_path": str(target),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run" if args.dry_run else "running",
        "task_verified": False,
        "exit_code": None,
    }
    run_file = output / "run.json"
    run_file.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    exit_code = 0
    with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
        if not args.dry_run:
            process = None
            try:
                process = subprocess.Popen(
                    command,
                    cwd=workspace,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                exit_code = process.wait(timeout=args.total_timeout + CLI_SHUTDOWN_GRACE_SECONDS)
                record["exit_code"] = exit_code
                record["status"] = {
                    0: "agent_finished",
                    3: "partial_or_needs_input",
                    124: "timed_out",
                    130: "cancelled",
                }.get(exit_code, "failed")
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                exit_code = 124 if isinstance(error, subprocess.TimeoutExpired) else 130
                record["status"] = "timed_out" if exit_code == 124 else "cancelled"
                record["cancellation"] = _cancel_task(binary, session_id, output)
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    record["exit_code"] = process.wait()
            except OSError as error:
                exit_code = 127
                record.update(status="launch_error", error=str(error))
                stderr.write(str(error) + "\n")
            record["session_artifacts"] = _copy_session(session_id, output)
            if record["session_artifacts"]["session.json"]["copied"]:
                try:
                    snapshot = json.loads((output / "session.json").read_text(encoding="utf-8"))
                    agent_result = snapshot.get("lastResult") or {}
                    if not isinstance(agent_result, dict):
                        raise ValueError("session lastResult must be an object")
                    record["agent_result"] = agent_result
                    final_text = _final_text(
                        snapshot.get("messages", []), agent_result.get("runId")
                    )
                    if final_text is not None:
                        (output / "final.txt").write_text(final_text + "\n", encoding="utf-8")
                except (OSError, ValueError, AttributeError) as error:
                    record["session_read_error"] = str(error)
    record.update(
        runner_exit_code=exit_code,
        ended_at=datetime.now(timezone.utc).isoformat(),
        duration_seconds=round(time.monotonic() - started, 3),
    )
    run_file.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(prompt)
    summary = {
        "output_dir": str(output),
        "status": record["status"],
        "exit_code": exit_code,
        "session_id": session_id,
    }
    agent_error = record.get("agent_result", {}).get("error") or {}
    if isinstance(agent_error, dict) and agent_error.get("message"):
        summary["error"] = agent_error["message"]
    print(json.dumps(summary, ensure_ascii=False))
    return exit_code if exit_code >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
