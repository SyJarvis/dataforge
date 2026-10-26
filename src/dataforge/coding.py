"""Check trusted, author-owned coding assets; never execute model code here.

Workers have fresh Python processes and a wall-clock timeout, not a security
sandbox. Network, filesystem access, and memory are not isolated or limited.
"""

import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Nonempty = Annotated[str, Field(min_length=1, pattern=r"\S")]
Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")]


class PackModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class CodingRuntime(PackModel):
    language: Literal["python"]
    version: Literal["3.13"]
    timeout_seconds: Literal[2]
    memory_mb: Literal[256]


class CodingCase(PackModel):
    id: Nonempty
    visibility: Literal["public", "hidden"]
    args: list[JsonValue]
    expected: JsonValue = None
    raises: Literal["ValueError"] | None = None

    @model_validator(mode="after")
    def validate_outcome(self):
        outcomes = self.model_fields_set & {"expected", "raises"}
        if len(outcomes) != 1 or ("raises" in outcomes and self.raises is None):
            raise ValueError("case requires exactly one of expected or raises='ValueError'")
        json.dumps(self.args, allow_nan=False)
        if "expected" in outcomes:
            json.dumps(self.expected, allow_nan=False)
        return self


class CodingMutant(PackModel):
    file: Annotated[str, Field(pattern=r"^mutants/[a-zA-Z0-9_][a-zA-Z0-9_-]*\.py$")]
    description: Nonempty


class CodingTask(PackModel):
    schema_version: Literal["1.0"]
    id: Identifier
    title: Nonempty
    task_kind: Literal["function_implementation", "bug_fix"]
    prompt: Nonempty
    entry_point: Identifier
    signature: Nonempty
    runtime: CodingRuntime
    preserve_inputs: bool
    cases: list[CodingCase]
    mutants: list[CodingMutant] = Field(min_length=3, max_length=3)
    starter_file: Literal["buggy.py"] | None = None

    @model_validator(mode="after")
    def validate_task(self):
        if not self.preserve_inputs:
            raise ValueError("preserve_inputs must be true")
        if (self.task_kind == "bug_fix") != (self.starter_file == "buggy.py"):
            raise ValueError("only bug_fix tasks require starter_file='buggy.py'")
        case_ids = [case.id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case IDs must be unique")
        if sum(case.visibility == "public" for case in self.cases) < 2:
            raise ValueError("task requires at least 2 public cases")
        if sum(case.visibility == "hidden" for case in self.cases) < 6:
            raise ValueError("task requires at least 6 hidden cases")
        files = [mutant.file for mutant in self.mutants]
        if len(set(files)) != len(files):
            raise ValueError("mutant paths must be unique")
        return self


def _asset_path(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"asset must be an existing file inside its task directory: {relative}")
    return path


def load_task(directory: str | Path) -> CodingTask:
    """Load strict task metadata and check all declared asset paths."""
    directory = Path(directory)
    task = CodingTask.model_validate_json(_asset_path(directory, "task.json").read_bytes())
    for relative in ["reference.py", *[mutant.file for mutant in task.mutants]]:
        _asset_path(directory, relative)
    if task.starter_file:
        _asset_path(directory, task.starter_file)
    return task


def public_task(directory: str | Path) -> dict:
    """Return only the statement, signature, public examples, and optional buggy source."""
    directory = Path(directory)
    task = load_task(directory)
    view = {
        "id": task.id,
        "title": task.title,
        "task_kind": task.task_kind,
        "prompt": task.prompt,
        "entry_point": task.entry_point,
        "signature": task.signature,
        "runtime": task.runtime.model_dump(),
        "preserve_inputs": task.preserve_inputs,
        "examples": [
            case.model_dump(exclude_unset=True)
            for case in task.cases
            if case.visibility == "public"
        ],
    }
    if task.starter_file:
        view["buggy_source"] = _asset_path(directory, task.starter_file).read_text(encoding="utf-8")
    return view


# This protocol is for trusted author assets only. It does not defend against a
# program that alters the harness, forges output, or accesses host resources.
_WORKER = r"""
import contextlib
import copy
import inspect
import io
import json
import sys
from pathlib import Path


def same_json(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(actual) is list:
        return len(actual) == len(expected) and all(map(same_json, actual, expected))
    if type(actual) is dict:
        return actual.keys() == expected.keys() and all(
            same_json(actual[key], expected[key]) for key in expected
        )
    return type(actual) in (str, bool, int, float, type(None)) and actual == expected


def check():
    payload = json.load(sys.stdin)
    path = Path(sys.argv[1])
    try:
        code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    except Exception as error:
        return {"status": "invalid_asset", "error": f"{type(error).__name__}: {error}"}
    results = []
    for case in payload["cases"]:
        args = copy.deepcopy(case["args"])
        namespace = {"__name__": "coding_asset", "__file__": str(path)}
        captured_out, captured_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(captured_out), contextlib.redirect_stderr(captured_err):
            try:
                exec(code, namespace)
                function = namespace.get(payload["entry_point"])
                if not callable(function):
                    raise ValueError("entry point must be callable")
                inspect.signature(function).bind(*args)
            except BaseException as error:
                return {"status": "invalid_asset", "error": f"{type(error).__name__}: {error}"}
            try:
                actual = function(*args)
            except Exception as error:
                passed = case.get("raises") == "ValueError" and type(error) is ValueError
                reason = None if passed else f"unexpected_exception:{type(error).__name__}"
            else:
                passed = "expected" in case and same_json(actual, case["expected"])
                reason = None if passed else "output_mismatch"
        if not same_json(args, case["args"]):
            passed, reason = False, "input_modified"
        results.append({
            "id": case["id"], "visibility": case["visibility"],
            "passed": passed, "reason": reason,
        })
    return {"status": "passed" if all(row["passed"] for row in results) else "failed",
            "cases": results}


print(json.dumps(check(), allow_nan=False))
"""


def _check_asset(path: Path, task: CodingTask) -> dict:
    payload = {
        "entry_point": task.entry_point,
        "cases": [case.model_dump(exclude_unset=True) for case in task.cases],
    }
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", _WORKER, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(
            json.dumps(payload, ensure_ascii=False, allow_nan=False),
            timeout=task.runtime.timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        return {"status": "timeout", "error": "asset exceeded wall-clock timeout"}
    if process.returncode:
        return {
            "status": "invalid_asset",
            "error": stderr[-2000:] or "worker exited unsuccessfully",
        }
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"status": "invalid_asset", "error": "worker returned invalid JSON"}


def check_coding_pack(root: str | Path) -> dict:
    """Validate a directory of trusted task packages and return an auditable report."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"coding pack directory does not exist: {root}")
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    if not directories:
        raise ValueError("coding pack contains no task directories")
    results, seen = [], set()
    for directory in directories:
        row = {"id": directory.name, "passed": False, "errors": []}
        results.append(row)
        try:
            if not directory.resolve().is_relative_to(root):
                raise ValueError("task directory cannot escape the coding pack")
            task = load_task(directory)
            if task.id in seen:
                raise ValueError(f"duplicate task ID: {task.id}")
            seen.add(task.id)
            row["id"] = task.id
            if task.id != directory.name:
                raise ValueError("task ID must match its directory name")
            actual_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            if actual_version != task.runtime.version:
                raise ValueError(
                    f"task requires Python {task.runtime.version}; actual Python {actual_version}"
                )
            row["runtime"] = task.runtime.model_dump()
            row["public_cases"] = sum(case.visibility == "public" for case in task.cases)
            row["hidden_cases"] = sum(case.visibility == "hidden" for case in task.cases)
            files = ["task.json", "reference.py", *[mutant.file for mutant in task.mutants]]
            if task.starter_file:
                files.append(task.starter_file)
            row["assets_sha256"] = {
                name: hashlib.sha256(_asset_path(directory, name).read_bytes()).hexdigest()
                for name in files
            }
            row["fingerprint"] = hashlib.sha256(
                json.dumps(row["assets_sha256"], sort_keys=True).encode()
            ).hexdigest()
            row["reference"] = _check_asset(_asset_path(directory, "reference.py"), task)
            row["mutants"] = []
            candidates = [(mutant.file, mutant.description) for mutant in task.mutants]
            if task.starter_file:
                candidates.append((task.starter_file, "provided buggy implementation"))
            for filename, description in candidates:
                result = _check_asset(_asset_path(directory, filename), task)
                hidden_failures = [
                    case["id"]
                    for case in result.get("cases", [])
                    if case["visibility"] == "hidden" and not case["passed"]
                ]
                result.update(
                    file=filename,
                    description=description,
                    hidden_failures=hidden_failures,
                    killed=bool(hidden_failures),
                )
                if filename == task.starter_file:
                    row["buggy"] = result
                else:
                    row["mutants"].append(result)
            checks = [row["reference"], *row["mutants"]]
            if "buggy" in row:
                checks.append(row["buggy"])
            row["errors"] = [
                check["error"]
                for check in checks
                if check["status"] in ("invalid_asset", "timeout")
            ]
            row["passed"] = (
                not row["errors"]
                and row["reference"]["status"] == "passed"
                and all(mutant["killed"] for mutant in row["mutants"])
                and ("buggy" not in row or row["buggy"]["killed"])
            )
        except (ValueError, OSError) as error:
            row["errors"].append(str(error))
    mutants = [mutant for row in results for mutant in row.get("mutants", [])]
    return {
        "schema_version": "1.0",
        "execution_mode": "trusted_authoring_check",
        "passed": all(row["passed"] for row in results),
        "runtime": {"python": platform.python_version(), "executable": sys.executable},
        "execution_limits": {
            "wall_clock_timeout": "2 seconds per asset",
            "memory_limit_enforced": False,
            "network_isolated": False,
            "filesystem_isolated": False,
        },
        "summary": {
            "tasks": len(results),
            "passed_tasks": sum(row["passed"] for row in results),
            "public_cases": sum(row.get("public_cases", 0) for row in results),
            "hidden_cases": sum(row.get("hidden_cases", 0) for row in results),
            "reference_passed": sum(
                row.get("reference", {}).get("status") == "passed" for row in results
            ),
            "mutants": len(mutants),
            "killed_mutants": sum(mutant["killed"] for mutant in mutants),
            "surviving_mutants": sum(not mutant["killed"] for mutant in mutants),
            "buggy_detected": sum(row.get("buggy", {}).get("killed", False) for row in results),
            "configuration_errors": sum(bool(row["errors"]) for row in results),
        },
        "tasks": results,
    }
