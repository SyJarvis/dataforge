import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from dataforge.cli import app
from dataforge.coding import _check_asset, check_coding_pack, load_task, public_task

PACK = Path(__file__).resolve().parents[1] / "examples" / "coding"
TASK_IDS = {
    "stable_unique",
    "chunk_list",
    "merge_intervals",
    "deep_merge",
    "parse_config",
    "max_false_run",
    "binary_search_first",
    "dedup_records",
}
requires_python313 = pytest.mark.skipif(
    sys.version_info[:2] != (3, 13),
    reason="Coding pack acceptance executes trusted assets requiring Python 3.13",
)


@pytest.fixture
def tiny_pack(tmp_path):
    directory = tmp_path / "pack" / "sample"
    (directory / "mutants").mkdir(parents=True)
    task = {
        "schema_version": "1.0",
        "id": "sample",
        "title": "Return the value",
        "task_kind": "function_implementation",
        "prompt": "返回传入的值，保持输入不变。",
        "entry_point": "sample",
        "signature": "sample(value: int) -> int",
        "runtime": {
            "language": "python",
            "version": "3.13",
            "timeout_seconds": 2,
            "memory_mb": 256,
        },
        "preserve_inputs": True,
        "cases": [
            {
                "id": f"case-{i}",
                "visibility": "public" if i < 2 else "hidden",
                "args": [i],
                "expected": i,
            }
            for i in range(8)
        ],
        "mutants": [
            {"file": f"mutants/wrong_{i}.py", "description": f"Incorrect constant {-i}"}
            for i in range(1, 4)
        ],
    }
    (directory / "task.json").write_text(json.dumps(task), encoding="utf-8")
    (directory / "reference.py").write_text("def sample(value):\n    return value\n")
    for index, mutant in enumerate(task["mutants"], 1):
        (directory / mutant["file"]).write_text(f"def sample(value):\n    return {-index}\n")
    return directory


def change_task(directory, edit):
    path = directory / "task.json"
    raw = json.loads(path.read_text())
    edit(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")


@requires_python313
def test_all_eight_real_tasks_have_valid_references_and_hidden_mutation_detection():
    report = check_coding_pack(PACK)
    assert report["passed"], json.dumps(report, ensure_ascii=False, indent=2)
    assert {task["id"] for task in report["tasks"]} == TASK_IDS
    assert report["summary"]["tasks"] == 8
    assert report["summary"]["reference_passed"] == 8
    assert report["summary"]["public_cases"] >= 16
    assert report["summary"]["hidden_cases"] >= 48
    assert report["summary"]["mutants"] == report["summary"]["killed_mutants"] == 24
    assert report["summary"]["surviving_mutants"] == 0
    assert report["summary"]["buggy_detected"] == 2
    assert report["runtime"]["python"].startswith("3.13.")
    assert report["execution_mode"] == "trusted_authoring_check"
    assert report["execution_limits"]["memory_limit_enforced"] is False
    for task in report["tasks"]:
        assert all(case["passed"] for case in task["reference"]["cases"])
        assert all(mutant["hidden_failures"] for mutant in task["mutants"])
        assert len(task["fingerprint"]) == 64
        assert "task.json" in task["assets_sha256"]


@pytest.mark.parametrize(
    "edit",
    [
        lambda task: task.pop("id"),
        lambda task: task["cases"][1].update(id="case-0"),
        lambda task: task["cases"][0].update(raises="ValueError"),
        lambda task: task["cases"][0].pop("expected"),
        lambda task: task["cases"][0].update(expected=float("nan")),
        lambda task: task["mutants"][0].update(file="../reference.py"),
        lambda task: task["mutants"][1].update(file="mutants/wrong_1.py"),
        lambda task: task.update(preserve_inputs=False),
        lambda task: task.update(preserve_inputs=1),
        lambda task: task.update(extra=True),
        lambda task: task.update(starter_file="buggy.py"),
        lambda task: task["cases"].pop(),
    ],
)
def test_strict_package_schema_rejects_invalid_contracts(tiny_pack, edit):
    change_task(tiny_pack, edit)
    with pytest.raises(ValidationError):
        load_task(tiny_pack)


@requires_python313
def test_null_expected_is_a_valid_json_outcome(tiny_pack):
    change_task(tiny_pack, lambda task: [case.update(expected=None) for case in task["cases"]])
    (tiny_pack / "reference.py").write_text("def sample(value):\n    return None\n")
    report = check_coding_pack(tiny_pack.parent)
    assert report["passed"]


def test_duplicate_task_ids_fail(tiny_pack):
    import shutil

    shutil.copytree(tiny_pack, tiny_pack.parent / "copy")
    report = check_coding_pack(tiny_pack.parent)
    assert not report["passed"]
    assert any("duplicate task ID" in error for task in report["tasks"] for error in task["errors"])


def test_asset_symlinks_cannot_escape_task_directory(tiny_pack, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("def sample(value):\n    return value\n")
    reference = tiny_pack / "reference.py"
    reference.unlink()
    reference.symlink_to(outside)
    with pytest.raises(ValueError, match="inside its task directory"):
        load_task(tiny_pack)


def test_missing_asset_and_empty_pack_fail_clearly(tiny_pack, tmp_path):
    (tiny_pack / "reference.py").unlink()
    with pytest.raises(ValueError, match="existing file"):
        load_task(tiny_pack)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no task directories"):
        check_coding_pack(empty)


@requires_python313
def test_public_view_only_exposes_public_examples_and_buggy_source(tiny_pack):
    view = public_task(tiny_pack)
    assert set(view) == {
        "id",
        "title",
        "task_kind",
        "prompt",
        "entry_point",
        "signature",
        "runtime",
        "preserve_inputs",
        "examples",
    }
    assert len(view["examples"]) == 2
    assert "case-2" not in json.dumps(view)
    assert "reference.py" not in json.dumps(view)
    assert "mutants" not in json.dumps(view)
    change_task(tiny_pack, lambda task: task.update(task_kind="bug_fix", starter_file="buggy.py"))
    buggy_source = "def sample(value):\n    return -99\n"
    (tiny_pack / "buggy.py").write_text(buggy_source)
    assert public_task(tiny_pack)["buggy_source"] == buggy_source
    report = check_coding_pack(tiny_pack.parent)
    assert report["passed"]
    assert report["tasks"][0]["buggy"]["killed"]
    (tiny_pack / "buggy.py").write_text("def sample(value):\n    return value\n")
    report = check_coding_pack(tiny_pack.parent)
    assert not report["passed"]
    assert not report["tasks"][0]["buggy"]["killed"]


@requires_python313
def test_surviving_mutant_and_public_only_failures_fail_acceptance(tiny_pack):
    (tiny_pack / "mutants" / "wrong_1.py").write_text(
        "def sample(value):\n    return -1 if value < 2 else value\n"
    )
    report = check_coding_pack(tiny_pack.parent)
    assert not report["passed"]
    result = report["tasks"][0]["mutants"][0]
    assert result["status"] == "failed"
    assert not result["killed"]
    assert result["hidden_failures"] == []
    assert report["summary"]["surviving_mutants"] == 1
    assert report["summary"]["configuration_errors"] == 0


@pytest.mark.parametrize(
    "source",
    [
        "def broken syntax",
        "other = 1",
        "raise RuntimeError('cannot load')",
        "def sample():\n    return 1\n",
    ],
)
@requires_python313
def test_unloadable_assets_are_configuration_errors_not_killed_mutants(tiny_pack, source):
    (tiny_pack / "mutants" / "wrong_1.py").write_text(source)
    report = check_coding_pack(tiny_pack.parent)
    assert not report["passed"]
    result = report["tasks"][0]["mutants"][0]
    assert result["status"] == "invalid_asset"
    assert not result["killed"]
    assert report["summary"]["configuration_errors"] == 1


@pytest.mark.parametrize(("actual", "expected"), [("True", 1), ("1.0", 1), ("(1,)", [1])])
def test_return_values_use_strict_json_types(tiny_pack, actual, expected):
    change_task(tiny_pack, lambda task: [case.update(expected=expected) for case in task["cases"]])
    source = tiny_pack / "reference.py"
    source.write_text(f"def sample(value):\n    return {actual}\n")
    result = _check_asset(source, load_task(tiny_pack))
    assert result["status"] == "failed"
    assert all(case["reason"] == "output_mismatch" for case in result["cases"])


def test_input_mutation_fails_even_when_return_value_matches(tiny_pack):
    change_task(
        tiny_pack,
        lambda task: [case.update(args=[[i]], expected=i) for i, case in enumerate(task["cases"])],
    )
    source = tiny_pack / "reference.py"
    source.write_text("def sample(value):\n    return value.pop()\n")
    result = _check_asset(source, load_task(tiny_pack))
    assert result["status"] == "failed"
    assert all(case["reason"] == "input_modified" for case in result["cases"])


def test_cases_reload_namespace_and_prints_do_not_corrupt_results(tiny_pack):
    change_task(tiny_pack, lambda task: [case.update(expected=1) for case in task["cases"]])
    source = tiny_pack / "reference.py"
    source.write_text(
        "import sys\nprint('not JSON')\nprint('diagnostic', file=sys.stderr)\nseen = []\n"
        "def sample(value):\n    seen.append(value)\n    print('running')\n    return len(seen)\n"
    )
    result = _check_asset(source, load_task(tiny_pack))
    assert result["status"] == "passed"


def test_exact_exception_type_and_mutation_on_error(tiny_pack):
    def raises_cases(task):
        for case in task["cases"]:
            case.pop("expected")
            case["raises"] = "ValueError"

    change_task(tiny_pack, raises_cases)
    source = tiny_pack / "reference.py"
    source.write_text("def sample(value):\n    raise ValueError('bad value')\n")
    assert _check_asset(source, load_task(tiny_pack))["status"] == "passed"
    source.write_text(
        "class CustomError(ValueError):\n    pass\n"
        "def sample(value):\n    raise CustomError('wrong type')\n"
    )
    assert _check_asset(source, load_task(tiny_pack))["status"] == "failed"


def test_asset_timeout_is_configuration_failure_and_worker_is_reaped(tiny_pack, monkeypatch):
    spawned, real_popen = [], subprocess.Popen

    def record_process(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr("dataforge.coding.subprocess.Popen", record_process)
    source = tiny_pack / "reference.py"
    source.write_text("import time\ndef sample(value):\n    time.sleep(30)\n")
    start = time.monotonic()
    result = _check_asset(source, load_task(tiny_pack))
    assert result["status"] == "timeout"
    assert spawned[0].returncode == -signal.SIGKILL
    assert time.monotonic() - start < 5


def test_runtime_mismatch_is_clear_and_does_not_execute(tiny_pack, monkeypatch):
    monkeypatch.setattr("dataforge.coding.sys.version_info", SimpleNamespace(major=3, minor=12))
    report = check_coding_pack(tiny_pack.parent)
    assert not report["passed"]
    assert "requires Python 3.13; actual Python 3.12" in report["tasks"][0]["errors"][0]
    assert "reference" not in report["tasks"][0]


@requires_python313
def test_cli_report_and_asset_fingerprint_change(tiny_pack, tmp_path):
    output = tmp_path / "results" / "report.json"
    result = CliRunner().invoke(
        app, ["check-coding-pack", str(tiny_pack.parent), "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    assert report == json.loads(result.output)
    assert not list(output.parent.glob(".report.json.*"))
    before = report["tasks"][0]["fingerprint"]
    (tiny_pack / "mutants" / "wrong_1.py").write_text("def sample(value):\n    return value\n")
    result = CliRunner().invoke(app, ["check-coding-pack", str(tiny_pack.parent)])
    assert result.exit_code == 1
    assert json.loads(result.output)["tasks"][0]["fingerprint"] != before
