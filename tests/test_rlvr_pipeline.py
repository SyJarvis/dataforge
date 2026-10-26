import json

import pytest
from jinja2 import UndefinedError
from pydantic import ValidationError
from typer.testing import CliRunner

from dataforge.cli import app
from dataforge.config import ModelConfig, QualityConfig, RunConfig
from dataforge.datasets.io import read_jsonl, write_jsonl
from dataforge.recipes.rlvr import ANSWER_FORMAT, prepare_seed
from dataforge.runner import run_pipeline
from dataforge.schemas import GenerationResult, RLVRSeed


class StubProvider:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []
        self.closed = False

    async def generate(self, request):
        self.calls.append(request)
        answer = next(self.answers)
        if isinstance(answer, GenerationResult):
            return answer
        return GenerationResult(text=answer, status="completed", input_tokens=2, output_tokens=3)

    async def aclose(self):
        self.closed = True


def seed(name, *, answer="42", verifier="exact_match", prompt=None, group_id=None):
    return {
        "id": name,
        "group_id": group_id,
        "prompt": [{"role": "user", "content": [{"type": "text", "text": prompt or name}]}],
        "reference": {"answer": answer},
        "verifier": verifier,
    }


def make_config(tmp_path, seeds, **overrides):
    input_path = tmp_path / "seeds.jsonl"
    write_jsonl(input_path, seeds)
    template = tmp_path / "generate.jinja2"
    template.write_text("Solve the task carefully.", encoding="utf-8")
    values = {
        "name": "rlvr-test",
        "task_type": "rlvr",
        "input_path": input_path,
        "media_root": tmp_path,
        "prompt_template": template,
        "output_dir": tmp_path / "runs",
        "model": ModelConfig(provider="mock", model="mock"),
        "concurrency": 1,
        "validation_fraction": 0.5,
    }
    values.update(overrides)
    return RunConfig(**values)


def rows(config, name):
    return list(read_jsonl(config.output_dir / config.name / name))


async def test_verification_filters_trial_answers_and_exports_only_tasks(tmp_path):
    config = make_config(
        tmp_path,
        [
            seed("exact"),
            seed("math", answer="0.5", verifier="math_answer"),
            seed("mismatch"),
            seed("malformed"),
        ],
    )
    provider = StubProvider(["Teacher reasoning.\n#### 42", "#### 1/2", "#### 41", "42"])
    report = await run_pipeline(config, provider=provider)
    assert provider.closed
    assert report["accepted"] == 2
    assert report["rejected"] == 2
    assert report["rejection_reasons"] == {"answer_mismatch": 1, "invalid_answer_format": 1}
    assert report["verification"] == {
        "evaluated": 4,
        "passed": 2,
        "failed": 2,
        "skipped": 0,
        "pass_rate": 0.5,
        "by_verifier": {
            "exact_match": {"evaluated": 3, "passed": 1, "failed": 2, "skipped": 0},
            "math_answer": {"evaluated": 1, "passed": 1, "failed": 0, "skipped": 0},
        },
    }
    assert report["token_usage"] == {"input": 8, "output": 12}
    assert report["accepted_dataset"]["verifiers"] == {"exact_match": 1, "math_answer": 1}
    quality = rows(config, "quality.jsonl")
    assert [item["verification"]["passed"] for item in quality] == [True, True, False, False]
    for rejected in rows(config, "rejected.jsonl"):
        assert rejected["verification"]["reason"] == rejected["reasons"][0]
    assert "Teacher reasoning." in rows(config, "raw.jsonl")[0]["result"]["text"]
    clean = rows(config, "clean.jsonl")
    exports = rows(config, "exports/train.jsonl") + rows(config, "exports/validation.jsonl")
    for item in clean:
        assert "completion" not in item
        assert "Teacher reasoning." not in json.dumps(item)
        assert item["prompt"][-1]["role"] == "user"
        assert item["prompt"][0]["content"][0]["text"] == ANSWER_FORMAT
    for item in exports:
        assert set(item) == {"id", "task_type", "group_id", "messages", "reference", "verifier"}
        assert item["task_type"] == "rlvr"
        assert [message["role"] for message in item["messages"]] == ["system", "user"]
        assert all(isinstance(message["content"], str) for message in item["messages"])
        internal = next(sample for sample in clean if sample["id"] == item["id"])
        assert item["reference"] == internal["reference"]
    clean_path = config.output_dir / config.name / "clean.jsonl"
    assert CliRunner().invoke(app, ["validate", str(clean_path)]).exit_code == 0
    assert CliRunner().invoke(app, ["analyze", str(clean_path)]).exit_code == 0


async def test_reference_is_absent_from_request_and_template_context(tmp_path):
    config = make_config(tmp_path, [seed("safe", answer="private-ground-truth")])
    config.prompt_template.write_text("Context: {{ seed | tojson }}", encoding="utf-8")
    original = config.input_path.read_bytes()
    provider = StubProvider(["#### private-ground-truth"])
    await run_pipeline(config, provider=provider)
    request = provider.calls[0]
    payload = request.model_dump_json()
    assert "private-ground-truth" not in payload
    assert "reference" not in payload
    assert "verifier" not in payload
    assert config.input_path.read_bytes() == original
    assert [message.model_dump(mode="json") for message in request.messages] == (
        rows(config, "clean.jsonl")[0]["prompt"]
    )


@pytest.mark.parametrize("field", ["reference", "verifier"])
async def test_reward_fields_cannot_be_read_by_generation_template(tmp_path, field):
    config = make_config(tmp_path, [seed("safe")])
    config.prompt_template.write_text("{{ seed." + field + " }}", encoding="utf-8")
    provider = StubProvider([])
    with pytest.raises(UndefinedError):
        await run_pipeline(config, provider=provider)
    assert provider.calls == []
    assert not config.output_dir.exists()


def test_answer_convention_preserves_original_seed_and_existing_system_prompt():
    item = seed("one")
    item["prompt"].insert(0, {"role": "system", "content": [{"type": "text", "text": "Be brief."}]})
    original = RLVRSeed.model_validate(item)
    before = original.model_dump()
    prepared = prepare_seed(original)
    assert original.model_dump() == before
    assert len(prepared.prompt) == 2
    system_text = "".join(block.text for block in prepared.prompt[0].content)
    assert system_text == "Be brief.\n" + ANSWER_FORMAT


async def test_multiple_candidates_record_verification_before_task_deduplication(tmp_path):
    config = make_config(tmp_path, [seed("one")], samples_per_seed=3)
    provider = StubProvider(["First proof.\n#### 42", "Another proof.\n#### 42", "#### 41"])
    report = await run_pipeline(config, provider=provider)
    assert report["accepted"] == 1
    assert report["pass_rate"] == 1 / 3
    assert report["verification"]["evaluated"] == 3
    assert report["verification"]["passed"] == 2
    assert report["verification"]["failed"] == 1
    assert report["verification"]["pass_rate"] == 2 / 3
    assert report["rejection_reasons"] == {"duplicate": 1, "answer_mismatch": 1}
    quality = rows(config, "quality.jsonl")
    assert quality[1]["verification"]["passed"] is True
    assert quality[1]["duplicate_of"] == quality[0]["sample_id"]


async def test_task_deduplication_includes_reference_and_verifier(tmp_path):
    config = make_config(
        tmp_path,
        [
            seed("one", prompt="same"),
            seed("two", prompt="same"),
            seed("other-answer", prompt="same", answer="43"),
            seed("other-verifier", prompt="same", verifier="math_answer"),
        ],
    )
    report = await run_pipeline(
        config, provider=StubProvider(["#### 42", "#### 42", "#### 43", "#### 42"])
    )
    assert report["accepted"] == 3
    assert report["rejection_reasons"] == {"duplicate": 1}


async def test_response_and_length_failures_skip_verification(tmp_path, monkeypatch):
    config = make_config(
        tmp_path,
        [seed(name) for name in ["incomplete", "refused", "empty", "short", "long"]],
        quality=QualityConfig(min_chars=8, max_chars=10),
    )

    def unexpected_verification(*args):
        pytest.fail("Invalid responses must not enter answer verification")

    monkeypatch.setattr("dataforge.runner.verify_answer", unexpected_verification)
    provider = StubProvider(
        [
            GenerationResult(text="#### 42", status="incomplete"),
            GenerationResult(text="#### 42", status="refused"),
            "  ",
            "#### 42",
            "Long response.\n#### 42",
        ]
    )
    report = await run_pipeline(config, provider=provider)
    assert report["accepted"] == 0
    assert report["verification"]["evaluated"] == 0
    assert report["verification"]["skipped"] == 5
    assert report["rejection_reasons"] == {
        "incomplete": 1,
        "refused": 1,
        "empty_completion": 1,
        "completion_too_short": 1,
        "completion_too_long": 1,
    }
    for item in rows(config, "quality.jsonl"):
        assert item["verification"]["evaluated"] is False
        assert item["verification"]["passed"] is None
        assert item["verification"]["reason"] == item["reasons"][0]


async def test_resume_preserves_group_splits_and_rejects_reference_changes(tmp_path):
    config = make_config(tmp_path, [seed(str(i), group_id=f"group-{i // 3}") for i in range(18)])
    provider = StubProvider(["#### 42"] * 18)
    original_report = await run_pipeline(config, provider=provider)
    train_groups = {item["group_id"] for item in rows(config, "exports/train.jsonl")}
    validation_groups = {item["group_id"] for item in rows(config, "exports/validation.jsonl")}
    assert train_groups and validation_groups
    assert not train_groups & validation_groups
    run_dir = config.output_dir / config.name
    artifacts = [
        "raw.jsonl",
        "clean.jsonl",
        "quality.jsonl",
        "exports/train.jsonl",
        "exports/validation.jsonl",
    ]
    before = {name: (run_dir / name).read_bytes() for name in artifacts}
    replay = StubProvider([])
    assert await run_pipeline(config, resume=True, provider=replay) == original_report
    assert replay.calls == []
    assert before == {name: (run_dir / name).read_bytes() for name in artifacts}
    seeds = list(read_jsonl(config.input_path))
    seeds[0]["reference"]["answer"] = "changed"
    write_jsonl(config.input_path, seeds)
    with pytest.raises(ValueError, match="has changed"):
        await run_pipeline(config, resume=True, provider=replay)
    assert replay.calls == []


@pytest.mark.parametrize("problem", ["missing_answer", "bad_number", "unknown_verifier", "image"])
async def test_invalid_rlvr_seeds_fail_before_api_or_output(tmp_path, problem):
    item = seed("invalid")
    if problem == "missing_answer":
        item["reference"] = {}
    elif problem == "bad_number":
        item["reference"] = {"answer": "not a number"}
        item["verifier"] = "math_answer"
    elif problem == "unknown_verifier":
        item["verifier"] = "unknown"
    else:
        item["prompt"][0]["content"].insert(0, {"type": "image", "uri": "missing.png"})
    config = make_config(tmp_path, [item])
    provider = StubProvider([])
    with pytest.raises(ValidationError):
        await run_pipeline(config, provider=provider)
    assert provider.calls == []
    assert not config.output_dir.exists()
