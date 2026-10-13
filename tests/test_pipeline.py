import asyncio
import fcntl
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from dataforge.cli import app
from dataforge.config import ModelConfig, RunConfig
from dataforge.datasets.io import read_jsonl, write_jsonl
from dataforge.quality.report import dataset_report
from dataforge.runner import run_pipeline
from dataforge.schemas import GenerationResult


class StubProvider:
    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []
        self.closed = False

    async def generate(self, request):
        self.calls.append(request)
        prompt = request.messages[-1].content[-1].text
        result = self.results.get(
            prompt, GenerationResult(text=f"Answer {prompt}", status="completed")
        )
        if isinstance(result, Exception):
            raise result
        return result

    async def aclose(self):
        self.closed = True


def make_config(tmp_path: Path, seeds=None, **overrides):
    media = tmp_path / "media"
    media.mkdir(exist_ok=True)
    input_path = tmp_path / "seeds.jsonl"
    write_jsonl(input_path, seeds or [seed("one"), seed("two")])
    template = tmp_path / "generate.jinja2"
    template.write_text("Generate a careful answer for {{ seed.id }}.", encoding="utf-8")
    values = dict(
        name="test",
        input_path=input_path,
        media_root=media,
        prompt_template=template,
        output_dir=tmp_path / "runs",
        model=ModelConfig(provider="mock", model="mock"),
        validation_fraction=0.5,
    )
    values.update(overrides)
    return RunConfig(**values)


def seed(name, *, prompt=None, group_id=None):
    return {
        "id": name,
        "group_id": group_id,
        "prompt": [{"role": "user", "content": [{"type": "text", "text": prompt or name}]}],
    }


def rows(config, name):
    return list(read_jsonl(config.output_dir / config.name / name))


async def test_resume_replays_raw_without_requests_or_synthesis_in_training(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    provider = StubProvider()
    import dataforge.runner as runner

    real_writer = runner.write_jsonl

    def crash_before_artifacts(*args, **kwargs):
        raise OSError("disk interrupted after journal")

    monkeypatch.setattr(runner, "write_jsonl", crash_before_artifacts)
    with pytest.raises(OSError, match="disk interrupted"):
        await run_pipeline(config, provider=provider)
    assert len(provider.calls) == 2
    assert len(rows(config, "raw.jsonl")) == 2
    assert provider.closed
    monkeypatch.setattr(runner, "write_jsonl", real_writer)
    replay_provider = StubProvider()
    report = await run_pipeline(config, resume=True, provider=replay_provider)
    assert replay_provider.calls == []
    assert report["accepted"] == 2
    samples = rows(config, "clean.jsonl")
    assert [row["metadata"]["seed_id"] for row in samples] == ["one", "two"]
    assert "careful answer" not in json.dumps([row["prompt"] for row in samples])
    assert "careful answer" in provider.calls[0].instructions
    with pytest.raises(ValueError, match="already exists"):
        await run_pipeline(config, provider=StubProvider())


@pytest.mark.parametrize("changed", ["template", "input", "config", "image"])
async def test_resume_rejects_changed_fingerprint(tmp_path, changed):
    item = seed("image")
    item["prompt"][0]["content"].insert(0, {"type": "image", "uri": "picture.png"})
    config = make_config(
        tmp_path, [item], model=ModelConfig(provider="mock", model="mock", vision=True)
    )
    image = config.media_root / "picture.png"
    Image.new("RGB", (2, 2), "red").save(image)
    await run_pipeline(config, provider=StubProvider())
    if changed == "template":
        config.prompt_template.write_text("Different instructions", encoding="utf-8")
    elif changed == "input":
        with config.input_path.open("a") as stream:
            stream.write(json.dumps(seed("extra")) + "\n")
    elif changed == "config":
        config.model.max_output_tokens = 900
    else:
        Image.new("RGB", (2, 2), "blue").save(image)
    provider = StubProvider()
    with pytest.raises(ValueError, match="has changed"):
        await run_pipeline(config, resume=True, provider=provider)
    assert not provider.calls


async def test_reject_status_and_count_all_token_usage(tmp_path):
    config = make_config(tmp_path, [seed(name) for name in ("short", "refusal", "empty", "good")])
    provider = StubProvider(
        {
            "short": GenerationResult(
                text="Partial", status="incomplete", input_tokens=5, output_tokens=2
            ),
            "refusal": GenerationResult(
                text="Cannot", status="refused", input_tokens=5, output_tokens=2
            ),
            "empty": GenerationResult(
                text="  \n", status="completed", input_tokens=5, output_tokens=2
            ),
            "good": GenerationResult(
                text="Good", status="completed", input_tokens=5, output_tokens=2
            ),
        }
    )
    report = await run_pipeline(config, provider=provider)
    assert report["accepted"] == 1
    assert report["rejection_reasons"] == {"incomplete": 1, "refused": 1, "empty_completion": 1}
    assert report["token_usage"] == {"input": 20, "output": 8}
    assert len(rows(config, "samples.jsonl")) == 1
    assert len(rows(config, "raw.jsonl")) == 4


async def test_dedup_preserves_code_and_uses_image_content(tmp_path):
    code = "```python\n    x = '  spaces  '\n\n    return x\n```\n"
    first, second = seed("one", prompt="Write code"), seed("two", prompt="Write code")
    for item, path in ((first, "a.png"), (second, "b.png")):
        item["prompt"][0]["content"].insert(0, {"type": "image", "uri": path})
    config = make_config(
        tmp_path, [first, second], model=ModelConfig(provider="mock", model="mock", vision=True)
    )
    for name in ("a.png", "b.png"):
        Image.new("RGB", (2, 2), "red").save(config.media_root / name)
    report = await run_pipeline(
        config,
        provider=StubProvider(
            {
                "Write code": GenerationResult(text=code, status="completed"),
            }
        ),
    )
    assert report["accepted"] == 1
    assert report["rejection_reasons"] == {"duplicate": 1}
    clean = rows(config, "clean.jsonl")[0]
    assert clean["completion"][0]["content"][0]["text"] == code
    exports = rows(config, "exports/train.jsonl") + rows(config, "exports/validation.jsonl")
    image_path = config.output_dir / config.name / "exports" / exports[0]["images"][0]
    original_image = image_path.read_bytes()
    image_path.write_bytes(b"corrupted derived image")
    await run_pipeline(config, resume=True, provider=StubProvider())
    assert image_path.read_bytes() == original_image


async def test_group_split_and_stable_order_with_bounded_concurrency(tmp_path):
    config = make_config(
        tmp_path, [seed(str(i), group_id=f"group-{i // 3}") for i in range(18)], concurrency=3
    )

    class DelayedProvider(StubProvider):
        active = 0
        peak = 0

        async def generate(self, request):
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.003 * (3 - int(request.messages[-1].content[0].text) % 3))
            self.active -= 1
            return await super().generate(request)

    provider = DelayedProvider()
    await run_pipeline(config, provider=provider)
    assert provider.peak == 3
    clean = rows(config, "clean.jsonl")
    assert [row["metadata"]["seed_id"] for row in clean] == [str(i) for i in range(18)]
    train_groups = {row["group_id"] for row in rows(config, "exports/train.jsonl")}
    validation_groups = {row["group_id"] for row in rows(config, "exports/validation.jsonl")}
    assert train_groups and validation_groups
    assert not train_groups & validation_groups


async def test_failed_requests_are_persisted_then_only_failures_retry(tmp_path):
    config = make_config(tmp_path)
    report = await run_pipeline(
        config, provider=StubProvider({"two": RuntimeError("temporary failure")})
    )
    assert report["failed_requests"] == 1
    assert rows(config, "errors.jsonl")[0]["seed_id"] == "two"
    assert rows(config, "errors.jsonl")[0]["message"] == "temporary failure"
    provider = StubProvider()
    report = await run_pipeline(config, resume=True, provider=provider)
    assert [r.messages[-1].content[0].text for r in provider.calls] == ["two"]
    assert report["failed_requests"] == 0
    assert report["accepted"] == 2
    assert len(rows(config, "errors.jsonl")) == 1


@pytest.mark.parametrize(
    "problem", ["duplicate_id", "missing_variable", "vision_off", "invalid_image"]
)
async def test_preflight_before_network_or_output_mutation(tmp_path, problem):
    config = make_config(tmp_path)
    if problem == "duplicate_id":
        write_jsonl(config.input_path, [seed("same"), seed("same")])
    elif problem == "missing_variable":
        config.prompt_template.write_text("{{ seed.variables.missing }}", encoding="utf-8")
    else:
        item = seed("image")
        item["prompt"][0]["content"].insert(0, {"type": "image", "uri": "missing.png"})
        write_jsonl(config.input_path, [item])
        config.model.vision = problem == "invalid_image"
    provider = StubProvider()
    with pytest.raises(Exception):
        await run_pipeline(config, provider=provider)
    assert not provider.calls
    assert not config.output_dir.exists()


async def test_only_truncated_final_json_is_recovered(tmp_path):
    config = make_config(tmp_path)
    await run_pipeline(config, provider=StubProvider())
    raw = config.output_dir / config.name / "raw.jsonl"
    original = raw.read_bytes()
    with raw.open("ab") as stream:
        stream.write(b'{"task_key": "unfinished')
    with pytest.warns(UserWarning, match="Recovered interrupted"):
        await run_pipeline(config, resume=True, provider=StubProvider())
    assert raw.read_bytes() == original
    with raw.open("ab") as stream:
        stream.write(b'{"task_key": "bad"}\n')
    with pytest.raises(ValueError, match="task_key"):
        await run_pipeline(config, resume=True, provider=StubProvider())
    raw.write_bytes(original + b"broken\n")
    with pytest.raises(ValueError, match="malformed journal"):
        await run_pipeline(config, resume=True, provider=StubProvider())


async def test_concurrent_run_lock(tmp_path):
    config = make_config(tmp_path)
    config.output_dir.mkdir()
    with (config.output_dir / ".test.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already active"):
            await run_pipeline(config, provider=StubProvider())


def test_cli_partial_failure_is_nonzero_and_validates_union(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    monkeypatch.setattr(
        "dataforge.runner.create_provider",
        lambda *args: StubProvider(
            {
                "two": RuntimeError("failure"),
            }
        ),
    )
    result = CliRunner().invoke(app, ["run", str(config_path)])
    assert result.exit_code == 1
    assert '"failed_requests": 1' in result.stdout
    sample_path = config.output_dir / config.name / "clean.jsonl"
    assert CliRunner().invoke(app, ["validate", str(sample_path)]).exit_code == 0
    assert CliRunner().invoke(app, ["analyze", str(sample_path)]).exit_code == 0
    preference = rows(config, "clean.jsonl")[0]
    preference["task_type"] = "preference"
    preference["chosen"] = preference.pop("completion")
    preference["rejected"] = preference["chosen"]
    write_jsonl(tmp_path / "preference.jsonl", [preference])
    assert CliRunner().invoke(app, ["validate", str(tmp_path / "preference.jsonl")]).exit_code == 0
    assert dataset_report([preference])["completion_characters"]["count"] == 0
