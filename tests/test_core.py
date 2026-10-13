import base64
import hashlib
import json

import pytest
import yaml
from jinja2 import UndefinedError
from PIL import Image
from pydantic import ValidationError

from dataforge.config import ModelConfig, QualityConfig, load_config
from dataforge.datasets.io import append_jsonl, read_jsonl, write_jsonl
from dataforge.datasets.media import image_data, image_sha256, resolve_image
from dataforge.prompting import render_prompt
from dataforge.schemas import Message, Seed, SFTSample, sample_adapter


def message(role="user", text="question"):
    return {"role": role, "content": [{"type": "text", "text": text}]}


def test_schema_preserves_training_text_and_rejects_unknown_fields():
    original = "  ```python\n    print('你好')\n```\n"
    seed = Seed(id="seed", prompt=[message(text=original)])
    assert seed.prompt[0].content[0].text == original
    with pytest.raises(ValidationError, match="Extra inputs"):
        Seed(id="seed", prompt=[message()], unexpected=True)


@pytest.mark.parametrize(
    "prompt",
    [[], [message("assistant")], [message(), message("system"), message()]],
)
def test_seed_requires_valid_prompt(prompt):
    with pytest.raises(ValidationError):
        Seed(id="seed", prompt=prompt)


@pytest.mark.parametrize("role", ["system", "assistant"])
def test_images_only_in_user_messages(role):
    with pytest.raises(ValidationError, match="images are supported only"):
        Message(role=role, content=[{"type": "image", "uri": "image.png"}])


def test_sample_types_and_assistant_completion():
    common = {"id": "sample", "prompt": [message()]}
    answer = [message("assistant", "answer")]
    for fields, expected in [
        ({"task_type": "sft", "completion": answer}, "sft"),
        ({"task_type": "preference", "chosen": answer, "rejected": answer}, "preference"),
        ({"task_type": "rl_task", "reference": {"answer": 2}, "verifier": "exact"}, "rl_task"),
    ]:
        sample = sample_adapter.validate_python(common | fields)
        assert sample.task_type == expected
    with pytest.raises(ValidationError, match="only assistant"):
        SFTSample(**common, completion=[message()])


def test_config_paths_are_relative_to_config(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "task.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "test-run",
                "input_path": "../seeds.jsonl",
                "media_root": "../assets",
                "prompt_template": "../prompts/generate.jinja2",
                "output_dir": "../runs",
                "model": {"provider": "mock", "model": "test"},
            }
        )
    )
    monkeypatch.chdir(tmp_path.parent)
    config = load_config(config_path)
    assert config.input_path == tmp_path / "seeds.jsonl"
    assert config.media_root == tmp_path / "assets"
    assert config.prompt_template == tmp_path / "prompts/generate.jinja2"
    assert config.output_dir == tmp_path / "runs"
    for name in ("../escape", ".", "a/b"):
        raw = yaml.safe_load(config_path.read_text())
        raw["name"] = name
        config_path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValidationError):
            load_config(config_path)


def test_config_rejects_secrets_and_impossible_quality_bounds():
    with pytest.raises(ValidationError, match="Extra inputs"):
        ModelConfig(provider="openai", model="test", api_key="must-not-be-in-yaml")
    with pytest.raises(ValidationError, match="omit this field"):
        ModelConfig(provider="anthropic", model="test", temperature=0.2)
    with pytest.raises(ValidationError, match="min_chars"):
        QualityConfig(min_chars=10, max_chars=1)


def test_prompt_missing_variables_fail_and_text_is_preserved(tmp_path):
    template = tmp_path / "prompt.jinja2"
    template.write_text("{{ seed.variables.topic }}\n")
    seed = Seed(id="seed", prompt=[message()])
    with pytest.raises(UndefinedError):
        render_prompt(template, seed)
    seed.variables["topic"] = "  test  "
    assert render_prompt(template, seed) == "  test  \n"


def test_jsonl_round_trip_and_helpful_line_errors(tmp_path):
    path = tmp_path / "data.jsonl"
    write_jsonl(path, [{"text": "你好\n世界"}])
    append_jsonl(path, {"text": "second"})
    assert list(read_jsonl(path)) == [{"text": "你好\n世界"}, {"text": "second"}]
    assert path.read_bytes().endswith(b"\n")
    with path.open("a") as stream:
        stream.write("broken\n")
    with pytest.raises(ValueError, match=r"data.jsonl:3: invalid JSON"):
        list(read_jsonl(path))


@pytest.mark.parametrize("content", ["\n", "[]\n", '"text"\n'])
def test_jsonl_requires_object_rows(tmp_path, content):
    path = tmp_path / "bad.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError, match="bad.jsonl:1:"):
        list(read_jsonl(path))


def test_failed_atomic_write_preserves_previous_dataset(tmp_path):
    path = tmp_path / "data.jsonl"
    write_jsonl(path, [{"original": True}])

    def broken_rows():
        yield {"new": True}
        raise RuntimeError("input stopped")

    with pytest.raises(RuntimeError, match="input stopped"):
        write_jsonl(path, broken_rows())
    assert list(read_jsonl(path)) == [{"original": True}]
    assert list(tmp_path.iterdir()) == [path]


def test_local_image_payload_and_content_hash(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), "red").save(image_path)
    mime_type, encoded = image_data("image.png", tmp_path)
    assert mime_type == "image/png"
    assert base64.b64decode(encoded) == image_path.read_bytes()
    expected_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert image_sha256("image.png", tmp_path) == expected_hash


def test_image_paths_cannot_escape_media_root(tmp_path):
    media_root = tmp_path / "assets"
    media_root.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2), "red").save(outside)
    (media_root / "linked.png").symlink_to(outside)
    for uri in ("../outside.png", "linked.png", str(outside), "https://example.com/img.png"):
        with pytest.raises(ValueError):
            resolve_image(uri, media_root)


def test_invalid_and_unsupported_images_are_rejected(tmp_path):
    (tmp_path / "fake.png").write_text(json.dumps({"not": "an image"}))
    with pytest.raises(ValueError, match="cannot read image"):
        resolve_image("fake.png", tmp_path)
    Image.new("RGB", (2, 2)).save(tmp_path / "image.bmp")
    with pytest.raises(ValueError, match="unsupported image format"):
        resolve_image("image.bmp", tmp_path)
