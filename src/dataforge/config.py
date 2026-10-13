"""Validated YAML configuration; paths are relative to the configuration file."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from dataforge.schemas import StrictModel


class ModelConfig(StrictModel):
    provider: Literal["openai", "anthropic", "mock"]
    model: str = Field(min_length=1)
    vision: bool = False
    max_output_tokens: int = Field(default=1024, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    timeout_seconds: float = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_provider_parameters(self):
        if self.provider == "anthropic" and self.temperature is not None:
            raise ValueError("the Anthropic adapter does not support temperature; omit this field")
        return self


class QualityConfig(StrictModel):
    min_chars: int = Field(default=1, ge=0)
    max_chars: int = Field(default=20000, gt=0)
    deduplicate: bool = True

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.min_chars > self.max_chars:
            raise ValueError("min_chars must be less than or equal to max_chars")
        return self


class RunConfig(StrictModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    task_type: Literal["sft"] = "sft"
    input_path: Path
    media_root: Path
    prompt_template: Path
    output_dir: Path
    model: ModelConfig
    concurrency: int = Field(default=4, gt=0)
    samples_per_seed: int = Field(default=1, gt=0)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    validation_fraction: float = Field(default=0.1, ge=0, lt=1)
    split_seed: int = 42


def load_config(path: str | Path) -> RunConfig:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        config = RunConfig.model_validate(yaml.safe_load(stream))
    for field in ("input_path", "media_root", "prompt_template", "output_dir"):
        value = getattr(config, field).expanduser()
        setattr(config, field, (path.parent / value).resolve())
    return config
