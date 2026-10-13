"""Render versionable generation instructions without changing training text."""

from pathlib import Path

from jinja2 import Environment, StrictUndefined

from dataforge.schemas import Seed


def render_prompt(template_path: str | Path, seed: Seed) -> str:
    environment = Environment(
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    template = environment.from_string(Path(template_path).read_text(encoding="utf-8"))
    return template.render(seed=seed.model_dump(mode="json"))
