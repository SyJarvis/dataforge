"""Command-line entry points."""

import asyncio
import json
from pathlib import Path

import typer

from dataforge.coding import check_coding_pack
from dataforge.config import load_config
from dataforge.datasets.io import read_jsonl, write_jsonl
from dataforge.quality.report import dataset_report
from dataforge.runner import run_pipeline
from dataforge.schemas import sample_adapter

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(config: Path, resume: bool = typer.Option(False, "--resume")) -> None:
    """Generate, clean, analyze, and export a configured SFT or RLVR run."""
    try:
        report = asyncio.run(run_pipeline(load_config(config), resume=resume))
    except Exception as error:
        typer.echo(f"Run failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if report["failed_requests"]:
        raise typer.Exit(1)


@app.command()
def validate(file: Path) -> None:
    """Validate an internal SFT, preference, RLVR, or legacy RL-task JSONL file."""
    try:
        count = 0
        for count, row in enumerate(read_jsonl(file), start=1):
            sample_adapter.validate_python(row)
    except Exception as error:
        typer.echo(f"Validation failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Valid: {count} samples")


@app.command()
def analyze(file: Path) -> None:
    """Print basic statistics for an internal dataset."""
    try:
        rows = [
            sample_adapter.validate_python(row).model_dump(mode="json") for row in read_jsonl(file)
        ]
        typer.echo(json.dumps(dataset_report(rows), ensure_ascii=False, indent=2))
    except Exception as error:
        typer.echo(f"Analysis failed: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("check-coding-pack")
def check_coding_pack_command(
    directory: Path,
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Check trusted author assets with timeouts; NOT a sandbox for model-generated code."""
    try:
        report = check_coding_pack(directory)
        if output is not None:
            write_jsonl(output, [report])
    except Exception as error:
        typer.echo(f"Coding pack check failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise typer.Exit(1)
