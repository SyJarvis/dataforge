"""Bounded synthesis with a durable response journal and deterministic replay."""

import asyncio
import fcntl
import hashlib
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import yaml

from dataforge.cleaning import sample_digest
from dataforge.config import RunConfig
from dataforge.datasets.export import export_splits
from dataforge.datasets.io import append_jsonl, read_jsonl, write_jsonl
from dataforge.datasets.media import image_sha256
from dataforge.prompting import render_prompt
from dataforge.providers import create_provider
from dataforge.quality.report import run_report
from dataforge.quality.rules import response_issues, sample_issues
from dataforge.recipes.rlvr import make_sample as make_rlvr_sample
from dataforge.recipes.rlvr import prepare_seed
from dataforge.recipes.sft import make_request, make_sample
from dataforge.schemas import GenerationResult, RLVRSeed, Seed
from dataforge.verifiers import verify_answer


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: dict) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _read_journal(path: Path) -> list[dict]:
    """Recover only an invalid final unterminated line caused by an interrupted append."""
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows = []
    consumed = 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as stream:
                    stream.truncate(consumed)
                    stream.flush()
                    os.fsync(stream.fileno())
                warnings.warn(
                    f"Recovered interrupted final append in {path}; discarded {len(line)} bytes.",
                    stacklevel=2,
                )
                break
            raise ValueError(f"{path}:{index + 1}: malformed journal JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index + 1}: expected a journal object")
        rows.append(row)
        consumed += len(line)
    else:
        if data and not data.endswith(b"\n"):
            with path.open("ab") as stream:
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
    return rows


def _preflight(config: RunConfig) -> tuple[list[Seed], list[str], dict[str, str], dict]:
    seed_model = RLVRSeed if config.task_type == "rlvr" else Seed
    seeds = [seed_model.model_validate(row) for row in read_jsonl(config.input_path)]
    if not seeds:
        raise ValueError("Input contains no seeds")
    ids = [seed.id for seed in seeds]
    if len(set(ids)) != len(ids):
        raise ValueError("Seed IDs must be unique")
    if config.task_type == "rlvr":
        seeds = [prepare_seed(seed) for seed in seeds]
    # Keep reward-only labels out of the generation template's context.
    instructions = [
        render_prompt(
            config.prompt_template,
            Seed.model_validate(seed.model_dump(include=set(Seed.model_fields))),
        )
        for seed in seeds
    ]
    image_hashes: dict[str, str] = {}
    for seed in seeds:
        for message in seed.prompt:
            for block in message.content:
                if block.type == "image":
                    if not config.model.vision:
                        raise ValueError("Image prompts require model.vision: true")
                    if block.uri not in image_hashes:
                        image_hashes[block.uri] = image_sha256(block.uri, config.media_root)
    ingredients = {
        "schema_version": "1.0",
        "config": config.model_dump(mode="json"),
        "input_sha256": _digest(config.input_path.read_bytes()),
        "prompt_sha256": _digest(config.prompt_template.read_bytes()),
        "images": image_hashes,
    }
    fingerprint = _digest(json.dumps(ingredients, sort_keys=True).encode())
    manifest = {
        "schema_version": "1.0",
        "dataforge_version": "0.1.0",
        "run_id": config.name,
        "fingerprint": fingerprint,
        "input_sha256": ingredients["input_sha256"],
        "prompt_sha256": ingredients["prompt_sha256"],
        "images": image_hashes,
    }
    return seeds, instructions, image_hashes, manifest


async def run_pipeline(config: RunConfig, *, resume: bool = False, provider=None) -> dict:
    """Return a report; failed requests are retained and retried only with explicit resume."""
    seeds, instructions, image_hashes, manifest = _preflight(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # The lock lives outside the run directory, so a newly interrupted run can be diagnosed.
    with (config.output_dir / f".{config.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Run is already active: {config.name}") from error
        run_dir = config.output_dir / config.name
        if resume:
            if not (run_dir / "manifest.json").exists():
                raise ValueError("Cannot resume: run manifest is missing")
            previous = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            if previous.get("fingerprint") != manifest["fingerprint"]:
                raise ValueError(
                    "Cannot resume: config, input, prompt, or image content has changed"
                )
            manifest["created_at"] = previous.get("created_at")
        else:
            if run_dir.exists():
                raise ValueError(f"Run directory already exists: {run_dir}; use --resume")
            run_dir.mkdir()
            manifest["created_at"] = datetime.now(timezone.utc).isoformat()
            _write_text(run_dir / "config.yaml", yaml.safe_dump(config.model_dump(mode="json")))

        tasks = []
        for seed_index, seed in enumerate(seeds):
            for candidate_index in range(config.samples_per_seed):
                task_key = _digest(
                    f"{manifest['fingerprint']}:{seed.id}:{candidate_index}".encode()
                )
                tasks.append((task_key, seed_index, candidate_index))
        expected = {key: (seed_index, candidate) for key, seed_index, candidate in tasks}
        journal = _read_journal(run_dir / "raw.jsonl")
        responses = {}
        for row in journal:
            key = row.get("task_key")
            if not isinstance(key, str) or key not in expected or key in responses:
                raise ValueError("Invalid, unknown, or duplicate task_key in raw journal")
            seed_index, candidate = expected[key]
            if (
                row.get("seed_id") != seeds[seed_index].id
                or row.get("candidate_index") != candidate
            ):
                raise ValueError("Raw journal task metadata does not match the run")
            GenerationResult.model_validate(row.get("result"))
            responses[key] = row
        _read_journal(run_dir / "errors.jsonl")
        for filename in ("raw.jsonl", "errors.jsonl"):
            (run_dir / filename).touch(exist_ok=True)
        manifest.update(status="running", expected=len(tasks))
        _write_json(run_dir / "manifest.json", manifest)
        pending = iter(task for task in tasks if task[0] not in responses)
        failed: set[str] = set()

        async def worker(active_provider):
            for task_key, seed_index, candidate_index in pending:
                seed = seeds[seed_index]
                try:
                    result = await active_provider.generate(
                        make_request(seed, instructions[seed_index])
                    )
                    result = GenerationResult.model_validate(result)
                except Exception as error:
                    failed.add(task_key)
                    append_jsonl(
                        run_dir / "errors.jsonl",
                        {
                            "task_key": task_key,
                            "seed_id": seed.id,
                            "candidate_index": candidate_index,
                            "kind": "api_error",
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    continue
                row = {
                    "task_key": task_key,
                    "seed_id": seed.id,
                    "candidate_index": candidate_index,
                    "result": result.model_dump(mode="json"),
                }
                append_jsonl(run_dir / "raw.jsonl", row)
                responses[task_key] = row

        if len(responses) < len(tasks):
            active_provider = provider or create_provider(config.model, config.media_root)
            try:
                async with asyncio.TaskGroup() as workers:
                    for _ in range(min(config.concurrency, len(tasks) - len(responses))):
                        workers.create_task(worker(active_provider))
            finally:
                await active_provider.aclose()

        samples, clean, rejected, quality, ordered_raw = [], [], [], [], []
        accepted_models = []
        seen: dict[str, str] = {}
        for task_key, seed_index, candidate_index in tasks:
            if task_key not in responses:
                continue
            row = responses[task_key]
            ordered_raw.append(row)
            result = GenerationResult.model_validate(row["result"])
            issues = response_issues(result)
            sample = None
            duplicate_of = None
            verification = None
            seed = seeds[seed_index]
            if config.task_type == "rlvr":
                verification = {
                    "verifier": seed.verifier,
                    "evaluated": False,
                    "passed": None,
                    "reason": issues[0] if issues else None,
                    "answer": None,
                }
            if not issues:
                metadata = {
                    "run_id": config.name,
                    "seed_id": seed.id,
                    "candidate_index": candidate_index,
                    "provider": config.model.provider,
                    "model": config.model.model,
                    "model_parameters": config.model.model_dump(mode="json"),
                    "prompt_sha256": manifest["prompt_sha256"],
                    "variables": seed.variables,
                    "request_id": result.request_id,
                }
                if config.task_type == "rlvr":
                    sample = make_rlvr_sample(seed, task_key, metadata)
                    if len(result.text) < config.quality.min_chars:
                        issues.append("completion_too_short")
                    elif len(result.text) > config.quality.max_chars:
                        issues.append("completion_too_long")
                    if issues:
                        verification["reason"] = issues[0]
                    else:
                        checked = verify_answer(seed.verifier, seed.reference, result.text)
                        verification.update(
                            evaluated=True,
                            passed=checked.passed,
                            reason=checked.reason,
                            answer=checked.answer,
                        )
                        if not checked.passed:
                            issues.append(checked.reason)
                else:
                    sample = make_sample(seed, result, task_key, metadata)
                    issues.extend(
                        sample_issues(sample, config.quality.min_chars, config.quality.max_chars)
                    )
                samples.append(sample.model_dump(mode="json"))
                if not issues and config.quality.deduplicate:
                    digest = sample_digest(sample, image_hashes)
                    duplicate_of = seen.get(digest)
                    if duplicate_of:
                        issues.append("duplicate")
                    else:
                        seen[digest] = task_key
            assessment = {"sample_id": task_key, "passed": not issues, "reasons": issues}
            if verification is not None:
                assessment["verification"] = verification
            if duplicate_of:
                assessment["duplicate_of"] = duplicate_of
            quality.append(assessment)
            if issues:
                rejected.append(
                    {
                        **assessment,
                        "seed_id": seeds[seed_index].id,
                        "sample": sample.model_dump(mode="json") if sample else None,
                    }
                )
            else:
                clean.append(sample.model_dump(mode="json"))
                accepted_models.append(sample)
        for name, rows in (
            ("samples", samples),
            ("quality", quality),
            ("clean", clean),
            ("rejected", rejected),
        ):
            write_jsonl(run_dir / f"{name}.jsonl", rows)
        report = run_report(
            samples,
            clean,
            rejected,
            ordered_raw,
            expected=len(tasks),
            failed=len(failed),
            quality=quality,
        )
        report["splits"] = export_splits(
            accepted_models,
            run_dir / "exports",
            config.media_root,
            image_hashes,
            config.validation_fraction,
            config.split_seed,
        )
        _write_json(run_dir / "report.json", report)
        manifest.update(status="partial" if failed else "completed", failed_requests=len(failed))
        _write_json(run_dir / "manifest.json", manifest)
        return report
