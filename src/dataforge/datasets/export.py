"""Portable messages export; image paths are relative to the exports directory."""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from dataforge.datasets.io import write_jsonl
from dataforge.datasets.media import resolve_image
from dataforge.schemas import RLVRSample, SFTSample


def export_splits(
    samples: list[SFTSample | RLVRSample],
    output_dir: Path,
    media_root: Path,
    image_hashes: dict[str, str],
    validation_fraction: float,
    split_seed: int,
) -> dict[str, int]:
    splits: dict[str, list[dict]] = {"train": [], "validation": []}
    for sample in samples:
        group = sample.group_id or sample.source_id or sample.id
        digest = hashlib.sha256(f"{split_seed}:{group}".encode()).digest()
        validation = int.from_bytes(digest, "big") / (1 << 256) < validation_fraction
        data = sample.model_dump(mode="json")
        messages = data["prompt"]
        if sample.task_type == "sft":
            messages += data["completion"]
        has_image = any(
            block["type"] == "image" for message in messages for block in message["content"]
        )
        images: list[str] = []
        for message in messages:
            if not has_image:
                message["content"] = "".join(block["text"] for block in message["content"])
                continue
            for block in message["content"]:
                if block["type"] != "image":
                    continue
                source = resolve_image(block["uri"], media_root)
                expected_hash = image_hashes[block["uri"]]
                if hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
                    raise ValueError(f"Image changed during this run: {block['uri']}")
                relative = Path("assets") / f"{expected_hash}{source.suffix.lower()}"
                destination = output_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if (
                    not destination.exists()
                    or hashlib.sha256(destination.read_bytes()).hexdigest() != expected_hash
                ):
                    with tempfile.NamedTemporaryFile(
                        dir=destination.parent, delete=False
                    ) as stream:
                        temporary = Path(stream.name)
                        try:
                            with source.open("rb") as source_stream:
                                shutil.copyfileobj(source_stream, stream)
                            stream.flush()
                            os.fsync(stream.fileno())
                        except BaseException:
                            temporary.unlink(missing_ok=True)
                            raise
                    os.replace(temporary, destination)
                block["uri"] = relative.as_posix()
                if block["uri"] not in images:
                    images.append(block["uri"])
        row = {"id": sample.id, "group_id": group, "messages": messages}
        if sample.task_type == "rlvr":
            row.update(task_type="rlvr", reference=data["reference"], verifier=sample.verifier)
        if has_image:
            row["images"] = images
        splits["validation" if validation else "train"].append(row)
    for name, rows in splits.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    return {name: len(rows) for name, rows in splits.items()}
