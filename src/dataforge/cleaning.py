"""Exact deduplication on canonical data, preserving source whitespace."""

import hashlib
import json

from dataforge.schemas import RLVRSample, SFTSample


def sample_digest(sample: SFTSample | RLVRSample, image_hashes: dict[str, str]) -> str:
    fields = (
        {"prompt", "reference", "verifier"}
        if sample.task_type == "rlvr"
        else {"prompt", "completion"}
    )
    content = sample.model_dump(mode="json", include=fields)
    message_fields = ("prompt",) if sample.task_type == "rlvr" else ("prompt", "completion")
    for field in message_fields:
        for message in content[field]:
            for block in message["content"]:
                if block["type"] == "image":
                    block["uri"] = image_hashes[block["uri"]]
    # Canonicalize serialization only: whitespace in code, formulas and prose is data.
    serialized = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
