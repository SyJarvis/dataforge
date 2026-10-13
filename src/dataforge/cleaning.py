"""Exact deduplication on canonical data, preserving source whitespace."""

import hashlib
import json

from dataforge.schemas import SFTSample


def sample_digest(sample: SFTSample, image_hashes: dict[str, str]) -> str:
    content = sample.model_dump(mode="json", include={"prompt", "completion"})
    for field in ("prompt", "completion"):
        for message in content[field]:
            for block in message["content"]:
                if block["type"] == "image":
                    block["uri"] = image_hashes[block["uri"]]
    # Canonicalize serialization only: whitespace in code, formulas and prose is data.
    serialized = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
