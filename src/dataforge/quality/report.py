"""Small JSON reports that do not require a database or notebook."""

from collections import Counter
from statistics import mean
from typing import Any


def dataset_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths: list[int] = []
    for row in rows:
        if "completion" not in row:
            continue
        messages = row["completion"]
        lengths.append(
            sum(len(block.get("text", "")) for message in messages for block in message["content"])
        )
    report = {
        "count": len(rows),
        "task_types": dict(Counter(row["task_type"] for row in rows)),
        "sources": dict(Counter(row.get("source_id") or "unknown" for row in rows)),
        "completion_characters": {
            "count": len(lengths),
            "min": min(lengths, default=0),
            "max": max(lengths, default=0),
            "mean": mean(lengths) if lengths else 0,
            "total": sum(lengths),
        },
    }
    verifiers = Counter(row["verifier"] for row in rows if "verifier" in row)
    if verifiers:
        report["verifiers"] = dict(verifiers)
    return report


def run_report(
    samples: list[dict],
    clean: list[dict],
    rejected: list[dict],
    raw: list[dict],
    *,
    expected: int,
    failed: int,
    quality: list[dict] | None = None,
) -> dict:
    report = {
        "expected": expected,
        "responses": len(raw),
        "failed_requests": failed,
        "samples": len(samples),
        "accepted": len(clean),
        "rejected": len(rejected),
        "pass_rate": len(clean) / expected if expected else 0,
        "rejection_reasons": dict(Counter(reason for row in rejected for reason in row["reasons"])),
        "token_usage": {
            "input": sum(row["result"]["input_tokens"] for row in raw),
            "output": sum(row["result"]["output_tokens"] for row in raw),
        },
        "accepted_dataset": dataset_report(clean),
    }
    verifications = [row["verification"] for row in quality or [] if "verification" in row]
    if verifications:
        by_verifier = {}
        for verifier in sorted({row["verifier"] for row in verifications}):
            results = [row for row in verifications if row["verifier"] == verifier]
            evaluated = sum(row["evaluated"] for row in results)
            passed = sum(row["passed"] is True for row in results)
            by_verifier[verifier] = {
                "evaluated": evaluated,
                "passed": passed,
                "failed": evaluated - passed,
                "skipped": len(results) - evaluated,
            }
        evaluated = sum(row["evaluated"] for row in verifications)
        passed = sum(row["passed"] is True for row in verifications)
        report["verification"] = {
            "evaluated": evaluated,
            "passed": passed,
            "failed": evaluated - passed,
            "skipped": len(verifications) - evaluated,
            "pass_rate": passed / evaluated if evaluated else 0,
            "by_verifier": by_verifier,
        }
    return report
