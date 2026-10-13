"""Conservative checks; text is never rewritten by these rules."""

from dataforge.schemas import GenerationResult, SFTSample


def response_issues(result: GenerationResult) -> list[str]:
    issues = []
    if result.status != "completed":
        issues.append(result.status)
    if not result.text.strip():
        issues.append("empty_completion")
    return issues


def sample_issues(sample: SFTSample, min_chars: int, max_chars: int) -> list[str]:
    length = sum(len(block.text) for message in sample.completion for block in message.content)
    if length < min_chars:
        return ["completion_too_short"]
    if length > max_chars:
        return ["completion_too_long"]
    return []
