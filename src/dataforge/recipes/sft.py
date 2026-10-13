"""Answer prepared prompts without including synthesis instructions in training data."""

from dataforge.schemas import (
    GenerationRequest,
    GenerationResult,
    Message,
    Seed,
    SFTSample,
    TextBlock,
)


def make_request(seed: Seed, instructions: str) -> GenerationRequest:
    return GenerationRequest(messages=seed.prompt, instructions=instructions)


def make_sample(
    seed: Seed,
    result: GenerationResult,
    task_key: str,
    metadata: dict,
) -> SFTSample:
    return SFTSample(
        id=task_key,
        source_id=seed.source_id or seed.id,
        group_id=seed.group_id or seed.source_id or seed.id,
        prompt=seed.prompt,
        completion=[Message(role="assistant", content=[TextBlock(text=result.text)])],
        metadata=metadata,
    )
