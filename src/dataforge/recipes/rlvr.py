"""Build verifiable tasks from trusted labeled seeds and teacher trial answers."""

from dataforge.schemas import Message, RLVRSample, RLVRSeed, TextBlock

ANSWER_FORMAT = "将最终答案单独写在最后一行，格式为 #### <答案>。"


def prepare_seed(seed: RLVRSeed) -> RLVRSeed:
    """Give the teacher and the exported task the same final-answer convention."""
    prepared = seed.model_copy(deep=True)
    if prepared.prompt[0].role == "system":
        prepared.prompt[0].content.append(TextBlock(text="\n" + ANSWER_FORMAT))
    else:
        prepared.prompt.insert(0, Message(role="system", content=[TextBlock(text=ANSWER_FORMAT)]))
    return prepared


def make_sample(seed: RLVRSeed, task_key: str, metadata: dict) -> RLVRSample:
    return RLVRSample(
        id=task_key,
        source_id=seed.source_id or seed.id,
        group_id=seed.group_id or seed.source_id or seed.id,
        prompt=seed.prompt,
        reference=seed.reference,
        verifier=seed.verifier,
        metadata=metadata,
    )
