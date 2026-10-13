"""Provider-independent inputs, training samples, and generation results."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextBlock(StrictModel):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(StrictModel):
    type: Literal["image"] = "image"
    uri: str = Field(min_length=1)


ContentBlock = Annotated[TextBlock | ImageBlock, Field(discriminator="type")]


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: list[ContentBlock] = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def validate_content(cls, content, info):
        if info.data.get("role") != "user" and any(
            isinstance(block, ImageBlock) for block in content
        ):
            raise ValueError("images are supported only in user messages")
        return content


def validate_prompt(messages: list[Message]) -> list[Message]:
    if not messages or messages[-1].role != "user":
        raise ValueError("prompt must be nonempty and end with a user message")
    if any(message.role == "system" for message in messages[1:]):
        raise ValueError("a system message is allowed only at the start of a prompt")
    return messages


def validate_completion(messages: list[Message]) -> list[Message]:
    if not messages or any(message.role != "assistant" for message in messages):
        raise ValueError("completion must contain only assistant messages")
    return messages


class Seed(StrictModel):
    id: str = Field(min_length=1)
    prompt: list[Message] = Field(min_length=1)
    source_id: str | None = None
    group_id: str | None = None
    variables: dict[str, Any] = Field(default_factory=dict)

    _validate_prompt = field_validator("prompt")(validate_prompt)


class SampleBase(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(min_length=1)
    source_id: str | None = None
    group_id: str | None = None
    prompt: list[Message] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _validate_prompt = field_validator("prompt")(validate_prompt)


class SFTSample(SampleBase):
    task_type: Literal["sft"] = "sft"
    completion: list[Message] = Field(min_length=1)

    _validate_completion = field_validator("completion")(validate_completion)


class PreferenceSample(SampleBase):
    task_type: Literal["preference"] = "preference"
    chosen: list[Message] = Field(min_length=1)
    rejected: list[Message] = Field(min_length=1)

    _validate_completion = field_validator("chosen", "rejected")(validate_completion)


class RLTaskSample(SampleBase):
    task_type: Literal["rl_task"] = "rl_task"
    reference: dict[str, Any]
    verifier: str = Field(min_length=1)


Sample = Annotated[SFTSample | PreferenceSample | RLTaskSample, Field(discriminator="task_type")]
sample_adapter = TypeAdapter(Sample)


class GenerationRequest(StrictModel):
    messages: list[Message] = Field(min_length=1)
    instructions: str = ""

    _validate_prompt = field_validator("messages")(validate_prompt)


class GenerationResult(StrictModel):
    text: str
    status: Literal["completed", "incomplete", "refused"]
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    request_id: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
