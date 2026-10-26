import json

import pytest
from pydantic import ValidationError

from dataforge.schemas import RLVRSample, RLVRSeed, sample_adapter
from dataforge.verifiers import validate_reference, verify_answer


def prompt():
    return [{"role": "user", "content": [{"type": "text", "text": "What is the answer?"}]}]


def test_exact_match_options_preserve_internal_whitespace():
    assert verify_answer("exact_match", {"answer": "Paris"}, "Reasoning\n#### Paris \n\n").passed
    assert not verify_answer("exact_match", {"answer": "Paris"}, "#### paris").passed
    assert verify_answer(
        "exact_match", {"answer": "Straße", "ignore_case": True}, "#### STRASSE"
    ).passed
    assert not verify_answer("exact_match", {"answer": "New York"}, "#### New  York").passed
    assert verify_answer(
        "exact_match", {"answer": " Paris ", "strip": False}, "####  Paris "
    ).passed
    mismatch = verify_answer("exact_match", {"answer": "Paris", "strip": False}, "#### Paris ")
    assert not mismatch.passed
    assert mismatch.reason == "answer_mismatch"
    assert mismatch.answer == "Paris "


@pytest.mark.parametrize(
    "response",
    [
        "",
        "   ",
        "The answer is 42.",
        "####",
        "#### ",
        "####   ",
        "####\t42",
        " #### 42",
        "#### 42\nMore reasoning",
        "#### 1\n#### 42",
        "####\n#### 42",
        "  #### 1\n#### 42",
        "##### 42",
        "42",
    ],
)
def test_final_answer_format_is_required_and_unambiguous(response):
    result = verify_answer("math_answer", {"answer": "42"}, response)
    assert not result.passed
    assert result.reason == "invalid_answer_format"
    assert result.answer is None


def test_verifier_uses_final_answer_without_extracting_reasoning_numbers():
    result = verify_answer("math_answer", {"answer": "42"}, "42 + 0 = 42\n#### 41")
    assert not result.passed
    assert result.reason == "answer_mismatch"
    assert result.answer == "41"
    result = verify_answer("math_answer", {"answer": "42"}, "The answer is 41.\n#### 42\n")
    assert result.passed
    assert result.reason is None
    assert result.answer == "42"


@pytest.mark.parametrize(
    ("reference", "answer"),
    [
        ("42", "+42"),
        ("-42", "-4.2e1"),
        ("0", "-0.0"),
        (".5", "1/2"),
        ("1/3", "2/6"),
        ("1/3", "-2/-6"),
        ("2.", "2e0"),
        ("1000", "1,000"),
        ("-1234.5", "-1,234.50"),
        ("2000", "2,000/1"),
        ("0.1", "1e-1"),
        ("1e1000", "10e999"),
        ("1e-1000", "1e-1000"),
        ("123456789012345678901234567890", "123456789012345678901234567890.0"),
    ],
)
def test_numeric_answers_compare_exactly(reference, answer):
    result = verify_answer("math_answer", {"answer": reference}, f"#### {answer}")
    assert result.passed


@pytest.mark.parametrize(
    ("answer", "tolerance", "passed"),
    [
        ("1.01", 0.01, True),
        ("0.99", 0.01, True),
        ("1.010000000000000001", 0.01, False),
        ("0.989999999999999999", 0.01, False),
        ("2", 1, True),
        ("1.000000000000000001", 0, False),
    ],
)
def test_numeric_tolerance_is_absolute_and_inclusive(answer, tolerance, passed):
    result = verify_answer("math_answer", {"answer": "1", "tolerance": tolerance}, f"#### {answer}")
    assert result.passed is passed
    assert result.reason == (None if passed else "answer_mismatch")


def test_rational_comparison_does_not_round_repeating_decimals():
    assert not verify_answer("math_answer", {"answer": "1/3"}, "#### 0.3333333333333333").passed
    assert verify_answer(
        "math_answer", {"answer": "1/3", "tolerance": 1e-16}, "#### 0.3333333333333333"
    ).passed


@pytest.mark.parametrize(
    "answer",
    [
        "NaN",
        "nan",
        "Inf",
        "-Infinity",
        "True",
        "1/0",
        "0/0",
        "1+1",
        "sqrt(4)",
        "1 or 2",
        "1,2",
        "12,34",
        "1,2345",
        "1_000",
        "2 kg",
        "1e1001",
        "1e-1001",
        "9" * 257,
        "1/2/3",
        "１",
        "0x10",
    ],
)
def test_unsafe_or_ambiguous_numeric_answers_fail(answer):
    result = verify_answer("math_answer", {"answer": "42"}, f"#### {answer}")
    assert not result.passed
    assert result.reason == "invalid_numeric_answer"


@pytest.mark.parametrize("model", [RLVRSeed, RLVRSample])
@pytest.mark.parametrize(
    ("verifier", "reference"),
    [
        ("exact_match", {}),
        ("exact_match", {"answer": ""}),
        ("exact_match", {"answer": "   "}),
        ("exact_match", {"answer": 42}),
        ("exact_match", {"answer": True}),
        ("exact_match", {"answer": "a\nb"}),
        ("exact_match", {"answer": "a\n"}),
        ("exact_match", {"answer": "a\r", "strip": False}),
        ("exact_match", {"answer": "a\r\n", "strip": False}),
        ("exact_match", {"answer": "A", "strip": 1}),
        ("exact_match", {"answer": "A", "ignore_case": "false"}),
        ("exact_match", {"answer": "A", "unexpected": True}),
        ("math_answer", {"answer": "NaN"}),
        ("math_answer", {"answer": "Inf"}),
        ("math_answer", {"answer": "1/0"}),
        ("math_answer", {"answer": "1+1"}),
        ("math_answer", {"answer": 1}),
        ("math_answer", {"answer": "1e1001"}),
        ("math_answer", {"answer": "9" * 257}),
        ("math_answer", {"answer": "1", "tolerance": -0.1}),
        ("math_answer", {"answer": "1", "tolerance": float("nan")}),
        ("math_answer", {"answer": "1", "tolerance": float("inf")}),
        ("math_answer", {"answer": "1", "tolerance": True}),
        ("math_answer", {"answer": "1", "tolerance": "0.01"}),
        ("math_answer", {"answer": "1", "ignore_case": True}),
        ("unsupported", {"answer": "1"}),
    ],
)
def test_invalid_references_fail_at_schema_validation(model, verifier, reference):
    with pytest.raises(ValidationError):
        model(id="task", prompt=prompt(), verifier=verifier, reference=reference)


@pytest.mark.parametrize("model", [RLVRSeed, RLVRSample])
def test_rlvr_schema_rejects_images(model):
    image_prompt = [{"role": "user", "content": [{"type": "image", "uri": "image.png"}]}]
    with pytest.raises(ValidationError, match="text-only"):
        model(id="task", prompt=image_prompt, verifier="exact_match", reference={"answer": "cat"})


def test_rlvr_reference_normalization_and_sample_adapter_round_trip():
    sample = sample_adapter.validate_python(
        {
            "id": "task",
            "task_type": "rlvr",
            "prompt": prompt(),
            "verifier": "exact_match",
            "reference": {"answer": "  A  "},
        }
    )
    assert isinstance(sample, RLVRSample)
    assert sample.reference == {"answer": "  A  ", "strip": True, "ignore_case": False}
    assert sample_adapter.validate_json(sample.model_dump_json()) == sample
    seed = RLVRSeed(id="task", prompt=prompt(), verifier="math_answer", reference={"answer": "1/3"})
    assert seed.reference == {"answer": "1/3", "tolerance": 0}
    assert json.loads(seed.model_dump_json())["reference"] == seed.reference


def test_legacy_rl_task_keeps_arbitrary_reference_and_verifier():
    sample = sample_adapter.validate_python(
        {
            "id": "legacy",
            "task_type": "rl_task",
            "prompt": prompt(),
            "verifier": "custom_environment",
            "reference": {"answer": 42, "custom": [1, 2]},
        }
    )
    assert sample.reference == {"answer": 42, "custom": [1, 2]}
    assert sample.verifier == "custom_environment"


def test_validation_rejects_unknown_verifier_and_non_object_reference():
    with pytest.raises(ValueError, match="unsupported"):
        validate_reference("unknown", {"answer": "42"})
    with pytest.raises(ValueError, match="object"):
        validate_reference("math_answer", [])
