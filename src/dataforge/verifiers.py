"""Deterministic answer checks for text-only RLVR tasks.

Numeric answers are single literals, limited to 256 characters and exponents
between -1000 and 1000. Integer fractions and strict thousands grouping are
supported; expressions, units, and symbolic mathematics are not.
"""

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

_INTEGER = r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)"
_NUMBER = re.compile(
    rf"(?:{_INTEGER}(?:\.[0-9]*)?|[+-]?\.[0-9]+)(?:[eE](?P<exponent>[+-]?[0-9]+))?"
)
_FRACTION = re.compile(rf"(?P<numerator>{_INTEGER})/(?P<denominator>{_INTEGER})")
_MAX_NUMBER_LENGTH = 256
_MAX_EXPONENT = 1000


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reason: str | None = None
    answer: str | None = None


def _parse_number(value: str) -> Fraction:
    value = value.strip()
    if not value or len(value) > _MAX_NUMBER_LENGTH:
        raise ValueError("numeric answer must contain between 1 and 256 characters")
    fraction = _FRACTION.fullmatch(value)
    if fraction:
        numerator = int(fraction["numerator"].replace(",", ""))
        denominator = int(fraction["denominator"].replace(",", ""))
        if denominator == 0:
            raise ValueError("numeric answer cannot have a zero denominator")
        return Fraction(numerator, denominator)
    number = _NUMBER.fullmatch(value)
    if not number:
        raise ValueError("numeric answer must be a finite number or integer fraction")
    if number["exponent"] is not None and abs(int(number["exponent"])) > _MAX_EXPONENT:
        raise ValueError("numeric answer exponent must be between -1000 and 1000")
    return Fraction(Decimal(value.replace(",", "")))


def validate_reference(verifier: str, reference: dict) -> dict:
    """Validate trusted task labels and return a JSON-compatible reference."""
    if verifier not in ("exact_match", "math_answer"):
        raise ValueError(f"unsupported RLVR verifier: {verifier}")
    if not isinstance(reference, dict):
        raise ValueError("reference must be an object")
    allowed = (
        {"answer", "strip", "ignore_case"} if verifier == "exact_match" else {"answer", "tolerance"}
    )
    if reference.keys() - allowed:
        raise ValueError("reference contains unknown fields")
    answer = reference.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("reference.answer must be a nonempty string")
    if answer.splitlines() != [answer]:
        raise ValueError("reference.answer must be a single line")
    if verifier == "exact_match":
        strip = reference.get("strip", True)
        ignore_case = reference.get("ignore_case", False)
        if not isinstance(strip, bool) or not isinstance(ignore_case, bool):
            raise ValueError("reference.strip and reference.ignore_case must be booleans")
        return {"answer": answer, "strip": strip, "ignore_case": ignore_case}
    _parse_number(answer)
    tolerance = reference.get("tolerance", 0)
    if type(tolerance) not in (int, float) or tolerance < 0:
        raise ValueError("reference.tolerance must be a nonnegative finite number")
    if isinstance(tolerance, float) and not math.isfinite(tolerance):
        raise ValueError("reference.tolerance must be a nonnegative finite number")
    _parse_number(str(tolerance))
    return {"answer": answer, "tolerance": tolerance}


def verify_answer(verifier: str, reference: dict, response: str) -> VerificationResult:
    """Check the sole final ``#### <answer>`` line, without parsing reasoning."""
    reference = validate_reference(verifier, reference)
    lines = [line for line in response.splitlines() if line.strip()]
    if (
        not lines
        or not lines[-1].startswith("#### ")
        or sum(line.lstrip().startswith("####") for line in lines) != 1
    ):
        return VerificationResult(False, "invalid_answer_format")
    answer = lines[-1][5:]
    if not answer.strip():
        return VerificationResult(False, "invalid_answer_format")
    if verifier == "exact_match":
        actual, expected = answer, reference["answer"]
        if reference["strip"]:
            actual, expected = actual.strip(), expected.strip()
        if reference["ignore_case"]:
            actual, expected = actual.casefold(), expected.casefold()
        passed = actual == expected
    else:
        try:
            actual = _parse_number(answer)
        except ValueError:
            return VerificationResult(False, "invalid_numeric_answer", answer)
        expected = _parse_number(reference["answer"])
        tolerance = _parse_number(str(reference["tolerance"]))
        passed = abs(actual - expected) <= tolerance
    return VerificationResult(passed, None if passed else "answer_mismatch", answer)
