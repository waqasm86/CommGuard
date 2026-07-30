from __future__ import annotations

import pytest

from commguard.artifacts.models import FieldReading
from commguard.exceptions import ValidationError


def test_unsupported_reading_requires_null_and_error_metadata() -> None:
    reading = FieldReading(
        value=None,
        unit="bytes/second",
        supported=False,
        error="NVML_ERROR_NOT_SUPPORTED",
    )
    reading.validate()
    assert reading.value is None
    assert not reading.supported
    assert reading.error == "NVML_ERROR_NOT_SUPPORTED"


def test_unsupported_reading_cannot_be_zero() -> None:
    with pytest.raises(ValidationError, match="must be null"):
        FieldReading(
            value=0,
            unit="bytes/second",
            supported=False,
            error="NVML_ERROR_NOT_SUPPORTED",
        ).validate()
