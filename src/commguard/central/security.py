"""Canonical HMAC signing, size limits, and telemetry privacy validation."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from commguard.exceptions import ValidationError
from commguard.schemas import FIELD_UNITS, TELEMETRY_FIELDS, FieldReading

MAX_PAYLOAD_BYTES = 1_048_576
SAMPLE_KEYS = frozenset(
    {
        "sample_id",
        "observed_at_utc",
        "monotonic_ns",
        "gpu_index",
        "gpu_uuid",
        "fields",
    }
)
READING_KEYS = frozenset({"value", "unit", "supported", "error"})
PROHIBITED_KEY_FRAGMENTS = (
    "prompt",
    "token",
    "dataset",
    "example",
    "content",
    "weight",
    "credential",
    "secret",
    "api_key",
)


def canonical_bytes(message: Mapping[str, Any], *, include_signature: bool = False) -> bytes:
    payload = dict(message)
    if not include_signature:
        payload.pop("signature", None)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sign_message(message: Mapping[str, Any], secret: bytes) -> str:
    if not secret:
        raise ValueError("HMAC secret must not be empty")
    return hmac.new(secret, canonical_bytes(message), hashlib.sha256).hexdigest()


def verify_message(message: Mapping[str, Any], secret: bytes) -> bool:
    supplied = message.get("signature")
    return isinstance(supplied, str) and hmac.compare_digest(
        supplied, sign_message(message, secret)
    )


def validate_payload_size(
    message: Mapping[str, Any], maximum_bytes: int = MAX_PAYLOAD_BYTES
) -> int:
    if maximum_bytes < 1:
        raise ValueError("maximum payload bytes must be positive")
    size = len(canonical_bytes(message, include_signature=True))
    if size > maximum_bytes:
        raise ValueError(f"payload_too_large: bytes={size} maximum={maximum_bytes}")
    return size


def _reject_prohibited_keys(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in PROHIBITED_KEY_FRAGMENTS):
                raise ValueError(f"privacy_violation: prohibited key {path}.{key}")
            _reject_prohibited_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_prohibited_keys(child, f"{path}[{index}]")


def validate_telemetry_samples(samples: tuple[dict[str, Any], ...]) -> None:
    _reject_prohibited_keys(samples)
    for index, sample in enumerate(samples):
        unknown = sorted(set(sample) - SAMPLE_KEYS)
        if unknown:
            raise ValueError(f"privacy_violation: sample {index} unknown keys {unknown}")
        missing = sorted(SAMPLE_KEYS - set(sample))
        if missing:
            raise ValueError(f"invalid_sample: sample {index} missing keys {missing}")
        try:
            observed = datetime.fromisoformat(str(sample["observed_at_utc"]).replace("Z", "+00:00"))
            monotonic_ns = int(sample["monotonic_ns"])
            gpu_index = int(sample["gpu_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid_sample: sample {index} identity/timestamp") from exc
        if observed.tzinfo is None or monotonic_ns < 0 or gpu_index < 0 or not sample["gpu_uuid"]:
            raise ValueError(f"invalid_sample: sample {index} identity/timestamp")
        fields = sample.get("fields")
        if not isinstance(fields, Mapping) or set(fields) != set(TELEMETRY_FIELDS):
            raise ValueError(f"invalid_sample: sample {index} telemetry fields must be exact")
        for name, reading in fields.items():
            if not isinstance(reading, Mapping) or set(reading) != READING_KEYS:
                raise ValueError(
                    f"invalid_sample: sample {index} field {name} reading keys must be exact"
                )
            try:
                parsed = FieldReading.from_dict(reading)
                parsed.validate(f"sample[{index}].fields.{name}")
            except (ValidationError, ValueError) as exc:
                raise ValueError(f"invalid_sample: sample {index} field {name}: {exc}") from exc
            if parsed.unit != FIELD_UNITS[name]:
                raise ValueError(f"invalid_sample: sample {index} field {name} unit mismatch")
