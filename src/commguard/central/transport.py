"""Deterministic offline transport and optional standard-library HTTP adapters."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from commguard.central.schemas import IngestionAck
from commguard.central.security import MAX_PAYLOAD_BYTES, canonical_bytes
from commguard.central.server import CentralIngestionService


def _message_id(message: dict[str, Any]) -> str:
    value = str(message.get("batch_id") or message.get("heartbeat_id") or "")
    if re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError("message ID is not safe for offline transport")
    return value


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


class OfflineFileTransport:
    """Create-only request/ack log around an in-process central service."""

    def __init__(self, root: str | Path, service: CentralIngestionService) -> None:
        self.root = Path(root).resolve()
        self.service = service

    def send(self, message: dict[str, Any]) -> IngestionAck:
        identifier = _message_id(message)
        _exclusive_json(self.root / "requests" / f"{identifier}.json", message)
        acknowledgment = self.service.ingest(message)
        _exclusive_json(
            self.root / "acknowledgments" / f"{identifier}.json",
            acknowledgment.to_dict(),
        )
        return acknowledgment


class HttpTransport:
    """Optional HTTPS client; production use requires a verified TLS URL."""

    def __init__(self, url: str, timeout_seconds: float = 10.0) -> None:
        if not url.startswith("https://"):
            raise ValueError("online central transport requires an https:// URL")
        self.url = url
        self.timeout_seconds = timeout_seconds

    def send(self, message: dict[str, Any]) -> IngestionAck:
        request = urllib.request.Request(
            self.url,
            data=canonical_bytes(message, include_signature=True),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return IngestionAck(**payload)


def build_http_handler(service: CentralIngestionService) -> type[BaseHTTPRequestHandler]:
    """Create an optional handler; callers own TLS wrapping and server lifecycle."""

    class IngestionHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            if self.path != "/v1/ingest":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(400)
                return
            if length < 1 or length > min(service.maximum_payload_bytes, MAX_PAYLOAD_BYTES):
                self.send_error(413)
                return
            try:
                payload = json.loads(self.rfile.read(length))
                acknowledgment = service.ingest(payload)
                encoded = canonical_bytes(
                    acknowledgment.to_dict(),
                    include_signature=True,
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                self.send_error(400)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            # Avoid standard-library request logging; deployment logging must be
            # explicitly configured to exclude signatures and credentials.
            return

    return IngestionHandler
