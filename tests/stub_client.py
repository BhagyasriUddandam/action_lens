"""Deterministic stand-in for anthropic.Anthropic, used by --dry-run.

Exists so the whole vlm_benchmark pipeline -- frame selection, base64 encoding,
request assembly, schema parsing, caching, metrics, cost accounting -- can be
exercised without an API key and without spending money.

It VALIDATES the request rather than just accepting it: a malformed image
block or a missing system prompt raises here, on a laptop, instead of showing
up as a paid 400 partway through a 160-clip run.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass

VALID_ACTIONS = ("walking", "sitting", "standing", "falling")


@dataclass
class StubUsage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class StubParsed:
    action: str
    confidence: float
    evidence: str


@dataclass
class StubResponse:
    usage: StubUsage
    parsed_output: StubParsed | None
    stop_reason: str = "end_turn"


class StubMessages:
    def __init__(self) -> None:
        self.calls = 0
        self._seen_system = False

    def parse(self, **kwargs):
        self.calls += 1
        _validate(kwargs)

        # Deterministic pseudo-prediction keyed on the image bytes, so the same
        # clip always yields the same label across dry runs.
        images = [
            b["source"]["data"]
            for b in kwargs["messages"][0]["content"]
            if b.get("type") == "image"
        ]
        digest = hashlib.sha256("".join(images).encode()).hexdigest()
        action = VALID_ACTIONS[int(digest[:8], 16) % len(VALID_ACTIONS)]

        n_images = len(images)
        # Roughly Anthropic's (w*h)/750 for a ~400x256 frame, plus prompt text.
        input_tokens = n_images * 137 + 120
        cache_read = 0 if not self._seen_system else 400
        cache_write = 400 if not self._seen_system else 0
        self._seen_system = True

        return StubResponse(
            usage=StubUsage(
                input_tokens=input_tokens,
                output_tokens=95,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_write,
            ),
            parsed_output=StubParsed(
                action=action,
                confidence=0.5 + (int(digest[8:10], 16) % 50) / 100,
                evidence="stubbed response; no model was called",
            ),
        )


class StubClient:
    def __init__(self) -> None:
        self.messages = StubMessages()


def _validate(kwargs: dict) -> None:
    """Fail loudly on anything the real API would reject."""
    for required in ("model", "max_tokens", "system", "messages", "output_format"):
        if required not in kwargs:
            raise AssertionError(f"request missing {required!r}")

    system = kwargs["system"]
    if not (isinstance(system, list) and system and system[0].get("text")):
        raise AssertionError("system must be a non-empty list of text blocks")

    messages = kwargs["messages"]
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise AssertionError("expected exactly one user message")

    content = messages[0]["content"]
    images = [b for b in content if b.get("type") == "image"]
    if not images:
        raise AssertionError("no image blocks in request")

    for block in images:
        src = block.get("source", {})
        if src.get("type") != "base64":
            raise AssertionError(f"bad image source type: {src.get('type')}")
        if src.get("media_type") != "image/jpeg":
            raise AssertionError(f"bad media_type: {src.get('media_type')}")
        data = src.get("data")
        if not isinstance(data, str) or not data:
            raise AssertionError("image data missing or not a string")
        try:
            raw = base64.b64decode(data, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"image data is not valid base64: {exc}") from exc
        if not raw.startswith(b"\xff\xd8\xff"):
            raise AssertionError("decoded image is not a JPEG (bad magic bytes)")

    if content[-1].get("type") != "text":
        raise AssertionError("last content block should be the instruction text")

    thinking = kwargs.get("thinking", {})
    effort = kwargs.get("output_config", {}).get("effort")
    # claude-opus-5 rejects disabled thinking above `high` effort.
    if thinking.get("type") == "disabled" and effort in ("xhigh", "max"):
        raise AssertionError(f"disabled thinking is invalid at effort={effort}")
