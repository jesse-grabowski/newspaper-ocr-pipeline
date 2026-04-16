from __future__ import annotations

import base64
import json
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..models import ImageRecord, OcrResult, PreprocessedRegion


class OcrEngine(Protocol):
    """Pluggable OCR stage operating on preprocessed line/region crops."""

    def recognize(self, image: ImageRecord, regions: list[PreprocessedRegion]) -> list[OcrResult]:
        """Return OCR results for selected layout regions."""


class NotImplementedOcrEngine:
    def recognize(self, image: ImageRecord, regions: list[PreprocessedRegion]) -> list[OcrResult]:
        raise NotImplementedError("Provide an OcrEngine implementation.")


class NoopOcrEngine:
    """No-op OCR stage for fast pipeline/debug runs."""

    def recognize(self, image: ImageRecord, regions: list[PreprocessedRegion]) -> list[OcrResult]:
        outputs: list[OcrResult] = []
        for region in regions:
            outputs.append(
                OcrResult(
                    region_id=region.region_id,
                    text="",
                    confidence=None,
                )
            )
        return outputs


class OllamaVisionOcrEngine:
    """
    OCR with a local Ollama-hosted vision model.

    Sends each preprocessed region crop as a PNG image to `/api/chat`.
    """

    DEFAULT_PROMPT = (
        "OCR newspaper text. Output text only. "
        "Keep paragraph breaks only (no line/column wraps). "
        "Merge drop caps into first word. "
        "Fix all OCR errors if context is clear; keep wording/meaning, punctuation, caps, and names faithful."
    )

    def __init__(
        self,
        model: str = "gemma4:e4b",
        host: str = "http://localhost:11434",
        prompt: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: int = 120,
        keep_alive: str = "30m",
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive

    def recognize(self, image: ImageRecord, regions: list[PreprocessedRegion]) -> list[OcrResult]:
        if not regions:
            return []

        outputs: list[OcrResult] = []
        for region in regions:
            if region.image_bgr is None or region.image_bgr.size == 0:
                outputs.append(OcrResult(region_id=region.region_id, text="", confidence=None))
                continue

            try:
                png_bytes = self._encode_crop_to_png_bytes(region.image_bgr)
                crop_b64 = base64.b64encode(png_bytes).decode("ascii")
                text = self._chat_ocr(crop_b64)
            except Exception as exc:
                page_id = image.metadata.page_id if image.metadata else image.image_id
                raise RuntimeError(
                    f"OCR failed for page={page_id}, region={region.region_id}: {exc}"
                ) from exc

            outputs.append(
                OcrResult(
                    region_id=region.region_id,
                    text=text.strip(),
                    confidence=None,
                )
            )
        return outputs

    def _encode_crop_to_png_bytes(self, crop_bgr) -> bytes:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("OllamaVisionOcrEngine requires `opencv-python`.") from exc

        ok, encoded = cv2.imencode(".png", crop_bgr)
        if not ok:
            raise ValueError("Failed to encode OCR crop to PNG.")
        return encoded.tobytes()

    def _chat_ocr(self, image_b64: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": self.prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
            },
        }
        req = Request(
            url=f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("Ollama returned non-JSON response.") from exc

        message = parsed.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            return ""
        return content
