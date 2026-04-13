from __future__ import annotations

from typing import Protocol

from ..models import ImageRecord, LayoutRegion, OcrResult


class OcrEngine(Protocol):
    """Pluggable OCR stage (may include line detection internally)."""

    def recognize(self, image: ImageRecord, regions: list[LayoutRegion]) -> list[OcrResult]:
        """Return OCR results for selected layout regions."""


class NotImplementedOcrEngine:
    def recognize(self, image: ImageRecord, regions: list[LayoutRegion]) -> list[OcrResult]:
        raise NotImplementedError("Provide an OcrEngine implementation.")
