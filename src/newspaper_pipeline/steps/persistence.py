from __future__ import annotations

from typing import Protocol

from ..models import ImageRecord, LayoutRegion, LegibilityResult, OcrResult, PersistedRecord


class PersistenceSink(Protocol):
    """Persistence stage for serialized outputs."""

    def persist(
        self,
        image: ImageRecord,
        regions: list[LayoutRegion],
        legibility: list[LegibilityResult],
        ocr: list[OcrResult],
    ) -> PersistedRecord:
        """Persist pipeline outputs for one image."""


class NotImplementedPersistenceSink:
    def persist(
        self,
        image: ImageRecord,
        regions: list[LayoutRegion],
        legibility: list[LegibilityResult],
        ocr: list[OcrResult],
    ) -> PersistedRecord:
        raise NotImplementedError("Provide a PersistenceSink implementation.")
