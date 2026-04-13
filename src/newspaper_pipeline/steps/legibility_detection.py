from __future__ import annotations

from typing import Protocol

from ..models import ImageRecord, LayoutRegion, LegibilityResult


class LegibilityDetector(Protocol):
    """Legibility classification stage."""

    def classify(self, image: ImageRecord, regions: list[LayoutRegion]) -> list[LegibilityResult]:
        """Return legibility labels/scores for regions."""


class PassthroughLegibilityDetector:
    """Default stub that marks all regions as legible."""

    def classify(self, image: ImageRecord, regions: list[LayoutRegion]) -> list[LegibilityResult]:
        return [LegibilityResult(region_id=r.region_id, label="Legible", score=1.0) for r in regions]
