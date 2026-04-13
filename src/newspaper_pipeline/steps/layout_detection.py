from __future__ import annotations

from typing import Protocol

from ..models import ImageRecord, LayoutRegion


class LayoutDetector(Protocol):
    """Pluggable layout detection stage."""

    def detect(self, image: ImageRecord) -> list[LayoutRegion]:
        """Return layout regions for a single image."""


class NotImplementedLayoutDetector:
    def detect(self, image: ImageRecord) -> list[LayoutRegion]:
        raise NotImplementedError("Provide a LayoutDetector implementation.")
