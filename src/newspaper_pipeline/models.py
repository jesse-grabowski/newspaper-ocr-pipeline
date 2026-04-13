from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageMetadata:
    """Strict metadata schema for one page image."""

    fetcher: str
    source_id: str
    issue_id: str
    publication_id: str | None
    issue_version: str | None
    lccn: str | None
    issue_date: str | None  # ISO-8601 date when available.
    page_number: int | None
    page_id: str
    scan_filename: str
    scan_url: str | None
    metadata_json_path: str | None
    mets_path: str | None
    article_id_prefix: str


@dataclass(frozen=True)
class ImageRecord:
    """Input page image and normalized metadata."""

    image_id: str
    source: str
    local_path: Path | None = None
    metadata: ImageMetadata | None = None


@dataclass(frozen=True)
class LayoutRegion:
    """Detected region on a page."""

    region_id: str
    label: str
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1)
    score: float | None = None


@dataclass(frozen=True)
class OcrResult:
    """Extracted text for a region."""

    region_id: str
    text: str
    confidence: float | None = None


@dataclass(frozen=True)
class PersistedRecord:
    """Persistence-layer acknowledgement."""

    image_id: str
    destination: str
    metadata: dict[str, Any] = field(default_factory=dict)
