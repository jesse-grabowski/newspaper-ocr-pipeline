from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from ..models import ImageRecord, LayoutRegion, OcrResult, PersistedRecord


class PersistenceSink(Protocol):
    """Persistence stage for serialized outputs."""

    def persist(
        self,
        image: ImageRecord,
        regions: list[LayoutRegion],
        ocr: list[OcrResult],
    ) -> PersistedRecord:
        """Persist pipeline outputs for one image."""


class NotImplementedPersistenceSink:
    def persist(
        self,
        image: ImageRecord,
        regions: list[LayoutRegion],
        ocr: list[OcrResult],
    ) -> PersistedRecord:
        raise NotImplementedError("Provide a PersistenceSink implementation.")


class DellJsonPersistenceSink:
    """
    Persist results in the AmericanStories/Dell JSON-like shape.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.output_dir = output_dir or (repo_root / "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def persist(
        self,
        image: ImageRecord,
        regions: list[LayoutRegion],
        ocr: list[OcrResult],
    ) -> PersistedRecord:
        text_by_region = self._aggregate_text_by_region_id(ocr)
        bboxes = []
        for idx, region in enumerate(regions):
            x0, y0, x1, y1 = region.bbox
            text = text_by_region.get(region.region_id, "")
            bbox_data = {
                "id": idx,
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "class": region.label,
                "raw_text": text,
                # Kept for shape compatibility with existing AmericanStories outputs.
                "legibility": "NA",
            }
            bboxes.append(bbox_data)

        payload = {
            "page_number": str(image.metadata.page_number) if image.metadata and image.metadata.page_number is not None else "na",
            "scan_url": image.metadata.scan_url if image.metadata and image.metadata.scan_url else image.source,
            "scan_ocr": "na",
            "scan": {},
            "bboxes": bboxes,
        }

        file_stem = image.image_id or (image.local_path.stem if image.local_path else Path(image.source).stem)
        output_path = self.output_dir / f"{file_stem}.json"
        with output_path.open("w") as outfile:
            json.dump(payload, outfile, indent=2)

        return PersistedRecord(
            image_id=image.image_id,
            destination=str(output_path),
            metadata={"format": "dell_json"},
        )

    def _aggregate_text_by_region_id(self, ocr: list[OcrResult]) -> dict[str, str]:
        """
        Merge OCR texts back to parent layout region id.

        Line-sliced region ids follow `<region_id>/line_<n>`.
        """
        bucket: dict[str, list[tuple[int, str]]] = {}
        for item in ocr:
            region_key, line_idx = self._split_region_and_line(item.region_id)
            text = (item.text or "").strip()
            if not text:
                continue
            bucket.setdefault(region_key, []).append((line_idx, text))

        merged: dict[str, str] = {}
        for region_key, values in bucket.items():
            values.sort(key=lambda x: x[0])
            merged[region_key] = "\n".join(text for _, text in values)
        return merged

    def _split_region_and_line(self, region_id: str) -> tuple[str, int]:
        marker = "/line_"
        if marker not in region_id:
            return region_id, 0

        base, suffix = region_id.rsplit(marker, 1)
        try:
            return base, int(suffix)
        except ValueError:
            return base, 0
