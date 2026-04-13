from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models import ImageRecord, LayoutRegion, PreprocessedRegion


class Preprocessor(Protocol):
    """Preprocess layout regions before OCR."""

    def preprocess(self, image: ImageRecord, regions: list[LayoutRegion]) -> list[PreprocessedRegion]:
        """Return region crops for OCR."""


class NoopPreprocessor:
    """
    Default preprocessing stage.

    Performs no image transformation; only crops regions.
    Optionally saves crops for debugging.
    """

    def __init__(
        self,
        save_debug_crops: bool = False,
        debug_output_dir: Path | None = None,
    ) -> None:
        self.save_debug_crops = save_debug_crops
        self.debug_output_dir = debug_output_dir
        if self.save_debug_crops and self.debug_output_dir is not None:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(self, image: ImageRecord, regions: list[LayoutRegion]) -> list[PreprocessedRegion]:
        if image.image_bgr is None:
            raise ValueError("NoopPreprocessor expects in-memory image data at image.image_bgr.")

        full = image.image_bgr
        h, w = full.shape[:2]
        out: list[PreprocessedRegion] = []

        for idx, region in enumerate(regions):
            x0, y0, x1, y1 = region.bbox
            x0 = max(0, min(x0, w - 1))
            y0 = max(0, min(y0, h - 1))
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            if x1 <= x0 or y1 <= y0:
                continue

            crop = full[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            rec = PreprocessedRegion(
                region_id=region.region_id,
                label=region.label,
                bbox=(x0, y0, x1, y1),
                image_bgr=crop,
                score=region.score,
            )
            out.append(rec)

            if self.save_debug_crops and self.debug_output_dir is not None:
                self._save_debug_crop(image.image_id, idx, rec)

        return out

    def _save_debug_crop(self, image_id: str, idx: int, region: PreprocessedRegion) -> None:
        try:
            import cv2
        except ImportError:
            return

        safe_region_id = region.region_id.replace('/', '_')
        out_path = self.debug_output_dir / f"{image_id}_{idx:04d}_{safe_region_id}.jpg"
        cv2.imwrite(str(out_path), region.image_bgr)
