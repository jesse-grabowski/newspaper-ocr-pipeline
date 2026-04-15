from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models import ImageRecord, LayoutRegion


class RegionFilter(Protocol):
    """Filter detected layout regions before preprocessing/OCR."""

    def filter(self, image: ImageRecord, regions: list[LayoutRegion]) -> list[LayoutRegion]:
        """Return only regions that should continue through the pipeline."""


class NotImplementedRegionFilter:
    def filter(self, image: ImageRecord, regions: list[LayoutRegion]) -> list[LayoutRegion]:
        raise NotImplementedError("Provide a RegionFilter implementation.")


class NoopRegionFilter:
    def filter(self, image: ImageRecord, regions: list[LayoutRegion]) -> list[LayoutRegion]:
        return regions


class DellLegibilityRegionFilter:
    """
    Filter layout regions with Dell's legibility ONNX classifier.

    Regions predicted as "illegible" are dropped.
    """

    DEFAULT_LABEL_MAP = {
        0: "legible",
        1: "questionable",
        2: "illegible",
    }

    def __init__(
        self,
        model_path: str,
        drop_labels: set[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.drop_labels = {self._normalize_label(label) for label in (drop_labels or {"illegible"})}
        self.label_map = dict(self.DEFAULT_LABEL_MAP)
        self._session = None
        self._input_name = None

    def filter(self, image: ImageRecord, regions: list[LayoutRegion]) -> list[LayoutRegion]:
        if not regions:
            return []
        if image.image_bgr is None:
            raise ValueError(
                "DellLegibilityRegionFilter expects in-memory image data at image.image_bgr."
            )

        import cv2
        import numpy as np

        session, input_name = self._get_session()
        full = image.image_bgr
        h, w = full.shape[:2]
        kept: list[LayoutRegion] = []

        for region in regions:
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

            model_input = self._prepare_input(crop, cv2, np)
            preds = session.run(None, {input_name: model_input})
            label = self._decode_label(preds, np)
            if label in self.drop_labels:
                continue

            kept.append(region)

        return kept

    def _get_session(self):
        if self._session is not None and self._input_name is not None:
            return self._session, self._input_name

        if not self.model_path.exists():
            raise FileNotFoundError(f"Dell legibility model not found: {self.model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("DellLegibilityRegionFilter requires `onnxruntime`.") from exc

        self._session = ort.InferenceSession(str(self.model_path))
        self._input_name = self._session.get_inputs()[0].name
        return self._session, self._input_name

    def _prepare_input(self, crop_bgr, cv2, np):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        target = 256

        if h < target or w < target:
            scale = max(target / max(h, 1), target / max(w, 1))
            new_w = max(int(round(w * scale)), target)
            new_h = max(int(round(h * scale)), target)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = rgb.shape[:2]

        y0 = (h - target) // 2
        x0 = (w - target) // 2
        crop = rgb[y0:y0 + target, x0:x0 + target]
        if crop.shape[0] != target or crop.shape[1] != target:
            crop = cv2.resize(crop, (target, target), interpolation=cv2.INTER_LINEAR)

        chw = crop.astype(np.float32) / 255.0
        chw = np.transpose(chw, (2, 0, 1))
        return np.expand_dims(np.ascontiguousarray(chw), axis=0)

    def _decode_label(self, preds, np) -> str:
        arr = np.array(preds[0])
        if arr.ndim == 0:
            return "legible"
        if arr.ndim == 1:
            cls_idx = int(arr.argmax())
        else:
            cls_idx = int(arr[0].argmax())
        return self._normalize_label(self.label_map.get(cls_idx, "legible"))

    @staticmethod
    def _normalize_label(label: str) -> str:
        return str(label).strip().lower().replace("-", "_").replace(" ", "_")
