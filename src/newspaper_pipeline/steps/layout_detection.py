from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models import ImageRecord, LayoutRegion


class LayoutDetector(Protocol):
    """Pluggable layout detection stage."""

    def detect(self, image: ImageRecord) -> list[LayoutRegion]:
        """Return layout regions for a single image."""


class NotImplementedLayoutDetector:
    def detect(self, image: ImageRecord) -> list[LayoutRegion]:
        raise NotImplementedError("Provide a LayoutDetector implementation.")


class DocLayoutYoloLayoutDetector:
    """
    Layout detection using DocLayout-YOLO (Ultralytics-compatible weights).

    Example model_source values:
    - local `.pt` path
    - an Ultralytics/HuggingFace model id if supported by your setup
    """

    DEFAULT_CLASS_MAP = {
        "Text": "article",
        "Title": "headline",
        "Section-header": "headline",
        "Caption": "image_caption",
        "Picture": "photograph",
        "Table": "table",
        "Footnote": "footnote",
        "Formula": "formula",
        "List-item": "list_item",
        "Page-header": "page_header",
        "Page-footer": "page_footer",
    }

    def __init__(
        self,
        model_source: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_map: dict[str, str] | None = None,
        keep_labels: set[str] | None = None,
    ) -> None:
        if not model_source:
            raise ValueError("model_source is required for DocLayoutYoloLayoutDetector")

        self.model_source = model_source
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_map = class_map or self.DEFAULT_CLASS_MAP
        self.keep_labels = keep_labels
        self._model = None

    def detect(self, image: ImageRecord) -> list[LayoutRegion]:
        image_path = self._resolve_image_path(image)
        model = self._get_model()

        predictions = model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        if not predictions:
            return []

        result = predictions[0]
        if result.boxes is None:
            return []

        names = getattr(result, "names", {}) or {}
        regions: list[LayoutRegion] = []
        for idx, box in enumerate(result.boxes):
            cls_idx = int(box.cls.item())
            raw_label = names.get(cls_idx, str(cls_idx))
            label = self.class_map.get(raw_label, raw_label)
            if self.keep_labels is not None and label not in self.keep_labels:
                continue

            x0, y0, x1, y1 = [int(round(v)) for v in box.xyxy[0].tolist()]
            score = float(box.conf.item()) if box.conf is not None else None
            regions.append(
                LayoutRegion(
                    region_id=f"{image.image_id}_region_{idx}",
                    label=label,
                    bbox=(x0, y0, x1, y1),
                    score=score,
                )
            )

        return regions

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "DocLayoutYoloLayoutDetector requires `ultralytics`. "
                "Install it in your environment before running layout detection."
            ) from exc
        self._model = YOLO(self.model_source)
        return self._model

    @staticmethod
    def _resolve_image_path(image: ImageRecord) -> Path:
        candidate = image.local_path or Path(image.source)
        if not candidate.exists():
            raise FileNotFoundError(f"Image path does not exist: {candidate}")
        return candidate
