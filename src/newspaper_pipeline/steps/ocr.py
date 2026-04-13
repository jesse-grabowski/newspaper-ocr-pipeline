from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models import ImageRecord, LayoutRegion, OcrResult


class OcrEngine(Protocol):
    """Pluggable OCR stage (may include line detection internally)."""

    def recognize(self, image: ImageRecord, regions: list[LayoutRegion]) -> list[OcrResult]:
        """Return OCR results for selected layout regions."""


class NotImplementedOcrEngine:
    def recognize(self, image: ImageRecord, regions: list[LayoutRegion]) -> list[OcrResult]:
        raise NotImplementedError("Provide an OcrEngine implementation.")


class PaddleOcrEngine:
    """
    OCR with PaddleOCR over layout regions.

    Runs OCR per region crop and returns one `OcrResult` per region.
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
    ) -> None:
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self._ocr = None

    def recognize(self, image: ImageRecord, regions: list[LayoutRegion]) -> list[OcrResult]:
        if not regions:
            return []

        full_image = self._load_image(image)
        ocr = self._get_ocr()
        height, width = full_image.shape[:2]

        outputs: list[OcrResult] = []
        for region in regions:
            x0, y0, x1, y1 = region.bbox
            x0 = max(0, min(x0, width - 1))
            y0 = max(0, min(y0, height - 1))
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            if x1 <= x0 or y1 <= y0:
                outputs.append(OcrResult(region_id=region.region_id, text="", confidence=None))
                continue

            crop = full_image[y0:y1, x0:x1]
            if crop.size == 0:
                outputs.append(OcrResult(region_id=region.region_id, text="", confidence=None))
                continue

            prediction = ocr.ocr(crop, cls=self.use_angle_cls)
            text, confidence = self._parse_prediction(prediction)
            outputs.append(
                OcrResult(
                    region_id=region.region_id,
                    text=text,
                    confidence=confidence,
                )
            )
        return outputs

    def _get_ocr(self):
        if self._ocr is not None:
            return self._ocr
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PaddleOcrEngine requires `paddleocr`. Install it in your environment first."
            ) from exc
        self._ocr = PaddleOCR(
            use_angle_cls=self.use_angle_cls,
            lang=self.lang,
        )
        return self._ocr

    @staticmethod
    def _load_image(image: ImageRecord):
        path = image.local_path or Path(image.source)
        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {path}")

        try:
            import cv2
        except ImportError as exc:
            raise ImportError("PaddleOcrEngine requires `opencv-python` (cv2).") from exc

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        return img

    @staticmethod
    def _parse_prediction(prediction) -> tuple[str, float | None]:
        """
        Convert PaddleOCR output to (text, confidence).

        Typical shape: [ [ [box], (text, score) ], ... ]
        """
        lines = prediction[0] if prediction and len(prediction) > 0 else []
        texts: list[str] = []
        scores: list[float] = []

        for line in lines or []:
            if not line or len(line) < 2:
                continue
            content = line[1]
            if not content or len(content) < 2:
                continue
            text = str(content[0])
            score = float(content[1])
            if text.strip():
                texts.append(text.strip())
                scores.append(score)

        joined = "\n".join(texts)
        avg_score = (sum(scores) / len(scores)) if scores else None
        return joined, avg_score
