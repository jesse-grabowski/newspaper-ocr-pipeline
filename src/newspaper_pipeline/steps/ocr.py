from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models import ImageRecord, OcrResult, PreprocessedRegion


class OcrEngine(Protocol):
    """Pluggable OCR stage (may include line detection internally)."""

    def recognize(self, image: ImageRecord, regions: list[PreprocessedRegion]) -> list[OcrResult]:
        """Return OCR results for selected layout regions."""


class NotImplementedOcrEngine:
    def recognize(self, image: ImageRecord, regions: list[PreprocessedRegion]) -> list[OcrResult]:
        raise NotImplementedError("Provide an OcrEngine implementation.")


class NoopOcrEngine:
    """No-op OCR stage for fast pipeline/debug runs."""

    def recognize(self, image: ImageRecord, regions: list[PreprocessedRegion]) -> list[OcrResult]:
        outputs: list[OcrResult] = []
        for region in regions:
            outputs.append(
                OcrResult(
                    region_id=region.region_id,
                    text="",
                    confidence=None,
                )
            )
        return outputs


class KrakenOcrEngine:
    """
    OCR with Kraken using a line recognition model (`.mlmodel`).
    """

    def __init__(
        self,
        model_path: str,
        binarize: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.binarize = binarize
        self._rec_model = None

    def recognize(self, image: ImageRecord, regions: list[PreprocessedRegion]) -> list[OcrResult]:
        if not regions:
            return []

        rec_model = self._get_model()

        outputs: list[OcrResult] = []
        for region in regions:
            if region.image_bgr is None or region.image_bgr.size == 0:
                outputs.append(OcrResult(region_id=region.region_id, text="", confidence=None))
                continue

            try:
                text, confidence = self._recognize_crop_with_kraken(region.image_bgr, rec_model)
            except Exception:
                text, confidence = "", None

            outputs.append(
                OcrResult(
                    region_id=region.region_id,
                    text=text,
                    confidence=confidence,
                )
            )
        return outputs

    def _get_model(self):
        if self._rec_model is not None:
            return self._rec_model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Kraken OCR model not found: {self.model_path}")
        try:
            from kraken.lib import models
        except ImportError as exc:
            raise ImportError("KrakenOcrEngine requires `kraken`.") from exc
        self._rec_model = models.load_any(str(self.model_path))
        return self._rec_model

    def _recognize_crop_with_kraken(self, crop_bgr, rec_model) -> tuple[str, float | None]:
        import cv2
        from PIL import Image
        from kraken import pageseg, rpred
        from kraken.binarization import nlbin

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        pil_im = Image.fromarray(gray)
        proc_im = nlbin(pil_im) if self.binarize else pil_im

        seg = pageseg.segment(proc_im)
        records = list(rpred.rpred(rec_model, proc_im, seg))

        texts: list[str] = []
        confs: list[float] = []
        for rec in records:
            pred = getattr(rec, "prediction", "")
            if pred and str(pred).strip():
                texts.append(str(pred).strip())
            for c in getattr(rec, "confidences", []) or []:
                try:
                    confs.append(float(c))
                except (TypeError, ValueError):
                    pass

        text = "\n".join(texts)
        confidence = (sum(confs) / len(confs)) if confs else None
        return text, confidence
