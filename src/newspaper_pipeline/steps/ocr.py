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

    def recognize(self, image: ImageRecord, regions: list[PreprocessedRegion]) -> list[OcrResult]:
        if not regions:
            return []

        ocr = self._get_ocr()

        outputs: list[OcrResult] = []
        for region in regions:
            if region.image_bgr is None or region.image_bgr.size == 0:
                outputs.append(OcrResult(region_id=region.region_id, text="", confidence=None))
                continue

            prediction = ocr.ocr(region.image_bgr)
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
    def _parse_prediction(prediction) -> tuple[str, float | None]:
        """
        Convert PaddleOCR output to (text, confidence) across PaddleOCR response variants.
        """
        texts: list[str] = []
        scores: list[float] = []

        def add_text_score(text, score=None):
            t = str(text).strip() if text is not None else ""
            if not t:
                return
            texts.append(t)
            try:
                if score is not None:
                    scores.append(float(score))
            except (TypeError, ValueError):
                pass

        # Variant A: dict-style output (PaddleOCR v3+ style)
        if isinstance(prediction, dict):
            rec_texts = prediction.get("rec_texts") or prediction.get("texts") or []
            rec_scores = prediction.get("rec_scores") or prediction.get("scores") or []
            if rec_texts:
                for i, t in enumerate(rec_texts):
                    s = rec_scores[i] if i < len(rec_scores) else None
                    add_text_score(t, s)
                return "\n".join(texts), (sum(scores) / len(scores) if scores else None)

        # Variant B: list of dict items
        if isinstance(prediction, list) and prediction and isinstance(prediction[0], dict):
            for item in prediction:
                if not isinstance(item, dict):
                    continue
                rec_texts = item.get("rec_texts") or item.get("texts") or []
                rec_scores = item.get("rec_scores") or item.get("scores") or []
                if rec_texts:
                    for i, t in enumerate(rec_texts):
                        s = rec_scores[i] if i < len(rec_scores) else None
                        add_text_score(t, s)
                    continue
                add_text_score(item.get("rec_text") or item.get("text"), item.get("rec_score") or item.get("score"))
            if texts:
                return "\n".join(texts), (sum(scores) / len(scores) if scores else None)

        # Variant C: classic format [[ [box], (text, score) ], ...]
        lines = prediction[0] if isinstance(prediction, list) and prediction else []
        for line in lines or []:
            if not line:
                continue
            if isinstance(line, dict):
                add_text_score(line.get("rec_text") or line.get("text"), line.get("rec_score") or line.get("score"))
                continue
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                content = line[1]
                if isinstance(content, (list, tuple)) and len(content) >= 2:
                    add_text_score(content[0], content[1])
                    continue
                if isinstance(content, str):
                    add_text_score(content, None)

        return "\n".join(texts), (sum(scores) / len(scores) if scores else None)


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
