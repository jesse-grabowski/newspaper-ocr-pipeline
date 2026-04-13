from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import traceback

from .models import PersistedRecord
from .steps.image_fetch import ImageFetcher, NotImplementedImageFetcher
from .steps.layout_detection import LayoutDetector, NotImplementedLayoutDetector
from .steps.ocr import OcrEngine, NotImplementedOcrEngine
from .steps.persistence import PersistenceSink, NotImplementedPersistenceSink


@dataclass
class ExtractionPipeline:
    """Main orchestrator for newspaper text extraction."""

    image_fetcher: ImageFetcher = NotImplementedImageFetcher()
    layout_detector: LayoutDetector = NotImplementedLayoutDetector()
    ocr_engine: OcrEngine = NotImplementedOcrEngine()
    persistence_sink: PersistenceSink = NotImplementedPersistenceSink()
    continue_on_error: bool = True

    def run(self, source: str) -> Iterator[PersistedRecord]:
        """Stream persisted records: fetch -> layout -> OCR -> persistence."""
        for image in self.image_fetcher.fetch(source):
            try:
                regions = self.layout_detector.detect(image)
                ocr_results = self.ocr_engine.recognize(image, regions)
                persisted = self.persistence_sink.persist(image, regions, ocr_results)
                self.image_fetcher.report_status(
                    image=image,
                    status="success",
                    message="Image processed and persisted.",
                    extra={"stage": "persistence"},
                )
                yield persisted
            except Exception as exc:
                self.image_fetcher.report_status(
                    image=image,
                    status="failed",
                    message=str(exc),
                    extra={"traceback": traceback.format_exc()},
                )
                if not self.continue_on_error:
                    raise
