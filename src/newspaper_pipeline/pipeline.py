from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import traceback

from .models import LayoutRegion, PersistedRecord
from .steps.image_fetch import ImageFetcher, NotImplementedImageFetcher
from .steps.layout_detection import LayoutDetector, NotImplementedLayoutDetector
from .steps.legibility_detection import LegibilityDetector, PassthroughLegibilityDetector
from .steps.ocr import OcrEngine, NotImplementedOcrEngine
from .steps.persistence import PersistenceSink, NotImplementedPersistenceSink


@dataclass
class ExtractionPipeline:
    """Main orchestrator for 5-phase newspaper text extraction."""

    image_fetcher: ImageFetcher = NotImplementedImageFetcher()
    layout_detector: LayoutDetector = NotImplementedLayoutDetector()
    legibility_detector: LegibilityDetector = PassthroughLegibilityDetector()
    ocr_engine: OcrEngine = NotImplementedOcrEngine()
    persistence_sink: PersistenceSink = NotImplementedPersistenceSink()
    legible_labels: set[str] | None = None
    continue_on_error: bool = True

    def run(self, source: str) -> Iterator[PersistedRecord]:
        """Stream persisted records: fetch -> layout -> legibility -> OCR -> persistence."""
        for image in self.image_fetcher.fetch(source):
            try:
                regions = self.layout_detector.detect(image)
                legibility = self.legibility_detector.classify(image, regions)
                selected_regions = self._filter_regions(regions, legibility)
                ocr_results = self.ocr_engine.recognize(image, selected_regions)
                persisted = self.persistence_sink.persist(image, regions, legibility, ocr_results)
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

    def _filter_regions(self, regions: list[LayoutRegion], legibility: list) -> list[LayoutRegion]:
        """Select regions eligible for OCR based on legibility labels."""
        allowed = self.legible_labels or {"Legible", "Questionable"}
        legible_region_ids = {item.region_id for item in legibility if item.label in allowed}
        return [region for region in regions if region.region_id in legible_region_ids]
