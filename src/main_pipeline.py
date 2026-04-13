from __future__ import annotations

import os

from newspaper_pipeline.pipeline import ExtractionPipeline
from newspaper_pipeline.steps.image_fetch import LocalDirectoryImageFetcher
from newspaper_pipeline.steps.layout_detection import DocLayoutYoloLayoutDetector


def build_pipeline() -> ExtractionPipeline:
    """Create pipeline with interface stubs. Swap in concrete implementations."""
    model_source = os.environ.get("DOCLAYOUT_YOLO_MODEL", "")
    if not model_source:
        return ExtractionPipeline(image_fetcher=LocalDirectoryImageFetcher())
    return ExtractionPipeline(
        image_fetcher=LocalDirectoryImageFetcher(),
        layout_detector=DocLayoutYoloLayoutDetector(model_source=model_source),
    )


def main(source: str) -> None:
    pipeline = build_pipeline()
    for _ in pipeline.run(source):
        pass


if __name__ == "__main__":
    raise SystemExit(
        "Wire concrete implementations into build_pipeline(), then call main(source)."
    )
