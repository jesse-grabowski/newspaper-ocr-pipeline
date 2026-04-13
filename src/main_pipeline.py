from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from newspaper_pipeline.pipeline import ExtractionPipeline
from newspaper_pipeline.steps.image_fetch import LocalDirectoryImageFetcher
from newspaper_pipeline.steps.layout_detection import DocLayoutYoloLayoutDetector
from newspaper_pipeline.steps.ocr import PaddleOcrEngine
from newspaper_pipeline.steps.persistence import MelissaJsonPersistenceSink

DEFAULT_SOURCE = "local"


def build_pipeline() -> ExtractionPipeline:
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")

    return ExtractionPipeline(
        image_fetcher=LocalDirectoryImageFetcher(),
        layout_detector=DocLayoutYoloLayoutDetector(
            model_source=os.environ.get("DOCLAYOUT_YOLO_MODEL", ""),
            keep_labels={"article"},
        ),
        ocr_engine=PaddleOcrEngine(
            lang=os.environ.get("PADDLE_OCR_LANG", "en"),
            use_angle_cls=os.environ.get("PADDLE_OCR_USE_ANGLE_CLS", "true").lower() == "true",
        ),
        persistence_sink=MelissaJsonPersistenceSink(
            output_dir=repo_root / os.environ.get("OUTPUT_DIR", "outputs")
        ),
    )


def main(source: str) -> None:
    pipeline = build_pipeline()
    for _ in pipeline.run(source):
        pass


if __name__ == "__main__":
    main(DEFAULT_SOURCE)
