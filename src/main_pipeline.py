from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from newspaper_pipeline.pipeline import ExtractionPipeline
from newspaper_pipeline.steps.image_fetch import LocalDirectoryImageFetcher
from newspaper_pipeline.steps.layout_detection import DellOnnxLayoutDetector
from newspaper_pipeline.steps.ocr import PaddleOcrEngine
from newspaper_pipeline.steps.preprocessing import NoopPreprocessor
from newspaper_pipeline.steps.persistence import MelissaJsonPersistenceSink


def build_pipeline() -> ExtractionPipeline:
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")

    return ExtractionPipeline(
        image_fetcher=LocalDirectoryImageFetcher(),
        layout_detector=DellOnnxLayoutDetector(
            model_path=os.environ.get(
                "DELL_LAYOUT_MODEL",
                str(repo_root / "weights" / "layout_model_new.onnx"),
            ),
            conf_threshold=0.05,
            iou_threshold=0.10,
            keep_labels={"article"},
            debug_output_dir=repo_root / "outputs" / "layout_debug",
        ),
        preprocessor=NoopPreprocessor(
            save_debug_crops=True,
            debug_output_dir=repo_root / "outputs" / "preprocess_debug",
        ),
        ocr_engine=PaddleOcrEngine(
            lang="en",
            use_angle_cls=True,
        ),
        persistence_sink=MelissaJsonPersistenceSink(
            output_dir=repo_root / "outputs"
        ),
    )


def main(source: str) -> None:
    pipeline = build_pipeline()
    for _ in pipeline.run(source):
        pass


if __name__ == "__main__":
    main("local")
