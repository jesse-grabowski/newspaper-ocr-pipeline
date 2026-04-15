from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from newspaper_pipeline.pipeline import ExtractionPipeline
from newspaper_pipeline.steps.filtering import DellLegibilityRegionFilter
from newspaper_pipeline.steps.image_fetch import LocalDirectoryImageFetcher
from newspaper_pipeline.steps.layout_detection import DellOnnxLayoutDetector
from newspaper_pipeline.steps.ocr import OllamaVisionOcrEngine
from newspaper_pipeline.steps.preprocessing import NoopPreprocessor
from newspaper_pipeline.steps.persistence import DellJsonPersistenceSink


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
            debug_output_dir=None,
        ),
        region_filter=DellLegibilityRegionFilter(
            model_path=os.environ.get(
                "DELL_LEGIBILITY_MODEL",
                str(repo_root / "weights" / "legibility_model_new.onnx"),
            ),
            drop_labels={"illegible"},
        ),
        preprocessor=NoopPreprocessor(
            save_debug_crops=False,
            debug_output_dir=None,
        ),
        ocr_engine=OllamaVisionOcrEngine(
            model=os.environ.get("OLLAMA_MODEL", "gemma4:e4b"),
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            temperature=0.0,
            timeout_seconds=120,
            keep_alive="30m",
        ),
        persistence_sink=DellJsonPersistenceSink(
            output_dir=repo_root / "outputs"
        ),
    )


def main(source: str) -> None:
    pipeline = build_pipeline()
    for _ in pipeline.run(source):
        pass


if __name__ == "__main__":
    main("local")
