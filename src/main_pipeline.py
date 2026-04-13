from __future__ import annotations

from newspaper_pipeline.pipeline import ExtractionPipeline
from newspaper_pipeline.steps.image_fetch import LocalDirectoryImageFetcher


def build_pipeline() -> ExtractionPipeline:
    """Create pipeline with interface stubs. Swap in concrete implementations."""
    return ExtractionPipeline(image_fetcher=LocalDirectoryImageFetcher())


def main(source: str) -> None:
    pipeline = build_pipeline()
    for _ in pipeline.run(source):
        pass


if __name__ == "__main__":
    raise SystemExit(
        "Wire concrete implementations into build_pipeline(), then call main(source)."
    )
