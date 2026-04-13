from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import ImageRecord


@dataclass(frozen=True)
class ProcessingEvent:
    """
    One durable status event for an image.

    Stored as JSONL to allow append-only writes at scale.
    """

    timestamp_utc: str
    status: str  # "success" | "failed"
    processing_key: str
    image_id: str
    source: str
    issue_id: str | None
    page_id: str | None
    message: str | None = None
    extra: dict[str, Any] | None = None


class ProcessingJournal:
    """
    Append-only processing log.

    Format:
    - One JSON object per line (`.jsonl`)
    - Each line is a `ProcessingEvent`
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def load_success_keys(self) -> set[str]:
        """
        Return keys whose latest observed status is success.

        If a key later receives a failure status, it is removed so it can retry.
        """
        latest_success: set[str] = set()
        with self.path.open() as infile:
            for raw in infile:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                key = row.get("processing_key")
                status = row.get("status")
                if not key or not status:
                    continue
                if status == "success":
                    latest_success.add(key)
                elif status == "failed":
                    latest_success.discard(key)
        return latest_success

    def append(self, event: ProcessingEvent) -> None:
        with self.path.open("a") as outfile:
            outfile.write(json.dumps(asdict(event), ensure_ascii=True))
            outfile.write("\n")


def build_processing_key(image: ImageRecord) -> str:
    """
    Stable key for de-duplication.

    Prefers page-level metadata if present; falls back to source path.
    """
    if image.metadata and image.metadata.page_id:
        return image.metadata.page_id
    return image.source


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
