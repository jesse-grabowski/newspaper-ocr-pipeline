from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any, Protocol
import xml.etree.ElementTree as ET

from ..models import ImageMetadata, ImageRecord
from ..state import ProcessingEvent, ProcessingJournal, build_processing_key, utc_now_iso


class ImageFetcher(Protocol):
    """Pluggable image ingestion/fetching stage."""

    def fetch(self, source: str) -> Iterator[ImageRecord]:
        """Stream image records from a source descriptor."""

    def report_status(
        self,
        image: ImageRecord,
        status: str,
        message: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Receive pipeline processing status for an image."""


class NotImplementedImageFetcher:
    def fetch(self, source: str) -> Iterator[ImageRecord]:
        raise NotImplementedError("Provide an ImageFetcher implementation.")

    def report_status(
        self,
        image: ImageRecord,
        status: str,
        message: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        _ = (image, status, message, extra)


class LocalDirectoryImageFetcher:
    """
    Fetch images from a local directory.

    Default directory is a sibling of `src`: `<repo>/input_images`.
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".jp2",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    }

    def __init__(self, image_dir: Path | None = None) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.image_dir = image_dir or (repo_root / "input_images")
        self._dir_metadata_cache: dict[Path, dict[str, Any]] = {}
        state_dir = repo_root / ".pipeline_state"
        self._journal = ProcessingJournal(state_dir / "image_fetch_status.jsonl")
        self._success_keys = self._journal.load_success_keys()

    def fetch(self, source: str) -> Iterator[ImageRecord]:
        _ = source  # Reserved for future source-routing.

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Local image directory not found: {self.image_dir}"
            )
        if not self.image_dir.is_dir():
            raise NotADirectoryError(
                f"Configured image path is not a directory: {self.image_dir}"
            )

        image_paths = sorted(
            path for path in self.image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        for path in image_paths:
            folder_meta = self._load_folder_metadata(path.parent)
            image_meta = self._build_image_metadata(path, folder_meta)
            image_bgr = self._load_image_array(path)
            image_record = ImageRecord(
                image_id=image_meta.article_id_prefix,
                source=str(path),
                local_path=path,
                image_bgr=image_bgr,
                metadata=image_meta,
            )
            processing_key = build_processing_key(image_record)
            if processing_key in self._success_keys:
                continue
            yield image_record

    def report_status(
        self,
        image: ImageRecord,
        status: str,
        message: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if status not in {"success", "failed"}:
            raise ValueError(f"Unsupported status: {status}")

        key = build_processing_key(image)
        if status == "success":
            self._success_keys.add(key)
        else:
            self._success_keys.discard(key)

        page_id = image.metadata.page_id if image.metadata else None
        issue_id = image.metadata.issue_id if image.metadata else None
        event = ProcessingEvent(
            timestamp_utc=utc_now_iso(),
            status=status,
            processing_key=key,
            image_id=image.image_id,
            source=image.source,
            issue_id=issue_id,
            page_id=page_id,
            message=message,
            extra=extra,
        )
        self._journal.append(event)

    def _load_folder_metadata(self, folder: Path) -> dict[str, Any]:
        cached = self._dir_metadata_cache.get(folder)
        if cached is not None:
            return cached

        metadata_json_path = folder / "metadata.json"
        mets_path = next(folder.glob("*.mets.xml"), None)

        metadata_json: dict[str, Any] = {}
        if metadata_json_path.exists():
            with metadata_json_path.open() as f:
                metadata_json = json.load(f)

        mets_data = self._parse_mets(mets_path) if mets_path else {}

        scan_url_by_filename = {
            item.get("filename"): item.get("url")
            for item in metadata_json.get("jp2_images", [])
            if item.get("filename")
        }

        merged = {
            "metadata_json_path": str(metadata_json_path) if metadata_json_path.exists() else None,
            "mets_path": str(mets_path) if mets_path else None,
            "metadata_json": metadata_json,
            "mets_data": mets_data,
            "scan_url_by_filename": scan_url_by_filename,
        }
        self._dir_metadata_cache[folder] = merged
        return merged

    def _build_image_metadata(self, image_path: Path, folder_meta: dict[str, Any]) -> ImageMetadata:
        metadata_json = folder_meta.get("metadata_json", {})
        mets_data = folder_meta.get("mets_data", {})

        source_id = (
            mets_data.get("source_id")
            or metadata_json.get("metadata_file", "").replace(".mets.xml", "")
            or f"{image_path.parent.name}_{image_path.stem}"
        )
        publication_id = mets_data.get("publication_id")
        issue_version = mets_data.get("issue_version") or metadata_json.get("id")
        issue_id = (
            mets_data.get("issue_id")
            or "_".join([part for part in [publication_id, issue_version] if part])
            or source_id
        )

        page_number = (
            mets_data.get("page_by_filename", {}).get(image_path.name)
            or self._infer_numeric_page_number(image_path.stem)
        )
        page_id = f"{issue_id}_p{page_number}" if page_number is not None else issue_id

        scan_url = folder_meta.get("scan_url_by_filename", {}).get(image_path.name)

        return ImageMetadata(
            fetcher="local_directory",
            source_id=source_id,
            issue_id=issue_id,
            publication_id=publication_id,
            issue_version=issue_version,
            lccn=mets_data.get("lccn"),
            issue_date=mets_data.get("issue_date"),
            page_number=page_number,
            page_id=page_id,
            scan_filename=image_path.name,
            scan_url=scan_url,
            metadata_json_path=folder_meta.get("metadata_json_path"),
            mets_path=folder_meta.get("mets_path"),
            article_id_prefix=page_id,
        )

    def _parse_mets(self, mets_path: Path) -> dict[str, Any]:
        tree = ET.parse(mets_path)
        root = tree.getroot()

        source_id = root.attrib.get("OBJID")
        lccn = None
        publication_id = None
        issue_version = None
        issue_date = None

        file_id_to_name: dict[str, str] = {}
        page_by_filename: dict[str, int] = {}

        for elem in root.iter():
            tag = self._local_name(elem.tag)

            if tag == "identifier" and elem.attrib.get("type") == "LCCN" and elem.text and lccn is None:
                lccn = elem.text.strip()
            elif tag == "recordIdentifier" and not source_id and elem.text:
                source_id = elem.text.strip()
            elif tag == "BibID" and elem.text:
                publication_id = elem.text.strip()
            elif tag == "VID" and elem.text:
                issue_version = elem.text.strip()
            elif tag == "File":
                fileid = elem.attrib.get("fileid")
                if fileid and "width" in elem.attrib:
                    # width/height exists, but mapping is done from fileSec below.
                    pass
            elif tag == "file":
                file_id = elem.attrib.get("ID")
                href = self._extract_href_from_file_element(elem)
                if file_id and href:
                    file_id_to_name[file_id] = Path(href).name

        issue_date = self._extract_issue_date(root)

        physical_struct_map = self._find_physical_struct_map(root)
        if physical_struct_map is not None:
            for div in physical_struct_map.iter():
                if self._local_name(div.tag) != "div":
                    continue
                if div.attrib.get("TYPE") != "Page":
                    continue

                order = div.attrib.get("ORDER")
                if not order or not order.isdigit():
                    continue
                page_num = int(order)

                for child in div:
                    if self._local_name(child.tag) != "fptr":
                        continue
                    file_id = child.attrib.get("FILEID")
                    if not file_id:
                        continue
                    filename = file_id_to_name.get(file_id)
                    if filename:
                        page_by_filename[filename] = page_num

        issue_id = "_".join([part for part in [publication_id, issue_version] if part]) or source_id

        return {
            "source_id": source_id,
            "publication_id": publication_id,
            "issue_version": issue_version,
            "issue_id": issue_id,
            "lccn": lccn,
            "issue_date": issue_date,
            "page_by_filename": page_by_filename,
        }

    @staticmethod
    def _local_name(tag: str) -> str:
        return tag.split("}", 1)[-1]

    @staticmethod
    def _extract_href_from_file_element(file_elem: ET.Element) -> str | None:
        for child in file_elem:
            if LocalDirectoryImageFetcher._local_name(child.tag) != "FLocat":
                continue
            for attr_name, attr_value in child.attrib.items():
                if attr_name.endswith("href"):
                    return attr_value
        return None

    def _find_physical_struct_map(self, root: ET.Element) -> ET.Element | None:
        for elem in root.iter():
            if self._local_name(elem.tag) == "structMap" and elem.attrib.get("TYPE") == "physical":
                return elem
        return None

    def _extract_issue_date(self, root: ET.Element) -> str | None:
        year: int | None = None
        month: int | None = None
        day: int | None = None

        for elem in root.iter():
            if self._local_name(elem.tag) != "SerialHierarchy":
                continue
            level = elem.attrib.get("level")
            order = elem.attrib.get("order")
            if not order or not order.isdigit():
                continue
            value = int(order)
            if level == "1":
                year = value
            elif level == "2":
                month = value
            elif level == "3":
                day = value

        if year and month and day:
            return date(year, month, day).isoformat()
        return None

    @staticmethod
    def _infer_numeric_page_number(stem: str) -> int | None:
        return int(stem) if stem.isdigit() else None

    @staticmethod
    def _load_image_array(image_path: Path):
        if image_path.suffix.lower() == ".jp2":
            try:
                import numpy as np
                from PIL import Image
            except ImportError as exc:
                raise ImportError(
                    "JPEG2000 (.jp2) decoding requires Pillow and NumPy. Install with: pip install pillow numpy"
                ) from exc

            with Image.open(image_path) as pil_image:
                rgb = pil_image.convert("RGB")
                rgb_array = np.asarray(rgb)
            # Pipeline uses OpenCV-style BGR arrays.
            return rgb_array[:, :, ::-1].copy()

        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "LocalDirectoryImageFetcher requires `opencv-python` (cv2) to decode images."
            ) from exc

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image file: {image_path}")
        return image
