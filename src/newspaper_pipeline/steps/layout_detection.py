from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Protocol

from ..models import ImageRecord, LayoutRegion


class LayoutDetector(Protocol):
    """Pluggable layout detection stage."""

    def detect(self, image: ImageRecord) -> list[LayoutRegion]:
        """Return layout regions for a single image."""


class NotImplementedLayoutDetector:
    def detect(self, image: ImageRecord) -> list[LayoutRegion]:
        raise NotImplementedError("Provide a LayoutDetector implementation.")


class DocLayoutYoloLayoutDetector:
    """
    Layout detection using DocLayout-YOLO (Ultralytics-compatible weights).

    Example model_source values:
    - local `.pt` path
    - an Ultralytics/HuggingFace model id if supported by your setup
    """

    DEFAULT_CLASS_MAP = {
        "Text": "article",
        "text": "article",
        "plain text": "article",
        "plain_text": "article",
        "body text": "article",
        "body_text": "article",
        "paragraph": "article",
        "Title": "headline",
        "title": "headline",
        "Section-header": "headline",
        "section-header": "headline",
        "section_header": "headline",
        "Caption": "image_caption",
        "caption": "image_caption",
        "Picture": "photograph",
        "picture": "photograph",
        "figure": "photograph",
        "image": "photograph",
        "Table": "table",
        "table": "table",
        "Footnote": "footnote",
        "footnote": "footnote",
        "Formula": "formula",
        "formula": "formula",
        "List-item": "list_item",
        "list-item": "list_item",
        "list_item": "list_item",
        "Page-header": "page_header",
        "page-header": "page_header",
        "page_header": "page_header",
        "Page-footer": "page_footer",
        "page-footer": "page_footer",
        "page_footer": "page_footer",
    }

    def __init__(
        self,
        model_source: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_map: dict[str, str] | None = None,
        keep_labels: set[str] | None = None,
        debug_output_dir: Path | None = None,
    ) -> None:
        if not model_source:
            raise ValueError("model_source is required for DocLayoutYoloLayoutDetector")

        self.model_source = model_source
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_map = class_map or self.DEFAULT_CLASS_MAP
        self.keep_labels = (
            {self._normalize_label(label) for label in keep_labels}
            if keep_labels is not None
            else None
        )
        self.debug_output_dir = debug_output_dir
        if self.debug_output_dir is not None:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    def detect(self, image: ImageRecord) -> list[LayoutRegion]:
        image_array = self._resolve_image_array(image)
        model = self._get_model()

        predictions = model.predict(
            source=image_array,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        if not predictions:
            return []

        result = predictions[0]
        if result.boxes is None:
            return []

        names = getattr(result, "names", {}) or {}
        regions: list[LayoutRegion] = []
        for idx, box in enumerate(result.boxes):
            cls_idx = int(box.cls.item())
            raw_label = names.get(cls_idx, str(cls_idx))
            normalized_raw = self._normalize_label(raw_label)
            label = self.class_map.get(raw_label) or self.class_map.get(normalized_raw) or normalized_raw
            normalized_label = self._normalize_label(label)
            if self.keep_labels is not None and normalized_label not in self.keep_labels:
                continue

            x0, y0, x1, y1 = [int(round(v)) for v in box.xyxy[0].tolist()]
            score = float(box.conf.item()) if box.conf is not None else None
            regions.append(
                LayoutRegion(
                    region_id=f"{image.image_id}_region_{idx}",
                    label=normalized_label,
                    bbox=(x0, y0, x1, y1),
                    score=score,
                )
            )

        if self.debug_output_dir is not None:
            self._write_debug_overlay(image, image_array, regions)

        return regions

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from doclayout_yolo import YOLOv10
        except ImportError as exc:
            raise ImportError(
                "DocLayoutYoloLayoutDetector requires `doclayout-yolo`. "
                "Install/upgrade it in your environment before running layout detection."
            ) from exc
        self._model = YOLOv10(self.model_source)
        return self._model

    @staticmethod
    def _resolve_image_array(image: ImageRecord):
        if image.image_bgr is None:
            raise ValueError(
                "DocLayoutYoloLayoutDetector expects in-memory image data at image.image_bgr."
            )
        return image.image_bgr

    @staticmethod
    def _normalize_label(label: str) -> str:
        return str(label).strip().lower().replace("-", "_").replace(" ", "_")

    def _write_debug_overlay(self, image: ImageRecord, image_array, regions: list[LayoutRegion]) -> None:
        try:
            import cv2
        except ImportError:
            return

        vis = image_array.copy()
        for region in regions:
            x0, y0, x1, y1 = region.bbox
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 3)
            label = region.label
            if region.score is not None:
                label = f"{label}:{region.score:.2f}"
            cv2.putText(
                vis,
                label,
                (x0, max(24, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        output_path = self.debug_output_dir / f"{image.image_id}_layout_debug.jpg"
        cv2.imwrite(str(output_path), vis)


class DellOnnxLayoutDetector:
    """
    Layout detection using Melissa Dell AmericanStories ONNX weights.
    """

    DEFAULT_LABEL_MAP = {
        0: "article",
        1: "author",
        2: "cartoon_or_advertisement",
        3: "headline",
        4: "image_caption",
        5: "masthead",
        6: "newspaper_header",
        7: "page_number",
        8: "photograph",
        9: "table",
    }

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.05,
        iou_threshold: float = 0.10,
        label_map: dict[int, str] | None = None,
        keep_labels: set[str] | None = None,
        input_size: int = 1280,
        debug_output_dir: Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.label_map = label_map or self.DEFAULT_LABEL_MAP
        self.keep_labels = (
            {self._normalize_label(label) for label in keep_labels}
            if keep_labels is not None
            else None
        )
        self.input_size = input_size
        self.debug_output_dir = debug_output_dir
        if self.debug_output_dir is not None:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)

        self._session = None
        self._input_name = None

    def detect(self, image: ImageRecord) -> list[LayoutRegion]:
        image_array = self._resolve_image_array(image)
        session, input_name = self._get_session()

        im, ratio, (dw, dh) = self._letterbox(image_array, (self.input_size, self.input_size), auto=False)
        im = im.transpose((2, 0, 1))[::-1]  # HWC->CHW, BGR->RGB
        im = self._as_float_input(im)

        raw = session.run(None, {input_name: im})[0]
        boxes = self._postprocess(raw)

        regions: list[LayoutRegion] = []
        orig_h, orig_w = image_array.shape[:2]
        for idx, det in enumerate(boxes):
            x0, y0, x1, y1, conf, cls_idx = det
            # Undo letterbox transform
            x0 = int(round((x0 - dw) / ratio))
            x1 = int(round((x1 - dw) / ratio))
            y0 = int(round((y0 - dh) / ratio))
            y1 = int(round((y1 - dh) / ratio))

            x0 = max(0, min(x0, orig_w - 1))
            x1 = max(0, min(x1, orig_w))
            y0 = max(0, min(y0, orig_h - 1))
            y1 = max(0, min(y1, orig_h))
            if x1 <= x0 or y1 <= y0:
                continue

            label = self.label_map.get(int(cls_idx), str(int(cls_idx)))
            norm_label = self._normalize_label(label)
            if self.keep_labels is not None and norm_label not in self.keep_labels:
                continue

            regions.append(
                LayoutRegion(
                    region_id=f"{image.image_id}_region_{idx}",
                    label=norm_label,
                    bbox=(x0, y0, x1, y1),
                    score=float(conf),
                )
            )

        if self.debug_output_dir is not None:
            self._write_debug_overlay(image, image_array, regions)
        return regions

    def _get_session(self):
        if self._session is not None and self._input_name is not None:
            return self._session, self._input_name

        if not self.model_path.exists():
            raise FileNotFoundError(f"Dell ONNX layout model not found: {self.model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("DellOnnxLayoutDetector requires `onnxruntime`.") from exc

        self._session = ort.InferenceSession(str(self.model_path))
        self._input_name = self._session.get_inputs()[0].name
        return self._session, self._input_name

    def _postprocess(self, raw: Any) -> list[tuple[float, float, float, float, float, int]]:
        """
        Handle YOLOv5-style or YOLOv8-style ONNX output and return xyxy + conf + cls.
        """
        import cv2
        import numpy as np

        pred = np.array(raw)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]

        # YOLOv8 often emits (C, N). Convert to (N, C).
        if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
            pred = pred.T

        dets: list[tuple[float, float, float, float, float, int]] = []

        if pred.ndim != 2 or pred.shape[1] < 6:
            return dets

        # Two common formats:
        # 1) [x,y,w,h,obj,cls1..clsN]
        # 2) [x,y,w,h,cls1..clsN]
        # Infer based on expected class count from label map.
        expected_nc = max(self.label_map.keys()) + 1 if self.label_map else 0
        has_obj = False
        if expected_nc > 0:
            if pred.shape[1] == 5 + expected_nc:
                has_obj = True
            elif pred.shape[1] == 4 + expected_nc:
                has_obj = False
            else:
                # Fallback: assume no objectness for modern YOLO exports.
                has_obj = False
        elif pred.shape[1] >= 6:
            # No label map; fallback heuristic.
            has_obj = pred.shape[1] > 6

        if has_obj:
            obj = pred[:, 4]
            cls_scores = pred[:, 5:]
            cls_idx = cls_scores.argmax(axis=1)
            cls_conf = cls_scores[np.arange(len(cls_scores)), cls_idx]
            conf = obj * cls_conf
        else:
            cls_scores = pred[:, 4:]
            cls_idx = cls_scores.argmax(axis=1)
            conf = cls_scores[np.arange(len(cls_scores)), cls_idx]

        keep = conf >= self.conf_threshold
        if not keep.any():
            return dets

        pred = pred[keep]
        conf = conf[keep]
        cls_idx = cls_idx[keep]

        boxes_xywh = pred[:, :4]
        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        # OpenCV NMS expects xywh
        boxes_cv = [[float(x), float(y), float(w), float(h)] for x, y, w, h in boxes_xywh.tolist()]
        conf_list = conf.astype(float).tolist()
        indices = cv2.dnn.NMSBoxes(boxes_cv, conf_list, self.conf_threshold, self.iou_threshold)
        if len(indices) == 0:
            return dets
        idxs = indices.flatten().tolist() if hasattr(indices, "flatten") else [int(i) for i in indices]

        for i in idxs:
            x0, y0, x1, y1 = boxes_xyxy[i].tolist()
            dets.append((x0, y0, x1, y1, float(conf[i]), int(cls_idx[i])))
        return dets

    @staticmethod
    def _xywh_to_xyxy(boxes):
        import numpy as np

        out = np.copy(boxes)
        out[:, 0] = boxes[:, 0] - (boxes[:, 2] / 2.0)
        out[:, 1] = boxes[:, 1] - (boxes[:, 3] / 2.0)
        out[:, 2] = boxes[:, 0] + (boxes[:, 2] / 2.0)
        out[:, 3] = boxes[:, 1] + (boxes[:, 3] / 2.0)
        return out

    @staticmethod
    def _resolve_image_array(image: ImageRecord):
        if image.image_bgr is None:
            raise ValueError(
                "DellOnnxLayoutDetector expects in-memory image data at image.image_bgr."
            )
        return image.image_bgr

    @staticmethod
    def _letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, stride=32):
        import cv2
        import numpy as np

        shape = im.shape[:2]  # h, w
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    @staticmethod
    def _as_float_input(chw_rgb):
        import numpy as np

        return np.expand_dims(np.ascontiguousarray(chw_rgb), axis=0).astype(np.float32) / 255.0

    @staticmethod
    def _normalize_label(label: str) -> str:
        return str(label).strip().lower().replace("-", "_").replace(" ", "_")

    def _write_debug_overlay(self, image: ImageRecord, image_array, regions: list[LayoutRegion]) -> None:
        try:
            import cv2
        except ImportError:
            return

        vis = image_array.copy()
        for region in regions:
            x0, y0, x1, y1 = region.bbox
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 180, 0), 3)
            label = region.label
            if region.score is not None:
                label = f"{label}:{region.score:.2f}"
            cv2.putText(
                vis,
                label,
                (x0, max(24, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )
        output_path = self.debug_output_dir / f"{image.image_id}_layout_debug.jpg"
        cv2.imwrite(str(output_path), vis)
