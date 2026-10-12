from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


TRAFFIC_SIGN_CLASS_NAMES = [
    "Green Light",
    "Red Light",
    "Speed Limit 100",
    "Speed Limit 110",
    "Speed Limit 120",
    "Speed Limit 20",
    "Speed Limit 30",
    "Speed Limit 40",
    "Speed Limit 50",
    "Speed Limit 60",
    "Speed Limit 70",
    "Speed Limit 80",
    "Speed Limit 90",
    "Stop",
]


def model_root() -> Path:
    model_dir = Path(os.getenv("MODEL_DIR", str(Path(__file__).resolve().parent.parent / "instance" / "models")))
    if not model_dir.is_absolute():
        model_dir = Path(__file__).resolve().parent.parent / model_dir
    return model_dir.resolve()


def normalize_traffic_sign_name(raw_name: str) -> str:
    label = str(raw_name or "").strip()
    lowered = label.lower().replace("_", " ")
    if lowered in {"stop", "stop sign"}:
        return "Stop Sign"
    if lowered in {"red light", "traffic light red", "redlight"}:
        return "Red Light"
    if lowered in {"green light", "traffic light green", "greenlight"}:
        return "Green Light"
    if lowered.startswith("speed limit"):
        parts = lowered.split()
        if len(parts) >= 3 and parts[-1].isdigit():
            return f"Speed Limit {parts[-1]}"
    if label:
        return label
    return "Traffic Sign"


class TrafficSignDetector:
    """Load and run the YOLOv8 traffic-sign model for this project.

    The checkpoint's traffic-sign labels are verified, but its training source and
    fit for Zambian or SADC roads are not independently verified. Evaluate it on
    local road footage before making production claims.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        threshold: float | None = None,
        image_size: int | None = None,
        max_detections: int | None = None,
        cpu_threads: int | None = None,
    ):
        configured_path = Path(model_path) if model_path is not None else Path(
            os.getenv(
                "TRAFFIC_SIGN_MODEL_PATH",
                str(model_root() / "traffic_signs" / "yolov8s-best.pt"),
            )
        )
        if not configured_path.is_absolute():
            configured_path = Path(__file__).resolve().parent.parent / configured_path
        self.model_path = configured_path.resolve()
        self.threshold = float(threshold if threshold is not None else os.getenv("TRAFFIC_SIGN_CONFIDENCE", "0.40"))
        self.image_size = int(
            image_size if image_size is not None else os.getenv("TRAFFIC_SIGN_IMAGE_SIZE", "416")
        )
        self.max_detections = int(
            max_detections
            if max_detections is not None
            else os.getenv("TRAFFIC_SIGN_MAX_DETECTIONS", "50")
        )
        self.cpu_threads = int(
            cpu_threads if cpu_threads is not None else os.getenv("TRAFFIC_SIGN_CPU_THREADS", "1")
        )
        if self.image_size <= 0 or self.max_detections <= 0 or self.cpu_threads <= 0:
            raise ValueError("YOLO image size, maximum detections, and CPU threads must be positive.")
        self.model = None
        self._model_lock = Lock()
        self._inference_lock = Lock()
        self._cpu_threads_initialized = False

    def is_available(self) -> bool:
        return self.model_path.is_file()

    def _ensure_model(self):
        if self.model is not None:
            return self.model
        with self._model_lock:
            if self.model is not None:
                return self.model
            if not self.is_available():
                raise RuntimeError(
                    "Traffic-sign YOLOv8 weights were not found. "
                    f"Expected file: {self.model_path}. Set TRAFFIC_SIGN_MODEL_PATH "
                    "to the model file or include it at the configured model path."
                )
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError("Ultralytics is required. Install it with pip install -r requirements.txt.") from exc
            try:
                self.model = YOLO(str(self.model_path), task="detect")
            except Exception as exc:
                logger.exception(
                    "Failed to load traffic-sign YOLOv8 weights from %s",
                    self.model_path,
                )
                raise RuntimeError(
                    "Traffic-sign YOLOv8 weights exist but could not be loaded from "
                    f"{self.model_path} ({type(exc).__name__}: {exc}). Verify that the "
                    "file is a valid Ultralytics detection checkpoint."
                ) from exc
        return self.model

    def predict(self, image, max_detections: int | None = None):
        limit = self.max_detections if max_detections is None else min(
            self.max_detections, max(1, int(max_detections))
        )
        detections = []
        with self._inference_lock:
            import torch

            if not self._cpu_threads_initialized:
                torch.set_num_threads(self.cpu_threads)
                self._cpu_threads_initialized = True
            model = self._ensure_model()
            results = model.predict(
                source=image,
                conf=self.threshold,
                imgsz=self.image_size,
                max_det=limit,
                device="cpu",
                verbose=False,
                stream=True,
            )
            try:
                for result in results:
                    names = getattr(result, "names", {}) or getattr(model, "names", {}) or {}
                    boxes = getattr(result, "boxes", None)
                    if boxes is not None:
                        for box in boxes:
                            xyxy = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = xyxy
                            confidence = box.conf[0]
                            class_id = box.cls[0]
                            confidence = float(
                                confidence.item()
                                if hasattr(confidence, "item")
                                else confidence
                            )
                            class_id = int(
                                class_id.item() if hasattr(class_id, "item") else class_id
                            )
                            raw_name = names.get(
                                class_id,
                                names[class_id]
                                if isinstance(names, list)
                                else f"class_{class_id}",
                            )
                            detections.append(
                                {
                                    "class_name": normalize_traffic_sign_name(raw_name),
                                    "confidence": confidence,
                                    "x_min": max(0, int(x1)),
                                    "y_min": max(0, int(y1)),
                                    "x_max": int(x2),
                                    "y_max": int(y2),
                                }
                            )
                            if len(detections) >= limit:
                                break
                    del boxes, result
                    if len(detections) >= limit:
                        break
            finally:
                close = getattr(results, "close", None)
                if close is not None:
                    close()
                del results
        return detections


class SpeedHumpDetector:
    """Separate detector stub for future speed-hump or speed-bump inference.

    No verified local YOLO speed-hump model is bundled with this project. The class
    is kept modular so a trustworthy weights file can be added later without changing
    the extraction and video-processing pipeline.
    """

    def __init__(self, model_path: str | os.PathLike[str] | None = None, threshold: float | None = None):
        self.model_path = Path(model_path) if model_path is not None else Path(
            os.getenv("SPEED_HUMP_MODEL_PATH", str(model_root() / "speed_humps" / "speed_hump_model.pt"))
        )
        self.threshold = float(threshold if threshold is not None else os.getenv("SPEED_HUMP_CONFIDENCE", "0.40"))
        self.model = None

    def is_configured(self) -> bool:
        return self.model_path.is_file()

    def _ensure_model(self):
        if self.model is not None:
            return self.model
        if not self.is_configured():
            return None
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("Ultralytics is required. Install it with pip install -r requirements.txt.") from exc
        self.model = YOLO(str(self.model_path))
        return self.model

    def predict(self, image):
        model = self._ensure_model()
        if model is None:
            return []
        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = 0
        except Exception:
            pass
        results = model(image, conf=self.threshold, device=device, verbose=False)
        detections = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            if hasattr(result, "boxes") and result.boxes is not None:
                for box in result.boxes:
                    if hasattr(box, "xyxy") and len(box.xyxy) > 0:
                        xyxy = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = xyxy
                        cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
                        raw_name = names.get(cls_id, "Speed Hump") if isinstance(names, dict) else "Speed Hump"
                        confidence = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                        detections.append(
                            {
                                "class_name": "Speed Hump" if str(raw_name).lower() not in {"speed hump", "speed bump"} else str(raw_name),
                                "confidence": confidence,
                                "x_min": max(0, int(x1)),
                                "y_min": max(0, int(y1)),
                                "x_max": int(x2),
                                "y_max": int(y2),
                            }
                        )
        return detections
