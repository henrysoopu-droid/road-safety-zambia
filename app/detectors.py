from __future__ import annotations

import os
from pathlib import Path


TRAFFIC_SIGN_CLASS_NAMES = [
    "Stop",
    "Red Light",
    "Green Light",
    "Speed Limit 10",
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
]


def model_root() -> Path:
    model_dir = Path(os.getenv("MODEL_DIR", str(Path(__file__).resolve().parent.parent / "instance" / "models")))
    return model_dir


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

    This selected model was trained on a general traffic-sign dataset and is not a
    guaranteed fit for Zambian or SADC road signs. Its performance must be evaluated
    on local road footage before any production claims are made.
    """

    def __init__(self, model_path: str | os.PathLike[str] | None = None, threshold: float | None = None):
        self.model_path = Path(model_path) if model_path is not None else Path(
            os.getenv("TRAFFIC_SIGN_MODEL_PATH", str(model_root() / "traffic_signs" / "yolov8s-best.pt"))
        )
        self.threshold = float(threshold if threshold is not None else os.getenv("TRAFFIC_SIGN_CONFIDENCE", "0.40"))
        self.model = None

    def is_available(self) -> bool:
        return self.model_path.is_file()

    def _ensure_model(self):
        if self.model is not None:
            return self.model
        if not self.is_available():
            raise RuntimeError(
                f"Traffic-sign model not found at {self.model_path}. Place the YOLOv8 weights file there or set TRAFFIC_SIGN_MODEL_PATH."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("Ultralytics is required. Install it with pip install -r requirements.txt.") from exc
        self.model = YOLO(str(self.model_path))
        return self.model

    def predict(self, image, max_detections: int | None = None):
        model = self._ensure_model()
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
            names = getattr(result, "names", {}) or getattr(model, "names", {}) or {}
            if hasattr(result, "boxes") and result.boxes is not None:
                for box in result.boxes:
                    if hasattr(box, "xyxy") and len(box.xyxy) > 0:
                        xyxy = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = xyxy
                        conf = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                        cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
                        raw_name = names.get(cls_id, names[cls_id] if isinstance(names, list) else f"class_{cls_id}")
                        label = normalize_traffic_sign_name(raw_name)
                        detections.append(
                            {
                                "class_name": label,
                                "confidence": conf,
                                "x_min": max(0, int(x1)),
                                "y_min": max(0, int(y1)),
                                "x_max": int(x2),
                                "y_max": int(y2),
                            }
                        )
        if max_detections is not None:
            detections = detections[:max_detections]
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
