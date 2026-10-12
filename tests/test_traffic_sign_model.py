import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app import create_app
from app.detectors import TRAFFIC_SIGN_CLASS_NAMES, TrafficSignDetector
from app.extensions import db


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "instance"
    / "models"
    / "traffic_signs"
    / "yolov8s-best.pt"
)


class TrafficSignModelTests(unittest.TestCase):
    def test_render_relative_model_directory_resolves_to_project_weights(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            database_path = Path(temporary_dir) / "model-config.db"
            environment = {
                "SECRET_KEY": "traffic-sign-model-test-secret",
                "DATABASE_URL": f"sqlite:///{database_path.as_posix()}",
                "MEDIA_ROOT": str(Path(temporary_dir) / "uploads"),
                "MODEL_DIR": "instance/models",
                "TRAFFIC_SIGN_MODEL": "yolov8s-best.pt",
            }
            with patch.dict(os.environ, environment):
                app = create_app()
                try:
                    self.assertEqual(
                        Path(app.config["TRAFFIC_SIGN_MODEL_PATH"]),
                        MODEL_PATH.resolve(),
                    )
                    self.assertTrue(Path(app.config["TRAFFIC_SIGN_MODEL_PATH"]).is_file())
                    self.assertEqual(
                        app.config["MODEL_CONFIG"]["traffic_sign"]["classes"],
                        TRAFFIC_SIGN_CLASS_NAMES,
                    )
                finally:
                    with app.app_context():
                        db.session.remove()
                        db.engine.dispose()

    def test_project_weights_load_with_expected_classes_and_run_inference(self):
        self.assertTrue(MODEL_PATH.is_file(), f"Missing project weights: {MODEL_PATH}")
        detector = TrafficSignDetector(model_path=MODEL_PATH)

        model = detector._ensure_model()

        self.assertEqual(
            model.names,
            dict(enumerate(TRAFFIC_SIGN_CLASS_NAMES)),
        )
        detections = detector.predict(np.zeros((128, 128, 3), dtype=np.uint8))
        self.assertIsInstance(detections, list)

    def test_missing_weights_error_reports_expected_path_and_configuration(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            missing_path = Path(temporary_dir) / "missing.pt"
            detector = TrafficSignDetector(model_path=missing_path)

            with self.assertRaisesRegex(
                RuntimeError,
                "Expected file: .*missing\\.pt.*TRAFFIC_SIGN_MODEL_PATH",
            ):
                detector._ensure_model()

    def test_invalid_weights_error_is_logged_with_path_and_cause(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            invalid_path = Path(temporary_dir) / "invalid.pt"
            invalid_path.write_bytes(b"not a YOLO checkpoint")
            detector = TrafficSignDetector(model_path=invalid_path)

            with (
                patch("ultralytics.YOLO", side_effect=ValueError("invalid checkpoint")),
                self.assertLogs("app.detectors", level="ERROR") as logs,
                self.assertRaisesRegex(
                    RuntimeError,
                    "could not be loaded.*invalid\\.pt.*ValueError: invalid checkpoint",
                ),
            ):
                detector._ensure_model()

            self.assertIn(
                "Failed to load traffic-sign YOLOv8 weights from",
                logs.output[0],
            )


if __name__ == "__main__":
    unittest.main()
