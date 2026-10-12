import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from sqlalchemy import MetaData, create_engine, inspect, UniqueConstraint

from app import create_app
from app.detectors import TrafficSignDetector
from app.extensions import db
from app.models import Frame, RoadMarkingDetection, TrafficSignDetection, User, Video
from app.videos import (
    detect_road_markings,
    detect_traffic_signs,
    generate_annotated_video,
    traffic_sign_model,
)


class FixedTrafficSignDetector:
    def is_available(self):
        return True

    def predict(self, image):
        return [
            {
                "class_name": "Stop Sign",
                "confidence": 0.91,
                "x_min": 10,
                "y_min": 10,
                "x_max": 80,
                "y_max": 80,
            },
            {
                "class_name": "Speed Limit 50",
                "confidence": 0.87,
                "x_min": 100,
                "y_min": 20,
                "x_max": 160,
                "y_max": 90,
            },
        ]


class DetectionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_env = {
            key: os.environ.get(key)
            for key in ("DATABASE_URL", "SECRET_KEY", "MEDIA_ROOT", "MODEL_DIR")
        }
        database_path = Path(self.temp_dir.name) / "legacy.db"
        database_url = f"sqlite:///{database_path.as_posix()}"
        os.environ["DATABASE_URL"] = database_url
        os.environ["SECRET_KEY"] = "detection-test-secret"
        os.environ["MEDIA_ROOT"] = str(Path(self.temp_dir.name) / "uploads")
        os.environ["MODEL_DIR"] = str(Path(self.temp_dir.name) / "models")

        legacy_metadata = MetaData()
        for table in db.metadata.sorted_tables:
            table.to_metadata(legacy_metadata)
        for table_name in ("traffic_sign_detections", "road_marking_detections"):
            table = legacy_metadata.tables[table_name]
            table.append_constraint(UniqueConstraint(table.c.annotated_filename))
        legacy_engine = create_engine(database_url)
        legacy_metadata.create_all(legacy_engine)
        legacy_engine.dispose()

        self.app = create_app()
        with self.app.app_context():
            user = User(username="test-admin", role="admin", active=True)
            user.set_password("test-password")
            db.session.add(user)
            db.session.commit()
            video = Video(
                title="Detection test",
                filename="source.mp4",
                original_filename="source.mp4",
                uploaded_by=user.id,
            )
            db.session.add(video)
            db.session.commit()
            frame = Frame(
                video_id=video.id,
                frame_number=0,
                timestamp_seconds=0.0,
                filename="frame.jpg",
            )
            db.session.add(frame)
            db.session.commit()

            image = np.full((240, 480, 3), 70, dtype=np.uint8)
            cv2.rectangle(image, (40, 100), (110, 120), (255, 255, 255), -1)
            cv2.rectangle(image, (260, 100), (340, 120), (255, 255, 255), -1)
            self.assertTrue(
                cv2.imwrite(
                    str(Path(self.app.config["FRAME_UPLOAD_FOLDER"]) / frame.filename),
                    image,
                )
            )

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.engine.dispose()
        for key, value in self.previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temp_dir.cleanup()

    def test_legacy_schema_and_detection_video_pipeline(self):
        with self.app.app_context():
            for table_name in (
                "traffic_sign_detections",
                "road_marking_detections",
            ):
                self.assertFalse(
                    any(
                        constraint.get("column_names") == ["annotated_filename"]
                        for constraint in inspect(db.engine).get_unique_constraints(
                            table_name
                        )
                    )
                )

            video = Video.query.one()
            marking_count, frame_count = detect_road_markings(video)
            self.assertEqual((marking_count, frame_count), (2, 1))
            markings = RoadMarkingDetection.query.filter_by(video_id=video.id).all()
            self.assertEqual(len({item.annotated_filename for item in markings}), 1)

            with patch("app.videos.traffic_sign_model", return_value=FixedTrafficSignDetector()):
                sign_count, frame_count = detect_traffic_signs(video)
            self.assertEqual((sign_count, frame_count), (2, 1))
            signs = TrafficSignDetection.query.filter_by(video_id=video.id).all()
            self.assertEqual(len({item.annotated_filename for item in signs}), 1)

            output_name = generate_annotated_video(video)
            output_path = Path(self.app.config["DETECTION_UPLOAD_FOLDER"]) / output_name
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_yolo_inference_uses_bounded_streaming_cpu_settings(self):
        detector = TrafficSignDetector(
            model_path="unused-for-mocked-model.pt",
            image_size=416,
            max_detections=2,
            cpu_threads=1,
        )

        class Box:
            def __init__(self, x):
                self.xyxy = [np.array([x, 10, x + 20, 30])]
                self.conf = [0.9]
                self.cls = [0]

        stream_closed = []

        def prediction_stream(**kwargs):
            try:
                yield SimpleNamespace(
                    names={0: "Stop"},
                    boxes=[Box(1), Box(30), Box(60)],
                )
            finally:
                stream_closed.append(True)

        detector.model = Mock()
        detector.model.names = {0: "Stop"}
        detector.model.predict.side_effect = prediction_stream
        with patch("torch.set_num_threads") as set_num_threads:
            detections = detector.predict(np.zeros((64, 64, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0]["class_name"], "Stop Sign")
        set_num_threads.assert_called_once_with(1)
        detector.model.predict.assert_called_once()
        self.assertEqual(detector.model.predict.call_args.kwargs["imgsz"], 416)
        self.assertEqual(detector.model.predict.call_args.kwargs["max_det"], 2)
        self.assertEqual(detector.model.predict.call_args.kwargs["device"], "cpu")
        self.assertTrue(detector.model.predict.call_args.kwargs["stream"])
        self.assertEqual(stream_closed, [True])

    def test_traffic_sign_detector_instance_is_reused_for_matching_config(self):
        with self.app.app_context():
            with (
                patch("app.videos._TRAFFIC_SIGN_DETECTOR", None),
                patch("app.videos._TRAFFIC_SIGN_DETECTOR_KEY", None),
                patch("app.videos.TrafficSignDetector") as detector_factory,
            ):
                detector_factory.return_value = object()

                first = traffic_sign_model()
                second = traffic_sign_model()

                self.assertIs(first, second)
                detector_factory.assert_called_once()


if __name__ == "__main__":
    unittest.main()