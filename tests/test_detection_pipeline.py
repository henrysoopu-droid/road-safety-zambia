import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from sqlalchemy import MetaData, create_engine, inspect, UniqueConstraint

from app import create_app
from app.extensions import db
from app.models import Frame, RoadMarkingDetection, TrafficSignDetection, User, Video
from app.videos import detect_road_markings, detect_traffic_signs, generate_annotated_video


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


if __name__ == "__main__":
    unittest.main()