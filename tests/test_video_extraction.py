import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app import create_app
from app.extensions import db
from app.models import Frame, RoadMarkingDetection, User, Video
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session


class VideoExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_env = {
            key: os.environ.get(key)
            for key in ("DATABASE_URL", "SECRET_KEY", "MEDIA_ROOT", "MODEL_DIR")
        }
        root = Path(self.temp_dir.name)
        os.environ["DATABASE_URL"] = f"sqlite:///{(root / 'extraction.db').as_posix()}"
        os.environ["SECRET_KEY"] = "video-extraction-test-secret"
        os.environ["MEDIA_ROOT"] = str(root / "uploads")
        os.environ["MODEL_DIR"] = str(root / "models")

        self.app = create_app()
        self.app.config["TESTING"] = True
        self.app.config["WTF_CSRF_ENABLED"] = False
        self.app.config["MAX_EXTRACTED_FRAMES"] = 20
        self.client = self.app.test_client()

        with self.app.app_context():
            user = User(username="extraction-test-admin", role="admin", active=True)
            user.set_password("test-password")
            db.session.add(user)
            db.session.commit()
            user_id = user.id
            self.video = Video(
                title="Extraction test",
                filename="source.mp4",
                original_filename="source.mp4",
                uploaded_by=user_id,
            )
            db.session.add(self.video)
            db.session.commit()
            self.video_id = self.video.id
            video_path = Path(self.app.config["VIDEO_UPLOAD_FOLDER"]) / self.video.filename
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64)
            )
            if not writer.isOpened():
                self.fail("OpenCV could not create the synthetic MP4 test video.")
            for frame_number in range(20):
                image = np.full(
                    (64, 64, 3), frame_number * 10, dtype=np.uint8
                )
                writer.write(image)
            writer.release()

        with self.client.session_transaction() as session:
            session["_user_id"] = str(user_id)
            session["_fresh"] = True

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

    def extract(self):
        return self.client.post(
            f"/videos/{self.video_id}/extract",
            data={f"extract-{self.video_id}-interval_seconds": "1"},
            follow_redirects=True,
        )

    def test_extract_route_creates_frames_from_uploaded_video(self):
        frame_folder = Path(self.app.config["FRAME_UPLOAD_FOLDER"])
        frame_folder.rmdir()

        response = self.extract()

        self.assertEqual(response.status_code, 200)
        self.assertIn("Extracted 2 frame(s)", response.get_data(as_text=True))
        with self.app.app_context():
            frames = Frame.query.filter_by(video_id=self.video_id).all()
            self.assertEqual(len(frames), 2)
            for frame in frames:
                self.assertTrue((frame_folder / frame.filename).is_file())

    def test_repeated_extraction_replaces_frames_and_cleans_old_records(self):
        first_response = self.extract()
        self.assertEqual(first_response.status_code, 200)

        with self.app.app_context():
            old_frame = Frame.query.filter_by(video_id=self.video_id).first()
            old_frame_id = old_frame.id
            old_frame_name = old_frame.filename
            old_image_path = (
                Path(self.app.config["FRAME_UPLOAD_FOLDER"]) / old_frame_name
            )
            old_annotation_name = "old-marking.jpg"
            old_annotation_path = (
                Path(self.app.config["DETECTION_UPLOAD_FOLDER"])
                / old_annotation_name
            )
            self.assertTrue(
                cv2.imwrite(
                    str(old_annotation_path),
                    np.zeros((64, 64, 3), dtype=np.uint8),
                )
            )
            db.session.add(
                RoadMarkingDetection(
                    video_id=self.video_id,
                    frame_id=old_frame_id,
                    class_name="lane marking",
                    confidence=0.8,
                    frame_number=old_frame.frame_number,
                    timestamp_seconds=old_frame.timestamp_seconds,
                    frame_filename=old_frame_name,
                    annotated_filename=old_annotation_name,
                )
            )
            db.session.commit()

        response = self.extract()

        self.assertEqual(response.status_code, 200)
        self.assertIn("Extracted 2 frame(s)", response.get_data(as_text=True))
        self.assertFalse(old_image_path.exists())
        self.assertFalse(old_annotation_path.exists())
        with self.app.app_context():
            self.assertIsNone(db.session.get(Frame, old_frame_id))
            self.assertEqual(
                RoadMarkingDetection.query.filter_by(video_id=self.video_id).count(),
                0,
            )
            frames = Frame.query.filter_by(video_id=self.video_id).all()
            self.assertEqual(len(frames), 2)
            self.assertTrue(
                all(
                    (Path(self.app.config["FRAME_UPLOAD_FOLDER"]) / frame.filename).is_file()
                    for frame in frames
                )
            )

    def test_invalid_video_retry_preserves_existing_frames_and_files(self):
        first_response = self.extract()
        self.assertEqual(first_response.status_code, 200)
        with self.app.app_context():
            old_frames = Frame.query.filter_by(video_id=self.video_id).all()
            old_frame_ids = {frame.id for frame in old_frames}
            old_frame_names = {frame.filename for frame in old_frames}
        frame_folder = Path(self.app.config["FRAME_UPLOAD_FOLDER"])
        video_path = Path(self.app.config["VIDEO_UPLOAD_FOLDER"]) / "source.mp4"
        video_path.write_bytes(b"not a valid video stream")

        response = self.extract()

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "OpenCV could not open this video.",
            response.get_data(as_text=True),
        )
        with self.app.app_context():
            self.assertEqual(
                {
                    frame.id
                    for frame in Frame.query.filter_by(video_id=self.video_id).all()
                },
                old_frame_ids,
            )
        self.assertEqual(
            {path.name for path in frame_folder.glob("*.jpg")},
            old_frame_names,
        )

    def test_database_failure_preserves_existing_frames_and_files(self):
        with self.app.app_context():
            old_frame = Frame(
                video_id=self.video_id,
                frame_number=99,
                timestamp_seconds=9.9,
                filename="old-frame.jpg",
            )
            db.session.add(old_frame)
            db.session.commit()
            old_frame_id = old_frame.id
            old_frame_path = (
                Path(self.app.config["FRAME_UPLOAD_FOLDER"]) / old_frame.filename
            )
            self.assertTrue(
                cv2.imwrite(
                    str(old_frame_path),
                    np.zeros((64, 64, 3), dtype=np.uint8),
                )
            )

        with patch.object(
            Session,
            "commit",
            side_effect=SQLAlchemyError("simulated commit failure"),
        ):
            response = self.extract()

        self.assertEqual(response.status_code, 200)
        self.assertIn("could not be saved to the database", response.get_data(as_text=True))
        self.assertTrue(old_frame_path.is_file())
        with self.app.app_context():
            self.assertIsNotNone(db.session.get(Frame, old_frame_id))
            self.assertEqual(
                {path.name for path in Path(self.app.config["FRAME_UPLOAD_FOLDER"]).glob("*.jpg")},
                {old_frame_path.name},
            )

    def test_extraction_storage_error_displays_actionable_message(self):
        with patch(
            "app.videos.extract_video_frames",
            side_effect=PermissionError("storage permission denied"),
        ):
            response = self.extract()

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "Check that the Render disk is mounted and that MEDIA_ROOT is writable.",
            response.get_data(as_text=True),
        )

    def test_missing_uploaded_video_displays_codec_or_path_error(self):
        video_path = Path(self.app.config["VIDEO_UPLOAD_FOLDER"]) / "source.mp4"
        video_path.unlink()

        response = self.extract()

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "The video file could not be found on the server.",
            response.get_data(as_text=True),
        )


if __name__ == "__main__":
    unittest.main()
