import os
import tempfile
import unittest
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from app import create_app
from app.cleanup import cleanup_expired_videos
from app.extensions import db
from app.models import (
    Frame,
    RoadMarkingDetection,
    TrafficSignDetection,
    User,
    Video,
    utc_now,
)


class VideoExpirationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_env = {
            key: os.environ.get(key)
            for key in (
                "DATABASE_URL",
                "SECRET_KEY",
                "MEDIA_ROOT",
                "MEDIA_STORAGE",
                "VIDEO_RETENTION_HOURS",
                "MODEL_DIR",
            )
        }
        root = Path(self.temp_dir.name)
        os.environ["DATABASE_URL"] = f"sqlite:///{(root / 'expiration.db').as_posix()}"
        os.environ["SECRET_KEY"] = "video-expiration-test-secret"
        os.environ["MEDIA_ROOT"] = str(root / "uploads")
        os.environ["MEDIA_STORAGE"] = "filesystem"
        os.environ["VIDEO_RETENTION_HOURS"] = "3"
        os.environ["MODEL_DIR"] = str(root / "models")

        self.app = create_app()
        self.app.config["TESTING"] = True
        self.app.config["WTF_CSRF_ENABLED"] = False
        self.client = self.app.test_client()
        with self.app.app_context():
            self.user = User(
                username="expiration-test-admin",
                role="admin",
                active=True,
            )
            self.user.set_password("test-password")
            db.session.add(self.user)
            db.session.commit()
            self.user_id = self.user.id
            self.video_id, self.frame_id = self.create_video(
                "retained-source.mp4",
                utc_now() + timedelta(hours=1),
            )

        with self.client.session_transaction() as session:
            session["_user_id"] = str(self.user_id)
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

    def create_video(self, filename, expires_at):
        video = Video(
            title="Retained road report",
            filename=filename,
            original_filename="original-road-video.mp4",
            uploaded_by=self.user_id,
            uploaded_at=utc_now(),
            expires_at=expires_at,
            processing_status="analysed",
            analysis_status="analysed",
        )
        db.session.add(video)
        db.session.flush()
        frame = Frame(
            video_id=video.id,
            frame_number=1,
            timestamp_seconds=0,
            filename="retained-frame.jpg",
        )
        db.session.add(frame)
        db.session.flush()
        db.session.add(
            TrafficSignDetection(
                video_id=video.id,
                frame_id=frame.id,
                class_name="stop sign",
                confidence=0.95,
                frame_number=1,
                timestamp_seconds=0,
                frame_filename=frame.filename,
                x_min=1,
                y_min=1,
                x_max=10,
                y_max=10,
            )
        )
        db.session.add(
            RoadMarkingDetection(
                video_id=video.id,
                frame_id=frame.id,
                class_name="lane marking",
                confidence=0.85,
                frame_number=1,
                timestamp_seconds=0,
                frame_filename=frame.filename,
                x_min=1,
                y_min=1,
                x_max=10,
                y_max=10,
            )
        )
        db.session.commit()
        source_path = (
            Path(self.app.config["VIDEO_UPLOAD_FOLDER"]) / filename
        )
        source_path.write_bytes(b"source video")
        return video.id, frame.id

    def test_cleanup_waits_for_expiration_then_removes_only_source_file(self):
        source_path = (
            Path(self.app.config["VIDEO_UPLOAD_FOLDER"]) / "retained-source.mp4"
        )
        with self.app.app_context():
            self.assertEqual(cleanup_expired_videos(), 0)
            self.assertTrue(source_path.is_file())

            video = db.session.get(Video, self.video_id)
            video.expires_at = utc_now() - timedelta(seconds=1)
            db.session.commit()
            self.assertEqual(cleanup_expired_videos(), 1)

            self.assertFalse(source_path.exists())
            video = db.session.get(Video, self.video_id)
            self.assertIsNotNone(video)
            self.assertEqual(video.video_status, "Expired")
            self.assertEqual(video.original_filename, "original-road-video.mp4")
            self.assertEqual(video.processing_status, "analysed")
            self.assertEqual(video.analysis_status, "analysed")
            self.assertEqual(len(video.frames), 1)
            self.assertEqual(
                TrafficSignDetection.query.filter_by(video_id=self.video_id).count(),
                1,
            )
            self.assertEqual(
                RoadMarkingDetection.query.filter_by(video_id=self.video_id).count(),
                1,
            )
            self.assertEqual(cleanup_expired_videos(), 0)

    def test_cleanup_failure_is_logged_and_fails_the_job(self):
        with self.app.app_context():
            video = db.session.get(Video, self.video_id)
            video.expires_at = utc_now() - timedelta(seconds=1)
            db.session.commit()

            with self.assertLogs(self.app.logger, level="ERROR"):
                with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                    with patch("app.cleanup.get_media_storage") as get_storage:
                        get_storage.return_value.delete.side_effect = OSError(
                            "storage unavailable"
                        )
                        cleanup_expired_videos()

            video = db.session.get(Video, self.video_id)
            self.assertIsNone(video.video_file_deleted_at)
            self.assertTrue(
                (
                    Path(self.app.config["VIDEO_UPLOAD_FOLDER"])
                    / video.filename
                ).is_file()
            )

    def test_cleanup_does_not_remove_a_video_while_processing(self):
        source_path = (
            Path(self.app.config["VIDEO_UPLOAD_FOLDER"]) / "retained-source.mp4"
        )
        with self.app.app_context():
            video = db.session.get(Video, self.video_id)
            video.expires_at = utc_now() - timedelta(seconds=1)
            video.processing_status = "processing"
            db.session.commit()

            self.assertEqual(cleanup_expired_videos(), 0)
            self.assertTrue(source_path.is_file())
            self.assertIsNone(video.video_file_deleted_at)

    def test_expired_video_keeps_public_results_and_reports_file_expiration(self):
        with self.app.app_context():
            video = db.session.get(Video, self.video_id)
            video.expires_at = utc_now() - timedelta(seconds=1)
            db.session.commit()

        response = self.client.get(f"/public/videos/{self.video_id}")
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("expired or was deleted", page)
        self.assertIn("stop sign", page)
        self.assertIn("lane marking", page)
        self.assertNotIn("<video class=\"video-player\"", page)

        response = self.client.get(f"/public/videos/{self.video_id}/file")
        self.assertEqual(response.status_code, 410)
        self.assertIn("analysis results remain available", response.get_data(as_text=True))

        response = self.client.get(f"/videos/{self.video_id}/file")
        self.assertEqual(response.status_code, 410)

    def test_missing_source_file_returns_clear_unavailable_response(self):
        source_path = (
            Path(self.app.config["VIDEO_UPLOAD_FOLDER"]) / "retained-source.mp4"
        )
        source_path.unlink()

        response = self.client.get(f"/videos/{self.video_id}/file")
        self.assertEqual(response.status_code, 410)
        self.assertIn("video file is unavailable", response.get_data(as_text=True))

    def test_expired_source_cannot_be_used_for_frame_extraction(self):
        with self.app.app_context():
            video = db.session.get(Video, self.video_id)
            video.expires_at = utc_now() - timedelta(seconds=1)
            db.session.commit()

        response = self.client.post(
            f"/videos/{self.video_id}/extract",
            data={f"extract-{self.video_id}-interval_seconds": "1"},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("source video file has expired", response.get_data(as_text=True))
        with self.app.app_context():
            self.assertEqual(Frame.query.filter_by(video_id=self.video_id).count(), 1)
            self.assertEqual(
                TrafficSignDetection.query.filter_by(video_id=self.video_id).count(),
                1,
            )
            self.assertEqual(
                RoadMarkingDetection.query.filter_by(video_id=self.video_id).count(),
                1,
            )

    def test_admin_manual_file_deletion_preserves_record_and_analysis(self):
        response = self.client.post(f"/videos/{self.video_id}/delete")
        self.assertEqual(response.status_code, 302)

        with self.app.app_context():
            video = db.session.get(Video, self.video_id)
            self.assertIsNotNone(video)
            self.assertIsNotNone(video.video_file_deleted_at)
            self.assertEqual(video.video_status, "Expired")
            self.assertEqual(len(video.frames), 1)
            self.assertEqual(
                TrafficSignDetection.query.filter_by(video_id=self.video_id).count(),
                1,
            )
            self.assertEqual(
                RoadMarkingDetection.query.filter_by(video_id=self.video_id).count(),
                1,
            )
            self.assertIsNotNone(db.session.get(User, self.user_id))

    def test_upload_records_configured_retention_expiration(self):
        response = self.client.post(
            "/videos",
            data={
                "title": "New uploaded road",
                "location": "Lusaka",
                "description": "",
                "video": (BytesIO(b"new source video"), "new-upload.mp4"),
                "submit": "Upload",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 302)

        with self.app.app_context():
            video = Video.query.filter_by(title="New uploaded road").one()
            self.assertEqual(video.original_filename, "new-upload.mp4")
            self.assertEqual(
                video.expires_at - video.uploaded_at,
                timedelta(hours=3),
            )

    def test_video_status_covers_available_processing_failed_and_expired(self):
        with self.app.app_context():
            video = db.session.get(Video, self.video_id)
            self.assertEqual(video.video_status, "Available")

            video.processing_status = "processing"
            self.assertEqual(video.video_status, "Processing")

            video.processing_status = "failed"
            self.assertEqual(video.video_status, "Failed")

            video.expires_at = utc_now() - timedelta(seconds=1)
            self.assertEqual(video.video_status, "Expired")


if __name__ == "__main__":
    unittest.main()
