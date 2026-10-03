import os
import tempfile
import unittest

from sqlalchemy import inspect, text

from app import create_app, ensure_database_schema, sqlite_uri

os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

class PublicAccessTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_public_pages_disable_coming_soon_placeholder(self):
        response = self.client.get("/signs")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Coming Soon", response.get_data(as_text=True))

        response = self.client.get("/markings")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Coming Soon", response.get_data(as_text=True))

        response = self.client.get("/results")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Coming Soon", response.get_data(as_text=True))

    def test_public_video_library_is_accessible_without_login(self):
        response = self.client.get("/public/videos")
        self.assertEqual(response.status_code, 200)

    def test_relative_sqlite_database_uri_resolves_inside_instance_dir(self):
        instance_dir = "/tmp/road_safety_instance"
        db_uri = sqlite_uri(instance_dir, "sqlite:///instance/road_safety.db")
        self.assertIn("sqlite:///", db_uri)
        self.assertNotIn("/instance/instance/", db_uri)
        self.assertTrue(db_uri.endswith("road_safety.db"))

    def test_relative_sqlite_database_uri_stays_under_project_instance(self):
        instance_dir = "/tmp/project/instance"
        db_uri = sqlite_uri(instance_dir, "sqlite:///road_safety.db")
        self.assertTrue(db_uri.endswith("road_safety.db"))
        self.assertIn("project/instance", db_uri)

    def test_legacy_sqlite_database_is_migrated_for_video_columns(self):
        original_database_url = os.environ.get("DATABASE_URL")
        temp_dir = tempfile.TemporaryDirectory()
        db_path = os.path.join(temp_dir.name, "legacy_road_safety_test.db")
        old_database = f"sqlite:///{db_path}"
        os.environ["DATABASE_URL"] = old_database

        try:
            app = create_app()
            from app.extensions import db

            with app.app_context():
                try:
                    db.drop_all()
                    db.session.execute(
                        text(
                            """
                            CREATE TABLE videos (
                                id INTEGER PRIMARY KEY,
                                title VARCHAR(160) NOT NULL,
                                filename VARCHAR(255) NOT NULL,
                                original_filename VARCHAR(255) NOT NULL,
                                location VARCHAR(160),
                                description TEXT,
                                uploaded_by INTEGER NOT NULL,
                                uploaded_at DATETIME NOT NULL,
                                processing_status VARCHAR(30) NOT NULL DEFAULT 'uploaded',
                                analysis_status VARCHAR(30) NOT NULL DEFAULT 'not_started'
                            )
                            """
                        )
                    )
                    db.session.commit()

                    inspector = inspect(db.engine)
                    self.assertNotIn("annotated_video_filename", {col["name"] for col in inspector.get_columns("videos")})

                    ensure_database_schema(app)
                    inspector = inspect(db.engine)
                    columns = {col["name"] for col in inspector.get_columns("videos")}
                    self.assertIn("annotated_video_filename", columns)
                finally:
                    db.session.remove()
                    db.engine.dispose()
        finally:
            if original_database_url is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = original_database_url
            temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
