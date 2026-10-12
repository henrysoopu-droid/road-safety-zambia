import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from flask import Flask
from botocore.exceptions import ClientError
from werkzeug.datastructures import FileStorage

from app.media_storage import MediaStorage


class FakeS3Client:
    def __init__(self):
        self.objects = {}

    def upload_fileobj(self, stream, bucket, key, ExtraArgs=None):
        self.objects[(bucket, key)] = stream.read()

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[(Bucket, Key)] = Body

    def upload_file(self, source, bucket, key, ExtraArgs=None):
        self.objects[(bucket, key)] = Path(source).read_bytes()

    def download_file(self, bucket, key, destination):
        self.last_download_path = destination
        try:
            content = self.objects[(bucket, key)]
        except KeyError as error:
            raise ClientError(
                {
                    "Error": {"Code": "NoSuchKey"},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "GetObject",
            ) from error
        Path(destination).write_bytes(content)

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError(
                {
                    "Error": {"Code": "NotFound"},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "HeadObject",
            )

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)


class MediaStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.roots = {
            category: self.root / category
            for category in ("videos", "frames", "detections", "models")
        }
        self.client = FakeS3Client()
        with patch("boto3.client", return_value=self.client):
            self.storage = MediaStorage(
                backend="s3",
                roots=self.roots,
                bucket="private-test-bucket",
                region="us-test-1",
                access_key_id="test-access-key",
                secret_access_key="test-secret-key",
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_s3_upload_and_opencv_staging_use_object_storage(self):
        uploaded = FileStorage(stream=io.BytesIO(b"video bytes"), filename="road.mp4")
        self.storage.save_upload("stored-road.mp4", uploaded)
        self.assertEqual(
            self.client.objects[("private-test-bucket", "videos/stored-road.mp4")],
            b"video bytes",
        )

        with self.storage.local_path("videos", "stored-road.mp4") as local_video:
            self.assertEqual(local_video.read_bytes(), b"video bytes")

    def test_s3_model_materialization_creates_a_local_cache(self):
        model_path = self.root / "traffic_signs" / "model.pt"
        self.client.objects[("private-test-bucket", "models/model.pt")] = b"weights"
        self.storage.materialize("models", "model.pt", model_path)
        self.assertEqual(model_path.read_bytes(), b"weights")
        self.assertFalse(list(model_path.parent.glob("*.tmp")))

    def test_s3_saves_frames_and_annotated_outputs_and_deletes_them(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        self.assertTrue(self.storage.save_image("frames", "frame.jpg", image))
        self.assertIn(("private-test-bucket", "frames/frame.jpg"), self.client.objects)

        output = self.root / "analysis.mp4"
        output.write_bytes(b"annotated video")
        self.storage.save_file("detections", "analysis.mp4", output)
        self.assertTrue(self.storage.exists("detections", "analysis.mp4"))

        self.storage.delete("frames", "frame.jpg")
        self.assertFalse(self.storage.exists("frames", "frame.jpg"))

    def test_s3_media_response_streams_private_object_and_cleans_staging_file(self):
        self.client.objects[("private-test-bucket", "videos/public.mp4")] = b"video bytes"
        app = Flask(__name__)
        with app.test_request_context("/public/videos/1/file"):
            response = self.storage.send("videos", "public.mp4", "video/mp4")
            self.assertEqual(b"".join(response.response), b"video bytes")
            staged_path = Path(self.client.last_download_path)
            self.assertTrue(staged_path.is_file())
            response.close()
            self.assertFalse(staged_path.exists())

    def test_s3_configuration_does_not_fall_back_when_missing(self):
        with self.assertRaisesRegex(RuntimeError, "S3_BUCKET"):
            MediaStorage(
                backend="s3",
                roots=self.roots,
                bucket="",
                region="us-test-1",
                access_key_id="test-access-key",
                secret_access_key="test-secret-key",
            )

    def test_filesystem_media_backend_remains_available_for_local_use(self):
        storage = MediaStorage("filesystem", self.roots)
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        self.assertTrue(storage.save_image("frames", "local-frame.jpg", image))
        self.assertTrue(storage.exists("frames", "local-frame.jpg"))
        with storage.local_path("frames", "local-frame.jpg") as path:
            self.assertIsNotNone(cv2.imread(str(path)))


if __name__ == "__main__":
    unittest.main()
