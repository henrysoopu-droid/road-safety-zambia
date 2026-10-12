from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from flask import current_app, send_file
from werkzeug.utils import secure_filename


class MediaStorage:
    def __init__(
        self,
        backend: str,
        roots: dict[str, Path],
        bucket: str | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ):
        self.backend = backend
        self.roots = roots
        self.bucket = bucket
        self._client = None
        if backend == "s3":
            if not bucket or not region or not access_key_id or not secret_access_key:
                raise RuntimeError(
                    "S3 media storage requires S3_BUCKET, S3_REGION, "
                    "AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY. "
                    "Configure them in the deployment environment."
                )
            import boto3

            self._client = boto3.client(
                "s3",
                region_name=region,
                endpoint_url=endpoint_url or None,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
            )

    @staticmethod
    def _safe_filename(filename: str) -> str:
        if (
            not filename
            or filename in {".", ".."}
            or Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
            or secure_filename(filename) != filename
        ):
            raise ValueError("Invalid media filename.")
        return filename

    def _key(self, category: str, filename: str) -> str:
        if category not in self.roots:
            raise ValueError("Unknown media category.")
        return f"{category}/{self._safe_filename(filename)}"

    def _path(self, category: str, filename: str) -> Path:
        return self.roots[category] / self._safe_filename(filename)

    def save_upload(self, filename: str, uploaded_file) -> None:
        key = self._key("videos", filename)
        if self.backend == "filesystem":
            path = self._path("videos", filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            uploaded_file.save(path)
            return
        import mimetypes

        self._client.upload_fileobj(
            uploaded_file.stream,
            self.bucket,
            key,
            ExtraArgs={
                "ContentType": mimetypes.guess_type(filename)[0]
                or "application/octet-stream"
            },
        )

    def save_image(self, category: str, filename: str, image) -> bool:
        if category not in {"frames", "detections"}:
            raise ValueError("Images can only be saved as frames or detections.")
        if self.backend == "filesystem":
            import cv2

            path = self._path(category, filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            return bool(cv2.imwrite(str(path), image))

        import cv2

        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            return False
        self._client.put_object(
            Bucket=self.bucket,
            Key=self._key(category, filename),
            Body=encoded.tobytes(),
            ContentType="image/jpeg",
        )
        return True

    def save_file(self, category: str, filename: str, source_path: Path) -> None:
        key = self._key(category, filename)
        if filename.lower().endswith(".mp4"):
            content_type = "video/mp4"
        elif filename.lower().endswith(".jpg"):
            content_type = "image/jpeg"
        else:
            content_type = "application/octet-stream"
        if self.backend == "filesystem":
            destination = self._path(category, filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.urandom(8).hex()}.tmp")
            try:
                with source_path.open("rb") as source, temporary.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            return
        self._client.upload_file(
            str(source_path),
            self.bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )

    def materialize(self, category: str, filename: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            with self.local_path(category, filename) as source_path:
                with source_path.open("rb") as source, temporary_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                    target.flush()
                    os.fsync(target.fileno())
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)

    @contextmanager
    def local_path(self, category: str, filename: str):
        key = self._key(category, filename)
        if self.backend == "filesystem":
            path = self._path(category, filename)
            if not path.is_file():
                raise FileNotFoundError(f"Stored {category[:-1]} is unavailable.")
            yield path
            return

        with tempfile.TemporaryDirectory(prefix="road-safety-media-") as temporary_dir:
            path = Path(temporary_dir) / self._safe_filename(filename)
            try:
                self._client.download_file(self.bucket, key, str(path))
            except Exception as error:
                if self._is_not_found(error):
                    raise FileNotFoundError(f"Stored {category[:-1]} is unavailable.") from error
                raise
            yield path

    def exists(self, category: str, filename: str) -> bool:
        key = self._key(category, filename)
        if self.backend == "filesystem":
            return self._path(category, filename).is_file()
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as error:
            if self._is_not_found(error):
                return False
            raise

    def delete(self, category: str, filename: str) -> None:
        key = self._key(category, filename)
        if self.backend == "filesystem":
            self._path(category, filename).unlink(missing_ok=True)
            return
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def send(self, category: str, filename: str, mimetype: str):
        if self.backend == "filesystem":
            path = self._path(category, filename)
            if not path.is_file():
                raise FileNotFoundError(f"Stored {category[:-1]} is unavailable.")
            return send_file(path, mimetype=mimetype, conditional=True)

        key = self._key(category, filename)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="road-safety-download-",
            suffix=Path(filename).suffix,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            self._client.download_file(self.bucket, key, str(temporary_path))
        except Exception as error:
            temporary_path.unlink(missing_ok=True)
            if self._is_not_found(error):
                raise FileNotFoundError(f"Stored {category[:-1]} is unavailable.") from error
            raise
        response = send_file(temporary_path, mimetype=mimetype, conditional=True)
        response.call_on_close(lambda: temporary_path.unlink(missing_ok=True))
        return response

    @staticmethod
    def _is_not_found(error: Exception) -> bool:
        from botocore.exceptions import ClientError

        if not isinstance(error, ClientError):
            return False
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = error.response.get("Error", {}).get("Code")
        return status == 404 or code in {"404", "NoSuchKey", "NotFound"}


def media_storage() -> MediaStorage:
    return current_app.extensions["media_storage"]
