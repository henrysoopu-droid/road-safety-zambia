from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from app.extensions import db

__all__ = ["User", "Video", "Frame", "TrafficSignDetection", "RoadMarkingDetection"]


def utc_now():
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=False)
    role = db.Column(db.String(20), nullable=False)
    active = db.Column(db.Boolean, nullable=False, default=True)
    videos = db.relationship("Video", backref="uploader", lazy="dynamic")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_active(self):
        return self.active

    def is_admin(self):
        return self.role == "admin"

    def is_analyst(self):
        return self.role == "analyst"


class Video(db.Model):
    __tablename__ = "videos"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(160), nullable=False)
    filename = db.Column(db.String(255), nullable=False, unique=True)
    content_sha256 = db.Column(db.String(64), nullable=True)
    original_filename = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(160), nullable=True, default="")
    description = db.Column(db.Text, nullable=True)
    uploaded_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    uploaded_at = db.Column(db.DateTime, nullable=False, default=utc_now)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=True)
    video_file_deleted_at = db.Column(db.DateTime(timezone=True), nullable=True)
    processing_status = db.Column(db.String(30), nullable=False, default="uploaded")
    analysis_status = db.Column(db.String(30), nullable=False, default="not_started")
    annotated_video_filename = db.Column(db.String(255), nullable=True)
    frames = db.relationship(
        "Frame",
        backref="video",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="Frame.frame_number",
    )
    detections = db.relationship(
        "TrafficSignDetection",
        backref="video",
        lazy=True,
        cascade="all, delete-orphan",
    )
    road_marking_detections = db.relationship(
        "RoadMarkingDetection",
        backref="video",
        lazy=True,
        cascade="all, delete-orphan",
    )
    __table_args__ = (
        db.Index("ix_videos_content_sha256", "content_sha256", unique=True),
    )

    @property
    def video_file_expired(self):
        if self.video_file_deleted_at is not None:
            return True
        if self.expires_at is None:
            return False
        expires_at = self.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return expires_at <= utc_now()

    @property
    def video_status(self):
        if self.video_file_expired:
            return "Expired"
        if self.processing_status == "processing":
            return "Processing"
        if self.processing_status == "failed":
            return "Failed"
        return "Available"


class Frame(db.Model):
    __tablename__ = "frames"

    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey("videos.id"), nullable=False, index=True)
    frame_number = db.Column(db.Integer, nullable=False)
    timestamp_seconds = db.Column(db.Float, nullable=False)
    filename = db.Column(db.String(255), nullable=False, unique=True)
    extracted_at = db.Column(db.DateTime, nullable=False, default=utc_now)
    detections = db.relationship(
        "TrafficSignDetection",
        backref="frame",
        lazy=True,
        cascade="all, delete-orphan",
    )


class TrafficSignDetection(db.Model):
    __tablename__ = "traffic_sign_detections"

    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey("videos.id"), nullable=False, index=True)
    frame_id = db.Column(db.Integer, db.ForeignKey("frames.id"), nullable=False, index=True)
    class_name = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    frame_number = db.Column(db.Integer, nullable=False)
    timestamp_seconds = db.Column(db.Float, nullable=False)
    frame_filename = db.Column(db.String(255), nullable=False)
    annotated_filename = db.Column(db.String(255), nullable=True)
    x_min = db.Column(db.Float, nullable=True)
    y_min = db.Column(db.Float, nullable=True)
    x_max = db.Column(db.Float, nullable=True)
    y_max = db.Column(db.Float, nullable=True)
    detected_at = db.Column(db.DateTime, nullable=False, default=utc_now)


class RoadMarkingDetection(db.Model):
    __tablename__ = "road_marking_detections"

    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey("videos.id"), nullable=False, index=True)
    frame_id = db.Column(db.Integer, db.ForeignKey("frames.id"), nullable=False, index=True)
    class_name = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    frame_number = db.Column(db.Integer, nullable=False)
    timestamp_seconds = db.Column(db.Float, nullable=False)
    frame_filename = db.Column(db.String(255), nullable=False)
    annotated_filename = db.Column(db.String(255), nullable=True)
    x_min = db.Column(db.Float, nullable=True)
    y_min = db.Column(db.Float, nullable=True)
    x_max = db.Column(db.Float, nullable=True)
    y_max = db.Column(db.Float, nullable=True)
    detected_at = db.Column(db.DateTime, nullable=False, default=utc_now)
