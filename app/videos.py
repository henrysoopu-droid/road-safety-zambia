from datetime import timedelta
from pathlib import Path
from functools import wraps
from math import isfinite
from hashlib import sha256
from itertools import chain
import tempfile
from threading import Lock
from uuid import uuid4

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.utils import secure_filename

from app.detectors import SpeedHumpDetector, TrafficSignDetector
from app.extensions import db
from app.forms import ALLOWED_VIDEO_EXTENSIONS, FrameExtractionForm, VideoUploadForm
from app.media_storage import media_storage
from app.models import Frame, RoadMarkingDetection, TrafficSignDetection, Video, utc_now

videos_bp = Blueprint("videos", __name__)
_EXTRACTION_LOCKS = {}
_EXTRACTION_LOCKS_GUARD = Lock()
_DETECTION_LOCKS = {}
_DETECTION_LOCKS_GUARD = Lock()
_TRAFFIC_SIGN_MODEL_LOCK = Lock()
_TRAFFIC_SIGN_DETECTOR = None
_TRAFFIC_SIGN_DETECTOR_KEY = None
_FRAME_QUERY_BATCH_SIZE = 64
_DETECTION_FLUSH_BATCH_SIZE = 16
_EXTRACTION_FLUSH_BATCH_SIZE = 50
_ROAD_MARKING_MAX_IMAGE_SIDE = 1280
TRAFFIC_SIGN_CLASS_ID = 11
TRAFFIC_SIGN_CLASS_NAME = "stop sign"

VIDEO_MIME_TYPES = {
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mov": "video/quicktime",
    "avi": "video/x-msvideo",
    "mkv": "video/x-matroska",
}


def analyst_or_admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if not current_user.is_admin() and not current_user.is_analyst():
            abort(403)
        return view(*args, **kwargs)

    return wrapped


def allowed_video_file(filename):
    if not filename or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[-1].lower()
    return extension in ALLOWED_VIDEO_EXTENSIONS


def stored_video_name(original_filename):
    extension = original_filename.rsplit(".", 1)[-1].lower()
    safe_stem = secure_filename(Path(original_filename).stem) or "video"
    return f"{uuid4().hex}_{safe_stem}.{extension}"


def annotated_video_name(video_id):
    return f"video_{video_id}_analysed.mp4"


def frame_name(video_id, frame_number):
    return f"video_{video_id}_{frame_number}_{uuid4().hex}.jpg"


def remove_media_files(category, filenames):
    for filename in set(filter(None, filenames)):
        try:
            media_storage().delete(category, filename)
        except Exception:
            current_app.logger.exception(
                "Could not remove unreferenced %s media %s", category, filename
            )


def media_response(category, filename, mimetype):
    try:
        return media_storage().send(category, filename, mimetype)
    except FileNotFoundError:
        abort(404)


def upload_sha256(uploaded_file):
    digest = sha256()
    stream = uploaded_file.stream
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def extraction_lock(video_id):
    with _EXTRACTION_LOCKS_GUARD:
        return _EXTRACTION_LOCKS.setdefault(video_id, Lock())


def detection_lock(video_id):
    with _DETECTION_LOCKS_GUARD:
        return _DETECTION_LOCKS.setdefault(video_id, Lock())


def iter_video_frames(video_id):
    last_id = 0
    while True:
        batch = (
            Frame.query.filter(
                Frame.video_id == video_id,
                Frame.id > last_id,
            )
            .order_by(Frame.id)
            .limit(_FRAME_QUERY_BATCH_SIZE)
            .all()
        )
        if not batch:
            return
        last_id = batch[-1].id
        yield from batch


def detection_annotation_filenames(model, video_id):
    names = (
        db.session.query(model.annotated_filename)
        .filter(
            model.video_id == video_id,
            model.annotated_filename.isnot(None),
        )
        .distinct()
        .yield_per(_FRAME_QUERY_BATCH_SIZE)
    )
    return {name for (name,) in names if name}


def traffic_sign_model():
    global _TRAFFIC_SIGN_DETECTOR, _TRAFFIC_SIGN_DETECTOR_KEY

    model_path = Path(current_app.config["TRAFFIC_SIGN_MODEL_PATH"])
    threshold = current_app.config.get("TRAFFIC_SIGN_CONFIDENCE", 0.40)
    image_size = current_app.config.get("TRAFFIC_SIGN_IMAGE_SIZE", 416)
    max_detections = current_app.config.get("TRAFFIC_SIGN_MAX_DETECTIONS", 50)
    cpu_threads = current_app.config.get("TRAFFIC_SIGN_CPU_THREADS", 1)
    detector_key = (
        str(model_path.resolve()),
        threshold,
        image_size,
        max_detections,
        cpu_threads,
    )
    with _TRAFFIC_SIGN_MODEL_LOCK:
        if (
            current_app.config["MEDIA_STORAGE_BACKEND"] == "s3"
            and not model_path.is_file()
        ):
            if not model_path.is_file():
                try:
                    media_storage().materialize(
                        "models", model_path.name, model_path
                    )
                except FileNotFoundError as error:
                    raise RuntimeError(
                        "Traffic-sign model weights are missing from the configured media bucket. "
                        f"Upload {model_path.name} to the bucket's models/ key."
                    ) from error
        if _TRAFFIC_SIGN_DETECTOR_KEY != detector_key:
            _TRAFFIC_SIGN_DETECTOR = TrafficSignDetector(
                model_path=model_path,
                threshold=threshold,
                image_size=image_size,
                max_detections=max_detections,
                cpu_threads=cpu_threads,
            )
            _TRAFFIC_SIGN_DETECTOR_KEY = detector_key
        return _TRAFFIC_SIGN_DETECTOR


def speed_hump_model():
    return SpeedHumpDetector(
        model_path=current_app.config.get("SPEED_HUMP_MODEL_PATH"),
        threshold=current_app.config.get("SPEED_HUMP_CONFIDENCE", 0.40),
    )


def predict_traffic_signs(image):
    detector = traffic_sign_model()
    return detector.predict(image)


def detect_traffic_signs(video):
    detector = traffic_sign_model()
    frame_count = Frame.query.filter_by(video_id=video.id).count()
    if not frame_count:
        raise RuntimeError("Extract frames before detecting traffic signs.")
    if not detector.is_available():
        raise RuntimeError(
            "Traffic-sign YOLOv8 weights were not found. "
            f"Expected file: {getattr(detector, 'model_path', current_app.config['TRAFFIC_SIGN_MODEL_PATH'])}. "
            "Set TRAFFIC_SIGN_MODEL_PATH to the model file or include it at the configured model path."
        )

    storage = media_storage()
    old_annotated = detection_annotation_filenames(
        TrafficSignDetection, video.id
    )
    db.session.query(TrafficSignDetection).filter_by(video_id=video.id).delete(
        synchronize_session="fetch"
    )
    db.session.flush()
    detection_count = 0
    created_files = []
    try:
        for frame_number, frame in enumerate(iter_video_frames(video.id), start=1):
            if not storage.exists("frames", frame.filename):
                image = None
                detections = None
            else:
                import cv2

                with storage.local_path("frames", frame.filename) as frame_path:
                    image = cv2.imread(str(frame_path))
                if image is None:
                    detections = None
                else:
                    detections = detector.predict(image)
                    if detections:
                        annotated_filename = (
                            f"traffic_{video.id}_frame_{frame.id}_{uuid4().hex}.jpg"
                        )
                        for detection in detections:
                            cv2.rectangle(
                                image,
                                (int(detection["x_min"]), int(detection["y_min"])),
                                (int(detection["x_max"]), int(detection["y_max"])),
                                (0, 200, 0),
                                2,
                            )
                            cv2.putText(
                                image,
                                f"{detection['class_name']} {detection['confidence']:.2f}",
                                (
                                    int(detection["x_min"]),
                                    max(20, int(detection["y_min"]) - 8),
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 200, 0),
                                2,
                            )
                        created_files.append(annotated_filename)
                        if not storage.save_image(
                            "detections", annotated_filename, image
                        ):
                            raise RuntimeError(
                                "The annotated traffic-sign image could not be saved."
                            )
                        for detection in detections:
                            db.session.add(
                                TrafficSignDetection(
                                    video_id=video.id,
                                    frame_id=frame.id,
                                    class_name=detection["class_name"],
                                    confidence=detection["confidence"],
                                    frame_number=frame.frame_number,
                                    timestamp_seconds=frame.timestamp_seconds,
                                    frame_filename=frame.filename,
                                    annotated_filename=annotated_filename,
                                    x_min=detection["x_min"],
                                    y_min=detection["y_min"],
                                    x_max=detection["x_max"],
                                    y_max=detection["y_max"],
                                )
                            )
                            detection_count += 1
                if image is not None:
                    del image
                if detections is not None:
                    del detections
            if frame_number % _DETECTION_FLUSH_BATCH_SIZE == 0:
                db.session.flush()
        db.session.commit()
    except Exception:
        db.session.rollback()
        remove_media_files("detections", created_files)
        raise
    remove_media_files("detections", old_annotated)
    return detection_count, frame_count


def predict_road_markings(image):
    import cv2
    import numpy as np

    image_height, image_width = image.shape[:2]
    scale = max(image_height, image_width) / _ROAD_MARKING_MAX_IMAGE_SIDE
    if scale > 1:
        processing_width = max(1, round(image_width / scale))
        processing_height = max(1, round(image_height / scale))
        processing_image = cv2.resize(
            image,
            (processing_width, processing_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0
        processing_image = image

    hsv = cv2.cvtColor(processing_image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 80, 255))
    yellow_mask = cv2.inRange(hsv, (15, 80, 80), (45, 255, 255))
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    del hsv, white_mask, yellow_mask, mask, kernel, processing_image
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour) * scale**2
        if area < 250:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        original_width = w * scale
        original_height = h * scale
        ratio = original_width / max(original_height, 1)
        if ratio >= 2.5:
            class_name = "lane marking"
        elif original_width > 40 and original_height > 25:
            class_name = "zebra crossing"
        else:
            class_name = "arrow marking"

        confidence = min(0.99, max(0.35, area / 8000.0))
        detections.append(
            {
                "class_name": class_name,
                "confidence": float(confidence),
                "x_min": max(0, round(x * scale)),
                "y_min": max(0, round(y * scale)),
                "x_max": min(image_width, round((x + w) * scale)),
                "y_max": min(image_height, round((y + h) * scale)),
            }
        )
    return detections


def detect_road_markings(video):
    frame_count = Frame.query.filter_by(video_id=video.id).count()
    if not frame_count:
        raise RuntimeError("Extract frames before detecting road markings.")

    storage = media_storage()
    old_annotated = detection_annotation_filenames(
        RoadMarkingDetection, video.id
    )
    db.session.query(RoadMarkingDetection).filter_by(video_id=video.id).delete(
        synchronize_session="fetch"
    )
    db.session.flush()
    detection_count = 0
    created_files = []
    try:
        for frame_number, frame in enumerate(iter_video_frames(video.id), start=1):
            if not storage.exists("frames", frame.filename):
                image = None
                detections = None
            else:
                import cv2

                with storage.local_path("frames", frame.filename) as frame_path:
                    image = cv2.imread(str(frame_path))
                if image is None:
                    detections = None
                else:
                    detections = predict_road_markings(image)
                    if detections:
                        annotated_filename = (
                            f"markings_{video.id}_frame_{frame.id}_{uuid4().hex}.jpg"
                        )
                        for detection in detections:
                            cv2.rectangle(
                                image,
                                (int(detection["x_min"]), int(detection["y_min"])),
                                (int(detection["x_max"]), int(detection["y_max"])),
                                (0, 165, 255),
                                2,
                            )
                            cv2.putText(
                                image,
                                f"{detection['class_name']} {detection['confidence']:.2f}",
                                (
                                    int(detection["x_min"]),
                                    max(20, int(detection["y_min"]) - 8),
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 165, 255),
                                2,
                            )
                        created_files.append(annotated_filename)
                        if not storage.save_image(
                            "detections", annotated_filename, image
                        ):
                            raise RuntimeError(
                                "The annotated road-marking image could not be saved."
                            )
                        for detection in detections:
                            db.session.add(
                                RoadMarkingDetection(
                                    video_id=video.id,
                                    frame_id=frame.id,
                                    class_name=detection["class_name"],
                                    confidence=detection["confidence"],
                                    frame_number=frame.frame_number,
                                    timestamp_seconds=frame.timestamp_seconds,
                                    frame_filename=frame.filename,
                                    annotated_filename=annotated_filename,
                                    x_min=detection["x_min"],
                                    y_min=detection["y_min"],
                                    x_max=detection["x_max"],
                                    y_max=detection["y_max"],
                                )
                            )
                            detection_count += 1
                if image is not None:
                    del image
                if detections is not None:
                    del detections
            if frame_number % _DETECTION_FLUSH_BATCH_SIZE == 0:
                db.session.flush()
        db.session.commit()
    except Exception:
        db.session.rollback()
        remove_media_files("detections", created_files)
        raise
    remove_media_files("detections", old_annotated)
    return detection_count, frame_count


def generate_annotated_video(video):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is not installed. Run pip install -r requirements.txt.") from error

    if not Frame.query.filter_by(video_id=video.id).count():
        raise RuntimeError("Extract frames before generating an analysed video.")

    storage = media_storage()
    first_frame = next(
        (
            frame
            for frame in iter_video_frames(video.id)
            if storage.exists("frames", frame.filename)
        ),
        None,
    )
    if first_frame is None:
        raise RuntimeError("No frames are available to create an analysed video.")

    output_name = f"{annotated_video_name(video.id).removesuffix('.mp4')}_{uuid4().hex}.mp4"
    previous_output = video.annotated_video_filename
    try:
        with tempfile.TemporaryDirectory(prefix="road-safety-video-") as temporary_dir:
            output_path = Path(temporary_dir) / output_name
            with storage.local_path("frames", first_frame.filename) as first_frame_path:
                first_image = cv2.imread(str(first_frame_path))
            if first_image is None:
                raise RuntimeError("The first extracted frame could not be read.")

            height, width = first_image.shape[:2]
            frame_interval = max(first_frame.timestamp_seconds, 0.1)
            fps = max(1.0, 1.0 / frame_interval)
            del first_image
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            try:
                if not writer.isOpened():
                    raise RuntimeError(
                        "OpenCV could not create the annotated output video."
                    )
                for frame in iter_video_frames(video.id):
                    if not storage.exists("frames", frame.filename):
                        continue
                    with storage.local_path("frames", frame.filename) as image_path:
                        image = cv2.imread(str(image_path))
                    if image is None:
                        continue
                    detections = chain(
                        TrafficSignDetection.query.filter_by(frame_id=frame.id)
                        .order_by(TrafficSignDetection.confidence.desc())
                        .yield_per(_FRAME_QUERY_BATCH_SIZE),
                        RoadMarkingDetection.query.filter_by(frame_id=frame.id)
                        .order_by(RoadMarkingDetection.confidence.desc())
                        .yield_per(_FRAME_QUERY_BATCH_SIZE),
                    )
                    for detection in detections:
                        label = (detection.class_name or "object").upper()
                        confidence = float(detection.confidence or 0.0)
                        left = max(0, int(getattr(detection, "x_min", 0) or 0))
                        top = max(0, int(getattr(detection, "y_min", 0) or 0))
                        right = min(
                            image.shape[1],
                            int(getattr(detection, "x_max", image.shape[1]) or image.shape[1]),
                        )
                        bottom = min(
                            image.shape[0],
                            int(getattr(detection, "y_max", image.shape[0]) or image.shape[0]),
                        )
                        color = (
                            (0, 200, 0)
                            if isinstance(detection, TrafficSignDetection)
                            else (0, 165, 255)
                        )
                        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
                        cv2.putText(
                            image,
                            f"{label} {confidence:.2f}",
                            (left, max(20, top - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )
                    writer.write(image)
                    del detections, image
            finally:
                writer.release()

            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("OpenCV did not produce an annotated output video.")
            storage.save_file("detections", output_name, output_path)

        video.annotated_video_filename = output_name
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
            remove_media_files("detections", [output_name])
            raise
    except Exception:
        db.session.rollback()
        if video.annotated_video_filename != output_name:
            remove_media_files("detections", [output_name])
        raise

    if previous_output:
        remove_media_files("detections", [previous_output])
    return output_name


def extract_video_frames(video, interval_seconds):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is not installed. Run pip install -r requirements.txt.") from error

    try:
        interval_seconds = float(interval_seconds)
    except (TypeError, ValueError):
        interval_seconds = 0
    if not isfinite(interval_seconds) or interval_seconds <= 0:
        raise RuntimeError("Choose a valid extraction interval greater than zero seconds.")
    if video.video_file_expired:
        raise RuntimeError(
            "The source video file has expired or was deleted. Existing analysis results remain available."
        )

    storage = media_storage()
    if not storage.exists("videos", video.filename):
        raise RuntimeError("The video file could not be found on the server.")
    capture = None
    old_frame_filenames = [
        filename
        for (filename,) in db.session.query(Frame.filename)
        .filter_by(video_id=video.id)
        .yield_per(_FRAME_QUERY_BATCH_SIZE)
    ]
    old_annotated_files = detection_annotation_filenames(
        TrafficSignDetection, video.id
    ) | detection_annotation_filenames(RoadMarkingDetection, video.id)
    old_annotated_video = video.annotated_video_filename
    saved_filenames = []
    pending_frames = []
    try:
        db.session.query(TrafficSignDetection).filter_by(video_id=video.id).delete(
            synchronize_session="fetch"
        )
        db.session.query(RoadMarkingDetection).filter_by(video_id=video.id).delete(
            synchronize_session="fetch"
        )
        db.session.query(Frame).filter_by(video_id=video.id).delete(
            synchronize_session="fetch"
        )
        db.session.flush()

        with storage.local_path("videos", video.filename) as video_path:
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                capture.release()
                capture = None
                raise RuntimeError(
                    "OpenCV could not open this video. Try a standard MP4 (H.264) or AVI file."
                )

            fps_value = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            fps = fps_value if isfinite(fps_value) and fps_value > 0 else None
            frame_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            total_frames = (
                int(frame_count_value)
                if isfinite(frame_count_value) and frame_count_value > 0
                else None
            )
            duration = total_frames / fps if total_frames and fps else None
            step_frames = max(1, round((fps or 1) * interval_seconds))
            max_frames = current_app.config["MAX_EXTRACTED_FRAMES"]
            extracted = 0
            frame_number = 0
            next_frame = 0

            try:
                while extracted < max_frames:
                    success, image = capture.read()
                    if not success:
                        break
                    if frame_number >= next_frame:
                        filename = frame_name(video.id, frame_number)
                        try:
                            frame_saved = storage.save_image("frames", filename, image)
                        except cv2.error as error:
                            raise RuntimeError(
                                f"OpenCV could not write extracted frame {frame_number} to media storage."
                            ) from error
                        if not frame_saved:
                            raise RuntimeError(
                                f"OpenCV could not write extracted frame {frame_number} to media storage."
                            )
                        saved_filenames.append(filename)
                        frame_record = Frame(
                            video_id=video.id,
                            frame_number=frame_number,
                            timestamp_seconds=frame_number / (fps or 1),
                            filename=filename,
                        )
                        db.session.add(frame_record)
                        pending_frames.append(frame_record)
                        extracted += 1
                        next_frame += step_frames
                        if len(pending_frames) >= _EXTRACTION_FLUSH_BATCH_SIZE:
                            db.session.flush()
                            for record in pending_frames:
                                db.session.expunge(record)
                            pending_frames.clear()
                    del image
                    frame_number += 1
            finally:
                capture.release()
                capture = None

        if extracted == 0:
            raise RuntimeError(
                "No frames could be extracted. The video may be empty, damaged, or encoded with an unsupported codec."
            )

        if pending_frames:
            db.session.flush()
            for record in pending_frames:
                db.session.expunge(record)
            pending_frames.clear()
        video.annotated_video_filename = None
        video.processing_status = "frames_extracted"
        video.analysis_status = "not_started"
        db.session.commit()
    except cv2.error as error:
        db.session.rollback()
        remove_media_files("frames", saved_filenames)
        raise RuntimeError(
            "OpenCV could not decode this video. Try converting it to a standard MP4 (H.264) or AVI file."
        ) from error
    except Exception:
        db.session.rollback()
        remove_media_files("frames", saved_filenames)
        raise
    finally:
        if capture is not None:
            capture.release()

    remove_media_files("frames", old_frame_filenames)
    remove_media_files("detections", old_annotated_files)
    if old_annotated_video:
        remove_media_files("detections", [old_annotated_video])

    return extracted, fps, total_frames, duration, extracted >= max_frames


@videos_bp.route("/videos", methods=["GET", "POST"])
@analyst_or_admin_required
def library():
    form = VideoUploadForm()
    if form.validate_on_submit():
        uploaded = form.video.data
        raw_name = (uploaded.filename or "upload").replace("\\", "/").split("/")[-1]
        original_name = raw_name[:255] or "upload"
        if not allowed_video_file(original_name):
            flash(
                "Unsupported video format. Use mp4, avi, mov, mkv, or webm.",
                "error",
            )
            return redirect(url_for("videos.library"))

        content_hash = upload_sha256(uploaded)
        existing = Video.query.filter_by(content_sha256=content_hash).first()
        if existing is not None:
            flash(
                f'This video is already stored as "{existing.title}". Reuse that record to rerun extraction or detection.',
                "info",
            )
            return redirect(url_for("videos.watch", video_id=existing.id))

        filename = stored_video_name(original_name)
        try:
            media_storage().save_upload(filename, uploaded)
            uploaded_at = utc_now()
            video = Video(
                title=form.title.data.strip(),
                filename=filename,
                original_filename=original_name,
                content_sha256=content_hash,
                location=(form.location.data or "").strip() or "Road location not provided",
                description=(form.description.data or "").strip() or None,
                uploaded_by=current_user.id,
                uploaded_at=uploaded_at,
                expires_at=uploaded_at + timedelta(
                    hours=current_app.config["VIDEO_RETENTION_HOURS"]
                ),
                processing_status="uploaded",
                analysis_status="not_started",
            )
            db.session.add(video)
            db.session.commit()
        except SQLAlchemyError:
            db.session.rollback()
            remove_media_files("videos", [filename])
            existing = Video.query.filter_by(content_sha256=content_hash).first()
            if existing is not None:
                flash(
                    f'This video is already stored as "{existing.title}". Reuse that record to rerun extraction or detection.',
                    "info",
                )
                return redirect(url_for("videos.watch", video_id=existing.id))
            current_app.logger.exception("Could not save uploaded video metadata")
            flash("The video could not be saved to the database. Please try again.", "error")
            return redirect(url_for("videos.library"))
        except Exception:
            db.session.rollback()
            remove_media_files("videos", [filename])
            current_app.logger.exception("Could not store uploaded video media")
            flash("The video could not be stored. Check the configured media storage and try again.", "error")
            return redirect(url_for("videos.library"))
        flash("Road video uploaded successfully.", "success")
        return redirect(url_for("videos.library"))

    videos = Video.query.order_by(Video.uploaded_at.desc()).all()
    extraction_forms = {
        video.id: FrameExtractionForm(prefix=f"extract-{video.id}")
        for video in videos
    }
    return render_template(
        "videos.html",
        form=form,
        videos=videos,
        extraction_forms=extraction_forms,
    )


@videos_bp.route("/videos/<int:video_id>")
@analyst_or_admin_required
def watch(video_id):
    video = db.session.get(Video, video_id)
    if video is None:
        abort(404)
    return render_template(
        "video_watch.html",
        video=video,
        extraction_form=FrameExtractionForm(prefix=f"extract-{video.id}"),
        detections=TrafficSignDetection.query.filter_by(video_id=video.id)
        .order_by(TrafficSignDetection.confidence.desc())
        .all(),
        road_markings=RoadMarkingDetection.query.filter_by(video_id=video.id)
        .order_by(RoadMarkingDetection.confidence.desc())
        .all(),
        media_type=VIDEO_MIME_TYPES.get(
            video.filename.rsplit(".", 1)[-1].lower(),
            "video/mp4",
        ),
    )


@videos_bp.route("/videos/<int:video_id>/annotated")
@analyst_or_admin_required
def annotated_media(video_id):
    video = db.session.get(Video, video_id)
    if video is None:
        abort(404)
    if not video.annotated_video_filename:
        abort(404)
    return media_response(
        "detections", video.annotated_video_filename, "video/mp4"
    )


@videos_bp.route("/videos/<int:video_id>/detect-signs", methods=["POST"])
@analyst_or_admin_required
def detect_signs(video_id):
    video = db.session.get(Video, video_id)
    if video is None:
        abort(404)
    lock = detection_lock(video.id)
    if not lock.acquire(blocking=False):
        flash("Traffic-sign detection is already in progress for this video.", "info")
        return redirect(url_for("videos.watch", video_id=video.id))
    try:
        try:
            video.processing_status = "processing"
            video.analysis_status = "processing"
            db.session.commit()
            detected, frame_count = detect_traffic_signs(video)
            try:
                generate_annotated_video(video)
            except Exception:
                current_app.logger.exception(
                    "Could not generate annotated video for video %s", video.id
                )
                flash(
                    "Traffic-sign detections were saved, but the annotated video could not be generated.",
                    "warning",
                )
            video.analysis_status = "analysed"
            video.processing_status = "analysed"
            db.session.commit()
        except RuntimeError as error:
            video.analysis_status = "failed"
            video.processing_status = "failed"
            db.session.commit()
            flash(str(error), "error")
            return redirect(url_for("videos.watch", video_id=video.id))
        except Exception:
            db.session.rollback()
            current_app.logger.exception(
                "Traffic-sign detection failed for video %s", video.id
            )
            video.analysis_status = "failed"
            video.processing_status = "failed"
            db.session.commit()
            flash(
                "Traffic-sign detection failed. Previously saved detections were preserved; the error was logged.",
                "error",
            )
            return redirect(url_for("videos.watch", video_id=video.id))
        flash(
            f"Traffic-sign detection completed: {detected} sign(s) found across {frame_count} extracted frame(s).",
            "success",
        )
        return redirect(url_for("videos.watch", video_id=video.id))
    finally:
        lock.release()


@videos_bp.route("/videos/<int:video_id>/detect-markings", methods=["POST"])
@analyst_or_admin_required
def detect_markings(video_id):
    video = db.session.get(Video, video_id)
    if video is None:
        abort(404)
    lock = detection_lock(video.id)
    if not lock.acquire(blocking=False):
        flash("Road-marking detection is already in progress for this video.", "info")
        return redirect(url_for("videos.watch", video_id=video.id))
    try:
        try:
            video.processing_status = "processing"
            video.analysis_status = "processing"
            db.session.commit()
            detected, frame_count = detect_road_markings(video)
            try:
                generate_annotated_video(video)
            except Exception:
                current_app.logger.exception(
                    "Could not generate annotated video for video %s", video.id
                )
                flash(
                    "Road-marking detections were saved, but the annotated video could not be generated.",
                    "warning",
                )
            video.analysis_status = "analysed"
            video.processing_status = "analysed"
            db.session.commit()
        except RuntimeError as error:
            video.analysis_status = "failed"
            video.processing_status = "failed"
            db.session.commit()
            flash(str(error), "error")
            return redirect(url_for("videos.watch", video_id=video.id))
        except Exception:
            db.session.rollback()
            current_app.logger.exception(
                "Road-marking detection failed for video %s", video.id
            )
            video.analysis_status = "failed"
            video.processing_status = "failed"
            db.session.commit()
            flash(
                "Road-marking detection failed. Previously saved detections were preserved; the error was logged.",
                "error",
            )
            return redirect(url_for("videos.watch", video_id=video.id))
        flash(
            f"Road-marking detection completed: {detected} marking(s) found across {frame_count} extracted frame(s).",
            "success",
        )
        return redirect(url_for("videos.watch", video_id=video.id))
    finally:
        lock.release()


@videos_bp.route("/videos/<int:video_id>/detections/<path:filename>")
@analyst_or_admin_required
def detection_media(video_id, filename):
    detection = TrafficSignDetection.query.filter_by(
        video_id=video_id, annotated_filename=filename
    ).first()
    if detection is None:
        abort(404)
    return media_response("detections", filename, "image/jpeg")


@videos_bp.route("/videos/<int:video_id>/markings/<path:filename>")
@analyst_or_admin_required
def marking_media(video_id, filename):
    detection = RoadMarkingDetection.query.filter_by(
        video_id=video_id, annotated_filename=filename
    ).first()
    if detection is None:
        abort(404)
    return media_response("detections", filename, "image/jpeg")


@videos_bp.route("/videos/<int:video_id>/extract", methods=["POST"])
@analyst_or_admin_required
def extract(video_id):
    video = db.session.get(Video, video_id)
    if video is None:
        abort(404)
    if video.video_file_expired:
        flash(
            "The source video file has expired or was deleted. Existing analysis results remain available.",
            "error",
        )
        return redirect(url_for("videos.watch", video_id=video.id))

    lock = extraction_lock(video.id)
    if not lock.acquire(blocking=False):
        flash("Frame extraction is already in progress for this video.", "info")
        return redirect(url_for("videos.watch", video_id=video.id))

    try:
        form = FrameExtractionForm(prefix=f"extract-{video.id}")
        if not form.validate_on_submit():
            errors = "; ".join(
                error for field_errors in form.errors.values() for error in field_errors
            )
            flash(errors or "Choose a valid extraction interval.", "error")
            return redirect(url_for("videos.watch", video_id=video.id))

        try:
            extracted, fps, total_frames, duration, capped = extract_video_frames(
                video, form.interval_seconds.data
            )
        except RuntimeError as error:
            db.session.rollback()
            current_app.logger.warning(
                "Frame extraction failed for video %s: %s", video.id, error
            )
            flash(str(error), "error")
            return redirect(url_for("videos.watch", video_id=video.id))
        except OSError:
            db.session.rollback()
            current_app.logger.exception(
                "Frame extraction could not access storage for video %s", video.id
            )
            flash(
                "The video or frame storage is unavailable. Check that the Render disk is mounted and that MEDIA_ROOT is writable. For S3 storage, verify the configured durable media-storage settings.",
                "error",
            )
            return redirect(url_for("videos.watch", video_id=video.id))
        except SQLAlchemyError:
            db.session.rollback()
            current_app.logger.exception(
                "Frame extraction could not save results for video %s", video.id
            )
            flash(
                "The extracted frames could not be saved to the database. Please try again.",
                "error",
            )
            return redirect(url_for("videos.watch", video_id=video.id))
        except Exception:
            db.session.rollback()
            current_app.logger.exception(
                "Unexpected frame extraction failure for video %s", video.id
            )
            flash(
                "Frame extraction failed unexpectedly. The error has been logged; please try again.",
                "error",
            )
            return redirect(url_for("videos.watch", video_id=video.id))

        details = [f"Extracted {extracted} frame(s)"]
        if fps:
            details.append(f"{fps:.2f} FPS")
        if total_frames:
            details.append(f"{total_frames} source frames")
        if duration is not None:
            details.append(f"{duration:.2f} seconds")
        if capped:
            details.append("maximum extraction limit reached")
        flash("; ".join(details) + ".", "success")
        return redirect(url_for("videos.watch", video_id=video.id))
    finally:
        lock.release()


@videos_bp.route("/videos/<int:video_id>/frames/<int:frame_id>/file")
@analyst_or_admin_required
def frame_media(video_id, frame_id):
    frame = db.session.get(Frame, frame_id)
    if frame is None or frame.video_id != video_id:
        abort(404)
    return media_response("frames", frame.filename, "image/jpeg")


@videos_bp.route("/videos/<int:video_id>/file")
@analyst_or_admin_required
def media(video_id):
    video = db.session.get(Video, video_id)
    if video is None:
        abort(404)
    if video.video_file_expired:
        return (
            "This video file has expired or was deleted. Its analysis results remain available.",
            410,
        )
    extension = video.filename.rsplit(".", 1)[-1].lower()
    try:
        return media_storage().send(
            "videos",
            video.filename,
            mimetype=VIDEO_MIME_TYPES.get(extension, "application/octet-stream"),
        )
    except FileNotFoundError:
        current_app.logger.warning("Source video file is missing for video %s", video.id)
        return (
            "This video file is unavailable. Its analysis results remain available.",
            410,
        )


@videos_bp.route("/videos/<int:video_id>/delete", methods=["POST"])
@analyst_or_admin_required
def delete(video_id):
    if not current_user.is_admin():
        flash("Only administrators can delete road videos.", "error")
        return redirect(url_for("videos.library"))

    video = db.session.get(Video, video_id)
    if video is None:
        abort(404)

    try:
        media_storage().delete("videos", video.filename)
        video.video_file_deleted_at = utc_now()
        db.session.commit()
    except Exception:
        db.session.rollback()
        current_app.logger.exception(
            "Could not delete source media for video %s", video.id
        )
        flash("The video file could not be deleted. Its record and analysis were preserved.", "error")
        return redirect(url_for("videos.library"))
    flash("Video file deleted. The video record and analysis history were preserved.", "success")
    return redirect(url_for("videos.library"))
