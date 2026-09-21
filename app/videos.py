from pathlib import Path
from math import isfinite
from threading import Lock
from uuid import uuid4

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    send_from_directory,
    url_for,
)
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from app.extensions import db
from app.forms import ALLOWED_VIDEO_EXTENSIONS, FrameExtractionForm, VideoUploadForm
from app.models import Frame, TrafficSignDetection, Video

videos_bp = Blueprint("videos", __name__)
_EXTRACTION_LOCKS = {}
_EXTRACTION_LOCKS_GUARD = Lock()
_DETECTION_LOCKS = {}
_DETECTION_LOCKS_GUARD = Lock()
TRAFFIC_SIGN_CLASS_ID = 11
TRAFFIC_SIGN_CLASS_NAME = "stop sign"

VIDEO_MIME_TYPES = {
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mov": "video/quicktime",
    "avi": "video/x-msvideo",
    "mkv": "video/x-matroska",
}


def allowed_video_file(filename):
    if not filename or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[-1].lower()
    return extension in ALLOWED_VIDEO_EXTENSIONS


def stored_video_name(original_filename):
    extension = original_filename.rsplit(".", 1)[-1].lower()
    safe_stem = secure_filename(Path(original_filename).stem) or "video"
    return f"{uuid4().hex}_{safe_stem}.{extension}"


def upload_folder():
    folder = Path(current_app.config["VIDEO_UPLOAD_FOLDER"])
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def frame_folder():
    folder = Path(current_app.config["FRAME_UPLOAD_FOLDER"])
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def detection_folder():
    folder = Path(current_app.config["DETECTION_UPLOAD_FOLDER"])
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def frame_name(video_id, frame_number):
    return f"video_{video_id}_{frame_number}_{uuid4().hex}.jpg"


def remove_frame_files(frames):
    folder = frame_folder()
    for frame in frames:
        path = folder / frame.filename
        if path.is_file():
            path.unlink()


def extraction_lock(video_id):
    with _EXTRACTION_LOCKS_GUARD:
        return _EXTRACTION_LOCKS.setdefault(video_id, Lock())


def detection_lock(video_id):
    with _DETECTION_LOCKS_GUARD:
        return _DETECTION_LOCKS.setdefault(video_id, Lock())


def traffic_sign_model():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is not installed. Run pip install -r requirements.txt.") from error

    model_path = Path(current_app.config["TRAFFIC_SIGN_MODEL"])
    if not model_path.is_file():
        raise RuntimeError(
            "Traffic-sign model weights are missing. Add yolov5s.onnx to the instance/models directory."
        )
    return cv2.dnn.readNetFromONNX(str(model_path))


def predict_stop_signs(model, image):
    import cv2
    import numpy as np

    image_height, image_width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image, scalefactor=1 / 255.0, size=(640, 640), swapRB=True, crop=False
    )
    model.setInput(blob)
    output = model.forward()
    predictions = output[0].T if output.ndim == 3 else output
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes = []
    confidences = []
    for prediction in predictions:
        objectness = float(prediction[4])
        class_scores = prediction[5:]
        class_id = int(np.argmax(class_scores))
        confidence = objectness * float(class_scores[class_id])
        if class_id != TRAFFIC_SIGN_CLASS_ID or confidence < 0.25:
            continue
        center_x, center_y, width, height = prediction[:4]
        left = int((center_x - width / 2) * image_width / 640)
        top = int((center_y - height / 2) * image_height / 640)
        box_width = int(width * image_width / 640)
        box_height = int(height * image_height / 640)
        boxes.append([left, top, box_width, box_height])
        confidences.append(confidence)

    selected = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    detections = []
    for index in selected:
        index = int(index)
        left, top, width, height = boxes[index]
        detections.append(
            {
                "class_name": TRAFFIC_SIGN_CLASS_NAME,
                "confidence": confidences[index],
                "x_min": max(0, left),
                "y_min": max(0, top),
                "x_max": min(image_width, left + width),
                "y_max": min(image_height, top + height),
            }
        )
    return detections


def detect_traffic_signs(video):
    model = traffic_sign_model()
    frames = list(video.frames)
    if not frames:
        raise RuntimeError("Extract frames before detecting traffic signs.")

    old_detections = TrafficSignDetection.query.filter_by(video_id=video.id).all()
    old_annotated = [d.annotated_filename for d in old_detections if d.annotated_filename]
    db.session.query(TrafficSignDetection).filter_by(video_id=video.id).delete(
        synchronize_session=False
    )
    db.session.flush()
    for filename in old_annotated:
        path = detection_folder() / filename
        if path.is_file():
            path.unlink()

    detection_count = 0
    created_files = []
    try:
        for frame in frames:
            frame_path = frame_folder() / frame.filename
            if not frame_path.is_file():
                continue
            import cv2

            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            detections = predict_stop_signs(model, image)
            if not detections:
                continue
            annotated_filename = f"video_{video.id}_frame_{frame.id}_{uuid4().hex}.jpg"
            annotated_path = detection_folder() / annotated_filename
            annotated = image.copy()
            for detection in detections:
                cv2.rectangle(
                    annotated,
                    (int(detection["x_min"]), int(detection["y_min"])),
                    (int(detection["x_max"]), int(detection["y_max"])),
                    (0, 200, 0),
                    2,
                )
                cv2.putText(
                    annotated,
                    f"{detection['class_name']} {detection['confidence']:.2f}",
                    (int(detection["x_min"]), max(20, int(detection["y_min"]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 0),
                    2,
                )
            if not cv2.imwrite(str(annotated_path), annotated):
                raise RuntimeError("The annotated detection image could not be saved.")
            created_files.append(annotated_path)
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
        db.session.commit()
    except Exception:
        for path in created_files:
            if path.is_file():
                path.unlink()
        db.session.rollback()
        raise
    return detection_count, len(frames)


def extract_video_frames(video, interval_seconds):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is not installed. Run pip install -r requirements.txt.") from error

    video_path = upload_folder() / video.filename
    if not video_path.is_file():
        raise RuntimeError("The video file could not be found on the server.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not open this video file.")

    fps_value = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    fps = fps_value if isfinite(fps_value) and fps_value > 0 else None
    frame_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_frames = int(frame_count_value) if frame_count_value > 0 else None
    duration = total_frames / fps if total_frames and fps else None
    step_frames = max(1, round((fps or 1) * interval_seconds))
    max_frames = current_app.config["MAX_EXTRACTED_FRAMES"]
    old_frames = list(video.frames)

    saved_paths = []
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
                path = frame_folder() / filename
                if not cv2.imwrite(str(path), image):
                    raise RuntimeError("OpenCV could not save an extracted frame.")
                saved_paths.append(path)
                db.session.add(
                    Frame(
                        video_id=video.id,
                        frame_number=frame_number,
                        timestamp_seconds=frame_number / (fps or 1),
                        filename=filename,
                    )
                )
                extracted += 1
                next_frame += step_frames
            frame_number += 1
        capture.release()
        if extracted == 0:
            raise RuntimeError("No frames could be extracted from this video.")
        remove_frame_files(old_frames)
        for old_frame in old_frames:
            db.session.delete(old_frame)
        db.session.commit()
    except Exception:
        capture.release()
        for path in saved_paths:
            if path.is_file():
                path.unlink()
        db.session.rollback()
        raise

    return extracted, fps, total_frames, duration, extracted >= max_frames


@videos_bp.route("/videos", methods=["GET", "POST"])
@login_required
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

        filename = stored_video_name(original_name)
        destination = upload_folder() / filename
        try:
            uploaded.save(destination)
            video = Video(
                title=form.title.data.strip(),
                filename=filename,
                original_filename=original_name,
                location=form.location.data.strip(),
                description=(form.description.data or "").strip() or None,
                uploaded_by=current_user.id,
            )
            db.session.add(video)
            db.session.commit()
        except Exception:
            db.session.rollback()
            if destination.is_file():
                destination.unlink()
            flash("The video could not be saved. Please try again.", "error")
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
@login_required
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
        media_type=VIDEO_MIME_TYPES.get(
            video.filename.rsplit(".", 1)[-1].lower(),
            "video/mp4",
        ),
    )


@videos_bp.route("/videos/<int:video_id>/detect-signs", methods=["POST"])
@login_required
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
            detected, frame_count = detect_traffic_signs(video)
        except RuntimeError as error:
            flash(str(error), "error")
            return redirect(url_for("videos.watch", video_id=video.id))
        flash(
            f"Traffic-sign detection completed: {detected} sign(s) found across {frame_count} extracted frame(s).",
            "success",
        )
        return redirect(url_for("videos.watch", video_id=video.id))
    finally:
        lock.release()


@videos_bp.route("/videos/<int:video_id>/detections/<path:filename>")
@login_required
def detection_media(video_id, filename):
    detection = TrafficSignDetection.query.filter_by(
        video_id=video_id, annotated_filename=filename
    ).first()
    if detection is None:
        abort(404)
    return send_from_directory(detection_folder(), filename, mimetype="image/jpeg")


@videos_bp.route("/videos/<int:video_id>/extract", methods=["POST"])
@login_required
def extract(video_id):
    video = db.session.get(Video, video_id)
    if video is None:
        abort(404)

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
            flash(str(error), "error")
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
@login_required
def frame_media(video_id, frame_id):
    frame = db.session.get(Frame, frame_id)
    if frame is None or frame.video_id != video_id:
        abort(404)
    folder = frame_folder()
    if not (folder / frame.filename).is_file():
        abort(404)
    return send_from_directory(folder, frame.filename, mimetype="image/jpeg")


@videos_bp.route("/videos/<int:video_id>/file")
@login_required
def media(video_id):
    video = db.session.get(Video, video_id)
    if video is None:
        abort(404)
    folder = upload_folder()
    path = folder / video.filename
    if not path.is_file():
        flash("The video file could not be found on the server.", "error")
        return redirect(url_for("videos.library"))
    extension = video.filename.rsplit(".", 1)[-1].lower()
    return send_from_directory(
        folder,
        video.filename,
        mimetype=VIDEO_MIME_TYPES.get(extension, "application/octet-stream"),
        as_attachment=False,
    )


@videos_bp.route("/videos/<int:video_id>/delete", methods=["POST"])
@login_required
def delete(video_id):
    if not current_user.is_admin():
        flash("Only administrators can delete road videos.", "error")
        return redirect(url_for("videos.library"))

    video = db.session.get(Video, video_id)
    if video is None:
        abort(404)

    path = upload_folder() / video.filename
    if path.is_file():
        path.unlink()

    remove_frame_files(video.frames)
    detections = TrafficSignDetection.query.filter_by(video_id=video.id).all()
    for detection in detections:
        if detection.annotated_filename:
            path = detection_folder() / detection.annotated_filename
            if path.is_file():
                path.unlink()
    for detection in detections:
        db.session.delete(detection)

    db.session.delete(video)
    db.session.commit()
    flash("Road video deleted.", "success")
    return redirect(url_for("videos.library"))
