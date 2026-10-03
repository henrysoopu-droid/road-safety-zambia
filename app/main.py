from flask import Blueprint, Response, current_app, redirect, render_template, send_from_directory, url_for
from flask_login import current_user, login_required

from app.extensions import db
from app.models import RoadMarkingDetection, TrafficSignDetection, Video

main_bp = Blueprint("main", __name__)


@main_bp.route("/healthz")
def healthz():
    return {"status": "ok"}, 200


@main_bp.route("/robots.txt")
def robots():
    sitemap_url = f"{current_app.config['SITE_URL']}/sitemap.xml"
    body = (
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /dashboard\n"
        "Disallow: /admin/\n"
        "Disallow: /analyst/\n"
        "Disallow: /users\n"
        "Disallow: /videos\n"
        "Disallow: /setup\n"
        "Disallow: /logout\n"
        f"Sitemap: {sitemap_url}\n"
    )
    return Response(body, mimetype="text/plain")


@main_bp.route("/sitemap.xml")
def sitemap():
    public_endpoints = (
        "main.index",
        "main.about",
        "main.signs",
        "main.markings",
        "main.results",
        "auth.login",
    )
    urls = "\n".join(
        f"  <url><loc>{current_app.config['SITE_URL']}{url_for(endpoint)}</loc></url>"
        for endpoint in public_endpoints
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}\n</urlset>\n"
    )
    return Response(xml, mimetype="application/xml")

@main_bp.route("/")
def index():
    if current_user.is_authenticated:
        endpoint = "main.admin_dashboard" if current_user.is_admin() else "main.analyst_dashboard"
        return redirect(url_for(endpoint))
    return render_template("home.html")


@main_bp.route("/dashboard")
@login_required
def dashboard():
    endpoint = "main.admin_dashboard" if current_user.is_admin() else "main.analyst_dashboard"
    return redirect(url_for(endpoint))


@main_bp.route("/admin/dashboard")
@login_required
def admin_dashboard():
    if not current_user.is_admin():
        from flask import abort

        abort(403)
    videos = Video.query.order_by(Video.uploaded_at.desc()).limit(3).all()
    video_count = Video.query.count()
    return render_template(
        "dashboard.html",
        dashboard_role="Admin",
        recent_videos=videos,
        video_count=video_count,
    )


@main_bp.route("/analyst/dashboard")
@login_required
def analyst_dashboard():
    if not current_user.is_analyst():
        from flask import abort

        abort(403)
    videos = Video.query.order_by(Video.uploaded_at.desc()).limit(3).all()
    video_count = Video.query.count()
    return render_template(
        "dashboard.html",
        dashboard_role="Analyst",
        recent_videos=videos,
        video_count=video_count,
    )


@main_bp.route("/public/videos")
def public_videos():
    videos = Video.query.order_by(Video.uploaded_at.desc()).all()
    return render_template("public_videos.html", videos=videos)


@main_bp.route("/public/videos/<int:video_id>")
def public_video_watch(video_id):
    video = db.session.get(Video, video_id)
    if video is None:
        return render_template("public_videos.html", videos=[], error="This video is not available."), 404
    return render_template(
        "public_video_watch.html",
        video=video,
        detections=TrafficSignDetection.query.filter_by(video_id=video.id)
        .order_by(TrafficSignDetection.confidence.desc())
        .all(),
        markings=RoadMarkingDetection.query.filter_by(video_id=video.id)
        .order_by(RoadMarkingDetection.confidence.desc())
        .all(),
        media_type="video/mp4",
    )


@main_bp.route("/public/videos/<int:video_id>/file")
def public_video_file(video_id):
    video = db.session.get(Video, video_id)
    if video is None:
        return "Video not found.", 404
    folder = current_app.config["VIDEO_UPLOAD_FOLDER"]
    return send_from_directory(folder, video.filename, mimetype="video/mp4")


@main_bp.route("/public/videos/<int:video_id>/annotated")
def public_annotated_video(video_id):
    video = db.session.get(Video, video_id)
    if video is None:
        return "Video not found.", 404
    if not video.annotated_video_filename:
        return "Analysed video is not available yet.", 404
    folder = current_app.config["DETECTION_UPLOAD_FOLDER"]
    return send_from_directory(folder, video.annotated_video_filename, mimetype="video/mp4")


@main_bp.route("/public/detections/<int:video_id>/<path:filename>")
def public_detection_media(video_id, filename):
    detection = TrafficSignDetection.query.filter_by(video_id=video_id, annotated_filename=filename).first()
    if detection is None:
        return "Detection not found.", 404
    folder = current_app.config["DETECTION_UPLOAD_FOLDER"]
    return send_from_directory(folder, filename, mimetype="image/jpeg")


@main_bp.route("/public/markings/<int:video_id>/<path:filename>")
def public_marking_media(video_id, filename):
    detection = RoadMarkingDetection.query.filter_by(video_id=video_id, annotated_filename=filename).first()
    if detection is None:
        return "Detection not found.", 404
    folder = current_app.config["DETECTION_UPLOAD_FOLDER"]
    return send_from_directory(folder, filename, mimetype="image/jpeg")


@main_bp.route("/signs")
def signs():
    detections = TrafficSignDetection.query.order_by(TrafficSignDetection.detected_at.desc()).all()
    summary = []
    seen = {}
    for detection in detections:
        key = detection.class_name
        if key not in seen:
            seen[key] = {"count": 0, "confidence": 0.0, "video_id": detection.video_id}
        seen[key]["count"] += 1
        seen[key]["confidence"] += detection.confidence
    for label, values in seen.items():
        values["average_confidence"] = (values["confidence"] / values["count"]) if values["count"] else 0.0
        summary.append({"label": label, **values})
    return render_template("signals.html", detections=detections, summary=summary)


@main_bp.route("/markings")
def markings():
    detections = RoadMarkingDetection.query.order_by(RoadMarkingDetection.detected_at.desc()).all()
    summary = []
    seen = {}
    for detection in detections:
        key = detection.class_name
        if key not in seen:
            seen[key] = {"count": 0, "confidence": 0.0}
        seen[key]["count"] += 1
        seen[key]["confidence"] += detection.confidence
    for label, values in seen.items():
        values["average_confidence"] = (values["confidence"] / values["count"]) if values["count"] else 0.0
        summary.append({"label": label, **values})
    return render_template("markings.html", detections=detections, summary=summary)


@main_bp.route("/results")
def results():
    videos = Video.query.order_by(Video.uploaded_at.desc()).all()
    sign_total = TrafficSignDetection.query.count()
    marking_total = RoadMarkingDetection.query.count()
    return render_template(
        "results.html",
        videos=videos,
        total_videos=Video.query.count(),
        traffic_sign_total=sign_total,
        road_marking_total=marking_total,
        sign_classes=TrafficSignDetection.query.with_entities(TrafficSignDetection.class_name).distinct().all(),
        marking_classes=RoadMarkingDetection.query.with_entities(RoadMarkingDetection.class_name).distinct().all(),
    )


@main_bp.route("/about")
def about():
    return render_template("about.html")
