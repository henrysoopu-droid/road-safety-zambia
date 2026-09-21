import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, flash, redirect, request, url_for
from werkzeug.middleware.proxy_fix import ProxyFix

from app.extensions import csrf, db, login_manager

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def sqlite_uri(instance_path, database_url):
    if not database_url.startswith("sqlite:///"):
        return database_url
    db_name = database_url.replace("sqlite:///", "", 1)
    db_path = Path(db_name)
    if not db_path.is_absolute():
        db_path = Path(instance_path) / db_path
    return "sqlite:///" + db_path.resolve().as_posix()


def create_app():
    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder="templates",
        static_folder="static",
    )
    os.makedirs(app.instance_path, exist_ok=True)

    secret_key = os.getenv("SECRET_KEY")
    if not secret_key:
        raise RuntimeError("SECRET_KEY is not set. Configure it in the deployment environment.")

    debug = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "500"))
    max_extracted_frames = int(os.getenv("MAX_EXTRACTED_FRAMES", "5000"))

    database_url = os.getenv("DATABASE_URL", "sqlite:///road_safety.db")
    if database_url.startswith("postgres://"):
        database_url = "postgresql+psycopg://" + database_url[len("postgres://") :]
    elif database_url.startswith("postgresql://"):
        database_url = "postgresql+psycopg://" + database_url[len("postgresql://") :]
    database_url = sqlite_uri(app.instance_path, database_url)

    media_root = Path(os.getenv("MEDIA_ROOT", Path(app.instance_path) / "uploads"))
    video_upload_folder = media_root / "videos"
    video_upload_folder.mkdir(parents=True, exist_ok=True)
    frame_upload_folder = media_root / "frames"
    frame_upload_folder.mkdir(parents=True, exist_ok=True)
    detection_upload_folder = media_root / "detections"
    detection_upload_folder.mkdir(parents=True, exist_ok=True)
    model_folder = Path(os.getenv("MODEL_DIR", Path(app.instance_path) / "models"))
    model_folder.mkdir(parents=True, exist_ok=True)

    app.config["SECRET_KEY"] = secret_key
    app.config["SITE_URL"] = os.getenv("SITE_URL", "http://127.0.0.1:5000").rstrip("/")
    app.config["DEBUG"] = debug
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = os.getenv("COOKIE_SECURE", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["WTF_CSRF_ENABLED"] = True
    app.config["VIDEO_UPLOAD_FOLDER"] = str(video_upload_folder)
    app.config["FRAME_UPLOAD_FOLDER"] = str(frame_upload_folder)
    app.config["DETECTION_UPLOAD_FOLDER"] = str(detection_upload_folder)
    app.config["TRAFFIC_SIGN_MODEL"] = str(
        model_folder / os.getenv("TRAFFIC_SIGN_MODEL", "yolov5s.onnx")
    )
    app.config["MAX_EXTRACTED_FRAMES"] = max_extracted_frames
    app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024
    app.config["MAX_UPLOAD_MB"] = max_upload_mb

    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    @app.errorhandler(413)
    def request_entity_too_large(error):
        if request.path.startswith("/videos"):
            flash(
                f"The video is too large. Maximum upload size is {app.config['MAX_UPLOAD_MB']} MB.",
                "error",
            )
            return redirect(url_for("videos.library"))
        return "Request is too large.", 413

    @app.errorhandler(500)
    def internal_server_error(error):
        app.logger.exception("Unhandled application error")
        return "An internal server error occurred. Please try again later.", 500

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    from app.models import Frame, TrafficSignDetection, User, Video  # noqa: F401

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    from app.auth import auth_bp
    from app.main import main_bp
    from app.videos import videos_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(videos_bp)

    with app.app_context():
        db.create_all()

    return app
