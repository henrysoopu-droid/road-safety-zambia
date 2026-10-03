import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, flash, redirect, request, url_for
from sqlalchemy import MetaData, inspect, text
from sqlalchemy.schema import CreateTable
from werkzeug.middleware.proxy_fix import ProxyFix

from app.extensions import csrf, db, login_manager

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def sqlite_uri(instance_path, database_url):
    if not database_url.startswith("sqlite:///"):
        return database_url

    db_name = database_url.replace("sqlite:///", "", 1)
    if not db_name:
        db_path = Path(instance_path) / "app.db"
        return "sqlite:///" + db_path.resolve().as_posix()

    db_path = Path(db_name)
    if db_path.is_absolute():
        return "sqlite:///" + db_path.resolve().as_posix()

    instance_dir = Path(instance_path).resolve()
    if db_name.startswith("instance/"):
        db_path = (instance_dir.parent / db_name).resolve()
    else:
        db_path = (instance_dir / db_name).resolve()

    return "sqlite:///" + db_path.as_posix()


def remove_detection_filename_uniqueness():
    inspector = inspect(db.engine)
    dialect = db.engine.dialect
    preparer = dialect.identifier_preparer
    table_names = ("traffic_sign_detections", "road_marking_detections")

    for table_name in table_names:
        if table_name not in inspector.get_table_names():
            continue
        unique_constraints = [
            constraint
            for constraint in inspector.get_unique_constraints(table_name)
            if constraint.get("column_names") == ["annotated_filename"]
        ]
        if not unique_constraints:
            continue

        quoted_table = preparer.quote(table_name)
        if dialect.name == "sqlite":
            source_table = db.metadata.tables[table_name]
            replacement_name = f"{table_name}_replacement"
            replacement_metadata = MetaData()
            for referenced_table in ("videos", "frames"):
                db.metadata.tables[referenced_table].to_metadata(
                    replacement_metadata
                )
            replacement_table = source_table.to_metadata(
                replacement_metadata, name=replacement_name
            )
            quoted_replacement = preparer.quote(replacement_name)
            quoted_columns = ", ".join(
                preparer.quote(column.name) for column in source_table.columns
            )
            with db.engine.begin() as connection:
                connection.execute(CreateTable(replacement_table))
                connection.execute(
                    text(
                        f"INSERT INTO {quoted_replacement} ({quoted_columns}) "
                        f"SELECT {quoted_columns} FROM {quoted_table}"
                    )
                )
                connection.execute(text(f"DROP TABLE {quoted_table}"))
                connection.execute(
                    text(
                        f"ALTER TABLE {quoted_replacement} "
                        f"RENAME TO {quoted_table}"
                    )
                )
                for index in source_table.indexes:
                    index.create(connection, checkfirst=True)
        elif dialect.name == "postgresql":
            with db.engine.begin() as connection:
                for constraint in unique_constraints:
                    name = constraint.get("name")
                    if name:
                        quoted_constraint = preparer.quote(name)
                        connection.execute(
                            text(
                                f"ALTER TABLE {quoted_table} DROP CONSTRAINT "
                                f"{quoted_constraint}"
                            )
                        )


def ensure_database_schema(app):
    with app.app_context():
        db.create_all()
        inspector = inspect(db.engine)
        if "videos" in inspector.get_table_names():
            video_columns = {column["name"] for column in inspector.get_columns("videos")}
            for column_name, column_definition in {
                "processing_status": "VARCHAR(30) NOT NULL DEFAULT 'uploaded'",
                "analysis_status": "VARCHAR(30) NOT NULL DEFAULT 'not_started'",
                "annotated_video_filename": "VARCHAR(255)",
            }.items():
                if column_name not in video_columns:
                    db.session.execute(text(f"ALTER TABLE videos ADD COLUMN {column_name} {column_definition}"))
        if "road_marking_detections" not in inspector.get_table_names():
            db.create_all()
        remove_detection_filename_uniqueness()
        db.session.commit()


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

    database_url = os.getenv("DATABASE_URL", "sqlite:///instance/road_safety.db")
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
    app.config["MODEL_DIR"] = str(model_folder)
    app.config["TRAFFIC_SIGN_MODEL_PATH"] = str(
        model_folder / "traffic_signs" / os.getenv("TRAFFIC_SIGN_MODEL", "yolov8s-best.pt")
    )
    app.config["SPEED_HUMP_MODEL_PATH"] = str(
        model_folder / "speed_humps" / os.getenv("SPEED_HUMP_MODEL", "speed_hump_model.pt")
    )
    app.config["TRAFFIC_SIGN_CONFIDENCE"] = float(os.getenv("TRAFFIC_SIGN_CONFIDENCE", "0.40"))
    app.config["SPEED_HUMP_CONFIDENCE"] = float(os.getenv("SPEED_HUMP_CONFIDENCE", "0.40"))
    app.config["MODEL_CONFIG"] = {
        "traffic_sign": {
            "name": os.getenv("TRAFFIC_SIGN_MODEL", "yolov8s-best.pt"),
            "format": "YOLOv8 PT weights",
            "inference": "Ultralytics YOLO",
            "classes": [
                "Stop",
                "Red Light",
                "Green Light",
                "Speed Limit 10",
                "Speed Limit 100",
                "Speed Limit 110",
                "Speed Limit 120",
                "Speed Limit 20",
                "Speed Limit 30",
                "Speed Limit 40",
                "Speed Limit 50",
                "Speed Limit 60",
                "Speed Limit 70",
                "Speed Limit 80",
                "Speed Limit 90",
            ],
        },
        "speed_hump": {
            "name": os.getenv("SPEED_HUMP_MODEL", "speed_hump_model.pt"),
            "format": "YOLO weights when available",
            "inference": "Ultralytics YOLO",
            "classes": ["Speed Hump", "Speed Bump"],
        },
        "road_marking": {
            "name": os.getenv("ROAD_MARKING_MODEL", "road_marking_model.onnx"),
            "format": "OpenCV/vision pipeline when available",
            "inference": "OpenCV image processing",
            "classes": ["lane marking", "zebra crossing", "arrow marking"],
        },
    }
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

    ensure_database_schema(app)
    return app
