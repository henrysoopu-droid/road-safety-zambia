from flask import current_app

from app import create_app
from app.extensions import db
from app.media_storage import media_storage as get_media_storage
from app.models import Video, utc_now


def cleanup_expired_videos(now=None):
    now = now or utc_now()
    expired_videos = (
        Video.query.filter(
            Video.expires_at.isnot(None),
            Video.expires_at <= now,
            Video.video_file_deleted_at.is_(None),
            Video.processing_status != "processing",
        )
        .order_by(Video.expires_at)
        .all()
    )
    deleted_count = 0
    failed_video_ids = []

    for video in expired_videos:
        video_id = video.id
        try:
            get_media_storage().delete("videos", video.filename)
            video.video_file_deleted_at = now
            db.session.commit()
            deleted_count += 1
            current_app.logger.info(
                "Expired source video file deleted for video %s", video.id
            )
        except Exception:
            db.session.rollback()
            failed_video_ids.append(video_id)
            current_app.logger.exception(
                "Could not clean up expired source video file for video %s",
                video_id,
            )

    if failed_video_ids:
        raise RuntimeError(
            "Expired-video cleanup failed for "
            f"{len(failed_video_ids)} video file(s); see preceding error logs."
        )
    return deleted_count


def main():
    app = create_app()
    with app.app_context():
        deleted_count = cleanup_expired_videos()
        app.logger.info(
            "Expired-video cleanup completed: %s file(s) deleted", deleted_count
        )


if __name__ == "__main__":
    main()
