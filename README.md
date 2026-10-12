# Road Safety in Zambia

Flask application for authorized road-video management, controlled frame extraction, and pretrained traffic-sign analysis.

## Local setup

1. Create and activate a Python 3.12 virtual environment.
2. Install dependencies:

   ```text
   pip install -r requirements.txt
   ```

3. Create `.env` with a strong local-only secret:

   ```text
   SECRET_KEY=replace-with-a-local-random-secret
   DATABASE_URL=sqlite:///road_safety.db
   FLASK_DEBUG=1
   ```

4. Start locally:

   ```text
   python run.py
   ```

The existing local SQLite database, Admin account, videos, frames, and model file remain under `instance/` and are not recreated by startup.

## Production configuration

Production uses `wsgi.py` and Gunicorn:

```text
gunicorn --workers 1 --threads 4 --timeout 300 wsgi:app
```

Required environment variables:

- `SECRET_KEY`: strong random value; never commit it.
- `DATABASE_URL`: PostgreSQL connection string for a multi-instance deployment, or a SQLite URL on a persistent single-instance disk.
- `FLASK_DEBUG=0`.
- `MEDIA_STORAGE`: `filesystem` for local development or `s3` for durable hosted storage.
- `MEDIA_ROOT`: local storage root when `MEDIA_STORAGE=filesystem`.
- `S3_BUCKET`, `S3_REGION`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY`: required for `MEDIA_STORAGE=s3`.
- `S3_ENDPOINT_URL`: optional for AWS S3; set it for S3-compatible providers.
- `MODEL_DIR`: model directory; relative paths resolve from the project root.
- `TRAFFIC_SIGN_MODEL`: model filename; Render uses the tracked `yolov8s-best.pt` checkpoint.
- `TRAFFIC_SIGN_MODEL_PATH`: optional full path override for the YOLOv8 checkpoint.
- `COOKIE_SECURE=1` when the deployment is served over HTTPS.
- `MAX_UPLOAD_MB=500` to configure the upload limit.
- `VIDEO_RETENTION_HOURS=24` to configure source-video retention. Set this to a smaller positive whole number for shorter retention; each upload records its exact scheduled expiration.
- `MAX_EXTRACTED_FRAMES=500` to cap saved frames per video. Each frame is decoded and written sequentially.
- `TRAFFIC_SIGN_IMAGE_SIZE=416` to limit YOLOv8 inference input dimensions.
- `TRAFFIC_SIGN_MAX_DETECTIONS=50` to cap detections returned for each frame.
- `TRAFFIC_SIGN_CPU_THREADS=1` to limit PyTorch CPU parallelism on low-memory hosts.
- `SITE_URL`: public HTTPS origin, for example `https://roadsafety-zambia.onrender.com`.

The app accepts Render's `postgres://` or `postgresql://` URL and converts it to the psycopg SQLAlchemy dialect.

Road-marking detection processes large frames at a maximum 1280-pixel side to limit OpenCV memory use, then maps detected boxes back to the original frame coordinates.

YOLOv8 traffic-sign inference runs on CPU with streaming results, a single cached model instance, and one frame at a time. A 416-pixel inference size lowers memory use but can miss small or distant signs; adjust `TRAFFIC_SIGN_IMAGE_SIZE` when the host has adequate memory.

## Render deployment

The included `render.yaml` creates a web service, a PostgreSQL database, and a Render Cron Job that runs `python -m app.cleanup` every 15 minutes. The web and cron services share the database and private media bucket. The cron service deletes only source video files whose recorded expiration has passed; records, extracted frames, detections, and user accounts are retained. Set `VIDEO_RETENTION_HOURS` on both services to the same positive whole number.

1. Push this repository to GitHub, excluding `.env`, local databases, uploads, and generated frame/detection files.
2. Create a new Render Blueprint from the GitHub repository.
3. Configure the required S3 bucket, region, access key, and secret key in Render's environment settings for both the web and cron services. These are separate `sync: false` values in Render and must be set to identical values manually; the Blueprint intentionally does not store or expose the secrets. Set `S3_ENDPOINT_URL` for a non-AWS S3-compatible provider. Keep the bucket private; the app serves authorized or public media through its own routes.
4. The project's trained `yolov8s-best.pt` checkpoint is tracked at `instance/models/traffic_signs/yolov8s-best.pt` and is deployed with the application; no model bucket upload is required.
5. Confirm `SECRET_KEY` is generated or set as a secret environment variable and `FLASK_DEBUG=0`.
6. Open `/healthz`; it should return `{"status":"ok"}`.
7. Set `SITE_URL` to the final public Render URL, then redeploy so canonical links, `robots.txt`, and `sitemap.xml` use the public origin.
8. Confirm the `road-safety-video-cleanup` Cron Job has the same S3 credentials and database connection as the web service, then check its Render job logs for `Expired-video cleanup completed`. If your Render plan does not support Cron Jobs, run `python -m app.cleanup` from an equivalent scheduled task with the same environment variables and network access; do not use an in-process web-worker scheduler.

Existing video records without an expiration are given the configured retention period from the first startup after this update, so deployment does not immediately remove historical source files. New uploads start their retention period at upload time. Manual admin deletion removes only the original source file and retains its record and analysis history.

## Google indexing

After the Render service is live:

1. Open the public URL and confirm `/`, `/about`, `/signs`, `/markings`, `/results`, `/login`, `/robots.txt`, and `/sitemap.xml` load without login.
2. Open Google Search Console at https://search.google.com/search-console and add the public URL-prefix property.
3. Verify ownership using one of Google's offered methods.
4. Open **Sitemaps**, enter `sitemap.xml`, and submit it.
5. Use **URL Inspection**, enter the public home URL, and select **Request indexing**.

Google indexing is not immediate or guaranteed. Search Console reports discovery and indexing status; do not expect the site to appear instantly in search results.

## Existing local data

The local SQLite database and media are intentionally ignored by Git. The Render Blueprint uses PostgreSQL, and existing local SQLite data is not automatically copied into it. `ensure_database_schema()` only adds compatible missing columns/indexes; it does not migrate accounts or analysis records between databases. If local records must be published, take verified backups of both databases, copy users (including password hashes and roles), videos, frames, and detections in foreign-key order, and upload each referenced media object to the configured bucket before switching the public service. Do not delete or reset either database during that transfer, and do not commit `.env`, passwords, database files, or uploaded media to GitHub.

## Security and storage notes

- No public registration route exists.
- Login, roles, CSRF, protected video routes, extraction locks, and detection locks remain enabled.
- Uploads are limited to 500 MB and supported video extensions.
- Hosted media uses the configured private S3-compatible bucket; there is no fallback to the web service's ephemeral filesystem. The database remains PostgreSQL. Processing stages S3 objects temporarily for OpenCV and cleans up local temporary files.
- `MEDIA_STORAGE=s3` requires `S3_BUCKET`, `S3_REGION`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY`; a missing setting prevents startup instead of silently storing uploads ephemerally. Configure credentials as Render secrets. Configure bucket access so only the application credentials can read/write objects.
- Render is configured to run the project's trained YOLOv8 traffic-sign model rather than the tracked ONNX model. The trained checkpoint is tracked with the application; only files tracked by Git are deployed with this repository.
- `db.create_all()` only creates missing tables; it does not migrate existing schemas or transfer local data.