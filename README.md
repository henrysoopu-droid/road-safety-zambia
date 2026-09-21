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
- `MEDIA_ROOT`: persistent storage root for uploaded videos, extracted frames, and annotated detection images.
- `MODEL_DIR`: directory containing `yolov5s.onnx`.
- `TRAFFIC_SIGN_MODEL`: model filename, normally `yolov5s.onnx`.
- `COOKIE_SECURE=1` when the deployment is served over HTTPS.
- `MAX_UPLOAD_MB=500` to configure the upload limit.
- `MAX_EXTRACTED_FRAMES=5000` to cap controlled frame extraction.
- `SITE_URL`: public HTTPS origin, for example `https://roadsafety-zambia.onrender.com`.

The app accepts Render's `postgres://` or `postgresql://` URL and converts it to the psycopg SQLAlchemy dialect.

## Render deployment

The included `render.yaml` creates a web service and PostgreSQL database. In the Render dashboard:

1. Push this repository to GitHub, excluding `.env`, local databases, uploads, and generated frame/detection files.
2. Create a new Render Blueprint from the GitHub repository.
3. Attach a Render persistent disk mounted at `/var/data` to the web service.
4. Keep `MODEL_DIR=instance/models`; the tracked `yolov5s.onnx` file is deployed with the application code.
5. Keep `MEDIA_ROOT=/var/data/uploads` so videos and derived images survive deploys.
6. Confirm `SECRET_KEY` is generated or set as a secret environment variable and `FLASK_DEBUG=0`.
7. Open `/healthz`; it should return `{"status":"ok"}`.
8. Set `SITE_URL` to the final public Render URL, then redeploy so canonical links, `robots.txt`, and `sitemap.xml` use the public origin.

## Google indexing

After the Render service is live:

1. Open the public URL and confirm `/`, `/about`, `/signs`, `/markings`, `/results`, `/login`, `/robots.txt`, and `/sitemap.xml` load without login.
2. Open Google Search Console at https://search.google.com/search-console and add the public URL-prefix property.
3. Verify ownership using one of Google's offered methods.
4. Open **Sitemaps**, enter `sitemap.xml`, and submit it.
5. Use **URL Inspection**, enter the public home URL, and select **Request indexing**.

Google indexing is not immediate or guaranteed. Search Console reports discovery and indexing status; do not expect the site to appear instantly in search results.

## Existing local data

The local SQLite database and media are intentionally ignored by Git. A new Render PostgreSQL database starts empty; it does not automatically contain the local Admin, Nationalist Road video, or 44 frames. Before presenting the production instance, migrate/import the existing data and media through a controlled process, or use a persistent SQLite disk for a single-instance demonstration. Do not commit `.env`, passwords, database files, or uploaded media to GitHub.

## Security and storage notes

- No public registration route exists.
- Login, roles, CSRF, protected video routes, extraction locks, and detection locks remain enabled.
- Uploads are limited to 500 MB and supported video extensions.
- Render's default filesystem is ephemeral. A persistent disk or object storage is required for videos, frames, annotated images, and model weights.
- `db.create_all()` only creates missing tables; it does not migrate existing schemas or transfer local data.