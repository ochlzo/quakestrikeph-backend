FROM python:3.12-slim

# Keep Python output unbuffered so logs stream to `docker logs` in real time.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app


WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (incl. the bundled GlobalSign intermediate cert).
COPY config.py scraper.py db.py celery_app.py csv_sync.py phivolcs_intermediate.pem ./


# Run as a non-root user.
RUN useradd --create-home appuser
USER appuser

# Single combined process: the worker also runs the beat scheduler. The beat
# schedule file lives in /tmp (writable, ephemeral, schedule is static).
CMD ["celery", "-A", "celery_app", "worker", "--beat", \
     "--loglevel=info", "--concurrency=1", \
     "--schedule=/tmp/celerybeat-schedule"]
