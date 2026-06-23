"""Celery application: beat schedule + the poll task.

A single combined ``worker --beat`` process runs this (see Dockerfile). Beat
fires ``phivolcs.poll_latest`` every ``POLL_INTERVAL_SECONDS`` (default 300s);
the worker executes it: scrape the PHIVOLCS page, upsert new events into
Supabase. There is no result backend -- the task is fire-and-forget, which also
keeps Upstash command usage minimal.

When ``REDIS_URL`` is a ``rediss://`` endpoint (Upstash), TLS is enabled. We set
``ssl_cert_reqs=CERT_NONE`` because Celery's redis transport requires the option
to be present for rediss URLs; the connection is still encrypted.
"""

import logging
import ssl

from celery import Celery

from config import POLL_INTERVAL_SECONDS, REDIS_URL

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Celery("phivolcs", broker=REDIS_URL)

app.conf.update(
    timezone="Asia/Manila",
    enable_utc=True,
    # Avoid the Celery 5.x startup warning and survive a broker not-yet-ready.
    broker_connection_retry_on_startup=True,
    # Fire-and-forget: no result backend needed.
    task_ignore_result=True,
    beat_schedule={
        "poll-phivolcs-latest": {
            "task": "phivolcs.poll_latest",
            "schedule": float(POLL_INTERVAL_SECONDS),
        }
    },
)

# TLS for Upstash / any rediss:// broker.
if REDIS_URL.startswith("rediss://"):
    app.conf.broker_use_ssl = {"ssl_cert_reqs": ssl.CERT_NONE}


@app.task(name="phivolcs.poll_latest")
def poll_latest() -> int:
    """Scrape the latest PHIVOLCS events and insert any new ones into Supabase."""
    # Imported lazily so the worker boots even if a transient import/network
    # issue exists, and to keep the task module light.
    from csv_sync import sync_events_to_csv
    from db import insert_events
    from scraper import fetch_events



    events = fetch_events()
    inserted = insert_events(events)
    csv_inserted = sync_events_to_csv(events)
    log.info(
        "poll_latest: %d scraped, %d new in Supabase, %d new in local CSV",
        len(events),
        inserted,
        csv_inserted,
    )
    return inserted

