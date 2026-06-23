"""Supabase insert layer for scraped earthquake events.

A fresh short-lived connection is opened per poll (every 5 min) rather than held
open, which plays nicely with the Supabase transaction pooler and avoids stale
sockets in the long-running Celery worker. Deduplication is delegated entirely
to the database: every scraped row is sent with a random UUID primary key and
``ON CONFLICT ON CONSTRAINT unique_event DO NOTHING``, so re-scraping the same
event (which happens on every poll, since the page lists the whole recent
catalog) is a no-op and only genuinely new events are inserted. ``RETURNING id``
plus ``execute_values(fetch=True)`` lets us count exactly how many rows were new.
"""

import logging

import psycopg2
from psycopg2.extras import execute_values

from config import DATABASE_URL

log = logging.getLogger(__name__)

# Column order shared by the INSERT statement and the row tuples below.
_COLUMNS = (
    "id",
    '"Date-Time"',
    '"Latitude"',
    '"Longitude"',
    '"Depth"',
    '"Magnitude"',
    '"Location"',
    '"Month"',
    '"Year"',
    "event_time",
)

_INSERT_SQL = f"""
    INSERT INTO public."RawEarthquakeEvents" ({", ".join(_COLUMNS)})
    VALUES %s
    ON CONFLICT ON CONSTRAINT unique_event DO NOTHING
    RETURNING id
"""


def _event_to_row(event: dict) -> tuple:
    return (
        event["id"],
        event["Date-Time"],
        event["Latitude"],
        event["Longitude"],
        event["Depth"],
        event["Magnitude"],
        event["Location"],
        event["Month"],
        event["Year"],
        event["event_time"],
    )


def insert_events(events: list[dict]) -> int:
    """Insert any new events, returning the number of rows actually inserted."""
    if not events:
        return 0

    rows = [_event_to_row(e) for e in events]

    connection = psycopg2.connect(DATABASE_URL)
    try:
        with connection:
            with connection.cursor() as cur:
                inserted = execute_values(
                    cur, _INSERT_SQL, rows, page_size=500, fetch=True
                )
        count = len(inserted)
    finally:
        connection.close()

    log.info("Inserted %d new events (%d scraped)", count, len(events))
    return count
