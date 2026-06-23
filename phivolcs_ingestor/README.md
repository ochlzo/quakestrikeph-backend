# PHIVOLCS Earthquake Ingestor

A Celery beat scheduler, packaged as a Docker container, that polls the PHIVOLCS
[Latest Earthquake Information](https://tsunami.phivolcs.dost.gov.ph/EQLatest.html)
page every 5 minutes and inserts any new events into the Supabase
`public."RawEarthquakeEvents"` table.

## How it works

- **`scraper.py`** — downloads the page (decoded as Windows-1252 to preserve the
  `°` in location strings), finds the events table by its `Latitude` header, and
  normalizes each row to the table schema (floats for lat/lon/magnitude, a
  float-formatted `Depth` string, the verbatim `Date - Time` text, and a naive
  Philippine-local `event_time`).
- **`db.py`** — opens a short-lived Supabase connection per poll and bulk-inserts
  with `ON CONFLICT ON CONSTRAINT unique_event DO NOTHING`. Each row gets a random
  UUID `id`; deduplication is handled entirely by the `unique_event` constraint
  on `(Date-Time, Latitude, Longitude, Depth, Magnitude)`, so re-scraping the same
  events on every poll inserts only genuinely new ones. It counts inserts via
  `RETURNING id`.
- **`celery_app.py`** — one combined `worker --beat` process. Beat fires
  `phivolcs.poll_latest` every `POLL_INTERVAL_SECONDS`; the worker runs it. No
  result backend (fire-and-forget), which keeps Upstash command usage low.

Times are stored as **Philippine local time** (the table's `event_time` is
`timestamp without time zone`), matching the human-readable `Date - Time` text.

## Configuration

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
| --- | --- |
| `DATABASE_URL` | Supabase Postgres pooler DSN. |
| `REDIS_URL` | Upstash **native Redis** `rediss://...` URL (from the Upstash console's Redis/TLS connect tab — *not* the REST URL/token). |
| `POLL_INTERVAL_SECONDS` | Poll interval; defaults to `300` (5 min). |

## Run

```bash
cd phivolcs_ingestor
cp .env.example .env   # then edit .env
docker compose up --build -d
docker compose logs -f
```

You should see a `poll_latest: N scraped, M new` line on each cycle. On the very
first run against an already-populated table, `M` will typically be 0 (or only
the handful of events newer than what's already stored).

### Trigger a poll immediately (without waiting for the next tick)

```bash
docker compose exec ingestor \
  celery -A celery_app call phivolcs.poll_latest
```

## Notes / variations

- **Bundled broker instead of Upstash:** add a `redis:7-alpine` service to
  `docker-compose.yml`, point `REDIS_URL` at `redis://redis:6379/0`, and the code
  automatically skips the TLS settings (only `rediss://` URLs enable them).
- **Separate beat and worker:** for multiple workers, split into two services —
  one `celery -A celery_app beat` and one or more `celery -A celery_app worker`.
  The single combined process here is the right choice for one instance.
