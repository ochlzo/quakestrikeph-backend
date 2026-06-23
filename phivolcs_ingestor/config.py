"""Runtime configuration, loaded once from the environment.

All secrets (Supabase DSN, Upstash Redis URL) come from env vars so nothing
sensitive is baked into the image. ``python-dotenv`` lets the same code run
locally from a ``.env`` file and unchanged inside the container.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Supabase Postgres connection string (transaction pooler DSN).
DATABASE_URL = os.environ["DATABASE_URL"]

# Native Redis (TCP/TLS) URL used as the Celery broker, e.g. an Upstash
# ``rediss://default:<token>@<host>:6379`` endpoint. NOT the REST URL.
REDIS_URL = os.environ["REDIS_URL"]

# How often the beat scheduler fires the poll task, in seconds.
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "300"))

# Source page listing the latest PHIVOLCS earthquake events.
PHIVOLCS_LATEST_URL = os.getenv(
    "PHIVOLCS_LATEST_URL",
    "https://tsunami.phivolcs.dost.gov.ph/EQLatest.html",
)

# Network timeout (seconds) for the scrape request.
HTTP_TIMEOUT_SECONDS = int(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))

# Path to the local historical catalog CSV file.
HISTORICAL_CSV_PATH = os.getenv(
    "HISTORICAL_CSV_PATH",
    os.path.join(os.path.dirname(__file__), "phivolcs_earthquake_2018_2026.csv"),
)

