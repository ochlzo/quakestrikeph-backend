"""Scrape the PHIVOLCS "Latest Earthquake Information" page.

The page is a static, FrontPage-generated HTML document (~7 MB of nested
``MsoNormalTable`` markup) served in Windows-1252, so the degree sign in the
location strings (``29° E``) arrives as byte 0xB0 and must be decoded as
``cp1252`` to match the existing catalog. We locate the one events table by its
``Latitude`` header rather than by position/class, then keep every row whose
first cell parses as a PHIVOLCS timestamp -- that naturally skips header and
spacer rows without hard-coding indices.

TLS note: the PHIVOLCS server sends an *incomplete* certificate chain (the leaf
only, omitting the ``GlobalSign RSA OV SSL CA 2018`` intermediate). Windows masks
this by fetching the missing intermediate via AIA, but Linux/OpenSSL/certifi do
not, so the request would fail inside the container. We fix it securely -- not by
disabling verification -- by verifying against certifi's roots plus the bundled
intermediate (``phivolcs_intermediate.pem``).

Each returned dict is already normalized to the ``RawEarthquakeEvents`` schema:
floats for the numeric columns, a float-formatted ``Depth`` string ("011" ->
"11.0") to match the historical CSV, the verbatim ``Date - Time`` text, and a
naive Philippine-local ``event_time`` datetime.
"""

import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import certifi
import requests
from bs4 import BeautifulSoup

from config import HTTP_TIMEOUT_SECONDS, PHIVOLCS_LATEST_URL

log = logging.getLogger(__name__)

# GlobalSign intermediate that PHIVOLCS fails to serve, shipped alongside this
# module. Combined with certifi's roots into one bundle so requests can verify
# the chain on Linux.
_INTERMEDIATE_PEM = Path(__file__).with_name("phivolcs_intermediate.pem")


def _ca_bundle() -> str:
    bundle = Path(tempfile.gettempdir()) / "phivolcs_ca_bundle.pem"
    bundle.write_text(
        Path(certifi.where()).read_text() + "\n" + _INTERMEDIATE_PEM.read_text()
    )
    return str(bundle)


_CA_BUNDLE = _ca_bundle()

# Same format string used by seis_model_suite/feature_engineering.py, e.g.
# "23 June 2026 - 12:53 AM".
PHIVOLCS_TIME_FORMAT = "%d %B %Y - %I:%M %p"

_USER_AGENT = (
    "Mozilla/5.0 (compatible; quakestrikeph-ingestor/1.0; "
    "+https://tsunami.phivolcs.dost.gov.ph/EQLatest.html)"
)


def _fetch_html() -> str:
    """Download the latest-events page and decode it as Windows-1252."""
    resp = requests.get(
        PHIVOLCS_LATEST_URL,
        headers={"User-Agent": _USER_AGENT},
        timeout=HTTP_TIMEOUT_SECONDS,
        verify=_CA_BUNDLE,
    )
    resp.raise_for_status()
    # Force cp1252: the page has no reliable charset declaration and contains
    # 0xB0 degree bytes that are invalid UTF-8.
    return resp.content.decode("cp1252", "replace")


def _find_events_table(soup: BeautifulSoup):
    """Return the <table> that holds the event rows (identified by its header)."""
    for table in soup.find_all("table"):
        if "Latitude" in table.get_text():
            return table
    return None


def _row_to_event(cells: list[str]) -> dict | None:
    """Convert one 6-cell row into a normalized event dict, or None to skip.

    Cell order on the page: Date-Time, Latitude, Longitude, Depth, Mag, Location.
    """
    date_time_text, lat_text, lon_text, depth_text, mag_text, location = cells

    # The first cell must be a real timestamp; header/spacer rows fail here and
    # are silently skipped.
    try:
        event_time = datetime.strptime(date_time_text, PHIVOLCS_TIME_FORMAT)
    except ValueError:
        return None

    # Latitude/Longitude/Magnitude are required numeric columns; drop the row if
    # any is unparseable (malformed/partial rows do occur).
    try:
        latitude = float(lat_text)
        longitude = float(lon_text)
        magnitude = float(mag_text)
    except ValueError:
        log.warning("Skipping row with non-numeric lat/lon/mag: %s", cells)
        return None

    # Depth is a text column but stored float-formatted ("011" -> "11.0") so it
    # matches the historical catalog and the unique_event constraint.
    try:
        depth = str(float(depth_text))
    except ValueError:
        depth = depth_text.strip()

    return {
        "id": str(uuid.uuid4()),
        "Date-Time": date_time_text,
        "Latitude": latitude,
        "Longitude": longitude,
        "Depth": depth,
        "Magnitude": magnitude,
        "Location": location or None,
        "Month": event_time.strftime("%B"),
        "Year": event_time.year,
        "event_time": event_time,
    }


def fetch_events() -> list[dict]:
    """Scrape and return all current events from the PHIVOLCS latest page."""
    html = _fetch_html()
    soup = BeautifulSoup(html, "html.parser")

    table = _find_events_table(soup)
    if table is None:
        raise RuntimeError(
            "Could not locate the earthquake events table on the PHIVOLCS page; "
            "the page layout may have changed."
        )

    events: list[dict] = []
    for row in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        if len(cells) != 6:
            continue
        event = _row_to_event(cells)
        if event is not None:
            events.append(event)

    log.info("Scraped %d events from PHIVOLCS latest page", len(events))
    return events
