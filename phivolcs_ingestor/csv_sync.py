"""Deduplication and atomic updates for the local historical catalog CSV."""

import logging
import os
import pandas as pd

from config import HISTORICAL_CSV_PATH

log = logging.getLogger(__name__)


def sync_events_to_csv(events: list[dict]) -> int:
    """Append new events to the local CSV, deduplicate, and write atomically.

    Returns the number of new events appended to the CSV.
    """
    if not events:
        return 0

    csv_path = HISTORICAL_CSV_PATH

    # 1. Create a DataFrame for new events (excluding internal fields like 'event_time')
    columns = [
        "id",
        "Date-Time",
        "Latitude",
        "Longitude",
        "Depth",
        "Magnitude",
        "Location",
        "Month",
        "Year",
    ]
    new_rows = []
    for e in events:
        # Create a dict with only the standard CSV columns
        row = {col: e.get(col) for col in columns}
        new_rows.append(row)

    new_df = pd.DataFrame(new_rows)

    # Convert numeric columns to float to ensure reliable deduplication
    numeric_cols = ["Latitude", "Longitude", "Depth", "Magnitude"]
    for col in numeric_cols:
        new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

    # 2. Load existing CSV if it exists
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            existing_df = pd.read_csv(csv_path)
            for col in numeric_cols:
                existing_df[col] = pd.to_numeric(existing_df[col], errors="coerce")

            initial_count = len(existing_df)
        except Exception as err:
            log.error(
                "Failed to read existing CSV at %s: %s. Overwriting with new events.",
                csv_path,
                err,
            )
            existing_df = pd.DataFrame(columns=columns)
            initial_count = 0
    else:
        existing_df = pd.DataFrame(columns=columns)
        initial_count = 0

    # 3. Concatenate existing and new records
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    # 4. Deduplicate based on the unique event signature:
    # (Date-Time, Latitude, Longitude, Depth, Magnitude).
    # keep='first' ensures we keep the older records (and their original random IDs).
    combined_df.drop_duplicates(
        subset=["Date-Time", "Latitude", "Longitude", "Depth", "Magnitude"],
        keep="first",
        inplace=True,
    )

    new_inserted_count = len(combined_df) - initial_count

    # 5. Write atomically using a temporary file (guaranteed safe on Linux/UNIX filesystems)
    temp_path = f"{csv_path}.tmp"
    try:
        combined_df.to_csv(temp_path, index=False)
        os.replace(temp_path, csv_path)
        log.info(
            "Successfully updated CSV at %s. Added %d new events.",
            csv_path,
            new_inserted_count,
        )
    except Exception as err:
        log.error("Failed to write updated CSV atomically to %s: %s", csv_path, err)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise err

    return new_inserted_count
