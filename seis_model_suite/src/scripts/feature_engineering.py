"""Single source of truth for SEIS aftershock feature engineering.

Both the training-dataset builder and every inference path compute their model
features from this module, so there is exactly one definition of each feature and
no risk of the train/serve drift that historically caused skew.

Two builder styles share the same primitives and constants:

  * Batch builders (``add_*``) -- vectorized over a whole sorted catalog. Used by
    ``build_training_dataset.py`` to label ~100k events quickly.
  * Per-event builders (``compute_*`` / ``build_prediction_features``) -- compute
    one event's features against prior history. Used by every ``predict_aftershock``
    path at serving time.

Parent / eta features are the one intentional split: at training time they come
from the authoritative C++ Zaliapin-Ben-Zion clustering (``add_parent_features``
reads the precomputed ``parent_id_key`` / ``eta`` columns), while at serving time
``compute_parent_features`` re-derives the nearest-neighbor parent in eta space
using the SAME eta formula and constants defined here. The two are verified to
agree (corr 1.000) because they run on the same catalog with the same parameters;
``investigate_feature_skew.py`` is the regression guard.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd


# --- Shared constants -------------------------------------------------------
PHIVOLCS_TIME_FORMAT = "%d %B %Y - %I:%M %p"
LOCAL_RADII_KM = [10.0, 25.0, 50.0, 100.0]
RECENT_WINDOWS_DAYS = [1, 7, 30]
NEAREST_RECENT_WINDOW_DAYS = 30

DEFAULT_HISTORICAL_CSV = Path("dataset/phivolcs_earthquake_2018_2026.csv")
DEFAULT_MIN_MAGNITUDE = 1.0
DEFAULT_B_VALUE = 1.0
DEFAULT_FRACTAL_DIMENSION = 1.6
DEFAULT_LOG10_ETA0 = -5.468679834899335
SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0

RAW_COLUMN_MAP = {
    "Date-Time": "origin_time",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Depth": "depth_km",
    "Magnitude": "magnitude",
}


# --- Catalog / geometry primitives ------------------------------------------
def parse_origin_time(series):
    parsed = pd.to_datetime(series, format=PHIVOLCS_TIME_FORMAT, errors="coerce")
    if parsed.isna().any():
        fallback = pd.to_datetime(series, errors="coerce")
        parsed = parsed.fillna(fallback)
    return parsed


def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(a))


def id_key(value):
    if pd.isna(value):
        return pd.NA
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        text = str(value).strip()
        return text if text else pd.NA


def normalize_raw_catalog(df):
    renamed = df.rename(columns={source: target for source, target in RAW_COLUMN_MAP.items()})
    required = {"origin_time", "latitude", "longitude", "depth_km", "magnitude"}
    missing = sorted(required - set(renamed.columns))
    if missing:
        raise ValueError(f"CSV is missing required raw columns: {missing}")

    normalized = renamed[list(required)].copy()
    normalized["event_time"] = parse_origin_time(normalized["origin_time"])
    for column in ["latitude", "longitude", "depth_km", "magnitude"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    before = len(normalized)
    normalized = normalized.dropna(
        subset=["event_time", "latitude", "longitude", "depth_km", "magnitude"]
    )
    if normalized.empty:
        raise ValueError("No usable historical rows after parsing raw catalog.")
    if len(normalized) != before:
        import sys

        skipped = before - len(normalized)
        print(f"Warning: skipped {skipped} malformed historical rows.", file=sys.stderr)

    return normalized.sort_values("event_time", kind="mergesort").reset_index(drop=True)


def filter_history_for_prediction(history, event_time, minimum_magnitude):
    history = history[
        (history["event_time"] < event_time)
        & (history["magnitude"] >= minimum_magnitude)
    ].copy()
    return history.sort_values("event_time", kind="mergesort").reset_index(drop=True)


def load_feature_columns(feature_columns_path):
    feature_columns_path = Path(feature_columns_path)
    if not feature_columns_path.exists():
        raise FileNotFoundError(f"Feature columns file does not exist: {feature_columns_path}")
    return [
        line.strip()
        for line in feature_columns_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- Per-event builders (serving) -------------------------------------------
def compute_parent_features(history, event, b_value, fractal_dimension, log10_eta0):
    """Re-derive the nearest-neighbor (eta-minimizing) parent for one event.

    Mirrors the C++ Zaliapin-Ben-Zion clustering's parent assignment using the
    same eta = t_years * dist_km**d * 10**(-b*m) metric, so the serving features
    match the training features sourced from the clustering.
    """
    defaults = {
        "eta": np.nan,
        "log10_eta": np.nan,
        "is_strong_link": 0,
        "has_parent": 0,
        "parent_time_gap_days": np.nan,
        "parent_distance_km": np.nan,
        "parent_magnitude": np.nan,
        "parent_depth_km": np.nan,
    }
    if history.empty:
        return defaults

    seconds = (event["event_time"] - history["event_time"]).dt.total_seconds().to_numpy()
    valid_time = seconds > 0
    if not valid_time.any():
        return defaults

    candidates = history.loc[valid_time].copy()
    seconds = seconds[valid_time]
    distances = haversine_km(
        event["latitude"],
        event["longitude"],
        candidates["latitude"].to_numpy(dtype=float),
        candidates["longitude"].to_numpy(dtype=float),
    )
    valid_distance = distances > 0.0
    if not valid_distance.any():
        return defaults

    candidates = candidates.loc[valid_distance].reset_index(drop=True)
    seconds = seconds[valid_distance]
    distances = distances[valid_distance]
    years = seconds / SECONDS_PER_YEAR
    log_eta = (
        np.log10(years)
        + fractal_dimension * np.log10(distances)
        - b_value * candidates["magnitude"].to_numpy(dtype=float)
    )
    best_position = int(np.nanargmin(log_eta))
    best_log_eta = float(log_eta[best_position])
    parent = candidates.iloc[best_position]

    return {
        "eta": float(10.0 ** best_log_eta),
        "log10_eta": best_log_eta,
        "is_strong_link": int(best_log_eta < log10_eta0),
        "has_parent": 1,
        "parent_time_gap_days": float(seconds[best_position] / 86400.0),
        "parent_distance_km": float(distances[best_position]),
        "parent_magnitude": float(parent["magnitude"]),
        "parent_depth_km": float(parent["depth_km"]),
    }


def compute_global_history_features(history, event_time):
    features = {}
    for days in RECENT_WINDOWS_DAYS:
        start_time = event_time - pd.Timedelta(days=days)
        features[f"events_past_{days}d"] = int(
            ((history["event_time"] >= start_time) & (history["event_time"] < event_time)).sum()
        )
    return features


def compute_local_history_features(history, event):
    features = {}
    for days in RECENT_WINDOWS_DAYS:
        for radius in LOCAL_RADII_KM:
            radius_token = int(radius)
            features[f"local_events_{radius_token}km_past_{days}d"] = 0
            features[f"local_max_mag_{radius_token}km_past_{days}d"] = np.nan
            features[f"local_log10_energy_{radius_token}km_past_{days}d"] = np.nan

    features[
        f"nearest_recent_event_distance_km_past_{NEAREST_RECENT_WINDOW_DAYS}d"
    ] = np.nan
    features[
        f"nearest_recent_event_magnitude_past_{NEAREST_RECENT_WINDOW_DAYS}d"
    ] = np.nan
    features[
        f"nearest_recent_event_age_days_past_{NEAREST_RECENT_WINDOW_DAYS}d"
    ] = np.nan

    if history.empty:
        return features

    max_window_days = max(RECENT_WINDOWS_DAYS)
    max_radius_km = max(LOCAL_RADII_KM)
    start_time = event["event_time"] - pd.Timedelta(days=max_window_days)
    candidates = history[
        (history["event_time"] >= start_time)
        & (history["event_time"] < event["event_time"])
    ].copy()
    if candidates.empty:
        return features

    lat_delta = max_radius_km / 111.32
    lon_scale = max(math.cos(math.radians(event["latitude"])), 0.1)
    lon_delta = max_radius_km / (111.32 * lon_scale)
    candidates = candidates[
        (np.abs(candidates["latitude"] - event["latitude"]) <= lat_delta)
        & (np.abs(candidates["longitude"] - event["longitude"]) <= lon_delta)
    ].copy()
    if candidates.empty:
        return features

    distances = haversine_km(
        event["latitude"],
        event["longitude"],
        candidates["latitude"].to_numpy(dtype=float),
        candidates["longitude"].to_numpy(dtype=float),
    )
    candidates["distance_km"] = distances
    candidates = candidates[candidates["distance_km"] <= max_radius_km].copy()
    if candidates.empty:
        return features

    nearest_window_start = event["event_time"] - pd.Timedelta(days=NEAREST_RECENT_WINDOW_DAYS)
    nearest_candidates = candidates[candidates["event_time"] >= nearest_window_start]
    if not nearest_candidates.empty:
        nearest = nearest_candidates.loc[nearest_candidates["distance_km"].idxmin()]
        features[
            f"nearest_recent_event_distance_km_past_{NEAREST_RECENT_WINDOW_DAYS}d"
        ] = float(nearest["distance_km"])
        features[
            f"nearest_recent_event_magnitude_past_{NEAREST_RECENT_WINDOW_DAYS}d"
        ] = float(nearest["magnitude"])
        features[
            f"nearest_recent_event_age_days_past_{NEAREST_RECENT_WINDOW_DAYS}d"
        ] = float((event["event_time"] - nearest["event_time"]).total_seconds() / 86400.0)

    for days in RECENT_WINDOWS_DAYS:
        window_start = event["event_time"] - pd.Timedelta(days=days)
        window = candidates[candidates["event_time"] >= window_start]
        if window.empty:
            continue
        for radius in LOCAL_RADII_KM:
            radius_token = int(radius)
            local = window[window["distance_km"] <= radius]
            if local.empty:
                continue
            magnitudes = local["magnitude"].to_numpy(dtype=float)
            features[f"local_events_{radius_token}km_past_{days}d"] = int(len(local))
            features[f"local_max_mag_{radius_token}km_past_{days}d"] = float(np.nanmax(magnitudes))
            features[f"local_log10_energy_{radius_token}km_past_{days}d"] = float(
                np.log10(np.nansum(10.0 ** (1.5 * magnitudes)))
            )

    return features


def build_prediction_features(history, event, args, feature_columns):
    """Assemble the full feature row for one event, in ``feature_columns`` order."""
    event_time = event["event_time"]
    features = {
        "magnitude": float(event["magnitude"]),
        "depth_km": float(event["depth_km"]),
        "latitude": float(event["latitude"]),
        "longitude": float(event["longitude"]),
        "event_year": int(event_time.year),
        "event_month": int(event_time.month),
        "event_dayofyear": int(event_time.dayofyear),
        "event_hour": int(event_time.hour),
        "event_weekday": int(event_time.weekday()),
    }
    features.update(
        compute_parent_features(
            history,
            event,
            args.b_value,
            args.fractal_dimension,
            args.log10_eta0,
        )
    )
    features.update(compute_global_history_features(history, event_time))
    features.update(compute_local_history_features(history, event))

    missing = sorted(set(feature_columns) - set(features))
    if missing:
        raise ValueError(f"Prediction builder did not create required features: {missing}")
    return pd.DataFrame([{column: features[column] for column in feature_columns}])


# --- Batch builders (training) ----------------------------------------------
def add_time_features(df):
    event_time = df["event_time"]
    df["event_year"] = event_time.dt.year
    df["event_month"] = event_time.dt.month
    df["event_dayofyear"] = event_time.dt.dayofyear
    df["event_hour"] = event_time.dt.hour
    df["event_weekday"] = event_time.dt.weekday
    return df


def add_parent_features(df):
    """Parent / eta features for training, sourced from the C++ ZBZ clustering.

    ``eta`` / ``log10_eta`` / ``is_strong_link`` are carried straight from the
    clustered input CSV; the rest are looked up from the precomputed
    ``parent_id_key``. The serving equivalent is ``compute_parent_features``.
    """
    by_event_id = df.set_index("event_id_key")
    parent = df["parent_id_key"].map(by_event_id["event_time"])
    df["parent_time_gap_days"] = (
        df["event_time"] - parent
    ).dt.total_seconds() / 86400.0

    for source_col, output_col in [
        ("magnitude", "parent_magnitude"),
        ("depth_km", "parent_depth_km"),
        ("latitude", "parent_latitude"),
        ("longitude", "parent_longitude"),
    ]:
        df[output_col] = df["parent_id_key"].map(by_event_id[source_col])

    has_parent_location = df[
        ["parent_latitude", "parent_longitude", "latitude", "longitude"]
    ].notna().all(axis=1)
    df["parent_distance_km"] = np.nan
    df.loc[has_parent_location, "parent_distance_km"] = haversine_km(
        df.loc[has_parent_location, "latitude"],
        df.loc[has_parent_location, "longitude"],
        df.loc[has_parent_location, "parent_latitude"],
        df.loc[has_parent_location, "parent_longitude"],
    )
    df["has_parent"] = df["parent_id_key"].notna().astype(int)
    return df


def add_recent_global_features(df):
    # Cast to ns before int64 so the hardcoded nanosecond windows below are
    # correct regardless of the column's datetime resolution. Pandas 3.0 parses
    # to datetime64[us]; a bare astype("int64") would yield microseconds and make
    # the window subtraction underflow, inflating counts to the row index.
    time_ns = df["event_time"].astype("datetime64[ns]").astype("int64").to_numpy()
    order = np.arange(len(df))

    for days in RECENT_WINDOWS_DAYS:
        window_ns = int(days * 86400 * 1_000_000_000)
        starts = np.searchsorted(time_ns, time_ns - window_ns, side="left")
        df[f"events_past_{days}d"] = order - starts

    return df


def add_recent_local_features(df):
    # See add_recent_global_features: cast to ns first so nanosecond windows match.
    time_ns = df["event_time"].astype("datetime64[ns]").astype("int64").to_numpy()
    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)
    magnitude = df["magnitude"].to_numpy(dtype=float)
    windows_ns = {
        days: int(days * 86400 * 1_000_000_000)
        for days in RECENT_WINDOWS_DAYS
    }
    nearest_window_ns = int(NEAREST_RECENT_WINDOW_DAYS * 86400 * 1_000_000_000)

    feature_data = {}
    for radius in LOCAL_RADII_KM:
        radius_token = int(radius)
        for days in RECENT_WINDOWS_DAYS:
            feature_data[f"local_events_{radius_token}km_past_{days}d"] = np.zeros(
                len(df),
                dtype=np.int32,
            )
            feature_data[f"local_max_mag_{radius_token}km_past_{days}d"] = np.full(
                len(df),
                np.nan,
            )
            feature_data[f"local_log10_energy_{radius_token}km_past_{days}d"] = np.full(
                len(df),
                np.nan,
            )

    nearest_distance = np.full(len(df), np.nan)
    nearest_magnitude = np.full(len(df), np.nan)
    nearest_age_days = np.full(len(df), np.nan)
    max_window_ns = windows_ns[max(RECENT_WINDOWS_DAYS)]
    max_radius_km = max(LOCAL_RADII_KM)
    max_window_starts = np.searchsorted(time_ns, time_ns - max_window_ns, side="left")
    lat_delta_degrees = max_radius_km / 111.32

    for row_index in range(len(df)):
        max_start = max_window_starts[row_index]
        if max_start == row_index:
            continue

        candidate_lat = lat[max_start:row_index]
        candidate_lon = lon[max_start:row_index]
        lon_scale = max(math.cos(math.radians(lat[row_index])), 0.1)
        lon_delta_degrees = max_radius_km / (111.32 * lon_scale)
        bounding_box_mask = (
            (np.abs(candidate_lat - lat[row_index]) <= lat_delta_degrees)
            & (np.abs(candidate_lon - lon[row_index]) <= lon_delta_degrees)
        )
        if not bounding_box_mask.any():
            continue

        candidates = max_start + np.flatnonzero(bounding_box_mask)
        candidate_times = time_ns[candidates]
        candidate_magnitudes = magnitude[candidates]
        distances = haversine_km(
            lat[row_index],
            lon[row_index],
            lat[candidates],
            lon[candidates],
        )
        radius_mask = distances <= max_radius_km
        if not radius_mask.any():
            continue

        candidates = candidates[radius_mask]
        candidate_times = candidate_times[radius_mask]
        candidate_magnitudes = candidate_magnitudes[radius_mask]
        distances = distances[radius_mask]

        nearest_window_mask = candidate_times >= time_ns[row_index] - nearest_window_ns
        if nearest_window_mask.any():
            nearest_positions = np.flatnonzero(nearest_window_mask)
            nearest_position = nearest_positions[
                int(np.nanargmin(distances[nearest_window_mask]))
            ]
            nearest_distance[row_index] = float(distances[nearest_position])
            nearest_magnitude[row_index] = float(candidate_magnitudes[nearest_position])
            nearest_age_days[row_index] = float(
                (time_ns[row_index] - candidate_times[nearest_position])
                / (86400 * 1_000_000_000)
            )

        for days in RECENT_WINDOWS_DAYS:
            window_mask = candidate_times >= time_ns[row_index] - windows_ns[days]
            if not window_mask.any():
                continue

            window_distances = distances[window_mask]
            window_magnitudes = candidate_magnitudes[window_mask]
            for radius in LOCAL_RADII_KM:
                radius_token = int(radius)
                local_mask = window_distances <= radius
                local_count = int(local_mask.sum())
                if not local_count:
                    continue

                local_magnitudes = window_magnitudes[local_mask]
                feature_data[f"local_events_{radius_token}km_past_{days}d"][
                    row_index
                ] = local_count
                feature_data[f"local_max_mag_{radius_token}km_past_{days}d"][
                    row_index
                ] = float(np.nanmax(local_magnitudes))
                feature_data[f"local_log10_energy_{radius_token}km_past_{days}d"][
                    row_index
                ] = float(np.log10(np.nansum(10.0 ** (1.5 * local_magnitudes))))

    for feature_name, values in feature_data.items():
        df[feature_name] = values
    df[f"nearest_recent_event_distance_km_past_{NEAREST_RECENT_WINDOW_DAYS}d"] = nearest_distance
    df[f"nearest_recent_event_magnitude_past_{NEAREST_RECENT_WINDOW_DAYS}d"] = nearest_magnitude
    df[f"nearest_recent_event_age_days_past_{NEAREST_RECENT_WINDOW_DAYS}d"] = nearest_age_days

    return df
