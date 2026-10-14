"""Build one feature row and one target row per unit-commitment case."""

from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from .data_loaders import DataLoader
except ImportError:  # Allows importing this module from a notebook in notebook/.
    from data_loaders import DataLoader


def _utc_naive(values: pd.Series) -> pd.Series:
    """Return timestamps as timezone-naive UTC values."""
    return pd.to_datetime(values, utc=True).dt.tz_localize(None)


def _naive_timestamp(value: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _indexed_frame(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Copy a loader frame and index it by unique, sorted UTC timestamps."""
    result = frame.copy()
    result[date_column] = _utc_naive(result[date_column])
    result = result.set_index(date_column).sort_index()
    if result.index.has_duplicates:
        raise ValueError(f"{date_column!r} contains duplicate timestamps")
    return result


def _window_is_complete(
    frame: pd.DataFrame, expected_index: pd.DatetimeIndex
) -> bool:
    """Check that every timestamp required by a case is present."""
    return frame.reindex(expected_index).notna().all().all()


def _add_hourly_features(
    features: dict[str, object],
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    case_start: pd.Timestamp,
    prefix: str,
) -> bool:
    expected_index = pd.date_range(
        start, end - pd.Timedelta(hours=1), freq="h"
    )
    if not _window_is_complete(frame, expected_index):
        return False

    values = frame.loc[expected_index]
    offsets = range(len(values))
    for column in values.columns:
        for offset, value in zip(offsets, values[column].to_numpy()):
            timestamp = start + pd.Timedelta(hours=offset)
            timestep = int((timestamp - case_start).total_seconds() // 3600)
            features[f"{prefix}{column}_t{timestep}"] = value
    return True


def _add_daily_features(
    features: dict[str, object],
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    case_start: pd.Timestamp,
    prefix: str,
) -> bool:
    first_day = start.normalize()
    last_day = (end - pd.Timedelta(nanoseconds=1)).normalize()
    expected_index = pd.date_range(first_day, last_day, freq="D")
    if not _window_is_complete(frame, expected_index):
        return False

    values = frame.loc[expected_index]
    case_day = case_start.normalize()
    for column in values.columns:
        for timestamp, value in values[column].items():
            day_offset = (timestamp - case_day).days
            features[f"{prefix}{column}_day{day_offset}"] = value
    return True


def prepare_data(
    start_date: datetime,
    end_date: datetime,
    hour_memory: int = 168,
    include_constraints: bool = False,
    data_source: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare fixed-width features and multilabel UC targets.

    ``hour_memory`` hours before each case and ``future_horizon`` hours from
    its start are included. Hourly inputs are represented at hourly offsets;
    daily volume and water-value inputs are represented at day offsets. Cases
    with incomplete source data are omitted rather than silently padded.

    ``end_date`` is inclusive for selecting case start times. The complete
    feature window must still be available in the source data.
    """

    future_horizon = 168 # Should not be changed as the data is always 168 hours long, and the model is trained to predict 168 hours into the future.

    if hour_memory < 0 or future_horizon <= 0:
        raise ValueError("hour_memory must be non-negative and future_horizon positive")

    selection_start = _naive_timestamp(start_date)
    selection_end = _naive_timestamp(end_date)
    if selection_end < selection_start:
        raise ValueError("end_date must not be earlier than start_date")

    data = DataLoader(data_source)
    decisions = data.commitment_decision_data.copy()
    decisions["starttime"] = _utc_naive(decisions["starttime"])
    decisions = decisions.sort_values("starttime").reset_index(drop=True)

    target_cols = [
        column
        for column in decisions.columns
        if column.startswith("result_committed")
    ]
    if not target_cols:
        raise ValueError("No result_committed target columns were found")

    hourly_frames = {
        "price_": _indexed_frame(data.day_ahead_price_data),
        "inflow_": _indexed_frame(data.inflow_data),
    }
    daily_frames = {
        "volume_": _indexed_frame(data.volume_data),
        "water_value_": _indexed_frame(data.synthetic_water_value_data),
    }
    if include_constraints:
        hourly_frames["min_flow_"] = _indexed_frame(data.min_flow_constraint_data)
        hourly_frames["min_volume_"] = _indexed_frame(
            data.min_volume_constraint_data
        )

    feature_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    for _, case in decisions.iterrows():
        case_start = case["starttime"]
        case_end = case_start + pd.Timedelta(hours=future_horizon)
        window_start = case_start - pd.Timedelta(hours=hour_memory)

        if not (selection_start <= case_start <= selection_end):
            continue

        features: dict[str, object] = {
            "Run No": case["Run No"],
            "starttime": case_start,
        }
        complete = True
        for prefix, frame in hourly_frames.items():
            complete &= _add_hourly_features(
                features, frame, window_start, case_end, case_start, prefix
            )
        for prefix, frame in daily_frames.items():
            complete &= _add_daily_features(
                features, frame, window_start, case_end, case_start, prefix
            )
        if not complete:
            continue

        feature_rows.append(features)
        target_rows.append(case[target_cols].to_dict())

    X = pd.DataFrame(feature_rows)
    y = pd.DataFrame(target_rows, columns=target_cols)
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    if not X.index.equals(y.index):
        raise RuntimeError("Feature and target indices are not aligned")
    return X, y


if __name__ == "__main__":
    X_train, y_train = prepare_data(
        start_date=datetime(2015, 1, 1),
        end_date=datetime(2022, 12, 31),
    )
    print("Feature matrix shape:", X_train.shape)
    print("Target matrix shape:", y_train.shape)
