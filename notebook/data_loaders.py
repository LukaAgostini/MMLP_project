# Main file for data loading
# Date: 2026-09-30

from abc import ABC, abstractmethod
from datetime import datetime
from functools import cached_property
from pathlib import Path
import pandas as pd
import yaml
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
data_dir = repo_root / "sintef_project" / "data"


class DataLoader:
    def __init__(self, data_source=None):
        self.data_source = Path(data_source) if data_source else data_dir

    def _csv(self, folder, filename, date_columns=()):
        file_path = self.data_source / folder / filename
        frame = pd.read_csv(file_path)
        for column in date_columns:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column])
        return frame

    def _yaml(self, folder, filename):
        file_path = self.data_source / folder / filename
        with file_path.open(encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    @cached_property
    def day_ahead_price_data(self):
        frame = self._csv(
            "kernel",
            "Historical_day_ahead_price_2015_2025.csv",
            ("date",),
        )
        frame.rename(columns={"Day-ahead price (EUR/MWh)": "day_ahead_price"}, inplace=True) # Rename for consistency
        return frame

    @cached_property
    def inflow_data(self):
        return self._csv("kernel", "Historical_inflow_1958_2025.csv", ("date",))

    @cached_property
    def volume_data(self):
        return self._csv("kernel", "Historical_volume_2015_2024.csv", ("date",))

    @cached_property
    def synthetic_water_value_data(self):
        return self._csv(
            "kernel",
            "Synthetic_water_value_2015_2024.csv",
            ("date",),
        )

    @cached_property
    def commitment_decision_data(self):
        return self._csv("kernel", "Unit_commitment_decisions.csv", ("starttime",))


    @cached_property
    def min_flow_constraint_data(self):
        return self._csv("extended", "Constraint_min_flow.csv", ("date",))

    @cached_property
    def min_volume_constraint_data(self):
        return self._csv("extended", "Constraint_min_volume.csv", ("date",))

    @cached_property
    def topology_data(self): #TODO: CHECK THIS ONE LATER
        return self._yaml("extended", "Tokke_Vinje_topology.yaml")

    ## Other functions
    def get_cached_property_data_attr(self):
        """
        Returns a list of attribute names that are cached_properties and return DataFrames.
        """
        cls = self.__class__
        dataframe_properties = []

        for attribute in dir(cls):
            if isinstance(getattr(cls, attribute, None), cached_property):
                val = getattr(self, attribute)
                if isinstance(val, pd.DataFrame):
                    dataframe_properties.append(attribute)
        return dataframe_properties

    def describe_columns(self):
        """
        prints the names of all cached_property attributes that return DataFrames along with their column names.
        """
        df_attributes = self.get_cached_property_data_attr()
        
        for attribute in df_attributes:
            val = getattr(self, attribute)
            
            print(f"Attribut: '{attribute}'")
            print(f"Columns:  {val.columns.tolist()}\n")


# Class to create the entire dataset in one go
class Training_set(DataLoader, ABC):
    """Container for raw datasets at their native time resolution.

    Hourly and daily tables remain separate. This avoids interpolating or
    accidentally dropping daily observations when building model features.
    """

    _raw_sources = {
        "day_ahead_price": ("day_ahead_price_data", "date"),
        "inflow": ("inflow_data", "date"),
        "volume": ("volume_data", "date"),
        "water_value": ("synthetic_water_value_data", "date"),
        "commitment_decision": ("commitment_decision_data", "starttime"),
        "min_flow": ("min_flow_constraint_data", "date"),
        "min_volume": ("min_volume_constraint_data", "date"),
    }

    @staticmethod
    def _index_frame(frame, timestamp_column):
        result = frame.copy()
        result[timestamp_column] = pd.to_datetime(
            result[timestamp_column], utc=True
        ).dt.tz_localize(None)
        result = result.set_index(timestamp_column).sort_index()
        if result.index.has_duplicates:
            raise ValueError(
                f"{timestamp_column!r} contains duplicate timestamps"
            )
        return result

    @cached_property
    def raw_data(self):
        """Return independent, indexed copies of every source table."""
        return {
            name: self._index_frame(
                getattr(self, attribute), timestamp_column
            )
            for name, (attribute, timestamp_column) in self._raw_sources.items()
        }

    @staticmethod
    def _data_frequency(name, frame):
        """Infer the native category from an indexed table."""
        if name == "commitment_decision":
            return "single"
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"{name!r} must have a DatetimeIndex")
        if len(frame.index) < 2:
            return "single"

        step = frame.index.to_series().diff().dropna().median()
        if step <= pd.Timedelta(hours=1):
            return "hourly"
        if step <= pd.Timedelta(days=1):
            return "daily"
        return "single"

    @classmethod
    def _categorize_data(cls, data):
        categorized = {"hourly": {}, "daily": {}, "single": {}}
        for name, frame in data.items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{name!r} must be a pandas DataFrame")
            category = cls._data_frequency(name, frame)
            categorized[category][name] = frame.copy()
        return categorized

    @cached_property
    def raw_data_by_frequency(self):
        """Raw tables grouped automatically as hourly, daily, or single."""
        return self._categorize_data(self.raw_data)

    def load_copies(self):
        """Backward-compatible alias for :attr:`raw_data`."""
        return {name: frame.copy() for name, frame in self.raw_data.items()}

    @abstractmethod
    def raw_preprocessing(self):
        """Return the preprocessed raw tables used as model inputs."""
        raise NotImplementedError

    @staticmethod
    def _timestamp(value):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        return timestamp

    @staticmethod
    def _window_is_complete(frame, expected_index):
        return frame.reindex(expected_index).notna().all().all()

    def _add_hourly_features(
        self, features, frame, start, end, case_start, prefix
    ):
        expected = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h")
        if not self._window_is_complete(frame, expected):
            return False
        values = frame.loc[expected]
        for offset, timestamp in enumerate(expected):
            timestep = int((timestamp - case_start).total_seconds() // 3600)
            for column in values.columns:
                features[f"{prefix}{column}_t{timestep}"] = values.iloc[
                    offset
                ][column]
        return True

    def _add_daily_features(
        self, features, frame, start, end, case_start, prefix
    ):
        expected = pd.date_range(
            start.normalize(),
            (end - pd.Timedelta(nanoseconds=1)).normalize(),
            freq="D",
        )
        if not self._window_is_complete(frame, expected):
            return False
        values = frame.loc[expected]
        case_day = case_start.normalize()
        for timestamp in expected:
            day_offset = (timestamp - case_day).days
            for column in values.columns:
                features[f"{prefix}{column}_day{day_offset}"] = values.loc[
                    timestamp, column
                ]
        return True

    def prepare_data(
        self,
        start_date: datetime,
        end_date: datetime,
        hour_memory: int = 168,
        include_constraints: bool = False,
    ):
        """Create model-ready feature and target matrices from this dataset."""
        if hour_memory < 0:
            raise ValueError("hour_memory must be non-negative")
        selection_start = self._timestamp(start_date)
        selection_end = self._timestamp(end_date)
        if selection_end < selection_start:
            raise ValueError("end_date must not be earlier than start_date")

        source = self.raw_preprocessing()
        if not isinstance(source, dict):
            raise TypeError("raw_preprocessing must return a dictionary of DataFrames")

        # Targets are always loaded from the original commitment table and
        # cannot be removed or altered by input feature preprocessing.
        source = dict(source)
        source["commitment_decision"] = self.raw_data["commitment_decision"].copy()
        categorized = self._categorize_data(source)

        required = set()
        if include_constraints:
            required.update({"min_flow", "min_volume"})
        missing = required.difference(source)
        if missing:
            raise ValueError(
                f"raw_preprocessing is missing required dataset(s): {sorted(missing)}"
            )
        if not include_constraints:
            source.pop("min_flow", None)
            source.pop("min_volume", None)
            categorized = self._categorize_data(source)

        decisions = categorized["single"]["commitment_decision"]
        target_cols = [
            column
            for column in decisions.columns
            if column.startswith("result_committed")
        ]
        if not target_cols:
            raise ValueError("No result_committed target columns were found")

        feature_rows = []
        target_rows = []
        for run_no, case in decisions.iterrows():
            case_start = run_no
            if not (selection_start <= case_start <= selection_end):
                continue
            case_end = case_start + pd.Timedelta(hours=168)
            window_start = case_start - pd.Timedelta(hours=hour_memory)
            start_month = case_start.month
            
            features = {"Run No": case["Run No"], 
                "starttime": case_start,
                "start_month_sin": np.sin(2 * np.pi * start_month / 12),
                "start_month_cos": np.cos(2 * np.pi * start_month / 12)}
            complete = True
            for name, frame in categorized["hourly"].items():
                prefix = "" if name == "day_ahead_price" else f"{name}_" # Manual override for the price for better naming consistency
                complete &= self._add_hourly_features(
                    features,
                    frame,
                    window_start,
                    case_end,
                    case_start,
                    prefix,
                )
            for name, frame in categorized["daily"].items():
                prefix = f"{name}_"
                complete &= self._add_daily_features(
                    features,
                    frame,
                    window_start,
                    case_end,
                    case_start,
                    prefix,
                )
            if complete:
                feature_rows.append(features)
                target_rows.append(case[target_cols].to_dict())

        X = pd.DataFrame(feature_rows)
        y = pd.DataFrame(target_rows, columns=target_cols)
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        return X, y


    def export_X_and_Y_summary(self, X, y, name: str, output_dir: str = "output"):
        """Export summary of feature and target matrices to files."""
        target_dir = repo_root / Path(output_dir) / name
        target_dir.mkdir(parents=True, exist_ok=True)

        x_csv_path = target_dir / "X_summary.csv"
        y_csv_path = target_dir / "y_summary.csv"

        # Export CSVs first
        X.to_csv(x_csv_path, index=False)
        y.to_csv(y_csv_path, index=False)

        # Get file sizes in MB
        x_size_mb = x_csv_path.stat().st_size / (1024 * 1024)
        y_size_mb = y_csv_path.stat().st_size / (1024 * 1024)


        # Extract base names and native resolutions
        feature_resolutions = {}
        for col in X.columns:
            if "_t" in col and col.rsplit("_t", 1)[1].lstrip("-").isdigit():
                base_name = col.rsplit("_t", 1)[0]
                feature_resolutions[base_name] = "hourly"
            elif "_day" in col and col.rsplit("_day", 1)[1].lstrip("-").isdigit():
                base_name = col.rsplit("_day", 1)[0]
                feature_resolutions[base_name] = "daily"
            else:
                feature_resolutions[col] = "single"


        # Write summary file
        with open(target_dir / "summary.txt", "w") as f:
            f.write(f"Summary of the feature and target matrices for the dataset for the {self.__class__.__name__} class:\n")
            f.write(f"The data spans from {X['starttime'].min()} to {X['starttime'].max()}\n")
            f.write(f"Feature matrix dimensions: {X.shape}\n")
            f.write(f"Target matrix dimensions: {y.shape}\n")
            f.write(f"Feature matrix CSV size: {x_size_mb:.2f} MB\n")
            f.write(f"Target matrix CSV size: {y_size_mb:.2f} MB\n")

            f.write("\n-------------------------------------------------------------------------\n\n")
            f.write("Feature matrix column categories:\n")
            for base_name, resolution in feature_resolutions.items():
                f.write(f"  {base_name} - {resolution}\n")



            f.write("\n-------------------------------------------------------------------------\n\n")
            f.write("Row starttimes:\n")
            for idx, start_time in enumerate(X['starttime']):
                f.write(f"  Row {idx}: {start_time}\n")

            f.write("\n-------------------------------------------------------------------------\n\n")
            f.write("Feature columns:\n")
            for col in X.columns:
                f.write(f"  {col}: {X[col].dtype}\n")

            f.write("\n-------------------------------------------------------------------------\n\n")
            f.write("Target columns:\n")
            for col in y.columns:
                f.write(f"  {col}: {y[col].dtype}\n")

class Training_set_example(Training_set):
    """Training set using only hourly price history as model input."""

    def raw_preprocessing(self):
        raw = self.load_copies()
        return {
            "day_ahead_price": raw["day_ahead_price"],
        }


class Training_set_all(Training_set):
    """Training set using all core raw inputs."""

    def raw_preprocessing(self):
        raw = self.load_copies()
        raw.pop("commitment_decision") # Commitment decision is always included as target, not input
        return raw

if __name__ == "__main__":
    start_time = datetime.now()
    loader = DataLoader()
    initialization_time = datetime.now() - start_time
    print(f"DataLoader initialized in {initialization_time.total_seconds():.2f} seconds.")


    error_string = ""
    for attr in loader.get_cached_property_data_attr():
        try:
            _ = getattr(loader, attr)
        except Exception as e:
            error_string += f"{attr},\n"

    if error_string != "":
        print(f"\nUnable to load data for the following attributes:\n{error_string}")
    else:
        print(f"All data could be loaded successfully in {(datetime.now() - start_time - initialization_time).total_seconds():.2f} seconds.")

    after_loading_timestamp = datetime.now()
    training_set = Training_set_all()

    initialization_time_Training_set = datetime.now() - after_loading_timestamp
    print(f"Training_set_all initialized in {initialization_time_Training_set.total_seconds():.2f} seconds.")

    X_test, y_test = training_set.prepare_data(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 12, 31),
        hour_memory=0,
        include_constraints=True,
    )

    data_preparation_time = datetime.now() - after_loading_timestamp - initialization_time_Training_set
    print(f"Data prepared in {data_preparation_time.total_seconds():.2f} seconds.")

    training_set.export_X_and_Y_summary(X_test, y_test, "test_data", "output")