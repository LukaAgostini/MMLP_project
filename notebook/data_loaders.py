# Main file for data loading
# Date: 2026-09-30

from functools import cached_property
from pathlib import Path
import pandas as pd
import yaml

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
        return self._csv(
            "kernel",
            "Historical_day_ahead_price_2015_2025.csv",
            ("date",),
        )

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


if __name__ == "__main__":
    DataLoader()

    # test all data loading methods
    attributes = [
        "day_ahead_price_data",
        "inflow_data",
        "volume_data",
        "synthetic_water_value_data",
        "commitment_decision_data",
        "min_flow_constraint_data",
        "min_volume_constraint_data",
        "topology_data",
    ]

    error_string = ""

    for attr in attributes:
        try:
            _ = getattr(DataLoader(), attr)
        except Exception as e:
            error_string += f"{attr},\n"

    if error_string != "":
        print(f"\nUnable to load data for the following attributes:\n{error_string}")
    else:
        print("All data loaded successfully.")