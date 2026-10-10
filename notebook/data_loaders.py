# Main file for data loading
# Date: 2026-09-30

from functools import cached_property
from pathlib import Path
import pandas as pd
import yaml

repo_root = Path(__file__).resolve().parents[1]
data_dir = repo_root / "sintef_project" / "data"


class DataLoader:
    # TODO: Change timestamp data from string to datatime object if available in the data source.

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
    def commitment_descision_data(self):
        return self._csv("kernel", "Unit_commitment_decisions.csv", ("starttime",))

    @cached_property
    def commitment_decision_data(self):
        return self.commitment_descision_data

    @cached_property
    def min_flow_constraint_data(self):
        return self._csv("extended", "Constraint_min_flow.csv", ("date",))

    @cached_property
    def min_volume_constraint_data(self):
        return self._csv("extended", "Constraint_min_volume.csv", ("date",))

    @cached_property
    def topology_data(self): #TODO: CHECK THIS ONE LATER
        return self._yaml("extended", "Tokke_Vinje_topology.yaml")



# Class to create the entire dataset in one go
class Training_set(DataLoader):
    '''
    It imports all the dataset, merging them using the the index (that will be the date)
    
    TO-DO: dataset cleaning and feature engineering
    '''
    
    def __init__(self, data_source=None):
        super().__init__(data_source)
        
        
    def load_copies(self):
        sources = {
            "price": (self.day_ahead_price_data, "date"),
            "inflow": (self.inflow_data, "date"),
            "volume": (self.volume_data, "date"),
            "wv": (self.synthetic_water_value_data, "date"),
            "uc": (self.commitment_decision_data, "starttime"),
        }
        out = {}
        for k, (df, col) in sources.items():
            df = df.copy().set_index(col)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            out[k] = df
        return out

    @cached_property
    def create_dataset(self):
        d = self.load_copies()

        price = d["price"].iloc[:, 0]
        price = price.groupby([price.index.normalize(), price.index.hour]).mean().unstack().add_prefix("price_h")

        X = d["volume"].add_prefix("volume_")
        X = X.join(d["inflow"].add_prefix("inflow_"))
        X = X.join(d["wv"].add_prefix("wv_"))
        X = X.join(price)
        X = X.join(d["uc"].drop(columns=["Run No", "total_calculation_time"]))
        return X.loc[:"2025-01-01"]
        



if __name__ == "__main__":
    DataLoader()
    '''
    print(DataLoader().day_ahead_price_data.head())
    print(DataLoader().inflow_data.head())
    print(DataLoader().volume_data.head())
    print(DataLoader().synthetic_water_value_data.head())
    print(DataLoader().commitment_decision_data.head())
    print(DataLoader().min_flow_constraint_data.head())
    print(DataLoader().min_volume_constraint_data.head())
    '''
    # testing the new class
    print(Training_set().create_dataset.head())
    d = Training_set().load_copies()
    for k, v in d.items():
        print(k, v.index.min(), v.index.max(), v.index.tz)

