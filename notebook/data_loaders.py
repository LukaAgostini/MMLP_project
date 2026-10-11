# Main file for data loading
# Date: 2026-09-30

from functools import cached_property
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
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
        '''
        Hourly input table (all times UTC): index = hour (UTC), columns = price, inflow_*, volume_*, wv_*.
        '''
        d = self.load_copies()
 
        price = d["price"].iloc[:, 0].resample("h").mean().rename("price")  # 15-min prices (from 2025-09) -> hourly
        idx = price.loc[:"2025-01-01 00:00"].index
 
        X = price.loc[idx].to_frame()
        X = X.join(d["inflow"].add_prefix("inflow_"))                        # already hourly
        X = X.join(d["volume"].add_prefix("volume_").reindex(idx, method="ffill"))
        X = X.join(d["wv"].add_prefix("wv_").reindex(idx, method="ffill"))
        return X
 
    @cached_property
    def create_labels(self):
        '''
        UC decisions in long format: index = (starttime, step), columns = units (0/1).
        '''
        uc = self.load_copies()["uc"].drop(columns=["Run No", "total_calculation_time"])
        units = list(dict.fromkeys(c.rsplit("_t", 1)[0].removeprefix("result_committed_") for c in uc.columns))
        Y = uc.values.reshape(len(uc), len(units), 168).transpose(0, 2, 1).reshape(-1, len(units))
        idx = pd.MultiIndex.from_product([uc.index, range(168)], names=["starttime", "step"])
        return pd.DataFrame(Y, index=idx, columns=units)
        
        
    def create_features(self, starts=None):
        '''
        - index = (starttime, step) 
        - one row per case and hour (all UTC)
        - starts: case start times
        - Columns: inputs of the real hour (starttime + step) from create_dataset, 
                plus week-context features of the case
                        - price rank
                        - relative price
                        - week and day statistics
                        - water value at end of week
                        - mean inflow of the week)
        '''
        
        X = self.create_dataset
        
        if starts is None:
            starts = self.create_labels.index.get_level_values("starttime").unique()
        
        starts = pd.DatetimeIndex(starts)
        
        # real hour of every (case, step)
        n, step = len(starts), np.tile(np.arange(168), len(starts))
        hour = np.repeat(starts, 168) + pd.to_timedelta(step, unit="h")           
        
        F = X.reindex(hour).set_axis(pd.MultiIndex.from_arrays([np.repeat(starts, 168), step], names=["starttime", "step"]))
        # per-day value -> its 24 hours
        rep = lambda a: np.repeat(a, 24, axis=1).ravel()                          
 
        # (cases, 168) prices of each case week
        P = swv(X.price.values, 168)[X.index.get_indexer(starts)]                
        D = P.reshape(n, 7, 24)
        F["step"], F["hour"] = step, step % 24
        
        # price rank within the week (0 = cheapest)
        F["rank"] = (P.argsort(1).argsort(1) / 167).ravel()                       
        F["price_rel"] = (P - P.mean(1, keepdims=True)).ravel()
        
        # weak mean and std
        F["wk_mean"], F["wk_std"] = np.repeat(P.mean(1), 168), np.repeat(P.std(1), 168)
        
        # day statistics
        F["day_mean"], F["day_max"], F["day_min"] = rep(D.mean(2)), rep(D.max(2)), rep(D.min(2))
        F["month"] = np.repeat(starts.month, 168)
 
        # water value at end of week
        wv_end = X.filter(like="wv_").reindex(starts + pd.Timedelta(hours=168)).set_axis(starts).add_suffix("_end")  
        
        # mean inflow of the week
        inflow = X.filter(like="inflow_").rolling(168).mean().shift(-167).reindex(starts).add_suffix("_wk")          
        
        # per-case -> per-row
        C = pd.concat([wv_end, inflow], axis=1).reindex(F.index.get_level_values(0)).set_axis(F.index)               
        return F.join(C)



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
    ts = Training_set()
    X, Y = ts.create_dataset, ts.create_labels
    print(X.shape, Y.shape)
    print("dataset:", X.index.min(), "->", X.index.max())
    assert X.index.max() == pd.Timestamp("2025-01-01 00:00"), "dataset must stop at 2025-01-01 00:00"
    assert X.index.is_monotonic_increasing and X.index.is_unique
    assert (X.index.to_series().diff().dropna() == pd.Timedelta("1h")).all(), "hourly gaps"
    print("NaN per column (non zero only):\n", X.isna().sum()[lambda v: v > 0])
    last = Y.index.get_level_values("starttime").max()
    print("labels: last case", last, "ends at", last + pd.Timedelta(hours=168))