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
    # TODO: move this in the notebooks (or make it simpler/remove it)

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
    
    TODO: simplify this process (or delete)
    '''
 
    def __init__(self, data_source=None):
        super().__init__(data_source)
 
 
    def load_copies(self):
        sources = {
            "day_ahead_price": self.day_ahead_price_data,
            "inflow": self.inflow_data,
            "volume": self.volume_data,
            "wv": self.synthetic_water_value_data,
            "uc": self.commitment_decision_data,
        }
        return sources
 
 
    def create_x(self):
        '''
        Create the complete X dataset
        '''
        
        data = self.load_copies()
        
        # setting the date as index and fixing the columns names
        for name in data.keys():
            if 'date' in list(data[name].columns):
                data[name]['date'] = pd.to_datetime(data[name]['date'], utc=True)
                data[name] = data[name].set_index(data[name]['date']).drop(columns='date')
                if 'Day-ahead price (EUR/MWh)' in data[name].columns:
                    data[name].columns = data[name].columns.str.replace(r'\s*\([^)]*\)', '', regex=True).str.replace(r'[\s-]+', '_', regex=True).str.lower()
                else:
                    data[name].columns = data[name].columns.str.lower().str.replace(' ', '_s')
                
            else:
                data[name]['starttime'] = pd.to_datetime(data[name]['starttime'], utc=True)
                data[name] = data[name].set_index(data[name]['starttime']).drop(columns='starttime')
                data[name].columns = data[name].columns.str.lower().str.replace(' ', '_')

        
        # creating the two datset
        #   y stored for future use
        self.y = data['uc']
        data.pop('uc')
        x = data['day_ahead_price']
        data.pop('day_ahead_price')
        for name in data.keys():
                x = x.join(data[name].add_suffix('_' + name), how='left')
                
        
        return x.copy()


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
    X = ts.create_x()
    print(X.head(5))