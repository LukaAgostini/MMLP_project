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


def build_cases(X, Y, T=168):
    '''
    Modify the dataset to have them case indexed
    Input:
    - X: x dataset
    - Y: y dataset
    '''
    
    # splitting between daily and hourly features
    hourly = ['day_ahead_price'] + [c for c in X.columns if c.endswith('_inflow')]
    vol = [c for c in X.columns if c.endswith('_volume')]
    wv  = [c for c in X.columns if c.endswith('_wv')]

    # getting the position of every case starting hour
    pos = X.index.get_indexer(Y.index)
    assert (pos >= 0).all() and (pos + T < len(X)).all()

    # creating the sliding windows
    win = swv(X[hourly].to_numpy(np.float32), T, axis=0)       
    xh = win[pos].reshape(len(pos), -1)                        
    
    wv_arr = X[wv].to_numpy()
    xs = np.hstack([
        X[vol].to_numpy()[pos],  # initial volume (s)
        wv_arr[pos],             # initial water value (pos)
        wv_arr[pos + T]          # final water value (pos+T)
    ])

    # getting the names for the columns of x
    hourly_cols = [f"{c}_t{h}" for c in hourly for h in range(T)]
    wv_init_cols = [f"{c}_init" for c in wv]
    wv_final_cols = [f"{c}_final" for c in wv]
    
    all_cols = hourly_cols + vol + wv_init_cols + wv_final_cols

    # excluding run_no e total_calculation_time
    y = pd.DataFrame(
        Y.filter(like='result_committed').to_numpy(np.uint8),
        index=Y.index,
        columns=Y.columns[2:]
    )
    
    x = pd.DataFrame(
        np.hstack([xh, xs]).astype(np.float32),
        index=Y.index,
        columns=all_cols
    )
    
    return x, y


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
    '''
 
    def __init__(self, data_source=None):
        super().__init__(data_source)
        self.y = None
        self.x = None
 
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
        
        TODO: feature engineering and selection
                + time split for easier computation (test model on smaller data)
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
                    data[name].columns = data[name].columns.str.lower().str.replace(' ', '_')
                
            else:
                data[name]['starttime'] = pd.to_datetime(data[name]['starttime'], utc=True)
                data[name] = data[name].set_index(data[name]['starttime']).drop(columns='starttime')
                data[name].columns = data[name].columns.str.lower().str.replace(' ', '_')

        # wv and volume have daily data
        #   need to repet thier value hourly
        hourly = data['day_ahead_price'].index
        for name in ('volume', 'wv'):
            data[name] = data[name].reindex(hourly, method='ffill')
        
        # creating the two datset
        #   y stored for future use
        self.y = data['uc'].copy()
        data.pop('uc')
        x = data['day_ahead_price']
        data.pop('day_ahead_price')
        for name in data.keys():
                x = x.join(data[name].add_suffix('_' + name), how='left')
                
        
        self.x = x.copy()
    
    
    def split_small_data(self, begin_train:str ='2015-01-01', end_train: str='2018-12-31', begin_test: str='2019-01-12', end_test: str='2019-12-31'):
        '''
        Return X and Y dataset
        Input:
            - beginnig_train (str, dafault=2015-01-01 ): strart date training, format 'yyyy-mm-dd'
            - beginnig_test (str, dafault=2018-12-31 ): strart date testing, format 'yyyy-mm-dd'
            - end_train (str, dafault=2019-01-01 ): end date of training, format 'yyyy-mm-dd'
            - end_test (str, dafault=2019-12-31 ): end date of testing, format 'yyyy-mm-dd'
        '''
        
        self.create_x()
        X, Y = self.x.copy(), self.y.copy()
        x_train, y_train = build_cases(X, Y.loc[begin_train:end_train])
        x_test,  y_test  = build_cases(X, Y.loc[begin_test:end_test])
        return x_train, y_train, x_test, y_test

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
    ts.create_x()
    X_1 = ts.x
    print(X_1.head(5))
    print(X_1.shape)
    
    #x_train, y_train, x_test, y_test = ts.split_small_data()
    #print(x_train.head(5),'\n', y_train.head(5),'\n',  x_test.head(5),'\n',  y_test.head(5))