
from datetime import datetime

from data_loaders import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

def prepare_data(start_date: datetime, end_date: datetime, hour_memory: int = 168, future_horizon: int = 168 ):
    # Initialize your data loader (assuming DataLoader class is defined in your environment)
    data = DataLoader()
    
    decisions = data.commitment_decision_data
    decisions['starttime'] = pd.to_datetime(decisions['starttime'])

    # Target variables (y)
    target_cols = [col for col in decisions.columns if col.startswith('result_committed')]
    y = decisions[target_cols]
    

    
    # Load and clean continuous time-series features
    prices = data.day_ahead_price_data.copy()
    prices['date'] = pd.to_datetime(prices['date']).dt.tz_convert(None)
    prices = prices.set_index('date')
    
    inflows = data.inflow_data.copy()
    inflows['date'] = pd.to_datetime(inflows['date']).dt.tz_convert(None)
    inflows = inflows.set_index('date')
    
    volumes = data.volume_data.copy()
    volumes['date'] = pd.to_datetime(volumes['date']).dt.tz_convert(None)
    volumes = volumes.set_index('date')
    
    water_values = data.synthetic_water_value_data.copy()
    water_values['date'] = pd.to_datetime(water_values['date']).dt.tz_convert(None)
    water_values = water_values.set_index('date')
    
    
    features_list = []
    target_list = []  
    
    # Slice 168-hour windows for each case
    for _, row in decisions.iterrows():
        start_time = row['starttime'] - pd.Timedelta(hours=hour_memory)
        end_time = row['starttime'] + pd.Timedelta(hours=future_horizon)
        if start_time < start_date:
            continue
        if end_time > end_date:
            continue

        # Validate that all required data is available for the given time window
        if start_time < min(prices.index.min(), inflows.index.min(), volumes.index.min(), water_values.index.min()) or \
            end_time > max(prices.index.max(), inflows.index.max(), volumes.index.max(), water_values.index.max()):
            continue

        price_slice = prices.loc[(prices.index >= start_time) & (prices.index < end_time)]
        inflow_slice = inflows.loc[(inflows.index >= start_time) & (inflows.index < end_time)]
        volume_slice = volumes.loc[(volumes.index >= start_time) & (volumes.index < end_time)]
        water_value_slice = water_values.loc[(water_values.index >= start_time) & (water_values.index < end_time)]
        
        feature_dict = {'Run No': row['Run No'], 'starttime': row['starttime']}
        
        # Flatten hourly prices
        n_past_prices = (price_slice.index < row['starttime']).sum()
        for i, val in enumerate(price_slice['Day-ahead price (EUR/MWh)'].values):
            timestep = i - n_past_prices
            feature_dict[f'price_t{timestep}'] = val
            
        # Flatten hourly inflows
        n_past_inflows = (inflow_slice.index < row['starttime']).sum()
        for col in inflows.columns:
            for i, val in enumerate(inflow_slice[col].values):
                timestep = i - n_past_inflows
                feature_dict[f'inflow_{col}_t{timestep}'] = val
                
        # Add volumes
        n_past_volumes = (volume_slice.index < row['starttime']).sum()
        for col in volumes.columns:
            for i, val in enumerate(volume_slice[col].values):
                timestep = i - n_past_volumes
                feature_dict[f'volume_{col}_day{timestep}'] = val
                
        # Add water values
        n_past_water_values = (water_value_slice.index < row['starttime']).sum()
        for col in water_values.columns:
            for i, val in enumerate(water_value_slice[col].values):
                timestep = i - n_past_water_values
                feature_dict[f'water_value_{col}_day{timestep}'] = val
                
        features_list.append(feature_dict)
        target_list.append(row[target_cols].to_dict())
        

        
    X = pd.DataFrame(features_list)
    y = pd.DataFrame(target_list)

    # Explicitly align indices
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    assert len(X) == len(y)
    assert X.index.equals(y.index)

    return X, y

    

if __name__ == "__main__":
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2022, 12, 31)
    X_train, y_train = prepare_data(start_date=start_date, end_date=end_date)
    print("Feature matrix shape:", X_train.shape)
    print("Target matrix shape:", y_train.shape)

    feature_cols = X_train.columns.tolist()

    # print("Feature columns:", feature_cols)
