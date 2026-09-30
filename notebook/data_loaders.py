
# Main file for data loading
# Date: 2026-09-30

from functools import cached_property



class DataLoader:
    def __init__(self, data_source):
        pass

    @cached_property
    def day_ahead_price_data(self):
        file_path = "" 
        pass

    @cached_property
    def inflow_data(self):
        file_path = ""
        pass 
    
    @cached_property
    def volume_data(self):
        file_path = ""
        pass

    @cached_property
    def synthetic_water_value_data(self):
        file_path = ""
        pass

    @cached_property
    def commitment_descision_data(self):
        file_path = ""
        pass

    @cached_property
    def min_flow_constraint_data(self):
        file_path = ""
        pass

    @cached_property
    def min_volume_constraint_data(self):
        file_path = ""
        pass#

    # TODO: Add topology data loader later?