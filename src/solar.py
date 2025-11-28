"""
Solar PV Module

Processes solar irradiance datasets and converts them into
capacity factor arrays or generation time series for the
optimization model.

Functions:
    load_solar_data(filepath)
    compute_capacity_factor(irradiance, derate=0.8)
    solar_generation(capacity, capacity_factor)
"""
