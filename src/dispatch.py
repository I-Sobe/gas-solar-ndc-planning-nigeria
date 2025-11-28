"""
Dispatch Module

Defines power balance constraints, gas generation limits,
solar dispatch, storage charge/discharge, and unmet load.

Used by the Pyomo/PyPSA optimization framework to ensure
system feasibility at each timestep.

Functions:
    power_balance(...)
    gas_generation_constraints(...)
    build_dispatch_model(params)
"""
