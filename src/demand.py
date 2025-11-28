"""
Demand Module

Handles national baseline demand projections and integrates
Energy-as-a-Service (EaaS) deployment scenarios as demand-side
modifiers. Produces net demand time series for system optimization.

Functions:
    load_baseline_demand(filepath)
    apply_eaas_intervention(demand, scenario_params)
    get_net_demand(profile, eaas_params)
"""
