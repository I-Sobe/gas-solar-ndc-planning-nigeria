"""
Gas Supply Module

Implements reservoir decline modeling using Arps equations
(exponential and hyperbolic) to generate annual gas availability
profiles for system-level optimization under supply constraints.

Functions:
    arps_exponential(qi, Di, t)
    arps_hyperbolic(qi, Di, b, t)
    generate_gas_supply_profile(params, years)
"""

