"""
Stochastic Simulation Module

Implements Monte Carlo sampling for uncertain variables including
gas price, carbon price, solar CAPEX, and demand growth. Wraps the
deterministic optimization model to generate risk-adjusted metrics.

Functions:
    sample_uncertainties(N, distributions)
    run_stochastic_optimization(base_model, samples)
    compute_risk_metrics(results)
"""
