"""Supplied concessional-commercial division (TASK 98): the counterfactual for Chapter 1 Question 2.

Under capital_treatment_case='f1_tranche' the division of each vintage's solar and storage capital between
concessional and commercial sources is a DECISION (m.<channel>_conc_vol[t] + m.<channel>_comm_vol[t] ==
capex_param[t] * add[t]), limited by one envelope constraint. This module lets a study SUPPLY the division
instead: for every eligible channel (solar_public, solar_eaas, storage) and vintage t it adds the linear equality

        m.<channel>_conc_vol[t] == s_t * m.<channel>_capex_param[t] * m.<channel>_add[t]

with s_t exogenous data. The coefficient s_t * capex_param[t] is data, so the programme stays linear (the
linearity invariant of the F1 block). Gas stays commercial-only (its conc_vol is fixed at 0 in the builder).

scenario["conc_share_rule"] is None (inert: nothing is added, every stored result and hash is unchanged) or a dict
        {"name": one of RULES,
         "reference": {"solar_add_mw_by_year": {year: MW}, "storage_add_mwh_by_year": {year: MWh}}   # split_prorata only
         "realised_year": int                                                                      # split_central_realised only (default 2035)
         "first_eligible_year": int}                                                               # split_frontload only (default: first model year)
Rules:
  split_prorata          constant share s = envelope / K_ref across all vintages, K_ref = sum_t capex_param[t] * reference add[t]
                         over solar and storage (the blend-ratio convention: a study divides the facility by the capital it plans).
  split_frontload        s = 1.0 at the first eligible vintage, nil thereafter (spend the facility as early as possible).
  split_zero             s = 0 everywhere (no concessional capital: the regression arm that must equal env_none).
  split_central_realised s = 1.0 at realised_year, nil elsewhere (the central endogenous solution's own division: the regression arm
                         that must reproduce the endogenous central).
The envelope constraint of the builder REMAINS. A supplied share vector that would draw more concessional volume than the
envelope allows therefore binds the plan (the LP stays feasible but is constrained); the arm is flagged where it binds
(m.f1_concessional_envelope_constraint slack) rather than the envelope being relaxed.
"""
import pyomo.environ as pyo

RULES = ("split_prorata", "split_frontload", "split_zero", "split_central_realised")
CHANNELS = ("solar_public", "solar_eaas", "storage")


def resolve_shares(rule, years, capex_solar, capex_storage, envelope_usd):
    """Return the exogenous share s_t for every model year (list aligned with `years`).
    capex_solar / capex_storage: callables year-index -> USD per MW / per MWh."""
    name = rule["name"]
    n = len(years)
    if name not in RULES:
        raise ValueError(f"unknown conc_share_rule {name!r}; choices {RULES}")
    if name == "split_zero":
        return [0.0] * n
    if name == "split_central_realised":
        y0 = int(rule.get("realised_year", 2035))
        if y0 not in years:
            raise ValueError(f"realised_year {y0} outside the horizon")
        return [1.0 if int(y) == y0 else 0.0 for y in years]
    if name == "split_frontload":
        y0 = int(rule.get("first_eligible_year", years[0]))
        if y0 not in years:
            raise ValueError(f"first_eligible_year {y0} outside the horizon")
        return [1.0 if int(y) == y0 else 0.0 for y in years]
    ref = rule.get("reference")
    if not ref:
        raise ValueError("split_prorata needs rule['reference'] (a plan whose eligible capital calibrates the share)")
    sol = {int(k): float(v) for k, v in ref["solar_add_mw_by_year"].items()}
    sto = {int(k): float(v) for k, v in ref["storage_add_mwh_by_year"].items()}
    k_ref = sum(capex_solar(t) * sol.get(int(y), 0.0) + capex_storage(t) * sto.get(int(y), 0.0) for t, y in enumerate(years))
    if k_ref <= 0:
        raise ValueError("reference plan has no eligible capital")
    s = float(envelope_usd) / k_ref
    if not (0.0 < s <= 1.0):
        raise ValueError(f"split_prorata share {s:.6g} outside (0, 1]: the envelope exceeds the reference plan's eligible capital")
    rule["_calibrated_share"] = s
    rule["_reference_eligible_capex_usd"] = k_ref
    return [s] * n


def add_supplied_division_constraints(m, scenario, T, rule):
    """Add the supplied-division equalities to a built f1_tranche model (called by both builders)."""
    years = [int(y) for y in scenario["years"]]
    shares = resolve_shares(rule, years, lambda t: pyo.value(m.solar_capex_param[t]), lambda t: pyo.value(m.storage_capex_param[t]),
                            scenario["concessional_envelope_usd"])
    m.supplied_conc_share = pyo.Param(T, initialize={t: shares[t] for t in T}, mutable=False)
    m.supplied_solar_public_conc = pyo.Constraint(T, rule=lambda m, t: m.solar_public_conc_vol[t] == m.supplied_conc_share[t] * m.solar_capex_param[t] * m.solar_public_add[t])
    m.supplied_solar_eaas_conc = pyo.Constraint(T, rule=lambda m, t: m.solar_eaas_conc_vol[t] == m.supplied_conc_share[t] * m.solar_capex_param[t] * m.solar_eaas_add[t])
    m.supplied_storage_conc = pyo.Constraint(T, rule=lambda m, t: m.storage_conc_vol[t] == m.supplied_conc_share[t] * m.storage_capex_param[t] * m.storage_add[t])
    return shares
