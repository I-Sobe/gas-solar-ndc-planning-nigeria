"""
diagnose_financing_endogeneity.py -- Part B of the financing-endogeneity task
(TASK_financing_endogeneity.md + TASK.md release note).

QUESTION
--------
Chapter 2 s2.5.2 calls a financing treatment endogenous only where financing
structure / cost of capital is (i) a decision variable, (ii) a solution
outcome, or (iii) a constraint interacting with the capacity solution. Part A
(docs/financing_endogeneity.md) established statically that no rate is a Var,
that every CRF/W is a precomputed float, and that no rate responds to risk.
This script runs the empirical tests that Part A could not settle.

TESTS
-----
 1   Envelope sweep (0, 0.5x, 1x, 5x, 10x env_dares), rate inputs fixed.
     Realised blended CRF / implied rate / portfolio rate. DEMOTED: shows the
     realised blend is capable of varying; calibrates the range.
 1b  DECISIVE. Envelope fixed at env_dares and every financing input fixed;
     vary only demand growth, gas regime, emissions cap, solar yield.
 2   Split freedom: envelope dual, slack-with-commercial-use, interior shares
     under slack AND binding envelope, distribution across cells.
 3   Bankability interaction with capacity (eaas regime only): (a) central
     dscr_case, (b) loosest dscr_case, (b2) a tighter case, (c) constraint
     deactivated via m.eaas_bankability.deactivate() (no src/ edit).
 4   Risk-to-rate feedback: static (Part A, A4). Restated in the report.
 5   Negative control: zero envelope across the SAME variations as 1b.

PANELS (regime x public capital case)
-------------------------------------
 P1_trad_unc    traditional, capital_case=unconstrained  (Phase B item 0: 01b_run_baseline_sliced.py)
 P2_eaas_mod    eaas,        capital_case=moderate       (Phase B item 2: 02_run_eaas.py)
 P3_eaas_unc    eaas,        capital_case=unconstrained  (isolates the public budget as a confound)
Every panel is reported; the public-budget constraint's slack/dual is recorded
in every solve.

USAGE
-----
    python scripts/diagnose_financing_endogeneity.py            # everything (~70 solves)
    python scripts/diagnose_financing_endogeneity.py --smoke    # ONE solve, prints metrics, no report
    python scripts/diagnose_financing_endogeneity.py --dry-run  # builds every model, no solve
    python scripts/diagnose_financing_endogeneity.py --report-only   # regenerate report from cache
Completed solves are cached in results/financing_endogeneity/runs/<id>.json
and skipped on re-run (delete the file to redo one).

Writes ONLY: results/financing_endogeneity/ and the marked Part B section of
docs/financing_endogeneity.md (Part A above the marker is never touched).
Part C (classification + paragraph) is written by hand after reading the output.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.scenarios import (                                    # noqa: E402
    load_scenario, MODEL_END_YEAR, concessional_envelope_scenarios,
    dscr_scenarios,
)
from src.optimize_model import solve_model                      # noqa: E402
from src.optimize_model_sliced import build_model_sliced        # noqa: E402
from src.io import load_econ, load_solar_capex_by_year          # noqa: E402
from src.utils import json_safe                                 # noqa: E402

OUT_DIR = ROOT / "results" / "financing_endogeneity"
RUN_DIR = OUT_DIR / "runs"
DOC_PATH = ROOT / "docs" / "financing_endogeneity.md"
CAP_PATH = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"
BEGIN = "<!-- BEGIN PART B AUTOGEN (scripts/diagnose_financing_endogeneity.py) -->"
END = "<!-- END PART B AUTOGEN -->"

# Central arms: copied from scripts/run_phase_b.sh lines 189-207 (C_*), the
# same values Phase B ran. Financing arms are NEVER varied except where a test
# says so.
CENTRAL = dict(
    peak_multiple_case="peak_unsuppressed", reserve_margin="nirp_spinning",
    storage_cc_case="storage_cc_central", gas_build_case="gas_unconstrained",
    new_gas_deliverability="constrained", gas_availability_case="gas_avail_nameplate",
    gas_retirement_case="gas_retire_vintage_30", gas_capex_case="gas_capex_central",
    ccgt_efficiency_case="ccgt_eff_central", capital_recovery_case="cr_central",
    dscr_case="dscr_central", us_inflation=0.020, capital_treatment_case="f1_tranche",
    conversion_case="conv_central", hydro_case="hydro_flat",
    tariff_case="tariff_central", om_case="om_central",
)
CENTRAL_GAS_DELIV = "baseline"
CENTRAL_DEMAND = "organic_central"
CENTRAL_CF = 0.19            # asserted against scenario["solar_cf"] at run time
CAP_ARM = "ndc3_conditional"  # emissions_cap.csv scenario used as the "binding cap" arm

PANELS = {
    "P1_trad_unc": dict(regime="traditional", capital_case="unconstrained"),
    "P2_eaas_mod": dict(regime="eaas", capital_case="moderate"),
    "P3_eaas_unc": dict(regime="eaas", capital_case="unconstrained"),
}

# Test 1b / 5 variations: (label, group, overrides)
VARIATIONS = [
    ("base",             "base", {}),
    ("b1_demand_low",    "b1", {"demand_case": "constrained_continuation"}),
    ("b1_demand_high",   "b1", {"demand_case": "organic_high"}),
    ("b2_gas_downside",  "b2", {"gas_case": "downside"}),
    ("b2_gas_upside",    "b2", {"gas_case": "upside"}),
    ("b2_gas_shock",     "b2", {"gas_case": "shock_recovery"}),
    ("b3_cap_ndc3cond",  "b3", {"cap_name": CAP_ARM}),
    ("b4_cf_0.16",       "b4", {"cf": 0.16}),
    ("b4_cf_0.20",       "b4", {"cf": 0.20}),
]
ENV_SWEEP = [  # (label, arm passed to load_scenario, override_usd or None)
    ("env_none", "env_none", None),
    ("env_0.5x", "env_dares", "half"),
    ("env_dares", "env_dares", None),
    ("env_5x", "env_5x", None),
    ("env_10x", "env_10x", None),
]
CHANNELS = ("solar_public", "solar_eaas", "storage", "gas")
ELIGIBLE = ("solar_public", "solar_eaas", "storage")
TECH = {"solar_public": "solar", "solar_eaas": "solar", "storage": "storage", "gas": "gas"}
TOL_USD = 1.0e3          # envelope / budget "binds" tolerance (USD; capex is O(1e9))
SHARE_EPS = 1.0e-6
NOISE_BP = 0.01          # below this a portfolio-rate range is treated as numerical noise


# =====================================================================
# Pure helpers (unit-testable without a solve)
# =====================================================================

def crf(r, L):
    """Replicates optimize_model_sliced._crf (lines 1542-1557), incl. the r<=0 floor."""
    if r <= 0:
        return 1.0 / L
    return r * (1.0 + r) ** L / ((1.0 + r) ** L - 1.0)


def crf_inverse(target, L):
    """Real rate r >= 0 with CRF(r, L) = target (bisection). Floors at 0 when target <= 1/L."""
    if target is None:
        return None
    if target <= 1.0 / L + 1e-12:
        return 0.0
    lo, hi = 0.0, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if crf(mid, L) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rate_terms(scenario):
    """Per-tech CRFs and rates exactly as the builder derives them (lines 1559-1613)."""
    infl = float(scenario["us_inflation"])
    r_conc = (1.0 + float(scenario["r_conc_nominal"])) / (1.0 + infl) - 1.0
    r_comm = (1.0 + float(scenario["r_comm_nominal"])) / (1.0 + infl) - 1.0
    lives = scenario["asset_lifetimes"]
    Lc, Lm = int(round(float(scenario["L_conc"]))), int(round(float(scenario["L_comm"])))
    t = {}
    for tech in ("solar", "storage", "gas"):
        life = int(round(float(lives[tech])))
        Le_c, Le_m = min(Lc, life), min(Lm, life)
        t[tech] = dict(L_conc=Le_c, L_comm=Le_m,
                       crf_conc=crf(r_conc, Le_c), crf_comm=crf(r_comm, Le_m))
    return dict(r_conc=r_conc, r_comm=r_comm, tech=t)


def window(DF, v, L, T_last):
    return sum(DF[t] for t in range(v, min(v + L, T_last + 1)))


def analyse(raw, terms):
    """raw: volumes etc. -> flat metrics dict + per-cell rows. No Pyomo here."""
    T = len(raw["years"])
    r_c, r_m = terms["r_conc"], terms["r_comm"]
    cells, flat = [], {}
    pv = {"elig": [0.0, 0.0, 0.0, 0.0], "all4": [0.0, 0.0, 0.0, 0.0]}  # vol, conc, crf*vol, r*vol
    for ch in CHANNELS:
        tt = terms["tech"][TECH[ch]]
        vol_ch = conc_ch = 0.0
        for v in range(T):
            conc = max(0.0, raw["ch"][ch]["conc"][v])
            comm = max(0.0, raw["ch"][ch]["comm"][v])
            vol = conc + comm
            if vol <= 1.0:      # < 1 USD: no capital raised
                continue
            c_real = (tt["crf_conc"] * conc + tt["crf_comm"] * comm) / vol
            r_vw = (r_c * conc + r_m * comm) / vol
            cells.append(dict(channel=ch, vintage=int(raw["years"][v]), conc=conc, comm=comm,
                              share=conc / vol, crf_real=c_real, r_vw=r_vw,
                              r_impl_Lcomm=crf_inverse(c_real, tt["L_comm"]),
                              r_impl_Lconc=crf_inverse(c_real, tt["L_conc"])))
            vol_ch += vol
            conc_ch += conc
            for key in (("all4",) + (("elig",) if ch in ELIGIBLE else ())):
                pv[key][0] += vol
                pv[key][1] += conc
                pv[key][2] += c_real * vol
                pv[key][3] += r_vw * vol
        if vol_ch > 0:
            cc = [c for c in cells if c["channel"] == ch]
            crf_ch = sum(c["crf_real"] * (c["conc"] + c["comm"]) for c in cc) / vol_ch
            flat[f"{ch}_vol_usd"] = vol_ch
            flat[f"{ch}_conc_share"] = conc_ch / vol_ch
            flat[f"{ch}_crf_real"] = crf_ch
            flat[f"{ch}_r_vw"] = sum(c["r_vw"] * (c["conc"] + c["comm"]) for c in cc) / vol_ch
            flat[f"{ch}_r_impl_Lcomm"] = crf_inverse(crf_ch, tt["L_comm"])
            flat[f"{ch}_r_impl_Lconc"] = crf_inverse(crf_ch, tt["L_conc"])
        else:
            for k in ("vol_usd", "conc_share", "crf_real", "r_vw", "r_impl_Lcomm", "r_impl_Lconc"):
                flat[f"{ch}_{k}"] = None
    for key, (vol, conc, crfv, rv) in pv.items():
        flat[f"port_{key}_vol_usd"] = vol
        flat[f"port_{key}_conc_share"] = conc / vol if vol > 0 else None
        flat[f"port_{key}_crf"] = crfv / vol if vol > 0 else None
        flat[f"port_{key}_r_vw"] = rv / vol if vol > 0 else None
        # implied rate inverted at the COMMERCIAL tenor (declared; see report)
        flat[f"port_{key}_r_impl_Lcomm"] = (
            crf_inverse(crfv / vol, terms["tech"]["solar"]["L_comm"]) if vol > 0 else None)
    # Portfolio including rebuild + conversion capital (commercial-only, no tranche Var)
    extra = raw.get("rebuild_conv_vol_usd", 0.0)
    vol, rv = pv["all4"][0] + extra, pv["all4"][3] + r_m * extra
    flat["port_all_incl_rebuild_vol_usd"] = vol
    flat["port_all_incl_rebuild_r_vw"] = rv / vol if vol > 0 else None
    return flat, cells


def compress(vs):
    vs = sorted(vs)
    if not vs:
        return "-"
    out, s, p = [], vs[0], vs[0]
    for x in vs[1:]:
        if x == p + 1:
            p = x
            continue
        out.append(f"{s}" if s == p else f"{s}-{p}")
        s = p = x
    out.append(f"{s}" if s == p else f"{s}-{p}")
    return ",".join(out)


def alloc_summary(cells):
    parts = []
    for ch in ELIGIBLE:
        cc = [c for c in cells if c["channel"] == ch]
        if not cc:
            continue
        full = [c["vintage"] for c in cc if c["share"] >= 1 - SHARE_EPS]
        zero = [c["vintage"] for c in cc if c["share"] <= SHARE_EPS]
        inter = [f'{c["vintage"]}:{c["share"]:.3f}' for c in cc if SHARE_EPS < c["share"] < 1 - SHARE_EPS]
        parts.append(f"{ch}: full[{compress(full)}] interior[{';'.join(inter) or '-'}] none[{compress(zero)}]")
    return " | ".join(parts)


# =====================================================================
# Run specification, build, solve, extract
# =====================================================================

def make_run(panel, env, var, t3="central", **extra):
    env_label, env_arm, env_ovr = env
    label, group, ovr = var
    p = PANELS[panel]
    spec = dict(panel=panel, regime=p["regime"], capital_case=p["capital_case"],
                env_label=env_label, env_arm=env_arm, env_override=env_ovr,
                variation=label, group=group, demand_case=CENTRAL_DEMAND,
                gas_case=CENTRAL_GAS_DELIV, cf=None, cap_name=None,
                dscr_case=CENTRAL["dscr_case"], bankability_off=False, t3=t3)
    spec.update(ovr)
    spec.update(extra)
    spec["id"] = "__".join([panel, env_label, label, t3])
    return spec


def build_run_list(panels):
    runs = {}

    def add(r):
        runs.setdefault(r["id"], r)

    for panel in panels:
        # Test 1: envelope sweep, base variation
        for env in ENV_SWEEP:
            add(make_run(panel, env, VARIATIONS[0]))
        # Test 1b (env_dares) and Test 5 (env_none): the variations
        for env in (ENV_SWEEP[2], ENV_SWEEP[0]):
            for var in VARIATIONS:
                add(make_run(panel, env, var))
        # Test 3 (eaas only)
        if PANELS[panel]["regime"] == "eaas":
            e = ENV_SWEEP[2]
            base = VARIATIONS[0]
            add(make_run(panel, e, base, t3="a_central"))
            add(make_run(panel, e, base, t3="b_loosest", dscr_case="dscr_regression"))
            add(make_run(panel, e, base, t3="b2_tighter", dscr_case="dscr_high"))
            add(make_run(panel, e, base, t3="c_bankability_off", bankability_off=True))
    # the Test 1 env_dares/base run is the same object as Test 3 (a) in spirit but
    # keep them separate ids so each table is self-contained; the cost is 1 solve per panel.
    return list(runs.values())


def build_scenario(spec):
    kw = dict(CENTRAL)
    kw["dscr_case"] = spec["dscr_case"]
    scenario = load_scenario(
        demand_level_case="served", demand_case=spec["demand_case"],
        capital_case=spec["capital_case"], gas_deliverability_case=spec["gas_case"],
        solar_build_case="deployment_unconstrained", land_case="loose",
        carbon_case="no_policy", start_year=2025, end_year=MODEL_END_YEAR,
        concessional_envelope_case=spec["env_arm"], **kw,
    )
    scenario["financing_regime"] = spec["regime"]
    if spec["env_override"] == "half":
        scenario["concessional_envelope_usd"] = 0.5 * concessional_envelope_scenarios()["env_dares"]
    if spec["cf"] is not None:
        scenario["solar_cf"] = spec["cf"]
    else:
        assert abs(float(scenario["solar_cf"]) - CENTRAL_CF) < 1e-9, (
            f"central solar_cf is {scenario['solar_cf']}, expected {CENTRAL_CF}")
    return scenario


def load_caps(name, years):
    df = pd.read_csv(CAP_PATH)
    df = df[(df["scenario"] == name) & (df["year"].isin([int(y) for y in years]))].sort_values("year")
    caps = df["cap_tco2"].astype(float).tolist()
    if len(caps) != len(years):
        raise ValueError(f"cap {name}: {len(caps)} values for {len(years)} years")
    return caps


def build(spec, econ, capex_tv):
    scenario = build_scenario(spec)
    caps = load_caps(spec["cap_name"], scenario["years"]) if spec["cap_name"] else None
    m = build_model_sliced(scenario=scenario, econ=econ, emissions_cap=None,
                           emissions_cap_by_year=caps, solar_capex_by_year=capex_tv)
    if spec["bankability_off"]:
        m.eaas_bankability.deactivate()
    return m, scenario


def _v(x):
    y = pyo.value(x, exception=False)
    return 0.0 if y is None else float(y)


def extract(m, scenario, econ):
    years = [int(y) for y in scenario["years"]]
    T = range(len(years))
    raw = dict(years=years, ch={}, adds={})
    names = {"solar_public": ("solar_public_conc_vol", "solar_public_comm_vol"),
             "solar_eaas": ("solar_eaas_conc_vol", "solar_eaas_comm_vol"),
             "storage": ("storage_conc_vol", "storage_comm_vol"),
             "gas": ("gas_conc_vol", "gas_comm_vol")}
    for ch, (a, b) in names.items():
        raw["ch"][ch] = dict(conc=[_v(getattr(m, a)[t]) for t in T],
                             comm=[_v(getattr(m, b)[t]) for t in T])
    for k in ("solar_public_add", "solar_eaas_add", "storage_add", "gas_add", "eaas_subsidy"):
        raw["adds"][k] = [_v(getattr(m, k)[t]) for t in T]
    raw["capex_param"] = [_v(m.solar_capex_param[t]) for t in T]
    raw["DF"] = [_v(m.DF[t]) for t in T]
    # rebuild + conversion capital (commercial only, no tranche Var): replicate
    # _rebuild_years (builder 1641-1648) volumes, undiscounted.
    lives, T_last = scenario["asset_lifetimes"], len(years) - 1

    def rebuild_years(b, life):
        life = int(round(float(life)))
        k, out = 1, []
        while b + k * life <= T_last:
            out.append(b + k * life)
            k += 1
        return out

    extra = 0.0
    for b in T:
        for rt in rebuild_years(b, lives["solar"]):
            extra += (raw["adds"]["solar_public_add"][b] + raw["adds"]["solar_eaas_add"][b]) * raw["capex_param"][rt]
        for rt in rebuild_years(b, lives["storage"]):
            extra += raw["adds"]["storage_add"][b] * float(m.storage_capex_param[rt])   # TASK 53 2.2: the rebuild year's price, as the builder charges it
        for rt in rebuild_years(b, lives["gas"]):
            extra += raw["adds"]["gas_add"][b] * float(scenario["gas_capex_per_mw"])
    if hasattr(m, "convert_comm_vol"):
        extra += sum(_v(m.convert_comm_vol[k]) for k in m.convert_comm_vol)
    raw["rebuild_conv_vol_usd"] = extra
    return raw


def constraint_state(m, scenario, raw, terms):
    """Envelope, public budget, bankability, emissions-cap diagnostics."""
    T = range(len(raw["years"]))
    out = {}
    env = m.f1_concessional_envelope_constraint
    B = float(scenario["concessional_envelope_usd"])
    drawn = sum(sum(raw["ch"][c]["conc"]) for c in ELIGIBLE)
    d = m.dual.get(env, None)
    out.update(envelope_usd=B, envelope_drawn_usd=drawn, envelope_slack_usd=B - drawn,
               envelope_binds=bool(B - drawn <= TOL_USD),
               envelope_dual_pv_per_undiscounted_usd=None if d is None else float(d))
    # public budget
    if hasattr(m, "public_budget_constraint"):
        c = m.public_budget_constraint
        body, up = _v(c.body), _v(c.upper)
        dd = m.dual.get(c, None)
        out.update(public_budget_built=True, public_budget_usd=up, public_budget_slack_usd=up - body,
                   public_budget_binds=bool(up - body <= TOL_USD),
                   public_budget_dual=None if dd is None else float(dd))
    else:
        out.update(public_budget_built=False, public_budget_usd=None, public_budget_slack_usd=None,
                   public_budget_binds=False, public_budget_dual=None)
    # bankability (vintage-years with eaas_add > 0 only; elsewhere it is 0 <= 0)
    n_bind = n_active = 0
    active = m.eaas_bankability.active
    for t in T:
        add = raw["adds"]["solar_eaas_add"][t]
        if add <= 1e-6:
            continue
        n_active += 1
        c = m.eaas_bankability[t]
        lhs = raw["capex_param"][t] * add
        slack = _v(c.upper) - _v(c.body)
        if slack <= 1e-6 * (abs(lhs) + 1.0):
            n_bind += 1
    out.update(bankability_active=bool(active), bankability_vintages_with_eaas=n_active,
               bankability_binding_vintage_years=n_bind)
    # subsidy
    sub = raw["adds"]["eaas_subsidy"]
    out.update(subsidy_total_usd=sum(sub), subsidy_npv_usd=sum(s * df for s, df in zip(sub, raw["DF"])))
    # emissions cap
    if hasattr(m, "emissions_constraint"):
        nb = 0
        for t in T:
            try:
                dd = m.dual.get(m.emissions_constraint[t], None)
            except Exception:
                dd = None
            if dd is not None and abs(dd) > 1e-9:
                nb += 1
        out["emissions_cap_binding_years"] = nb
    else:
        out["emissions_cap_binding_years"] = None
    return out


def charge_check(m, raw, terms):
    """Recompute the three tranche capital charges from the extracted volumes and
    this script's own CRF x W; compare with the model's Expressions. Verifies that
    the CRFs used in the REPORT are the CRFs used in the OBJECTIVE."""
    DF, T_last = raw["DF"], len(raw["years"]) - 1
    worst = 0.0
    for ch, expr in (("solar_public", m.cost_public_solar_capital_charge_npv),
                     ("solar_eaas", m.cost_eaas_solar_capital_charge_npv),
                     ("storage", m.cost_storage_capital_charge_npv),
                     ("gas", m.cost_gas_capital_charge_npv)):
        tt = terms["tech"][TECH[ch]]
        mine = 0.0
        for v in range(len(DF)):
            mine += tt["crf_conc"] * window(DF, v, tt["L_conc"], T_last) * raw["ch"][ch]["conc"][v]
            mine += tt["crf_comm"] * window(DF, v, tt["L_comm"], T_last) * raw["ch"][ch]["comm"][v]
        model = _v(expr)
        worst = max(worst, abs(mine - model) / max(1.0, abs(model)))
    return worst


def plan_metrics(m, raw, scenario):
    a = raw["adds"]
    return dict(
        npv_cost_usd=_v(m.system_cost_npv),
        solar_public_mw=sum(a["solar_public_add"]), solar_eaas_mw=sum(a["solar_eaas_add"]),
        solar_total_mw=sum(a["solar_public_add"]) + sum(a["solar_eaas_add"]),
        storage_mwh=sum(a["storage_add"]), gas_add_mw=sum(a["gas_add"]),
        emissions_tco2=_v(m.emissions),
        unserved_twh=sum(_v(m.unserved[t, s]) for t in range(len(raw["years"])) for s in m.S),
    )


def run_one(spec, econ, capex_tv, solver):
    t0 = time.time()
    rec = dict(spec=spec, status="error")
    try:
        m, scenario = build(spec, econ, capex_tv)
        terms = rate_terms(scenario)
        rec["inputs"] = dict(
            r_conc_nominal=float(scenario["r_conc_nominal"]), r_comm_nominal=float(scenario["r_comm_nominal"]),
            us_inflation=float(scenario["us_inflation"]), L_conc=scenario["L_conc"], L_comm=scenario["L_comm"],
            r_conc_real=terms["r_conc"], r_comm_real=terms["r_comm"],
            required_margin=float(scenario["required_margin"]),
            tariff_usd_per_twh=float(scenario["solar_service_tariff_usd_per_twh"]),
            envelope_usd=float(scenario["concessional_envelope_usd"]),
            financing_regime=scenario["financing_regime"], solar_cf=float(scenario["solar_cf"]),
            bank_tenor_case=scenario.get("bank_tenor_case"),
            public_solar_budget_npv=scenario.get("public_solar_budget_npv"),
            demand_case=spec["demand_case"], gas_case=spec["gas_case"], cap_name=spec["cap_name"],
        )
        try:
            st = solve_model(m, scenario=scenario, solver=solver)
        except RuntimeError as e:
            rec.update(status="infeasible_or_failed", message=str(e)[:300], seconds=time.time() - t0)
            return rec
        if not st["optimal"]:
            rec.update(status="not_optimal", message=str(st), seconds=time.time() - t0)
            return rec
        raw = extract(m, scenario, econ)
        flat, cells = analyse(raw, terms)
        rec.update(status="optimal", flat=flat, cells=cells, alloc=alloc_summary(cells),
                   constraints=constraint_state(m, scenario, raw, terms),
                   plan=plan_metrics(m, raw, scenario),
                   charge_check_rel_err=charge_check(m, raw, terms),
                   solver_version=st.get("solver_version"), seconds=time.time() - t0)
    except Exception as e:  # recorded, not swallowed: shows in the report's failure list
        rec.update(status="error", message=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:],
                   seconds=time.time() - t0)
    return rec


# =====================================================================
# Reporting
# =====================================================================

def pct(x, nd=4):
    return "n/a" if x is None else f"{100 * x:.{nd}f}%"


def bp(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def usd(x):
    return "n/a" if x is None else f"{x / 1e6:,.1f}M"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def get(recs, **kw):
    for r in recs:
        s = r["spec"]
        if all(s.get(k) == v for k, v in kw.items()):
            return r
    return None


def ok(r):
    return r is not None and r["status"] == "optimal"


def render(recs, args):
    L = []
    w = L.append
    w(BEGIN)
    w("")
    w("# Part B — empirical tests (generated by `scripts/diagnose_financing_endogeneity.py`)")
    w("")
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Solver `{args.solver}`. "
      f"{sum(1 for r in recs if r['status'] == 'optimal')} optimal / {len(recs)} runs. "
      "These are VERIFICATION artefacts, not thesis findings. Part C is written by hand after this section.")
    w("")
    # ---- manifest
    w("## B0. Run manifest — exactly what each run used")
    w("")
    w("Held at the Phase B central arms in every run (from `scripts/run_phase_b.sh` 189-207): "
      + ", ".join(f"`{k}={v}`" for k, v in CENTRAL.items()) + f". Solar build `deployment_unconstrained`, "
      f"carbon `no_policy`, `voll_mid`, `genset_central`, capex `solar_low`.")
    w("")
    w("**Regimes Phase B actually ran** (read, not assumed): item 0 `01b_run_baseline_sliced.py` — "
      "`financing_regime` is never set, so `load_scenario()`'s default `\"traditional\"` applies "
      "(`src/scenarios.py:3373`), with `capital_case=\"unconstrained\"` (`01b…py:297`). Item 2 "
      "`02_run_eaas.py` sets `scenario[\"financing_regime\"] = \"eaas\"` (`02_run_eaas.py:260`) with "
      "`capital_case=\"moderate\"` (`02_run_eaas.py:227`).")
    w("")
    first = next((r for r in recs if r["status"] in ("optimal",) and "inputs" in r), None)
    if first:
        i = first["inputs"]
        w(f"Fixed financing inputs (identical in every run unless the row says otherwise): "
          f"r_conc nominal {pct(i['r_conc_nominal'], 2)} / real {pct(i['r_conc_real'], 4)}, L_conc {i['L_conc']}; "
          f"r_comm nominal {pct(i['r_comm_nominal'], 2)} / real {pct(i['r_comm_real'], 4)}, L_comm {i['L_comm']}; "
          f"inflation {pct(i['us_inflation'], 2)}; required_margin {i['required_margin']}; "
          f"tariff {i['tariff_usd_per_twh']:,.0f} USD/TWh; bank_tenor_case `{i['bank_tenor_case']}`.")
        w("")
    pb = {}
    for r in recs:
        if "inputs" in r:
            pb[r["spec"]["capital_case"]] = r["inputs"]["public_solar_budget_npv"]
    w("**Public budget** (`capital_envelope_scenarios()`): "
      + "; ".join(f"`{k}` = {('none (constraint NOT built)' if v is None else f'{v:,.0f} USD NPV (constraint built, sliced:2005-2016)')}"
                  for k, v in pb.items()) + ".")
    w("")
    rows = []
    for r in recs:
        s = r["spec"]
        ovr = ""
        if s["env_override"] == "half":
            ovr = f"override {0.5 * concessional_envelope_scenarios()['env_dares']:,.0f}"
        rows.append([s["id"], s["regime"], s["capital_case"],
                     f"{s['env_label']} (arm `{s['env_arm']}` {ovr}) = {usd(r.get('inputs', {}).get('envelope_usd'))}",
                     s["variation"], s["demand_case"], s["gas_case"], s["cf"] if s["cf"] else "central 0.19",
                     s["cap_name"] or "none", s["dscr_case"], "OFF" if s["bankability_off"] else "on", r["status"]])
    w(md_table(["run id", "regime", "capital_case", "envelope", "variation", "demand", "gas deliv.",
                "solar CF", "emissions cap", "dscr_case", "bankability", "status"], rows))
    w("")

    bad = [r for r in recs if r["status"] != "optimal"]
    chk = [r for r in recs if r["status"] == "optimal" and r["charge_check_rel_err"] > 1e-6]
    if chk:
        w(f"**WARNING — charge check failed** (script's CRF×W ≠ model's capital-charge Expression, rel. err > 1e-6) in: "
          + ", ".join(r['spec']['id'] for r in chk) + ". Realised-rate figures for these runs are NOT trustworthy.")
        w("")
    else:
        w("Charge check: in every optimal run the tranche capital charges recomputed from this script's CRF×W and "
          "the extracted volumes match the model's `cost_*_capital_charge_npv` Expressions to < 1e-6 relative, "
          "so the CRFs used below are the CRFs in the objective.")
        w("")

    w("**Definitions (declared choices).** `CRF_realised(c,v)` is exactly the task's formula. Its implied real "
      "rate `r_impl` inverts `CRF(r, L)` and needs a single tenor, but a blended cell has two (solar 25/10, "
      "storage 15/10, gas n/a). Reported at the COMMERCIAL tenor `L_comm` (and, in CSV, also at `L_conc`); "
      "this makes `r_impl` a tenor-convention-dependent summary (when a blended CRF falls below 1/L_comm, because the "
      "concessional CRF is on a 25-yr tenor, the inversion has no solution at L_comm and floors at 0.000% — a tenor artefact, "
      "not a zero rate). The tenor-free alternative reported beside it, "
      "`r_vw`, is the capital-weighted average of the two tranche REAL rates (the blended-WACC sense of "
      "`blended_finance.blended_wacc`). Conclusions about whether the rate MOVES should be read from both. "
      "Portfolio weights are undiscounted capital volumes. `elig` = public solar + EaaS solar + storage "
      "(concessional-eligible); `all4` adds gas (100% commercial by construction); `+rebuild` additionally "
      "adds rebuild and conversion capital (commercial, no tranche Var).")
    w("")

    for panel, p in PANELS.items():
        pr = [r for r in recs if r["spec"]["panel"] == panel]
        if not pr:
            continue
        w(f"---\n\n## Panel `{panel}` — regime `{p['regime']}`, capital_case `{p['capital_case']}`\n")
        # ---------------- Test 1
        w("### Test 1 — envelope sweep (limb ii, weak form; DEMOTED: calibrates the range)\n")
        rows = []
        for env in ENV_SWEEP:
            r = get(pr, env_label=env[0], variation="base", t3="central")
            if not ok(r):
                rows.append([env[0], "FAILED"] + [""] * 9)
                continue
            f = r["flat"]
            rows.append([env[0], usd(r["inputs"]["envelope_usd"]), pct(f["port_elig_conc_share"], 2),
                         *[f"{f[f'{c}_crf_real']:.5f}" if f[f"{c}_crf_real"] is not None else "–" for c in CHANNELS],
                         *[pct(f[f"{c}_r_impl_Lcomm"], 3) if f[f"{c}_r_impl_Lcomm"] is not None else "–" for c in CHANNELS],
                         pct(f["port_elig_r_vw"], 4), pct(f["port_all4_r_vw"], 4)])
        w(md_table(["envelope", "B (USD)", "conc. share (elig.)",
                    "CRF solar_pub", "CRF solar_eaas", "CRF storage", "CRF gas",
                    "r_impl solar_pub", "r_impl solar_eaas", "r_impl storage", "r_impl gas",
                    "portfolio r_vw (elig.)", "portfolio r_vw (all4)"], rows))
        base = [get(pr, env_label=e[0], variation="base", t3="central") for e in ENV_SWEEP]
        base = [r for r in base if ok(r)]
        if len(base) >= 2:
            for key in ("port_elig_r_vw", "port_all4_r_vw", "port_elig_r_impl_Lcomm"):
                vals = [r["flat"][key] for r in base if r["flat"][key] is not None]
                w(f"\n- `{key}` across the envelope sweep: min {pct(min(vals), 4)}, max {pct(max(vals), 4)}, "
                  f"**range {bp(1e4 * (max(vals) - min(vals)))} bp**.")
            w("\n**Interpretation rule (stated in advance).** If `r_realised` varies while every rate input is fixed, "
              "the cost of capital is a quantity the solution determines (limb ii, WEAK form: it responds to the "
              "analyst's financing dial). If invariant, limb ii is not satisfied.")
        w("")

        # ---------------- Test 1b
        w("### Test 1b — DECISIVE: financing inputs all fixed (envelope = env_dares); vary only demand / gas / cap / solar yield\n")
        rows, vals_el, vals_a4, vals_ir = [], [], [], []
        b0 = get(pr, env_label="env_dares", variation="base", t3="central")
        for var in VARIATIONS:
            r = get(pr, env_label="env_dares", variation=var[0], t3="central")
            if not ok(r):
                rows.append([var[0], r["status"] if r else "missing"] + [""] * 8)
                continue
            f, c, pl = r["flat"], r["constraints"], r["plan"]
            d_el = (1e4 * (f["port_elig_r_vw"] - b0["flat"]["port_elig_r_vw"])) if ok(b0) else None
            vals_el.append(f["port_elig_r_vw"])
            vals_a4.append(f["port_all4_r_vw"])
            vals_ir.append(f["port_all_incl_rebuild_r_vw"])
            rows.append([var[0], f"{pl['solar_total_mw']:,.0f}", f"{f['port_elig_vol_usd'] / 1e9:.3f}",
                         pct(f["port_elig_conc_share"], 3), pct(f["port_elig_r_vw"], 4), bp(d_el),
                         pct(f["port_all4_r_vw"], 4), pct(f["port_all_incl_rebuild_r_vw"], 4),
                         pct(f["port_elig_r_impl_Lcomm"], 4),
                         "yes" if c["envelope_binds"] else "no",
                         "n/a" if c["envelope_dual_pv_per_undiscounted_usd"] is None else f"{c['envelope_dual_pv_per_undiscounted_usd']:.4g}"])
        w(md_table(["variation", "solar built (MW)", "eligible capex (bn USD)", "conc. share (elig.)",
                    "portfolio r_vw (elig.)", "Δ vs base (bp)", "portfolio r_vw (all4)", "r_vw (+rebuild)",
                    "r_impl@L_comm (elig.)", "envelope binds", "envelope dual"], rows))
        if len(vals_el) >= 2:
            rg = {k: 1e4 * (max(v) - min(v)) for k, v in
                  (("elig", vals_el), ("all4", vals_a4), ("incl_rebuild", vals_ir))}
            w(f"\n**Magnitude across base + b1–b4:** range of portfolio real rate = **{bp(rg['elig'])} bp** (eligible), "
              f"{bp(rg['all4'])} bp (all four channels), {bp(rg['incl_rebuild'])} bp (incl. rebuild/conversion). "
              f"Numerical-noise floor taken as {NOISE_BP} bp.")
            verdict = ("MOVES by more than the noise floor → limb (ii) STRONG form is supported on this panel "
                       "(the realised cost of capital responds to quantities with no financing content)"
                       if rg["elig"] > NOISE_BP else
                       "does NOT move beyond the noise floor → limb (ii) strong form NOT supported on this panel")
            w(f"\n**Reading (eligible portfolio):** the realised rate {verdict}. "
              "Whether that movement is economically large is the bp figure above; this script does not set a threshold for 'large'.")
        # mechanism note
        w("\n**Mechanism note — cell-level allocation of the pooled envelope.** (`full` = share 1; `interior` = vintage:share; `none` = share 0; vintage index = model year.)\n")
        rows = []
        for var in VARIATIONS:
            r = get(pr, env_label="env_dares", variation=var[0], t3="central")
            if ok(r):
                rows.append([var[0], r["alloc"]])
        w(md_table(["variation", "allocation of concessional dollars across (channel, vintage)"], rows))
        w("\nThe eligible-portfolio `r_vw` is an exact function of the eligible concessional share "
          "(`r_comm − share·(r_comm − r_conc)`, both techs share the same two real rates), so movement in it can arise ONLY "
          "from the pooled fixed envelope being spread over a larger or smaller eligible capex total — it is not caused by "
          "reallocation across cells. Reallocation across (channel, vintage) cells changes the CRF-weighted figures "
          "(`CRF_realised`, `r_impl`) through tenor/channel mix but not `r_vw`. The `all4` and `+rebuild` portfolios additionally "
          "move with the commercial-only gas/rebuild capital share (channel mix), which is plan-dependent but is not a change "
          "in any channel's rate.\n")

        # ---------------- Test 2
        w("### Test 2 — does the split have real freedom? (limb i)\n")
        rows = []
        interior_bind, interior_slack, slack_comm = [], [], []
        for r in pr:
            if not ok(r) or r["spec"]["t3"] != "central":
                continue
            c, cells = r["constraints"], r["cells"]
            elig = [x for x in cells if x["channel"] in ELIGIBLE]
            interior = [x for x in elig if SHARE_EPS < x["share"] < 1 - SHARE_EPS]
            comm_used = any(x["comm"] > TOL_USD for x in elig)
            slack = c["envelope_slack_usd"] > TOL_USD
            sc = slack and comm_used
            if sc:
                slack_comm.append(r["spec"]["id"])
            if interior and slack:
                interior_slack.append(r["spec"]["id"])
            if interior and not slack:
                interior_bind.append((r["spec"]["id"], interior))
            rows.append([f"{r['spec']['env_label']}/{r['spec']['variation']}", usd(c["envelope_usd"]),
                         usd(c["envelope_drawn_usd"]), "yes" if c["envelope_binds"] else "no",
                         "n/a" if c["envelope_dual_pv_per_undiscounted_usd"] is None else f"{c['envelope_dual_pv_per_undiscounted_usd']:.4g}",
                         "YES" if sc else "no", len(interior),
                         "slack" if slack else "binding"])
        w(md_table(["run (envelope/variation)", "B", "drawn", "binds", "dual", "slack AND commercial used",
                    "# interior cells", "envelope state"], rows))
        w(f"\n- Runs with the envelope **slack while eligible commercial capital is still raised**: "
          f"{', '.join(slack_comm) if slack_comm else 'none'}.")
        w(f"- Runs with an interior share (0<share<1) **while the envelope is slack**: "
          f"{', '.join(interior_slack) if interior_slack else 'none'}.")
        w(f"- Runs with interior shares **while the envelope binds**: {len(interior_bind)}. "
          "Distribution across cells (channel vintage:share), first 12 runs:")
        for rid, cells_i in interior_bind[:12]:
            w(f"  - `{rid}`: " + "; ".join(f"{x['channel']} {x['vintage']}:{x['share']:.3f}" for x in cells_i))
        w("\nLP note: with one pooled linear envelope, basic solutions have at most a handful of fractional cells "
          "(the marginal ones); a long list of interior shares would indicate degeneracy/alternative optima rather than "
          "economic structure. The solver's choice among tied cells is arbitrary.\n")
        w("**Interpretation rule (stated in advance).** If the split is always 'fill the envelope, then commercial' the "
          "split is a decision variable in form but near-deterministic in content. If the share varies for reasons other "
          "than envelope exhaustion the freedom is substantive.\n")

        # ---------------- Test 3
        if p["regime"] == "eaas":
            w("### Test 3 — does the bankability constraint interact with the capacity solution? (limb iii)\n")
            rows = []
            t3s = [("a_central", "(a) central dscr"), ("b_loosest", "(b) loosest dscr (dscr_regression, 1.10)"),
                   ("b2_tighter", "(b2) tighter dscr_high (1.40)"), ("c_bankability_off", "(c) bankability OFF")]
            t3r = {}
            for key, lab in t3s:
                r = get(pr, t3=key)
                t3r[key] = r
                if not ok(r):
                    rows.append([lab, r["status"] if r else "missing"] + [""] * 8)
                    continue
                c, pl = r["constraints"], r["plan"]
                rows.append([lab, r["spec"]["dscr_case"], f"{pl['solar_eaas_mw']:,.1f}", f"{pl['solar_total_mw']:,.1f}",
                             f"{pl['npv_cost_usd']:,.0f}", f"{c['bankability_binding_vintage_years']}/{c['bankability_vintages_with_eaas']}",
                             usd(c["subsidy_total_usd"]), usd(c["subsidy_npv_usd"]),
                             "yes" if c["public_budget_binds"] else ("no" if c["public_budget_built"] else "not built")])
            w(md_table(["configuration", "dscr_case", "EaaS solar (MW)", "total solar (MW)", "npv_cost (USD)",
                        "binding vintage-years / vintages with EaaS", "subsidy drawn (undisc.)", "subsidy (NPV)",
                        "public budget binds"], rows))
            a, cc = t3r.get("a_central"), t3r.get("c_bankability_off")
            if ok(a) and ok(cc):
                dcap = cc["plan"]["solar_total_mw"] - a["plan"]["solar_total_mw"]
                deaas = cc["plan"]["solar_eaas_mw"] - a["plan"]["solar_eaas_mw"]
                dcost = cc["plan"]["npv_cost_usd"] - a["plan"]["npv_cost_usd"]
                w(f"\n**(c) − (a):** Δ EaaS solar = {deaas:,.2f} MW; Δ total solar = {dcap:,.2f} MW; Δ npv_cost = {dcost:,.0f} USD "
                  f"({100 * dcost / a['plan']['npv_cost_usd']:.6f}%).")
                same = abs(dcap) < 1e-3 and abs(deaas) < 1e-3 and abs(dcost) < 1e-3 * max(1.0, abs(a['plan']['npv_cost_usd'])) * 1e-6
                w("Reading: " + ("the capacity solution is IDENTICAL with the constraint removed → on this panel the "
                                 "bankability constraint is inert in the central case (limb iii not shown here)."
                                 if same else
                                 "the capacity solution DIFFERS when the constraint is removed → on this panel the "
                                 "constraint interacts with the capacity solution (limb iii supported here)."))
            w("\n**On the subsidy under (c).** `eaas_subsidy` has NO objective coefficient anywhere in the model (Part A, A1); "
              "it appears only in `eaas_bankability`, `eaas_subsidy_limit` and (if built) the public budget. With bankability "
              "deactivated it has no constraint to relax, so a zero (or any value within `eaas_subsidy_limit`) is expected and "
              "the reported value carries no information — it is not a solver artefact and not a finding. "
              "The same zero-cost property holds under (a): the subsidy is free to the objective, and is limited only by "
              "`gap·add` and, where built, the public budget. **Structural prediction to check against the table:** when the "
              "public budget is not built, a free subsidy up to `gap·add` makes the bankability constraint satisfiable at "
              "equality for any build, so removing it should not change capacity; the constraint can interact with capacity "
              "only through the budget (compare the two eaas panels).\n")

        # ---------------- Test 5 (negative control)
        w("### Test 5 — negative control: zero envelope vs env_dares across the SAME variations\n")
        rows, zs, ds = [], [], []
        for var in VARIATIONS:
            rz = get(pr, env_label="env_none", variation=var[0], t3="central")
            rd = get(pr, env_label="env_dares", variation=var[0], t3="central")
            if not (ok(rz) and ok(rd)):
                rows.append([var[0], "failed/missing", "", "", ""])
                continue
            zs.append(rz["flat"]["port_elig_r_vw"])
            ds.append(rd["flat"]["port_elig_r_vw"])
            rows.append([var[0], pct(rz["flat"]["port_elig_r_vw"], 4), pct(rz["flat"]["port_all4_r_vw"], 4),
                         pct(rd["flat"]["port_elig_r_vw"], 4), pct(rd["flat"]["port_all4_r_vw"], 4)])
        w(md_table(["variation", "ZERO envelope: portfolio r (elig.)", "ZERO: portfolio r (all4)",
                    "env_dares: portfolio r (elig.)", "env_dares: portfolio r (all4)"], rows))
        if zs:
            w(f"\nZero envelope: range across variations = **{bp(1e4 * (max(zs) - min(zs)))} bp** "
              f"(commercial real rate = {pct(first['inputs']['r_comm_real'], 4) if first else 'n/a'}). "
              f"env_dares: range = **{bp(1e4 * (max(ds) - min(ds)))} bp**.")
            w("The zero-envelope column is the treatment of the review's exogenous studies (all capital at a fixed "
              "commercial rate); its invariance is expected by construction and is checked, not assumed.")
        w("")

    # ---- Test 4
    w("---\n\n## Test 4 — risk-to-rate feedback (restated from Part A, A4)\n")
    w("No rate, CRF, tenor or discount window in either builder depends on technology mix, capacity, emissions, reliability or "
      "any other solved quantity (A3, A4; `optimize_model_sliced.py` 1559-1634, `blended_finance.py` 212-244). "
      "The risk-to-rate limb is **absent**. Test 1b is consistent with this: any movement it shows is a change in the "
      "volume weights on two fixed rates, not a change in either rate.\n")

    # ---- Inconclusive / failures
    w("## Runs that did not produce a result\n")
    if bad:
        w(md_table(["run id", "status", "message"], [[r["spec"]["id"], r["status"], str(r.get("message", ""))[:200]] for r in bad]))
    else:
        w("None.")
    w("")
    w(END)
    return "\n".join(L)


def write_outputs(recs, args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # runs.csv -- FLAT fields (CLAUDE.md: CSV runners need flat fields)
    rows = []
    for r in recs:
        s = r["spec"]
        row = {k: s[k] for k in ("id", "panel", "regime", "capital_case", "env_label", "env_arm", "variation",
                                 "group", "demand_case", "gas_case", "cf", "cap_name", "dscr_case",
                                 "bankability_off", "t3")}
        row["env_override"] = s["env_override"]
        row["status"] = r["status"]
        if r["status"] == "optimal":
            row.update(r["flat"])
            row.update({f"c_{k}": v for k, v in r["constraints"].items()})
            row.update({f"p_{k}": v for k, v in r["plan"].items()})
            row["charge_check_rel_err"] = r["charge_check_rel_err"]
            row["alloc"] = r["alloc"]
            row.update({f"in_{k}": v for k, v in r["inputs"].items()})
        else:
            row["message"] = r.get("message")
        rows.append(row)
    cols = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    with open(OUT_DIR / "runs.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    with open(OUT_DIR / "cells.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["run_id", "channel", "vintage_year", "conc_usd", "comm_usd", "conc_share", "crf_realised",
                     "r_vw", "r_impl_Lcomm", "r_impl_Lconc"])
        for r in recs:
            if r["status"] == "optimal":
                for c in r["cells"]:
                    wr.writerow([r["spec"]["id"], c["channel"], c["vintage"], c["conc"], c["comm"], c["share"],
                                 c["crf_real"], c["r_vw"], c["r_impl_Lcomm"], c["r_impl_Lconc"]])
    text = render(recs, args)
    (OUT_DIR / "report_partB.md").write_text(text)
    doc = DOC_PATH.read_text()
    if BEGIN in doc and END in doc:
        pre, rest = doc.split(BEGIN, 1)
        post = rest.split(END, 1)[1]
        doc = pre.rstrip("\n") + "\n\n" + text + post
    else:
        doc = doc.rstrip("\n") + "\n\n" + text + "\n"
    DOC_PATH.write_text(doc)
    print(f"Wrote {OUT_DIR}/runs.csv, cells.csv, report_partB.md and the Part B section of {DOC_PATH}")


# =====================================================================
# Main
# =====================================================================

def print_manifest(runs):
    print("=" * 100)
    print("RUN MANIFEST (arms / overrides / regimes) -- central arms:", json.dumps(CENTRAL))
    print("=" * 100)
    for s in runs:
        print(f"{s['id']:<70} regime={s['regime']:<11} capital={s['capital_case']:<13} "
              f"env={s['env_label']}({s['env_arm']}{'*0.5' if s['env_override'] == 'half' else ''}) "
              f"demand={s['demand_case']} gas={s['gas_case']} cf={s['cf'] or 'central'} "
              f"cap={s['cap_name'] or '-'} dscr={s['dscr_case']} bankability={'OFF' if s['bankability_off'] else 'on'}")
    print(f"{len(runs)} runs")
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser(description="Financing-endogeneity Part B diagnostics")
    ap.add_argument("--panels", nargs="+", default=list(PANELS), choices=list(PANELS))
    ap.add_argument("--solver", default="cbc")
    ap.add_argument("--smoke", action="store_true", help="solve ONE run and print its metrics; write no report")
    ap.add_argument("--dry-run", action="store_true", help="build every model, solve nothing")
    ap.add_argument("--report-only", action="store_true", help="regenerate outputs from cached runs")
    args = ap.parse_args()

    specs = build_run_list(args.panels)
    print_manifest(specs)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    econ = load_econ("voll_mid", genset_case="genset_central")
    capex_tv = load_solar_capex_by_year(scenario_name="solar_low", start_year=2025, end_year=MODEL_END_YEAR)

    if args.dry_run:
        for s in specs:
            t0 = time.time()
            m, sc = build(s, econ, capex_tv)
            print(f"  built {s['id']}  envelope={sc['concessional_envelope_usd']:,.0f}  regime={sc['financing_regime']}  "
                  f"budget={sc.get('public_solar_budget_npv')}  ({time.time() - t0:.1f}s)")
        return

    if args.smoke:
        s = next(x for x in specs if x["env_label"] == "env_dares" and x["variation"] == "base")
        rec = run_one(s, econ, capex_tv, args.solver)
        print(json.dumps(json_safe({k: v for k, v in rec.items() if k != "cells"}), indent=1)[:6000])
        return

    recs = []
    for i, s in enumerate(specs, 1):
        f = RUN_DIR / (hashlib.md5(s["id"].encode()).hexdigest()[:10] + "__" + s["id"][:80].replace("|", "_") + ".json")
        if f.exists():
            rec = json.loads(f.read_text())
            print(f"[{i}/{len(specs)}] cached  {s['id']}  ({rec['status']})")
        elif args.report_only:
            continue
        else:
            print(f"[{i}/{len(specs)}] solving {s['id']} ...", flush=True)
            rec = run_one(s, econ, capex_tv, args.solver)
            f.write_text(json.dumps(json_safe(rec)))
            print(f"    -> {rec['status']} in {rec.get('seconds', 0):.0f}s", flush=True)
        recs.append(rec)
    write_outputs(recs, args)


if __name__ == "__main__":
    main()
