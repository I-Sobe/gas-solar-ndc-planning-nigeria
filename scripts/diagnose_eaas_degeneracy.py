"""
diagnose_eaas_degeneracy.py -- is the public/EaaS solar split determinate?

Context: docs/financing_endogeneity.md Part B Test 3 gave a public/EaaS split that
moves non-monotonically with the coverage margin while total solar and npv_cost
are invariant. Part C reads that as alternative optima. This script tests it.
F-12 (docs/findings_frozen.md) already established non-identification at
ndc3_conditional by lexicographic min/max; this script measures it at the
configurations the task names. Report: docs/eaas_degeneracy.md.

PER CONFIGURATION (all with financing_regime="eaas", env_dares, dscr_central,
Phase B central arms from scripts/diagnose_financing_endogeneity.py:CENTRAL)
 A0  CBC solve (tolerances as solve_model()), duals + reduced costs.
 A1  Reduced costs of solar_public_add[t], solar_eaas_add[t]; duals on
     eaas_bankability[t], eaas_subsidy_limit[t], the concessional envelope,
     the public budget (where built).
 A2  Same model, second solver: HiGHS 1.15 via its CLI (the Pyomo 'highs'
     interface needs highspy, which is not installed). LP written with
     symbolic labels, solution read back through the symbol map.
 A3  Perturbation ladder: objective + delta * cost_eaas_solar_capital_charge_npv
     for delta in +/-{1e-6,1e-5,1e-4,1e-3,1e-2}, applied to the BUILT model (no
     src/ edit). This scales the EaaS-solar initial-build tranche charge by 1+delta;
     the rebuild charge (which pools public and EaaS adds in one term) and the
     split-identity/bankability coefficients are NOT perturbed. Reported npv_cost
     is always the UNPERTURBED system_cost_npv.
 A4  Lexicographic min / max of total EaaS MW subject to
     system_cost_npv <= Z*(1+1e-6)  (F-12's method and tolerance).
Configurations: see CONFIGS (central on both Part B EaaS panels; an unconditional
cap; a tight public budget; a cap + tight budget; F-34's gas regime in both
solar-build designs).

PART C: slack-envelope case (45 bn, 100 bn) on traditional and eaas, unconstrained.

STABILITY VERDICT (rule fixed in advance; inputs are printed beside it):
  lexi_w   = (max EaaS - min EaaS) / total solar          [A4]
  solv_d   = |EaaS_CBC - EaaS_HiGHS| / total solar        [A2]
  ladder   = knife-edge | smooth | insensitive | mixed     [A3, see classify_ladder]
  tied         : lexi_w >= 1%  and (solv_d >= 1% or ladder == knife-edge)
  determinate  : lexi_w <= 0.1% and solv_d <= 0.1% and ladder in {smooth, insensitive}
  inconclusive : anything else (including any failed solve)

USAGE
  python scripts/diagnose_eaas_degeneracy.py --smoke      # 1 config, CBC solve only
  python scripts/diagnose_eaas_degeneracy.py --dry-run    # build every model, no solve
  python scripts/diagnose_eaas_degeneracy.py              # everything (~100 solves)
  python scripts/diagnose_eaas_degeneracy.py --report-only
Finished configurations are cached in results/eaas_degeneracy/runs/.
Writes only results/eaas_degeneracy/ and the AUTOGEN block of docs/eaas_degeneracy.md.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts"))

from src.scenarios import load_scenario, MODEL_END_YEAR, concessional_envelope_scenarios  # noqa: E402
from src.optimize_model_sliced import build_model_sliced                                # noqa: E402
from src.io import load_econ, load_solar_capex_by_year                                 # noqa: E402
from src.utils import json_safe                                                         # noqa: E402
import diagnose_financing_endogeneity as pb                                             # noqa: E402

OUT_DIR = ROOT / "results" / "eaas_degeneracy"
RUN_DIR = OUT_DIR / "runs"
DOC_PATH = ROOT / "docs" / "eaas_degeneracy.md"
BEGIN = "<!-- BEGIN AUTOGEN (scripts/diagnose_eaas_degeneracy.py) -->"
END = "<!-- END AUTOGEN -->"

DELTAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
LEXI_TOL = 1e-6
RC_TOL = 1.0         # USD per MW (NPV): |rc| below this counts as zero (coefficients are O(1e5-1e6))
MW_EPS = 1e-6

# id, description, overrides
CONFIGS = [
    dict(id="central_P2", desc="central, eaas, capital_case=moderate (Part B P2)",
         capital_case="moderate", cap_name=None, gas_case="baseline", solar_build="deployment_unconstrained"),
    dict(id="central_P3", desc="central, eaas, capital_case=unconstrained (Part B P3)",
         capital_case="unconstrained", cap_name=None, gas_case="baseline", solar_build="deployment_unconstrained"),
    dict(id="cap_ndc3unc_unc", desc="ndc3_unconditional cap, capital_case=unconstrained",
         capital_case="unconstrained", cap_name="ndc3_unconditional", gas_case="baseline",
         solar_build="deployment_unconstrained"),
    dict(id="cap_ndc3unc_tight", desc="ndc3_unconditional cap, capital_case=tight (03_run_ndc_eaas / F-35 setting)",
         capital_case="tight", cap_name="ndc3_unconditional", gas_case="baseline",
         solar_build="deployment_unconstrained"),
    dict(id="budget_tight_nopolicy", desc="no policy, capital_case=tight (tightest public budget, 5.137 bn)",
         capital_case="tight", cap_name=None, gas_case="baseline", solar_build="deployment_unconstrained"),
    dict(id="gasF34_capped", desc="F-34 regime: gas shock_recovery, no policy, moderate, solar_build=nirp_trajectory "
                                  "(GAS-2's stored design, where EaaS worsens gas scarcity by 10.56%)",
         capital_case="moderate", cap_name=None, gas_case="shock_recovery", solar_build="nirp_trajectory"),
    dict(id="gasF34_uncapped", desc="F-34 regime: gas shock_recovery, no policy, moderate, solar_build=deployment_unconstrained "
                                    "(L81: relief +35.59% in this design)",
         capital_case="moderate", cap_name=None, gas_case="shock_recovery", solar_build="deployment_unconstrained"),
]
SLACK_CONFIGS = [
    dict(id="slack_trad_45bn", regime="traditional", env_usd=45e9),
    dict(id="slack_trad_100bn", regime="traditional", env_usd=100e9),
    dict(id="slack_eaas_45bn", regime="eaas", env_usd=45e9),
    dict(id="slack_eaas_100bn", regime="eaas", env_usd=100e9),
]


# =====================================================================
# Build / solve helpers
# =====================================================================

def make_scenario(regime, capital_case, cap_name, gas_case, solar_build, env_arm="env_dares", env_usd=None):
    kw = dict(pb.CENTRAL)
    sc = load_scenario(
        demand_level_case="served", demand_case=pb.CENTRAL_DEMAND, capital_case=capital_case,
        gas_deliverability_case=gas_case, solar_build_case=solar_build, land_case="loose",
        carbon_case="no_policy", start_year=2025, end_year=MODEL_END_YEAR,
        concessional_envelope_case=env_arm, **kw)
    sc["financing_regime"] = regime
    if env_usd is not None:
        sc["concessional_envelope_usd"] = float(env_usd)
    return sc


def build(cfg, econ, capex_tv, regime="eaas", env_arm="env_dares", env_usd=None):
    sc = make_scenario(regime, cfg["capital_case"], cfg["cap_name"], cfg["gas_case"], cfg["solar_build"],
                       env_arm, env_usd)
    caps = pb.load_caps(cfg["cap_name"], sc["years"]) if cfg["cap_name"] else None
    m = build_model_sliced(scenario=sc, econ=econ, emissions_cap=None, emissions_cap_by_year=caps,
                           solar_capex_by_year=capex_tv)
    if not hasattr(m, "rc"):
        m.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m, sc


def cbc_solve(m):
    """CBC with the same recorded tolerances as src solve_model(); raises on non-optimal."""
    s = pyo.SolverFactory("cbc")
    if not s.available():
        raise RuntimeError("cbc not available")
    s.options["primalTolerance"] = 1e-7
    s.options["dualTolerance"] = 1e-7
    s.options["integerTolerance"] = 1e-7
    res = s.solve(m, tee=False)
    tc = res.solver.termination_condition
    if tc != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"CBC termination: {tc}")


def highs_solve(m, tag):
    """HiGHS via CLI. Loads primal values back into m. Returns status string."""
    work = OUT_DIR / "tmp"
    work.mkdir(parents=True, exist_ok=True)
    lp, sol, opt = work / f"{tag}.lp", work / f"{tag}.sol", work / f"{tag}.opt"
    _, sid = m.write(str(lp), io_options={"symbolic_solver_labels": True})
    smap = m.solutions.symbol_map[sid]
    opt.write_text(f"write_solution_style = 0\nsolution_file = {sol}\n")
    if sol.exists():
        sol.unlink()
    subprocess.run(["highs", str(lp), "--options_file", str(opt)], capture_output=True, text=True,
                   stdin=subprocess.DEVNULL, timeout=3600)
    if not sol.exists():
        raise RuntimeError("HiGHS wrote no solution file")
    lines = sol.read_text().splitlines()
    if len(lines) < 2 or lines[1].strip() != "Optimal":
        raise RuntimeError(f"HiGHS status: {lines[1] if len(lines) > 1 else '?'}")
    i = lines.index("# Primal solution values")
    j = i + 1
    while not lines[j].startswith("# Columns"):
        j += 1
    n = int(lines[j].split()[2])
    for ln in lines[j + 1: j + 1 + n]:
        name, val = ln.rsplit(" ", 1)
        ref = smap.bySymbol.get(name)
        if ref is None:
            continue
        v = ref()
        if hasattr(v, "fixed") and not v.fixed:
            v.set_value(float(val), skip_validation=True)
    lp.unlink(missing_ok=True)
    sol.unlink(missing_ok=True)
    opt.unlink(missing_ok=True)
    return "Optimal"


def set_objective(m, expr, sense=pyo.minimize):
    for name in ("pert_obj",):
        if hasattr(m, name):
            m.del_component(name)
    m.objective.deactivate()
    m.pert_obj = pyo.Objective(expr=expr, sense=sense)


def restore_objective(m):
    if hasattr(m, "pert_obj"):
        m.del_component("pert_obj")
    m.objective.activate()


def snapshot(m, T):
    pub = [pb._v(m.solar_public_add[t]) for t in T]
    eaas = [pb._v(m.solar_eaas_add[t]) for t in T]
    return dict(public_mw=sum(pub), eaas_mw=sum(eaas), total_mw=sum(pub) + sum(eaas),
                npv_cost_usd=pb._v(m.system_cost_npv), public_by_vintage=pub, eaas_by_vintage=eaas)


# =====================================================================
# Analysis
# =====================================================================

def classify_ladder(base_eaas, total, lad):
    """lad: list of dicts(delta, sign, eaas_mw, total_mw). Returns label + inputs."""
    if not lad:
        return "inconclusive", {}
    dev = {(r["sign"], r["delta"]): abs(r["eaas_mw"] - base_eaas) for r in lad}
    full = max(dev.values())
    step1 = max(dev.get((s, DELTAS[0]), 0.0) for s in (+1, -1))
    tot_span = max(r["total_mw"] for r in lad) - min(r["total_mw"] for r in lad)
    info = dict(full_dev_mw=full, step1_dev_mw=step1, total_span_mw=tot_span)
    if full < 1e-3 * total:
        return "insensitive", info
    if full >= 1e-2 * total and step1 >= 0.5 * full:
        return "knife-edge", info
    # monotone non-increasing in delta on the + side (more expensive EaaS -> less EaaS)
    plus = [r["eaas_mw"] for r in sorted((r for r in lad if r["sign"] == +1), key=lambda r: r["delta"])]
    minus = [r["eaas_mw"] for r in sorted((r for r in lad if r["sign"] == -1), key=lambda r: r["delta"])]
    mono = all(a >= b - 1e-6 for a, b in zip(plus, plus[1:])) and all(a <= b + 1e-6 for a, b in zip(minus, minus[1:]))
    if mono and step1 < 0.1 * full:
        return "smooth", info
    return "mixed", info


def verdict(rec):
    try:
        base, hi, lx = rec["cbc"], rec["highs"], rec["lexi"]
        total = base["total_mw"]
        lexi_w = (lx["max_eaas_mw"] - lx["min_eaas_mw"]) / total
        solv_d = abs(base["eaas_mw"] - hi["eaas_mw"]) / total
        lab, _ = classify_ladder(base["eaas_mw"], total, rec["ladder"])
    except Exception:
        return "inconclusive", {}
    inputs = dict(lexi_width_frac=lexi_w, solver_diff_frac=solv_d, ladder=lab)
    if lexi_w >= 0.01 and (solv_d >= 0.01 or lab == "knife-edge"):
        return "tied", inputs
    if lexi_w <= 0.001 and solv_d <= 0.001 and lab in ("smooth", "insensitive"):
        return "determinate", inputs
    return "inconclusive", inputs


# =====================================================================
# Per-configuration run
# =====================================================================

def run_config(cfg, econ, capex_tv, smoke=False):
    t0 = time.time()
    rec = dict(cfg=cfg, status="error")
    try:
        m, sc = build(cfg, econ, capex_tv)
        T = range(len(sc["years"]))
        terms = pb.rate_terms(sc)
        rec["inputs"] = dict(envelope_usd=float(sc["concessional_envelope_usd"]), required_margin=float(sc["required_margin"]),
                             tariff=float(sc["solar_service_tariff_usd_per_twh"]), financing_regime=sc["financing_regime"],
                             public_budget=sc.get("public_solar_budget_npv"))
        cbc_solve(m)
        base = snapshot(m, T)
        raw = pb.extract(m, sc, econ)
        cs = pb.constraint_state(m, sc, raw, terms)
        rc_pub = [m.rc.get(m.solar_public_add[t], None) for t in T]
        rc_eaas = [m.rc.get(m.solar_eaas_add[t], None) for t in T]
        bank = [m.dual.get(m.eaas_bankability[t], None) for t in T]
        sub = [m.dual.get(m.eaas_subsidy_limit[t], None) for t in T]
        years = [int(y) for y in sc["years"]]
        vint = []
        for t in T:
            vint.append(dict(year=years[t], public_mw=base["public_by_vintage"][t], eaas_mw=base["eaas_by_vintage"][t],
                             rc_public=rc_pub[t], rc_eaas=rc_eaas[t], dual_bankability=bank[t], dual_subsidy_limit=sub[t]))
        def zero_rc(v, rc):
            return rc is not None and abs(rc) <= RC_TOL
        n_pub0 = sum(1 for r in vint if r["public_mw"] <= MW_EPS and zero_rc(0, r["rc_public"]))
        n_eaas0 = sum(1 for r in vint if r["eaas_mw"] <= MW_EPS and zero_rc(0, r["rc_eaas"]))
        n_both = sum(1 for r in vint if r["public_mw"] > MW_EPS and r["eaas_mw"] > MW_EPS)
        nz = [abs(x) for x in rc_pub + rc_eaas if x is not None and abs(x) > RC_TOL]
        rec.update(cbc=base, constraints=cs, vintages=vint,
                   rc_summary=dict(n_vint_public_zero_with_rc0=n_pub0, n_vint_eaas_zero_with_rc0=n_eaas0,
                                   n_vint_both_positive=n_both, min_nonzero_abs_rc=min(nz) if nz else None),
                   emissions_cap_binding_years=cs.get("emissions_cap_binding_years"))
        rec["cbc"].pop("public_by_vintage"); rec["cbc"].pop("eaas_by_vintage")
        if smoke:
            rec["status"] = "smoke"
            return rec
        # A2 HiGHS on a fresh copy
        try:
            m2, _ = build(cfg, econ, capex_tv)
            highs_solve(m2, cfg["id"])
            h = snapshot(m2, T); h.pop("public_by_vintage"); h.pop("eaas_by_vintage")
            rec["highs"] = h
        except Exception as e:
            rec["highs_error"] = f"{type(e).__name__}: {e}"
        # A3 ladder (re-uses m)
        lad = []
        for d in DELTAS:
            for sign in (+1, -1):
                try:
                    set_objective(m, m.system_cost_npv + sign * d * m.cost_eaas_solar_capital_charge_npv)
                    cbc_solve(m)
                    s = snapshot(m, T)
                    lad.append(dict(delta=d, sign=sign, eaas_mw=s["eaas_mw"], public_mw=s["public_mw"],
                                    total_mw=s["total_mw"], npv_cost_usd=s["npv_cost_usd"]))
                except Exception as e:
                    lad.append(dict(delta=d, sign=sign, error=f"{type(e).__name__}: {e}"))
        restore_objective(m)
        rec["ladder"] = [r for r in lad if "error" not in r]
        rec["ladder_errors"] = [r for r in lad if "error" in r]
        # A4 lexicographic bounds at Z*(1+tol)
        Z = base["npv_cost_usd"]
        m.lexi_cost = pyo.Constraint(expr=m.system_cost_npv <= Z * (1.0 + LEXI_TOL))
        tot_eaas = pyo.quicksum(m.solar_eaas_add[t] for t in T)
        lx = {}
        for nm, sense in (("min_eaas_mw", pyo.minimize), ("max_eaas_mw", pyo.maximize)):
            set_objective(m, tot_eaas, sense)
            cbc_solve(m)
            s = snapshot(m, T)
            lx[nm] = s["eaas_mw"]
            lx[nm.replace("eaas", "total")] = s["total_mw"]
        rec["lexi"] = lx
        rec["verdict"], rec["verdict_inputs"] = verdict(rec)
        rec["status"] = "ok"
    except Exception as e:
        rec.update(status="error", message=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
    rec["seconds"] = time.time() - t0
    return rec


def run_slack(sc_cfg, econ, capex_tv):
    t0 = time.time()
    rec = dict(cfg=sc_cfg, status="error")
    try:
        cfg = dict(capital_case="unconstrained", cap_name=None, gas_case="baseline", solar_build="deployment_unconstrained")
        m, sc = build(cfg, econ, capex_tv, regime=sc_cfg["regime"], env_arm="env_10x", env_usd=sc_cfg["env_usd"])
        terms = pb.rate_terms(sc)
        cbc_solve(m)
        raw = pb.extract(m, sc, econ)
        flat, cells = pb.analyse(raw, terms)
        cs = pb.constraint_state(m, sc, raw, terms)
        elig = [c for c in cells if c["channel"] in pb.ELIGIBLE]
        rec.update(status="ok", flat=flat, constraints=cs, alloc=pb.alloc_summary(cells),
                   n_interior=sum(1 for c in elig if pb.SHARE_EPS < c["share"] < 1 - pb.SHARE_EPS),
                   eligible_comm_usd=sum(c["comm"] for c in elig),
                   max_eligible_comm_cell_usd=max([c["comm"] for c in elig] or [0.0]),
                   charge_check_rel_err=pb.charge_check(m, raw, terms),
                   npv_cost_usd=pb._v(m.system_cost_npv))
    except Exception as e:
        rec.update(status="error", message=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
    rec["seconds"] = time.time() - t0
    return rec


# =====================================================================
# Report
# =====================================================================

def f1(x, nd=1):
    return "n/a" if x is None else f"{x:,.{nd}f}"


def render(recs, slack):
    L = []
    w = L.append
    w(BEGIN)
    w("")
    w(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `scripts/diagnose_eaas_degeneracy.py`. "
      "Verification artefacts, not thesis findings.*")
    w("")
    w("## Parts A and B — per-configuration results")
    w("")
    w("Rule for the stability verdict (fixed in advance): **tied** if the lexicographic EaaS interval is ≥ 1% of total solar "
      "and (CBC/HiGHS differ by ≥ 1% of total solar or the ladder is knife-edge); **determinate** if interval ≤ 0.1%, solvers "
      "agree within 0.1%, and ladder is smooth/insensitive; otherwise **inconclusive**. Ladder labels: *insensitive* (max "
      "deviation < 0.1% of total), *knife-edge* (≥ 1% and ≥ half of it already at δ=1e-6), *smooth* (monotone and first "
      "step < 10% of full), else *mixed*.")
    w("")
    rows = []
    for r in recs:
        c = r["cfg"]
        if r["status"] != "ok":
            rows.append([c["id"], r["status"], "", "", "", "", "", "", r.get("message", "")[:80]])
            continue
        vi = r["verdict_inputs"]
        rows.append([c["id"], f1(r["cbc"]["total_mw"]), f1(r["cbc"]["eaas_mw"]),
                     f1(r["highs"]["eaas_mw"]) if "highs" in r else "FAILED",
                     f"{f1(r['lexi']['min_eaas_mw'])} – {f1(r['lexi']['max_eaas_mw'])}",
                     f"{100 * vi['lexi_width_frac']:.2f}%" if vi else "n/a",
                     f"{100 * vi['solver_diff_frac']:.2f}%" if vi else "n/a",
                     vi.get("ladder", "n/a") if vi else "n/a", f"**{r['verdict']}**"])
    w(pb.md_table(["configuration", "total solar (MW)", "EaaS MW (CBC)", "EaaS MW (HiGHS)",
                   "A4 EaaS interval at Z*(1+1e-6) (MW)", "interval / total", "abs(CBC−HiGHS) / total",
                   "A3 ladder", "verdict"], rows))
    w("")
    for r in recs:
        c = r["cfg"]
        w(f"### `{c['id']}` — {c['desc']}")
        w("")
        if r["status"] != "ok":
            w(f"**Run status: {r['status']}** — {r.get('message', '')}")
            w("")
            continue
        i, cs = r["inputs"], r["constraints"]
        w(f"Inputs: envelope {i['envelope_usd']:,.0f} USD, required_margin {i['required_margin']}, tariff {i['tariff']:,.0f}, "
          f"public budget {i['public_budget'] if i['public_budget'] is not None else 'not built'}. "
          f"Emissions-cap binding years: {cs.get('emissions_cap_binding_years')}. "
          f"Concessional envelope: binds={cs['envelope_binds']}, dual={cs['envelope_dual_pv_per_undiscounted_usd']}. "
          f"Public budget: built={cs['public_budget_built']}, binds={cs['public_budget_binds']}, dual={cs['public_budget_dual']}.")
        w("")
        w(f"**A0/A2** CBC: EaaS {f1(r['cbc']['eaas_mw'])} MW, public {f1(r['cbc']['public_mw'])} MW, total {f1(r['cbc']['total_mw'])} MW, "
          f"npv_cost {r['cbc']['npv_cost_usd']:,.2f} USD. "
          + (f"HiGHS: EaaS {f1(r['highs']['eaas_mw'])} MW, public {f1(r['highs']['public_mw'])} MW, total {f1(r['highs']['total_mw'])} MW, "
             f"npv_cost {r['highs']['npv_cost_usd']:,.2f} USD (Δcost vs CBC {r['highs']['npv_cost_usd'] - r['cbc']['npv_cost_usd']:+,.2f} USD)."
             if "highs" in r else f"HiGHS FAILED: {r.get('highs_error')}"))
        w("")
        rs = r["rc_summary"]
        w(f"**A1** vintages (of 30): solar_public_add = 0 with reduced cost ≈ 0 (|rc| ≤ {RC_TOL} USD/MW): {rs['n_vint_public_zero_with_rc0']}; "
          f"solar_eaas_add = 0 with rc ≈ 0: {rs['n_vint_eaas_zero_with_rc0']}; both positive: {rs['n_vint_both_positive']}; "
          f"smallest non-zero |rc| anywhere: {rs['min_nonzero_abs_rc']}. A variable at 0 with rc ≈ 0 could enter at no cost; "
          "full per-vintage table in `results/eaas_degeneracy/reduced_costs.csv`. This is evidence of indifference, not proof — "
          "A4 is the direct test.")
        w("")
        if c["id"] in ("central_P2", "central_P3"):
            w(pb.md_table(["year", "public MW", "EaaS MW", "rc public", "rc EaaS", "dual bankability", "dual subsidy limit"],
                          [[v["year"], f1(v["public_mw"]), f1(v["eaas_mw"]), v["rc_public"], v["rc_eaas"],
                            v["dual_bankability"], v["dual_subsidy_limit"]] for v in r["vintages"]]))
            w("")
        w("**A3 ladder** (EaaS capital charge × (1+sign·δ); npv_cost is the unperturbed objective):")
        w("")
        w(pb.md_table(["δ", "sign", "EaaS MW", "public MW", "total MW", "npv_cost (USD)"],
                      [[f"{x['delta']:.0e}", "+" if x["sign"] > 0 else "−", f1(x["eaas_mw"]), f1(x["public_mw"]),
                        f1(x["total_mw"], 3), f"{x['npv_cost_usd']:,.2f}"]
                       for x in sorted(r["ladder"], key=lambda x: (x["delta"], -x["sign"]))]))
        lab, info = classify_ladder(r["cbc"]["eaas_mw"], r["cbc"]["total_mw"], r["ladder"])
        w(f"\nLadder reads **{lab}** ({json.dumps(json_safe(info))}). Failed ladder solves: {len(r['ladder_errors'])}.")
        w("")
        lx = r["lexi"]
        w(f"**A4** EaaS MW range at cost ≤ Z*(1+{LEXI_TOL:g}): {f1(lx['min_eaas_mw'])} – {f1(lx['max_eaas_mw'])} "
          f"(total solar at the two ends: {f1(lx['min_total_mw'])}, {f1(lx['max_total_mw'])}).")
        w(f"\n**Verdict: {r['verdict']}** — inputs {json.dumps(json_safe(r['verdict_inputs']))}.")
        w("")

    w("## Part C — slack-envelope case")
    w("")
    w("Prediction under test (Part A of the endogeneity task): with a slack envelope no interior share arises and all "
      "eligible capital is concessional.")
    w("")
    rows = []
    for s in slack:
        c = s["cfg"]
        if s["status"] != "ok":
            rows.append([c["id"], s["status"], s.get("message", "")[:100]] + [""] * 8)
            continue
        cs, f = s["constraints"], s["flat"]
        rows.append([c["id"], c["regime"], usd_b(c["env_usd"]), f"{usd_b(cs['envelope_drawn_usd'])}",
                     "yes" if cs["envelope_slack_usd"] > pb.TOL_USD else "no",
                     "n/a" if cs["envelope_dual_pv_per_undiscounted_usd"] is None else f"{cs['envelope_dual_pv_per_undiscounted_usd']:.3g}",
                     " / ".join(pb.pct(f[f"{ch}_conc_share"], 4) if f[f"{ch}_conc_share"] is not None else "–" for ch in pb.CHANNELS),
                     s["n_interior"], f"{s['eligible_comm_usd']:,.0f}",
                     pb.pct(f["port_elig_r_vw"], 4), pb.pct(f["port_all4_r_vw"], 4)])
    w(pb.md_table(["run", "regime", "B", "eligible drawn", "slack?", "envelope dual",
                   "conc. share: solar_pub / solar_eaas / storage / gas", "# interior cells",
                   "eligible commercial USD", "portfolio r_vw (elig.)", "portfolio r_vw (all4)"], rows))
    w("")
    for s in slack:
        if s["status"] == "ok":
            w(f"- `{s['cfg']['id']}`: eligible capex {s['flat']['port_elig_vol_usd'] / 1e9:.3f} bn; allocation: {s['alloc']}; "
              f"charge-check rel. err {s['charge_check_rel_err']:.1e}.")
    w("")
    w(END)
    return "\n".join(L)


def usd_b(x):
    return f"{x / 1e9:,.1f} bn"


def write_outputs(recs, slack):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "reduced_costs.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["config", "year", "public_mw", "eaas_mw", "rc_public", "rc_eaas", "dual_bankability", "dual_subsidy_limit"])
        for r in recs:
            if "vintages" in r:
                for v in r["vintages"]:
                    wr.writerow([r["cfg"]["id"], v["year"], v["public_mw"], v["eaas_mw"], v["rc_public"], v["rc_eaas"],
                                 v["dual_bankability"], v["dual_subsidy_limit"]])
    with open(OUT_DIR / "ladder.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["config", "delta", "sign", "eaas_mw", "public_mw", "total_mw", "npv_cost_usd"])
        for r in recs:
            for x in r.get("ladder", []):
                wr.writerow([r["cfg"]["id"], x["delta"], x["sign"], x["eaas_mw"], x["public_mw"], x["total_mw"], x["npv_cost_usd"]])
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["config", "status", "total_mw", "eaas_cbc", "eaas_highs", "public_cbc", "npv_cbc", "npv_highs",
                     "lexi_min_eaas", "lexi_max_eaas", "lexi_width_frac", "solver_diff_frac", "ladder", "verdict"])
        for r in recs:
            if r["status"] == "ok":
                vi = r["verdict_inputs"]
                h = r.get("highs", {})
                wr.writerow([r["cfg"]["id"], "ok", r["cbc"]["total_mw"], r["cbc"]["eaas_mw"], h.get("eaas_mw"),
                             r["cbc"]["public_mw"], r["cbc"]["npv_cost_usd"], h.get("npv_cost_usd"),
                             r["lexi"]["min_eaas_mw"], r["lexi"]["max_eaas_mw"], vi.get("lexi_width_frac"),
                             vi.get("solver_diff_frac"), vi.get("ladder"), r["verdict"]])
            else:
                wr.writerow([r["cfg"]["id"], r["status"]])
    with open(OUT_DIR / "slack_envelope.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["run", "regime", "envelope_usd", "status", "drawn_usd", "slack_usd", "dual", "n_interior",
                     "eligible_comm_usd", "port_elig_r_vw", "port_all4_r_vw", "elig_conc_share"])
        for s in slack:
            if s["status"] == "ok":
                cs, f_ = s["constraints"], s["flat"]
                wr.writerow([s["cfg"]["id"], s["cfg"]["regime"], s["cfg"]["env_usd"], "ok", cs["envelope_drawn_usd"],
                             cs["envelope_slack_usd"], cs["envelope_dual_pv_per_undiscounted_usd"], s["n_interior"], s["eligible_comm_usd"],
                             f_["port_elig_r_vw"], f_["port_all4_r_vw"], f_["port_elig_conc_share"]])
            else:
                wr.writerow([s["cfg"]["id"], s["cfg"]["regime"], s["cfg"]["env_usd"], s["status"]])
    text = render(recs, slack)
    (OUT_DIR / "report_autogen.md").write_text(text)
    doc = DOC_PATH.read_text() if DOC_PATH.exists() else ""
    if BEGIN in doc and END in doc:
        pre, rest = doc.split(BEGIN, 1)
        post = rest.split(END, 1)[1]
        doc = pre.rstrip("\n") + "\n\n" + text + post
    else:
        doc = doc.rstrip("\n") + "\n\n" + text + "\n"
    DOC_PATH.write_text(doc)
    print(f"Wrote {OUT_DIR}/{{summary,ladder,reduced_costs,slack_envelope}}.csv, report_autogen.md and the AUTOGEN block of {DOC_PATH}")


# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="EaaS/public solar split determinacy diagnostics")
    ap.add_argument("--smoke", action="store_true", help="central_P3, CBC solve only (A0/A1); prints and exits")
    ap.add_argument("--dry-run", action="store_true", help="build every model, solve nothing")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--configs", nargs="+", default=None, help="subset of config ids")
    ap.add_argument("--skip-slack", action="store_true")
    args = ap.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cfgs = [c for c in CONFIGS if args.configs is None or c["id"] in args.configs]
    print("=" * 100)
    print("RUN MANIFEST. Central arms:", json.dumps(pb.CENTRAL))
    for c in cfgs:
        print(f"  {c['id']:<24} regime=eaas capital={c['capital_case']:<13} cap={c['cap_name'] or '-':<20} gas={c['gas_case']:<15} "
              f"solar_build={c['solar_build']}  :: {c['desc']}")
    for s in SLACK_CONFIGS:
        print(f"  {s['id']:<24} regime={s['regime']:<11} capital=unconstrained envelope override={s['env_usd']:,.0f} (arm env_10x label)")
    print(f"  per config: CBC + HiGHS + ladder {len(DELTAS) * 2} + lexi 2; deltas={DELTAS}; lexi tol={LEXI_TOL}")
    print("=" * 100)

    econ = load_econ("voll_mid", genset_case="genset_central")
    capex_tv = load_solar_capex_by_year(scenario_name="solar_low", start_year=2025, end_year=MODEL_END_YEAR)

    if args.dry_run:
        for c in cfgs:
            m, sc = build(c, econ, capex_tv)
            print(f"  built {c['id']}: budget={sc.get('public_solar_budget_npv')} envelope={sc['concessional_envelope_usd']:,.0f} "
                  f"solar_cumulative_cap={'yes' if sc.get('solar_cumulative_cap_mw') is not None else 'no'}")
        for s in SLACK_CONFIGS:
            m, sc = build(dict(capital_case="unconstrained", cap_name=None, gas_case="baseline",
                               solar_build="deployment_unconstrained"), econ, capex_tv,
                          regime=s["regime"], env_arm="env_10x", env_usd=s["env_usd"])
            print(f"  built {s['id']}: envelope={sc['concessional_envelope_usd']:,.0f} regime={sc['financing_regime']}")
        return

    if args.smoke:
        rec = run_config(CONFIGS[1], econ, capex_tv, smoke=True)
        rec.pop("vintages", None) if rec["status"] != "smoke" else None
        print(json.dumps(json_safe({k: v for k, v in rec.items() if k != "vintages"}), indent=1)[:5000])
        if "vintages" in rec:
            print("first vintages:", json.dumps(json_safe(rec["vintages"][:6]), indent=1)[:2000])
        return

    recs, slack = [], []
    for c in cfgs:
        f = RUN_DIR / f"{c['id']}.json"
        if f.exists():
            rec = json.loads(f.read_text())
            print(f"cached  {c['id']} ({rec['status']})")
        elif args.report_only:
            continue
        else:
            print(f"running {c['id']} ...", flush=True)
            rec = run_config(c, econ, capex_tv)
            f.write_text(json.dumps(json_safe(rec)))
            print(f"   -> {rec['status']} {rec.get('verdict', '')} in {rec['seconds']:.0f}s", flush=True)
        recs.append(rec)
    if not args.skip_slack:
        for s in SLACK_CONFIGS:
            f = RUN_DIR / f"{s['id']}.json"
            if f.exists():
                rec = json.loads(f.read_text())
                print(f"cached  {s['id']} ({rec['status']})")
            elif args.report_only:
                continue
            else:
                print(f"running {s['id']} ...", flush=True)
                rec = run_slack(s, econ, capex_tv)
                f.write_text(json.dumps(json_safe(rec)))
                print(f"   -> {rec['status']} in {rec['seconds']:.0f}s", flush=True)
            slack.append(rec)
    write_outputs(recs, slack)


if __name__ == "__main__":
    main()
