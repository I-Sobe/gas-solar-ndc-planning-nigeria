"""
52_invariants.py -- TASK 52 Part 2: an executable invariant suite over the
load-bearing layer.  REPORT ONLY: nothing here fixes anything.

    python scripts/52_invariants.py --list
    python scripts/52_invariants.py --stage A [--only ID[,ID...]]

Every invariant is a function returning PASS / FAIL / SKIP (class `R`).
FAIL carries expected, observed and where (file::construct).  SKIP carries
the reason and, when the invariant cannot be asserted on a reduced
instance, the real-scale configuration it would need.

Rules the harness enforces (TASK 52 Part 2 section 1):
  * No invariant reads a stored summary file.  Every "at the solution"
    assertion runs against a model the harness solved itself.
  * Reduced instances only (seconds each); dimensions are printed in the
    report by `fixture_dims()`.
  * Solver: CBC (what produced the results), through src.optimize_model.
    solve_model(), which sets primal/dual/integer tolerance 1e-7 explicitly.
    HiGHS is not used; if a verdict could depend on the choice the
    invariant says so.
  * Independent recomputation: formulas below (CRF, windows, rebuild years,
    O&M, fuel) are re-implemented here from the formulation, not imported
    from the builder.  Where the builder's own nested function is itself the
    object under test (A1, G3-1) it is extracted by AST and executed.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn

from src.io import load_econ, load_solar_capex_by_year
from src.optimize_model import solve_model
from src.optimize_model_sliced import build_model_sliced
from src.scenarios import BASELINE_CENTRAL_ARM, load_scenario

SOLVER = "cbc"
SOLVER_NOTE = "CBC 2.10.13 via solve_model(): primalTolerance=dualTolerance=integerTolerance=1e-7"


# ---------------------------------------------------------------------------
# result type and registry
# ---------------------------------------------------------------------------
@dataclass
class R:
    status: str                      # PASS | FAIL | SKIP
    detail: str = ""
    expected: str = ""
    observed: str = ""
    where: str = ""
    extra: dict = field(default_factory=dict)


def PASS(detail=""):
    return R("PASS", detail)


def FAIL(expected, observed, where, detail=""):
    return R("FAIL", detail, str(expected), str(observed), where)


def SKIP(reason, needs=""):
    return R("SKIP", reason + (f" | real-scale configuration needed: {needs}" if needs else ""))


REGISTRY = {}   # id -> (stage, text, func)


def inv(id_, stage, text):
    def deco(fn):
        REGISTRY[id_] = (stage, text, fn)
        return fn
    return deco


# ---------------------------------------------------------------------------
# independent formulation helpers (NOT imported from the builder)
# ---------------------------------------------------------------------------
def crf(r, L):
    if r <= 0:
        return 1.0 / L
    return r * (1 + r) ** L / ((1 + r) ** L - 1)


def annuity(r, L):
    """a(r, L) = sum_{k=1..L} (1+r)^-k, ordinary annuity."""
    return sum((1 + r) ** -k for k in range(1, L + 1))


def real_rate(nominal, infl):
    return (1 + nominal) / (1 + infl) - 1


def df(t, rs):
    return 1.0 / (1.0 + rs) ** t


def window_W(v, L, case, T_last, rs):
    """Discounted-instalment sum for vintage index v under capital_window_case.
    advance: t = v .. v+L-1 (capped at T_last); arrears: the same COUNT one year
    later, discount factor extended past T_last; due_crf: as advance."""
    off = 1 if case == "arrears" else 0
    return sum(df(t, rs) for t in range(v + off, min(v + off + L, T_last + 1 + off)))


def crf_case(r, L, case):
    c = crf(r, L)
    return c / (1 + r) if case == "due_crf" else c


def rebuild_years(b, life, T_last):
    life = int(round(life))
    out, k = [], 1
    while b + k * life <= T_last:
        out.append(b + k * life)
        k += 1
    return out


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------
FIX = dict(end_year=2036, lives={"solar": 8.0, "storage": 4.0, "gas": 8.0}, L_conc=6, L_comm=3)
CHANNELS = ("solar", "storage", "gas")


def fixture_dims():
    n = FIX["end_year"] - 2025 + 1
    return (f"horizon 2025-{FIX['end_year']} ({n} model years, T_last={n-1}); six slices "
            "(the builder's season x period structure is hard-wired, so the two-slice instance "
            "asked for is not constructible without patching the builder); lifetimes "
            f"solar/storage/gas {FIX['lives']['solar']:.0f}/{FIX['lives']['storage']:.0f}/{FIX['lives']['gas']:.0f} yr, "
            f"tenors concessional/commercial {FIX['L_conc']}/{FIX['L_comm']} yr so that L_eff is "
            f"{min(FIX['L_conc'],FIX['lives']['solar'])}/{FIX['L_comm']} (solar), "
            f"{min(FIX['L_conc'],FIX['lives']['storage'])}/{FIX['L_comm']} (storage); windows truncate from vintage "
            f"index {n-FIX['L_conc']+1} and storage rebuilds at v+4 (v <= {n-1-4}); solves ~0.3-0.6 s each")


class Fx:
    pass


def make_scenario(end_year=None, capital_case="unconstrained", arms=None, solar_build_case="deployment_unconstrained",
                  load_kw=None):
    end_year = FIX["end_year"] if end_year is None else end_year
    kw = dict(BASELINE_CENTRAL_ARM)
    kw["peak_multiple_case"] = kw.pop("peak_case")
    arms = dict(arms or {})
    lk_over = {k: arms.pop(k) for k in list(arms) if k in ("tariff_case", "om_case", "reserve_margin", "us_inflation")}
    kw.update(arms)
    lk = dict(demand_level_case="served", demand_case="organic_central", capital_case=capital_case,
              solar_build_case=solar_build_case, land_case="loose", carbon_case="no_policy",
              start_year=2025, end_year=end_year, reserve_margin="nirp_spinning", us_inflation=0.02,
              tariff_case="tariff_central", om_case="om_central")
    lk.update(lk_over)
    lk.update(load_kw or {})
    with contextlib.redirect_stdout(io.StringIO()):
        sc = load_scenario(**lk, **kw)
    return sc


def build_fx(window=None, capital_case="unconstrained", eaas=False, force=True, reduced_life=True,
             end_year=None, arms=None, sc_over=None, builder_kw=None, load_kw=None, solar_build_case="deployment_unconstrained",
             econ_over=None, force_lb=None, sc_fn=None, capex_scale=1.0):
    """Build (not solve) the reduced fixture.  Returns Fx."""
    x = Fx()
    sc = make_scenario(end_year, capital_case, arms, solar_build_case, load_kw)
    if window is not None:          # None = the load_scenario() default (TASK 53: arrears)
        sc["capital_window_case"] = window
    if reduced_life:
        sc["asset_lifetimes"] = dict(FIX["lives"])
        sc["L_conc"] = FIX["L_conc"]
        sc["L_comm"] = FIX["L_comm"]
    if eaas:
        sc["financing_regime"] = "eaas"
    sc.update(sc_over or {})
    if sc_fn is not None:
        sc_fn(sc)
    econ = load_econ("voll_mid", genset_case="genset_central")
    econ.update(econ_over or {})
    ys = sc["years"]
    capex = load_solar_capex_by_year(scenario_name="solar_low", start_year=int(ys[0]), end_year=int(ys[-1]))
    if capex_scale != 1.0:
        capex = {y: c * capex_scale for y, c in capex.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        m = build_model_sliced(scenario=sc, econ=econ, solar_capex_by_year=capex, **(builder_kw or {}))
    T = range(len(ys))
    if force:
        lb = dict(solar_public=50.0, storage=30.0, gas=10.0, solar_eaas=20.0)
        lb.update(force_lb or {})
        for t in T:
            m.solar_public_add[t].setlb(lb["solar_public"])
            m.storage_add[t].setlb(lb["storage"])
            m.gas_add[t].setlb(lb["gas"])
            if eaas:
                m.solar_eaas_add[t].setlb(lb["solar_eaas"])
    x.m, x.sc, x.econ, x.T, x.years, x.capex = m, sc, econ, T, [int(y) for y in ys], capex
    x.Tlast = len(T) - 1
    x.solved = False
    return x


ENGINE = "cbc"          # "cbc" (solve_model, what produced the results) or "highs" (CLI; Stage C G10-4 only)


def solve_highs_cli(m):
    """Solve a built model with the HiGHS command-line binary (the Python interface highspy is not installed):
    write the LP with symbolic labels, run `highs`, read primal values back into the Pyomo variables.
    Raises RuntimeError on a non-optimal model status (same contract as solve_model)."""
    import subprocess
    with tempfile.TemporaryDirectory() as d:
        lp, sol = os.path.join(d, "m.lp"), os.path.join(d, "s.sol")
        fname, smid = m.write(lp, io_options={"symbolic_solver_labels": True})
        smap = m.solutions.symbol_map[smid]
        r = subprocess.run(["highs", "--model_file", lp, "--solution_file", sol, "--presolve", "on"], capture_output=True, text=True)
        txt = open(sol).read().splitlines() if os.path.exists(sol) else []
        if not txt or txt[1].strip() != "Optimal":
            raise RuntimeError(f"HiGHS status: {txt[1].strip() if len(txt) > 1 else r.stdout[-200:]}")
        i = txt.index(next(l for l in txt if l.startswith("# Columns")))
        for l in txt[i + 1:]:
            if l.startswith("#") or not l.strip():
                break
            name, val = l.rsplit(" ", 1)
            if name in smap.bySymbol:
                ref = smap.bySymbol[name]
                if ref is not None and not ref.fixed:
                    ref.set_value(float(val))
    for var in m.component_data_objects(pyo.Var, descend_into=True):
        if var.value is None:
            var.set_value(0.0)
    return {"status": "optimal", "optimal": True, "solver": "highs_cli"}


def solve_fx(x):
    if ENGINE == "highs":
        x.status = solve_highs_cli(x.m)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            st = solve_model(x.m, scenario=x.sc, solver=SOLVER)
        x.status = st
    x.solved = True
    return x


def obj(x):
    return float(pyo.value(x.m.system_cost_npv))


def v(e):
    return float(pyo.value(e))


def lp_md5(x):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.lp")
        x.m.write(p, io_options={"symbolic_solver_labels": True})
        return hashlib.md5(open(p, "rb").read()).hexdigest()


_CACHE = {}


def cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def solved(key, **kw):
    """A solved fixture, built once per key."""
    return cached(("solved", key), lambda: solve_fx(build_fx(**kw)))


def near(a, b, rel=1e-9, abs_=0.0):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b), 1e-300))


# ---------------------------------------------------------------------------
# AST utilities
# ---------------------------------------------------------------------------
def src_of(path):
    return Path(ROOT / path).read_text()


def extract_nested_function(path, name):
    """Compile a (nested) function's source standalone so the code under test is run as is."""
    tree = ast.parse(src_of(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(src_of(path), node)
            import textwrap
            ns = {}
            exec(textwrap.dedent(seg), {"pyo": pyo}, ns)
            return ns[name]
    raise KeyError(f"{name} not found in {path}")


# ===========================================================================
# STAGE A -- G7: arm, scenario and builder plumbing
# ===========================================================================
BUILDERS = {"sliced": "src/optimize_model_sliced.py", "annual": "src/optimize_model.py"}


def _const_str(n):
    return n.value if isinstance(n, ast.Constant) and isinstance(n.value, str) else None


def scenario_gets(path, var="scenario"):
    """Every `<var>.get(key, default)` call: (key, default_source, lineno)."""
    s = src_of(path)
    tree = ast.parse(s)
    out = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
                and isinstance(n.func.value, ast.Name) and n.func.value.id == var and n.args):
            key = _const_str(n.args[0])
            dflt = ast.get_source_segment(s, n.args[1]) if len(n.args) > 1 else "None (implicit)"
            out.append((key if key is not None else "<dynamic>", dflt, n.lineno))
    return out


def scenario_subscripts(path, var="scenario"):
    s = src_of(path)
    tree = ast.parse(s)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == var:
            sl = n.slice
            k = _const_str(sl)
            out.append((k if k is not None else "<dynamic>", n.lineno))
    return out


_FUNC_INDEX = None


def func_index():
    """name -> [(path, FunctionDef)] over src/*.py."""
    global _FUNC_INDEX
    if _FUNC_INDEX is None:
        _FUNC_INDEX = {}
        for p in sorted((ROOT / "src").glob("*.py")):
            rel = str(p.relative_to(ROOT))
            for n in ast.walk(ast.parse(p.read_text())):
                if isinstance(n, ast.FunctionDef):
                    _FUNC_INDEX.setdefault(n.name, []).append((rel, n))
    return _FUNC_INDEX


def keys_in_node(node, var):
    ks = set()
    dyn = False
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
                and isinstance(n.func.value, ast.Name) and n.func.value.id == var and n.args):
            k = _const_str(n.args[0])
            if k is None:
                dyn = True
            else:
                ks.add(k)
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == var:
            k = _const_str(n.slice)
            if k is None:
                dyn = True
            else:
                ks.add(k)
    return ks, dyn


def builder_keys(path, entry):
    """Scenario keys read by the builder function `entry` and by every helper it hands `scenario` to."""
    idx = func_index()
    seen, keys, dyn_any = set(), set(), False
    reached = []
    stack = [(path, entry, "scenario")]
    while stack:
        p, name, var = stack.pop()
        if (p, name, var) in seen:
            continue
        seen.add((p, name, var))
        for (pp, node) in idx.get(name, []):
            if p and pp != p and name != entry:
                continue
            ks, dyn = keys_in_node(node, var)
            keys |= ks
            dyn_any |= dyn
            reached.append(f"{pp}::{name}({var})")
            for c in ast.walk(node):
                if isinstance(c, ast.Call):
                    fname = c.func.id if isinstance(c.func, ast.Name) else (c.func.attr if isinstance(c.func, ast.Attribute) else None)
                    if fname is None or fname not in idx:
                        continue
                    for i, a in enumerate(c.args):
                        if isinstance(a, ast.Name) and a.id == var:
                            for (cp, cn) in idx[fname]:
                                params = [x.arg for x in cn.args.args]
                                if i < len(params):
                                    stack.append((cp, fname, params[i]))
                    for kw in c.keywords:
                        if isinstance(kw.value, ast.Name) and kw.value.id == var and kw.arg:
                            stack.append((None, fname, kw.arg))
    return keys, dyn_any, reached


@inv("G7-1", "A", "Every scenario.get(k, default) in both builders: key, default, whether load_scenario sets it; flag differing defaults and fallback-only keys")
def g7_1():
    base = make_scenario()
    alt = make_scenario(arms={"conversion_case": "conv_none"})   # arm-dependent keys take their default here
    rows, bad = [], []
    for bname, path in BUILDERS.items():
        for key, dflt, ln in scenario_gets(path):
            in_sc = key in base
            sc_val = base.get(key, "<absent>")
            try:
                dv = ast.literal_eval(dflt)
                lit = True
            except Exception:
                dv, lit = None, False
            flag = ""
            if not in_sc:
                flag = "FALLBACK-ONLY (load_scenario never sets it)"
            elif lit and dv is None:
                flag = ""   # `.get(k, None)` is a presence guard (followed by a raise), not a value fallback
            elif lit and dv != sc_val and dv != alt.get(key, "<absent>") and not (isinstance(sc_val, (list, dict, tuple)) or sc_val is None):
                flag = "DEFAULT DIFFERS from load_scenario value"
            elif not lit:
                flag = "non-literal default (not compared)"
            rows.append((bname, ln, key, dflt, in_sc, repr(sc_val)[:60], flag))
            if flag.startswith(("FALLBACK", "DEFAULT DIFFERS")):
                bad.append((bname, ln, key, dflt, flag, repr(sc_val)[:40]))
    r = FAIL("no builder fallback differing from, or standing in for, load_scenario", f"{len(bad)} flagged of {len(rows)} .get() sites",
             "src/optimize_model_sliced.py + src/optimize_model.py::scenario.get(...)",
             detail="; ".join(f"{b}:{ln} {k} default={d} [{f}] load_scenario={s}" for b, ln, k, d, f, s in bad)) if bad else PASS(f"{len(rows)} .get() sites")
    r.extra["rows"] = rows
    r.extra["flagged"] = bad
    return r


@inv("G7-3", "A", "Every scenario key: live or inert per builder (static reachability incl. helpers); flag live-in-annual/inert-in-sliced and dead keys")
def g7_3():
    base = make_scenario()
    live = {}
    dyn = {}
    for bname, path in BUILDERS.items():
        entry = "build_model_sliced" if bname == "sliced" else "build_model"
        ks, d, reached = builder_keys(path, entry)
        live[bname], dyn[bname] = ks, d
    # keys read elsewhere in src/ and scripts/ (string literal search), excluding scenarios.py itself
    other_files = [p for p in list((ROOT / "src").glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
                   if p.name not in ("scenarios.py", "52_invariants.py")]
    texts = {p: p.read_text() for p in other_files}
    rows, flags = [], []
    for k in sorted(base):
        ls, la = k in live["sliced"], k in live["annual"]
        elsewhere = [p.name for p, t in texts.items() if f'"{k}"' in t or f"'{k}'" in t]
        if ls and la:
            cls = "live in both"
        elif ls:
            cls = "live in SLICED only"
        elif la:
            cls = "live in ANNUAL only (inert in the reporting builder)"
            flags.append((k, cls))
        else:
            cls = "inert in both builders" + (" (read elsewhere)" if elsewhere else " AND read nowhere else: DEAD")
            if not elsewhere:
                flags.append((k, cls))
        rows.append((k, cls, ",".join(elsewhere[:4])))
    extra_only = sorted((live["sliced"] | live["annual"]) - set(base))
    r = FAIL("no key live only in the non-reporting builder, no dead key",
             f"{len(flags)} flagged", "src/scenarios.py::load_scenario dict vs builders",
             detail="; ".join(f"{k}: {c}" for k, c in flags)) if flags else PASS("map complete")
    r.extra.update(rows=rows, builder_only_not_in_scenario=extra_only, dynamic_access=dyn)
    return r


def _arm_value_sets():
    from src import scenarios as S
    from src.cli_arms import _arm_specs
    arms = {}
    for kwarg, flag, dest, choices, typ, h in _arm_specs():
        if choices:
            arms[kwarg] = ("load", list(choices))
    arms["reserve_margin"] = ("load", ["nirp_spinning", 0.15])
    arms["us_inflation"] = ("load", [0.02, 0.03])
    arms["capital_case"] = ("cap", list(S.capital_envelope_scenarios().keys()))
    arms["demand_case"] = ("loadkw", list(S.demand_growth_scenarios().keys()))
    arms["land_case"] = ("loadkw", list(S.land_scenarios().keys()))
    arms["carbon_case"] = ("loadkw", list(S.carbon_policy_scenarios().keys()))
    arms["blend_case"] = ("loadkw", list(S.blended_finance_scenarios().keys()))
    arms["bank_tenor_case"] = ("loadkw", ["bank_tenor_comm", "bank_tenor_legacy"])
    arms["solar_build_case"] = ("sbc", list(S.solar_build_scenarios().keys()))
    arms["capital_window_case"] = ("window", list(S.capital_window_scenarios().keys()))
    arms["financing_regime"] = ("regime", ["traditional", "eaas"])
    return arms


CONTEXTS = {
    "A": "traditional regime, unconstrained capital",
    "B": "eaas regime, moderate capital",
    "C": "eaas regime, moderate capital, new_gas_deliverability=exempt",
    "D": "eaas regime, moderate capital, bank_tenor_case=bank_tenor_legacy",
}


def _build_for(kind, name, val, ctx):
    base = dict(eaas=(ctx != "A"), capital_case="unconstrained" if ctx == "A" else "moderate", force=False,
                arms={}, load_kw={})
    if ctx == "C":
        base["arms"]["new_gas_deliverability"] = "exempt"
    if ctx == "D":
        base["load_kw"]["bank_tenor_case"] = "bank_tenor_legacy"
    if name == "capital_treatment_case":
        base["arms"]["conversion_case"] = "conv_none"      # an available conversion arm requires f1_tranche
    if kind == "load":
        base["arms"][name] = val
    elif kind == "loadkw":
        base["load_kw"][name] = val
    elif kind == "cap":
        base["capital_case"] = val
    elif kind == "sbc":
        base["solar_build_case"] = val
    elif kind == "window":
        base["window"] = val
    elif kind == "regime":
        base["eaas"] = (val == "eaas")
    return build_fx(**base)


@inv("G7-2", "A", "For every arm a live runner sets: the built model (LP file) differs between at least two values; report arms that change nothing")
def g7_2():
    arms = _arm_value_sets()
    rows, dead = [], []
    for name, (kind, vals) in arms.items():
        res = {}
        live_ctx = None
        for ctx in CONTEXTS:
            md5s = {}
            err = None
            for val in vals:
                try:
                    md5s[val] = lp_md5(_build_for(kind, name, val, ctx))
                except Exception as e:      # an invalid combination is itself information
                    md5s[val] = f"ERR:{type(e).__name__}"
                    err = str(e)[:80]
            ok = {k: x for k, x in md5s.items() if not str(x).startswith("ERR")}
            distinct = len(set(ok.values()))
            res[ctx] = (distinct, len(ok), len(vals), err)
            if distinct >= 2:
                live_ctx = ctx
                break
        status = "LIVE (context %s)" % live_ctx if live_ctx else "NO EFFECT on the built model in either context"
        rows.append((name, vals, status, res))
        if not live_ctx:
            dead.append(name)
    r = FAIL("every arm changes the built model in at least one context", f"{len(dead)} arms changed nothing: {dead}",
             "scenario -> build_model_sliced plumbing", detail="contexts tried in order A,B,C,D: " + "; ".join(f"{k} = {v}" for k, v in CONTEXTS.items())) if dead else PASS(f"{len(rows)} arms all live")
    r.extra["rows"] = rows
    return r


def _runner_files():
    out = []
    for p in sorted((ROOT / "scripts").glob("*.py")):
        t = p.read_text()
        if "load_scenario(" in t and p.name not in ("52_invariants.py", "51_check_advance_identity.py"):
            out.append(p)
    return out


@inv("G7-4", "A", "For every arm a runner varies (CLI flag fed to load_scenario): its value reaches the runner's output key/hash; report silent-overwrite risks")
def g7_4():
    from src.cli_arms import _arm_specs
    spec_dest = {k: (f, d) for k, f, d, *_ in _arm_specs()}
    arm_names = set(spec_dest) | {"capital_window_case", "capital_case", "solar_build_case", "conc_share_rule"}   # TASK 98: the supplied-division flag must reach the output key
    documented = {"tariff_case": "01b tag omits it by design (66cd29e comments)",}
    rows, flagged = [], []
    for p in _runner_files():
        t = p.read_text()
        tree = ast.parse(t)
        # arms the runner takes as flags: std arms via add_standard_arms (minus exclude/with include), or explicit flags
        flags = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "add_standard_arms":
                inc = exc = None
                for kw in n.keywords:
                    if kw.arg in ("include", "exclude"):
                        try:
                            val = set(ast.literal_eval(kw.value))
                        except Exception:
                            val = None
                        if kw.arg == "include":
                            inc = val
                        else:
                            exc = val
                std = set(spec_dest)
                flags |= (inc if inc is not None else std - (exc or set()))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument" and n.args:
                a0 = _const_str(n.args[0])
                if a0 and a0.startswith("--"):
                    dest = a0[2:].replace("-", "_")
                    if dest in arm_names:
                        flags.add(dest)
        if not flags:
            continue
        if not any(w in t for w in ("to_csv(", "json.dump(", "write_text(", "open(", "savefig(")):
            documented[p.name] = "writes no output file (stdout probe); nothing to overwrite"
            continue
        # output key: any assignment to a name containing 'tag' (+ calls to hash8_of/short_tag/om_tag_suffix)
        tag_src = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and any(isinstance(tg, ast.Name) and "tag" in tg.id.lower() for tg in n.targets):
                tag_src.append(ast.get_source_segment(t, n) or "")
        keyed = "hash8_of" in t or "short_tag" in t or "append_index" in t
        tagtxt = "\n".join(tag_src)
        generic_all = ("standard_arms_kwargs" in tagtxt) or ("std.items()" in tagtxt) or ("standard_arms.items()" in tagtxt) or ("sorted(std" in tagtxt) \
            or ("complete_arm_tag" in t) or ("arm_key(" in t) or ("resolve_results_dir(" in t and "keyed=False" not in t)   # TASK 53 2.5: the generic keys, verified dynamically below
        in_key, not_in_key = [], []
        for a in sorted(flags):
            present = generic_all or (f"args.{a}" in tagtxt) or (a in tagtxt) or (f"{a}_tag_suffix" in tagtxt) or \
                      (a == "om_case" and "om_tag_suffix" in tagtxt) or (a == "capital_window_case" and "capital_window" in tagtxt)
            (in_key if present else not_in_key).append(a)
        rows.append((p.name, keyed, len(flags), in_key, not_in_key))
        if not_in_key:
            und = [a for a in not_in_key if a not in documented]
            if und or not keyed:
                flagged.append((p.name, "no per-arm output key" if not keyed else "", und or not_in_key))
    # dynamic half (TASK 53 2.5): the generic key must actually move when ANY single arm moves
    import argparse as _ap
    from src.cli_arms import arm_key, complete_arm_tag, standard_arms_kwargs
    base = {}
    for k, (flag, dest) in {k: (f, d) for k, f, d, *_ in _arm_specs()}.items():
        pass
    specs = _arm_specs()
    ns = _ap.Namespace(**{dest: (choices[0] if choices else (0.02 if typ is float else None)) for kw, flag, dest, choices, typ, _h in specs})
    ns.us_inflation = 0.02
    h0 = arm_key(ns)[1]
    not_moving = []
    for kw, flag, dest, choices, typ, _h in specs:
        alt = (choices[1] if choices and len(choices) > 1 else (0.03 if kw == "us_inflation" else None))
        if alt is None:
            continue
        n2 = _ap.Namespace(**vars(ns)); setattr(n2, dest, alt)
        if arm_key(n2)[1] == h0:
            not_moving.append(kw)
    for extra_arm in ("capital_window_case", "storage_capex_case"):
        n2 = _ap.Namespace(**vars(ns)); setattr(n2, extra_arm, "x_probe")
        if arm_key(n2)[1] == h0:
            not_moving.append(extra_arm)
    if not_moving:
        flagged.append(("src/cli_arms.py::arm_key", "dynamic probe", not_moving))
    r = FAIL("every arm a runner accepts reaches its output key", f"{len(flagged)} runners with arms outside the key",
             "scripts/*.py output-key construction (static read of `*tag*` assignments)",
             detail="; ".join(f"{n}: {('['+m+'] ' if m else '')}{a}" for n, m, a in flagged)) if flagged else PASS(f"{len(rows)} runners: every arm each accepts reaches its output key (generic arm_key / complete_arm_tag, or an explicit tag); the generic key moves when any single standard arm, capital_window_case or storage_capex_case moves")
    r.extra.update(rows=rows, documented=documented)
    return r


# ===========================================================================
# STAGE A -- item 1 + G3: the capital chain
# ===========================================================================
def _repn(expr):
    return generate_standard_repn(expr, compute_values=True)


def _coef_map(expr):
    r = _repn(expr)
    return {id(vv): c for vv, c in zip(r.linear_vars, r.linear_coefs)}, r.constant


def real_rates(sc):
    infl = float(sc["us_inflation"])
    return real_rate(float(sc["r_conc_nominal"]), infl), real_rate(float(sc["r_comm_nominal"]), infl)


def leff(sc):
    lives = sc["asset_lifetimes"]
    return ({ch: min(int(sc["L_conc"]), int(round(float(lives[ch])))) for ch in CHANNELS},
            {ch: min(int(sc["L_comm"]), int(round(float(lives[ch])))) for ch in CHANNELS})


def storage_price_vec(sc, econ):
    """Storage capex (USD per MWh) by model year, recomputed independently of the builder:
    'flat' = the econ scalar; 'declining_nrel' = the Storage_low rows of storage_capex.csv read here,
    held flat past the last sourced year (TASK 53 2.2)."""
    ys = [int(y) for y in sc["years"]]
    if sc["storage_capex_case"] == "flat":
        return [float(econ["STORAGE_COST_PER_MWH"])] * len(ys)
    import csv
    d = {}
    for r in csv.DictReader(open(ROOT / "data/cost/processed/storage_capex.csv", encoding="utf-8-sig")):
        if r["Scenario"] == "Storage_low":
            d[int(r["Year"])] = float(str(r["Storage_capex_usd_per_mwh"]).replace("$", "").replace(",", "").strip())
    last = max(d)
    return [d[min(y, last)] for y in ys]


def unit_capex(x, ch, t):
    if ch == "solar":
        return float(x.capex[x.years[t]])
    if ch == "storage":
        return storage_price_vec(x.sc, x.econ)[t]
    return float(x.sc["gas_capex_per_mw"])


@inv("A1", "A", "CRF(r,L) x a(r,L) = 1 for the builder's own _crf across a grid of r in (0,0.3] and L in 1..40, to 1e-12 (no model)")
def a1():
    worst, arg = 0.0, None
    for path in BUILDERS.values():
        f = extract_nested_function(path, "_crf")
        for r in [1e-4, 1.176e-3, 0.005, 0.0088, 0.02, 0.05, 0.08, 0.0882353, 0.11, 0.2, 0.3]:
            for L in range(1, 41):
                d = abs(f(r, L) * annuity(r, L) - 1.0)
                if d > worst:
                    worst, arg = d, (path, r, L, f(r, L) * annuity(r, L))
    # declared floor: r <= 0 returns 1/L
    f = extract_nested_function(BUILDERS["sliced"], "_crf")
    neg = f(-0.01, 10) * annuity(-0.01, 10)
    note = f"r<=0 floor (declared in the docstring): CRF(-0.01,10) x a = {neg:.6f} (identity does not hold below zero; not in the grid)"
    if worst <= 1e-12:
        return PASS(f"max |CRF x a - 1| = {worst:.2e} over both builders; {note}")
    return FAIL("<= 1e-12", f"{worst:.2e} at {arg}", "src/optimize_model*.py::_crf")


def _a2_for(case):
    x = build_fx(window=case, force=False, sc_over={})
    rs = float(x.sc["social_discount_rate"])
    nom = (1 + rs) * (1 + float(x.sc["us_inflation"])) - 1          # real rate == r_social
    # rebuild with that rate (the scenario dict is mutated before the build)
    x = build_fx(window=case, force=False, sc_over={"r_conc_nominal": nom, "r_comm_nominal": nom})
    m = x.m
    Le_c, Le_m = leff(x.sc)
    rows, worst, worst_trunc = [], 0.0, 0.0
    chans = [("solar_public", "solar", m.cost_public_solar_capital_charge_npv, m.solar_public_conc_vol, m.solar_public_comm_vol),
             ("solar_eaas", "solar", m.cost_eaas_solar_capital_charge_npv, m.solar_eaas_conc_vol, m.solar_eaas_comm_vol),
             ("gas", "gas", m.cost_gas_capital_charge_npv, m.gas_conc_vol, m.gas_comm_vol),
             ("storage", "storage", m.cost_storage_capital_charge_npv, m.storage_conc_vol, m.storage_comm_vol)]
    for name, tech, expr, vc, vm in chans:
        cm, _ = _coef_map(expr)
        for tr, var, L in (("conc", vc, Le_c[tech]), ("comm", vm, Le_m[tech])):
            if var[0].fixed:
                continue    # gas_conc_vol is fixed at 0 (A4); a fixed variable is a constant to the repn, coefficient not observable
            for t in x.T:
                coef = cm.get(id(var[t]), 0.0)
                ratio = coef / df(t, rs)          # present value at the vintage year, per unit of capital
                dev = abs(ratio - 1.0)
                truncated = t + L > x.Tlast + 1
                if truncated:
                    worst_trunc = max(worst_trunc, dev)
                else:
                    worst = max(worst, dev)
                if t == 0:
                    rows.append((name, tr, ratio))
    return worst, worst_trunc, rows


def _make_a2(case):
    ident = "default" if case is None else case
    @inv(f"A2-{ident}", "A", (f"CRF x W / DF[v] = 1 for every channel, tranche and untruncated vintage when r_conc = r_comm = r_social, window arm "
                              + ("= load_scenario() DEFAULT (TASK 53: must be 'arrears' and must pass)" if case is None else f"'{case}'")
                              + (" (retained arm: characterised, passes iff the deviation is exactly r_s = 8%)" if case == "advance" else "")))
    def f():
        worst, wt, rows = _a2_for(case)
        sample = "; ".join(f"{n}/{tr}: {r:.6f}" for n, tr, r in rows[:4])
        if case is None:
            dflt = make_scenario()["capital_window_case"]
            if dflt != "arrears":
                return FAIL("load_scenario() default capital_window_case == 'arrears'", dflt, "src/scenarios.py::load_scenario")
        if case == "advance":
            rs = float(make_scenario()["social_discount_rate"])
            if abs(worst - rs) <= 1e-9:
                return PASS(f"retained arm 'advance' carries exactly the (1 + r_s) factor: max |CRF x W / DF[v] - 1| = {worst:.9f} = r_s = {rs} (TASK 50 V1; no longer the default)")
            return FAIL(f"deviation == r_s = {rs}", f"{worst:.9f}", "src/optimize_model_sliced.py::_df_window_*")
        if worst <= 1e-10:
            return PASS(f"max |CRF x W / DF[v] - 1| = {worst:.2e} over all untruncated vintages (truncated vintages max dev {wt:.2e}; they are cut by design)")
        return FAIL("1.000000 (|dev| <= 1e-10)", f"max dev {worst:.6f}; v=0 ratios: {sample}",
                    "src/optimize_model_sliced.py::build_model_sliced _crf_tc x _df_window_* (coefficients read from cost_*_capital_charge_npv)",
                    detail="at r_f = r_s an ordinary-annuity charge over the window must have PV/I = 1 at the vintage year; "
                           "deviation = 1 + r_s for the advance window (TASK 50 V1). Normalised by DF[v] because the objective discounts to year 0.")
    return f


for _c in (None, "advance", "arrears", "due_crf"):
    _make_a2(_c)


def _volumes(x):
    m = x.m
    out = {}
    for name, tech, add_f, vc, vm in (
            ("solar_public", "solar", lambda t: v(m.solar_public_add[t]), m.solar_public_conc_vol, m.solar_public_comm_vol),
            ("solar_eaas", "solar", lambda t: v(m.solar_eaas_add[t]), m.solar_eaas_conc_vol, m.solar_eaas_comm_vol),
            ("gas", "gas", lambda t: v(m.gas_add[t]), m.gas_conc_vol, m.gas_comm_vol),
            ("storage", "storage", lambda t: v(m.storage_add[t]), m.storage_conc_vol, m.storage_comm_vol)):
        out[name] = (tech, [add_f(t) for t in x.T], [v(vc[t]) for t in x.T], [v(vm[t]) for t in x.T])
    return out


def S_base(window=None):
    return solved(("S", window), window=window, eaas=True, force=True)


@inv("A3", "A", "At the solution V_conc + V_comm = I x add for every channel and vintage (rel 1e-6)")
def a3():
    x = S_base()
    worst, where = 0.0, None
    n_pos = {}
    for name, (tech, add, vc, vm) in _volumes(x).items():
        for t in x.T:
            I = unit_capex(x, tech, t)
            lhs, rhs = vc[t] + vm[t], I * add[t]
            d = abs(lhs - rhs) / max(1.0, abs(rhs))
            n_pos[name] = n_pos.get(name, 0) + (rhs > 0)
            if d > worst:
                worst, where = d, (name, t, lhs, rhs)
    if min(n_pos.values()) == 0:
        return SKIP("a channel has zero build in the fixture, the identity would be vacuous", "n/a")
    if worst <= 1e-6:
        return PASS(f"max rel dev {worst:.2e}; positive-build vintages per channel {n_pos}")
    return FAIL("<= 1e-6", f"{worst:.3e} at {where}", "src/optimize_model_sliced.py::*_split_identity")


@inv("A4", "A", "V_conc[gas, v] = 0 exactly for every v (variable fixed, value 0.0)")
def a4():
    x = S_base()
    vals = [v(x.m.gas_conc_vol[t]) for t in x.T]
    fixed = all(x.m.gas_conc_vol[t].fixed for t in x.T)
    if all(a == 0.0 for a in vals) and fixed:
        return PASS("all 12 gas_conc_vol fixed at 0.0")
    return FAIL("0.0 and fixed", f"values {vals}, fixed={fixed}", "src/optimize_model_sliced.py::gas_conc_vol")


def independent_ccap(x):
    """C^cap recomputed from the solution's tranche volumes and the formulation, independently of the builder."""
    sc, m = x.sc, x.m
    rs = float(sc["social_discount_rate"])
    rc, rm = real_rates(sc)
    Lc, Lm = leff(sc)
    case = sc["capital_window_case"]
    T, Tl = x.T, x.Tlast
    tot, parts = 0.0, {}
    vols = _volumes(x)
    for name, (tech, add, vcv, vmv) in vols.items():
        s = 0.0
        for t in T:
            s += crf_case(rc, Lc[tech], case) * window_W(t, Lc[tech], case, Tl, rs) * vcv[t]
            s += crf_case(rm, Lm[tech], case) * window_W(t, Lm[tech], case, Tl, rs) * vmv[t]
        parts[name] = s
    reb = 0.0
    lives = sc["asset_lifetimes"]
    for tech, addf in (("solar", [v(m.solar_public_add[t]) + v(m.solar_eaas_add[t]) for t in T]),
                       ("storage", [v(m.storage_add[t]) for t in T]),
                       ("gas", [v(m.gas_add[t]) for t in T])):
        for b in T:
            for rt in rebuild_years(b, lives[tech], Tl):
                reb += crf_case(rm, Lm[tech], case) * window_W(rt, Lm[tech], case, Tl, rs) * addf[b] * unit_capex(x, tech, rt)
    parts["rebuild"] = reb
    conv = 0.0
    if hasattr(m, "convert_mw"):
        capex_kw = float(sc["conversion_capex_per_kw"])
        for tier in m.convert_mw:
            conv += crf_case(rm, Lm["gas"], case) * window_W(0, Lm["gas"], case, Tl, rs) * capex_kw * 1000.0 * v(m.convert_mw[tier])
    parts["conversion"] = conv
    return sum(parts.values()), parts


@inv("A5", "A", "C^cap recomputed independently from the solution's tranche volumes (incl. rebuild and conversion) equals the objective's capital term, all three window arms")
def a5():
    msgs, bad = [], []
    for case in ("advance", "arrears", "due_crf"):
        x = S_base(case)
        ind, parts = independent_ccap(x)
        mod = v(x.m.capital_charge_total_npv)
        ok = near(ind, mod, 1e-9)
        msgs.append(f"{case}: independent {ind:.4f} vs model {mod:.4f} (rel {abs(ind-mod)/max(mod,1):.1e})")
        if not ok:
            bad.append((case, ind, mod, {k: round(a, 2) for k, a in parts.items()}))
    if not bad:
        return PASS("; ".join(msgs))
    return FAIL("independent == model (rel 1e-9)", str(bad), "src/optimize_model_sliced.py::capital_charge_total_npv", detail="; ".join(msgs))


@inv("G3-1", "A", "_rebuild_years(b, life) returns exactly the years b+k*life <= T_last; none for an asset whose life reaches the horizon (builder's own function, extracted)")
def g3_1():
    bad = []
    for path in BUILDERS.values():
        try:
            tree = ast.parse(src_of(path))
        except Exception:
            continue
        has = any(isinstance(n, ast.FunctionDef) and n.name == "_rebuild_years" for n in ast.walk(tree))
        if not has:
            continue
        import textwrap
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == "_rebuild_years":
                seg = textwrap.dedent(ast.get_source_segment(src_of(path), n))
                for T_last in (11, 29):
                    ns = {}
                    exec(seg, {"_T_last": T_last, "T_last": T_last}, ns)
                    f = ns["_rebuild_years"]
                    for life in (3, 4, 8, 12, 15, 30, 40):
                        for b in range(T_last + 1):
                            got, exp = list(f(b, life)), rebuild_years(b, life, T_last)
                            if got != exp:
                                bad.append((path, T_last, b, life, got, exp))
                            if b + life > T_last and got:
                                bad.append((path, "life reaches horizon but rebuilds", b, life, got))
    return PASS("builder's _rebuild_years agrees with the independent definition (T_last 11 and 29, lives 3..40)") if not bad else FAIL("[] mismatches", f"{bad[:3]}", "src/optimize_model*.py::_rebuild_years")


def _make_g32(case):
    @inv(f"G3-2-{case}", "A", f"The rebuild charge uses the SAME window and CRF convention as the initial charge, window arm '{case}' (coefficients on the add variables vs an independent recomputation)")
    def f():
        x = build_fx(window=case, force=False, eaas=True)
        sc, m = x.sc, x.m
        rs = float(sc["social_discount_rate"])
        _, rm = real_rates(sc)
        _, Lm = leff(sc)
        lives = sc["asset_lifetimes"]
        cm, _ = _coef_map(m.cost_rebuild_capital_charge_npv)
        worst, where, n = 0.0, None, 0
        for tech, vars_ in (("solar", (m.solar_public_add, m.solar_eaas_add)), ("storage", (m.storage_add,)), ("gas", (m.gas_add,))):
            for b in x.T:
                exp = sum(crf_case(rm, Lm[tech], case) * window_W(rt, Lm[tech], case, x.Tlast, rs) * unit_capex(x, tech, rt)
                          for rt in rebuild_years(b, lives[tech], x.Tlast))
                for var in vars_:
                    got = cm.get(id(var[b]), 0.0)
                    n += exp > 0
                    d = abs(got - exp) / max(1.0, abs(exp))
                    if d > worst:
                        worst, where = d, (tech, b, got, exp)
        if n == 0:
            return SKIP("no rebuilds in fixture")
        if worst <= 1e-9:
            return PASS(f"{n} vintage/technology rebuild coefficients match; max rel dev {worst:.1e}")
        return FAIL("rebuild coeff == crf x W_arm(rebuild_t) x capex", f"{worst:.2e} at {where}",
                    "src/optimize_model_sliced.py::cost_rebuild_capital_charge_npv",
                    detail="a divergence means the V1 correction would have to be applied twice")
    return f


for _c in ("advance", "arrears", "due_crf"):
    _make_g32(_c)


@inv("G3-3", "A", "Rebuild capital appears exactly once in the objective (objective coefficient on each add variable = rebuild coefficient, with O&M switched off)")
def g3_3():
    x = build_fx(window="advance", force=False, eaas=True, arms={"om_case": "om_none"})
    m = x.m
    cm_rb, _ = _coef_map(m.cost_rebuild_capital_charge_npv)
    cm_ob, _ = _coef_map(m.system_cost_npv)
    worst, where, n = 0.0, None, 0
    for var in (m.solar_public_add, m.solar_eaas_add, m.storage_add, m.gas_add):
        for b in x.T:
            a, o = cm_rb.get(id(var[b]), 0.0), cm_ob.get(id(var[b]), 0.0)
            n += a > 0
            d = abs(o - a) / max(1.0, abs(a))
            if d > worst:
                worst, where = d, (var.name, b, o, a)
    if n == 0:
        return SKIP("no rebuilds in fixture")
    if worst <= 1e-12:
        return PASS(f"{n} rebuild-carrying variables: objective coefficient == rebuild coefficient exactly (once)")
    return FAIL("objective coef == rebuild coef", f"{worst:.3e} at {where} (ratio o/a = {where[2]/where[3] if where[3] else 'n/a'})",
                "src/optimize_model_sliced.py::system_cost_npv")


# ===========================================================================
# STAGE A -- item 6 objective assembly, and G2 (O&M)
# ===========================================================================
def independent_om(x, cap=None):
    """O&M NPV from the capacity vector and the rates (formulation: FOM USD/kW-yr x 1000 on power; storage FOM
    per kW of POWER converted with the duration; VOM USD/MWh on generation, TWh x 1e6)."""
    sc, m = x.sc, x.m
    rs = float(sc["social_discount_rate"])
    om = sc["om_cost_by_year"]
    dur = float(sc["storage_duration_hours"])
    tot = 0.0
    conv_mw = sum(v(m.convert_mw[k]) for k in m.convert_mw) if hasattr(m, "convert_mw") else 0.0
    S = list(m.S)
    for t in x.T:
        y = x.years[t]
        d = df(t, rs)
        solar = v(m.solar_capacity_mw[t]) if cap is None else cap["solar"][t]
        stor = v(m.storage_capacity_mwh[t]) if cap is None else cap["storage"][t]
        gas_tot = v(m.gas_capacity_mw[t]) if cap is None else cap["gas"][t]
        gas_unc = gas_tot - conv_mw
        gen_unc = sum(v(m.gas_generation[t, sp]) for sp in S) if cap is None else cap["gen_unc"][t]
        gen_cc = (sum(v(m.convert_generation[k, t, sp]) for k in m.convert_mw for sp in S) if (hasattr(m, "convert_mw") and cap is None) else (0.0 if cap is None else cap["gen_cc"][t]))
        tot += d * (solar * om["solar_utility"][y]["fom_usd_per_kw_yr"] * 1000.0
                    + stor * om["storage"][y]["fom_usd_per_kw_yr"] * 1000.0 / dur
                    + gas_unc * om["gas_ct"][y]["fom_usd_per_kw_yr"] * 1000.0
                    + conv_mw * om["gas_cc"][y]["fom_usd_per_kw_yr"] * 1000.0
                    + gen_unc * 1e6 * om["gas_ct"][y]["vom_usd_per_mwh"]
                    + gen_cc * 1e6 * om["gas_cc"][y]["vom_usd_per_mwh"])
    return tot


def independent_terms(x):
    sc, m, econ = x.sc, x.m, x.econ
    rs = float(sc["social_discount_rate"])
    S = list(m.S)
    fuel = sum(df(t, rs) * v(m.gas_to_power[t, sp]) * econ["GAS_COST_PER_TWH_TH"] for t in x.T for sp in S)
    if hasattr(m, "convert_to_power"):
        fuel += sum(df(t, rs) * v(m.convert_to_power[k, t, sp]) * econ["GAS_COST_PER_TWH_TH"]
                    for k in m.convert_mw for t in x.T for sp in S)
    price = econ.get("GENSET_COST_PER_TWH_E", econ["UNSERVED_ENERGY_PENALTY"])
    selfgen = sum(df(t, rs) * v(m.selfgen[t, sp]) * price for t in x.T for sp in S)
    uns = sum(df(t, rs) * v(m.unserved[t, sp]) * econ["UNSERVED_ENERGY_PENALTY"] for t in x.T for sp in S)
    carbon = 0.0
    if sc.get("carbon_active", False):
        carbon = sum(df(t, rs) * v(m.emissions_by_year[t]) * float(sc["carbon_price"]) for t in x.T)
    ccap, parts = independent_ccap(x)
    return {"fuel (gas opex)": fuel, "capital charge C^cap": ccap, "self-generation": selfgen,
            "unserved (VoLL)": uns, "carbon": carbon, "O&M": independent_om(x)}, parts


@inv("A6", "A", "Every cost term recomputed from the solution; their sum equals m.system_cost_npv (rel 1e-6); each term's share reported")
def a6():
    x = S_base()
    terms, parts = independent_terms(x)
    tot = sum(terms.values())
    mod = obj(x)
    shares = "; ".join(f"{k} {100*a/mod:.2f}%" for k, a in terms.items())
    capshare = "; ".join(f"{k} {100*a/mod:.2f}%" for k, a in parts.items())
    r = PASS(f"sum {tot:.4f} vs model {mod:.4f} (rel {abs(tot-mod)/mod:.1e}); shares of the objective: {shares}; C^cap components: {capshare}") \
        if near(tot, mod, 1e-6) else FAIL("sum of independent terms == system_cost_npv", f"{tot:.4f} vs {mod:.4f}", "src/optimize_model_sliced.py::system_cost_npv", detail=shares)
    r.extra["shares"] = {k: a / mod for k, a in terms.items()}
    r.extra["capital_parts"] = {k: a / mod for k, a in parts.items()}
    return r


@inv("A7", "A", "Each objective term enters exactly once (objective == sum of the named terms, coefficient-wise); every decision variable's objective coefficient has the expected sign")
def a7():
    x = S_base()
    m = x.m
    names = ["cost_gas_opex_npv", "capital_charge_total_npv", "cost_selfgen_npv", "cost_unserved_voll_npv", "cost_carbon_npv", "cost_om_npv"]
    cm_obj, c_obj = _coef_map(m.system_cost_npv)
    acc, c_acc = {}, 0.0
    for n in names:
        cm, c = _coef_map(getattr(m, n))
        c_acc += c
        for k, a in cm.items():
            acc[k] = acc.get(k, 0.0) + a
    worst = max([abs(cm_obj.get(k, 0.0) - acc.get(k, 0.0)) / max(1.0, abs(cm_obj.get(k, 0.0))) for k in set(cm_obj) | set(acc)] + [abs(c_obj - c_acc) / max(1.0, abs(c_obj))])
    cap_names = ["cost_public_solar_capital_charge_npv", "cost_eaas_solar_capital_charge_npv", "cost_gas_capital_charge_npv",
                 "cost_storage_capital_charge_npv", "cost_rebuild_capital_charge_npv", "cost_conversion_capital_charge_npv"]
    cm_cap, _ = _coef_map(m.capital_charge_total_npv)
    acc2 = {}
    for n in cap_names:
        for k, a in _coef_map(getattr(m, n))[0].items():
            acc2[k] = acc2.get(k, 0.0) + a
    worst2 = max([abs(cm_cap.get(k, 0.0) - acc2.get(k, 0.0)) / max(1.0, abs(cm_cap.get(k, 0.0))) for k in set(cm_cap) | set(acc2)])
    # sign / presence table by variable family
    fam = {}
    for comp in m.component_objects(pyo.Var, descend_into=False):
        co = [cm_obj.get(id(comp[i]), 0.0) for i in comp]
        fam[comp.name] = (min(co), max(co), sum(1 for i in comp if cm_obj.get(id(comp[i]), 0.0) != 0.0), len(co))
    comp_fixed = {c.name: [c[i].fixed for i in c] for c in m.component_objects(pyo.Var, descend_into=False)}
    expect_zero = {"gas_curtail", "hydro_gen", "storage_charge", "storage_discharge", "eaas_subsidy"}
    expect_pos = {"gas_to_power", "convert_to_power", "selfgen", "unserved", "solar_public_conc_vol", "solar_public_comm_vol",
                  "solar_eaas_conc_vol", "solar_eaas_comm_vol", "gas_comm_vol", "storage_conc_vol", "storage_comm_vol", "convert_comm_vol"}
    bad = []
    for n, (lo, hi, nnz, N) in fam.items():
        if n in expect_zero and (lo != 0.0 or hi != 0.0):
            bad.append((n, "expected absent from objective", lo, hi))
        if n in expect_pos and lo <= 0.0 and not all(comp_fixed.get(n, [False])):
            bad.append((n, "expected strictly positive", lo, hi))
        if lo < 0.0:
            bad.append((n, "negative coefficient", lo, hi))
    r = None
    if worst <= 1e-12 and worst2 <= 1e-12 and not bad:
        r = PASS(f"objective == sum of 6 named terms (max coefficient dev {worst:.1e}); capital == sum of 6 components ({worst2:.1e}); no negative coefficient; zero-cost variables absent")
    else:
        r = FAIL("objective == named terms (once each), signs as expected", f"term dev {worst:.2e}, capital dev {worst2:.2e}, sign problems {bad}",
                 "src/optimize_model_sliced.py::system_cost_npv / capital_charge_total_npv")
    r.extra["families"] = fam
    return r


def _fixed_point_model(arm_om="om_central", vals=None):
    """An UNSOLVED model with all variables set to a prescribed point (for O&M evaluation)."""
    x = build_fx(force=False, arms={"om_case": arm_om})
    m = x.m
    for comp in m.component_objects(pyo.Var, descend_into=False):
        for i in comp:
            comp[i].set_value(0.0)
    return x


@inv("G2-1", "A", "O&M NPV recomputed independently from the capacity vector and the rates equals the objective's O&M term (rel 1e-9)")
def g2_1():
    x = S_base()
    ind, mod = independent_om(x), v(x.m.cost_om_npv)
    if near(ind, mod, 1e-9):
        return PASS(f"independent {ind:.4f} vs model {mod:.4f}")
    return FAIL("independent == model", f"{ind:.4f} vs {mod:.4f}", "src/io.py::compute_om_cost_npv / optimize_model_sliced.py::cost_om_npv")


@inv("G2-2", "A", "O&M is affine-linear in capacity: O&M(2P) - O&M(0) = 2 (O&M(P) - O&M(0)) on a fixed point (generation held at zero)")
def g2_2():
    def at(k):
        x = _fixed_point_model()
        for t in x.T:
            x.m.solar_public_add[t].set_value(k * 100.0)
            x.m.storage_add[t].set_value(k * 50.0)
            x.m.gas_add[t].set_value(k * 20.0)
        return v(x.m.cost_om_npv)
    o0, o1, o2 = at(0), at(1), at(2)
    if o1 - o0 <= 0:
        return SKIP("fixed point carries no O&M increment")
    ratio = (o2 - o0) / (o1 - o0)
    if abs(ratio - 2.0) <= 1e-12:
        return PASS(f"O&M(0)={o0:.2f}, O&M(P)={o1:.2f}, O&M(2P)={o2:.2f}; increment ratio {ratio:.15f}")
    return FAIL("2.0", f"{ratio}", "src/io.py::compute_om_cost_npv")


@inv("G2-3", "A", "Every technology that carries capacity carries O&M; report any with zero and whether that is declared")
def g2_3():
    x = build_fx(force=False)
    om = x.sc["om_cost_by_year"]
    y0 = x.years[0]
    carries = {"solar_utility": om["solar_utility"][y0]["fom_usd_per_kw_yr"], "storage": om["storage"][y0]["fom_usd_per_kw_yr"],
               "gas_ct": om["gas_ct"][y0]["fom_usd_per_kw_yr"], "gas_cc": om["gas_cc"][y0]["fom_usd_per_kw_yr"]}
    zeros = [k for k, a in carries.items() if not a > 0]
    # technologies with capacity in the model but no entry in om_cost_by_year
    cap_techs = {"solar", "storage", "gas (unconverted)", "gas (converted)", "hydro"}
    hydro_cap = x.sc.get("hydro_capacity_mw_by_year")
    hydro_has_cap = hydro_cap is not None and max(hydro_cap) > 0
    in_om = {"solar", "storage", "gas (unconverted)", "gas (converted)"}
    missing = sorted(cap_techs - in_om) if hydro_has_cap else []
    declared = ""
    for fp in sorted((ROOT / "docs").glob("*.md")):
        for i, line in enumerate(fp.read_text().splitlines(), 1):
            l = line.lower()
            if "hydro" in l and ("o&m" in l or " fom" in l or "operation and maintenance" in l):
                declared = f"{fp.name}:{i}: {line.strip()[:110]}"
                break
        if declared:
            break
    if zeros or missing:
        return FAIL("every capacity-carrying technology has O&M > 0", f"zero-rate technologies {zeros}; capacity without any O&M term: {missing} (hydro capacity max {max(hydro_cap):.0f} MW)",
                    "src/io.py::compute_om_cost_npv (no hydro argument) / sliced builder om_cost_npv call",
                    detail=("no declaration located: the closest line in docs/*.md mentioning hydro with O&M is "
                            f"[{declared or 'none'}], which I read: it is an epoch note on O&M figures, not a statement that hydro carries no O&M"))
    return PASS(f"FOM > 0 for {list(carries)}; hydro capacity is absent from the model")


@inv("G2-4", "A", "The om_case arms give distinct O&M totals on the same capacity point; om_none gives zero")
def g2_4():
    tot = {}
    for arm in ("om_none", "om_low", "om_central", "om_high"):
        x = _fixed_point_model(arm)
        for t in x.T:
            x.m.solar_public_add[t].set_value(100.0)
            x.m.storage_add[t].set_value(50.0)
            x.m.gas_add[t].set_value(20.0)
            for sp in x.m.S:
                x.m.gas_to_power[t, sp].set_value(1.0)
        tot[arm] = v(x.m.cost_om_npv)
    distinct = len(set(round(a, 6) for a in tot.values()))
    if distinct == 4 and tot["om_none"] == 0.0:
        return PASS("; ".join(f"{k} {a:.2f}" for k, a in tot.items()))
    return FAIL("4 distinct totals, om_none == 0", str(tot), "src/scenarios.py::om_scenarios / load_om_cost_by_year")


# ===========================================================================
# STAGE A -- item 2 (envelope), G6 (public budget), item 3 (bankability)
# ===========================================================================
def try_solve(**kw):
    """Build + solve a fixture; return Fx, or None if the solver reports infeasible/unbounded."""
    x = build_fx(**kw)
    try:
        solve_fx(x)
    except RuntimeError as e:
        x.error = str(e)[:100]
        return None
    return x


def conc_total(x):
    m = x.m
    return sum(v(m.solar_public_conc_vol[t]) + v(m.solar_eaas_conc_vol[t]) + v(m.storage_conc_vol[t]) for t in x.T)


def plan_vec(x):
    m = x.m
    return ([v(m.solar_public_add[t]) + v(m.solar_eaas_add[t]) for t in x.T] + [v(m.storage_add[t]) for t in x.T] + [v(m.gas_add[t]) for t in x.T])


def con_status(x, con, tol=1e-6):
    body, up = v(con.body), (v(con.upper) if con.upper is not None else None)
    d = x.m.dual.get(con, None)
    slack = (up - body) if up is not None else None
    return dict(body=body, upper=up, slack=slack, binding=(slack is not None and slack <= tol * max(1.0, abs(up))), dual=(float(d) if d is not None else None))


@inv("B1", "A", "Concessional envelope = 0  =>  every concessional volume (solar public/EaaS, storage) = 0 at the solution")
def b1():
    x = try_solve(eaas=True, sc_over={"concessional_envelope_usd": 0.0})
    if x is None:
        return FAIL("feasible with envelope 0", "solver infeasible", "src/optimize_model_sliced.py::f1_concessional_envelope_constraint")
    tot = conc_total(x)
    if tot <= 1e-6:
        return PASS(f"total concessional volume = {tot:.2e} USD")
    return FAIL("0", f"{tot:.4f}", "src/optimize_model_sliced.py::f1_concessional_envelope_constraint")


ENV_GRID = [0.0, 2.5e8, 7.5e8, 2.0e9, 1e13]


@inv("B2", "A", "Cost is monotone non-increasing in the concessional envelope across a coarse grid (reduced instance)")
def b2():
    costs = []
    for e in ENV_GRID:
        x = try_solve(eaas=True, sc_over={"concessional_envelope_usd": e})
        costs.append(None if x is None else obj(x))
    if None in costs:
        return FAIL("all grid points feasible", f"{costs}", "f1_concessional_envelope_constraint")
    bad = [(ENV_GRID[i], ENV_GRID[i + 1], costs[i], costs[i + 1]) for i in range(len(costs) - 1) if costs[i + 1] > costs[i] * (1 + 1e-9)]
    if not bad:
        return PASS("envelope grid " + ", ".join(f"{e:g}" for e in ENV_GRID) + " -> cost " + ", ".join(f"{c:.1f}" for c in costs))
    return FAIL("non-increasing", str(bad), "f1_concessional_envelope_constraint")


def _dual_check(points):
    """points: list of (label, con_status).  Slack => dual 0 (<=1e-9); binding => dual <= 1e-9 (Pyomo sign: shadow price = -dual >= 0)."""
    bad, notes = [], []
    for lab, st in points:
        d = st["dual"]
        if st["binding"]:
            if d is None or d > 1e-9:
                bad.append((lab, "binding but dual", d))
            if d is not None and abs(d) <= 1e-9:
                notes.append(f"{lab}: binding with zero dual (degenerate)")
        else:
            if d is None or abs(d) > 1e-9:
                bad.append((lab, "slack but dual", d))
    return bad, notes


@inv("B3", "A", "Envelope dual: zero exactly when the constraint is slack; sign of a minimisation shadow price otherwise (Pyomo dual <= 0), at every grid point")
def b3():
    pts = []
    for e in ENV_GRID:
        x = try_solve(eaas=True, sc_over={"concessional_envelope_usd": e})
        if x is None:
            return FAIL("feasible", f"infeasible at envelope {e}", "f1_concessional_envelope_constraint")
        pts.append((f"env={e:g}", con_status(x, x.m.f1_concessional_envelope_constraint)))
    bad, notes = _dual_check(pts)
    desc = "; ".join(f"{l}: slack={s['slack']:.3g} dual={s['dual']:.4g}" for l, s in pts)
    return PASS(desc + ("; " + "; ".join(notes) if notes else "")) if not bad else FAIL("dual 0 iff slack; <= 0 if binding", str(bad), "f1_concessional_envelope_constraint", detail=desc)


BUD_GRID = [3e8, 1e9, 2e9, 4e9, 1e15]


@inv("G6-1", "A", "Public budget set very large reproduces the unconstrained configuration (objective and plan); bitwise status reported")
def g6_1():
    a = try_solve(eaas=True, capital_case="moderate", sc_over={"public_solar_budget_npv": 1e15})
    b = try_solve(eaas=True, capital_case="unconstrained")
    if a is None or b is None:
        return FAIL("both feasible", "infeasible", "public_budget_constraint")
    ca, cb = obj(a), obj(b)
    dv = max(abs(p - q) for p, q in zip(plan_vec(a), plan_vec(b)))
    bit = ca == cb
    if near(ca, cb, 1e-9) and dv <= 1e-6:
        return PASS(f"objective {ca:.6f} vs {cb:.6f} (bitwise identical: {bit}; rel diff {abs(ca-cb)/cb:.1e}); max |plan diff| {dv:.1e}")
    return FAIL("equal objective and plan", f"{ca} vs {cb}; plan diff {dv}", "src/optimize_model_sliced.py::public_budget_constraint")


@inv("G6-2", "A", "Public budget = 0  =>  public solar build = 0 exactly (and public storage, EaaS subsidy)")
def g6_2():
    x = try_solve(eaas=True, force=False, capital_case="moderate", sc_over={"public_solar_budget_npv": 0.0})
    if x is None:
        return FAIL("feasible at budget 0 (gas/EaaS can serve)", "solver infeasible", "public_budget_constraint")
    ps = sum(v(x.m.solar_public_add[t]) for t in x.T)
    st = sum(v(x.m.storage_add[t]) for t in x.T)
    sub = sum(v(x.m.eaas_subsidy[t]) for t in x.T)
    if ps <= 1e-9 and st <= 1e-9 and sub <= 1e-9:
        return PASS(f"public solar {ps:.1e} MW, storage {st:.1e} MWh, EaaS subsidy {sub:.1e}; EaaS solar built {sum(v(x.m.solar_eaas_add[t]) for t in x.T):.1f} MW")
    return FAIL("0", f"public solar {ps}, storage {st}, subsidy {sub}", "public_budget_constraint")


@inv("G6-3", "A", "Cost is monotone non-increasing in the public budget (reduced instance)")
def g6_3():
    costs, rows = [], []
    for b in BUD_GRID:
        x = try_solve(eaas=True, force=False, capital_case="moderate", sc_over={"public_solar_budget_npv": b})
        rows.append((b, None if x is None else obj(x)))
    feas = [(b, c) for b, c in rows if c is not None]
    bad = [(feas[i], feas[i + 1]) for i in range(len(feas) - 1) if feas[i + 1][1] > feas[i][1] * (1 + 1e-9)]
    if len(feas) < 3:
        return SKIP(f"fewer than 3 feasible grid points: {rows}")
    return PASS(", ".join(f"{b:g}:{('infeasible' if c is None else f'{c:.1f}')}" for b, c in rows)) if not bad else FAIL("non-increasing", str(bad), "public_budget_constraint")


@inv("G6-4", "A", "Public budget dual: zero when slack; <= 0 (Pyomo) when binding, at every grid point")
def g6_4():
    pts = []
    for b in BUD_GRID:
        x = try_solve(eaas=True, force=False, capital_case="moderate", sc_over={"public_solar_budget_npv": b})
        if x is None:
            continue
        pts.append((f"budget={b:g}", con_status(x, x.m.public_budget_constraint)))
    if len(pts) < 3 or not any(s["binding"] for _, s in pts) or all(s["binding"] for _, s in pts):
        return SKIP(f"grid did not straddle binding/slack: {[(l, s['binding']) for l, s in pts]}")
    bad, notes = _dual_check(pts)
    desc = "; ".join(f"{l}: binding={s['binding']} dual={s['dual']:.4g}" for l, s in pts)
    return PASS(desc) if not bad else FAIL("dual 0 iff slack; <= 0 binding", str(bad), "public_budget_constraint", detail=desc)


@inv("G6-5", "A", "EaaS solar capital does not appear in the public budget constraint's left-hand side (coefficients on solar_eaas_add and the EaaS tranche volumes are 0)")
def g6_5():
    x = build_fx(eaas=True, capital_case="moderate", force=False)
    m = x.m
    cm, _ = _coef_map(m.public_budget_constraint.body)
    nz = {name: [(t, cm[id(var[t])]) for t in x.T if cm.get(id(var[t]), 0.0) != 0.0]
          for name, var in (("solar_eaas_add", m.solar_eaas_add), ("solar_eaas_conc_vol", m.solar_eaas_conc_vol), ("solar_eaas_comm_vol", m.solar_eaas_comm_vol))}
    sub = [cm.get(id(m.eaas_subsidy[t]), 0.0) for t in x.T]
    bad = {k: a for k, a in nz.items() if a}
    if not bad:
        return PASS(f"EaaS solar capital absent from the LHS. NOTE: the EaaS *subsidy* is in it (coefficient DF[t]: {sub[0]:.3f} .. {sub[-1]:.3f}), "
                    "so EaaS relieves the public budget by the capital it moves off-budget, less the subsidy it draws")
    return FAIL("0", str(bad), "src/optimize_model_sliced.py::public_budget_constraint")


@inv("G6-6", "A", "By coefficients: a dollar of concessional public-solar capital consumes both the envelope and the budget; a commercial dollar the budget only")
def g6_6():
    x = build_fx(eaas=True, capital_case="moderate", force=False)
    m = x.m
    rs = float(x.sc["social_discount_rate"])
    cb, _ = _coef_map(m.public_budget_constraint.body)
    ce, _ = _coef_map(m.f1_concessional_envelope_constraint.body)
    rows, bad = [], []
    for t in x.T:
        cap = unit_capex(x, "solar", t)
        env_conc, env_comm = ce.get(id(m.solar_public_conc_vol[t]), 0.0), ce.get(id(m.solar_public_comm_vol[t]), 0.0)
        # per dollar of overnight capital via the split identity (conc + comm = capex x add): budget sees DF*capex*add
        bud_per_dollar = cb.get(id(m.solar_public_add[t]), 0.0) / cap
        bud_direct = (cb.get(id(m.solar_public_conc_vol[t]), 0.0), cb.get(id(m.solar_public_comm_vol[t]), 0.0))
        ok = (abs(env_conc - 1.0) < 1e-12 and env_comm == 0.0 and abs(bud_per_dollar - df(t, rs)) < 1e-12 and bud_direct == (0.0, 0.0))
        if not ok:
            bad.append((t, env_conc, env_comm, bud_per_dollar, bud_direct))
        if t in (0, x.Tlast):
            rows.append(f"t={t}: envelope coef conc {env_conc:g}, comm {env_comm:g}; budget per capital dollar {bud_per_dollar:.6f} (DF={df(t, rs):.6f})")
    if not bad:
        return PASS("; ".join(rows) + ". The envelope counts undiscounted dollars, the budget NPV dollars (DF[t]); both tranches draw the budget identically via the split identity")
    return FAIL("envelope: conc 1, comm 0; budget: DF per capital dollar, 0 on tranche vars", str(bad[:2]), "f1_concessional_envelope_constraint / public_budget_constraint")


def _same_solution(a, b):
    """(bitwise_objective_equal, max plan diff)"""
    return obj(a) == obj(b), max(abs(p - q) for p, q in zip(plan_vec(a), plan_vec(b)))


DEAD_COEF = [
    # (label, what it varies, why it should be inert, kw-builder)
    ("dscr_case dscr_low vs dscr_high", "EaaS bankability cushion", "EaaS channel disabled (traditional regime)",
     lambda: [dict(eaas=False, arms={"dscr_case": "dscr_low"}), dict(eaas=False, arms={"dscr_case": "dscr_high"})]),
    ("tariff_case tariff_low vs tariff_high", "EaaS service tariff", "EaaS channel disabled",
     lambda: [dict(eaas=False, arms={"tariff_case": "tariff_low"}), dict(eaas=False, arms={"tariff_case": "tariff_high"})]),
    ("blend_case commercial_only vs concessional_heavy", "blended-finance rate (read only under bank_tenor_legacy)", "bank_tenor_case = bank_tenor_comm (default)",
     lambda: [dict(eaas=True, load_kw={"blend_case": "commercial_only"}), dict(eaas=True, load_kw={"blend_case": "concessional_heavy"})]),
    # TASK 78 D.1: the entry "ccgt_efficiency_case ccgt_eff_iso vs ccgt_eff_low, inert under new_gas_deliverability =
    # constrained" (TASK 61-66, register L120) is REMOVED. With gas_vintage_dispatch on by default (TASK 77 C.4) the arm
    # sets the added stream's efficiency and is LIVE under constrained deliverability: objective 12,156,792,461.50 against
    # 12,227,032,785.96, plan difference 55.1. That is the intended change; G7-2 asserts the arm reaches the model.
    ("r_conc_nominal 2.12% vs 6%, L_conc 25 vs 10", "concessional leg of the capital recovery arm", "concessional envelope = 0 (no concessional volume)",
     lambda: [dict(eaas=False, sc_over={"concessional_envelope_usd": 0.0, "r_conc_nominal": 0.0212}), dict(eaas=False, sc_over={"concessional_envelope_usd": 0.0, "r_conc_nominal": 0.06, "L_conc": 10})]),
    ("conversion capex 600 vs 850 USD/kW", "conversion retrofit cost", "convert_mw fixed at 0 (conversion available but unused)",
     lambda: None),
]


@inv("B4", "A", "Dead-coefficient table: vary each parameter the chapter says affects only mechanism X, with X disabled; objective and plan bit-identical")
def b4():
    rows, bad = [], []
    for label, what, why, kws in DEAD_COEF:
        specs = kws()
        if specs is None:
            continue
        sols = [try_solve(**s) for s in specs]
        if any(s is None for s in sols):
            rows.append((label, "infeasible", why))
            bad.append((label, "infeasible"))
            continue
        bit, dv = _same_solution(*sols)
        ca, cb = obj(sols[0]), obj(sols[1])
        verdict = "inert (bit-identical)" if bit and dv == 0.0 else (f"inert to 1e-9 but NOT bitwise (dobj={ca-cb:.3e}, dplan={dv:.1e})" if near(ca, cb, 1e-9) and dv <= 1e-6 else f"LIVE: objective {ca:.4f} vs {cb:.4f}, plan diff {dv:.3g}")
        rows.append((label, verdict, why))
        if verdict.startswith("LIVE"):
            bad.append((label, verdict))
    r = FAIL("every listed parameter inert when its mechanism is disabled", str(bad), "scenario -> build_model_sliced", detail="; ".join(f"{a}: {b} [{c}]" for a, b, c in rows)) if bad \
        else PASS("; ".join(f"{a}: {b}" for a, b, c in rows))
    r.extra["rows"] = rows
    return r


@inv("B5", "A", "Tariff very large => bankability slack at every EaaS vintage; subsidy fixed 0 with a positive per-MW gap => x_eaas = 0 exactly")
def b5():
    # part 1: huge tariff
    x = try_solve(eaas=True, force=True, sc_over={"solar_service_tariff_usd_per_twh": 1e15})
    if x is None:
        return FAIL("feasible", "infeasible", "eaas_bankability")
    worst_slack = []
    for t in x.T:
        if v(x.m.solar_eaas_add[t]) > 0:
            st = con_status(x, x.m.eaas_bankability[t])
            worst_slack.append(st["slack"] / max(1.0, abs(st["upper"])))
    ok1 = bool(worst_slack) and min(worst_slack) > 1e-6
    # part 2: low tariff, subsidy fixed at 0, positive gap
    y = build_fx(eaas=True, force=False, sc_over={"solar_service_tariff_usd_per_twh": 1e6})
    for t in y.T:
        y.m.eaas_subsidy[t].fix(0.0)
    solve_fx(y)
    gap = []
    rs = None
    for t in y.T:
        bankable = (1e6 * y.sc["solar_cf"] * 8760 / 1e6 * v(y.m.remaining_npv_factor[t])) / y.sc["required_margin"]
        gap.append(float(y.capex[y.years[t]]) - bankable)
    xe = sum(v(y.m.solar_eaas_add[t]) for t in y.T)
    ok2 = min(gap) > 0 and xe <= 1e-9
    if ok1 and ok2:
        return PASS(f"huge tariff: min relative slack over {len(worst_slack)} EaaS vintages {min(worst_slack):.3f}; low tariff, subsidy 0: min per-MW gap {min(gap):.0f} USD/MW > 0, EaaS build {xe:.1e} MW")
    return FAIL("slack at all vintages; x_eaas = 0", f"part1 ok={ok1} (slacks {worst_slack}); part2 ok={ok2} (gap min {min(gap)}, x_eaas {xe})", "src/optimize_model_sliced.py::eaas_bankability_rule")


# ===========================================================================
# STAGE A -- items 4/5 and G5, G11, G1
# ===========================================================================
import src.optimize_model_sliced as _osm
_ORIG_GAS_AVAIL = _osm.gas_available_power


@contextlib.contextmanager
def gas_scale(lam):
    """Scale the gas deliverability series A_t by lam for builds made inside the block."""
    def f(**kw):
        r = dict(_ORIG_GAS_AVAIL(**kw))
        r["available_twh_th"] = r["available_twh_th"] * lam
        return r
    _osm.gas_available_power = f
    try:
        yield
    finally:
        _osm.gas_available_power = _ORIG_GAS_AVAIL


def solve_scaled(lam, **kw):
    with gas_scale(lam):
        x = build_fx(**kw)
    try:
        solve_fx(x)
    except RuntimeError as e:
        x.error = str(e)[:100]
        return None
    x.lam = lam
    return x


def thermal_burn(x):
    m = x.m
    b = sum(v(m.gas_to_power[t, sp]) for t in x.T for sp in m.S)
    if hasattr(m, "convert_to_power"):
        b += sum(v(m.convert_to_power[k, t, sp]) for k in m.convert_mw for t in x.T for sp in m.S)
    return b


def slice_demand(x):
    return {(t, sp): float(pyo.value(x.m.balance[t, sp].lower)) for t in x.T for sp in x.m.S}


@inv("C1", "A", "Energy balance holds at the solution for every (t, s); busbar factor on self-generation is the stated one; demand sums to annual demand")
def c1():
    x = S_base()
    m = x.m
    conv = hasattr(m, "convert_generation")
    busbar = float(x.sc.get("selfgen_busbar_factor", 1.0))
    worst_viol, worst_sur, n = 0.0, 0.0, 0
    for t in x.T:
        for sp in m.S:
            sup = (v(m.gas_generation[t, sp]) + (sum(v(m.convert_generation[k, t, sp]) for k in m.convert_mw) if conv else 0.0)
                   + v(m.solar_generation[t, sp]) + v(m.hydro_gen[t, sp]) + v(m.storage_discharge[t, sp]) - v(m.storage_charge[t, sp])
                   + v(m.selfgen[t, sp]) * busbar + v(m.unserved[t, sp]))
            dem = float(pyo.value(m.balance[t, sp].lower))
            worst_viol = max(worst_viol, dem - sup)
            worst_sur = max(worst_sur, sup - dem)
            n += 1
    cm, _ = _coef_map(m.balance[0, list(m.S)[0]].body)
    coef = cm.get(id(m.selfgen[0, list(m.S)[0]]), None)
    ok_bus = coef is not None and abs(coef - busbar) < 1e-12 and busbar == 1.0
    note = (f"{n} (t,s) balances; max shortfall {worst_viol:.2e} TWh, max surplus {worst_sur:.2e} TWh (constraint is >=, free disposal); "
            f"selfgen coefficient in the balance = {coef} (selfgen_busbar_factor {busbar}, a scenario.get default: G7-1/L50)")
    if worst_viol <= 1e-6 and ok_bus:
        return PASS(note)
    return FAIL("shortfall <= 1e-6 and selfgen coefficient == selfgen_busbar_factor", note, "src/optimize_model_sliced.py::m.balance")


def _scarce(lam=0.6, **kw):
    return solve_scaled(lam, force=False, **kw)


@inv("C2", "A", "Gas deliverability holds at the solution; its dual is zero when slack and <= 0 (Pyomo) when binding, on a scarcity grid")
def c2():
    pts, viol = [], 0.0
    for lam in (0.5, 0.7, 1.0, 3.0):
        x = _scarce(lam)
        if x is None:
            return FAIL("feasible", f"infeasible at lambda={lam}", "gas_balance")
        for t in x.T:
            st = con_status(x, x.m.gas_balance[t])
            viol = max(viol, -st["slack"] if st["slack"] is not None else 0.0)
            pts.append((f"lam={lam},t={t}", st))
    bad, notes = _dual_check(pts)
    nb = sum(1 for _, s in pts if s["binding"])
    if nb == 0 or nb == len(pts):
        return SKIP(f"grid did not straddle binding/slack ({nb} binding of {len(pts)})")
    if viol <= 1e-6 and not bad:
        return PASS(f"{len(pts)} (lambda,t) points, {nb} binding; max deliverability violation {viol:.1e} TWh_th; no dual-sign or slack/dual failures; degenerate: {len(notes)}")
    return FAIL("slack => dual 0; binding => dual <= 0; no violation", f"violation {viol}; {bad[:3]}", "src/optimize_model_sliced.py::m.gas_balance")


@inv("C3", "A", "Scale A_t by lambda > 1, all else fixed: gas burn non-decreasing, cost non-increasing, sum of |deliverability duals| non-increasing (reduced instance)")
def c3():
    rows = []
    for lam in (0.5, 0.6, 0.8, 1.0, 1.5, 3.0):
        x = _scarce(lam)
        if x is None:
            rows.append((lam, None, None, None))
            continue
        rows.append((lam, thermal_burn(x), obj(x), sum(abs(float(x.m.dual.get(x.m.gas_balance[t], 0.0) or 0.0)) for t in x.T)))
    ok = [r for r in rows if r[1] is not None]
    bad = []
    for a, b in zip(ok[:-1], ok[1:]):
        if b[1] < a[1] - 1e-6:
            bad.append(("burn decreased", a[0], b[0], a[1], b[1]))
        if b[2] > a[2] * (1 + 1e-9):
            bad.append(("cost increased", a[0], b[0], a[2], b[2]))
        if b[3] > a[3] + 1e-6:
            bad.append(("dual grew", a[0], b[0], a[3], b[3]))
    desc = "; ".join(f"lam {l}: burn {b:.1f} TWh_th, cost {c/1e9:.3f} bn, sum|dual| {d:.3g}" for l, b, c, d in ok)
    return PASS(desc) if not bad else FAIL("burn up, cost down, dual down as lambda rises", str(bad[:3]), "src/optimize_model_sliced.py::gas_balance", detail=desc)


@inv("C7", "A", "REGRESSION GUARD (TASK 50 V4): gamma_t (gas_curtail) has zero objective coefficient, and forcing it to 0 reproduces the solve exactly")
def c7():
    x = S_base()
    cm, _ = _coef_map(x.m.system_cost_npv)
    coefs = [cm.get(id(x.m.gas_curtail[t]), 0.0) for t in x.T]
    y = build_fx(eaas=True, force=True)
    for t in y.T:
        y.m.gas_curtail[t].fix(0.0)
    solve_fx(y)
    ca, cb = obj(x), obj(y)
    dv = max(abs(p - q) for p, q in zip(plan_vec(x), plan_vec(y)))
    if all(c == 0.0 for c in coefs) and near(ca, cb, 1e-12) and dv <= 1e-9:
        return PASS(f"objective coefficients all 0; objective {ca:.6f} vs gamma=0 {cb:.6f} (bitwise {ca == cb}); max |plan diff| {dv:.1e}; free solve gamma sum = {sum(v(x.m.gas_curtail[t]) for t in x.T):.3g} TWh_th")
    return FAIL("coefficient 0 and identical solve", f"coefs {coefs}; {ca} vs {cb}; plan diff {dv}", "src/optimize_model_sliced.py::gas_curtail")


def _retirement_expected(sc, life):
    import csv
    plants = [r for r in csv.DictReader(open(ROOT / "data/capacity/raw/gas_plant_vintages.csv")) if r["plant_nerc_name"] not in ("AES_1", "Gbarain_1")]
    ys = [int(y) for y in sc["years"]]
    by = {}
    for p in plants:
        ry = int(p["commissioning_year"]) + life
        by[ry] = by.get(ry, 0.0) + float(p["installed_mw"])
    cum, out = 0.0, []
    for y in ys:
        cum += by.get(y, 0.0)
        cum = min(cum, 10947.0)
        out.append(cum)
    return out, sum(float(p["installed_mw"]) for p in plants)


@inv("G5-1", "A", "Capacity accounting identity at the solution: K_t = K_(t-1) - retirements_t + builds_t, every t")
def g5_1():
    x = S_base()
    m, sc = x.m, x.sc
    ser = sc["gas_baseline_retirement_series_mw"]
    base = float(sc["gas_baseline_mw"])
    worst = 0.0
    for t in x.T:
        K = v(m.gas_capacity_mw[t])
        prev = v(m.gas_capacity_mw[t - 1]) if t else base
        ret_t = min(ser[t], base) - (min(ser[t - 1], base) if t else 0.0)
        worst = max(worst, abs(K - (prev - ret_t + v(m.gas_add[t]))))
    return PASS(f"max |K_t - (K_t-1 - ret_t + build_t)| = {worst:.2e} MW (baseline {base:.0f} MW)") if worst <= 1e-6 else FAIL("0", f"{worst}", "src/optimize_model_sliced.py::gas_cap_rule")


@inv("G5-2", "A", "K_t >= 0 every t; the vintage retirement series is non-decreasing, <= the baseline, and equals an independent plant-by-plant recomputation and its stated total")
def g5_2():
    x = S_base()
    sc = x.sc
    ser = sc["gas_baseline_retirement_series_mw"]
    Kmin = min(v(x.m.gas_capacity_mw[t]) for t in x.T)
    sc30 = make_scenario(end_year=2054)
    out = []
    for life, arm in ((30, "gas_retire_vintage_30"), (40, "gas_retire_vintage_40"), (50, "gas_retire_vintage_50")):
        s2 = make_scenario(end_year=2054, arms={"gas_retirement_case": arm})
        exp, total = _retirement_expected(s2, life)
        got = s2["gas_baseline_retirement_series_mw"]
        d = max(abs(a - b) for a, b in zip(exp, got))
        mono = all(b >= a - 1e-9 for a, b in zip(got, got[1:]))
        out.append((arm, d, mono, max(got), total))
    bad = [o for o in out if o[1] > 1e-6 or not o[2] or o[3] > 10947 + 1e-6]
    if Kmin >= -1e-9 and not bad:
        return PASS(f"min K_t = {Kmin:.1f} MW; " + "; ".join(f"{a}: series == independent (max dev {d:.1e}), monotone, final {mx:.0f} MW of {tot:.0f} MW fleet" for a, d, _, mx, tot in out))
    return FAIL("K >= 0; series == independent", f"Kmin {Kmin}; {bad}", "src/scenarios.py::gas_vintage_retirement_series_mw")


@inv("G5-3", "A", "gas_build_case = gas_none  =>  no gas build (gas_add = 0 for every t)")
def g5_3():
    x = try_solve(eaas=True, force=True, force_lb={"gas": 0.0}, arms={"gas_build_case": "gas_none"})
    if x is None:
        return FAIL("feasible", "infeasible", "gas_build_case")
    tot = sum(v(x.m.gas_add[t]) for t in x.T)
    return PASS(f"total gas build {tot:.1e} MW; gas_add fixed: {all(x.m.gas_add[t].fixed for t in x.T)}") if tot == 0.0 else FAIL("0", f"{tot}", "src/optimize_model_sliced.py::gas_build_case")


@inv("G5-4", "A", "The new-gas exempt fuel allowance appears in exactly one constraint (name it)")
def g5_4():
    from pyomo.core.expr.visitor import identify_components
    x = build_fx(eaas=True, force=False, arms={"new_gas_deliverability": "exempt"})
    m = x.m
    allow = m.gas_exempt_fuel_allowance_twh_th
    hits = {}
    for con in m.component_data_objects(pyo.Constraint, active=True, descend_into=True):
        found = list(identify_components(con.body, {type(allow[0])})) + list(identify_components(con.upper, {type(allow[0])}) if con.upper is not None and hasattr(con.upper, "is_expression_type") else [])
        if any(f.parent_component() is allow for f in found):
            hits.setdefault(con.parent_component().name, []).append(con.name)
    names = {k: len(vv) for k, vv in hits.items()}
    return PASS(f"allowance referenced by exactly one constraint family: {names}") if list(names) == ["gas_balance"] else FAIL("{'gas_balance': 12}", str(names), "src/optimize_model_sliced.py::gas_exempt_fuel_allowance_twh_th")


@inv("G5-5", "A", "Gas generation <= alpha x K^g x hours / 1e6 for every (t,s) at the solution (fleet net of converted nameplate)")
def g5_5():
    x = S_base()
    m, sc = x.m, x.sc
    alpha = float(sc["gas_availability_factor"])
    days, hrs = {"dry": 181, "wet": 184}, {"night": 7, "day": 13, "peak": 4}
    conv_mw = sum(v(m.convert_mw[k]) for k in m.convert_mw) if hasattr(m, "convert_mw") else 0.0
    worst, n = -1e9, 0
    for t in x.T:
        for (s, p) in m.S:
            H = days[s] * hrs[p]
            cap = (v(m.gas_capacity_mw[t]) - conv_mw) * H * alpha / 1e6
            worst = max(worst, v(m.gas_generation[t, (s, p)]) - cap)
            n += 1
    return PASS(f"{n} slices; max (generation - limit) = {worst:.2e} TWh_e (alpha {alpha}, converted {conv_mw:.0f} MW netted out as the builder states)") if worst <= 1e-6 else FAIL("<= 0", f"{worst}", "src/optimize_model_sliced.py::gas_capacity_constraint")


@inv("C4", "A", "E_t = E_grid_t + E_self_t exactly at the solution, and each recomputed from thermal burn / genset fuel x emission factor, every t")
def c4():
    x = S_base()
    m, econ = x.m, x.econ
    efth = float(econ["EF_TCO2_PER_MWH_TH"])
    gens = float(econ["GENSET_EF_TH_HHV"]) / float(econ["GENSET_ETA_HHV"])
    worst_sum, worst_g, worst_s = 0.0, 0.0, 0.0
    for t in x.T:
        Et, Eg, Es = v(m.emissions_by_year[t]), v(m.emissions_grid_only_by_year[t]), v(m.emissions_selfgen_by_year[t])
        worst_sum = max(worst_sum, abs(Et - (Eg + Es)) / max(1.0, abs(Et)))
        burn = sum(v(m.gas_to_power[t, sp]) for sp in m.S) + (sum(v(m.convert_to_power[k, t, sp]) for k in m.convert_mw for sp in m.S) if hasattr(m, "convert_to_power") else 0.0)
        worst_g = max(worst_g, abs(Eg - burn * 1e6 * efth) / max(1.0, abs(Eg)))
        worst_s = max(worst_s, abs(Es - sum(v(m.selfgen[t, sp]) for sp in m.S) * 1e6 * gens) / max(1.0, abs(Es)))
    if max(worst_sum, worst_g, worst_s) <= 1e-9:
        return PASS(f"E_t == E_grid + E_self (max rel dev {worst_sum:.1e}); E_grid == thermal burn x EF_th x 1e6 ({worst_g:.1e}); E_self == genset electricity x EF/eta ({worst_s:.1e})")
    return FAIL("<= 1e-9", f"sum {worst_sum:.2e}, grid {worst_g:.2e}, self {worst_s:.2e}", "src/optimize_model_sliced.py::emissions_*_by_year")


@inv("C5", "A", "Cap set to +infinity reproduces the uncapped solve (objective, plan); bitwise status reported (reduced instance)")
def c5():
    a = try_solve(eaas=True)
    b = try_solve(eaas=True, builder_kw={"emissions_cap_by_year": [1e15] * (FIX["end_year"] - 2024)})
    c = try_solve(eaas=True, builder_kw={"emissions_cap": 1e18})
    if a is None or b is None or c is None:
        return FAIL("feasible", "infeasible", "emissions_constraint")
    msgs, bad = [], []
    for nm, y in (("per-year cap 1e15", b), ("total cap 1e18", c)):
        dv = max(abs(p - q) for p, q in zip(plan_vec(a), plan_vec(y)))
        okk = near(obj(a), obj(y), 1e-9) and dv <= 1e-6
        msgs.append(f"{nm}: objective {obj(y):.6f} vs {obj(a):.6f} (bitwise {obj(a) == obj(y)}), plan diff {dv:.1e}")
        if not okk:
            bad.append(nm)
    return PASS("; ".join(msgs)) if not bad else FAIL("identical solve", str(bad), "src/optimize_model_sliced.py::emissions_constraint", detail="; ".join(msgs))


@inv("C6", "A", "Every cap value <= E_base (the file's 2025 uncapped baseline) and each arm's trajectory is non-increasing in its own stated shape")
def c6():
    import csv
    rows = list(csv.DictReader(open(ROOT / "data/cost/processed/emissions_cap.csv")))
    by = {}
    for r in rows:
        by.setdefault(r["scenario"], []).append((int(r["year"]), float(r["cap_tco2"])))
    ebase = max(c for ser in by.values() for y, c in ser if y == 2025)
    firsts = {k: sorted(s)[0][1] for k, s in by.items()}
    bad = []
    for k, ser in by.items():
        ser = sorted(ser)
        if max(c for _, c in ser) > ebase + 1e-6:
            bad.append((k, "exceeds E_base", max(c for _, c in ser)))
        for (y0, c0), (y1, c1) in zip(ser, ser[1:]):
            if c1 > c0 + 1e-6:
                bad.append((k, "increases", y0, y1, c0, c1))
        if len(ser) != 30:
            bad.append((k, "length", len(ser)))
    if bad:
        return FAIL("cap <= E_base, non-increasing, 30 years", str(bad[:4]), "data/cost/processed/emissions_cap.csv (built by scripts/00_build_emissions_cap.py)")
    return PASS(f"{len(by)} arms x 30 years; E_base taken as the 2025 value in the file ({ebase/1e6:.6f} MtCO2/yr; all arms start at it). "
                "NOT asserted here: that E_base equals a recomputed baseline (the script derives it from an annual-builder conv_none run; TASK 52 solve set)")


def _demand_annual(x):
    d = slice_demand(x)
    return {t: sum(d[(t, sp)] for sp in x.m.S) for t in x.T}


@inv("G11-1", "A", "Obligation epsilon = 1 => slack at the solution; reproduces the unconstrained solve")
def g11_1():
    a = try_solve(eaas=True)
    b = try_solve(eaas=True, builder_kw={"max_offgrid_fraction": 1.0, "grid_service_mode": "annual"})
    if a is None or b is None:
        return FAIL("feasible", "infeasible", "grid_service_obligation")
    sl = min(con_status(b, b.m.grid_service_obligation[t])["slack"] for t in b.T)
    dv = max(abs(p - q) for p, q in zip(plan_vec(a), plan_vec(b)))
    if sl > 0 and near(obj(a), obj(b), 1e-9) and dv <= 1e-6:
        return PASS(f"min slack {sl:.3g} TWh; objective {obj(b):.6f} vs {obj(a):.6f} (bitwise {obj(a) == obj(b)}); plan diff {dv:.1e}")
    return FAIL("slack and identical", f"slack {sl}, {obj(a)} vs {obj(b)}", "src/optimize_model_sliced.py::grid_service_obligation")


@inv("G11-2", "A", "Obligation epsilon = 0 => self-generation + unserved = 0 exactly (even when gas is scarce)")
def g11_2():
    out = []
    for lam in (1.0, 0.5):
        x = solve_scaled(lam, force=False, eaas=True, builder_kw={"max_offgrid_fraction": 0.0, "grid_service_mode": "annual"})
        if x is None:
            out.append((lam, "infeasible"))
            continue
        sg = sum(v(x.m.selfgen[t, sp]) + v(x.m.unserved[t, sp]) for t in x.T for sp in x.m.S)
        out.append((lam, sg))
    bad = [o for o in out if o[1] != "infeasible" and o[1] > 1e-9]
    if bad:
        return FAIL("0", str(bad), "grid_service_obligation")
    if all(o[1] == "infeasible" for o in out):
        return SKIP("infeasible at epsilon 0 on this fixture")
    return PASS("; ".join(f"lambda {l}: selfgen+unserved = {s if s == 'infeasible' else format(s, '.1e')} TWh" for l, s in out))


@inv("G11-3", "A", "Total vs annual mode: annual => every year satisfies the bound; total => the horizon aggregate does and at least one year need not")
def g11_3():
    found = None
    for lam in (0.2, 0.15, 0.25):
        for eps in (0.0002, 0.0005, 0.001, 0.002, 0.004):
            xa = solve_scaled(lam, force=False, eaas=True, builder_kw={"max_offgrid_fraction": eps, "grid_service_mode": "annual"})
            xt = solve_scaled(lam, force=False, eaas=True, builder_kw={"max_offgrid_fraction": eps, "grid_service_mode": "total"})
            if xa is None or xt is None:
                continue
            Da, Dt = _demand_annual(xa), _demand_annual(xt)
            ua = {t: sum(v(xa.m.selfgen[t, sp]) + v(xa.m.unserved[t, sp]) for sp in xa.m.S) for t in xa.T}
            ut = {t: sum(v(xt.m.selfgen[t, sp]) + v(xt.m.unserved[t, sp]) for sp in xt.m.S) for t in xt.T}
            ann_ok = all(ua[t] <= eps * Da[t] + 1e-9 for t in xa.T)
            tot_ok = sum(ut.values()) <= eps * sum(Dt.values()) + 1e-9
            viol_years = [t for t in xt.T if ut[t] > eps * Dt[t] + 1e-9]
            if sum(ut.values()) > 1e-6 and viol_years:
                found = (lam, eps, ann_ok, tot_ok, viol_years, round(sum(ut.values()), 4), round(eps * sum(Dt.values()), 4))
                break
        if found:
            break
    if not found:
        return SKIP("no (lambda, epsilon) on the grid made total mode use self-generation unevenly across years on this fixture", "a scarcer, year-varying gas series at anchor scale")
    lam, eps, ann_ok, tot_ok, vy, used, allow = found
    if ann_ok and tot_ok and vy:
        return PASS(f"lambda={lam}, epsilon={eps}: annual mode satisfied every year; total mode used {used} TWh <= allowance {allow} TWh and exceeded the per-year bound in model years {vy} (F-33's content as a property)")
    return FAIL("annual: all years ok; total: aggregate ok and some year exceeds", f"annual ok {ann_ok}, total ok {tot_ok}, years {vy}", "grid_service_obligation")


@inv("G11-4", "A", "Obligation dual: zero when slack, <= 0 (Pyomo) when binding, across an epsilon grid (annual mode, scarce gas)")
def g11_4():
    pts = []
    for eps in (0.0, 0.0005, 0.002, 0.05, 1.0):
        x = solve_scaled(0.2, force=False, eaas=True, builder_kw={"max_offgrid_fraction": eps, "grid_service_mode": "annual"})
        if x is None:
            continue
        for t in x.T:
            pts.append((f"eps={eps},t={t}", con_status(x, x.m.grid_service_obligation[t])))
    nb = sum(1 for _, s in pts if s["binding"])
    if not pts or nb == 0 or nb == len(pts):
        return SKIP(f"grid did not straddle binding/slack ({nb}/{len(pts)})")
    bad, notes = _dual_check(pts)
    return PASS(f"{len(pts)} points, {nb} binding; no failures; degenerate (binding, zero dual): {len(notes)}") if not bad else FAIL("slack=>0; binding<=0", str(bad[:3]), "grid_service_obligation")


@inv("G1-1", "A", "No component is indexed beyond T_last; the only reference past T_last is the discount-factor extension under arrears (instalment windows reach T_last+1 only there)")
def g1_1():
    x = build_fx(window="advance", force=False)
    Tl = x.Tlast
    over = []
    for comp in x.m.component_objects([pyo.Var, pyo.Constraint, pyo.Expression, pyo.Param], descend_into=True):
        if not comp.is_indexed():
            continue
        for idx in comp.keys():
            ints = [i for i in (idx if isinstance(idx, tuple) else (idx,)) if isinstance(i, int) and not isinstance(i, bool)]
            # the first int is the year index for all T-indexed components; tier labels are strings
            if ints and max(ints[:2] if comp.name.startswith("convert_to_power") else ints[:1]) > Tl:
                over.append((comp.name, idx))
                break
    reach = {}
    for case in ("advance", "arrears", "due_crf"):
        y = build_fx(window=case, force=False)
        mx = 0
        for tr in ("conc", "comm"):
            for ch in CHANNELS:
                for t in y.T:
                    off = 1 if case == "arrears" else 0
                    L = leff(y.sc)[0 if tr == "conc" else 1][ch]
                    mx = max(mx, max(range(t + off, min(t + off + L, Tl + 1 + off))))
        reach[case] = mx
    if over:
        return FAIL("no component indexed past T_last", str(over[:4]), "src/optimize_model_sliced.py")
    if reach["advance"] <= Tl and reach["due_crf"] <= Tl and reach["arrears"] == Tl + 1:
        return PASS(f"T_last = {Tl}; no component index exceeds it; max window year index: advance {reach['advance']}, due_crf {reach['due_crf']}, arrears {reach['arrears']} (= T_last + 1, the extension)")
    return FAIL("advance/due_crf <= T_last; arrears == T_last+1", str(reach), "capital window")


@inv("G1-2", "A", "Every summary quantity the baseline runner writes is labelled with the year it is taken from; 2054 anchors also carry their 2045 value (TASK 53 2.4)")
def g1_2():
    t = src_of("scripts/01b_run_baseline_sliced.py")
    tree = ast.parse(t)
    keys, table_keys = [], set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Dict):
            ks = [(_const_str(k) if k is not None else None) for k in n.keys]
            ks = [k for k in ks if k]
            if "year" in ks:                       # a per-year table (its 'year' column labels every row)
                table_keys.update(ks)
            keys.extend(ks)
    scalar_q = [k for k in keys if k not in table_keys and any(w in k for w in ("final_", "total", "npv_", "cumulative", "emissions_tco2", "storage_capacity", "built_mw"))]
    labelled = [k for k in scalar_q if any(w in k for w in ("by_year", "report_window", "full_horizon", "2045", "2054", "window"))]
    unl = sorted(set(scalar_q) - set(labelled))
    # npv_total_cost_usd is one objective over the whole horizon: it keeps its name (~86 readers) and the runner must record the window
    windowed_ok = "npv_total_cost_window" in keys
    unl = [k for k in unl if not (k == "npv_total_cost_usd" and windowed_ok)]
    uses_helpers = "labelled_decision_totals" in t and "labelled_period_totals" in t
    # the helpers' output on a 2025-2054 horizon: every key names its year, and every last-year key has its 2045 twin
    from src.summary_years import labelled_decision_totals, labelled_period_totals
    yrs = list(range(2025, 2055))
    one = {y: 1.0 for y in yrs}
    produced = list(labelled_decision_totals(yrs, one, one, gas_add_by_year=one)) + list(labelled_period_totals(yrs, one, one))
    no_year = [k for k in produced if not any(w in k for w in ("2045", "2054"))]
    no_twin = [k for k in produced if "2054" in k and k.replace("2054", "2045") not in produced]
    if unl or no_year or no_twin or not uses_helpers or not windowed_ok:
        return FAIL("every aggregate/snapshot key names its year or window; a 2045 value beside every 2054 anchor",
                    f"unlabelled: {unl}; helpers used: {uses_helpers}; npv window recorded: {windowed_ok}; produced keys without a year: {no_year}; without a 2045 twin: {no_twin}",
                    "scripts/01b_run_baseline_sliced.py (summary/decision_variables dicts), src/summary_years.py")
    return PASS(f"all scalar aggregates labelled; {len(produced)} helper keys, each last-year key with a 2045 twin: {sorted(produced)}; the objective's window is recorded in `npv_total_cost_window` (no 2045 objective exists: it is one solve over the horizon); per-year table columns are labelled by their `year` column")


@inv("G1-3", "A", "Buffer invariance: extending T_last by 5 (and by 9) years leaves the plan to 2045 unchanged within 1% of the cumulative 2045 build (central arms, unforced; not quotable)")
def g1_3():
    def plan(end):
        x = build_fx(end_year=end, force=False, reduced_life=False, eaas=False)
        solve_fx(x)
        ys = x.years
        keep = [i for i, y in enumerate(ys) if y <= 2045]
        m = x.m
        return ([v(m.solar_public_add[i]) + v(m.solar_eaas_add[i]) for i in keep], [v(m.storage_add[i]) for i in keep], [v(m.gas_add[i]) for i in keep], ys)
    base = plan(2045)
    out, bad = [], []
    for end in (2050, 2054):
        ext = plan(end)
        for nm, a, b in (("solar", base[0], ext[0]), ("storage", base[1], ext[1]), ("gas", base[2], ext[2])):
            dmax = max(abs(p - q) for p, q in zip(a, b))
            cum = max(sum(a), 1.0)
            rel = sum(abs(p - q) for p, q in zip(a, b)) / cum
            last = max([i for i, (p, q) in enumerate(zip(a, b)) if abs(p - q) > 1.0] or [-1])
            out.append(f"end {end} {nm}: L1 diff {100*rel:.2f}% of cumulative to 2045, max annual diff {dmax:.1f}, last differing year {2025+last if last >= 0 else 'none'}")
            if rel > 0.01:
                bad.append((end, nm, rel))
    note = ("The specified test (extend the real 2054 horizon by five years) is not constructible: gas deliverability data stop at 2054 "
            "(`ValueError: Missing gas availability values ... 2055-2059`). This is the analogue on the same arms: base horizon 2045 extended by 5 (2050) and by 9 (2054) years. "
            "Plan differences at 2054 begin in the 2037-38 storage vintages, which are rebuilt (15-yr life) only if the horizon reaches 2052-53.")
    if not bad:
        return PASS("; ".join(out) + " | " + note)
    return FAIL("<= 1% L1 difference in the plan to 2045", f"{[(e, n, round(r, 4)) for e, n, r in bad]}", "src/optimize_model_sliced.py (horizon end / rebuild inside horizon)", detail="; ".join(out) + " | " + note)


@inv("G1-3s", "A", "Buffer invariance of STOCKS: the cumulative solar, gas and storage build to 2045 is unchanged when the horizon end is extended by 5 and 9 years, to 0.1 MW (MWh) absolute (central arms, unforced; companion to G1-3, which tests annual timing and is not weakened)")
def g1_3s():
    # TASK 79 B.2. G1-3 asks whether the ANNUAL additions to 2045 are horizon-invariant (L1 criterion 1%) and fails under vintage
    # dispatch by re-timing (task_log_78 D.4). This asks the different question the thesis also reports: are the cumulative STOCKS
    # at 2045 horizon-invariant. Tolerance 0.1 MW (MWh for storage), absolute: the observed differences are <= 2e-4 (CBC solver
    # noise: 1e-7 relative on stocks of 1e4), three orders inside the tolerance, and the tolerance is far below the smallest annual
    # re-timing the same solves show (25.5 MW), so it cannot hide timing as stock.
    TOL = 0.1
    def stocks(end):
        x = build_fx(end_year=end, force=False, reduced_life=False, eaas=False)
        solve_fx(x)
        keep = [i for i, y in enumerate(x.years) if y <= 2045]
        m = x.m
        return {"solar": sum(v(m.solar_public_add[i]) + v(m.solar_eaas_add[i]) for i in keep),
                "storage": sum(v(m.storage_add[i]) for i in keep),
                "gas": sum(v(m.gas_add[i]) for i in keep)}
    base = stocks(2045)
    out, bad = [], []
    for end in (2050, 2054):
        ext = stocks(end)
        for nm in ("solar", "storage", "gas"):
            d = abs(ext[nm] - base[nm])
            out.append(f"end {end} {nm}: stock {ext[nm]:.3f} against {base[nm]:.3f}, |diff| {d:.2e}")
            if d > TOL:
                bad.append((end, nm, d))
    note = "annual timing is tested by G1-3 (unchanged); this asserts only what the model identifies: the stock"
    if not bad:
        return PASS("; ".join(out) + f" | tolerance {TOL} MW (MWh) | " + note)
    return FAIL(f"|stock difference| <= {TOL}", f"{[(e, n, round(d, 3)) for e, n, d in bad]}", "src/optimize_model_sliced.py (horizon end)", detail="; ".join(out) + " | " + note)


@inv("G1-3m", "A", "Addendum A-1: test G1-3's mechanism (in-horizon storage rebuilds): first divergent horizon, rebuild-inside flags for 2036-2039 vintages, and the no-rebuild isolating run at 2054")
def g1_3m():
    def plan(end, storage_life=None):
        over = {"asset_lifetimes": {"solar": 30.0, "storage": float(storage_life), "gas": 30.0}} if storage_life else None
        x = build_fx(end_year=end, force=False, reduced_life=False, eaas=False, sc_over=over)
        solve_fx(x)
        k = [i for i, y in enumerate(x.years) if y <= 2045]
        mm = x.m
        return dict(solar=[v(mm.solar_public_add[i]) + v(mm.solar_eaas_add[i]) for i in k],
                    storage=[v(mm.storage_add[i]) for i in k], gas=[v(mm.gas_add[i]) for i in k]), x

    def diff(a, b):
        return {n: (max(abs(p - q) for p, q in zip(a[n], b[n])), sum(abs(p - q) for p, q in zip(a[n], b[n])) / max(sum(a[n]), 1.0)) for n in a}

    base, bx = plan(2045)
    life = int(round(float(bx.sc["asset_lifetimes"]["storage"])))
    vint = {y: base["storage"][bx.years.index(y)] for y in (2036, 2037, 2038, 2039)}
    rows, first = [], None
    for H in (2050, 2051, 2052, 2053, 2054):
        p, _ = plan(H)
        d = diff(base, p)
        inside = {y: (y + life <= H) for y in vint}
        diverges = any(a > 1e-6 for a, _ in d.values())
        if diverges and first is None:
            first = H
        rows.append((H, inside, {n: (round(a, 2), round(100 * b, 3)) for n, (a, b) in d.items()}, diverges))
    b30, _ = plan(2045, 30)
    p30, _ = plan(2054, 30)
    iso = diff(b30, p30)
    ctl = diff(base, b30)
    iso_ok = all(a <= 1e-6 for a, _ in iso.values())
    ctl_ok = all(a <= 1e-6 for a, _ in ctl.values())
    first_rebuild_vol = min((y + life) for y, vol in vint.items() if vol > 0)
    desc = "; ".join(f"horizon {H}: rebuilds inside for vintages {[y for y, f in ins.items() if f]} (volumes built {{{', '.join(f'{y}: {vint[y]:.1f}' for y, f in ins.items() if f)}}}), plan to 2045 diverges: {dv}, max abs diff {dd}" for H, ins, dd, dv in rows)
    none_inside_no_div = all((not dv) for H, ins, dd, dv in rows if not any(vint[y] > 0 and f for y, f in ins.items()))
    inside_at_first = any(vint[y] > 0 and f for y, f in dict((H, ins) for H, ins, _, _ in rows)[first].items()) if first else False
    verdict = ("CONFIRMED" if (first is not None and inside_at_first and none_inside_no_div and iso_ok and ctl_ok)
               else ("REFUTED" if (not iso_ok) else "UNDETERMINED"))
    detail = (f"VERDICT: {verdict}. Storage baseline volumes (2045-horizon plan): {{{', '.join(f'{y}: {vol:.1f} MWh' for y, vol in vint.items())}}}, life {life} yr. First divergent horizon: {first} "
              f"(earliest in-horizon rebuild of a vintage that was actually built: {first_rebuild_vol}, which at horizon {first_rebuild_vol} is a single final-year instalment and did not by itself move the plan; the plan first moves one horizon later, when the 2037 vintage, the largest, is rebuilt). {desc}. ISOLATION: horizon 2054 with storage life 30 (no storage rebuild inside the horizon) vs 2045 horizon with life 30: "
              f"max abs plan diff {max(a for a, _ in iso.values()):.2e} (solar/storage/gas) -> plan to 2045 {'returns exactly to' if iso_ok else 'does NOT return to'} the 2045-horizon baseline. "
              f"CONTROL: life 30 vs life 15 at the 2045 horizon (changes L_eff for storage, no rebuild in either): max diff {max(a for a, _ in ctl.values()):.2e}, so the L_eff change is not what removes the divergence. "
              "Configuration: central arms, unforced, REAL lifetimes/tenors, traditional regime, capital_case unconstrained; the 2054 run is the anchor-scale configuration (see report).")
    r = R({"CONFIRMED": "PASS", "REFUTED": "FAIL", "UNDETERMINED": "SKIP"}[verdict], detail)
    if verdict != "CONFIRMED":
        r.expected, r.observed, r.where = "divergence first at the earliest in-horizon rebuild; none without rebuilds", detail[:300], "optimize_model_sliced.py::_rebuild_years"
    return r


# ===========================================================================
# STAGE B -- Tier 2 (items 7-12, G4, G9, G8)
# ===========================================================================
def slices_of(x):
    return list(x.m.S)


@inv("D1", "B", "Credited firm capacity recomputed from the solution >= peak x omega x (1 + rho_t) every t, at the reported credits")
def d1():
    x = S_base()
    m, sc = x.m, x.sc
    cc = sc["capacity_credit"]
    om = float(sc.get("within_block_peaking_factor", 1.3448))
    rho = sc["reserve_margin"]
    rho = [float(a) for a in rho] if isinstance(rho, (list, tuple)) else [float(rho)] * len(x.T)
    alpha = float(sc["gas_availability_factor"])
    tiers = sc["conversion_tiers"]
    upl = float(sc["conversion_uplift"])
    worst, bind = 1e30, 0
    for t in x.T:
        lhs = (cc["solar"] * v(m.solar_capacity_mw[t]) + cc["gas"] * v(m.gas_capacity_mw[t])
               + cc["storage"] * v(m.storage_power_mw[t]) + cc["hydro"] * float(sc["hydro_capacity_mw_by_year"][t]))
        if hasattr(m, "convert_mw"):
            lhs += sum(cc["gas"] * tiers[k]["paf"] / alpha * upl * v(m.convert_mw[k]) for k in m.convert_mw)
        rhs = float(sc["peak_mw"][t]) * om * (1 + rho[t])
        worst = min(worst, lhs - rhs)
        bind += (lhs - rhs) <= 1e-3 * rhs
    credits = {k: round(float(a), 4) for k, a in cc.items()}
    if worst >= -1e-6:
        return PASS(f"min (credited - required) = {worst:.2f} MW over {len(x.T)} years ({bind} binding); credits {credits}; omega {om} (builder fallback, L50); rho_t {rho[0]:.4f}..{rho[-1]:.4f}")
    return FAIL(">= 0", f"{worst}", "src/optimize_model_sliced.py::firm_capacity_constraint")


@inv("D2", "B", "REGRESSION GUARD (TASK 50 V5): a converted OCGT megawatt enters the firm-capacity LHS once at nameplate (inside gross K^g) and its uplift once (convert_mw coefficient = credit x PAF / alpha x uplift)")
def d2():
    x = build_fx(force=False)
    m, sc = x.m, x.sc
    cc, alpha, upl, tiers = sc["capacity_credit"], float(sc["gas_availability_factor"]), float(sc["conversion_uplift"]), sc["conversion_tiers"]
    bad, rows = [], []
    for t in (0, x.Tlast):
        cm, _ = _coef_map(m.firm_capacity_constraint[t].body)
        for k in m.convert_mw:
            got = cm.get(id(m.convert_mw[k]), 0.0)
            exp = cc["gas"] * tiers[k]["paf"] / alpha * upl
            rows.append(f"t={t} tier {k}: coef {got:.6f} vs {exp:.6f}")
            if abs(got - exp) > 1e-12:
                bad.append((t, k, got, exp))
        gk = [cm.get(id(m.gas_add[b]), 0.0) for b in range(t + 1)]
        if any(abs(g - cc["gas"]) > 1e-12 for g in gk):
            bad.append((t, "gas_add coefficient != gas credit", gk))
    return PASS("; ".join(rows[:3]) + f"; gas_add coefficient = credit {cc['gas']}: converted nameplate sits inside the gross fleet K^g (counted once) and the uplift once") if not bad else \
        FAIL("convert_mw coef = credit x PAF/alpha x uplift (uplift only)", str(bad[:3]), "src/optimize_model_sliced.py::firm_capacity_constraint")


@inv("D3", "B", "Slice weights sum to 1; slice hours sum to 8,760; derived block PAR equals the peak period factor (section 3.2.5's identity)")
def d3():
    from src.optimize_model_sliced import slice_hours, _normalised_demand_weights
    from src import scenarios as S
    H = slice_hours()
    sh = sum(H.values())
    x = build_fx(force=False)
    w = _normalised_demand_weights({"season_demand_factor": x.sc["season_demand_factor"], "period_demand_factor": x.sc["period_demand_factor"]})
    sw = sum(w.values())
    # independent weights: hours x season factor x period factor, normalised
    raw = {sp: H[sp] * x.sc["season_demand_factor"][sp[0]] * x.sc["period_demand_factor"][sp[1]] for sp in H}
    tot = sum(raw.values())
    wdev = max(abs(w[sp] - raw[sp] / tot) for sp in H)
    fpk = float(x.sc["period_demand_factor"]["peak"])
    par = float(x.sc["par_block"])
    # hours-weighted means of the factors
    hs = {s: sum(H[(s, p)] for p in ("night", "day", "peak")) for s in ("dry", "wet")}
    hp = {p: sum(H[(s, p)] for s in ("dry", "wet")) for p in ("night", "day", "peak")}
    mean_season = sum(hs[s] * x.sc["season_demand_factor"][s] for s in hs) / sh
    mean_period = sum(hp[p] * x.sc["period_demand_factor"][p] for p in hp) / sh
    msgs = f"sum weights {sw:.15f}; sum hours {sh}; weights vs independent max dev {wdev:.1e}; hours-weighted mean season factor {mean_season:.6f}, period factor {mean_period:.6f}; peak period factor {fpk:.6f}; par_block {par:.6f}"
    bad = []
    if abs(sw - 1.0) > 1e-12: bad.append("weights do not sum to 1")
    if sh != 8760: bad.append("hours != 8760")
    if wdev > 1e-12: bad.append("weights differ from independent")
    if abs(par - fpk) > 1e-9: bad.append(f"par_block {par} != peak factor {fpk}")
    if bad:
        return FAIL("weights 1, hours 8760, PAR == f_peak (to 1e-9)", f"{bad}", "src/scenarios.py::load_scenario par_block / src/optimize_model_sliced.py::_normalised_demand_weights",
                    detail=msgs + "; chapter claim (section 3.2.5): when both factor sets are normalised to an hours-weighted mean of one the season factors cancel and PAR_block equals f_peak")
    return PASS(msgs)


def _storage_check(x):
    m = x.m
    eta = float(x.sc.get("storage_round_trip_eff", 0.90))
    days, hrs = {"dry": 181, "wet": 184}, {"night": 7, "day": 13, "peak": 4}
    worst_bal, worst_pw, active, n = -1e30, -1e30, 0, 0
    for t in x.T:
        for season in ("dry", "wet"):
            ch = sum(v(m.storage_charge[t, sp]) for sp in m.S if sp[0] == season)
            dis = sum(v(m.storage_discharge[t, sp]) for sp in m.S if sp[0] == season)
            worst_bal = max(worst_bal, dis - eta * ch)
            active += ch > 1e-9
            n += 1
        P = v(m.storage_power_mw[t])
        for sp in m.S:
            Hs = days[sp[0]] * hrs[sp[1]]
            lim = P * Hs / 1e6
            worst_pw = max(worst_pw, v(m.storage_charge[t, sp]) - lim, v(m.storage_discharge[t, sp]) - lim)
    return worst_bal, worst_pw, active, n


@inv("D4", "B", "Storage: the seasonal balance holds, no variable links seasons (no inter-seasonal transfer), power limits respected at the solution (on a fixture where storage actually cycles)")
def d4():
    x = solved(("Scycle",), eaas=True, force=True, force_lb={"solar_public": 3000.0, "storage": 3000.0})
    wb, wp, active, n = _storage_check(x)
    link = []
    for t in (0, x.Tlast):
        for season in ("dry", "wet"):
            cm, _ = _coef_map(x.m.storage_season_balance[t, season].body)
            other = "wet" if season == "dry" else "dry"
            foreign = {id(x.m.storage_charge[t, sp]) for sp in x.m.S if sp[0] == other} | {id(x.m.storage_discharge[t, sp]) for sp in x.m.S if sp[0] == other}
            if [k for k in cm if k in foreign]:
                link.append((t, season))
    tot_dis = sum(v(x.m.storage_discharge[t, sp]) for t in x.T for sp in x.m.S)
    if active == 0:
        return SKIP("storage never charges on this fixture, the balance would be vacuous")
    if wb <= 1e-6 and wp <= 1e-6 and not link:
        return PASS(f"fixture: solar >= 3000 MW/yr and storage >= 3000 MWh/yr forced; {active} of {n} (year, season) pairs cycle ({tot_dis:.1f} TWh discharged); max (discharge - eta x charge) = {wb:.2e} TWh (binding where cycling); max power-limit excess {wp:.2e} TWh; no balance term crosses seasons. NOTE: the balance is an inequality per season with no state of charge (no within-season ordering of charge before discharge)")
    return FAIL("<= 1e-6 and no cross-season term", f"balance {wb}, power {wp}, links {link}", "src/optimize_model_sliced.py::storage_season_balance / storage_*_power_limit")


@inv("D5", "B", "Hydro: the seasonal budget holds as an equality (sums to the annual energy series); minimum flow respected every slice; power limit respected")
def d5():
    x = S_base()
    m, sc = x.m, x.sc
    days, hrs = {"dry": 181, "wet": 184}, {"night": 7, "day": 13, "peak": 4}
    Hs = {(s, p): days[s] * hrs[p] for s in days for p in hrs}
    seas_h = {s: sum(Hs[(s, p)] for p in hrs) for s in days}
    mu = float(sc.get("hydro_min_flow_fraction", 0.5))
    wb, wm, wp, wsum = 0.0, 1e30, -1e30, 0.0
    for t in x.T:
        E = float(sc["hydro_energy_twh_by_year"][t])
        tot = 0.0
        for season in days:
            gen = sum(v(m.hydro_gen[t, (season, p)]) for p in hrs)
            bud = float(pyo.value(m.hydro_season_budget[t, season].upper))
            wb = max(wb, abs(gen - bud))
            tot += gen
            for p in hrs:
                g = v(m.hydro_gen[t, (season, p)])
                wm = min(wm, g - mu * bud * Hs[(season, p)] / seas_h[season])
                wp = max(wp, g - float(sc["hydro_capacity_mw_by_year"][t]) * Hs[(season, p)] / 1e6)
        wsum = max(wsum, abs(tot - E))
    if wb <= 1e-6 and wm >= -1e-6 and wp <= 1e-6 and wsum <= 1e-6:
        return PASS(f"budget equality max dev {wb:.1e} TWh; seasons sum to the annual series (max dev {wsum:.1e}); min over slices of (gen - floor) = {wm:.2e}; max power-limit excess {wp:.1e}; mu = {mu}")
    return FAIL("equality, floor, limit", f"budget {wb}, annual {wsum}, floor {wm}, power {wp}", "src/optimize_model_sliced.py::hydro_season_budget / hydro_min_flow")


@inv("D6", "B", "Solar generation is zero in every peak (and night) slice; positive only in day slices")
def d6():
    x = S_base()
    m = x.m
    peak = max(abs(v(m.solar_generation[t, sp])) for t in x.T for sp in m.S if sp[1] == "peak")
    night = max(abs(v(m.solar_generation[t, sp])) for t in x.T for sp in m.S if sp[1] == "night")
    day = min(v(m.solar_generation[t, sp]) for t in x.T for sp in m.S if sp[1] == "day")
    return PASS(f"max |peak| {peak:.1e}, max |night| {night:.1e}, min day {day:.4f} TWh") if peak == 0.0 and night == 0.0 and day > 0 else \
        FAIL("peak 0, night 0, day > 0", f"peak {peak}, night {night}, day {day}", "src/optimize_model_sliced.py::solar_generation / SOLAR_PERIODS")


DOC_GENSET_EF = {"genset_srmc": 0.729, "genset_central": 0.902, "genset_high": 4.638}


@inv("D7", "B", "Self-generation emission factors: for each genset arm, implied tCO2/MWh_e = EF_th / eta recomputed and compared with the model's objective-side coefficient and with the declared values (0.729 / 0.902 / 4.638)")
def d7():
    import csv
    rows = {r["Scenario"]: r for r in csv.DictReader(open(ROOT / "data/cost/processed/genset_cost.csv"))}
    out, bad = [], []
    dens = {}
    for arm, decl in DOC_GENSET_EF.items():
        r = rows[arm]
        eta, efth = float(r["genset_eta_hhv"]), float(r["ef_th_hhv_tco2_per_mwh_th"])
        implied = efth / eta
        e2 = load_econ("voll_mid", genset_case=arm)
        x = build_fx(force=False, econ_over={k: e2[k] for k in ("GENSET_EF_TH_HHV", "GENSET_ETA_HHV", "GENSET_COST_PER_TWH_E") if k in e2})
        cm, _ = _coef_map(x.m.emissions_selfgen_by_year[0])
        sp = list(x.m.S)[0]
        model = cm.get(id(x.m.selfgen[0, sp]), 0.0) / 1e6          # tCO2 per MWh_e
        sfc = float(r["sfc_l_per_kwh"])
        dens[arm] = 3.6 / (eta * sfc)                              # implied HHV MJ per litre from eta and specific fuel consumption
        ok = abs(model - implied) < 1e-9 and abs(round(implied, 3) - decl) < 1e-9
        out.append(f"{arm}: EF_th {efth} / eta {eta} = {implied:.4f} tCO2/MWh_e; model coefficient {model:.4f}; declared {decl}; implied HHV {dens[arm]:.1f} MJ/L")
        if not ok:
            bad.append((arm, implied, model, decl))
    agree_ago = abs(dens["genset_srmc"] - dens["genset_central"]) / dens["genset_srmc"]
    if agree_ago > 0.01:
        bad.append(("AGO arms imply different fuel energy density", dens["genset_srmc"], dens["genset_central"]))
    tail = (f"; the two diesel (AGO) arms imply the same fuel energy density within {100*agree_ago:.2f}%; genset_high implies {dens['genset_high']:.1f} MJ/L (gasoline). "
            "genset_high's 4.638 tCO2/MWh_e follows from a 5.11% observed efficiency (A2EI metered); it is declared, internally consistent, and about 5x the diesel arms")
    return PASS("; ".join(out) + tail) if not bad else FAIL("model == EF/eta == declared", str(bad), "data/cost/processed/genset_cost.csv / src/io.py::load_econ", detail="; ".join(out))


@inv("G4-1", "B", "solar cumulative cap binds as stated (additions <= cap every t; equality somewhere when it binds); cap set to zero => no solar build")
def g4_1():
    free = try_solve(force=False, eaas=True, reduced_life=False)
    if free is None:
        return FAIL("feasible", "infeasible", "solar_cumulative_cap")
    fl = [v(free.m.solar_public_add[t]) + v(free.m.solar_eaas_add[t]) for t in free.T]
    cumfree = [sum(fl[:t + 1]) for t in free.T]
    cap = [0.7 * c for c in cumfree]
    x = try_solve(force=False, eaas=True, reduced_life=False, sc_over={"solar_cumulative_cap_mw": cap})
    z = try_solve(force=False, eaas=True, reduced_life=False, sc_over={"solar_cumulative_cap_mw": [0.0] * len(free.T)})
    n_ = try_solve(force=False, eaas=True, solar_build_case="nirp_trajectory", reduced_life=False)
    if x is None or z is None or n_ is None:
        return FAIL("feasible", "infeasible", "solar_cumulative_cap")
    cum, worst, bound = 0.0, -1e30, 0
    for t in x.T:
        cum += v(x.m.solar_public_add[t]) + v(x.m.solar_eaas_add[t])
        worst = max(worst, cum - cap[t])
        bound += (cap[t] > 0 and abs(cum - cap[t]) <= 1e-6 * max(1.0, cap[t]))
    zsol = sum(v(z.m.solar_public_add[t]) + v(z.m.solar_eaas_add[t]) for t in z.T)
    ncap = n_.sc["solar_cumulative_cap_mw"]
    ncum = 0.0
    nworst = -1e30
    for t in n_.T:
        ncum += v(n_.m.solar_public_add[t]) + v(n_.m.solar_eaas_add[t])
        nworst = max(nworst, ncum - ncap[t])
    ok = worst <= 1e-6 and bound >= 1 and zsol <= 1e-9 and nworst <= 1e-6 and obj(x) >= obj(free) - 1e-6
    note = (f"cap = 0.7 x the free cumulative plan: max (cumulative - cap) {worst:.1e} MW, equality in {bound} of {len(x.T)} years, objective {obj(x):.0f} >= free {obj(free):.0f}; "
            f"zero cap: solar build {zsol:.1e} MW; the NIRP-trajectory arm's own cap is respected (max excess {nworst:.1e}) but does not bind on a 12-year horizon (cap above the free plan in every year)")
    return PASS(note) if ok else FAIL("additions <= cap; equality when binding; zero cap => zero build", note, "src/optimize_model_sliced.py::solar_cumulative_cap")


@inv("G4-2", "B", "The min-build floor forces build when it binds, and is distinguishable from the cumulative cap (different constraint, per-year not cumulative)")
def g4_2():
    base = try_solve(force=False, eaas=True)
    fl = try_solve(force=False, eaas=True, sc_over={"solar_min_build_mw_per_year": 100.0})
    if base is None or fl is None:
        return FAIL("feasible", "infeasible", "solar_min_build_constraint")
    b = [v(base.m.solar_public_add[t]) + v(base.m.solar_eaas_add[t]) for t in base.T]
    f = [v(fl.m.solar_public_add[t]) + v(fl.m.solar_eaas_add[t]) for t in fl.T]
    forced = [t for t in base.T if b[t] < 100 - 1e-6]
    ok_floor = min(f) >= 100 - 1e-6 and len(forced) > 0 and obj(fl) >= obj(base) - 1e-6
    names = (hasattr(fl.m, "solar_min_build_constraint"), hasattr(base.m, "solar_min_build_constraint"))
    # distinguishable: floor 100/yr with a cumulative cap of 50 MW is infeasible; the cap alone leaves annual flexibility
    cap_floor = try_solve(force=False, eaas=True, sc_over={"solar_min_build_mw_per_year": 100.0, "solar_cumulative_cap_mw": [50.0] * len(base.T)})
    cap_only = try_solve(force=False, eaas=True, sc_over={"solar_cumulative_cap_mw": [50.0] * len(base.T)})
    dist = cap_floor is None and cap_only is not None
    ok = ok_floor and names == (True, False) and dist
    note = (f"floor 100 MW/yr: min build {min(f):.1f}, binds in {len(forced)} years where the free plan builds less; objective {obj(fl):.0f} >= {obj(base):.0f}; constraint present only with the floor {names}; "
            f"floor + cumulative cap 50 MW: {'infeasible' if cap_floor is None else 'feasible'}; cap alone: {'feasible' if cap_only is not None else 'infeasible'}")
    return PASS(note) if ok else FAIL("floor forces build; distinct from cap", note, "src/optimize_model_sliced.py::solar_min_build_constraint / solar_cumulative_cap")


@inv("G4-3", "B", "Land constraint: units close (km2/MW x MW, km2/MWh x MWh against km2), the solution respects it, and an artificially tightened allowance binds it")
def g4_3():
    free = try_solve(force=False, eaas=True)
    if free is None:
        return FAIL("feasible", "infeasible", "land_constraint")
    sc = free.sc
    ls, lg, lst, A = (float(sc["land_intensity_solar_km2_per_mw"]), float(sc["land_intensity_gas_km2_per_mw"]),
                      float(sc["land_intensity_storage_km2_per_mwh"]), float(sc["land_available_km2"]))

    def land(x, t):
        return v(x.m.solar_capacity_mw[t]) * ls + v(x.m.gas_capacity_mw[t]) * lg + v(x.m.storage_capacity_mwh[t]) * lst
    u = max(land(free, t) for t in free.T)
    anchor_land = 39030.0 * ls + 7221.0 * lg + 63056.0 * lst
    A2 = 0.6 * u
    tight = try_solve(force=False, eaas=True, sc_over={"land_available_km2": A2})
    plaus = (0.005 <= ls <= 0.1) and (0.0005 <= lg <= 0.05) and (1e-6 <= lst <= 1e-2)
    if tight is None:
        return FAIL("feasible with allowance = 0.6 x free peak use", "infeasible", "land_constraint")
    tl = max(land(tight, t) for t in tight.T)
    dual = min(float(tight.m.dual.get(tight.m.land_constraint[t], 0.0) or 0.0) for t in tight.T)
    note = (f"solar {ls} km2/MW, gas {lg} km2/MW, storage {lst} km2/MWh, allowance {A:.0f} km2; free plan peak use {u:.1f} km2 (<= {A:.0f}); arithmetic anchor plan (39,030 MW, 7,221 MW, 63,056 MWh) = {anchor_land:.0f} km2 = {100*anchor_land/A:.2f}% of the allowance (the constraint cannot bind at the anchor); "
            f"allowance cut to {A2:.1f} km2: peak use {tl:.1f} km2 (binding: {abs(tl-A2) <= 1e-6*A2}), objective {obj(tight):.0f} >= {obj(free):.0f}, most negative land dual {dual:.3g}")
    ok = u <= A + 1e-6 and plaus and tl <= A2 * (1 + 1e-6) and abs(tl - A2) <= 1e-6 * A2 and obj(tight) >= obj(free) - 1e-6 and dual < 0
    return PASS(note) if ok else FAIL("respected, plausible magnitudes, binds when tightened", note, "src/optimize_model_sliced.py::land_constraint")


@inv("G9-1", "B", "Every exogenous series on the 30-year scenario: correct length, finite, within a stated plausible range")
def g9_1():
    sc = make_scenario(end_year=2054)
    n = len(sc["years"])
    capex = load_solar_capex_by_year(scenario_name="solar_low", start_year=2025, end_year=2054)
    series = {
        "solar_capex_usd_per_mw (io.load_solar_capex_by_year)": ([capex[y] for y in range(2025, 2055)], 3e5, 2e6),
        "peak_mw": (sc["peak_mw"], 2e3, 4e4),
        "hydro_energy_twh_by_year": (sc["hydro_energy_twh_by_year"], 5, 40),
        "hydro_capacity_mw_by_year": (sc["hydro_capacity_mw_by_year"], 1e3, 2.2e4),
        "gas_baseline_retirement_series_mw": (sc["gas_baseline_retirement_series_mw"], 0, 10947),
        "reserve_margin (per year)": (sc["reserve_margin"], 0.0, 0.5),
        "solar_cumulative_cap_mw": (sc.get("solar_cumulative_cap_mw") or [0.0] * n, 0, 2e5),
    }
    from src.gas_supply import gas_available_power
    series["gas_available_twh_th"] = (list(gas_available_power(2025, 2054, "baseline")["available_twh_th"]), 10, 300)
    for tech in ("solar_utility", "storage", "gas_ct", "gas_cc"):
        series[f"om fom {tech} (USD/kW-yr)"] = ([sc["om_cost_by_year"][tech][y]["fom_usd_per_kw_yr"] for y in range(2025, 2055)], 0, 300)
        series[f"om vom {tech} (USD/MWh)"] = ([sc["om_cost_by_year"][tech][y]["vom_usd_per_mwh"] for y in range(2025, 2055)], 0, 50)
    bad, rows = [], []
    for name, (vals, lo, hi) in series.items():
        arr = [float(a) for a in vals]
        okk = len(arr) == n and all(math.isfinite(a) for a in arr) and min(arr) >= lo and max(arr) <= hi
        rows.append(f"{name}: n={len(arr)} [{min(arr):.4g}, {max(arr):.4g}] in [{lo:g}, {hi:g}]")
        if not okk:
            bad.append((name, len(arr), min(arr), max(arr), lo, hi))
    r = PASS(f"{len(series)} series, length {n}, finite, in range") if not bad else FAIL("length, finite, range", str(bad), "src/scenarios.py / src/io.py series")
    r.extra["rows"] = rows
    return r


@inv("G9-2", "B", "Series claimed monotone are monotone, series claimed flat are flat: solar capex declines, storage capex declines as the source series does and the model reads it (TASK 53 2.2; was the assertion form of TASK 50 V8), hydro_flat is flat; section 3.7.3's text vs the model")
def g9_2():
    capex = load_solar_capex_by_year(scenario_name="solar_low", start_year=2025, end_year=2054)
    sv = [capex[y] for y in range(2025, 2055)]
    solar_dec = all(b <= a + 1e-9 for a, b in zip(sv, sv[1:])) and sv[-1] < 0.5 * sv[0]
    x = build_fx(force=False)
    unit = [abs(_coef_map(x.m.storage_split_identity[t].body)[0].get(id(x.m.storage_add[t]), 0.0)) for t in x.T]
    expected = storage_price_vec(x.sc, x.econ)
    dev = max(abs(a - b) / b for a, b in zip(unit, expected))
    storage_dec = all(b <= a + 1e-9 for a, b in zip(unit, unit[1:])) and unit[-1] < 0.75 * unit[0]
    sc = make_scenario(end_year=2054)
    hydro_flat = max(sc["hydro_energy_twh_by_year"]) == min(sc["hydro_energy_twh_by_year"]) and max(sc["hydro_capacity_mw_by_year"]) == min(sc["hydro_capacity_mw_by_year"])
    txt = src_of("docs/chapters/CHAPTER_3_assembled.md")
    claim_decl = "Solar and storage capital costs\nare read as per-year series which decline" in txt or "Solar and storage capital costs are read as per-year series" in txt.replace("\n", " ")
    note = (f"solar capex non-increasing, {sv[0]:.0f} -> {sv[-1]:.0f} USD/MW ({100*(sv[-1]/sv[0]-1):.0f}%): {solar_dec}; storage unit cost in the model (storage_split_identity coefficient) {unit[0]:.0f} -> {unit[-1]:.0f} USD/MWh, "
            f"non-increasing {storage_dec}, equals the independently read source series to {dev:.1e} (storage_capex_case = {x.sc['storage_capex_case']}); hydro_flat series flat: {hydro_flat}; section 3.7.3 asserts the declines: {claim_decl}")
    if solar_dec and storage_dec and dev <= 1e-12 and hydro_flat and claim_decl:
        return PASS(note)
    return FAIL("both solar AND storage decline in the model, as section 3.7.3 states", note, "src/optimize_model_sliced.py::storage_capex_param / docs/chapters/CHAPTER_3_assembled.md")


_G8_CACHE = {}


def _diag(x):
    from src.optimize_experiments import extract_planning_diagnostics
    key = id(x)
    if key not in _G8_CACHE:
        with contextlib.redirect_stdout(io.StringIO()):
            _G8_CACHE[key] = extract_planning_diagnostics(x.m, x.sc, x.econ, x.capex)
    return _G8_CACHE[key]


@inv("G8-1", "B", "Each derived diagnostic extract_planning_diagnostics reports is recomputed from the primal and dual solution and compared (annual-by-year series and scalars)")
def g8_1():
    x = solved(("Smod",), eaas=True, capital_case="moderate", force=True)
    d = _diag(x)
    m, sc, econ = x.m, x.sc, x.econ
    ys = x.years
    rs = float(sc["social_discount_rate"])
    S = list(m.S)
    checks = {}

    def series(name, exp):
        got = d[name]
        dev = max(abs(float(got[y]) - e) / max(1.0, abs(e)) for y, e in zip(ys, exp) if got[y] is not None)
        checks[name] = dev

    series("solar_capacity_mw_by_year", [float(sc["solar_baseline_mw"]) + sum(v(m.solar_public_add[k]) + v(m.solar_eaas_add[k]) for k in range(t + 1)) for t in x.T])
    series("solar_generation_twh_e_by_year", [sum(v(m.solar_generation[t, sp]) for sp in S) for t in x.T])
    series("selfgen_twh_by_year", [sum(v(m.selfgen[t, sp]) for sp in S) for t in x.T])
    series("emissions_tco2_by_year", [v(m.emissions_by_year[t]) for t in x.T])
    series("unserved_twh_by_year", [sum(v(m.unserved[t, sp]) for sp in S) for t in x.T])
    series("fleet_gas_to_power_twh_th_by_year", [sum(v(m.gas_to_power[t, sp]) for sp in S) for t in x.T])
    series("storage_discharge_twh_e_by_year", [sum(v(m.storage_discharge[t, sp]) for sp in S) for t in x.T])
    series("discount_factor_by_year", [df(t, rs) for t in x.T])
    series("demand_twh_by_year", [sum(float(pyo.value(m.balance[t, sp].lower)) for sp in S) for t in x.T])
    series("gas_shadow_pv_usd_per_twh_th_by_year", [max(0.0, -float(m.dual.get(m.gas_balance[t], 0.0) or 0.0)) for t in x.T])
    series("gas_shadow_yr_usd_per_twh_th_by_year", [max(0.0, -float(m.dual.get(m.gas_balance[t], 0.0) or 0.0)) / df(t, rs) for t in x.T])
    ncap = lambda t: float(pyo.value(m.solar_capex_param[t]))
    pub = sum(df(t, rs) * v(m.solar_public_add[t]) * ncap(t) for t in x.T)
    sub = sum(df(t, rs) * v(m.eaas_subsidy[t]) for t in x.T)
    spv = storage_price_vec(x.sc, econ)
    spend = sum(df(t, rs) * (v(m.solar_public_add[t]) * ncap(t) + v(m.storage_add[t]) * spv[t] + v(m.eaas_subsidy[t])) for t in x.T)
    budget = float(sc["public_solar_budget_npv"])
    sc_ = {"ndc_finance_public_solar_npv_usd": pub, "ndc_finance_eaas_subsidy_npv_usd": sub, "public_budget_realised_spend_pv_usd": spend,
           "public_budget_realised_spend_undiscounted_usd": sum(v(m.solar_public_add[t]) * ncap(t) + v(m.storage_add[t]) * spv[t] + v(m.eaas_subsidy[t]) for t in x.T),
           "public_budget_utilisation": spend / budget,
           "storage_final_capacity_mwh": v(m.storage_capacity_mwh[x.Tlast]),
           "storage_throughput_twh": sum(v(m.storage_discharge[t, sp]) for t in x.T for sp in S)}
    for k, e in sc_.items():
        got = d.get(k)
        checks[k] = None if got is None else abs(float(got) - e) / max(1.0, abs(e))
    cdz = d["cost_decomposition"]
    terms, _ = independent_terms(x)
    for k, e in (("cost_total_npv", obj(x)), ("om_npv", terms["O&M"]), ("selfgen_npv", terms["self-generation"])):
        checks["cost_decomposition." + k] = abs(float(cdz[k]) - e) / max(1.0, abs(e))
    # concessional share by vintage
    vols = _volumes(x)
    cs = []
    for t in x.T:
        conc = sum(vols[n][2][t] for n in ("solar_public", "solar_eaas", "storage"))
        elig = sum(vols[n][2][t] + vols[n][3][t] for n in ("solar_public", "solar_eaas", "storage"))
        cs.append(conc / elig if elig > 0 else 0.0)
    series("concessional_share_by_vintage", cs)
    bad = {k: dv for k, dv in checks.items() if dv is not None and dv > 1e-6}
    skipped = [k for k, dv in checks.items() if dv is None]
    if not bad:
        return PASS(f"{len(checks) - len(skipped)} diagnostics recomputed independently, max rel dev {max(dv for dv in checks.values() if dv is not None):.1e}" + (f"; reported None: {skipped}" if skipped else ""))
    return FAIL("rel dev <= 1e-6", str({k: f"{dv:.3g}" for k, dv in bad.items()}), "src/optimize_experiments.py::extract_planning_diagnostics",
                detail="; ".join(f"{k}: {dv:.3g}" for k, dv in checks.items() if dv is not None))


@inv("G8-1b", "B", "Reliability fields (TASK 53 2.6): unserved_only_* recomputed from the solution; the canonical grid_service_reliability_* equals 1 - (unserved + self-generation)/demand and agrees with the model's own grid-service constraint; the two are NOT made to agree")
def g8_1b():
    x = solve_scaled(0.2, force=False, eaas=True, builder_kw={"max_offgrid_fraction": 0.5, "grid_service_mode": "annual"})
    d = _diag(x)
    m = x.m
    ys, S = x.years, list(m.S)
    D = [sum(float(pyo.value(m.balance[t, sp].lower)) for sp in S) for t in x.T]
    U = [sum(v(m.unserved[t, sp]) for sp in S) for t in x.T]
    G = [sum(v(m.selfgen[t, sp]) for sp in S) for t in x.T]
    rel_u = [1 - u / dd for u, dd in zip(U, D)]
    rel_g = [1 - (u + g) / dd for u, g, dd in zip(U, G, D)]
    dev = {
        "unserved_only_annual_reliability_by_year": max(abs(d["unserved_only_annual_reliability_by_year"][y] - r) for y, r in zip(ys, rel_u)),
        "unserved_only_worst_year_reliability": abs(d["unserved_only_worst_year_reliability"] - min(rel_u)),
        "unserved_only_reliability": abs(d["unserved_only_reliability"] - (1 - sum(U) / sum(D))),
        "unserved_only_expected_unserved_energy_twh": abs(d["unserved_only_expected_unserved_energy_twh"] - sum(U)),
        "grid_service_reliability_by_year": max(abs(d["grid_service_reliability_by_year"][y] - r) for y, r in zip(ys, rel_g)),
        "grid_service_reliability_worst_year": abs(d["grid_service_reliability_worst_year"] - min(rel_g)),
        "grid_service_reliability_horizon": abs(d["grid_service_reliability_horizon"] - (1 - (sum(U) + sum(G)) / sum(D))),
        "grid_shortfall_total_twh": abs(d["grid_shortfall_total_twh"] - (sum(U) + sum(G))),
    }
    # against the model's own constraint: the shortfall it bounds is selfgen + unserved, year by year
    con_ok = True
    if hasattr(m, "grid_service_obligation"):
        con_ok = all(abs(float(pyo.value(m.grid_service_obligation[t].body)) - d["grid_shortfall_twh_by_year"][y]) <= 1e-6 for t, y in zip(x.T, ys))
    retired = [k for k in ("horizon_reliability", "worst_year_reliability", "annual_reliability_by_year", "expected_unserved_energy_twh") if k in d]
    differ = abs(d["unserved_only_reliability"] - d["grid_service_reliability_horizon"]) > 1e-9      # the two are different quantities on this fixture
    ok = all(v_ <= 1e-9 for v_ in dev.values()) and con_ok and not retired and differ
    desc = (f"max deviation of the eight fields from an independent recomputation {max(dev.values()):.1e}; on the scarce-gas fixture (A_t x 0.2, annual obligation 50%) "
            f"self-generation is {100*sum(G)/sum(D):.3f}% of demand and unserved {100*sum(U)/sum(D):.3f}%: unserved_only_reliability {d['unserved_only_reliability']:.6f}, "
            f"grid_service_reliability_horizon {d['grid_service_reliability_horizon']:.6f} (different quantities, different names); shortfall equals the grid_service_obligation body: {con_ok}; "
            f"retired old names still present: {retired}")
    return PASS(desc) if ok else FAIL("renamed fields correct, canonical reliability = 1 - (unserved + selfgen)/demand, old names gone", desc, "src/optimize_experiments.py::compute_unserved_only_metrics / compute_grid_service_metrics")


@inv("G8-1d", "B", "Peak adequacy diagnostic (split out of G8-1b by TASK 53): the diagnostic's peak and firm capacity against the model's own firm-capacity constraint. NOT in the fix epoch's scope; L31/L103")
def g8_1d():
    x = solve_scaled(0.2, force=False, eaas=True)
    d = _diag(x)
    m, sc = x.m, x.sc
    ys = x.years
    pa = d["peak_adequacy_by_year"]
    om = float(sc.get("within_block_peaking_factor", 1.3448))
    cc = sc["capacity_credit"]
    t = x.Tlast
    model_peak = float(sc["peak_mw"][t]) * om
    rep_peak = pa[ys[t]]["peak_demand_mw"]
    rep_firm = pa[ys[t]]["firm_capacity_mw"]
    credited = (cc["solar"] * v(m.solar_capacity_mw[t]) + cc["gas"] * v(m.gas_capacity_mw[t]) + cc["storage"] * v(m.storage_power_mw[t])
                + cc["hydro"] * float(sc["hydro_capacity_mw_by_year"][t]))
    mult_used = float(sc.get("peak_demand_multiple", 1.82))
    peak_ok = abs(rep_peak - model_peak) / model_peak < 0.01
    firm_ok = abs(rep_firm - credited) / credited < 0.01
    desc = (f"last-year peak adequacy: reported peak {rep_peak:.0f} MW (scenario peak_demand_multiple {mult_used} x average) vs the model's firm-capacity peak (peak_mw x omega) {model_peak:.0f} MW; "
            f"reported firm capacity {rep_firm:.0f} MW (gas + storage nameplate) vs the model's credited capacity {credited:.0f} MW")
    if peak_ok and firm_ok:
        return PASS(desc)
    return FAIL("reported peak adequacy equals the model's firm-capacity test", f"peak ok: {peak_ok}; firm ok: {firm_ok}",
                "src/optimize_experiments.py::extract_planning_diagnostics peak_adequacy_by_year (peak_demand_multiple, gas + storage power)", detail=desc)


@inv("G8-1c", "B", "Addendum B-1 (EaaS / bankability group): multipliers, private investment, subsidy per MW, bankable capex and the bankability threshold recomputed; bankable capex checked against the model's bankability constraint")
def g8_1c():
    x = solved(("Smod",), eaas=True, capital_case="moderate", force=True)
    d = _diag(x)
    m, sc, econ = x.m, x.sc, x.econ
    rs = float(sc["social_discount_rate"])
    ys = x.years
    cap = lambda t: float(pyo.value(m.solar_capex_param[t]))
    priv = sum(df(t, rs) * v(m.solar_eaas_add[t]) * cap(t) for t in x.T)
    sub = sum(df(t, rs) * v(m.eaas_subsidy[t]) for t in x.T)
    pubsol = sum(df(t, rs) * v(m.solar_public_add[t]) * cap(t) for t in x.T)
    spv = storage_price_vec(x.sc, econ)
    stor = sum(df(t, rs) * v(m.storage_add[t]) * spv[t] for t in x.T)
    exp = {
        "eaas_private_investment_pv_usd": priv,
        "eaas_private_investment_undiscounted_usd": sum(v(m.solar_eaas_add[t]) * cap(t) for t in x.T),
        "eaas_multiplier_vs_subsidy": priv / sub if sub > 0 else None,
        "eaas_multiplier_vs_public_budget": priv / (pubsol + stor + sub),
    }
    dev = {}
    for k, e in exp.items():
        got = d.get(k)
        dev[k] = None if (got is None and e is None) else (abs(float(got) - e) / max(1.0, abs(e)) if got is not None and e is not None else float("inf"))
    spm = max(abs(d["subsidy_per_mw_usd_by_year"][y] - (v(m.eaas_subsidy[t]) / v(m.solar_eaas_add[t]) if v(m.solar_eaas_add[t]) > 1e-6 else 0.0)) for t, y in enumerate(ys))
    dev["subsidy_per_mw_usd_by_year"] = spm
    # bankable capex per MW from the CONSTRAINT: capex0 - (coefficient of solar_eaas_add in the bankability row), at t=0
    cm, _ = _coef_map(m.eaas_bankability[0].body)
    coef = cm.get(id(m.solar_eaas_add[0]), None)
    bankable_model = cap(0) - coef                # revenue per MW the constraint allows
    tau = float(sc["solar_service_tariff_usd_per_twh"])
    rep_b = float(d["bankable_capex_per_mw"])
    dev["bankable_capex_per_mw vs constraint"] = abs(rep_b - bankable_model) / bankable_model
    thr = float(d["bankability_threshold"])
    # at the threshold tariff the bankable capex equals the capex: bankable(tau*) = bankable(tau) * thr / tau = capex0
    dev["bankability_threshold (bankable at threshold == capex)"] = abs(rep_b * thr / tau - cap(0)) / cap(0)
    bad = {k: dv for k, dv in dev.items() if dv is not None and dv > 1e-6}
    desc = "; ".join(f"{k}: {('None' if dv is None else format(dv, '.1e'))}" for k, dv in dev.items())
    extra = f"private investment {priv:.0f} USD, subsidy {sub:.0f}, multiplier vs public budget {exp['eaas_multiplier_vs_public_budget']:.3f}; threshold {thr:.0f} USD/TWh vs tariff {tau:.0f}; bankable capex {rep_b:.0f} vs 2025 capex {cap(0):.0f} USD/MW"
    return PASS(f"{desc}. {extra}") if not bad else FAIL("rel dev <= 1e-6", str(bad), "src/optimize_experiments.py::extract_planning_diagnostics", detail=desc + ". " + extra)


@inv("G8-2", "B", "'Capital mobilised (report window)' uses 2045, not the model horizon end: recomputed from the solution on a 2050-horizon fixture and compared with npv_gross_capex_report_window_usd")
def g8_2():
    x = solved(("G82",), end_year=2050, force=True, eaas=True)
    d = _diag(x)
    m, sc, econ = x.m, x.sc, x.econ
    rs = float(sc["social_discount_rate"])

    def capex_at(t):
        c = (v(m.solar_public_add[t]) + v(m.solar_eaas_add[t])) * float(pyo.value(m.solar_capex_param[t]))
        c += v(m.storage_add[t]) * storage_price_vec(sc, econ)[t] + v(m.gas_add[t]) * float(sc["gas_capex_per_mw"])
        if hasattr(m, "convert_mw") and t == 0:
            c += float(sc["conversion_capex_per_kw"]) * 1000.0 * sum(v(m.convert_mw[k]) for k in m.convert_mw)
        return c
    rep = sum(df(t, rs) * capex_at(t) for t in x.T if x.years[t] <= 2045)
    full = sum(df(t, rs) * capex_at(t) for t in x.T)
    got_r, got_f, endy = float(d["npv_gross_capex_report_window_usd"]), float(d["npv_gross_capex_full_horizon_usd"]), d["report_window_end_year"]
    ok = near(got_r, rep, 1e-9) and near(got_f, full, 1e-9) and endy == 2045 and abs(full - rep) > 1.0
    note = f"horizon 2025-2050; report window {got_r:.2f} vs independent 2045 sum {rep:.2f}; full horizon {got_f:.2f} vs {full:.2f}; the two differ by {full-rep:.2f} so the test discriminates; report_window_end_year = {endy}"
    return PASS(note) if ok else FAIL("report window sums years <= 2045 only", note, "src/optimize_experiments.py::extract_planning_diagnostics (_report_T)")


@inv("G8-3", "B", "Dual-to-shadow-price conversion: sign and scaling checked against a finite difference of the objective; the _pv_ series equals -dObj/dA_t and the _yr_ series equals that divided by DF[t] (TASK 53 2.3)")
def g8_3():
    from src.optimize_experiments import (extract_gas_shadow_pv_usd_per_twh_th as g_pv,
                                          extract_gas_shadow_yr_usd_per_twh_th as g_yr)
    import src.optimize_model_sliced as osm
    orig = _ORIG_GAS_AVAIL

    def solve_p(lam, tidx, delta):
        def f(**kw):
            r = dict(orig(**kw))
            a = (r["available_twh_th"] * lam).copy()
            a[tidx] += delta
            r["available_twh_th"] = a
            return r
        osm.gas_available_power = f
        try:
            x = build_fx(force=False)
        finally:
            osm.gas_available_power = orig
        solve_fx(x)
        return x
    x0 = solve_p(0.4, 0, 0.0)
    pv, yr = g_pv(x0.m, x0.years), g_yr(x0.m, x0.years)
    rs = float(x0.sc["social_discount_rate"])
    rows, bad = [], []
    for t in (1, 5, 9):
        d = 0.05
        fd = (obj(solve_p(0.4, t, d)) - obj(solve_p(0.4, t, -d))) / (2 * d)        # d objective / d A_t
        y = x0.years[t]
        rows.append((t, fd, pv[y], yr[y], df(t, rs)))
        if not (abs(pv[y] + fd) <= 1e-4 * abs(fd) and pv[y] >= 0):
            bad.append(("pv", t, fd, pv[y]))
        if not (abs(yr[y] - pv[y] / df(t, rs)) <= 1e-9 * max(1.0, abs(yr[y])) and abs(yr[y] + fd / df(t, rs)) <= 1e-4 * abs(fd / df(t, rs))):
            bad.append(("yr", t, fd, yr[y]))
    desc = "; ".join(f"t={t}: dObj/dA = {fd/1e6:.3f} M, _pv_ {p_/1e6:.3f} M, _yr_ {y_/1e6:.3f} M, DF {d_:.3f}" for t, fd, p_, y_, d_ in rows)
    if bad:
        return FAIL("_pv_ == -dObj/dA_t (>= 0) and _yr_ == _pv_ / DF[t]", str(bad), "src/optimize_experiments.py::extract_gas_shadow_{pv,yr}_usd_per_twh_th")
    return PASS(desc + ". Both bases are exposed and named; consumers that compare with year-t benchmarks use _yr_ (06, GAS-1 in optimize_experiments).")


# ===========================================================================
# STAGE C -- Tier 3 and the dimensional audit
# ===========================================================================
UNIT_VALUES = {1e6, 1000000, 1e3, 1000, 8760, 1e-6, 3.412, 3.6, 365, 24, 1e9, 1e-3}

# E1 annotations: substring of the source line -> (unit identity asserted, closes?)
E1_ANN = [
    ("annual_mwh_per_mw = cf_annual * 8760", "MWh per MW-year = (dimensionless capacity factor) x (hours per year)", True),
    ("/ 1_000_000   # TWh per MW", "TWh per MW = MWh per MW / (1e6 MWh per TWh)", True),
    ("* H[(s, p)] * gas_availability_factor / 1e6", "TWh_e = MW x h x (availability) / 1e6 (MWh per TWh): matches gas_generation in TWh_e", True),
    ("* H[(s, p)] * _conv_tiers[tier][\"paf\"] / 1e6", "TWh_e = MW x h x PAF / 1e6", True),
    ("<= m.gas_capacity_mw[t] * H[(s, p)] * gas_availability_factor / 1e6", "TWh_e = MW x h x availability / 1e6", True),
    ("m.gas_new_capacity_mw[t] * 8760 * gas_availability_factor", "MWh_e per year at availability (MW x h/yr x alpha), numerator of the exempt allowance", True),
    ("/ eta / 1e6", "TWh_th = MWh_e / eta (MWh_th per MWh_e) / 1e6: closes to the TWh_th of gas_balance; eta is the dispatch efficiency (TASK 66; was gas_new_efficiency)", True),
    ("<= hydro_capacity_mw_by_year[t] * H[(s, p)] / 1e6", "TWh = MW x h / 1e6, against hydro_gen in TWh", True),
    ("<= m.storage_power_mw[t] * H[(s, p)] / 1e6", "TWh = MW x h / 1e6, against storage flows in TWh_e", True),
    ("solar_energy_per_mw_annual = scenario[\"solar_cf\"] * 8760 / 1_000_000", "TWh per MW-year = cf x h/yr / 1e6; multiplied by tariff USD/TWh gives USD per MW-year", True),
    ("solar_energy_per_mw = scenario[\"solar_cf\"] * 8760 / 1_000_000", "TWh per MW-year (annual builder), same identity", True),
    ("== _conv_capex_per_kw * 1000.0 * m.convert_mw[tier]", "USD = (USD per kW) x (1000 kW per MW) x MW", True),
    ("_conv_capex_per_kw * 1000.0 * m.convert_mw[tier]", "USD = (USD per kW) x 1000 kW/MW x MW (overnight diagnostic)", True),
    ("float(capex_per_kw) * 1000.0 * total_mw", "USD = USD/kW x 1000 kW/MW x MW", True),
    ("* 1e6 * ef_e", "tCO2 = TWh_e x (1e6 MWh per TWh) x (tCO2 per MWh_e)", True),
    ("_converted_gen_year(m, t) * 1e6 * _ef_e_retrofit", "tCO2 = TWh_e x 1e6 x (tCO2 per MWh_e at retrofit efficiency)", True),
    ("* 1e6 * selfgen_ef_e", "tCO2 = TWh_e x 1e6 x tCO2 per MWh_e", True),
    ("m.gas_generation[t] * 1e6 * ef_e", "tCO2 = TWh_e x 1e6 x tCO2/MWh_e (annual builder)", True),
    ("m.selfgen[t] * 1e6 * selfgen_ef_e", "tCO2 = TWh_e x 1e6 x tCO2/MWh_e (annual builder)", True),
    ("* 1e6 * _ef_th", "tCO2 = TWh_th x 1e6 (MWh_th per TWh_th) x tCO2/MWh_th (gas_vintage_dispatch sensitivity, TASK 66)", True),
    ("<= m.gas_vintage_new_capacity_mw[t] * 8760 * gas_availability_factor / 1e6", "TWh_e = gas_new_efficiency x TWh_th stream <= MW added x 8760 h x alpha / 1e6 (annual builder, vintage dispatch, TASK 77)", True),
    ("<= (m.gas_capacity_mw[t] - m.gas_vintage_new_capacity_mw[t]) * 8760 * gas_availability_factor / 1e6", "TWh_e = eta x legacy TWh_th stream <= surviving legacy MW x 8760 h x alpha / 1e6 (annual builder, vintage dispatch, TASK 77)", True),
    ("m.gas_to_power[t] * 1e6 * econ[\"EF_TCO2_PER_MWH_TH\"]", "tCO2 = TWh_th x 1e6 (MWh_th per TWh_th) x tCO2/MWh_th (annual builder, vintage dispatch, TASK 77)", True),
    ("<= (m.storage_power_mw[t] * storage_deployable_hours) / 1e6", "TWh = MW x (deployable hours per year, 700) / 1e6 (annual builder)", True),
    ("<= m.gas_capacity_mw[t] * 8760 * gas_availability_factor / 1e6", "TWh_e = MW x 8760 h x alpha / 1e6 (annual builder)", True),
    ("storage_deployable_hours > 8760", "bound check: hours per year", True),
    ("if total != 8760", "slice hours must sum to hours per year", True),
    ("mip_feasibility_tolerance", "not a unit conversion: HiGHS feasibility tolerance (dimensionless, relative), recorded explicitly", True),
    ("_getter(t) > 1e-6", "not a unit conversion: numerical threshold (MW or MWh) in the in-horizon retirement guard", True),
]
E1_IO = [("fom_solar", "gas", "USD/kW-yr x 1000 = USD per MW-yr; x MW"), ]


@inv("E1", "C", "Every unit-conversion literal in both builders (1e6, 1e3, 8760, 1e-6, 3.412, ...): the expression it sits in and the unit identity it asserts; flag any that does not close")
def e1():
    rows, unannotated = [], []
    for bname, path in BUILDERS.items():
        src = src_of(path)
        lines = src.splitlines()
        tree = ast.parse(src)
        seen = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool) and n.value in UNIT_VALUES:
                key = (n.lineno, n.value)
                if key in seen:
                    continue
                seen.add(key)
                line = lines[n.lineno - 1].strip()
                ann = next((a for a in E1_ANN if a[0] in line), None)
                if ann is None and n.lineno + 1 <= len(lines):          # multi-line expressions: look at the neighbours
                    ctx = " ".join(l.strip() for l in lines[max(0, n.lineno - 3):n.lineno + 1])
                    ann = next((a for a in E1_ANN if a[0] in ctx), None)
                rows.append((bname, n.lineno, n.value, line[:100], ann[1] if ann else "UNANNOTATED", ann[2] if ann else None))
                if ann is None:
                    unannotated.append((bname, n.lineno, n.value, line[:80]))
    # the shared O&M helper (src/io.py) is a third site of unit conversions used by both builders
    io_src = src_of("src/io.py")
    for ln, l in enumerate(io_src.splitlines(), 1):
        if ("1000.0" in l or "1e6" in l) and 252 <= ln <= 338:
            rows.append(("io(shared)", ln, "1000/1e6", l.strip()[:100], "FOM: USD/kW-yr x 1000 kW/MW = USD per MW-yr (x MW); storage divides by duration (h) to reach USD per MWh-yr (x MWh); VOM: TWh x 1e6 MWh/TWh x USD/MWh = USD", True))
    nbad = [r for r in rows if r[5] is False]
    r = PASS(f"{len(rows)} unit-conversion literal sites enumerated ({sum(1 for r_ in rows if r_[0]=='sliced')} sliced, {sum(1 for r_ in rows if r_[0]=='annual')} annual, {sum(1 for r_ in rows if r_[0]=='io(shared)')} shared O&M helper); every identity closes; none unannotated") \
        if not nbad and not unannotated else FAIL("every site annotated and closed", f"unannotated {unannotated}; not closing {nbad}", "src/optimize_model*.py")
    r.extra["rows"] = rows
    return r


E2_ANN = {
    # name: (LHS units, RHS units, closes, note)
    "gas_new_stream_within_total": ("TWh_th (added-capacity stream)", "TWh_th (total gas draw)", True, "vintage dispatch (TASK 66)"),
    "gas_new_stream_capacity": ("TWh_e (gas_new_efficiency x added stream TWh_th)", "MW added x h x alpha / 1e6 = TWh_e", True, "vintage dispatch (TASK 66)"),
    "gas_legacy_stream_capacity": ("TWh_e (eta x legacy stream TWh_th)", "MW surviving legacy capacity x h x alpha / 1e6 = TWh_e", True, "the tightened row (TASK 74/75); slack positive at the real efficiency, >= -1e-6 at e = gas_eta"),
    "gas_capacity_constraint": ("TWh_e", "MW x h x alpha / 1e6 = TWh_e", True, "net of converted nameplate when conversion is on"),
    "convert_capacity_constraint": ("TWh_e", "(1+uplift) x MW x h x PAF / 1e6 = TWh_e", True, ""),
    "gas_balance": ("TWh_th (burn + converted burn + curtailment)", "A_t in TWh_th (+ exempt allowance TWh_th)", True, ""),
    "gas_new_capacity_cap_constraint": ("MW", "MW", True, "cumulative cap"),
    "hydro_season_budget": ("TWh_e", "TWh_e (annual series x season share)", True, "equality"),
    "hydro_power_limit": ("TWh_e", "MW x h / 1e6 = TWh_e", True, ""),
    "hydro_min_flow": ("TWh_e", "fraction x TWh_e x (h/h)", True, ""),
    "storage_season_balance": ("TWh_e discharged", "eta x TWh_e charged", True, "inequality, no state of charge"),
    "storage_charge_power_limit": ("TWh_e", "MW x h / 1e6 = TWh_e", True, ""),
    "storage_discharge_power_limit": ("TWh_e", "MW x h / 1e6 = TWh_e", True, ""),
    "storage_energy_balance": ("TWh_e discharged (annual builder)", "eta x TWh_e charged", True, "annual builder only"),
    "storage_power_limit": ("TWh_e discharged (annual builder)", "MW x deployable hours / 1e6 = TWh_e", True, "annual builder only"),
    "storage_charge_power_limit(annual)": ("TWh_e", "MW x deployable hours / 1e6", True, "annual builder"),
    "balance": ("TWh_e supply (gas, converted, solar, hydro, discharge - charge, selfgen x busbar, unserved)", "TWh_e demand per slice (or per year, annual builder)", True, ""),
    "eaas_bankability": ("USD (capex USD/MW x MW - subsidy USD)", "USD (tariff USD/TWh x TWh/MW-yr x npv factor x MW / margin)", True, ""),
    "eaas_subsidy_limit": ("USD", "USD/MW x MW", True, ""),
    "eaas_disable": ("MW", "0", True, ""),
    "land_constraint": ("km2 (MW x km2/MW + MWh x km2/MWh)", "km2", True, ""),
    "firm_capacity_constraint": ("MW (credit x MW; storage power = MWh / duration h)", "MW (peak MW x omega x (1 + rho))", True, "storage MWh/h gives MW"),
    "solar_public_split_identity": ("USD (conc + comm volume)", "USD/MW x MW", True, ""),
    "solar_eaas_split_identity": ("USD", "USD/MW x MW", True, ""),
    "gas_split_identity": ("USD", "USD/MW x MW", True, ""),
    "storage_split_identity": ("USD", "USD/MWh x MWh", True, ""),
    "convert_split_identity": ("USD", "USD/kW x 1000 x MW", True, ""),
    "f1_concessional_envelope_constraint": ("USD (undiscounted commitment stock)", "USD", True, "undiscounted, unlike the budget"),
    "public_budget_constraint": ("USD, present value (DF x capex + subsidy)", "USD, present value (public_solar_budget_npv)", True, "discounted, unlike the envelope"),
    "grid_service_obligation": ("TWh_e (selfgen + unserved)", "epsilon x TWh_e demand", True, ""),
    "emissions_constraint": ("tCO2 (capped boundary)", "tCO2 cap", True, ""),
    "solar_cumulative_cap": ("MW cumulative additions", "MW", True, ""),
    "solar_min_build_constraint": ("MW additions", "MW", True, ""),
    "deployment_constraint": ("MW", "MW", True, "annual builder build cap if present"),
}


def _constraint_families():
    fam = {}
    variants = [
        dict(eaas=True, capital_case="moderate", force=False),
        dict(eaas=True, force=False, arms={"new_gas_deliverability": "exempt", "gas_build_case": "gas_ndc_17gw"}, sc_over={"solar_min_build_mw_per_year": 10.0}),
        dict(eaas=True, force=False, builder_kw={"max_offgrid_fraction": 0.01, "grid_service_mode": "annual"}),
        dict(eaas=True, force=False, builder_kw={"max_offgrid_fraction": 0.01, "grid_service_mode": "total"}),
        dict(eaas=True, force=False, builder_kw={"emissions_cap_by_year": [1e9] * (FIX["end_year"] - 2024)}),
        dict(eaas=True, force=False, builder_kw={"emissions_cap": 1e12}),
        dict(eaas=True, force=False, sc_over={"solar_cumulative_cap_mw": [1e5] * (FIX["end_year"] - 2024)}),
    ]
    for kw in variants:
        x = build_fx(**kw)
        for c in x.m.component_objects(pyo.Constraint, descend_into=False):
            fam.setdefault(c.name, "sliced")
    return fam


@inv("E2", "C", "Every constraint family in both builders: units of the left and right sides; flag mismatches and unannotated families")
def e2():
    sl = _constraint_families()
    # annual builder
    from src.optimize_model import build_model
    sc = make_scenario(arms={"conversion_case": "conv_none"})
    sc["capital_window_case"] = "advance"
    econ = load_econ("voll_mid", genset_case="genset_central")
    capex = load_solar_capex_by_year(scenario_name="solar_low", start_year=2025, end_year=FIX["end_year"])
    an = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ma = build_model(scenario=sc, econ=econ, solar_capex_by_year=capex)
        for c in ma.component_objects(pyo.Constraint, descend_into=False):
            an.setdefault(c.name, "annual")
    except Exception as e:
        an = {"__ERR__": f"{type(e).__name__}: {str(e)[:80]}"}
    names = dict(sl)
    for k, v_ in an.items():
        names.setdefault(k, v_)
    un = [k for k in names if k not in E2_ANN and not k.startswith("__")]
    bad = [k for k in names if k in E2_ANN and not E2_ANN[k][2]]
    rows = [(k, names[k], *E2_ANN.get(k, ("?", "?", None, "UNANNOTATED"))) for k in sorted(names)]
    ok = not un and not bad
    r = PASS(f"{len(names)} constraint families ({len(sl)} sliced, {len(an)} annual); every left/right pair closes dimensionally. Note: the envelope is an undiscounted USD stock and the budget a present-value USD sum (both close; they are different objects)") if ok else \
        FAIL("every family annotated and closing", f"unannotated {un}; not closing {bad}", "src/optimize_model*.py", detail=str(an.get("__ERR__", "")))
    r.extra["rows"] = rows
    return r


@inv("E3-gap", "C", "GAP REPORT: E1/E2 alone do not flag the G8-3 present-value dual; the class is closed iff E4 (the third audit class, TASK 53) exists and passes")
def e_gap():
    r1, r2 = e1(), e2()
    flagged = r1.status == "FAIL" or r2.status == "FAIL"
    r4 = e4()
    if r4.status == "PASS":
        return PASS(f"E1/E2 {'flag' if flagged else 'do not flag'} a dual-basis defect (they audit builder literals and constraint sides only); the third class E4 exists, measures every reported dual by finite difference, and passes after TASK 53 2.3. The gap is closed by E4, not by E1/E2.")
    return FAIL("E4 passes (the third audit class closes the gap E1/E2 leave)", f"E4: {r4.status} - {r4.observed}", "scripts/52_invariants.py::E4")


# ---------------------------------------------------------------------------
# E4 -- the third dimensional audit class (TASK 52 section 2.3).
# For every quantity the code reports from a DUAL or from an objective-derived
# NPV: is it present-value (year-0 USD) or year-t USD, and what divisor converts
# it?  The objective is entirely PV (every cost term carries m.DF[t]); the dual
# of ANY constraint therefore inherits PV units on its objective side, whatever
# units the constraint itself is written in.  Rows below are MEASURED when the
# status column says so (finite difference of the re-solved objective against
# the reported number) and BY CONSTRUCTION otherwise.
# ---------------------------------------------------------------------------
def _fd_central(make, p0, rel=1e-4):
    """(obj(p0+h) - obj(p0-h)) / 2h for a builder `make(p)->solved Fx or None`."""
    h = rel * abs(p0)
    a, b = make(p0 + h), make(p0 - h)
    if a is None or b is None:
        return None, h
    return (obj(a) - obj(b)) / (2 * h), h


def _basis_from_ratio(rep, fd, d):
    """Classify a reported dual `rep` against -dObj/dRHS `fd` at discount factor d."""
    pv = abs(rep - abs(fd)) <= 1e-4 * max(abs(fd), 1e-12)
    yr = abs(rep - abs(fd) / d) <= 1e-4 * max(abs(fd) / d, 1e-12)
    if abs(d - 1.0) < 1e-9:
        return "PV (DF=1 at t=0: not informative)" if pv else "NEITHER"
    return "PV (year-0 USD)" if pv and not yr else ("year-t USD" if yr and not pv else ("indistinguishable" if pv and yr else "NEITHER"))


@inv("E4", "C", "Third dimensional audit class: every reported dual and objective-derived NPV is classified PV or year-t USD with its divisor; duals are measured by finite difference of the re-solved objective")
def e4():
    from src.optimize_experiments import (extract_gas_shadow_pv_usd_per_twh_th as g_sh,
                                          extract_gas_shadow_yr_usd_per_twh_th as g_sh_yr,
                                          extract_carbon_shadow_pv_usd_per_tco2 as c_sh,
                                          extract_carbon_shadow_yr_usd_per_tco2 as c_sh_yr,
                                          extract_carbon_shadow_capwide_pv_usd_per_tco2 as c_cap,
                                          extract_reliability_dual)
    import src.optimize_model_sliced as osm
    rows = []          # (quantity, reported key / location, declared basis, measured basis, divisor, status, evidence)
    fails = []

    def add(q, where, declared, measured, divisor, status, ev):
        rows.append((q, where, declared, measured, divisor, status, ev))
        if status == "FAIL":
            fails.append(q)

    free = try_solve(force=False, eaas=True)
    rs = float(free.sc["social_discount_rate"])
    D = lambda t: df(t, rs)
    ys = free.years

    # ---- 1. gas shadow (gas_balance[t], TWh_th)
    def gas_solve(tidx, delta):
        def f(**kw):
            r = dict(_ORIG_GAS_AVAIL(**kw))
            a = (r["available_twh_th"] * 0.4).copy()
            a[tidx] += delta
            r["available_twh_th"] = a
            return r
        osm.gas_available_power = f
        try:
            x = build_fx(force=False)
        finally:
            osm.gas_available_power = _ORIG_GAS_AVAIL
        return solve_fx(x)
    x0 = gas_solve(0, 0.0)
    sh, sh_yr = g_sh(x0.m, x0.years), g_sh_yr(x0.m, x0.years)
    ev, cls_pv, cls_yr = [], set(), set()
    for t in (1, 5, 9):
        d = 0.05
        fd = (obj(gas_solve(t, d)) - obj(gas_solve(t, -d))) / (2 * d)
        rep, rep_yr = sh[x0.years[t]], sh_yr[x0.years[t]]
        cls_pv.add(_basis_from_ratio(rep, fd, D(t)))
        cls_yr.add(_basis_from_ratio(rep_yr, fd, D(t)))
        ev.append(f"t={t}: _pv_ {rep/1e6:.3f} M, _yr_ {rep_yr/1e6:.3f} M, |dObj/dA| {abs(fd)/1e6:.3f} M, DF {D(t):.3f}")
    add("gas shadow price (USD per TWh_th)", "gas_shadow_pv_usd_per_twh_th_by_year and gas_shadow_yr_usd_per_twh_th_by_year (TASK 53 2.3); 01b writes both; timeseries gas_shadow_{pv,yr}_usd_per_twh_th",
        "both bases named in the key", f"_pv_ measured {', '.join(sorted(cls_pv))}; _yr_ measured {', '.join(sorted(cls_yr))}", "_yr_ = _pv_ / DF[t]",
        "PASS" if cls_pv == {"PV (year-0 USD)"} and cls_yr == {"year-t USD"} else "FAIL", "; ".join(ev) + " (= G8-3)")

    # ---- 2. carbon shadow, annual cap
    e = [v(free.m.emissions_by_year[t]) for t in free.T]
    cap0 = [0.9 * a if a > 1 else 1e12 for a in e]

    def carbon_make(tidx):
        def mk(c):
            cap = list(cap0)
            cap[tidx] = c
            return try_solve(force=False, eaas=True, builder_kw={"emissions_cap_by_year": cap})
        return mk
    xc = try_solve(force=False, eaas=True, builder_kw={"emissions_cap_by_year": cap0})
    shc, shc_yr = c_sh(xc.m, xc.years), c_sh_yr(xc.m, xc.years)
    ev, cls_pv, cls_yr = [], set(), set()
    for t in (0, 4, 9):
        fd, h = _fd_central(carbon_make(t), cap0[t])
        rep, rep_yr = shc[xc.years[t]], shc_yr[xc.years[t]]
        cls_pv.add(_basis_from_ratio(rep, fd, D(t)))
        cls_yr.add(_basis_from_ratio(rep_yr, fd, D(t)))
        ev.append(f"t={t}: _pv_ {rep:.3f}, _yr_ {rep_yr:.3f} USD/t, |dObj/dcap| {abs(fd):.3f}, DF {D(t):.3f}")
    cls_pv = {c for c in cls_pv if "not informative" not in c} or cls_pv
    cls_yr = {c for c in cls_yr if "not informative" not in c} or cls_yr
    add("carbon shadow price, annual cap (USD per tCO2)", "carbon_shadow_pv_usd_per_tco2_by_year and carbon_shadow_yr_usd_per_tco2_by_year (TASK 53 2.3); timeseries carbon_shadow_{pv,yr}_usd_per_tco2; carbon_shadow_{pv,yr}_*_mean",
        "both bases named in the key", f"_pv_ measured {', '.join(sorted(cls_pv))}; _yr_ measured {', '.join(sorted(cls_yr))}", "_yr_ = _pv_ / DF[t]",
        "PASS" if cls_pv == {"PV (year-0 USD)"} and cls_yr == {"year-t USD"} else "FAIL", "; ".join(ev))

    # ---- 3. carbon shadow, scalar cap
    tot0 = 0.9 * sum(e)
    def scal_make(c):
        return try_solve(force=False, eaas=True, builder_kw={"emissions_cap": c})
    xs = scal_make(tot0)
    shs = c_sh(xs.m, xs.years)
    capw = c_cap(xs.m)
    fd, h = _fd_central(scal_make, tot0)
    per_year_none = all(val is None for val in shs.values()) and all(val is None for val in c_sh_yr(xs.m, xs.years).values())
    ok = capw is not None and abs(capw - abs(fd)) <= 1e-4 * abs(fd) and per_year_none
    emitted = [v(xs.m.emissions_by_year[t]) for t in xs.T]
    add("carbon shadow price, single cap (USD per tCO2)", "carbon_shadow_capwide_pv_usd_per_tco2 (TASK 53 2.3); the per-year series are None under a scalar cap",
        "name says cap-wide present value", ("PV (equals |dObj/dcap|)" if ok else "NEITHER") + "; no per-year value is written",
        "none: the marginal tonne is abated in whichever year is cheapest (endogenous); no _yr_ counterpart by design",
        "PASS" if ok else "FAIL", f"cap-wide {capw}, |dObj/dcap| {abs(fd):.3f}; per-year series None: {per_year_none}; emissions by year {[round(a/1e6,2) for a in emitted]} Mt")

    # ---- 4. land shadow (land_constraint[t], km2)
    ls_, lg_, lst_ = (float(free.sc["land_intensity_solar_km2_per_mw"]), float(free.sc["land_intensity_gas_km2_per_mw"]), float(free.sc["land_intensity_storage_km2_per_mwh"]))
    u = max(v(free.m.solar_capacity_mw[t]) * ls_ + v(free.m.gas_capacity_mw[t]) * lg_ + v(free.m.storage_capacity_mwh[t]) * lst_ for t in free.T)
    A2 = 0.6 * u
    land_make = lambda a: try_solve(force=False, eaas=True, sc_over={"land_available_km2": a})
    xl = land_make(A2)
    dg = _diag(xl)
    shl = dg["land_shadow_price_by_year"]
    fd, h = _fd_central(land_make, A2)
    sum_rep = sum(shl[y] for y in xl.years)
    nbind = sum(1 for y in xl.years if shl[y] > 1e-9)
    bind_df = [D(t) for t, y in enumerate(xl.years) if shl[y] > 1e-9]
    add("land shadow price (USD per km2)", "extract_planning_diagnostics land_shadow (`float(-raw)`); key land_shadow_price_by_year",
        "none in the key", ("PV (year-0 USD)" if abs(sum_rep - abs(fd)) <= 1e-4 * abs(fd) else "NEITHER") + f" (sum over years of the reported values equals |dObj/d(allowance)| because one allowance feeds all {len(xl.T)} constraints)",
        "divide by DF[t] for the year-t value" + (f" (binding years' DF {min(bind_df):.3f}..{max(bind_df):.3f})" if bind_df else ""),
        "DECLARED", f"option 3 (TASK 53 2.3): basis declared in register L106, key not renamed, no result reads it; sum reported {sum_rep:.4g}, |dObj/dA| {abs(fd):.4g}, {nbind} binding year(s)")

    # ---- 5. public budget shadow (sum_t DF[t] x spend <= budget_npv: constraint is itself in PV USD)
    pts = []
    for b in BUD_GRID[:4]:
        xb = try_solve(eaas=True, force=False, capital_case="moderate", sc_over={"public_solar_budget_npv": b})
        if xb is not None and con_status(xb, xb.m.public_budget_constraint)["binding"]:
            pts.append((b, xb))
    if pts:
        b, xb = pts[len(pts) // 2]
        mk = lambda bb: try_solve(eaas=True, force=False, capital_case="moderate", sc_over={"public_solar_budget_npv": bb})
        fd, h = _fd_central(mk, b)
        rep = _diag(xb)["public_budget_shadow_usd_per_usd"]
        ok = abs(rep - abs(fd)) <= 1e-4 * abs(fd)
        add("public budget shadow (USD of PV cost per USD of PV budget)", "extract_planning_diagnostics public_budget_shadow; key public_budget_shadow_usd_per_usd",
            "usd_per_usd: PV cost per PV budget dollar", "PV / PV (dimensionless; both sides discounted)", "none; equals 1 / DF[t] only if one wants year-t USD of spend (spend in year t consumes DF[t] of the budget)",
            "PASS" if ok else "FAIL", f"budget {b:g}: reported {rep:.5f}, |dObj/dB| {abs(fd):.5f}")
    else:
        add("public budget shadow", "", "", "not measured: no binding budget on the grid", "", "SKIP", f"grid {BUD_GRID[:4]}")

    # ---- 6. concessional envelope shadow (F1 envelope: UNDISCOUNTED USD stock; objective PV)
    cfree = conc_total(try_solve(force=False, eaas=True, capital_case="moderate"))
    xe = try_solve(force=False, eaas=True, capital_case="moderate", sc_over={"concessional_envelope_usd": 0.5 * cfree})
    if xe is not None:
        env_make = lambda B: try_solve(force=False, eaas=True, capital_case="moderate", sc_over={"concessional_envelope_usd": B})
        # TASK 80 A: relative step 1e-3, not the default 1e-4. NOT a tolerance widening (the 1e-4 criterion below is unchanged): at 1e-4 the
        # perturbation changes the objective by ~10,000 USD against CBC's ~2,500 USD noise floor (signal to noise ~4), so the old step measured noise;
        # at 1e-3 the forward and backward slopes agree to 5e-6 and with the dual to < 3e-5 (task_log_79 C.2).
        fd, h = _fd_central(env_make, 0.5 * cfree, rel=1e-3)
        raw = xe.m.dual.get(xe.m.f1_concessional_envelope_constraint, None)
        rep = None if raw is None else max(0.0, -float(raw))
        draws = [t for t in xe.T if v(xe.m.solar_public_conc_vol[t]) + v(xe.m.solar_eaas_conc_vol[t]) + v(xe.m.storage_conc_vol[t]) > 1e-6]
        dfs = [D(t) for t in draws]
        ok = rep is not None and abs(rep - abs(fd)) <= 1e-4 * abs(fd)
        add("concessional envelope shadow (USD of PV cost per USD of envelope)", "scripts/18_run_pareto_frontier.py f1_envelope_shadow_pv_per_undiscounted_usd (`-dual`); diagnose_* envelope_dual_pv_per_undiscounted_usd (raw); the dead extract_planning_diagnostics concessional_shadow_usd_per_usd is L108",
            "the key says PV per undiscounted USD (TASK 53 2.3)", ("MIXED: PV objective per UNDISCOUNTED USD of commitment" if ok else "NEITHER") + "; reported equals |dObj/dB|",
            f"no single divisor: the marginal envelope dollar is drawn in a vintage-dependent year (concessional draws at DF {min(dfs):.3f}..{max(dfs):.3f}); year-v USD value = reported / DF[v]" if dfs else "none drawn",
            "PASS" if ok else "FAIL", f"B = {0.5*cfree:,.0f}: reported {rep}, |dObj/dB| {abs(fd):.6g}; named to say its basis, not forced into the _pv_/_yr_ family; the diagnostics-dict field concessional_shadow_usd_per_usd is still None under f1_tranche (L108, not in this task)")
    else:
        add("concessional envelope shadow", "", "", "not measured: infeasible", "", "SKIP", "")

    # ---- 7. grid-service obligation (the one quantity that already carries both bases)
    # find the (gas scale, eps) at which the annual obligation binds in the most years on the current default model
    xg, eps0, lam_g, best_n = None, None, None, 0
    for lam_try in (0.2, 0.15, 0.25, 0.3):
        for eps_try in (0.0005, 0.0002, 0.0001, 0.001, 0.002, 0.0):
            with gas_scale(lam_try):
                cand = try_solve(force=False, eaas=True, builder_kw={"max_offgrid_fraction": eps_try, "grid_service_mode": "annual"})
            if cand is None:
                continue
            pvc = extract_reliability_dual(cand.m, _diag(cand))["dual_by_year_pv"]
            n_b = sum(1 for val in pvc.values() if val > 1e-6)
            if n_b > best_n or (n_b == best_n and n_b > 0 and eps_try > (eps0 or 0.0)):
                xg, eps0, lam_g, best_n = cand, eps_try, lam_try, n_b
    if xg is not None:
        dg = _diag(xg)
        r = extract_reliability_dual(xg.m, dg)
        pv = r["dual_by_year_pv"]
        dem = dg["demand_twh_by_year"]
        tb = [t for t in pv if pv[t] > 1e-6][-3:]
        if not tb:
            return FAIL("a binding obligation to measure", "no binding year found", "E4 fixture")
        ev, cls = [], set()
        for t in tb:
            def mk(e_t, t=t):
                lst = [eps0] * len(xg.T)
                lst[t] = e_t
                with gas_scale(lam_g):
                    return try_solve(force=False, eaas=True, builder_kw={"max_offgrid_fraction": lst, "grid_service_mode": "annual"})
            h = 0.25 * eps0 if eps0 > 0 else 1e-4
            up, dn = mk(eps0 + h), mk(max(0.0, eps0 - h))
            # one-sided slopes bracket the dual at a kink (degenerate vertex); the reported value must lie inside the bracket
            sl_r = (obj(up) - obj(xg)) / h / dem[xg.years[t]]
            sl_l = (obj(xg) - obj(dn)) / h / dem[xg.years[t]]
            lo_, hi_ = min(abs(sl_l), abs(sl_r)), max(abs(sl_l), abs(sl_r))
            inside_pv = lo_ * (1 - 1e-3) <= pv[t] <= hi_ * (1 + 1e-3)
            inside_yr = lo_ / D(t) * (1 - 1e-3) <= pv[t] <= hi_ / D(t) * (1 + 1e-3)
            cls.add("PV (year-0 USD)" if inside_pv and not inside_yr else ("year-t USD" if inside_yr and not inside_pv else ("inconclusive: both bases inside the kink bracket" if inside_pv and inside_yr else "NEITHER")))
            fd_rhs = (lo_ + hi_) / 2
            ev.append(f"t={t}: dual_by_year_pv {pv[t]/1e6:.3f} M, one-sided |dObj/dRHS| [{lo_/1e6:.3f}, {hi_/1e6:.3f}] M, year-t value {pv[t]/D(t)/1e6:.3f} M")
        cls = {c for c in cls if not c.startswith("inconclusive")} or cls
        yr_ok = abs(r["dual_yr_usd_per_twh_all_mean"] - (sum(pv[t] / D(t) for t in pv) / len(pv))) <= 1e-6 * abs(r["dual_yr_usd_per_twh_all_mean"] + 1e-12) if r.get("dual_yr_usd_per_twh_all_mean") is not None else None
        add("grid-service obligation dual (USD per TWh of shortfall allowed)", "extract_reliability_dual: dual_by_year_pv / dual_pv_* and dual_yr_*; key dual_pv_usd_per_pct_pt_* / dual_yr_usd_per_pct_pt_* in the runners",
            "both bases named in the key and returned separately (TASK 02)", ", ".join(sorted(cls)) + f"; year basis = PV / m.DF[t] (all-years mean checks: {yr_ok})", "dual_yr = dual_pv / DF[t]",
            "PASS" if cls == {"PV (year-0 USD)"} and yr_ok else "FAIL", "; ".join(ev))
    else:
        add("grid-service obligation dual", "", "", "not measured: infeasible", "", "SKIP", "")

    # ---- 8. by construction rows (no finite difference: degenerate or no reported use)
    add("EaaS bankability / subsidy-limit duals", "scripts/diagnose_eaas_degeneracy.py (dual_bankability, dual_subsidy_limit)",
        "none", "BY CONSTRUCTION: PV (constraints written in year-t USD, objective PV)", "divide by DF[t]; the duals are degenerate on the flat optimal face (F-12) so their value is not identified", "DECLARED",
        "not measured: the face is flat in the objective, so a finite difference returns 0 whichever basis the label means; reported only in a diagnostic script, quoted nowhere")
    add("reduced costs (solar_public_add, solar_eaas_add)", "scripts/diagnose_eaas_degeneracy.py (rc_public, rc_eaas)", "none", "BY CONSTRUCTION: PV (USD per MW of vintage t, in year-0 USD)", "divide by DF[t]", "DECLARED",
        "diagnostic script only")

    # ---- 9. objective-derived NPVs
    xm = solved(("Smod",), eaas=True, capital_case="moderate", force=True)
    dgm = _diag(xm)
    T = xm.T
    ncap = lambda t: float(pyo.value(xm.m.solar_capex_param[t]))
    flow_eaas = [v(xm.m.solar_eaas_add[t]) * ncap(t) for t in T]
    pv_eaas = sum(D(t) * flow_eaas[t] for t in T)
    und_eaas = sum(flow_eaas)
    got = dgm["eaas_private_investment_pv_usd"]
    got_u = dgm["eaas_private_investment_undiscounted_usd"]
    ok_pv = abs(got - pv_eaas) <= 1e-6 * max(1.0, abs(pv_eaas))
    ok_un = abs(got_u - und_eaas) <= 1e-6 * max(1.0, und_eaas)
    add("eaas_private_investment_{pv,undiscounted}_usd", "extract_planning_diagnostics private_investment (TASK 53 2.3; was eaas_private_investment_usd, L107)",
        "both bases named", f"_pv_ is {'PV (year-0 USD)' if ok_pv else 'NEITHER'}; _undiscounted_ is {'the plain sum' if ok_un else 'WRONG'}", "_undiscounted_ = sum without DF[t]",
        "PASS" if ok_pv and ok_un else "FAIL", f"_pv_ {got:,.0f}; _undiscounted_ {got_u:,.0f} (independent: {pv_eaas:,.0f} / {und_eaas:,.0f})")
    capm = dgm["npv_gross_capex_report_window_usd"]
    full = dgm["npv_gross_capex_full_horizon_usd"]
    add("npv_gross_capex_report_window_usd / _full_horizon_usd (capital mobilised)", "extract_planning_diagnostics npv_capex_report / npv_capex_full (sum DF[t] x gross capex)",
        "'npv' in the name", "PV (year-0 USD), report window = years <= 2045 (G8-2)", "undiscounted capital = sum without DF[t]; the two are not interchangeable", "PASS",
        f"fixture report {capm:,.0f}, full {full:,.0f}; bstar_triangulation.md states both NPV and undiscounted")
    for k, where in (("ndc_finance_public_solar_npv_usd", "sum DF x solar_public_add x capex"), ("ndc_finance_eaas_subsidy_npv_usd", "sum DF x eaas_subsidy"),
                     ("eaas_subsidy_npv_usd", "sum DF x eaas_subsidy"), ("public_budget_realised_spend_pv_usd", "sum DF x (public solar + storage + subsidy): the budget constraint's own LHS (TASK 53 2.3; was public_budget_realised_spend_usd, L107; _undiscounted_ beside it)")):
        add(k, "extract_planning_diagnostics: " + where, "'npv' or '_pv_' in the name", "PV (year-0 USD)", "divide each year's term by DF[t]", "PASS", "recomputed independently in G8-1")
    add("every objective component NPV (cost_decomposition.*, cost_*_npv, capital_charge_*_npv, system_cost_npv, npv_total_cost_usd, salvage_npv)", "both builders",
        "'npv' in the name", "PV (year-0 USD); DF[0] = 1 so the base year is 2025; salvage is credited at DF[T_last]", "none needed while the name says npv", "PASS",
        f"DF[0] = {D(0)}; component sum identities are G2-1 / G8-1")
    add("concessional / envelope volumes (concessional_volume_by_vintage_usd, f1_envelope_utilisation)", "extract_planning_diagnostics; 18_run_pareto_frontier", "USD", "year-v USD at vintage v (undiscounted nominal-real USD), NOT PV", "multiply by DF[v] for PV", "PASS", "the envelope is an undiscounted stock by design (L45); the budget (PV) and envelope (undiscounted) must not be compared (G6)")

    bad = [r_ for r_ in rows if r_[5] == "FAIL"]
    table = "\n".join(f"| {q} | {w} | {dcl} | {ms} | {dv} | {st} | {evd} |" for q, w, dcl, ms, dv, st, evd in rows)
    det = "| quantity | where | declared basis | measured basis | divisor to year-t | verdict | evidence |\n|---|---|---|---|---|---|---|\n" + table
    if bad:
        return FAIL("every reported dual / NPV states its basis and the basis matches the measured one",
                    f"{len(bad)} of {len(rows)} rows: " + "; ".join(r_[0] for r_ in bad),
                    "src/optimize_experiments.py (extractors), scripts/18, scripts/01b, scripts/diagnose_*", detail=det)
    nd = sum(1 for r_ in rows if r_[5] == "DECLARED")
    r_ok = PASS(f"{len(rows)} rows: every reported dual and objective-derived NPV states its basis and the stated basis was measured ({nd} declared under option 3, no result reads them); table in the detail file")
    r_ok.extra["e4_table"] = det
    return r_ok


@inv("G10-1", "C", "Identification (script 26): the min/max re-solves return the base objective within the declared tolerance, and the interval converges as the tolerance shrinks (anchor-scale, f3_moderate-like configuration)")
def g10_1():
    import importlib.util
    spec = importlib.util.spec_from_file_location("s26_f3", ROOT / "scripts" / "26_run_f3_identification.py")
    f3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f3)
    x = build_fx(eaas=True, force=False, capital_case="moderate", reduced_life=False, end_year=2054)
    solve_fx(x)
    Z = obj(x)
    rows, bad = [], []
    for tol in (1e-9, 1e-8, 1e-7, 1e-6):
        with contextlib.redirect_stdout(io.StringIO()):
            lo = f3.reoptimise_eaas(x.m, x.sc, x.T, x.years, Z, tol, pyo.minimize)
            hi = f3.reoptimise_eaas(x.m, x.sc, x.T, x.years, Z, tol, pyo.maximize)
        ex_lo, ex_hi = lo["system_cost_npv"] / Z - 1.0, hi["system_cost_npv"] / Z - 1.0
        rows.append((tol, lo["total_eaas_mw"], hi["total_eaas_mw"], ex_lo, ex_hi))
        if max(abs(ex_lo), abs(ex_hi)) > tol + 1e-9:
            bad.append((tol, ex_lo, ex_hi))
    widths = [r[2] - r[1] for r in rows]
    desc = "; ".join(f"tol {t:g}: EaaS [{lo:.1f}, {hi:.1f}] MW, cost excess lo {a/t:+.2f} tol, hi {b/t:+.2f} tol" for t, lo, hi, a, b in rows)
    conv = abs(rows[0][1] - rows[1][1]) < 5.0 and abs(rows[0][2] - rows[1][2]) < 5.0
    note = (f"base objective {Z:.2f} (the central anchor value; non-quotable harness run); at the script's own tolerances (1e-6 and up) the extremes spend the whole allowance (excess = +1.00 tol, {rows[-1][3]*Z:,.0f} USD) and the lower end sits {rows[0][1]-rows[-1][1]:.0f} MW below its tol -> 0 limit")
    if bad or not conv:
        return FAIL("cost within tol of base; interval converges as tol -> 0", f"{bad}; converged {conv}", "scripts/26_run_f3_identification.py::reoptimise_eaas", detail=desc + ". " + note)
    return PASS(desc + ". " + note)


@inv("G10-2", "C", "--fix-gas banding (script 30) takes on the variables it claims to fix: the real code block is extracted and executed on a model, tolerance 0 and > 0, and a banded solve stays inside the band")
def g10_2():
    import textwrap
    src = src_of("scripts/30_run_stochastic_redispatch.py")
    tree = ast.parse(src)
    blk = None
    for n in ast.walk(tree):
        if isinstance(n, ast.If) and "args.fix_gas" in ast.get_source_segment(src, n.test) and '"fixed"' in ast.get_source_segment(src, n.test):
            seg = ast.get_source_segment(src, n)
            if "gas_add_bounds" in seg and "setlb" in seg:
                blk = textwrap.dedent(seg)
                break
    if blk is None:
        return FAIL("banding block found in script 30", "not found", "scripts/30_run_stochastic_redispatch.py")
    base = solved(("Sfree",), force=False, eaas=False, arms={"conversion_case": "conv_none"})
    base_gas = [v(base.m.gas_add[t]) for t in base.T]
    if max(base_gas) <= 0:
        base = solved(("Sfree2",), force=True, eaas=False)
        base_gas = [v(base.m.gas_add[t]) for t in base.T]
    out = []
    for tol in (0.0, 1e-5, 0.1):
        # TASK 79 C.1: y is built on the SAME conversion arm as `base` (it was built on the fixture default, conv_central,
        # against a conv_none base: two different models, which passed only because both built zero gas under blended dispatch)
        y = build_fx(force=False, eaas=False, arms={"conversion_case": "conv_none"})
        ns = dict(args=type("A", (), {"fix_gas": "fixed", "fix_tolerance": tol})(), T=y.T, m=y.m, base_gas_add=base_gas, gas_add_bounds={}, i=0, RuntimeError=RuntimeError)
        exec(blk, {"RuntimeError": RuntimeError}, ns)
        okk = True
        for t in y.T:
            lb, ub = ns["gas_add_bounds"][t]
            if tol == 0.0:
                okk &= y.m.gas_add[t].fixed and abs(v(y.m.gas_add[t]) - base_gas[t]) < 1e-12
            else:
                okk &= abs(float(y.m.gas_add[t].lb) - lb) < 1e-12 and abs(float(y.m.gas_add[t].ub) - ub) < 1e-12 and ub - lb <= 2 * tol * base_gas[t] + 1e-9
        solve_fx(y)
        inband = all(ns["gas_add_bounds"][t][0] - 1e-6 <= v(y.m.gas_add[t]) <= ns["gas_add_bounds"][t][1] + 1e-6 for t in y.T)
        out.append((tol, okk, inband))
    ok = all(a and b for _, a, b in out)
    desc = "; ".join(f"tol {t:g}: variable state as claimed {a}, solved value inside band {b}" for t, a, b in out) + f"; base gas build total {sum(base_gas):.1f} MW"
    return PASS(desc) if ok else FAIL("bounds/fixing take; solve inside band", desc, "scripts/30_run_stochastic_redispatch.py (--fix-gas fixed branch)")


@inv("G10-3", "C", "The stochastic sampler is reproducible from its seed, and the draw-set hash is stable (also compared with the stored draw file's hash: a read of that artefact)")
def g10_3():
    import csv as _csv, hashlib as _h
    from src.stochastic import sample_uncertainties
    sc = make_scenario(end_year=2054, capital_case="moderate")
    def draws(seed):
        s = sample_uncertainties(100, sc, 3.9, 0.3, demand_sigma=0.01, seed=seed)
        return [(i, d["demand_growth"], d["gas_scenario"], d.get("carbon_price") if d.get("carbon_active", False) else "") for i, d in enumerate(s)]
    a, b, c = draws(42), draws(42), draws(43)
    def sha(rows):
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "d.csv")
            with open(fp, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["draw_id", "demand_growth", "gas_scenario", "carbon_price"])
                w.writerows(rows)
            return _h.sha256(open(fp, "rb").read()).hexdigest()
    ha, hb, hc = sha(a), sha(b), sha(c)
    stored = ROOT / "results/stochastic/draws__bd65b0bb.csv"
    sh_stored = _h.sha256(stored.read_bytes()).hexdigest() if stored.exists() else None
    ok = a == b and ha == hb and ha != hc
    desc = f"seed 42 twice: identical ({a == b}), sha256 {ha[:16]}; seed 43 differs ({ha != hc}); stored draws__bd65b0bb.csv sha256 {str(sh_stored)[:16]} -> regenerated == stored: {ha == sh_stored}"
    if not ok:
        return FAIL("same seed => same draws and hash; different seed => different", desc, "src/stochastic.py::sample_uncertainties")
    if sh_stored is not None and ha != sh_stored:
        return FAIL("regenerated draws hash equals the stored draw set's hash", desc, "src/stochastic.py / results/stochastic/draws__bd65b0bb.csv",
                    detail="the sampler is seed-reproducible within this code state, but does not reproduce the stored draw file (base-scenario demand_growth, gas weights or numpy state differ)")
    return PASS(desc)


def _highs_flip_table():
    """Re-run the five solver-conditional invariants (and an F-12-style identification) with HiGHS."""
    global ENGINE
    res = {}
    base_status = {k: REGISTRY[k][2]() for k in ("G6-1", "C5", "G11-1", "C7", "B4")}
    _CACHE.clear(); _G8_CACHE.clear()
    ENGINE = "highs"
    try:
        for k in base_status:
            try:
                res[k] = REGISTRY[k][2]()
            except Exception as e:
                res[k] = R("FAIL", f"HiGHS run raised {type(e).__name__}: {str(e)[:120]}")
    finally:
        ENGINE = "cbc"
        _CACHE.clear(); _G8_CACHE.clear()
    return base_status, res


@inv("G10-4", "C", "Solver dependence: do the five CBC-conditional passes (G6-1, C5, G11-1, C7, B4) flip under HiGHS, and does F-12's identification interval change (flat optimal face)?")
def g10_4():
    cbc, hi = _highs_flip_table()
    rows, flips, bitwise = [], [], []
    for k in cbc:
        a, b = cbc[k], hi[k]
        bw = lambda r: ("bitwise True" in r.detail or "bitwise identical: True" in r.detail or "bit-identical" in r.detail) and "NOT bitwise" not in r.detail and "bitwise False" not in r.detail and "bitwise identical: False" not in r.detail
        rows.append((k, a.status, b.status, "bitwise under CBC" if bw(a) else "n/a", "bitwise under HiGHS" if bw(b) else "NOT bitwise under HiGHS"))
        if a.status != b.status:
            flips.append(k)
        if bw(a) and not bw(b):
            bitwise.append(k)
    # F-12-style identification under both solvers on the anchor-scale moderate/EaaS configuration
    import importlib.util
    spec = importlib.util.spec_from_file_location("s26_f3b", ROOT / "scripts" / "26_run_f3_identification.py")
    f3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f3)

    def ident(engine):
        global ENGINE
        ENGINE = engine
        try:
            x = build_fx(eaas=True, force=False, capital_case="moderate", reduced_life=False, end_year=2054)
            solve_fx(x)
            Z = obj(x)
            plan = plan_vec(x)
            out = {}
            for sense, name in ((pyo.minimize, "min"), (pyo.maximize, "max")):
                for nm in ("f3_zstar_con", "f3_eaas_obj"):
                    if x.m.component(nm) is not None:
                        x.m.del_component(nm)
                x.m.objective.deactivate()
                x.m.f3_zstar_con = pyo.Constraint(expr=x.m.system_cost_npv <= Z * (1 + 1e-6))
                x.m.f3_eaas_obj = pyo.Objective(expr=pyo.quicksum(x.m.solar_eaas_add[t] for t in x.T), sense=sense)
                solve_fx(x)
                out[name] = sum(v(x.m.solar_eaas_add[t]) for t in x.T)
                x.m.del_component("f3_eaas_obj"); x.m.del_component("f3_zstar_con"); x.m.objective.activate()
            return Z, plan, out
        finally:
            ENGINE = "cbc"
    Zc, pc, ic = ident("cbc")
    Zh, ph, ih = ident("highs")
    dplan = max(abs(p - q) for p, q in zip(pc, ph))
    desc = "; ".join(f"{k}: CBC {a}, HiGHS {b}, {c} / {d}" for k, a, b, c, d in rows)
    ident_desc = (f"anchor-scale moderate/EaaS base objective CBC {Zc:.4f} vs HiGHS {Zh:.4f} (rel {abs(Zc-Zh)/Zc:.1e}); base-plan max abs difference {dplan:.1f} (MW or MWh); "
                  f"identification at tol 1e-6: CBC EaaS [{ic['min']:.1f}, {ic['max']:.1f}] MW, HiGHS [{ih['min']:.1f}, {ih['max']:.1f}] MW")
    if flips:
        return FAIL("no verdict flips", f"status flips {flips}; bitwise statements lost {bitwise}", "solver vertex selection (CBC vs HiGHS)", detail=desc + ". " + ident_desc)
    return PASS(desc + ". " + ident_desc + (f". Bitwise statement lost under HiGHS (no verdict flips; difference at machine precision): {bitwise}" if bitwise else ""))


@inv("G10-5-frontier", "C", "Frontier non-dominance: dominated points recomputed from the raw (cost, emissions) points per arm and compared with the reported dominated counts (a read of stored frontier files)")
def g10_5():
    import csv as _csv
    raw = list(_csv.DictReader(open(ROOT / "results/pareto_frontier/pareto_raw.csv")))
    dom_rep = list(_csv.DictReader(open(ROOT / "results/pareto_frontier/pareto_dominated_points.csv")))
    by = {}
    for r in raw:
        if r["status"] != "optimal":
            continue
        cf = float(r["cap_fraction"]) if r["cap_fraction"] else 9.9          # 9.9 marks the unconstrained anchor
        by.setdefault(r["arm"], []).append((cf, float(r["npv_cost"]), float(r["cum_emissions"])))
    rep_cnt = {}
    for r in dom_rep:
        rep_cnt[r["arm"]] = rep_cnt.get(r["arm"], 0) + 1
    out, bad = [], []
    for arm, pts in sorted(by.items()):
        dom = 0
        for i, (f, c, e) in enumerate(pts):
            if any((c2 <= c + 1e-6 and e2 <= e + 1e-6) and (c2 < c - 1e-6 or e2 < e - 1e-6) for j, (f2, c2, e2) in enumerate(pts) if j != i):
                dom += 1
        out.append(f"{arm}: {len(pts)} points (grid + tight-end extension + unconstrained anchor), recomputed dominated {dom}, reported {rep_cnt.get(arm, 0)}, non-dominated {len(pts)-dom}")
        if dom != rep_cnt.get(arm, 0):
            bad.append((arm, dom, rep_cnt.get(arm, 0)))
    return PASS("; ".join(out)) if not bad else FAIL("recomputed == reported dominated counts", str(bad), "results/pareto_frontier/pareto_dominated_points.csv / scripts/18_run_pareto_frontier.py", detail="; ".join(out))


@inv("G10-6-sobol", "C", "Sobol: first-order indices sum to ~1 (within the sampling error), total >= first, the solve count equals N(d+2), and the draws file holds exactly that many rows (a read of stored Sobol files)")
def g10_6():
    s = json.load(open(ROOT / "results/parameter_uncertainty/arms_09bd6dbc/summary_n256_seed42.json"))
    n, d, tot = s["n_base"], s["d"], s["total_solves"]
    S1, ST = s["sobol_indices"]["first_order"], s["sobol_indices"]["total_order"]
    lines = sum(1 for _ in open(ROOT / "results/parameter_uncertainty/arms_09bd6dbc/draws_n256_seed42.jsonl"))
    rows = []
    bad = []
    for q, (a, b) in enumerate(zip(S1, ST)):
        rows.append(f"QoI {s['sobol_indices']['qoi_order'][q]}: sum S1 = {sum(a):.4f}, sum ST = {sum(b):.4f}, min(ST-S1) = {min(y - x for x, y in zip(a, b)):.4f}")
        if abs(sum(a) - 1.0) > 0.05:
            bad.append(("sum S1", q, sum(a)))
        if min(y - x for x, y in zip(a, b)) < -0.02:
            bad.append(("ST < S1", q))
    solves_ok = tot == n * (d + 2)
    draws_ok = lines == n * (d + 2)
    ci = s["sobol_indices"].get("first_order_ci95", {})
    desc = "; ".join(rows) + f"; N={n}, d={d}: N(d+2) = {n*(d+2)}, reported total_solves {tot}, draws file rows {lines}"
    if bad or not solves_ok or not draws_ok:
        return FAIL("sum S1 ~ 1, ST >= S1, total_solves == N(d+2) == rows in the draws file", f"index problems {bad}; solves ok {solves_ok}; draws file rows {lines} vs {n*(d+2)}",
                    "scripts/29_run_parameter_uncertainty.py (appends to the draws file: L77)", detail=desc)
    return PASS(desc)


@inv("G10-7-threshold", "C", "Threshold detection (compute_feasibility_threshold): reported threshold is the tightest FEASIBLE grid value, floor-bound flag set when it is the grid's tightest value, grid extent recoverable; checked on synthetic rows and a small real sweep")
def g10_7():
    from src.optimize_experiments import compute_feasibility_threshold
    def rows_for(feas_map):
        return [dict(gas_case="g", solar_build_case="s", reliability_mode="total", eps=e, status=("feasible" if f else "infeasible"),
                     npv_cost=float(100 - e * 10) if f else None, solar_total_mw=1.0, **{k: 0.0 for k in __import__("src.optimize_experiments", fromlist=["DUAL_FLAT_KEYS"]).DUAL_FLAT_KEYS}) for e, f in feas_map.items()]
    msgs, bad = [], []
    # interior threshold
    r1 = compute_feasibility_threshold(rows_for({0.001: False, 0.01: False, 0.05: True, 0.1: True, 0.2: True})).iloc[0]
    ok1 = abs(r1["feasibility_threshold_eps"] - 0.05) < 1e-12 and not r1["threshold_is_floor_bound"]
    # floor bound
    r2 = compute_feasibility_threshold(rows_for({0.001: True, 0.01: True, 0.05: True})).iloc[0]
    ok2 = abs(r2["feasibility_threshold_eps"] - 0.001) < 1e-12 and bool(r2["threshold_is_floor_bound"])
    # fully infeasible
    r3 = compute_feasibility_threshold(rows_for({0.001: False, 0.01: False})).iloc[0]
    ok3 = r3["fully_infeasible"] and r3["feasibility_threshold_eps"] is None
    # real mini-sweep: feasibility must be monotone in eps (loosening cannot create infeasibility) and cost non-increasing
    real = []
    for eps in (0.0, 0.0002, 0.001, 0.01, 0.1):
        xx = solve_scaled(0.2, force=False, eaas=True, builder_kw={"max_offgrid_fraction": eps, "grid_service_mode": "annual"})
        real.append((eps, xx is not None, obj(xx) if xx is not None else None))
    feas = [f for _, f, _ in real]
    mono = all(feas[i] <= feas[i + 1] for i in range(len(feas) - 1))
    costs = [c for _, f, c in real if f]
    cmono = all(costs[i] >= costs[i + 1] - 1e-6 for i in range(len(costs) - 1))
    rr = compute_feasibility_threshold([dict(gas_case="g", solar_build_case="s", reliability_mode="annual", eps=e, status=("feasible" if f else "infeasible"),
                                             npv_cost=c, solar_total_mw=1.0, **{k: 0.0 for k in __import__("src.optimize_experiments", fromlist=["DUAL_FLAT_KEYS"]).DUAL_FLAT_KEYS}) for e, f, c in real if e > 0]).iloc[0]
    finest_feasible = min(e for e, f, _ in real if f and e > 0)
    ok4 = mono and cmono and abs(rr["feasibility_threshold_eps"] - finest_feasible) < 1e-12
    desc = (f"synthetic interior {ok1} (0.05), floor-bound {ok2}, all-infeasible {ok3}; real sweep eps {[e for e, _, _ in real]}: feasibility {feas}, monotone {mono}, cost non-increasing {cmono}, function returns {rr['feasibility_threshold_eps']} (finest feasible {finest_feasible}), floor-bound flag {bool(rr['threshold_is_floor_bound'])}. "
            "Limit: the function reports the tightest feasible value ON THE GRID tested; the grid's extent is recoverable only from the rows (eps_floor), not from the output row except through threshold_is_floor_bound")
    return PASS(desc) if (ok1 and ok2 and ok3 and ok4) else FAIL("tightest feasible grid value, floor flag, monotone feasibility", desc, "src/optimize_experiments.py::compute_feasibility_threshold")



@inv("F1", "C", "Two-builder agreement: where annual and sliced should coincide (conv_none, no solar, no storage build, hydro flat, flat demand shape) the objectives agree; and a configuration where they must differ, with the term that differs")
def f1():
    from src.optimize_model import build_model
    def cfg(end=2036, solar=False):
        sc = make_scenario(end_year=end, arms={"conversion_case": "conv_none"})
        for k in ("season_demand_factor", "period_demand_factor"):
            sc[k] = {a: 1.0 for a in sc[k]}
        if not solar:
            sc["solar_cumulative_cap_mw"] = [0.0] * len(sc["years"])
        econ = load_econ("voll_mid", genset_case="genset_central")
        ys = sc["years"]
        capex = load_solar_capex_by_year(scenario_name="solar_low", start_year=int(ys[0]), end_year=int(ys[-1]))
        return sc, econ, capex

    def both(solar_lb, nostorage):
        sc, econ, capex = cfg(solar=True)
        with contextlib.redirect_stdout(io.StringIO()):
            ms = build_model_sliced(scenario=sc, econ=econ, solar_capex_by_year=capex)
            ma = build_model(scenario=sc, econ=econ, solar_capex_by_year=capex)
        for mod in (ms, ma):
            for t in range(len(sc["years"])):
                if nostorage:
                    mod.storage_add[t].fix(0.0)
                mod.solar_public_add[t].setlb(solar_lb)
                if solar_lb == 0.0:
                    mod.solar_public_add[t].setub(0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            solve_model(ms, scenario=sc); solve_model(ma, scenario=sc)
        terms = {}
        for nm in ("cost_gas_opex_npv", "cost_om_npv", "capital_charge_total_npv", "cost_selfgen_npv", "cost_unserved_voll_npv"):
            if hasattr(ms, nm) and hasattr(ma, nm):
                terms[nm] = (float(pyo.value(getattr(ms, nm))), float(pyo.value(getattr(ma, nm))))
        return float(pyo.value(ms.system_cost_npv)), float(pyo.value(ma.system_cost_npv)), terms
    a, b, terms = both(0.0, True)
    rel0 = abs(a - b) / a
    a2, b2, terms2 = both(3000.0, True)
    rel1 = abs(a2 - b2) / a2
    top = max(terms2.items(), key=lambda kv: abs(kv[1][0] - kv[1][1]))
    desc = (f"coincident configuration (12 years, conv_none, solar cumulative cap 0, storage fixed 0, hydro flat, flat demand shape, f1_tranche): sliced {a:.4f} vs annual {b:.4f}, rel diff {rel0:.1e}; "
            f"contrast configuration (3,000 MW of solar built every year, no storage): sliced {a2:.2f} vs annual {b2:.2f}, rel diff {100*rel1:.2f}%, largest term difference {top[0]} ({top[1][0]-top[1][1]:+.4g} USD)")
    if rel0 > 1e-7:
        return FAIL("agree to 1e-7", f"{rel0:.2e}", "src/optimize_model.py vs src/optimize_model_sliced.py", detail=desc)
    if rel1 < 1e-4:
        return FAIL("the contrast configuration differs (the annual builder cannot value storage/solar timing)", f"{rel1:.2e}", "builders", detail=desc)
    return PASS(desc + ". Tier 1 invariants that would need repeating on the annual builder to close the E_base gap: " + str(len(ANNUAL_REPEAT)) + " (listed in the closing report)")


@inv("F1b", "C", "TASK 53 2.2: the annual builder carries the same capital-window and storage-series coefficients as the sliced builder (every window arm x both storage arms: capital-charge expressions compared coefficient by coefficient, by variable name)")
def f1b():
    from src.optimize_model import build_model
    exprs = ("cost_public_solar_capital_charge_npv", "cost_eaas_solar_capital_charge_npv", "cost_gas_capital_charge_npv",
             "cost_storage_capital_charge_npv", "cost_rebuild_capital_charge_npv")
    worst, rows, bad, ncmp, stor = 0.0, [], [], 0, {}
    for window in ("advance", "arrears", "due_crf"):
        for scase in ("flat", "declining_nrel"):
            sc = make_scenario(end_year=2036, arms={"conversion_case": "conv_none"}, load_kw={"capital_window_case": window, "storage_capex_case": scase})
            sc["asset_lifetimes"] = dict(FIX["lives"]); sc["L_conc"] = FIX["L_conc"]; sc["L_comm"] = FIX["L_comm"]
            econ = load_econ("voll_mid", genset_case="genset_central")
            ys = sc["years"]
            capex = load_solar_capex_by_year(scenario_name="solar_low", start_year=int(ys[0]), end_year=int(ys[-1]))
            with contextlib.redirect_stdout(io.StringIO()):
                ms = build_model_sliced(scenario=sc, econ=econ, solar_capex_by_year=capex)
                ma = build_model(scenario=sc, econ=econ, solar_capex_by_year=capex)
            dev = 0.0
            for nm in exprs:
                cs, ks = _coef_map(getattr(ms, nm)); ca, ka = _coef_map(getattr(ma, nm))
                vs = {id(vv): vv.name for vv in ms.component_data_objects(pyo.Var)}
                va = {id(vv): vv.name for vv in ma.component_data_objects(pyo.Var)}
                bs = {vs[i]: c for i, c in cs.items() if i in vs}
                ba = {va[i]: c for i, c in ca.items() if i in va}
                for k in set(bs) | set(ba):
                    ncmp += 1
                    a_, b_ = bs.get(k, 0.0), ba.get(k, 0.0)
                    if nm == "cost_rebuild_capital_charge_npv" and window == "arrears":
                        stor.setdefault(scase, {})[k] = a_
                    dev = max(dev, abs(a_ - b_) / max(abs(a_), abs(b_), 1e-300))
            worst = max(worst, dev)
            rows.append(f"{window}/{scase}: {dev:.1e}")
            if dev > 1e-12:
                bad.append((window, scase, dev))
            if scase == "declining_nrel" and ma.storage_capex_param[ma.storage_add.index_set().last()] == ma.storage_capex_param[0]:
                bad.append((window, scase, "annual storage_capex_param is flat under declining_nrel"))
    live = ncmp > 0 and stor["flat"] != stor["declining_nrel"]
    if not live:
        bad.append(("vacuous", ncmp, "storage charge coefficients do not differ between flat and declining_nrel"))
    if bad:
        return FAIL("identical coefficients (<= 1e-12)", str(bad), "src/optimize_model.py vs src/optimize_model_sliced.py", detail="; ".join(rows))
    return PASS(f"{ncmp} coefficient pairs compared; storage charge coefficients differ between flat and declining_nrel (so the comparison is live); max relative difference by arm: " + "; ".join(rows))


ANNUAL_REPEAT = ["A2-default", "A2-advance", "A3", "A4", "A5", "G3-1", "G3-2-advance", "G3-3", "A6", "A7", "B1", "B2", "B3", "B4", "B5",
                 "G6-1", "G6-2", "G6-3", "G6-4", "G6-5", "G6-6", "C1", "C2", "C3", "C4", "C5", "C6", "C7",
                 "G5-1", "G5-2", "G5-3", "G5-4", "G5-5", "G11-1", "G11-2", "G11-3", "G11-4", "G2-1", "G2-2", "G2-4", "G1-1", "G1-3"]

S1_ARMS = {"conversion_case": "conv_none", "capital_treatment_case": "pre_f1_overnight", "dscr_case": "dscr_regression",
           "tariff_case": "tariff_legacy_65m", "om_case": "om_none"}
CFGS = {
    "S1 regression anchor": dict(arms=S1_ARMS, capital_case="unconstrained", eaas=False),
    "S2 central anchor": dict(arms={}, capital_case="unconstrained", eaas=False),
    "S3 proposed (tight, EaaS)": dict(arms={}, capital_case="tight", eaas=True),
}


def _mutations():
    def scale(key, f):
        return lambda sc: sc.__setitem__(key, sc[key] * f)
    return [
        ("capital chain (rate)", "r_conc_nominal 2.12% -> 6.12%", dict(sc_over={"r_conc_nominal": 0.0612})),
        ("capital chain (tenor)", "L_comm 10 -> 8", dict(sc_over={"L_comm": 8})),
        ("capital chain (window convention)", "capital_window_case advance -> arrears", dict(window="arrears")),
        ("rebuild / replacement", "storage life 15 -> 30", dict(sc_fn=lambda sc: sc.__setitem__("asset_lifetimes", {**sc["asset_lifetimes"], "storage": 30.0}))),
        ("concessional envelope", "envelope x 0.5", dict(sc_fn=scale("concessional_envelope_usd", 0.5))),
        ("bankability", "tariff x 0.5", dict(sc_fn=scale("solar_service_tariff_usd_per_twh", 0.5))),
        ("bankability (DSCR)", "dscr_case -> dscr_high", dict(arms={"dscr_case": "dscr_high"})),
        ("public budget", "budget x 0.5", dict(sc_fn=lambda sc: sc.__setitem__("public_solar_budget_npv", 0.5 * sc["public_solar_budget_npv"]) if sc.get("public_solar_budget_npv") else None)),
        ("gas deliverability", "A_t x 0.95", dict(gas=0.95)),
        ("gas fuel price (objective)", "GAS_COST_PER_TWH_TH x 1.01", dict(econ_scale=("GAS_COST_PER_TWH_TH", 1.01))),
        ("self-generation price", "GENSET_COST x 1.01", dict(econ_scale=("GENSET_COST_PER_TWH_E", 1.01))),
        ("self-generation emission factor", "GENSET_EF_TH_HHV x 1.5", dict(econ_scale=("GENSET_EF_TH_HHV", 1.5))),
        ("O&M layer", "om_case -> om_low", dict(arms={"om_case": "om_low"})),
        ("horizon", "end year 2054 -> 2053", dict(end_year=2053)),
        ("firm capacity (credit)", "gas credit x 0.9", dict(sc_fn=lambda sc: sc["capacity_credit"].__setitem__("gas", sc["capacity_credit"]["gas"] * 0.9))),
        ("firm capacity (omega)", "omega 1.3448 -> 1.40", dict(sc_over={"within_block_peaking_factor": 1.40})),
        ("reserve margin", "reserve_margin nirp_spinning -> 0.15", dict(arms={"reserve_margin": 0.15})),
        ("demand projection", "demand_case -> next case", dict(demand_alt=True)),
        ("slice weights", "season_demand_factor dry 1.08 -> 1.10", dict(sc_fn=lambda sc: sc.__setitem__("season_demand_factor", {**sc["season_demand_factor"], "dry": sc["season_demand_factor"]["dry"] * 1.02}))),
        ("storage season balance", "round-trip efficiency 0.90 -> 0.85", dict(sc_over={"storage_round_trip_eff": 0.85})),
        ("hydro minimum flow", "min-flow fraction 0.5 -> 0.4", dict(sc_over={"hydro_min_flow_fraction": 0.4})),
        ("hydro energy", "hydro series x 1.01", dict(sc_fn=lambda sc: sc.__setitem__("hydro_energy_twh_by_year", [a * 1.01 for a in sc["hydro_energy_twh_by_year"]]))),
        ("conversion", "conversion_case -> conv_low", dict(arms={"conversion_case": "conv_low"})),
        ("land constraint", "land allowance x 0.5", dict(sc_fn=scale("land_available_km2", 0.5))),
        ("solar build rules", "min build 10 MW/yr", dict(sc_over={"solar_min_build_mw_per_year": 10.0})),
        ("solar capex series", "solar capex x 1.01", dict(capex_scale=1.01)),
        ("storage capex", "storage capex series x 1.01", dict(sc_fn=lambda sc: sc.__setitem__("storage_capex_by_year", {y: c * 1.01 for y, c in sc["storage_capex_by_year"].items()}))),
    ]


def _solve_mut(cfgname, mut):
    c = dict(CFGS[cfgname])
    kw = dict(force=False, reduced_life=False, end_year=2054, eaas=c["eaas"], capital_case=c["capital_case"], arms=dict(c["arms"]))
    ctx = contextlib.nullcontext()
    if mut:
        if "arms" in mut:
            kw["arms"].update(mut["arms"])
        for k in ("sc_over", "sc_fn", "capex_scale", "window", "end_year"):
            if k in mut:
                kw[k] = mut[k]
        if "econ_scale" in mut:
            e0 = load_econ("voll_mid", genset_case="genset_central")
            key, f = mut["econ_scale"]
            if key in e0:
                kw["econ_over"] = {key: e0[key] * f}
        if mut.get("demand_alt"):
            from src import scenarios as S
            names = list(S.demand_growth_scenarios().keys())
            kw["load_kw"] = {"demand_case": next(n for n in names if n != "organic_central")}
        if "gas" in mut:
            ctx = gas_scale(mut["gas"])
    try:
        with ctx:
            x = build_fx(**kw)
        solve_fx(x)
    except Exception as e:
        return None, str(e)[:60]
    return x, None


def _sig(x):
    return obj(x), v(x.m.storage_capacity_mwh[x.Tlast])


@inv("F2", "C", "Guard coverage matrix: for each Tier 1/2 mechanism, which existing regression signature would fail if it broke (a parameter owned by the mechanism is perturbed in each signature's configuration; detection = objective or final storage moves). Anchor-scale solves, non-quotable")
def f2():
    base, rows = {}, []
    for cfg in ("S1 regression anchor", "S2 central anchor"):
        x, err = _solve_mut(cfg, None)
        if x is None:
            return FAIL("base solves", f"{cfg}: {err}", "harness")
        base[cfg] = (x, _sig(x))
    anchor_s1 = abs(base["S1 regression anchor"][1][0] - 21328493384.730568)
    anchor_s2 = abs(base["S2 central anchor"][1][0] - 27317670155.72187)
    matrix = {}
    for mech, label, mut in _mutations():
        cells = {}
        for cfg in ("S1 regression anchor", "S2 central anchor"):
            x, err = _solve_mut(cfg, mut)
            if x is None:
                cells[cfg] = f"infeasible/err ({err})"
                continue
            o, st = _sig(x)
            bo, bs = base[cfg][1]
            if abs(o - bo) / bo > 1e-9 or abs(st - bs) > 1e-6:
                cells[cfg] = "DETECTED"
            else:
                cells[cfg] = "not detected"
        matrix[(mech, label)] = cells
    r = PASS(f"regression anchor reproduces 21,328,493,384.73 (dev {anchor_s1:.2e}); central anchor reproduces 27,317,670,155.72 (dev {anchor_s2:.2e}); {len(matrix)} perturbations")
    r.extra["matrix"] = matrix
    r.extra["anchor_dev"] = (anchor_s1, anchor_s2)
    return r


@inv("F3", "C", "Proposed second regression signature for the f1 path (tight capital, EaaS regime, central arms, 2054): coverage by the same perturbations and its three objective values (one per window convention); anchor-scale, non-quotable")
def f3():
    cfg = "S3 proposed (tight, EaaS)"
    out, cov = {}, {}
    for w in ("advance", "arrears", "due_crf"):
        mut = dict(window=w)
        x, err = _solve_mut(cfg, mut)
        if x is None:
            return FAIL("solves", err, "harness")
        out[w] = (obj(x), v(x.m.storage_capacity_mwh[x.Tlast]), sum(v(x.m.solar_eaas_add[t]) for t in x.T), conc_total(x))
    x0, _ = _solve_mut(cfg, None)
    b0 = _sig(x0)
    for mech, label, mut in _mutations():
        x, err = _solve_mut(cfg, mut)
        cov[(mech, label)] = "infeasible" if x is None else ("DETECTED" if (abs(_sig(x)[0] - b0[0]) / b0[0] > 1e-9 or abs(_sig(x)[1] - b0[1]) > 1e-6) else "not detected")
    must = ["capital chain (rate)", "capital chain (tenor)", "concessional envelope", "public budget", "bankability", "O&M layer", "conversion", "rebuild / replacement"]
    got = {m_: any(v_ == "DETECTED" for (mm, _), v_ in cov.items() if mm == m_) for m_ in must}
    r = PASS("; ".join(f"{w}: objective {o:.2f} USD, 2054 storage {st:.2f} MWh, EaaS {e:.1f} MW, concessional volume {c:.0f} USD" for w, (o, st, e, c) in out.items())
             + f". Mechanisms covered (objective moves): {[k for k, v_ in got.items() if v_]}; not covered: {[k for k, v_ in got.items() if not v_]}")
    r.extra["values"] = out
    r.extra["coverage"] = cov
    return r



# ===========================================================================
# driver
# ===========================================================================
@inv("G12-1", "B", "Supplied concessional-commercial division (TASK 98, src/supplied_division.py): inert when conc_share_rule is None, present in BOTH builders when set, split_zero collapses to a zero envelope, split_prorata exhausts the envelope at its reference plan")
def g12_1():
    from src.optimize_model import build_model
    notes = []
    x0 = build_fx()
    if hasattr(x0.m, "supplied_conc_share"):
        return FAIL("no supplied-division block when conc_share_rule is None", "block present", "src/optimize_model_sliced.py")
    # (b) present in both builders when set
    rule = {"name": "split_zero"}
    x1 = build_fx(sc_over={"conc_share_rule": dict(rule)})
    names = ("supplied_solar_public_conc", "supplied_solar_eaas_conc", "supplied_storage_conc")
    if not all(hasattr(x1.m, n) and len(getattr(x1.m, n)) == len(x1.T) for n in names):
        return FAIL("three supplied constraints over every vintage in the sliced builder", "missing", "src/optimize_model_sliced.py")
    # the annual builder does not implement conversion: build it on the conv_none scenario, as the G7-3 reachability check does
    sca = make_scenario(arms={"conversion_case": "conv_none"})
    sca["capital_window_case"] = "advance"
    with contextlib.redirect_stdout(io.StringIO()):
        ma = build_model(scenario={**sca, "conc_share_rule": dict(rule)}, econ=x1.econ, solar_capex_by_year=x1.capex)
        ma0 = build_model(scenario={**sca, "conc_share_rule": None}, econ=x1.econ, solar_capex_by_year=x1.capex)
    if not all(hasattr(ma, n) for n in names) or hasattr(ma0, "supplied_conc_share"):
        return FAIL("annual builder: constraints present when set, absent when inert", f"present {[hasattr(ma, n) for n in names]}, inert-has-block {hasattr(ma0, 'supplied_conc_share')}", "src/optimize_model.py")
    notes.append("both builders: 3 constraints x %d vintages when set, none when inert" % len(x1.T))
    # (c) split_zero collapses to a zero envelope
    xe = solve_fx(build_fx(sc_over={"concessional_envelope_usd": 0.0}))
    xz = solve_fx(x1)
    oe, oz = obj(xe), obj(xz)
    conc_z = sum(v(x1.m.solar_public_conc_vol[t]) + v(x1.m.solar_eaas_conc_vol[t]) + v(x1.m.storage_conc_vol[t]) for t in x1.T)
    rel = abs(oz - oe) / oe
    if rel > 1e-7 or conc_z > 1e-6:
        return FAIL("split_zero objective equals the zero-envelope objective to 1e-7 with no concessional volume", f"rel {rel:.2e}, concessional volume {conc_z:.3g}", "src/supplied_division.py")
    notes.append(f"split_zero vs zero envelope: rel diff {rel:.1e}")
    # (d) split_prorata exhausts the envelope at its reference plan
    xr = solve_fx(build_fx())
    ref = {"solar_add_mw_by_year": {int(y): v(xr.m.solar_public_add[t]) + v(xr.m.solar_eaas_add[t]) for t, y in enumerate(xr.years)},
           "storage_add_mwh_by_year": {int(y): v(xr.m.storage_add[t]) for t, y in enumerate(xr.years)}}
    xp = build_fx(sc_over={"conc_share_rule": {"name": "split_prorata", "reference": ref}})
    r = xp.sc["conc_share_rule"]
    s_, k_ = r["_calibrated_share"], r["_reference_eligible_capex_usd"]
    E = float(xp.sc["concessional_envelope_usd"])
    if abs(s_ * k_ / E - 1.0) > 1e-9 or not (0 < s_ <= 1):
        return FAIL("share x reference eligible capital == envelope, share in (0, 1]", f"share {s_:.6g}, share*K/E {s_ * k_ / E:.12g}", "src/supplied_division.py")
    notes.append(f"split_prorata share {s_:.4f} exhausts the envelope at the reference plan")
    return PASS("; ".join(notes))


# Passes whose verdict is "bitwise identical" and were obtained on CBC alone (TASK 52 addendum 4.2);
# G10-4 reports whether any flips under HiGHS.
SOLVER_CONDITIONAL = {"G6-1", "C5", "G11-1", "C7", "B4"}

STAGE_A_ORDER = ["G7-1", "G7-3", "G7-2", "G7-4",
                 "A1", "A2-default", "A2-advance", "A2-arrears", "A2-due_crf", "A3", "A4", "A5", "G3-1", "G3-2-advance", "G3-2-arrears", "G3-2-due_crf", "G3-3",
                 "A6", "A7",
                 "B1", "B2", "B3", "G6-1", "G6-2", "G6-3", "G6-4", "G6-5", "G6-6",
                 "B4", "B5",
                 "C1", "C2", "C3", "C7", "G5-1", "G5-2", "G5-3", "G5-4", "G5-5",
                 "C4", "C5", "C6",
                 "G11-1", "G11-2", "G11-3", "G11-4",
                 "G2-1", "G2-2", "G2-3", "G2-4",
                 "G1-1", "G1-2", "G1-3", "G1-3s", "G1-3m"]


def ordered(stage):
    if stage == "A":
        missing = [k for k in STAGE_A_ORDER if k not in REGISTRY]
        extra = [k for k, (st, _, _) in REGISTRY.items() if st == "A" and k not in STAGE_A_ORDER]
        assert not missing and not extra, (missing, extra)
        return STAGE_A_ORDER
    return [k for k, (st, _, _) in REGISTRY.items() if st == stage]


def write_detail(results, outdir, stage="A"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    L = [f"# TASK 52 Stage {stage} -- detail tables (generated by scripts/52_invariants.py --stage {stage} --detail)", ""]
    _e4 = results.get("E4") if isinstance(results, dict) else None
    if _e4 is not None and "e4_table" in getattr(_e4, "extra", {}):
        L += ["## E4 -- basis of every reported dual and objective-derived NPV (TASK 53 2.3)", "", _e4.extra["e4_table"], ""]
    r = results.get("G9-1")
    if r and "rows" in r.extra:
        L += ["## G9-1 exogenous series (30-year central scenario): length, range", ""] + [f"- {x}" for x in r.extra["rows"]] + [""]
    r = results.get("G7-1")
    if r and "rows" in r.extra:
        L += ["## G7-1 scenario.get(k, default) in both builders", "", "| builder | line | key | default | in load_scenario | load_scenario value | flag |", "|---|---|---|---|---|---|---|"]
        L += [f"| {b} | {ln} | `{k}` | `{d}` | {ins} | `{sv}` | {fl} |" for b, ln, k, d, ins, sv, fl in r.extra["rows"]]
        L.append("")
    r = results.get("G7-3")
    if r and "rows" in r.extra:
        live_arms = {}
        r2 = results.get("G7-2")
        if r2 and "rows" in r2.extra:
            live_arms = {n: st for n, vals, st, res in r2.extra["rows"] if st.startswith("LIVE")}
        L += ["## G7-3 live / inert per builder", "",
              "> **Static classifications (reachability of `scenario[...]`/`.get(...]` in the builder and its helpers) are SUPERSEDED by G7-2 (the dynamic test: does the built model change?) wherever the two disagree.** "
              "A key consumed inside `load_scenario` (an arm) is invisible to the static read. Rows corrected this way are marked SUPERSEDED.", "",
              "| key | class | also read in |", "|---|---|---|"]
        corrected = []
        for k, c, e in r.extra["rows"]:
            if k in live_arms and ("inert in both" in c or "DEAD" in c):
                corrected.append(k)
                c = f"~~{c}~~ **SUPERSEDED by G7-2: {live_arms[k]}** (consumed inside `load_scenario`; changes the built model)"
            L.append(f"| `{k}` | {c} | {e} |")
        L += ["", f"Rows corrected from the static label by G7-2: {corrected or 'none'}. Rows still classed DEAD (inert in both builders and read nowhere else; not an arm that changes the model): "
              + ", ".join(f"`{k}`" for k, c, e in r.extra["rows"] if "DEAD" in c and k not in live_arms) + "."]
        L += ["", f"Keys the builders read that `load_scenario` never sets: {r.extra.get('builder_only_not_in_scenario')}",
              f"Dynamic (non-literal) scenario key access seen in reached code: {r.extra.get('dynamic_access')}", ""]
    r = results.get("G7-2")
    if r and "rows" in r.extra:
        L += ["## G7-2 arms: does the built model change?", "", "| arm | values | verdict | distinct LP files / built / tried per context |", "|---|---|---|---|"]
        L += [f"| `{n}` | {vals} | {st} | {res} |" for n, vals, st, res in r.extra["rows"]]
        L.append("")
    r = results.get("G7-4")
    if r and "rows" in r.extra:
        L += ["## G7-4 runner output keys", "", "| runner | has hash/tag key | arm flags | in key | NOT in key |", "|---|---|---|---|---|"]
        L += [f"| {n} | {k} | {nf} | {len(i)} | {', '.join(o) if o else '-'} |" for n, k, nf, i, o in r.extra["rows"]]
        L.append("")
    r = results.get("A6")
    if r and "shares" in r.extra:
        L += ["## A6 share of the objective (reduced fixture, forced builds: NOT the anchor's shares)", ""]
        L += [f"- {k}: {100*a:.2f}%" for k, a in r.extra["shares"].items()]
        L += ["", "Capital charge components:"] + [f"- {k}: {100*a:.2f}% of the objective" for k, a in r.extra["capital_parts"].items()] + [""]
    r = results.get("A7")
    if r and "families" in r.extra:
        L += ["## A7 objective coefficient by variable family (min, max, non-zero, count)", ""]
        L += [f"- `{k}`: min {lo:.6g}, max {hi:.6g}, non-zero {nz}/{N}" for k, (lo, hi, nz, N) in r.extra["families"].items()] + [""]
    r = results.get("E1")
    if r and "rows" in r.extra:
        L += ["## E1 unit-conversion literals", "", "| builder | line | literal | source line | identity asserted | closes |", "|---|---|---|---|---|---|"]
        L += [f"| {b} | {ln} | {val} | `{src}` | {ident} | {ok} |" for b, ln, val, src, ident, ok in r.extra["rows"]] + [""]
    r = results.get("E2")
    if r and "rows" in r.extra:
        L += ["## E2 constraint families", "", "| constraint | builder(s) | left side | right side | closes | note |", "|---|---|---|---|---|---|"]
        L += [f"| `{k}` | {b} | {l} | {rr} | {ok} | {nt} |" for k, b, l, rr, ok, nt in r.extra["rows"]] + [""]
    r = results.get("F2")
    if r and "matrix" in r.extra:
        L += ["## F2 guard coverage: perturbation detected by the signature?", "", "(S1 = regression anchor, pre_f1_overnight; S2 = central anchor; DETECTED = objective or final storage moved; anchor-scale solves, non-quotable)", "",
              "| mechanism | perturbation | S1 regression anchor | S2 central anchor |", "|---|---|---|---|"]
        L += [f"| {m_} | {lab} | {c['S1 regression anchor']} | {c['S2 central anchor']} |" for (m_, lab), c in r.extra["matrix"].items()] + [""]
    r = results.get("F3")
    if r and "values" in r.extra:
        L += ["## F3 proposed signature S3 (tight capital, EaaS regime, central arms, 2025-2054)", "", "| window convention | objective (USD) | 2054 storage (MWh) | EaaS MW | concessional volume (USD) |", "|---|---|---|---|---|"]
        L += [f"| {w} | {o_:.2f} | {st:.2f} | {e:.1f} | {c:.0f} |" for w, (o_, st, e, c) in r.extra["values"].items()]
        L += ["", "Coverage of S3 by the same perturbations:", ""] + [f"- {m_} ({lab}): {v_}" for (m_, lab), v_ in r.extra["coverage"].items()] + [""]
    r = results.get("B4")
    if r and "rows" in r.extra:
        L += ["## B4 dead-coefficient table", ""] + [f"- {a}: **{b}** (disabled by: {c})" for a, b, c in r.extra["rows"]] + [""]
    (outdir / f"stage{stage}_detail.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=["A", "B", "C"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--detail", default=None, help="directory for the generated detail tables")
    a = ap.parse_args()
    if a.list:
        for st in ("A", "B", "C"):
            ids = ordered(st)
            for k in ids:
                print(f"{st}  {k:14s} {REGISTRY[k][1]}")
            if not ids:
                print(f"{st}  (no invariants built yet: stage not started)")
        return 0
    if not a.stage:
        ap.error("--stage or --list required")
    ids = ordered(a.stage)
    if a.only:
        want = set(a.only.split(","))
        ids = [k for k in ids if k in want]
    if not ids:
        print(f"Stage {a.stage}: nothing built (hard stop: stages are built one at a time).")
        return 0
    print(f"solver: {SOLVER_NOTE}")
    print(f"fixture: {fixture_dims()}")
    results = {}
    t_all = time.time()
    for k in ids:
        stage, text, fn = REGISTRY[k]
        t0 = time.time()
        try:
            with warnings_ignored():
                r = fn()
        except Exception as e:
            r = R("FAIL", f"HARNESS EXCEPTION {type(e).__name__}: {e}", "no exception", f"{type(e).__name__}: {e}", "scripts/52_invariants.py::" + k,)
            r.extra["trace"] = traceback.format_exc(limit=4)
        results[k] = r
        lab = r.status + ("*" if (r.status == "PASS" and k in SOLVER_CONDITIONAL) else "")
        print(f"{k:14s} {lab:6s} {time.time()-t0:5.1f}s  {text[:90]}")
    c = {s: sum(1 for r in results.values() if r.status == s) for s in ("PASS", "FAIL", "SKIP")}
    if any(k in SOLVER_CONDITIONAL for k in results):
        print("(* = PASS is solver-conditional: bitwise-identity verdict obtained on CBC 2.10.13 only; HiGHS not run)")
    print(f"\nSTAGE {a.stage}: {c['PASS']} PASS, {c['FAIL']} FAIL, {c['SKIP']} SKIP in {time.time()-t_all:.0f}s")
    for k, r in results.items():
        if r.status != "PASS":
            print(f"\n[{r.status}] {k}: {REGISTRY[k][1]}")
            for lab, val in (("expected", r.expected), ("observed", r.observed), ("where", r.where), ("detail", r.detail)):
                if val:
                    print(f"   {lab}: {val}")
    if a.detail:
        write_detail(results, a.detail, a.stage)
        print(f"detail tables: {a.detail}/stage{a.stage}_detail.md")
    import json
    (ROOT / "scratch_52_results.json").write_text("") if False else None
    return 0


@contextlib.contextmanager
def warnings_ignored():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


if __name__ == "__main__":
    sys.exit(main())
