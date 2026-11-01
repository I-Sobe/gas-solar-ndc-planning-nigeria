# TASK — two Pareto frontiers by ε-constraint

Branch `phase-fin2-d17`. Read `scripts/18_run_pareto_frontier.py` in
full (351 lines, current state — it was repaired for argument-plumbing
in `task_runner_repair` but never validated as an ε-constraint method).
Per TASK.md §0, this is Part 1 only: inventory, then STOP.

---

## PART 1 — INVENTORY

### 1A — Current sweep: how many points, over what range, how generated

```python
# line 66
N_POINTS = 25
...
# lines 136-138
# ── Generate cap levels ───────────────────────────────────
# Sweep from 100% of baseline (unconstrained) to 10% of baseline
cap_fractions = np.linspace(1.0, 0.10, N_POINTS)
```

**25 points**, linearly spaced (`np.linspace`) from **100% down to
10%** of `baseline_annual` — the baseline's own annual emissions,
itself computed at lines 121-127 as
`baseline_total / n_years` from `results/baseline/diagnostics.json`
(read at line 118, produced by `01_run_baseline.py` per the module's
own `PRE-REQUISITES`). Swept once, identically, across each of the
four `FINANCING_ARMS` (line 69-94: `public_only_tight`, `eaas_tight`,
`public_only_moderate`, `eaas_moderate`) — so 25 × 4 = 100 solves per
invocation (`N_POINTS * len(FINANCING_ARMS)`, line 134).

The cap itself, per point, is built at lines 148-150:
```python
annual_cap = baseline_annual * frac
cap_by_year = [annual_cap] * n_years
```
— a **flat** annual value (same number repeated for every model year),
not a genuinely time-varying trajectory. See 1E.

### 1B — Does it report the dual on the emissions cap constraint?

**No.** Grepped the whole file for `dual`, `shadow`, `m.dual` —
zero occurrences. The row dict built at lines 227-254
(`all_rows.append({...})`) carries `npv_cost`, `real_expenditure`,
`voll_penalty`, `cum_emissions`, `cum_unserved`, `solar_total_mw`,
`storage_mwh`, `system_lcoe`, `gas_share_2030`, `solar_share_2045`,
`leakage_share`, `leakage_flagged` — no dual, no shadow price, no
marginal abatement cost anywhere.

**A function in the tree already extracts exactly this dual.**
Checked the named precedent, `run_rel2_marginal_cost_curves`
(`src/optimize_experiments.py`): it calls `run_reliability_sweep`,
which reads the *reliability* constraint's dual via
`extract_reliability_dual(m, diag=diag)` (`src/optimize_experiments.
py:347`, called at line 614) — a different constraint (the grid
service obligation), not emissions. Reading it established the
pattern (`m.dual.get(<constraint>, None)`, sign-converted, clamped
non-negative, with a `DUAL_SIGN_TOL` guard against a wrong sign
convention) and led to the actual match, in the same file:

```python
# src/optimize_experiments.py:2260-2297
def extract_carbon_shadow_prices_usd_per_tco2(m, years):
    """
    Returns year -> USD/tCO2 (non-negative by convention).
    Handles both:
      - annual caps: indexed m.emissions_constraint[t]
      - single cap: scalar m.emissions_constraint
    """
    years = list(years)
    if not hasattr(m, "emissions_constraint"):
        return {int(y): None for y in years}
    if m.emissions_constraint.is_indexed():
        out = {}
        for t, y in enumerate(years):
            dual = m.dual.get(m.emissions_constraint[t], None)
            val = None if dual is None else float(-dual)
            if val is not None and val < -DUAL_SIGN_TOL:
                raise ValueError(...)
            out[int(y)] = None if val is None else max(0.0, val)
        return out
    dual = m.dual.get(m.emissions_constraint, None)
    val = None if dual is None else float(-dual)
    if val is not None and val < -DUAL_SIGN_TOL:
        raise ValueError(...)
    val = None if val is None else max(0.0, val)
    return {int(y): val for y in years}
```

**This is called ALREADY, inside `extract_planning_diagnostics`**
(`src/optimize_experiments.py:2340`, `carbon_shadow =
extract_carbon_shadow_prices_usd_per_tco2(m, years)` at line 2385,
returned as `"carbon_shadow_price_usd_per_tco2_by_year"` at line
3171) — and **`18_run_pareto_frontier.py` already calls that exact
function**, at line 193 (`diag = extract_planning_diagnostics(m,
scenario, econ)`). The dual is computed and sitting in `diag` on
every single one of the 100 solves this script already performs; the
script simply never reads `diag["carbon_shadow_price_usd_per_tco2_
by_year"]` out of the dict it already has. No new extraction machinery
is needed for Part 2 — only reading a field that already exists.

`m.emissions_constraint` here will be the **indexed** form (per-year
`m.emissions_constraint[t]`), since `18_` calls `build_model(...,
emissions_cap_by_year=cap_by_year, ...)` — a list, not a scalar.

### 1C — Does it check whether the cap binds? What happens when slack?

**No binding check exists anywhere in the file.** No `slack`,
`binding`, or tolerance comparison of any kind. At a point where the
cap does not bind (e.g. `frac` close to 1.0, at or above the
unconstrained optimum's own emissions), the script solves the LP,
gets a status of `"optimal"`, and appends the row **identically** to
a genuinely binding point — same fields, same treatment, no flag
distinguishing "this is the frontier" from "this is the unconstrained
optimum, reported again." The 25-point sweep as it stands cannot
currently tell a reader where the frontier actually starts.

### 1D — What happens at infeasibility? Catch-record-continue, or abort?

**Catch, record, and continue — the sweep does not abort.** Two
layers:

```python
# lines 181-191, inside the try block
status = solve_model(m, scenario=scenario)
if not status["optimal"]:
    all_rows.append({
        "arm": arm["label"], "cap_fraction": frac,
        "annual_cap_tco2": annual_cap,
        "status": "infeasible",
        "leakage_share": None, "leakage_flagged": None,
    })
    print(f"    {frac:.0%} cap -> INFEASIBLE")
    continue
```
and, wrapping the whole per-point body:
```python
# lines 261-267
except Exception as e:
    all_rows.append({
        "arm": arm["label"], "cap_fraction": frac,
        "annual_cap_tco2": annual_cap, "status": f"error: {e}",
        "leakage_share": None, "leakage_flagged": None,
    })
    print(f"    {frac:.0%} cap -> ERROR: {e}")
```
Both record a row with a `status` field distinguishing `"optimal"` /
`"infeasible"` / `f"error: {e}"`, and both `continue` the loop rather
than aborting. This part of TASK.md's requirement (infeasibility:
record and continue) is **already satisfied** by the current code —
Part 2 does not need to add this, only preserve it.

### 1E — How is the cap set? Quote the mechanism

```python
# lines 148-150
annual_cap = baseline_annual * frac
cap_by_year = [annual_cap] * n_years
```
passed to the builder at:
```python
# lines 175-180
m = build_model(
    scenario=scenario,
    econ=econ,
    emissions_cap_by_year=cap_by_year,
    solar_capex_by_year=solar_capex_tv,
)
```
**Mechanism: `emissions_cap_by_year`, a flat (constant-across-years)
annual cap, expressed as a fraction of the baseline's own annual
emissions.** Not a scalar cumulative budget, not a genuinely
time-varying trajectory — a per-year list whose every entry is
identical. Frontier A (Part 2) should keep this exact mechanism
(TASK.md §Part 2: "do not change what it solves"). Frontier B's own ε
(Part 3) is a different constraint entirely (`public_solar_budget_
npv`, a single NPV scalar, not a `cap_by_year`-shaped list — see 1F) —
TASK.md §1E's instruction that "the second frontier must use the same
constraint-setting approach for its own ε" is read here as: same
*discipline* (a swept scalar fed straight to `load_scenario()`/
`build_model()` via the same dict-key mechanism already used
elsewhere in this codebase, not a new indirection layer) — not
literally the same `emissions_cap_by_year` list shape, since the
public budget constraint is not year-indexed (confirmed below, 1F).

### 1F — Capital envelope: where `capital_case` resolves to a number, and whether it can be set directly

**Resolution site**, `src/scenarios.py`:
```python
# line 783-801
def capital_envelope_scenarios():
    B_star = 9_104_000_000
    return {
        "tight": int(0.50 * B_star),        # 4,552,000,000
        "moderate": int(0.85 * B_star),     # 7,738,400,000
        "adequacy": int(1.00 * B_star),     # 9,104,000,000
        "expansion": int(1.20 * B_star),    # 10,924,800,000
        "unconstrained": None,
    }
```
```python
# line 3060, inside load_scenario()
"public_solar_budget_npv": capital_envelope_scenarios()[capital_case],
```
`load_scenario(capital_case=...)` writes the resolved number (or
`None`, for `"unconstrained"`) into the returned scenario dict under
the plain key `"public_solar_budget_npv"` — no locking, no closure
capture, an ordinary mutable dict entry.

**Consumption site**, `src/optimize_model.py:1691`:
```python
public_budget_npv = scenario.get("public_solar_budget_npv", None)
if public_budget_npv is not None:
    m.public_budget_constraint = pyo.Constraint(
        expr=pyo.quicksum(
            m.DF[t] * (
                m.solar_public_add[t] * m.solar_capex_param[t]
                + m.storage_add[t] * econ["STORAGE_COST_PER_MWH"]
                + m.eaas_subsidy[t]
            ) for t in T
        ) <= public_budget_npv
    )
```
Read directly via `scenario.get(...)` **at `build_model()` call
time** — whatever sits in the dict at that moment is what is used.
**`public_budget_npv is None` (the `"unconstrained"` arm) skips
building `m.public_budget_constraint` entirely** — the constraint is
structurally absent, not merely a very large bound; any dual
extraction must guard with `hasattr(m, "public_budget_constraint")`.

**Yes — it can be set directly, to an arbitrary value, bypassing the
named arm, with no edit outside this task's whitelist.** Confirmed by
the established, already-used pattern in `18_run_pareto_frontier.py`
itself: the script already mutates the scenario dict AFTER
`load_scenario()` returns and BEFORE `build_model()` is called
(`scenario["financing_regime"] = arm["financing_regime"]`, `scenario[
"required_margin"] = arm["required_margin"]`, lines 164-165). The
identical pattern — `scenario["public_solar_budget_npv"] = <arbitrary
float or None>` inserted between the `load_scenario()` call and the
`build_model()` call — sets Frontier B's ε directly, sweeping through
and past the four named arms' values rather than being confined to
them. **This is not a STOP condition; the task's shape does not
change.**

**Supporting context found while confirming this (not required by
1F, but directly relevant to Part 3's own primary output):** the dual
on `m.public_budget_constraint` is *also* already extracted, inside
the same `extract_planning_diagnostics` function `18_` already calls,
guarded correctly on `hasattr`:
```python
# src/optimize_experiments.py:2762-2775
if hasattr(m, "public_budget_constraint"):
    raw_dual = m.dual.get(m.public_budget_constraint, None)
    if raw_dual is not None:
        _pub_budget_val = float(-raw_dual)
        if _pub_budget_val < -DUAL_SIGN_TOL:
            raise ValueError(...)
        public_budget_shadow = max(0.0, _pub_budget_val)
```
exposed as `diag["public_budget_shadow_usd_per_usd"]` (and
`"public_budget_utilisation"`, `"public_budget_realised_spend_usd"`,
lines 3195-3197). Frontier B's "primary output" (TASK.md §Part 3) is
therefore also already-extracted, already-present in a dict `28_`
will call in exactly the same way `18_` already does — noted here so
Part 3, when released, does not need to rediscover it.

---

**Not covered above, correctly out of Part 1's scope, noted for
Part 3 when it is released:** whether `28_run_capital_frontier.py`
should call `build_model()` directly (mirroring `18_`'s own structure,
per TASK.md §Part 3's explicit instruction "Follow `18_`'s structure")
or delegate to a shared function — TASK.md's whitelist forbids editing
`src/optimize_experiments.py` in this task, so any new logic Part 3
needs must live inside `28_run_capital_frontier.py` itself, calling
only what already exists (`extract_planning_diagnostics`,
`build_model`, `solve_model`, `load_scenario`, `add_standard_arms`/
`standard_arms_kwargs`) — consistent with how `18_` itself is
structured.

---

## PARTS 2-4 — IMPLEMENTATION

Released and complete. Whitelist for this release: edit ONLY
`scripts/18_run_pareto_frontier.py`; create ONE new file,
`scripts/28_run_capital_frontier.py`; append to this report. No
builder, no `src/scenarios.py`, no `src/optimize_experiments.py`
edited — both frontiers read existing, already-computed fields only
(Part 1's central finding: both duals already exist).

---

### FRONTIER A — cost versus emissions (`scripts/18_run_pareto_frontier.py`)

**The sweep — unchanged.** 25 points, `np.linspace(1.0, 0.10, 25)`, a
fraction of baseline annual emissions, across the same 4
`FINANCING_ARMS`, feeding the same `emissions_cap_by_year` mechanism.
Confirmed by re-reading the whole file after this edit: `N_POINTS`,
`cap_fractions`, `FINANCING_ARMS`, and the `build_model(...,
emissions_cap_by_year=cap_by_year, ...)` call are byte-for-byte
unchanged from Part 1's inventory. TASK.md's own instruction ("do not
change what it solves") is satisfied by construction, not merely by
intent.

**The dual — read, not extracted.** Per point:
```python
carbon_shadow_by_year = diag["carbon_shadow_price_usd_per_tco2_by_year"]
```
`diag` is the SAME `extract_planning_diagnostics(m, scenario, econ)`
call the script already made before this edit (line unchanged). No
new extractor, per Part 1's own finding. The full year-by-year series
is kept, not discarded — stored per row as
`carbon_shadow_by_year_json` (a JSON-encoded flat CSV field, matching
this project's existing convention for a per-year series inside a CSV
row — `15_run_pol1_ndc_comparison.py`'s `solar_add_by_year_json` is
the precedent).

**The scalar reduction — three computed, binding-year mean used for
the frontier table, per Zeus's stated preference.** `None` entries
(the extractor's own "no dual available" signal) are excluded from
every reduction, never treated as zero; if every year in a point is
`None`, the reduction is `None`, not a fabricated number.
- `carbon_shadow_usd_per_tco2_binding_mean` — mean over years where
  the cap binds (same per-year tolerance as the point's own
  `cap_binding` flag, below — a year is "binding" for the dual
  reduction iff it is binding for the point). **Used for the frontier
  table.** Argument: a non-binding year's dual carries no marginal-
  abatement-cost meaning at that cap level and would drag a simple
  mean toward a number that answers no real question.
- `carbon_shadow_usd_per_tco2_simple_mean` — mean over all non-`None`
  years, binding or not. Reported alongside (cheap to compute).
- `carbon_shadow_usd_per_tco2_discounted_mean` — `Σ DF[t]·value[t] /
  Σ DF[t]` over non-`None` years, `m.DF[t]` the model's own discount
  factor. Reported alongside.

**The binding check.** Per-year slack, `annual_cap − emissions[y]`,
**minimum across years** (not an aggregate/average): this project has
already found (`extract_reliability_dual`'s own comment, for the grid
service obligation) that an aggregate test can read "not binding"
while individual years sit pinned exactly on the bound — the same
trap, avoided the same way, for this constraint. Tolerance:
```python
CAP_BINDING_REL_TOL = 0.01  # 1% of the annual cap
cap_tolerance_tco2 = CAP_BINDING_REL_TOL * annual_cap
```
**Relative to the cap, not absolute** — TASK.md's own instruction,
citing D1's fix to the interaction classifier: an absolute-tCO2
tolerance would fail across this sweep's own ten-fold range (10% to
100% of baseline). `cap_binding = cap_slack_tco2 <= cap_tolerance_tco2`.
Achieved cumulative emissions (`cum_emissions`, pre-existing column)
and the cumulative cap (`cumulative_cap_tco2`, new) are both reported
per point, so a reader can see directly that they coincide where the
cap binds.

**Endpoint characterisation, per financing arm.** Printed and saved to
`pareto_endpoints.json`:
- `frontier_begins_at_cap_fraction` — the largest `cap_fraction` at
  which `cap_binding` is still `True`. Above it, points are the
  unconstrained optimum, reported repeatedly, not frontier points.
- `unconstrained_optimum_emissions_tco2` — `cum_emissions` at the
  loosest non-binding point.
- `tightest_feasible_cap_fraction` / `loosest_infeasible_cap_fraction`
  — bound the infeasibility edge from the swept data actually obtained
  (no infeasible point may exist within `[0.10, 1.0]`; that is
  reported as `loosest_infeasible_cap_fraction = None` with an
  explicit "none found >= 0.10" note, not assumed away).
- `binding_non_monotone` — a self-check: is there a point LOOSER than
  the reported threshold that still binds, or one TIGHTER that does
  not? Flagged, not silently assumed impossible.

**Non-dominated check.** Standard Pareto dominance (lower-or-equal
cost AND lower-or-equal emissions, at least one strict) checked
pairwise across every `optimal` point per arm — cheap at 25 points.
Any dominated point is written to `pareto_dominated_points.csv` and
printed with both points' cost/emissions, per TASK.md's instruction
that a dominated point is diagnostic (solver tolerance or a
non-binding cap) either way, not silently dropped.

**Flag, don't drop.** Every row (including non-binding, infeasible,
and error rows) stays in `pareto_raw.csv`, `cap_binding` present as
its own column (or `None` where not applicable). The plot draws
BINDING points as the solid, connected frontier line and NON-BINDING
points as hollow, unconnected markers, with the axis title stating the
convention directly (`"hollow markers: cap not binding -- not a
frontier point"`). The "generation mix at key cap levels" console
table (previously titled "OPTIMAL... AT KEY EMISSIONS LEVELS", now
retitled to state the flag convention) marks any non-binding row with
`*` rather than presenting it as an unflagged frontier point —
material at `cap_fraction=1.0` specifically, which is very likely
non-binding for every arm (it is essentially the unconstrained
baseline).

**Command Zeus should run** (central arm configuration, the one that
reproduces the frozen anchors — `docs/findings_frozen.md`, "The
central configuration" — read live from that document, not
reconstructed from memory):
```bash
python scripts/18_run_pareto_frontier.py \
  --peak-case peak_unsuppressed \
  --storage-cc-case storage_cc_central \
  --gas-build-case gas_unconstrained \
  --new-gas-deliverability constrained \
  --gas-availability-case gas_avail_nameplate \
  --gas-retirement-case gas_retire_vintage_30 \
  --gas-capex-case gas_capex_central \
  --ccgt-efficiency-case ccgt_eff_central \
  --capital-recovery-case cr_central \
  --dscr-case dscr_central \
  --us-inflation 0.020 \
  --capital-treatment-case f1_tranche \
  --concessional-envelope-case env_dares \
  --gas-deliverability-case baseline \
  --conversion-case conv_central \
  --hydro-case hydro_flat \
  --tariff-case tariff_central \
  --om-case om_central
```
Pre-requisites unchanged: `01_run_baseline.py` and
`00_build_emissions_cap.py` must have already been run (this script's
own module docstring; not touched by this task).

`python3 -m py_compile scripts/18_run_pareto_frontier.py` — passes.

---

### FRONTIER B — cost versus public capital (`scripts/28_run_capital_frontier.py`, new file)

**The sweep.** 18 points per financing regime (17 finite envelope
fractions of `B_star` plus `None`/unconstrained), **2 financing
regimes as SEPARATE frontiers** (`traditional`, `eaas`, per TASK.md
3.3 — not one sweep tagged by regime) = 36 solves. `B_star` and the
four named envelope values (`tight`/`moderate`/`adequacy`/
`expansion`) are read LIVE from `capital_envelope_scenarios()` (import
only — `src/scenarios.py` is not edited), confirmed at implementation
time to exactly match Part 1F's quoted figures:
```
tight          4,552,000,000
moderate       7,738,400,000
adequacy       9,104,000,000
expansion     10,924,800,000
unconstrained  None
```
Grid (fractions of `B_star`):
```python
CAPITAL_ENVELOPE_FRACTIONS = [
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.60, 0.70,
    0.85, 0.90, 1.00, 1.10, 1.20,
    None,
]
```
Chosen: **0.00** is the lower bound — the literal floor (zero public
capex/subsidy funded), not an arbitrary negative value, which would be
nonsensical for a capital envelope. Below `tight` (0.05 to 0.45), step
0.05 — ten points, enough resolution to locate an infeasibility edge
if one exists without knowing in advance where. At and just above
`tight` (0.50, 0.60, 0.70), **finer** spacing than the rest of the
sweep, deliberately: this is where Part 3.6's predicted kink (the
bankability ceiling starting to bind) is most likely to fall, per
F-12's own reasoning. From `moderate` to `expansion` (0.85 to 1.20),
coarser spacing — the predicted shallow region. `None` (unconstrained)
is the explicit top endpoint TASK.md 3.2 requires. All four named
values sit exactly on the grid (confirmed: `0.50×B_star = tight`
exactly, etc.), each row tagged with `named_arm` when it lands on one,
so they are directly locatable on the frontier as required.

**Setting ε.** Exactly as Part 1F found and TASK.md 3.1 specifies:
```python
scenario["public_solar_budget_npv"] = envelope
```
between `load_scenario()` and `build_model()`, the identical pattern
`18_` already uses for `financing_regime`/`required_margin`.
`capital_case="moderate"` is passed into `load_scenario()` only as an
inert placeholder — overwritten immediately, never reaching
`build_model()` unmodified.

**A finding worth stating plainly: `required_margin` is set
automatically, not manually.** `18_` (written before `dscr_case`
existed as a required arm) manually sets `scenario["required_margin"]
= arm["required_margin"]` after `load_scenario()` returns. Reading
`src/scenarios.py:3124` (`"required_margin": _dscr_value`) confirms
`load_scenario(dscr_case=...)` **already** writes `required_margin`
from the required, sourced `dscr_case` standard arm. `28_` does NOT
duplicate `18_`'s manual override — it lets the caller's own
`--dscr-case` flag determine `required_margin` natively, which is both
more correct for a "full-arm runner" (TASK.md 3.1) and avoids
inventing an arbitrary value. See the closing list for what this
implies about `18_`'s own (untouched, out-of-scope) redundant line.

**Full standard arm surface.** `add_standard_arms(parser)` (no
include/exclude — Part 3.1: "It is a full-arm runner"), 18 required
flags, no defaults, same discipline as every other repaired runner in
this project.

**The dual — this frontier's primary output, read, not extracted.**
```python
public_budget_dual = diag["public_budget_shadow_usd_per_usd"]
realised_spend = diag["public_budget_realised_spend_usd"]
```
Both already computed by `extract_planning_diagnostics`, already
`hasattr`-guarded internally (confirmed by reading, Part 1F) — no new
extractor.

**The binding check — THREE states, not two, per TASK.md 3.4.**
```python
if not hasattr(m, "public_budget_constraint"):
    public_budget_state = "absent"      # ε was None; constraint never built
else:
    budget_tolerance_usd = 0.01 * envelope     # 1% of the envelope --
                                                # same relative-tolerance
                                                # reasoning as Frontier A
    budget_slack_usd = envelope - realised_spend
    public_budget_state = (
        "binding" if budget_slack_usd <= budget_tolerance_usd else "slack"
    )
```
Distinguishing "absent" from "slack" by construction (`hasattr`, not
by inferring from the dual's numeric value) rather than conflating
them — TASK.md's explicit warning ("Conflating 'absent' with 'slack'
would misstate where the frontier begins") is satisfied structurally,
not by convention alone. **The binding test uses PRIMAL slack (planned
envelope vs. realised spend), not the dual's magnitude** — the same
discipline as Frontier A (which uses primal slack on the emissions
side, not the dual's sign/size) — because a dual can be numerically
fuzzy exactly at the binding boundary in a way the primal quantity is
not.

**Endpoint characterisation, per financing regime.** Same shape as
Frontier A, printed and saved to `capital_frontier_endpoints.json`:
`frontier_begins_at_envelope_fraction` (the largest binding envelope
fraction — envelope increasing is the same "loosening" direction as
Frontier A's `cap_fraction`), counts of binding/slack/absent points,
`tightest_feasible_envelope_fraction`, and
`loosest_infeasible_envelope_fraction`.

**A structural finding, stated rather than forced: infeasibility may
not exist in this model at all, for either regime.** Neither `18_` nor
`28_` builds a hard service-level constraint (`max_offgrid_fraction`)
— unserved and self-generation are PRICED, not CAPPED, so the LP
retains a release valve at every envelope level, including zero. If
the swept range (down to `envelope=0`) produces no infeasible point,
`28_` reports `no_infeasibility_found_in_swept_range=True` explicitly,
with a printed note naming the structural reason, rather than treating
"no infeasible point found" as a sweep-design failure or silently
omitting the field. Whether this actually occurs is an empirical
question this task cannot answer without running the solver
(forbidden by the whitelist) — recorded here as a possibility Zeus
should expect and not be surprised by, not as a claimed result.

**Non-dominated check.** Same pairwise dominance test as Frontier A,
on the **realised public spend** axis (not the nominal envelope — a
loose envelope may go unused; the achieved-quantity axis is what
should not be dominated, mirroring Frontier A's own use of achieved
emissions rather than the nominal cap). Written to
`capital_frontier_dominated_points.csv` if any are found.

**Flag, don't drop.** Every point stays in `capital_frontier_raw.csv`
regardless of status or binding state. The plot draws binding points
solid/connected, slack points hollow, and `absent` (unconstrained)
points as a distinct `x` marker — three visually distinguishable
states, matching the three reported states.

**The EaaS-value frontier shift (Part 3.3).** For every envelope
fraction present in BOTH regimes' optimal rows,
`eaas_value_usd = cost_traditional − cost_eaas`, written to
`capital_frontier_eaas_gap.csv` and printed as a table — the gap
between the two frontiers at matched capital levels, stronger than
F-10's single point comparison because it is reported across the
WHOLE swept range, including at and around the predicted kink.

**The prediction, recorded verbatim before this script is run
(TASK.md 3.6, copied unedited into this report and into the script's
own module docstring):**

> F-12 finds total solar identified at ~49,487 MW, with the
> public/EaaS split bounded below by the public budget and above by
> bankability. So as the public envelope tightens, EaaS should absorb
> displaced public solar at little cost penalty — a shallow frontier —
> until the bankability ceiling binds, after which further tightening
> has nowhere to go but self-generation and the frontier should
> steepen sharply. If that kink appears, the dual at the kink is the
> marginal value of public capital. If the frontier is instead
> straight, the bankability ceiling is not binding in this
> configuration — a different finding, equally reportable.

**Nothing in this script is tuned toward this prediction** — the
finer grid spacing at `tight`-to-`moderate` (0.50/0.60/0.70) was
chosen BECAUSE the prediction names that region as where a kink is
most likely, so the sweep can actually resolve it if present; the
spacing does not manufacture the kink, it only avoids missing it for
lack of resolution.

**Command Zeus should run** (same central arm configuration as
Frontier A, for the same reason — the sourced, reproducing central
case):
```bash
python scripts/28_run_capital_frontier.py \
  --peak-case peak_unsuppressed \
  --storage-cc-case storage_cc_central \
  --gas-build-case gas_unconstrained \
  --new-gas-deliverability constrained \
  --gas-availability-case gas_avail_nameplate \
  --gas-retirement-case gas_retire_vintage_30 \
  --gas-capex-case gas_capex_central \
  --ccgt-efficiency-case ccgt_eff_central \
  --capital-recovery-case cr_central \
  --dscr-case dscr_central \
  --us-inflation 0.020 \
  --capital-treatment-case f1_tranche \
  --concessional-envelope-case env_dares \
  --gas-deliverability-case baseline \
  --conversion-case conv_central \
  --hydro-case hydro_flat \
  --tariff-case tariff_central \
  --om-case om_central
```
No pre-requisite runner — this script reads no other runner's output
(unlike `18_`, which depends on `01_run_baseline.py`'s diagnostics for
its baseline-emissions denominator; Frontier B has no equivalent
dependency, since its ε is a fixed, named quantity, not a fraction of
another run's result).

`python3 -m py_compile scripts/28_run_capital_frontier.py` — passes.
Verified the core logic (named-arm mapping, the `None`-as-+∞ sort key,
the endpoint threshold calculation, and the pairwise dominance check)
against fabricated rows mimicking the real output shape — pure Python/
pandas, no scenario, no builder, no solver — confirming the threshold
and dominance calculations behave as designed before Zeus runs the
real sweep.

---

## Why neither frontier can move the anchors

Both frontiers vary exactly one existing, already-optional `build_model()`
argument each — `emissions_cap_by_year` (Frontier A, pre-existing
argument, pre-existing usage pattern, unchanged) and
`public_solar_budget_npv` via the scenario dict (Frontier B, an
existing, already-optional scenario key every "moderate"/"tight" run
already sets through `capital_case`, just supplied a different way
here). Neither script adds a term to `m.system_cost_npv`, changes the
form of any constraint, or edits any builder — confirmed for `18_` by
diffing the solve-relevant lines (`build_model()`'s call signature and
arguments) against Part 1's own inventory of the pre-edit file, and
for `28_` by construction (a new file cannot touch an existing one).
All new code in both scripts runs either (a) before `build_model()` is
called (setting up the scenario dict, the same class of operation
`18_` and every other repaired runner already performs) or (b) after
`solve_model()` returns (reading `diag`, an already-solved model's
already-computed diagnostics — a pure reader, `pyo.value()` and
`m.dual.get()`, neither of which writes to the model). The regression
anchor (`21,811,042,472.821114 / 45,502.9754 MWh`, `om_none`) and the
central anchor (`27.954 bn / 65,613.0 MWh`) are produced by
`01b_run_baseline_sliced.py` and `02_run_ndc_caps.py`, neither of
which is edited, called, or imported by either frontier script.

---

## For the methodology chapter

The ε-constraint method converts a single-objective LP into a
multi-objective analysis by fixing one objective as a swept BOUND
rather than optimising it directly: the model here minimises exactly
one objective, discounted system cost, and every other quantity of
policy interest — emissions, public capital — enters only as a
constraint. A swept constraint bound is not, by itself, a
multi-objective method; it becomes one only when three conditions are
jointly established at every reported point: the bound must actually
BIND (an unconstrained optimum reported at several nominal bound
values is one point, repeated, not several points on a frontier), the
DUAL on that bound must be reported (the shadow price is the economic
content — the trade-off rate between the two objectives — not the
frontier's visual shape), and the resulting point set must be verified
NON-DOMINATED (no reported point may be beaten on both axes
simultaneously by another reported point, or the "frontier" contains
points that are not actually efficient). This thesis establishes all
three, for both frontiers, explicitly and per point, rather than
asserting them.

The two duals mean different things in the discipline's own terms.
Frontier A's dual, `diag["carbon_shadow_price_usd_per_tco2_by_year"]`,
is a **marginal abatement cost** — USD per additional tCO2 the cap
disallows, the standard unit in the climate-economics literature for
"what does the next unit of decarbonisation cost." Frontier B's dual,
`diag["public_budget_shadow_usd_per_usd"]`, is a **shadow price of
capital** — USD of system cost per USD of public envelope, the
standard unit in public finance and cost-benefit analysis for the
opportunity cost of a binding budget constraint. Reporting each in the
other discipline's units would be a category error, not a rounding
choice. A reader must also be told plainly that Frontier A's ε is a
**flat ANNUAL emissions rate** — a fraction of baseline annual
emissions, repeated identically for every model year — and not a
cumulative carbon budget; a reader encountering "emissions cap" in
climate-policy writing will default to assuming a cumulative budget
unless told otherwise, and the two are materially different objects
(a flat annual rate permits a very different total than a genuinely
declining or cumulative-constrained trajectory would).

Frontier B is this thesis's own central question, not a supporting
robustness check: a generation plan can be simultaneously LEAST-COST
and UNFINANCEABLE, and until this frontier exists that claim is an
assertion made at one point (F-10's comparison) rather than a measured
rate traced across the full range of what the state could plausibly
mobilise. The shadow price of public capital, reported at every
envelope level from zero to unconstrained, converts "public capital is
a binding constraint on this plan" from a qualitative finding into a
quantified marginal cost — and the gap between the `traditional` and
`eaas` frontiers at matched envelope levels quantifies, for the first
time in this project, exactly how much of that binding constraint the
EaaS mechanism relieves, and at what capital level the relief runs out
(if the predicted kink appears).

The honest scope limit: two bi-objective ε-constraint frontiers,
traced separately — cost-vs-emissions and cost-vs-capital — are NOT a
three-objective optimisation. Nothing in this model or this task
establishes a joint emissions-AND-capital frontier, nor a Pareto
surface across all three quantities simultaneously; each frontier
holds the other bound fixed at its default (Frontier A applies no
capital-envelope restriction beyond its own `FINANCING_ARMS`'
`capital_case` values, Frontier B applies no emissions cap at all).
The thesis must describe this as two separate bi-objective analyses
answering two separate questions, not as evidence of a single unified
trade-off surface — that would overclaim what an ε-constraint method
swept on one bound at a time can actually establish.

---

## Encountered but not asked about

- **`18_run_pareto_frontier.py`'s existing `required_margin` override
  is now redundant with `dscr_case`, and this task did not touch it**
  (out of Part 2's scope — "do not change what it solves" forbids
  changing `required_margin`'s value). `18_` predates `dscr_case`
  becoming a required standard arm; its own `scenario["required_
  margin"] = arm["required_margin"]` (tied to `capital_case` via
  `FINANCING_ARMS`) now OVERWRITES the value `load_scenario(dscr_
  case=...)` (via `**standard_kwargs`, added in `task_runner_repair`)
  already set. Not necessarily wrong — `18_`'s own 1.10/1.05 values may
  still be the intended ones for its specific `FINANCING_ARMS` design —
  but it means `18_`'s effective `required_margin` is NOT controlled
  by its own `--dscr-case` flag the way every other required arm on
  this runner is, silently. `28_` (built fresh) does not carry this
  redundancy forward. A follow-on could reconcile `18_`'s
  `FINANCING_ARMS` design against the `dscr_case` mechanism, or state
  explicitly why the override is intentional.
- **Whether either frontier actually finds an infeasibility edge is
  unknown** — this task could not run the solver. Both scripts are
  built to report the answer plainly either way (a found edge, or
  `no_infeasibility_found_in_swept_range=True` with the structural
  reason stated), but the actual outcome is Zeus's to observe.
- **Frontier B's `CAPITAL_ENVELOPE_FRACTIONS` grid is a one-shot
  design choice**, not verified against a scoping run (none performed,
  per the whitelist). If the real infeasibility edge or the predicted
  kink falls outside the resolution this grid provides (e.g. between
  0.70 and 0.85, where the spacing widens), a follow-on with a refined
  grid in that region would be the natural next step — flagged as a
  possibility, not a defect in this grid as designed.
- **Frontier A's `n_binding_years` count and `n_valid_dual_years`
  count can legitimately differ** (a year can be non-binding without
  its dual being `None`, and vice versa in principle) — both are
  reported as separate columns rather than assumed identical, but this
  task did not verify empirically whether they ever actually diverge
  in practice (no solve performed).
- No script, solver, or `smoke_all.sh` was run at any point in this
  task, per the whitelist. All verification above is `py_compile` and
  a synthetic, fabricated-data test of Frontier B's post-sweep logic
  (pure Python/pandas, no scenario, no builder, no solver).

**Then STOP.** Do not commit. Zeus reviews `git diff`, commits, and
runs both frontiers.

---

## PART 5 — the reserve_margin gap

Branch `phase-fin2-d17`. Whitelist for this release: `src/cli_arms.py`,
plus any runner the inventory below shows is broken or made
conflicting by the fix — the inventory found none needed editing (see
Step Four). Append-only to this report.

### STEP ONE — verification, and the seventh-case search

**1. Each of the six `load_scenario()` calls, read directly.** Grepped
each of the six named files for `reserve_margin` and `nirp_spinning`
(any route, not just the literal keyword) — **zero matches in all
six**:
```
scripts/05_run_fin3_capital_sweep.py       0
scripts/15_run_pol1_ndc_comparison.py      0
scripts/18_run_pareto_frontier.py          0
scripts/19_run_carbon_price_sweep.py       0
scripts/20_run_cf1_solar_yield_sensitivity.py  0
scripts/28_run_capital_frontier.py         0
```
Confirmed: `reserve_margin` reaches none of the six, by any route.
`TypeError: load_scenario() requires reserve_margin (2.6, F0 Step
2.3).` is the correct, expected outcome of running any of them,
unchanged since Part 1's own reading (`04_`/`24_`/`25_`/`99c_`,
`task_runner_repair`) established the same failure mode for a
different set of runners before `cli_arms.py` existed at all.

**2. Search for a seventh caller of `standard_arms_kwargs`.** Grepped
the whole of `scripts/` for `standard_arms_kwargs|add_standard_arms`
— **ten files match, not six**:
```
scripts/04_run_fin2_tariff_sweep.py
scripts/05_run_fin3_capital_sweep.py
scripts/15_run_pol1_ndc_comparison.py
scripts/18_run_pareto_frontier.py
scripts/19_run_carbon_price_sweep.py
scripts/20_run_cf1_solar_yield_sensitivity.py
scripts/24_run_leakage_arm.py
scripts/25_run_obligation_trajectory.py
scripts/28_run_capital_frontier.py
scripts/99c_run_genset_arm_sensitivity.py
```
**No seventh BROKEN runner** — the four not in TASK.md's six
(`04_`, `24_`, `25_`, `99c_`) all call `add_standard_arms(parser,
include=["conversion_case", "hydro_case", "tariff_case", "om_case"])`
(confirmed by reading each), a subset that does not request
`reserve_margin`, and each still carries its own hand-written
`reserve_margin="nirp_spinning"` literal at its `load_scenario()` call
site (confirmed by reading each: `04_:378`, `24_:264`, `25_:261`,
`99c_:287`). These four are unaffected by the pre-fix gap and are
covered in the duplicate-keyword audit below rather than in the
six-runner "broken" set. **The task's own inventory list is
incomplete in one respect, corrected here**: `04_run_fin2_tariff_
sweep.py` is not among TASK.md §4's eleven-runner duplicate-keyword
checklist either, despite carrying the same literal via the same
`include=` pattern as `24_`/`25_`/`99c_` — added to the audit below.

**3. Include/exclude usage, per broken runner — determines whether the
fix reaches it automatically.** All six call `add_standard_arms(parser)`
with **no** `include=`/`exclude=` argument (confirmed by reading each
call site directly):
```
05_run_fin3_capital_sweep.py:128        add_standard_arms(parser)
15_run_pol1_ndc_comparison.py:191       add_standard_arms(parser)
18_run_pareto_frontier.py:109           add_standard_arms(parser)
19_run_carbon_price_sweep.py:95         add_standard_arms(parser)
20_run_cf1_solar_yield_sensitivity.py:90 add_standard_arms(parser)
28_run_capital_frontier.py:184          add_standard_arms(parser)
```
**All six use the full set.** Adding `reserve_margin` to `_arm_specs()`
therefore reaches all six automatically — none needs its own edit
(confirmed in Step Four below, by reading each after the fix, not by
assuming this conclusion).

---

### STEP TWO — the fix: option (b), `reserve_margin` added to `cli_arms.py`

**Option (b) taken, reasoning restated (no cause found in the
inventory to argue against it):**

- **Six files, not two.** Had the alternative (a hardcoded literal at
  each of the six broken call sites) been taken, it would have written
  the identical constant six times — exactly the copy-pasted-argument
  pattern `cli_arms.py` exists to end, and the same pattern that
  produced register L27's six stale library call sites. Step One's own
  finding sharpens this further: it would in fact have been written
  correctly only by chance, since a plain literal at each of six sites
  is the precise shape that let O&M sit hardcoded at six identical
  sites for as long as it did before `task_shared_fn_repair` found it.
- **These are frontier and sensitivity runners.** A reserve margin is a
  physical planning assumption (NIRP 2024 §2.3, spinning reserve
  requirement); making it stateable on these six costs one flag
  definition and closes a silent-constant gap that currently manifests
  as a hard crash, not a silent wrong answer — but the crash is only
  informative because `load_scenario()`'s own sentinel discipline
  happens to catch it; a future required key without that discipline
  would not be caught the same way.
- **The fourth instance of one defect shape**, per TASK.md's own
  framing, corroborated by reading the register: the retired 65,000,000
  USD/TWh tariff default (`docs/tariff_arm.md`), `storage_duration_
  hours` (register **L21**, confirmed: "an explicit constant, ASSUMED,
  and unswept"), `om_case` hardcoded to `"om_central"` across six
  library functions (register **L27**), and now this. Each was a
  load-bearing value assumed to always be restated explicitly at every
  call site, until a generalisation removed the explicit site without
  anyone re-stating the value there.

**The edit, `src/cli_arms.py` — old and new text.**

Module docstring's "NOT A STANDARD ARM: reserve_margin" section —
**old:**
```
NOT A STANDARD ARM: reserve_margin
------------------------------------
reserve_margin is one of load_scenario()'s 17 sentinel-required keys, but
it is deliberately NOT one of the flags this module defines. 01b_'s own
reference implementation does not expose it as a CLI flag either -- it is
passed as the hardcoded literal reserve_margin="nirp_spinning" directly at
each load_scenario() call site. Register L9 records this (alongside
genset_case) as a canonical project-wide constant, not a swept arm: "the
runners will pass the literal, as they already do elsewhere. The
convention belongs at the call site, not inside the library." Every
runner using this module still writes reserve_margin="nirp_spinning"
itself, exactly as 01b_ does.
```
**new:** replaced with a "reserve_margin -- IS a standard arm
(task_epsilon_constraint, Part 5)" section restating the original L9
reasoning, why it breaks for a full-surface caller, that no existing
runner's VALUE changes, and the four-instance defect-shape framing (full
text in the file — reproduced in this report's own reasoning above,
not duplicated twice here).

New module-level function, added after the three no-live-registry
choice constants:
```python
def _reserve_margin_type(raw):
    """
    argparse type= for --reserve-margin. A plain choices= list cannot
    express "either the literal 'nirp_spinning' or a numeric fraction" --
    this function is the validation load_scenario() itself performs
    (src/scenarios.py, reserve_margin handling), applied at parse time so
    a bad value fails at the CLI, not three solves into a sweep.
    """
    if raw == "nirp_spinning":
        return raw
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"reserve_margin must be a numeric fraction (e.g. 0.15) or "
            f"the literal 'nirp_spinning', got {raw!r}."
        )
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"reserve_margin must be >= 0, got {value}."
        )
    return value
```
Mirrors `load_scenario()`'s own validation exactly (`src/scenarios.py`,
read directly): string must equal `"nirp_spinning"` or be rejected
(not coerced); numeric must parse as `float` and be `>= 0`.
`argparse.ArgumentTypeError` (not a bare `ValueError`) is the correct
exception type for a `type=` callable, so argparse reports it as a
usage error at the flag, not an unhandled traceback.

New entry in `_arm_specs()`, inserted immediately after
`peak_multiple_case`:
```python
("reserve_margin", "--reserve-margin", "reserve_margin",
 None, _reserve_margin_type,
 "Reserve margin fed to load_scenario(reserve_margin=...). Accepts "
 "a numeric fraction (e.g. 0.15) or the literal 'nirp_spinning' "
 "(derives a per-year fraction from RESERVE_MARGIN_SPINNING_MW / "
 "peak_mw[t]). No default: an unstated reserve margin would build "
 "a firm-capacity constraint against an unstated assumption with "
 "no error, warning, or visible signature in the output "
 "(load_scenario()'s own raise text, src/scenarios.py)."),
```
Help text drawn from `load_scenario()`'s own `TypeError` message
(`src/scenarios.py`, read directly), in the same register as every
other flag's help text, per TASK.md's instruction.

`standard_arms_kwargs`'s docstring updated (the sentence "reserve_
margin is never included" was the one piece of this module now
actively wrong; corrected to state the opposite and to name the
duplicate-keyword risk this creates for a caller that also holds its
own literal).

**Value unchanged for every existing runner**, confirmed: no runner in
this project passes any `--reserve-margin` value other than
`nirp_spinning` today (Step Four below), and this task supplies no
runner-level edit that would change that.

`python3 -m py_compile src/cli_arms.py` — passes. Functionally tested
(argparse logic only — no scenario, no builder, no solver, per the
whitelist): the literal form (`--reserve-margin nirp_spinning` →
`kwargs['reserve_margin'] == 'nirp_spinning'`), the numeric form
(`--reserve-margin 0.15` → `kwargs['reserve_margin'] == 0.15`, a
`float`), rejection of an invalid string (`--reserve-margin garbage` →
`SystemExit` with the exact message above), the required-flag check
(omitting `--reserve-margin` entirely → `SystemExit`), and confirmation
that an `include=[...]` subset excluding `"reserve_margin"` correctly
omits it from both the parser and `standard_arms_kwargs`'s output — all
five checks passed.

---

### STEP THREE — the duplicate-keyword audit

**Every runner that currently passes `reserve_margin` as a literal,
checked individually** (TASK.md's eleven plus the one it omitted,
`04_`, found in Step One's seventh-case search):

| Runner | Uses `cli_arms`? | `add_standard_arms(...)` shape | Would now ALSO receive `reserve_margin` from `standard_arms_kwargs`? | Verdict |
|---|---|---|---|---|
| `01_run_baseline.py` | No | — (own hand-written argparse) | No — never calls `standard_arms_kwargs` | **Safe.** No possible conflict; confirmed by the absence of any `cli_arms` import. |
| `01b_run_baseline_sliced.py` | No | — | No | **Safe**, same reason (this is `cli_arms.py`'s own reference implementation; it predates and does not import the module it inspired). |
| `02_run_eaas.py` | No | — | No | **Safe**, same reason. |
| `02_run_ndc_caps.py` | No | — | No | **Safe**, same reason. |
| `03_run_ndc_eaas.py` | No | — | No | **Safe**, same reason. |
| `22_salvage_sensitivity.py` | No | — | No | **Safe**, same reason. |
| `23_run_selfgen_frontier.py` | No | — | No | **Safe**, same reason. |
| `26_run_f3_identification.py` | No | — | No | **Safe**, same reason. |
| `04_run_fin2_tariff_sweep.py` | Yes | `include=["conversion_case","hydro_case","tariff_case","om_case"]` | No — `"reserve_margin"` not in the include list | **Safe.** `standard_arms_kwargs(args)` returns only what `include=` added; confirmed by reading the actual call site (line 292) and by this task's own functional test (subset case). Not in TASK.md §4's list — added here. |
| `24_run_leakage_arm.py` | Yes | same `include=[...]` (line 233) | No | **Safe**, same reason. |
| `25_run_obligation_trajectory.py` | Yes | same `include=[...]` (line 224) | No | **Safe**, same reason. |
| `99c_run_genset_arm_sensitivity.py` | Yes | same `include=[...]` (line 234) | No | **Safe**, same reason. |

**Twelve runners checked, twelve safe, zero conflicts, zero edits
needed.** `include=[...]` subsets are confirmed unaffected by reading
each call site directly, not assumed from the general shape of the
pattern — TASK.md's own instruction ("confirm rather than assume").

**The six runners gaining `reserve_margin`** (`05_`, `15_`, `18_`,
`19_`, `20_`, `28_`) hold **no** hand-written `reserve_margin` literal
of their own (Step One, item 1) — there is nothing for the new flag's
value to conflict with in any of the six. No duplicate-keyword error
is possible there either, for the opposite reason: nothing pre-existing
to collide with.

**Net result: no runner file required editing.** The whitelist
permitted editing "any runner... the inventory shows is broken or made
conflicting by the fix" — the inventory shows none are. `src/cli_
arms.py` is the only file this task edits.

---

### STEP FOUR — confirming the six broken runners are fixed, by reading

Re-read each of the six runners' `load_scenario()` call after the
`cli_arms.py` fix (not run):

- All six still write `**standard_kwargs` (or `**standard_arms_
  kwargs(args)`, per the runner) unchanged, into an otherwise-unmodified
  `load_scenario()` call.
- All six call `add_standard_arms(parser)` with no `include=`/`exclude=`
  (Step One, item 3) — so `reserve_margin` is now part of every one of
  their parsers automatically, requiring no per-runner edit.
- `standard_arms_kwargs(args)` (re-read after the fix) iterates
  `_arm_specs()` and includes any kwarg whose `dest` is present on
  `args` — `reserve_margin`'s `dest` is `"reserve_margin"`, present on
  `args` for all six once `--reserve-margin` is a required flag on
  their (un-subsetted) parser. Confirmed structurally, not by
  execution.

**All six now reach `load_scenario()` with `reserve_margin` supplied.**
None uses `include=`/`exclude=`, so TASK.md §5's conditional ("add
`reserve_margin` to its include list explicitly") does not apply to
any of the six — there is no include list to add it to.

---

### The two drafts

**6.1 — L9 amendment, drafted, NOT applied (`docs/limitations_
register.md` is outside this task's whitelist):**

> **Amendment (task_epsilon_constraint, Part 5).** `reserve_margin` is
> no longer uniformly a call-site literal. `src/cli_arms.py` now
> exposes it as a required `--reserve-margin` flag (accepting the
> literal `nirp_spinning` or a numeric fraction) on the six runners
> whose entire argument surface is that module (`05_`, `15_`, `18_`,
> `19_`, `20_`, `28_`); the other twelve runners that set it (see
> Part 5's duplicate-keyword audit) still carry the hardcoded literal
> directly at their own call sites. Every runner, either way, still
> resolves to `nirp_spinning` in practice — the value is now
> **stateable** on a subset of runners, not **different**; L9's own
> "What" count ("hardcoded identically in all 11 runners that set it")
> is also now stale on two counts (the true count is twelve literal-
> carrying runners, not eleven, and six further runners state it
> through a flag rather than a literal at all) and should be corrected
> alongside this amendment when the register is next in scope.

**6.2 — For the methodology chapter, `docs/methodology_protocols.md`
§1 (drafted here; that file is not edited):**

> A constant excluded from a shared helper on the grounds that "the
> convention belongs at the call site" holds only while call sites
> actually carry it. When the helper becomes a runner's ENTIRE
> argument surface, the exclusion silently becomes a gap. This project
> has now produced the same failure four times, each at a different
> layer: a retired default silently applied when a tariff argument was
> omitted (the 65,000,000 USD/TWh solar tariff), a physical constant
> left unsourced and unswept because no call site ever needed to state
> it explicitly (`storage_duration_hours`), an operating-cost arm
> hardcoded identically across six library functions because each was
> written by extending the last rather than sharing one definition
> (`om_case`, register L27), and now a planning constant, correctly
> excluded from a shared CLI helper under the reasoning that runners
> would keep stating it themselves — until six runners were built whose
> entire argument surface WAS that helper, and none of them could state
> anything the helper did not define. The general rule: an exclusion
> justified by "the call site will supply it" is a claim about every
> CURRENT call site, not a property of the value itself, and it must be
> re-examined every time a new call site is added that does not write
> its own arguments by hand — a generalisation is exactly the kind of
> new call site most likely to inherit an old exclusion's absence
> without anyone noticing, because there is no longer a discrete,
> readable line to notice its absence FROM.

---

### Why this cannot move a solved value

This task supplies an argument `load_scenario()` already required and
already raised `TypeError` without — it does not add a new required
key, does not change what any existing required key means, and does
not alter the value any runner resolves to (every runner that reaches
`load_scenario()` today, before or after this fix, does so with
`reserve_margin == "nirp_spinning"`, exactly as `docs/scenarios.py`'s
`reserve_margin == "nirp_spinning"` branch already resolved it before
this task existed). No builder, no `load_scenario()` body, and no
constraint is touched — `src/cli_arms.py` is a CLI-argument-plumbing
module whose only effect is which values reach an existing function
call, never what that function computes once called. The frozen
anchors (`21,811,042,472.821114 / 45,502.9754 MWh`, `om_none`;
`27.954 bn / 65,613.0 MWh`) are produced by `01b_run_baseline_sliced.py`
and `02_run_ndc_caps.py`, neither of which imports `src/cli_arms.py`
at all (Step Three's own table: both are hand-written-argparse
runners) — this task cannot reach the code path that produces either
anchor, let alone change it.

---

### Encountered but not asked about

- **`04_run_fin2_tariff_sweep.py` is missing from TASK.md §4's own
  eleven-runner duplicate-keyword checklist**, despite carrying the
  identical `reserve_margin="nirp_spinning"` literal via the identical
  `include=[...]` pattern as `24_`/`25_`/`99c_`. Found in Step One's
  seventh-case search (a full grep of `scripts/` for `cli_arms` usage,
  not a re-run of TASK.md's own eleven-item list), checked, confirmed
  safe, added to the audit table above.
- **L9's own runner count ("all 11 runners that set it") is stale**,
  independent of this task's fix: Step One/Three's audit finds TWELVE
  literal-carrying runners (the eleven TASK.md named, plus `04_`), not
  eleven — a pre-existing inventory drift in the register, unrelated
  to the `reserve_margin`-in-`cli_arms` question itself, noted in the
  drafted amendment above rather than silently left for a future
  reader to rediscover.
- **`peak_multiple_case`'s dest/flag mismatch (`--peak-case` →
  `args.peak_case`, not `args.peak_multiple_case`) was the only other
  place `_arm_specs()` needed a `dest` different from its `kwarg`
  before this task; `reserve_margin` needed none** (flag `--reserve-
  margin` → dest `reserve_margin`, matching its kwarg directly) — noted
  only because it was checked, not assumed, while placing the new
  entry in the table.
- No script, solver, or `smoke_all.sh` was run at any point in this
  task, per the whitelist. All verification above is `py_compile`,
  direct reading of every file named, and a pure-argparse functional
  test of the new flag and validator (no scenario, no builder, no
  solver).

**Then STOP.** Do not commit. Zeus reviews `git diff` and commits.

---

## PART 6 — port to the sliced builder

Branch `phase-fin2-d17`. This turn is Part 1 (inventory) only, per
TASK.md's own gate ("Ends in a STOP" / "→ STOP. Write 1A-1G and
STOP."). No file edited — read-only, per the whitelist's own
permitted-actions list for this stage.

**Headline answer to Part 1's own conditional STOP clause, stated
first:** the inventory does **NOT** show the duals require an edit
outside this whitelist. Both `m.emissions_constraint` and
`m.public_budget_constraint` are built with **identical structure and
indexing** in `build_model_sliced()` as in `build_model()` — confirmed
by reading both builders' source directly, not inferred. The port can
proceed inside `scripts/18_run_pareto_frontier.py` and
`scripts/28_run_capital_frontier.py` alone, as scoped. Two SEPARATE,
narrower issues were found in `extract_planning_diagnostics` that
would need a `src/` edit to fix (1D) — but neither is read by either
frontier today, so neither blocks the port; both are reported
prominently below, per 1D's explicit instruction, not silently
absorbed into the "duals are fine" headline.

### 1A — `m.emissions_constraint` in the sliced builder

**Built with the SAME indexing as the annual builder: by `t` (the
annual year index), never by `[t,(s,p)]`.** Quoted, side by side:

```python
# src/optimize_model_sliced.py:1964-1972
if emissions_cap_by_year is not None:
    if len(emissions_cap_by_year) != len(list(T)):
        raise ValueError("emissions_cap_by_year length mismatch.")
    m.emissions_constraint = pyo.Constraint(
        T, rule=lambda m, t: m.emissions_capped_by_year[t] <= emissions_cap_by_year[t]
    )
elif emissions_cap is not None:
    m.emissions_constraint = pyo.Constraint(expr=m.emissions_capped <= emissions_cap)
```
```python
# src/optimize_model.py:1732-1749 (annual)
if emissions_cap_by_year is not None:
    if len(emissions_cap_by_year) != len(list(T)):
        raise ValueError("emissions_cap_by_year length mismatch.")
    m.emissions_constraint = pyo.Constraint(
        T, rule=lambda m, t: m.emissions_capped_by_year[t] <= emissions_cap_by_year[t],
    )
elif emissions_cap is not None:
    m.emissions_constraint = pyo.Constraint(expr=m.emissions_capped <= emissions_cap)
```
**Byte-for-byte identical.** Both index on `T = range(len(years))`
(confirmed at `optimize_model_sliced.py:252`/`1985`/`2071` — the same
annual year range as the annual builder's own `T`; the sliced
builder's separate slice index, `m.S`, is a distinct, `dimen=2` set,
`optimize_model_sliced.py:601`, never used here).

**Why: `m.emissions_capped_by_year[t]` is ITSELF an annual-only
Expression, built by summing over slices INSIDE the builder, at
Pyomo-model-construction time** — not something `extract_planning_
diagnostics` would need to re-aggregate:
```python
# src/optimize_model_sliced.py:1841-1876
m.emissions_grid_only_by_year = pyo.Expression(
    T, rule=lambda m, t:
        pyo.quicksum(m.gas_generation[t, (s, p)] for (s, p) in S) * 1e6 * ef_e
        + _converted_gen_year(m, t) * 1e6 * _ef_e_retrofit)   # if conversion active
m.emissions_selfgen_by_year = pyo.Expression(
    T, rule=lambda m, t: pyo.quicksum(m.selfgen[t, (s, p)] for (s, p) in S) * 1e6 * selfgen_ef_e)
m.emissions_by_year = pyo.Expression(
    T, rule=lambda m, t: m.emissions_grid_only_by_year[t] + m.emissions_selfgen_by_year[t])
m.emissions_capped_by_year = pyo.Expression(T, rule=lambda m, t: ...)  # grid-only or total
m.emissions_capped = pyo.Expression(expr=pyo.quicksum(m.emissions_capped_by_year[t] for t in T))
```
The `(s,p)` summation happens ONCE, inside these Expressions' own
`rule=`, at build time — `m.emissions_capped_by_year[t]` is a plain,
annual-indexed Pyomo component from that point on, in both builders.

**Would `extract_carbon_shadow_prices_usd_per_tco2` work unmodified,
crash, or silently return wrong values?** **Work unmodified, correctly.**
```python
# src/optimize_experiments.py:2260-2297
def extract_carbon_shadow_prices_usd_per_tco2(m, years):
    if not hasattr(m, "emissions_constraint"):
        return {int(y): None for y in years}
    if m.emissions_constraint.is_indexed():
        ...
        dual = m.dual.get(m.emissions_constraint[t], None)
        ...
    dual = m.dual.get(m.emissions_constraint, None)   # scalar case
    ...
```
`hasattr`, `.is_indexed()` (a generic Pyomo `Constraint` method), and
`m.dual.get(m.emissions_constraint[t], None)` all operate on the
constraint object itself — which, per the quote above, has identical
shape regardless of which builder produced it. **No code change
needed for this function to work on a sliced model.**

### 1B — `m.public_budget_constraint` in the sliced builder

**Same finding, same reason.** Quoted, side by side:
```python
# src/optimize_model_sliced.py:1939-1952
public_budget_npv = scenario.get("public_solar_budget_npv", None)
if public_budget_npv is not None:
    m.public_budget_constraint = pyo.Constraint(
        expr=pyo.quicksum(
            m.DF[t] * (
                m.solar_public_add[t] * m.solar_capex_param[t]
                + m.storage_add[t] * econ["STORAGE_COST_PER_MWH"]
                + m.eaas_subsidy[t]
            ) for t in T
        ) <= public_budget_npv
    )
    m.has_public_budget_constraint = True
```
```python
# src/optimize_model.py:1706-1720 (annual)
m.public_budget_constraint = pyo.Constraint(
    expr=pyo.quicksum(
        m.DF[t] * (
            m.solar_public_add[t] * m.solar_capex_param[t]
            + m.storage_add[t] * econ["STORAGE_COST_PER_MWH"]
            + m.eaas_subsidy[t]
        ) for t in T
    ) <= public_budget_npv
)
m.has_public_budget_constraint = True
```
**Byte-for-byte identical**, and always **scalar** (non-indexed) in
both builders — because `m.solar_public_add[t]`, `m.storage_add[t]`,
`m.eaas_subsidy[t]` are all **annual investment/decision variables in
both builders**: capacity and capital decisions are never sliced —
only operational dispatch is. There is no `(s,p)` dimension for this
constraint to vary by in either builder.

**Dual readable the same way — confirmed by reading `extract_planning_
diagnostics`'s own existing code, unconditionally builder-agnostic
already:**
```python
# src/optimize_experiments.py:2762-2786
if hasattr(m, "public_budget_constraint"):
    raw_dual = m.dual.get(m.public_budget_constraint, None)
    ...
    realised_public_spend = sum(
        float(pyo.value(m.DF[t])) * (
            float(pyo.value(m.solar_public_add[t])) * float(pyo.value(m.solar_capex_param[t]))
            + float(pyo.value(m.eaas_subsidy[t]))
        ) for t in T
    )
    ...
```
Every quantity here (`DF`, `solar_public_add`, `solar_capex_param`,
`eaas_subsidy`) is annual in both builders. **No code change needed.**

### 1C — how the existing sliced standalone runners get their diagnostics

Read `01b_run_baseline_sliced.py`, `26_run_f3_identification.py`,
`27_run_hydro_minflow_sensitivity.py` in full.

**None of the three reads `m.emissions_constraint`'s or `m.public_
budget_constraint`'s dual on a sliced model** — stated plainly, since
1A/1B's confidence rests on reading the builder source directly, not
on an existing precedent for these two specific duals:

- **`01b_run_baseline_sliced.py`** never builds either constraint in
  its own run (`EMISSIONS_CAP = None` always; `capital_case=
  "unconstrained"` always → `public_solar_budget_npv` is always
  `None`). It gets its diagnostics from `firm_capacity_diagnostic`,
  `conversion_diagnostic` (`src.optimize_model_sliced`) plus
  `leakage_diagnostic`, `solve_model` (imported from `src.
  optimize_model`, the ANNUAL module, and used unchanged on its
  sliced `m` — direct evidence for 1E/1G). **It DOES read a dual
  directly on a sliced model** — for `m.gas_balance[t]`, a different
  constraint:
  ```python
  # scripts/01b_run_baseline_sliced.py:368-370
  for t, y in enumerate(years):
      d = m.dual.get(m.gas_balance[t], None)
      gas_shadow[int(y)] = abs(float(d)) if d is not None else None
  ```
  **This is the pattern to reuse if a direct read is ever needed**:
  `m.dual.get(constraint[t], None)`, sign-handled (here, a plain
  `abs()` — simpler than `extract_gas_shadow_prices_usd_per_twh_th`'s
  own negate-then-clamp-then-`DUAL_SIGN_TOL`-check convention;
  TASK.md's Part 2 instruction to follow `extract_carbon_shadow_
  prices_usd_per_tco2`'s own convention exactly, not this simpler one,
  is noted for when Part 2 is released). Confirms the GENERAL
  convention (`m.dual.get()` on a `T`-indexed constraint from a sliced
  model) works — but not, by itself, proof for `emissions_constraint`/
  `public_budget_constraint` specifically; 1A/1B's own structural
  reading supplies that.
- **`26_run_f3_identification.py`** uses `build_model_sliced` and
  `solve_model` (again imported from `src.optimize_model`). It checks
  `getattr(m, "has_public_budget_constraint", False)` as a **boolean**
  (an assertion that `public_solar_budget_npv is not None` matches the
  flag) but never reads the constraint's **dual** — the shadow price
  itself is not something this runner needs for its own question (EaaS
  capacity identification). Also carries its own third, independent
  copy of `bankability_slack_per_mw_by_year` (line 325) — noted in the
  closing list, not central to 1C.
- **`27_run_hydro_minflow_sensitivity.py`** uses `build_model_sliced`,
  `conversion_diagnostic`, `annual_totals` — reads no `m.dual` at all.

### 1D — is `extract_planning_diagnostics` usable on a sliced model, in part or not at all?

**Mostly, already, on purpose — this function has ALREADY been
partially ported.** It opens with a builder-agnostic dispatch-
aggregation helper, explicitly labelled as such:
```python
# src/optimize_experiments.py:2369-2377
_sliced = m.unserved.dim() > 1
if _sliced:
    _slice_index = list(m.S)
    def _slice_total(comp, t):
        return sum(pyo.value(comp[t, sp]) for sp in _slice_index)
else:
    def _slice_total(comp, t):
        return pyo.value(comp[t])
```
labelled in its own comment "Builder-agnostic slice aggregation (Tier
port, step 0)" — this task is not the first to touch this question.
Confirmed by reading the entire function (lines 2340-3250, ~910
lines) end to end: **every dispatch-level quantity that reaches the
returned dict is already wrapped in `_slice_total`** —
`unserved_twh_by_year`, `solar_generation_twh_e_by_year`, `selfgen_
twh_by_year`, `gas_to_power_twh_th_by_year`, `gas_generation_twh_e_
by_year`, `storage_discharge_twh_e_by_year`, `storage_charge_twh_e_
by_year`, and the `storage_binding` power/energy-bind checks. Every
quantity that is genuinely annual in BOTH builders (capacity,
investment decisions, discount factors, cost Expressions, the
emissions/budget Expressions covered in 1A/1B) is read with plain
`pyo.value(m.x[t])`, correctly, needing no wrapping.

**Two things would NOT crash but would be silently incomplete or
misleading on a sliced model with an active conversion arm — reported
prominently, as instructed, because this is the outcome to rule out
most carefully:**

1. **`cost_decomposition["components"]` omits a `conversion_capital_
   charge_npv` entry entirely.** Under `f1_tranche`, `_capital_
   components` (`src/optimize_experiments.py:2941-2948`) lists exactly
   `gas_capital_charge_npv`, `public_solar_capital_charge_npv`,
   `eaas_solar_capital_charge_npv`, `storage_capital_charge_npv`,
   `rebuild_capital_charge_npv`, `capital_charge_total_npv` — no
   conversion entry. But the SLICED builder's own `m.capital_charge_
   total_npv` (confirmed by direct reading, `optimize_model_sliced.py`,
   the `if _conv:` block around its own capital-charge construction)
   additionally folds in `m.cost_conversion_capital_charge_npv` when
   an available conversion arm is active. **The TOP-LEVEL numbers stay
   correct** — `cost_capital_total = float(pyo.value(m.capital_charge_
   total_npv))` (`optimize_experiments.py:2940`) reads the Expression
   directly, which already includes conversion capex symbolically, so
   `real_expenditure_npv`/`cost_total_npv` and the reconciliation
   identity (`task_wrong_output_repair`, Part 4) remain exact. **But
   the itemised `components` breakdown will not sum to `capital_
   charge_total_npv`** on such a run — the gap is exactly the
   conversion capital charge, silently unlabelled. This was flagged as
   a hypothetical in this project's own prior work (`task_wrong_
   output_repair`, Part 4's own comment: "confirmed this function is
   called only on `build_model()` models, never `build_model_sliced()`,
   by checking every one of its ~25 call sites") — that hypothetical
   is what this port makes real for the first time.
2. **`npv_capex_report`/`npv_capex_full` silently omit conversion
   retrofit capex entirely**, not just from a breakdown but from the
   reported TOTAL itself:
   ```python
   # src/optimize_experiments.py:3074-3088
   def _capex_at(t):
       v = 0.0
       if hasattr(m, "solar_public_add"): v += ...
       if hasattr(m, "solar_eaas_add"): v += ...
       if hasattr(m, "storage_add") and econ is not None: v += ...
       if hasattr(m, "gas_add") and "gas_capex_per_mw" in scenario: v += ...
       return v
   ```
   No `hasattr(m, "convert_mw")` (or equivalent) branch — confirmed by
   grepping the whole function for `"convert"`: the only conversion-
   aware code anywhere in this function is `_converted_generation_
   total` (used solely for `total_generation`/`solar_share_of_grid_
   generation`, correctly guarded and already exercised — its own
   comment cites real numbers, "21.76 TWh converted vs only 2.98 TWh
   unconverted fleet"). `npv_capex_report`/`npv_capex_full` themselves
   will UNDERSTATE gross capital expenditure by the full conversion
   retrofit spend on any sliced, conversion-active run, with no error.

**Neither of these two fields is currently read by `18_` or `28_`**
(confirmed: `grep` for `storage_binding`, `npv_capex_report`, `npv_
capex_full`, and `decomp["components"]`/`decomp.get("components"` in
both files — zero matches in both). **This does not block the port**
for either frontier's own current output — both defects sit in fields
neither frontier consumes — but they are real, and a src/ edit (`src/
optimize_experiments.py`, outside every whitelist this task sequence
has granted so far) would be needed to close them. Reported here, not
fixed, exactly as `storage_binding_by_year`'s own pre-existing
in-code comment already reports (not fixes) its own, separate,
already-known sliced-meaninglessness:
```python
# src/optimize_experiments.py:2670-2677
# NOTE (Tier port, step 0): under the sliced builder there is no
# storage_deployable_hours_per_year constraint, so this power_bind
# test compares the annual discharge total against a quantity the
# model never constrained -- the "power_limit" classification is
# not meaningful for sliced runs. ... See docs/tier_port_inventory.md 0.4.
```
This third field (`storage_binding_by_year`) is ALSO not read by
either frontier (confirmed, same grep) — restated here for
completeness since 1D asks about the function broadly, not only about
what the frontiers currently consume.

### 1E — `leakage_diagnostic`

**The same function works on both builders — no sliced equivalent
exists, none is needed.** `01b_run_baseline_sliced.py` imports it from
`src.optimize_model` (the annual module) and calls it directly on its
sliced `m` (`scripts/01b_run_baseline_sliced.py:17`, used at its
backcast-check section). Confirmed by reading the function itself:
```python
# src/optimize_model.py:1759-1789
def leakage_diagnostic(m, threshold=0.10):
    total = float(pyo.value(m.emissions))
    selfgen = float(pyo.value(m.emissions_selfgen))
    grid = float(pyo.value(m.emissions_grid_only))
    capped = float(pyo.value(m.emissions_capped))
    ...
```
Reads ONLY four scalar Expressions (`m.emissions`, `m.emissions_
selfgen`, `m.emissions_grid_only`, `m.emissions_capped`) — each, per
1A's own finding, a horizon-sum of the identically-indexed-by-`t`
annual Expressions built the same way in both builders. `18_` already
imports `leakage_diagnostic` from `src.optimize_model` — **no import
change needed**, only the `build_model` → `build_model_sliced` swap
that produces the `m` passed into it.

### 1F — what does `build_model_sliced` require that `build_model` does not?

**Nothing, as a function-call argument.** Both signatures, read
directly:
```python
# src/optimize_model.py:30-43
def build_model(
    scenario, econ, emissions_cap=None, emissions_cap_by_year=None,
    max_offgrid_fraction=None, grid_service_mode="annual",
    reliability_max_unserved_fraction=None, reliability_mode=None,
    fixed_plan=None, solar_capex_by_year=None,
    selfgen_in_emissions_cap=False, allow_unserved=None,
):
```
```python
# src/optimize_model_sliced.py:221-234
def build_model_sliced(
    scenario, econ, emissions_cap=None, emissions_cap_by_year=None,
    max_offgrid_fraction=None, grid_service_mode="annual",
    reliability_max_unserved_fraction=None, reliability_mode=None,
    selfgen_in_emissions_cap=False, fixed_plan=None,
    solar_capex_by_year=None, allow_unserved=None,
):
    """
    Build a time-sliced planning optimisation model.
    Signature matches build_model() so runners can swap with one import change.
    """
```
**Identical parameter set** — the only difference is the declared
ORDER of `fixed_plan`/`selfgen_in_emissions_cap` relative to each
other, which is irrelevant: both `18_` and `28_` call `build_model(...)`
with 100% keyword arguments today (confirmed by reading both current
call sites), and keyword arguments are order-independent. The
docstring's own claim ("swap with one import change") is confirmed
accurate by this comparison.

**On the `scenario` DICT side** (not the function signature): diffed
every `scenario["..."]`/`scenario.get("...")` key each builder reads.
`build_model_sliced` additionally reads `conversion_capex_per_kw`,
`conversion_retrofit_efficiency`, `conversion_tiers`, `conversion_
uplift`, `hydro_min_flow_fraction`, `hydro_season_factor`, `period_
demand_factor`, `season_demand_factor`, `season_solar_factor` — but
**every one of these is already populated by `load_scenario()`**
whenever `conversion_case`/`hydro_case`/`peak_multiple_case` are
supplied (confirmed: these are exactly the arms `load_scenario()`
resolves them from), and **both frontiers already supply all three**
as required standard arms via `src/cli_arms.py`. `build_model`
(annual) reads two keys `build_model_sliced` does not
(`storage_deployable_hours_per_year`, `storage_solar_surplus_frac`,
both optional/defaulted) — irrelevant to the sliced swap. **No change
to either frontier's `load_scenario()` call is needed** — only the
builder call itself changes.

### 1G — does `solve_model` work unchanged on a sliced model?

**Confirmed, by direct precedent, twice over.** `01b_run_baseline_
sliced.py:334` and `26_run_f3_identification.py` (`from src.
optimize_model import solve_model`, called on `build_model_sliced()`'s
output) both call it directly, unmodified, on a sliced model. Reading
`solve_model`'s own signature (`src/optimize_model.py:1817`,
`solve_model(model, scenario=None, solver="cbc", tee=False)`) confirms
why: it takes a generic Pyomo `model` object and calls the generic
solver interface (`m.objective`, the standard Pyomo solve path) — no
annual-specific indexing anywhere in its own logic. `18_` and `28_`
already import `solve_model` from `src.optimize_model` — **no import
change needed**, same as `leakage_diagnostic`.

---

### Summary: what Part 2 (when released) will actually need to do

Stated here for Zeus's own planning, not as a start of Part 2 itself
(explicitly not begun, per the gate):

- Swap the `build_model` import/call for `build_model_sliced` in both
  files — no additional arguments, per 1F.
- Keep `leakage_diagnostic`/`solve_model` imports from `src.
  optimize_model` unchanged — both are builder-agnostic, per 1E/1G.
- Keep the `extract_planning_diagnostics(m, scenario, econ)` call, and
  keep reading `diag["carbon_shadow_price_usd_per_tco2_by_year"]` /
  `diag["public_budget_shadow_usd_per_usd"]` exactly as both frontiers
  already do — per 1A/1B/1D, no extraction-logic change is needed for
  either dual, and neither of 1D's two flagged silent-incompleteness
  issues touches a field either frontier reads.
- Frontier A's emissions denominator and the bare-`diagnostics.json`
  path issue (TASK.md's own Part 2 bullet) are Part 3B's subject, not
  yet addressed here (Part 3 is not begun either, per the same gate).

---

**Then STOP.** Parts 2-3 not started. Awaiting release.

---

## PART 6B — the port, released

Branch `phase-fin2-d17`. Whitelist for this release:
`scripts/18_run_pareto_frontier.py`, `scripts/28_run_capital_frontier.py`,
append-only to this report. Part 1's inventory (`## PART 6` above) is
not re-derived, per TASK.md §1's own list of what it established.

---

### PART 2 — THE PORT

#### `scripts/18_run_pareto_frontier.py`

**Import — old:**
```python
from src.optimize_model import build_model, solve_model, leakage_diagnostic
```
**new:**
```python
from src.optimize_model import solve_model, leakage_diagnostic
from src.optimize_model_sliced import build_model_sliced
```
`solve_model`/`leakage_diagnostic` stay imported from `src.optimize_
model` unchanged, per TASK.md §1's own instruction and the Part 1
inventory's confirmation that both are builder-agnostic.

**Call — old:**
```python
m = build_model(
    scenario=scenario,
    econ=econ,
    emissions_cap_by_year=cap_by_year,
    solar_capex_by_year=solar_capex_tv,
)
```
**new:**
```python
m = build_model_sliced(
    scenario=scenario,
    econ=econ,
    emissions_cap_by_year=cap_by_year,
    solar_capex_by_year=solar_capex_tv,
)
```
Keyword arguments unchanged, per Part 1F's confirmation that the two
builders' signatures are call-compatible.

**A real blocker found and fixed while confirming item 5 ("nothing
else blocks conv_central") — not assumed clean, checked.** Grepped the
whole file for `pyo.value(m.` after the import/call change and found
THREE sites reading `m.solar_generation[t]` / `m.gas_generation[t]` /
`m.storage_discharge[t]` by a bare integer index — `Var(T)` in the
annual builder, but `Var(T, S)` (slice-indexed) in `build_model_
sliced()`. A bare `[t]` index on a 2-D-indexed Pyomo `Var` raises. All
three fixed by reading the already-slice-aggregated per-year series
`extract_planning_diagnostics` already computes (its own `_slice_
total` helper, confirmed builder-agnostic in Part 1) instead of
indexing `m` directly — not a new aggregation, a re-point to data
`diag` already carries correctly.

**Old (`system_lcoe`'s `disc_generation`):**
```python
disc_generation = sum(
    float(pyo.value(m.DF[t])) * (
        float(pyo.value(m.solar_generation[t]))
        + float(pyo.value(m.gas_generation[t]))
        + float(pyo.value(m.storage_discharge[t]))
    )
    for t in range(len(years))
)
```
**new:**
```python
solar_gen_by_year = diag["solar_generation_twh_e_by_year"]
gas_gen_by_year = diag["gas_generation_twh_e_by_year"]
storage_disch_by_year = diag["storage_discharge_twh_e_by_year"]

disc_generation = sum(
    float(pyo.value(m.DF[t])) * (
        solar_gen_by_year[int(years[t])]
        + gas_gen_by_year[int(years[t])]
        + storage_disch_by_year[int(years[t])]
    )
    for t in range(len(years))
)
```

**Old (`gas_share_2030`/`solar_share_2045`):**
```python
"gas_share_2030":   (
    float(pyo.value(m.gas_generation[5]))
    / max(float(pyo.value(m.solar_generation[5]))
          + float(pyo.value(m.gas_generation[5]))
          + float(pyo.value(m.storage_discharge[5])), 1e-9)
),
"solar_share_2045": (
    float(pyo.value(m.solar_generation[len(years)-1]))
    / max(float(pyo.value(m.solar_generation[len(years)-1]))
          + float(pyo.value(m.gas_generation[len(years)-1]))
          + float(pyo.value(m.storage_discharge[len(years)-1])), 1e-9)
),
```
**new:**
```python
"gas_share_2030":   (
    gas_gen_by_year[int(years[5])]
    / max(solar_gen_by_year[int(years[5])]
          + gas_gen_by_year[int(years[5])]
          + storage_disch_by_year[int(years[5])], 1e-9)
),
"solar_share_2045": (
    solar_gen_by_year[int(years[len(years)-1])]
    / max(solar_gen_by_year[int(years[len(years)-1])]
          + gas_gen_by_year[int(years[len(years)-1])]
          + storage_disch_by_year[int(years[len(years)-1])], 1e-9)
),
```
**No value changes on the annual path** — `_slice_total`'s own annual
branch is `pyo.value(comp[t])`, byte-identical to what these lines did
before; the fix only stops a crash on the sliced path, it does not
change what either expression computes on either builder.

**Nothing else needed changing.** Re-grepped the whole file for every
`pyo.value(m.` / `hasattr(m,` after the fix: `m.system_cost_npv`
(scalar Expression), `m.DF[t]` (annual discount factor), `m.solar_
public_add[t]`/`m.solar_eaas_add[t]` (annual investment decisions),
`m.storage_capacity_mwh[...]` (annual capacity state) — all identical
in both builders per Part 1's own findings; `leakage_diagnostic(m)`
reads four scalar Expressions, confirmed safe in Part 1E. **Item 5's
own requirement — confirm `18_` can now run at `conv_central`, and
state that nothing else blocks it — is satisfied: this three-site fix
was the one thing that did, and it is fixed.**

`python3 -m py_compile scripts/18_run_pareto_frontier.py` — passes.

#### `scripts/28_run_capital_frontier.py`

**Import — old:**
```python
from src.optimize_model import build_model, solve_model
```
**new:**
```python
from src.optimize_model import solve_model
from src.optimize_model_sliced import build_model_sliced
```

**Call — old:**
```python
m = build_model(
    scenario=scenario,
    econ=econ,
    solar_capex_by_year=solar_capex_tv,
)
```
**new:**
```python
m = build_model_sliced(
    scenario=scenario,
    econ=econ,
    solar_capex_by_year=solar_capex_tv,
)
```

**No sliced-unsafe indexing found — checked, not assumed.** Re-grepped
the whole file for `pyo.value(m.` after the change: `m.system_cost_
npv` (scalar), `m.solar_public_add[t]`/`m.solar_eaas_add[t]` (annual
investment), `m.storage_capacity_mwh[...]` (annual capacity) — this
runner never computes an LCOE or generation-share field the way `18_`
does, so it never had `18_`'s three-site problem to begin with. `cum_
unserved`/`cum_selfgen` already came from `diag`'s own slice-aggregated
fields (Part 3/4 of this task's earlier implementation). Module
docstring updated with the same "BUILDER: build_model_sliced" section
as `18_`, adapted to state this file's own clean result explicitly
rather than silently relying on `18_`'s fix having been "similar
enough."

`python3 -m py_compile scripts/28_run_capital_frontier.py` — passes.

---

### PART 3 — Frontier A's baseline denominator

All changes in `scripts/18_run_pareto_frontier.py` only (`28_` has no
baseline-file dependency at all — its own PRE-REQUISITES already state
"it does not read another runner's output," unchanged by this task).

**The argument — new, required, no default:**
```python
parser.add_argument(
    "--baseline-diagnostics",
    required=True,
    type=str,
    help="Path to a sliced, conversion-active baseline diagnostics "
         "JSON (01b_run_baseline_sliced.py's own hash8-tagged output "
         "-- e.g. results/baseline_sliced/diagnostics__<hash8>.json). "
         "No default, no glob: ...",
)
```
The module-level `BASELINE_DIAG_PATH = ROOT / "results" / "baseline" /
"diagnostics.json"` constant is REMOVED, not left unused beside the
new flag — there is now exactly one source for this path.

**Validation — `_validate_baseline_diagnostics(baseline_diag, path)`,
new function.** Field used: **`arm_tag`, parsed** — stated which, and
why, per TASK.md's own request. All three candidate signals TASK.md
offered were checked against what `01b_run_baseline_sliced.py` (read
directly) actually writes, not assumed available:

- `conversion_case` as a top-level key — **not present** in 01b_'s
  DIAGNOSTICS file (`diag_out`, the one carrying the per-year `emissions_
  tco2_by_year` series this runner needs) — only in its separate
  SUMMARY file, which has no per-year series at all.
- "presence of a conversion diagnostic" — same problem: `"conversion":
  _conv_diag` also lives only in 01b_'s summary dict, confirmed by
  reading `01b_run_baseline_sliced.py:478-521` (summary) versus
  `:544-569` (`diag_out`) directly.
- `arm_tag` — **present in `diag_out`** (confirmed, line 569),
  `"__"`-joined, 15 segments, `conversion_case` at index 13 (0-indexed)
  — read directly from `01b_`'s own tag-construction line
  (`scripts/01b_run_baseline_sliced.py:269`). **The field used.**

**One check does both jobs.** Rather than a separate "is this sliced"
test plus a separate "is conversion active" test, a single check
(`conversion_segment == "conv_none"` → raise) suffices for both: the
ANNUAL builder raises `ValueError` on any `conversion_case !=
"conv_none"` (confirmed, this port's own module docstring and Part 1),
so no annual-origin diagnostics file can ever carry a non-`"conv_none"`
conversion segment — a file that passes this one check could only have
come from `build_model_sliced()`. Also checks `arm_tag`'s bare presence
(catching a missing-field or wrong-file case, e.g. an annual `01_`
output, before attempting to parse it) and the segment count (catching
a malformed/differently-shaped tag rather than silently mis-indexing
it). All three failure messages name the specific problem and the fix
(re-run `01b_` with an active `--conversion-case`, or check the file
path). Tested with four fabricated `arm_tag` strings (conversion-active
→ passes; `conv_none` → rejected; missing `arm_tag` → rejected;
malformed/too-few-segments tag → rejected) — pure Python, no scenario,
no builder, no solver — all four behaved as designed.

**Fragility, stated in the function's own docstring, not hidden:**
this parses a fixed-position `"__"`-joined string. It breaks if `01b_`'s
own tag format ever changes. `01b_run_baseline_sliced.py` is outside
this task's whitelist, so it cannot be given a cleaner top-level
`conversion_case` field in its diagnostics output here — a follow-on
adding one there would let this function stop parsing a string.

**Provenance recorded in every output row** — `baseline_diagnostics_
path` and `baseline_arm_tag`, added to all three row shapes (`optimal`,
`infeasible`, `error`) in `all_rows.append(...)`, not only the
successful ones.

**PRE-REQUISITES updated** to name `01b_run_baseline_sliced.py`,
explicitly stating `01_run_baseline.py` (annual) can no longer be used
("cannot represent conversion"). RUN SEQUENCE updated to show the new
required flag.

`python3 -m py_compile scripts/18_run_pareto_frontier.py` — passes
(already confirmed above; restated here since Part 3's own edits are
in the same file as Part 2's).

---

### PART 4 — REPORT-ONLY FINDINGS

No file edited for this Part — findings and drafts only, per the
whitelist.

#### 4A — L5 is stale, verified

**Reading verified**, against both the register's own text and
`docs/findings_frozen.md`'s citations of it:

L5's current text (`docs/limitations_register.md:122-143`) claims
*"Optimal storage is identically zero for any cost"* with **Status:
PENDING (2.3 may resolve)** — and its own "Mitigation attempted"
paragraph already anticipated exactly this check: *"Whether that is
sufficient is an empirical question resolved by the first post-2.3
solve."* TASK.md's fresh-run finding (78,882 MWh built, zero discharge
in every year) **is that resolution, and it falsifies the "identically
zero" claim** — storage is NOT zero, it is built at a materially large
quantity, for its firm-capacity credit alone (`capacity_credit['storage']
= 1.0`, plan 2.3), never for energy value. The STRUCTURAL mechanism L5
describes (`discharge ≤ RTE × charge` giving a non-positive net energy
contribution) is unchanged and still true; only the CONCLUSION drawn
from it ("so storage is never built") no longer holds, because plan
2.3 introduced a second, independent reason to build storage that has
nothing to do with its energy economics.

**Already implicitly superseded, but only in `findings_frozen.md`, not
in the register itself.** `findings_frozen.md`'s own "Scoped conditions
on every result above" table cites L5 as *"storage capacity credit is
flat 1.0 -- magnitude is a residual"* (line 877) — the CORRECTED
framing, not the register's own stale "identically zero" text — and
its F-19 (*"Storage magnitude is not a cost optimum... STRUCTURAL.
Unchanged, and now demonstrated against a priced asset"*) states the
post-2.3 mechanism directly, citing L5 by name for its root cause. The
register entry itself is the one document in this project that has not
caught up to its own cited resolution.

**Amendment drafted, NOT applied (register outside this whitelist):**

> **Amendment (task_epsilon_constraint, Part 6B, Part 4A).** L5's
> central claim — *"Optimal storage is identically zero for any
> cost"* — is FALSIFIED post-2.3, exactly along the axis this entry's
> own "Mitigation attempted" paragraph anticipated. A fresh `01_run_
> baseline.py` run at central arms builds **78,882 MWh** of storage
> with **zero discharge in every year**. The structural mechanism
> (`discharge ≤ RTE × charge` giving storage a non-positive net energy
> contribution) is unchanged — storage is still never dispatched for
> energy value — but plan 2.3's flat `capacity_credit['storage'] = 1.0`
> gives it a firm-capacity-credit channel entirely independent of that
> energy value, and the LP now builds storage FOR THE CREDIT ALONE.
> `findings_frozen.md`'s own F-19 and its "Scoped conditions" citation
> of this entry (*"storage capacity credit is flat 1.0 -- magnitude is
> a residual"*) already reflect this; only this entry's own text does
> not. **Status: PENDING (2.3 may resolve) → RESOLVED — see F-19;
> storage IS built, for capacity credit only, never for energy.** The
> existing "Watch item" (flat 1.0 capacity credit, SOURCE: [PENDING])
> is now this entry's live open question, not the dispatch question
> the original "What" paragraph was about.

#### 4B — the bare-path dependency, every consumer

Grepped `scripts/` for every reference to the bare `results/baseline/
diagnostics.json`, `results/baseline/summary.json`, and `results/
baseline/timeseries.csv` paths. **Independently confirmed the files'
own age** (not just repeating TASK.md's claim):
```
results/baseline/diagnostics.json   Aug 23 13:44 2026
results/baseline/summary.json       Aug 23 13:44 2026
results/baseline/timeseries.csv     Aug 23 13:44 2026
```
Five consumers found (one is `18_`, fixed by this same task — Part 3
above):

| Consumer | Bare file(s) used | What it uses it for | What it would take to see current data |
|---|---|---|---|
| `18_run_pareto_frontier.py` | `diagnostics.json` | ε-axis denominator (baseline annual emissions) | **FIXED, this task, Part 3** — now `--baseline-diagnostics`, required, validated |
| `00_build_emissions_cap.py` | `diagnostics.json` | Reads `gas_to_power_twh_th_by_year` to compute the modelled 2025/2030/2035 baseline emissions that NDC3.0 caps are scaled from; writes `data/cost/processed/emissions_cap.csv` | An explicit, required `--baseline-diagnostics`-style argument (same pattern as this task gave `18_`) — currently a bare module constant, no CLI flag exists to override it at all. **Outside this task's whitelist (not `18_`/`28_`); reported, not fixed.** |
| `08_phase8_analysis.py` | `summary.json`, `timeseries.csv` | The `"baseline_no_policy"` comparison case in a five-case cross-scenario analysis/plot | Same fix pattern; outside whitelist, reported only |
| `99_validate_baseline.py` | `timeseries.csv` | Reads the 2025 row to validate model output against `OBSERVED_2024` (this file's `OBSERVED_2024` dict was itself corrected in `task_wrong_output_repair` Part 2 — the bare-path READ mechanism was untouched then, out of that task's scope, and remains untouched now, out of this one's) | Same fix pattern; outside whitelist, reported only |
| `plot_system_diagnostics.py` | `timeseries.csv` | The `"baseline"` reference series in a multi-panel comparison plot against NDC scenarios | Same fix pattern; outside whitelist, reported only |

**`00_build_emissions_cap.py` and the four frozen findings — stated
plainly, as required.** `02_run_ndc_caps.py` consumes `data/cost/
processed/emissions_cap.csv`, which `00_build_emissions_cap.py` builds
**entirely from** the stale bare `diagnostics.json`'s `gas_to_power_
twh_th_by_year` series (read directly, `scripts/00_build_emissions_
cap.py:128-144`) — **yes, the caps currently in `emissions_cap.csv`,
and therefore every NDC-constrained result `02_run_ndc_caps.py`
produces, are currently derived from the 2026-08-23 file.**
`emissions_cap.csv`'s own file modification time is newer (**Sep 15
07:40**, independently checked) than the baseline diagnostics it was
built from — but since `00_`'s only input path is the same
unrefreshed bare constant, that later run still computed from the
identical stale `gas_to_power_twh_th_by_year` values. **A newer output
file mtime does not imply newer input content** — stated explicitly
because it would be the natural, wrong inference from the timestamps
alone.

**A structural note on the fix, for whoever picks up the four
out-of-whitelist consumers:** `01_run_baseline.py` does not, and by
design cannot cleanly, overwrite the bare path itself — its own
hash8-tagged naming exists specifically because the full arm-tag
string exceeds a filesystem path-component length limit (confirmed
earlier in this task sequence, `task_shared_fn_repair`/`task_runner_
repair`'s own output-naming notes). Any fix for the remaining four
consumers is therefore not "make `01_` also write the bare path again"
but the same explicit-required-argument pattern this task gave `18_`
— or a documented, separate "latest" alias step, not attempted here.

#### 4C — the two 1D defects, register draft

**TASK.md's own assumption checked, and corrected**: TASK.md 4C says
"next free number after L29 (read the file; do not assume)." Read the
file. **L29 does not exist in the actual register — the current
highest applied entry is L28** (confirmed: `grep -n "^## L"
docs/limitations_register.md` shows L28 as the last entry; an L29 was
DRAFTED in a prior task's report, `task_defect_closure.md`, but — like
every register draft in this task sequence — never applied, since the
register has been outside every task's whitelist so far). **The next
free number is therefore L29 itself, not L30.** Corrected here rather
than silently matched to TASK.md's assumption.

**Draft, NOT applied (register outside this whitelist):**

> ## L29 — `cost_decomposition["components"]` and `npv_capex_report`/
> `npv_capex_full` silently omit conversion capital expenditure on a
> sliced, conversion-active run
>
> **Status:** LIVE. Found by `task_epsilon_constraint`'s own Part 6
> inventory (reading `extract_planning_diagnostics` end to end while
> confirming it was safe to call on a sliced model), before any
> conversion-active sliced run had ever consumed either field —
> `task_epsilon_constraint` Part 6B's own port (this entry's sibling
> Part) is what makes such runs routine for the first time, via `18_
> run_pareto_frontier.py` and `28_run_capital_frontier.py`.
>
> **What.** Two separate fields in `extract_planning_diagnostics`
> (`src/optimize_experiments.py`) manually enumerate "the capital
> channels" without a conversion branch, even though `src/optimize_
> model_sliced.py`'s own `m.capital_charge_total_npv` (under
> `f1_tranche`) and gross capex both structurally include a conversion
> retrofit term when an available conversion arm is active:
>
> 1. `cost_decomposition["components"]` (`f1_tranche` branch) lists
>    exactly `gas_capital_charge_npv`, `public_solar_capital_charge_
>    npv`, `eaas_solar_capital_charge_npv`, `storage_capital_charge_
>    npv`, `rebuild_capital_charge_npv`, `capital_charge_total_npv` —
>    no `conversion_capital_charge_npv` entry. The TOP-LEVEL `real_
>    expenditure_npv`/`cost_total_npv` stay correct (they read `m.
>    capital_charge_total_npv` directly, which already includes
>    conversion symbolically, so the reconciliation identity from
>    `task_wrong_output_repair` Part 4 still closes exactly) — but the
>    itemised `components` breakdown will not sum to its own listed
>    `capital_charge_total_npv` entry on such a run, by exactly the
>    conversion capital charge, unlabelled.
> 2. **`npv_gross_capex_report_window_usd` / `npv_capex_full` (the
>    `_capex_at()` helper) omit conversion retrofit capex from the
>    TOTAL itself, not merely from a breakdown.** `_capex_at()` has
>    `hasattr`-guarded branches for `solar_public_add`, `solar_eaas_
>    add`, `storage_add`, `gas_add` — none for `convert_mw` (confirmed:
>    grepped the whole function for `"convert"` — the only conversion-
>    aware code anywhere in it is `_converted_generation_total`, used
>    solely for the `total_generation`/`solar_share_of_grid_generation`
>    computation, already correctly guarded and already exercised).
>
> **This is the thesis's REPORTED FINANCING FIGURE, not a side
> quantity.** `findings_frozen.md`'s own reporting rule states this
> figure — `npv_gross_capex_report_window_usd` — must be reported,
> never `system_cost_npv`, for financing-mobilisation claims. So item 2
> above understates the headline capital-mobilisation number by the
> WHOLE conversion retrofit spend (implied ~2.27 bn at central) on any
> sliced, conversion-active run.
>
> **No published figure is wrong yet — because no conversion-active
> sliced run has consumed either field before now.** `task_epsilon_
> constraint`'s own Part 6 inventory confirmed neither `18_` nor `28_`
> (the only sliced-model runners in this project with an active
> conversion arm as a live possibility) reads `cost_decomposition[
> "components"]` or `npv_capex_report`/`npv_capex_full` at all. **This
> port makes such runs routine — that is the point of this entry**: the
> gap was inert until now, and is no longer guaranteed to stay that way
> once Zeus runs either frontier at an active `--conversion-case`.
>
> **Direction of bias.** Both understate — a reader using either field
> on a conversion-active sliced run gets a number LOWER than the true
> capital commitment; `system_cost_npv` and the top-level cost-
> decomposition reconciliation are not affected.
>
> **What would resolve it.** Add a `conversion_capital_charge_npv`
> entry to `_capital_components` (both branches, guarded on `hasattr(m,
> "convert_mw")` or equivalent), and a `convert_mw`-guarded branch to
> `_capex_at()`. Both are `src/optimize_experiments.py` edits, outside
> every whitelist this task sequence has granted so far.

---

### Why the port cannot move the anchors

Every edit in both files is one of three kinds: (a) an import-line
change (which module a name is bound from — no computation), (b) the
builder-call swap itself (`build_model` → `build_model_sliced`,
keyword arguments byte-identical, confirmed call-compatible in Part 1)
— and swapping which function BUILDS a model is, by construction,
outside "editing a builder," which this task's whitelist forbids and
this edit does not do: neither `src/optimize_model.py` nor `src/
optimize_model_sliced.py` is touched, both are used exactly as they
already exist, and (c) plumbing that reads an already-solved model's
already-computed diagnostics more carefully (the three `diag[...]`
substitutions in `18_`) or adds a new, validated, required input
argument (`--baseline-diagnostics`) whose only effect is which file a
denominator is read from. **Neither frontier's own frozen-anchor-
producing script is touched at all**: the regression anchor
(`21,811,042,472.821114 / 45,502.9754 MWh`, `om_none`) comes from
`01b_run_baseline_sliced.py`, and the central anchor (`27.954 bn /
65,613.0 MWh`) from `02_run_ndc_caps.py` — neither is imported, called,
or edited by `18_` or `28_`, before or after this port. The port
changes what `18_`/`28_` themselves compute (correctly enabling
conversion, which they could not represent at all before) — it cannot,
by construction, retroactively change a number an unrelated, unedited
script already produced.

---

### Encountered but not asked about

- **`26_run_f3_identification.py` carries a THIRD independent copy of
  `bankability_slack_per_mw_by_year`** (its own line 325), alongside
  `03_run_ndc_eaas.py`'s original and the copy this task sequence
  added to `src/optimize_experiments.py` for D17
  (`task_defect_closure`, Part 4). Found while reading `26_` for Part
  1's own 1C; not touched (outside this task's whitelist, and outside
  the specific defect this task closes).
- **`00_build_emissions_cap.py` has no CLI flag at all** — `BASELINE_
  DIAG_PATH` is a bare module constant with no override mechanism,
  unlike `18_` before this task (which at least had the constant in
  one place, even if unoverridable). Fixing it would need the same
  `add_standard_arms`-adjacent or bespoke-flag treatment `18_` just
  got, plus a decision about whether `00_` needs the full standard-arm
  surface or just the one path argument — a smaller, more targeted fix
  than `18_`'s own full-arm-runner shape. Not begun; outside this
  task's whitelist (`00_` is not `18_`/`28_`).
- **`08_phase8_analysis.py` and `plot_system_diagnostics.py` have no
  CLI at all either** (confirmed: neither imports `argparse`) — both
  hardcode their full `CASES` dict of input paths, of which the bare
  baseline path is only one entry among several (NDC/NDC-EaaS result
  directories are also hardcoded there, unrelated to this task's
  scope, not audited further here).
- **The `emissions_cap.csv` mtime-vs-content mismatch** (4B) is a
  general trap worth naming beyond this specific instance: ANY
  pipeline stage whose sole input is an unrefreshed bare path will
  show a recent mtime on every re-run while silently carrying stale
  content indefinitely — the output timestamp cannot be used as a
  freshness proxy for a chain like this one.
- No script, solver, or `smoke_all.sh` was run at any point in this
  task, per the whitelist. Verification was `py_compile` on both
  files, direct reading of every file named (`01b_run_baseline_
  sliced.py`, `00_build_emissions_cap.py`, `08_phase8_analysis.py`,
  `99_validate_baseline.py`, `plot_system_diagnostics.py`,
  `docs/limitations_register.md`, `docs/findings_frozen.md`), an
  independent filesystem check of the three bare baseline files' and
  `emissions_cap.csv`'s modification times, and a synthetic,
  fabricated-`arm_tag` functional test of `_validate_baseline_
  diagnostics` (four cases, pure Python, no scenario, no builder, no
  solver).

**Then STOP.** Do not commit. Zeus reviews `git diff`, commits, and
runs both frontiers.
