# Phase 2.3b — replacement capex and storage capacity credit arms

Scope: two defects that were harmless while storage was structurally zero and
are load-bearing now that storage builds (77,388 MWh in 2045 under
peak_unsuppressed). **Diagnose and edit only — no solver, no runner, no smoke
suite was executed.** Verification was limited to `python -m py_compile` on
every touched file, plain-import checks, `load_scenario()` unit-level calls
(pure Python dict construction, not a model build/solve), and `--help` on all
11 runners (argparse only, exits before `main()` body runs). Peak-shape arms,
`par_block` derivation, `within_block_peaking_factor`, and the firm-capacity
constraint itself were **not** touched.

---

## STEP 0 — inventory

### 0.1 — the retirement warning

`src/optimize_model.py:1180-1214` (identical logic in
`src/optimize_model_sliced.py`'s equivalent guard, called from the same
`solve_model()` in `src/optimize_model.py`):

```python
    # ---- Post-solve guard: in-horizon retirement (plan 2.5)
    # Storage life (15 yr) is shorter than the horizon (21 yr). Any vintage
    # built early enough to retire before the horizon ends is still dispatched
    # by the model, and no replacement capex is charged. Inert as at Phase 2.5
    # (first storage build 2038) but Phases 2.6/2.7 may shift the optimum
    # earlier. This warns rather than raises so that exploratory runs proceed.
    try:
        _lives = scenario.get("asset_lifetimes", {})
        _T = list(range(len(scenario["years"])))
        _T_last = _T[-1]
        _checks = [
            ("storage", _lives.get("storage"), lambda t: pyo.value(model.storage_add[t])),
            ("solar",   _lives.get("solar"),
             lambda t: pyo.value(model.solar_public_add[t]) + pyo.value(model.solar_eaas_add[t])),
            ("gas",     _lives.get("gas"),   lambda t: pyo.value(model.gas_add[t])),
        ]
        for _name, _life, _getter in _checks:
            if not _life:
                continue
            for t in _T:
                if (_T_last - t + 1) > float(_life) and _getter(t) > 1e-6:
                    print(
                        f"  WARNING [plan 2.5]: {_name} vintage {scenario['years'][t]} "
                        f"retires within the horizon (life {_life:.0f} yr < "
                        f"{_T_last - t + 1} yr served). No replacement capex is "
                        f"charged and the asset is still dispatched. Salvage for "
                        f"this vintage is 0. Replacement capex is now required."
                    )
                    break
```

Checked for **storage, solar, AND gas** — not storage alone.

### 0.2 — per-technology life vs. the horizon

From `asset_lifetimes()` (`src/scenarios.py:610-643`) and
`MODEL_END_YEAR=2054` (`src/scenarios.py:282`); horizon = 2025-2054 =
**30 years**, `T_last = 29`:

| tech | life | retires within horizon? | why (from code) |
|---|---|---|---|
| solar | 30 yr | **No** | `(T_last - t + 1) > life` never fires for any `t`: the earliest vintage (`t=0`) serves exactly 30 yr = life, not more. Boundary case, not "within." |
| storage | 15 yr | **Yes**, `t=0..14` (build years 2025-2039) | `15 < 30`; first rebuild due at `t+15`. |
| gas | 30 yr | **No, and structurally inert regardless of life** | `m.gas_add = pyo.Param(T, initialize=0.0, within=pyo.NonNegativeReals)` (`src/optimize_model.py:327`) — a fixed Param, not a decision variable. `gas_add[t]` is 0 for every `t`, so any expression built on it evaluates to 0 no matter what the life comparison says. |
| hydro | n/a | n/a | Not a "built" asset: no addition variable, no capex, no lifetime treatment anywhere in either builder — an exogenous fixed series (`hydro_baseline_twh`, `hydro_growth`). Excluded from Step 1. |

**So under today's parameters, only storage actually retires within the
horizon.** This is a fact about the current numbers (30-yr horizon exactly
matching solar/gas life, `gas_add` pinned at 0), not something to hard-code —
Step 1's implementation is generic across all three techs, so it stays
correct if the horizon is extended past 30 years or `gas_add` is ever freed
(flagged in the code itself as "required for the plan 5.3 experiment").

### 0.3 — salvage computation

`src/optimize_model.py:879-907` (byte-identical logic in
`src/optimize_model_sliced.py:873-901`, before this task's edits):

```python
    def _salvage_fraction(t, life):
        """Straight-line residual fraction at end of horizon for vintage t.
        Vintage t operates years t..T_last inclusive = (T_last - t + 1) years.
        Returns 0.0 for vintages that retire within the horizon."""
        served = T_last - t + 1
        return max(0.0, float(life) - served) / float(life)

    m.salvage_npv = pyo.Expression(
        expr=m.DF[T_last] * (
            pyo.quicksum(
                (m.solar_public_add[t] + m.solar_eaas_add[t])
                * m.solar_capex_param[t]
                * _salvage_fraction(t, lives["solar"])
                for t in T
            )
            + pyo.quicksum(
                m.storage_add[t] * econ["STORAGE_COST_PER_MWH"]
                * _salvage_fraction(t, lives["storage"])
                for t in T
            )
            + pyo.quicksum(
                m.gas_add[t] * scenario["gas_capex_per_mw"]
                * _salvage_fraction(t, lives["gas"])
                for t in T
            )
        )
    )
```

Straight-line residual at `T_last`, correctly zero for a vintage that fully
depreciates within the horizon (`served >= life`). Applied to `system_cost_npv`
(the objective) only — not the public budget constraint (`m.public_budget_constraint`
continues to see gross capex; see the comment immediately above this block).

### 0.4 — capacity accumulation from vintages ⚠️

`src/optimize_model.py:343-361` (identical pattern in
`src/optimize_model_sliced.py:443-461`):

```python
    def gas_cap_rule(m, t):
        year_t = int(years[t])
        if retirement_start is not None and year_t >= retirement_start:
            retired = retirement_rate * (year_t - retirement_start + 1)
            retired = min(retired, gas_baseline)
        else:
            retired = 0.0
        return gas_baseline - retired + sum(m.gas_add[k] for k in range(0, t + 1))
    m.gas_capacity_mw = pyo.Expression(T, rule=gas_cap_rule)

    def storage_cap_rule(m, t):
        return sum(m.storage_add[k] for k in range(0, t+1))
    m.storage_capacity_mwh = pyo.Expression(T, rule=storage_cap_rule)
```

and `solar_cap_rule` (`:513-518`):
`baseline_mw + sum(m.solar_public_add[k] + m.solar_eaas_add[k] for k in range(0, t+1))`.

**Answer: a retired vintage is never removed. All three are pure cumulative
sums of every addition ever made.** Only the *baseline* gas fleet has an
explicit retirement schedule (`gas_baseline - retired`); nothing ever leaves
`solar_capacity_mw`, `gas_capacity_mw`, or `storage_capacity_mwh` once added.

**This is flagged prominently per the task's instruction, because it is a
more serious defect than the missing capex, and Step 1 does NOT fix it.**
These same expressions feed **both**:
- the generation/dispatch expressions (e.g. `m.solar_generation = m.solar_capacity_mw[t] * solar_energy_per_mw`), and
- `m.firm_capacity_constraint` directly (`src/optimize_model.py:823-832`):
  ```python
  m.firm_capacity_constraint = pyo.Constraint(
      T,
      rule=lambda m, t:
          capacity_credit["solar"] * m.solar_capacity_mw[t]
          + capacity_credit["gas"] * m.gas_capacity_mw[t]
          + capacity_credit["storage"] * m.storage_power_mw[t]
          + capacity_credit["hydro"] * hydro_capacity_mw
          >= peak_mw[t] * within_block_peaking_factor * (1.0 + _reserve_margin_by_year[t])
  )
  ```
  A retired vintage stays fully dispatchable **and** fully firm-capacity-credited
  forever. Step 1 (below) charges the missing rebuild *cost*, which corrects the
  economics, but it does **not** remove the retired vintage from these capacity
  expressions — that would change what the constraint and dispatch see, which
  is out of this task's scope (peak-shape arms and the firm-capacity constraint
  itself are explicitly off-limits) and belongs with the Phase 3.0 F1
  annuitised-capex rework. **Do not read Step 1 as having closed this.**

---

## STEP 1 — rebuild-on-retirement

Implemented identically in `src/optimize_model.py` and
`src/optimize_model_sliced.py`, inside the salvage block (both builders share
byte-identical variable names there: `T`, `T_last`, `lives`,
`m.solar_public_add/eaas_add`, `m.storage_add`, `m.gas_add`,
`m.solar_capex_param`, `m.DF`, `econ`, `scenario`).

**Mechanism**: for a vintage built at `build_t` with life `life`, every
in-horizon rebuild lands at `build_t + k*life` for `k = 1, 2, ...` while that
index stays `<= T_last`. At each such `rebuild_t`, full capex is charged
again — at the CAPEX rate applicable in `rebuild_t` (`m.solar_capex_param[rebuild_t]`
for solar, since ATB capex is time-varying; flat scalars for storage/gas) —
discounted at `m.DF[rebuild_t]` (the social rate in the year the rebuild
occurs, per the task's instruction). The replacement vintage then gets the
**same** `_salvage_fraction()` treatment as an original build, evaluated at
its own `rebuild_t`, so a rebuild that doesn't fully depreciate before
`T_last` still earns a residual credit at the horizon end.

Under current parameters (Step 0.2) only storage ever actually rebuilds
(`k=1` only, since storage life 15 yr × 2 = 30 yr = the full horizon); solar
and gas terms are present but evaluate to an empty rebuild set / zero
respectively — this is not hard-coded, it falls out of the same generic loop.

**`m.cost_replacement_capex_npv`** is a separate `pyo.Expression`, not folded
into `gas_capex_npv` / `storage_capex_npv` / `public_solar_capex_npv` /
`eaas_solar_capex_npv`, added into `system_cost_npv` as its own term — so it
changes the objective (closing the defect for real, not just for reporting)
and is independently readable via `pyo.value(m.cost_replacement_capex_npv)`.
Naming note: the task's Step 1 suggested `npv_replacement_capex_usd` as an
illustrative name; the codebase's own established convention for this exact
purpose is `m.cost_<x>_npv` (`m.cost_gas_capex_npv`, `m.cost_storage_capex_npv`,
`m.cost_salvage_npv` in the sliced builder, etc.), so `m.cost_replacement_capex_npv`
was used instead, for consistency with the sibling fields it sits next to —
flagged here rather than silently deviating.

**Diff — `src/optimize_model.py`** (identical structural diff in
`src/optimize_model_sliced.py`, shown together below since both are
byte-for-byte the same change):

```diff
     def _salvage_fraction(t, life):
         served = T_last - t + 1
         return max(0.0, float(life) - served) / float(life)

+    # ------------------------------------------------------------
+    # Replacement capex on in-horizon retirement (2.6, phase_2_3b Step 1).
+    # A vintage built at build_t with technical life `life` reaches end of
+    # life at build_t + life; if that falls inside the horizon, the asset
+    # is rebuilt -- full capex charged again, discounted at m.DF[rebuild_t]
+    # (the social rate in the year the rebuild occurs). Repeats for as many
+    # rebuild cycles as fit before T_last. Generic across solar/storage/gas
+    # so it stays correct if the horizon changes or gas_add is ever freed
+    # (plan 5.3) -- under CURRENT lifetimes (solar/gas 30 yr == the 30-yr
+    # horizon, gas_add pinned at 0) only storage actually rebuilds; see
+    # docs/phase_2_3b_replacement_capex.md Step 0.2.
+    #
+    # PROVISIONAL: Phase 3.0 F1 will annuitise capex at owner-specific WACC
+    # via a capital recovery factor truncated at the horizon, superseding
+    # both this rebuild-on-retirement treatment and the salvage credit
+    # below. Do not extend this mechanism past phase_2_3b.
+    # ------------------------------------------------------------
+    def _rebuild_years(build_t, life):
+        """rebuild_t for every in-horizon replacement of the vintage built
+        at build_t, `life` years apart, for as many cycles as fit <= T_last."""
+        life_int = int(round(float(life)))
+        k = 1
+        while build_t + k * life_int <= T_last:
+            yield build_t + k * life_int
+            k += 1
+
     m.salvage_npv = pyo.Expression(
         expr=m.DF[T_last] * (
             pyo.quicksum(
                 (m.solar_public_add[t] + m.solar_eaas_add[t])
                 * m.solar_capex_param[t]
                 * _salvage_fraction(t, lives["solar"])
                 for t in T
             )
             + pyo.quicksum(
                 m.storage_add[t] * econ["STORAGE_COST_PER_MWH"]
                 * _salvage_fraction(t, lives["storage"])
                 for t in T
             )
             + pyo.quicksum(
                 m.gas_add[t] * scenario["gas_capex_per_mw"]
                 * _salvage_fraction(t, lives["gas"])
                 for t in T
             )
+            # Replacement vintages get the SAME straight-line salvage
+            # treatment as original builds, credited at T_last (Step 1).
+            + pyo.quicksum(
+                (m.solar_public_add[build_t] + m.solar_eaas_add[build_t])
+                * m.solar_capex_param[rebuild_t]
+                * _salvage_fraction(rebuild_t, lives["solar"])
+                for build_t in T
+                for rebuild_t in _rebuild_years(build_t, lives["solar"])
+            )
+            + pyo.quicksum(
+                m.storage_add[build_t] * econ["STORAGE_COST_PER_MWH"]
+                * _salvage_fraction(rebuild_t, lives["storage"])
+                for build_t in T
+                for rebuild_t in _rebuild_years(build_t, lives["storage"])
+            )
+            + pyo.quicksum(
+                m.gas_add[build_t] * scenario["gas_capex_per_mw"]
+                * _salvage_fraction(rebuild_t, lives["gas"])
+                for build_t in T
+                for rebuild_t in _rebuild_years(build_t, lives["gas"])
+            )
         )
     )

+    # Replacement capex itself: full capex again at the rebuild year,
+    # discounted at m.DF[rebuild_t]. A SEPARATE, visible cost -- not folded
+    # into gas/storage/solar capex_npv above (2.6, phase_2_3b Step 1). Read
+    # independently via m.cost_replacement_capex_npv (mirrors the
+    # m.cost_<x>_npv reporting pattern used for the other cost components
+    # below).
+    m.cost_replacement_capex_npv = pyo.Expression(
+        expr=(
+            pyo.quicksum(
+                m.DF[rebuild_t]
+                * (m.solar_public_add[build_t] + m.solar_eaas_add[build_t])
+                * m.solar_capex_param[rebuild_t]
+                for build_t in T
+                for rebuild_t in _rebuild_years(build_t, lives["solar"])
+            )
+            + pyo.quicksum(
+                m.DF[rebuild_t] * m.storage_add[build_t] * econ["STORAGE_COST_PER_MWH"]
+                for build_t in T
+                for rebuild_t in _rebuild_years(build_t, lives["storage"])
+            )
+            + pyo.quicksum(
+                m.DF[rebuild_t] * m.gas_add[build_t] * scenario["gas_capex_per_mw"]
+                for build_t in T
+                for rebuild_t in _rebuild_years(build_t, lives["gas"])
+            )
+        )
+    )
+
     # Unserved energy penalty (VoLL in USD/TWh)
```

And in `system_cost_npv` (both files):

```diff
     system_cost_npv = (
         gas_opex_npv
         + public_solar_capex_npv
         + eaas_solar_capex_npv
         + gas_capex_npv
         + storage_capex_npv
+        + m.cost_replacement_capex_npv
         + selfgen_npv
         + unserved_npv
         + carbon_cost_npv
         - m.salvage_npv
     )
     m.system_cost_npv = pyo.Expression(expr=system_cost_npv)
```

**Which technologies replacement capex now applies to**: the mechanism is
wired for **solar, storage, and gas** (all three the warning checks) in both
builders. Under current lifetimes/horizon (Step 0.2), only **storage**
produces a nonzero `m.cost_replacement_capex_npv` contribution; solar and gas
terms are live code, not dead code, and will activate automatically if the
horizon is extended or `gas_add` is freed (plan 5.3) without further edits.
Hydro is excluded (Step 0.2 — not a built asset).

**Where `m.cost_replacement_capex_npv` is reported**: as a model-level Pyomo
`Expression`, in both `src/optimize_model.py` and `src/optimize_model_sliced.py`,
alongside the existing `m.cost_gas_capex_npv` / `m.cost_storage_capex_npv` /
`m.cost_public_solar_capex_npv` / `m.cost_eaas_solar_capex_npv` fields, and
folded into `m.system_cost_npv`. **It is not yet wired into
`extract_planning_diagnostics()` (`src/optimize_experiments.py`) or the
sliced runner's `diag_out` dict** — doing so would mean editing a
2,800-line file (`optimize_experiments.py`) and/or `scripts/01b_run_baseline_sliced.py`'s
diagnostics block, which Step 1 does not ask for ("Implement... in both
src/optimize_model.py and src/optimize_model_sliced.py"). Flagged here per
the task's "STOP and report rather than expanding scope" instruction, rather
than silently expanding into the diagnostics pipeline.

**Provisional note (verbatim as instructed)**: this treatment is
provisional. Phase 3.0 F1 will annuitise capex at owner-specific WACC via a
capital recovery factor truncated at the horizon, which supersedes both
rebuild-on-retirement and the salvage credit. This was not attempted here —
the comment block above `_rebuild_years()` in both files carries the same
note so a future reader hits it in the code, not only in this doc.

---

## STEP 2 — storage capacity credit as a swept arm

### What changed in `src/scenarios.py`

1. New sentinel `_STORAGE_CC_REQUIRED`, alongside `_PEAK_MULTIPLE_REQUIRED` /
   `_RESERVE_MARGIN_REQUIRED`.
2. New required parameter `storage_cc_case=_STORAGE_CC_REQUIRED` on
   `load_scenario()`, with the matching no-default `TypeError` (unknown value
   → `ValueError`) — same two-stage validation pattern as `peak_multiple_case`
   and `reserve_margin`.
3. New `storage_cc_scenarios()`:
   ```python
   def storage_cc_scenarios() -> dict[str, float]:
       return {
           "storage_cc_low":     0.50,
           "storage_cc_central": 0.80,
           "storage_cc_high":    1.00,
       }
   ```
   with the SOURCE: DECLARED JUDGEMENT docstring the task specified (4-hour
   battery covers a 4-hour peak block in principle —
   `min(1.0, duration/peak_hours) = 1.0` — but effective load-carrying
   capability declines with penetration, and 77 GWh is not a low-penetration
   system; `storage_cc_high` reproduces the old flat 1.0 as an explicit upper
   bound).
4. `capacity_credit["storage"]` now reads `storage_cc` (resolved from
   `storage_cc_case`) instead of the old
   `min(1.0, _storage_duration_hours / PERIOD_HOURS["peak"])`.
5. `storage_cc_case` added to the returned scenario dict and to `labels`.
6. `PERIOD_HOURS` dropped from the `optimize_model_sliced` import — it was
   only used in the line just replaced, and is now unused; removed rather
   than left as a dead import (a direct, minimal byproduct of this exact
   edit, not separate scope).

Full diff (`src/scenarios.py`):

```diff
 from src.optimize_model_sliced import (
     SEASON_DAYS,
-    PERIOD_HOURS,
     DEFAULT_SEASON_DEMAND_FACTOR,
     slice_hours,
     _normalised_demand_weights,
     _hydro_season_energy_share,
 )
@@
+def storage_cc_scenarios() -> dict[str, float]:
+    """
+    Storage's firm-capacity credit (fraction of storage_power_mw counted
+    toward the firm-capacity constraint). Named levels low/central/high
+    (2.6, phase_2_3b Step 2).
+
+    SOURCE: DECLARED JUDGEMENT, not measured. A 4-hour battery covers a
+    4-hour peak block in principle -- min(1.0, duration/peak_hours) = 1.0
+    at the model's current 4-hour storage duration and 4-hour peak slice --
+    but effective load-carrying capability declines with storage
+    penetration, and 77 GWh (2045, peak_unsuppressed arm) is not a
+    low-penetration system: the simple duration-coverage ratio overstates
+    what a fleet that size can actually firm.
+
+    storage_cc_central (0.80) matches gas's availability-factor credit,
+    treating storage as no better than fuel-agnostic thermal capacity at
+    this penetration -- the central case, previously implicit.
+    storage_cc_high (1.00) reproduces the old flat duration/peak_hours
+    figure as an explicit, named UPPER bound, not a default.
+    storage_cc_low (0.50) brackets the low end: a materially discounted
+    credit if penetration effects are more severe than the central case
+    assumes.
+    """
+    return {
+        "storage_cc_low":     0.50,
+        "storage_cc_central": 0.80,
+        "storage_cc_high":    1.00,
+    }
+
+
 def _solar_cumulative_cap(solar_build_case: str, years) -> list[float] | None:
@@
 _PEAK_MULTIPLE_REQUIRED = object()
 _RESERVE_MARGIN_REQUIRED = object()
+# Same mechanism, same reason (2.6, phase_2_3b Step 2): storage's
+# firm-capacity credit was a bare, unsourced 1.0 sitting above gas's 0.80 --
+# a silent default here would keep crediting storage as the model's best
+# firm-capacity resource on an undeclared assumption.
+_STORAGE_CC_REQUIRED = object()
@@
     peak_multiple_case=_PEAK_MULTIPLE_REQUIRED,
     reserve_margin=_RESERVE_MARGIN_REQUIRED,
+    storage_cc_case=_STORAGE_CC_REQUIRED,
     start_year: int = MODEL_START_YEAR,
     end_year: int = MODEL_END_YEAR,
 ) -> dict:
@@
+    storage_cc_case   : key of storage_cc_scenarios() -- REQUIRED, no
+                        default (2.6, phase_2_3b Step 2, mirrors
+                        genset_case/peak_multiple_case discipline). One of
+                        'storage_cc_low' / 'storage_cc_central' /
+                        'storage_cc_high'. Storage's firm-capacity credit
+                        (capacity_credit['storage']) is a DECLARED
+                        JUDGEMENT, not a measured constant -- see
+                        storage_cc_scenarios().
     start_year, end_year : planning horizon bounds.
     """
@@
     elif reserve_margin < 0:
         raise ValueError(f"reserve_margin must be >= 0, got {reserve_margin}")

+    # ---- storage_cc_case (required, no default; 2.6, phase_2_3b Step 2).
+    # Mirrors genset_case/peak_multiple_case discipline: storage's
+    # firm-capacity credit is a declared judgement call (see
+    # storage_cc_scenarios()), not a measured constant, so it must be named
+    # explicitly rather than silently defaulting to the old flat 1.0.
+    if storage_cc_case is _STORAGE_CC_REQUIRED:
+        raise TypeError(
+            "load_scenario() requires storage_cc_case (2.6, phase_2_3b Step "
+            "2). Pass one of 'storage_cc_low' / 'storage_cc_central' / "
+            "'storage_cc_high' (see storage_cc_scenarios()). There is no "
+            "default: omitting it would build a firm-capacity constraint "
+            "crediting storage at an unstated, unsourced fraction."
+        )
+    if storage_cc_case not in storage_cc_scenarios():
+        raise ValueError(f"Unknown storage_cc_case: {storage_cc_case}")
+    storage_cc = storage_cc_scenarios()[storage_cc_case]
+
     years = planning_horizon(start_year, end_year)
@@
     capacity_credit = {
         "solar": 0.0,                                                  # SOURCE: [PENDING]
         "gas": _gas_availability_factor,                                # SOURCE: [PENDING] (= gas_availability_factor)
-        "storage": min(1.0, _storage_duration_hours / PERIOD_HOURS["peak"]),  # SOURCE: [PENDING]
+        "storage": storage_cc,                          # SOURCE: DECLARED JUDGEMENT -- see storage_cc_scenarios()
         "hydro": _hydro_dry_avg_mw / _hydro_capacity_mw,                # SOURCE: [PENDING]
     }
@@
         "peak_mw": peak_mw,
         "reserve_margin": reserve_margin_resolved,
         "capacity_credit": capacity_credit,
+        "storage_cc_case": storage_cc_case,
@@
             "peak_multiple": peak_multiple_case,
+            "storage_cc": storage_cc_case,
         },
     }
```

### Builder-side validation ("matching the existing pattern")

Both `build_model()` and `build_model_sliced()` get a guard mirroring the
existing `peak_multiple_case` guard, right after the existing
`capacity_credit` presence/key check — closing the same hand-built-scenario
bypass (a caller that sets `capacity_credit['storage']` directly without
going through `load_scenario()`'s new validation):

```diff
     for _tech in ("solar", "gas", "storage", "hydro"):
         if _tech not in capacity_credit:
             raise ValueError(
                 f"scenario['capacity_credit'] is missing '{_tech}'."
             )
+    # storage_cc_case (2.6, phase_2_3b Step 2): closes the same
+    # hand-built-scenario bypass as the peak_multiple_case guard above, for
+    # a caller that sets capacity_credit['storage'] directly without going
+    # through load_scenario()'s storage_cc_case validation.
+    if scenario.get("storage_cc_case", None) is None:
+        raise ValueError(
+            "scenario['storage_cc_case'] is missing or None (2.6, "
+            "phase_2_3b Step 2). Required so storage's firm-capacity "
+            "credit has a traceable, sourced justification -- see "
+            "scenarios.storage_cc_scenarios()."
+        )
```

(Identical in both `src/optimize_model.py` and `src/optimize_model_sliced.py`.)

### `--storage-cc-case` on the eleven runners

`STORAGE_CC_CASE_CHOICES = ("storage_cc_low", "storage_cc_central", "storage_cc_high")`
and a required `--storage-cc-case` `argparse` argument were added to all
eleven runners that already carry `--peak-case`:
`01_run_baseline.py`, `01b_run_baseline_sliced.py`, `02_run_eaas.py`,
`02_run_ndc_caps.py`, `03_run_ndc_eaas.py`, `04_run_fin2_tariff_sweep.py`,
`22_salvage_sensitivity.py`, `23_run_selfgen_frontier.py`,
`24_run_leakage_arm.py`, `25_run_obligation_trajectory.py`,
`99c_run_genset_arm_sensitivity.py`. Every `load_scenario(...)` call in these
files (including inside per-file helper functions that previously took only
`peak_case` — `run_arm`, `make_scenario`/`solve_point`, `solve_one` in four
files — all now also take `storage_cc_case`) now passes
`storage_cc_case=args.storage_cc_case`. Confirmed with
`python <script>.py --help` for all eleven (argparse exits before `main()`
runs — no solver touched) — all show the new flag correctly.

**Output tagging**: every runner now tags its output filenames (or, for the
two runners that already group outputs into a per-case directory,
`02_run_ndc_caps.py`/`03_run_ndc_eaas.py`, its `case_dir` name) with both
arms, and every summary/row dict gets an explicit `"storage_cc_case"` field
alongside the existing `"peak_case"` field — mirroring how `peak_case` itself
is tagged in each file today (nine of eleven files: a `tag = f"{args.peak_case}__{args.storage_cc_case}"`
variable replacing bare `{args.peak_case}` in filenames; the two `case_dir`
files: `f"{c}_{voll_case}_{args.peak_case}_{args.storage_cc_case}"`;
`04_run_fin2_tariff_sweep.py`: extended its existing `output_tag` variable).
One deliberate non-change: `04_run_fin2_tariff_sweep.py`'s per-row dicts
(`rows`, from `run_fin2_blended_bankability_sweep`) already don't carry a
per-row `"peak_case"` field — only the output filenames are tagged in that
file — so `storage_cc_case` was matched to that same existing pattern rather
than added inconsistently.

Full per-file diffs for all eleven runners plus the three edited files under
`src/` are in `git diff` for this change (1,247 lines); the complete unified
diff was generated with:
```
git diff -- src/optimize_model.py src/optimize_model_sliced.py src/scenarios.py \
  scripts/01_run_baseline.py scripts/01b_run_baseline_sliced.py scripts/02_run_eaas.py \
  scripts/02_run_ndc_caps.py scripts/03_run_ndc_eaas.py scripts/04_run_fin2_tariff_sweep.py \
  scripts/22_salvage_sensitivity.py scripts/23_run_selfgen_frontier.py scripts/24_run_leakage_arm.py \
  scripts/25_run_obligation_trajectory.py scripts/99c_run_genset_arm_sensitivity.py
```
and is the authoritative, line-exact record of every change in this task
(this document's inline diffs are drawn from the same edits but the `git
diff` output is what to cite for exact line numbers, since several edits
shifted line numbers within a file).

### ⚠️ Scope boundary found during the Step 2 inventory — reported, not fixed

Per CLAUDE.md ("Before editing N consumers of a changed quantity, inventory
all of them first") and this task's "If a step is larger than described,
STOP and report rather than expanding scope": making `storage_cc_case`
required-with-no-default in `load_scenario()` (exactly as Step 2 specifies)
means **every** `load_scenario()` call that doesn't pass it now raises
`TypeError`. A repo-wide inventory (`grep -rn "load_scenario("`) found
roughly 20 more call sites beyond the eleven canonical runners:

- `tests/test_run.py` (module-level script)
- `scripts/17_run_monte_carlo.py`, `09b_run_rel1_retirement_sensitivity.py`,
  `18_run_pareto_frontier.py`, `05_run_fin3_capital_sweep.py`,
  `probe_2_6_structure.py`, `19_run_carbon_price_sweep.py`,
  `06_run_gas1_shadow_benchmarks.py`, `09b_run_rel1_mode_sensitivity.py`,
  `15_run_pol1_ndc_comparison.py`, `07_run_gas2_eaas_gas_relief.py` (×2),
  `20_run_cf1_solar_yield_sensitivity.py`
- Ten more `load_scenario()` calls inside `src/optimize_experiments.py`,
  inside `run_bottleneck_sensitivity`, `run_rel3_financing_frontier`,
  `run_dem1_demand_sensitivity`, `run_dem2_growth_gas_matrix`,
  `run_str1_storage_parameter_sweep`, `run_rel2_marginal_cost_curves`,
  `run_rel1_feasibility_matrix`, `run_gas_regime_ndc_matrix`,
  `run_all_deterministic_scenarios`, `run_financing_vs_resource_test`

**None of these were touched.** Checked and confirmed important:
`grep -n "peak_multiple_case\|reserve_margin=" ` across all of them returns
**nothing** — i.e. every one of these call sites was **already** raising
`TypeError` before this task, from the pre-existing `peak_multiple_case`/
`reserve_margin` required-arguments introduced in the earlier "2.6, F0 Step
2.4" round. `tests/test_run.py` in particular is a bare module-level script
(not a `def test_...` a runner would collect) that would already fail at
import. So: **this change does not newly break any currently-functional code
path** — it adds one more missing argument to call sites that were already
non-functional. Also confirmed the three functions the eleven canonical
runners actually depend on for diagnostics —
`extract_planning_diagnostics` (`optimize_experiments.py:1817-2405`),
`run_deterministic_scenario` (`:246-335`), and
`run_fin2_blended_bankability_sweep` (`:827-1001`) — contain **no**
`load_scenario()` call of their own; they consume a scenario dict the
runner already built. Fixing the ~20 already-stale call sites (which would
mean touching a 2,800-line file plus a dozen more scripts) is out of this
task's stated scope and was not attempted.

---

## Where the two changes interact

- The replacement-capex mechanism (Step 1) and the storage capacity-credit
  arm (Step 2) are otherwise independent — Step 1 changes `system_cost_npv`
  (the objective), Step 2 changes `capacity_credit["storage"]` (a coefficient
  in `m.firm_capacity_constraint`), and neither reads the other's output.
- They interact through the LP's optimum, though: `storage_cc_case` changes
  how much storage the firm-capacity constraint is willing to credit per MW,
  which changes how much storage the model chooses to build, which changes
  how many in-horizon rebuilds `_rebuild_years()` actually triggers (more
  storage built early in the horizon → more vintages retiring before 2054 →
  larger `m.cost_replacement_capex_npv`). `storage_cc_low` (0.50) makes
  storage a worse firm-capacity resource than before, which should reduce
  build and therefore reduce replacement capex; `storage_cc_high` (1.00)
  reproduces the old, pre-Step-2 credit exactly, so under `storage_cc_high`
  the *capacity-credit* effect is a no-op and any change in storage build
  volume relative to the pre-Step-1 baseline is attributable to Step 1's new
  replacement-capex cost alone. This makes `storage_cc_high` the natural arm
  to isolate Step 1's effect, and `storage_cc_low` vs `storage_cc_high` (both
  with Step 1 active) the natural pair to isolate Step 2's effect. Neither
  comparison was run — "Run nothing" — but is worth stating for whoever runs
  the sweep next.
- Both changes touch the same `m.salvage_npv` / `system_cost_npv` region of
  code in both builders, so a future editor touching that region should read
  Step 1's `_rebuild_years()` block before changing `_salvage_fraction()` —
  the replacement vintages' salvage terms call the same helper with a
  different index (`rebuild_t` instead of `t`) and would silently go stale
  if `_salvage_fraction()`'s signature changed without updating both call
  sites.
