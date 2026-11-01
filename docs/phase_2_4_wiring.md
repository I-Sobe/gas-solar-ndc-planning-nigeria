# Phase 2.4 Wiring — `peak_multiple_case` / `reserve_margin` into the runners

Scope: the eleven runners named in TASK.md STEP 1. Nothing was run — no
solver, no `NN_*.py` runner, no `smoke_all.sh`. `src/scenarios.py`'s arm
definitions, constants, and values were not touched.

---

## STEP 0 — inventory

### 1. Does `reserve_margin="nirp_spinning"` exist?

Yes, in `src/scenarios.py`'s `load_scenario()`. Two parts, quoted verbatim.

Validation (raises on required-arg omission, rejects any string other than
`'nirp_spinning'`, rejects a negative numeric fraction):

```python
    # ---- 2.6, F0 Step 2.3: reserve_margin (required, no default). Either a
    # numeric fraction (used as-is) or the literal string 'nirp_spinning'
    # (derives a per-year fraction from RESERVE_MARGIN_SPINNING_MW /
    # peak_mw[t], resolved further down once peak_mw is known).
    if reserve_margin is _RESERVE_MARGIN_REQUIRED:
        raise TypeError(
            "load_scenario() requires reserve_margin (2.6, F0 Step 2.3). "
            "Pass a fractional reserve margin (e.g. 0.15), or the literal "
            "'nirp_spinning' to derive it from RESERVE_MARGIN_SPINNING_MW / "
            "peak_mw[t]. There is no default: omitting it would build a "
            "firm-capacity constraint against an unstated reserve "
            "assumption with no error, warning, or visible signature in "
            "the output."
        )
    if isinstance(reserve_margin, str):
        if reserve_margin != "nirp_spinning":
            raise ValueError(
                f"Unknown reserve_margin mode: '{reserve_margin}'. Pass a "
                "numeric fraction, or the literal 'nirp_spinning'."
            )
    elif reserve_margin < 0:
        raise ValueError(f"reserve_margin must be >= 0, got {reserve_margin}")
```

Resolution (once `peak_mw` is known, later in the same function):

```python
    # Resolve reserve_margin now that peak_mw is known (2.6, F0 Step 2.3).
    # 'nirp_spinning': a FIXED absolute MW requirement implies a SHRINKING
    # per-year fraction as peak_mw[t] grows -- both builders accept either a
    # flat scalar or a per-year sequence here (same pattern already used for
    # max_offgrid_fraction).
    if reserve_margin == "nirp_spinning":
        reserve_margin_resolved = [
            RESERVE_MARGIN_SPINNING_MW / mw for mw in peak_mw
        ]
    else:
        reserve_margin_resolved = reserve_margin
```

`RESERVE_MARGIN_SPINNING_MW = 900.0` (comment: "NIRP 2024 Section 2.3, p.19
-- spinning reserve requirement"). Confirmed unambiguous — proceeded past
STEP 0.1 without stopping.

### 2. `01b_run_baseline_sliced.py`'s original `load_scenario()` call, verbatim

```python
    scenario = load_scenario(
        demand_level_case="served",
        demand_case="organic_central",
        capital_case="unconstrained",
        gas_deliverability_case="baseline",
        solar_build_case="deployment_unconstrained",
        land_case="loose",
        carbon_case="no_policy",
        start_year=2025,
        end_year=MODEL_END_YEAR,
    )
```

No argparse anywhere in the original file: no `import argparse`, no
`ArgumentParser(`. `main()` took no arguments; the CLI entry point was
`if __name__ == "__main__": main()`.

### 3. The eleven runners, before this task's edits

| Script | argparse before? | `load_scenario()` call sites (source, not runtime invocations) |
|---|---|---|
| `01_run_baseline.py` | No | 1 |
| `01b_run_baseline_sliced.py` | No | 1 |
| `02_run_eaas.py` | No | 1 |
| `02_run_ndc_caps.py` | No | 1 (inside a `for c, cfg in NDC_CASES.items():` loop — one source call site, invoked once per NDC case at runtime) |
| `03_run_ndc_eaas.py` | No | 1 (same loop pattern, over its own `NDC_CASES`) |
| `04_run_fin2_tariff_sweep.py` | **Yes** (`--concessional-envelope`, `--deployable-fraction`) | 1 (inside a `for ndc_name, cfg in NDC_CASES.items():` loop) |
| `22_salvage_sensitivity.py` | No | 1 (inside `run_arm()`, called once per lifetime arm × per model type) |
| `23_run_selfgen_frontier.py` | No | 1 (inside `make_scenario()`, called once per `solve_point()` invocation) |
| `24_run_leakage_arm.py` | No | 1 (inside `solve_one()`, called once per genset arm × boundary) |
| `25_run_obligation_trajectory.py` | No | 1 (inside `solve_one()`, called once per trajectory) |
| `99c_run_genset_arm_sensitivity.py` | No | 1 (inside `solve_one()`, called once per genset arm × financing arm × eps) |

Confirmed by direct read of every file: **none of the eleven passed
`peak_multiple_case` or `reserve_margin` anywhere**, before this task's edits.
Unambiguous — proceeded to STEP 1.

---

## STEP 1 + STEP 2 — what changed, file by file

Common pattern applied to every script (verbatim in each file):

```python
PEAK_CASE_CHOICES = ("peak_suppressed", "peak_unsuppressed", "peak_nirp", "peak_high")

def parse_args():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument(
        "--peak-case",
        required=True,
        choices=PEAK_CASE_CHOICES,
        help="Peak-demand shape arm fed to load_scenario(peak_multiple_case=...). "
             "No default: an unstated demand shape would build a firm-capacity "
             "constraint silently.",
    )
    return parser.parse_args()
```

`--peak-case` has no default (`required=True`), so an omitted arm fails
argparse's own error immediately (`the following arguments are required:
--peak-case`) — before any scenario is built. Every `load_scenario()` call
in every file now receives `peak_multiple_case=<the CLI value>,
reserve_margin="nirp_spinning"`.

### `01_run_baseline.py`

- Added `import argparse`, `PEAK_CASE_CHOICES`, `parse_args()`.
- `main()` now starts with `args = parse_args()`.
- `load_scenario(...)` call gained `peak_multiple_case=args.peak_case,
  reserve_margin="nirp_spinning"`.
- Output tagging (STEP 2): `diagnostics.json` → `diagnostics__{peak_case}.json`,
  `summary.json` → `summary__{peak_case}.json` (and added a `"peak_case"` key
  inside the summary dict itself, belt-and-suspenders), `timeseries.csv` →
  `timeseries__{peak_case}.csv`. Print statements updated to match.
- Straightforward: the `load_scenario()` call and the three output writes
  are all directly in `main()`.

### `01b_run_baseline_sliced.py`

- Same argparse addition as above (own `parse_args()`, own description).
- `load_scenario(...)` call gained the two new kwargs.
- Output tagging: `summary.json`, `diagnostics.json`, `timeseries.csv`,
  `slice_detail.csv` all gained a `__{peak_case}` suffix; `"peak_case"` added
  into the `summary` dict.
- Straightforward, same shape as `01_run_baseline.py`.

### `02_run_eaas.py`

- Same argparse addition.
- `load_scenario(...)` call gained the two new kwargs.
- Output tagging: `diagnostics.json`, `summary.json` (+ `"peak_case"` field),
  `timeseries.csv` all suffixed.
- Straightforward.

### `02_run_ndc_caps.py`

- Same argparse addition; `main()` now starts with `args = parse_args()`.
- The single `load_scenario(...)` call site (inside the `NDC_CASES` loop)
  gained the two new kwargs — applies to every NDC case, since the arm is
  the same across the whole run.
- Output tagging: this script already namespaces its output per NDC case
  into a subdirectory, `case_dir = RESULTS_DIR / f"{c}_{voll_case}"`. Extended
  in place to `RESULTS_DIR / f"{c}_{voll_case}_{args.peak_case}"` rather than
  renaming the files inside it (`summary.json`, `diagnostics.json`,
  `timeseries.csv` stay unqualified, now inside a peak_case-tagged
  directory). Added `"peak_case": args.peak_case` into the `summary.json`
  payload.
- Not fully mechanical: had to extend an *existing* tagging convention
  rather than invent one, to stay consistent with the directory-per-case
  pattern already in the file.

### `03_run_ndc_eaas.py`

- Same treatment as `02_run_ndc_caps.py` (near-identical file structure):
  argparse added, `load_scenario(...)` call gained the two kwargs,
  `case_dir` extended to `f"{c}_{voll_case}_{args.peak_case}"`, `"peak_case"`
  added to the `summary.json` payload.

### `04_run_fin2_tariff_sweep.py`

- **Already had argparse** — the only one of the eleven. Added `--peak-case`
  to the *existing* parser (`required=True`, same choices/help text as
  everywhere else) rather than building a new one.
- `load_scenario(...)` call (inside the `NDC_CASES` loop) gained the two new
  kwargs.
- Output tagging: this script already tags every output file via
  `output_tag = f"{envelope_arm}__df{args.deployable_fraction:g}"` — this is
  the pattern STEP 2 pointed to as the template. Extended to
  `f"{envelope_arm}__df{args.deployable_fraction:g}__{args.peak_case}"`, so
  every file this script writes (`tariff_sweep_results__{tag}.csv/.json`,
  `tariff_sweep_summary__{tag}.json`) now also carries the peak arm.
- Mutation check (STEP 1's explicit ask): confirmed at lines 211–212 (now
  shifted slightly by the new argparse line), `scenario["financing_regime"]
  = "eaas"` and `scenario["required_margin"] = cfg["required_margin"]` —
  **neither is a demand-shape key** (`period_demand_factor`,
  `season_demand_factor`, `par_block`, `peak_mw`). Confirmed safe.

### `22_salvage_sensitivity.py`

- Same argparse addition; `main()` starts with `args = parse_args()`.
- **Not mechanical at the call site**: `load_scenario()` is called inside
  `run_arm(lives, sliced, econ, capex_tv)`, a helper invoked once per
  lifetime arm × per model type (12 calls at runtime from one source
  call site). Added `peak_case` as an explicit fifth parameter to
  `run_arm()`'s signature and threaded `args.peak_case` into its one call
  site in `main()`, rather than reaching into a closure or module global.
- Output tagging: `trajectories.csv` → `trajectories__{peak_case}.csv`,
  `salvage_sensitivity.json` → `salvage_sensitivity__{peak_case}.json`
  (+ `"peak_case"` key added into the `out` dict written to that JSON).

### `23_run_selfgen_frontier.py`

- Same argparse addition.
- **Not mechanical**: `load_scenario()` lives inside `make_scenario()`,
  which is itself called from `solve_point(builder, pricing, eps, sliced)`,
  which is called from a triple-nested loop in `main()` (builder × pricing ×
  eps — up to 48 calls at runtime). Added `peak_case` as a parameter to both
  `make_scenario(peak_case)` and `solve_point(..., peak_case)`, threaded from
  `main()`'s loop.
- Output tagging: `frontier.csv` → `frontier__{peak_case}.csv`,
  `frontier.json` → `frontier__{peak_case}.json` (+ `"peak_case"` added to
  the JSON's metadata block). Print statements at the end updated to match.

### `24_run_leakage_arm.py`

- Same argparse addition.
- `load_scenario()` lives inside `solve_one(genset_arm, selfgen_in_cap)`,
  called from a comprehension in `main()` over `GENSET_ARMS × (False, True)`.
  Added `peak_case` as a third parameter, threaded through.
- Also added `peak_case=peak_case` into the `base` dict so it appears as a
  field on every row of `leakage_arm.csv` (this script already carries
  several other arm labels — `genset_arm`, `selfgen_in_cap` — as row fields,
  so a field was the more consistent choice here than a filename suffix for
  *that* dimension; the filename itself is *also* tagged, per below, since
  every row in one invocation shares one peak_case and a second run must not
  silently overwrite the first).
- Output tagging: `leakage_arm.csv` → `leakage_arm__{peak_case}.csv`.
- Noted, not touched: `solve_one()` also has a conditional,
  environment-variable-gated mutation, `if os.environ.get("BUSBAR_ARM"):
  sc["selfgen_busbar_factor"] = 1 / (1 - 0.07)`. `selfgen_busbar_factor` is
  not one of the four demand-shape keys, so this is unaffected by the
  inversion — flagged here only for completeness since STEP 1 asked about
  scenario-key mutation generally.

### `25_run_obligation_trajectory.py`

- Same argparse addition.
- `load_scenario()` lives inside `solve_one(traj_name)`, called from a loop
  over `TRAJECTORIES` (3 calls at runtime). Added `peak_case` as a second
  parameter.
- Added `peak_case=peak_case` into the `base` summary dict (same
  field-on-every-row reasoning as `24_run_leakage_arm.py` — this script
  already carries `trajectory` as a row field on both its summary and
  annual CSVs).
- Output tagging: `obligation_trajectory_summary.csv` →
  `..._summary__{peak_case}.csv`, `obligation_trajectory_annual.csv` →
  `..._annual__{peak_case}.csv`.

### `99c_run_genset_arm_sensitivity.py`

- Same argparse addition.
- `load_scenario()` lives inside `solve_one(genset_arm, fin_arm, eps)`,
  called from a triple loop over `GENSET_ARMS × FINANCING_ARMS × EPS_LEVELS`
  (36 calls at runtime). Added `peak_case` as a fourth parameter.
- Added `peak_case=peak_case` into the `base` dict (row field, same
  reasoning as above).
- Output tagging: `genset_arm_sensitivity.csv` →
  `genset_arm_sensitivity__{peak_case}.csv`. Also added `peak_case` to the
  progress print line in `main()`.

---

## Which runners already had argparse, which needed one added

Only **`04_run_fin2_tariff_sweep.py`** already had argparse. The other ten
(`01`, `01b`, `02_run_eaas`, `02_run_ndc_caps`, `03`, `22`, `23`, `24`, `25`,
`99c`) had none and each got a minimal, per-file `parse_args()` — no shared
helper module was introduced, and no module-level constant stands in for the
CLI flag anywhere: every file requires `--peak-case` on its own command
line, so the arm choice is forced into shell history and job logs, per the
task's explicit instruction.

## Runners where the wiring was not mechanical, and why

- **`22`, `23`, `24`, `25`, `99c`**: in every one of these, the
  `load_scenario()` call lives inside a helper function invoked from a loop
  or comprehension in `main()`, not directly in `main()` itself. Each
  needed `peak_case` threaded as an explicit function parameter through one
  or two call layers, rather than a one-line kwarg addition.
- **`02_run_ndc_caps.py` / `03_run_ndc_eaas.py`**: already had an
  established per-NDC-case output-directory tagging convention
  (`case_dir = RESULTS_DIR / f"{c}_{voll_case}"`); extended it in place
  rather than introducing filename suffixes that would have fought the
  existing convention.
- **`04_run_fin2_tariff_sweep.py`**: the one script with pre-existing
  argparse and an pre-existing `output_tag` convention — extended both
  rather than adding a second, parallel mechanism.
- **`24` and `25`, `99c`**: chose to add `peak_case` as a *row field* (these
  scripts already carry other arm labels as row fields on flat, multi-row
  CSVs) in addition to a filename tag, since a single invocation's rows all
  share one peak_case and the filename tag is what actually prevents one
  run's results from silently overwriting another's.

## Mutation of scenario keys after `load_scenario()` — surveyed across all eleven

TASK.md named `04_run_fin2_tariff_sweep.py` specifically. The survey covered
all eleven; here is every post-construction scenario-key mutation found,
none of which touch a demand-shape key
(`period_demand_factor`/`season_demand_factor`/`par_block`/`peak_mw`):

| Script | Keys mutated after `load_scenario()` |
|---|---|
| `01_run_baseline.py` | `solar_min_build_mw_per_year` |
| `01b_run_baseline_sliced.py` | `solar_min_build_mw_per_year` |
| `02_run_eaas.py` | `financing_regime`, `solar_service_tariff_usd_per_twh`, `required_margin`, `solar_min_build_mw_per_year` |
| `02_run_ndc_caps.py` | `solar_min_build_mw_per_year` |
| `03_run_ndc_eaas.py` | `solar_min_build_mw_per_year`, `financing_regime`, `solar_service_tariff_usd_per_twh`, `required_margin`, `public_solar_budget_npv` |
| `04_run_fin2_tariff_sweep.py` | `financing_regime`, `required_margin` |
| `22_salvage_sensitivity.py` | `solar_min_build_mw_per_year`, `asset_lifetimes` |
| `23_run_selfgen_frontier.py` | `solar_min_build_mw_per_year` |
| `24_run_leakage_arm.py` | `solar_min_build_mw_per_year`, `financing_regime`, `required_margin`, and conditionally `selfgen_busbar_factor` (only if env var `BUSBAR_ARM` is set) |
| `25_run_obligation_trajectory.py` | `solar_min_build_mw_per_year`, `financing_regime`, `required_margin` |
| `99c_run_genset_arm_sensitivity.py` | `solar_min_build_mw_per_year`, `financing_regime`, `required_margin` |

**Conclusion: no in-scope runner mutates a demand-shape key.** `par_block`
and `peak_mw`, as derived and stored by `load_scenario()`, are never
invalidated by any of the eleven runners' post-construction edits.

---

## `SOURCE: [PENDING]` markers encountered

- **`src/scenarios.py`**: `PEAK_NIRP_PERIOD_FACTOR = None` /
  `PEAK_NIRP_SEASON_FACTOR = None` — the `peak_nirp` arm. Per TASK.md this is
  correct as-is: NIRP 2024 states no daily load profile anywhere in the
  document, so block factors cannot be derived from it, and selecting
  `--peak-case peak_nirp` on any of the eleven wired runners will raise
  `load_scenario()`'s own `ValueError` (`"...['period_demand_factor'] is
  None -- SOURCE: [PENDING]..."`) by design, not a bug introduced here.
- **`src/scenarios.py`**: `solar_min_build_default()` is explicitly marked
  `STATUS: [UNSOURCED]` in its own docstring — an arbitrary floor pending
  the test described there. Ten of the eleven runners call it and mutate
  `solar_min_build_mw_per_year` with its return value (see mutation table
  above); this task did not touch that call, only confirmed it isn't a
  demand-shape key.
- **`src/scenarios.py`**, inside `load_scenario()`'s `capacity_credit` dict
  (feeds the firm-capacity constraint guarded in both builders):
  `capacity_credit["solar"]`, `["gas"]`, `["storage"]`, `["hydro"]` are all
  commented `# SOURCE: [PENDING]` — declared assumptions, not yet sourced.
- **`within_block_peaking_factor`**: read inside both builders via
  `scenario.get("within_block_peaking_factor", 1.0)` with an inline `#
  SOURCE: [PENDING]` comment, the last time this document's author read
  `src/optimize_model.py`/`src/optimize_model_sliced.py` directly (before
  commit `a2717ac`). TASK.md's own context section states the default is
  now `1.3448` in both builders, and commit `a2717ac`'s message ("2.4:
  sourced peak-shape arms (Oluwole 2016), within-block peaking factor
  1.3448") corroborates that the default changed as part of the same
  commit that sourced the arms. **This was not independently re-read in
  this pass** — STEP 1 of this task only touches the eleven runner scripts,
  not the two builders, and the human's explicit instruction was to stop
  running/checking further and write this report. Flagged so the number is
  traceable to its source (commit message + TASK.md) rather than presented
  as independently re-verified.
- Pre-existing, unrelated to this task's wiring, still outstanding in
  `src/scenarios.py`'s docstrings: `season_solar_factor`,
  `hydro_season_factor`, `hydro_min_flow_fraction` (all `[SOURCE NEEDED]`),
  `asset_lifetime_sweep()`'s band widths (`[SOURCE NEEDED]`), the
  `solar_build_scenarios()` 2.0x `nirp_accelerated` multiplier (`[SOURCE
  NEEDED]`), and every rate in `blended_finance_scenarios()` (marked "ALL
  RATES ARE PLACEHOLDERS pending primary-source verification").

---

## Process note (not part of STEP 1–3, recorded for the record)

Getting to a place where these edits could be made safely took more than
the eleven scripts themselves. This session's background-isolation policy
blocks direct edits to the shared checkout, but a fresh `EnterWorktree`
branches from a git ref and cannot carry uncommitted working-tree state.
Three files this task depends on
(`scripts/04_run_fin2_tariff_sweep.py`, `src/io.py`,
`src/optimize_experiments.py`) were still uncommitted on top of the arm-
sourcing commit (`a2717ac`, already present before this task started). At
the human's explicit direction, those three were committed as `b496d77`
("FIN-2: multi-blend/multi-NDC tariff sweep, GENSET_CASE always set,
zero-subsidy diagnostic"), `worktree.baseRef` was set to `head` for this
repo (in `.claude/settings.local.json`), and only then was a worktree
entered and the eleven scripts edited there. No file governed by TASK.md's
"do not modify" instruction (`src/scenarios.py`'s arm definitions,
constants, values) was part of that commit or touched since.

All eleven edits above are currently uncommitted in the worktree at
`.claude/worktrees/phase-2-4-wiring` (branch `worktree-phase-2-4-wiring`).
Nothing has been committed there and nothing has been run.
