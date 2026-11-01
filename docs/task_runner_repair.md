# TASK — repair the nine directly-broken runners

Branch `phase-fin2-d17`. Supersedes `docs/task_shared_fn_repair.md` (the
six shared functions, now repaired and committed at `b074715`; both
anchors reproduce there). This report covers the nine runners named in
TASK.md, plus the new `src/cli_arms.py` helper module.

Reported incrementally, per TASK.md §0's own warning: "Nine runners plus
a new module is enough to lose a report to."

---

## STEP ONE — `src/cli_arms.py`

### Why it exists

The defect this whole repair sequence closes (register L27) propagated
because the 17/18-flag argparse block feeding `load_scenario()`'s
required arms was hand-copied file to file, not defined once — six
library functions in `src/optimize_experiments.py` (Part 3) and, this
task establishes, at least four of the nine runners here carried their
own hand-copied copy of the same block already (see Shape A below).
`src/cli_arms.py` is the fix for the copying, not for the arms
themselves: it changes nothing about what `load_scenario()` requires.

### API

```python
def add_standard_arms(parser, *, include=None, exclude=None): ...
def standard_arms_kwargs(args) -> dict: ...
```

`add_standard_arms` adds argparse flags for the standard arm set to a
parser; every flag `required=True`, no defaults — matches
`01b_run_baseline_sliced.py`'s discipline exactly (read first, per
TASK.md §2, as the reference implementation). `standard_arms_kwargs`
reads back whichever of those flags are present on the parsed
`Namespace` and returns a `load_scenario()`-ready kwargs dict.
`include=`/`exclude=` (mutually exclusive, `ValueError` if both given)
take an iterable of `load_scenario()` kwarg names to add only a subset;
unknown keys raise `ValueError` rather than silently no-op.

### Which flags it defines, and two things it deliberately does NOT do

**18 flags**, one per arm: `peak_multiple_case` (flag text `--peak-case`,
matching 01b_'s own flag naming, not `--peak-multiple-case`),
`storage_cc_case`, `gas_build_case`, `new_gas_deliverability`,
`gas_availability_case`, `gas_retirement_case`, `gas_capex_case`,
`ccgt_efficiency_case`, `capital_recovery_case`, `dscr_case`,
`us_inflation` (float, not choices), `capital_treatment_case`,
`concessional_envelope_case`, `gas_deliverability_case`,
`conversion_case`, `hydro_case`, `tariff_case`, `om_case`. Choices are
read live from each arm's `*_scenarios()` registry function in
`src/scenarios.py` at call time (a function, not a module constant, so
a future registry change is picked up automatically) — confirmed by
direct comparison (`python3 -c "from src.scenarios import ...; print(tuple(...))"`
under the `nigeria-energy` env) that every live registry's keys
currently match 01b_'s own hardcoded tuples byte-for-byte; no drift
found today, but the point of sourcing live is that this module will
not silently go stale the next time an arm changes.

**Does NOT expose `reserve_margin` as a flag.** It is one of
`load_scenario()`'s 17 sentinel-required keys, but `01b_`'s own
reference implementation does not flag it either — it is written as the
literal `reserve_margin="nirp_spinning"` directly at the `load_scenario()`
call site. Register L9 records this (with `genset_case`) as a canonical
project-wide constant, not a swept arm, and TASK.md Part 3 §2 said so
explicitly for the shared functions: "The runners will pass the literal,
as they already do elsewhere. The convention belongs at the call site,
not inside the library." Every runner touched below still writes
`reserve_margin="nirp_spinning"` itself.

**Three arms hardcoded, not sourced live — a real gap, reported rather
than hidden.** `gas_build_case`, `new_gas_deliverability` and
`capital_treatment_case` are required flags in `01b_`'s own reference
implementation, but `src/scenarios.py` defines **no `*_scenarios()`
registry function for any of the three** — confirmed by listing every
`^def .*scenarios` in the file and separately grepping `src/*.py` for
`gas_ndc_17gw` / `pre_f1_overnight` / `"exempt"`: the only definitions
are inline tuple/string-literal checks inside `load_scenario()` and the
two builders. `src/cli_arms.py` therefore hardcodes these three exactly
as `01b_` does (`GAS_BUILD_CASE_CHOICES`, `NEW_GAS_DELIVERABILITY_CHOICES`,
`CAPITAL_TREATMENT_CASE_CHOICES`) — the same staleness class TASK.md
asked this module to close, but with no live source available to close
it against. Adding a registry function for these three would be a
`src/scenarios.py` edit, outside this task's whitelist; flagged here and
in the closing list rather than worked around silently.

### Verification

`python3 -m py_compile src/cli_arms.py` — passes. Functional check (not
a script/solver run — pure argparse/dict logic, read-only per TASK.md
§0.5): built a parser with `add_standard_arms(parser)`, parsed all 18
flags, confirmed `standard_arms_kwargs()` returns exactly 18 keys with
`peak_case` correctly bridged to `peak_multiple_case` and
`reserve_margin` absent; confirmed `add_standard_arms(parser,
include=[...])` returns only the requested subset; confirmed a missing
required flag raises `SystemExit` (argparse's own enforcement) and an
unknown `include`/`exclude` key raises `ValueError`.

---

## STEP TWO — SHAPE A: the four three-key runners

**Shape confirmed against the live files, not just the inventory**, for
all four: each already carried the identical 14-flag argparse block
(`--peak-case` through `--concessional-envelope-case`) and an identical
`load_scenario()` call passing all 14 plus the hardcoded literal
`om_case="om_central"` — and, in every one, `conversion_case`,
`hydro_case` and `tariff_case` were the ONLY sentinel-required keys
absent (confirmed by reading each file's full `load_scenario()` call
site and cross-checking against the 17-key list in
`docs/task_shared_fn_repair.md`). No fifth or sixth omission found in
any of the four — Shape A holds.

**Judgement on `include=` vs literal `add_argument()` calls (TASK.md
§3, "your judgement, stated either way"):** used
`add_standard_arms(parser, include=["conversion_case", "hydro_case",
"tariff_case", "om_case"])` in all four, bundling `om_case`'s promotion
into the same call rather than a separate one. Cleaner than four more
hand-copied `add_argument()` blocks, and it means these four flags'
help text and choices are now sourced from the one place that will be
updated if an arm's registry ever changes — the exact defect this
module exists to prevent from recurring. The other 14 pre-existing
flags in each of these four files are UNCHANGED (still their own
hand-copied `add_argument()` calls, not migrated onto
`add_standard_arms()`); migrating those is a larger, separate edit
outside "add the three flags" and is noted in the closing list.

### `04_run_fin2_tariff_sweep.py`

Flags added: `add_standard_arms(parser, include=["conversion_case",
"hydro_case", "tariff_case", "om_case"])`, after the existing
`--concessional-envelope-case` argument.

`load_scenario()` call (single call site, in `main()`) — old tail:
```python
    capital_treatment_case=args.capital_treatment_case,
    concessional_envelope_case=args.concessional_envelope_case,
    om_case="om_central",
)
```
new tail:
```python
    capital_treatment_case=args.capital_treatment_case,
    concessional_envelope_case=args.concessional_envelope_case,
    conversion_case=args.conversion_case,
    hydro_case=args.hydro_case,
    tariff_case=args.tariff_case,
    om_case=args.om_case,
)
```

**`tariff_case` here is base-value-only, not the load-bearing arm.**
This script's own tariff sweep is performed by
`run_fin2_blended_bankability_sweep()` (`src/optimize_experiments.py`,
NOT one of Part 3's six repaired functions — it takes a pre-built
`base_scenario` dict and overwrites `solar_service_tariff_usd_per_twh`
directly per grid point; confirmed by reading its body, no internal
`load_scenario()` call). So `--tariff-case` supplies the scenario's
initial tariff value, immediately superseded once the sweep starts.
Recorded as a real, minor finding (not fixed — no `load_scenario()`
defect exists inside that function to fix): the flag is required but
functionally inert for this particular script's purpose, distinct from
the "inert because no EaaS channel" reason documented elsewhere.

**Caller must now additionally pass:** `--conversion-case`,
`--hydro-case`, `--tariff-case`, `--om-case`.

### `24_run_leakage_arm.py`

Flags added: same `include=[...]` call, after
`--concessional-envelope-case`.

`solve_one()` signature: 4 new trailing parameters —
`conversion_case, hydro_case, tariff_case, om_case` — inserted after
`concessional_envelope_case`.

`load_scenario()` call — old tail: `om_case="om_central",` — new tail:
`conversion_case=conversion_case, hydro_case=hydro_case,
tariff_case=tariff_case, om_case=om_case,`.

`base` dict (the row echoed to `leakage_arm__{tag}.csv`) — the four new
values are also recorded there (`conversion_case=conversion_case,
hydro_case=hydro_case, tariff_case=tariff_case, om_case=om_case`),
matching how every other already-required arm in this script is
already echoed into that same dict for output traceability — not
requested explicitly, but the same pattern the file already applies
uniformly to its other 14 arms, so leaving these four out would be an
inconsistency in the output schema, not a smaller diff.

`main()`'s call site to `solve_one()` updated to pass
`args.conversion_case, args.hydro_case, args.tariff_case, args.om_case`
positionally, matching the new signature.

**`tariff_case` is INERT here** (documented inline): this is a
`financing_regime="public"` runner with no EaaS channel, so
`m.eaas_disable` fixes `solar_eaas_add` to 0 regardless of the tariff
value — the same inertness `01b_`'s own docstring records for
`tariff_case` under `financing_regime='traditional'`.

**Caller must now additionally pass:** `--conversion-case`,
`--hydro-case`, `--tariff-case`, `--om-case`.

### `25_run_obligation_trajectory.py`

Identical shape and identical edit pattern to `24_`: `solve_one()`
gains the same 4 trailing parameters, the `load_scenario()` call and
the `base` dict both gain the same 4 keyword arguments, and `main()`'s
call site is updated to match. `tariff_case` is inert here for the
same reason as `24_` (public-only, no EaaS channel).

**Caller must now additionally pass:** `--conversion-case`,
`--hydro-case`, `--tariff-case`, `--om-case`.

### `99c_run_genset_arm_sensitivity.py`

Identical edit pattern to `24_`/`25_`, with one difference worth
stating explicitly: **`tariff_case` is NOT inert for this script.**
Unlike `24_`/`25_`, `FINANCING_ARMS` here includes an
`"eaas"` entry (`financing_regime="eaas"`), under which
`m.eaas_disable` does not fix `solar_eaas_add` to 0 — so the EaaS half
of this script's sweep genuinely reads the tariff arm now, not just
carries it for schema consistency. This is the sharper case of the
same silent-gap defect: before this edit, `99c_`'s EaaS rows were
solving with `tariff_case` entirely absent from `load_scenario()`,
which would have raised a `TypeError` immediately (the sentinel has no
default) rather than silently using a wrong value — so this was a
hard failure blocking the whole script from running at all, not a
silent-wrong-answer risk. Confirmed by reading `load_scenario()`'s
signature: `tariff_case=_TARIFF_REQUIRED` has no default, so any prior
invocation of this script (with the pre-existing 14-flag block only)
must already have raised before ever reaching a solve.

**Caller must now additionally pass:** `--conversion-case`,
`--hydro-case`, `--tariff-case`, `--om-case`.

### Verification, all four

`python3 -m py_compile` on each of the four files individually —
passes. `grep -n 'om_central"'` across all four — zero remaining
matches (the literal is fully retired from each). Confirmed
`FINANCING_ARMS`, `NDC_SCENARIO`, `EPS`/`EPS_LEVELS`, `CAPITAL_CASE`,
`GENSET_ARMS`, `TRAJECTORIES`, `GRID_SERVICE_MODE` and every other
pre-existing module constant in these four files are byte-for-byte
unchanged (diff review, not just grep) — none of TASK.md §0/§5's
DO-NOT-TOUCH constants were touched.

---

## STEP THREE — SHAPE B: the five full-surface runners

**Correction to TASK.md, found reading the live files, not just the
inventory:** TASK.md §4 says "05_ has no argparse at all" as if this
were the one exception among the five. It is not — **none of the five
(`05_`, `15_`, `18_`, `19_`, `20_`) has ANY argparse**: no `import
argparse`, no `parse_args()`, `main()` took no arguments in every one
of them (confirmed by grepping all five for `argparse` and `def main(`
before making any edit — zero matches for the former across all five).
Built a fresh `parse_args()` for each of the five, not just `05_`, per
the actual state of the files.

**Second correction:** TASK.md §4 says "Two of these delegate to the
now-repaired shared functions... 15_ does not... check each against
the specification in `docs/task_shared_fn_repair.md` Part 3." Grepped
all five files for all six repaired function names
(`run_rel3_financing_frontier`, `run_dem1_demand_sensitivity`,
`run_dem2_growth_gas_matrix`, `run_rel2_marginal_cost_curves`,
`run_rel1_feasibility_matrix`, `run_gas_regime_ndc_matrix`) — **zero
matches in any of the five.** None of `05_`/`15_`/`18_`/`19_`/`20_`
delegates to any of Part 3's six functions; all five call
`load_scenario()` and `build_model()` directly. Whatever two runners
TASK.md had in mind, they are not among these five as the files
actually stand today — reported rather than silently assumed away or
forced to match.

**Design choice, applied uniformly to all five:** each existing
`load_scenario()` call already named several arms explicitly
(`demand_level_case`, `demand_case`, `capital_case`, `solar_build_case`,
`land_case`, `carbon_case`, `start_year`, `end_year` — all either fixed
research-design constants or runner-owned sweep variables, none of
them standard arms) plus, in every one, the two flagged literals
`gas_deliverability_case="baseline"` and `om_case="om_central"` (`15_`
additionally pinned `tariff_case=CANONICAL_TARIFF_CASE`). Rather than
add 16-18 more explicit `key=args.key` lines per call site, each
call now ends `**standard_kwargs` (`standard_kwargs =
standard_arms_kwargs(args)`, computed once in `main()`) — TASK.md
§2's literal instruction ("`add_standard_arms(parser)` for the full
set, then `standard_arms_kwargs(args)` into the `load_scenario()`
call"). The two (three, for `15_`) flagged literals are removed
outright, not kept as now-redundant duplicate keywords (Python raises
`TypeError: got multiple values` on a duplicate keyword, so keeping
them was not an option even had it been desirable).

### `05_run_fin3_capital_sweep.py`

Built `parse_args()`: `argparse.ArgumentParser(description=...)`,
`add_standard_arms(parser)` for the full 18-flag set, `parser.
parse_args()`. Wired `args = parse_args()` and `standard_kwargs =
standard_arms_kwargs(args)` at the top of `main()`.

**Literal replaced:** `gas_deliverability_case="baseline"` and
`om_case="om_central"` — both removed from the explicit call, now
supplied via `standard_kwargs`. This changes a default invocation from
"silently `baseline` gas deliverability, silently `om_central`" to
"both must be stated on the command line."

`load_scenario()` call — old:
```python
scenario = load_scenario(
    demand_level_case="served",
    demand_case="organic_central",
    gas_deliverability_case="baseline",
    capital_case=capital_case,
    carbon_case="no_policy",   # no carbon price — cap only
    start_year=2025,
    end_year=MODEL_END_YEAR,
    om_case="om_central",
)
```
new:
```python
scenario = load_scenario(
    demand_level_case="served",
    demand_case="organic_central",
    capital_case=capital_case,
    carbon_case="no_policy",   # no carbon price — cap only
    start_year=2025,
    end_year=MODEL_END_YEAR,
    **standard_kwargs,
)
```

`CAPITAL_SWEEP_ORDER`, `NDC_CAP_SCENARIO`, `DOMESTIC_SHARE`,
`COLLECTION_RATE_SCENARIOS` — this runner's own subject —
UNCHANGED.

**Caller must now additionally pass all 18 standard-arm flags**
(the full set: `--peak-case` through `--om-case`; see the STEP ONE
flag list above).

### `18_run_pareto_frontier.py`

Same `parse_args()` pattern (description names MIX-2). Literals
replaced: `gas_deliverability_case="baseline"`, `om_case="om_central"`.

`load_scenario()` call — old tail: `gas_deliverability_case="baseline",
... om_case="om_central",` — new: both removed, call ends
`**standard_kwargs,`. `capital_case=arm["capital_case"]` (from
`FINANCING_ARMS`) UNCHANGED. `N_POINTS`, `FINANCING_ARMS`,
`BASELINE_DIAG_PATH` UNCHANGED.

**Caller must now additionally pass all 18 standard-arm flags.**

### `19_run_carbon_price_sweep.py`

Identical pattern (description names MIX-3). Same two literals
replaced the same way. `CARBON_PRICES`, `FINANCING_ARMS` UNCHANGED —
including the post-`load_scenario()` mutation of
`scenario["carbon_active"]`/`scenario["carbon_price"]` per sweep point
(the runner's whole subject), untouched.

**Caller must now additionally pass all 18 standard-arm flags.**

### `20_run_cf1_solar_yield_sensitivity.py`

Identical pattern (description names CF-1). Literal replaced:
`gas_deliverability_case="baseline"` and `om_case="om_central"`.
`capital_case="unconstrained"` (matching `01b_`'s own convention that
`capital_case` is never itself a CLI arm, and this script's own stated
design of holding everything but `solar_cf` at the `01_run_baseline.py`
configuration) UNCHANGED. `CF_SWEEP` UNCHANGED.

**Caller must now additionally pass all 18 standard-arm flags.**

### `15_run_pol1_ndc_comparison.py`

Built `parse_args()` (description names POL-1). Three items replaced,
not two: `gas_deliverability_case="baseline"`, `tariff_case=
CANONICAL_TARIFF_CASE`, and `om_case="om_central"`.

**`CANONICAL_TARIFF_CASE` / `CANONICAL_TARIFF` module constants
removed outright**, not left dead. Their sole purpose was supplying
the `tariff_case` literal this edit retires; the docstring comment
directly above them ("this runner is non-functional independent of
this task... A proper required `--tariff-case` CLI flag is deferred to
whenever this runner is revived") described exactly the gap this task
closes, so keeping the now-unreachable constants and the comment
describing a still-open gap would have been stale documentation, not
attribution — unlike `docs/limitations_register.md` L27's Status line
(Part 3's report), which was left exactly as drafted on explicit
instruction; no such instruction exists here, and TASK.md §4 itself
gives the "replace the literal" instruction that makes these constants
dead. The now-unused `tariff_scenarios` import (used only to compute
the removed `CANONICAL_TARIFF`) was removed with it; the also-unused
`asset_lifetime_sweep` import in the same statement is a pre-existing,
unrelated dead import and was left alone (not something this edit
caused or was asked to clean up).

`run_one()` gained a fourth parameter, `standard_kwargs`, threaded
from `main()` (`run_one(ndc_cfg, fin_cfg, econ, standard_kwargs)`).

`load_scenario()` call — old:
```python
scenario = load_scenario(
    demand_level_case="served",
    demand_case="organic_central",
    gas_deliverability_case="baseline",
    capital_case=capital,
    solar_build_case="deployment_unconstrained",
    carbon_case="no_policy",
    start_year=2025,
    end_year=MODEL_END_YEAR,
    tariff_case=CANONICAL_TARIFF_CASE,
    om_case="om_central",
)
```
new:
```python
scenario = load_scenario(
    demand_level_case="served",
    demand_case="organic_central",
    capital_case=capital,
    solar_build_case="deployment_unconstrained",
    carbon_case="no_policy",
    start_year=2025,
    end_year=MODEL_END_YEAR,
    **standard_kwargs,
)
```
`NDC_CONFIGS`, `FINANCING_CONFIGS` — this runner's own subject —
UNCHANGED.

**Caller must now additionally pass all 18 standard-arm flags.**

### Verification, all five

`python3 -m py_compile` on each of the five files individually —
passes. Full-repo compile sweep of all nine runners plus
`src/cli_arms.py` together — nine `OK`, one module `OK`. `grep -n
'om_central"\|gas_deliverability_case="baseline"'` across all nine
files — every remaining match is inside an explanatory comment this
task added, none in live code. `git diff --stat` — 9 files changed,
200 insertions, 34 deletions, plus 2 new untracked files
(`src/cli_arms.py`, this report); no file outside the whitelist
touched, confirmed by `git status --short`.

---

## NO COMPUTATION WAS CHANGED

Every edit in this task was one of exactly four kinds: (1) creating
`src/cli_arms.py`, whose only effect is to add argparse flags and
return a `dict` of parsed values — no model, no `load_scenario()`
internals, no builder touched; (2) adding a `parse_args()` function and
wiring `args`/`standard_kwargs` into each runner's `main()`; (3) adding
keyword arguments to existing `load_scenario()` calls (either
individually, or via `**standard_kwargs`), each one supplying a value
that a required, no-default sentinel parameter was already demanding —
before this task, every one of these nine runners' `load_scenario()`
calls raised `TypeError` before ever reaching `build_model()` or
`build_model_sliced()`, so **none of the nine ever produced a solved
model, at any point up to and including this edit**; (4) removing two
now-dead module constants in `15_` and rewriting comments. No
`build_model()`/`build_model_sliced()`/`solve_model()` call signature
changed, no Pyomo expression changed, no diagnostic-extraction line
changed, and no output field changed anywhere in this diff — confirmed
by reading every hunk of every file's `git diff` before writing this
report, the same discipline Part 3 applied to `src/optimize_experiments.py`.

**Why the two frozen anchors cannot have moved.** The regression anchor
(`21,811,042,472.821114 / 45,502.9754 MWh`, `om_none`,
`01b_run_baseline_sliced.py`) and the central anchor (`27.954 bn`,
`02_run_ndc_caps.py`) are produced by two runners that are **not among
the nine edited here** — confirmed against TASK.md §0's own list, and
independently by this task never opening either file. A script that
was never edited cannot change a result it alone produces. TASK.md §0
states this independently: "None of these nine is in the frozen sweep
and none produces a reported figure, so this repair cannot move an
anchor" — restated here because the report asks it stated plainly, not
because the claim needed re-deriving. Additionally: none of the nine
runners touched here was even capable of reaching a solve before this
edit (see above) — there is no "before" result from any of them for
this edit to have changed, only a `TypeError` that a caller supplying
the CLI's own well-formed arguments will no longer hit.

---

## Encountered but not asked about

- **TASK.md's own framing did not match the live files, twice, both
  corrected in Step Three above rather than followed as written:** "05_
  has no argparse at all" undersold the actual state — all five Shape B
  runners had none; and "two of these delegate to the now-repaired
  shared functions" — none of the five do, confirmed by grep for all
  six function names across all five files.
- **Three arms (`gas_build_case`, `new_gas_deliverability`,
  `capital_treatment_case`) have no live `*_scenarios()` registry in
  `src/scenarios.py`** to source `src/cli_arms.py`'s `choices=` from —
  hardcoded verbatim from `01b_`'s own literals instead, flagged in
  Step One rather than silently treated as satisfying TASK.md §2's
  "never hardcoded lists" instruction. Adding the missing registry
  functions is a `src/scenarios.py` edit, outside this task's
  whitelist.
- **Output filenames in `04_`, `24_`, `25_`, `99c_` do not encode the
  three newly-added arms or `om_case`** (their existing `tag`/`output_tag`
  construction was left untouched, per "leave every existing hardcoded
  value alone" — but it means two runs of the same script differing
  only in `--conversion-case`/`--hydro-case`/`--tariff-case`/`--om-case`
  will silently overwrite each other's result files). Not fixed: naming
  scheme changes were not part of "thread arguments through," and `05_`/
  `15_`/`18_`/`19_`/`20_` have the identical pre-existing gap at a larger
  scale (their output filenames carry no arm-encoding at all, for any
  arm, old or new) — a single, pre-existing, project-wide output-naming
  issue, not something introduced or worsened by this task.
- **`04_run_fin2_tariff_sweep.py`'s new `--tariff-case` flag is
  required but functionally inert for this script's own purpose** — its
  tariff sweep is performed by overwriting `solar_service_tariff_usd_per_twh`
  directly inside `run_fin2_blended_bankability_sweep()`
  (`src/optimize_experiments.py`, confirmed not one of Part 3's six
  repaired functions and confirmed to call no `load_scenario()`
  internally), so the flag only supplies scenario's initial,
  immediately-superseded tariff value.
- **`src/optimize_experiments.py:run_str1_storage_parameter_sweep` and
  `run_bottleneck_sensitivity`** still carry the same missing-keys
  defect — restated from Part 3's closing list, not rediscovered here;
  neither is called by any of this task's nine runners and neither is
  on this task's edit whitelist.
- **`reliability_mode="total"`** — grepped the nine runners and found no
  occurrence at all (it lives only in `run_rel1_feasibility_matrix`,
  Part 3's territory, not this task's). Confirmed absent rather than
  assumed absent.
- `docs/runner_inventory.md`, from a prior task, remains untracked
  (`??`) alongside this report — not touched this turn, not part of
  this task's whitelist.

**Then STOP.** No commit, no staging (`git status --short` above shows
only the nine modified runners plus two new untracked files). Zeus
reviews `git diff`, commits, and confirms both anchors reproduce.
