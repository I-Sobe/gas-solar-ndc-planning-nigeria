# TASK — three wrong-output defects: cost decomposition, baseline
# validation, smoke test

Branch `phase-fin2-d17`. Reported incrementally per TASK.md §0.

---

## PART 1 — `cost_decomposition` omits O&M

### The defect, as found

Function: `extract_planning_diagnostics(m, scenario, econ=None,
solar_capex_series=None)`, `src/optimize_experiments.py:2087`. This is
the ONE function this task's whitelist permits editing in this file.

**Before this edit**, the cost-reading block
(`src/optimize_experiments.py:2598-2658` prior to this edit) read:

```python
capital_treatment_case = scenario["capital_treatment_case"]
cost_gas_opex     = float(pyo.value(m.cost_gas_opex_npv))
cost_voll_penalty = float(pyo.value(m.cost_unserved_voll_npv))
cost_selfgen      = float(pyo.value(m.cost_selfgen_npv))
cost_carbon       = float(pyo.value(m.cost_carbon_npv))
cost_salvage      = float(pyo.value(m.salvage_npv)) if hasattr(m, "salvage_npv") else 0.0

if capital_treatment_case == "pre_f1_overnight":
    cost_gas_capital       = float(pyo.value(m.cost_gas_capex_npv))
    cost_pub_solar_capital = float(pyo.value(m.cost_public_solar_capex_npv))
    cost_eaas_solar_capital = float(pyo.value(m.cost_eaas_solar_capex_npv))
    cost_storage_capital   = float(pyo.value(m.cost_storage_capex_npv))
    _capital_components = {
        "gas_capex_npv":          round(cost_gas_capital, 0),
        "public_solar_capex_npv": round(cost_pub_solar_capital, 0),
        "eaas_solar_capex_npv":   round(cost_eaas_solar_capital, 0),
        "storage_capex_npv":      round(cost_storage_capital, 0),
    }
else:
    cost_gas_capital       = float(pyo.value(m.cost_gas_capital_charge_npv))
    cost_pub_solar_capital = float(pyo.value(m.cost_public_solar_capital_charge_npv))
    cost_eaas_solar_capital = float(pyo.value(m.cost_eaas_solar_capital_charge_npv))
    cost_storage_capital   = float(pyo.value(m.cost_storage_capital_charge_npv))
    _capital_components = {
        "gas_capital_charge_npv":          round(cost_gas_capital, 0),
        "public_solar_capital_charge_npv": round(cost_pub_solar_capital, 0),
        "eaas_solar_capital_charge_npv":   round(cost_eaas_solar_capital, 0),
        "storage_capital_charge_npv":      round(cost_storage_capital, 0),
        "capital_charge_total_npv":        round(float(pyo.value(m.capital_charge_total_npv)), 0),
    }

cost_real_expenditure = (
    cost_gas_opex + cost_gas_capital + cost_pub_solar_capital
    + cost_eaas_solar_capital + cost_storage_capital + cost_carbon
)
cost_total = (cost_real_expenditure + cost_selfgen
              + cost_voll_penalty - cost_salvage)

cost_decomposition = {
    "real_expenditure_npv":        round(cost_real_expenditure, 0),
    "voll_penalty_npv":            round(cost_voll_penalty, 0),
    "voll_penalty_share":          round(cost_voll_penalty / cost_total, 4) if cost_total > 0 else 0,
    "selfgen_npv":                 round(cost_selfgen, 0),
    "selfgen_share":               round(cost_selfgen / cost_total, 4) if cost_total > 0 else 0,
    "salvage_npv":                 round(cost_salvage, 0),
    "cost_total_npv":              round(cost_total, 0),
    "components": {
        "gas_opex_npv":            round(cost_gas_opex, 0),
        **_capital_components,
        "selfgen_npv":             round(cost_selfgen, 0),
        "voll_penalty_npv":        round(cost_voll_penalty, 0),
        "carbon_cost_npv":         round(cost_carbon, 0),
        "salvage_npv":             round(cost_salvage, 0),
    },
}
```

**No O&M term anywhere, in either `capital_treatment_case` branch** —
confirmed by reading the whole block; `m.cost_om_npv` (or any O&M
reference) does not appear once. The objective includes O&M (+6.602 bn
at central, +30.9%, `docs/task_om_implementation.md`); this
decomposition did not.

**Consequence, confirmed by reading `16_plot_cost_decomposition.py`'s
own reconciliation guard** (not run — reading only, per the
whitelist): its check compares `real + selfgen + voll − salvage`
against `npv_total_cost_usd`. Since `npv_total_cost_usd` (=
`m.system_cost_npv`) includes O&M and the four terms on the left did
not, the guard will read a residual of exactly the O&M cost on every
`om_central`/`om_low`/`om_high` result (i.e. every result computed
since `task_om_implementation` shipped) and flag MISMATCH. **The guard
was working correctly; the decomposition it was checking was wrong.**

### Finding the O&M Expression — named exactly, not guessed

`docs/task_om_implementation.md` records a diagnostics Expression
added next to `m.cost_carbon_npv`. Grepped both builders directly:

```
src/optimize_model.py:1688:   m.cost_carbon_npv = pyo.Expression(expr=carbon_cost_npv)
src/optimize_model.py:1689:   m.cost_om_npv     = pyo.Expression(expr=om_cost_npv)
src/optimize_model_sliced.py:1924:  m.cost_om_npv = pyo.Expression(expr=om_cost_npv)
src/optimize_model_sliced.py:1937:  m.cost_carbon_npv = pyo.Expression(expr=carbon_cost_npv)
```

`m.cost_om_npv` — confirmed the exact name in both builders, sitting
immediately beside `m.cost_carbon_npv` in both, exactly as
`task_om_implementation.md` records.

**Confirmed unconditional in both branches of both builders** — this
matters for how it should be read. In `optimize_model.py`:
`om_cost_npv` enters `system_cost_npv` at line 1647
(`pre_f1_overnight` branch) AND line 1656 (`f1_tranche` branch); `m.
cost_om_npv` itself is built at line 1689, outside either branch's
`if`. Identical structure in `optimize_model_sliced.py` (lines
1910/1919/1924). So `m.cost_om_npv` can be read the same way
`cost_carbon` already is — no `capital_treatment_case` branch, no
`hasattr` guard (unlike `cost_salvage`, which genuinely is absent
under `f1_tranche` — `m.salvage_npv` is built only inside the
`pre_f1_overnight` block in both builders, confirmed by direct
reading: `optimize_model.py:1175`, `optimize_model_sliced.py:1434`).

### The edit

Added, immediately after `cost_carbon`:
```python
cost_om           = float(pyo.value(m.cost_om_npv))
```

**Decision: O&M is its own top-level line, not folded into
`cost_real_expenditure`.** Argued both ways, per TASK.md §3's request:

- *For folding it in:* O&M is a real resource cost like fuel — paid
  out of the same operating budget as gas opex, with the same
  "this is money actually spent, not a penalty" character as the rest
  of `cost_real_expenditure`. On pure cost-accounting grounds, lumping
  it in is defensible.
- *Against, and the position taken:* `cost_decomposition` does not
  exist for accounting purposes alone — it feeds a **stacked chart**
  (`16_plot_cost_decomposition.py`) that a reader interprets visually.
  A term that is 30.9% of the central objective, folded silently into
  an existing category, changes what that category means without
  changing its label — a reader comparing this decomposition to the
  pre-O&M literature, or to a run under a different `om_case`, has no
  visual signal that the "real expenditure" bar grew for a reason
  unrelated to fuel, capital, or carbon. Salvage and carbon — both
  much smaller terms — already get their own line for exactly this
  reason. O&M, being the single largest correction in the project, is
  the strongest possible case for the same treatment, not a weaker
  one.

Given its own top-level `om_npv` field, plus `om_share` (mirroring the
existing `voll_penalty_share`/`selfgen_share` convention — no such
convention exists for `salvage_npv` since it is a credit, not a
positive cost share), and `om_cost_npv` inside `components` (named to
mirror `carbon_cost_npv`'s existing swap of `cost_X_npv` →
`X_cost_npv`, keeping the two Expression-adjacent entries visually
paired the way their source Expressions are).

`cost_total` updated from `real + selfgen + voll − salvage` to `real +
selfgen + voll − salvage + om`.

### Reconciliation — closes for the requested defect, but NOT
### completely, and I am reporting the remainder rather than patching it

**The identity requested:** `real + selfgen + voll − salvage + om ==
objective`. **Every term the O&M fix was asked to add is now present
and correctly wired.** But checking "every objective term now appears
exactly once on the left" (TASK.md §4's own instruction) surfaced a
**second, separate, pre-existing gap**, unrelated to O&M, that I am
reporting rather than silently folding into this edit:

**`m.cost_replacement_capex_npv`** (`pre_f1_overnight` branch,
`optimize_model.py:1223`) and its `f1_tranche` counterpart
**`m.cost_rebuild_capital_charge_npv`** (`optimize_model.py:1509`,
entering `m.capital_charge_total_npv` at `optimize_model.py:1533-1539`
alongside the four channel capital charges) are **real, charged
objective terms that `extract_planning_diagnostics` never reads** —
confirmed by grepping the whole file for `"rebuild"` and
`"replacement"`: zero matches, in either the pre-edit or post-edit
version.

**This is not a theoretical edge case.** `docs/limitations_register.md`
L18 already documents (found while checking this, not rediscovered
independently) that "for every lifetime value this project actually
supplies anywhere (`asset_lifetimes()`, `asset_lifetime_sweep()`,
and `22_salvage_sensitivity.py`'s arms)... no vintage was found where
[a rebuild is not generated]" — i.e. **a nonzero rebuild/replacement
charge is present in essentially every run this project can produce**,
in both capital-treatment regimes, including the regime used for the
frozen central figure (`f1_tranche`, confirmed against
`docs/findings_frozen.md:26`) and the regime used for the frozen
regression anchor (`pre_f1_overnight`, confirmed against
`docs/findings_frozen.md:37`).

**Consequence for the identity:**
- Under `pre_f1_overnight`: `system_cost_npv` includes
  `m.cost_replacement_capex_npv` as an additive term (`optimize_model.
  py:1642`); `cost_real_expenditure` does not read it at all. `cost_
  total` (even after this fix) is short of the true objective by
  exactly this amount.
- Under `f1_tranche`: `system_cost_npv` includes
  `m.capital_charge_total_npv`, which itself equals the four channel
  capital charges **plus** `m.cost_rebuild_capital_charge_npv`
  (`optimize_model.py:1533-1539`). `cost_real_expenditure`'s
  `cost_pub_solar_capital + cost_eaas_solar_capital +
  cost_gas_capital + cost_storage_capital` sums only the four channel
  charges — **this can be checked directly inside the existing
  `components` dict itself**, without running anything: `components`
  already carries `capital_charge_total_npv` (the true, full total)
  alongside the four separate `*_capital_charge_npv` entries: summing
  the four will not equal `capital_charge_total_npv` whenever the
  rebuild charge is nonzero — which L18 says is always, in this
  project. This built-in cross-check was already sitting in the output
  before this edit; it just required reading the numbers rather than
  assuming the labelled "total" and the "sum of parts" agree.

**I have not added either term to the decomposition.** TASK.md §4 is
explicit: *"If a term is still missing or double-counted, report it —
do not patch the identity to make it balance."* This finding was
surfaced by exactly the discipline that instruction asks for, and I am
following it literally: reported, not patched. Fixing it is a natural
follow-on (see "For the limitations register" below) — likely with the
same "own line vs folded in" design question this task just resolved
for O&M, since replacement/rebuild capex is arguably even more
naturally read as capital expenditure than as a separate category, but
that argument belongs to whoever does that follow-on, not asserted
here as settled.

I did **not** verify the replacement/rebuild term's magnitude
numerically for either frozen anchor — doing so would require running
the solver, forbidden by this task's whitelist. The finding above is
structural (read from the code and cross-checked against the existing
`components` dict's own redundant total), not empirical.

### Why this cannot move the objective

`extract_planning_diagnostics` is called **after** `solve_model()`
returns (confirmed by reading every call site: it takes an already-
solved `m` and only calls `pyo.value(...)` on already-built
Expressions — it contains no `pyo.Var`, no constraint, no `+=` to any
`m.system_cost_npv`-contributing term, and assigns to no Pyomo
component). It is a **post-solve diagnostic**: every line in this
function reads a value out of an LP that has already been solved to
optimality: `pyo.value()` is a pure reader. Adding `cost_om =
float(pyo.value(m.cost_om_npv))` reads an Expression that already
existed, unconditionally, in the model that produced both frozen
anchors — it does not create it, does not change what it evaluates
to, and does not touch `m.system_cost_npv` itself. The anchors
(`21,811,042,472.821114 / 45,502.9754 MWh`, `27.954 bn / 65,613.0
MWh`) are properties of the solved LP, unaffected by what a
diagnostic function reads out of it afterward.

`python3 -m py_compile src/optimize_experiments.py` — passes.

---

## PART 2 — `99_validate_baseline.py`

### Every hardcoded observation, as found

| Key | Value (before) | Comment (before) | Verdict |
|---|---|---|---|
| `demand_twh` | 23.08 | "NBS Q1 2024 annualized served energy" | **STALE — the specifically retired demand base.** `methodology_demand_horizon.md` §1.1: this is *collected* energy, a revenue quantity, understating system scale by ~40% against gross generation. |
| `gas_generation_twh` | 25.6 | "NERC 2024 grid annual gen (~27-30 TWh) (update with actual)" | **STALE placeholder with its own TODO.** The project has had a precise, sourced figure (25.62 TWh) since the demand-base correction; this file was never updated to use it. |
| `solar_generation_twh` | 0.3 | "rooftop + utility estimate" | **CURRENT — left unchanged.** See note below. |
| `unserved_twh` | 3.0 | "from access gap estimates" | **STALE AND STRUCTURALLY MEANINGLESS.** Predates the genset backstop (2.6). `unserved` is now ~0 by construction under any genset case, so a −100% gap is reported against a comparison that has stopped meaning anything, every single run. |

Module-level docstring (lines 8-11, before this edit) also asserted
`"Gas generation: NERC 2024 annual grid generation (~27-30 TWh)"` and
`"Demand: NBS Q1 2024 annualized = 23.08 TWh"` — both stale in the
same way as the constants below them, and corrected alongside them
(a docstring citing the same superseded figures the code uses is not
a separate defect, but leaving it stale while fixing the code would
have created a second, smaller inconsistency of exactly the kind this
whole task is about).

### `demand_twh`: corrected to 37.09 TWh

Source: `src/scenarios.py`'s module header docstring — `"[SOURCE: NERC
2024 — gross generation 37,093.70 GWh]"`, metering point explicitly
GROSS GENERATION at the busbar, not collected/billed energy (see the
loss-ladder note in that same docstring). Added as a source comment
directly on the constant, plus expanded in the module docstring.

### `gas_generation_twh`: corrected to 25.62 TWh

Same source, same docstring: `"of which gas 25.62 TWh"` — a split of
the identical NERC 2024 gross-generation total used for `demand_twh`,
not an independent figure. The old value's own comment, `"(update
with actual)"`, is retired along with the value it was flagging as
provisional.

### `unserved_twh`: removed, not re-based

TASK.md §3 gave two options and an explicit fallback: re-base on
`selfgen + unserved` (`methodology_genself_backstop.md`'s recorded
reportable quantity — the grid SERVICE shortfall, not raw `unserved`),
or remove the check; **"if re-basing needs an observed figure you
cannot source, remove it — do not invent a comparison value."**

Searched this project's documents for a sourced, current, observed
2024 access-gap or self-generation figure to re-base against. The only
candidate is NIRP's ~27 TWh self-generation estimate, and
`docs/limitations_register.md` L4 records it in terms that rule it out
as a citable comparison value here: *"stipulated, not derived: no
methodology, no SFC assumption, no citation"*, plus an independent
memo attempting to rebuild it (`docs/tier2_reconciliation.md`, D26)
containing an unresolved 10× unit error on one of its three routes.
**No sourceable figure exists to re-base against.** Per the explicit
fallback, the check is **removed**, with the reasoning above recorded
as a comment at its former place in the dict, and the now-unused
`MODEL_2025["unserved_twh"]` entry removed alongside it (kept out
rather than left in as a dead, unread key — the validation loop only
ever iterates `OBSERVED_2024`'s keys).

### `solar_generation_twh`: checked, left unchanged

Not one of the two defects TASK.md named. Checked whether any current
document supersedes 0.3 TWh: none does. Specifically checked, and
rejected, the tempting-looking "existing solar 0.88 TWh" figure that
recurs across several docs (`src/scenarios.py`'s
`solar_min_build_default()` docstring,
`docs/phase_2_11_minbuild_retirement.md`,
`docs/methodology_demand_horizon.md`) — that is a **model input**
(the assumed existing-capacity baseline's notional dispatch output
inside the 2025 backcast arithmetic), not an independently observed
2024 generation statistic, and is a different quantity from what
`OBSERVED_2024` represents. Several of the documents that cite "0.88"
(`docs/task_methodology_repair.md`, line ~195) additionally record
that *specific* backcast arithmetic passage as itself stale (a 2-step-
old gas figure, −4.1% vs the current −3.2% central backcast).
Substituting it here would have swapped one stale-adjacent number for
a different quantity entirely, not corrected a stale one — closer to
inventing a comparison value than sourcing one. Left as the pre-
existing "rooftop + utility estimate," flagged as current rather than
silently left with no comment at all.

### Other observations checked against the frozen state

TASK.md §4 named three figures to check against: gas 25.62 TWh_e
(now the corrected `gas_generation_twh` value, addressed above), hydro
11.47 TWh, and backcast −3.2%. **Hydro is not one of this file's four
`OBSERVED_2024` keys at all** — `99_validate_baseline.py` checks
demand, gas, solar and (formerly) unserved only, never hydro — so
there is no hydro constant in this file to correct or confirm; noted
rather than silently skipped. **"Backcast −3.2%"** is not a constant
in this file either — it is the *name* of the same comparison this
file's own `gap_pct` computation performs for `gas_generation_twh`
(model 2025 gas output vs. observed 25.62 TWh, `docs/findings_frozen.
md:34`); it is not a separate observation to add, and this file
already computes the equivalent `gap_pct` for `gas_generation_twh`
generically. Confirmed, not something requiring a code change.

### Epoch stamp

Added: `"EPOCH: current as of HEAD 2a9226b (task_wrong_output_repair,
Part 2)."` in the module docstring, with a note to re-check the stamp
if this file or the sources it cites move again.

`python3 -m py_compile scripts/99_validate_baseline.py` — passes.

**Why this cannot move a solved value:** this script does not build
or solve a model at all — it reads a pre-existing CSV
(`results/baseline/timeseries.csv`) and a fixed dict of constants, and
writes a comparison table. No `build_model`, no `load_scenario`, no
Pyomo object anywhere in the file.

---

## PART 3 — `smoke_all.sh` and `run_verification_sweep.sh`

### 3a — `smoke_all.sh`: three-outcome logic

**Before:** binary pass/fail via `if ... ; then PASS ; else FAIL`,
which scores an argparse rejection (a healthy runner correctly
refusing to run unconfigured) identically to a real crash.

**After:** exit code captured explicitly (`code=$?`), then classified
three ways:

```bash
if [ $code -eq 0 ]; then
  echo "  PASS        $f"; pass=$((pass+1))
elif [ $code -eq 2 ] && grep -q "error: the following arguments are required" /tmp/smoke.log; then
  echo "  NEEDS-ARGS  $f"; needs_args=$((needs_args+1))
else
  echo "  FAIL        $f  -- $(grep -m1 'Error\|error:' /tmp/smoke.log | cut -c1-80)"
  fail=$((fail+1))
fi
```

`NEEDS-ARGS` requires **both** conditions — exit code 2 **and** the
literal argparse message text on stderr — matching TASK.md §3a's
explicit instruction not to rely on exit code 2 alone (Python/argparse
uses 2 as a generic command-line-misuse code; matching on the message
text is what actually identifies "this was argparse's own required-
argument rejection," not some other exit-code-2 failure). Summary line
now reports all three counts. The script's header comment is rewritten
to state plainly both what it tests (import errors, missing constants,
renamed attributes, stale kwargs — the 2.6-era defect class — plus,
now, an argument-discipline census) and what it does not (correctness:
a PASS says nothing about whether the runner's output is right, and a
NEEDS-ARGS says only that argparse itself is wired, not that the
demanded flags are individually correct).

`bash -n scripts/smoke_all.sh` — passes.

### 3b — `run_verification_sweep.sh`: missing-file guard and RUNNERS audit

**Defect confirmed:** `RUNNERS` named `scripts/99c_run_voll_sensitivity.
py`; the file does not exist (renamed to
`scripts/99c_run_genset_arm_sensitivity.py`, per the 99c_ runner's own
module docstring: *"REPLACES 99c_run_voll_sensitivity.py (correction
plan 2.6, F1a)"*). The script's own missing-file guard (`[ -f "$f" ] ||
MISSING=...`) fires on this and `exit 1`s before running anything.

**Fix:** corrected the one entry to `scripts/99c_run_genset_arm_
sensitivity.py`.

**Full audit of the `RUNNERS` list against files that actually
exist** (`[ -f ... ]` on every one of the 19 entries, individually):
all 18 other entries exist exactly as named. `99c_run_voll_
sensitivity.py` → `99c_run_genset_arm_sensitivity.py` is the **only**
missing-or-renamed entry in the list.

`bash -n scripts/run_verification_sweep.sh` — passes.

**Not attempted, per TASK.md's explicit instruction** ("Do not attempt
to make every runner in that list pass. ... Fixing the abort so the
sweep can report per-runner status is the whole job."): several
`RUNNERS` entries will still fail or reject bare invocation for
reasons unrelated to the missing-file guard — see the closing list
below for the specific, newly-relevant instance this task's own recent
history created.

### Why neither shell-script edit can move a solved value

Both are **test harnesses that invoke other scripts as subprocesses**;
neither contains a Pyomo model, a `load_scenario()` call, or any
numeric computation of its own. `smoke_all.sh`'s edit only changes how
an *already-captured exit code and log* are classified and printed —
the runners it invokes (and what they compute) are untouched.
`run_verification_sweep.sh`'s edit is a one-line filename string
correction in a list of arguments to `python`, executed exactly as
before for every other entry.

---

## For the limitations register

**Requested draft (next free number: L28 — L27 is the current
highest):**

> ## L28 — `cost_decomposition` silently omitted a 6.602 bn objective
> term; found by a consuming script's own guard, not by inspection
>
> **Status:** RESOLVED (task_wrong_output_repair, Part 1) for the O&M
> term named below; a second, related gap remains OPEN (see "Also
> found" below).
>
> **What.** `extract_planning_diagnostics`'s `cost_decomposition`
> (`src/optimize_experiments.py`) built its reconciliation total from
> `real_expenditure + selfgen + voll − salvage`, omitting O&M entirely
> in both `capital_treatment_case` branches, even though `m.cost_om_
> npv` had existed in both builders, unconditionally, since `task_om_
> implementation` shipped — the correction that added +6.602 bn
> (+30.9%) to the central objective, the single largest correction in
> this project.
>
> **How it was found.** Not by inspection or by a targeted audit of
> `extract_planning_diagnostics` — `docs/runner_inventory.md` found it
> as a side effect of documenting `16_plot_cost_decomposition.py`'s own
> reconciliation guard (`total = real + selfgen + voll − salvage`
> checked against `npv_total_cost_usd`), which had been printing
> MISMATCH on every `om_central`/`om_low`/`om_high` result since the
> O&M correction shipped. The guard was correct; the function fills it
> with was wrong. Nothing about the guard's own output — a bare
> MISMATCH flag — named O&M as the cause; that required reading the
> objective's own term list against the decomposition's, which is what
> this task's Part 1 did.
>
> **Why this matters beyond this one bug.** A reconciliation guard that
> checks a derived total against an independent objective value is
> exactly the kind of structural cross-check that catches a silent
> omission a unit test on the decomposition function alone would not
> — a test of `cost_decomposition` written against its OWN (wrong)
> expectations would have passed forever. The guard caught this because
> it compares against a DIFFERENT, independently-computed quantity
> (`m.system_cost_npv`) that the decomposition function does not
> control. **Keep this class of guard, and read what it says even when
> "everything looks fine otherwise"** — the O&M gap produced no
> exception, no wrong-looking number in isolation, and no crash; every
> `om_central` result before this fix was a plausible, structurally
> unremarkable, understated total.
>
> **Also found, while verifying the fix (not the original defect this
> entry is about, but surfaced by the same reconciliation check and
> reported here rather than opened as a second, disconnected entry):**
> `m.cost_replacement_capex_npv` (`pre_f1_overnight`) and `m.cost_
> rebuild_capital_charge_npv` (`f1_tranche`, folded into `m.capital_
> charge_total_npv`) are ALSO real, charged objective terms that
> `extract_planning_diagnostics` never reads, in either branch. Per
> register L18, a nonzero rebuild/replacement charge is present in
> essentially every run this project can produce (both regimes,
> including both frozen anchors' own regimes). This is directly
> verifiable inside the existing `components` dict without running
> anything: under `f1_tranche`, summing the four channel `*_capital_
> charge_npv` entries will not equal the dict's own `capital_charge_
> total_npv` entry whenever the rebuild charge is nonzero. **Not fixed
> in this task** (out of Part 1's scope, and TASK.md's own instruction
> was to report rather than patch a still-open gap) — recorded here so
> it is not lost, and to make explicit that closing the O&M gap did not
> make this decomposition fully reconcile.
>
> **Direction of bias.** The pre-fix decomposition understated
> `real_expenditure`-adjacent reporting by the full O&M term (now
> fixed) and continues to understate it by the unread replacement/
> rebuild capital charge (not yet fixed) — both omissions bias the
> DECOMPOSITION low relative to the true objective; `m.system_cost_
> npv` itself, the actual optimised and reported quantity, was never
> wrong.
>
> **What would resolve the remainder.** Add `cost_replacement_capex_
> npv` / `cost_rebuild_capital_charge_npv` to `cost_decomposition`,
> following the same "own line vs. folded into an existing category"
> design question this task resolved for O&M — arguably resolved even
> more easily in that direction, since replacement/rebuild capex reads
> naturally as capital expenditure rather than a genuinely new
> category. Re-verify the reconciliation identity closes exactly (to
> within floating-point tolerance) after that change, for both
> capital-treatment regimes.

---

## Encountered but not asked about

- **The replacement/rebuild capital-charge gap** (`m.cost_replacement_
  capex_npv` / `m.cost_rebuild_capital_charge_npv`), detailed above and
  in the L28 draft — the most significant thing found this task beyond
  what was asked, and not fixed, per TASK.md's explicit "report, do not
  patch" instruction.
- **`run_verification_sweep.sh`'s `om_args` allowlist is now stale**,
  as a direct consequence of last task's nine-runner repair (`docs/
  task_runner_repair.md`, committed at `ecb7e54`). The script special-
  cases `--om-case om_central` for exactly five runners (`01_`, `01b_`,
  `02_run_eaas`, `02_run_ndc_caps`, `03_run_ndc_eaas`) with a comment
  asserting "every other runner in RUNNERS above takes no extra arg
  here." That was true when written; it is no longer true for `05_`,
  `15_`, `18_`, `19_`, `20_` — all five now require the full 18-flag
  standard-arm surface (`src/cli_arms.py`) and will reject a bare
  invocation just as hard as before, for a much longer list of missing
  flags than just `--om-case`. Not fixed here: TASK.md Part 3b's scope
  is explicitly "correct the name" plus "audit... for missing or
  renamed" files, and separately says "do not attempt to make every
  runner in that list pass" — updating `om_args` to match the current
  argument surface of five already-repaired runners is exactly that
  kind of broader repair, not a missing-file fix.
- **`run_verification_sweep.sh`'s `RUNNERS` list has no entry at all
  for `24_`, `25_`, or `99c_`'s ACTUAL research runners beyond the one
  corrected filename** — i.e. the audit found the list internally
  consistent (every remaining name matches an existing file) but did
  not check whether the list's *coverage* of the nine recently-repaired
  runners is complete; that was not asked (the task scoped the audit to
  missing/renamed files, not list completeness) and is a different
  question from the one Part 3b poses.
- **Part 2's module docstring** now cites `task_wrong_output_repair`
  by name in-line — consistent with how Part 1 and Part 3's comments
  cite it, for a reader who finds this file later without this report
  open beside it.
- No script, solver, or `smoke_all.sh` was run at any point in this
  task, per the whitelist. All verification above is `py_compile`,
  `bash -n`, and direct reading/grepping of source and docs.

---

## PART 4 — close the reconciliation: replacement/rebuild capex

Follow-on to Part 1's own finding. Whitelist this time: `src/
optimize_experiments.py` (`extract_planning_diagnostics` only) and
`docs/limitations_register.md` (the L28 entry only).

### Reading both terms in both builders

**`m.cost_replacement_capex_npv`** — `optimize_model.py:1223`,
`optimize_model_sliced.py:1482`. Built **only** inside the `if
capital_treatment_case == "pre_f1_overnight":` block in both builders
— the same block that builds `m.salvage_npv`. Confirmed by reading the
surrounding structure directly (the block's `else:`, introducing the
`f1_tranche` path, appears at `optimize_model.py:1245`, after both
`m.salvage_npv` and `m.cost_replacement_capex_npv`). **Needs the same
`hasattr` guard `cost_salvage` already uses** — reading it
unconditionally would raise `AttributeError` under `f1_tranche`.

**`m.cost_rebuild_capital_charge_npv`** — `optimize_model.py:1509`,
`optimize_model_sliced.py:1753`. Built **only** inside the `else:`
(`f1_tranche`) branch. Immediately followed by:

```python
# optimize_model.py:1533-1539
m.capital_charge_total_npv = pyo.Expression(
    expr=(
        m.cost_public_solar_capital_charge_npv
        + m.cost_eaas_solar_capital_charge_npv
        + m.cost_gas_capital_charge_npv
        + m.cost_storage_capital_charge_npv
        + m.cost_rebuild_capital_charge_npv
    )
)
```

**`m.capital_charge_total_npv` already equals the four channel charges
plus the rebuild charge, exactly.** It does not need its own `hasattr`
guard beyond the existing branch split (it, like `cost_rebuild_
capital_charge_npv`, exists only under `f1_tranche` — but the
decomposition already branches on `capital_treatment_case`, so this
falls out of the existing structure for free).

**One further, unrequested-but-relevant finding from reading both
builders side by side:** `optimize_model_sliced.py`'s own `capital_
charge_total_npv` (built as a Python-accumulated `_capital_charge_
total`, `optimize_model_sliced.py:1785-1810`) is NOT identical in
shape to the annual builder's — it optionally adds a further `m.cost_
conversion_capital_charge_npv` term when `conversion_case != "conv_
none"` (a Phase-2.8 OCGT→CCGT retrofit charge that does not exist in
`optimize_model.py` at all — confirmed by grepping `optimize_model.py`
for `convert_mw`/`_conv`/`cost_conversion_capital_charge_npv`: zero
matches; the annual builder has no conversion mechanism). This matters
for the design choice below, not as a defect to fix here: confirmed,
by grepping every call site of `extract_planning_diagnostics` (~25 of
them, across `scripts/` and `src/optimize_experiments.py`), that it is
**always** called on a `build_model()` (annual) result, **never** on
`build_model_sliced()` — consistent with `01b_run_baseline_sliced.py`'s
own comment that this function "is annual-only and would crash on the
sliced `[t,(s,p)]` indexing." So the conversion term is structurally
unreachable from this function today; noted because it directly
motivates reading `capital_charge_total_npv` rather than re-summing
channels (below), not because it needs fixing here.

### Which quantity the decomposition should sum: total, not parts

**Chose `m.capital_charge_total_npv` directly**, not "the four
channels plus rebuild." Two reasons:

1. **Correctness by construction, not by re-derivation.** The pre-Part-
   4 code summed the four channel charges by hand — the exact shape of
   defect this repair sequence keeps finding (a quantity re-derived at
   a second site drifts from its source of truth). Reading the
   Expression the objective itself uses removes that risk entirely:
   whatever `capital_charge_total_npv` is defined to include, this
   line matches it, by definition.
2. **It generalises for free.** The sliced builder's own equivalent
   Expression already includes a term (`conversion`) the four-channel
   list does not name. `extract_planning_diagnostics` cannot reach that
   builder today (confirmed above), so this is not a live bug — but
   reading the total rather than the parts means that if this function
   is ever pointed at a different model, or the annual builder ever
   grows a new capital term, this line keeps matching the objective
   without needing to be found and edited a third time.

**Not double-counted:** the fix does NOT add `cost_rebuild_capital_
charge_npv` on top of `capital_charge_total_npv` — it replaces the
four-channel manual sum with the single total, which already contains
the rebuild charge once. `cost_rebuild_capital_charge_npv` is read
separately only to display it as its own named field in `_capital_
components` (see below), not summed again into `cost_real_
expenditure`.

### Own line, or folded in? Applying Part 1's own test

Part 1 argued O&M deserved its own top-level `cost_decomposition` key
because it is 30.9% of the objective and this dict feeds a stacked
chart a reader interprets visually — burying a term that large inside
an existing category would mislead. Applying the same test here:

**Magnitude class, determined structurally (no run performed).** Under
this project's CENTRAL lifetimes (`asset_lifetimes()`: solar 30 /
storage 15 / gas 30 years, against a 30-year model horizon), solar and
gas life exactly equal the horizon length. `_rebuild_years(build_t,
life)` only yields a rebuild when `build_t + life <= T_last`; for
solar/gas, `build_t + 30` always exceeds `T_last = 29` (the horizon's
last index), so **neither ever rebuilds** — stated directly in
`optimize_model.py`'s own comment immediately above `_rebuild_years()`
("under CURRENT lifetimes... only storage actually rebuilds"). Only
**storage** — this project's smallest capital channel by construction
(register L5: storage is structurally dominated/near-zero at annual
resolution; even where it builds, register L21 records its role as
comparatively small) — rebuilds, and only for vintages built before
roughly the horizon's midpoint, each rebuilding at most once.

**Conclusion: fold it in, not its own line.** Replacement/rebuild
capex is capital expenditure by nature (TASK.md's own framing) AND,
unlike O&M, structurally bounded to a small fraction of one already-
small channel under the lifetimes this project actually uses centrally
— it does not clear the bar Part 1 set for "large enough that folding
it in hides something a reader needs to see." It is folded into `cost_
real_expenditure`'s capital total in both branches. Visibility is
preserved a different way: named explicitly inside `_capital_
components` (`"replacement_capex_npv"` under `pre_f1_overnight`,
`"rebuild_capital_charge_npv"` under `f1_tranche`) — a reader
inspecting the breakdown can see the term and its value; it is simply
not large enough to warrant a dedicated top-level share field the way
`om_share`/`voll_penalty_share`/`selfgen_share` are.

**Caveat, stated rather than hidden:** this magnitude-class argument
holds for the CENTRAL lifetimes this project's frozen results use. A
dedicated sensitivity run under `asset_lifetime_sweep()`'s alternative
lifetimes (e.g. a 25-year solar life) could make solar rebuild too,
changing the magnitude class for those specific runs. Not re-litigated
here — the fold-in decision is about the standard case, and the
underlying values are read correctly (and reconcile exactly, see
below) regardless of which lifetime arm produced them.

### The edit — old and new

**Old** (`pre_f1_overnight` branch):
```python
if capital_treatment_case == "pre_f1_overnight":
    cost_gas_capital       = float(pyo.value(m.cost_gas_capex_npv))
    cost_pub_solar_capital = float(pyo.value(m.cost_public_solar_capex_npv))
    cost_eaas_solar_capital = float(pyo.value(m.cost_eaas_solar_capex_npv))
    cost_storage_capital   = float(pyo.value(m.cost_storage_capex_npv))
    _capital_components = {
        "gas_capex_npv":          round(cost_gas_capital, 0),
        "public_solar_capex_npv": round(cost_pub_solar_capital, 0),
        "eaas_solar_capex_npv":   round(cost_eaas_solar_capital, 0),
        "storage_capex_npv":      round(cost_storage_capital, 0),
    }
```

**New:**
```python
if capital_treatment_case == "pre_f1_overnight":
    cost_gas_capital       = float(pyo.value(m.cost_gas_capex_npv))
    cost_pub_solar_capital = float(pyo.value(m.cost_public_solar_capex_npv))
    cost_eaas_solar_capital = float(pyo.value(m.cost_eaas_solar_capex_npv))
    cost_storage_capital   = float(pyo.value(m.cost_storage_capex_npv))
    cost_replacement_capex = (
        float(pyo.value(m.cost_replacement_capex_npv))
        if hasattr(m, "cost_replacement_capex_npv") else 0.0
    )
    cost_capital_total = (
        cost_gas_capital + cost_pub_solar_capital + cost_eaas_solar_capital
        + cost_storage_capital + cost_replacement_capex
    )
    _capital_components = {
        "gas_capex_npv":          round(cost_gas_capital, 0),
        "public_solar_capex_npv": round(cost_pub_solar_capital, 0),
        "eaas_solar_capex_npv":   round(cost_eaas_solar_capital, 0),
        "storage_capex_npv":      round(cost_storage_capital, 0),
        "replacement_capex_npv":  round(cost_replacement_capex, 0),
    }
```

**Old** (`f1_tranche` branch, the `else:`):
```python
else:
    cost_gas_capital       = float(pyo.value(m.cost_gas_capital_charge_npv))
    cost_pub_solar_capital = float(pyo.value(m.cost_public_solar_capital_charge_npv))
    cost_eaas_solar_capital = float(pyo.value(m.cost_eaas_solar_capital_charge_npv))
    cost_storage_capital   = float(pyo.value(m.cost_storage_capital_charge_npv))
    _capital_components = {
        "gas_capital_charge_npv":          round(cost_gas_capital, 0),
        "public_solar_capital_charge_npv": round(cost_pub_solar_capital, 0),
        "eaas_solar_capital_charge_npv":   round(cost_eaas_solar_capital, 0),
        "storage_capital_charge_npv":      round(cost_storage_capital, 0),
        "capital_charge_total_npv":        round(float(pyo.value(m.capital_charge_total_npv)), 0),
    }
```

**New:**
```python
else:
    cost_gas_capital       = float(pyo.value(m.cost_gas_capital_charge_npv))
    cost_pub_solar_capital = float(pyo.value(m.cost_public_solar_capital_charge_npv))
    cost_eaas_solar_capital = float(pyo.value(m.cost_eaas_solar_capital_charge_npv))
    cost_storage_capital   = float(pyo.value(m.cost_storage_capital_charge_npv))
    cost_rebuild_capital = float(pyo.value(m.cost_rebuild_capital_charge_npv))
    cost_capital_total = float(pyo.value(m.capital_charge_total_npv))
    _capital_components = {
        "gas_capital_charge_npv":          round(cost_gas_capital, 0),
        "public_solar_capital_charge_npv": round(cost_pub_solar_capital, 0),
        "eaas_solar_capital_charge_npv":   round(cost_eaas_solar_capital, 0),
        "storage_capital_charge_npv":      round(cost_storage_capital, 0),
        "rebuild_capital_charge_npv":      round(cost_rebuild_capital, 0),
        "capital_charge_total_npv":        round(cost_capital_total, 0),
    }
```

**Old** (both branches converge here):
```python
cost_real_expenditure = (
    cost_gas_opex + cost_gas_capital + cost_pub_solar_capital
    + cost_eaas_solar_capital + cost_storage_capital + cost_carbon
)
```

**New:**
```python
cost_real_expenditure = (
    cost_gas_opex + cost_capital_total + cost_carbon
)
```

`cost_total`, `cost_decomposition`'s top-level keys, and `om_npv`/
`om_share` are **unchanged from Part 1** — no new top-level key was
added (per the "fold in" decision above).

### The identity — verified, closes exactly, both regimes

**`pre_f1_overnight`:** `system_cost_npv = gas_opex_npv +
public_solar_capex_npv + eaas_solar_capex_npv + gas_capex_npv +
storage_capex_npv + m.cost_replacement_capex_npv + selfgen_npv +
unserved_npv + carbon_cost_npv − m.salvage_npv + om_cost_npv`
(`optimize_model.py:1636-1648`).

`cost_total` (this function, post-fix) = `cost_real_expenditure +
cost_selfgen + cost_voll_penalty − cost_salvage + cost_om` = `(gas_opex
+ [gas_capex + pub_solar_capex + eaas_solar_capex + storage_capex +
replacement_capex] + carbon) + selfgen + voll − salvage + om`.

Matching term for term against `system_cost_npv`: `gas_opex` ✓,
`pub_solar_capex` ✓, `eaas_solar_capex` ✓, `gas_capex` ✓,
`storage_capex` ✓, `replacement_capex_npv` ✓ (now included),
`selfgen_npv` ✓ (`cost_selfgen` reads the same Expression), `unserved_
npv` ✓ (`cost_voll_penalty` reads `m.cost_unserved_voll_npv`, defined
as `expr=unserved_npv`), `carbon_cost_npv` ✓, `−salvage_npv` ✓,
`+om_cost_npv` ✓. **Every term on both sides matches, exactly once,
with no residual.**

**`f1_tranche`:** `system_cost_npv = gas_opex_npv +
m.capital_charge_total_npv + selfgen_npv + unserved_npv +
carbon_cost_npv + om_cost_npv` (`optimize_model.py:1649-1657`; no
salvage term at all in this regime).

`cost_total` (post-fix) = `(gas_opex + capital_charge_total_npv +
carbon) + selfgen + voll − 0 (salvage guarded to 0.0 under this
regime) + om`. Matching term for term: `gas_opex` ✓, `capital_charge_
total_npv` ✓ (read directly — already includes the four channels and
the rebuild charge, confirmed against `optimize_model.py:1533-1539`
above), `selfgen_npv` ✓, `unserved_npv` ✓, `carbon_cost_npv` ✓,
`om_cost_npv` ✓. **Every term on both sides matches, exactly once,
with no residual.**

**No residual to report in either regime** — the identity closes
exactly (to floating-point rounding in the `round(..., 0)` calls,
which apply identically to both sides of every comparison above and do
not introduce a structural mismatch).

`python3 -m py_compile src/optimize_experiments.py` — passes.

### L28 updated

`docs/limitations_register.md`'s L28 entry (added directly in this
Part — it had been drafted in this report's own Part 1 section but,
under the previous task's whitelist, never actually appended to the
register file itself; confirmed by grepping the live file for `^## L2`
before this edit — only L26 and L27 existed there) now records BOTH
the O&M and the replacement/rebuild findings, with **Status: RESOLVED**
for both, and an updated closing paragraph stating the identity closes
exactly in both regimes rather than flagging the rebuild/replacement
gap as still open. `grep -n "^## L" docs/limitations_register.md`
confirms L1 through L28, ascending, no gaps, both before and after
this append.

### Why a post-solve diagnostic cannot move the anchors

Identical argument to Part 1, restated because it is asked for again
here: `extract_planning_diagnostics` runs strictly **after**
`solve_model()` returns. Every value this Part reads —
`m.cost_replacement_capex_npv`, `m.cost_rebuild_capital_charge_npv`,
`m.capital_charge_total_npv` — is a `pyo.Expression` that already
existed, unconditionally (within its branch), in the exact model that
produced both frozen anchors; `pyo.value()` is a pure reader with no
side effect on the solved LP. This edit adds and rearranges what this
function reads out of an already-solved model; it creates no new
Pyomo component, writes to no existing one, and touches no line inside
either builder. The anchors (`21,811,042,472.821114 / 45,502.9754
MWh`; `27.954 bn / 65,613.0 MWh`) are properties of the solved LP,
unaffected by what a diagnostic reads out of it afterward.

### Encountered but not asked about (Part 4)

- The sliced builder's `capital_charge_total_npv` including an extra
  conversion-capex term the annual builder has no equivalent of —
  irrelevant to this function today (confirmed unreachable from it),
  but worth remembering if `extract_planning_diagnostics` is ever
  extended to accept a sliced-model `m` directly (currently the sliced
  runner builds its own diagnostics separately, per `01b_`'s own
  comment).
- No script, solver, or `smoke_all.sh` was run in this Part either,
  per the whitelist; the identity above is verified structurally (by
  reading every term on both sides), not empirically.

**Then STOP.** No commit, no staging. Zeus reviews `git diff` and
commits.
