# Task report — storage O&M: sourcing gate, then wire it

Branch `phase-fin2-d17`. Written incrementally per TASK.md.

---

## PART 1 — two-site subscript fix (BYTE-IDENTICAL)

**Line-number note.** TASK.md cites `optimize_model_sliced.py:461`; fresh
read finds the site at line 467 (drift from intervening edits in an
earlier task, consistent with prior line-drift findings in this
codebase). `optimize_model.py:386` matches exactly.

### Edit 1 — `src/optimize_model.py:386`

Old text:
```python
    storage_duration_hours = scenario.get("storage_duration_hours", 4.0)  # kept for storage_power_mw only
```

New text:
```python
    storage_duration_hours = scenario["storage_duration_hours"]  # kept for storage_power_mw only
```

### Edit 2 — `src/optimize_model_sliced.py:467`

Old text:
```python
    storage_duration_hours = scenario.get("storage_duration_hours", 4.0)
```

New text:
```python
    storage_duration_hours = scenario["storage_duration_hours"]
```

### Value-preservation argument

`storage_duration_hours` is always present on every scenario dict:
`load_scenario()` sets it unconditionally at `scenarios.py:2666`
(`_storage_duration_hours = 4.0`) and surfaces it on the returned dict
at `scenarios.py:3021` — there is no code path through `load_scenario()`
that omits the key, and §1 records that no hand-built scenario dict
literal reaches either builder. Therefore `scenario.get("storage_duration_hours", 4.0)`
and `scenario["storage_duration_hours"]` return the identical object on
every existing path: the `.get()` fallback of `4.0` is never actually
exercised, only ever the value already in the dict. The change removes
a rival literal (a second `4.0` that could in principle drift from
`scenarios.py:2666` if either were edited without the other) and
converts a future accidental omission of the key from a silent
fallback to a `KeyError`, which is louder, not different, in every case
that can currently occur.

**Why this cannot change a computed value.** Both edits replace a
`.get(key, default)` call with a `[key]` subscript where the key is
statically guaranteed present; no arithmetic, constraint, or control
flow is touched. Neither `load_scenario()`'s signature nor
`scenarios.py:2666` was changed, per TASK.md's instruction.

**Syntax check:**
```
$ python3 -m py_compile src/optimize_model.py src/optimize_model_sliced.py
(clean, no output)
```

**These two edits do not affect Part 2 or Part 3 below and are reported
separately, as instructed, so they can be committed and verified
(against both frozen anchors) independently of anything else in this
task.**

---

## PART 2 — SOURCING GATE

### 2A — Every storage O&M definition in the tree

Search terms `om`, `o_and_m`, `opex`, `fom`, `vom`, `fixed_om`,
`variable_om` (case-insensitive) against `src/io.py` and every CSV under
`data/cost/`, plus a full-tree grep for the resulting symbol name.
Exactly **one** definition exists anywhere in the tree:

`src/io.py:237-240`, quoted with its surrounding comment in full:

```python
    # Storage annual fixed O&M. Primary purpose: break LP degeneracy in storage
    # sizing (without this, any storage capacity >= ~10 GWh is equally optimal).
    # Value is conservative (NREL ATB range: 10,000–18,000 USD/MWh-yr).
    econ["STORAGE_OM_PER_MWH_YR"] = 2_000.0
```

No storage O&M column or value exists in any cost CSV
(`data/cost/processed/storage_capex.csv`, `storage_capex.xlsx`,
`cost_registry.yaml`, or any other file under `data/cost/`) — the CSV
carries only `Storage_capex_usd_per_mwh` (a capital, not operating,
figure) per year and scenario (`Storage_low` / `_mid` / `_high`).

### 2B — Provenance

**Quoted verbatim, no provenance marker beyond the inline comment
above.** There is no `SOURCED` / `DERIVED` / `ASSUMED` marker (contrast
with `genset_cost.csv`'s `note` column, e.g. "eta SOURCED = mean of
Caterpillar C15/C18/3512 PRIME at 25% load..."), no retrieval date, and
no `[SOURCE NEEDED]` tag. **Absence is the finding, not an inference
from plausibility.**

What the comment *does* say is itself a red flag, not a citation:

- The stated **purpose** is "break LP degeneracy in storage sizing" —
  i.e. the value's job is to make the LP's storage-sizing decision
  non-degenerate (pick a unique optimum) among otherwise-tied storage
  capacities, not to represent a real operating cost.
- The value used, **2,000 USD/MWh-yr, is itself outside the range the
  comment cites** ("NREL ATB range: 10,000–18,000 USD/MWh-yr") — it is
  roughly a fifth of the cited range's low end, not "conservative"
  within it. "Conservative" here evidently means "small enough not to
  distort results, while still breaking the tie," which is a numerical-
  conditioning design goal, not a cost estimate.
- No NREL ATB year, technology assumption (chemistry, duration), or
  page/table reference is given for the cited 10,000–18,000 range
  itself, so even that range cannot be independently checked from this
  comment alone.

**Conclusion: this value has no usable provenance.** It was chosen to
serve a numerical role (tie-breaking) in a code path that (per 2D below)
never actually uses it, and it does not match a citable NREL ATB figure
on its own terms.

### 2C — Units and basis

**USD/MWh-yr — determined from the variable name and inline comment
("Storage annual fixed O&M" / `_PER_MWH_YR` suffix), not inferred.**
This is an ENERGY-capacity basis (USD per MWh of installed storage
energy capacity, per year), not a power-capacity basis (it is not
USD/kW-yr) and not a throughput basis (it is not USD/MWh discharged).
No unit comment gives a conversion factor to any other basis; the value
would need to be multiplied by cumulative installed `storage_capacity_mwh`
in each year to enter a cost expression, matching its own name directly.

### 2D — Is it read anywhere at all?

**No functional reader anywhere in the tree.** `grep -rn "STORAGE_OM_PER_MWH_YR"`
across `src/`, `scripts/`, and `tests/` returns exactly two hits:

1. `src/io.py:240` — the definition itself.
2. `scripts/14_run_str1_storage_role.py:30` — a comment, not code, in a
   **SUSPENDED** runner's module docstring, stating plainly: *"STORAGE_OM_PER_MWH_YR
   : never entered the objective"* (listed there alongside
   `storage_solar_surplus_frac` as one of two "no-op" sweep axes that
   made that runner's entire prior sweep produce identical LPs).

Neither `optimize_model.py` nor `optimize_model_sliced.py` contains any
`om`/`opex`/`fom`/`vom` term at all (confirmed by the same search
against both files directly — zero matches in either). `econ["STORAGE_OM_PER_MWH_YR"]`
is set and then never subscripted by either builder. This independently
corroborates plan 2.2's note ("defined in io.py, never read") and the
suspended runner's own comment.

### 2E — How is GAS O&M handled in the objective?

**It is not.** Gas carries no O&M term of any kind — only fuel cost.
The sole gas operating-cost term in the objective is `gas_opex_npv`
(`src/optimize_model.py:1064-1067`), quoted with its preceding comment:

```python
    # Gas cost is in USD per TWh_th. Fuel OPEX is unconditional -- it is
    # untouched by F1 (F1 only replaces the CAPEX/financing side).
    gas_opex_npv = pyo.quicksum(
        m.DF[t] * m.gas_to_power[t] * econ["GAS_COST_PER_TWH_TH"]
        for t in T
    )
```

This is fuel cost only (`gas_to_power[t]`, an energy/throughput
quantity, times a USD/TWh fuel price), discounted by the social discount
factor `m.DF[t]`. There is no separate fixed-O&M-per-MW or
variable-O&M-per-MWh term added to it anywhere in either builder. The
comment's own label — "Fuel OPEX" — is accurate to what the code does;
it is `gas_opex_npv`'s *name* that is potentially misleading if read as
"all gas operating costs," since it is fuel only.

**This means Part 2E's premise does not hold: there is no existing
gas-O&M pattern for Part 3 to follow**, because gas has no O&M term to
follow. The only "basis" this reveals is that gas's operating cost is
carried on an *energy* (generation) basis, not a capacity basis — but
that is a fuel-cost basis question, not an O&M-treatment template.

### 2F — How is SOLAR O&M handled?

**Also not at all — solar carries no O&M term either**, widening the
defect beyond storage, exactly as TASK.md's 2F anticipated as a
possible outcome. Solar's only cost terms in the objective are capital:
`public_solar_capex_npv` (`optimize_model.py:704-708`) and
`eaas_solar_capex_npv` (immediately following it), both of the form
`m.DF[t] * <capacity added in year t> * <capex per MW>` — capital
expenditure on newly-added capacity, discounted, with no fixed or
variable operating term for either publicly- or EaaS-financed solar.

**Net finding: no technology in either LP builder's objective carries a
fixed or variable O&M cost term.** Gas pays fuel only; solar and storage
pay capital cost only (plus, for storage/gas under `pre_f1_overnight`,
replacement capex and salvage — still capital-side, not O&M). The
omission plan 2.2 flagged for storage specifically is, on this
inventory, a system-wide omission, not a storage-specific one.

---

## → STOP HERE, per TASK.md's instruction

**The gate does not pass in the form TASK.md anticipated, for two
independent reasons, either of which is sufficient to withhold release
of Part 3 pending Zeus's and the supervisor's decision:**

1. **No usable provenance.** `STORAGE_OM_PER_MWH_YR = 2_000.0` carries
   no SOURCED/DERIVED/ASSUMED marker, no citation Zeus could verify
   independently, and is explicitly a numerical-conditioning constant
   (chosen to "break LP degeneracy," and itself inconsistent with the
   only range its own comment cites) rather than a cost estimate. Wiring
   this specific value into the objective would inject an unsourced
   parameter directly into a reported result — the outcome TASK.md's
   preamble says is "worse than the omission it fixes."
2. **No template exists.** Part 3's instruction is to wire storage O&M
   "following exactly the pattern Part 2E found for gas." Part 2E finds
   that gas has no O&M pattern — no technology has one. There is
   nothing to replicate structurally, only a decision to introduce a
   wholly new cost category into both builders' objectives, for one,
   two, or three technologies at once.

**Part 3 is NOT begun.** No edit has been made to either builder's
objective. This section of the report is written for Zeus and the
supervisor to decide, per TASK.md: "Zeus and the supervisor decide; you
do not."

---

## PART 3 — WIRE IT

**Not released. Not started.** No content to report here beyond the
gate finding above.

---

## PART 4 — required writing deliverables

### 4A — For the methodology chapter

Every cost term that enters this model's objective is a discounted net
present value: a stream of annual dollar amounts, one per model year, is
multiplied by a social-discount factor `DF[t] = 1 / (1 + r_social)^t`
(`r_social` = 8% central) and summed across the twenty-one year planning
horizon. What varies between cost categories is not the discounting
mechanism but which quantity each dollar amount is proportional to.
Capital costs — solar, gas, and storage capacity additions — are priced
per unit of newly-added capacity in the year it is added, multiplied by
a capital cost per MW or MWh; this is a one-off charge at build time (in
the pre-F1 overnight-capex treatment) or an annuitised tranche charge
spread over a private lending tenor (under the current `f1_tranche`
treatment). Operating costs are, in principle, a distinct category:
recurring charges tied either to how much a technology is used
(variable, energy-basis) or to how much capacity exists regardless of
use (fixed, capacity-basis).

In this model, that second category is populated for exactly one
technology and one purpose: gas fuel cost, `gas_opex_npv`, charges each
year's gas generation at a USD-per-TWh fuel price. No fixed or variable
operating-and-maintenance charge exists anywhere in either builder's
objective, for any of solar, storage, or gas capacity. A parameter for
storage O&M is defined in the cost-loading module (`STORAGE_OM_PER_MWH_YR`,
2,000 USD per MWh of storage energy capacity per year) but is never read
by either builder — it was introduced, by its own inline comment, to
break a degenerate tie in the storage-sizing decision at annual
resolution, not to represent storage's true recurring cost, and a
suspended sensitivity runner's own docstring independently confirms it
"never entered the objective."

This matters specifically for storage because storage is the asset
class this thesis's financing argument concerns. An asset that is
capital-cost-only in the objective is systematically cheaper, over its
lifetime, than one that also carries a recurring maintenance charge —
and because gas's only operating cost (fuel) scales with use while
storage's absent operating cost would have scaled with installed
capacity regardless of use, the omission does not bias the gas-versus-
storage comparison symmetrically. Any conclusion this model draws about
storage's cost-competitiveness against gas, or about the scale of
storage deployment the optimum selects, is drawn without a term that
would, if included, make storage relatively more expensive than it
currently appears. Because the omission spans all three technologies
rather than storage alone, it is better understood as a general
under-specification of operating cost in this model's objective than as
a storage-specific gap — but storage is the technology for which that
gap carries the most weight, because it is the one whose economics this
thesis's central financing argument turns on.

### 4B — For the limitations register

The gate did not pass this task, so Part 3 was not released. This is
therefore a declared omission, not a wired-and-sourced term. Draft entry,
following the register's established format:

> ## L22 — No technology in the LP objective carries an O&M cost term; a storage O&M parameter exists but is unread
>
> **Status:** LIVE
>
> **What.** Neither builder's objective (`src/optimize_model.py`,
> `src/optimize_model_sliced.py`) contains a fixed or variable
> operating-and-maintenance term for any technology. Gas's only
> operating cost is fuel (`gas_opex_npv`, USD-per-TWh fuel price ×
> generation); solar and storage carry capital cost only. A storage O&M
> value, `econ["STORAGE_OM_PER_MWH_YR"] = 2_000.0` (`src/io.py:240`), is
> defined during cost loading but is never subscripted by either
> builder — it has zero functional readers anywhere in the tree.
>
> **Classification: ASSUMED, and unusable as sourced even if wired.**
> The inline comment at the definition site states its purpose as
> breaking LP degeneracy in storage sizing at annual resolution, not
> representing a real operating cost, and cites an "NREL ATB range" of
> 10,000-18,000 USD/MWh-yr that the chosen value (2,000) itself falls
> outside of, with no ATB year, technology, or table reference given.
> There is no SOURCED/DERIVED/ASSUMED marker on it, unlike e.g.
> `genset_cost.csv`'s per-parameter provenance notes.
>
> **What it controls (were it wired).** Would add a recurring,
> capacity-proportional charge to the objective for whichever
> technologies received one — here, only storage's parameter exists at
> all, so wiring it alone (without an equivalent for gas and solar)
> would introduce an asymmetric operating-cost treatment rather than
> resolve the omission uniformly.
>
> **Direction of bias — for the period of omission.** All three
> technologies are understated in lifetime cost relative to a model that
> charged O&M, but not uniformly: gas's recurring cost is partially
> captured through fuel, which scales with generation; solar and storage
> have no recurring cost captured at all. Storage is the asset class
> most affected in relative terms, because it is the one this thesis's
> financing argument turns on, and because the LP's only reason to
> prefer it over gas is a comparison from which its own recurring costs
> are entirely absent.
>
> **Direction of bias — for the unsourced value, if it were wired
> as-is.** Unknown and not defensible either way: the value is not a
> cost estimate, so using it would not correct the omission in a known
> direction — it would replace an omission of unknown size with an
> arbitrary charge of unknown accuracy, which could over- or
> under-correct depending on where the true figure sits relative to
> 2,000 (or relative to the cited but uncorroborated 10,000-18,000
> range).
>
> **What would resolve it.** Source a real fixed and/or variable O&M
> figure for each of solar, storage, and gas (an ATB citation with year,
> technology, and table reference, at minimum, for storage), decide the
> basis (capacity vs. energy) for each consistently with how each
> technology's other costs are already carried, and wire all three
> together so the gas-vs-storage-vs-solar comparison is not corrected
> asymmetrically. `docs/task_storage_om.md` Part 2 is the inventory this
> would start from.

---

## Closing list — things encountered but not asked about

- `optimize_model_sliced.py`'s `storage_duration_hours` line has drifted
  from TASK.md's cited 461 to 467; no other line-number drift was found
  in Part 1's two sites.
- The `STORAGE_OM_PER_MWH_YR` comment's cited "NREL ATB range" (10,000-
  18,000 USD/MWh-yr) has no year, technology, or table reference and was
  not independently checked against IRENA/NREL ATB source material in
  this task (out of scope: sourcing a real figure is Part 3's
  precondition, not this gate's job).
- `scripts/14_run_str1_storage_role.py` is SUSPENDED pending Phase 2
  intra-annual structure; its docstring's broader claim (that annual-
  resolution storage is always weakly dominated and never adds value)
  was not re-verified here, since it is orthogonal to the O&M question.
- No check was made of whether `genset_cost.csv`'s `fixed_om_usd_per_kw_yr`
  / `om_cost_usd_per_mwh` columns (private self-generation backstop, not
  the grid-gas fleet) are read anywhere or feed into the LP objective —
  out of scope for this task, which was scoped to storage O&M and the
  gas/solar comparators named in 2E/2F specifically.
