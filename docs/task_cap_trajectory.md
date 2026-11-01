# Plan 4.2 — post-2035 cap trajectory as an additive arm

Branch `phase-fin2-d17`. No script, solver, or `smoke_all.sh` run. No git
staging or commit performed — Zeus reviews `git diff`, commits, confirms the
four existing objectives reproduce, then runs the new arm.

Files edited: `scripts/00_build_emissions_cap.py`,
`scripts/02_run_ndc_caps.py`. Both diffs are **pure insertions** (89 lines
added, 0 removed — `git diff --stat` confirms). `src/scenarios.py` was
**not edited** — see Step 3B/5 for why. `docs/limitations_register.md` was
**not edited** — it is outside this task's whitelist; §7B below is a draft
for Zeus to place, not an applied change.

---

## STEP ONE — inventory (3A–3E)

### 3A — the cap builder, and the code for 2036–2045 vs. 2046–2054

File: `scripts/00_build_emissions_cap.py`. Two scenario "types" exist, each
with its own loop (lines 206–273 before this task's edit).

**NDC 3.0 branch** (`scen_type == "ndc3_proportional"`, pre-edit lines
248–273), the one relevant to plan 4.2:

```python
elif y <= TARGET_YEAR_2:
    cap_y = _lerp(y, TARGET_YEAR, cap_2030, TARGET_YEAR_2, cap_2035)
else:
    cap_y = cap_2035     # held flat after 2035
```

**Finding, stated plainly because it changes what "the code for 2036–2045"
and "the code for 2046–2054" means: there is no separate code for the two.**
One `else` branch (pre-edit line 260) produces every year from 2036 through
2054 — the in-window years (2036–2045) and the buffer years (2046–2054)
are the *same line*, `cap_y = cap_2035`. The distinction TASK.md's step 3A
asks about is not a code distinction in the original file; it is a
distinction this task's new branch had to *introduce* (see Step 2 below).
The pre-edit comment immediately above that line (pre-edit lines 262–266)
explains explicitly why buffer years are folded into the same flat value
rather than released to an unbound value — this is the D10 arbitrage-sink
fix, cross-referenced in the same terms in `merged_correction_plan_v2.md`'s
DO-NOT-REPORT list (buffer-year row).

The `bau_relative` branch (NDC 2.0, `scen_type == "bau_relative"`, pre-edit
lines 210–245) has the identical structure and an almost verbatim duplicate
of the same comment: `else: cap_y = cap_2030` (pre-edit line 220), one
branch for both 2036–2045 and 2046–2054. Not touched by this task — the new
arm is NDC 3.0-conditional-anchored only, per TASK.md §4.

### 3B — how the four scenarios are named and selected

**Registry:** the `SCENARIOS` dict, `scripts/00_build_emissions_cap.py`
(pre-edit lines 170–201), keyed by the four scenario-name strings
(`ndc3_unconditional`, `ndc3_conditional`, `ndc2_unconditional`,
`ndc2_conditional`), each value a config dict (`type`, the type-specific
anchor fields, `description`).

**Both a dict key and a CSV column value.** The row-generation loop writes
the key verbatim into the output: `"scenario": scen_name` (pre-edit line
242 / 270). So the same string is simultaneously the Python dict key that
selects a scenario's *construction* and the literal value a downstream
reader filters on in the *output*. Confirmed by reading both write and
read sides, not assumed.

**Naming convention.** All four names follow `ndc{2,3}_{unconditional,
conditional}` — base case plus one qualifier, no existing precedent for a
third-level qualifier. TASK.md's suggested name, `ndc3_conditional_decline`
(base name + a descriptive suffix), matches the pattern this project uses
elsewhere for shape variants of a base case (e.g. `gas_retire_vintage_30`
vs `_40`/`_50`, `tariff_central` vs `tariff_high`) — appending a qualifier
to an existing base name, not inventing a new naming axis. Used as given.

**A second registry exists and is the one that actually matters for
"selection."** `NDC_CASES`, `scripts/02_run_ndc_caps.py` (pre-edit lines
403–408), keyed by the *same* four strings, mapping each to a
`capital_case` ("tight" or "moderate"). This is the consumer-side registry
— see 3C.

### 3C — where `02_run_ndc_caps.py` reads the cap series and chooses a scenario

**Read:** `load_annual_caps()` (pre-edit lines 266–291) —
`pd.read_csv(CAP_PATH)` then `df[df["scenario"] == scenario_name]`.
`CAP_PATH = ROOT/"data"/"cost"/"processed"/"emissions_cap.csv"` (pre-edit
line 32) — the same path `00_build_emissions_cap.py` writes to (its
`OUT_PATH`, pre-edit line 109). Confirmed identical path in both files, not
assumed from the filename alone.

**Choice: not a CLI choice at all.** `main()` (pre-edit line 410):

```python
for c, cfg in NDC_CASES.items():
```

Every invocation of `02_run_ndc_caps.py` unconditionally solves **every**
entry in the hardcoded `NDC_CASES` dict, back to back, in one call. The
script's CLI flags (`--peak-case`, `--gas-build-case`, etc.) select the
*surrounding* scenario configuration (gas, hydro, financing, …); nothing
selects *which cap scenarios* run. This is the fact that makes Step 5
("report whether extending the list changes what a default invocation
runs") answer **yes** — see Step 5 below.

### 3D — is the cap series a committed CSV, and would regenerating it overwrite existing columns?

`data/cost/processed/emissions_cap.csv`. **Committed** — `git ls-files`
confirms it is tracked, and `git log -- <path>` shows its last touching
commit (`a834765`) is the *same* commit that last touched
`scripts/00_build_emissions_cap.py`. The two are in sync; the CSV on disk
reflects the current (pre-this-task) code exactly, not a stale prior
version.

**Regeneration overwrites the whole file, not a delta.** `main()` builds
one `rows` list across *all* `SCENARIOS.items()` and writes it once:
`df.to_csv(OUT_PATH, index=False)` (pre-edit line 278) — there is no
append mode, no per-scenario file. Regenerating after this task's edit
would therefore rewrite the entire CSV. **Why this is still additive despite
being a full rewrite:** Python dicts preserve insertion order (3.7+); the
new `SCENARIOS` entry was inserted *last*, after the four existing ones,
and none of the four existing entries' values or the code paths that
consume them were touched. So the first 120 rows (4 scenarios × 30 years,
in the original order) are produced by unmodified code reading unmodified
inputs and will be byte-identical; the new entry's 30 rows are appended
after them. Expected diff on regeneration: **+30 lines at the end of the
file, 0 lines changed or removed.** This task did **not** run the build
script — the CSV currently on disk is still the pre-edit 120-row file, and
was not touched by this task.

### 3E — do the code's 2030/2035 anchors match 13.037 / 12.195 MtCO2?

**Confirmed match — for `ndc3_unconditional`, specifically.** Read the
existing CSV (already in sync with current code, per 3D) with a plain
`csv.DictReader` (no project code executed):

    ndc3_unconditional   2030 = 13.037004027946463 MtCO2
                          2035 = 12.195033869033656 MtCO2

matches TASK.md's established 13.037 / 12.195 to the decimal.

**Worth flagging precisely, because it matters for Step 2:** these two
figures are `ndc3_unconditional`'s anchors, not `ndc3_conditional`'s.
`ndc3_conditional`'s own anchors, read the same way, are:

    ndc3_conditional      2030 = 11.786877468310331 MtCO2
                          2035 = 10.039368724458964 MtCO2

TASK.md §4 directs the new arm to anchor on `ndc3_conditional`'s 2035 value
— i.e. **10.039 MtCO2, not 12.195.** Established fact §1's "NDC-anchored
caps: 2030 = 13.037, 2035 = 12.195" is correct as stated (it is
`ndc3_unconditional`'s pair, and matches code) but is **not** the pair this
task's new scenario is built from. No STOP condition triggered — 3E's check
was "do the code's anchors match the established figures," which they do;
the distinct point above is that §4's construction instructions name a
*different* scenario's anchor, correctly, and this report distinguishes the
two rather than conflating them. The implementation below references
`ndc3_conditional`'s `cap_2035` **programmatically** (`scen_cfg["cap_2035"]`
sourced from the existing `ndc3_cond_cap_2035` variable), never as a
hardcoded literal — so it tracks whichever value that scenario actually
carries, rather than freezing today's 10.039 into a second, driftable copy.

---

## STEP TWO — the new trajectory: edits made

### Edit 1 — `scripts/00_build_emissions_cap.py`, `SCENARIOS` dict

**Section:** the `SCENARIOS` dict (Step 8, "Build scenario definitions").
**Old text:** dict ended after the `ndc2_conditional` entry (pre-edit line
200, closing `}` at line 201).
**New text:** one new entry appended after `ndc2_conditional`, keyed
`"ndc3_conditional_decline"`, `"type": "ndc3_declining"`, `cap_2030`/
`cap_2035` set to `ndc3_cond_cap_2030`/`ndc3_cond_cap_2035` (the same
in-scope variables `ndc3_conditional`'s own entry uses — not re-derived, not
hardcoded), plus a `description` string and a comment block recording: the
2025–2035 identity with `ndc3_conditional`; that the 2060 endpoint is
policy-anchored (Nigeria's own stated net-zero-2060 target) while the linear
interpolation to it is an analyst construction, stated "in the same terms
plan 4.1 uses for the proportional apportionment rule" per TASK.md §4; and
why the buffer years depart from this file's flat-hold convention (reopens
the D10 arbitrage sink under a flat buffer + declining in-window cap; a flat
hold would also contradict the trajectory's own premise). Full text in the
file.

### Edit 2 — `scripts/00_build_emissions_cap.py`, row-generation loop

**Section:** Step 9, "Build annual cap trajectories," after the
`bau_relative` and `ndc3_proportional` branches, before "Step 10, Write
output."
**Old text:** loop ended after the `ndc3_proportional` branch's `continue`.
**New text:** a third branch, `if scen_type == "ndc3_declining":`,
implementing exactly the formula TASK.md §4 gives:

```python
cap_y = cap_2035 * (1.0 - (y - TARGET_YEAR_2) / (NET_ZERO_YEAR - TARGET_YEAR_2))
```

for every `y > 2035` (2025–2030 and 2030–2035 reuse the same `_lerp` calls
the `ndc3_proportional` branch uses, against the *same* `cap_2030`/
`cap_2035` values — this is how "2025–2035 unchanged from `ndc3_conditional`"
is actually enforced, not merely stated). `NET_ZERO_YEAR = 2060` is a local
constant, commented as policy-anchored. No `max(cap_y, 0.0)` special-casing
needed or added beyond the existing row-append line — the decline stays
positive throughout the model's 2025–2054 span (multiplier ranges
0.96 at 2036 down to 0.24 at 2054; only reaches exactly 0 at 2060, past the
model horizon).

**Verified by hand** (given values, not re-run): at 2045,
`10.039368724458964 × (1 − 10/25) = 6.0236 MtCO2` = `cap(2035) × 0.60`,
matching TASK.md's stated "0.60 × cap(2035), a 40% reduction against the
current flat hold" exactly. At 2054 (terminal buffer year),
`× (1 − 19/25) = × 0.24 → 2.409 MtCO2`.

**Syntax-checked** (read-only, authorised): `python3 -m py_compile
scripts/00_build_emissions_cap.py` — passes.

---

## STEP THREE — wiring: edits made

### Edit 3 — `scripts/02_run_ndc_caps.py`, new CLI flag

Added `--include-decline-arm` (`action="store_true"`, `default=False`) to
`parse_args()`, after `--om-case`. Help text states plainly that a default
invocation is unaffected and that this arm is never in
`run_central_sweep.sh`.

### Edit 4 — `scripts/02_run_ndc_caps.py`, `NDC_CASES`

**Old text:** `NDC_CASES` dict defined with exactly four entries, then
immediately `for c, cfg in NDC_CASES.items():`.
**New text:** between the dict literal and the loop:

```python
if args.include_decline_arm:
    NDC_CASES["ndc3_conditional_decline"] = {"capital_case": "moderate"}
```

**Why this preserves the default run exactly.** The four-entry dict literal
is untouched. Without the new flag (`args.include_decline_arm` is `False`
by construction unless a caller explicitly passes `--include-decline-arm`),
this `if` body never executes, `NDC_CASES` is the identical 4-entry dict it
was before this task, and the loop that follows runs the identical code,
in the identical order, over identical scenario configs. **This is a
stronger additivity guarantee than the CSV case in 3D**: it is not "the
same computation happens to produce the same output," it is "the exact
same code path executes, unconditionally, unless a new opt-in flag is
passed."

**Design decision not specified in TASK.md, made and flagged here:** the
new entry's `capital_case` is set to `"moderate"` — the same value
`ndc3_conditional` uses. Reasoning given in the code comment: this arm
shares `ndc3_conditional`'s 2025–2035 caps and NDC ambition tier exactly;
only the post-2035 cap *shape* differs, so there is no basis to also vary
the capital ceiling. TASK.md did not specify `capital_case` for the new
arm — this was inferred from the anchor relationship stated in §4, not
guessed independently.

**Syntax-checked:** `python3 -m py_compile scripts/02_run_ndc_caps.py` —
passes.

### `src/scenarios.py` — confirmed not to need editing

TASK.md's whitelist named this file as a candidate ("or whichever file
builds the cap series — verify first"). Checked: `grep -n
"ndc3_conditional\|ndc2_unconditional\|ndc3_unconditional\|ndc2_conditional\|
emissions_cap\|cap_scenario\|NDC_CASES" src/scenarios.py` returns nothing.
The emissions-cap pipeline is self-contained in the producer
(`00_build_emissions_cap.py`) and the one consumer this task was asked to
wire (`02_run_ndc_caps.py`); `src/scenarios.py` (the `load_scenario()`
registry of demand/gas/financing/etc. arms) has no cap-scenario concept at
all. Not edited, per the "no other file without reporting why" rule — this
is the report of why.

### `run_central_sweep.sh` — confirmed not touched

Not in the whitelist and not edited. `--include-decline-arm` defaults off,
so even if the sweep script invoked `02_run_ndc_caps.py` (it does not — the
sweep is `01b`-based; `02_run_ndc_caps.py` is not part of
`run_central_sweep.sh` at all, confirmed by grep) it would be unaffected.

---

## STEP FOUR — the additivity proof

**Existing cap series, current state.** `data/cost/processed/
emissions_cap.csv` on disk is **byte-identical to before this task** — no
script was run, so nothing regenerated it. It holds the original 4
scenarios × 30 years = 120 rows.

**Expected diff once Zeus regenerates it.** Run:

    python scripts/00_build_emissions_cap.py

Expected: the first 120 rows unchanged (same scenario order, same values —
argued in 3D/Edit 1: none of the four existing `SCENARIOS` entries, their
computation branches, or their upstream inputs were touched); 30 new rows
appended at the end, `scenario == "ndc3_conditional_decline"`, years
2025–2054. `git diff data/cost/processed/emissions_cap.csv` should show
only additions.

**Command to confirm the four existing capped objectives reproduce.**
Reconstructed from the frozen central configuration
(`docs/findings_frozen.md`, "The central configuration") mapped onto this
script's own `argparse` flag names — **not run by this task**, so treat as
a best-evidence reconstruction for Zeus to execute, not a confirmed result:

```bash
python scripts/02_run_ndc_caps.py \
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

Expected: `npv_total_cost_usd` in the four `summary.json` outputs reproduces
53.821 / 63.374 / 58.698 / 51.573 bn (ndc2_unconditional / ndc2_conditional
/ ndc3_unconditional / ndc3_conditional). Requires
`results/baseline/diagnostics.json` to already exist (unchanged by this
task) and `data/cost/processed/emissions_cap.csv` to already be regenerated
per the step above.

**Command to run the new arm.** Same command, plus one flag:

```bash
python scripts/02_run_ndc_caps.py \
  --peak-case peak_unsuppressed --storage-cc-case storage_cc_central \
  --gas-build-case gas_unconstrained --new-gas-deliverability constrained \
  --gas-availability-case gas_avail_nameplate \
  --gas-retirement-case gas_retire_vintage_30 \
  --gas-capex-case gas_capex_central --ccgt-efficiency-case ccgt_eff_central \
  --capital-recovery-case cr_central --dscr-case dscr_central \
  --us-inflation 0.020 --capital-treatment-case f1_tranche \
  --concessional-envelope-case env_dares --gas-deliverability-case baseline \
  --conversion-case conv_central --hydro-case hydro_flat \
  --tariff-case tariff_central --om-case om_central \
  --include-decline-arm
```

This solves all five cases (the original four, byte-identically per the
`NDC_CASES` argument in Edit 4, plus `ndc3_conditional_decline`).

---

## 7A — For the methodology chapter

The model's emissions cap is not a single sourced series; it is built in
two layers, and NDC 3.0 only supplies the first. NDC 3.0 states two
numerical targets — an economy-wide abatement of 168.2 MtCO2e by 2030 and
184.9 MtCO2e by 2035 — and an energy-sector share of each. This model
apportions a further share of that energy-sector abatement to gas-fired
power generation specifically, in proportion to gas-power's own share of
modelled energy-sector emissions (plan 4.1). That apportionment rule is
this project's construction, not NDC 3.0's; NDC 3.0 says nothing about how
much of its energy-sector target the power sector alone should carry, let
alone gas-fired power within it. The result is two anchor points per cap
scenario — a 2030 cap and a 2035 cap — built from real NDC 3.0 numbers
through one stated, but unavoidably analytical, apportionment step.

Past 2035, NDC 3.0 gives no target at all. Every cap value from 2036
onward — ten of the twenty-one years this model reports, plus the entire
2046–2054 buffer — is therefore a second-layer construction, stacked on
the first. Until this task, that construction was a single, largely
unstated choice: hold the 2035 cap flat through the rest of the horizon.
Flat is not a neutral default. It encodes an implicit assumption that
climate ambition stops accumulating in 2035, which understates the
abatement a country actually committed to net-zero by 2060 would need to
deliver across the back two-thirds of a thirty-year power-sector planning
horizon. The flat hold was silently doing policy work it was never
declared to be doing.

This task adds a second, declared construction alongside the first rather
than replacing it: a cap that declines linearly from the `ndc3_conditional`
2035 anchor toward zero at 2060 — the year Nigeria has itself named as its
net-zero target, making the endpoint policy-anchored rather than
analyst-chosen. The straight line between the 2035 anchor and that endpoint
is, in exactly the same sense as the plan 4.1 apportionment rule, this
project's own construction: no Nigerian document specifies that the path
between those two points is linear rather than front- or back-loaded. It is
reported here as a sensitivity arm precisely because of that — a shape
assumption introduced three days before a model freeze is evidence that a
range exists, not a claim that this particular range is correct. Reporting
it as a bound alongside the flat hold, rather than substituting one
unstated assumption for one declared one, is the more defensible position
this close to submission.

## 7B — For the limitations register (drafted, not applied)

**Numbering note, read before placing this entry.** TASK.md's established
facts state "Register L20–L23 exist. Do not renumber." Checked against the
file (`grep -n "^## L[0-9]" docs/limitations_register.md`): L20 and L23
exist as headed entries; **L21 and L22 do not** — they are referenced by
number three times in the body of L23 ("compare... L22", "unlike L19 and
L21", "the O&M omission (L22)") as though they were written, written entries
never exist. This is a pre-existing gap, not something this task caused or
was asked to fix — flagged below under "encountered but not asked about,"
not repaired. Given L21/L22 are referenced-but-absent, and L23 is the
highest entry that actually exists, this task treats **L24** as the next
free number, rather than filling the L21/L22 gap (whose intended content
this task has no basis to reconstruct). `docs/limitations_register.md`
itself was **not edited** — this file is outside this task's whitelist.
The text below is a draft for Zeus to place.

> ## L24 — Post-2035 cap shape is a second analyst construction stacked on the apportionment rule
>
> **Status:** LIVE — declared, sensitivity arm added (plan 4.2)
>
> **What.** NDC 3.0 anchors only 2030 and 2035. Every capped year from 2036
> onward — 2036–2045 in the reporting window, all of 2046–2054 in the
> buffer — is therefore unanchored by NDC 3.0 itself, for every cap
> scenario in this model. Two constructions for that unanchored span now
> coexist: the original **flat hold** (2035 cap held constant to the
> horizon end; `ndc2_unconditional`, `ndc2_conditional`,
> `ndc3_unconditional`, `ndc3_conditional`), and the new
> **`ndc3_conditional_decline`** arm (plan 4.2), which instead declines
> linearly from the `ndc3_conditional` 2035 cap toward zero at 2060 —
> Nigeria's own stated net-zero target — continued through the buffer
> rather than held flat.
>
> **Why unresolved.** NDC 3.0 (Section 3, pp.14–15) states numerical
> targets only for 2030 and 2035. No Nigerian document sourced during this
> project specifies a power-sector-level post-2035 pathway shape, so
> neither construction can be upgraded to SOURCED.
>
> **Interaction with plan 4.1 — two stacked constructions, stated
> plainly.** The cap value at *any* year, including 2030 and 2035
> themselves, already rests on plan 4.1's apportionment rule: gas-fired
> power's share of energy-sector abatement, derived from this model's own
> baseline gas-to-power split — a methodological choice, not something in
> NDC 3.0's text (4.1's own wording: "your methodological construction, not
> something in NDC 3.0"). Past 2035, a *second* construction stacks on top
> of the first: which shape the cap takes once NDC 3.0's own numbers run
> out. Ten of the twenty-one reported years, and the entire buffer, carry
> both constructions at once. State both, together, wherever a post-2035
> capped result is reported — citing only the apportionment rule
> understates how much of the number is this project's own construction.
>
> **Direction of bias.** The two constructions are not symmetric.
> **The flat hold is not neutral** — it is an implicit "climate ambition
> stops in 2035" assumption, understating the abatement a country actually
> committed to net-zero-2060 would need across the back two-thirds of the
> horizon. **The linear decline's endpoint and direction are sourced**
> (net-zero-2060 is Nigeria's own target), **but its shape is not** — a
> real decarbonisation pathway is not necessarily linear (most observed
> trajectories are convex, slow-then-fast), so the straight line is a
> declared simplification, not a second sourced figure. Net: flat
> understates second-half ambition in one direction; the decline's
> endpoint is directionally correct but its interpolation is exactly as
> analyst-chosen as the apportionment rule it stacks on.
>
> **Mitigation.** `ndc3_conditional_decline` carried as an additive
> sensitivity (plan 4.2), opt-in only via `--include-decline-arm` on
> `02_run_ndc_caps.py` (default off), never central, never in
> `run_central_sweep.sh`. The four original flat-hold scenarios are
> unmodified — see `docs/task_cap_trajectory.md` for the additivity proof.
>
> **What would resolve it.** A Nigerian 2036–2060 sectoral decarbonisation
> pathway (an LT-LEDS or equivalent, if one specifying power-sector shape
> exists) would replace the straight line with a sourced curve. Absent
> that, report the flat-hold and declining arms as a declared band, not a
> single central figure.

---

## Encountered but not asked about

- **`docs/limitations_register.md` L21 and L22 do not exist**, despite
  being referenced three times by number inside the existing L23 entry (as
  though already written) and despite TASK.md's established facts stating
  "Register L20–L23 exist." Verified with `grep -n "^## L[0-9]"` — only
  L20 and L23 are present in that number range (plus L16 and L18, each
  still duplicated under two different bodies, flagged in the previous
  task's report and still unresolved). This task numbered its draft
  register entry L24, treating L23 as the true highest entry, rather than
  attempting to fill the L21/L22 gap — reconstructing what those two
  entries were meant to contain is a separate investigation this task was
  not asked to perform.
- Several other runners hardcode the same four cap-scenario names
  (`03_run_ndc_eaas.py`, `04_run_fin2_tariff_sweep.py`,
  `05_run_fin3_capital_sweep.py`, `06_`, `07_`, `08_phase8_analysis.py`,
  `08_run_gas3_regime_ndc_feasibility.py`, `11_`, `12_`, `13_`, `15_`,
  `16_`, `24_`, `25_`, `26_`, `99c_`, `scripts/deferred/17_run_monte_carlo.py`)
  and would need their own wiring to select `ndc3_conditional_decline` if
  a future task wants the new arm exercised through them. Most of these
  overlap with register L10's dead-runner list. Not touched — TASK.md's
  whitelist named only `02_run_ndc_caps.py`.
- The `bau_relative` branch (NDC 2.0 scenarios) carries the identical
  "one `else` line covers both 2036–2045 and 2046–2054" structure and an
  almost-verbatim duplicate of the buffer-arbitrage comment already in the
  `ndc3_proportional` branch (pre-edit lines 221–238 vs 262–266) — the same
  duplication pattern register L14 already documents for `_rebuild_years`
  in `src/optimize_model*.py`. Not touched; TASK.md scoped this task to
  the NDC 3.0 conditional anchor only.
- This task could not execute `00_build_emissions_cap.py` or
  `02_run_ndc_caps.py` (barred by TASK.md's whitelist), so every numeric
  claim above the code-reading level — the 0.60× / 0.24× decline
  multipliers, the reconstructed central-configuration CLI command, the
  expected 53.821/63.374/58.698/51.573 bn reproduction — is verified by
  hand arithmetic or by cross-referencing already-committed documents, not
  by running anything. Flagged per standing instruction that results
  produced during correction (here: none, since nothing was run) are
  verification artefacts, not thesis findings — the actual verification
  step is Zeus's, per the commands in Step Four.
