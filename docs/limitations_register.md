# Limitations register

**Standing document. Every limitation that cannot be engineered away gets an
entry here at the moment it is identified — not at write-up time.**

Each entry states: what the limitation is, why it cannot be resolved within
scope, **which direction it biases results**, and what would resolve it. A
limitation whose direction is named is defensible; one whose direction is
unknown is not.

**Result generations (TASK 86 B.4).** Two generations of stored results coexist (L142). *Blended*: dispatch without the vintage split, `gas_vintage_dispatch` absent from the arm tag or `False`; central objective 24,986,620,949.18 USD, hash `6046f759`. *Vintage*: the arm tag carries `gas_vintage_dispatch=True`; central objective 23,323,068,340.46 USD, hash `661b03c1`. A figure in this document describes the generation of the result file it is taken from. Where an entry cites a hash-keyed result file, a **Generation** line at its end says which generation those files belong to, checked against each file's arm record by `scripts/task86_generation_tags.py`; fixed-name and pattern citations are not assessed, because a fixed name is overwritten by later runs. An entry with no Generation line cites no hash-keyed file and describes structure or provenance, or its generation is not established by a file.

Status values: `LIVE` (stands, goes into the thesis) / `RESOLVED` (engineered
away, kept for the record) / `PENDING` (may yet be resolved).

---

## L1 — Demand shape is a 2016 vintage

**Status:** LIVE

**What.** The intra-day and seasonal demand shape driving the six-slice
structure and the firm-capacity constraint is derived from Oluwole (2023),
*Nigeria National Electricity Demand Timeseries*, DOI 10.17632/pxvdm26rn7.2 —
8,784 hourly observations for calendar 2016. Planning horizon begins 2024.

**Why unresolved.** No alternative exists. NIRP 2024 contains no daily load
profile, no typical-day curve, no peak-hour timing and no statement of its
model's temporal resolution; the only hourly dataset it names is for hydro
seasonality. NERC publishes period averages, not shapes. The alternative to a
2016 shape is an undeclared assumption, which is what the model previously
carried (period factors 0.65 / 1.05 / 1.20, unsourced).

**What is actually taken from 2016.** The *shape* only — the ratios
`period_demand_factor` and `season_demand_factor`. Demand *levels* come from
NERC 2024 and the growth arms. Load shapes are structurally persistent
(near-constant equatorial daylight, work and school schedules, cooking and
lighting habits, sectoral mix); levels are not.

**Direction of bias.** Three post-2016 changes plausibly moved the shape, and
all three move it the same way: distributed solar under DARES/REA and C&I EaaS
shifts daytime load behind the meter; 2023 subsidy removal changed
self-generation economics; 2024 Band A tariff reform reallocated served supply.
Each concentrates grid draw in the evening. **So the true 2024 shape is
expected to be more evening-peaked than 2016, meaning the central arm
understates firm-capacity requirements and therefore understates the cost of a
solar-heavy pathway.**

**Mitigation in place.** `peak_high` arm sweeps the direction of the drift by
raising the evening factor to 1.20. The vintage risk is therefore carried as a
sensitivity, not as an unquantified caveat.

**What would resolve it.** A post-2023 hourly national demand series with an
unsuppressed reconstruction. NISO/TCN hold the underlying operational data.

---

## L2 — Suppressed series is a supply shape and cannot anchor demand

**Status:** LIVE (methodological, and reported as a finding)

**What.** The served (suppressed) 2016 series cannot be used as a demand
anchor. Measured directly: suppression flattens the evening specifically —
peak factor falls 1.088 → 0.983 while night *rises* 0.906 → 1.022. Suppression
ratio 0.4192; only 41.9% of demand was served.

**Why it matters.** The suppressed series carries a *higher* instantaneous PAR
(1.6891 vs 1.4588) but a *lower* block PAR (0.9807 vs 1.0847) — brief supply
spikes against a suppressed base. That is a supply signature, not a load curve.

**Treatment.** `peak_suppressed` is retained as a declared floor, explicitly
not a candidate central case. This is the same discipline applied to
`tier2_low` and to the rejection of generation CAGR as a demand growth anchor.

**Reportable.** This is the constraint-relief asymmetry as measurement rather
than argument, and belongs in the critical-review chapter.

---

## L3 — Firm capacity is sized on block averages, not instantaneous peak

**Status:** LIVE, mitigated

**What.** With six slices, the firm-capacity constraint sizes against the
4-hour peak *block average*, not the instantaneous system peak.

**Magnitude.** Measured within-block peaking factor = 1.3448 (PAR_inst 1.4588
÷ PAR_block 1.0847, unsuppressed 2016). Sizing on the block average alone
would undersize firm capacity by roughly a third.

**Mitigation.** Applied explicitly as `within_block_peaking_factor`, kept
separate from NIRP's 900 MW spinning reserve (§2.3, p.19) so the two
components remain traceable.

**Residual.** A deterministic reserve margin is a proxy for NIRP's
probabilistic criterion (LOLE 100 h/yr in 2024 falling to 24 h/yr by 2035).
Equivalence is asserted, not demonstrated.

---

## L4 — Tier 2 self-generation magnitude uncertain by ~40%

**Status:** LIVE — see `docs/tier2_reconciliation.md` (D26)

**What.** NIRP's ~27 TWh self-generation figure is stipulated, not derived: no
methodology, no SFC assumption, no citation. Routes A and B are the same NIRP
number on different bases; the IEA household route is methodologically
independent but covers households only.

**Open defects in that memo, to correct before submission.** Route C contains a
unit error of 10× (the stated multiplier stack yields 90,081 GWh = 90.1 TWh,
recorded as 8.9 TWh), and the memo conflates λ = 0.542 with λ = 0.5645
(= NIRP's 35,000/62,000 read directly). Route C also multiplies rated genset
capacity by running hours, implying a 100% load factor.

**Direction of bias.** Unresolved pending the corrections above.

**Finding that survives regardless of arm.** NIRP's demand forecast has
self-generation going to zero by 2035 (Table 2). The elimination assumption,
not the 2024 baseline magnitude, is the critical-review finding.

---

## L5 — Storage is structurally dominated at annual resolution

**Status:** FALSIFIED by the test this entry itself nominated. The
original structural claim ("optimal storage is identically zero for
any cost") is WRONG — storage is built, substantially, at every
`f1_tranche`/`conv_none` central-arm solve. What is true, and now the
entry's live content, is the Watch Item below: storage is built for
its firm-capacity credit alone and never dispatched.

**Correction (task_register_amendment).** This entry named its own
test: *"whether [the 2.3 firm-capacity constraint] is sufficient is an
empirical question resolved by the first post-2.3 solve."* That solve
exists and was read: `results/baseline/summary__..._conv_none__hydro_
flat.json` (2026-09-16, central arms, `conv_none`/`hydro_flat`) shows
`final_storage_capacity_mwh: 78,882.13` — not zero, built from 2035
onward, reaching 78,882 MWh by the 2054 horizon end. Its own
`timeseries__..._conv_none__hydro_flat.csv` shows `storage_discharge_
twh_e = 0.0` in EVERY year, 2025 through 2054, while `storage_capacity_
mwh` rises monotonically across the same span. **Both halves are
confirmed directly from the run, not inferred:** the original claim
("optimal storage is identically zero") is false at the CAPACITY level
and true at the DISCHARGE level — a distinction the original entry did
not draw.

**What actually happens.** The algebraic argument in the original
**What** below is correct as far as it goes — `discharge[t] ≤ RTE ×
charge[t]` does make storage's net ENERGY contribution non-positive,
so the LP has no economic reason to ever discharge it, and none of the
78,882 MWh built is ever dispatched. But the entry's conclusion
("optimal storage is identically zero") does not follow, because
storage's OTHER contribution — capacity, via `capacity_credit[
'storage']` in the firm-capacity/reserve-margin constraint — is not
subject to the same identity. The entry's own **Watch Item** already
named the mechanism (`capacity_credit['storage'] = 1.0`, flat,
un-declining): with a flat 1.0, storage is the model's single most
capacity-efficient firm resource, at zero marginal energy cost since
it is never asked to actually deliver, so the LP builds it purely to
satisfy the capacity constraint cheaply, then leaves it idle. **The
Watch Item is not a caveat on a pending result — it IS the mechanism
that explains the confirmed one**, and this entry's own further audit
work (`docs/findings_frozen.md` F-19, cited below) already reached
this same conclusion independently, addressing what the register
itself had not yet updated to say.

**Original claim, preserved for the record, now marked FALSIFIED.**
In the annual builder, `discharge[t] ≤ RTE × charge[t]` with the
balance taking `+discharge − charge` gives storage a net contribution of at
most `(RTE − 1) × charge ≤ 0`. Storage can only subtract energy while costing
capex. ~~Optimal storage is identically zero for any cost~~ — **FALSE, see
above: optimal DISCHARGE is identically zero; optimal CAPACITY is not, and
is substantial (78,882 MWh at the 2054 horizon end under central arms).**
Confirmed independently by `14_run_str1_storage_role.py`, which is suspended
for exactly the discharge-side reason (its own suspension text does not
claim capacity is zero either, on rereading it — it argues storage has "no
valid role," which is a normative claim about value, not the register's
now-falsified factual claim about magnitude).

**Why it matters.** No parameter sweep changes the DISCHARGE-is-zero
identity — that remains structural. But the CAPACITY magnitude is very
much swept by, and sensitive to, `capacity_credit['storage']` — exactly
the register's own Watch Item, now confirmed live rather than pending.
Register **L31** already documents that the peak-adequacy diagnostic
using this same flat multiple is itself flawed — the two entries now
describe the same underlying mechanism from two angles and should be
read together.

**Mitigation attempted, and result.** The 2.3 firm-capacity constraint
gives storage a capacity-value channel — CONFIRMED sufficient, in the
sense that it alone, not any energy-value channel, now explains 100%
of built storage capacity.

**Watch item, promoted to live content, no longer pending.**
`capacity_credit['storage'] = min(1.0, duration/peak_hours) = 1.0` at 4h
duration against a 4h peak block, placing storage *above* gas (0.80).
This flat 1.0 is the entire reason storage is built at all — with no
declining ELCC, storage's marginal firm-capacity contribution never
falls, so the LP has no reason to ever stop building it once it is
cheaper per MW of firm capacity than the alternatives, and no reason to
ever discharge it once built. SOURCE: [PENDING] — this was already the
status before falsification and remains so; falsifying the structural
claim does not source the multiple.

**Citations in `docs/findings_frozen.md`, checked, quoted before and
after — NEITHER NEEDS AMENDMENT.** Both already describe the corrected
mechanism, not the register's now-falsified one:
- F-19 ("Storage magnitude is not a cost optimum"): *"Once gas reaches
  its floor, storage absorbs the entire firm-capacity residual, so its
  quantity is insensitive to further financial variation (F-3). Root
  cause is the flat `capacity_credit['storage'] = 1.0` (register L5):
  with no declining ELCC, storage's marginal firm contribution never
  falls and no interior optimum exists."* — this is exactly this
  amendment's own corrected account, already in `findings_frozen.md`,
  unchanged, needing no edit.
- Scoped-conditions list: *"L5 storage capacity credit is flat 1.0 --
  magnitude is a residual"* — also already consistent; also unchanged.

Both citations were evidently written from a more current understanding
of the mechanism than the register entry itself carried until this
amendment — the register was the stale document here, not the
findings.

---

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L6 — EaaS channel excludes storage by construction

**Status:** LIVE unless C10 is implemented

**What.** `storage_add` is a single regime-agnostic variable charged to the
public budget in every financing regime. The EaaS channel is solar-only.

**Why it matters for the contribution.** The thesis claims EaaS decouples
*solar and storage* financing from the public ceiling. The storage half is not
merely untested — it is structurally impossible in the current formulation.

**Why this misrepresents Nigeria.** DARES funds solar *hybrids* throughout —
Component 1 mini-grids ($410m) are solar-battery; Component 2 standalone
systems are solar-battery by definition. Nigerian C&I EaaS providers finance
batteries on their own balance sheets. In Nigeria storage is *more* privately
financed than solar, not less. The model has this inverted.

**Resolution.** Split `storage_add` into EaaS and public channels with bundled
bankability on delivered energy. If not implemented, the thesis claim must be
narrowed to solar and the exclusion stated as a scope boundary with this
reason, not buried.

---

## L7 — Environment was not reproducible before 2026-08-23

**Status:** RESOLVED, retained for the record

**What.** `environment.yml` omitted `pyyaml` and both solvers (CBC, HiGHS)
while listing `pypsa`, which no module imports. An AST scan established the
true dependency set as five packages: numpy, pandas, pyomo, matplotlib, yaml.

**Why it matters.** Results produced before the Linux migration ran on an
environment that could not be recreated from the repository. Windows Smart App
Control blocking unsigned BLAS binaries forced a WSL2 migration, during which
numpy was briefly and silently upgraded 1.26.4 → 2.2.6.

**Resolution.** Pinned to numpy 1.26.4; `environment_linux.yml` committed;
smoke suite reproduces the Windows result exactly (6 passed, 27 failed on the
`peak_multiple_case` guard).

**Reportable.** One sentence in methods on environment pinning as a
reproducibility control.

---

## Entry template

```
## Lnn — one-line statement

**Status:** LIVE / RESOLVED / PENDING

**What.** Precise statement, with the number or expression at issue.

**Why unresolved.** What was tried or why it is out of scope.

**Direction of bias.** Which way it moves results, and why. If unknown, say so
explicitly — that is itself a weaker position.

**Mitigation.** Sensitivity arm, declared bound, or none.

**What would resolve it.** The data or work required.
```

## L8 — Gas fleet retirement schedule is an analyst estimate

**Status:** LIVE

**What.** Plant commissioning years in
`data/capacity/raw/gas_plant_vintages.csv` are an analyst estimate from
general knowledge of Nigerian project histories, not traced to NERC,
NDPHC, TCN, or company filings. Confidence by capacity: 35.8% high,
47.2% medium, 17.0% low. Largest single weakness is Afam IV-V (726 MW),
where the 1990 value blends a 1970s and a ~2002 vintage that should be
separate rows.

**Why unresolved.** Nigerian commissioning-date records were not
obtainable within the project timeframe. Evidentiary status is identical
to NIRP's stipulated 27 TWh and λ: used because nothing better exists,
carried as a declared assumption rather than a citation. Named
`gas_retire_vintage_*`, never `_sourced`.

**Reconciliation.** The CSV totals 11,361 MW against
`gas_baseline_mw = 10,947`. Resolved by excluding AES_1 (294 MW) and
Gbarain_1 (120 MW), both licensed under NERC 2024 Table 4.3 but absent
from NERC's availability-tracked fleet and from the 2026/Q1 table. Not
by scaling, which would retire fractions of plants that retire as whole
units. A structural guard raises if the surviving total ever ceases to
equal `GAS_BASELINE_MW`.

**Magnitude — EPOCH AND CONFIGURATION STAMPED, see the two warnings
below.**

*Produced under:* `peak_unsuppressed`, `storage_cc_central`, `gas_none`,
`gas_avail_nameplate`. All other keys UNDECLARED at the time of writing.

*Epoch:* PRE-shape-factor-sourcing. Dated by objective band-matching
against `findings_provisional.md` (21.x), **not** from a recorded run
tag — the entry was written without one. Treat the epoch as inferred.

| arm | objective | storage 2045 |
|---|---|---|
| `gas_retire_none` | 19.278 bn | 52,947 MWh |
| `gas_retire_vintage_50` | 21.127 bn | 68,011 MWh |
| `gas_retire_vintage_30` (central) | 21.597 bn | 82,071 MWh |
| `gas_retire_proxy` (SUPERSEDED, retired arm) | 22.873 bn | 96,735 MWh |

**⚠ NOT COMMENSURABLE with the central-configuration lever tables.**
This table runs `gas_none`; the central configuration runs
`gas_unconstrained`. The `none → vintage_30` span here is 2.32 bn
against 2.76 bn in `findings_provisional.md` F-5 and 2.71 bn in the
post-sourcing handover. Only the 2.76 → 2.71 step is the shape-factor
sourcing (~2%); the remainder is the gas-build arm. Comparing this
table's span against either lever table would attribute a configuration
difference to the sourcing event.

**⚠ CORRECTION to the previous summary sentence.** It read: "the full
range across all retirement assumptions is 2.32 bn (12.0%) and 43,788
MWh (82.7%)". Those two figures span DIFFERENT ARM SETS. The 2.32 bn is
`none → vintage_30`, live arms. The 43,788 MWh is `none → proxy`, and
the proxy is a retired arm. Across live arms the storage range is
**29,124 MWh (55.0%)**, not 82.7%.

The arm set above is also stale in both directions: it carries
`gas_retire_proxy`, since retired, and predates `gas_retire_vintage_40`
(21.362 bn at the central configuration, F-20), whose storage value was
never measured at this epoch.

**Against the retired proxy.** The 680 MW/yr proxy overstated system
cost by 1.28 bn (5.9%) and storage by 14,664 MWh (17.9%) against
`vintage_30` at this epoch.

**⚠ NARROWED CLAIM.** This entry previously read "retirement is the
largest single driver of the storage result." That is **withdrawn as a
current claim** on three grounds: it rested on the 82.7% figure
corrected above; L5 declares storage magnitude non-reportable as an
optimisation result, so a driver ranking over it has no standing; and
the shape-factor sourcing acts directly on the channel retirement
operates through — the hydro seasonal correction (dry 0.90 → 1.045)
added firm generation to the dry peak block, which is where the
firm-capacity constraint binds and therefore where retired gas is
replaced. The direction of that interaction is not established and has
not been measured post-sourcing. Retirement remains a LARGE lever on
storage at this epoch; it is not established as the largest, and the
magnitude is not reportable in any case.

**Direction of bias.** Every vintage arm retires substantially less than
the proxy (365 / 292 / 167 MW/yr mean over 2035–54, against 680).
Results reported under the proxy therefore overstate retirement, storage
need and system cost. Under the central arm the remaining uncertainty is
roughly symmetric, bounded by `none` below and the proxy above.

**Mitigation.** Four arms carried as a sensitivity. A flat technical life
is itself wrong for a rehabilitated fleet — the 40-year case retires
Delta in 2025 and Egbin in 2026, both operating — so 30 is central on the
grounds that rehabilitation (extending life) and under-utilisation (also
extending calendar life) partly cancel, not on the grounds that 30 years
is correct.

**Open anomaly.** `gas_retire_vintage_50` binds in 23 of 30 years against
21 for both neighbouring arms, despite retiring less gas. Probably a
timing effect — the 50-year schedule retires Sapele Steam in 2030 and
Delta in 2035, early and lumpy, where the 30-year schedule retires
nothing until 2031. Unconfirmed, and measured at the pre-sourcing epoch.

**What would resolve it.** Commissioning dates from NERC quarterly
reports (best — same source as Table 4.3, so bases would match), NDPHC
NIPP completion records, or filings from Geregu Power Plc, Transcorp,
Sahara/KEPCO, Azura-Edo, NAOC. The CSV is structured for partial
replacement.

**Process note.** This entry was written without a run tag or epoch
stamp, which is what made it undatable by anything but band-matching.
RULE: every magnitude table entering this register carries the arm set
it was produced under and the epoch, at the moment it is written.

## L9 — `reserve_margin` and `genset_case` are canonical constants, not swept arms

**Status:** LIVE

**What.** `reserve_margin="nirp_spinning"` is hardcoded identically in all
11 runners that set it; `genset_case=CANONICAL_GENSET` in ~20, with
deliberate exceptions for the VoLL path (`22`, `23` pass `None`) and the
genset sensitivity arms (`24`, `25`, `99c`). `peak_multiple_case` is
aliased to `--peak-case` rather than exposed separately. Neither
`reserve_margin` nor `genset_case` appears in any output tag.

**Why unresolved.** Both are required no-default keys in
`load_scenario()`, but the requirement is satisfied at the call site with
a constant rather than a CLI arm. Changing this moves every output
filename, breaking the provenance chain on results already verified
against reference values. It spans 20+ files. Deferred as a phase, not a
fix.

**Direction of bias.** None. Every result to date used the canonical
value; there is no incorrect number. The gap is reproducibility, not
correctness: a run under a different reserve margin or genset arm would
produce output indistinguishable from the canonical one.

**Mitigation.** The canonical values are stated here and in methods.
`99c_run_genset_arm_sensitivity` does sweep `genset_case` explicitly, so
that dimension is not untested — only untagged elsewhere.

**What would resolve it.** Promote both to named constants with a single
definition site and append them to the output tag; or declare them fixed
by design and document the values in methods.

---

## L10 — Ten runners are dead code predating F1

**Status:** CLOSED (TASK 28 Part E.1). The two items this status called live are both closed: the `09b` pair were moved to `scripts/retired/` ("RETIRED (Tier work A, step 1)" -- the mode axis they swept no longer has two live options) and `17_run_monte_carlo.py` now imports argparse and the standard-arm helper, threads the required keys, builds with `build_model_sliced`, and ran as Phase B item 18 (2026-10-03, 500 solves, 0 infeasible; `logs/phase_b/18_A_monte_carlo.log`). All nine items are now repaired or retired; the entry is historical. (The amendments below are the record of when each was repaired.)

**Original status line, SUPERSEDED (kept for the record; TASK 28 Part E.1, 2026-10-04):**
~~**Status:** LIVE for two of the nine items below; historical for the~~
~~other seven (see amendments).~~

**task_audit_fixes_p2 (audit finding A23) amendment, added without
renumbering this entry.** Five of the nine items in **What** below have
since been repaired: `05_run_fin3_capital_sweep.py`,
`15_run_pol1_ndc_comparison.py`, `18_run_pareto_frontier.py`,
`19_run_carbon_price_sweep.py`, and `20_run_cf1_solar_yield_sensitivity.py`
now import `argparse` and thread the full standard-arm surface via
`src/cli_arms.py` (`add_standard_arms` / `standard_arms_kwargs`), the
same helper `01b_run_baseline_sliced.py` established. Repaired in commit
`ecb7e54` ("Repair nine directly-broken runners: add src/cli_arms.py
shared arm-flag helper; thread required keys through all nine; promote
om_case from literal to flag"); `18_run_pareto_frontier.py` received a
further commit, `f4dd48d` ("Port both frontiers to build_model_sliced;
fix three sliced-unsafe indexing sites in 18_; add required
--baseline-diagnostics with arm_tag validation"), needed to make it work
against the sliced builder specifically. For these five, everything
below this amendment is now historical — it describes a state that no
longer holds for them.

**task_register_amendment, second amendment, extending the one above
without rewriting it.** Two more of the nine items are now also
repaired: `06_run_gas1_shadow_benchmarks.py` and `07_run_gas2_eaas_gas_
relief.py` both now `import argparse` and call `add_standard_arms`/
`standard_arms_kwargs`, verified directly against the current files.
Repaired in commit **`0012f16`** ("Revive nine dead runners: 02_run_
eaas, 06_, 07_ (+A37 and a second TypeError bug), 08_gas3, 09_, 10_,
11_, 13_ (A55 closed), probe_2_6_structure; all route through the
annual builder and must run conv_none"). For these two, the entry below
is now equally historical. **The remaining two items — `09b` (both),
`17` — were not touched by either commit; the entry below remains live
and accurate for those two only.**

**What.** `05`, `06`, `07`, `09b` (both), `15`, `17`, `18`, `19`, `20`
call `load_scenario()` without any of the eight required keys introduced
in phases 2.4–2.13 (`peak_multiple_case`, `reserve_margin`,
`storage_cc_case`, `gas_build_case`, `gas_availability_case`,
`gas_retirement_case`, `gas_capex_case`, `ccgt_efficiency_case`). None
imports argparse. Each raises `TypeError` on the first missing argument.

**Why unresolved.** Wiring them requires replicating the full per-runner
scaffold from five earlier phases across ten files, not adding the five
F1 keys. Out of scope at three weeks to submission.

**Direction of bias.** None on existing results — these runners have not
produced output since the phases that broke them. The cost is
capability, not correctness.

**Consequence for planned analysis.** `17_run_monte_carlo`,
`18_run_pareto_frontier` and `19_run_carbon_price_sweep` are on the
planned-outputs list. Each must be wired before those analyses can run,
or the analysis dropped.

**What would resolve it.** Full argparse scaffold + `load_scenario()`
threading + output-tag wiring per runner, following
`scripts/01b_run_baseline_sliced.py` as reference.
---

## L11 — `storage_binding` diagnostic is not meaningful under the sliced builder

**Status:** LIVE

**What.** `extract_planning_diagnostics`'s `storage_binding` classifier
tests `power_bind` as `discharge == storage_power_mw *
storage_deployable_hours_per_year / 1e6`. The sliced builder imposes no
such constraint — throughput falls out of per-slice charge/discharge
power limits plus a per-season energy balance. The test therefore
compares annual discharge against a quantity the model never
constrained.

**Why unresolved.** A slice-native storage-binding diagnostic would have
to be re-derived from the actual per-slice constraints. Out of scope at
the port stage.

**Direction of bias.** None on any reported result — no runner consumes
`storage_power_limit_years`. The field is serialised into
`diagnostics.json` by 02 and 03 and read by nothing.

**Mitigation.** Field is inert. Do not report it.

**What would resolve it.** Re-derive the classifier from the sliced
builder's own per-slice power and per-season energy constraints.

---

## L12 — The annual baseline has no recorded reference objective

**Status:** LIVE

**What.** `01b_run_baseline_sliced` has a documented reference (21.4748
bn, 44,377 MWh) used as the regression instrument throughout the F1/F2
corrections. `01_run_baseline` (annual) has none. When
`extract_planning_diagnostics` was made builder-aware, the annual path's
byte-identity could only be argued from the code (the `_slice_total`
helper reduces to `pyo.value(comp[t])` under annual indexing,
grep-verified), not verified against a number.

**Why unresolved.** `results/` is gitignored, so no pre-change annual
result is recoverable.

**Direction of bias.** Unknown. The code argument is sound but is not a
measurement.

**Mitigation, superseded figure preserved, current figure recorded
(task_register_amendment).** ~~Post-change annual baseline recorded at
21.3748 bn / 63,055 MWh as the reference from this point forward.~~
**That figure is itself now superseded — read against the current
tree, it matches neither the bare `results/baseline/summary.json`
(16,926,222,807.78 USD / 0.0 MWh storage, 2026-08-23) nor any later
tagged run, most likely because O&M and other corrections landed after
it was recorded. Current reference, verified directly: `results/
baseline/summary__peak_unsuppressed__storage_cc_central__gas_
unconstrained__constrained__gas_avail_nameplate__gas_retire_vintage_
30__gas_capex_central__ccgt_eff_central__cr_central__dscr_central__f1_
tranche__env_dares__baseline__conv_none__hydro_flat.json` (2026-09-16),
central arms, `conv_none`: `npv_total_cost_usd = 32,448,782,287.34`,
`final_storage_capacity_mwh = 78,882.13`. This is the CENTRAL-arm-set,
`conv_none` annual reference — a different arm combination from, and
not to be confused with, the project's frozen REGRESSION anchor
(21,811,042,472.821114, `om_none`).**

**Caution, added per instruction — the F10 bare-path hazard, now
demonstrated with real, disagreeing numbers rather than argued
abstractly.** `results/baseline/summary.json` and `diagnostics.json`
remain frozen at 2026-08-23 (pre-O&M, pre-storage-fix: 16.93bn NPV, 0
MWh terminal storage) and are NOT updated by any tagged run — the
tagged-output convention writes alongside the bare path, never over
it. Anything reading the bare path gets a third, oldest, and now
demonstrably wrong state; anyone citing "the annual baseline" must name
which of the (at least) three disagreeing figures they mean.

**What would resolve it.** Nothing retrospectively. Forward: treat the
recorded value above as the annual regression anchor for the CURRENT
central/`conv_none` configuration, re-recording it again whenever a
future correction moves the objective, and never read the bare
`results/baseline/summary.json` path as current.

---

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L13 — Gas deliverability arms span two epistemic families

**Status:** LIVE

**What.** `gas_deliverability_case` now offers two families that are not
comparable: historical-drift extrapolation (`downside`, `baseline`,
`upside`, `shock_recovery`, plus `flat_*` level-equivalents) and GMP 2026
policy-target discounting (`gas_deliv_flat`, `gas_deliv_gmp_partial`,
`gas_deliv_gmp_full`). The GMP family reaches **140.5-191.7** TWh_th by
2030 (`gas_deliv_gmp_partial` to `gas_deliv_gmp_full`); the entire drift
family spans 81.5-94.7.

**Correction (task_audit_fixes_p3, A29).** This entry previously quoted
the GMP family's lower bound as 114.9, not 140.5. Recomputed directly
from `build_series()` (`scripts/build_gas_deliverability.py`) against
the current `data/gas/processed/gas_deliverability_cases.yaml`:
`gas_deliv_gmp_partial` (I=0.5, the family's CENTRAL arm) is 140.51
TWh_th at 2030, not 114.9 — the entry's argument was always right, the
one number was not. The upper bound (191.7, `gas_deliv_gmp_full`) and
the drift family's range (81.5-94.7) were both independently
re-verified correct and are unchanged. Separately verified: this
generator reproduces its own checked-in CSV (`data/gas/processed/
gas_available_power_annual_twh_th.csv`) byte-for-byte — all 300 rows
(10 scenarios × 30 years), zero discrepancies, re-derived directly from
the current script and YAML, not merely inspected. `build_gas_
deliverability.py` itself was not the source of the error.

**Why unresolved.** Both are legitimate. Drift arms carry the only
downside and shock-recovery shapes, and GAS-3's `flat_*` controls depend
on them. GMP arms carry the policy-target case.

**Direction of bias.** None if labelled. A result mixing the families
without stating which it came from would be uninterpretable.

**Mitigation.** Family stated in the YAML header and in the CLI help
text. Never present a cross-family comparison unlabelled.

---

## L14 — `_rebuild_years` is duplicated across the F1 branch

**Status:** LIVE

**What.** `_rebuild_years(build_t, life)` exists twice in each builder:
once inside the `pre_f1_overnight` block, once inside `f1_tranche`.
Python scoping prevents the `else:` branch reaching the `if:` branch's
locals, and hoisting would edit the regression instrument.

**Amendment (L18, D6 closure).** A fifth copy of this arithmetic now
exists: `solve_model()`'s (`src/optimize_model.py`) post-solve rebuild-
invariant guard replicates `_rebuild_years`'s logic inline, since the
closure itself is local to the builders and unreachable from
`solve_model()`. Comment there names all four builder-side copies to
keep in sync with. Same risk as below, one more site.

**Direction of bias.** None currently — the copies are verbatim
identical. Risk is future divergence: an edit to one will not propagate.

**Mitigation.** Keep-in-sync comment at both (now five) sites.

---

## L15 — Stale "2045" naming across four sites

**Status:** RESOLVED (`9605c82`).

**Verified (task_register_amendment).** All four sites named below are
now fixed: the variable is `slice_terminal_year` (was `slice_2045`,
`:486`); the surrounding comments document the rename and state both
years are now read dynamically (`:482-483`, `:636-639`); the
`diagnostics.json` key is `slice_detail_terminal_year` (was `slice_
detail_2045`, `:600`), with the schema change confirmed checked against
consumers before renaming; and the exported per-slice CSV row label
uses `str(years[-1])` (was the hardcoded literal `"2045"`, `:641`).
Commit `9605c82` ("L15: terminal-year naming read dynamically;
diagnostics key slice_detail_2045 -> slice_detail_terminal_year
(output-schema change)").

**Original entry, preserved for the record.**

**What.** `01b_run_baseline_sliced.py` names the terminal-year slice
`slice_2045`, comments it as 2045, writes it to `diagnostics.json` under
the key `slice_detail_2045`, and tags the exported CSV
`slice_detail__{tag}.csv` with the literal string "2045". All four hold
**2054** data — the naming predates the horizon extension (74d34f2).

**Why unresolved.** Fixing the JSON key changes the output schema beyond
the scope of the commit that found it.

**Direction of bias.** Any figure read from those files or that key is
mislabelled by nine years. **No longer applies — all four sites now
read/label correctly, confirmed above.**

**Mitigation.** The printed summary label was fixed to read the terminal
year dynamically. The four remaining sites are declared here. **All
four are now fixed; see the verification above.**

---

## L16 — EaaS and public solar are physically identical in the model

**Status:** LIVE — declared scope boundary, not an unfinished task

**What.** The two solar channels differ only in financing. They share
`m.solar_capex_param[t]`, the same capacity factor, the same CRF path,
and enter `m.solar_capacity_mw` additively and symmetrically
(`optimize_model_sliced.py:705-710`). Two sourced corrections that
would distinguish them are unimplemented:

- **[DIST-2] Loss advantage.** Behind-the-meter generation avoids the
  7% TLF and distribution losses. Favours EaaS.
- **[DIST-5] Distributed premium.** NREL ATB Distributed Commercial PV
  carries a CAPEX premium of +16% (2025) rising to +37% (2045), and a
  distributed capacity factor differing from utility-scale 0.20.
  Disfavours EaaS.

**Why unresolved.** Out of scope, deliberately. The thesis's EaaS claim
concerns FINANCING STRUCTURE, not distributed-versus-utility-scale
physics; siting is not in the research question. DIST-2 is a
distribution-loss effect, and this model's metering point is declared
at the busbar specifically so distribution falls outside the boundary —
implementing it would reach back across a boundary drawn on purpose.
Both are sourced from the ATB workbook and implementable; the
constraint is scope, not data.

**Direction of bias.** The two corrections OPPOSE each other, so the
net direction is not established. Implementing either alone would bias
in a known direction; that is why they must be done together or not at
all. Their joint absence makes the two channels harder to distinguish,
which makes degeneracy in the public/EaaS split MORE likely — a
scoped condition on F3.

**What the model therefore cannot say.** Nothing about WHERE solar is
sited, only about how it is paid for. Any result comparing EaaS to
public solar is a financing comparison, not a siting one.

**What would resolve it.** Implement both together: a distributed
capex multiplier on `solar_eaas_add`'s capex path, a separate
`solar_cf` for the EaaS channel, and a TLF credit on EaaS generation.
Any F3 or EaaS-split result would then need re-running — not because
the test is invalid, but because the interval depends on how
distinguishable the channels are.

---

## L17 — The bankability constraint discounts at the retired pre-F1 blend, over a horizon the objective contradicts

**Status:** RESOLVED IN PART (rate-coherence task, Part 1, declared
2026-09-25) — see the update at the end of this entry. First half
(rate AND horizon incoherence, for every live path except FIN-2)
CLOSED; second half (`blended_finance_scenarios()`'s `r_commercial` =
0.18, naira-basis, now reaching FIN-2 only) OPEN.

**What.** `m.remaining_npv_factor[t]`
(`src/optimize_model_sliced.py:439`, `src/optimize_model.py:352`) is the
annuity factor inside `m.eaas_bankability` and `m.eaas_subsidy_limit`.
It is built as

    sum(1 / (1 + r_private)^(k - t) for k in range(t, len(T)))
    r_private = resolve_private_rate(scenario)

with two incoherences against F1:

- **Rate.** `resolve_private_rate()` returns the PRE-F1 exogenous
  blended WACC. Under the central configuration
  (`blend_case = blended_central`, the default, which no runner
  overrides) that is `0.30 x 0.06 + 0.70 x 0.18 = 0.144`. Seventy per
  cent of it is the 18% commercial anchor F1 retired from the objective
  as naira-basis and dimensionally incoherent against real-USD
  cashflows. The F0 audit's headline was that `blended_wacc` never
  reached the objective; F1 replaced it there, but it still reaches
  this constraint.
- **Horizon.** It discounts to the MODEL HORIZON (30 years). The F1
  capital charge amortises commercial capital over `L_comm` = 10 years,
  SOURCED from Afrigreen's 8-10 year Nigerian C&I lending and Nigerian
  bank tenors of 3-7 years absent credit enhancement. The same asset
  therefore carries a 10-year tenor in the capital charge and a 30-year
  tenor in the admissibility test.

**Why unresolved.** The correct treatment — discount at `r_comm` real
over `min(L_comm, T_last - t + 1)`, keeping the year-t convention
because bankability asks whether an investor AT VINTAGE t recovers
capex — requires a new required scenario key threaded through 22
`load_scenario()` call sites. An attempt consumed over an hour of agent
time across sixteen files and produced no report; the working tree was
discarded (`git stash`, "3.4 attempt, unverified"). At the measured
magnitude below, that cost is not justified before submission.

**Direction of bias.** The two errors PARTLY CANCEL at early vintages
and diverge thereafter.

At t = 0: the legacy rate is HIGHER (14.4% vs 8.82% real) while the
legacy horizon is LONGER (30 vs 10 years). Annuity factors 6.822
(legacy) against 6.469 (correct) — bankable revenue is OVERSTATED by
about 5%.

The offset does not hold away from t = 0, and breaks down entirely once
truncation would bite from 2046 (a 2050 vintage has a 5-year window
under the correct rule against 5 years of remaining horizon at 14.4%
under the legacy rule). So the residual bias is toward OVER-CREDITING
LATE-VINTAGE EaaS REVENUE, meaning bankability binds less often than it
should and EaaS deployment is permitted more freely late in the horizon
than a commercial lender would allow.

Two errors partly cancelling is the same pattern as the Phase 1
calibration finding — undetectable from internal consistency alone, and
found only by checking each input against its own basis.

**What it affects.** Within the EaaS channel, everything:
`m.eaas_bankability`, `m.eaas_subsidy_limit`, and the
`bankability_threshold` / `bankable_capex_per_mw` diagnostics in
`extract_planning_diagnostics`. Note the subsidy-limit coupling runs the
other way: a lower factor would REDUCE `max_bankable_at_t`, widening the
subsidy gap and permitting more subsidy, which draws on the public
budget. So a fix would have two opposing effects on EaaS deployment.

Outside EaaS, nothing. Gas, storage, hydro, conversion, the
firm-capacity constraint, the F1 capital charge and the objective do not
read this factor.

**Consequence for reported findings.** F-26 through F-29 (numbers of `findings_provisional_pre_sweep.md`, which is superseded and not to be quoted; `findings_frozen.md` has no F-23 to F-29) all rest on the
bankability constraint and are therefore conditional on this entry.
**F3's identification ceiling is NOT set by this factor** (methodology
claims ledger C-2, corrected in F-12/methodology.md §7.7): the EaaS
maximum equals total solar (`solar_public_add_mw_total` = 0.0 at the
maximising solve, `results/f3/*/summary.json`), is identical across
tariff arms, and rises with the cost tolerance — properties a
bankability ceiling could not have. This entry's core point stands
regardless: the bankability constraint itself discounts at the retired
pre-F1 blend over an incoherent horizon, and F-26 through F-29 (provisional-file numbers, see the first mention in this entry) remain
conditional on it for that reason, independent of whether it happens to
set F3's ceiling.

**Known second instance, same defect.**
`run_fin2_blended_bankability_sweep`
(`src/optimize_experiments.py:853, 887, 907-908, 981`) discounts at the
same pre-F1 blend in both its analytical T* and its solves. Its runner
`04_run_fin2_tariff_sweep.py` is dead on unrelated required args.

**What would resolve it.** A `bankability_tenor_case` key with
`bank_tenor_legacy` (the current form, as a regression arm) and
`bank_tenor_comm` (r_comm real over `min(L_comm, T_last - t + 1)`),
threaded through `load_scenario()`. Builders-only with a default is the
cheaper route and was rejected here on the no-silent-defaults rule.
Step-0 inventory is complete and preserved in
`docs/bankability_tenor.md` — a future attempt starts from there, not
from scratch.

**Process note (appended to L17).** Making a scenario key REQUIRED breaks
every `load_scenario()` caller, not only those that consume the key
meaningfully. The `tariff_case` migration analysed liveness for
TARIFF-SETTING runners and wired the two that were live, but fifteen
further runners call `load_scenario()` without setting a tariff and
began raising — five of them live, including `01b`, the primary sliced
runner and the source of the regression anchor. Repaired in
`docs/tariff_arm.md` Step 4.

RULE: any future required-key addition must be wired to EVERY
`load_scenario()` caller, not only to consumers of the key. The liveness
question is "does it call load_scenario", not "does it use this
parameter".

**UPDATE (rate-coherence task, Part 1, 2026-09-25).** Implemented
almost exactly as "What would resolve it" specified above: a
`bank_tenor_case` key (`BANK_TENOR_CASE_CHOICES`, `src/scenarios.py`),
dispatched through `resolve_remaining_npv_factor()`
(`src/blended_finance.py`), called identically at the single
unconditional `m.remaining_npv_factor` build site in both builders
(`src/optimize_model.py:345-351`, `src/optimize_model_sliced.py:467-475`).

- `bank_tenor_comm` [NEW DEFAULT]: $F_t = \sum_{j=0}^{L^{eff}-1}(1+r^{comm,real})^{-j}$,
  $L^{eff}=\min(L_{comm}, T_{last}-t+1)$ — the objective's OWN
  commercial tranche (`r_comm_nominal`/`L_comm`,
  `capital_recovery_scenarios()`), deflated independently at the build
  site. This is the "correct treatment" this entry originally
  specified, applied. At $t=0$: $F_0 = 7.038499$ (real rate 8.824%,
  $L^{eff}=10$) against the legacy $F_0 = 8.349709$ (`blended_central`,
  13.24% — see below) — ratio 0.842963, close to but not identical to
  this entry's earlier "6.469 (correct)" estimate, which was a rough
  figure from an abandoned attempt, not this implementation.
- `bank_tenor_legacy`: unchanged in form —
  `sum(1/(1+r_private)^(k-t) for k in range(t, len(T)))`,
  `r_private = resolve_private_rate(scenario)` — retained so every
  pre-correction result stays attributable, NOT the default any more.
  **Note this entry's own worked example above used `r_private = 14.4%`**
  (the pre-L53 `blended_central`, which itself used the since-retired
  `r_concessional = 0.06`); at HEAD, `blended_central` is 13.24% (register
  L53 / `docs/task_log_elasticity_sourcing.md`), so a fresh run under
  `bank_tenor_legacy` today reproduces 13.24%-based figures, not the
  14.4% quoted above — the entry's original worked numbers are
  historical, from before that unrelated correction landed.
- **FIN-2** (`run_fin2_blended_bankability_sweep`,
  `src/optimize_experiments.py`) is explicitly PINNED to
  `bank_tenor_legacy` — its own analytics
  (`blended_wacc(share, r_conc, r_comm)`) are not rebuilt in this task,
  and letting its LP silently pick up `bank_tenor_comm` would have
  made its analytics and its solves price bankability on two different
  bases. This is a declared scope condition, not an oversight — see
  `docs/task_log_rate_coherence.md`.
- **Consequence for `03_run_ndc_eaas.py` and `26_run_f3_identification.py`**:
  neither passes `bank_tenor_case` explicitly, so both silently adopt
  `bank_tenor_comm` on their next run. Both are bankability-priced and
  both are named above as resting on this defect (F-26 through F-29 (provisional-file numbers, see the first mention in this entry);
  F3's ceiling, per this entry's own correction, is NOT set by this
  factor, so F3's headline conclusion is unaffected in kind, though its
  reported MW figure will still move because the underlying solve
  changes). **This movement is EXPECTED, not a new defect** — it is
  this entry closing, working as intended. Register **L57** records the
  `bank_tenor_case` arm itself in full; **L58** records the anchor
  values already observed to have moved for an unrelated reason
  (`organic_central`, L54) and flags that Part 2's concessional-rate
  correction will move the central anchor again.
- **What remains open**: `blended_finance_scenarios()`'s
  `r_commercial = 0.18` is still naira-basis (unresolved, `src/blended_finance.py:50-53`,
  `scripts/rpublic_diagnostic.py:74,168`) and, after this task, reaches
  exactly one live consumer — FIN-2, via the pin above. This is the
  "remaining half" this entry's status line refers to; it needs its
  own decision and its own FIN-2 re-run, not made here.

---

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L18 — D6 (buffer-year storage replacement capex) is CLOSED; its guard message was stale

**Status:** RESOLVED, retained for the record

**What.** `merged_correction_plan_v2.md`'s D6 (`[OPEN]` `[SHOULD]`) read:
"Storage life (15 yr) is shorter than the 30-year model horizon, so
buffer-year vintages retire in-model, are still dispatched, and are
never charged replacement capex." That was true when written (plan 2.5,
before the rebuild mechanism existed) but has been false since Phase 2.6
for the `pre_f1_overnight` branch (`m.cost_replacement_capex_npv`,
entering `system_cost_npv`) and since F2 for the `f1_tranche` branch
(`m.cost_rebuild_capital_charge_npv`, entering
`m.capital_charge_total_npv`). Both sum a rebuild at every `build_t +
k*life <= T_last` via the builder's `_rebuild_years()` closure, for
solar, storage, and gas alike (not storage-only, despite D6's title).
The post-solve guard in `solve_model()` (`src/optimize_model.py`, "plan
2.5") kept firing on every run — storage life 15 yr is shorter than
every horizon this project uses — with a message asserting the ORIGINAL,
now-false claim ("No replacement capex is charged"). A guard that fires
on every run with a message the run itself contradicts trains its
reader to ignore it.

**Why it was closed, not just re-worded.** Worked through the arithmetic
(`docs/d6_storage_guard.md`, step 0c): the guard's fire condition,
`(T_last - build_t + 1) > life`, is algebraically identical to
`_rebuild_years(build_t, life)` yielding at least one in-horizon rebuild
year, for every lifetime value this project actually supplies anywhere
(`asset_lifetimes()`, `asset_lifetime_sweep()`, and
`22_salvage_sensitivity.py`'s arms are all whole-number years — checked,
not assumed). So the guard's old condition was not merely mis-worded, it
was TESTING FOR an event (uncompensated retirement) that cannot occur
under any scenario this codebase can build. No vintage was found where
the guard's condition holds and no rebuild is generated.

**Resolution.** The guard now tests that invariant directly — for each
channel/vintage where the old condition holds, it replicates
`_rebuild_years`'s own arithmetic inline (that closure is local to
`build_model()`/`build_model_sliced()` and not reachable from
`solve_model()`) and asserts the resulting rebuild list is non-empty.
When non-empty (every case this project can produce), it prints an
INFORMATIONAL note naming the rebuild year(s) and which cost expression
charges them — not a warning, since this is now correctly-handled,
expected behaviour. If the list were ever empty (not reachable today,
but genuinely tested rather than assumed away, in case a future
non-integer lifetime or horizon change opens the gap identified in step
0c's caveat), it raises `RuntimeError` rather than printing a swallowed
warning — an actually-uncompensated retirement is a real defect and
should stop a run, not scroll past in the log.

**Direction of bias.** None. This closes a documentation/reporting
defect (a stale warning), not a modelling one — no constraint, variable,
or cost expression was touched. The underlying rebuild mechanism this
entry is about has been charging correctly since 2.6/F2; only the guard
that describes it was wrong.

**Also corrected:** TASK.md named `docs/phase_2_8_conversion.md` as
carrying the note that the guard "still fires under `f1_tranche`" without
recording that its message was wrong — that file has no such note
(checked: no match for "plan 2.5", "retirement", "storage life",
"replacement capex", "fires", or "WARNING" anywhere in it). The actual
note is in `docs/f1_implementation.md`, two places: the STEP 0b
observation ("Whether it still fires under F1 ... is deferred") and the
dedicated "## Storage post-solve guard under f1_tranche" section ("The
guard still fires under `f1_tranche`, unchanged ... Not fixed, per the
task's instruction to report only") — true as far as it went, but it did
not record that the guard's MESSAGE was wrong, only that it fired. Both
corrected in that file.

**Related, still open:** L14 (`_rebuild_years` duplicated across the F1
branch) — the guard rewrite here added a fifth copy of that helper's
arithmetic (replicated inline in `solve_model()`, since the closure
itself is unreachable from there); L14 amended to note it.

**What would resolve it further.** Nothing required — CLOSED. If
`_rebuild_years` is ever promoted out of its four builder-local closures
into a shared, importable function (the resolution L14 names), the
inline replica in `solve_model()` could be replaced with a direct call,
removing the "kept in sync" burden.

---

## L19 — `26_run_f3_identification.py` sets two required keys internally

**Status:** LIVE — small, fix before submission

**What.** The runner does not expose `--dist5-case` or
`--solar-cf-case` despite both the DIST-5 and CF-band tasks reporting
that it was wired. It resolves both internally.

**Why it matters.** This is the silent-default class that produced the
65M tariff defect: a load-bearing parameter present in every run,
absent from every output tag, and unselectable. The F3 identification
results therefore cannot be swept across the CF band or DIST-5 from the
command line, and their arm tags do not record which values were used.

**Direction of bias.** None on the reported F3 results — the internal
values match the central configuration. The exposure is
reproducibility.

**What would resolve it.** Add both flags following the `--tariff-case`
pattern, required with choices derived live from the registry, appended
to the summary dict.

## L20 — hydro_min_flow_fraction is ASSUMED, binding, and immaterial

**Status:** DECLARED, measured, closed.

**What.** `hydro_min_flow_fraction = 0.50` floors per-slice hydro
dispatch at 50% of an hours-proportional share of that season's energy
budget. Classification ASSUMED, not SOURCED.

**Why not sourced.** A documented search found no Nigerian minimum-flow
obligation expressed as a fraction. Nigerian dam-operation sources give
ABSOLUTE DISCHARGE RATES. The nearest analogue is JICA, Nigeria National
Water Resources Master Plan, Vol. 5, SR6, p. SR6-7, Table SR6-9,
"Assumed Operation Rules for Release from Shiroro Dam" — 315/100/50 m3/s
at reservoir storage 20%/10%/below 10%. JICA labels these ASSUMED
operating rules, not a mandated obligation.

Not convertible into this parameter, for two independent reasons: it is
an absolute discharge at one plant against a fleet-wide energy-share
fraction; and it is storage-state dependent, while this model carries no
reservoir state variable (verified — the only hydro decision variable is
m.hydro_gen[t,(s,p)], and all three hydro constraints are
contemporaneous).

CORRECTION recorded so it is not repeated: turbine discharge at 95%
reliability (Kainji 440, Jebba 500, Shiroro 120 m3/s) is PLANT
CAPABILITY, not release obligation. An earlier reading of the 500 m3/s
figure as a minimum-flow obligation was incorrect.

**The floor BINDS.** Verified by arithmetic against
results/baseline_sliced/slice_detail__02c17f55.csv, independent of any
solve: wet_night sits exactly on its floor in 2025 (0.8060181 both
sides); both day slices sit exactly on theirs in the terminal year
(dry_day 1.6095677, wet_day 1.4968907). The binding slice migrates from
night toward day as solar builds out, because solar displaces hydro's
daytime value and the LP would otherwise strip day slices further.

**And it is IMMATERIAL.** scripts/27_run_hydro_minflow_sensitivity.py,
central configuration, three solves at 0.30 / 0.50 / 0.70:

| arm | objective | storage 2045 |
|---|---|---|
| 0.30 | 21.337190 bn | identical |
| 0.50 (central) | 21.351610 bn | identical |
| 0.70 | 21.367301 bn | identical |

EPOCH: these three objectives were measured PRE-O&M (om_none-equivalent,
before om_case existed). The band and the zero-storage response are the
findings, not the levels; both are expected to hold at om_central since
the min-flow floor and the O&M term act on different constraints. Not
re-measured — the verdict was IMMATERIAL and re-running to confirm an
immaterial result is not a good use of the remaining budget.

Objective band 30.1M USD = 0.141% of central. Storage at 2045 moves
0.000%. Materiality thresholds (0.5% objective, 2% storage) were FIXED
BEFORE any solve ran. Response is monotonic and mildly convex.

**Mechanism for the zero storage response.** Storage is sized by the
firm-capacity constraint, which reads hydro's MW capacity credit, not
its per-slice energy allocation. The floor redistributes energy within a
season whose total is exogenously fixed, so it cannot alter hydro's firm
capacity, the firm-capacity residual, or storage. Zero is the predicted
result, not a coincidence.

**Direction of bias.** A lower floor would reduce system cost slightly
by permitting more hydro concentration into high-value slices. The
central 0.50 is therefore mildly conservative on cost. Bounded by the
measured 0.141% band.

**Finding worth reporting.** A constraint can bind exactly, in multiple
slices, across the horizon, and still carry negligible shadow value.
Binding and material are distinct properties.

**CONSEQUENCE FOR THE SLICED RESULTS.** This was the last shape
parameter carrying [SOURCE NEEDED]. findings_provisional.md's standing
condition — "NO SLICED RESULT IS FINAL until they are sourced" — is
LIFTED: four parameters are sourced and this one is declared ASSUMED
with its immateriality measured.

---

> **Generation (TASK 86):** the hash-keyed result files this entry cites are blended dispatch (no `gas_vintage_dispatch=True` in the cited files' arm records). Checked: `02c17f55` (blended). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142. [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L21 — storage_duration_hours is an explicit constant, ASSUMED, and unswept

**Status:** LIVE

**What.** `storage_duration_hours = 4.0` (`src/scenarios.py:2666`,
comment: "Storage discharge duration (hours)") sets the storage
capacity credit's ceiling and is used to derive storage power capacity
from storage energy capacity in both LP builders
(`optimize_model.py:386`, `optimize_model_sliced.py:461`, both
currently reached via `scenario.get("storage_duration_hours", 4.0)`).
It is not exposed on the scenario dict as a required, no-default key —
it is set once inside `load_scenario()` and surfaced on the returned
dict (`scenarios.py:3021`) — and it does not appear in any arm tag or
output filename.

**Classification: ASSUMED.** Determined from the code's own commentary
at the definition site: unlike `_hydro_baseline_twh` and
`_hydro_capacity_mw` two lines above it, which carry `# SOURCE: see
dict entry below` comments pointing to a citation, `_storage_duration_hours`
carries no source citation — only a note that it was "previously an
IMPLICIT default... made explicit here" to keep the builders' fallback
and the capacity-credit calculation from silently drifting apart. No
derivation from another sourced quantity is shown either, so it is
neither SOURCED nor DERIVED by the register's own definitions.

**Why unresolved.** A fresh inventory for this task found the key
threaded through 25 distinct files and 36 `load_scenario()` call sites
— more than a prior task's estimate of 22 — so promoting it to a
required, explicitly-threaded key (same treatment already given to
`tariff_case`, `hydro_case`, and 18 others) was not completed in this
task; see Part A of `docs/task_bucketA_parity.md`. It remains reached
by `.get()` fallback in both builders.

**What it controls.** The storage energy-to-power ratio used to (a)
cap the capacity credit the LP grants to storage and (b) derive
storage power capacity from storage energy capacity, in both LP
builders.

**Direction of bias.** Unknown/unswept, not one-directional. A shorter
assumed duration (batteries that discharge faster relative to their
energy content) would raise the derived power-to-energy ratio and grant
more capacity credit per MWh of storage, favouring storage as a firm-
capacity substitute; a longer duration would do the reverse. No arm
currently varies it, so which direction (if any) the true value departs
from 4.0 is not established, and no reported result has been checked
against a different value.

**What would resolve it.** Promote it to a required `load_scenario()`
parameter following the `tariff_case` pattern (name, cite why an
unstated value is unsafe, no default), across all 36 call sites
identified in this task — and, separately from that promotion, decide
whether it should also become a swept arm.

---

## L22 — Fixed + variable O&M is now SOURCED and wired; the storage conversion inherits L21's uncertainty

**Status:** LIVE. The measured O&M band is corrected to the post-epoch arms (TASK 70 B.1). *(Prior status, retained as history: LIVE.)*

**Corrected 2026-10-23 (TASK 70 B.1).** The MEASURED BAND below was taken at `ecb7e54` on the pre-epoch configuration (objective 27.954 bn). Post-epoch, from `results/baseline_sliced/summary__792f8dce.json` (`om_low`), `summary__6046f759.json` (`om_central`) and `summary__f6926482.json` (`om_high`): 24.560, 24.987 and 25.612 bn; span 1.052 bn, 4.2% of central; storage 2054 identical at 63,056.31 MWh and `storage_add_mwh_by_year` identical in all three. The claims that survive are unchanged: storage is identical across the arms, the circularity argument against scaling fixed operating cost to Nigerian availability, and the two-signed bias. The pre-epoch figures are struck below and kept as history.

**What.** Both LP builders now charge fixed O&M (FOM, USD/kW-yr or
the converted USD/MWh-yr for storage) on installed capacity and
variable O&M (VOM, USD/MWh) on gas generation, for solar, storage,
gas_ct (unconverted), and gas_cc (OCGT-to-CCGT converted), discounted
alongside every other operating cost. This replaces a prior omission:
no technology previously carried any O&M term in either objective
(`docs/task_storage_om.md`), and a placeholder constant
(`STORAGE_OM_PER_MWH_YR`) had zero readers for the life of the
project.

**Classification: SOURCED**, NREL ATB 2024, `2024_v3_Workbook.xlsx`,
`Summary_FOM` sheet (fixed) and the per-technology sheets (variable),
three cost cases (Advanced / Moderate / Conservative, exposed as
`om_low` / `om_central` / `om_high`), 2025-2050, extracted to
`data/cost/processed/om_cost.csv`. Storage's zero VOM was verified
against the technology sheet directly (`Utility-Scale Battery Storage`,
row 137) rather than the unreliable `Summary_VOM` sheet, which shows a
nonzero figure for storage that does not represent variable O&M (per
TASK.md's own sourcing note — not re-derived in this task).

**Unscaled US new-build figures, applied to a Nigerian fleet operating
at far lower availability — deliberate, not an oversight.** ATB's cost
figures are calibrated to US utility-scale fleets running at US
availability factors; Nigeria's gas fleet in particular operates at a
fraction of that (this model's own `gas_availability_factor` arms are
well below nameplate). A tempting "correction" is to scale FOM down by
the availability shortfall, on the reasoning that a less-available
plant needs less maintenance spend. This model does NOT do that, for
the same reason `docs/tier2_reconciliation.md` and the demand-growth
arms refuse to calibrate an input to a supply-constrained outcome:
Nigerian gas plants likely run below nameplate availability BECAUSE of
chronic O&M underspend relative to their nameplate design, not
instead of it. Scaling FOM down to match observed availability would
treat the underspend as if it were an efficiency the model should
reward, when the underspend is plausibly the CAUSE of the very
availability figure the model already takes as exogenous. That is
circular in exactly the shape the register already declines elsewhere:
an input is not calibrated to a phenomenon the input itself may be
producing.

**Direction of bias.** Two directions, not one:
(a) Against the un-scaled-US-rates choice: if Nigerian O&M cost per
installed kW is genuinely lower than the US ATB figure (labour is
cheaper; some maintenance categories may be deferred rather than
foregone), this model OVERSTATES gas and solar/storage operating
cost in absolute USD terms, and therefore overstates total system
cost and possibly the relative attractiveness of a lower-capacity
pathway.
(b) Against the (correctly, in this model's judgement) un-scaled
availability choice: if the true relationship runs the other way —
Nigerian plants are simply worse-maintained per dollar spent, not
spending less per dollar of value received — this model's cost figure
could still UNDERSTATE the true cost of keeping gas plant available at
Nigerian reliability standards, since ATB's dollars buy US-grade
maintenance outcomes, not Nigerian ones.
Net direction is therefore NOT signed here; both biases point through
the same parameter in opposite directions and neither dominates by
construction.

**The storage conversion inherits L21.** Storage's FOM is converted
from ATB's USD/kW-yr to this model's USD/MWh-yr basis by dividing by
`storage_duration_hours` (= 4.0, canonical `scenarios.py:2666`). L21
classifies that value ASSUMED and unswept. Before this task it set
only a capacity-credit ceiling; it now also scales a cost term. A
shorter true duration than 4.0 would understate storage's true FOM in
USD/MWh-yr terms (the same kW-yr charge spread over less assumed
energy capacity, i.e. UNDERSTATED here since the model would be
dividing by a number larger than the truth); a longer true duration
would overstate it.

**MEASURED BAND, added at HEAD `ecb7e54`.** The three ATB cost cases
were sourced when O&M was implemented but only `om_central` had ever
been run. Measured at the central configuration:

~~| arm | ATB case | objective | vs central | storage 2054 |~~
~~| `om_low` | Advanced | 27.531 bn | −0.423 (−1.51%) | 65,613.0 MWh |~~
~~| `om_central` | Moderate | 27.954 bn | — | 65,613.0 MWh |~~
~~| `om_high` | Conservative | 28.542 bn | +0.588 (+2.10%) | 65,613.0 MWh |~~

**Post-epoch (TASK 70 B.1):**

| arm | ATB case | objective | vs central | storage 2054 |
|---|---|---|---|---|
| `om_low` | Advanced | 24.560 bn | −0.427 (−1.71%) | 63,056.31 MWh |
| `om_central` | Moderate | 24.987 bn | — | 63,056.31 MWh |
| `om_high` | Conservative | 25.612 bn | +0.625 (+2.50%) | 63,056.31 MWh |

~~**Span 1.011 bn, 3.6% of central.**~~ **Span 1.052 bn, 4.2% of central.**
Storage is identical to the decimal across all three; the band is a
pure cost effect with no quantity response (see `findings_frozen.md`
F-3).

**What this settles.** ~~Adding O&M moved the objective by 6.602 bn; the
sourced uncertainty around that addition is 1.011 bn — about 15% of the
correction itself.~~ **Post-epoch, switching O&M off (`om_none`, the synthetic
arm, 18.466 bn, `results/lever_ranking/arms_6046f759/lever_ranking.csv`) to `om_central` (24.987 bn)
moves the objective by 6.520 bn; the sourced uncertainty around the
addition is 1.052 bn, about 16% of it.** **The omission was material;
the choice of cost case is not.** A referee disputing ATB Moderate as
the central assumption is disputing at most 1.052 bn on a 24.987 bn
objective, and the direction is bounded in both senses by sourced
endpoints rather than by an analyst's judgement.

**What it does not settle.** This band is the uncertainty in ATB's own
cost-case projections for US plant. It is NOT the uncertainty in
applying US rates to a Nigerian fleet, which is the unscaled-rates
question above and remains unquantified — the two are independent, and
the second is almost certainly the larger of the two. Do not cite the
~~3.6%~~ **4.2%** band as though it bounded the Nigerian applicability question.

**What would resolve it.** For the scaling question: a Nigeria-specific
O&M cost survey (none is known to exist at the technology granularity
this model needs). For the L21 coupling: promote `storage_duration_hours`
to a required, swept key (declined in `docs/task_bucketA_parity.md`
for this project's remaining timeline, not because it is unimportant).

---

> **Generation (TASK 86):** the hash-keyed result files this entry cites are blended dispatch (no `gas_vintage_dispatch=True` in the cited files' arm records). Checked: `6046f759` (blended); `792f8dce` (blended); `f6926482` (blended). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142. [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L23 — F1 ships two capital tranches; the design specified three

**Status:** LIVE. Declared here for the first time — this gap was
previously recorded nowhere in the register, the methodology, or the
findings.

**What.** The tranche formulation (plan 3.2, the item the plan itself
calls "the single most important fix for the financing thesis")
specifies a capital merit order of THREE tranches:

| Tranche | Rate anchor | Envelope anchor |
|---|---|---|
| Concessional (DFI/climate funds) | named facility terms | facility size x deployable fraction [TASK 57: as implemented the fraction is an identity at 1.0 and is not applied; `04_run_fin2_tariff_sweep.py --deployable-fraction` is RETIRED, recorded not removed] |
| **Guaranteed commercial (PRG/FGN-backed)** | **commercial − guarantee spread** | **World Bank PRG headroom / FGN contingent-liability ceiling** |
| Unguaranteed commercial | Nigeria power CoC | unbounded |

**The implementation ships TWO.** The capital charge is
`CRF(r_conc, L_conc) x conc_vol + CRF(r_comm, L_comm) x comm_vol`.
There is no guaranteed-commercial tranche in any form — not present and
inert, not fixed at zero, not commented out. A tree-wide search for
`guarantee`, `PRG`, `credit_enh`, `headroom` and `contingent` returns
only ordinary English usage in unrelated comments; the only real
occurrences are in `merged_correction_plan_v2.md` itself.

**Why it is missing.** Blocked on sourcing, per the plan's own ledger:
*Guarantee headroom | World Bank PRG / FGN contingent-liability ceiling
| blocks 3.2*. Neither the deployable PRG headroom for Nigeria's power
sector nor the FGN contingent-liability ceiling is published at the
granularity this tranche needs — a facility size and a guarantee spread.
Without both, the tranche would be two invented numbers wearing a
citation, which is the failure mode this register exists to prevent
(compare the retired `STORAGE_OM_PER_MWH_YR`, L22, and the
non-convertible JICA discharge figures, L20).

**What is lost by its absence.** Three things, in descending order of
cost to the thesis:

1. **The guarantee-headroom dual does not exist.** Plan 3.2 names it
   explicitly as a policy output and notes the literature is
   "near-unoccupied" on it. It would have priced the marginal dollar of
   sovereign guarantee capacity — the quantity a finance ministry
   actually rations. This is a forgone contribution, not an error.
2. **The capital merit order has two steps where the design has three.**
   Marginal cost of capital still becomes an output (the F1 contribution
   holds), but it steps from concessional to unguaranteed commercial
   with nothing between. Real Nigerian power financing has that middle
   rung; the model does not.
3. **Credit enhancement cannot be evaluated as a policy instrument.**
   The model can price concessional volume and commercial rate, but
   cannot answer "what is a PRG worth here" — which is among the more
   actionable questions a Nigerian power-sector planner would ask.

**Direction of bias.** Omitting the middle tranche makes the capital
merit order **more expensive than reality at the margin**: capital that
would in practice be guaranteed at commercial-minus-spread is instead
priced at full unguaranteed commercial. So the model **overstates the
cost of the commercial tranche** and, with it, overstates the value of
the concessional envelope relative to a world where guarantees exist.

The direction matters for F-2. Concessional finance is reported as
inframarginal — it changes who pays, not what is built. A guaranteed
tranche sitting between concessional and commercial would narrow the gap
the envelope is measured across, so **F-2's conclusion is unlikely to
reverse, but its magnitude (2.46 bn across the 0-to-10x envelope span)
is an upper bound on what a two-tranche world implies.** State this
where F-2 is reported.

**Post-epoch figure (TASK 70 C.2).** The 2.46 bn is pre-epoch. Post-epoch the concessional-envelope span is 2.147 bn (`env_10x` 23.109 to `env_none` 25.256 bn; `results/lever_ranking/arms_6046f759/lever_ranking.csv`). It remains an upper bound on a two-tranche world; the condition is in "Reporting conditions" at the end of this register.

**Not a silent default.** Unlike L19 and L21, nothing here resolves to
an unstated value at runtime. The tranche is absent, not defaulted. No
result is contaminated; the model does exactly what its code says. The
defect is that the code did not do what the design said, and until this
entry that divergence was recorded nowhere.

**What would resolve it.** Named facility terms for a Nigerian
power-sector PRG (World Bank IDA guarantee allocation, or an equivalent
MIGA/ATI instrument), giving both a deployable envelope and a guarantee
spread against the commercial anchor. Failing that, an explicitly
DECLARED spread and envelope, swept as a band — which would at least
produce the headroom dual as a conditional result rather than not at
all. The LP structure already supports a third tranche: the capital
charge is a sum over tranches and rates rise as cheap tranches exhaust,
so the formulation stays convex and remains a pure LP.

**Process note.** This gap survived from the plan's authorship to the
model freeze without appearing in any document that a reader of the
results would consult. It was found by comparing the plan's
specification against the code, not by reading either alone — the same
method that found the O&M omission (L22) and the retirement inversion
(F-20). RULE: when a design document specifies N components and the
implementation ships fewer, the shortfall is declared at the moment it
is noticed, in the register, not left to the plan's own status markers.

---

> **Generation (TASK 86):** the hash-keyed result files this entry cites are blended dispatch (no `gas_vintage_dispatch=True` in the cited files' arm records). Checked: `6046f759` (blended). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142. [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L24 — Post-2035 cap shape is a second analyst construction stacked on the apportionment rule

**Status:** LIVE — declared, sensitivity arm added (plan 4.2)

**What.** NDC 3.0 anchors only 2030 and 2035. Every capped year from 2036
onward — 2036–2045 in the reporting window, all of 2046–2054 in the
buffer — is therefore unanchored by NDC 3.0 itself, for every cap
scenario in this model. Two constructions for that unanchored span now
coexist: the original **flat hold** (2035 cap held constant to the
horizon end; `ndc2_unconditional`, `ndc2_conditional`,
`ndc3_unconditional`, `ndc3_conditional`), and the new
**`ndc3_conditional_decline`** arm (plan 4.2), which instead declines
linearly from the `ndc3_conditional` 2035 cap toward zero at 2060 —
Nigeria's own stated net-zero target — continued through the buffer
rather than held flat.

**Why unresolved.** NDC 3.0 (Section 3, pp.14–15) states numerical
targets only for 2030 and 2035. No Nigerian document sourced during this
project specifies a power-sector-level post-2035 pathway shape, so
neither construction can be upgraded to SOURCED.

**Interaction with plan 4.1 — two stacked constructions, stated
plainly.** The cap value at *any* year, including 2030 and 2035
themselves, already rests on plan 4.1's apportionment rule: gas-fired
power's share of energy-sector abatement, derived from this model's own
baseline gas-to-power split — a methodological choice, not something in
NDC 3.0's text (4.1's own wording: "your methodological construction, not
something in NDC 3.0"). Past 2035, a *second* construction stacks on top
of the first: which shape the cap takes once NDC 3.0's own numbers run
out. Ten of the twenty-one reported years, and the entire buffer, carry
both constructions at once. State both, together, wherever a post-2035
capped result is reported — citing only the apportionment rule
understates how much of the number is this project's own construction.

**Direction of bias.** The two constructions are not symmetric.
**The flat hold is not neutral** — it is an implicit "climate ambition
stops in 2035" assumption, understating the abatement a country actually
committed to net-zero-2060 would need across the back two-thirds of the
horizon. **The linear decline's endpoint and direction are sourced**
(net-zero-2060 is Nigeria's own target), **but its shape is not** — a
real decarbonisation pathway is not necessarily linear (most observed
trajectories are convex, slow-then-fast), so the straight line is a
declared simplification, not a second sourced figure. Net: flat
understates second-half ambition in one direction; the decline's
endpoint is directionally correct but its interpolation is exactly as
analyst-chosen as the apportionment rule it stacks on.

**Mitigation.** `ndc3_conditional_decline` carried as an additive
sensitivity (plan 4.2), opt-in only via `--include-decline-arm` on
`02_run_ndc_caps.py` (default off), never central, never in
`run_central_sweep.sh`. The four original flat-hold scenarios are
unmodified — see `docs/task_cap_trajectory.md` for the additivity proof.

**What would resolve it.** A Nigerian 2036–2060 sectoral decarbonisation
pathway (an LT-LEDS or equivalent, if one specifying power-sector shape
exists) would replace the straight line with a sourced curve. Absent
that, report the flat-hold and declining arms as a declared band, not a
single central figure.

---

> **Decision recorded 2026-10-23 (TASK 69 A.4): the capped lever ranking is measured under NDC 3.0, both arms.** `ndc3_unconditional` and `ndc3_conditional`, the conditional reported as the binding commitment and the unconditional as a robustness check. NDC 2.0 is excluded as superseded; the `ndc3_conditional_decline` arm is excluded because this entry records the post-2035 shape as a second analyst construction, not a commitment. ~~Consequence: Question 1's capped ranking and Question 3's leakage results are meant to sit on the same cap.~~ **Correction (TASK 70 C.4): the decision does NOT put Question 1's capped ranking and Question 3's leakage results on one cap, and the sentence is withdrawn so that it is not repeated.** F-21 is the flat-hold against decline pair (it rests on the arm this decision excludes) and F-43 uses flat fractions of baseline (`cap_fraction` 1.0 to 0.01), neither an NDC 3.0 cap. The NDC 3.0 decision itself stands. **Two qualifications from the files, not decided here:** (i) F-21 compares the flat hold with the decline arm, so it rests on the arm this decision excludes; it stands as a trajectory sensitivity (the declared band above), not as a commitment result. (ii) F-43 uses flat caps at fractions of the baseline (`cap_fraction` 1.0 to 0.01, `scripts/18_run_pareto_frontier.py`), not the NDC 3.0 caps; F-31 uses NDC 3.0 unconditional. So only F-31 and the existing `02_`/`08_` NDC arms sit on an NDC 3.0 cap as stored.

## L25 — WITHDRAWN IN PART: DIST-5 is sourced but NOT implemented; the
previously reported measurement was never computed (C-10)

**Status:** CORRECTED 2026-09-2x (methodology claims ledger C-10). This
entry previously stated "DIST-5 is now implemented" and "the direction
IS established," disfavouring EaaS, with a measured consequence of
under 0.03%. **Both claims are false.** No code path at HEAD, or on any
branch, implements DIST-5. **L16 (LIVE, "unimplemented") is the
accurate entry; this entry contradicted it.**

**What exists.** The sourced numbers below are real — they sit in
`docs/shape_factors_and_dist5.md` as data, retrieved from Global Solar
Atlas (2026-09-05) at identical Kano coordinates so siting is held
constant:

    ground-mounted large scale, 16 deg S (optimum)  1,723.7 kWh/kWp
    medium commercial,          16 deg S            1,670.7    -3.1%
    medium commercial,           0 deg flat         1,646.7    -4.5%
    medium commercial,          30 deg azimuth  90  1,491.3   -13.5%
    medium commercial,          30 deg azimuth 270  1,485.7   -13.8%

30 degrees is the representative Nigerian roof pitch. The penalty is
almost entirely azimuth, not tilt: a flat roof costs 4.5%, an off-south
30-degree roof costs 13.6% (the file rounds this "-13.6%" in prose
against "-13.5%" in the table; both are the same GSA figure). Together
with the capex premium (+16% 2025 rising to +37% 2045, NREL ATB
Distributed Commercial PV) and the capacity-factor cut (0.19 -> 0.164,
ratio 0.8635), these are the three numbers that would constitute DIST-5
if implemented.

**What does not exist: any implementation, and any measurement.**
`docs/shape_factors_and_dist5.md` Part B is an inventory of what
implementing DIST-5 would require; it stops at "STOP — B0 finding" and
states explicitly, later in the same file, that DIST-5 "has not"
shipped. Neither builder (`optimize_model.py`, `optimize_model_sliced.py`)
nor `scenarios.py` carries a distributed capex multiplier, a distributed
`solar_cf`, or a distributed seasonal profile for the EaaS channel — the
two solar channels remain identical except for financing (L16). The
"<0.03% objective movement, EaaS deployment identical to the megawatt"
figure previously reported here was therefore never computed by any
solve.

**How the false claim entered `findings_frozen.md`.** Commit chain:
`b7f7bdb` put the sourced figures above into
`docs/shape_factors_and_dist5.md` (also touching `scripts/` and `src/`
for the *shape-factor* corrections A1-A3, which are real and separate
from DIST-5 itself) -> `977a682`, whose message claims seven runners
were changed to expose `--solar-cf-case` and derive the EaaS CF by
ratio, but whose diff is 172 lines inside
`docs/shape_factors_and_dist5.md` alone — no `src/`, no `scripts/` -> `0c2cd5b`,
which revised this register entry to "implemented" on the strength of
that commit message rather than a diff -> `d55f814`, which propagated
"implemented and measured inframarginal" into F-13. Severity
LIVE-REPORTED (forensic audit A63): `977a682` is the mechanism by which
an unmeasured figure entered the frozen findings.

**Direction of bias: not established**, per L16. DIST-2 (favours EaaS)
and DIST-5 (disfavours EaaS) are both sourced and both unimplemented;
because they oppose, their joint absence does not have a known net
direction. This entry's earlier claim that the model is "now biased
AGAINST the EaaS channel" is withdrawn.

**Still not implemented, still out of scope**, per L16: DIST-2, for the
metering-boundary reason there. And the SITING component of DIST-5: a
southern rooftop is -19.3% against northern utility ground-mount
(Anambra 1,390.3 vs Kano 1,723.7 kWh/kWp), which combined with
orientation is -30.3%, CF 0.137 — unmeasured and, like all siting
effects, outside a model with no transmission representation.

**What would resolve it.** As L16 states: implement DIST-2 and DIST-5
together, then re-run any F3/EaaS-split result that depends on the
distinguishability of the two channels.

---

## L26 — Solar capacity factor is a national average with no siting

**Status:** LIVE

**What.** `solar_cf` is now SOURCED at central 0.19, band 0.16-0.20,
from Global Solar Atlas ground-mounted large scale at each site's own
optimum fixed tilt:

    Kano (Kumbotso)  11.887 N   0.1968   national maximum measured
    FCT               9.165 N   0.1743   (run at 10 deg vs optimum 14)
    Anambra           6.620 N   0.1587   southern

**What was corrected.** The retired central of 0.20 was the national
MAXIMUM — it assumed every plant is built at the best site in the
country. NIRP 2024 itself uses 0.19 falling to 0.18 (Annex F Table 19
with Table 8 CF), so the model sat above the national plan's own
assumption. The retired high arm of 0.22 was above EVERY measured
Nigerian site and is retired as unattainable rather than kept as an
unreachable bound.

**Why unresolved.** The model has no transmission and no siting, so a
single national CF must stand for a spatially heterogeneous resource.
Best-resource land is northern and overlaps insecurity zones; demand is
southern. The model cannot represent that trade-off.

**Direction of bias.** Unresolved rather than one-directional. Using a
northern CF assumes solar is sited for resource and ignores both
security and transmission; using a southern CF assumes it is sited at
load and forgoes 19% of yield. The 0.16-0.20 band is the honest span
and both ends are measured.

**Unexplained observation.** `cf_low` (0.16) gives a CHEAPER system
than `cf_central` (0.19) — 20.086 bn against 20.856. Single-variable,
same runner, so it is real. It suggests solar sits at or near a cost
knife-edge in the corrected model. Not diagnosed.

---

## L27 — Six shared analysis functions carried a stale `load_scenario()` calling convention

**Status:** REPAIRED at the six shared functions, AND at the nine
directly-broken runners (`ecb7e54`, verified — see amendment). Two
library functions genuinely remain unrepaired, now named explicitly:
`run_str1_storage_parameter_sweep` and `run_bottleneck_sensitivity`.

**Amendment (task_register_amendment).** The nine directly-broken
runners this status line refers to are `04_run_fin2_tariff_sweep.py`,
`05_run_fin3_capital_sweep.py`, `15_run_pol1_ndc_comparison.py`,
`18_run_pareto_frontier.py`, `19_run_carbon_price_sweep.py`, `20_run_
cf1_solar_yield_sensitivity.py`, `24_run_leakage_arm.py`, `25_run_
obligation_trajectory.py`, and `99c_run_genset_arm_sensitivity.py` —
confirmed via `git show --stat ecb7e54` and, independently, via `add_
standard_arms` present and called in all nine current files. **All
nine are repaired.** The "two further library functions" this status
line has always alluded to, without naming, are named here for the
first time: `run_str1_storage_parameter_sweep` (`src/optimize_
experiments.py:1940`) and `run_bottleneck_sensitivity` (`:750`).
Verified via `inspect.signature` and direct reading: `run_bottleneck_
sensitivity(scenario, econ)` calls `load_scenario()` internally
(`:759-766`) supplying only 7 of the 17 sentinel keys; `run_str1_
storage_parameter_sweep` calls `load_scenario()` internally (`:2019`)
with its own self-diagnosing comment (`:2042`) confirming the same
omission. **Both genuinely still carry the stale convention.**
`run_str1_storage_parameter_sweep`'s dead status is the already-
declared, expected consequence of **L5**'s STR-1 suspension (see also
**L32**, which already names this same function as one of its six dead
library functions) — not a new, independent concern; `run_bottleneck_
sensitivity` has no such prior declaration and is the more genuinely
open of the two.

**What.** Six functions in `src/optimize_experiments.py`
(`run_gas_regime_ndc_matrix`, `run_rel1_feasibility_matrix`,
`run_rel2_marginal_cost_curves`, `run_rel3_financing_frontier`,
`run_dem1_demand_sensitivity`, `run_dem2_growth_gas_matrix`) each
carry their own `load_scenario()` call that omits 15 or 16 of
`load_scenario()`'s 17 sentinel-required keys — the same keys every
healthy runner in this project (`01_`, `01b_`, `02_run_eaas`,
`02_run_ndc_caps`, `03_`, `22_`, `23_`, `26_`) threads through in
full. Every call raises `TypeError` before any solve.

**What it meant.** Six research questions have been unanswerable
since roughly the phase where the keys they omit were introduced as
required (2.6 through 2.13, by the functions' own internal dating):
GAS-1 (gas shadow price vs. export/domestic benchmarks), GAS-2 (EaaS
gas-scarcity relief), GAS-3 (gas decline shape vs. level), REL-1
(feasibility threshold determinant), REL-2 (marginal cost of
reliability), REL-3 (EaaS financing-frontier improvement), DEM-1
(served-vs-latent demand sensitivity — additionally a stub,
`raise NotImplementedError`), and DEM-2 (demand-growth × gas
interaction) — eight questions across six functions, since
`run_gas_regime_ndc_matrix` alone serves GAS-1, GAS-2, and GAS-3.

**How it was found.** Not by any single runner failing in isolation —
a full runner inventory (`docs/runner_inventory.md`, 32 runners plus
associated library code) that read every runner's `load_scenario()`
call, or traced it to the shared function it delegates to, found the
same 15-16-key omission recurring identically across what first
looked like twelve independent failures. Two of the six functions
(`run_rel1_feasibility_matrix`, `run_gas_regime_ndc_matrix`) already
carried inline comments self-diagnosing the defect at the point it
occurs — evidence the pattern was noticed locally, previously, but
never traced to its full extent or fixed at the source.

**Why this is a different defect class from twelve independent
bugs.** A copy-pasted calling convention, wrong once and propagated,
has a different repair shape than twelve unrelated defects would:
fixing six functions (plus, separately, nine runners that own the
identical defect directly with no shared function involved) resurrects
every dependent runner in one pass, and a future addition to
`load_scenario()`'s required set will re-break all six identically
unless the fix also converts these call sites to the same
fail-loudly-on-omission discipline the rest of the project uses
(`docs/methodology_protocols.md` §1). Treating this as "twelve broken
scripts" would have led to twelve separate, redundant repairs;
treating it as "one convention, six library call sites, nine direct
call sites" is both more accurate and considerably cheaper to fix.

**Related.** Register **L10**'s dead-runner list was confirmed
incomplete by the same inventory: it names `05`, `06`, `07`, `09b`
(both), `15`, `17`, `18`, `19`, `20`, but not `10`, `24`, `25`, or
`99c` — all four independently confirmed dead for the same or an
analogous reason. L10 itself notes it "predates the O&M work and is
likely incomplete"; this entry and the runner inventory are the
confirmation.

**What would resolve it.** This repair, Part 3 (released separately
from this inventory step): thread every omitted key through each of
the six functions as a required parameter with no default, per
`docs/methodology_protocols.md` §1's sentinel discipline — except
`om_case`, already present as a hardcoded literal, which must be
promoted from literal to required parameter specifically so no future
caller can silently revert the O&M correction. Then, separately (out
of this repair's scope), the nine directly-broken runners and the
`NotImplementedError` stub each need their own repair to actually
call the fixed functions/`load_scenario()` correctly.

---

## L28 — `cost_decomposition` silently omitted 6.602 bn of O&M and an always-present rebuild/replacement charge; found by a consuming script's own guard, not by inspection

**Status:** RESOLVED (`task_wrong_output_repair`, Parts 1 and 4).
Both terms identified below are now read and reconciled; the identity
`real_expenditure + selfgen + voll_penalty − salvage + om ==
system_cost_npv` closes exactly, for both `capital_treatment_case`
regimes, confirmed structurally (by reading every term on both sides,
not by running anything).

**What.** `extract_planning_diagnostics`'s `cost_decomposition`
(`src/optimize_experiments.py`) omitted TWO real, charged objective
terms, in both `capital_treatment_case` branches:

1. **O&M** (`m.cost_om_npv`) — omitted entirely. +6.602 bn (+30.9%) at
   the central case, the single largest correction in this project
   (`docs/task_om_implementation.md`), present unconditionally in both
   builders and both branches since that correction shipped.
2. **Mid-horizon asset-replacement capex** — `m.cost_replacement_
   capex_npv` (`pre_f1_overnight`) and `m.cost_rebuild_capital_charge_
   npv` (`f1_tranche`, folded inside `m.capital_charge_total_npv`).
   Per register **L18**, a rebuild/replacement charge is generated
   "for every lifetime value this project actually supplies anywhere"
   — i.e. present in essentially every run this project can produce,
   not a theoretical edge case.

**How each was found.** Not by inspection or a targeted audit of
`extract_planning_diagnostics` — `docs/runner_inventory.md` found the
O&M omission as a side effect of documenting
`16_plot_cost_decomposition.py`'s own reconciliation guard (`total =
real + selfgen + voll − salvage`, checked against `npv_total_cost_
usd`), which had been printing MISMATCH on every `om_central`/
`om_low`/`om_high` result since the O&M correction shipped, for
reasons the guard's own output never named. The replacement/rebuild
gap was found while VERIFYING the O&M fix (Part 1's own reconciliation
check, applying TASK.md's instruction to confirm every objective term
appears exactly once) — directly readable from the `components` dict's
own internal cross-check: under `f1_tranche`, summing the four channel
`*_capital_charge_npv` entries did not equal the same dict's own
`capital_charge_total_npv` entry.

**Why this matters beyond these two bugs.** A reconciliation guard
that checks a derived total against an INDEPENDENTLY computed
objective value (not a re-derivation of the same inputs) catches a
silent omission that a unit test written against the decomposition
function's own (wrong) expectations never would have. Neither omission
produced an exception, a visibly wrong number in isolation, or a
crash — every `om_central` result before Part 1, and every `f1_
tranche` result before Part 4, was a plausible, structurally
unremarkable, understated total. **Keep this class of guard, and read
what it says even when "everything looks fine otherwise."**

**How Part 4 resolved the remainder.** `pre_f1_overnight`:
`m.cost_replacement_capex_npv` read with a `hasattr` guard (it is
built only in this branch, mirroring `m.salvage_npv`'s existing
guard), folded into `cost_real_expenditure`'s capital total (capital
expenditure by nature — argued, not assumed: under this project's
central lifetimes, solar and gas life exactly equal the model horizon
and never rebuild, so only storage, the smallest capital channel,
ever does — a small charge relative to the objective, unlike O&M's
30.9%, so it did not warrant O&M's own-top-level-line treatment).
`f1_tranche`: `cost_real_expenditure`'s capital total now reads `m.
capital_charge_total_npv` DIRECTLY rather than re-summing the four
channel charges, since that Expression already equals their sum plus
`m.cost_rebuild_capital_charge_npv` — summing only the four channels
would have silently dropped the rebuild charge a second time. Reading
the model's own total (rather than re-listing its parts by hand) also
means this line cannot go stale the way a hand-copied list could,
regardless of what future term the objective's own capital total might
one day include.

**Direction of bias, now closed.** Both omissions biased the
DECOMPOSITION low relative to the true objective (never `m.system_
cost_npv` itself, the actually-optimised and reported quantity, which
was always correct). Resolved in both directions, not just one: fixing
O&M without also closing the replacement/rebuild gap would have left
the identity failing to close under `f1_tranche` and (for runs with an
early storage vintage) under `pre_f1_overnight` — Part 4 confirmed
neither residual remains.

## L29 — `build_model_sliced()` silently dropped its own declared
> reliability-alias parameters; a real sweep's sliced half was
> empirically dead across twelve values
>
> **Status:** RESOLVED (`task_audit_fixes_p1`) — the translation is now
> ported into `build_model_sliced()`; the historical defect and its
> consequence are recorded below for the record.
>
> **What.** `build_model_sliced()` accepted `reliability_max_unserved_
> fraction` and `reliability_mode`, both commented `# DEPRECATED alias
> for ...`, and never read either anywhere in its ~1,700-line body. The
> annual builder, `build_model()`, has the identical two parameters
> with identical comments and correctly translates them into
> `max_offgrid_fraction`/`grid_service_mode` (including a `ValueError`
> guard against passing both an alias and its canonical parameter).
> Only the sliced builder was missing the translation.
>
> **Consequence, verified empirically, not merely argued.**
> `scripts/23_run_selfgen_frontier.py` sweeps a 12-point `eps` grid
> across BOTH builders in one run, passing ONLY
> `reliability_max_unserved_fraction=eps` (never `max_offgrid_
> fraction`) to whichever builder is active that iteration. In the
> already-persisted `results/selfgen_frontier/frontier.csv` (48 rows),
> every `sliced` row for a given `pricing` arm is bit-identical to full
> float precision across all twelve swept `eps` values (`objective_usd`
> and `shortfall_fraction_realised` unchanged past the 13th significant
> digit) — the sliced half of the sweep solved the same unconstrained
> problem twelve times. The `annual` half of the SAME file, same run,
> varies correctly (infeasible below `eps=0.3`, then a smoothly
> declining objective from `1.4667e12` to `1.1468e12` as `eps`
> increases) — proving the sweep mechanism itself and the annual
> builder both worked, isolating the defect to the sliced builder's
> silently-dropped aliases.
>
> **Fingerprint.** A parameter accepted and then discarded — the
> caller's intent vanished silently, with no error, warning, or
> visible signature in the output (every `sliced` row still printed
> `feasible=True` with a plausible-looking objective). The annual
> builder's CORRECT implementation of the identical translation is
> precisely what made this detectable at all: without a working
> sibling to diff against, twelve identical numbers in a row could as
> easily have been mistaken for "the constraint doesn't bind in this
> range" as for "the constraint was never wired."
>
> **Direction of bias.** The sliced-builder half of `results/
> selfgen_frontier/frontier.csv` understated the sensitivity of every
> reported quantity to the reliability threshold — not a directional
> bias on a level, but a total absence of the swept relationship where
> one was intended and reported as present.
>
> **What is void, and must be regenerated.** `results/selfgen_
> frontier/frontier.csv`'s `sliced` rows, in full — see the DO-NOT-
> REPORT addition, 4E below. No number in `findings_frozen.md` or
> `frozen_state_2026_09_10.md` was found to derive from this file
> (checked: `grep -rn "selfgen_frontier\|23_run_selfgen\|EPS_GRID"`
> across both — no hits), so no published figure requires correction;
> the file itself does.
>
> **What would resolve it further.** Nothing — closed by the port.
> `23_run_selfgen_frontier.py` must be re-run (outside every
> whitelist so far; Zeus's own next step) before its sliced-builder
> output can be cited.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L30 — A live function carried the retired gas efficiency (0.43)
> as a silent fallback; inert only because its one caller never
> omitted the key
>
> **Status:** RESOLVED (`task_audit_fixes_p1`).
>
> **What.** `run_deterministic_scenario()` (`src/optimize_
> experiments.py`, a legacy non-LP dispatch simulator, explicitly
> labelled "NOT for thesis result tables") read `scenario.get(
> "gas_eta", 0.43)`. `0.43` is not an arbitrary placeholder: it is the
> exact figure this project's own sourcing ledger
> (`docs/merged_correction_plan_v2.md`) records as retired — a CCGT
> efficiency figure misapplied to this project's OCGT-dominated fleet,
> replaced project-wide by the measured **0.287** derived from NERC
> 2024 generation ÷ gas-to-power volumes. A fallback that fired would
> silently overstate fleet efficiency by 50%, understating gas fuel
> consumption and emissions by the same margin in this function's own
> (non-reported) cost/emissions figures.
>
> **Why it never fired.** `load_scenario()` unconditionally sets
> `scenario["gas_eta"] = 0.287` in its returned dict, and this
> function's one live caller (`scripts/02_run_eaas.py:301`) always
> builds its scenario via `load_scenario()`. The fallback was
> therefore reachable in principle, never in the run history this
> project can inspect.
>
> **Why "never fires today" is not a safety property.** `run_
> deterministic_scenario(scenario, econ, capacity_paths)` takes a bare
> `scenario` dict as a parameter — nothing in its signature or body
> requires the caller to have built it via `load_scenario()`. The
> function's own docstring describes its purpose as "quick feasibility
> sketches" from a HAND-SUPPLIED capacity path — precisely the use
> case most likely to be invoked with a hand-built, partial scenario
> dict that omits `gas_eta`. This is the same shape this project's own
> register already names for other parameters (L9, L19, L21): a
> load-bearing value whose correctness depends on every caller
> happening to go through one specific code path, with nothing
> enforcing that they do.
>
> **Resolution.** Changed to `scenario.get("gas_eta", None)` +
> `raise ValueError(...)` if absent, matching `build_model_sliced()`'s
> own existing guard for the same key verbatim. A future caller that
> omits `gas_eta` now fails loudly, at the point of failure, naming the
> key — not months later, in a plausible-looking but 50%-wrong number.
>
> **Direction of bias, historical.** None realised — confirmed no run
> in this project's history reached the fallback branch (the one live
> caller always supplies the key). Recorded as a TRAP class finding,
> not a LIVE-REPORTED one: correct in every run to date, wrong the
> first time a caller does what this function's own documentation
> invites.
>
> **Sibling checked, not a defect.** The same function's other numeric
> `.get()` default, `scenario.get("storage_duration_hours", 4.0)`,
> matches the canonical value set by `load_scenario()` and used
> identically by both LP builders (register L21) — not fixed, since it
> is not a divergent or retired figure.

## L31 — The reported "peak adequacy" diagnostic uses a flat,
> arm-invariant multiplier while the constraint it is named after uses
> an arm-sensitive one
>
> **Status:** LIVE (merged 2026-10-23, TASK 69 A.3: this is the single entry for the peak-adequacy diagnostic; the peak-adequacy half of L103 is folded in below and L103 now points here. The earlier number is kept.) The merged defect is an **Appendix B statement**: the field is a defect and not a result; no Chapter 4 quantity depends on it.
>
> **What.** `extract_planning_diagnostics`'s `peak_adequacy_by_year`
> field (`src/optimize_experiments.py`) computes its own `peak_demand_
> mw` as `avg_demand_twh_converted_to_mw × peak_demand_multiple`, where
> `peak_demand_multiple` is a flat **2.5**, set unconditionally by
> `load_scenario()` (`scenarios.py:3157`) OUTSIDE any `peak_multiple_
> case` branch — identical for `peak_suppressed`, `peak_unsuppressed`,
> `peak_nirp`, and `peak_high`. The ACTUAL firm-capacity constraint
> both LP builders build sizes against a structurally different
> quantity, `peak_mw[t]` — a REQUIRED scenario key, DERIVED from the
> selected `peak_multiple_case` arm's own `season_demand_factor`/
> `period_demand_factor` (register L1's subject), and genuinely
> sensitive to which arm is chosen.
>
> **Already known to the code, not to the register.** `scenarios.
> py:3150-3156` carries the author's own comment stating this is a
> "PRE-EXISTING, DIFFERENT quantity from peak_multiple_case/peak_mw...
> flagged here so the two are not confused" — an intentional, declared-
> in-a-comment design choice, not an oversight. It has never appeared
> in this register, `findings_frozen.md`, or `frozen_state_2026_
> 09_10.md`.
>
> **Why it matters.** `peak_adequacy_by_year`'s field names
> (`peak_demand_mw`, `firm_capacity_mw`, `reserve_margin`) read as a
> direct report on capacity adequacy against the model's own
> peak-demand assumption. They are not: the SAME configuration solved
> under two different `peak_multiple_case` arms produces two genuinely
> different firm-capacity requirements internally, but IDENTICAL
> `peak_demand_mw` figures in this diagnostic, because the diagnostic
> never reads `peak_multiple_case` at all.
>
> **Direction of bias.** Not one-directional — the diagnostic simply
> does not track the quantity its name claims to describe. Whether
> `reserve_margin` reads too high or too low relative to the true,
> arm-specific adequacy depends entirely on how far `2.5` sits from
> whatever `par_block`-derived multiple the selected `peak_multiple_
> case` arm actually implies, which varies by arm.
>
> **Why unresolved.** No fix is proposed by this task — Part 1 of the
> preceding audit found this, this entry only declares it. A fix would
> mean either deriving `peak_demand_multiple` from the same `season_
> demand_factor`/`period_demand_factor` machinery `peak_mw` already
> uses (closing the gap) or renaming the diagnostic's fields to state
> plainly that they use a fixed, arm-independent proxy (the L19/L21
> pattern) — a design decision, not made here.
>
> **Mitigation.** None currently — `peak_adequacy_by_year` has zero
> consumers in `scripts/` today (checked), so no published figure is
> affected. Do not cite `peak_adequacy_by_year`/`reserve_margin` across
> more than one `peak_multiple_case` arm as though it reflected the
> arm difference.
>
> **What would resolve it.** Derive `peak_demand_multiple` from the
> same per-arm shape factors `peak_mw[t]` uses, or rename the
> diagnostic's fields (e.g. `flat_proxy_peak_demand_mw`) to state
> plainly that it does not track `peak_multiple_case`.
>
> **Merged from L103 (2026-10-23, TASK 69 A.3).** L103 registered the same defect from a second side: the diagnostic uses `peak_demand_multiple` (2.5 x average) and counts gas plus storage power only, against the model's firm-capacity test (`peak_mw x omega`, credited capacity including hydro and credits). Measured in the invariant test (TASK 52 G8-1b, addendum B-1): reported peak 16,124 MW against the model's 9,434 MW, reported firm 12,476 MW against 11,348 MW. The diagnostic is unchanged (invariant `G8-1d`, FAIL, in the suite's 6 expected failures); it is quoted nowhere. L103's other half (reliability metrics ignoring self-generation) is RESOLVED (TASK 53 2.6) and stays in L103.
>
> **Consumers, updated 2026-10-23.** Zero in `scripts/` (this entry), and zero in `docs/findings_frozen.md`, `docs/chapters/` (Chapters 1 to 4 and drafts, apart from the Appendix B copy of this register) and both handovers (audit pass 3 C.4, `docs/chapter4_audit_pass_3.md`). Question 4's quantities use the model's own firm-capacity condition (F-32), the REL runners' shortfall (F-33) and costs (F-42), not this field.

## L32 — Three entire dead modules and six dead library functions;
> one of the three dead modules must not be revived without redesign
>
> **Status:** LIVE (declared, not deleted — this entry documents, it
> does not remove anything).
>
> **What.** Three files in `src/` define functions with zero callers
> anywhere in `src/` or `scripts/`, confirmed by direct search, not
> inference:
>
> 1. **`src/postprocess.py`** — `export_benchmark_comparison`, the
>    file's only function. Reads as support for GAS-1 (gas shadow
>    price vs export/domestic benchmarks) but `06_run_gas1_shadow_
>    benchmarks.py` computes its own comparison inline and never
>    imports this module.
> 2. **`src/gas_balance.py`** — `project_gas_availability_th` and
>    `gas_to_power_electricity_twh`, the file's only two functions. An
>    earlier, single-parameter exponential-decline model of gas
>    availability, superseded by the CSV/arm-based system in
>    `src/gas_supply.py` (`gas_available_power()`, keyed by `gas_
>    deliverability_case`) that every live caller actually uses.
> 3. **`src/stochastic.py`** — `sample_uncertainties`, `run_
>    stochastic_simulation`, `compute_risk_metrics`, all three
>    functions in the file. A complete, documented "fixed-plan
>    re-dispatch" Monte Carlo mechanism, distinct from both
>    `scripts/deferred/17_run_monte_carlo.py` [revived as `scripts/17_run_monte_carlo.py`, commit `36ddf55`, TASK 06] (dead on required
>    arguments, register L10) and `scripts/29_run_parameter_
>    uncertainty.py` (the Sobol parameter-uncertainty sweep) — never
>    wired to any runner at all.
>
> Additionally, six top-level functions in `src/optimize_experiments.py`
> (out of 33) have zero callers: `run_annual_cap_case`, `identify_
> bottleneck`, `run_bottleneck_sensitivity`, `run_str1_storage_
> parameter_sweep`, `run_all_deterministic_scenarios`, `run_financing_
> vs_resource_test`. One of the six (`run_str1_storage_parameter_
> sweep`) is the expected, already-covered consequence of register
> L5's declared STR-1 suspension. The other five are an orphaned
> exploratory layer predating the current runner-script architecture.
>
> **`stochastic.py` specifically must NOT be revived as-is.**
> `sample_uncertainties` draws `demand_growth` and `gas_scenario` (the
> gas deliverability regime) as two INDEPENDENT random variables per
> Monte Carlo sample, despite both responding to a shared macro driver
> — the identical failure shape this project already retires a whole
> script over (`scripts/deferred/17_run_monte_carlo.py` [revived as `scripts/17_run_monte_carlo.py`, commit `36ddf55`, TASK 06], per its own
> declared defects and register L10). `run_stochastic_simulation` also
> calls `build_model()` (annual) exclusively, never `build_model_
> sliced()`, so any revival would additionally inherit register L5's
> "storage is structurally dominated at annual resolution — optimal
> storage is identically zero" for every draw. Both defects are
> currently inert only because nothing calls the module.
>
> **Direction of bias.** None currently realised — every function
> named above has zero callers, confirmed. Purely a "declared, not
> deleted" entry: nothing here changes a reported number.
>
> **What would resolve it.** A scoped decision on each: delete as dead
> weight, or retain with a stated reason (as `src/EaaS.py` already does
> for its own, larger dead surface, citing `docs/task_bucketA_
> parity.md` Part B1). If `stochastic.py` is ever revived, redesign the
> sampling to draw demand growth and gas regime from a joint or
> conditional distribution reflecting their shared driver, and decide
> deliberately whether to port it to `build_model_sliced()` first.

## L33 — `04_run_fin2_tariff_sweep.py` still computes a retired public-
> financing mechanism; two diagnostic columns are unconditionally null,
> and pre-retirement output with real values from that mechanism sits
> on disk unlabelled (Phase 3 A52, A53)

> **Status:** LIVE (declared, not fixed — this script is outside this
> task's whitelist; `04_` was already established as NOT dead this same
> task, see A51/D25's correction, which changes the priority of this
> entry: this is now a live, executable script's defect, not a dormant
> one).

> **What.** `04_run_fin2_tariff_sweep.py:336-344` computes `envelope_
> npv` from `--concessional-envelope`/`--deployable-fraction` and
> threads it into `run_fin2_blended_bankability_sweep(...,
> concessional_envelope_npv=envelope_npv, ...)`. `src/optimize_model.py:
> 726-736` states this mechanism (`m.concessional_envelope_constraint`,
> `m.concessional_drawdown_npv`) is RETIRED per envelope resolution
> Decision 2 (`588a734`), superseded by the separate, F1-owned
> `concessional_envelope_case` mechanism. `extract_planning_
> diagnostics`'s own guard (`hasattr(m, "concessional_envelope_
> constraint")`) can never be true for a model built by the current
> builders, so `concessional_shadow_usd_per_usd`/`concessional_
> utilisation` are unconditionally `None` on any fresh run of this
> script, regardless of which `--concessional-envelope` arm is chosen.
> 04_'s own comment (`:149-157`) already concedes the flag now "feed[s]
> an inert scenario key" — the retirement is known locally, but not the
> concrete, checkable null-column consequence.

> **Compounding, on disk right now.** `results/fin2/tariff_sweep_
> results__dares_only__df1.csv` and `results/fin2/tariff_sweep_
> results__unbounded__df1.csv` (both mtime 2026-08-23 13:44) PREDATE the
> retirement commit `588a734` (2026-08-28) by five days. Their
> `concessional_utilisation` column carries 150 non-null real values
> (e.g. `1.0` for `blended_central`) — computed by the pre-retirement
> code path. Re-running `04_` today would silently produce all-null
> values for the same column in a new file, with no version marker
> distinguishing old from new in either.

> **Why unresolved.** Fixing the null-column issue requires either
> removing the retired parameter from `04_`'s call (a `scripts/`-only
> change, but outside this task's whitelist, and not "cheap" in the
> sense this task defines it — it touches this script's CLI surface and
> its interaction with `run_fin2_blended_bankability_sweep`, not a
> single line) or accepting the columns as permanently null and
> documenting that. Deferred to a dedicated task.

> **Direction of bias.** None on any currently-published number —
> `findings_frozen.md` cites nothing from `results/fin2/`. A future
> reader treating the pre-retirement CSVs as current output would be
> misled into believing `concessional_utilisation` is a live, working
> diagnostic for this script, when a fresh run cannot produce it at all.

> **Mitigation, applied this task.** `docs/merged_correction_plan_v2.
> md`'s DO-NOT-REPORT list now names both stale files explicitly (see
> that document) — screened before either could be cited.

> **What would resolve it.** Either strip the retired `--concessional-
> envelope`/`--deployable-fraction` plumbing from `04_` entirely (it
> cannot do anything under the current builders), or replace it with the
> current `concessional_envelope_case` arm's own diagnostic fields.
> Either way, regenerate `results/fin2/*` afterward and delete the
> pre-retirement files so no stale-but-plausible CSV survives alongside
> the fix.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L34 — The retired `required_margin` double-count will resurface the
> moment either of two dormant runners is repaired for its required
> arguments (Phase 3 A45, A56; see also L43, `13_`'s own reason it
> cannot even reach this defect yet — L43/L34/L35 are one "trap set for
> a future reviver," per the audit's own framing)

> **Status:** **CLOSED** (TASK 29; override removed in code commit `4ec43b4`,
> 2026-10-04; every affected runner re-run 2026-10-04 at `cded909`: Phase B
> items 5, 8, 13, 16 (both arms), 18 and 20, all exit 0; recorded 2026-10-08).
> `src/optimize_experiments.py`'s `run_rel3_financing_frontier` and
> `run_dem2_growth_gas_matrix` (and the dormant dem1/str1 paths) no longer
> overwrite `required_margin`; the literals are gone from `11_`, `13_`, `17_`,
> `18_`, `19_` and `99c_` (and the inert ones in `24_`, `25_`, `12_`); every
> arm takes the `dscr_case`-derived margin (1.30 at `dscr_central`). The
> defect's measured size is in `docs/task_log_29_1_required_margin_change.md`
> and `docs/task_log_29_2_findings_update.md`: **the forced 1.10/1.05 was a
> laxer bankability test taken only by the EaaS arm, so it biased results
> toward overstating EaaS feasibility, and the corrected EaaS costs rose
> (never fell beyond solver noise)**: Pareto `eaas_tight` at cap_fraction 0.10
> +14.3% (75.57 → 86.35 bn; largest grid-point shift +18.2%) and `eaas_moderate`
> +23.1% (55.19 → 67.92 bn); Monte Carlo unconditional EaaS P90 +0.57%; DEM-2
> EaaS interaction +0.13 bn; carbon-sweep `eaas_tight` at $150
> +9.6 Mt; **no REL-3 threshold and no F-35 frontier shift moved** (at eps-grid
> resolution). **Residue, stated rather than dropped:** the stored outputs of
> `07_run_gas2_eaas_gas_relief.py` (Phase B item 10; F-34) still carry a
> `required_margin` column of 1.10/1.05 on solves that used 1.30. That is a
> label on results that are correct (the override was never applied in `07_`);
> the script now writes the margin actually used, and the stored file is
> relabelled by the next run of item 10. It is closed because it affects no
> number, not because the stored file is fixed.
>
> ~~**Status (original, superseded):** LIVE — ESCALATED, ACTIVE, not dormant
> (task_register_
> amendment). Both `11_run_rel3_financing_frontier.py` and `13_run_
> dem2_growth_gas_interaction.py` were revived at commit `0012f16` and
> are now executable — the precondition this entry's "currently inert"
> framing depended on ("the runners that would trigger them are dead
> on required arguments") no longer holds. **Verified: the override
> lines below are unchanged in `src/optimize_experiments.py` (outside
> every whitelist available since this defect was found), and both
> runners' own hardcoded `1.10`/`1.05` literals are unchanged.** This
> double-count now fires on any real solve of either runner's `eaas`
> arm.~~
>
> **Precisely which rows/runs were affected (historical; see the CLOSED status
> above).** Every `13_run_dem2_growth_gas_interaction.py` output row with
> `financing_arm == "eaas"` (36 of the 72 total solves, per its own
> experiment structure) carries a `required_margin` silently forced to
> `1.10`, regardless of whatever `--dscr-case` was passed on the
> command line. Every `11_run_rel3_financing_frontier.py` row for the
> `eaas` financing arm, across every `POLICY_CONFIGS` entry, carries the
> same silent override (`1.10` or `1.05` depending on the policy arm,
> per `:161,169,177,185,193`). **Do not read any `eaas`-arm financing
> cost, `required_margin`, or frontier-shift figure from either runner
> as reflecting its own `--dscr-case` choice.** The `public_only`/
> `traditional` arm of both runners is unaffected by this entry and may
> be read normally.

> **What.** `dscr_scenarios()` (`src/scenarios.py:1790-1804`) retired
> the per-case `required_margin` split (1.10 unconditional / 1.05
> conditional) as a double-count once `capital_recovery_case`'s CRF
> rates began carrying that distinction directly. `02_run_eaas.py`,
> `03_run_ndc_eaas.py`, `04_run_fin2_tariff_sweep.py`, and (this task)
> `15_run_pol1_ndc_comparison.py` were all fixed to stop overriding it.
> Two more call sites still do:
> - `src/optimize_experiments.py:1840-1842`, inside `run_dem2_growth_
>   gas_matrix` — `13_run_dem2_growth_gas_interaction.py`'s own
>   `FINANCING_CONFIGS` (`:128-134`) supplies the retired `1.10` literal
>   for its `eaas` arm, overwriting the correctly `dscr_case`-sourced
>   value.
> - `src/optimize_experiments.py`, inside `run_reliability_sweep`'s
>   `POLICY_CONFIGS`-driven path used by `11_run_rel3_financing_
>   frontier.py:161,169,177,185,193` — the same `1.10`/`1.05` literals,
>   same overwrite mechanism.
> ~~Both `13_` and `11_` are currently dead on required arguments
> (missing 16 keyword arguments each, confirmed via `inspect.
> signature(...).bind(...)`), so neither override currently fires on a
> real solve.~~ **task_register_amendment: NO LONGER TRUE. Both were
> revived at `0012f16` and are now executable — see the escalated
> Status above. The override fires on every real `eaas`-arm solve of
> either runner now.**

> **Why unresolved.** The obvious, shallow fix for either runner's
> dead-on-args problem (thread the missing arguments through, mirroring
> the `cli_arms.py` pattern already applied to `05_/15_/18_/19_/20_`)
> ~~would make the script RUN while silently reintroducing this exact
> double-count~~ **DID make both scripts run (`0012f16`), and DID
> silently reintroduce this exact double-count, exactly as predicted —
> a trap for whoever repairs either runner next, unless this entry is
> read first.** The override sites themselves remain unfixed because
> they live inside `src/optimize_experiments.py`, outside every
> whitelist available to the tasks that revived `11_`/`13_`.

> **Direction of bias.** ~~None realised today — both runners are dead,
> confirmed.~~ **REALISED NOW: both runners are alive (`0012f16`).**
> Their `eaas`-arm financing cost IS biased by a double-counted margin,
> as of the revival landing — this entry's warning was not heeded
> first, because the revival task's whitelist did not include this
> entry (register edits were out of scope for that task, per its own
> bounding) and could not have fixed the override even had it been
> read.

> **What would resolve it.** When `11_`/`13_` are repaired for required
> arguments, remove the `POLICY_CONFIGS`/`FINANCING_CONFIGS` `required_
> margin`/`eaas_margin` overrides in the same commit — not as a
> follow-up — following the pattern already applied to `02_/03_/04_/
> 15_`. **This has NOT yet happened — both runners are repaired for
> arguments but the override removal described here was never done, and
> cannot be done from a docs-only whitelist. It requires an `src/`-edit
> task scoped specifically to `optimize_experiments.py`.**

## L35 — GAS-1's and GAS-2's dead runners carry three further defects
> that are invisible while the runners cannot execute, and would bite
> the moment either is repaired (Phase 3 A36, A37, A38; see also L34,
> L43 — the same "trap set for a future reviver" shape)

> **Status:** `06_`/`07_` REVIVED at `0012f16` (task_register_
> amendment). Item 2 (A37) is RESOLVED as part of that same commit.
> Items 1 and 3 remain LIVE — no longer dormant/inert, since both
> runners can now execute. A fourth item, the same capital-tier shape
> as the already-fixed A47, was found in `07_` during the revival and
> is added below.

> **What.**
> 1. **Duplicate source of truth.** The gas-price benchmark dictionary
>    (`dom_power_2p4` through `lng_export_10`, keyed off `MMBTU_PER_
>    TWH_TH = 1e9/293.071`) is hand-typed independently in FOUR places:
>    `06_run_gas1_shadow_benchmarks.py:108-118`, `src/optimize_
>    experiments.py:3772-3782` (the shared `run_gas_regime_ndc_matrix`,
>    which 06_'s own Arm 2 also calls), and TWICE within `07_run_gas2_
>    eaas_gas_relief.py` (`:114-124` and `:233-243`) — line numbers
>    re-verified at HEAD `2f11f3f` before B2. Values identical across all
>    four (diffed directly); unenforced. **RESOLVED once B2 lands
>    (docs/task_log_bstar_gas1.md, row B2): all four sites import
>    `BENCHMARKS` and `MMBTU_PER_TWH_TH` from `src/gas_benchmarks.py`.
>    The conversion constant has a fifth copy at `scripts/08_phase8_
>    analysis.py:75`, outside this item.** **Confirmed still
>    present, unchanged, and now LIVE rather than dormant — both files
>    execute as of `0012f16`.**
> 2. ~~**Copy-paste status bug.** `07_run_gas2_eaas_gas_relief.py:187-
>    188` (`run_no_policy_gas_matrix_eaas`'s own result dict) hardcodes
>    `"financing_regime": "traditional"` and `"status": "traditional"` —
>    an unedited copy of the sibling non-EaaS function's dict
>    (`:288-289`, correct THERE). `baseline_no_policy` is the SOLE
>    caller of this function; every downstream read gates on `r_yes.
>    get("status") == "optimal"` (`:365,382-385`), always False, so
>    `yes_shadow`/`eaas_binding_years`/`eaas_unserved_twh`/`eaas_solar_
>    total_mw`/`eaas_npv_cost` all silently collapse to `None` for this
>    arm, and the summary filter (`:407-408`) matches zero rows for it
>    — `baseline_no_policy` is GAS-2's cleanest test case (no emissions
>    cap confound at all), and this bug guarantees it reports
>    nothing.~~ **RESOLVED at `0012f16` (audit finding A37) — verified
>    directly: the result dict now reads `"financing_regime": "eaas"`,
>    `"status": "optimal"`, fixed as part of the same commit that
>    revived this runner.**
> 3. **Schema gap.** The same function's result dict has no `"solar_
>    total_mw"` key, unlike the NDC-arm schema the other four arms use —
>    `:384`'s `r_yes.get("solar_total_mw")` silently defaults to `None`
>    via `dict.get`, populated for 4 of 5 arms, always blank for the
>    5th. **Confirmed still present, unchanged, now LIVE.**
> 4. **NEW (task_register_amendment): a fourth capital-tier discrepancy,
>    the same shape as the already-fixed A47.** `07_run_gas2_eaas_gas_
>    relief.py`'s `NDC_CASES["ndc3_conditional"]["capital_case"]` is
>    `"tight"` (`:104`), where the canonical convention (`02_run_ndc_
>    caps.py`, `03_run_ndc_eaas.py`, `11_run_rel3_financing_frontier.
>    py`) uses `"moderate"` for every conditional arm. Found during the
>    `0012f16` revival's own verification pass, reported there and left
>    unfixed (a "constant that looks wrong," out of scope for a
>    plumbing-only revival) — confirmed still present, unchanged.

> **Why unresolved.** All four require editing `src/`-adjacent
> `scripts/` files outside this task's whitelist. ~~None is observable
> without a working solve — dormant until `06_`/`07_` are repaired for
> required arguments~~ **NO LONGER TRUE for items 1, 3, 4 — both
> runners are repaired for required arguments (`0012f16`) and can now
> execute; these three items are live and observable on any real run.**

> **Direction of bias.** ~~None realised today (both files dead,
> confirmed).~~ **Realised now for items 1, 3, 4 (both files execute as
> of `0012f16`).** (1) is a divergence risk, not yet triggered — the
> four copies remain byte-identical today. (3) silently blanks GAS-2's
> `baseline_no_policy` `eaas_solar_total_mw` column specifically. (4)
> gives `07_`'s `ndc3_conditional` arm **41.2% LESS** public capital
> than every sibling script's same-named arm (`tight` = $4.552B against
> `moderate` = $7.7384B, per `capital_envelope_scenarios()`), biasing
> that arm toward the envelope **binding** — and therefore toward
> **overstating** measured EaaS relief, since EaaS's modelled value is
> precisely decoupling from a ceiling that binds. The error flatters
> the thesis's own hypothesis, which is the direction that most needs
> declaring.

> **What would resolve it.** When `06_`/`07_` are repaired: ~~(1) import
> the benchmark dict from one shared location~~ (1) import the
> benchmark dict from one shared location (e.g. `src/optimize_
> experiments.py`'s own copy) rather than re-typing it — still
> unresolved; ~~(2) fix the two mislabelled literals at
> `07_run_gas2_eaas_gas_relief.py:187-188` to `"eaas"`/`"optimal"`~~
> **DONE, see item 2 above**; (3) add a `"solar_total_mw"` key to
> `run_no_policy_gas_matrix_eaas`'s result dict, matching the NDC-arm
> schema — still unresolved; (4) align `NDC_CASES["ndc3_conditional"]`'s
> `capital_case` to `"moderate"`, matching every sibling script.

## L36 — `08_run_gas3_regime_ndc_feasibility.py` is dead on required
> arguments by the same mechanism as nine other runners but was never
> itself added to either dead-runner list, and separately carries two
> stale pre-recalibration anchors (Phase 3 A41, A42, A43)

> **Status:** Item 1 RESOLVED at `0012f16` (task_register_amendment).
> Items 2 and 3 remain LIVE, unchanged — and are now MORE exposed than
> before, precisely because the runner they sit in can actually be
> executed today.

> **What.**
> 1. ~~**Undeclared dead-runner membership.** `run_gas_regime_ndc_
>    matrix` (`src/optimize_experiments.py`) requires 20 keyword
>    arguments (confirmed via `inspect.signature`); `08_run_gas3_
>    regime_ndc_feasibility.py:261-268,292-299` supplies only 3, plus
>    optional defaults — `TypeError` on the first call in `main()`, no
>    `argparse` anywhere.~~ **RESOLVED at `0012f16` — `08_run_gas3_
>    regime_ndc_feasibility.py` now has `argparse`/`add_standard_arms`
>    and threads all 17 required keys into both call sites, verified
>    via `inspect.signature(...).bind(...)`.** Register L27 explicitly
>    names `run_gas_regime_ndc_matrix`
>    as serving GAS-1, GAS-2, AND GAS-3, and L27's own corrected
>    dead-runner list (extending L10's nine with `10, 24, 25, 99c`) does
>    not add this file, even in the same breath that names GAS-3 as
>    broken by this exact mechanism. `results/gas3/*` is STILL entirely
>    dated 2026-08-23 13:44 — the same pre-2.6/pre-O&M epoch already
>    flagged elsewhere for `results/baseline/`/`results/ndc/*_voll_
>    mid` — the runner can now regenerate output, but has not yet been
>    run since its own repair, so this stale output remains the only
>    thing on disk today.
> 2. **Stale gas anchor.** `:120-121`, `GAS_ANCHOR_TWH_TH = 40.66` — the
>    PRE-recalibration value; the current, live anchor is 89.27 TWh_th
>    (`data/gas/processed/gas_deliverability_cases.yaml`'s own comment
>    names 40.66 as the superseded figure this script still hardcodes).
>    Actively used: gates `"gas_disrupted"` (`:158`) and `"cap_and_gas_
>    disruption_coincide"` (`:160-164`, the script's own comment: "Key
>    for GAS-3 thesis") — pinned at under half the current anchor, this
>    threshold would essentially never fire correctly against
>    current-scale data even after (1) above is repaired.
> 3. **Stale docstring totals.** `:16-19,29-31`'s "robustness gaps
>    fixed" docstring quotes cumulative gas totals (baseline 853.9
>    TWh_th, downside 640.4, upside 959.6, shock_recovery 914.0; flat_*
>    means 30.50/45.69/43.52) off by roughly 2.3-3.4x from the current
>    `gas_available_power_annual_twh_th.csv` (summed directly: 2,881.6 /
>    1,886.0 / 3,486.1 / 2,716.4; flat means 70.76/105.6/89.18) — the
>    script's own stated justification for its `flat_*` control design
>    no longer describes the data it would run against.

> **Why unresolved.** ~~(1) requires the same `cli_arms.py`-pattern
> repair as the nine (now effectively twelve, per L27/L35) other dead
> runners — out of this task's whitelist and non-trivial.~~ (1) is
> done. (2) and (3) are one-line/docstring fixes in principle but sit
> in a file outside every whitelist available since they were found
> (the `0012f16` revival's own whitelist explicitly excluded constant/
> docstring changes — "change no sweep grid, no constant, no
> computation" — and this task's whitelist is docs-only).

> **Direction of bias.** ~~None realised today (script cannot
> execute).~~ **The runner CAN execute now (`0012f16`). Once it is next
> run without (2) also being fixed, GAS-3's own disruption-coincidence
> diagnostic WILL be silently miscalibrated** — no longer a future risk
> contingent on a repair that hasn't happened; the repair has happened
> and the miscalibration risk is live on the very next run.

> **What would resolve it.** ~~Repair required arguments via `cli_
> arms.py` (as L27's pattern prescribes)~~ — **done, `0012f16`** —
> correct `GAS_ANCHOR_TWH_TH` to 89.27 in a future commit, and refresh
> the docstring's illustrative totals against a fresh run (now
> possible, since (1) is resolved — neither has been attempted yet).
> Also: add this file to L27's own dead-runner enumeration so a future
> reader does not miss it the way this audit nearly did — **also not
> yet done; L27's own amendment (this task) enumerates the ORIGINAL
> nine plus the two remaining library functions, but does not add
> `08_run_gas3_regime_ndc_feasibility.py` to any list, since it was
> never on L10's or L27's original nine to begin with — a fresh,
> explicit cross-reference from L27 to this entry would still close
> that gap and was not made, since L27's own amendment in this task was
> scoped to what TASK.md asked for it specifically.**

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L37 — `05_run_fin3_capital_sweep.py`: a scenario key written and
> never read, and two CSV columns from the same row reporting different
> bases under confusingly parallel names (Phase 3 A34, A35)

> **Status:** LIVE (declared). The dead-key defect is unchanged. The claim in this entry that no finding cites FIN-3 is false; the `budget_usd` / `budget_b_usd` basis hazard now bears on F-31's tier labels. *(Prior status, retained as history: LIVE (declared; this runner is NOT dead — it uses `cli_arms.py` correctly, register L10/A23 — so both defects are live on every run, not dormant).)*

> **Corrected 2026-10-23 (TASK 69 A.2).** The claim 'no finding cites FIN-3' was false: F-31's Source line is `05_run_fin3_capital_sweep.py`, re-derived post-epoch from `results/fin3/arms_92390a63/capital_sweep_results.csv` (`docs/findings_frozen.md`, F-31; audit pass 3 A.2/A.4, `docs/chapter4_audit_pass_3.md`). Nothing else in the entry changed.

> **What.**
> 1. `scripts/05_run_fin3_capital_sweep.py:232` sets `scenario["disco_
>    collection_rate"] = cr_value`, a key nothing in `src/` reads
>    (confirmed by grep). The script's ACTUAL collection-rate mechanism
>    is a separate, hand-computed capital-budget haircut (`:212-217`,
>    `scenario["public_solar_budget_npv"] = effective_budget`), applied
>    before the dead write. `docs/merged_correction_plan_v2.md` (item
>    3.3, `[OPEN]`) already documents `disco_collection_rate` as an
>    unused `scenarios.py` stub intended for a DIFFERENT, demand-side
>    revenue mechanism (`effective revenue = tariff × collection_
>    rate`) — this runner uses a supply-side capital haircut instead,
>    while still writing to the named-but-unwired key.
> 2. `:301-305`: `budget_usd` (the collection-rate-ADJUSTED effective
>    budget) and `budget_b_usd`/`budget_fraction_bstar` (the raw NOMINAL
>    `budget`) differ by roughly 23% at `cr_label="current"` (0.35),
>    despite names reading as unit-rescaled copies of one figure. The
>    console table correctly separates "Nominal B$"/"Effective B$"; the
>    CSV column names do not.

> **Why unresolved.** Both are one-file, plausibly cheap fixes, but
> `05_run_fin3_capital_sweep.py` is outside this task's whitelist.
> Deferred rather than rushed.

> **Direction of bias.** ~~No number currently published depends on either (checked: no "fin3"/"FIN-3" citation anywhere in `findings_frozen.md`).~~ **F-31 is sourced from this runner (see the correction above); the dead `disco_collection_rate` write affects no number, but the `budget_b_usd` / `budget_fraction_bstar` (nominal) against `budget_usd` (effective) basis hazard bears on F-31's capital-tier labels.** A future reader of `capital_sweep_results.csv` who reads
> `budget_b_usd` as a rescaled `budget_usd` would use the wrong (nominal
> rather than effective) capital figure without any signal.

> **What would resolve it.** Remove the dead `disco_collection_rate`
> write (or wire it to something that reads it, if the demand-side
> mechanism `merged_correction_plan_v2.md` 3.3 describes is ever built
> instead of the current supply-side haircut). Rename `budget_b_usd`/
> `budget_fraction_bstar` to make clear they are the NOMINAL basis (e.g.
> `budget_nominal_b_usd`), or add an explicit `budget_effective_b_usd`
> alongside them.

> **Generation (TASK 86):** the hash-keyed result files this entry cites are blended dispatch (no `gas_vintage_dispatch=True` in the cited files' arm records). Checked: `92390a63` (blended). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142. [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L38 — `rpublic_diagnostic.py` hardcodes the wrong solar asset life
> (Phase 3 A33)

> **Status:** LIVE (declared).

> **What.** `scripts/rpublic_diagnostic.py:80`, `LIFETIMES = {"solar":
> 25, "gas": 30, "storage": 15}`, against the live, sourced `asset_
> lifetimes()` (`src/scenarios.py:672-676`, IRENA 2026): `{"solar": 30.0,
> "storage": 15.0, "gas": 30.0}`. `gas`/`storage` match exactly; `solar`
> does not — 25 vs 30, a 20% difference that enters every capital-
> recovery factor in this standalone tool inversely. The tool's own
> Section 5 compounds this: it exists specifically to test exposure to
> "an unsourced asset lifetime propagat[ing] into every CRF... silently"
> and uses 25, not the model's actual 30, as ITS OWN reference point
> (`ref = crf(real, 25)`, printed as "vs L=25").

> **Why unresolved.** `rpublic_diagnostic.py` is a standalone, no-
> repo-dependency exploratory script outside this task's whitelist. A
> one-constant fix, but not authorised this task.

> **Direction of bias.** No persisted output depends on this script (it
> prints to console only, self-contained, no CLI, not called by
> anything). Every solar-related CRF/USD-per-MW figure this tool prints
> is computed against the wrong asset life until fixed — would mislead
> exactly the kind of ad-hoc "is r_public load-bearing" judgement call
> the tool exists to support.

> **What would resolve it.** Change `LIFETIMES["solar"]` from `25` to
> `30`, and `ref = crf(real, 25)` to `crf(real, 30)`, matching `src/
> scenarios.py`'s `asset_lifetimes()` exactly.

## L39 — `14_run_str1_storage_role.py`'s own "re-enable when"
> precondition has already been met, and the file gives no indication
> of this (Phase 3 A57)

> **Status:** LIVE (declared; the underlying SUSPENSION itself remains
> correct and is not being questioned here — see L5).

> **What.** `scripts/14_run_str1_storage_role.py:33-39`: *"RE-ENABLE
> WHEN — Phase 2 introduces representative intra-annual slices...
> storage then shifts day->evening within a season and acquires a
> genuine physical role... STR-1 is rebuilt against the time-sliced
> model."* `src/optimize_model_sliced.py` — self-labelled "Phase 2.1" —
> already exists, is mature, and is imported by roughly a dozen live
> scripts. It implements exactly the mechanism this file says doesn't
> exist yet (`m.storage_season_balance` lets storage charge from midday
> solar and discharge into the same season's evening peak). A future
> reviver reading only this file — exactly what its own suspension
> notice invites — has no way to know the precondition has shipped.

> **Why unresolved.** Updating the notice is cheap in principle, but
> `14_run_str1_storage_role.py` is outside this task's whitelist, and
> whether STR-1 is actually WORTH rebuilding against the sliced model
> (the genset backstop "caps storage's marginal value",
> `optimize_model_sliced.py:632-636`) is a substantive empirical
> question this task is not positioned to answer alongside a text fix.

> **Direction of bias.** None on any published number (STR-1 has been
> suspended throughout; `run_str1_storage_parameter_sweep`'s dead status
> is separately confirmed, A17/L5). The risk is purely to a future
> reviver's time and starting assumptions.

> **What would resolve it.** Update the "RE-ENABLE WHEN" text to point
> at `build_model_sliced()` as already satisfying the structural
> precondition, and state explicitly that whether storage wins
> economically against the genset backstop under slices remains open
> and would need to be tested, not assumed, before STR-1 is actually
> rebuilt.

## L40 — `21_compute_resolution_premium.py` carries a safety check that
> cannot act on the data it is given, and an untimestamped "verified"
> figure in a warning string (Phase 3 A58, A59)

> **Status:** LIVE (declared; distinct from D8, which concerns the
> resolution premium figure itself being stale — both remain true at
> once).

> **What.**
> 1. **Vacuous "same scenario configuration" check.** The docstring
>    (`:15-21`) promises this script "refuses to report unless it can
>    verify... both runs used the same demand arm, gas regime and VoLL
>    case." Check 3 (`:158-166`) tests five field names (`demand_case`,
>    `demand_growth_rate`, `gas_case`, `gas_scenario`, `voll_case`) that
>    exist in NEITHER `results/baseline/summary.json` nor `results/
>    baseline_sliced/summary.json` (confirmed directly against both
>    files' actual keys) — the `and`-chained blocker condition
>    short-circuits every time and the check silently never fires, with
>    or without staleness.
> 2. **Untimestamped figure in a warning string.** `:116-121`: *"Salvage
>    IS in the sliced objective (verified: 18.31 -> 14.286 bn)"* — fires
>    on every run (the sliced diagnostics file has zero salvage-related
>    keys), carries no commit or date stamp, in a file whose own D8
>    entry already warns that the horizon extension moved adjacent
>    salvage-dependent figures.

> **Why unresolved.** Both are plausibly cheap fixes but sit in a file
> outside this task's whitelist. (1) in particular needs a real
> decision (which field names should the check actually test against
> the current `summary.json` schema?), not a one-line patch, so
> deferred deliberately rather than guessed at under this task's "cheap
> and safe only" principle.

> **Direction of bias.** Neither affects the computed `premium`/
> `reportable` output — (1) means the check has never once been a real
> safety property, staleness or not; (2) is text only. Current
> staleness (D8's own Check 4) already blocks this script from
> reporting at all, so nothing is presently misreported; (1) would
> matter the instant a fresh, non-stale pair of baseline files is
> supplied and genuinely differ in scenario configuration.

> **What would resolve it.** Rewrite Check 3 against the field names
> `summary.json` actually emits today (or add the missing fields to the
> writer), and add a commit/date reference to the Section-4-style
> salvage figure in the warning string, or remove the specific numbers
> and state the mechanism only.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L41 — `plot_storage_regime.py` cannot run today, and would fail a
> second, less obvious way if its input path were patched (Phase 3 A31)

> **Status:** LIVE (declared).

> **What.** `RESULT_PATH = Path("results/ndc/ndc_unconditional_20_
> voll_low/diagnostics.json")` (`:6`) does not exist under any current
> output-naming convention (confirmed via direct `ls`) — `FileNotFound
> Error` on the script's second line. Separately: `mapping = {"none": 0,
> "power_limit": 1, "cycle_limit": 2, "solar_limit": 3}` (`:16-23`) does
> not match `storage_binding_by_year`'s actual current vocabulary
> (`"none"`, `"power_limit"`, `"energy_limit"` — `src/optimize_
> experiments.py:2699-2709`); any year classified `"energy_limit"`
> would raise `KeyError` the moment the path defect is fixed. The field
> this script plots is also the one register L11 already declares "not
> meaningful under the sliced builder" and "read by nothing" — this
> script IS a reader, contradicting the "read by nothing" half of that
> claim even while broken.

> **Why unresolved.** Fixing the path is trivial; fixing the vocabulary
> requires deciding what a fixed script should actually show given L11
> already disclaims the underlying diagnostic's meaning under the
> sliced builder — not a decision this task's "cheap and safe" principle
> should make casually, and the file is outside this task's whitelist
> regardless.

> **Direction of bias.** None (script cannot run). The compounding risk
> is that a future, partial fix (path only) trades a loud crash for a
> second loud crash (`KeyError`), and a THIRD, less careful fix (papering
> over the `KeyError` with a guessed `"energy_limit"` mapping entry
> without reading L11 first) could produce a plausible-looking figure of
> a diagnostic the project's own register says carries no real meaning
> under the builder it would be plotted from.

> **What would resolve it.** Fix the path to a real, current `results/
> ndc/` directory; update `mapping` to the current three-value
> vocabulary; and add an explicit comment citing L11, so a future fixer
> does not produce a plausible-looking figure of a diagnostic the
> project's own register already discounts.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L42 — `12_run_dem1_demand_sensitivity.py`'s detailed rationale is an
> orphaned second docstring, declared rather than merged (Phase 3 A54)

> **Status:** LIVE (declared; a one-line "fix" was considered and
> rejected as unsafe this task — see below).

> **What.** `ast.get_docstring()` confirms this file's live `__doc__`
> is only its first 12 lines (the short "[SUSPENDED]" banner). Lines
> 14-101 — the full "RESEARCH QUESTION"/"WHY THE CHOICE OF DEMAND BASE
> MATTERS"/"EXPERIMENT STRUCTURE"/"OUTPUTS" sections — form a SECOND,
> separate string-literal expression statement, never bound to
> `__doc__`: readable by opening the file, invisible to `help()`/pydoc
> or anything introspecting `__doc__` programmatically.

> **Why not fixed as a one-line move.** Considered merging the two
> literals into a single docstring this task; rejected. The orphaned
> block is itself internally stale and self-contradictory — it opens
> with its OWN "STALE NUMBERS" warning (predating the eta correction),
> then continues to cite the PRE-recalibration gas anchor (40.66 TWh_th
> at eta=0.43) that is stale by a SECOND, later recalibration the
> "STALE NUMBERS" warning itself does not know about. Merging this
> content into the live, tool-visible `__doc__` would make confusing,
> self-contradicting numeric content MORE prominent, not less — the
> opposite of what a docstring fix should do. Not "obviously safe" by
> this task's own standard.

> **Direction of bias.** None — the file is fully readable by opening
> it, and the substantive, ACCURATE deprecation reason survives in the
> live, bound docstring regardless of this defect.

> **What would resolve it.** A deliberate rewrite of the orphaned block
> (not a merge) that either strips the stale numeric content entirely
> or refreshes it against current anchors, THEN binds the result as a
> single docstring — a content decision, not a one-line move, and
> better done alongside whatever eventually resolves this script's own
> `NotImplementedError` suspension than as an isolated text edit now.

## L43 — `13_run_dem2_growth_gas_interaction.py`: executable since
> `0012f16` (A55 resolved), but its `eaas`-arm output is WRONG, not an
> error, and must not be used (A56, see L34)

> **Status:** **CLOSED** (TASK 29; code `4ec43b4`, 2026-10-04; `13_` re-run as
> Phase B item 13 on 2026-10-04 at `cded909`, 72 solves, exit 0; recorded
> 2026-10-08). Both halves are closed: A55 (the 16 missing arguments) at
> `0012f16`, and A56 (the `eaas`-arm `required_margin` forced to 1.10) by
> removing the override from `run_dem2_growth_gas_matrix` and from `13_`'s
> `FINANCING_CONFIGS` — the same fix as L34, which holds the full account.
> The `eaas` arm of `13_` is now reportable. Measured effect on this runner
> (F-37, F-38): the EaaS downside interaction +0.80 → +0.93 bn; EaaS value at
> 6.0% growth / downside 63.4 → 63.2 bn; the central case and the whole
> `public_only` arm unchanged at the printed precision.
>
> ~~**Status (original, superseded):** PARTIALLY RESOLVED
> (task_register_amendment). A55 (the
> 16 missing arguments this entry documents) is RESOLVED at `0012f16`.
> A56 (the consequence of naively fixing A55 without also fixing the
> `required_margin` override — predicted by this entry's own original
> text below) is NOT resolved; its root cause is inside `src/optimize_
> experiments.py`, outside every whitelist available since it was
> found. **This runner now executes. Its `public_only` arm is sound.
> Its `eaas` arm produces silently wrong output and must not be read.**~~

> **Verified (task_register_amendment).** `scripts/13_run_dem2_growth_
> gas_interaction.py` now has `add_standard_arms`/`standard_arms_
> kwargs`; `inspect.signature(run_dem2_growth_gas_matrix).bind(...)`
> succeeds with the 16 keys now threaded. Separately re-confirmed `src/
> optimize_experiments.py:1839-1841`'s override
> (`scenario["required_margin"] = fin_cfg["required_margin"]` under
> `financing_regime == "eaas"`) is unchanged, and `13_`'s own
> `FINANCING_CONFIGS["eaas"]["required_margin"] = 1.10` is unchanged —
> **this is the SAME mechanism L34 already documents in full for both
> `13_` and `11_`; see L34 for the complete explanation rather than
> duplicating it here.** Precisely for `13_`: every one of the 36 (of
> 72 total) rows with `financing_arm == "eaas"` carries `required_
> margin` forced to `1.10` regardless of `--dscr-case`. The other 36
> rows (`financing_arm == "public_only"`) never reach the override and
> are sound.

> **What (original text, preserved for the record).** ~~`scripts/13_
> run_dem2_growth_gas_interaction.py:393-400` calls `run_dem2_growth_
> gas_matrix` with only 6 of its 20 required keyword arguments
> (confirmed via `inspect.signature(...).bind(...)`, side-effect-free —
> no execution): `TypeError: missing a required argument: 'peak_
> multiple_case'`. No `argparse` anywhere in the file — no path exists
> to supply the other 14. `main()` fails immediately, before building
> or solving anything.~~ **RESOLVED, `0012f16` — see above.**

> **This is a THIRD, independent trap alongside L34/L35**, named
> together per the audit's own framing: L34 (`11_`/`13_`'s **now-
> active, not dormant** `required_margin` double-count) and L35
> (`06_`/`07_`'s **now-live, not dormant** GAS-1/GAS-2 defects) are
> each traps that fire the moment their runner is repaired; THIS entry
> WAS the reason `13_` could not even reach L34's own defect. ~~Fixing
> this entry's problem... is exactly the shallow fix that would
> silently re-expose L34's `required_margin` double-count in the same
> commit if done carelessly.~~ **This is exactly what happened: `0012f16`
> fixed this entry's problem, and L34's double-count was re-exposed in
> the same commit, precisely as predicted — confirmed above, not merely
> foreseen.**

> **Why unresolved (A56 only — A55 is resolved).** Repairing A56
> requires editing `src/optimize_experiments.py`'s `run_dem2_growth_
> gas_matrix` (removing or reworking the override at `:1839-1841`) —
> outside every whitelist available since the audit found it, including
> this one (docs-only).

> **Direction of bias.** ~~None realised (the script cannot produce a
> single row of output today).~~ **Realised now: every `eaas`-arm row
> (36 of 72) carries a biased `required_margin`, as of `0012f16`.**
> `docs/task_shared_fn_repair.md` already recorded, in an earlier
> repair task, that this runner was deliberately left unrepaired when
> `run_dem2_growth_gas_matrix`'s own signature was hardened — this
> entry was the first place that decision's consequence for `13_`
> itself was recorded in the register, and is now the first place the
> SECOND-order consequence (the double-count firing for real) is
> recorded too.

> **What would resolve it.** ~~Full `cli_arms.py` argparse scaffold for
> `13_`... AND, in the same commit, the `required_margin` fix L34
> describes~~ — **the first half is done (`0012f16`); the second half
> (the `required_margin` fix inside `optimize_experiments.py`) was NOT
> done in that commit, exactly the risk this entry warned about, and
> remains the one thing standing between `13_`'s `eaas` arm and
> reportable output.**

## L44 — A cross-runner field-name mismatch produces a silent NaN in
> `08_phase8_analysis.py`'s own, currently-on-disk master results table
> (Phase 3 A40)

> **Status:** LIVE (declared; confirmed present in real, dated output,
> not merely predicted from the code).

> **What.** `scripts/01_run_baseline.py:359-360` writes the terminal
> storage figure as `"final_storage_capacity_mwh"`. `scripts/02_run_ndc_
> caps.py:370-371` writes the SAME underlying quantity under the
> DIFFERENT key `"storage_capacity_mwh"`. `scripts/08_phase8_analysis.
> py:139` hardcodes one name only: `dv.get("storage_capacity_mwh",
> np.nan)`. Confirmed directly in `results/phase8/master_results.csv`
> (dated 2026-09-15, genuinely regenerated, not a stale artefact): the
> `baseline_no_policy` row's `storage_capacity_mwh` cell is empty (NaN)
> — fed by `01_`'s schema, which lacks that exact key — while all eight
> other rows (fed by `02_`'s schema) correctly show a value.

> **Why unresolved.** The root cause is a naming inconsistency between
> two upstream runners (`01_`, `02_`), both outside this task's
> whitelist and both established, committed conventions with other
> consumers — renaming either is not a "cheap and safe" fix within this
> task's scope. `08_phase8_analysis.py` itself, the file where the
> symptom actually appears, is also outside this task's whitelist.

> **Direction of bias.** Not cited in `findings_frozen.md` (checked, no
> "phase8"/"master_results" citation found) — this is a real defect in
> an actual output file, not (yet) a published number, but the sharper
> and more actionable of the two: a NaN cell in a dated, genuinely-
> regenerated CSV is one copy-paste away from a chapter table.

> **What would resolve it.** Either standardise on one key name across
> `01_`/`02_`'s summary schemas (a coordinated, multi-file change, not
> attempted here), or have `08_phase8_analysis.py` check both names
> (`dv.get("storage_capacity_mwh", dv.get("final_storage_capacity_mwh",
> np.nan))`) — the second is cheaper and more local, and should be
> preferred when this is next in scope.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L45 — The public capital envelope's basis, B\* has no derivation
anywhere in the repository, and the envelope it scales is not an
adequacy constraint

> **Status:** DECLARED NORMALISATION CONSTANT; residual LIVE (TASK 28 Part E.1). The premise "no derivation exists" no longer holds: B\* was DERIVED on 2026-09-20 (10,273,343,610 USD, realised public spend at the traditional-regime slack point; F-22) and, after the rate-coherence and `organic_central` corrections moved the model, re-ruled on 2026-09-25 to a DECLARED NORMALISATION CONSTANT (not re-derived) reported with the current bracket [9,246,009,182 - 9,761,349,093]. **Still live:** the constant is pre-rate-coherence and only bracketed, and `adequacy`/`moderate` tiers are fractions of it (`src/scenarios.py:933`); the self-consistency property it was defined by no longer holds at the current state. The statement that this "blocks `merged_correction_plan_v2.md` §3.5" is superseded by the 2026-09-25 ruling (see the UPDATE at the end of this entry).
>
> **Original status line, SUPERSEDED (kept for the record; TASK 28 Part E.1, 2026-10-04):**
> ~~**Status:** LIVE, undeclared until now. Blocks~~
> ~~`merged_correction_plan_v2.md` §3.5 (`[PARTIAL]` `[MUST]`), whose own~~
> ~~sourcing ledger lists "B\* reconciliation — own derivation, documented~~
> ~~— ❌ blocks 3.5".~~

> **What.**
> 1. **No derivation exists.** `capital_envelope_scenarios()` sets
>    `B_star = 9_104_000_000` and returns `tight` 0.50, `moderate` 0.85,
>    `adequacy` 1.00, `expansion` 1.20 of it. The constant entered at
>    `96b50cf` (2026-04-03, "large refactor with results") with no stated
>    basis. No derivation appears anywhere in `docs/` — every hit is a
>    restatement or a flag.
> 2. **The figure it is said to conflict with was never code.**
>    `git log -S` finds neither `6.13e9` nor `6_130_000_000` anywhere in
>    the repository's history. 6.13 bn existed only as a docstring claim.
>    The ledger item "reconcile B\* (6.13bn docstring vs 9.104bn coded)"
>    is therefore unanswerable as posed: there are not two competing
>    derivations, there is one undocumented constant and one figure that
>    was never coded. **The item should be restated as *derive and
>    document B\**, not *reconcile*.**
> 3. **"Adequacy" is a misnomer.** The envelope charges public solar
>    CAPEX, storage CAPEX and the EaaS subsidy. **Gas CAPEX is not
>    charged to it** (both builders). Adequacy is met by capacity, and
>    gas capacity is free of the envelope, so the constraint has no
>    adequacy interpretation. What it bounds is public *clean-energy*
>    capital.
> 4. **The quantity it bounds is plan-dependent, and the uncapped plan
>    is not the cheap case.** B\* was derived from an uncapped run, which
>    nonetheless requires 10,273,343,610 USD of public clean-energy
>    capital and builds 41,031 MW of solar with 65,613 MWh of storage.
>    Solar at that scale is least-cost in this model without any climate
>    policy; an emissions cap changes the composition and the leakage
>    boundary, not the existence of a large capital requirement. B\* is
>    therefore the requirement of the named uncapped reference plan, and a
>    capped plan's requirement has not been derived.
>    **[Values superseded, TASK 43, 2026-10-10: 41,031 MW and 65,613 MWh are
>    the pre-rate-coherence plan. Current epoch, uncapped reference plan
>    (`results/capital_frontier/capital_frontier_raw.csv`): 39,030.07 MW
>    solar, 63,056.31 MWh storage; the traditional regime's public capital
>    requirement is the bracket 9,246,009,182.24 (last binding) to
>    9,761,349,093.39 (first slack) of realised spend, not 10,273,343,610,
>    which is the 1.0 × B\* envelope.]**
> 5. **The envelope's scope broadened after B\* was set.** Storage CAPEX
>    was added to `public_budget_constraint` at `ddc8f74` (2026-04-26) in
>    the annual builder; the sliced builder was created at `a101aaa`
>    (2026-08-14) with storage already present — both dates after
>    `96b50cf` (2026-04-03), when B\* itself was set. The same nominal
>    tiers then covered a cost category they had not been set against,
>    and no one could observe it, because the diagnostic that would have
>    shown the additional consumption is the one that omits storage
>    (L46).
> 6. **No scope statement.** No arm name or docstring in
>    `capital_envelope_scenarios()` states whether storage is inside
>    B\*'s intended scope. The scenario key is named
>    `public_solar_budget_npv` — "solar" — and was never renamed when the
>    constraint's scope widened.

> **Why unresolved.** Resolving it requires either an own derivation
> against a named reference plan (the ledger's stated requirement) or a
> conscious amendment of that ledger requirement. Both are author
> decisions, not code fixes.

> **Direction of bias.** The tier VALUES are unaffected — `28_` sweeps
> absolute USD envelopes read live from the function, so every solved
> point remains valid, and F-21's binding $7.738 bn envelope is a real
> quantity that really binds. What is unsupported is the NORMATIVE
> reading: that 9.104 bn is a requirement and that 0.85 of it is a
> "moderate" provision. The tiers are also effectively tighter than when
> set, biasing toward the envelope binding and therefore toward
> overstating EaaS relief — the same direction as L35 clause 4.

> **What would resolve it.** (a) Derive the requirement from the model
> under named reference plans — a large finite envelope makes the
> constraint exist without binding, so realised spend reports it in one
> solve per plan (`results/bstar_derivation.json`); (b) report
> capital-envelope results in absolute USD with fraction-of-B\* as a
> secondary axis; (c) retire "adequacy" as a label, or restate it as a
> normalisation constant of stated provenance; (d) restate the ledger
> item as *derive*, not *reconcile*.

> **See also:** L35 (GAS-1/GAS-2 benchmark and capital-tier defects),
> L46 (the storage omission that concealed clause 5), A48
> (the retired B\* propagated into `15_`). GAS-1 runs under a hardcoded
> `capital_case="moderate"` — 0.85 of this same constant — so every
> GAS-1 figure inherits this entry as a scope condition.
>
> **Derived (2026-09-20).** B\* = **$10,273,343,610** — realised public
> clean-energy spend at Frontier B's traditional slack point (uncapped
> central configuration (`carbon_case="no_policy"`, no
> `emissions_cap_by_year`, `solar_build_case="deployment_unconstrained"`),
> sliced builder, `om_central`, traditional financing regime; arms per
> `results/capital_frontier_run.log`).
> The coded 9,104,000,000 **understates the requirement by 12.84%**.
> Clause 3 now has direct support: at the tier named `adequacy` the
> envelope still **binds**, dual **0.2147 USD/USD** (CSV 0.21469358);
> it does not go slack until 1.20 × B\*. Source:
> `results/capital_frontier/capital_frontier_raw.csv`
> (`realised_public_spend_usd` = 10,273,343,610.147528 at
> `envelope_usd` = 10,924,800,000). See F-22.

> **UPDATE (B\* triangulation task, 2026-09-25) — STATUS CHANGED to
> DECLARED NORMALISATION CONSTANT; self-consistency NO LONGER HOLDS.**
> Following the `organic_central` (L54) and `cr_central` concessional
> rate/tenor (L53) corrections, a fresh read of
> `results/capital_frontier/capital_frontier_raw.csv` (re-run, 0
> failures, 2026-09-25) shows the constant's own defining property —
> realised spend equals the envelope exactly, dual zero, at 1.00×B\* —
> **no longer holds**. At `envelope_fraction=0.9` the traditional arm is
> still binding (dual 0.10102134, realised spend 9,246,009,182.24); at
> `envelope_fraction=1.0` (the unchanged constant itself,
> 10,273,343,610) it is already slack (dual 0.0, realised spend
> 9,761,349,093.39) — the true crossover lies somewhere in between, and
> the swept grid (0.05 steps to 0.5, then 0.6/0.7/0.85/0.9/1.0/1.1/1.2)
> does not resolve it further.
>
> **Zeus's ruling: retain B\* = 10,273,343,610 as a DECLARED
> NORMALISATION CONSTANT** (not re-derived, not replaced), stating its
> derivation date and state explicitly wherever cited — **derived
> 2026-09-20, at the pre-rate-coherence, pre-`organic_central`-correction
> state (register L53/L54 postdate it)** — and reporting the current
> bracket **[9,246,009,182 – 9,761,349,093]** alongside it as the
> up-to-date reading, rather than re-running the full envelope-grid
> sweep for a new fixed point that would itself drift again at the next
> parameter correction (`docs/bstar_triangulation.md`, Part C
> recommendation, reasons given both for and against). The tier
> VALUES (`tight` 0.50, `moderate` 0.85, `adequacy` 1.00, `expansion`
> 1.20 of the constant) are UNCHANGED by this update — this is a status
> and documentation change, not a value change; `B_star` in
> `src/scenarios.py` was not edited.
>
> **Externally corroborated in scale, not in a specific figure.**
> `docs/bstar_triangulation.md` triangulates B\* against NIRP 2024's own
> stated investment programme (Part A/B/C of that report) and finds the
> model's bottom-up solar+storage total is **0.68× (ATB capex basis) to
> 0.50× (IRENA capex basis)** of NIRP's own $59bn undiscounted
> solar+storage figure — a COMPOSITIONAL disagreement (model solar
> 0.43× NIRP's dollar allocation, model storage ≥5.3×), not a uniform
> scale difference, traced to the model's zero-capacity-credit
> treatment of solar (`docs/chapters/chapter3_3_draft.md` [removed `009d230`; held at `a993f15`; successor `CHAPTER_3_assembled.md` §3.3] §3.3) and its
> adequacy-driven storage sizing (§3.3.5) against a different storage
> duration basis (NIRP 3hr vs this model's 4.0hr). Storage's UNIT cost
> specifically agrees closely and independently (model $252/kWh vs
> NIRP's implied $272.73/kWh, 7.6% apart) — the one clean external
> agreement the triangulation finds. NIRP cannot and does not supply a
> replacement value for B\* (no NIRP figure is both NPV-basis and
> solar+storage-isolated — see `docs/bstar_triangulation.md` Part A) —
> only a scale-plausibility check, which B\* passes.
>
> **See also:** `docs/bstar_triangulation.md` (full triangulation, all
> three parts), `docs/parameter_provenance.md` (new row, B\*), L53, L54.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L46 — Storage CAPEX was omitted from the realised-spend
diagnostic, which the capital envelope charges

> **Status:** FIXED (`src/optimize_experiments.py`, storage term added
> to `realised_public_spend`, discounted by `m.DF[t]`, guarded by
> `hasattr(m, "storage_add") and econ is not None`; ~:2791). Declared
> here because it governs how capital-envelope binding is measured.
> **The fix is committed at `5beaef0`; this entry landed in
> `2f11f3f`.**

> **What.** `public_budget_constraint` charges public solar CAPEX,
> storage CAPEX and the EaaS subsidy (both builders). The diagnostic
> summed only solar CAPEX and subsidy. Storage was omitted.

> **Magnitude.** At `STORAGE_COST_PER_MWH` = 252,000, across
> Frontier B's binding rows: **20 rows carry non-zero storage; 14 have
> positive duals and would have been reported slack** had the fix not
> been in place. Narrowest margin ~7.5× the 1% classification
> tolerance (traditional, $5.4624 bn envelope, 1,620 MWh); all others
> clear by 25×-400×. Figure is a lower bound — `storage_mwh` is
> end-of-horizon stock, and 2037-2039 rebuild vintages add further
> uncharged capex.

> **Realised impact: none, by one run.** `28_run_capital_frontier.py`
> is the only runner that is both sliced (storage dispatched) and
> envelope-swept. Its first execution post-dates the fix.
> `05_run_fin3_capital_sweep.py` is annual — storage is not dispatched
> and none is built under a binding envelope, so its 2026-08-23 output
> was never affected (realised spend equals envelope exactly on every
> binding row, only possible with storage zero).

> **Direction of bias (pre-fix):** toward under-reporting capital
> scarcity — the envelope looked less binding than it was.

> **Residual.** Scenario key still named `public_solar_budget_npv`
> though the envelope is no longer solar-only. Cosmetic; unrenamed to
> keep the fix isolated. See L45 (B\* basis), A60.

## L47 — The solar capex basis is US-sourced and has not been validated
against any African or Nigerian evidence

> **Status:** LIVE. Declared 2026-09-20. Sensitivity arm available;
> no default changed.

> **What.** Every runner's solar CAPEX is `solar_low`: NREL ATB 2024,
> Utility-Scale PV, Class 5, Market financial case (`src/io.py`,
> `load_solar_capex_by_year`; `data/cost/processed/solar_capex.csv`),
> USD 1,456/kW in 2025 falling to 603/kW in 2045. It is a US basis. No
> Nigerian or African cost evidence enters the model, and there is no
> cost adder of any kind (none exists in the code).

> **The gap.** IRENA (2025) puts Africa's utility-scale solar total
> installed cost at USD 837/kW (`docs/sourcing_pack_02_ledger.md`,
> section 1; secondary citation, unverified). ATB `solar_low` 2025 is
> therefore **at least 1.74x** the African figure (1,456 / 837), and
> `solar_low` does not fall to that level until 2034. **1.74x is a LOWER
> BOUND, not a point estimate:** the comparison mixes ATB in real 2022 USD
> (raw workbook, `Solar - Utility PV`, Base Year 2022) against an IRENA
> figure in a later dollar year, and ATB re-expressed in common dollars
> would be higher still. No deflator is sourced in the repository.

> **What now exists.** `solar_irena_africa` (26 rows, 2025-2050, in
> `solar_capex.csv`): ATB `solar_low`'s year-on-year shape re-anchored to
> USD 837/kW, `837000 * solar_low[t] / solar_low[2025]`, scale 0.5748626;
> 2035 = 393,000 and 2045 = 347,000 USD/MW. It is an ALTERNATIVE arm;
> `solar_low` remains every runner's default and nothing has been re-run.

> **Direction of bias, both ways.**
> (1) The ATB basis overstates the African-average solar cost by at least
> the factor above. If Nigerian cost is near the African average, ATB
> inflates the capital requirement, and hence B\* (L45), and makes the
> public capital envelope bind harder, which FLATTERS EaaS (EaaS's value
> is decoupling from a binding envelope).
> (2) IRENA's African average is dominated by the region's larger, more
> mature markets (South Africa, Egypt, Morocco) and plausibly UNDERSTATES
> Nigerian cost. Whether ATB overstates *Nigerian* cost by the same
> factor is therefore not established; only the ATB-versus-Africa gap
> is bounded.

> **Conclusion.** Neither basis is a Nigeria figure. The spread between
> them is a SENSITIVITY to be reported, not a correction to apply. B\*
> (10,273,343,610 USD) and every capital-envelope result are conditional
> on the ATB basis until that sensitivity has been run.

> **What would resolve it.** A Nigeria-specific installed-cost source; a
> sourced deflator to put ATB and IRENA in common dollars; then the
> sensitivity run (`solar_irena_africa` against `solar_low`) reported
> alongside B\* and the Frontier B results. Sourcing-pack citations are
> secondary and unverified against primaries (see each pack's header).

> **See also:** L45 (B\* basis), L46, `docs/sourcing_pack_01_rates.md`
> (the r_commercial question that interacts with this one: a higher
> commercial rate on an already-high capex basis compounds),
> `docs/sourcing_pack_02_ledger.md`.

> **UPDATE (B\* triangulation task, 2026-09-25) — the blend-versus-level
> distinction STRENGTHENS this entry, not weakens it.** The 1.74×
> figure above (1,456/837, both 2025 LEVELS) is confirmed unchanged and
> independently, via `docs/bstar_triangulation.md`'s own NIRP
> triangulation: nothing in that report's separate computation alters
> or contests it. That report also computes a SECOND, different ratio —
> the model's own realised solar spend, build-weighted across
> 2025-2045, divided by MW built ($618.41/kW, ATB basis) — and found it
> sits BELOW the $837/kW IRENA level this entry compares against,
> despite resting on the same ATB `solar_low` series this entry already
> flags as 1.74× the IRENA level. **This is not a contradiction of the
> 1.74× finding; it demonstrates the finding's own point more sharply**:
> the model's blended cost looks cheap only because its build is
> back-loaded onto the DECLINING ATB series' late, cheap years — the
> underlying 2025 PRICE BASIS this entry flags as expensive-relative-to-
> Africa is unchanged, and would apply in full to any build concentrated
> in the model's early years instead. A reader comparing the model's
> blended $618/kW against IRENA's $837/kW and concluding "the model's
> solar is already cheaper than the African reference, so this entry's
> concern is moot" would be **drawing the wrong conclusion from mixing
> a blend against a level** — exactly the error this update corrects
> before it could be made. The two ratios (1.74× level-vs-level; 0.67×
> blend-vs-NIRP's-own-blend) are BOTH true, are NOT the same comparison,
> and this entry's own 1.74× finding is the one that answers what this
> entry is actually about — whether the PRICE basis is validated against
> African/Nigerian evidence. It is not, still. See
> `docs/bstar_triangulation.md` Part C, "Does the model's implied solar
> unit cost..." for the full two-pairing derivation.

## L48 — GAS-1's benchmark comparison: five declared limitations, and a
hardcoded binding-years denominator

> **Status:** LIVE. GAS-1 (F-30) is reportable with these declared. The hardcoded binding-years denominator described below is fixed; F-30's NDC legs carry no leakage flag (L51). *(Prior status, retained as history: LIVE. GAS-1 (F-30) is reportable with these declared.)*

> **Corrected 2026-10-23 (TASK 69 A.2).** The `/21` denominator defect is fixed: the header of `scripts/06_run_gas1_shadow_benchmarks.py` (lines 30-44) records TASK 20 Part B making both arms report evaluable years (`len(gs_vals)`); the file prints `binding=n/evaluable` (`:275`). The other four declared limitations, the hardcoded `capital_case="moderate"`, and the absence of a leakage check on the NDC legs (L51; audit pass 3 B.5) stand.

> **1. Asymmetric truncation, against power.** The shadow price is
> `d(system cost)/d(gas delivered)` — marginal system cost *avoided*.
> Demand is exogenous and inelastic: every MWh is served regardless, by
> grid or by diesel at VoLL. **The model therefore cannot value
> electricity that would not otherwise exist** — suppressed industrial
> output, productivity, firms that never open. The export benchmark is
> marginal *revenue*, a complete measure of its own side. **Consequence:
> the exceedance share is a LOWER BOUND on the case for domestic
> allocation, not a complete comparison.** Stating it that way is
> stronger than defending the comparison as complete.

> **2. Benchmark provenance — mixed, all declared.** `dom_power_2p4` and
> `dom_commercial_2p9` are SOURCED to the NMDPRA Domestic Base Price
> series (power $2.42 → $2.13 → $2.18; commercial $2.92 → $2.63 → $2.68
> for 2024 → 2025 → 2026; `docs/sourcing_pack_01_rates.md`, section 7;
> secondary citation, unverified) and are rounded 2024 values, so stale-
> high; the bias is conservative. `dom_high_3p3` is ASSUMED — it sits
> above the regulated industry ceiling and matches no tier. `opp_3/5/7`
> are ASSUMED. The delivered LNG anchor (10.05, used for the netbacks) is
> SOURCED (interim) to a single month's Japan landed spot (pack 01,
> section 4) and should be superseded by a JKM or TTF assessment over a
> stated window. The `lng_export_10` continuity key is 10.00, not 10.05.
> The GECF liquefaction and World Bank shipping/regas components are in
> pack 01, section 8 (vintage unverified).

> **3. Dimensional inconsistency — CORRECTED.** The shadow price is a
> marginal value at the domestic delivery point; `lng_export_10` is a
> *delivered* price. Comparing them overstated the export side by the
> full liquefaction and shipping cost. Netbacks are now derived
> (`src/gas_benchmarks.py`), basis-explicit (DES: regas not deducted;
> HUB: deducted), from liquefaction (2.86, band 1.0–3.5, GECF 2026) and
> shipping (0.70, Nigeria → NW Europe, World Bank). **Measured effect
> at the central netbacks: one year in thirty** (36.7% delivered → 40.0%
> DES/HUB central). The correction matters for correctness, not for the
> conclusion.

> **4. Static benchmarks over thirty years.** Every benchmark is a
> single scalar applied unchanged from 2025 to 2054, against a shadow
> price that moves year by year. No escalation, no discounting in the
> comparison.

> **5. Configuration scope condition.** `run_no_policy_gas_matrix`
> hardcodes `capital_case="moderate"`, `demand_level_case="served"`,
> `demand_case="organic_central"` and `carbon_case="no_policy"` inside
> the function rather than threading them from the CLI. GAS-1 therefore
> runs under a binding public capital envelope at 0.85 × B\*, unlike
> frozen central. **This is not inert:** the mean is sensitive to B\*
> through the hardcoded `capital_case="moderate"`; the pre-B\* artefact
> was not retained, so the magnitude is not measured.

> **Defect found in the same run, separately declared:** the binding-
> years denominator is hardcoded twice — `/30` at
> `06_run_gas1_shadow_benchmarks.py:243` (no-policy path) and `/21` at
> `:259` (NDC path). `gas_shadow_by_year.csv` confirms **30 years
> (2025–2054) for every case**, so `/21` is stale. NDC `downside`
> binding is **18/30 = 60%**, not 18/21 = 85.7%. ~~Not yet fixed.~~ **Fixed (TASK 20 Part B, `docs/task_log_20b_binding_denominator.md`): both arms report evaluable years dynamically.**

> **See also:** L35 (item 1, benchmark dictionary now single-sourced),
> L45 (B\* basis), F-30, F-9.

## L49 — FIN-3's "bottleneck" test keys on a quantity that cannot move,
so its printed conclusion is unsupported

> **Status:** LIVE. The printed interpretation must not be cited.

> **What.** `05_run_fin3_capital_sweep.py` reports a "bottleneck
> disappears at" capital case and prints, at every collection rate:
> *"Public capital sufficient at this collection rate. EaaS is
> CONVENIENT but not strictly necessary."* Unserved energy is `0.000 TWh`
> at all fifteen solves. The bottleneck test keys on unserved, which under
> a priced genset case is fixed at zero by construction — neither priced at
> VoLL nor free. The test therefore cannot fire for a structural reason
> rather than a behavioural one, which makes the conclusion stronger, not
> weaker: it is incapable of finding insufficiency at any capital case.

> **Consequence.** The test cannot fire. It is not finding sufficiency;
> it is structurally incapable of finding insufficiency, and it returns
> `tight` — the first, most constrained case — for every arm
> (`capital_sweep_summary.json`: `bottleneck_disappears_at: "tight"`).

> **It contradicts the duals in its own output.** The same table reports
> budget shadow prices of 6.39 / 2.87 / 0.40 (no_friction: tight /
> moderate / adequacy) and **7.82 / 4.56 / 3.92 / 1.31** (current: tight
> / moderate / adequacy / expansion). A dual of 7.82 says one more dollar
> of public capital would cut system NPV cost by $7.82. That is capital
> scarcity, reported alongside a verdict of sufficiency. The leakage
> results (F-31) say the same: 42–50% of emissions sit outside the cap at
> `tight`.

> **Direction of bias.** Toward understating the case for EaaS — the
> opposite of the direction a reader would suspect the thesis of
> favouring, but wrong either way.

> **What would resolve it.** Key the test on the **budget shadow price**
> (bottleneck where the dual exceeds a stated threshold), not on
> unserved energy. Until then the printed sentence is an artefact and
> the duals are the reportable quantity.

## L50 — `within_block_peaking_factor` is a builder-side `dict.get` default
with no scenario entry, no arm and no provenance record, and it multiplies
firm capacity in every binding year

> **Status:** LIVE. Declared 2026-09-20 from the methodology claims ledger
> (C-5, docs/methodology_claims_ledger.md, Part 3 row 146). **Forensic
> audit A62** (`docs/forensic_audit.md`). Not fixed.

> **What.** The firm-capacity constraint is
> `Σ capacity_credit × capacity ≥ peak_mw[t] × within_block_peaking_factor ×
> (1 + reserve_margin[t])`. The factor is read as
> `float(scenario.get("within_block_peaking_factor", 1.3448))` in BOTH
> builders (`src/optimize_model_sliced.py:1262-1263`;
> `src/optimize_model.py:1039-1040`). `load_scenario()` never sets the key
> (`scenarios.py` mentions it only in comments: "read inside the builders,
> NOT set here"), no runner sets it (grep of `scripts/`: none), no arm or
> CLI flag exposes it, and it appears in no output tag. Every result at
> every arm therefore uses the literal 1.3448.

> **Provenance.** `methodology_physical_layer.md` §2 calls it "DERIVED from
> the same dataset": 1.4588 (instantaneous PAR, 2016 unsuppressed series)
> ÷ 1.0847 (block PAR) = 1.3448. The inputs exist only in comments and
> the register (L2); the instantaneous PAR is not computed anywhere in the
> code. At HEAD `load_scenario()` derives the block PAR as 1.0877, not
> 1.0847 (ledger C-4), so the same ratio would be 1.3411 (−0.28%).

> **Why it matters.** It is the same silent-default failure class that
> `methodology_protocols.md` §1 says the required-key discipline exists to
> prevent (cf. the retired `solar_tariff_case` 65M default): a
> load-bearing constant reached through `dict.get(key, DEFAULT)` inside a
> builder, invisible to every runner. It scales the firm-capacity
> requirement in every binding year (20 of 30 years at the central
> configuration, 2035-2054; dual 23,925-80,309 USD/MW) and is the single
> multiplier that converts the block-average peak into the instantaneous
> peak the constraint should size against.

> **It is arm-independent though its derivation is not.** The 1.3448 was
> derived from the UNSUPPRESSED series only, yet the builders apply it
> unchanged under `peak_suppressed` (register L2: instantaneous PAR
> 1.6891 vs block 0.9807, ratio 1.722) and `peak_high` (a constructed
> shape whose instantaneous peak is not measured). Sweeping the peak arm
> therefore changes the block peak but never the within-block factor.

> **Direction of bias.** Not established. A larger factor raises the
> required firm capacity and hence storage/gas cost; the 1.3411 that
> HEAD's own block ratio implies would lower it by 0.28%. The
> suppressed-arm ratio (1.722) would raise it by 28%.

> **Comment contradicts code in BOTH builders.** `src/optimize_model.py`
> :1032-1035 and `src/optimize_model_sliced.py:1255-1261` say "SOURCE:
> [PENDING]; DEFAULT 1.0 is a declared conservative simplification
> (assumes no within-block peaking above the block average)" — but the
> code that follows (:1039-1040 and :1262-1263) defaults to 1.3448, not
> 1.0. A reader trusting the comment would believe the builders add no
> within-block peaking; they add 34.5%.

> **What would resolve it.** Promote it to a required, arm-tagged
> load_scenario parameter (or derive it per peak arm from the shape data)
> with its provenance stated at the definition site, and re-run the
> regression anchor to confirm the promotion is value-preserving (per the
> protocols document's promotion rule).
>
> **See also:** L1, L2, C-4, C-5.

## L51 — The leakage diagnostic is not applied to every capped run

> **Status:** LIVE.

> **What.** `methodology_genself_backstop.md` §4.4 states the post-solve
> leakage check is applied to every run with a real cap. It is not.
> `leakage_diagnostic()` (`src/optimize_model.py:1759`, threshold 0.10) is
> called by `02_run_ndc_caps`, `03_run_ndc_eaas`, `05_run_fin3_capital_sweep`
> and `15_` (and also by `18_` and `99c_`), but three scripts
> that pass a real `emissions_cap_by_year` never call it:
> `24_run_leakage_arm.py:317`, `25_run_obligation_trajectory.py:315`,
> `26_run_f3_identification.py:273`. Four further scripts that appear capped
> pass `emissions_cap_by_year=None` and are uncapped, so no leakage check
> applies to them by construction (`02_run_eaas:291`,
> `20_run_cf1_solar_yield_sensitivity:135`, `23_run_selfgen_frontier:468`,
> `29_run_parameter_uncertainty:476`).

> **Consequence.** Capped results from those three carry no leakage check, so
> a non-compliant plan can be reported without the flag. Note separately that
> Frontier B (`28_`) applies no emissions cap at all, so leakage is undefined
> there rather than unchecked (see C-13 in
> `docs/methodology_claims_ledger.md`).

> **UPDATE (TASK 32, 2026-10-08): the count "three" is out of date.** At HEAD
> the diagnostic is also absent behind **shared experiment functions** that
> pass a real cap and never call it (no reference to `leakage_diagnostic` in
> `src/optimize_experiments.py`). The scripts that reach a real cap without the
> diagnostic are **nine**: directly, `24_run_leakage_arm.py:319`,
> `25_run_obligation_trajectory.py:317`, `26_run_f3_identification.py:273`
> (the line numbers above have moved by two for the first two); through shared
> functions, `04_run_fin2_tariff_sweep.py` (`run_fin2_blended_bankability_sweep`),
> `06_run_gas1_shadow_benchmarks.py`, `07_run_gas2_eaas_gas_relief.py` and
> `08_run_gas3_regime_ndc_feasibility.py` (`run_gas_regime_ndc_matrix`),
> `11_run_rel3_financing_frontier.py` (`run_rel3_financing_frontier`) and
> `13_run_dem2_growth_gas_interaction.py` (`run_dem2_growth_gas_matrix`). Their
> NDC legs carry no leakage flag; findings affected: F-30 (NDC legs), F-34, F-35,
> F-37, F-38 (and F-12 via `26_`). `17_` (F-36) records `leakage_flagged`;
> `09_`/`10_` (REL-1/REL-2) and `28_` pass no cap. The dead `run_dem1_demand_
> sensitivity` and `run_str1_storage_parameter_sweep` also pass caps but have no
> live caller. Direction of bias unchanged.
>
> **Re-verified (TASK 34, at `85d820d`):** the nine-script list above is current. The only change since TASK 32 is that `05_run_fin3_capital_sweep.py` is now Phase B item 23; it calls `leakage_diagnostic` (as it always did), so it is not among the nine, and its re-run flagged 8 of 15 cells.

> **Direction of bias.** Toward under-reporting non-compliance.

> **What would resolve it.** Either call the diagnostic in the three capped
> runners, or amend §4.4 to state which runners apply it and why. (Now: call it
> inside the shared functions, which fixes six of the nine at once.)

## L52 — The objective's commercial rate (`cr_central`) and the
bankability blend's commercial rate price the same quantity 0.11 vs
0.18, and both may sit below the sourced floor

> **Status:** LIVE — declared, not resolved. Declared 2026-09-2x from
> the methodology claims ledger (A4, `docs/methodology_claims_ledger.md`).
> **Forensic audit A61** (`docs/forensic_audit.md`). No value changed by
> this entry.

> The objective's commercial rate (`cr_central`, `r_comm` = 0.11,
> feeding the F1 tranche CRF) and the bankability blend's commercial
> rate (`r_comm` = 0.18) are not merely two rates for two mechanisms: **a
> WACC cannot lie below its own cost of debt.** CPI (2023) gives
> climate-project cost of debt at 18.5% (Jamaica, B+) and 19.0%
> (Bangladesh, BB-), with Nigeria rated at or below B, implying a
> commercial WACC near 20% (about 16% after a tax shield). The 0.11
> rests on an IEA 8.5-9% WACC for Kenya and Senegal plus a 2-2.5pp
> uplift that `capital_recovery_scenarios()`'s own docstring marks as an
> analyst estimate, not sourced. `cr_comm_high` (0.14) also sits below
> the CPI range. Unresolved; affects every F1 objective result using
> `cr_central` / `cr_ida21` / `cr_comm_*`. See
> `docs/sourcing_pack_01_rates.md` §2.

> **Why unresolved.** This is a genuine open sourcing question, not a
> coding error: two different literatures (IEA Cost of Capital
> Observatory regional averages vs CPI sovereign-rating-based cost of
> debt) were used for what the code's own comments describe as the same
> conceptual quantity, and the discrepancy was noted in code
> (`src/scenarios.py:152-156`, commit `85f2598`) without being resolved
> or escalated to the register until now.

> **Direction of bias, if the CPI figures are the correct anchor.** Both
> 0.11 and 0.18 would be understated, which would understate the true
> cost of both public F1 commercial-tranche capital and EaaS private
> capital — direction on the net EaaS-vs-public comparison not
> established without re-solving, since both channels' commercial cost
> would rise together.

> **What would resolve it.** An author decision on which sourcing
> anchor governs `r_comm`/`r_commercial` project-level Nigerian
> commercial power-sector debt, followed by a re-run of any result that
> depends on `cr_central`, `cr_ida21`, `cr_comm_*`, or the bankability
> blend, per this task's FORBIDDEN clause: no value is changed by this
> entry itself.
>
> **See also:** L17, C-4.

> **UPDATE (rate-coherence task, Part 1, 2026-09-25) — the MECHANISM
> half of this entry is closed; the SOURCING half is not.** This entry
> conflated two distinct problems: (1) two different code mechanisms
> reading two different literals for "the same quantity," and (2)
> whether either literal (0.11 or 0.18) is itself defensible against
> the CPI cost-of-debt floor. `bank_tenor_case` (L17) resolves (1) for
> every live path except FIN-2: under the new default
> `bank_tenor_comm`, the bankability discount factor now reads
> `r_comm_nominal` — the SAME `cr_central` value (0.11) the objective's
> own capital charge uses — not `blended_finance_scenarios()`'s 0.18.
> The two mechanisms price commercial capital identically now, for
> every caller that does not explicitly pin `bank_tenor_legacy`.
> **Problem (2) is completely untouched**: whether 0.11 sits below its
> own sourced cost-of-debt floor is a sourcing question this task does
> not decide, and 0.18 (still naira-basis, per L17) still governs FIN-2
> specifically, since FIN-2 is pinned to `bank_tenor_legacy`. No value
> in `capital_recovery_scenarios()`'s commercial fields (`r_comm`)
> changed in this task — see the FORBIDDEN clause of
> `docs/task_log_rate_coherence.md`'s governing task.

## L53 — The objective's concessional rate (`cr_central`) and the
bankability blend's concessional rate price the same instrument at two
different World Bank IDA vintages, 2.80% (IDA20) vs 2.12% (IDA21), and
a same-vintage alternative already exists unused

> **Status:** RESOLVED (rate-coherence task, Parts 2 and 3, 2026-09-25; TASK 28 Part E.1). `cr_central`'s concessional rate moved from 2.80% (IDA20) to 2.12% (IDA21) and its `L_conc` from 30 to 25 years, matching IDA21's own maturity ("TENOR MISMATCH NOW CLOSED" in this entry's own UPDATEs); the objective and the bankability blend now price the same instrument at the same IDA vintage. The net effect on the concessional capital charge was +8.2% against the IDA20 baseline, not the decrease first predicted. What still needed re-running (the central sweep and anchors) was done: see L58 and Phase B item 0.
>
> **Original status line, SUPERSEDED (kept for the record; TASK 28 Part E.1, 2026-10-04):**
> ~~**Status:** LIVE — declared, not resolved. Declared as the~~
> ~~concessional twin of L52/A61. No value changed by this entry.~~

> **Correction to the premise this entry was requested under: `r_conc`
> (2.80%, `cr_central`) is NOT unsourced.** `capital_recovery_scenarios()`'s
> own docstring (`src/scenarios.py:1877-1879`) marks it "SOURCED. World
> Bank IDA Terms, USD single-currency blend, FY24Q3 (IDA20, DARES
> vintage Dec-2023): 2.80% all-in (1.39% service + 1.41% interest), 30yr
> maturity / 5yr grace." This is independently corroborated: claims
> ledger row 68 verdicts it VERIFIED against the same docstring and
> value. **Both rates in this pair are sourced.** The actual defect is
> that they are sourced to two different vintages of the same
> instrument (World Bank IDA blend terms for Nigeria) and never
> reconciled.

> **What.** The F1 objective's concessional CRF uses `r_conc_nominal` =
> `capital_recovery_scenarios()["cr_central"]["r_conc"]` = **0.0280**
> (`scenarios.py:1899`), IDA20, FY24Q3, DARES vintage Dec-2023. The EaaS
> bankability constraint's blended WACC uses `r_concessional` =
> `blended_finance_scenarios()["blended_central"][1]` = **0.0212**
> (`scenarios.py:167-168`), IDA21 Blend Fixed Rate, effective
> 2026-01-01, sourced to `docs/sourcing_pack_01_rates.md` §1 (itself a
> secondary citation, per that file's own header). A 32% difference in
> the concessional rate (0.0280 vs 0.0212) between the two mechanisms,
> with no documented reason to use a different IDA replenishment cycle
> in one than the other.

> **A same-vintage reconciliation already exists in the code, unused as
> the central arm.** `capital_recovery_scenarios()` already carries
> `"cr_ida21": {"r_conc": 0.0213, "L_conc": 25, "r_comm": 0.11,
> "L_comm": 10}` (`scenarios.py:1900`) — the IDA21 rate, matching the
> bankability blend's vintage to the digit (0.0213 vs 0.0212-0.0213
> band). The central configuration (`cr_central`) does not select it;
> it uses IDA20 instead. No comment in either function states why the
> objective anchors to IDA20 while the bankability constraint anchors
> to IDA21, or why `cr_ida21` — which would remove the mismatch — is
> not the central arm.

> **Tenor mismatch travels with the rate mismatch.** IDA20 (`cr_central`)
> carries 30yr maturity / 5yr grace; IDA21 (`cr_ida21`, and the
> bankability blend's own sourcing) carries 25yr maturity / 5yr grace.
> `L_conc` in the F1 objective (30) and the tenor implicit in the
> bankability blend's IDA21 sourcing (25) therefore also differ, for
> the same underlying reason.

> **Every consumer of `r_conc_nominal` (the objective-side rate),
> traced exhaustively:**
> 1. `src/scenarios.py:3247` — `load_scenario()` sets
>    `scenario["r_conc_nominal"] = _cr_terms["r_conc"]`, where
>    `_cr_terms = capital_recovery_scenarios()[capital_recovery_case]`.
> 2. `src/optimize_model.py:1280` (annual builder, `f1_tranche` branch)
>    — reads `scenario["r_conc_nominal"]`, Fisher-deflates it
>    (`:1286`), feeds `_crf_conc` (`:1331`).
> 3. `src/optimize_model_sliced.py:1559` (sliced builder, `f1_tranche`
>    branch) — identical pattern: read (`:1559`), deflate (`:1565`),
>    feed `_crf_conc` (`:1610`).
>
> No script reads `r_conc_nominal` directly — `scripts/03_run_ndc_eaas.py`,
> `04_run_fin2_tariff_sweep.py` and `15_run_pol1_ndc_comparison.py` only
> pass `capital_recovery_case` through to `load_scenario()`
> (`grep -rn "r_conc_nominal" scripts/*.py`: no hits). Downstream,
> `_crf_conc` feeds exactly four capital-charge Expressions per builder:
> `m.cost_public_solar_capital_charge_npv`,
> `m.cost_eaas_solar_capital_charge_npv`,
> `m.cost_gas_capital_charge_npv` (always zero — gas is
> commercial-only, `m.gas_conc_vol` fixed at 0), and
> `m.cost_storage_capital_charge_npv` — active only under
> `capital_treatment_case="f1_tranche"`, the central configuration.
> Confirmed identical by direct comparison of both builder files.

> **Why unresolved.** Already flagged, undocumented, in the claims
> ledger's own coverage audit: row 5a records "the blend/bankability
> `r_concessional` (IDA21) and the two-vintage split are not documented"
> in any of the five methodology documents. No register entry existed
> for it before this one.

> **Direction of bias.** Not established without a decision on which
> vintage is correct. If IDA21 (the more recent, and the one the
> bankability blend already uses) is the right anchor throughout, the
> objective's concessional CRF is currently charged at a **higher**
> rate (2.80% vs 2.12%) over a **longer** tenor (30yr vs 25yr) than it
> should be, which would overstate the F1 tranche's concessional
> capital charge and understate how cheap concessional-financed
> solar/storage is relative to what is currently reported.

> **What would resolve it.** An author decision on whether the F1
> objective's concessional rate should track the same IDA vintage as
> the bankability blend (in which case `cr_central` should use
> `cr_ida21`'s terms, or `cr_ida21` should be promoted to the central
> arm), or whether the two are deliberately different because they
> represent different disbursement mechanisms — and if so, that
> rationale needs to be written down, not left implicit. No value is
> changed by this entry.
>
> **See also:** L52, A61, C-4.

> **UPDATE (rate-coherence task, Part 2, 2026-09-25) — rate moved
> first, tenor followed.** Zeus's ruling: adopt 2.12% (IDA21) as the
> single concessional rate. `capital_recovery_scenarios()["cr_central"]["r_conc"]`
> changed 0.0280 → 0.0212. The objective's per-tranche concessional CRF
> and the bankability blend's `r_concessional` now price concessional
> capital IDENTICALLY on rate (0.0212 both). Justification, written
> into `capital_recovery_scenarios()`'s docstring verbatim: *"DARES is
> modelled at current IDA terms (IDA21, 2.12%) rather than its IDA20
> origination vintage (2.80%), because the model evaluates forward
> pathways rather than repricing an existing commitment. The IDA20 rate
> is retained in the register as the superseded value."*
>
> Applying the rate alone left `cr_central`'s `L_conc` at IDA20's 30yr
> — a hybrid, IDA21's rate on IDA20's tenor, flagged in this entry at
> the time and deliberately NOT corrected in that same pass (the
> instruction named the rate only).

> **UPDATE (rate-coherence task, Part 3, same day) — TENOR MISMATCH NOW
> CLOSED, not merely declared.** `cr_central`'s `L_conc` changed
> 30 → 25, matching IDA21's own maturity. `cr_central` now carries a
> single vintage's terms throughout (2.12%/25yr), not a hybrid.
>
> **Why the hybrid could not stand — arithmetic, both tenors, confirmed
> by direct computation** (`CRF(r,L) = r(1+r)^L/((1+r)^L-1)`,
> `r = 0.1176%` real, deflated from 2.12% nominal at `us_inflation =
> 0.020`):
>
> | tenor | CRF | vs IDA20 baseline (2.80% nom / 30yr, CRF 0.037538) |
> |---|---|---|
> | 30yr (the hybrid, now retired) | 0.033943 | **−9.6%** |
> | 25yr (`cr_central` now) | 0.040616 | **+8.2%** |
>
> The tenor works AGAINST the rate: shortening the recovery window from
> 30 to 25 years raises the annuity factor even though the rate fell,
> because the same principal is recovered over fewer years. **The
> hybrid (30yr) priced concessional capital cheaper than either real
> vintage** — cheaper than the old IDA20 terms (2.80%/30yr, CRF
> 0.037538) AND cheaper than the correct IDA21 terms (2.12%/25yr, CRF
> 0.040616) — which is exactly why a rate-only correction could not be
> left standing on its own.
>
> **Effect on $L_{conc}^{eff}$ per channel**
> ($L_{conc}^{eff}(\text{channel}) = \min(L_{conc},
> \text{asset\_lifetimes}[\text{channel}])$, `asset_lifetimes()`:
> solar 30yr, storage 15yr, gas 30yr but commercial-only throughout so
> moot here):
>
> | channel | before ($L_{conc}=30$) | after ($L_{conc}=25$) |
> |---|---|---|
> | solar | $\min(30,30)=30$ | $\min(25,30)=25$ — the ceiling no longer binds; the arm's own 25yr now governs |
> | storage | $\min(30,15)=15$ | $\min(25,15)=15$ — **unaffected**, the technical-life ceiling already bound tighter than either tenor |
>
> **`cr_ida21` — now near-identical to `cr_central`, not redundant.**
> Term-by-term: `cr_central` (0.0212, 25yr, 0.11, 10yr) vs `cr_ida21`
> (0.0213, 25yr, 0.11, 10yr) — identical in every field except `r_conc`
> in the fourth decimal. That difference is not noise: it is IDA21's
> own two 2026 effective dates (2.12% from 2026-01-01, `cr_central`;
> 2.13% from 2026-07-01, `cr_ida21`), confirmed by direct computation
> to move CRF by 0.127% relative (`CRF(0.0213 nom, 25yr) = 0.040666`
> against `cr_central`'s `0.040616`) — a real, if narrow, distinction
> between two effective dates of the SAME IDA21 facility, not two
> different instruments. Retained, not deleted, per instruction: this
> is a case where the arms differ only by effective date, and that is
> stated rather than used as a reason to delete either.
>
> **Arithmetic effect, full sequence (both Part 2 and Part 3, cumulative
> against the original IDA20 baseline).** CRF: 0.037538 (IDA20, 2.80%/30yr)
> → 0.033943 (hybrid, 2.12%/30yr, −9.6%) → 0.040616 (IDA21, 2.12%/25yr,
> **+8.2% net against the IDA20 baseline**). The concessional capital
> charge per dollar of `conc_vol` is therefore HIGHER, not lower, than
> the pre-correction figure, for every channel not ceiling-bound by
> `asset_lifetimes` (solar; not storage, per the table above) — the
> opposite direction from Part 2's rate-only reading, which understated
> the true effect by looking at the rate in isolation.
>
> **Direction-of-bias statement above, fully resolved.** This entry
> originally predicted the objective's concessional CRF was overstated
> by BOTH a higher rate and a longer tenor than the bankability blend's
> vintage. The rate WAS overstated and is now corrected. The tenor
> claim does not carry through unchanged: correcting the tenor to
> IDA21's true 25yr RAISES the CRF relative to the 30yr hybrid (shorter
> window, same principal), so the net effect of fully correcting to
> IDA21 terms is a net increase in the concessional charge relative to
> the old IDA20 baseline, not a straightforward decrease. The two-error
> framing this entry used ("higher rate over a longer tenor") was a
> useful diagnostic but not a linear one — the corrected tenor's own
> effect dominates the corrected rate's.
>
> **What still needs re-running.** Unchanged from the note above: every
> result built under `capital_recovery_case="cr_central"` and
> `capital_treatment_case="f1_tranche"` — the declared central
> configuration (`scripts/run_central_sweep.sh`) and everything
> downstream, including the `results/anchor_central.log` central
> anchor (**L58**). Not run here.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L54 — The demand-growth income elasticity was carried without a
locator through two citations before being traced against its primary;
`organic_central` has moved as a result

> **Status:** RESOLVED as far as the current arm goes; the process
> defect (citing a paper with no table/page/locator) is the reason this
> entry, and `docs/parameter_provenance.md`, exist. Declared
> 2026-09-25.

> **The value and where it is set.** `demand_growth_scenarios()["organic_central"]`,
> `src/scenarios.py:547` — **now 0.039** (was 0.040). Feeds
> `demand_growth_prior()`'s `mean` (`:650`, by reference, so it moved
> automatically) and every runner default via `demand_case="organic_central"`
> (`src/scenarios.py:2179`). Derived from Route A: `g = ε·g_GDP +
> (1-ε)·g_pop`, ε = 0.80, g_GDP = 4.5% (declared assumption), g_pop =
> 1.77% → 3.95%; Route B (per-capita convergence to Kenya) → 3.67%;
> adopted 3.9%.

> **Citation history — the real sequence, not the one an earlier task
> draft asserted.** Attributed to Liddle, Smyth & Zhang (2020), *Energy
> Economics* 86, 104681 → corrected to Liddle & Hasanov (2023),
> *Environmental and Ecological Statistics* 30(4), 827–849, **on Chat's
> inference from search results**, without a locator → traced against
> the primary PDF (`~/thesis-code/literature/Liddle_Hasanov.pdf`,
> `.gitignore`d, not committed). **No revert to LSZ (2020) was ever
> applied** — a revert was proposed in an earlier, unexecuted
> instruction and then mistakenly written into a later task's preamble
> as history; that revert never happened in this repository, and its
> non-occurrence was fortunate, since LSZ (2020) itself fails to trace
> (see below).

> **Trace verdict (full detail: `docs/task_log_elasticity_sourcing.md`,
> Part 1).** Liddle & Hasanov (2023) state, verbatim, twice — p.829
> (Introduction) and p.843 (Conclusions §6) — "the income elasticity of
> economy-wide electricity demand for these middle-income countries was
> on average between 0.7 and 0.9"; the Abstract (p.827) gives "on
> average 0.8." The model's previously-carried figures do **not** trace
> to this paper as a pair: **0.76 belongs to a different paper —
> Sadorsky (2012) — quoted in Liddle & Hasanov's own literature review
> (p.830: "reported time-invariant, statistically significant long-run
> elasticities of 0.76 for income..."), a 19-country panel using CPI
> instead of electricity prices, not Liddle & Hasanov's own estimate.**
> **0.90** is a real Liddle & Hasanov statistic — the mid-point average
> of their eight individually-analyzed countries' elasticity ranges
> (Table 4, p.840, discussed p.841) — but is not presented as a bound of
> "the range" in that paper. **0.83 appears in neither paper.** LSZ
> (2020) itself also fails independently (reported for completeness,
> traced in the same session): its Table 1 gives long-run GDP
> elasticities of 0.68 and 0.57 for total final consumption (TFC, not
> electricity) — a different dependent variable — and neither 0.76 nor
> 0.90 appears in it either.

> **Adopted:** ε = 0.80, sourced to Liddle & Hasanov (2023), p.827 /
> p.829 / p.843. `organic_central` moved **4.0% → 3.9%** (Route A 3.95%,
> Route B 3.67%, adopted 3.9%) — the one model value this correction
> changes.

> **Direction of bias of the superseded value.** The untraced 0.83 sat
> above the paper's own central tendency (0.8) and above the CCE point
> estimate (0.70), so it **overstated** demand growth and therefore
> overstated downstream capacity requirements (solar build is a residual
> of demand net of gas and hydro, per `docs/methodology_demand_horizon.md`
> §1.3's own amplification limitation — a small change in `g` moves the
> headline solar build by a larger percentage).

> **The excluded sub-Saharan range remains unsourced.** The model
> excludes SSA-estimated elasticities (0.60–0.76) as rationing-
> contaminated (same reasoning as the demand-shape and generation-CAGR
> rejections elsewhere in §1). That excluded range itself has **no
> citation anywhere in the repository** — unchanged by this entry, and
> now stated as such directly in the corrected code comment and
> methodology text rather than implied. Separately, the panel Liddle &
> Hasanov actually use is **not** SSA-free: it includes South Africa and
> Côte d'Ivoire (2 of 31 countries), so the model's "non-SSA" framing is
> now stated as the model's own choice, not a property of the cited
> panel, and is flagged as a partial exclusion for that reason.

> **What would resolve the remaining gap.** A citation for the excluded
> 0.60–0.76 sub-Saharan range, if that exclusion is to be kept as
> reasoned rather than merely asserted.

> **See also:** `docs/parameter_provenance.md` (row 1), L55, C-10
> (the DIST-5 precedent for the same failure mode — a citation carried
> without ever checking whether the underlying mechanism or measurement
> existed).

## L55 — Two Nigerian electricity-generation series coexist in the
repo, disagree by up to 11.5% in a given year, and the data pipeline
still pulls the non-canonical one

> **Status:** LIVE — declared, partially resolved. Ember is now the
> canonical series for every demand-growth derivation in
> `docs/chapters/chapter3_2_draft.md` [removed `009d230`; held at `a993f15`; successor `CHAPTER_3_assembled.md` §3.2] §3.2.2–3.2.3 (see that chapter and
> L54). `data/make_data.py` is unchanged and still produces the other
> series by default.

> **The two files.**
> 1. `data/demand/electricity_generation_owid.csv` — fetched by
>    `data/make_data.py:24-63` (`fetch_owid_demand()`) directly from
>    OWID's own compiled `owid-energy-data.csv`
>    (`raw.githubusercontent.com/owid/energy-data`), filtered to Nigeria.
>    This is the file `data/make_data.py` produces when run, and the
>    only generation file that existed in the repo before this task.
> 2. `data/demand/electricity_generation_ember_diagnostic.csv` — a
>    manual transcription of Ember's Electricity Data Explorer figures,
>    recorded in `data/demand/step1_per_capita_diagnostic.xlsx` (sheet
>    `Raw_Ember_WDI`, column B) and extracted to this file in the
>    previous task. **Canonical**, per Zeus's decision: it reproduces
>    every row of the §3.2.3 decomposition table, including the
>    `constrained_continuation` arm value (3.14%), to ten significant
>    figures against the workbook's own `Decomposition` sheet.

> **The two series are not close.** 2010 and 2014 match exactly between
> the two files (26.13 and 32.28 TWh respectively), but they diverge
> materially from 2022 onward. **2024 level: Ember 37.56 TWh vs OWID
> 40.13 TWh — a 6.8% difference** (40.13/37.56 − 1). Per the diagnostic
> workbook's own `Raw_NERC`-vs-`Raw_Ember_WDI` comparison (also recorded
> against NERC, a third series — see L1/§3.2.2), the NERC-to-Ember gap
> alone moves from +8.8% (2022) to +11.5% (2023) to +1.3% (2024) to
> +6.0% (2025): none of the three series (NERC, Ember, OWID) can be
> treated as a stable re-scaling of another.

> **`data/make_data.py` still pulls OWID, not Ember.** No change was
> made to the data pipeline by this task (out of scope — Ember was
> extracted by hand from the diagnostic workbook, not re-plumbed into
> the automated pipeline). Anyone re-running `make_data.py` regenerates
> the non-canonical file; the canonical Ember file exists only as the
> hand-extracted CSV.

> **Consequence.** `electricity_generation_owid.csv` is not the source
> of any figure now stated in `docs/chapters/chapter3_2_draft.md` [removed `009d230`; held at `a993f15`; successor `CHAPTER_3_assembled.md` §3.2]. **Any
> other figure in this repository derived from OWID's generation series
> — rather than from the diagnostic workbook's Ember column — must be
> re-derived against Ember or explicitly labelled as OWID-sourced.** No
> systematic search for such figures elsewhere in `docs/*.md` or
> `src/*.py` was performed as part of this entry; `src/scenarios.py`'s
> own `constrained_continuation` derivation names "Ember" without a file
> path (see `docs/task_log_elasticity_sourcing.md`, Part 1/Job 10, for
> the prior session's full file:line inventory of every site naming
> Ember, OWID or WDI).

> **What would resolve it.** Either point `data/make_data.py` at an
> Ember-direct source (the Electricity Data Explorer, not OWID's
> compiled file) so the automated pipeline and the hand-extracted
> diagnostic agree, or formally retire the OWID file with a comment
> explaining why it is not used, so a future reader does not assume the
> two are interchangeable.

> **See also:** L54, `docs/task_log_elasticity_sourcing.md`.

## L56 — `demand_growth_prior()`'s SIGMA is derived from a superseded
arm set and was deliberately not recomputed when `organic_central` moved

> **Status:** LIVE — declared, not fixed. Declared 2026-09-25, the
> same session `organic_central` moved (L54).

> **What.** `demand_growth_prior()` (`src/scenarios.py:570`) returns
> `{"mean": demand_growth_scenarios()["organic_central"], "sigma": 0.0120}`.
> `mean` is read **by reference**, so it moved automatically when
> `organic_central` changed (0.040 → 0.039, L54) — it is currently
> 0.039 with no further action needed. **`SIGMA` is a separately stored
> literal, 0.0120, derived by a worked calculation in the function's own
> docstring from the three-arm set `{0.0314, 0.0400, 0.0600}`
> (equal-weighted standard deviation of the arms). That arm set is now
> superseded** — the live arms are `{0.0314, 0.0390, 0.0600}` — **and
> SIGMA was not recomputed**, on the grounds that recomputing it would
> be a second value change beyond the one this task's instruction
> authorized (only `organic_central` itself).

> **What recomputing it would change.** Recomputed on the live arm set:
>
>     arms          : 0.0314, 0.0390, 0.0600
>     arm mean      : 0.043467   (was 0.0438)
>     deviations    : -0.012067, -0.004467, +0.016533
>                      (was -0.0124, -0.0038, +0.0162)
>     sigma         : sqrt(mean of squared deviations) = 0.012096 -> 0.0121
>                      (was 0.011981 -> 0.0120)
>
> **The change is small: SIGMA would move from 0.0120 to 0.0121**, a
> shift of +0.0001 (about 0.8% relative). The arm mean moves from 0.0438
> to 0.043467, negligibly. This is a mechanical consequence of
> `organic_central` moving 0.1 percentage points closer to
> `constrained_continuation` and further from `organic_high`, which very
> slightly increases the dispersion of the three-arm set around its own
> mean.

> **Which results use `demand_growth_prior()`.** Exactly one consumer
> in the whole repository: `scripts/deferred/17_run_monte_carlo.py` [revived as `scripts/17_run_monte_carlo.py`, commit `36ddf55`, TASK 06]
> (confirmed by `grep -rln "demand_growth_prior" --include="*.py" .`,
> excluding the `.claude/worktrees` mirror — two hits, `src/scenarios.py`
> itself, which defines it, and this one caller). **This script is
> `deferred`, not a live runner** — it is the same implementation the
> claims ledger already documents as superseded because it draws demand
> growth and gas regime independently despite their sharing a driver
> (methodology claims ledger row 29; `docs/limitations_register.md`
> discussion of the shared-driver confound; `docs/task_monte_carlo.md`).
> **No currently-reported finding, and no live runner, depends on
> `SIGMA` at all.** The live parameter-uncertainty analysis
> (`29_run_parameter_uncertainty.py`, Sobol) samples five other
> continuous parameters and does not call `demand_growth_prior()`.

> **Direction of bias, if left unrecomputed.** Negligible and, if
> anything, conservative: the stale SIGMA (0.0120) is very slightly
> *narrower* than the correct one (0.0121), so a symmetric normal built
> from it would understate the dispersion implied by the live arms by
> less than 1%. Given `scripts/deferred/17_run_monte_carlo.py` [revived as `scripts/17_run_monte_carlo.py`, commit `36ddf55`, TASK 06] is not a
> live runner, this has no effect on any currently reportable result
> either way.

> **What would resolve it.** Recompute `SIGMA` to 0.0121 and update the
> worked docstring calculation, if and when `scripts/deferred/17_run_monte_carlo.py` [revived as `scripts/17_run_monte_carlo.py`, commit `36ddf55`, TASK 06]
> (or a successor) is revived — there is no case for doing so before
> then, since no live consumer reads the value.

> **See also:** L54.

## L57 — `bank_tenor_case` (rate-coherence task, Part 1): the bankability discount-factor basis is now an explicit, named arm

**Status:** RESOLVED (new arm, live default). Declared 2026-09-25.

**What.** `m.remaining_npv_factor[t]` (the annuity factor inside
`m.eaas_bankability` / `m.eaas_subsidy_limit`, L17) is now selected by
`scenario["bank_tenor_case"]` (`BANK_TENOR_CASE_CHOICES`,
`src/scenarios.py`), dispatched through `resolve_remaining_npv_factor()`
(`src/blended_finance.py`), called at the same unconditional site —
before any `capital_treatment_case` branch — in both builders.

- **`bank_tenor_comm`** [NEW DEFAULT, `load_scenario(bank_tenor_case=
  "bank_tenor_comm")`]: the objective's OWN commercial tranche.
  $F_t = \sum_{j=0}^{L^{eff}-1}(1+r^{comm,real})^{-j}$,
  $L^{eff}=\min(L_{comm}, T_{last}-t+1)$, `r_comm_real` deflated
  independently at this call site from `r_comm_nominal`/`us_inflation`
  — NOT reused from the F1 per-tranche capital-charge block, whose own
  real-rate local variable is only defined under `f1_tranche` (would be
  a `NameError` under `pre_f1_overnight` if reused).
- **`bank_tenor_legacy`**: the pre-correction form, unchanged —
  `resolve_private_rate()` (blended concessional/commercial WACC),
  summed undiscriminated to the model horizon. Retained, not deleted,
  for attribution: **every EaaS/bankability result produced before
  2026-09-25 used this basis by construction**, whether or not it named
  it, since no other option existed.

**Why a plain default, not a REQUIRED sentinel.** Unlike the sixteen
`_XXX_REQUIRED` arms in `load_scenario()`, `bank_tenor_case` defaults
silently to `bank_tenor_comm` — a deliberate departure from this
project's own no-silent-defaults discipline (see L17's "Process note"),
made because the corrected basis, not the historical one, is what
should govern every caller that does not know this arm exists yet.
This is the SAME direction of asymmetry as `new_gas_deliverability`'s
plain default, but the opposite VALUE choice: that default reproduces
old behaviour; this one does not.

**Consequence — flagged, not fixed.** `scripts/03_run_ndc_eaas.py` and
`scripts/26_run_f3_identification.py` both build models that read
`m.remaining_npv_factor` (confirmed, Part 0 inventory) and neither
passes `bank_tenor_case`. Their next run silently adopts
`bank_tenor_comm`. Both are bankability-priced (L17: "F-26 through
F-29 all rest on the bankability constraint", F-numbers of `findings_provisional_pre_sweep.md`, superseded and not to be quoted), so **F-12's
identification interval and F3's 47,528 MW ceiling will move** on
their next run. This is the DEFECT L17 described actually closing, not
a new one — movement here is expected and should not be read as a
regression when it appears in a future re-run's diff.

**`run_fin2_blended_bankability_sweep`** (FIN-2,
`src/optimize_experiments.py`) is explicitly pinned to
`bank_tenor_legacy` — see L17's update and the arithmetic-divergence
reasoning there. Not silently defaulted; deliberately exempted.

**What would extend it.** A decision on `blended_finance_scenarios()`'s
naira-basis `r_commercial` (0.18) — the half of L17 this entry does not
close — would change what `bank_tenor_legacy` (and therefore FIN-2)
actually computes, but would not change `bank_tenor_comm`'s formula,
which never reads `r_commercial` at all.

**See also:** L17, L52, L53, L58.

## L58 — Anchor re-baselining: the regression and central anchors have moved, both OBSERVED, from two distinct causes

**Status:** OBSERVED (re-run complete, 0 failures, 2026-09-25) —
promoted from the prior DECLARED/predicted status. The values below are
read directly from `logs/sweep/A_REGRESSION_ANCHOR.log` and
`logs/sweep/A_central_baseline.log`, both dated 2026-09-25 12:22, not
predicted or extrapolated.

**What moved, and to what — two separate movements, two separate
causes.**

| anchor | quantity | superseded value | new value | cause |
|---|---|---|---|---|
| regression (`A_REGRESSION_ANCHOR`, `treat=pre_f1_overnight`) | objective | 21,811,042,472.821114 USD (21.811 bn) | **21.328 bn** | `organic_central` only (L54) — `pre_f1_overnight` has no CRF mechanism, cannot read `r_conc_nominal`/`bank_tenor_case` |
| regression | storage 2054 | 45,502.9754 MWh | **43,539.97566 MWh** | `organic_central` only |
| central (`A_central_baseline`, `treat=f1_tranche`) | objective | 27.954 bn USD | **27.318 bn** | BOTH: `organic_central` (first move, to 27.303 bn, not separately re-observed here) AND `cr_central`'s concessional rate+tenor correction (second move, 27.303 → 27.318 bn) |
| central | storage 2054 | 65,612.985 MWh | **63,056.31077 MWh** | both, as above |
| central | cumulative solar (2054) | 41,031.4 MW | not separately confirmed here | — |

**The second movement's DIRECTION was predicted wrong by this entry's
own prior version, and is corrected here.** The prior draft of this
entry stated: *"a CRF-driven reduction (~9.57% on the concessional
tranche charge per dollar of `conc_vol`)... should be expected on the
next re-run."* That was the RATE-ONLY effect (register L53's Part 2
update), reported before the tenor correction (`L_conc` 30→25,
register L53's Part 3 update, same day) was applied. **The tenor
correction dominates and reverses the sign**: CRF at 2.12% real over
30yr = 0.033943 (−9.6% vs the IDA20 baseline); over the CORRECT 25yr
= 0.040616 (**+8.2%** vs the same baseline). The observed central
anchor move, 27.303 → 27.318 bn, is small and POSITIVE — consistent
with the net +8.2% CRF effect landing on the concessional-tranche
share of capital charge only (a small fraction of the total objective,
not the whole capital charge), not the large negative move the
rate-only prediction implied. **Both movements are now OBSERVED, not
predicted**: `organic_central` alone moves the regression anchor
(21.811 → 21.328 bn) and is the FIRST of two moves on the central
anchor (27.954 → ~27.303 bn, not independently isolated in this
re-run); the concessional rate+tenor correction is the SECOND move on
the central anchor only (27.303 → 27.318 bn, net positive, tenor-
dominated) and does not touch the regression anchor at all.

**Provenance, named explicitly** (per instruction): the two commits are
`6d5a076` (`organic_central` 4.0%→3.9%, L54) and `30561ce` (`cr_central`
concessional RATE to IDA21, L53) — the tenor correction (`L_conc`
30→25) that produces the observed +8.2% CRF effect above is, as of
this re-run, an uncommitted working-tree edit on top of `30561ce`, not
its own commit; see `docs/task_log_rate_coherence.md` for the exact
sequence.

**`scripts/run_central_sweep.sh` — UPDATED, no longer on the pending
list.** Its `A_REGRESSION_ANCHOR` comment and completion-trailer check
now assert 21.328 bn / 43,539.97566 MWh, with the two commits named
inline. This is the one file from the list below that this task DOES
edit, per Zeus's explicit instruction (item 1) — every other file
remains untouched.

**Every OTHER file quoting the superseded values, still pending a
one-pass update (re-grepped 2026-09-25, `.git`/`literature/` excluded;
unchanged from the prior version of this entry except
`run_central_sweep.sh` removed, now fixed):**

`docs/chapters/chapter3_1_v2.md` [removed `009d230`; held at `a993f15`; successor `CHAPTER_3_assembled.md` §3.1], `docs/drafts_gas1_l48_leakage_f22.md`,
`docs/findings_frozen.md`, `docs/forensic_audit.md`,
`docs/frozen_state_2026_09_10.md`, `docs/handover_session_close_sept.md`,
`docs/limitations_register.md` (this file — prior entries citing the
anchor as a currently-valid figure, e.g. A61's cross-reference; this
entry itself is exempt, it discusses the value historically),
`docs/merged_correction_plan_v2.md`, `docs/methodology.md`,
`docs/methodology_claims_ledger.md`, `docs/methodology_physical_layer.md`,
`docs/methodology_protocols.md`, `docs/runner_inventory.md`,
`docs/task_audit_fixes_p2.md`, `docs/task_audit_fixes_p3.md`,
`docs/task_bucketA_parity.md`, `docs/task_capex_reporting_field.md`,
`docs/task_conversion_capex_exposure.md`,
`docs/task_conversion_diagnostics.md`, `docs/task_defect_closure.md`,
`docs/task_epsilon_constraint.md`, `docs/task_log_elasticity_sourcing.md`
(already discusses the anchor as historical, from the earlier
re-baselining report — exempt in the same sense),
`docs/task_log_methodology_phase2.md`, `docs/task_methodology_repair.md`,
`docs/task_om_implementation.md`, `docs/task_register_amendment.md`,
`docs/task_runner_repair.md`, `docs/task_runner_revival.md`,
`docs/task_shared_fn_repair.md`, `docs/task_wrong_output_repair.md`.

**Not applied.** No file in the list above was edited by this entry.
Per instruction, they still await a one-pass update — now, since the
central anchor's second move is OBSERVED (27.318 bn), that pass can
use final numbers rather than waiting on a further re-run.

**See also:** L54 (`organic_central`), L17/L52/L53/L57 (the
rate-coherence work), `docs/task_log_elasticity_sourcing.md` (the
earlier, report-only anchor inventory this entry supersedes).

> **Generation (TASK 86):** not established. This entry quotes blended-generation or pre-epoch figures (27.318, 41,031, 63,056) without citing a hash-keyed result file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L59 — Discount-rate sweep re-run at the current state: physical-plan invariance to $r_s$ confirmed, monetary results remain conditional on 8%

**Status:** RESOLVED (re-run complete). Declared 2026-09-25.

**What.** `docs/chapters/chapter3_1_v2.md` [removed `009d230`; held at `a993f15`; successor `CHAPTER_3_assembled.md` §3.1] §3.1.2 claims the model's
*physical* plan is invariant to the social discount rate $r_s$, while
every monetary result is conditional on it. The sweep supporting that
claim (4%, 6%, 8%, 10%, 12%) had been run at the pre-correction state —
its central (8%) point matched the SUPERSEDED central anchor exactly
(41,031.4 MW solar, 65,613 MWh storage), from before both the
`organic_central` demand correction (L54) and the `cr_central`
concessional rate/tenor correction (L53) landed.

**Re-run, five solves, current state** (`organic_central` = 3.9%,
`cr_central` = IDA21 2.12%/25yr throughout), logs in
`results/_sdr_sweep_v3/`:

| $r_s$ | objective (bn USD) | cumulative solar to 2054 (MW) | solar to 2045 (MW) | storage 2054 (MWh) |
|---|---|---|---|---|
| 4% | 48.022 | 39,030.065 | 18,841.7 | 63,056 |
| 6% | 35.679 | 39,030.065 | 18,927.5 | 63,056 |
| 8% (central) | 27.318 | 39,030.065 | 18,990.3 | 63,056 |
| 10% | 21.520 | 39,030.065 | 19,240.8 | 63,056 |
| 12% | 17.416 | 39,030.065 | 19,240.8 | 63,056 |

**[VERIFY: this table is a pre-epoch, blended-dispatch state (27.318 bn at 8% is the advance-convention, flat-storage objective of L88). No social-discount-rate arm with `gas_vintage_dispatch=True` exists (`docs/task_log_85.md`), so the solar-to-2045 column (18,841.7 to 19,240.8 MW) and the 63,056 MWh storage figure are not confirmed against a vintage file.]**

**Physical-plan invariance confirmed at the current state, more
tightly than the original claim.** Cumulative solar to 2054 agrees
across all five rates to within **0.0003 MW** (39,030.0650 to
39,030.0653); storage in 2054 is identical at 63,056 MWh; the first
solar-build year is 2032 at every rate. Only the TIMING of additions
shifts, confined to the 2039-2048 window (the original, pre-correction
claim had this window at 2039-2051): solar installed by 2045 ranges
18,841.7-19,240.8 MW, a **2.1% spread** (the original claim: "within
±1%" — the current spread is wider in relative terms, though the
underlying finding — timing redistributes, cumulative totals do not —
is unchanged and, if anything, more tightly confirmed on the
cumulative-to-2054 figure specifically).

**Monetary results remain fully conditional on $r_s$.** Objective
ranges from 48.022 bn USD (4%) to 17.416 bn USD (12%) — a **factor of
2.76**, "reflecting present-value scale rather than any change in what
is built" (unchanged conclusion from the original claim, re-confirmed
at the current state).

**Conclusion, restated at the current state:** every monetary result in
the thesis is conditional on $r_s = 8\%$; every physical result is not.
The chapter's own open item — that 8% itself is unsourced, and a
Ramsey-type derivation would likely land lower (3-6%) — is unaffected
by this re-run and remains open; the sweep continues to show a lower
rate would not change the physical plan.

**What changed in the chapter.** `docs/chapters/chapter3_1_v2.md` [removed `009d230`; held at `a993f15`; successor `CHAPTER_3_assembled.md` §3.1]
§3.1.2's sweep paragraph and its worked figures updated to the table
above; the paragraph's own concessional-rate "Note to Zeus" item
(asking whether `cr_central`'s 2.80% needed a recorded source) removed
as resolved (register L53); the 8%-unsourced open item retained.

**See also:** L53 (`cr_central` IDA21 rate/tenor), L54 (`organic_central`),
L58 (anchor re-baselining, same underlying corrections).

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L60 — NDC 3.0's gas-power apportionment rebuilt on an inventory-derived share; the equal-proportional-effort rule and the conditional arm's 573.5 Mt anchor remain declared assumptions

**Status:** RESOLVED IN PART — the share is now sourced; two declared
assumptions remain, stated explicitly rather than left implicit.
Declared 2026-09-25.

**What changed.** `scripts/00_build_emissions_cap.py`'s
`GAS_POWER_SHARE` — gas-fired power's share of NDC 3.0's energy-sector
abatement — was `E_base / 217.0`, where `E_base` is the MODEL's own
baseline gas-to-power emissions and `217.0` was an UNSOURCED "~38% of
573.5, round conservative estimate" tracing to no page or table in
NDC 3.0, BTR1 or the NID (`docs/task_log_ndc_extraction.md` Part 4,
`docs/task_log_btr1_denominator.md`). A model quantity divided by an
unsourced figure expressed no actual share of anything.

**Replaced with `POWER_SHARE_OF_ENERGY_2018 = 16,234.0 / 193,346.72 =
8.3963%`** — inventory-over-inventory, both terms 2018 figures from the
same December-2024 inventory round: numerator NID Table 3.9, `1.A.1.a
Public electricity and heat production` (p.36); denominator BTR1 Table
2.10, Sector 1 `Energy` (p.33). Verified independently: both AR5 GWP,
2006 IPCC Guidelines + 2019 Refinement; nesting arithmetic exact
(`1.A.1.a` 16,234 + `1.A.1.b` 990 + `1.A.1.c` 12,640 = `1.A.1` 29,864,
matching BTR1's own 29,863.81 to the digit, and cross-confirmed a
second, independent way via the NID's own Table 3.3). Scope match
confirmed: NDC 3.0's own formal definition of "Energy" (p.38) is
structurally identical to BTR1/NID's `1. Energy`, so the pairing with
the p.15 energy-sector-abatement share (22.0%/26.3%, itself stated as
covering "Energy sector (including buildings and transport)") is
correct, not merely convenient.

**Also corrected**: the module docstring's illustrative `E_base ≈ 7.364
MtCO2/yr` was stale — verified against the live `emissions_cap.csv`'s
stored 2030 value, the actual figure is **15.717 MtCO2/yr**
(`Ebase_2025`, `results/baseline/diagnostics.json`), roughly double the
stale illustration. Docstring corrected; no code depended on the
stale illustrative number itself (it was prose only).

**New cap values** (2030 and 2035, unconditional and conditional; not
yet applied at the time this entry was first written —
`00_build_emissions_cap.py` had not been re-run, per Zeus's own "report
first" instruction). **These NEW figures were projected ahead of the
actual run and are themselves now superseded** — see the correction
below, added once commit `7908d09` actually rebuilt
`emissions_cap.csv`:

| | OLD (217.0, unsourced) | NEW, as projected here (SUPERSEDED) | % difference, as projected (SUPERSEDED) |
|---|---|---|---|
| Unconditional 2030 | 13,037,004.03 tCO2 | 12,610,209.44 tCO2 | −3.27% |
| Unconditional 2035 | 12,195,033.87 tCO2 | 11,634,162.98 tCO2 | −4.60% |
| Conditional 2030 | 11,786,877.47 tCO2 | 11,161,011.29 tCO2 | −5.31% |
| Conditional 2035 | 10,039,368.72 tCO2 | 9,135,227.26 tCO2 | −9.01% |

**Correction (2026-09-26): the actually-applied values differ from the
projection above.** Read directly from `data/cost/processed/
emissions_cap.csv` as regenerated by commit `7908d09`, against the
pre-rebuild snapshot `data/cost/processed/_emissions_cap_PRE_L60.csv`
(both on disk):

| | OLD | NEW, actually applied | % difference, actually applied |
|---|---|---|---|
| Unconditional 2030 | 13,037,004.03 tCO2 | **12,638,030.79** tCO2 | **−3.06%** |
| Unconditional 2035 | 12,195,033.87 tCO2 | **11,661,984.33** tCO2 | **−4.37%** |
| Conditional 2030 | 11,786,877.47 tCO2 | **11,188,832.64** tCO2 | **−5.07%** |
| Conditional 2035 | 10,039,368.72 tCO2 | **9,163,048.61** tCO2 | **−8.73%** |

Every cell in the projected table above was off — not by a rounding
error, but consistently, in the same direction, by a small margin
(roughly 0.2-0.3 pp of the OLD base in each case). The projection was
computed by hand ahead of the actual rebuild; the code that ran later
evidently used a marginally different intermediate value somewhere in
the chain (candidates: the regenerated $E^{base}$ or a rounding
difference in the power-share arithmetic — not traced further here).
**The applied CSV is authoritative; the projected table is retained
only as a record of what was expected, not what happened.** Anything
citing this entry's percentages elsewhere (register L63's own
correction note, `docs/findings_frozen.md`'s F-21 addition,
`docs/task_log_findings_reverify_L60.md`) must use the "actually
applied" row above, not the projection — this was gotten backwards in
at least L63 and the reverify task log, both corrected in the same
pass as this note.

All four caps TIGHTEN either way (direction consistent with the old
217.0 denominator overstating BTR1's true Energy total by 12.2%, per
`docs/task_log_btr1_denominator.md` Part 1).

**Two declared assumptions remain — stated explicitly, not resolved
by this change:**

1. **Equal-proportional-effort allocation.** The construction still
   apportions abatement to gas-fired power in EQUAL PROPORTION to its
   share of Energy-sector emissions — an intensity-matching rule, not
   an NDC-stated power-sector target (NDC 3.0 states no power-sector-
   specific abatement share anywhere found). A plausible alternative —
   power abates MORE than proportionally, since grid electricity is
   generally the cheapest sector to decarbonise — would TIGHTEN the
   cap further. Sensitivity-testable; not tested here. See
   `docs/parameter_provenance.md` row 11.
2. **The conditional arm's 573.5 Mt anchor is unsourced against BTR1
   and the NID.** `docs/task_log_btr1_denominator.md` Part 1 item 2:
   573.5 appears in neither document (exhaustive search, zero
   matches); NDC 3.0 itself cites 614 Mt (BTR1's own figure) on p.13
   then switches to 573.5 on p.14 with no stated adjustment. The
   UNCONDITIONAL caps above are now fully sourced end to end; the
   CONDITIONAL caps retain this one unsourced anchor, since the
   conditional multiplier (Reading A, adopted — see below) is built
   from 573.5. See `docs/parameter_provenance.md` row 12.

**Conditional multiplier — Reading A adopted, with its basis stated.**
NDC 3.0 p.14: *"Nigeria maintains its 20% unconditional objective and
views the conditional reduction at 80% of the 29% and 32% reduction."*
Two readings exist. **Reading A** (adopted): unconditional = 20% of the
573.5 Mt 2018 base = 114.7 Mt; multipliers 168.2/114.7 = 1.466×
(2030), 184.9/114.7 = 1.612× (2035) — varies by year because it is
anchored to a fixed base against changing targets. **Reading B**
(rejected): unconditional = 20% of the target itself; multiplier =
target/(0.20×target) = 1/0.20 = **5.0 exactly, in every year, by
construction** — it cannot vary with the NDC figures it is supposedly
derived from, and therefore measures nothing about the NDC. Reading A
is adopted because it is the only one of the two that is actually
sensitive to NDC 3.0's own stated numbers.

**See also:** L45/L53/L54/L58/L59 (the wider rate-coherence/demand
correction sequence this entry's cap values are computed against),
`docs/task_log_ndc_extraction.md`, `docs/task_log_btr1_denominator.md`
(both parts).

---

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L61 — The A18 diagnostics-key guard caught the L60 rebuild's stale baseline by coincidence, not by design; the bare-path/no-provenance dependency remains live

**Status:** LIVE

**What.** `scripts/00_build_emissions_cap.py` reads a bare, untracked
`results/baseline/diagnostics.json`. Nothing writes that bare path:
`scripts/01_run_baseline.py` writes only arm-hashed filenames
(`diagnostics__{tag}.json`), by its own docstring's own admission
(`00_build_emissions_cap.py:206`, "01_run_baseline.py writes only
diagnostics__{tag}.json; see task_epsilon_constraint.md, Part 4B, for
why it cannot cleanly write the bare path itself"). Placing the correct
run's output at the bare path is manual, and nothing records which arm
set, commit state, or model version produced whatever file currently
sits there.

**What happened (L60 rebuild, commit `7908d09`, 2026-09-26).** The bare
file on disk had mtime 2026-08-23 13:44 — stale in content (it
predated the rate-coherence corrections, L52/L53/L57-L59, and the
demand corrections, L54/L56, both upstream of the baseline dispatch
that determines `fleet_gas_to_power_twh_th_by_year`) and, separately,
still carried the pre-rename key `gas_to_power_twh_th_by_year` rather
than `fleet_gas_to_power_twh_th_by_year`. The A18 guard (`docs/
forensic_audit.md` A18; guard added in `docs/task_audit_fixes_p2.md`
A18) raised on the missing renamed key and forced a fresh
`01_run_baseline.py` run before the rebuild could proceed.

**Why the catch does not resolve the risk.** The guard's condition is
"is the renamed key present" — a proxy for "was this file regenerated
after the A18 rename," not for "was this file regenerated after every
correction currently upstream of it." Those two facts coincided here
only because the Aug-23 file predated both the rename and the later
rate/demand corrections. A bare file regenerated any time after the
rename but before a subsequent upstream correction — e.g. a
`01_run_baseline.py` run placed today, followed tomorrow by a rate or
demand parameter change with no matching re-run — would carry the
current key by construction and pass the guard silently while still
being stale in content, exactly as it would have failed to catch
anything before the rename existed at all. The guard checks a key
name, not a provenance stamp, a content hash, or a timestamp against
its own declared upstream inputs.

**Direction of risk.** Unquantifiable in advance — a stale baseline
could move `E_base`, and therefore every NDC 3.0 cap, in either
direction depending on which upstream correction it missed. Recorded
as a live process gap, not a signed bias like L1.

**What would resolve it.** Either (a) `01_run_baseline.py` writes the
bare path itself, stamped with its arm tag and source-commit hash, so
`00_` can assert the arm rather than trust a hand-placed file; or (b)
`00_` checks a content hash or mtime floor against its own declared
upstream dependencies (gas-availability CSV, demand/rate scenario
files), not only a key name. Neither is implemented. This is the
still-open half of A18: `docs/forensic_audit.md` already called the
guard "correct today ... will crash the moment anyone tries to refresh
it" — the crash is necessary but not sufficient, and it caught this
particular staleness only because the file happened to also be
pre-rename.

**See also:** L60 (the rebuild during which this fired), L52/L53/L54/
L56/L57/L58/L59 (the corrections the stale file missed), `docs/
forensic_audit.md` A18, `docs/task_audit_fixes_p2.md` A18.

---

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L62 — `run_central_sweep.sh`'s anchor check is a hand-maintained literal, not a derived assertion

**Status:** LIVE

**What.** The `A_REGRESSION_ANCHOR` value `run_central_sweep.sh` treats
as ground truth is not read from any stamped record — it is typed
twice into the script itself: once in the pre-run comment
(`run_central_sweep.sh:213-217`) and once in the post-run completion
message (`run_central_sweep.sh:318-320`, `echo "CHECK FIRST:
A_REGRESSION_ANCHOR must read 21.328 bn..."`). Both are plain text for
a human to compare against the log by eye; **neither is a shell
assertion** (no `test`/`diff`/non-zero exit on mismatch) — a wrong
anchor does not fail the script, it only prints a claim that may by
then be false, next to the log the reader is meant to check it
against.

**Same defect class as L61, different file.** A correctness guard
whose own reference value is maintained by hand can go stale silently,
then asserts a claim — true or false depending on whether someone
remembered to update it — to whoever reads it next.

**Correction to the note this entry was drafted from.** It described
the anchor as "re-baselined twice in two days." Checked against this
file's own commit history: the anchor text in `run_central_sweep.sh`
has been edited exactly twice — `c30e479` (2026-09-07) and `1170b4c`
(2026-09-25 16:43) — **18 days apart, not two.** Across the wider
register, the anchor VALUE has moved twice on record: 21.475 → 21.811
bn (pre-2026-09-10, `docs/frozen_state_2026_09_10.md`) and 21.811 →
21.328 bn (L58, 2026-09-25). "Twice" is right; "in two days" is not —
kept out of the finding below since it doesn't change the defect, but
recorded here so the correction isn't silently absorbed.

**Direction of risk**, as with L61: a reader who trusts a passing
check against a now-stale literal concludes the sweep is attributable
when it isn't; a reader who sees a *failing* check against a literal
that itself went stale (not the sweep) concludes the opposite — that
the sweep is broken when only the check's memory is.

**What would resolve it.** Either (a) derive the printed/compared
value at run time from a stamped anchor record (e.g. a small JSON file
written by whatever last validated the anchor, carrying its own
source-commit hash) rather than a literal typed by hand in two places,
or (b) have the completion check fail loudly — not merely print — when
the literal's own last-edited commit predates the newest commit
touching the modules it depends on (`src/scenarios.py`, the rate/demand
arms), i.e. detect its own staleness rather than assume freshness.

**See also:** L58 (the anchor re-baselining this literal tracks), L61
(same defect class, different file — `00_build_emissions_cap.py`'s
bare-path diagnostics dependency).

---

## L63 — Leakage rises with cap stringency, confirmed by a second instrument (the L60 cap rebuild); two of fifteen FIN-3 cells cross the flag threshold

**Status:** OBSERVED (re-run complete, 2026-09-26). This is a
VERIFICATION cross-check produced during correction, not a thesis
finding in its own right — it corroborates an existing one (F-21).

**What.** The L60 cap rebuild (register L60, commit `7908d09`) tightens
the cap `02_run_ndc_caps.py`/`03_run_ndc_eaas.py` enforce. Re-running
`05_run_fin3_capital_sweep.py` at the new cap
(`results/fin3/capital_sweep_results.csv`, `results/fin3_L60.log`, both
2026-09-26 12:09 — after the L60 rebuild at 11:17, so run against the
applied state, not the pre-rebuild one) against the pre-L60 snapshot
(`results/_pre_L60_20260926_1158/fin3/capital_sweep_results.csv`,
2026-09-26 11:58) shows leakage share rising in **all fifteen FIN-3
cells** (5 capital cases × 3 collection-rate scenarios; confirmed by
direct row-by-row comparison of both CSVs, not sampled). Two cells
cross the declared 0.10 flag threshold (`leakage_diagnostic`,
`src/optimize_model.py:1760`) from compliant to flagged:

| cell | pre-L60 leakage | post-L60 leakage | flag |
|---|---|---|---|
| moderate / no_friction | 9.85% | 13.53% | compliant → **flagged** |
| adequacy / reform_target | 8.95% | 12.56% | compliant → **flagged** |

**This corroborates F-21's mechanism** (`docs/findings_frozen.md`,
tightening a grid-only cap raises emissions outside the modeled
boundary) **through an instrument that was not a deliberate
tightening** — L60's purpose was sourcing the power-sector share, not
testing cap stringency; the leakage response is a side effect of that
correction, independent corroboration rather than a designed test.

**Correction to the note this entry was drafted from, on one point —
and a correction to this entry's OWN prior correction, on the same
point:**

1. **This entry previously said the source note's unconditional cap
   deltas (−3.06% at 2030, −4.37% at 2035) "do not match L60's own
   recorded table" and used L60's table (−3.27%/−4.60%) as the
   reference instead. That was backwards.** L60's table was a
   PROJECTION, written before `00_build_emissions_cap.py` was actually
   re-run (L60's own text says so: "not yet applied"). Checked directly
   against `data/cost/processed/emissions_cap.csv` (as regenerated by
   commit `7908d09`) against the pre-rebuild snapshot
   `_emissions_cap_PRE_L60.csv`: the applied unconditional deltas are
   **−3.06% (2030) and −4.37% (2035)** — the source note's figures were
   right; L60's projected table (now corrected in place, see L60 above)
   was not. Trust the applied CSV over any register entry's prose table
   when the two disagree — a lesson this entry itself is now the
   evidence for.
2. **Carbon shadow prices did not rise "2-11% across every cell."**
   Recomputed from `carbon_shadow_mean_usd_per_tco2` across all fifteen
   cells in both CSVs, the true range is **+2.55% to +14.85%**, and
   four of fifteen cells exceed the claimed 11% upper bound:
   moderate/no_friction (+13.12%), moderate/reform_target (+12.21%),
   adequacy/reform_target (+14.85%), adequacy/current (+13.82%). ~~The
   direction (universal increase) and the qualitative conclusion
   (tightening raises the carbon shadow price) both hold;~~ **[TASK 29
   Part 3.2: these are all-year means. On the binding-year basis (25 binding
   years before, 26 after, every cell) the range is −1.39% to +10.43% and
   four cells FALL (`unconstrained` at every collection rate, and
   `expansion` at 1.00, all 115.05 → 113.45 USD/tCO₂). "Universal increase"
   does not hold; the rise is concentrated in the `tight`, `moderate` and
   `adequacy` cells. See F-31.]** The
   magnitude claim did not hold either, and is corrected here rather than
   repeated.

**Two qualifications, both from the source note, both confirmed
against the CSVs:**

1. **Part of the rise is a level shift, not scarcity.** The
   `unconstrained` cell — never cap-bound, `leakage_flagged=False` by
   construction, no budget constraint to tighten against — also rose,
   5.81% → 7.37%. This has no cap-stringency channel to explain it; it
   is consistent with L60's baseline recomputation (`E_base`
   regenerated) shifting `leakage_share`'s denominator, not with any
   capital-sweep dynamic. Not independently traced further here.
2. **`expansion/current` remaining unflagged is a property of the
   0.10 threshold, not evidence of compliance.** It rose 5.84% → 8.10%,
   still under the cutoff — evidence the threshold isn't yet binding on
   this cell, not that its leakage is acceptable on any independent
   basis.

**Not applied further.** No cap file, runner, or chapter text was
edited by this entry — it only re-runs an existing script against
already-applied L60 output and records the comparison.

**See also:** L60 (the cap rebuild this cross-check runs against),
`docs/findings_frozen.md` F-21 (the mechanism corroborated), L49
(FIN-3's bottleneck-quantity limitation, a separate concern from
leakage), L37 (FIN-3 scenario-key defect, repaired).

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L64 — `src/stochastic.py` wired as `scripts/30_run_stochastic_redispatch.py`: a genuine two-stage evaluation, with three declared gaps in what "fixed" means

**Status:** RAN, and its three declared gaps are now split (TASK 28 Part E.1). The runner ran as Phase B item 17, both legs (`logs/phase_b/17_A_..._fixed.log`, `17_B_..._free.log`, 2026-10-03, HEAD `2b22fe9`, 100 draws each; runs `d1cd080f`, `5d29ee61`). It builds with `build_model_sliced` since commit `c549bba` (2026-10-01), so the "builder scope condition (annual only)" below is superseded. **Gap 1 (`gas_add` free by default) is CLOSED:** `--fix-gas fixed` exists, was run, the exact `.fix()` was infeasible and replaced by a banded fixation (L65), and the difference between the two settings was measured (F-32 Result 5: 0 of 100 vs 50 of 100 infeasible). It was closed by the runner's own `--fix-gas`/`--fix-tolerance` work and the item 17 run -- **not** by the sliced port and **not** by TASK 08 (`docs/task_log_08_ws_comparability.md` concerned WS comparability and touched neither the builders nor `fixed_plan`). **Gap 2 (`eaas_subsidy` never fixed) remains OPEN.** **Gap 3 remains OPEN in changed form:** the sliced port made `convert_mw` exist in the runner's builder (it also earns firm-capacity credit), but `fixed_plan` holds only `solar_public_add`, `solar_eaas_add`, `storage_add` and `gas_add_fixed`, so conversion is free recourse and the wording "absent from this builder" is false. Still live: the ASSUMED `demand_sigma`, `carbon_mu`, `carbon_sigma`, and `gas_probability_weights()` as an unverified prior.

**Original status line, SUPERSEDED (kept for the record; TASK 28 Part E.1, 2026-10-04):**
~~**Status:** WRITTEN, NOT YET RUN. Declared 2026-09-27. The runner has~~
~~been verified statically (AST parse, `--help`) only — no solve has been~~
~~run by Claude Code; Zeus runs it (task_log_stochastic_runner.md, role~~
~~separation).~~

**What the method measures.** Fixed-plan re-dispatch: solve once at a
base scenario for the here-and-now capacity plan, freeze it, re-solve
under Monte Carlo draws of demand growth, gas deliverability regime,
and (if active) carbon price, so only dispatch adjusts. Reports mean,
sd, p5/p50/p95, **VaR and CVaR at 95%** on system cost, and **counts
and records every infeasible draw with the realisation that caused
it** — infeasibility under a fixed plan is a reliability-risk finding
in its own right, not merely a dropped observation.

**What it does not claim**, quoted verbatim from `src/stochastic.py`'s
own docstring: *"it does NOT find the plan that is optimal under
uncertainty (that would require stochastic programming with
recourse)."* This is two-stage EVALUATION of a plan already chosen,
never two-stage optimisation.

**Builder scope condition (annual only).** `stochastic.py` imports
only `src.optimize_model.build_model`, never the sliced builder. Under
the annual builder: no conversion mechanism exists at all (not merely
unfixed — the concept is absent); hydro is a fixed exogenous annual
series with no dispatch, seasonal shape, or minimum-flow constraint;
and storage cannot value its ENERGY-ARBITRAGE contribution (confirmed
against the sliced builder's own docstring: *"storage structurally
incapable of value [at annual resolution]... there is no intra-annual
time for it to shift energy across"*) but DOES still receive its
firm-capacity CREDIT identically in both builders — a fixed
`storage_add` still counts toward a tightened peak/reserve-margin
requirement under a draw, even though building it for energy-shifting
value alone would never be chosen. **Flag for Chapter 3, not edited
here**: §3.1.8 and §3.3.5 both say "cannot value storage" with no
qualifier; both need the precise "energy-arbitrage vs. firm-capacity
credit" distinction.

**The fixed plan is partial, three ways — not "capacity frozen, only
dispatch adjusts":**
1. `gas_add` (new gas capacity) is free by default (`--fix-gas free`);
   the runner adds `--fix-gas fixed`, which additionally freezes it at
   the base solve's per-year values via a direct Pyomo `.fix()` call
   the runner itself makes on the built model — `build_model`'s
   `fixed_plan` kwarg does not accept a `gas_add` key (verified by
   reading `optimize_model.py:476-481`; the builder was NOT edited to
   accept one, per the task's explicit instruction to stop rather than
   edit it). The gap between the two `--fix-gas` settings is itself a
   reportable result: the value of gas flexibility as recourse.
2. `eaas_subsidy` is permanently excluded from fixing, per the
   builder's own code comment (`optimize_model.py:477`:
   `# eaas_subsidy is excluded`) — no flag controls this.
3. `convert_mw` does not exist in the annual builder at all.

**Every ASSUMED parameter this runner introduces, all required on the
CLI with no default** (matching `dscr_case`/`storage_cc_case`
discipline): `demand_sigma` (a reproducible CALIBRATION against this
model's own demand arms — 0.01 on the 3.9% `organic_central` central
arm brackets `constrained_continuation` (3.14%) at ±2σ and approaches
`organic_high` (6.0%) — not an independently sourced figure);
`carbon_mu`/`carbon_sigma` (ASSUMED, no source, and INERT at this
runner's hardcoded `carbon_case="no_policy"` base configuration —
required regardless, per the required-key discipline). A candidate,
UNWIRED calibration source exists (`data/stochastic/
carbon_price_reference.csv`, a non-Nigeria-specific World Bank
Carbon Pricing Dashboard extract with a generator,
`data/make_data.py`, but zero consumers) — recorded, not used;
calibrating against it would mean inventing a mapping.

**`gas_probability_weights()` is a prior, not a forecast** — quoted
from its own docstring, and its own sourcing note flags itself as
UNVERIFIED against a primary ("Confirm each citation against a primary
source before submission"). The runner exposes `--gas-weights
{sourced,symmetric}` for exactly this reason.

**A second instance of the interface-insufficiency class Ruling 1
already covered**: `sample_uncertainties()` has no parameter to
override the gas-regime weights, so `--gas-weights sourced` calls it
unmodified but `--gas-weights symmetric` uses a local helper in the
runner that reproduces its three documented distributions exactly,
parameterised by weights. `run_stochastic_simulation()` itself is
never called — it discards infeasible draws' realisations and most of
each feasible draw's realisation, which cannot satisfy the per-draw
`draws.jsonl` output. No line of `src/stochastic.py` changes in either
case.

**See also:** `docs/task_log_stochastic_runner.md` (full Part 1-3
trace), `docs/parameter_provenance.md` rows for `demand_sigma`,
`carbon_mu`, `carbon_sigma`, `gas_probability_weights`.

## L65 — Fixed-plan re-dispatch by exact variable fixation is infeasible against the model's own optimal solution

**Status:** RESOLVED by a banded fixation; recorded because it is a
property of the method, not a defect, and will recur.

**What.** Fixing `gas_add[t]` at the base solution's own values, in a
freshly built model at the base realisation, returns **infeasible** —
the model rejects its own optimum. Established by minimal
reproduction: base solve optimal; re-solve with solar and storage
fixed, still optimal; additionally fix `gas_add`, infeasible.

**Cause.** `firm_capacity_constraint` is exactly binding — zero slack
to solver precision — in every year the base solution builds gas,
because gas is the marginal firm-capacity instrument. Exact `.fix()`
removes the LP's own slack cushion at the one point that had none, and
the sub-1e-5 MW residual inherent in round-tripping a continuous
solution value through `pyo.value()` and back through `.fix()` then
violates the constraint.

**Evidence.** Bisection: years 0–9 alone feasible; year 10 alone
feasible; **0–9 together with 10 infeasible** — combinatorial, not
per-year. Tolerance sweep: infeasible at 0, 1e-9, 1e-8, 1e-7 and 1e-6;
feasible from 1e-5 upward, with cost matching to four significant
figures across 1e-5 to 1.0. A floating-point threshold, not a physical
one.

**Resolution.** `--fix-tolerance`, required with no default, bands
`gas_add[t]` to `[v(1−ε), v(1+ε)]`; exact `.fix()` is retained at
tolerance 0. At `ε=1e-5` the absolute band width is a fraction of 1 MW
— orders of magnitude below the smallest capacity decision in the
model — so the plan remains fixed in every economically meaningful
sense regardless of whether the band technically binds.

**The band DOES bind, under every real draw — this corrects an
earlier over-broad claim.** A standalone, zero-perturbation
reconstruction (base scenario, unmodified) found the band non-binding:
0 of 20 nonzero-gas years sat at an edge across ε = 1e-5, 1e-4 and
1e-3, with identical cost throughout — consistent with the band merely
absorbing floating-point residual around an UNCHANGED optimum. That
result was then generalised, in an earlier draft of this entry, to "the
band is verified non-binding," stated without qualification. **The
actual production data contradicts the unqualified version of that
claim directly**: `gas_add_band_diagnostics` on every real run in
`results/stochastic_redispatch/` — `ae0a8d24`, `42ccf7c7`, and all
four `--demand-sigma` sweep runs (`10127e1b`, `ae0a8d24`, `c57b5de4`,
`a9f1e7a3`) — shows **1000 of 1000 nonzero-gas-year checks landing at a
band edge, zero in the interior**, with complete consistency across
every run. **Reconciled**: under a genuinely UNCHANGED draw there is no
directional pressure on `gas_add`, so it lands wherever the LP
relaxation puts it inside the tiny band (interior). Under ANY real
perturbation (a different `demand_growth` or `gas_scenario`), the LP's
marginal-cost logic wants to move `gas_add` at least slightly away from
the base value — gas being the model's cheapest lever to adjust — and
the razor-thin band stops that movement almost immediately, at the
edge. **"Binds" (the LP sense) and "is economically material" are NOT
the same test here**: the band binds at essentially every real draw,
but the absolute deviation it permits before binding is still a
fraction of a MW, so a plan pinned at that edge is indistinguishable,
economically, from one pinned exactly. The correct, narrower claim is:
the band is non-binding ONLY in the specific zero-perturbation
reconstruction used to verify it is not silently distorting the base
case; it binds, negligibly, at essentially every actual draw. The
edge/interior diagnostic is computed on every production run
(`summary.json.gas_add_band_diagnostics`) for exactly this reason — a
one-off zero-perturbation check does not describe production
behaviour, and should not have been reported as if it did.

**Generality.** Not gas-specific and not a model defect. Any
fixed-plan method that pins a continuous variable an LP has driven to
exact bindingness will show this. Reported in §3.6 as a property of
the method.

**See also:** F-32, whose Result 1 is the same zero-slack finding read
as an economic result rather than a numerical one.
`docs/task_log_stochastic_runner.md` (full derivation, Items 1 and 4).

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L66 — `peak_mw` was not recomputed per draw, so demand realisations tested energy adequacy only

**Status:** FIXED at the commit introducing `_recompute_peak_mw()` in
`scripts/30_run_stochastic_redispatch.py`; recorded because it
invalidates earlier output.

**What.** `scripts/30_run_stochastic_redispatch.py` perturbed
`demand_growth` per draw but left `peak_mw` at its base-case value, so
the firm-capacity requirement (3.21) did not move with the
realisation. Because self-generation is unbounded and priced, energy
adequacy can always be met at a cost — so demand draws tested energy
adequacy only and **never capacity adequacy**, which is the quantity
the method exists to stress.

**Reasoning for the correction.** The plan's capacity is fixed at the
investment stage, correctly. The peak it must meet is a property of
the drawn world, not of the plan. The two were conflated.

**Effect.** `peak_mw` is now recomputed per draw using
`load_scenario()`'s own derivation, verified byte-identical before
use. Under recourse (`--fix-gas free`, σ=0.01), comparing the pre-fix
run (`76965a0b`) against the post-fix run (`49cc69e0`), the tail
heavies — CVaR(95%) 121.22 → **126.68 bn**, p95 97.81 → **102.77 bn**,
sd 23.88 → **25.84 bn** (all verified directly against both runs'
`summary.json`, to the reported precision) — while infeasibility stays
at 0/100, since gas can still be built. Under a committed plan the
correction is what makes F-32's threshold observable at all: with a
static peak, `--fix-gas fixed` would have shown the same 0/100 the
recourse case shows, for the same reason (no capacity-side channel for
demand to bind against).

**Superseded.** Every `30_` result produced before this correction,
including the first `--fix-gas free` pilot (`76965a0b`) and its
gas-regime cost breakdown.

**See also:** F-32 (the finding this correction makes observable),
L65 (the separate, still-live numerical issue in how the frozen plan's
`gas_add` is represented, orthogonal to this one).

## L67 — VOLL (`UNSERVED_ENERGY_PENALTY`) has no source, no SOURCED/DERIVED/ASSUMED tag, and its cost file lacked the provenance columns every sibling cost file carries

> **Status:** LIVE. Declared TASK 01 item 20
> (`docs/task_log_01_omnibus_inventory.md`); the provenance gap closed
> (not the sourcing gap — no value changed) by TASK 05
> (`docs/task_log_05_voll_provenance.md`): `data/cost/processed/
> unserved_energy_penalty.csv` now carries `source`/`notes` columns,
> `docs/parameter_provenance.md` row 20 now exists, and this entry is
> that row's required register cross-reference.

> **What.** VOLL (`econ["UNSERVED_ENERGY_PENALTY"]`) is read by
> `load_econ(voll_case=...)` (`src/io.py:424-430`) from
> `data/cost/processed/unserved_energy_penalty.csv` —
> `voll_low`/`voll_mid`/`voll_high` = 5/10/30 USD/kWh =
> 5,000,000,000/10,000,000,000/30,000,000,000 USD/TWh, flat across every
> one of the model's 21 years (2025-2045, checked directly). Before
> TASK 05 the CSV's header was `year,scenario,voll_usd_per_kwh,
> voll_usd_per_twh` — no `source` column, no `notes` column — unlike
> `solar_capex.csv`/`storage_capex.csv` (`notes`) and `om_cost.csv`
> (`source`) beside it in the same directory. No SOURCED/DERIVED/ASSUMED
> classification existed anywhere in the repository for this parameter,
> and no VoLL methodology or study is referenced in any `docs/*.md`
> (exhaustive search, TASK 01 item 20).

> **Why it matters.** VOLL enters `system_cost_npv` two ways: directly,
> via `unserved_npv` in both builders (`src/optimize_model.py:1545-1548`;
> `src/optimize_model_sliced.py:1842-1845`), unconditionally; and
> conditionally, via `selfgen_price`'s own fallback
> (`econ.get("GENSET_COST_PER_TWH_E", voll)`, both builders) whenever
> `genset_case=None` — live, by direct check of every `load_econ(...)`
> call site in `scripts/`, at `22_salvage_sensitivity.py`,
> `23_run_selfgen_frontier.py`'s own "voll"-pricing arm (deliberate, by
> that script's design), `probe_2_6_structure.py` (its `--genset-case`
> flag defaults to `None` specifically "to probe the commit-A fallback")
> and `scripts/verify_2_6_commit_a.sh` (an explicit regression-test
> fallback). Every live REL runner (`09_run_rel1_feasibility.py`,
> `10_run_rel2_marginal_cost.py`, `11_run_rel3_financing_frontier.py`)
> and `30_run_stochastic_redispatch.py` pass a real `genset_case`
> (`"genset_central"`), so for those four the fallback never triggers —
> but these are exactly the runners (TASK 01/05) where unserved energy
> is not suppressed to near-zero by construction the way it is in a
> typical capacity-planning runner, so VOLL's precise value is more
> directly load-bearing there than elsewhere.

> **The backstop multiple, recorded not applied.** `voll_low`/`voll_mid`/
> `voll_high` are 11.23x / 22.45x / 67.35x the model's own
> self-generation price under the canonical genset arm (`genset_central`,
> 445,416,246 USD/TWh_e, `data/cost/processed/genset_cost.csv`, itself
> tagged **PARTIAL**, not SOURCED) — arithmetic from parameters already
> in the model (TASK 05 Part 2B), recorded as the basis a later sourcing
> decision could argue from. **No VOLL value was changed on the strength
> of this ratio**, and the ratio is not itself a source — it compares
> one unsourced/partially-sourced figure against another.

> **No VOLL sensitivity runner exists.** Checked `scripts/` and
> `scripts/deferred/` directly (TASK 05 Part 1): no runner sweeps
> `voll_case` or overrides `UNSERVED_ENERGY_PENALTY`.
> `02_run_ndc_caps.py` and `03_run_ndc_eaas.py` each carry dormant
> sweep scaffolding (`voll_cases = [CANONICAL_VOLL]`, a single-element
> list looped trivially) that has never evaluated more than
> `"voll_mid"` in practice — not a sweep. Unlike the social-discount-rate
> gap (row 15, `docs/parameter_provenance.md`), where a sweep (L59)
> showed the physical plan is invariant across the full tested range,
> no equivalent evidence exists yet for VOLL — whether this parameter's
> value matters for the physical plan, the monetary results, or both is
> not established.

> **Direction of bias.** Not established — there is nothing to compare
> the current value against in either direction.

> **What would resolve it.** A primary VoLL study (ideally Nigeria- or
> West-Africa-specific), or an explicit ruling to formalise VOLL as a
> declared policy ceiling expressed as a multiple of the genset backstop
> price (the ratio above is the input such a ruling would use) rather
> than treat it as a measured figure awaiting a citation. A VOLL
> sensitivity sweep (`voll_low`/`voll_mid`/`voll_high`, paired across
> the REL runners and `30_` where the parameter is most load-bearing)
> is explicitly out of scope for TASK 05 and is a post-Phase-B item.

> **See also:** `docs/parameter_provenance.md` row 20. L50 (the same
> silent-default/undeclared-provenance failure class, for
> `within_block_peaking_factor`). L59 (the social-discount-rate sweep
> this entry's "what would resolve it" is modelled on).

---

## L68 — `gas_avail_central` remains unsourced: both Nigerian routes failed; TASK 16's candidate figure was rejected for sample-selection bias; `gas_avail_nameplate` stays ASSUMED, now corroborated by a full-fleet benchmark

> **Status:** LIVE — current and ongoing, not foreclosed. `gas_avail_central`
> stays `None` in `gas_availability_scenarios()` (`src/scenarios.py`)
> and still raises `ValueError` if selected (`load_scenario()`'s own
> guard, unchanged across TASK 11, 12, 15, 16 and 17). TASK 11
> (`docs/task_log_11_gas_availability.md`) established no Nigerian
> source decomposes gas-plant unavailability by cause. TASK 12
> (`docs/task_log_12_gas_availability_sourcing.md`) attempted to source
> `gas_avail_central` from a Texas Reliability Entity (**Texas RE**, the
> North American Electric Reliability Corporation's regional entity for
> the ERCOT Interconnection — not to be confused with the **Nigerian
> Electricity Regulatory Commission (NERC Nigeria)**, source of the
> 0.295 [0.3061 since TASK 73, L128] blended Plant Availability Factor already in
> `gas_avail_observed`) benchmark but fetched the wrong Texas RE
> document and stopped. TASK 15 Part E
> (`docs/task_log_15e_gas_availability.md`) redesigned the search to
> require a Nigerian-grounded route first, with the Texas RE figure
> demoted to corroboration only — attempted both Nigerian routes; both
> failed. TASK 16 Part A then attempted a fleet-weighted derivation
> from the only plant-level technology data available at the time and
> produced a candidate (0.8880) that TASK 17 Part A.0 showed was biased
> by its own sample selection — **rejected, not adopted, recorded below
> rather than silently dropped.** TASK 17 Part A instead found a true
> full-fleet plant-level inventory and computed an unbiased benchmark
> (B = 0.8307) — **not used to fill `gas_avail_central`**, and, per
> TASK 18 Part C's correction of TASK 17 A.3's own instruction, **not
> used to retag `gas_avail_nameplate` as DERIVED either**: 0.80 predates
> B entirely, so B can corroborate 0.80 (check it against an external
> benchmark) but cannot have produced it. `gas_avail_nameplate` keeps
> its ASSUMED tag (`docs/parameter_provenance.md` row 21, corrected);
> `gas_avail_central` remains exactly as unsourced as before. The two
> arms are conceptually different and this distinction matters either
> way.

> **Route A attempted — dedicated-supply Nigerian subsample. Failed:
> no documentary basis for the classification.** The method needed
> grid-connected Nigerian gas plants identifiable, from primary
> documents, as drawing on dedicated or ring-fenced upstream gas supply
> (for which feedstock unavailability is near-negligible, so their
> blended PAF would approximate the fuel-agnostic quantity the model
> needs). Checked all four 2025 NERC Nigeria quarterly reports
> (`data/gas/raw/NERC_report/*.pdf`, full-text extracted) for any
> plant-level gas-supply-arrangement narrative — none exists; every
> occurrence of "dedicated"/"ring-fenced" in these documents concerns
> the Commission's own complaint-handling unit or DisCo operations, not
> gas supply. Checked every other file in `data/gas/` (assumptions
> notes on heating value, conversion efficiency, and the deliverability
> fuel-volume series; raw NUPRC/thermal-generation spreadsheets) — none
> attributes a supply arrangement to any named plant either. Separately,
> even a documented classification could not currently be weighted from
> `data/capacity/raw/gas_plant_availability.csv`: that file holds only
> 16 open-cycle plants and **excludes Okpai_1 entirely**, despite
> Okpai_1 appearing in the primary NERC report itself (Q4 2025, Table 1:
> 480 MW installed, PAF 68.12%/57.72%) — the combined-cycle and
> gas-steam fleet is absent from the processed data pipeline (the same
> gap `docs/methodology_claims_ledger.md` row 162 already names). Route
> A is reported as blocked, not worked around with an outside-the-documents
> classification the task's own instructions excluded.

> **Route B attempted — System Operator constraint decomposition.
> Failed: no stable primary located.** Nigeria's grid operator has
> historically reported daily constrained-capacity figures by cause
> (gas/line/frequency/water) under NESIStat, the Power Sector Recovery
> Programme, or National Control Centre daily reports. Searched directly
> (web search, four targeted queries; two site fetches — NISO's main
> site, which redirect-looped, and its Grid Notices page, which holds
> only individual outage/force-majeure notices, not a systematic
> cause-coded daily series) — found no stable, downloadable primary.
> Every hit naming the figures this method needs (e.g. "2,648.7mw to
> gas, line constraint") is press coverage quoting an unarchived daily
> report, which the task's own instructions excluded as a source. The
> Rural Electrification Agency's statistics page, which search results
> suggested might host per-plant constraint data, returned HTTP 404.

> **Route C (Texas RE) — fully verified; now used for a narrower
> purpose than originally framed.** `literature/2021_Assessment_of_Reliability_Performance_Appendices.pdf`
> (the correct Appendices volume — TASK 12 had fetched the narrative
> annual report by mistake) was checked page-by-page against every
> figure in the TASK 12 sourcing memo: Table A.3 (p.13, fleet-wide
> 2017-2021 Equivalent Availability Factor: 85.3/85.2/86.1/84.1/83.8,
> 5-yr avg 84.5%), Table A.4 (p.14, by technology: Gas 72.4% (40 units),
> Gas Turbine/Jet Engine 88.8% (109), Combined Cycle Block 76.8% (18),
> Combined Cycle GT 82.3% (149), Combined Cycle ST 82.0% (61)), Section H
> "Fuel Constraints" (p.24) and Figures A.27/A.28 (p.25, reported
> separately from Section E's availability metrics, as claimed), and
> Table A.6 (p.18, "External" forced-outage cause: 42,933.8 MW of
> 487,129.6 MW total, 8.8%, matching Table A.5's own total). **Every
> figure confirmed exactly, zero discrepancies.** The 2021-vs-Uri
> question is also resolved in the document's own data: 2021's own EAF
> (83.8%) sits 0.7 percentage points below the 5-yr average (84.5%),
> confirming the task's own expectation that the storm's fleet-level
> effect is under one point. This verified table is what both TASK 16
> and TASK 17 drew their technology-class EAF values from — Route C
> itself was never the problem; how each task used it is what differs
> (see below). **Still not used to fill `gas_avail_central`.**

> **TASK 16 Part A's candidate (0.8880) — measured, then rejected for
> sample-selection bias, not silently dropped.** TASK 16 Part A weighted
> Table A.4's technology EAFs by the only plant-level technology data
> then available: `data/capacity/raw/gas_plant_availability.csv`, 16
> plants, 6,482 MW. Every one of those 16 plants is `turbine_type=OC`
> (open-cycle) — the file's own population is the *conversion-eligible
> open-cycle subset* (Phase 2.8's OCGT-to-CCGT candidates), not the
> fleet. Weighting a 100%-open-cycle sample against Table A.4
> necessarily returns the GT/Jet Engine row's own EAF (88.8%) with no
> actual cross-class weighting taking place — TASK 17 Part A.0's
> diagnosis: **the selection criterion drives the result**, and
> simple-cycle carries the highest EAF of any class in Table A.4 (88.8%
> against 72.4% for gas steam and 76.8–82.3% for combined cycle), so
> 0.8880 is biased upward by the US-to-Nigeria transfer *and* by the
> sample composition, compounding in the direction that makes the
> model's most variance-dominant parameter more permissive. **Rejected.
> Not used to fill `gas_avail_central` or any other arm.** Full
> derivation and the rejection reasoning: `docs/task_log_16a_gas_avail_derivation.md`,
> `docs/task_log_17a_fleet_benchmark.md` A.0.

> **TASK 17 Part A's full-fleet benchmark (B) — corroboration, not a
> derivation (corrected by TASK 18 Part C).** `data/capacity/raw/gas_plant_vintages.csv`
> turned out to be the full-fleet, plant-level, technology-labelled
> inventory Route A needed all along — 23 plants (after excluding
> AES_1/Gbarain_1, the same exclusion `GAS_BASELINE_MW` itself
> applies), 10,947 MW, carrying Okpai_1 (CC, 480 MW), Afam VI/`Afam_2`
> (CC, 650 MW) and Egbin_1 (GS, 1,320 MW) — exactly the plants Route A's
> dedicated-supply search and TASK 16's sample both lacked. Mapped
> cleanly to all three Table A.4 classes with zero unmapped rows (CC
> 21.7411%, GS 19.0463%, OC 59.2126% of fleet MW) and weighted: **B =
> 0.8307** (0.830675 unrounded). TASK 17 A.3 originally instructed
> writing this up as a DERIVED provenance row for `gas_avail_nameplate`
> (B minus a declared Nigeria derate). **That instruction was wrong,
> and TASK 18 Part C corrected it**: 0.80 predates B entirely — it is
> the pre-2.10 unsourced assumption, set long before this benchmark
> existed — so (B − 0.80) is a residual, not an estimate, and tagging
> it DERIVED would assert the value was computed from a source found
> only after the value was already fixed. `docs/parameter_provenance.md`
> row 21 now tags `gas_avail_nameplate` **ASSUMED**, with B recorded as
> **corroboration**: 0.80 sits 0.0307 below B (3.07 points, 3.83%
> relative), on the conservative side, with no "derate" constructed
> from that gap. **Does not resolve this entry either way**:
> `gas_avail_central` is a conceptually different arm — the model's
> intended unbiased central estimate, not a declared upper bound — and
> TASK 17 A.5 explicitly left it unfilled, `None`, and raising; nothing
> in TASK 18 Part C changes that.

> **Direction and magnitude of the gap this leaves, updated.** Now
> partially known where it was previously unknown: a full,
> technology-correct Nigerian-fleet weighting of the Texas RE benchmark
> lands at B = 0.8307, not at TASK 16's rejected 0.8880 — so *if*
> `gas_avail_central` were ever filled by a similar transfer
> methodology, 0.8307 is the better-grounded candidate figure, not
> 0.8880. Still unmeasured: the same US-to-Nigeria transfer bias that
> applies to any version of this benchmark applies to B as much as it
> applied to 0.8880 — B is a sourced *transfer*, not a Nigerian
> *measurement*. The 0.0307 gap between 0.80 and B is **not** a
> measurement of the true Nigeria-specific shortfall (TASK 18 Part C: it
> is a residual against an assumption that predates B, not an estimate
> of anything) — maintenance regime, parts supply chain, ambient/dust
> conditions and plant age profile remain reasons to *expect* a true
> Nigerian figure below B, not a quantification of how far below. Route
> A's and B's original failures (above) still stand: there is still no
> Nigerian *measurement* to compare any transferred figure against.

> **What would resolve it.** Unchanged in kind, sharpened in target:
> (a) a Nigerian document attributing gas supply arrangements to
> specific plants (a GenCo annual report, an NUPRC or NNPC
> gas-allocation record, or a future NERC Nigeria quarterly that adds
> cause-coding to its PAF table), now re-attemptable against the full
> 23-plant inventory `gas_plant_vintages.csv` provides, including
> Okpai_1 and Afam VI; or (b) a located, stable primary for the System
> Operator's constrained-capacity-by-cause series (Route B). Absent
> either, `gas_avail_central` should stay `None` and raising — an
> honest gap, per this task's own framing, beats a filled field resting
> on a transfer nobody has grounded in a Nigerian measurement, B = 0.8307
> included.

> **See also:** `docs/task_log_11_gas_availability.md`,
> `docs/task_log_12_gas_availability_sourcing.md`,
> `docs/task_log_15e_gas_availability.md`,
> `docs/task_log_16a_gas_avail_derivation.md`,
> `docs/task_log_17a_fleet_benchmark.md`, `docs/task_log_18c_gas_avail_tag.md`.
> `docs/parameter_provenance.md` row 21 (`gas_avail_nameplate`, ASSUMED,
> corroborated by B — not DERIVED from it).
> `docs/methodology_claims_ledger.md` row 162 (the same
> `gas_plant_availability.csv` coverage gap, found independently
> there).

---

## L69 — A results manifest's `Tag SHA` line is a caller-supplied claim, not a verified fact: nothing checks it against `git rev-parse`

> **Status:** LIVE. `scripts/make_results_manifest.py` writes a
> `Tag SHA` line into every manifest's header, meant to pin the
> manifest to a specific point in the repository's history — the tag it
> precedes, or current HEAD (labelled as such) if that tag does not
> exist yet. TASK 15 Part F found this line hardcoded to resolve the
> literal tag name `pre-phase-b`: a manifest meant to precede a
> *different* tag (`phase-b-start`) silently reported the wrong SHA —
> the old `pre-phase-b` tag's commit, 35 commits behind the actual HEAD
> at generation time — with no error of any kind.
> `docs/task_log_15f_phase_b_hygiene.md` records the fix Zeus chose: a
> `--tag` CLI argument, defaulted to `pre-phase-b` to leave TASK 00's
> original invocation unchanged, verified correct in both directions
> (commit `2146740`). **That instance is fixed.**

> **What the class of defect is, which parameterization alone does not
> close.** The script has no way to verify, at generation time, that
> the `--tag` value a caller supplies is actually the tag that will be
> created at that commit, or to catch a typo, a stale leftover value,
> or a tag-name collision. The manifest's header simply echoes back
> whatever string the caller passed; it carries identical-looking
> authority whether the caller got it right or wrong. A manifest's
> self-reported `Tag SHA` is therefore a **claim**, not a checked fact —
> the same undeclared-provenance failure shape as L50 and L67, applied
> here to a file whose entire purpose is provenance.

> **Why it matters.** These manifests are what a later chapter cites to
> tie a specific results figure to a specific state of the repository
> tree (TASK 16's own framing, in the context of citing
> `docs/results_manifest_phase_b_start.md`). If a manifest's `Tag SHA`
> line is wrong — whether from this specific hardcoding bug, now fixed,
> or from a future `--tag` typo or stale argument — nothing in the
> manifest itself signals the mismatch. A reader, including a thesis
> examiner checking provenance, has no way to detect it without
> independently re-running `git rev-parse` and comparing by hand. A
> manifest that misidentifies its own tree is a provenance failure, not
> a cosmetic bug: it breaks the exact chain of custody the file exists
> to provide.

> **Direction of the bias.** Not directional. The mismatch can point to
> an earlier or a later commit than the one actually manifested,
> depending on what the caller's `--tag` value (or its omission)
> happens to resolve to at generation time — unlike most entries in
> this register, there is no single predictable direction to correct
> for.

> **What would resolve it.** Procedurally, immediately available: before
> citing any `results_manifest_*.md` file as evidence tying a figure to
> a tree, check its stated `Tag SHA` against `git rev-parse <tag-name>`
> (or `git rev-parse HEAD` if the tag does not exist yet) independently,
> rather than trusting the file's own header. A structural fix, not
> built here because it is outside this task's scope, would have the
> script itself verify at generation time that its `--tag` argument
> resolves to an actual reachable commit consistent with the caller's
> stated intent, or refuse to run silently mismatched — closing the gap
> parameterization alone left open.

> **See also:** L50, L67 (the same undeclared-provenance failure class,
> for other quantities). `docs/task_log_15f_phase_b_hygiene.md` (the
> instance this entry generalises from). `docs/task_log_16c_manifest_register.md`
> (this entry's own report).

## L70 — A binding-year mean is a composition artefact: it can fall while every binding year gets worse

**Status:** LIVE (reporting rule, not a model defect)

**What.** REL-2 at `gas_avail_observed`, `downside`
(`results/rel2/gas_avail_observed/cost_curve_binding.csv`): as eps
tightens 0.15 -> 0.10 -> 0.08, `binding_years_n` goes 2 -> 3 -> 9 and
`dual_pv_usd_per_twh_binding_max` rises monotonically 1.63e8 -> 4.47e8
-> 5.09e8, but `..._binding_mean` goes 1.07e8 -> 2.56e8 -> **1.27e8** and
`..._binding_median` 1.07e8 -> 1.76e8 -> **1.23e7**. Newly binding years
carry small duals and dilute the mean; the statistic changes because the
population it averages over changes.

**Why unresolved.** It is a property of the statistic, not of the model.

**Direction of bias.** A binding-year mean understates tightening at the
point where the binding set grows; the maximum does not.

**Mitigation.** Chapter 4 must plot the **maximum and the median**, not the
mean, for any binding-conditioned dual; the same applies to
`gas_shadow_mean`-type quantities (F-30 gives both all-year and
binding-year means).

**What would resolve it.** Nothing to resolve; carry the reporting rule.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L71 — Feasibility and frontier results are invariant to the gas availability arm; cost and dual results are not

**Status:** LIVE

**What.** Between `gas_avail_nameplate` and `gas_avail_observed`: REL-3
(`results/rel3/gas_avail_*/frontier_shift.csv`) is identical in every
threshold and shift. REL-1's `feasibility_threshold_eps` and feasibility
status are identical (`results/rel1/{total,annual}/gas_avail_*/`), but
**costs, duals, `solar_mw_at_threshold` and the shortfall-by-year
distribution differ** (e.g. F-33 Part 1: worst year 0.2353 vs 0.2099) —
so "REL-1 output is identical" is true of attainability only. REL-2
changes completely: two binding regimes (downside onset 0.20,
shock_recovery 0.05; baseline and upside never bind) become four, with
onsets ordered 0.15 / 0.10 / 0.08 / 0.05 (downside / shock_recovery /
baseline / upside). Under unconstrained gas build, lower availability is
answered by building more capacity, which raises the cost of a standard
without changing its attainability.

**Why unresolved.** Structural to `gas_build_case=gas_unconstrained`.

**Direction of bias.** Attainability results are insensitive to the
availability assumption; cost and dual results are conditional on it.

**Mitigation.** This bounds the exposure of §3.7's firm-capacity passage
to the cost-based results only. Report the arm with every cost/dual
figure.

**What would resolve it.** A sourced `gas_avail_central` (L68).

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L72 — Threshold grids stopped short three times and three conclusions changed; a fourth runner has the same pattern unflagged

**Status:** PARTLY RESOLVED. The Pareto frontier's tight end was extended and its loose end located (below). The REL floor-bound flagging stands; the 'decision for Zeus' on item 20 is taken. *(Prior status, retained as history: LIVE (item 20 open; decision for Zeus))*

**Corrected 2026-10-23 (TASK 69 A.2).** `results/pareto_frontier/arms_92390a63/pareto_raw.csv` (2026-10-20, 128 solves = 4 arms x [anchor + 25 grid + 6 tight-end points]) has the tight-end points and the uncapped anchor (504.1 Mt), all `optimal`; F-43 carries the 0.01 end (audit pass 3 A.2). The 'tight end is untested' direction-of-bias sentence is therefore superseded; the concern becomes that no feasibility boundary exists for a grid to stop short of. Residual: F-43's quantities are 2025-2054 totals (pass 3 B.2), a separate matter.

**What.** REL-2 (TASK 13), REL-3 (TASK 22) and REL-1 (TASK 23) each
reported a clean negative at the original grid floor. Verified here for
REL-3 (floor 0.20: 16/20 cells floor-bound, max shift 0.05 -> 8
advantaged cells, max 0.17 at floor 0.05) and REL-1 `total` mode
(floor 0.05 `INDETERMINATE` -> `GAS_REGIME` at 0.001; the `annual` mode
kept `gas_regime` and moved 0.03 -> 0.079). The REL-2 reversal rests on
`docs/task_log_13_rel2_grid.md` and was not re-derived in TASK 25.
Floor-bound flagging now exists in all three. **Item 20's Pareto
frontier (`scripts/18_run_pareto_frontier.py`) shows the same pattern
unflagged:** `cap_fractions = np.linspace(1.0, 0.10, 25)`
(line 308); all four arms report `tightest_feasible_cap_fraction=0.1`,
`loosest_infeasible=null`; all 25 of 25 points per arm are `optimal`, so
the 0.10 boundary (1 point in 25 per arm, 4 of 100 overall) is the grid
floor, not a feasibility limit. Every point including `cap_fraction=1.0`
has `cap_binding=True`, and `unconstrained_optimum_emissions_tco2` is
null, so the loose end is also not located. ~~Grid not changed.~~ **Grid extended (TASK 26 Part B; re-run as Phase B item 20): six tight-end points per arm, cap_fraction 0.08 to 0.01, and an uncapped anchor per arm; 128 solves, all optimal; the frontier never reaches infeasibility.**

**Direction of bias.** The cost-emissions frontier's tight end is
untested, not shown unreachable; cost at the tight end is a lower bound
on the true frontier's extent.

**Mitigation.** None yet. Do not quote the frontier's tight end as a
limit.

**What would resolve it.** ~~Extend the grid below 0.10 (a decision for Zeus; it affects a headline bi-objective result).~~ **Done; see the correction above.**

> **Generation (TASK 86):** the hash-keyed result files this entry cites are blended dispatch (no `gas_vintage_dispatch=True` in the cited files' arm records). Checked: `92390a63` (blended). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142. [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L73 — The genset arm is a segment switch, not a sensitivity range

**Status:** LIVE

**What.** `data/cost/processed/genset_cost.csv`: `genset_srmc` and
`genset_central` are large, part-loaded C&I diesel at eta_HHV 0.3474 and
0.2809; `genset_high` is a 3 kVA petrol set at 0.0511, metered by A2EI at
7.40 kWh/month. Implied EF_e = EF_th/eta = 0.729, 0.902 and 4.638
tCO2/MWh. Total cost $276.8 / $445.4 / $3,323.3 per MWh_e (a 12.0x
range). The 1,359.2 Mt figure is `genset_high`'s self-generation
emissions in the leakage arm (`results/leakage_arm/leakage_arm__*.csv`,
item 6; `genset_srmc` 213.8 Mt, `genset_central` 264.4 Mt). Self-generation
volume is 293.04 TWh in all three arms (0.00% spread), solar built and
grid emissions are identical, and the objective moves 43.9 / 54.0 /
225.8 bn: the arm's cost dimension moves the objective and nothing else;
its emission-factor dimension moves self-generation emissions.
`selfgen_in_cap=True` is infeasible for all three arms. (Item 8's
`genset_arm_sensitivity.csv` is a different test: public_only infeasible
at every eps, EaaS feasible with 0.0 TWh self-generation in every arm.)

**Direction of bias.** Quoting the petrol arm's emissions as a
sensitivity bound on the C&I population overstates leakage emissions
about 5x.

**Mitigation.** Chapter 4 should quote `genset_central` and report the
range with the segment interpretation attached.

**What would resolve it.** A sourced population split between the two
segments.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L74 — The Pareto frontier's anchors and tight-end extension have no pre-correction counterpart, so the margin correction's effect on them is unmeasured

**Status:** LIVE

**What.** The pre-correction snapshot of Phase B
(`/home/zeus/thesis-snapshots/results_pre_margin_fix/pareto_frontier/`) holds
the 100 original grid points (cap_fraction 1.0 to 0.10). The four uncapped
anchors and the six tight-end extension points per arm (cap_fraction 0.08 to
0.01; 24 points) were added by TASK 26 Part B and first solved under the
corrected `required_margin`, so there is no before-and-after for them. F-43
carries the statement; TASK 29's quantification of the correction on the EaaS
arms (+14% to +23% in cost at cap 0.10) is confined to the original grid.

**Why unresolved.** Producing the missing "before" means re-solving the 24 +
4 points with the forced 1.10 margin restored, a deliberately wrong model,
which the project removed (L34). Not worth a code revival.

**Direction of bias.** Applies to the EaaS arms only (the public arms carry no
margin and are bit-identical where checked). The correction raised EaaS cost
and emissions at every moved grid point, increasingly toward tight caps, so
the *uncorrected* tight-end extension would have shown lower EaaS cost and
emissions than the corrected one; the size is unknown and the grid-point trend
(+14% and +23% at 0.10; `eaas_tight` peaks at +18.2% before 0.10 and
`eaas_moderate` is still rising at 0.10) gives no reason to think it was smaller.

**Mitigation.** Quote the extension only as corrected values; never as a
comparison with the pre-correction frontier.

**What would resolve it.** Nothing needed; it is a record of what cannot be
compared.

## L75 — The documented route to regenerate the emissions cap was unsatisfiable after the A5 port

> **Status:** CLOSED (TASK 32, 2026-10-08; docstring rewritten and the route tested). Residue: L61 (the bare `diagnostics.json` handoff is manual and unstamped) stays LIVE.

> **What.** `scripts/00_build_emissions_cap.py`'s docstring (and its `KeyError` message) told a reader to regenerate the cap by re-running `01_run_baseline.py` under `conv_none` and insisted the baseline be "an ANNUAL conv_none run ... not a sliced run". Since the A5 port (`c549bba`, 2026-10-01) `01_run_baseline.py` builds sliced, so the documented route could not produce what the script said it required, and the cap could not be reproduced as described. The file in use (`results/baseline/diagnostics.json`, 2026-09-26) was an annual-builder result, which made the cap's `E_base` (15.745 MtCO₂/yr) the one annual-builder-derived input behind every capped finding (`task_log_31a`, addendum).

> **Resolution.** The equivalence the old wording could not assume was tested: `01_run_baseline.py` at HEAD under `conv_none` (sliced), cap regenerated to a comparison path: `E_base` **15.7450032** against the stored **15.7450033** MtCO₂/yr, and all 150 cap values agree to a relative 8e-9 (largest absolute difference 0.073 tCO₂ on caps of order 1e7). `emissions_cap.csv` was not overwritten. The requirement was always "no conversion" (the annual builder could only ever run `conv_none`); it is stated that way now, with the exact three-step regeneration route, its two overwrite hazards, and the test result.

> **Direction of bias.** None realised: the cap values did not move. The risk was irreproducibility (and, had `E_base` moved, every capped finding with it).

> **What would resolve it further.** Nothing required. L61 (stamping the bare handoff) remains the open part of the same dependency.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L76 — `environment.yml` lists dependencies the study never uses and omits ones it needs; the Sobol indices come from SciPy, not SALib

> **Status:** LIVE (recorded 2026-10-08; `environment.yml` deliberately not edited, because editing it would change the environment the results were produced in).

> **What.** `environment.yml` lists `SALib` (pip section). The sensitivity indices (M-4; Phase B item 22, `scripts/29_run_parameter_uncertainty.py`) are computed by **`scipy.stats.sobol_indices`** (SciPy 1.15.2, estimator `saltelli_2010`); the script imports it at line 218 and says "SALib is NOT installed and is not used". `SALib` is not installed in the environment the results were produced in and is not imported anywhere in `src/`, `scripts/`, `tests/`, `data/make_data.py` or the notebooks. A reader of the dependency list would infer SALib produced the indices; a person rebuilding from `environment.yml` installs a package that does nothing. Same finding for six more entries: **`plotly`, `pypsa`, `tqdm`, `joblib`, `seaborn`, `pvlib` are listed, not installed, and not imported anywhere** (seven in all with SALib).

> **The other direction (more consequential).** The repository has **two** environment files. The one that matches the active environment (`nigeria-energy`, conda prefix `~/miniconda3/envs/nigeria-energy`) is **`environment_linux.yml`** (commit `31aaf3f`, "Linux environment pinned after WSL migration"): it lists `highs`, `coincbc` and `pyyaml` and none of the unused packages. **`environment.yml` omits the two solvers (`cbc`, which `solve_model` calls by default, and `highs`) and `pyyaml` (imported by `src/io.py` and `scripts/build_gas_deliverability.py`).** So `environment.yml` alone does not reproduce the results: it would install unused packages and lack the solvers and a required import. Not a defect in any result.

> **Also found.** `lxml`, `xlrd` and `openpyxl` (in both files) are installed but imported nowhere (no `read_excel`, `ExcelFile` or `read_html` call in `src/` or `scripts/`); `requests` is used only by `data/make_data.py` (a data-acquisition script outside `src/` and `scripts/`). Used by the study's code: `numpy`, `pandas`, `scipy` (one import, the Sobol runner), `matplotlib`, `pyomo`, plus the solver binaries and `pyyaml`.

> **Direction of bias.** None on any result: the code that ran is the code that is cited. The risk is misattributed method (SALib vs SciPy) and an irreproducible environment from the wrong file.

> **Mitigation.** Cite SciPy for the indices (§3.6.7 does: Virtanen et al., 2020). Use `environment_linux.yml` to rebuild.

> **What would resolve it.** After the thesis results are frozen: make `environment_linux.yml` the single environment file (or generate `environment.yml` from `conda env export` of the active environment) and remove the unused entries. Not done now by instruction.

## L77 — The Sobol runner appends to its draws file and builds `qoi_summary` and `feasibility` from the whole file

> **Status:** RESOLVED in the runner (TASK 53 2.7) (was: LIVE (recorded 2026-10-09, TASK 38; script deliberately not edited; the defect was first found in TASK 26 Part C and handled in the findings, not registered until now)).

> **What.** `scripts/29_run_parameter_uncertainty.py` opens `draws_<n>_seed<s>.jsonl` in append mode (`run_one_draw`) and `main()` reads the whole file back after `sobol_indices` returns to build `feasibility` and `qoi_summary`. `results/parameter_uncertainty/draws_n256_seed42.jsonl` holds 5,376 rows: three appended runs of the same 1,792 draws (rep 0 from an earlier code state, reps 1 and 2 bit-identical). `summary_n256_seed42.json` therefore pools them: cost mean 31.733 bn, sd 2.786, `feasibility.n_total` 5,376. The logged run alone (rep 2): mean 31.493, sd 2.751, p5 27.796, p50 30.928, p95 36.439, 1,792 draws, 0 failed.

> **What is not affected.** The Sobol indices are computed from `func`'s in-memory return values (`sobol_indices(func=func, …)`), not from the file (read in TASK 38 Part D.3; TASK 26 Part C replayed rep 2 through scipy, max abs difference 0.0).

> **Hit list (TASK 38 Part D).** Pooled figures appear in `docs/task_log_25_phase_b_findings.md` lines 99–100 (erratum at line 127; original not struck) and are quoted as the thing to correct in `task_log_26c_sobol_rows.md` and `task_log_27c_sobol_correction.md`; `findings_frozen.md` M-4 quotes them only to warn. No chapter, appendix or skeleton text quotes them.

> **Direction of bias.** Pooled `qoi_summary` overstates the mean by 0.24 bn (0.8%) and the sd by 0.035 bn. No effect on any index.

> **Mitigation.** Quote `qoi_summary` only from the rep-2 rows (or recompute). Do not re-run item 22 without deleting or renaming the draws file.

> **What would resolve it.** Truncate the file at the start of `main()` (or write a per-run file) and build the summary from the in-process results. Not done: the indices do not depend on it.

> **TASK 53 update.** `29_run_parameter_uncertainty.py` now writes to `results/parameter_uncertainty/arms_<hash8>/` (keyed by every arm, `n_base` and `seed`), opens the draws log exclusively (`"x"`), aborts before solving if the directory holds anything but `ARMS.json`, and raises unless the log holds exactly N(d+2) rows. The stored `results/parameter_uncertainty/draws_n256_seed42.jsonl` (5,376 rows) is untouched and is superseded by the re-run; `qoi_summary` and `feasibility` computed from it remain contaminated until then.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L78 — The handover's "NDC cap 15.8–28.5 bn" EaaS value has no source (documentation defect, not a model defect)

> **Status:** RESOLVED by annotation: the unsourced range is withdrawn in the handover itself and Table 4.4 is rebuilt without it. *(Prior status, retained as history: LIVE (recorded 2026-10-09, TASK 38 Part B.4).)*

> **Corrected 2026-10-23 (TASK 69 A.2).** `docs/HANDOVER_chapter_4.md:177` ('Further superseded, 2026-10-18 (TASK 48 B.6; TASK 47 C28) ... Withdrawn (register L78)'); Table 4.4 is rebuilt from `results/dem2/eaas_value.csv` alone (the entry's own mitigation) (audit pass 3 A.2).

> **What.** `docs/HANDOVER_chapter_4.md` §4.8 (line 152; commit `423b0af`, the only commit touching it) and the chapter-4 skeleton's Table 4.4 draft quote "an NDC cap (15.8–28.5 bn)". No document or result file states that range or values for it. The nearest stored numbers come from different designs: `results/pol1/eaas_rescue.csv` 15787639411.96 (POL-1, ndc3_conditional) and the pre-TASK-29 F-36 NDC3 unconditional mean saving 28.53 bn (now 28.43). Whether either is the origin is not established. The same handover paragraph's "high growth (36–63 bn)" omits F-38's 23.1 bn cell, and "conditional NDC ~nothing" has no stored USD value (F-35's quantity is an ε-threshold shift).

> **Direction of bias.** None on any model result. Risk: a table row quoted from a figure that no result file supports.

> **Mitigation.** Table 4.4 is rebuilt from `results/dem2/eaas_value.csv` alone; no row uses the handover range.

> **What would resolve it.** ~~Correct the handover §4.8 text.~~ **Done by annotation (not deletion): `docs/HANDOVER_chapter_4.md:177` carries 'Withdrawn (register L78); do not quote it' (TASK 48 B.6, 2026-10-18); the historical paragraph keeps the range by design.**

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L79 — GAS-2 (F-34) was cleared for the `required_margin` correction by reading code, not by re-running; its stored files carry the wrong label, and its configuration differs from DEM-2's in the solar build arm

> **Status:** RESOLVED: GAS-2 (F-34) was re-run post-epoch (40 solves); the mislabelled files are the superseded pre-epoch ones. The capped-solar design caveat lives on in L81. *(Prior status, retained as history: LIVE (recorded 2026-10-09, TASK 38 Part C; nothing re-run).)*

> **Corrected 2026-10-23 (TASK 69 A.2).** F-34 TASK 55 block (`docs/findings_frozen.md`): 'Post-epoch `results/gas2/arms_be9472ef/eaas_vs_noeaas_gas_shadow.csv` (40 solves ...)'; the directory exists (`ls results/gas2`). The entry's 'nothing re-run' and 'outside TASK 29's re-run set' were true on 2026-10-09 only (audit pass 3 A.2).

> **What.** Phase B item 10 (`07_run_gas2_eaas_gas_relief.py`, `45587af`) is outside TASK 29's re-run set. It was scoped out in `task_log_28a_required_margin.md` and the F-34 "TASK 29 CHECK" on the reading that the `NDC_CASES` margin fields (1.10/1.05) never reached a solve. ~~`results/gas2/*.csv` still label 12 rows 1.1 and 8 rows 1.05.~~ **Those are the pre-epoch files; superseded by `results/gas2/arms_be9472ef/` (post-epoch re-run).** It is one of the five findings of handover §8 (enumerated in `task_log_27d_register_triage.md`: F-34, F-35, F-36, F-37, F-38); the other four were re-run. F-34 is the only `findings_frozen.md` entry citing GAS-2. The file stores no budget dual or utilisation. **GAS-2 and DEM-2 differ in `solar_build_case`** (GAS-2's no-policy legs take the `load_scenario` default `nirp_trajectory`, a cumulative solar cap; DEM-2 passes `deployment_unconstrained`). GAS-2 downside no-policy: public 38049381729.35, EaaS 33409095158.12 (difference 4.64 bn); DEM-2: 37261137811.13, 29386251600.72 (7.87 bn). A re-run would cost 40 solves, about 50 s.

> **Direction of bias.** Not measured. The margin cannot be the cause of the GAS-2/DEM-2 gap (never applied).

> **Mitigation.** Quote F-34 as a GAS-2 result with its own configuration; do not set it beside DEM-2 values (including the 7.87 bn gas-downside EaaS value).

> **What would resolve it.** ~~Re-run item 10 with the corrected labels (40 solves).~~ **Done: F-34's TASK 55 block re-derives it from `results/gas2/arms_be9472ef/eaas_vs_noeaas_gas_shadow.csv` (40 solves).**

> **Generation (TASK 86):** the hash-keyed result files this entry cites are blended dispatch (no `gas_vintage_dispatch=True` in the cited files' arm records). Checked: `be9472ef` (blended). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142. [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L80 — F-10 rests on the pre-B\* archive; its stated design caveat contradicts the stored summaries; three stored sets give three different NDC EaaS savings

> **Status:** LIVE (recorded 2026-10-09, TASK 38 Part B.1; nothing repaired).

> **What.** F-10's four figures (19.79, 12.58, 22.36, 9.66 bn) reproduce at full precision from `results/_pre_bstar_20260920_1132/ndc/` and `ndc_eaas/` (19787514134.83, 12577690409.11, 22358559488.35, 9661414167.999). The same configurations in the current `results/ndc/` and `ndc_eaas/` give 14.151, 6.848, 17.851, 5.421 bn; POL-1 (`results/pol1/pol1_results.csv`) gives 17.672, 21.234, 24.195, 15.788 bn. The differences were not diagnosed. F-10's caveat says the public legs ran `capital_case` `unconstrained`; the stored `summary.json` files of both archive and current legs say `tight` (unconditional) or `moderate` (conditional), identical in the two legs, with only `required_margin` and `solar_service_tariff_usd_per_twh` (EaaS-leg keys) differing.

> **Direction of bias.** Unknown.

> **Mitigation.** Do not quote F-10's range as current-epoch or as a mixed-capital comparison without the file check above.

> **What would resolve it.** Refresh F-10 from the current files and state the design from them.

> **Update (TASK 39 Part C, 2026-10-09).** The three stored sets are two designs. The archive and `results/ndc*/` (scripts `02_`/`03_`) pass no `solar_build_case`, so `load_scenario` uses `nirp_trajectory` (a cumulative solar cap); POL-1 (`15_`) passes `deployment_unconstrained`. A GAS-2 re-run (scratch, 40 solves) reproduces `results/ndc*/` to the digit with the cap and POL-1 to the digit without it (three of four cases; the fourth, `ndc3_conditional`, differs because GAS-2 runs it at `tight` and POL-1 at `moderate`). Stored `summary.json` configuration keys are identical between archive and current in all eight files (`required_margin` is 1.3 in both EaaS epochs); the archive-to-current differences are in outcomes only. F-10's `unconstrained` statement is contradicted by all three sets. Detail: `docs/task_log_39_dual_mechanism.md` Part C.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L81 — GAS-2 and DEM-2 are not comparable: GAS-2's solar arm is capped, DEM-2's is not, and F-34's negative case reverses when the cap is removed

> **Status:** LIVE (recorded 2026-10-09, TASK 39 Part B; supersedes the last sentence of L79's "What", which recorded the difference but not its size).

> **What.** `07_run_gas2_eaas_gas_relief.py` takes `load_scenario`'s default `solar_build_case` (`nirp_trajectory`) in all 40 solves (the NDC legs through `run_gas_regime_ndc_matrix`, which passes none); `13_run_dem2_growth_gas_interaction.py` passes `deployment_unconstrained`. A scratch re-run of GAS-2 with only that default changed (outputs in the session scratchpad, `results/gas2/` untouched) moves `gas_shadow_relief_pct` for `no_policy` as follows: baseline 27.07 → 27.07, upside 0.0 → 0.0, downside 36.05 → 86.46, **shock_recovery −10.56 → +35.59**. A capped control re-run at the current HEAD reproduces every stored GAS-2 value (cost and relief) to the digit. F-34's 4.64 bn and F-38's 7.875 bn gas-downside EaaS values come from different solar arms (capped 38049381729.35 / 33409095158.12; uncapped 37261137811.13 / 29386251600.72). The cumulative solar cap binds in 7 years (downside) and 3 / 4 years (shock_recovery, public / EaaS arm) of the capped no-policy runs and in none of the baseline and upside runs.

> **Direction of bias.** F-34's downside relief is understated by the cap (36.05 against 86.46 uncapped) and its shock_recovery sign is a cap effect.

> **Mitigation.** Quote F-34 as a capped-design result with its configuration. Do not set any GAS-2 cost beside a DEM-2 cost.

> **What would resolve it.** Decide which solar arm the thesis's gas-relief claim is made under; if uncapped, replace F-34 with the uncapped run (40 solves, about 45 s).

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L82 — Handover §4.4 states F-12 more narrowly than F-12 does (documentation defect, not a model defect)

> **Status:** RESOLVED by annotation: the handover now carries the correction beside §4.4. *(Prior status, retained as history: LIVE (recorded 2026-10-09, TASK 39 Part D.2).)*

> **Corrected 2026-10-23 (TASK 69 A.2).** `docs/HANDOVER_chapter_4.md:130` ('Corrected 2026-10-10 (TASK 43; L82). F-12 as corrected 2026-10-08 ...'). The §4.4 heading text itself still carries the narrower wording, by annotation rather than rewrite (audit pass 3 A.2).

> **What.** `docs/HANDOVER_chapter_4.md` §4.4 (line 122) reads "The EaaS/public split is unidentified where the budget does not bind (F-12)" and tells the reader to check whether the budget binds before making a statement about what EaaS builds. F-12 as corrected 2026-10-08 says the split is unidentified in every capital case, binding ones included (tight-budget interval 10,969 MW wide at the frozen state; 16,703.8 MW at the current epoch, TASK 44). The narrower wording was carried into the chapter-4 skeleton (line 123, C-4.4.9) and TASK 38 A.2; both amended in TASK 39. An earlier instance of the same wording, in the F-35 text, was corrected 2026-10-08.

> **Direction of bias.** None on any model result. Risk: a composition statement made at a binding cell on the strength of the handover.

> **Mitigation.** F-12 governs. Budget status is reported as a fact about the constraint; it is not an identification condition.

> **What would resolve it.** ~~Correct the handover §4.4 text.~~ **Done by annotation: `docs/HANDOVER_chapter_4.md:130` 'Corrected 2026-10-10 (TASK 43; L82)'; F-12 governs.**

## L83 — B\* no longer equals the traditional regime's measured requirement; the definition (requirement vs reference scale) is undecided

> **Status:** DECIDED, option (b), 2026-10-23 (TASK 69 A.4); entry LIVE as a standing caveat. *(Prior status: LIVE (recorded 2026-10-10, TASK 45; no change made to `capital_envelope_scenarios()` or any result).)*

> **What.** B\* (10,273,343,610 USD, `capital_envelope_scenarios()["adequacy"]`, `src/scenarios.py:896-938`) was derived as the public envelope at which the traditional regime's budget constraint stops binding, and F-22 claimed that the derived value reproduces that property. At the current epoch (`results/capital_frontier/capital_frontier_raw.csv`, 2026-10-03) it does not: at 1.00 × B\* the traditional regime is slack with 511,994,517 USD (4.98%) of the envelope unspent; its measured requirement is the bracket 9,246,009,182.24 (0.9 × B\*, last binding, dual 0.10102134) to 9,761,349,093.39 (1.0 × B\*, first slack, dual 0) of realised spend. B\* therefore overstates the requirement by about five per cent; the tier names (`tight` 0.50, `moderate` 0.85, `adequacy` 1.00, `expansion` 1.20) are fractions of B\*, so every tier moves with the definition.

> ~~**Decision open (Chapter 3).**~~ **Decision taken 2026-10-23 (TASK 69 A.4): option (b).** B\* stays as Chapter 1 §1.6.5 defines it, a reference scale, with the tier names as grid labels; no tier is re-derived and nothing is re-solved for this. The roughly 5 per cent offset (B\* overstates the traditional regime's measured requirement) is stated wherever a tier is called 'fully funded' or 'adequate'. Option (a), not taken, was to redefine B\* at the envelope where the dual first reaches zero (re-derive every tier and re-solve everything solved under a tier). **Open item found while recording this:** the decision text says F-31's title carries the qualifier, but the title in `docs/findings_frozen.md` reads 'Under realistic collection friction a fully-funded capital envelope still leaves a fifth of emissions outside the cap', with no offset stated in the title; the qualifier still has to be added to F-31 (not done here; findings are outside this task). **Closed 2026-10-23 (TASK 70 B.4, `c6a8673`):** F-31's title and opening now carry the qualifier.

> **Places that treat B\* as the requirement, or as a denominator** (forward-facing; not edited): `chapters/chapter3_5_draft.md:417` ("fully-funded envelope (`adequacy`, 1.00 × B\*)") and the F-31 title/entry (`findings_frozen.md` F-31, "fully-funded capital envelope"); `chapters/chapter3_4_draft.md:~273` and `methodology.md:397,401` (0.40 × B\* as the EaaS saturation fraction; a ratio to B\*, valid under (b), to be re-expressed under (a)); `chapters/chapter3_3_draft.md:277,302` [removed `009d230`; held at `a993f15`; successor `CHAPTER_3_assembled.md` §3.3] and `docs/bstar_triangulation.md` (the B\* triangulation against NIRP); `chapters/chapter_4_skeleton.md` C-4.4.4 and C-4.5.5; register L45 and L46; every capital-case result (`tight`, `moderate`, `adequacy`, `expansion`) in F-10, F-12, F-13, F-31, F-35, F-37, F-38, F-39 and F-40, which are in USD terms and unaffected under (b).

> **Direction of bias.** Under (b) none on any USD result; the labels "adequacy" and "fully funded" overstate the requirement by about 5%.

> **Mitigation.** F-22 compares against the measured bracket, not against B\*.

> **What would resolve it.** The Chapter 3 decision above; under (a), a re-derivation and re-run of every capital-case result.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L84 — 32 sweep-aggregate result files are admitted as CURRENT on path-plus-log evidence, a weaker standard than arm-keys-in-artefact

> **Status:** LIVE (recorded 2026-10-18, TASK 48 Part A; admitted by Zeus's decision at the A.5 stop).

> **What.** Sweep-aggregate summaries (`rel1/2/3`, `gas1/2/3`, `fin2/3`, `dem2`, `pol1`, `carbon_price_sweep`, `parameter_uncertainty`, including their `_gate` and `_uncapped` twins) contain experiment results only: no arm key and no `arm_tag`. Their arm and epoch cannot be read from the file. Rule C2 in `scripts/build_results_inventory.py` admits a file as CURRENT when (i) the directory path identifies the arm, (ii) a Phase B or Phase C runner log whose header records a HEAD has an mtime within 15 minutes of the file's, and (iii) every arm token in the directory path (`nameplate`/`observed`/`annual`/`total`) appears in the log's name. The matched log and its HEAD are recorded per file (`epoch_log`, `epoch_head` in `docs/results_inventory.csv`) and carried into `docs/results_manifest.json`.

> **Why it is weaker.** Rule C1 (arm keys recorded in the artefact) is evidence the file carries. C2 is evidence about the file made elsewhere: a directory name and a log. A file overwritten after its log (same path, later mtime) fails (ii), which is the only protection. A file overwritten within 15 minutes of the log, or by a run that wrote no log, would pass.

> **Where it already caught something.** `results/rel2/gas_avail_nameplate/rel2_summary.json` (and `curve_summary.csv`, `cost_curve_binding.csv`) have mtime 2026-10-03 20:35:28; the only matching log, `logs/phase_b/15_A_rel2_gas_avail_nameplate.log`, ends 00:35:12, twenty hours earlier. The file was rewritten after its log by something that left no log. It stays UNCLASSIFIABLE and is not admitted.

> **Direction of bias.** Unknown for any one file. The risk is a stale or hand-rewritten file passing as current.

> **Mitigation.** Findings citing a C2 file carry the C2 tag in the manifest. The staleness checker (Part G) reports md5 change against the manifest, which does not depend on C2.

> **What would resolve it.** Have each sweep runner write its `arm_tag` and `om_case`/`tariff_case`/`solar_build_case` into its own summary JSON, then re-run; C2 then retires.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L85 — `om_case` was not in the hashed arm tag; Phase C Group 1 overwrote the `om_low` result with `om_high` at `02c17f55`

> **Status:** RESOLVED in the result tree (TASK 59 Part 5.1, 2026-10-22; was: FIX COMMITTED `5aad6ff`, 2026-10-18, TASK 48 B.1, re-solve not yet run). The fix is `5aad6ff`. The Group 4 three-solve re-run executed on 2026-10-18 at 20:57 (`logs/phase_c/4_A_om_central.log`, `4_B_om_low.log`, `4_C_om_high.log`, mtimes 20:57:27, 20:57:28, 20:57:30), after which `results/baseline_sliced/summary__02c17f55.json` holds `om_central` (27,317,670,155.72187) and `om_low` / `om_high` sit at their own hashes `9137a106` (26,909,457,846.59) and `b4106a3e` (27,884,317,679.69). The post-epoch generation also has four distinct files, `143725ce` (om_none), `792f8dce` (om_low), `6046f759` (om_central) and `f6926482` (om_high), with the recomputed `sha1(arm_tag)[:8]` equal to each filename (`docs/task_log_59.md` Part 2.2). The collision is gone in both generations. Detected in TASK 48 Part A, 8 days after the run.

> **What.** The runners that take `--om-case` (`01b_`, `02_`, `03_`) build their arm tag without it, by documented design ("recorded for auditability ... NOT appended to the output tag", commit `66cd29e`, 2026-09-07). The design assumed one O&M value per tag. Phase C Group 1 ran `om_low` and `om_high` at the central arm set, the hash is `sha1(tag)[:8]`, and both wrote `summary__02c17f55.json` (13:08:39, `om_high` last). `results/baseline_sliced/INDEX.md` line 1 still maps `02c17f55` to the O&M-free central tag. `run_phase_c.sh` was written on the opposite premise ("the om arm is part of the hashed arm tag") and said so; that premise was not checked against the runner.

> **Consequence.** The `om_low` result file does not exist (its objective, 26.909 bn, survives only in `logs/phase_c/1_A_om_low.log`). The `om_central` baseline is no longer at its own hash: `summary__02c17f55.json` holds `om_high` (27,884,317,679.69). Anything that reads that file as the central baseline is wrong by +0.567 bn against the 27.318 bn central. F-3's O&M-band paragraph has no valid current-epoch `om_low`.

> **Same class, not yet fixed.** `tariff_case` and `solar_build_case` are likewise absent from the tags; the `_uncapped` directories share hashes with the capped ones and differ only by directory name (16 hashes, TASK 48 A.3). No live overwrite exists for those today.

> **Fix.** `src/output_naming.py:om_tag_suffix()` appends the arm for any non-central value; `om_central` keeps its tag string and so every existing hash is unchanged. Applied in `01b_`, `02_`, `03_`. `26_` hard-codes `om_central` and is unchanged.

> **Detection.** Part A of TASK 48: the file's own recorded `om_case` disagreed with the Phase C log naming that hash. The recorded key is what made it detectable.

> **What would resolve it.** Run Group 4 (`bash scripts/run_phase_c.sh --only 4`; success check `python3 scripts/check_om_arms.py`), which writes `om_central`, `om_low`, `om_high` to three distinct hashes and restores the central at `02c17f55`.

> **Generation (TASK 86):** the hash-keyed result files this entry cites are blended dispatch (no `gas_vintage_dispatch=True` in the cited files' arm records). Checked: `02c17f55` (blended). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142. [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L86 — Script 26 recorded no `om_case` in its summaries; 7 f3 files are admitted on a verified log-to-commit chain, 10 stay out

> **Status:** FIX COMMITTED for new runs (TASK 48, 2026-10-18); the 17 existing `f3` summaries still lack the key. Admission rule C3 is a **weaker standard than arm-keys-in-artefact** (see L84).

> **What.** `scripts/26_run_f3_identification.py` has no `--om-case` flag and hard-coded `om_case="om_central"` inside its `load_scenario` call (line 253 at `3f0b382`) without writing it to `summary.json` or into the tag. No `f3` summary could therefore say which O&M arm produced it; the only record was a literal in a script that could change. The same fact (a required arm absent from the tag) is L85's class. Unlike the baseline summaries, the artefact offered nothing to detect a mismatch with.

> **Admission (rule C3, `scripts/build_results_inventory.py`).** An `f3` summary is CURRENT only if a Phase B/C runner log names its directory, was written within 15 minutes of it, records a HEAD **with a clean working tree**, and `git show <that HEAD>:scripts/26_run_f3_identification.py` contains the `om_case="om_central"` literal in the call and no `args.om_case`. The literal is checked at the log's commit, not at HEAD. The chain is stored per file (`verification` in `docs/results_inventory.csv` and `docs/results_manifest.json`). **Result: 7 admitted** (`results/f3_uncapped/*`, logs `3_26_f3_*`, HEAD `3f0b382`, clean, literal at line 253). **10 not admitted:** the 2 `f3_gate` files (log `0_26_f3_*`, HEAD `983a479`: the literal is present at that commit but the log header records `Working tree: DIRTY`, so the commit does not describe the code that ran) and the 8 `f3` files from 2026-10-03 (no runner log names them: `logs/phase_b` holds no `26_` item).

> **Fix.** One constant `OM_CASE` now feeds both the `load_scenario` call and `summary["om_case"]`. No result was re-run.

> **Consequence of the fix.** Script 26 is in `GATE_CODE_FILES` of `run_phase_c.sh`, so the stored fingerprint is stale and Group 3 refuses until Group 0 is re-run, and `compare_phase_c_gate.py` is a byte-identity test: the new key makes every `f3` `summary.json` differ from the stored ones. Group 0 will report the 26_ runners as not matching for that one reason. Either regenerate the stored `f3` set (the 11 solves in `_ndc_runners`' f3 block) or teach the comparator to ignore the added key; neither was done here.

> **What would resolve it.** Re-run the eight 2026-10-03 `f3` configurations and the two gate configurations with the fixed script; C3 then retires for `f3`.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L87 — Results are produced faster than they are recorded; the lag has been caught only by hand (TASK 47 C34, C14, C4). Detection: the staleness gate

> **Status:** LIVE, with a detection method in place (TASK 48 Part G, 2026-10-18). Not a model defect.

> **What.** `results/` is gitignored, runs are executed by hand in batches, and `findings_frozen.md` is edited later by a different party. Nothing connected an entry to the file version it was written against. Consequences found only by manual verification: C34 (F-21 quoted a pair whose decline arm had since been re-run), C14 (a comparator with no current-epoch run), C4 (F-30's scope condition described the superseded annual-builder measurement). Part A of TASK 48 found a further instance of the same shape: the Phase C driver was written on a premise about the hash that was never checked against the runner (L85).

> **Detection method.** `scripts/build_results_inventory.py` classifies every summary/diagnostics file CURRENT / LEGACY / UNCLASSIFIABLE with a named rule; `scripts/build_results_manifest.py` writes `docs/results_manifest.json` (path, md5, arm, mtime, HEAD where a log records it, and the evidence standard that admitted each file); `scripts/check_findings_freshness.py` reads each findings entry's `Source.` line and reports (A) cited files absent from the manifest, (B) cited files newer than the entry's last dated block, (C) entries with no parseable `Source.` line. Exit status 1 on any hit. Run it after every batch of solves and before any findings edit.

> **Limits of the gate.** (i) B compares mtime to the entry's last *date string*, not md5 to a recorded md5: no entry records the md5 it was written against, so a file rewritten on the day of the last block, or an entry whose last block merely says it was re-derived, passes. Entries should record the md5 of each source from now on; the checker can then compare it exactly. (ii) It tests that a source is current and cited, not that the quoted number matches it. (iii) The manifest is only as good as the inventory's epoch rules (L84, L86 record their weaker standards). (iv) `results/` is not in git: the manifest is the only anchor and it is tied to a repository tag only if one is created by hand after the manifest is committed.

> **Mitigation.** Regenerate inventory, manifest and ranking after every batch; a findings entry is not rewritten without a clean run of the checker's A/C classes for that entry.

> **What would resolve it.** Findings entries carry per-source md5 (so B becomes exact); sweep runners write their own arm keys into their summaries (retiring C2 and C3); and a tag is cut at each manifest commit.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L88 — The first capital instalment is charged in the build year against an ordinary-annuity CRF; PV/I is 1 + r_s at r_f = r_s, not 1

> **Status:** RESOLVED by decision and change (TASK 53 2.1) (was: LIVE. Found by TASK 50 V1 (2026-10-18). Measurement arm `capital_window_case` built (TASK 51 M1); the six solves are not yet run. Nothing corrected).

> **What.** `optimize_model_sliced.py`, f1_tranche branch: `_df_window_conc` / `_df_window_comm` sum `m.DF[t]` over `range(v, min(v + L_eff, T_last + 1))` (first instalment in the build year, DF[2025] = 1) while `_crf` is the ordinary-annuity form r(1+r)^L / ((1+r)^L - 1). For the 2025 solar vintage CRF x W is 0.468236 (concessional, IDA21) and 1.120455 (commercial); an ordinary annuity with the first instalment one year after the build would give 0.4336 and 1.0375. At r_f = r_s the product is 1.08, not 1. §3.1.2's "equals one when r_f = r_s", its "about 0.42" and its "about 4%" describe the other convention. The factor applies to C^cap, the conversion charge (vintage 0, same window) and the rebuild charges; fuel, O&M, self-generation and carbon are unscaled.

> **Two internally consistent corrections** (TASK 51): *arrears* (window one year later; removes the factor (1 + r_s) = 8% from all capital uniformly, does not move the concessional-to-commercial ratio) and *annuity-due* (CRF / (1 + r_f); removes about 0.12% on concessional and 8.8% on commercial, so it moves the ratio). The code matches neither.

> **Direction of bias.** Capital is over-charged relative to fuel, O&M and every non-capital term, so the objective is biased toward capital-light, fuel-heavy plans; under the annuity-due reading the over-charge is also larger on commercial than on concessional capital, biasing against commercial finance. Size not yet measured.

> **Why the regression anchor did not catch it.** The anchor runs `pre_f1_overnight` and never reaches the f1 path.

> **What would resolve it.** Zeus decides the convention after the TASK 51 M1 cells; one structural-fix epoch; one re-run. See also L89 and L93.

> **TASK 53 update.** `capital_window_case` defaults to `arrears` in `load_scenario`; `advance` and `due_crf` remain arms. A2 passes at the default (`A2-default`: max deviation 0) and `advance` is characterised as carrying exactly the (1 + r_s) factor (0.080000000); G3-2 passes in all three arms (the rebuild charge uses the same convention). The annual builder was taught all three arms. Central objective, harness, 2025-2054: 27,317,670,155.72 (advance, flat storage) -> 26,332,543,636.30 (arrears) -> 24,986,620,949.18 (arrears + declining storage, 2.2); non-quotable until reproduced by `run_phase_b.sh` item 0. M1's six cells now measure the consequence of the decision on the corrected model, after the re-run.

> **Generation (TASK 86):** not established. This entry quotes blended-generation or pre-epoch figures (24,986,620,949, 27,317,670) without citing a hash-keyed result file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L89 — Concessional recovery windows are truncated from the 2031 solar vintage and the 2041 storage vintage, inside the reporting window; the tenor result does not account for it

> **Status:** LIVE. Found by TASK 50 V2/V3. The §3.4.2 statement is made; the untruncated-window solve and the Chapter 4 statement are not. *(Prior status, retained as history: LIVE. Found by TASK 50 V2/V3.)*

> **Corrected 2026-10-23 (TASK 70 B.3).** `docs/chapters/CHAPTER_3_assembled.md:1782-1788` ('What the truncation in (3.34) does'). The quantitative content and the direction of bias are unchanged. The commercial start year is stated there as 2045; this entry says 2046 (v + 10 exceeds 30 from the vintage after 2045 under its own arithmetic) and the difference is not resolved here.

> **What.** L_eff = min(tranche tenor, technical life): concessional solar and gas 25 yr, storage 15 yr (`asset_lifetimes()` storage = 15, not 30), commercial 10 yr. A window is truncated when v + L_eff exceeds the 30-year horizon: concessional solar from 2031, concessional storage from 2041, commercial from 2046. Concessional solar instalments: 2030 25, 2035 20, 2040 15, 2045 10, 2050 5. ~~§3.4.2's "truncation does not touch reported vintages" is true of the commercial tranche only.~~ **§3.4.2 now says so itself: concessional solar windows truncate from vintage 2031, concessional storage from 2041, commercial from 2045 (`CHAPTER_3_assembled.md:1782-1788`).** In the central solve the $750M concessional volume is drawn entirely in 2032 (23 of 25 instalments charged), so the effect on the central anchor is small; it grows with the envelope and with later draw.

> **Not a confound of F-4.** The 50.1% tenor share is closed-form on CRF levels (TASK 47 C20), not read from solves. It is a confound of any claim that the *solved* concessional advantage reflects tenor economics alone.

> **Direction of bias.** Late concessional vintages are cheaper because the model stops charging them at 2054; this favours concessional finance. Not quantified.

> **What would resolve it.** A solve with untruncated windows, ~~or an explicit statement in §3.4.2 and Chapter 4~~ **or the statement in Chapter 4 (the §3.4.2 statement is made)**.

## L90 — Storage capital cost is held flat at the 2025 `Storage_low` value although the source series falls by about 42% to 2045

> **Status:** RESOLVED: the central case has moved (TASK 53 2.2) (was: LIVE. Found by TASK 50 V8).

> **What.** `econ["STORAGE_COST_PER_MWH"]` (`src/io.py`) is the `Storage_low` 2025 row, 252,000 USD/MWh, used as a scalar at every vintage and in every rebuild. `data/cost/processed/storage_capex.csv` carries a declining Storage_low series (252 -> 199 -> 181 -> 163 -> 146 USD/kWh at 2025/2030/2035/2040/2045) that the model never reads. Solar capex is a per-year series (1,456,000 -> 563,000 USD/MW, -61%, held flat after 2050). §3.7.3 says both decline; only solar does.

> **Direction of bias.** Storage is over-priced for every post-2025 vintage, so storage-heavy results (63,056 MWh by 2054 at the anchor) are conservative. Not quantified.

> **What would resolve it.** Pass a storage series as solar's is, or declare the omission in §3.7.3 and §3.3.3.

> **TASK 52 section 2.1 addendum.** No commit message, task log or register entry records a decision to hold storage flat. The documents say the opposite of what the code does: `docs/model_capability_inventory.md` (Technology learning) states that solar and storage capex "are read as BY-YEAR series (`load_solar_capex_by_year`/`load_storage_capex_by_year`)"; `docs/forensic_audit.md` calls `load_storage_capex_by_year` "confirmed live and heavily used"; `docs/merged_correction_plan_v2.md` says "storage flat-holds from 2045". The function has no caller in `src/` or `scripts/` today. It was wired in once: commit `72cad20` (2026-03-21) built `m.storage_capex_param` from it inside `build_model` (with a `storage_capex_scenario` scenario key, default `Storage_low`), and commit `96b50cf` (2026-04-03, message "large refactor with results") removed that block while rewriting the builder around the `solar_capex_by_year` argument, leaving storage on the scalar. Neither the commit message nor any document records dropping storage; the same diff comments the solar fallback to a fixed 2025 scalar as "conservative". The "flat is conservative" argument in `merged_correction_plan_v2.md` and `methodology_demand_horizon.md` concerns the buffer years past the source's coverage, not 2025-2045. The arm `storage_capex_case` (`flat` default, `declining_nrel`) now exists; `bash scripts/run_storage_capex.sh` measures the bound.

> **TASK 53 update.** `storage_capex_case` defaults to `declining_nrel` on every runner; `flat` is the declared sensitivity (`bash scripts/run_storage_capex.sh --with-flat`, reported in section 3.7). The record shows this was never a decision (`72cad20` wired the series, `96b50cf` removed it in a refactor; no commit message mentions storage), and the supervisor's second reason (rejecting a US learning trajectory for storage while solar is also NREL-derived and read) was an asymmetry with nothing behind it. The annual builder reads the series as well (F1b). At the central configuration the 2054 storage stock is unchanged to five digits (firm capacity sets it); the regression anchor, which has no f1 chain, moved in quantity (43,539.98 -> 74,435.22 MWh).

> **Generation (TASK 86):** not established. This entry quotes blended-generation or pre-epoch figures (63,056) without citing a hash-keyed result file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L91 — The planning reserve margin is a fixed 900 MW spinning-reserve requirement divided by peak demand; it falls from 19.5% to 6.4% across the horizon

> **Status:** LIVE. Found by TASK 50 V11. Extends L9 (which records only that the margin is hard-coded). Chapter 3 §3.7.4 now declares the margin. *(Prior status, retained as history: LIVE. Found by TASK 50 V11. Extends L9 (which records only that the margin is hard-coded).)*

> **Corrected 2026-10-23 (TASK 69 A.2).** `docs/chapters/CHAPTER_3_assembled.md:3244-3251` (audit pass 3 A.2). The quantitative content of the entry, the direction of bias and the missing fixed-fraction arm are unchanged.

> **What.** `RESERVE_MARGIN_SPINNING_MW = 900.0` (NIRP 2024 section 2.3, p.19), `reserve_margin = 900 / peak_mw[t]` (`scenarios.py`). Peak demand triples (4,605 -> 13,967 MW), so rho_t falls 19.54% (2025), 13.33% (2035), 9.09% (2045), 6.44% (2054). A spinning reserve is an operating quantity; the adequacy constraint uses it as a planning margin. The margin then multiplies the block-average peak, which is scaled by omega = 1.3448 through a `scenario.get` default (L50), so the margin on the instantaneous peak is different again. ~~§3.7.4 names three weakly identified adequacy inputs and not this one.~~ **§3.7.4 now declares it (`CHAPTER_3_assembled.md:3244-3251`, 'The reserve margin is an operating quantity used as a planning standard', with the 19.5 to 6.4 per cent fall).**

> **Direction of bias.** Late-horizon firm capacity and storage requirements are lower than a fixed-fraction margin would give; not quantified. `01b` hard-codes the arm, so the baseline cannot vary it (L9).

> **What would resolve it.** A fixed-fraction margin arm, or a stated justification.

## L92 — The VSS legs (scripts 30 and 17) run at `capital_case="moderate"`; the central anchor runs at `"unconstrained"`; "47% more" mixes configurations

> **Status:** DISSOLVED post-epoch for the central plan (the comparator problem no longer exists); retained as history. Open only: whether `moderate` binds in individual recourse draws, which is unmeasured. *(Prior status, retained as history: LIVE. Found by TASK 50 V13; legs checked against each other by TASK 51 M2.)*

> **Corrected 2026-10-23 (TASK 69 A.2).** F-42 TASK 55 block (`docs/findings_frozen.md`) and `results/vss_bound/vss_bound__d6f19a23.json`: the frozen first-stage plan equals `summary__6046f759.json` (solar 39,030.1 MW, storage 63,056.3 MWh), no divergent year, deterministic base anchor + 6.75 USD. Recomputed from `results/stochastic_redispatch/draws__8fc2e490.jsonl` (100 of 100 feasible): mean 37,846,482,445.52, median 27,501,838,741.14, P10 22.17 bn, P90 72.97 bn (audit pass 3 C.3). By construction the budget should not bind in recourse (the solar and storage plan is frozen), but that is inference, not a stored result; F-42 says so.

> **What.** `scripts/30_run_stochastic_redispatch.py:628-637` and the matched WS case in `scripts/17_run_monte_carlo.py` call `load_scenario(capital_case="moderate")`; `scripts/01b_run_baseline_sliced.py:294` uses `"unconstrained"`. The only scenario difference is `public_solar_budget_npv` = 8,732,342,068 USD. ~~The ECP leg's own base cost is 27,426,913,099 USD against the anchor's 27,317,670,156 (+109.2 M, +0.40%)~~ **post-epoch the legs' deterministic base is 24,986,620,955.93 USD against the anchor's 24,986,620,949.18 (+6.75 USD)**. ~~40.203 bn / 27.318 bn - 1 = 47.2%~~ **post-epoch: 37.846 bn / 24.987 bn - 1 = 51.47% (mean) and 27.502 bn / 24.987 bn - 1 = 10.07% (median)**; against the legs' own base it is 46.6%; ~~the ECP median is 29.85 bn (+8.8% over base) against a mean of 40.20 bn~~ **post-epoch the ECP median is 27.502 bn against a mean of 37.846 bn (mean/median 1.38)**, so the figure is carried by the downside tail. TASK 51 M2 found the two legs agree with each other on every standard arm and on `case_fields`; the bound ECP - WS is on one basis. The frozen first-stage plan is the moderate plan, not the anchor's.

> **Direction of bias.** None on the bound itself. ~~The "committed plan costs 47% more" comparison is against the wrong baseline by 0.6 percentage points and describes a tail-driven mean.~~ **Post-epoch the legs' base equals the anchor, so the comparison is on one basis (+51.47% mean, +10.07% median against both); the tail-driven reading stands (mean/median 1.38).**

> **What would resolve it.** State the comparator in section 3.6.8, report the median, ~~or re-run both legs at `unconstrained` (about 200 LPs, Zeus).~~ **Not needed for the central plan: the frozen first-stage plan is the unconstrained anchor.**

> **Generation (TASK 86):** the hash-keyed result files this entry cites are blended dispatch (no `gas_vintage_dispatch=True` in the cited files' arm records). Checked: `6046f759` (blended); `8fc2e490` (blended). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142. [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L93 — The five VSS guards compare `case_fields` but not the legs' standard arms

> **Status:** LIVE. Found by TASK 51 M2.

> **What.** `scripts/31_compute_vss_bound.py`'s guards are `same_draws_sha256`, `fix_gas_free`, `gas_deliverability_baseline`, `ws_case_scenario_matches_stage1` and `zero_infeasible_both_legs`. Guard 4 compares `capital_case`, `financing_regime`, `required_margin` and `has_emissions_cap` only. No guard compares `dscr_case`, `capital_recovery_case`, `om_case` or any other standard arm. In the stored legs the arms agree (the ECP leg ran at `2b22fe9`, before the `required_margin` override was removed at `4ec43b4`; the WS leg at `cded909`, after; the matched case never carried the override, so they agree), but the agreement rests on the two runners being driven with the same flags, not on a check.

> **Direction of bias.** None today; a future divergence would go undetected.

> **What would resolve it.** Compare `standard_arms` in guard 4, or have both legs write the same arm record and compare it.

## L94 — The plan to 2045 depends on the horizon end: a 2054 horizon and a 2050 horizon give different 2045 plans

> **Status:** LIVE. Two records, on different generations and different quantities. **Blended model, reduced instance (TASK 53): not reproduced** (the plan to 2045 no longer changed with the horizon end after the declining storage capex). **Vintage model: reproduced in a different form** (TASK 78 D.4, L135): cumulative stocks at 2045 are invariant across horizon ends 2045, 2050 and 2054 to 0.1 MW, while annual addition timing moves 1.5 to 1.9 per cent in L1. This is why invariants G1-3 and G1-3m remain failing by decision and G1-3s passes; no assertion was changed. See L135. *(Prior status, retained as history: NOT REPRODUCED on the corrected model, reduced instance only (TASK 53); before that LIVE. Found by TASK 52 G1-3 (reduced-instance analogue; real-scale test not constructible: gas deliverability data stop at 2054).)*

> **What.** Central arms, horizon 2045 versus 2050: identical plan to 2045. Horizon 2045 versus 2054: storage differs by 26.3% of cumulative to 2045, solar 2.6%, gas by up to 1,136 MW in a year, last differing year 2045. The 2054 solve builds 373.6 MWh of storage in 2037 where the shorter horizons build 4,919.6. TASK 52 A-1 tested the cause and CONFIRMED it: the plan to 2045 is identical at horizons 2050 and 2051 and first diverges at 2052, when the 2037 vintage's rebuild (v + 15) falls inside the horizon; with storage life set to 30 (no rebuild inside) the 2054-horizon plan to 2045 equals the 2045-horizon plan exactly. §3.1.1 presents 2046-2054 as an end-effect buffer; the buffer is not neutral.

> **Direction of bias.** Unknown; plans to 2045 are horizon-conditional.

> **What would resolve it.** A real-scale pair of solves that isolates the rebuild (Zeus), or a decision on how in-horizon rebuilds are charged.

> **TASK 53 update.** After 2.2 (declining storage capex) the reduced-instance `G1-3` passes: the plan to 2045 no longer changes with the horizon end (2050 or 2054), and the mechanism test `G1-3m` is UNDETERMINED (no divergence to attribute). This is a 12-year, 8-year-life fixture, the same on which the divergence was found, so the comparison is like for like, but whether the real plan's dependence on the horizon end disappears is for B-3's eight cells on the corrected model (Zeus, after the re-run). The buffer claim in section 3.1.1 is NOT restored by this.

## L95 — Hydro carries neither a capital cost nor an operation-and-maintenance cost

> **Status:** LIVE. Found by TASK 52 G2-3. Chapter 3 §3.7.3 now declares the omission and cites this entry; a default-off flag exists for the term with a null rate (TASK 69 B.1). The omission itself stands: no non-zero rate is set. *(Prior status, retained as history: LIVE. Found by TASK 52 G2-3.)*

> **Corrected 2026-10-23 (TASK 70 B.2).** `docs/chapters/CHAPTER_3_assembled.md:3160-3164` ('Hydro carries no operating cost') and `:1448-1453` (§3.3.5, 'an upper bound (Appendix B, L95)'). The flag: `compute_om_cost_npv(hydro_capacity_mw=..., hydro_fom_usd_per_kw_yr=...)` in `src/io.py`, both builders, `scenario["hydro_om_term"]` (default off) and `scenario["hydro_fom_usd_per_kw_yr"]` (default 0.0), runner flags `--hydro-om-term` and `--hydro-fom-usd-per-kw-yr` (commit `df4594f`, `docs/task_log_69.md` B.1). Off is byte-identical to `summary__6046f759.json`; on at the null rate 0.0 gives the same objective, 24,986,620,949.18249 (`results/deferred_flag_sensitivity/summary__b9ff3478.json`). A rate would need a row in `docs/parameter_provenance.md` and invariant G2-3's `in_om` set extended.

> **What.** `compute_om_cost_npv` has solar, storage and gas arguments only ~~.~~ **by default, and a hydro term only when the flag is on.** Hydro capacity (2,678 MW under `hydro_flat`) enters firm capacity and the energy balance and carries no fixed or variable O&M, and no capital cost (TASK 48 Part C). ~~No document declares this for O&M.~~ **Chapter 3 §3.7.3 declares it for O&M and cites L95.**

> **Direction of bias.** Hydro is costless to the objective, so every hydro-contingent result is favourable to hydro; F-18 and the hydro lever are upper bounds on hydro's benefit.

> **What would resolve it.** ~~A hydro FOM term from the ATB or a stated declaration.~~ **The declaration is done; the term is built and off. What remains is a sourced rate (ATB hydropower classes are US new-build, not the legacy Kainji/Jebba/Shiroro fleet; any rate is a transfer, as in L22).**

> **Generation (TASK 86):** the hash-keyed result files this entry cites are blended dispatch (no `gas_vintage_dispatch=True` in the cited files' arm records). Checked: `6046f759` (blended); `b9ff3478` (blended). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142. [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L96 — `selfgen_busbar_factor` is read only through a builder `scenario.get` default (extends L50)

> **Status:** LIVE. Found by TASK 52 G7-1.

> **What.** Both builders read `scenario.get("selfgen_busbar_factor", 1.0)`; `load_scenario` never sets it. The default is declared in a code comment ("a declared conservative simplification"), not in the register or any arm. It multiplies self-generation in the electricity balance.

> **Direction of bias.** 1.0 under-credits self-generation by the 7% transmission loss factor it avoids; this biases against the self-generation backstop. Not quantified.

## L97 — Scenario keys that no builder or script reads, and a key live in only one builder

> **Status:** LIVE. Found by TASK 52 G7-3.

> **What.** `hydro_baseline_twh`, `hydro_capacity_mw`, `hydro_growth` and `storage_baseline_mwh` (four keys; `blend_case`, flagged by the static read, is live) are set by `load_scenario` and read by neither builder nor any other script or module. `storage_deployable_hours_per_year` (700 h) is read by the annual builder and not by the sliced one. A reader (or a sensitivity run) that varies them sees no effect.

> **Direction of bias.** None on results; a trap for sensitivity analysis. The static read does not resolve dynamic key access.

## L98 — Most runners write fixed-name outputs; no keyed runner records `us_inflation`

> **Status:** RESOLVED for arms (TASK 53 2.5) (was: LIVE. Found by TASK 52 G7-4. Related to L85 and L87).

> **What.** 28 runners take arm flags. 22 write outputs with no per-arm key (a re-run at a different arm overwrites it). The 6 keyed runners (`01b`, `02_run_ndc_caps`, `03`, `26`, `30`, `51`) omit `us_inflation` except `51`; `01b`/`02`/`03` also omit `tariff_case`. `us_inflation` changes every capital charge.

> **Direction of bias.** None on stored results (each was run at one central configuration); a silent-overwrite risk for any varied re-run.

> **TASK 53 update.** Every runner's output location now carries a hash of the full arm set (`complete_arm_tag` for runners that key their own files, `resolve_results_dir` -> `arms_<hash8>/` for those that wrote fixed names; runner 29 keyed by arms, `n_base` and `seed`); `us_inflation` is in every key. G7-4 passes statically and by a dynamic probe. Residual: flags that are not arms (L110).

## L99 — Summary quantities carry no year label, and no 2045 value sits beside the 2054 anchors

> **Status:** RESOLVED for the baseline family (TASK 53 2.4) (was: LIVE. Found by TASK 52 G1-2; TASK 50 V12).

> **What.** `final_storage_capacity_mwh` (63,056.31, the 2054 value), `solar_total_built_mw`, `gas_total_built_mw`, `npv_total_cost_usd`, `actual_emissions_tco2_total` and `cumulative_unserved_twh` are horizon aggregates or last-year snapshots including the 2046-2054 buffer, with no year or window in the key. Only the two capex keys name their window.

> **What would resolve it.** Write a 2045 companion for every 2054 anchor.

> **TASK 53 update.** `01b`, `01_run_baseline`, `02_run_eaas`, `51`, `54` and `56` write `solar_cumulative_built_{2054,2045}_mw`, `gas_cumulative_built_{2054,2045}_mw`, `storage_capacity_{2054,2045}_mwh`, `cumulative_unserved_2025_{2054,2045}_twh`, `cumulative_emissions_2025_{2054,2045}_tco2`; the old unlabelled names are removed. `npv_total_cost_usd` keeps its name and gains `npv_total_cost_window`; it has no 2045 counterpart (one optimisation). Residual: L111.

> **Generation (TASK 86):** not established. This entry quotes blended-generation or pre-epoch figures (63,056) without citing a hash-keyed result file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L100 — The derived block PAR differs from the peak period factor by 3e-5, not "exactly" as section 3.2.5 states

> **Status:** RESOLVED by wording: Chapter 3 §3.2.5 now states the rounding residual. Renormalisation of the factor sets is not done (immaterial, 0.003%). *(Prior status, retained as history: LIVE (immaterial in size). Found by TASK 52 D3.)*

> **Corrected 2026-10-23 (TASK 69 A.2).** `docs/chapters/CHAPTER_3_assembled.md:990-995`: 'the season factors cancel in (3.27) and PAR_block reduces to the peak period factor f_peak to four decimal places — 1.087668 against a declared 1.0877, the residual arising from rounding in the published factor sets' (audit pass 3 A.2).

> **What.** `par_block` = 1.087668 against `PEAK_UNSUPPRESSED_PERIOD_FACTOR["peak"]` = 1.0877. The factor sets are rounded to four decimals; their hours-weighted means are 0.999917 (season) and 1.000029 (period), so the season factors do not cancel exactly. Slice weights and hours are exact.

> **Direction of bias.** peak_mw is 0.003% below what an exact normalisation would give. Negligible.

> **What would resolve it.** ~~State "to four decimals" in section 3.2.5, or renormalise the factors (moves every peak_mw).~~ **Wording done (`CHAPTER_3_assembled.md:990-995`); renormalisation not done and not needed.**

## L101 — The gas shadow price is a present-value (year-0) quantity labelled USD per TWh_th, and is compared with year-t benchmarks

> **Status:** RESOLVED (TASK 53 2.3) (was: LIVE. Found by TASK 52 G8-3. Bears on F-30 and TASK 47 C4).

> **What.** `extract_gas_shadow_prices_usd_per_twh_th` returns `-dual` of `gas_balance[t]`, which a finite difference confirms equals -dObjective/dA_t: the marginal cost in year-0 dollars, because the objective discounts every term by DF[t]. The year-t value is the reported value divided by DF[t] (DF[2054] = 0.107). `06_run_gas1_shadow_benchmarks.py` compares the reported series with `BENCHMARKS` (USD per TWh_th) directly, and `01b` writes `abs(dual)` under `gas_shadow_price_usd_per_twh_th_by_year`.

> **Direction of bias.** Late-year shadow prices are understated against year-t benchmarks, so the benchmark-exceedance counts (C4: 7 to 16 of 30) are lower bounds. Binding-year counts and the zero-shadow results are unaffected.

> **What would resolve it.** Relabel as present value, or divide by DF[t] in the extractor and recompute the exceedance table from the stored series (no re-solve).

> **TASK 53 update.** `gas_shadow_pv_usd_per_twh_th_by_year` (the solver's number, year-0 USD) and `gas_shadow_yr_usd_per_twh_th_by_year` (divided by `m.DF[t]`) replace the basis-less key; GAS-1's benchmark exceedance, the gas3 paradox test and the plot use the year-t series. G8-3 passes. The stored pre-epoch tables (F-30, TASK 47 C4) compared PV values with year-t benchmarks and are replaced at the findings re-derivation, which recomputes the exceedance table from the re-run.

## L102 — The storage seasonal balance has no state of charge: discharge need not follow charge within a season

> **Status:** LIVE. Found by TASK 52 D4 (elevated at Stage B acceptance).

> **What.** `storage_season_balance[t, season]`: sum of discharge in the season <= round-trip efficiency x sum of charge in the season. Nothing orders charging before discharging within a season or tracks a stored quantity, so the model may discharge in a slice that precedes the charging slice. Section 3.1.8's wording that charge and discharge are tied to distinct slices of the same season implies an ordering the formulation does not enforce. The annual builder bounds storage differently (power x 700 deployable hours).

> **Direction of bias.** Optimistic about storage's energy value. Not quantified.

> **What would resolve it.** A state-of-charge variable and an ordering of slices, or a stated declaration.

## L103 — Two reported diagnostics do not measure what the model enforces: reliability ignores self-generation, peak adequacy uses a different peak and capacity

> **Status:** reliability half RESOLVED (TASK 53 2.6); peak-adequacy half MERGED INTO L31 on 2026-10-23 (TASK 69 A.3), where the earlier number and the Appendix B statement now live (was: peak-adequacy half OPEN; before that LIVE. Found by TASK 52 G8-1b (addendum B-1)).

> **What.** `compute_reliability_metrics` returns 1 - unserved/demand. In the sliced builder unserved energy is fixed at zero whenever self-generation is priced, so `annual_reliability_by_year`, `worst_year_reliability` and `horizon_reliability` equal 1.0 in every priced-genset result (GAS-1/2/3, FIN-3 and POL-1 write `horizon_reliability` from it); the grid-service shortfall the model constrains is selfgen + unserved (the REL runners compute their own). On a scarce-gas instance self-generation was 0.032% of demand and the reported reliability was 1.000000. `peak_adequacy_by_year` uses `peak_demand_multiple` (2.5 x average) and counts gas + storage power only, against the model's firm-capacity test (peak_mw x omega, credited capacity including hydro and credits): in the test, reported peak 16,124 MW against 9,434 MW and reported firm 12,476 MW against 11,348 MW.

> **Direction of bias.** The reliability columns overstate reliability; the peak-adequacy margin is not the model's. Neither enters the LP.

> **What would resolve it.** Count self-generation in the reliability metrics or relabel them "unserved-only" (done, TASK 53), and drop or realign the peak-adequacy diagnostic (now tracked in L31).

> **TASK 53 update.** The four reliability fields are renamed `unserved_only_*` (no alias) and the canonical `grid_service_reliability_*` (unserved + self-generation, the REL definition) sits beside them; they are not made to agree. Recovered post hoc from the 81 stored diagnostics files that carry the series: the stored `horizon_reliability` is 1.0 in all 81, the recovered canonical value is below 1 in 70 (minimum 0.6658). The peak-adequacy diagnostic is unchanged (invariant `G8-1d`, FAIL); it is quoted nowhere, and Appendix B should state that the field is a defect and not a result.

## L104 — Mechanisms with no regression signature, and a stale test file

> **Status:** stale-test-file clause RESOLVED (TASK 53 2.11); coverage clauses unchanged (was: LIVE. Found by TASK 52 F2 (guard coverage matrix)).

> **What.** Perturbing a parameter owned by each mechanism in the two existing signatures (the regression anchor `19c81f7c`, which runs `pre_f1_overnight`; the central anchor `02c17f55`) shows what each would catch. Neither catches: the bankability constraint, the DSCR cushion, the subsidy limit and the whole EaaS regime (both anchors are traditional regime); the public capital budget (absent from both); the emissions cap and carbon price (absent); the grid-service obligation (absent); self-generation price and emission factor (self-generation is unused at the anchors); the land constraint (slack at 16% of the allowance). The regression anchor also misses the entire f1 capital chain, the concessional envelope, conversion and the window convention. The central anchor detects the chain, the envelope, rebuilds, conversion, O&M, deliverability, firm capacity, demand and slice weights, storage and hydro. The Phase C byte-identity gate covers more configurations but detects change, not error. `tests/test_run.py` is a script, not a test, and no longer runs (it calls `load_scenario` without required arms). The reported diagnostics (`extract_planning_diagnostics`) are exercised by no signature. A mechanism with no guard is a blind spot whether or not a defect sits in it.

> **What would resolve it.** The proposed second signature S3 (tight capital, EaaS regime) covers the budget, bankability, envelope, chain, rebuild, O&M and conversion; it still leaves DSCR (objective-flat), self-generation, caps and land uncovered.

> **TASK 53 update.** `tests/test_run.py` was removed (it failed at import: `generate_epsilon_pareto` no longer exists) and `tests/README.md` points to `scripts/52_invariants.py`. S3 has its runner (`scripts/56_run_capital_regime_cell.py`, `run_signature_s3.sh`); its adoption waits for the re-run. The unsignatured mechanisms (emissions cap, carbon price, grid-service obligation, self-generation, land) and DSCR (objective-flat) remain uncovered.

## L105 — The salvage sensitivity cannot isolate salvage, has no results, and horizon truncation is not a salvage value

> **Status:** LIVE for the script (no results, the `no_salvage` confound); STALE for the §3.4.2 claim: Chapter 3 now states the truncation correctly. *(Prior status, retained as history: LIVE. Found by TASK 52 addendum B-2 (read, not run).)*

> **Corrected 2026-10-23 (TASK 69 A.2).** `docs/chapters/CHAPTER_3_assembled.md:1782-1800`: truncation windows from the 2031 solar and 2041 storage vintages, and 'the plan to 2045 is identical in all eight cells (§3.1.1), while cost is not', the 15- against 30-year storage cost difference of 91,477,408 USD reproducing the reported replacement charge; eight cells in `results/horizon_rebuild/` (audit pass 3 A.2; pass 2 item 0). The script, its 44-byte placeholder result and the salvage-versus-replacement confound are unchanged.

> **What.** `scripts/22_salvage_sensitivity.py` varies asset lifetimes over six arms on both builders, under `pre_f1_overnight` only (the script says salvage is structurally absent under `f1_tranche`). Its `no_salvage` arm sets every life to 1 year; under `pre_f1_overnight` that also rebuilds every asset every year (`_rebuild_years` yields every later year, each rebuild at a salvage fraction of 0), so the arm switches off salvage and switches on a replacement charge at once. No stored result exists: `results/salvage_sensitivity/salvage_sensitivity.json` is a 44-byte placeholder (2026-08-23, `models: {}`), the script is not in `run_phase_b.sh`, and its predicates predate Phase B. Under `f1_tranche`, no arm or scenario key credits residual value at the horizon (`salvage_npv` is 0). TASK 52 A-1 shows horizon truncation does not act as a salvage value: it charges vintages whose rebuild year falls inside the horizon and leaves others untouched. ~~Section 3.4.2 claims the opposite.~~ **§3.4.2 now says truncation holds 'in cost and not in the plan' (`CHAPTER_3_assembled.md:1790-1800`), using the eight horizon cells.**

> **Direction of bias.** Unknown; no measured bound exists.

> **What would resolve it.** An arm that credits a stated residual at the horizon under `f1_tranche` (including rebuilt vintages), or a declaration in section 3.7.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L106 — Dual-derived shadow prices other than the gas series (L101) are also present-value, and their keys state no basis

> **Status:** RESOLVED for the four quantities that feed results; the rest DECLARED (TASK 53 2.3) (was: LIVE. Found by TASK 52 E4 (the third dimensional audit class); extends L101).

> **What.** The objective is entirely present-value (every cost term carries `m.DF[t]`), so the dual of any constraint is a year-0 USD marginal whatever units the constraint is written in. E4 re-solved the reduced instance with a perturbed right-hand side and compared the reported number with the central difference of the objective. Carbon shadow, annual cap (`extract_carbon_shadow_prices_usd_per_tco2`): t=4 reported 94.668 against 94.665, t=9 reported 149.186 against 149.186 USD/tCO2, with DF 0.735 and 0.500; PV. Carbon shadow, single cap: one value (88.007) written to every year; it equals the derivative of the objective with respect to the cap, so it is PV and also not a per-year price. Land shadow (`land_shadow_price_by_year`): the reported values sum to the derivative with respect to the allowance (2.519e7 against 2.519e7), PV. Concessional envelope (`f1_envelope_shadow_usd_per_usd`, `envelope_dual`): PV objective per UNDISCOUNTED USD of commitment, a mixed basis with no single divisor (the divisor is DF of the vintage that draws the marginal dollar). EaaS bankability and subsidy-limit duals and reduced costs (diagnostic scripts only): PV by construction, not measured (flat optimal face, F-12). Exempt, because they state their basis or are PV/PV: the public budget shadow (1.95944 against 1.95943, PV per PV) and the grid-service obligation dual (`dual_pv_*` and `dual_yr_*`, TASK 02; the year basis is PV / DF[t]).

> **Exposure.** F-21 quotes carbon shadows in USD/tCO2 (115.05 to 113.45; `tight`/`no_friction` 212.11). Those are binding-year means of PV values. Percentage changes between cells that share the same binding years are unaffected by the divisor to first order; the levels are not year-t USD, and a year-t level is each binding year's value divided by DF[t] (DF[2054] = 0.107). Nothing found compares a carbon shadow with a carbon price, so no stated conclusion is known to depend on it; the carbon-price sweep reports prices, not shadows.

> **Direction of bias.** Reported levels understate the year-t value by DF[t], increasingly late in the horizon.

> **What would resolve it.** Divide by DF[t] in each extractor and name the basis in the key, or rename the keys `*_pv_*` as the REL series already is.

> **TASK 53 update.** Option 2 was applied to the gas shadow and the annual-cap carbon shadow (`_pv_` and `_yr_`), `eaas_private_investment` and `public_budget_realised_spend` (`_pv_` and `_undiscounted_`); the single-cap carbon dual is `carbon_shadow_capwide_pv_usd_per_tco2` and is no longer written to every year; the concessional envelope shadow is named `f1_envelope_shadow_pv_per_undiscounted_usd` (divisor `DF[v]` of the drawing vintage). Option 3, DECLARED and not renamed: the land shadow (`land_shadow_price_by_year`) and the bankability, subsidy-limit and reduced-cost duals in `diagnose_eaas_degeneracy.py`; all are present-value (land: the sum over years of the reported values equals the derivative with respect to the allowance); no result reads them. Record for Appendix B. F-21's carbon shadows (FIN-3, annual branch) were binding-year means of the PV series; the re-run reports both bases.

## L107 — Two diagnostics are present-value sums whose names do not say so

> **Status:** RESOLVED (TASK 53 2.3) (was: LIVE. Found by TASK 52 E4).

> **What.** `eaas_private_investment_usd` is the sum over vintages of DF[t] x EaaS solar MW x capex (fixture: 1,659,438,046 reported; undiscounted 3,303,100,567). `public_budget_realised_spend_usd` is the budget constraint's own left-hand side, also discounted. Neither name carries `npv`, so both read as nominal USD; their siblings (`eaas_subsidy_npv_usd`, `ndc_finance_*_npv_usd`, `npv_gross_capex_*`) do. `eaas_multiplier_vs_public_budget` is a ratio of two PV sums and is unaffected.

> **Direction of bias.** A reader taking the first as capital mobilised understates it by the discounting.

> **What would resolve it.** Rename to `eaas_private_investment_npv_usd` and `public_budget_realised_spend_npv_usd` (CLAUDE.md: rename when the meaning is not what the name says); no stored result is quoted under the old names that I found, but the inventory of consumers is part of the fix.

> **TASK 53 update.** `eaas_private_investment_pv_usd` / `_undiscounted_usd` and `public_budget_realised_spend_pv_usd` / `_undiscounted_usd`; FIN-3's `budget_realised_spend_{pv,undiscounted}_usd` and the capital frontier's `realised_public_spend_{pv,undiscounted}_usd`.

## L108 — `concessional_shadow_usd_per_usd` is None in every `f1_tranche` result

> **Status:** LIVE, unchanged (not in the TASK 53 list) (was: LIVE. Found by TASK 52 E4).

> **What.** `extract_planning_diagnostics` reads `m.concessional_envelope_constraint`, the retired pre-F1 envelope; under `f1_tranche` the live constraint is `m.f1_concessional_envelope_constraint`, so the `hasattr` guard is false and `concessional_shadow_usd_per_usd`, `concessional_drawdown` and `concessional_utilisation` are None. Stored diagnostics confirm: all three are None in `results/ndc_uncapped/*/diagnostics.json` and `results/eaas/diagnostics.json`. The FIN-2 and DEM-1 rows that pass the field through (`optimize_experiments.py:1586`, `:2075`, `:2089`) therefore carry None. The live dual is extracted only in `scripts/18_run_pareto_frontier.py` (`f1_envelope_shadow_usd_per_usd`) and the diagnostic scripts, with the L106 mixed basis.

> **Direction of bias.** None on any quoted number found; a column that reads as empty rather than as zero.

> **What would resolve it.** Read the F1 constraint in the extractor, or drop the field.

> **TASK 53 update.** `concessional_shadow_usd_per_usd` is still None under `f1_tranche`; the live dual is read under the new name `f1_envelope_shadow_pv_per_undiscounted_usd` by `18_run_pareto_frontier.py` and `envelope_dual_pv_per_undiscounted_usd` by the `diagnose_*` scripts.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L109 — `18_run_pareto_frontier.py`'s carbon-shadow `discounted_mean` DF-weighted a series that was already discounted

> **Status:** FIXED at the source (TASK 53 2.3); the stored pre-epoch frontier file carries the defective column.

> **What.** `_discounted_mean(carbon_shadow_by_year)` weighted each year's shadow by `m.DF[t]`. The shadow was the present-value dual (L106), so the weighting discounted it a second time (an effective DF-squared weighting). Found while building the consumer inventory for 2.3. The column is now `carbon_shadow_yr_usd_per_tco2_discounted_mean`, the DF-weighted mean of the YEAR-t series (equal to sum of the PV values over sum of DF).

> **Direction of bias.** The old column understated late-year weight relative to a correct discounted mean. No finding was found that quotes it (`grep` of `findings_frozen.md` and the chapters).

> **What would resolve it.** Nothing further; the re-run regenerates the column.

## L110 — Own flags that are not arms are not in the output key of fixed-name runners

> **Status:** LIVE. Found by TASK 53 2.5.

> **What.** Arm values now reach every output key (G7-4). Flags that are not arms (17's `--draws-file`, `--n-draws`, seeds; 05's capital loop is internal) are not, for the eighteen runners keyed through `resolve_results_dir`. Two invocations at the same arms and different own-flags share one `arms_<hash8>/` directory and the second overwrites the first. Was true before; it is the same silent-overwrite shape as L98.

> **What would resolve it.** Pass each runner's own flags as `extra=` to `resolve_results_dir`.

## L111 — The unlabelled aggregate names remain in the runners G1-2 did not cover

> **Status:** LIVE. Found by TASK 53 2.4.

> **What.** `02_run_ndc_caps.py`, `03_run_ndc_eaas.py`, `27_run_hydro_minflow_sensitivity.py`, `29_run_parameter_uncertainty.py` and `optimize_experiments.summarize_run` still write `solar_total_built_mw`, `final_storage_capacity_mwh` and `actual_emissions_tco2_total`, which are values at, or cumulative to, the last model year (2054) with no year in the name and no 2045 counterpart. `08_phase8_analysis.py` accepts either naming. `scripts/_compare_2_6.py` reads the old baseline names and is stale.

> **What would resolve it.** `labelled_decision_totals` / `labelled_period_totals` (`src/summary_years.py`) in each; mechanical.

## L112 — Several analysis scripts read result files no current runner writes

> **Status:** LIVE. Found by TASK 53 2.5.

> **What.** `21_compute_resolution_premium.py` (`results/baseline/summary.json`), `99_validate_baseline.py` (`results/baseline/timeseries.csv`), `diag_fin2.py` (`tariff_sweep_results__unbounded__df1.csv`) and `verify_2_6_commit_a.sh` read fixed legacy names. The current baseline runners write `summary__<hash8>.json`; the fixed names have not been written for several epochs. They failed before TASK 53 and still do.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L113 — Pre-epoch and post-epoch result files have different schemas

> **Status:** LIVE (a property of the epoch, recorded so nothing reads across it silently).

> **What.** After TASK 53 the following differ from every earlier result file: the gas and carbon shadow keys and every derived `*_shadow_*` column (now `_pv_`/`_yr_`), `eaas_private_investment` and `public_budget_realised_spend` (now `_pv_`/`_undiscounted_`), the reliability fields (`unserved_only_*`, `grid_service_reliability_*`), the baseline summary aggregates (year-labelled), every output path (arm-keyed), the central hash (`6046f759`, was `02c17f55`), and the values (arrears + declining storage). `compare_phase_c_gate.py` and any byte-identity gate against stored results will report every change; that gate detects change, not correctness.

## L114 — `09_run_rel1_feasibility.py` writes in place and carries no arm tag; TASK 53 §2.5's list of keyed runners missed it

> **Status:** LIVE. Found by TASK 55 Part 0 (2026-10-20); recorded, not fixed.

> **What.** TASK 53 §2.5 gave eighteen fixed-name runners an arm-keyed directory (`resolve_results_dir` -> `<results dir>/arms_<hash8>/`, with `ARMS.json` and an `INDEX.md` line). `09_run_rel1_feasibility.py` was not among them: it writes to `results/rel1/<reliability_mode>/<gas_availability_case>/` and nothing else, so only two of its arms (the reliability mode and the gas-availability case) reach the path. The capital-window and storage-capex defaults, `us_inflation`, `tariff_case`, `om_case` and every other standard arm do not. Phase B (item 14, four invocations, 2026-10-19 23:59 to 2026-10-20 00:04) therefore **overwrote the pre-epoch files at the same paths**: all 16 files under `results/rel1/` are post-epoch, none carries an `arm_tag` (`rel1_summary.json` has no arm, mode or window field), and the pre-epoch files survive only in `~/results_pre_fix_epoch.tar.gz`. The G7-4 invariant passes for it because it counts a nested directory as an explicit tag. Consequence for the findings: F-33 is re-derivable from the files as they stand (they are post-epoch by mtime and by the Phase B log), but its counterpart can be matched only by path and mtime, never by tag.

> **Which other runners share the defect.** A scan of `scripts/` for scripts that write result files and call none of `resolve_results_dir`, `complete_arm_tag`, `arm_key`, `hash8_of`, `short_tag` finds, among the runners, only **`09_`** and **`12_run_dem1_demand_sensitivity.py`** (suspended; not in `run_phase_b.sh`; last written 2026-08-23, so nothing was overwritten). The remaining hits are not runners that solve a model: `00_build_emissions_cap.py` (writes `data/cost/processed/emissions_cap.csv`, handled by TASK 53 §4), `08_phase8_analysis.py` and `16_plot_cost_decomposition.py` (post-process stored timeseries into `results/phase8` and `results/figures`), `21_compute_resolution_premium.py` and `99_validate_baseline.py` (L112: read files no runner writes), and the TASK 51-55 report scripts (`51_report_capital_window.py`, `53_epoch_demo.py`, `53_report_horizon_rebuild.py`, `55_*`). The list proved incomplete once, so this scan is by behaviour (does the script write a file and call a key function), not by §2.5's table; it should be re-run whenever a runner is added.

> **What would resolve it.** In `09_`: call `resolve_results_dir(ROOT / "results" / "rel1", args, extra={"reliability_mode": args.reliability_mode})` after argument parsing (it needs `add_standard_arms` to have been called with every arm; `09_` already takes the standard arms) and write under `<that dir> / args.gas_availability_case`, the pattern `10_` and `11_` already use. Readers to update: `scripts/diagnose_f33_window.sh` (reads `results/rel1/{form}/{arm}/...` at lines 34 and 82); `scripts/build_results_inventory.py` (line 61, by log prefix only); `run_phase_b.sh` item 14 (no change; it passes the arms). Re-running the four invocations is 384 solves. `12_` takes the same change if it is ever un-suspended.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L115 — Chapter 3 reference list: entries the repo cannot close

> **Status:** LIVE (found by TASK 56, 2026-10-20; recorded, not fixed).

> **What.** After TASK 56's §2.1, `docs/chapters/CHAPTER_3_assembled.md` still carries a `[VERIFY]`/`[REF]`/`[TITLE]`/`[VINTAGE]` marker on the entries listed in `docs/task_log_56_chapter3_close.md` §3 (the single external-source worklist) and §2 (resolved-in-part entries). None can be answered from `data/`, `literature/`, the sourcing packs or `docs/parameter_provenance.md`. In-text markers in the body (lines 33, 120, 131, 449, 723, 759, 761, 931, 1010, 1107, 1146, 1263, 1338, 1358, 1436, 1605, 1617, 1662, 1709, 1811, 1843, 2053, 2455) were not touched; those that duplicate a reference-list marker will go stale if the list is closed first.

> **What would resolve it.** The worklist in the TASK 56 log, one external lookup per row. Line 2053 also needs a register number for the cited limitation ("L[VERIFY: resolve this register number]").

## L116 — The DARES envelope is a 750M IDA grant-and-TA programme; the model treats it as an unconditional concessional debt stock

> **Status:** LIVE (found by TASK 57, 2026-10-21; recorded, not fixed).

> **What.** The PAD (PAD00037, P179687) gives 750M IDA = 150M Credit + 600M Shorter Maturity Loan, states no terms for either, delivers the money to developers as results-based capex grants (PBG, catalytic grants) and TA, makes 600M contingent on three PBCs, and finances only distributed generation (isolated and DISCO-interconnected mini grids, rooftop, standalone solar; largest unit 5 MWp). `src/scenarios.py` uses 750M as a fully available, capital-eligible, grid-connected concessional tranche priced at IDA21 Blend 2.12%/25yr (`env_dares`, `cr_central`). Technical assistance (40M, 5.3%) is not separated from capital. Comments at `scenarios.py:117-120` and `:2128-2129` misdescribe the instrument and the approval. Prose elsewhere ("domestic deployable portion", `financing_endogeneity.md:649`, `limitations_register.md:1262`) implies a fraction the code does not apply. See `docs/task_log_57_dares.md` §3.

> **What would resolve it.** Chat decides what the thesis claims for `env_dares` (a stylised concessional tranche sized to the DARES headline, not DARES itself). Any model change (a 150M core arm, a TA deduction, a grant-versus-debt treatment) is a separate task with a solve.

## L117 — No source in the repository for the "multilateral concessional finance does not fund unabated fossil generation" convention

> **Status:** LIVE (found by TASK 57, 2026-10-21; recorded, not fixed).

> **What.** The gas exclusion (`gas_conc_vol` fixed to 0.0; docstring "DARES is a distributed-renewables facility") rests on a declared ASSUMPTION, not a lender rule. The PAD states no technology exclusion and finances solar-battery-diesel hybrids. Direction of bias: excluding gas from the concessional tranche denies gas the cheap capital that solar and storage receive, so it favours solar/storage and understates gas's financing-side competitiveness (my reading; check against the §1 text this entry was asked to follow, which I did not have). The general convention is not supported by anything in `literature/`, the sourcing packs, `docs/references*.md` or the provenance tables, and none at the lending institution (the PAD, `docs/task_log_57_dares.md` §4).

> **What would resolve it.** One external primary source (a World Bank Group climate action plan, IDA energy policy or MDB joint statement) with page, for §3.4.3.

## L118 — The Electricity Act 2023 is not held as the gazetted Act: `literature/Electricity_Act_2023.pdf` is the enrolled Bill

> **Status:** LIVE (found by TASK 58 B.1, 2026-10-22; recorded, not fixed).

> **What.** The file held at `literature/Electricity_Act_2023.pdf` (151 pages, image-only scan, SHA-256 `1ab4958b24c45bf4629f7dc4713bbe929ffe89267faeb245c27f7d7621172a6d`) is the enrolled copy of the Electricity Bill, 2023, not the gazetted Act. It is headed "A Bill — For"; the commencement bracket is empty; the Clerk's certification (5 June 2023) calls it a true copy of the Bill; the assent block is signed but the day digit of the assent date is illegible in the scan. It carries no Act number or gazette particulars. The text of the Fifth Alteration (constitutional amendment) is not present. Published secondary sources give the assent date variously as 8, 9 and 10 June 2023, so the date cannot be resolved by secondary citation. No section number, provision or definition has been taken from this file for the thesis (TASK 58 B.2 to B.5 not run). Chapter 1 §1.2.3 keeps its existing [VERIFY] marks.

> **What would resolve it.** The gazetted Act (Act number, commencement date, gazette particulars) and the Fifth Alteration text, supplied as primary documents; then B.1 is repeated against them and B.2 to B.5 are run.

## L119 — A suite assertion that reads chapter prose by string match tracks wording and file paths, not the property it tests (class; instance G9-2)

> **Status:** LIVE (found by TASK 58 A.3 and TASK 59 Parts 0 and 1, 2026-10-22; recorded, not redesigned).

> **What.** `scripts/52_invariants.py` G9-2 tests a claim about the model (solar and storage capital costs decline over the horizon; hydro series are flat) and, as one of five conjuncts, string-matches the sentence in the chapter that states it (`:2269-2270`). Three properties of that class, each established by TASK 59 Part 1:
>
> 1. **Wording coupling.** The pass status of the prose conjunct tracks the wording of one sentence, including the position of a line break (pattern A requires a newline after "costs"; pattern B tolerates it). Any revision of that sentence breaks the assertion with nothing in the model changing.
> 2. **Path coupling.** G9-2 failed because commit `009d230` (2026-10-20) deleted `docs/chapters/SECTION_3_7_final.md` in a deliberate consolidation into `CHAPTER_3_assembled.md`, not because any claim became untrue. The harness raised on the missing file, so the assertion reported FAIL from 2026-10-20 until the repoint (`d306a15`), and TASK 58 found the tally at 79 / 7 / 1 against §3.6.10's 80 / 6 / 1. G9-2 has no register entry of its own.
> 3. **Loss of specificity introduced by the repair.** `d306a15` points the search at the whole assembled chapter, not §3.7 alone, so a pass no longer demonstrates by construction that §3.7.3 states the claim. Checked at the time of the repair (TASK 59, against the working tree): the phrase "Solar and storage capital costs" occurs once in the chapter, at line 3165, inside §3.7.3 (lines 3064 to 3207), and both patterns match once. The loss is therefore not currently exposed, but it is not guaranteed by the assertion; a second occurrence elsewhere in the chapter would satisfy it.
>
> The same class is open wherever the suite reads prose. TASK 59 Part 0 found that the suite names a deleted chapter file only in G9-2; I did not audit the other assertions for string matches against documents.

> **What would resolve it.** An assertion that tests the model's series directly and cites the chapter as documentation rather than as evidence (the four model conjuncts of G9-2 already do this), or one that slices the named section (`## 3.7 Limitations` to the next `## ` heading) before matching. Not done here; the repair in `d306a15` is a path change only.

## L120 — `ccgt_efficiency_case` reached no model term under the blended dispatch representation; it is a live lever under vintage dispatch

> **Status:** RESOLVED by the dispatch correction (TASK 74–81). Under the blended representation the parameter was read only inside the `new_gas_deliverability = "exempt"` branch, so its three arms at the `constrained` baseline were identical and the family's span was zero — a property of the wiring, not a null result. Under vintage dispatch the parameter enters the dispatch of capacity added within the model, and the family spans 0.3309 bn USD unconstrained (rank 14) and 4.0030 bn under the NDC 3.0 conditional cap (rank 4). The zero-span rows reported before TASK 81 are not to be cited as null results. *(Prior status, retained as history: LIVE: the sizing defect is RESOLVED (TASK 66, commit `d9ed430`); what remains live is that `ccgt_efficiency_case` reaches no model term by default, so the `ccgt_efficiency` zero-span rows are inert by construction and invariant G7-2 fails on it, correctly. *(Prior status, retained as history: LIVE (found by TASK 61 Part A, 2026-10-22; recorded, not repaired).)*)*

> **Corrected 2026-10-23 (TASK 69 A.2).** The entry's own TASK 66 update ends 'Status: RESOLVED for the sizing defect; see L123 for the representation gap it leaves' while the header said LIVE. Evidence: `src/optimize_model_sliced.py:935` (`/ eta / 1e6`), `results/baseline_sliced/summary__{768c2796,9d0e06c3,86eea714}.json` (identical 23601372250.73667), invariant G7-2 (`docs/task_log_66.md` Part B) (audit pass 3 A.2, A.4 item 5).

> **What.** `gas_new_efficiency` (set from `ccgt_efficiency_case`, `src/scenarios.py:2865`, `:3579`) is read in one place in each builder, inside `if new_gas_deliverability == "exempt":` (`src/optimize_model_sliced.py:918-949`, `src/optimize_model.py:675-683`). At `constrained` (the central value) it is not read and no term depends on it; dispatch OPEX and emissions use the fleet `gas_eta` throughout. The three arms `ccgt_eff_iso` (`f25d5493`), `ccgt_eff_central` (`6046f759`), `ccgt_eff_low` (`3c01d259`) in `results/baseline_sliced/` therefore have the same objective to the cent (24986620949.18249) and identical diagnostics apart from the arm tag. The code documents the conditional use (`scenarios.py:1864-1868`), but the family is declared as a lever (`LEVER_FAMILIES`, `ccgt_efficiency`, `in_f5_ranking: False`) and its docstring calls it "a large lever". ~~The effect under `exempt` is untested: the two stored `exempt` baseline files (`768c2796`, `6f3780de`) both carry `ccgt_eff_central`.~~ **Tested in TASK 62 B (live under the old expression), then made inert again by the TASK 66 repair; see the updates below.**

> **Affected rows.** The `ccgt_efficiency` row (span 0.0000 bn, three arms, one plan) of the one-plan block in `results/lever_ranking/arms_6046f759/lever_ranking.csv`. It measures nothing about CCGT efficiency and must not be published as a null finding.

> **What would resolve it.** Two solves, `ccgt_eff_iso` and `ccgt_eff_low` at `new_gas_deliverability = exempt` and otherwise central, compared with `768c2796`; or the family declared as conditional on `exempt` in `LEVER_FAMILIES`.

> **Update (TASK 62 Part B, 2026-10-22).** The two solves were run (`summary__9d0e06c3.json` `ccgt_eff_iso`, `86eea714` `ccgt_eff_low`, against `768c2796`, all `exempt`). Under `exempt` the parameter is live: span 152,354,404.56 USD (0.1524 bn), plans differ from 2032 (`docs/task_log_62.md` Part B.2). The inference above is confirmed: the zero span at `constrained` was an artefact of the branch. Status stays LIVE for the family declaration and the published row, which still presents `ccgt_efficiency` as a lever at `constrained`; the model is not at fault.

> **Verdict on the sign (TASK 63 Part B, 2026-10-22).** Under `exempt` the more efficient arm costs more (`ccgt_eff_low` 24,228,389,758.82 < central 24,287,967,211.48 < `ccgt_eff_iso` 24,380,744,163.37) and builds less gas and more solar. Traced in `docs/task_log_63.md` B.1-B.5: `gas_new_efficiency` is a **divisor** in `m.gas_exempt_fuel_allowance_twh_th` (`src/optimize_model_sliced.py:928-932`), so the allowance on the `m.gas_balance` right-hand side (`:941`, `:949`) falls as efficiency rises, while dispatch converts new capacity's fuel at the blended `gas_eta` 0.287 (`:824`). The allowance is `0.287 / efficiency` (52.7% / 55.5% / 57.3% for iso / central / low) of the fuel the model's own dispatch burns to run the added capacity. **Verdict: a defect in the expression, a basis inconsistency** (allowance sized at one efficiency, dispatch at another), contradicting the design record's claim of exact equivalence to a vintage split (`docs/phase_2_9_gas_addition.md`, Step 3) and the lever description at `src/scenarios.py:1895-1903`. The 0.1524 bn efficiency span measures that divisor, not an efficiency effect on dispatch. All `exempt` results to date (the `new_gas_deliverability` family span 0.6987 bn at `768c2796`; the TASK 62 B arms `9d0e06c3`, `86eea714`, `332649ba`, `5c22ca7f`) are conditional on this expression. Not repaired: either sizing the allowance at the dispatch `eta` or dispatching new capacity at its own efficiency is a design change that invalidates them. Status LIVE.

> **Update (TASK 66 Part B, 2026-10-22): REPAIRED, Option 1.** `m.gas_exempt_fuel_allowance_twh_th` is now sized at the dispatch efficiency `eta` (`src/optimize_model_sliced.py:935`, `src/optimize_model.py:686`, commit `d9ed430`); `gas_new_efficiency` is read in neither builder. Constrained arms byte-identical (`6046f759`, 24986620949.18249). Under the repaired expression the `ccgt_efficiency` arms are identical under `exempt` as under `constrained` (span 0.0, `768c2796`, `9d0e06c3`, `86eea714`, all 23601372250.73667): the parameter no longer enters the model anywhere, which is the intended consequence of Option 1 and not an efficiency finding. Exempt results re-solved: central 23601372250.73667 (was 24287967211.476887), gas-balance dual positive in 11 of 30 years (was 22); capital-recovery span over four arms 1.8787 bn (was 2.2799 over three; constrained 3.1300); `new_gas_deliverability` family span 1.3852 bn (was 0.6987). The old exempt figures are not citable. Invariant G7-2 now fails on `ccgt_efficiency_case` ("changes nothing"), correctly. The design record `docs/phase_2_9_gas_addition.md` (Step 3) is accurate under the repaired expression. The Chapter 3 passage at `CHAPTER_3_assembled.md:460-465` (allowance written with $\eta^{new}$) no longer matches the code. Status: RESOLVED for the sizing defect; see L123 for the representation gap it leaves.

> **Generation (TASK 86):** the hash-keyed result files this entry cites are blended dispatch (no `gas_vintage_dispatch=True` in the cited files' arm records). Checked: `6046f759` (blended); `9d0e06c3` (blended). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142. [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L121 — `dscr_case` acts only on EaaS-financed capacity; at the `traditional`-regime baseline its three arms cannot differ

> **Status:** LIVE (found by TASK 61 Part B, 2026-10-22; recorded, not repaired).

> **What.** `scenario["required_margin"]` (from `dscr_case`, `src/scenarios.py:3515`) is used at two places in the sliced builder, `eaas_bankability_rule` (`src/optimize_model_sliced.py:1130`) and `subsidy_limit_rule` (`:1144`), both on `m.solar_eaas_add`. With `financing_regime = "traditional"` (`scenarios.py:3464`) the builder adds `m.eaas_disable` (`:1136-1138`), fixing `solar_eaas_add <= 0`, so the parameter multiplies a variable that cannot move. It is not in the objective. There is **no coverage-ratio constraint on the concessional/commercial tranche split**: that split's only constraint is the undiscounted envelope (`:1787-1794`). The three arms `dscr_central` (`6046f759`), `dscr_low` (`37449688`), `dscr_high` (`3f01567d`) have the same objective to the cent and the same diagnostics apart from the arm tag; `concessional_volume_by_vintage_usd` and `concessional_share_by_vintage` are equal in all three (2032: 750,000,000.0, 0.9324). The constraints' duals are not stored and were not solved for. The chapter already states that the objective does not register the margin (§3.6.10), but §3.4's overview describes the layer as a tranche split "constrained by ... a coverage-ratio test" (`CHAPTER_3_assembled.md` :1590-1593, :1666-1668), which the code does not implement for the tranche split.

> **Affected rows.** The `dscr` row (span 0.0000 bn, three arms, one plan) of the one-plan block in `results/lever_ranking/arms_6046f759/lever_ranking.csv`. It is a statement about the `traditional` regime, not about the size of the coverage ratio; the parameter's effect, if any, is in the `eaas` regime runners.

> **What would resolve it.** Reading the bankability constraint's dual at an `eaas`-regime arm, or varying `dscr_case` there; and Chat deciding the §3.4 wording.

> **Generation (TASK 86):** the hash-keyed result files this entry cites are blended dispatch (no `gas_vintage_dispatch=True` in the cited files' arm records). Checked: `6046f759` (blended). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142. [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L122 — A source-reading assertion (G1-2) caught a diagnostic addition that the model, the summary and the objective were all unaffected by; the addition had been predicted not to affect the suite

> **Status:** RESOLVED (TASK 62 A.4, 2026-10-22; was: LIVE for the interval between TASK 60 `3dca52b` and TASK 62 `de3cee9`).

> **What.** TASK 60 added a `cost_decomposition` block to `scripts/01b_run_baseline_sliced.py`'s diagnostics, with two new scalar keys that name a full-horizon NPV without a window: `capital_charge_total_npv` (in `components_unrounded`) and `capital_charge_total_minus_six_lines_usd` (in `reconciliation`). Invariant G1-2 (`scripts/52_invariants.py:1841`) reads the runner's source and requires every scalar aggregate key to carry its year or window; it flagged exactly those two. The suite went from 80 / 6 / 1 to 79 / 7 / 1 (TASK 61 D.1). The model, the objective (24986620949.18249), `decision_variables`, the summary file and the CSVs were unchanged throughout (TASK 60 and TASK 62 A.3), so no result was affected; only a naming convention was. The change was a diagnostic addition made in the same session that asserted it would not affect the suite (TASK 61's brief and TASK 60's closing note both expected 80 / 6 / 1).

> **Why it is recorded.** The assertion behaved correctly. The convention it enforces is the one this project has already been caught by (a reporting-window numerator read against a full-horizon denominator), and a full-horizon NPV is window-dependent. Its firing was predicted not to occur, for the second time in two tasks (the first was TASK 59's G9-2 repoint, L119, whose pass status tracked wording). The lesson is that "touches nothing the model or the summary reads" does not establish "touches nothing the suite reads": assertions that scan the runner's source see its key names.

> **Resolution.** The two keys were window-labelled (`capital_charge_total_full_horizon_npv`, `capital_charge_total_minus_six_lines_full_horizon_usd`), the assertion's scope was not changed, and the annual block's `components` keys were left alone (TASK 62 Part A, `de3cee9`). The suite is back at 80 / 6 / 1 with G1-2 the only row that moved (`docs/task_log_62.md` A.4).

## L123 — The model does not represent new-build thermal efficiency: all gas generation converts at the blended fleet value

> **Status:** LIVE (named by TASK 66 Part B.8, 2026-10-22; limitation and further work).

> **What.** `m.gas_generation = eta * m.gas_to_power` (`src/optimize_model_sliced.py:824`), emissions use the same `eta` (`:1925`), and the exempt allowance is sized at the same `eta` (`:935`), with `eta = scenario["gas_eta"]` = 0.287 (`src/scenarios.py:3214`), whatever the vintage. New combined-cycle capacity converts at `ccgt_eff_central` = 0.5173 in reality (about 1.8 times the electricity per unit of fuel), so on a fuel-bound system (the deliverability dual is positive in 23 of 30 years at the constrained central arm, `diagnostics__6046f759.json`) the model understates what added gas capacity yields at the margin. `ccgt_efficiency_case` is carried in the arm tag and the scenario (`scenario["gas_new_efficiency"]`) and reaches no model term by default, so the three arms of the `ccgt_efficiency` family are identical (L120). Direction of bias, not measured here: against new gas.

> **What would resolve it.** Split `m.gas_to_power` by vintage (added capacity at `gas_new_efficiency`, legacy at `gas_eta`) with a regression arm reproducing current behaviour exactly when the two efficiencies are equal. A bounded sensitivity behind a default-off flag is specified in TASK 65 Parts B and C (`docs/task_log_66.md` Part C). A full treatment invalidates every stored result and belongs with the forced re-run (see "Deferred structural changes").

> **Sensitivity measured (TASK 66 Part C = TASK 65 B to E, 2026-10-22).** A default-off flag `gas_vintage_dispatch` (`src/optimize_model_sliced.py`, `scripts/01b_run_baseline_sliced.py --gas-vintage-dispatch`, commit `d6da3b3`) converts capacity added in the model at `gas_new_efficiency` and pre-existing capacity at `gas_eta`; emissions follow the thermal draw; capacity stays pooled. Regression: flag off, byte-identical (`6046f759`, 24986620949.18249); flag on with the new efficiency set to `gas_eta`, identical objective and plan (`results/vintage_dispatch_sensitivity/summary__8470793b.json`). At `ccgt_eff_central` (0.5173) the stored pair `6046f759` (`gas_unconstrained`) and `4c3166b3` (`gas_none`) behaves differently: cumulative gas generation to 2045 is 366.698 TWh_e with new gas against 247.382 TWh_e without (+119.315 TWh_e, +48.2%) where blended dispatch gives −0.104 TWh_e; the central objective falls 1.664 bn (−6.66%) to 23,323,068,344.06 (`summary__33957d4d.json`); solar is lower by 7,453 MW (2045) and 8,091 MW (2054), storage by 12,717 MWh and 6,766 MWh, gas built is 3,825 MW against 646 MW at 2045; the gas-balance dual is positive in 23 of 30 years in both modes; cumulative emissions to 2045 are unchanged (the thermal draw is the same). **No frozen finding states the no-net-generation claim** (it was only P2-4 in a superseded document, retired as L124), so the sensitivity tests a measured property of the stored results, not a published finding; that property does not survive vintage dispatch. Blended dispatch for all vintages is a modelling choice that matters for any conclusion about what new gas adds. Not adopted as the model; stored results are blended-dispatch results.

> **Generation (TASK 86):** the hash-keyed result files this entry cites are MIXED: some cited files are vintage, some blended; each is named where its figure is quoted. Checked: `33957d4d` (both); `6046f759` (blended); `8470793b` (both). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142. [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L124 — P2-4 retired: "New gas displaces storage, not solar, and adds no generation" is withdrawn

> **Status:** RESOLVED as a retirement (TASK 66 Part C.1, 2026-10-22). The claim is withdrawn, not carried.

> **What.** P2-4 (`docs/findings_provisional_pre_sweep.md:717-726`, a superseded file, no F-number, no `Source` line, "STRUCTURAL. Pre-F1 statement of what F-7 confirms under F1 arms") stated four clauses about `gas_none` against `gas_unconstrained`: (1) gas generation is bit-identical through 2045 and so are the gas shadow prices; (2) cumulative emissions to 2045 are identical to the tonne (358.8 Mt); (3) storage falls 46% for a 0.6% cost reduction, i.e. new gas displaces storage rather than solar; (4) new thermal capacity is bought for firm capacity only because there is no more fuel. The pre-F1 numbers are not on disk. Against the post-epoch stored pair `6046f759` / `4c3166b3` (`results/baseline_sliced/summary__*.json`, `diagnostics__*.json`):

> | Clause | Post-epoch files |
> |---|---|
> | (1) bit-identical gas generation and duals | Not so: the sum to 2045 differs by 0.104 TWh_e (0.042%, `gas_none` the higher) but individual years differ by up to 4.05 TWh_e (2045: 14.731 against 10.684); the gas-balance dual is positive in 23 years against 12 |
> | (2) emissions identical to the tonne | Not so: 338,177,278.30 against 333,717,859.33 tCO₂ to 2045, 4.46 Mt (1.3%) apart |
> | (3) displaces storage, not solar, for 0.6% | Not so: solar is lower with new gas by 2,887.6 MW (−13.0%) at 2045 and 13,266.4 MW (−25.4%) at 2054, more than storage in relative terms at 2054 (−15.3%); the cost saving is 1.90% (482.7 M USD) |
> | (4) firm capacity only, no added generation | Holds only as cumulative generation to 2045 under blended dispatch (−0.104 TWh_e), and not under vintage dispatch (+119.3 TWh_e; L123) |

> **Where it was found live.** Nowhere: the search of TASK 66 Part A found no statement of any of the four clauses in the abstract, Chapters 1 to 4 (drafts included), `findings_frozen.md`, this register or the handovers (`docs/task_log_66.md` Part A). The adjacent F-19 composition sentence ("gas exits, storage displaces it at 4.0 MWh per MW", `findings_frozen.md:1069-1071`) is a different clause, about the financing arms, and is untouched. P2-4 and its restatements in the superseded file are not to be quoted.

## Deferred structural changes

> An index over existing entries, not new limitations (TASK 65 Part E, 2026-10-22). Each is a change to the formulation or its parameters that would alter stored results and so should ride along with any full re-run that is forced for another reason. "Flag and regression arm" says whether the change can already be switched on behind a default-off option with a tested reproduction of current behaviour.

| Change | Flag and regression arm | Would invalidate | Register entry |
|---|---|---|---|
| Vintage-differentiated gas dispatch (added capacity converts at `gas_new_efficiency`, legacy at `gas_eta`; emissions on the thermal draw) | **Yes**: `scenario["gas_vintage_dispatch"]`, `01b --gas-vintage-dispatch`; off = byte-identical to `6046f759`, on at `gas_new_efficiency = gas_eta` = identical objective and plan (TASK 66 C, commit `d6da3b3`). Sliced builder only; validated at `constrained` only | Every stored result that contains gas, i.e. all of them; the `ccgt_efficiency` family would stop being inert; invariant G7-2 would return to passing for that arm | L123 (sensitivity result), L120 |
| A hydro operation-and-maintenance term (and capital, if declared) in the objective | **Yes (TASK 69 B.1)**: `scenario["hydro_om_term"]`, `io.compute_om_cost_npv(hydro_capacity_mw=..., hydro_fom_usd_per_kw_yr=...)`, `01b --hydro-om-term`; off byte-identical to `6046f759`, on at the null rate 0.0 identical; **no non-zero rate set** (no provenance row yet; candidate NREL ATB 2024 hydropower FOM, USD/kW-yr, workbook not in the repository). G2-3 needs the hydro entry added when the term is switched on | Every hydro-contingent objective and difference, including the hydro lever (13.65 bn) and F-18, and the central objective itself | L95 |
| A state of charge in the storage seasonal balance (ordering of charge before discharge within a season) | No. **Considered for this epoch and declined (TASK 69 B.3):** it adds decision variables and changes the representation of the 63,056 MWh of storage in the plan, and Chapter 3 §3.7.3 declares the limitation | Storage quantity and every storage-dependent result (F-19, F-3 composition, capital and fuel comparisons) | L102 |
| `selfgen_busbar_factor` set from the 7% transmission loss factor (1/(1-0.07) = 1.0753) instead of 1.0 | **Yes (TASK 69 B.2)**: now an explicit `load_scenario()` value (1.0) read directly by both builders; `01b --selfgen-busbar-factor`; default byte-identical to `6046f759`, explicit 1.0 identical; the central arm dispatches no self-generation, so the effect lives only in capped and leakage runs; `24_` still sets it by environment variable without recording it | Every result in which priced self-generation is dispatched (the baseline backstop, the leakage and cap arms) | L96 |
| Renormalising the peak and season factors (only if that, not the wording fix, is chosen) | No | Every `peak_mw` (by 0.003%), so formally every stored result | L100 |
| The capped lever ranking: 40 arms per cap case under NDC 3.0, `ndc3_conditional` as the binding commitment and `ndc3_unconditional` as a robustness check (decision TASK 69 A.4; NDC 2.0 and the decline arm excluded) | No flag needed; `02_run_ndc_caps.py` runs the cap cases, but at `tight`/`moderate` capital cases and writing a layout `scripts/build_lever_ranking.py` does not read (audit pass 3 D.1, D.3); about 80 solves for the two cap cases | Adds results, invalidates none; depends on the caps (L61) and on the efficiency basis (L123, L125) being settled first | L24, L51, L61 |

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L125 — `gas_eta` and `ccgt_efficiency` are on different bases; the 1.80 uplift the sensitivity rests on overstates a like-for-like ratio

> **Status:** LIVE (TASK 67, 2026-10-22). No verdict on the epoch; that is for Zeus.

> **What.** `gas_eta` = 0.287 (`src/scenarios.py:3214`) is a realised fleet-average efficiency: NERC grid generation (total minus hydro, 25.62 TWh_e) over gas-to-power volumes at an HHV of 1,036 Btu/scf (89.27 TWh_th); open cycle 59.2%, combined cycle 21.7%, gas-steam 19.0% of the 10,947 MW fleet; it embeds part load, degradation, cycling and composition, and the project's own unresolved billed-versus-combusted gas-volume boundary (swept 0.277-0.298 in `scripts/29_run_parameter_uncertainty.py`). `ccgt_efficiency` (0.5445 / 0.5173 / 0.5010, `scenarios.py:1858-1913`) is the design-point efficiency of one new H-class combined-cycle unit, NREL ATB heat rate 6,266 Btu/kWh at ISO conditions with an ASSUMED relative ambient derate of 0, 5% and 8%; it embeds no operation. Neither has a row in `docs/parameter_provenance.md`; the primary NERC 2024 and NGC volume documents and the ATB workbook are not in the repository, and the 1,036 Btu/scf has no cited source. Whether either is gross or net of station auxiliaries is not stated anywhere in the repository, and the two cannot be compared on that dimension. On a common basis the uplift is between 1.53 and 1.80: 1.613 if the new unit realises the same fraction of its rating (0.287 / 0.3375 = 0.850) as the existing fleet, against the 1.802 used by the vintage-dispatch sensitivity (L123), about 10.5% lower; the 0.3375 fleet ISO expectation is itself indicative and its per-type inputs are not in the repository. The converted-plant stream converts at 0.46 (`conv_central`; 0.42 / 0.50), a design-point figure, giving conversion an uplift of 1.603 on the fleet value, while new build earns 1.00 by default and 1.802 under the flag. `gas_add` is declared combined cycle in Chapter 3 (:1220) but its capex is technology-blind (NIRP-derived, `scenarios.py:1839-1841`); nothing in the repository settles the efficiency a generic new unit should carry.

> **Also found.** `data/gas/processed/gas_to_power_reconciliation_2025H1.xlsx` stores an implied efficiency of 0.4349 computed from Q1-Q3 thermal generation (20,320 GWh) against six months of gas (153,879 MMscf); period-matched it is 0.3015. The stale notes `data/gas/assumptions/gas_to_power_efficiency.md` (η ≈ 0.43) and `gas_heating_value.md` (≈1,030 Btu/scf labelled LHV) contradict the live values.

> **What would resolve it.** Provenance rows for both parameters; the primary ATB and NERC/NGC documents in `literature/`; a stated common basis (gross or net, design or realised) for all three efficiencies (fleet, new build, retrofit), or a sensitivity of the vintage-dispatch result to the ratio over 1.36-1.80. `docs/task_log_67.md`.

> **Decision recorded 2026-10-23 (TASK 69 A.4): `gas_add` is combined cycle.** Consequence: when the epoch applies the efficiency basis (L123, this entry), new build takes the combined-cycle efficiency (`ccgt_efficiency_case`, on whatever common basis is settled), and the capital cost of new build **remains technology-blind** (NIRP-derived `gas_capex_central` 1,317,000 USD/MW, `src/scenarios.py:1839-1841`; NIRP does not state open or combined cycle), so the cost and the efficiency of the same unit rest on different evidence. Chapter 3's declaration (`CHAPTER_3_assembled.md:1220`, 'New gas is declared combined cycle') is now the model's position, not an inference. The generic-unit efficiency question TASK 67 C.4 left open is closed by this decision, not by a source.

## L126 — `24_run_leakage_arm.py` sets the busbar factor from the environment variable `BUSBAR_ARM` and records it nowhere

> **Status:** LIVE (TASK 70 Part D, 2026-10-23; found in TASK 69 B.2). Not repaired in this task.

> **What.** `scripts/24_run_leakage_arm.py:289-290`: `if os.environ.get("BUSBAR_ARM"): sc["selfgen_busbar_factor"] = 1 / (1 - 0.07)`. Any non-empty value of the variable switches the factor from 1.0 to 1.0753; the value is not written to the output (no key, no summary field) and not to the log (the script has no other reference to the factor; `grep busbar` returns only these two lines). The value any stored leakage-arm result used therefore cannot be established from its files, and two runs that differ only in the variable write the same output location. Recorded earlier as an unresolved item in `docs/task_log_58_invariants_and_act.md:204` (item 6) and described in `docs/phase_2_4_wiring.md:245`; this entry makes it a register defect. Related: L96 (the factor), L98 (runner keys).

> **Direction of bias.** None on a result known to have been run with the variable unset (factor 1.0, the declared default); unknown for any stored `24_` result whose invocation was not otherwise recorded. The factor multiplies self-generation, so it bears on the leakage quantities Question 3 reports.

> **What would resolve it.** Not a repair of the runner as it stands. The epoch's capped runs must carry the factor in their key (the `selfgen_busbar_factor` runner option built in TASK 69 B.2 enters the key as `__selfgen_busbar_factor=<v>`), which is where the fix belongs; until then no stored `24_` leakage-arm result should be cited for a quantity that depends on the factor.

## L127 — `ccgt_eff_iso` was attributed to the NREL ATB for the life of the project; the attribution was false and the value was right

> **Status:** RESOLVED (TASK 73 Part A, 2026-10-23). The value was never wrong; its citation was.

> **What.** `src/scenarios.py` (`ccgt_efficiency_scenarios()` docstring), `docs/methodology_physical_layer.md:473-475`, `docs/methodology_claims_ledger.md:366`, `docs/phase_2_13_gas_cost_efficiency.md:248-250`, `docs/model_capability_inventory.md:1342-1344, 2100` and `docs/parameter_provenance.md` row 24 cited `ccgt_eff_iso` = 0.5445 (net heat rate 6,266 Btu/kWh) to "NREL ATB 2024 ... new-build H-class NGCC", with "EIA / Sargent & Lundy 2023" as supporting. The ATB workbook does not contain 6,266: TASK 72A searched all 39 sheets of the v3 workbook (and v4), every non-empty cell, cached values and formulas, and found no value within ±0.001 MMBtu/MWh (nearest 6.269, F-frame class), which is what made the real source findable. The figure is in the EIA-published Sargent & Lundy report SL-018001 Rev A (6 December 2023), Table 5-1 (PDF p.76, printed p.34) and Table 1-2 (PDF p.24): CC 2x2x1, H-class, 1,227 MW net, net heat rate 6,266 Btu/kWh on a higher-heating-value basis, $868/kW, fixed O&M $12.12/kW-year, variable O&M $3.41/MWh, 42-month lead time, 40-year operating life (`literature/capital_cost_AEO2025.pdf`, SHA-256 `ad912713...96ec`).

> **Basis, as the document states it.** HHV (Table 5-1 row label); net of internal auxiliary power (PDF p.73: two CTs about 436 MW gross each and an STG about 393 MW gross, "after deducting internal auxiliary power demand, the net output of the plant is 1227 MW"); quoted at 60 F and 60% relative humidity at a generic US site, not named ISO. The 5% and 8% derates remain ASSUMED and are not from the report.

> **Direction of bias.** None on any result: the value is unchanged. The ISO label in the code docstring and in the name `ccgt_eff_iso` overstated the document's condition label (60 F / 60% RH); no value moves.

> **Resolved by.** Citations corrected in every place above; provenance row 24 returned from ASSUMED to SOURCED (base) with report, table and page. The 19 September 2023 EIA slides the task names were not in `literature/` and were not read; the primary report is sufficient.

## L128 — `gas_avail_observed` was 0.295 against a stated derivation that gives 0.3061; how 0.295 was produced could not be traced

> **Status:** LIVE (TASK 73 Part B, 2026-10-23). The code value is corrected; the origin of 0.295 is not recovered, and every stored result was run at 0.295.

> **What.** `gas_avail_observed` was 0.295 (`src/scenarios.py` `gas_availability_scenarios()`), documented as "NERC-measured 2024 (3,351 / 10,947 MW)" (`src/scenarios.py:1134`, `docs/task_log_10_availability_ceiling.md:14`, `docs/model_capability_inventory.md`, `docs/phase_2_9_gas_addition.md`). 3,351.16 / 10,947 = **0.3061**. Both numbers are the sums of the same 23 gas plants in Table 7.1 of the NERC (Nigerian Electricity Regulatory Commission) 2024 Annual Report (PDF p.149, printed p.143): 3,351.16 MW is 4,853.69 (Total row) less the five hydro rows (378.25 + 25.60 + 327.02 + 372.12 + 399.55 = 1,502.54); 10,947 MW is 13,625 less 2,678 hydro. **0.295 equals 3,351.16 / 11,361 = 0.29498 numerically**, where 11,361 adds AES_1 (294 MW) and Gbarain_1 (120 MW), licensed (Table 4.3) but absent from Table 7.1. Whether 0.295 was made that way could not be traced: no file, log or commit message states a computation giving 0.295.

> **Evidence on the two plants.** AES_1 and Gbarain_1 appear in the report only in the licence and renaming tables (Tables 4.3 and D.1), not in Table 7.1 (availability), Table D.3 (load factor) or Table D.5 (plant-level generation, 2024). The 27 rows of D.5 plus Zungeru_1 (7.48% share, Figure 7.3) account for the stated 37,093.70 GWh to within rounding (residual between -2 and +1 GWh), so the two plants delivered at most a few GWh. The report states no reason (it never says either was idle, repaired or decommissioned).

> **Decision (TASK 73).** `gas_avail_observed` = **0.3061**, numerator and denominator over the same 23 plants, which is also the capacity base the model already uses (`GAS_BASELINE_MW` = 10,947). **0.2950 is recorded as a coherent alternative**: it is the availability of the licensed 25-plant gas fleet if AES_1 and Gbarain_1 had zero available capacity. The report does not state that; it shows only that they generated nothing.

> **Same convention elsewhere.** 2025/Q4: the Q4 2025 quarterly report (`data/gas/raw/NERC_report/2025_Q4-Report.pdf`, Table 1, printed p.17) gives gas available capacity 3,173.18 MW over the same 23 plants, 3,173.18 / 10,947 = 0.2899, which reproduces the repository's "29.0%": the convention matches the corrected 2024 figure, not 0.295. 2026/Q1 (26.2%): the 2026/Q1 report is not in the repository, `gas_plant_availability.csv` holds only the 16 open-cycle plants (6,482 MW; capacity-weighted 26.4%), so 26.2% could not be reproduced and is unverified.

> **Headline availability.** NERC's headline 2024 plant availability factor, 37.43%, is the simple (unweighted) mean of the 28 plants' PAFs; the capacity-weighted figure is 4,853.69 / 13,625 = 35.62%. The Q4 2025 quarterly report's 39.64% is capacity-weighted (5,400.38 / 13,625), so the annual and quarterly reports differ in convention. Gas only, 2024: simple mean 32.97%, capacity-weighted 30.61%. The repository quotes neither headline number (searched `docs/`, `src/`, `scripts/` for 37.43, 35.62, 37.4%); it refers to NERC's "headline Availability Factor" in prose only (`src/scenarios.py` docstring, `docs/methodology_physical_layer.md`).

> **Direction of bias.** Raising the observed arm from 0.295 to 0.3061 raises the lower bound of availability by 3.8%; the `gas_availability` lever span (Question 1 row 1, 7.059 bn at 0.295) narrows by an amount not computed here. **Every stored result, finding and chapter figure that includes `gas_avail_observed` was run at 0.295 and is stale against the code until re-run** (epoch). The Monte Carlo parameter-uncertainty prior `U(0.295, 0.80)` was hardcoded in `scripts/29_run_parameter_uncertainty.py` and now reads `U(0.3061, 0.80)`.

> **What would resolve it.** Locating how 0.295 was computed, if it matters; re-running the arms that use `gas_avail_observed`. Not done in this task.

## L129 — The `conversion` lever family is compound in three fields; its 5.220 bn span is a conversion programme's worth, not the 0.46's

> **Status:** LIVE as a labelling condition (TASK 76 Part C.3, 2026-10-23). Nothing is wrong with the arms; the row must be labelled for what it is.

> **What.** `conv_low` / `conv_central` / `conv_high` (`src/scenarios.py` `conversion_scenarios()`) move **three fields together**: retrofit capital cost (850 / 600 / 450 USD/kW of OCGT nameplate), capacity uplift (0.40 / 0.50 / 0.50) and retrofit efficiency (0.42 / 0.46 / 0.50). The `conversion` family in `LEVER_FAMILIES` (`conv_none` to `conv_high`, 5.220 bn: `conv_high` 23,900,759,294.20 to `conv_none` 29,120,942,321.18, `results/lever_ranking/arms_6046f759/lever_ranking.csv`) therefore measures what a conversion programme is worth, moving extent, cost and efficiency together and including the step from no conversion to any, **not what the unsourced 0.46 is worth**. Converted capacity by tier is a decision variable, not an arm input.

> **Efficiency alone.** The new family `conversion_efficiency` (`conv_eff_low` / `conv_central` / `conv_eff_high`, capex 600 and uplift 0.50 fixed, efficiency 0.42 / 0.46 / 0.50) gives **1.134 bn (4.54% of central)**: 25,623,673,320.39 / 24,986,620,949.18 / 24,489,931,741.09 USD; converted capacity is identical in all three (571 + 3,215 = 3,786 MW); the plans differ from 2031; maximum build differences at 2054 are solar 1,496.59 MW, gas 312.89 MW, storage 1,251.55 MWh (`docs/task_log_76.md` C.3). At the blended-dispatch model that produced the central anchor (flag off).

> **How Chapter 4 labels it.** The conversion row is "a conversion programme (capex, uplift and efficiency together)", and the efficiency assumption's worth is the 1.134 bn family, not 5.220 bn.

> **Generation (TASK 86):** the hash-keyed result files this entry cites are blended dispatch (no `gas_vintage_dispatch=True` in the cited files' arm records). Checked: `6046f759` (blended). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142. [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L130 — "NGC range 800-860 MMscf/d" is cited as sourced and has no source

> **Status:** LIVE (TASK 77 Part A.1, 2026-10-23). The range, and the "triangulated three ways" claim built on it, are not supported by any document.

> **What.** The 2024 gas-to-power volume range "800-860 MMscf/d", attributed to "NGC" (Nigerian Gas Company), is called "sourced" in `data/gas/processed/gas_deliverability_cases.yaml:6-8` and `docs/chapter3_3_inputs.md:89, 96-99, 249-250`, and "reported" in `docs/merged_correction_plan_v2.md:100`, `docs/nirp_defence_sourced.md:19` and `scripts/29_run_parameter_uncertainty.py:65`; `docs/findings_frozen.md:2684` called the 0.277-0.298 sweep "DERIVED from" it (corrected, TASK 77 A.3). **No document, table or file holds it.** It first appears in commit `986f467` (the Phase 1 re-anchor, `git log -S"800-860"`) with no source. `docs/chapter3_3_inputs.md:252` already records "no NNPC or NGC citation anywhere in the yaml or build script". The "805 MMscf/d" inside it is 89.27 TWh_th converted back (805.49), so it is not a second observation (`docs/task_log_76.md` Part A).

> **Extent of the search that found no source.** `docs/*.md`, `src/`, `scripts/` and every `.yaml/.csv/.txt/.md` under `data/` for `800-860`, `800–860`, `800 to 860`; every cell of every `.xlsx` under `data/` (regular expression `800.{0,3}860|NGC|860 MM`: the only hits are "NGCC"/"NGCT" in the ATB workbooks and a carbon-tax sheet, none a volume); the extracted text of 32 PDFs (the 18 in `literature/`, the four NERC 2025 quarterly reports, ten under `data/cost/raw`) for `800-860` in any spacing and for volumes of the form `8xx MMscf`: no hit; `git log -S` on the string; and the nearest data on disk, `data/gas/processed/gas_to_power_suppy_2025.xlsx` (NUPRC, **January-July 2025** only, 780-887 MMscf/d by month, a URL-sourced series for another year). Not searched: documents not on disk (NNPC monthly reports, NGC publications).

> **Chapters** (not edited; for correction by the author): `docs/chapters/CHAPTER_3_assembled.md:1234-1241` ("triangulated three ways", "the sourced NGC range of 800 to 860 MMscf/d. Three independent routes -- NERC generation, NGC volumes and the efficiency identity -- agree to within about 1%"); `docs/chapters/Chapter1.md:22-29` ("a sourced industry range of 800 to 860. Three independent routes, being the regulator's generation data, the transmission company's gas volumes and the efficiency identity"; the "transmission company" is a third misattribution: NGC is not the transmission company); `docs/chapters/CHAPTER_3_assembled.bak.md:1233-1241` (backup copy). `docs/chapters/Abstract_1.md` and `CHAPTER_4_section_4_*.md`: no occurrence (extent: `grep` for `800`, `860`, `triangulat`, `Three independent`).

> **Direction of bias.** The anchor's size rests on one sourced quantity (NERC 2024 generation, 25.62 TWh_e) and one chosen efficiency (0.287); with the range unsourced there is no independent check on 89.27 for 2024. TASK 77 Part B rebuilds the chain on the 2025 NUPRC volumes, the only measurement on disk.

## L131 — The fuel layer is 2025 (when the chain moves) while demand, hydro and the generation share stay 2024: a declared base-year mismatch

> **Status:** LIVE (TASK 78 A.3, 2026-10-23). Declared, not resolved. Pre-existing: the demand base has always been 2024; moving the fuel layer to 2025 makes it visible.

> **What.** The model's physical layer carries NERC 2024 demand (37.09 TWh, `src/scenarios.py:28`), 2024 hydro (11.47 TWh) and the 2024 gas share, labelled as the 2025 start. The NUPRC volume that the TASK 77 rebuild would use is a 2025 series (January-July). Measured by TASK 77 (`docs/task_log_77.md` B.5): the model's 2025 gas output is 24.788 TWh_e under every chain tested; against 2024 observed gas generation (25.62) that is **-3.2%**; against H1 2025 annualised thermal generation (28.40 TWh_e) it is **-12.7%**. The 2025 backcast is insensitive to `gas_eta` and the anchor (it is set by demand, capacity and the converted stream), so it does not test the chain.

> **Not done.** Demand and hydro are not re-based; that is a separate decision outside this epoch. A chapter that reports the chain must say which year each layer is.

## L132 — The end-effect buffer is adequate under blended dispatch and not under vintage dispatch at the current horizon

> **Status:** LIVE (TASK 78 A.4, 2026-10-23). Cross-references L94. Diagnosis is TASK 78 Part D.4 (`docs/task_log_78.md`); no Chapter 4 figure to 2045 may be quoted on the vintage model before it is read.

> **What.** L94's buffer test (G1-3: the plan to 2045 unchanged within 1% when the horizon end is extended to 2050 and 2054) passed on the blended-dispatch model (TASK 76 suite 79 / 7 / 1, where G1-3 passes). On the merged vintage-dispatch model (flag default on, TASK 77 C.4) it fails: solar +1.48% (2050) and +1.52% (2054), gas +1.86% and +1.88% of cumulative to 2045, storage +0.33%, last differing year 2041; G1-3m's mechanism verdict is REFUTED (first divergent horizon 2050). The criterion is **not relaxed**.

> **Direction of bias.** Plans to 2045 on the vintage model are horizon-conditional by 1.5-1.9% in solar and gas.

## L133 — The NUPRC gas-to-power series is a hand-compiled spreadsheet with a dead URL as its only backing, and it has June and July swapped

> **Status:** LIVE (TASK 78 Part B, 2026-10-23). The chain is not set on it.

> **What.** `data/gas/processed/gas_to_power_suppy_2025.xlsx` (January-July 2025) is backed only by `data/gas/raw/nuprc_gas_to_power_2025.xlsx`, which holds one URL (`nuprc.gov.ng/nigerias-gas-production-hits-daily-average-of-7-59billion-scf-as-nuprc-releases-july-production/`) that returns 404. Two press reports of NUPRC's July status update (BusinessDay, 3 September 2025; Blueprint, 2 September 2025, the latter attributing it to NUPRC's Head of Media) give the same seven monthly gas-to-power figures **with June and July the other way round**: June 833.86 and July 862.86 MMscf/d (BusinessDay: July "up 3.48% from June"; 862.86 / 833.86 = 1.0348). **The sheet has June 862.86 and July 833.86: swapped.** The sheet's other five months agree with the press. A primary NUPRC document ("2025 Monthly Gas Data for Publication, December 2025", `nuprc.gov.ng/wp-content/uploads/2026/01/...pdf`, found by search) could not be fetched (404 from this environment), and I could not confirm it breaks out gas-to-power.

> **Consequence.** H1 volume on the press order is 153,044.86 MMscf (not 153,914.86: -870.00 MMscf, -0.565%); eta(H1 2025) at 1,036 Btu/scf is **0.3031** (not 0.3014); the annualised anchor 93.71 TWh_th (not 94.24). January-July totals differ by 29.00 MMscf (the swap changes the day weights 30/31).

> **Also unreconciled.** The sheet's columns do not reproduce the press year-to-date shares (sheet export 33.66% and domestic 28.78% of total production, against 35.88% and 27.8%); the sheet's July total production 250,880.54 MMscf is 8,092.9 MMscf/d against the press headline "7.59 BSCFD" for July (-6.2%); neither is explained (a flaring-adjusted 7.51 is not the headline either). So the sheet's total, export and domestic columns cannot be taken as NUPRC's published figures; only the gas-to-power column is corroborated, by press.

## L134 — Vintage-differentiated gas dispatch is the default model (TASK 74-77): what it is, where it lives, what it makes stale

> **Status:** RESOLVED as a declaration (TASK 78 Part E, 2026-10-23); the conditions below stand. Updates L123 (the model did not represent new-build thermal efficiency) and L120.

> **What.** `load_scenario(gas_vintage_dispatch=True)` (default; forced False under `new_gas_deliverability='exempt'`, which the builders refuse with the flag on). Capacity added in the model dispatches at `ccgt_eff_central` 0.5173 (`gas_new_efficiency`), pre-existing capacity at `gas_eta` 0.287; `m.gas_to_power` is the total thermal draw and `m.gas_to_power_new` the added-capacity part; generation = eta x draw + (gas_new_efficiency - eta) x added stream; emissions follow the thermal draw (`tCO2 = TWh_th x 1e6 x EF`), so the added stream emits less per MWh_e; converted capacity keeps its own stream. Three capacity rows: added stream within the total draw; added stream within capacity added in the model; **legacy stream within surviving legacy capacity** (the tightened row: `eta x (draw - added stream) <= (gas_capacity_mw - cumulative additions) x h x alpha`; converted nameplate is deliberately not netted, because it exceeds surviving legacy capacity late in the horizon, an overlap the model does not prevent). Both builders carry it (`optimize_model_sliced.py`; `optimize_model.py` since TASK 77 C.2). `scripts/01b_run_baseline_sliced.py --gas-vintage-dispatch {on,off}`; the effective value is in every runner's output key (`src/cli_arms.py`), so no old result shares a hash with a vintage result.

> **Collapse check (CLAUDE.md).** Flag on with `gas_new_efficiency = gas_eta`: decision block identical; objective within 1e-7 relative of 24,986,620,949.18249 (sliced; observed +7.98 USD, 3.2e-10); legacy-row slack positive at the real efficiency and >= -1e-6 TWh_e at `gas_eta`. Annual builder, flag off: LP file byte-identical to the pre-change builder; objective 28,876,674,784.69965 (the `conv_none` central arm), flag on at `gas_eta`: identical.

> **The legacy row is a guard, not an active constraint at the arms run** (TASK 76 C.2, corrected TASK 77): legacy stream positive in 27-167 of 180 (year, slice) rows depending on the arm, maximum utilisation 0.29-0.74, minimum slack 0.56-3.42 TWh_e. (TASK 76 reported the legacy stream as zero everywhere; that was a measurement error.)

> **What it makes stale.** **Every stored result, finding and chapter figure produced before TASK 74 is on blended dispatch** (the central anchor 24,986,620,949.18249 included; the vintage-model central anchor is 23,323,068,340.46 USD at the sliced builder, `docs/task_log_74.md` D.4, measured but not adopted as the quoted anchor). A chapter must say which dispatch each reported figure carries. The lever builder's tag derivation (`scripts/build_lever_ranking.py`, `src/scenarios.py` `BASELINE_ARM_TAG_FIELDS`) predates the flag key and will refuse vintage files until the epoch's re-run. The cap series (`00_build_emissions_cap.py`) reads baseline thermal input; it must be rebuilt on the epoch's baseline. Open under the flag: L132 (the plan to 2045 depends on the horizon end, 1.5-1.9% in solar and gas timing).

> **Generation (TASK 86):** not established. This entry quotes blended-generation or pre-epoch figures (24,986,620,949) without citing a hash-keyed result file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L135 — Stocks to 2045 are horizon-invariant under vintage dispatch; annual additions are horizon-sensitive (1.5-1.9% L1), by lumpiness near the threshold at which new gas becomes worth building

> **Status:** LIVE (TASK 79 Part B, 2026-10-23). Cross-references L94 (the original horizon-end finding, whose storage-rebuild mechanism is refuted for the vintage model) and L132 (the buffer finding this entry measures). Evidence: `docs/task_log_78.md` D.4.

> **What.** On the vintage-dispatch model at the central efficiency (0.5173): the **cumulative solar, gas and storage build to 2045 is identical across horizon ends 2045, 2050 and 2054 to 0.1 MW (MWh)** (observed differences at most 1.9e-4; solar 11,787.67, gas 3,825.30 MW, storage 24,697.69 MWh), asserted by the new invariant **G1-3s**. **Annual additions are not**: L1 differences of 1.48-1.52% (solar), 1.86-1.88% (gas), 0.33-0.34% (storage), a re-timing of 47.7 MW of solar and 25.5 MW of gas between 2035 and 2037 and smaller moves in 2039-2045; G1-3 and G1-3m (annual timing, 1% L1) are **kept as they are and fail**.

> **Mechanism (evidence, not proof).** The effect is **zero when the new-build efficiency equals `gas_eta`** (and under blended dispatch), **largest at 0.35-0.45** (gas L1 160.5% at 0.35 where the build is only just worth making, 4.3-9.5% at 0.40-0.45), 1.9% at 0.5173 and **near zero at 0.5445** (gas 0.00%, solar 0.38%). So the sensitivity is **lumpiness around the threshold at which new gas becomes worth building**: a discrete build decision whose payoff extends past a truncated horizon, present only when there is an efficiency differential. It is not the storage rebuild of L94: the first divergent horizon is 2050, before any in-horizon rebuild of a built storage vintage (earliest 2054).

> **Not answerable.** Whether a horizon beyond 2054 restores invariance: gas deliverability data stop at 2054, and a synthetic 0.5%-a-year extension to 2059 perturbed the blended control (gas L1 180.5%, which is exactly invariant to 2054), so the 2059 result is uninterpretable. It needs real 2055-2059 inputs and a check of every other horizon-bound series.

> **Reporting consequence:** see Chapter 4 condition C.5.

## L136 — The fuel chain's high arm rests on a parameter with no 2024 measurement and on press-reported 2025 figures, one of which was transcribed wrongly in the repository's own copy

> **Status:** LIVE (TASK 79 Part D, 2026-10-23). What the `fuel_chain` span measures; cross-references L130, L133, L131, L125.

> **What.** The two-field family (central `gas_eta` 0.287 with anchor 89.27 TWh_th; high 0.3031 with 93.71) measures the model's dependence on the **gas-to-power volume**, for which **2024 has no measurement** (the 805 MMscf/d is the anchor converted back, L130) and **2025 has regulator figures known only through two newspaper reports** of NUPRC's July update (no primary document, L133), against **a documented June/July transcription swap in the repository's own spreadsheet**, corrected in `data/gas/processed/gas_to_power_2025_corrected.csv` (the source `.xlsx` is untouched). Heating value (1,036 Btu/scf, assumed) and the gross-or-net basis of the generation figure (TASK 75) are not in the span. The 2025 pair also mixes years (2025 fuel, 2024 demand and generation share: L131).

> **Numbers.** `docs/task_log_79.md` Part D.3.

## L137 — E4's envelope-shadow check measured solver noise at its default step; the step, not the tolerance, was repaired

> **Status:** RESOLVED (TASK 80 Part A, 2026-10-23).

> **What.** `52_invariants.py` E4 compared the concessional-envelope dual (0.27062338) with a central finite difference at a relative step of 1e-4 and failed at 3.0e-4 against its 1e-4 tolerance. **This is not a tolerance widening: the 1e-4 criterion is unchanged.** At the 1e-4 step the envelope perturbation changes the objective by about 10,000 USD against a CBC noise floor of about 2,500 USD (signal-to-noise near four), so the old step measured noise; at a step of 1e-3 the forward and backward slopes agree with each other to 5e-6 and with the dual to under 3e-5 (`docs/task_log_79.md` C.2: no kink; TASK 78 D.2's and the TASK 79 brief's "kink" diagnosis was wrong). E4 now uses `_fd_central(..., rel=1e-3)` for that row. E3-gap follows E4 and passes. Suite after repair: 80 PASS, 8 FAIL, 0 SKIP.

## L138 — The lever generator does not admit the `new_gas_deliverability=exempt` arm of the vintage generation (TASK 81 Batch 1)

> **Status:** RESOLVED (TASK 82 A.1, 2026-10-23). `scripts/build_lever_ranking.py` now writes `gas_vintage_dispatch=False` for the exempt member (`arm_for`) and admits that token on an exempt tag only (`epoch_tokens_ok`). The family is ranked 15th of 15 at 0.2783 bn. **Residual condition, not a defect:** the exempt arm runs the single-stream dispatch (the builder raises if the flag is on under exempt, `src/optimize_model_sliced.py:672`, and the allowance is sized at `eta`, TASK 66), the constrained arm runs the vintage dispatch, so the span (constrained 23.323 bn, exempt 23.601 bn) mixes two dispatch models and exempt is dearer than constrained. It is not a clean measure of deliverability; state that wherever the row is reported. Evidence: `docs/task_log_82.md` A.1.

Under `exempt` the vintage flag is forced off (`cli_arms.full_arm_dict`), so the arm's stored tag carries `gas_vintage_dispatch=False`, and the generator's epoch token is `gas_vintage_dispatch=True`. The arm solved (`results/baseline_sliced/summary__58728c98.json`, objective 23,601,372,250.74, equal to the blended exempt value on the same arm) but is reported "declared but not ranked", so the `new_gas_deliverability` family is absent from `results/lever_ranking/arms_661b03c1/lever_ranking.md` (57 of 58 members admitted, 16 of 17 families ranked or reported). Not fixed (discovery closed, TASK 80): the omission is stated on the page and no reported figure is false. Fix when wanted: let `arm_for` write `gas_vintage_dispatch=False` for the exempt member and admit that token for that family only.

> **Generation (TASK 86):** the hash-keyed result files this entry cites are MIXED: some cited files are vintage, some blended; each is named where its figure is quoted. Checked: `58728c98` (blended); `661b03c1` (vintage). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142. [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L139 — Runners 06_, 07_ and 08_ compute the leakage diagnostic but their saved outputs do not carry it (TASK 81 Batch 3)

TASK 80 C.1 wired `leakage_flat(diag)` into the per-run result dicts of `06_run_gas1_shadow_benchmarks.py` (line 205) and `07_run_gas2_eaas_gas_relief.py` (lines 215, 316), and the shared matrix builder behind `08_`. The files these runners write are projections of those dicts (`gas_shadow_matrix.csv` builds its rows field by field, `gas1_summary.json` and `gas2_summary.json` are keyed by case with a fixed field list, `regime_ndc_matrix.csv` has no leakage column), so no leakage share or flag is recoverable from `results/gas1/arms_f30c5990`, `results/gas2/arms_f30c5990` or `results/gas3/arms_b7144908`; the logs print nothing (`LEAKAGE FLAG` appears only in the pol1 and fin3 logs). The capped rows in those runs (ndc3_unconditional in 06_; the four NDC scenarios in 07_ and 08_) therefore have no stored leakage figure. Not fixed (discovery closed, TASK 80): no reported figure is false; the Batch 3 C.3 table says "not stored" for these rows. Fix when wanted: add the `leakage_share`, `leakage_flagged` fields to each row projection. Extent of search: the saved CSV and JSON files of the three results directories (column and key names containing "leak"), the three driver logs, and the row-building code at `scripts/06_run_gas1_shadow_benchmarks.py:296-318`; 07_ and 08_ code paths were checked by their output files only.

> **Generation (TASK 86):** the hash-keyed result files this entry cites are vintage (`gas_vintage_dispatch=True`). Checked: `b7144908` (vintage); `f30c5990` (vintage). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142.

## L140 — `27_run_hydro_minflow_sensitivity.py` compares its central arm to the blended anchor (TASK 81 Batch 4)

On the vintage model the runner prints `WARNING: central arm (hydro_min_flow_fraction=0.5) objective 23,323,068,340.46 does NOT reproduce the reference 24,986,620,949.18`. The reference is the blended central (`summary__6046f759.json`); the runner's central arm equals the vintage central (`661b03c1`) to the cent, so the warning is the stale constant, not a failure. Verdict unchanged: IMMATERIAL (objective band 0.150% of central against a 0.5% threshold; storage at 2045 band 0.022% against 2.0%; `results/hydro_minflow/arms_e1ca2888/`). Not fixed (discovery closed): no reported figure is false. Also stale, same cause: the Sobol summary's `note_on_parameter_1` quotes F-5's "~8.60 bn" for the gas-availability lever (`results/parameter_uncertainty/arms_09bd6dbc/summary_n256_seed42.json`); the vintage span is 8.4896 bn (`results/lever_ranking/arms_661b03c1/`).

> **Generation (TASK 86):** the hash-keyed result files this entry cites are MIXED: some cited files are vintage, some blended; each is named where its figure is quoted. Checked: `09bd6dbc` (vintage); `6046f759` (blended); `661b03c1` (vintage); `e1ca2888` (vintage). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142. [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L141 — The results-inventory epoch rules end at 2026-10-10, so every file written by the epoch-1 and vintage re-runs is classed UNCLASSIFIABLE (TASK 81 E.1)

> **Status:** RESOLVED with named residuals (TASK 82 A.2, 2026-10-23). Rules L6, C4, C5 added to `scripts/build_results_inventory.py` and S3 to `scripts/build_results_manifest.py`; counts and the residual files are in `docs/task_log_82.md` A.2.

`scripts/build_results_inventory.py` rule U1 classes any file with mtime after 2026-10-10 as UNCLASSIFIABLE; the rules predate both the 2026-10-19/20 epoch-1 set and this task's vintage set. Rebuilt on 2026-10-23: 1,705 rows (LEGACY 1,039; UNCLASSIFIABLE 488; CURRENT 178; the prior file had 1,221 rows, 203 CURRENT, 12 UNCLASSIFIABLE). Of the 473 files with mtime on or after 2026-10-19, 449 are UNCLASSIFIABLE by U1 and the other 24 are LEGACY (L1, L3); none is CURRENT. `scripts/build_results_manifest.py` admits only CURRENT rows and their siblings, so `docs/results_manifest.json` fell from 422 to 328 entries and lists none of the vintage generation. Neither the old nor the new manifest records the vintage set (the old one predates it; the new one excludes it); I did not check whether any old entry's md5 was overwritten by the re-runs. Not fixed (discovery closed): no reported figure is false; the vintage results are identified by the epoch token in L142 and by `docs/task_log_81_*.md`. Fix when wanted: add rules for the 10-19/20 and 10-23 generations keyed on the arm tags (`capital_window_case=arrears`, `gas_vintage_dispatch=True`), then rebuild.

## L143 — Vintage-generation files still outside the manifest after TASK 82 A.2 (TASK 82)

> **Status:** RESOLVED (TASK 83 B.2, 2026-10-23). rel1: retrofitted `ARMS.json` sidecars (`scripts/retrofit_rel1_sidecars.py`, labelled as retrofitted; rule C5/S3). vss_bound and the stochastic draws: rule S4 (the draws meta carries the vintage token; the vss file is admitted through its stage-2 draws file). Everything written 2026-10-23 or later and not in the manifest is now blended-generation (LEGACY).

`results/rel1/{annual,total}/gas_avail_{nameplate,observed}/` (four `rel1_summary.json` plus 12 CSVs, written 2026-10-23 by the vintage re-run of `09_run_rel1_feasibility.py`) carry no `arm_tag` and have no `ARMS.json` sidecar, so no rule can show they are vintage; they stay UNCLASSIFIABLE (U1). `results/vss_bound/vss_bound__9e2eda70.json` and `results/stochastic/draws__6b686491*` also carry no vintage token (the draws are model-independent inputs). Not fixed (discovery closed). Fix when wanted: give 09_ an `ARMS.json`.

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L142 — The repository holds two result generations: blended dispatch and vintage dispatch (TASK 81 E.3)

- **Blended** (epoch 1, 2026-10-19/20): `gas_vintage_dispatch` absent from the arm tag, or `False`. Central objective 24,986,620,949.18 USD, hash `6046f759`; reproducible with `--gas-vintage-dispatch off` (`dec1ea9c`).
- **Vintage** (this task, 2026-10-23): arm tag carries **`gas_vintage_dispatch=True`**. Central objective 23,323,068,340.46 USD, hash `661b03c1`. The epoch-filter token is `gas_vintage_dispatch=True`, as `capital_window_case=arrears` and `storage_capex_case=declining_nrel` were epoch 1's; `scripts/build_lever_ranking.py` admits on it (`EPOCH_TOKENS[4]`).
- Both generations coexist in `results/` under different hash8 directories; the blended files were not overwritten except where a runner writes fixed names (the central `661b03c1` and 01b twins were re-solved with identical code). Exempt-deliverability arms are not vintage by construction (flag forced off, tag `gas_vintage_dispatch=False`; L138).
- Not re-run, therefore blended-generation only: `22_`, `23_`, `51_`, `53_`, `54_`, `56_`, `58_`, `12_` and `14_` (suspended stubs, never produced results in either generation), and `09b` (retired in `efa433e`).
- Findings in `docs/findings_frozen.md` still quote blended values; they are not re-derived here (TASK 81 rule).

> **Generation (TASK 86):** not established. The result files this entry cites are fixed-name, directory or pattern citations (overwritten by later runs), absent, or carry no arm record, so the entry cannot be assigned to blended or vintage dispatch from a file (L142). [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

## L144 — G10-6-sobol reads the stale unkeyed Sobol files and fails on the draws-file row count, not on the indices (TASK 82 C.2)

`scripts/52_invariants.py` G10-6-sobol opens `results/parameter_uncertainty/summary_n256_seed42.json` and `draws_n256_seed42.jsonl` (both unkeyed, 2026-10-03). The draws file has 5,376 rows (three appended runs, L77) against N(d+2) = 1,792; the index tests pass (sum S1 1.0013 inside 1 +/- 0.05; min(ST-S1) -0.0021 above -0.02). The vintage run is at `results/parameter_uncertainty/arms_09bd6dbc/` (1,792 rows; sum S1 1.0001 and 1.0116, min(ST-S1) -0.0025 and -0.0019). TASK 82's brief described the failure as a sampling-error question; it is a stale-file question. Not fixed (assertion not changed, discovery closed). Fix when wanted: point the assertion at the keyed directory.

> **Generation (TASK 86):** the hash-keyed result files this entry cites are vintage (`gas_vintage_dispatch=True`). Checked: `09bd6dbc` (vintage). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142.

## L145 — Conditions on the capped lever ranking (TASK 82 Part B)

(1) **Leakage:** 53 of 58 arms (NDC 3.0 conditional) and 57 of 58 (unconditional) are flagged by the runner's own leakage guard (share above 0.10), the central arm included (0.1049 conditional); the capped objectives are the cost of plans that guard calls not NDC-compliant. (2) **Exempt row:** the `new_gas_deliverability` span compares a vintage-dispatch arm with a single-stream arm; exempt equals the blended-dispatch constrained objective to the cent at both caps, so the span (9.7426 bn conditional, 17.5400 unconditional) is the dispatch-model difference (L138). (3) **Envelope:** capped solves carry `capital_case` tight (unconditional) or moderate (conditional); `capped - uncapped` is cap plus envelope. (4) **Plan comparison:** 02_ stores no gas or storage additions; capped "plan" is solar additions, storage 2054 and gas thermal draw, compared with a tolerance (1e-3 MW/MWh, 1e-6 TWh_th), a choice made in TASK 82, not the baseline's exact rule. (5) The infeasible-arm path of the generator was not exercised (no arm was infeasible).

## L146 — F-43's turning point and the tight-capital branch differ on the vintage frontier (TASK 82 C.1; not re-derived)

`results/pareto_frontier/arms_1d68be49/pareto_raw.csv` (vintage): `public_only_moderate` minimum total emissions at cap_fraction 0.625 (322.1 Mt), not 0.8125; `public_only_tight` minimum 420.6 Mt at 0.8125 against 480.9 Mt at cap 1.0, i.e. a falling branch where F-43 says none. All four arms have `endpoints_degenerate` True (tight end unlocated: feasible down to 0.01). For the finding re-derivation task.

> **Generation (TASK 86):** the hash-keyed result files this entry cites are vintage (`gas_vintage_dispatch=True`). Checked: `1d68be49` (vintage). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142.

## L147 — Vintage-model observations recorded by TASK 84, not investigated

(1) **Retirement ordering:** `gas_retire_vintage_40` (22.6851 bn) is cheaper than `gas_retire_vintage_50` (22.8144 bn) on the vintage model (F-20 REVERSED); cause not examined. (2) **Shared alternative vertex:** `cr_comm_high`, `om_high` and `gas_capex_low` return the same 2054 plan (-143.91 MW solar, -120.35 MWh storage, +30.09 MW gas) against central; whether it is a degenerate optimum was not examined (F-3). (3) **Inversion count:** the TASK 55 'zero inversions' of F-38 used a definition that is not recorded; under the stated pairwise definition the vintage file gives 2 / 2 / 0 and the blended file 0 / 3 / 0 (F-38). (4) **Conversion no longer lowers the capital programme** (7.5600 bn without, 7.7003 bn with; F-6) and `conv_low` converts tier A only (F-7). (5) **F-33:** all 8 reliability cells are floor-bound at 0.001 on the vintage model, so the horizon-against-annual comparison cannot be made. (6) **F-39/F-40 window:** the carbon-price sweep stores no annual total emissions, so its reporting-window figures cannot be computed without a re-run. (7) The 06_ gas-shadow runner hard-codes `capital_case=moderate`; F-30's cap-alone figures come from the 02_ NDC 3.0 runs. Evidence: `docs/task_log_84_part*.md`.

## L148 — `53_run_horizon_rebuild_isolation.py`'s `anchor_check` compares the 2054, 15-year cell with the blended anchor (TASK 91, recorded TASK 92)

**Status:** LIVE. On the vintage run the script prints `anchor_check {'npv_this_run': 23323068340.46025, 'npv_anchor': 24986620949.18249, 'npv_bit_identical': False}`. `ANCHOR_PATH` is hard-coded to `results/baseline_sliced/summary__6046f759.json` (blended central, 24,986,620,949.18249). The vintage central is `results/baseline_sliced/summary__661b03c1.json`, **23,323,068,340.46025**; the vintage 2054, 15-year cell (`results/horizon_rebuild/summary__e4afbf07.json`) equals it bit for bit and reproduces its 2025-2045 solar, gas and storage addition series with a maximum absolute difference of 0.0 (TASK 91 A.4). So `npv_bit_identical: False` is the stale constant, not a failure of reproduction. The summary's `anchor_check` field in `e4afbf07` therefore records a false negative. Not changed (discovery closed): no reported figure rests on the field.

> **Generation (TASK 92):** MIXED. The hash-keyed files cited are `e4afbf07` (vintage), `661b03c1` (vintage) and `6046f759` (blended).

## L149 — `53_report_horizon_rebuild.py` mixes blended and vintage cells and overwrites `docs/task_log_52_horizon_rebuild_results.md` (TASK 91, recorded TASK 92)

**Status:** LIVE. The reporter reads every `summary__*.json` in `results/horizon_rebuild/` with no key on the result generation. With the blended set (`3338302f`, `da437cc1`, `ced5ab1b`, `0f71fa22`, `465aaeae`, `c057aa8f`, `ae79658d`, `b17f4558`) and the vintage set (`edc2e7dc`, `d6a1d105`, `02d3bc2c`, `c8690670`, `8ff6b1c1`, `e4afbf07`, `ddc60cd9`, `d39e722b`) side by side it tabulates the two together, and it writes its table over the tracked file `docs/task_log_52_horizon_rebuild_results.md`. **The reporter must not be run against the vintage set until it is keyed by generation.** It was run once in TASK 91 to inspect the vintage cells; the overwritten file was restored with `git checkout -- docs/task_log_52_horizon_rebuild_results.md` and is unchanged from its committed (blended) state. The vintage comparison in TASK 91 A.2 was computed separately from the eight summaries (`docs/task_log_91.md`). Not changed (discovery closed): no reported figure rests on the reporter's mixed table.

> **Generation (TASK 92):** MIXED. Both sets of hash-keyed files are named above.

## L150 — `leakage_diagnostic` computes the out-of-boundary share and stores it as "leakage" (TASK 94 Part A, recorded TASK 95)

**Status:** LIVE. `leakage_diagnostic` (`src/optimize_model.py:1883-1912`) computes `leakage_share = selfgen / total` (the out-of-boundary share of cumulative physical emissions, a **level**), `leakage_tco2 = total - capped` and `flagged = share > 0.10`, and stores them under the names `leakage_share`, `leakage_tco2` and `leakage_flagged`; its docstring calls the quantity "carbon leakage". Section 3.5.6 defines that quantity as a share, "not a displacement", and defines **leakage** as the increment of the share over the same configuration solved without the cap. The code contains no uncapped counterfactual; the increment exists only where a paired no-cap run is made and subtracted afterwards. Two further quirks: (1) equation (3.42) is **per year** ($\lambda_t = E^{self}_t/E_t$) while the code is **cumulative over the horizon**; (2) `leakage_tco2` is `total - capped`, which equals the self-generation emissions only under the default grid-only boundary: with `selfgen_in_emissions_cap=True` capped = total, so `leakage_tco2` is **zero while `leakage_share` stays positive**. **Chapter 4 labels what it reports** (the out-of-boundary share, or the increment where stated); the fields are **not renamed** (a rename touches every stored result and runner; discovery is closed).
*Where the fields are carried.* Code that writes them: `src/optimize_model.py`, `src/optimize_experiments.py` (`leakage_flat`), `scripts/01_run_baseline.py`, `01b_run_baseline_sliced.py`, `02_run_ndc_caps.py`, `03_run_ndc_eaas.py`, `05_run_fin3_capital_sweep.py`, `15_run_pol1_ndc_comparison.py`, `17_run_monte_carlo.py`, `18_run_pareto_frontier.py`, `24_run_leakage_arm.py`, `25_run_obligation_trajectory.py`, `99c_run_genset_arm_sensitivity.py`; code that reads them: `scripts/build_lever_ranking.py`, `batch2_report.py`, `batch3_report.py`, `task84_entries.py`, `task84_pareto.py`, and `src/pareto_plot.py` (which computes grid emissions as `(1 - leakage_share) * cum_emissions` and so relies on the field being the level). Output files carrying any of the three names (counts of files under `results/`, searched by text in csv, json and md): `ndc/` 178; `ndc_eaas/` 8; `fin3/` 8 (including the vintage `arms_1d68be49` and `arms_7f91f752`); `f3/` 8; `rel1/` 4; `lever_ranking_capped/` 4 (`ndc3_conditional`, `ndc3_unconditional` and the two `_noenvelope` directories, `capped_arms.csv` and `lever_ranking_summary.json`); `genset_arm_sensitivity/` 4; `pol1/` 3; `pareto_frontier/` 3; `monte_carlo/` 3; `_attrib/` 3; `fin3_gate/` 2; `fin2/` 2; `emissions_baseline_agreement/` 2; the dated snapshots `_pre_rerun_20260925_1221/fin3/`, `_pre_bstar_20260920_1132/fin3/`, `_pre_L60_20260926_1158/fin3/` 2 each; and one each in `obligation_trajectory/`, `leakage_arm/`, `fin3_uncapped/`, `eaas/`, `dem2/` and `baseline/`.
*Measured on the vintage generation (TASK 95 A): increment = level.* FIN-3 no-cap run `results/fin3/arms_7f91f752` (`gas_vintage_dispatch=True`, `emissions_cap=none`, 13 cells): the uncapped out-of-boundary share is 0.000% in all 13 cells, including tight at 35% collection (capped 38.953%). In `lever_ranking_capped`, all 42 distinct uncapped counterpart runs (`results/baseline_sliced`) have zero cumulative self-generation (41 vintage; the 42nd, `58728c98`, is the `exempt` arm, which cannot run vintage dispatch). So on the vintage generation the share and its increment coincide in every cell and arm measured; this is a measurement, not a property of the code.

> **Generation (TASK 95):** vintage for the measurements above (`7f91f752`, `1d68be49`, `3fbc75ac`, `ae31b29c`, `4ca4832d`, `60585e1f`, `661b03c1`); the older directories listed as carrying the fields are blended or unclassified and are not assessed.

## L151 — The quantity Chapter 1's second question names, the value of determining each vintage's concessional-commercial division within the model, has never been measured; F-38 measures the public/private division (TASK 96 Part A, recorded TASK 97)

**Status:** LIVE. (1) **The question.** Chapter 1 §1.4, Question 2 (`Chapter1.md:247-255`) asks what is changed "when the division of each vintage's capital between concessional and commercial sources is determined within the model rather than supplied to it", and excludes the public/private division explicitly: "It is not asked of the division between publicly and privately financed solar, which lies on a flat optimal face and is not identified by the formulation (§3.4.4)" (lines 253-255). (2) **What F-38 measures.** F-38 (and the thirty-six-cell separation, `results/dem2/arms_1fa34539/`) differences the **public-only** and **service-financed** arms (`scripts/13_run_dem2_growth_gas_interaction.py:316-364`, `FINANCING_CONFIGS` :123-146) at a fixed `moderate` capital tier, with `capital_treatment_case=f1_tranche` and `concessional_envelope_case=env_dares` in both arms, i.e. **with the concessional-commercial split determined within the model in both**. It therefore measures the value of moving capital off the public account, as a function of whether the public-capital budget binds (budget dual zero: value at most ⟦−0.61⟧ USD in 18 cells; positive: never below ⟦653,600,175⟧ USD in 18 cells). It does not vary the treatment of the split and cannot measure the value of determining it. (3) **The measurement does not exist.** No arm fixes a vintage's concessional share (`concessional_share` occurs only in `blended_finance_scenarios()`, `src/scenarios.py:98-160` and `:3515`, where it feeds the EaaS bankability WACC, not the capital-charge split), and no paired run exists comparing a supplied split with an endogenous one. **Extent of the search (TASK 96 A.4):** `results/` searched by text over every `ARMS.json`, `INDEX.md` and summary or arm-tag JSON for `capital_treatment_case=pre_f1_overnight` (0 hash-keyed hits; the string occurs only in logs `results/anchor_regression.log`, `fin3_L60.log`, `capital_frontier_run.log`, `capital_frontier_postbstar.log`, `baseline_regen_L60.log`, `_sdr_sweep_v2/` and `_sdr_sweep_v3/*.log`, and one blended-era snapshot under `results/_pre_L60_20260926_1158/baseline_sliced/`); `pre_f1_overnight` is in any case a different objective form (overnight capital with salvage) and `load_scenario` refuses it with conversion available (`src/scenarios.py:3010`); `src/scenarios.py`, `src/cli_arms.py` and `src/optimize_model_sliced.py` searched for any arm fixing the concessional share (none); `docs/*.md` searched for "value of determining", "determined within", "split supplied", "endogenous split" (hits only in the figure schedule, the TASK 95 log and Chapter 1's question itself). By text search of files; no scanned documents were involved. (4) **What is measured instead, and what each quantity is.** The **capital-recovery family** (`capital_recovery`, arms `cr_central`, `cr_ida21`, `cr_comm_low`, `cr_comm_high`): span ⟦2.675⟧ billion USD (`results/lever_ranking/arms_661b03c1/lever_ranking.csv`): the value of the **rates and tenors** at which concessional and commercial capital are priced. The **concessional-envelope family** (`concessional_envelope`, arms `env_none` to `env_10x`): span ⟦1.612⟧ billion USD: the value of the **amount of concessional capital available**. **F-38's conditionality threshold**: the value of the **service-financed channel relative to public-only financing**, switching on where the public budget dual becomes positive. None of the three is the value of determining the split. The central solution draws its whole 750 million USD envelope in a single vintage (2035, concessional share 1.0 of that vintage's solar and storage capital; `summary__661b03c1.json::concessional_share_by_vintage`), which is what the endogenous determination does at the central arm; what a supplied division would have cost has not been solved.

> **TASK 98 update (2026-10-29).** Point (3) is no longer true: the paired measurement now exists. `src/supplied_division.py` (scenario key `conc_share_rule`, both builders, `scripts/01b_run_baseline_sliced.py --conc-share-rule`, invariant G12-1) supplies the division; at the central configuration (`capital_case=unconstrained`, `env_dares`) supplying a constant share that exhausts the envelope (`split_prorata`, share 0.02797) raises the objective by 135,925,768.96 USD (0.5828%) and supplying nothing concessional raises it by 221,870,588 USD (0.9513%); handing the model its own realised division reproduces the endogenous solution (8.86 USD, 3.8e-10 relative, decision block identical). Results: `results/supplied_division/`, table `docs/task_log_98_results.csv`, report `docs/task_log_98.md`. What remains unmeasured: the dispersion of cost across realisations that Question 2 also asks for, and any cell with a binding public capital budget (the supplied-division runner solves only `capital_case=unconstrained`, where no budget exists). The scope-gap statement (2) stands: F-38 still measures the public/private division.

> **Generation (TASK 97):** vintage for every file named with a hash (`1fa34539`, `661b03c1`); the older log files and the `_pre_L60` snapshot are blended or pre-epoch and are cited only as the extent of the search.

## Reporting conditions (Chapter 4)

> Not defects. Conditions that Chapter 4 must carry wherever the named result is reported, recorded in TASK 70 Part C (2026-10-23) so they are not rediscovered at writing time. Each points to the entry that holds the evidence.

> **C.1 Question 1 row 1 is an asymmetric span.** The `gas_availability` row (7.059 bn, rank 1 under every ordering in `results/lever_ranking/arms_6046f759/lever_ranking.csv`) runs from `gas_avail_nameplate` (0.80, ASSUMED, corroborated by a US-to-Nigeria transfer rather than measured; objective 24.987 bn, the central arm) to `gas_avail_observed` (0.295 when the stored results were run, 0.3061 in the code since TASK 73 and register L128, a NERC fleet-wide measurement, not the sixteen-plant open-cycle file; 32.046 bn at 0.295). It is the distance between an assumed upper endpoint and a measured lower one, and the epoch does not change that. State the asymmetry beside the row. Evidence: L68.

> **C.2 F-2's magnitude is an upper bound** on what a two-tranche world implies (L23 middle: concessional finance is inframarginal, and a guaranteed tranche between concessional and commercial would narrow the gap across which the envelope is measured). The post-epoch concessional-envelope span is 2.147 bn (`env_10x` to `env_none`, `lever_ranking.csv`); the pre-epoch 2.46 bn is superseded. State it wherever F-2 is reported. Evidence: L23.

> **C.3 Post-2035 capped results carry two constructions**, the apportionment rule and the cap shape. Ten of the twenty-one reported years and the entire buffer carry both. State both wherever a post-2035 capped result is reported. Evidence: L24.

> **C.4 No common cap for Questions 1 and 3.** The NDC 3.0 decision (L24, TASK 69 A.4) does not put Question 1's capped ranking and Question 3's leakage results on one cap: F-21 is the flat-hold against decline pair and F-43 uses flat fractions of baseline; only F-31 and the `02_`/`08_` NDC arms sit on an NDC 3.0 cap. Correction to the TASK 69 brief, recorded against L24; the NDC 3.0 decision itself stands.

> **C.5 Annual capacity-addition series carry a horizon-sensitivity bound; stocks do not (TASK 79 B.4, L135, L132).** Under vintage dispatch, **cumulative stocks of solar, gas and storage at 2045 and 2054 are reportable as results** (invariant to the horizon end to 0.1 MW at the central arm, G1-3s). **Any annual addition series (solar, gas, storage MW or MWh per year) must carry the horizon-sensitivity bound wherever it appears**: 1.5-1.9% L1 in solar and gas, 0.3% in storage at the central efficiency, with single-year re-timings of up to 48 MW; at other new-build efficiencies the bound is larger and not characterised. The sensitivity is declared, not removed.

> **Generation (TASK 86):** the hash-keyed result files this entry cites are blended dispatch (no `gas_vintage_dispatch=True` in the cited files' arm records). Checked: `6046f759` (blended). Fixed-name and pattern citations in the entry are not assessed (a fixed name is overwritten by later runs, L142). Tokens: L142. [VERIFY: result-derived figures in this entry are not confirmed against a vintage file.]

