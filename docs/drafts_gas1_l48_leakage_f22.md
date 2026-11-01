# DRAFTS — GAS-1, L48, leakage finding, F-22 update

Drafted 2026-09-20 from the post-B\* run set. **Figures marked `[MEDIAN
PENDING]` need the re-grouped distribution query** (the first pass
pooled `ndc_scenario`, so its medians are artefacts). Everything else
is from the run output as printed.

---

## 1. FINDING — GAS-1 (next F-number)

> ### F-<n> — The gas shadow price collapses to zero under an NDC cap,
> confirming F-9 from the dual side; and the case for domestic gas
> allocation is insensitive to the export benchmark
>
> **Source.** `06_run_gas1_shadow_benchmarks.py`, run 2026-09-20 at
> `conv_none` (annual builder), `capital_case="moderate"` hardcoded
> (= 0.85 × B\*), NDC-3 unconditional caps. Eight solves, 0 failures.
> Benchmarks from `src/gas_benchmarks.py`; sourcing in
> `docs/sourcing_pack_01_rates.md`.
>
> **Result 1 — the cap binds before gas deliverability does.**
>
> | NDC scenario | gas case | mean shadow (USD/TWh_th) | USD/MMBtu | binding years |
> |---|---|---|---|---|
> | no policy | baseline | 28,124,906 | **8.24** | 29/30 |
> | no policy | upside | 26,629,441 | 7.80 | 29/30 |
> | no policy | downside | 38,533,131 | 11.29 | 29/30 |
> | no policy | shock_recovery | 28,129,433 | 8.24 | 29/30 |
> | NDC-3 | baseline | **0** | **0** | **0/30** |
> | NDC-3 | upside | **0** | **0** | **0/30** |
> | NDC-3 | downside | 10,316,417 | 3.02 | 18/30 |
> | NDC-3 | shock_recovery | **0** | **0** | **0/30** |
>
> Under an NDC cap the gas shadow price is **exactly zero** and binding
> years go from 29/30 to 0/30: the emissions cap binds first and gas
> deliverability goes slack. Only `downside` retains a positive shadow
> — the crossover where gas supply is scarce enough to bind before the
> cap does.
>
> **This independently confirms F-9 from the dual side.** F-9
> established the same conclusion by comparing objectives across
> deliverability arms; GAS-1 shows it from the dual of the
> deliverability constraint. Two methods, two runners, one conclusion.
>
> **Result 2 — the distribution, not the mean.** The uncapped
> distribution is heavily right-skewed: median `[MEDIAN PENDING]`
> against a mean of 8.24 USD/MMBtu, with a maximum of 29.79.
> **The mean must not be quoted alone.**
>
> **Result 3 — the conclusion is insensitive to the export benchmark.**
> Exceedance shares for `baseline_no_policy`:
>
> | benchmark | USD/MMBtu | years exceeding |
> |---|---|---|
> | `opp_7` | 7.00 | 40.0% |
> | LNG netback, DES central | 6.49 | 40.0% |
> | LNG netback, HUB central | 5.79 | 40.0% |
> | `lng_export_10` (delivered) | 10.05 | 36.7% |
>
> Three thresholds spanning 5.79–7.00 return **identical** counts, and
> the delivered price differs by one year in thirty. The distribution is
> bimodal — years sit near zero or well above 7 — so **the unsourced
> delivered LNG price does not propagate into the finding.** The
> netback correction is therefore a correctness fix, not a result
> driver, and the conclusion is robust across the whole plausible
> benchmark range.
>
> **Scope conditions.** Annual builder (`conv_none`): conversion, the
> second-largest lever, is not representable and storage is not
> dispatched. `capital_case` is hardcoded to `moderate` inside
> `run_no_policy_gas_matrix`, so every figure here sits under a binding
> public capital envelope at 0.85 × B\*. The mean fell from 35.34M
> (pre-B\*) to 28.12M on this run **because** B\* rose: a looser
> envelope buys more solar and reduces gas scarcity rent. See L<GAS-1
> entry>.

---

## 2. REGISTER — GAS-1 (next L-number)

> ## L<n> — GAS-1's benchmark comparison: five declared limitations
>
> **Status:** LIVE. GAS-1 is reportable with these declared.
>
> **1. Asymmetric truncation, against power.** The shadow price is
> `d(system cost)/d(gas delivered)` — marginal system cost *avoided*.
> Demand is exogenous and inelastic: every MWh is served regardless, by
> grid or by diesel at VoLL. **The model therefore cannot value
> electricity that would not otherwise exist** — suppressed industrial
> output, productivity, firms that never open. The export benchmark is
> marginal *revenue*, a complete measure of its own side.
> **Consequence: the exceedance share is a LOWER BOUND on the case for
> domestic allocation, not a complete comparison.** Stating it that way
> is stronger than defending the comparison as complete.
>
> **2. Benchmark provenance — mixed, all declared.** `dom_power_2p4`
> and `dom_commercial_2p9` are SOURCED to the NMDPRA 2024 Domestic Base
> Price ($2.42 power, $2.92 commercial, rounded); both are stale-high
> since the regulator cut to $2.13/$2.63 in 2025 and $2.18/$2.68 in
> 2026, so the bias is conservative. `dom_high_3p3` is ASSUMED — it
> sits above the regulated industry ceiling and matches no tier.
> `opp_3/5/7` are ASSUMED. The delivered LNG anchor (10.05) is SOURCED
> (interim) to a single month's Japan landed spot and should be
> superseded by a JKM or TTF assessment over a stated window. See
> `docs/sourcing_pack_01_rates.md`.
>
> **3. Dimensional inconsistency — CORRECTED.** The shadow price is a
> marginal value at the domestic delivery point; `lng_export_10` is a
> *delivered* price. Comparing them overstated the export side by the
> full liquefaction and shipping cost. Netbacks now derived
> (`src/gas_benchmarks.py`), basis-explicit (DES: regas not deducted;
> HUB: deducted), from sourced liquefaction (2.86, band 1.0–3.5, GECF
> 2026) and shipping (0.70, Nigeria→NW Europe, World Bank).
> **Measured effect: one year in thirty** (36.7% → 40.0%). The
> correction matters for correctness, not for the conclusion.
>
> **4. Static benchmarks over thirty years.** Every benchmark is a
> single scalar applied unchanged from 2025 to 2054, against a shadow
> price that moves year by year. No escalation, no discounting in the
> comparison.
>
> **5. Configuration scope condition.** `run_no_policy_gas_matrix`
> hardcodes `capital_case="moderate"`, `demand_level_case="served"`,
> `demand_case="organic_central"` and `carbon_case="no_policy"` inside
> the function rather than threading them from the CLI. GAS-1 therefore
> runs under a binding public capital envelope at 0.85 × B\*, unlike
> frozen central. **This is not inert:** when B\* rose from 9.104 bn to
> 10.273 bn the uncapped mean fell from 35.34M to 28.12M USD/TWh_th
> (−20%), purely through the envelope.
>
> **Defect found in the same run, separately declared:** the binding-
> years denominator is hardcoded twice — `/30` at
> `06_run_gas1_shadow_benchmarks.py:243` (no-policy path) and `/21` at
> `:259` (NDC path). `gas_shadow_by_year.csv` confirms **30 years
> (2025–2054) for every case**, so `/21` is stale. NDC `downside`
> binding is **18/30 = 60%**, not 18/21 = 85.7%. Not yet fixed.

---

## 3. REGISTER — FIN-3's bottleneck test (next L-number)

> ## L<n> — FIN-3's "bottleneck" test keys on a quantity that cannot
> move, so its printed conclusion is unsupported
>
> **Status:** LIVE. The printed interpretation must not be cited.
>
> **What.** `05_run_fin3_capital_sweep.py` reports a "bottleneck
> disappears at" capital case and prints, at every collection rate:
> *"Public capital sufficient at this collection rate. EaaS is
> CONVENIENT but not strictly necessary."* The test keys on **unserved
> energy**, which is `0.000 TWh` at all fifteen solves — because the LP
> has no hard service constraint (unserved is priced at VoLL, not
> forbidden; `28_run_capital_frontier.py`'s docstring states the same
> and its sweep finds no infeasible point down to a zero envelope).
>
> **Consequence.** The test cannot fire. It is not finding sufficiency;
> it is structurally incapable of finding insufficiency, and it returns
> `tight` — the loosest possible verdict — for every arm.
>
> **It contradicts the duals in its own output.** The same table
> reports budget shadow prices of 6.39 / 2.87 / 0.40 (no_friction) and
> **7.82 / 4.56 / 3.92 / 1.31** (current). A dual of 7.82 says one more
> dollar of public capital would cut system NPV cost by $7.82. That is
> capital scarcity, reported alongside a verdict of sufficiency.
>
> **Direction of bias.** Toward understating the case for EaaS — the
> opposite of the direction a reader would suspect the thesis of
> favouring, but wrong either way.
>
> **What would resolve it.** Key the test on the **budget shadow
> price** (bottleneck where the dual exceeds a stated threshold), not on
> unserved energy. Until then the printed sentence is an artefact and
> the duals are the reportable quantity.

---

## 4. FINDING — collection-rate leakage (next F-number)

> ### F-<n> — Under realistic collection friction a fully-funded
> capital envelope still leaves a fifth of emissions outside the cap
>
> **Source.** `05_run_fin3_capital_sweep.py`, sliced builder, run
> 2026-09-20 at the new B\*, 15 solves, NDC-3 unconditional caps.
>
> **Result.** Emissions falling outside the cap boundary, by capital
> envelope and DisCo collection rate:
>
> | capital case | no_friction (100%) | reform_target (60%) | current (35%) |
> |---|---|---|---|
> | tight | 42.1% ⚠ | 47.2% ⚠ | 50.0% ⚠ |
> | moderate | 13.7% ⚠ | 24.9% ⚠ | 30.2% ⚠ |
> | adequacy | 6.6% | 12.7% ⚠ | **21.3% ⚠** |
> | expansion | 6.6% | 6.6% | 7.9% |
> | unconstrained | 6.6% | 6.6% | 6.6% |
>
> ⚠ = flagged NOT NDC-compliant by the runner's own leakage diagnostic.
>
> **Two readings, both material.** First, **capital scarcity is an
> emissions mechanism**: at `tight` half of national emissions sit
> outside a cap the plan nominally satisfies. Second, **collection
> friction is too**: at a fully-funded envelope (`adequacy`) leakage
> rises from 6.6% to 21.3% purely by moving the collection rate from
> 100% to 35%, and the NDC-compliance flag flips from clean to
> non-compliant.
>
> **This is F-21's mechanism in a second, independent runner.** F-21
> established that tightening a grid-only cap raises national emissions
> because abatement is pushed across the boundary. Here the same
> boundary leaks under two different pressures — capital rationing and
> revenue friction — neither of which is a cap-tightening. The
> mechanism is a property of the grid-only boundary itself, not of the
> cap level.
>
> **Scope condition.** `disco_collection_rate` scales only the domestic
> 35% share of the public capital envelope
> (`effective_budget = base × (0.35·cr + 0.65)`); it is a fiscal-
> realisation rate, not a DisCo collection efficiency, and its arm
> values are ASSUMED with provenance unknown (see L<PACK-3 entry>).
> The tariff × collection-rate revenue mechanism (plan 3.3) is not
> implemented.

---

## 5. F-22 UPDATE

F-22's envelope figures were solved on the **old** B\* grid. Replace
with the post-B\* run, and add the self-consistency result:

> **Re-run 2026-09-20 at the derived B\* (10,273,343,610).** Both
> regimes still reach the identical optimum, **$27.954 bn** at
> 41,031 MW solar and 65,613 MWh storage. Public capital required:
>
> | regime | public capital to reach $27.954 bn |
> |---|---|
> | traditional | **$10,273,343,610** (1.00 × B\*) |
> | eaas | **$4,109,337,444** (0.40 × B\*) |
>
> **EaaS reaches the same optimum on $6,164,006,166 less public
> capital — 60.0%.** Note 0.40 is a sweep grid point, so the EaaS
> requirement lies in (0.35, 0.40] and 60.0% is an upper bound at this
> resolution.
>
> **B\* is now self-consistent.** At `adequacy` (1.00 × B\*) the
> envelope is exactly consumed — realised spend 10,273,343,610, dual
> **0.0000** — and goes slack above it. Under the superseded constant
> the same tier bound at a dual of 0.2147. The tier name is now earned
> by the model's own dual rather than asserted.
>
> **One figure moved sharply and is not yet explained:** the EaaS dual
> at a zero envelope rose from 11.26 to **147.21** between runs. The
> `r_concessional` correction (0.06 → 0.0212) is the likely cause, but
> this has not been isolated. Do not cite the 147.21 figure until it is.
>
> The degeneracy scope condition is unchanged and still applies: above
> 0.40 × B\* the EaaS arm is cost-flat, reports `binding` with a zero
> dual because a costless transfer absorbs the envelope, and the
> non-dominated check's six flagged points are that artefact.
