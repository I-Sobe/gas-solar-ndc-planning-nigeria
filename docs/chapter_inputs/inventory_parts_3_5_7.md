<!-- VERBATIM EXTRACT (TASK 31 Part B). Source: docs/model_capability_inventory.md
     at commit daf02e0. Three blocks below, each byte-for-byte the source lines
     named in its marker; nothing summarised, reworded or reordered. The only
     added text is the HTML-comment markers. For §3.6 (Parts 3 and 5) and
     §3.7.3 (Part 7). -->

<!-- ===== BEGIN source lines 1251-1452: PART 3 -->
## PART 3 — the arm system, complete

`src/scenarios.py` read in full (3,465 lines). Of its 36 top-level
`def`s, **23 return arm-value dictionaries** consumed by
`load_scenario()` via a `*_case` parameter. The rest are plumbing,
excluded from the arm count for the reason stated: `planning_horizon`
(computes the year index, not a case dict), `demand_growth_benchmark`
and `demand_growth_prior` (external benchmarks / a Monte Carlo prior
over the SAME arms — explicitly "NOT scenario arms," quoted from
`demand_growth_benchmark`'s own docstring, kept out of the registry so
consumers iterating it are unaffected), `nirp_solar_cumulative_mw` (a
data table `solar_build_scenarios()`'s multipliers are applied to, not
itself a case dict), `asset_lifetimes`/`asset_lifetime_sweep` (a
by-technology lookup table and its sensitivity band, not a `*_case`
selector — no `asset_lifetime_case` parameter exists in
`load_scenario()`), `solar_min_build_default` (a bare float, no case
selection), `gas_probability_weights` (a Monte Carlo prior, consumed
only by the orphaned `stochastic.py`, Part 1B), `_load_gas_plant_vintages`/
`gas_conversion_tiers`/`gas_vintage_retirement_series_mw`/
`gas_ndc_17gw_trajectory_mw`/`hydro_trajectory`/`_solar_cumulative_cap`
(private or public HELPER functions that derive a time series from an
arm already selected elsewhere, not arms themselves), and `load_scenario`
itself (the assembler).

### Arm table — 23 arms, every value, source status, exercise status

**Method for the "exercised" column**: grepped every named value across
`results/` (all subdirectories) for its literal string. Many runners
embed the FULL arm-tag string in output filenames or in a JSON field
(e.g. `...__ccgt_eff_central__cr_central__dscr_central__f1_t...`,
confirmed by direct inspection of hit files) — a hit in this scheme is
high-confidence evidence of exercise. **But three arms
(`bank_tenor_case`, `gas_price_case`, `voll_case`) returned ZERO hits
for every one of their values, including the DEFAULT/most-common
choice** — confirmed by checking a sample `diagnostics.json`'s own key
list directly: it records derived dollar/rate quantities, not the
originating arm-name string, for these three. **A zero hit for these
three arms means "not part of this tag/field scheme," not "never run"**
— genuinely indeterminate by this method, stated plainly rather than
guessed at. For every other arm, a zero hit is treated as reasonably
confident evidence of non-exercise, EXCEPT where the value's own code
raises a `ValueError` if selected at all (`peak_nirp`, `gas_avail_central`)
— those are certain-never-exercised by construction, not by absence of
evidence.

| arm (`*_case` parameter) | values | meaning (condensed; sourcing quoted where the file states it) | source status | exercised? |
|---|---|---|---|---|
| `demand_level_case` | `served` (37.09 TWh) | Tier 1: "observed GROSS GENERATION at busbar" [NERC 2024]. Not a bound; measured. | SOURCED | yes (1,953 files) |
| | `latent_low` (38.47), `latent_high` (76.93) | DEPRECATED — raise in `load_scenario`; retained "for provenance only" | marked DEPRECATED, unsourced basis (built as multiples of a since-corrected base) | present in 5 files each, but these are references to the deprecation itself (raise messages), not live runs — cannot raise and also produce output |
| `demand_case` | `constrained_continuation` (0.0314) | Ember 2010-2025 grid-generation CAGR — "a CONSTRAINT-PERSISTENCE arm... NOT organic demand growth" | SOURCED | yes (4) |
| | `organic_central` (0.039) [CENTRAL] | Route A (Liddle & Hasanov 2023, elasticity 0.80) + Route B (Kenya per-capita convergence); GDP 4.5% is "a DECLARED ASSUMPTION" | SOURCED method, ASSUMED GDP input | yes (4) |
| | `organic_high` (0.060) | Route A (eps 0.90, GDP 6.8%) / Route B (Ghana convergence) / Route C (NIRP low case) | SOURCED method, ASSUMED GDP input | yes (4) |
| `peak_multiple_case` | `peak_suppressed` | 2016 Oluwole reconstruction, rationing-flattened shape — "DECLARED FLOOR, NOT a candidate central case" | SOURCED (shape); vintage-declared | yes (37) |
| | `peak_unsuppressed` [CENTRAL] | 2016 Oluwole reconstruction, unsuppressed | SOURCED (shape, 2016 vintage) | yes (1,038) |
| | `peak_nirp` | `None` — "raises if selected, by design"; NIRP gives a load factor but no daily profile | SOURCE: [PENDING] | **never — cannot be, raises unconditionally** |
| | `peak_high` | Evening factor pinned at 1.20, "CONSTRUCTED upper bound, not measured" | ASSUMED/CONSTRUCTED (declared) | yes (27) |
| `reserve_margin` | float, or literal `"nirp_spinning"` | NIRP 2024 §2.3 p.19, 900 MW fixed spinning-reserve requirement, converted to a shrinking fractional margin | SOURCED | not separately checked (not a `*_case` dict; a plain float/string param) |
| `solar_build_case` | `nirp_trajectory` (1.0×) [effectively CENTRAL] | NIRP's own schedule; "REPORT UNSERVED ENERGY AND FEASIBILITY FROM THIS ARM, NOT NPV" | SOURCED (NIRP Annex F / Table 8) | yes (3) |
| | `nirp_accelerated` (2.0×) | "[SOURCE NEEDED] for the 2.0x... currently a declared assumption" | stated ASSUMED, explicitly flagged `[SOURCE NEEDED]` | **0 hits — not found** |
| | `deployment_unconstrained` (`None`) | "the counterfactual that isolates what the deployment limit itself costs" | n/a (no cap) | yes (4) |
| `carbon_case` | `no_policy` [CENTRAL] | `carbon_active=False`, price 0.0 | n/a (policy stance) | yes (356) |
| | `moderate_policy` (50 USD/tCO2), `stringent_policy` (75 USD/tCO2) | no sourcing note given for either price in the docstring — unmarked | **unmarked — not classified in-file** | **0 hits each — not found** |
| `gas_deliverability_case` | `downside`/`baseline`/`upside`/`shock_recovery` (structural) | must match `data/gas/processed/gas_available_power_annual_twh_th.csv`'s `scenario` column | data-driven (CSV); shapes not classified in scenarios.py itself | yes (65–966 files each) |
| | `flat_downside`/`flat_upside`/`flat_shock_recovery` | "Level-equivalent flat controls (GAS-3 only)... to isolate SHAPE vs LEVEL" | data-driven (CSV) | yes (3 each) |
| | `gas_deliv_flat`/`gas_deliv_gmp_partial`/`gas_deliv_gmp_full` | GMP 2026 policy-target family, "planned increment: 0.0 / 0.50 / 1.0" | data-driven (CSV / `gas_deliverability_cases.yaml`) | yes (24–92 each) |
| `land_case` | `tight` (1,000 km²), `moderate` (3,000), `loose` (6,000) | "Land availability envelopes" — no source cited in-file | **unmarked — not classified in-file** | tight/moderate yes (175 each); loose yes (4) |
| `capital_case` | `tight` (0.50×B\*), `moderate` (0.85×), `adequacy` (1.00×), `expansion` (1.20×), `unconstrained` (`None`) | B\* = 10,273,343,610 USD, "Derived value... Source: `28_run_capital_frontier.py`, run 2026-09-20" — superseded an undocumented constant that understated the requirement by 12.84% (L45) | SOURCED (B\* itself is a model-internal derived quantity, declared as such — see `docs/parameter_provenance.md` row 10) | all yes (175–1108 depending on tier) |
| `tariff_case` | `tariff_low` (75M), `tariff_central` (95M) [CENTRAL], `tariff_high` (115M) | Nigeria's 2016 NBET solar IPP negotiation: 115 = NBET's agreed rate; 75 = FGN counter-position; 95 = "the MIDPOINT... A CONSTRUCTION, not an independently sourced figure" | SOURCED (low/high); central is a DECLARED construction | yes (11/324/11) |
| | `tariff_legacy_95m`, `tariff_legacy_65m` | "RETAINED FOR ATTRIBUTION ONLY, both UNSOURCED" | explicitly UNSOURCED, kept for attribution | yes (12 each) |
| `om_case` | `om_none` (synthetic, all-zero regression arm) | forces every O&M rate to 0.0, "both builders' O&M term is identically 0.0" | n/a (synthetic control) | yes (4) |
| | `om_low`/`om_central`[CENTRAL]/`om_high` | NREL ATB 2024 Advanced/Moderate/Conservative cost cases, data-driven from `data/cost/processed/om_cost.csv` | SOURCED (NREL ATB) | `om_central` yes (219); `om_high` yes (8); **`om_low` 0 hits — not found** |
| `storage_cc_case` | `storage_cc_low` (0.50), `storage_cc_central` (0.80) [CENTRAL], `storage_cc_high` (1.00) | "SOURCE: DECLARED JUDGEMENT, not measured" — central "matches gas's availability-factor credit" | DECLARED JUDGEMENT (explicit) | low/high yes (4 each); central yes (1,068) |
| `gas_availability_case` | `gas_avail_observed` (0.295 at the time; 0.3061 since TASK 73, whose derivation 3,351.16/10,947 it now matches) | NERC-measured 2024 (3,351/10,947 MW). "DECLARED LOWER BOUND — double-counts fuel" | SOURCED (NERC), with a stated double-counting caveat | yes (24) |
| | `gas_avail_central` (`None`) | "SOURCE: [PENDING]... Raises if selected until sourced" | SOURCE: [PENDING] | **never — cannot be, raises unconditionally** (6 files hit the STRING "gas_avail_central" but these are the code's own docstring/error text appearing in log output, not a completed run using it — confirmed by the arm's own raise) |
| | `gas_avail_nameplate` (0.80) | "the pre-2.10 unsourced assumption, retained as an explicit UPPER BOUND" | explicitly UNSOURCED, declared as an upper bound | yes (1,040) |
| `conversion_case` | `conv_none` (regression) | `available=False`; reproduces pre-2.8 results; annual builder raises on any other value | n/a (control) | yes (174) |
| | `conv_central` [CENTRAL] (600 USD/kW, 0.50 uplift, 0.46 eff) | "DERIVED, not SOURCED... greenfield CC-minus-SC increment" (Gas Turbine World 2024) + a declared brownfield premium | DERIVED (method sourced, premium ASSUMED) | yes (734) |
| | `conv_low` (850/0.40/0.42), `conv_high` (450/0.50/0.50) | same method, bracketing the premium/efficiency/uplift assumptions | DERIVED/ASSUMED, same basis | yes (27 each) |
| `gas_retirement_case` | `gas_retire_none` (0 MW/yr) | "a DECLARED UPPER BOUND on the surviving fleet, not a forecast" | DECLARED (explicit) | yes (24) |
| | `gas_retire_proxy` (680 MW/yr from 2035) | "the pre-2.11 'brownfield retirement proxy'... SOURCE: [PENDING]" | SOURCE: [PENDING] | yes (4) — exercised despite being unsourced |
| | `gas_retire_vintage_30` [CENTRAL] (30 yr) | "SOURCE: ASSUMED — analyst vintage estimate, not traced to a Nigerian source"; central because two opposite errors partly cancel at this figure | ASSUMED (explicit) | yes (992) |
| | `gas_retire_vintage_40` (40 yr), `gas_retire_vintage_50` (50 yr) | same ASSUMED basis; the 40-yr arm retires two currently-operating plants (Delta_1, Egbin_1), "reported as a limitation, not corrected" | ASSUMED (explicit) | yes (20/24) |
| `gas_build_case` (`GAS_BUILD_CASE_CHOICES`, not a `_scenarios()` dict) | `gas_ndc_17gw` [intended CENTRAL] | NDC 3.0's 17 GW-by-2035 target, on a NIRP-derived trajectory — "DERIVED, not stated by NIRP" | DERIVED | yes (4) |
| | `gas_unconstrained` | no cap; cost-optimal build | n/a | yes (1,028) |
| | `gas_none` | fixed at 0; "Reproduces every pre-2.9 result" | n/a (control) | yes (32) |
| `hydro_case` | `hydro_flat` [REGRESSION ARM] (11.47 TWh flat) | NERC 2024, 30.92% of gross generation, held flat | SOURCED | yes (826) |
| | `hydro_committed` (~12.8 TWh) | "SOURCED 2025 figure... 'held flat after 2025' is DECLARED" | SOURCED level, DECLARED shape | yes (23) |
| | `hydro_ndc` (→10,400 MW by 2030) | NDC's sourced 2030 target; "linear interpolation SHAPE and the post-2030 flat treatment are DECLARED, not sourced" | SOURCED anchor, DECLARED shape | yes (27) |
| `gas_capex_case` | `gas_capex_low` (900,000 USD/MW) | "the pre-2.13 bare literal... never sourced" — retained as an explicit lower bound | explicitly UNSOURCED (declared as such) | **0 hits — not found** |
| | `gas_capex_central` [CENTRAL] (1,317,000) | DERIVED from NIRP 2024 Tables 19/22/23, method cross-validated against NIRP's own solar figures | DERIVED (reproducible arithmetic quoted in-file) | yes (1,036) |
| | `gas_capex_high` (1,400,000) | upper end of the same NIRP-derived year-by-year cluster | DERIVED | **0 hits — not found** |
| `ccgt_efficiency_case` | `ccgt_eff_iso` (0.5445) | NREL ATB 2024 new-build H-class NGCC, ISO conditions, no derate | SOURCED (ATB) | **0 hits — not found** |
| | `ccgt_eff_central` [CENTRAL] (0.5173) | ATB ISO figure with "a 5% Nigerian ambient derate... a DECLARED ASSUMPTION, NOT from ATB" | SOURCED base, ASSUMED derate | yes (1,036) |
| | `ccgt_eff_low` (0.5010) | ATB ISO figure with an 8% derate, "a declared assumption, not from ATB" | SOURCED base, ASSUMED derate | **0 hits — not found** |
| `capital_recovery_case` | `cr_central` [CENTRAL] (r_conc 2.12%/25yr, r_comm 11%/10yr) | r_conc: World Bank IDA21 terms, SOURCED; r_comm: "SOURCED BASE + ASSUMED UPLIFT" (IEA Cost of Capital Observatory base, +2-2.5pp uplift "AN ANALYST ESTIMATE, NOT SOURCED") | mixed, precisely stated per-component | yes (1,000) |
| | `cr_ida21` (2.13%/25yr, same r_comm) | "NEARLY IDENTICAL to cr_central... differing only by which of IDA21's two 2026 effective dates" | SOURCED | yes (27) |
| | `cr_comm_low`/`cr_comm_high` (2.80% IDA20/30yr, r_comm 9%/14%) | pre-Part-2 IDA20 rate/tenor, unchanged; r_comm brackets the same assumed uplift | mixed (r_conc SOURCED-superseded-vintage; r_comm ASSUMED) | yes (35 each) |
| `dscr_case` | `dscr_regression` (1.10) | "reproduces the pre-F1 commercial-arm value exactly" | n/a (regression control) | yes (201) |
| | `dscr_central` [CENTRAL] (1.30), `dscr_low` (1.20), `dscr_high` (1.40) | no explicit source/assumed marker given for the three named levels themselves | **unmarked — not classified in-file** | yes (799/23/23) |
| `concessional_envelope_case` | `env_dares` [CENTRAL] (750M) | "STIPULATED, not SOURCED, within this repository"; external corroboration exists (World Bank DARES, $750M IDA credit, Dec 2023) but is not traced inside the repo | STIPULATED (external corroboration noted, not itself verified in-repo) | yes (963) |
| | `env_dares_plus` (1,050M) | "The 1,050M figure has no external corroboration and remains STIPULATED" | STIPULATED, weaker than `env_dares` | yes (23) |
| | `env_none` (0), `env_5x`/`env_10x` (3,750M/7,500M) | declared counterfactuals — "do NOT claim DARES itself is 5x or 10x its stated size" | DECLARED COUNTERFACTUAL (explicit) | yes (35/27/27) |
| `capital_treatment_case` (`CAPITAL_TREATMENT_CASE_CHOICES`) | `pre_f1_overnight` | reproduces the pre-F1 objective exactly (four overnight-capex terms + salvage + replacement) | n/a (structural choice) | yes (66) |
| | `f1_tranche` [CENTRAL] | per-tranche CRF annuity, replaces all six terms | n/a (structural choice) | yes (1,023) |
| `bank_tenor_case` (`BANK_TENOR_CASE_CHOICES`) | `bank_tenor_comm` [NEW DEFAULT] | objective's own commercial tranche, truncated at a real tenor — corrects a prior mismatch (bankability priced at 18% naira-basis, no tenor, vs. the objective's 11%/10yr) | n/a (structural correction) | **indeterminate — see method note above** |
| | `bank_tenor_legacy` | "the pre-correction blended-WACC factor... undiscriminated by any tenor. Every result produced before this task used this basis" | n/a | **indeterminate — see method note above** |
| `blend_case` (`blended_finance_scenarios`) | `commercial_only` (0%), `blended_central` [ADOPTED] (30%), `blended_optimistic` (50%), `concessional_heavy` (75%) | concessional-share tuples feeding the bankability blended WACC; `r_concessional` SOURCED, `r_commercial`/`concessional_share` "remain PLACEHOLDERS pending primary-source verification" | mixed — r_conc SOURCED, r_comm/share explicitly flagged PLACEHOLDER | yes (6 each) |
| `new_gas_deliverability` (plain default, not a `_scenarios()` dict) | `constrained` [DEFAULT] | new gas capacity shares the existing fleet's fuel budget | n/a (structural choice) | yes (1,108) |
| | `exempt` | new capacity gets its own fuel-equivalent allowance, sized off `ccgt_efficiency_case` | n/a | **0 hits — not found** |

### `io.py`'s arms

- **`voll_case`** (`load_econ`, default `"voll_low"`): values
  `voll_low`/`voll_mid`/`voll_high`, data-driven from
  `data/cost/processed/unserved_energy_penalty.csv`'s `scenario` column
  (year-2025 row read at `io.py:424-430`) — the SET of values is
  data-driven, not enumerable from `io.py`'s own source without reading
  the CSV. Exercise status: indeterminate (see method note above);
  `voll_mid` alone showed 497 `results/` hits on grep, but that count
  cannot be trusted to mean the OTHER two were never run, only that
  `voll_mid`'s string happens to coincide with some other field/tag —
  not independently confirmed.
- **`gas_price_case`** (`load_econ`, default `"gas_low"`): values
  `gas_low` (7.27M USD/TWh_th, no transport tariff), `gas_mid` (11.01M,
  domestic tariff), `gas_high` (14.87M, export-parity) — figures quoted
  directly from the docstring, data-driven from
  `data/cost/processed/gas_cost.csv`'s `Scenario` column. Exercise
  status: indeterminate, zero hits for all three by the tag-string
  method (see method note).
- **`genset_case`** (`load_econ`, REQUIRED — no default, confirmed
  already in `docs/task_log_ch36_inputs.md` item B this session):
  values `genset_srmc`/`genset_central`[CENTRAL]/`genset_high`, or an
  explicit `None` ("run without the self-generation backstop... the
  regression harness... and the 'voll' arm of
  `23_run_selfgen_frontier.py` both rely on it"), data-driven from
  `data/cost/processed/genset_cost.csv`. Exercised: `genset_srmc` (3),
  `genset_central` (46), `genset_high` (3) — all confirmed present.
- **`om_case`** handling: already covered in the main arm table
  (`om_scenarios()` is defined in `scenarios.py`, but the actual cost
  data it names is loaded by `io.load_om_cost_by_year`/
  `compute_om_cost_npv` from `data/cost/processed/om_cost.csv`) —
  included here for completeness since the question spans both files.

### Non-arm modelling choices — hardcoded constants, `.get()` defaults, in-function literals

`docs/chapters/chapter3_1_v2.md` §3.1.7 lists constraint EQUATIONS, not
hardcoded constants as such, and names `within_block_peaking_factor` as
a live example of the pattern this section extends.
`docs/chapters/chapter3_3_draft.md` §3.3.7, quoted in full, already
lists five:

> "`new_gas_deliverability` (`"constrained"`, both builders);
> `conversion_available` (`False`, sliced builder); `hydro_season_factor`
> (dry 1.045 / wet 0.956, sliced builder); `hydro_min_flow_fraction`
> (0.50, sliced builder); `storage_duration_hours` (4.0, `load_scenario`)."

**Grepped every `.get(` call with a literal (non-`None`) fallback in
both builders and `scenarios.py`, plus every bare module-level numeric
constant, to find what §3.3.7 does NOT yet list. Complete set below,
§3.3.7's five marked accordingly:**

| parameter | value | where | already in §3.3.7? |
|---|---|---|---|
| `new_gas_deliverability` | `"constrained"` | both builders | yes |
| `conversion_available` | `False` | sliced builder | yes |
| `hydro_season_factor` | dry 1.045 / wet 0.956 | sliced builder (`DEFAULT_HYDRO_SEASON_FACTOR`) | yes |
| `hydro_min_flow_fraction` | 0.50 | sliced builder (`DEFAULT_HYDRO_MIN_FLOW_FRACTION`) | yes |
| `storage_duration_hours` | 4.0 | `load_scenario` | yes |
| **`within_block_peaking_factor`** | **1.3448** | both builders, `optimize_model.py:1041`/`optimize_model_sliced.py:1265` | **NEW — named in §3.1.7's prose as the motivating example, but its own value is not tabulated in §3.3.7** — comment at the call site: "Set scenario[\"within_block_peaking_factor\"] once sourced," i.e. SOURCE: [PENDING], currently silently defaulted on every run that doesn't set it explicitly |
| **`social_discount_rate`** | **0.08** | both builders, `r_social = float(scenario.get("social_discount_rate", 0.08))` | **NEW** — this is the widely-discussed $r_s$=8% (register/Chapter 3.1.2's own stated open item, "8% itself is unsourced"); confirmed here to be a plain `.get()` fallback, not a required, explicitly-passed arm — nothing stops a caller from omitting it silently |
| **`storage_round_trip_eff`** | **0.90** | both builders | **NEW** — no `storage_efficiency_case` arm exists; this round-trip efficiency is a bare fallback |
| **`selfgen_busbar_factor`** | **1.0** | both builders | **NEW** |
| **`solar_min_build_mw_per_year`** | **0.0** | both builders, sourced from `solar_min_build_default()` | **NEW as a `.get()` site**, though the function itself (and its "no runner calls this any more" status) is already documented in `scenarios.py`'s own docstring, Part 1's module map |
| **`season_solar_factor`** | (`DEFAULT_SEASON_SOLAR_FACTOR`, value not re-derived here — a sibling default to `hydro_season_factor`) | sliced builder | **NEW**, and explicitly flagged by §3.3.7's own closing sentence as "one of four copies of the same two numbers, alongside a sibling solar shape factor" — this IS that sibling |
| **`EF_TCO2_PER_MWH_TH`** | **0.1823** | `io.load_econ`, in-function literal (not a `.get()`, an unconditional assignment) | **NEW** — SOURCED (IPCC default stationary-combustion factor, HHV-adjusted, quoted in-file: "IPCC default, HHV-adjusted. I'd Replace with Nigeria NIR factor if found") but hardcoded, not arm-selectable |
| **`RESERVE_MARGIN_SPINNING_MW`** | **900.0** | `scenarios.py` module constant | **NEW** — SOURCED (NIRP 2024 §2.3 p.19), feeds the `reserve_margin="nirp_spinning"` special-string path |
| **`GAS_BASELINE_MW`** | **10,947** | `scenarios.py` module constant | **NEW** — SOURCED (NERC Quarterly Reports, 23-plant tracked fleet), a structural guard target multiple other functions validate against |
| **`MODEL_START_YEAR`/`MODEL_END_YEAR`/`REPORT_END_YEAR`** | **2025 / 2054 / 2045** | `scenarios.py` module constants | **NEW** — the 2046-2054 buffer years are explicitly declared extrapolations, "Acceptable precisely because those years are never reported" |
| **`TARIFF_SWEEP_GRID`** | 10-point grid, 30M–110M | `scenarios.py` module constant | **NEW** — carries its own STALE-DOC warning in-file: "fine resolution was placed around the old 44-46M crossing... Widen and re-centre this grid in the final pass once rates are frozen" |
| **`_CONVERSION_NAMEPLATE_TOTAL_MW`** | **6,482.0** | `scenarios.py` module constant | **NEW** — a structural guard target (16 convertible plants), not a modelling assumption in the same sense as the others, but a hardcoded literal nonetheless |
| **`_HYDRO_EXPLOITABLE_MW`** | **17,620.0** | `scenarios.py` module constant | **NEW** — SOURCED (Energy Commission of Nigeria), a validity-guard ceiling, RAISES if exceeded |
| **`_HYDRO_BUILD_RATE_WARN_MW_PER_YR`** | **700.0** | `scenarios.py` module constant | **NEW** — DECLARED (one Zungeru-equivalent/yr), WARNS only, never raises |
| **`# storage_solar_surplus_frac`** | **0.20 (commented out)** | `optimize_model.py:386` | **NEW, and notable for being DEAD, not live** — a parameter that WAS apparently a modelling choice, now commented out; not exercised by definition since the line does not execute |

**Not counted as new findings, checked and excluded**: the `None`-default
`.get()` calls on `peak_mw`, `capacity_credit`, `hydro_capacity_mw_by_year`,
`hydro_energy_twh_by_year`, `gas_availability_factor`,
`solar_cumulative_cap_mw`, `gas_new_capacity_cap_mw`,
`gas_baseline_retirement_series_mw`, and `tariff` — every one of these
is followed by an explicit `is None` check that raises or branches
loudly (confirmed by direct reading, same discipline as the sentinel-
guarded arms in Part 1), so `None` here is a REQUIRED-effectively
guard, not a silent numeric fallback; including them would overstate
this section's finding.

**Total: 15 non-arm hardcoded constants/defaults found beyond §3.3.7's
five** (16 counting the dead, commented-out `storage_solar_surplus_frac`
separately), for **20 in total** across both sections once §3.3.7's
five are folded in.

---


<!-- ===== END PART 3; BEGIN source lines 2182-2407: PART 5 -->
## PART 5 — uncertainty and trade-off methods, precisely

### 1. The arm system / scenario sweep — scenario uncertainty

Full evidence base: Part 3 (23 arms, ~95 values) and Part 3B (their
sourcing audit). What it claims, in the codebase's own words
(`docs/methodology_protocols.md:363-368`, quoted):

> "These are not samples from a distribution. They are named,
> internally coherent states of the world, and the constraint-relief
> cross-tab rules out combinations that would be incoherent — a world
> where demand booms while the gas that would serve it collapses, for
> instance."

What it does NOT claim: a probability, a confidence interval, or
coverage of the full joint space of arm combinations — Part 3 found
several values (e.g. `nirp_accelerated`, `moderate_policy`/
`stringent_policy`) with zero `results/` exercise, so even the DISCRETE
set of named worlds is not exhaustively populated in practice. Section
C's interaction cells (Part 1, C-1 correction) are the only place two
dimensions move at once in a controlled way; everywhere else, one arm
moves against the central configuration at a time (Section B) or not
at all (Section A).

### 2. Sobol (`29_`) — parameter uncertainty by Monte Carlo over deterministic solves

Docstring's own distinction, quoted in full (`scripts/29_run_
parameter_uncertainty.py:5-27`; identical text already captured this
session in `docs/task_log_ch36_inputs.md` item D — repeated here since
TASK.md asks for it in this Part specifically):

> "PARAMETER uncertainty vs SCENARIO uncertainty -- keep these separate
> --------------------------------------------------------------------
> This model is DETERMINISTIC with structured scenario analysis. The
> project's 38-arm sweep answers SCENARIO uncertainty -- which discrete
> world happens... This script answers a DIFFERENT question: PARAMETER
> uncertainty -- five physical/cost quantities whose TRUE value is
> unknown but FIXED (not a scenario branch), sampled continuously at
> the CENTRAL scenario configuration, decomposed by Sobol variance
> analysis to find which parameter's uncertainty drives the output
> band. The two kinds of uncertainty must not be conflated..."

What it claims: a variance decomposition over five CONTINUOUS physical/
cost parameters (gas availability/capacity-credit, gas efficiency,
solar CF, O&M level, conversion capex), each independently re-optimised
(full LP re-solve per draw, not fixed-plan). What it does NOT claim:
coverage of scenario uncertainty (explicitly disclaimed above), nor
genset efficiency uncertainty (deliberately excluded — Part 1B/2's own
reading of this docstring already covers why: the three published
genset efficiency values are different LOAD POINTS, an operating
choice, not a parameter uncertainty).

### 3. `src/stochastic.py` — fixed-plan re-dispatch

Full coverage already exists at two points in this document — Part 1B
(`### src/stochastic.py (227 lines)`, line 479) and Part 2's addition 1
(`### Part 2, addition 1 — stochastic.py: a runner away, not a
rewrite`, line 1096) — restated here in this Part's own terms since
TASK.md asks for the method, distributions, caller, results/ status,
and limitation note specifically in Part 5:

**Full method**: two-stage fixed-plan re-dispatch. Solve the LP once at
a base scenario to obtain the optimal capacity plan
(`solar_public_add`/`solar_eaas_add`/`storage_add` per year); `.fix()`
those three variables in every subsequent draw via `build_model(...,
fixed_plan=fixed_plan)`; re-solve under each draw so only dispatch
variables adjust. Infeasible draws are dropped and counted, not
silently discarded.

**Uncertainties and their distributions** (`sample_uncertainties()`,
Part 1B's reading, quoted there): demand growth — truncated-at-zero
NORMAL on the growth rate, `demand_sigma=0.01` default; gas regime —
DISCRETE draw from `gas_probability_weights()` (`{"baseline": 0.50,
"downside": 0.25, "upside": 0.20, "shock_recovery": 0.05}`, confirmed
still current against `gas_deliverability_scenarios()` in Part 1B);
carbon price — LOGNORMAL, only when `carbon_active`.

**Who calls it**: nobody. Confirmed zero importers anywhere in `src/`,
`scripts/` (including `deferred`/`retired`), `tests/`, `notebooks/`
(Part 1's module map). There is no `stochastic`-specific `results/`
directory at all — unlike `EaaS.py`/`gas_balance.py`/`postprocess.py`,
none of which have ever been called either but which at least share a
call path with something that HAS produced output (`run_deterministic_
scenario`'s `results/eaas/`), `stochastic.py` has produced literally
nothing, ever.

**Its own limitation note, verbatim** (`src/stochastic.py:31-37`):

> "Limitations
> -----------
> - This measures the robustness of a plan optimized for the BASE case;
>   it does NOT find the plan that is optimal under uncertainty (that
>   would require stochastic programming with recourse). State this
>   explicitly in the thesis.
> - Regime probabilities are a sensitivity prior, not a forecast (see
>   gas_probability_weights). Report symmetric-weight results
>   alongside."

**What wiring it would need**: already answered precisely in Part 2's
addition 1 — confirmed **"a runner away, not a rewrite"**: every
interface (`fixed_plan`'s three keys, `demand_growth`/`gas_scenario`/
`carbon_active`/`carbon_price` scenario keys, `gas_probability_
weights()`) checked directly against HEAD and found current. What's
missing is a `scripts/XX_run_stochastic.py` that builds a scenario via
`load_scenario()`/`load_econ()` through the standard arm surface, calls
`run_stochastic_simulation()`, and writes results — with the one scope
caveat that it only calls the ANNUAL builder (`optimize_model.
build_model`), never the sliced/reporting builder.

### 4. ε-constraint (`18_`, `28_`) — bi-objective by constraint

Full detail already in `docs/task_log_ch36_inputs.md` item C (this
session's own report — point counts, ε grids, duals, binding checks,
non-dominated verification, all quoted from source) and refined by
this document's own Part 2 and Part 4:

- **Method, restated**: one single-objective LP (`system_cost_npv`,
  unconditionally minimised — Part 2 item 15) per swept point on a
  SECOND bound (the emissions cap for `18_`, the public capital
  envelope for `28_`) — never a genuine multi-objective solve. `18_`:
  25 points × 4 financing arms = 100 solves. `28_`: 19 non-uniform
  points × 2 regimes = 38 solves.
- **What Part 2 adds since the §3.6 report**: both frontiers run on
  `build_model_sliced` (`18_`) — Part 2 established the sliced
  builder's hydro/storage/conversion mechanisms are the ones these
  frontiers' points actually reflect, not the annual builder's cruder
  versions; Frontier B (`28_`) explicitly runs `carbon_case="no_
  policy"`, uncapped, so it never touches the emissions side at all
  (confirmed already in Part 2's EaaS/Part 1B discussion).
- **What Part 4 adds**: currency. Both frontiers' `results/` output was
  found current at HEAD in Part 4's per-runner currency check (cross-
  reference Part 4's own entries for `18_run_pareto_frontier.py` and
  `28_run_capital_frontier.py` — both among the small actively-
  maintained September-2026 population, not the August-23 stale
  majority).
- **What it does NOT claim**: a genuine Pareto FRONTIER discovered by a
  multi-objective solver — it is a scan of single-objective solves at
  a grid of ε values, with a post-hoc pairwise non-dominance CHECK
  (quoted in both Part 2 and the §3.6 report) that can and does find
  dominated points (a solver/grid artefact, both docstrings say so
  explicitly), not a guarantee of Pareto-efficiency by construction.

### 5. Anything else addressing uncertainty, robustness or trade-offs

A genuine fresh check, not a re-citation — each item below judged on
whether it produces a FRONTIER/distribution/robustness statement (a
method, in TASK.md's sense) or a plain one-at-a-time point sensitivity
(which item 1 already covers conceptually, and adds nothing new as a
METHOD even if it's a real, reportable result):

- **`scripts/20_run_cf1_solar_yield_sensitivity.py`** — a POINT
  sensitivity (solar capacity factor swept across its `solar_cf`
  range), one arm at a time, no frontier or distribution produced. Not
  a new method.
- **`scripts/27_run_hydro_minflow_sensitivity.py`** — same shape,
  sweeps `hydro_min_flow_fraction`. Not a new method.
- **`scripts/19_run_carbon_price_sweep.py`** — sweeps `carbon_price`
  as a continuous dial rather than the two named `carbon_case` policy
  points; this is CLOSER to a frontier (a cost-vs-carbon-price curve)
  than the CF-1/hydro sensitivities, but it is still one dimension
  swept holding everything else fixed, methodologically identical in
  KIND to Section B of the arm sweep (item 1), just on a continuous
  rather than discrete grid. Not counted as a distinct METHOD, though
  it is the one entry here that produces a curve rather than a table
  of points.
- **`scripts/22_salvage_sensitivity.py`** — a basis-consistency check
  (confirms both arms compared share the same salvage/demand/gas/VoLL
  basis before reporting a premium), not an uncertainty or trade-off
  method at all — closer to a regression-arm discipline check (Part
  2/methodology_protocols.md §6's "one variable at a time" rule) than
  to items 1-4.
- **`scripts/retired/09b_run_rel1_*.py`** (both, retired) — one-at-a-
  time mode/retirement sensitivities on the REL-1 feasibility matrix,
  same shape as the CF-1/hydro-minflow entries; retired, and not a
  distinct method even when they were live.
- **`scripts/deferred/17_run_monte_carlo.py`** — the one genuine
  exception in this section: a JOINT demand-growth × gas-regime
  Monte Carlo that RE-OPTIMISES (not fixed-plan) at every draw — a
  real third method, distinct from both `stochastic.py` (fixed-plan,
  not full re-optimisation) and Sobol (parameter uncertainty on
  physical/cost quantities, not joint scenario-variable resampling).
  Deferred/dead per Part 4's own finding, for the two defects both
  `stochastic.py`'s and `29_`'s docstrings cite it as the cautionary
  example of (LCOE accounting bug; unsourced demand-growth prior) —
  its capability, if those two defects were fixed, is not duplicated
  by anything live today.

**No other uncertainty/robustness/trade-off method found** after this
sweep — the five items above, plus items 1-4, are the complete set.

### What the model is

**The model is deterministic, single-objective at every individual
solve, and perfect-foresight within each solve's horizon — recovering
multi-objective trade-offs only through repeated single-objective
solves, never through a genuine multi-objective solve, and covering
outcome uncertainty only through separate, mostly-unwired methods
layered on top of that deterministic core.** It is deterministic: one
Pyomo LP, one optimal plan per named scenario, no random element in
either builder (Part 2's own reading of both builders' construction;
`src/stochastic.py` and `scripts/29_run_parameter_uncertainty.py` are
the only places `numpy.random`/sampling enters anywhere in `src/`, and
neither is part of the reported central path). It is single-objective
at every solve: `system_cost_npv` alone, unconditionally minimised —
the weighted-sum alternative (`weight_cost`/`weight_emissions`) was
removed and confirmed fully gone (Part 2's removed-capabilities
section). Multi-objective trade-offs (cost-vs-emissions, cost-vs-
capital) are recovered only as ε-constraint SCANS — many single-
objective solves at swept bound values (item 4 above), not a Pareto
solve. It is perfect-foresight within each solve's horizon: `build_
model`/`build_model_sliced` each construct one Pyomo model over the
ENTIRE year range (`T = range(len(scenario["years"]))`,
`optimize_model.py:116-117`, confirmed identical in shape in the
sliced builder) and solve it once, simultaneously, for all years — a
grep of `src/*.py` for "rolling"/"receding"/"myopic" returns zero hits,
confirming no rolling- or receding-horizon mechanism exists anywhere to
contradict this. Outcome uncertainty is addressed by two further,
separately-scoped methods bolted onto this deterministic core rather
than built into it: Sobol parameter-uncertainty decomposition (wired,
current) and fixed-plan stochastic re-dispatch (fully specified,
zero callers, "a runner away" per Part 2).

---

---


<!-- ===== END PART 5; BEGIN source lines 2659-2867: PART 7 -->
## PART 7 — structural assumptions and omissions

Not parameters — the assumptions embodied in the model's FORM, derived
from what the two builders' variable and constraint sets contain (Part
2) and, where the absence of an arm is itself evidence (Part 3), from
what they don't. Each point: the assumption, where it's embodied (or
where its absence is confirmed), and its direction of bias.

**Spatial resolution.** Nigeria is a single node. Grepped both builders
for `zone`/`node`/`transmission`/`region` as constraint or variable
names: the only hits are a `selfgen_busbar_factor` comment (below) and
a docstring line about hydro commissioning — no locational variable,
no transmission-capacity constraint, no zonal balance anywhere.
Generation, storage, and demand are summed to one national balance
every period. **Bias**: understates the cost and physical difficulty of
building solar/storage/gas WHERE the grid can absorb or evacuate it —
a plan that is optimal at the national level could require transmission
reinforcement the model never prices or constrains against. No sign is
assertable without knowing whether Nigeria's real bottlenecks favour
solar (typically more distributed, arguably eases this) or gas/hydro
(typically more concentrated, arguably worsens it) — stated as
directionally uncertain, not zero.

**Foresight.** One Pyomo LP is built over the ENTIRE horizon
(2025-2054, `T = range(len(years))`, both builders) and solved once;
every year's investment and dispatch decision sees every other year's
data simultaneously. Grepped `src/` for "rolling"/"receding"/"myopic":
zero hits. **This is perfect foresight**, not sequential/adaptive
planning. **Bias**: UNDERSTATES the cost of uncertainty and OVERSTATES
how efficiently a real planner could sequence investment — a real
planner commits capital without knowing future demand, gas
deliverability, or cost realisations, and would build some insurance
against being wrong that a perfect-foresight plan has no reason to
carry. This is the same gap `src/stochastic.py` (Part 1B, Part 5) exists
to probe and cannot, being unwired.

**Linearity / absent integer structure.** Grepped both builders for
`pyo.Binary`, `pyo.Integer`, and any `domain=` other than
`pyo.NonNegativeReals`: **every decision variable in both builders is
continuous** (`domain=pyo.NonNegativeReals`, confirmed for every `Var`
declaration checked — solar/gas/storage additions, dispatch, curtailment,
tranche volumes, conversion variables). No binary or integer variable
exists anywhere. This single fact implies four absent structures, each
biasing differently:
- **No unit commitment** (no binary on/off state for gas plants) —
  UNDERSTATES the operational cost and emissions of keeping thermal
  plant synchronised/cycling, and OVERSTATES how freely gas generation
  can track a residual-demand shape net of solar.
- **No minimum-stable-generation floor** — UNDERSTATES gas plants'
  real inflexibility at low output and OVERSTATES how far dispatch can
  fall to make room for solar/storage, which is the same direction as
  unit commitment's omission and compounds it.
- **No ramping constraint** — UNDERSTATES the cost/risk of steep net-load
  swings around the solar-driven daytime dip and evening peak (a
  concern the sliced builder's own slice structure is otherwise
  designed to expose, per Part 2 item 1, but ramping WITHIN a slice
  transition is still untouched).
- **No start-up cost, no lumpy/discrete build** — capacity is added in
  arbitrary continuous MW; UNDERSTATES the true cost of a system that
  must round up to the next real, discrete unit size (a plant comes in
  one size, not any size), which generally biases total system cost
  DOWN relative to a world with realistic unit granularity.

**Outage treatment.** There is a firm-capacity DERATE
(`capacity_credit`, Part 2 item 8; Part 3), but reading its
construction directly (`scenarios.py:3038-3039`): **`capacity_credit["gas"]
= _gas_availability_factor` — the SAME scalar used for annual energy
deliverability is reused, unmodified, as the capacity-planning firmness
derate.** This is a PLANNING-LEVEL derate applied to a fixed reserve-
margin CONSTRAINT, not a stochastic forced-outage model with its own
distribution — there is no `forced_outage`/`EFOR`/`outage_rate` concept
anywhere that generates a distribution of outage EVENTS a dispatch
would have to react to within a period. **Bias**: cannot be signed in
general — a single reused derate could overstate or understate true
firmness depending on whether the real correlation structure of gas
outages (which cluster, e.g. during scheduled maintenance windows or
fuel shortages) is better or worse than a flat annual average implies;
what's certain is that the model cannot show a SHORT, sharp outage
event forcing unserved energy or self-generation within an
otherwise-adequate year, since nothing in either builder models outage
TIMING at all.

**Demand elasticity and exogeneity.** `src/demand.py`'s own docstring
states this outright (quoted in full): *"No price elasticity or
sectoral disaggregation"* and, further down, *"No saturation, rebound,
or elasticity effects are modeled."* Demand is a fixed exogenous path,
`base_demand * (1+growth_rate)^t`, entering the electricity balance as
a hard right-hand side every builder builds against — tariff appears
only on the FINANCING side (the EaaS bankability constraint, Part 1B/2
item 12), never as a term demand responds to. **One nuance worth
stating precisely, not conflating**: `scenarios.py`'s demand-GROWTH
arms (Part 3) were themselves CALIBRATED using an income elasticity of
electricity demand w.r.t. GDP (`eps = 0.80`, `scenarios.py:454-458`,
"long-run income elasticity... panel, 1990-2019, income elasticity
averaged 0.8") — this is a real elasticity concept, but it is a
ONE-TIME EXTERNAL INPUT fixing a growth RATE before the model runs, not
a live mechanism inside the LP; it does not make the model
price-responsive. **Bias**: a fixed demand path cannot show a
tariff-driven reduction in load — a high-tariff EaaS/financing scenario
that would, in reality, suppress some demand instead shows up entirely
as a capacity/cost consequence, which OVERSTATES the capacity Nigeria
would need to build under a high-tariff regime and UNDERSTATES demand's
own capacity to adjust.

**System boundary (imports, exports, self-generation, T&D).** No
`import_case`/`export_case` arm exists anywhere in Part 3's 23-arm
inventory, and grepping `scenarios.py` for "import"/"export" as a
scenario dimension returns nothing — **cross-border trade is absent
from the system boundary entirely**, not merely zero-valued by
assumption within an arm. Self-generation (Part 2 item 7) IS modelled,
but as an unbounded backstop OUTSIDE the emissions cap boundary (this
is the F-21/F-31 leakage mechanism this whole session's earlier work
concerned). T&D losses: there is a named concept
(`selfgen_busbar_factor`, `optimize_model.py:415-427`, quoted: *"A
genset delivers at the point of use, downstream of the transmission
loss factor, so 1 TWh of genset output displaces MORE than 1 TWh of
busbar generation... DEFAULT IS 1.0 -- a declared conservative
simplification... It understates the genset by roughly the TLF
(~7%)"*) — but this factor applies ONLY to the genset/busbar
equivalence, defaults OFF (factor 1.0, i.e. transmission losses NOT
credited unless a caller explicitly overrides it), and there is no
general transmission or distribution loss term applied anywhere else
in either builder's main electricity balance — solar, gas, storage,
and hydro all sum directly to demand at busbar with no loss factor in
between. **Bias**: with the default `selfgen_busbar_factor=1.0`, the
model UNDERSTATES how much genset output actually displaces (by
~7%, the TLF itself, per the comment's own arithmetic) — a
conservative bias AGAINST the self-generation/leakage finding, not
inflating it; absent losses elsewhere are a smaller, harder-to-sign
simplification of the same kind as single-node resolution.

**Technology learning.** Solar and storage CAPEX are read as
BY-YEAR series (`load_solar_capex_by_year`/`load_storage_capex_by_year`,
Part 1/Part 6), and Part 6 confirmed these processed CSVs are
hand-maintained, not model-generated — the by-year values DO decline
over the horizon in the data (an exogenous, time-indexed cost trajectory
consistent with an assumed learning curve), but **no constraint or
expression anywhere ties `solar_capex_param[t]` (or its storage
equivalent) to any quantity the model itself has built** — confirmed
by reading both capex terms' construction in Part 2 (`m.DF[t] *
m.solar_public_add[t] * m.solar_capex_param[t]`, a straight read of an
exogenous per-year parameter, no lagged-capacity term anywhere in the
expression). **Exogenous, time-varying cost decline exists; ENDOGENOUS
(deployment-linked) learning does not** — the model's own build decisions
never feed back into the unit cost it or any other scenario faces
later. **Bias**: if real-world deployment (in Nigeria or globally) would
respond to how much THIS plan builds, the model cannot capture that
feedback in either direction — understating cost declines a large
domestic programme might help induce, or, just as plausibly, ignoring
supply-chain cost pressure a large build-out might create. Not
signable from the code alone; flagged as an omission, not a biased
one.

**Policy feedback.** The NDC cap trajectory (`00_build_emissions_cap.py`,
run by hand, Part 6) and the carbon price (`carbon_policy_scenarios()`,
Part 3) are both fixed scenario inputs, constructed BEFORE the solve
and passed into `build_model`/`build_model_sliced` as arguments — never
computed from within the LP, and never fed back into a subsequent solve
within the same run. There is no rule anywhere of the form "if leakage
exceeds X%, tighten the cap further" or "if cost exceeds Y, relax the
carbon price" — policy is exogenous throughout. **Bias**: cannot be
signed in general (a real policymaker might tighten OR loosen in
response to observed cost/leakage, and which the model's silence
favours depends on which direction the counterfactual policymaker
would have moved) — but it does mean every "the cap causes leakage"
finding in this project (F-21, F-31, this session's L60/L63 work) is a
COMPARATIVE-STATICS result across pre-specified cap paths, never a
finding about how policy would actually evolve in response to the
leakage it discovers.

**Capacity-credit treatment.** Restated in structural-assumption terms
from Part 2 item 8: each technology's contribution to the firm-capacity/
reserve-margin constraint is a FIXED, exogenous derate (100% for
storage's power rating; `_gas_availability_factor` reused for gas, per
the outage-treatment point above; near-zero/`[PENDING]` for solar per
Part 3) — none of these derates is ITSELF estimated from any
loss-of-load or reliability calculation internal to this model. Firm
capacity is an assumption fed to the LP, never a result the LP could in
principle revise (e.g. by discovering that a technology mix it
otherwise favours would need a different derate to actually keep the
lights on at the assumed reliability standard).

**Two more, from what else the builders' form assumes**:

- **No fuel-price uncertainty within a single solve.** Gas price is a
  scenario-level arm (`gas_price_case`, resolved once via
  `load_econ()` before the solve), constant for the entire horizon
  within one run — it cannot represent a gas-price shock arriving mid-
  horizon inside a single optimisation the way, say, the sliced
  builder's intra-year gas allocation can react to seasonal
  deliverability. Distinct from (and narrower than) `stochastic.py`'s
  unwired ability to VARY gas regime ACROSS separate solves (Part 5) —
  within any one solve, price is a constant.
- **No explicit water-year / climate trajectory for hydro beyond the
  fixed `hydro_case` arm.** Hydro's annual energy and (in the sliced
  builder) seasonal shares are exogenous named scenarios (Part 2 item
  6, Part 3), not a stochastic hydrological process or a climate-
  trend adjustment over the horizon — a run-of-river-style planning
  assumption that treats historical hydrology as stationary across
  2025-2054.

No single directional bias summarises Part 7 as a whole — some
omissions push cost/capacity DOWN (no lumpy build, no ramping/start-up
cost, no learning feedback risk), some push it UP or are ambiguous
(single-node, outage timing, T&D losses net of the conservative
`selfgen_busbar_factor` default), and demand exogeneity specifically
overstates capacity NEED under high-tariff scenarios. Each is stated
with its own direction above rather than netted into one claim the
formulation does not support.

<!-- ===== END PART 7 -->
