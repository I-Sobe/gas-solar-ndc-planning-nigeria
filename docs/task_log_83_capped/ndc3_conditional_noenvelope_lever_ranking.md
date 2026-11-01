# Lever ranking -- post-epoch, generated 2026-10-23T22:10:21 at HEAD `dabd2a7` (src/scripts have uncommitted changes) by `scripts/build_lever_ranking.py`

Central `4ca4832d` objective 34,995,532,338.04 USD. Files read 42, admitted 42. A verification artefact, not a finding.

## Ranked levers (arms produce different build decisions)

Ordered by column 3 (live arms). `rank (col 1)` is the rank by all-arms span.

| rank (col 3) | rank (col 1) | family | span col 3 live (bn) | span col 1 all arms (bn) | col 3 low arm | col 3 high arm | unasserted (excluded) | % central (col 3) | arms | in F-5 | plans | col 2 (bn) | col 2 differs | max d solar 2054 MW | max d gas thermal draw 2025-54 TWh_th | max d storage 2054 MWh | first year |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | gas_availability | 10.3358 | 10.3358 | gas_avail_nameplate (34.9955) | gas_avail_observed (45.3313) | none | 29.535 | 2 | yes | 2 | 10.3358 | no | 1228.1151 | 995.3892 | 18280.8451 | 2025 |
| 2 | 2 | om | 8.3687 | 8.3687 | om_none (27.1668) | om_high (35.5355) | none | 23.914 | 4 | no | 3 | 8.3687 | no | 56.3764 | 147.9727 | 656.2131 | 2031 |
| 3 | 3 | gas_build | 7.5550 | 7.5550 | gas_unconstrained (34.9955) | gas_none (42.5505) | none | 21.588 | 3 | no | 3 | 7.5550 | no | 5777.8454 | 1166.8871 | 14990.9762 | 2025 |
| 4 | 4 | ccgt_efficiency | 4.0030 | 4.0030 | ccgt_eff_iso (32.5914) | ccgt_eff_low (36.5945) | none | 11.439 | 3 | no | 3 | 4.0030 | no | 1478.9094 | 132.3481 | 2684.5981 | 2030 |
| 5 | 5 | capital_recovery | 3.5783 | 3.5783 | cr_comm_low (33.0264) | cr_comm_high (36.6046) | none | 10.225 | 4 | yes | 3 | 3.5783 | no | 291.5702 | 42.3464 | 656.2131 | 2031 |
| 6 | 6 | concessional_envelope | 2.3435 | 2.3435 | env_10x (32.9603) | env_none (35.3038) | none | 6.696 | 5 | yes | 2 | 2.3435 | no | 0.0000 | 0.2740 | 0.0000 | 2030 |
| 7 | 7 | conversion | 2.1030 | 2.1030 | conv_high (33.4950) | conv_none (35.5980) | none | 6.009 | 4 | yes | 4 | 2.1030 | no | 439.0113 | 1269.6864 | 8755.7855 | 2025 |
| 8 | 8 | storage_cc | 1.6833 | 1.6833 | storage_cc_high (34.5683) | storage_cc_low (36.2516) | none | 4.810 | 3 | no | 3 | 1.6833 | no | 222.2899 | 11.4747 | 42641.2363 | 2031 |
| 9 | 9 | gas_capex | 1.6145 | 1.6145 | gas_capex_low (33.6391) | gas_capex_high (35.2535) | none | 4.613 | 3 | no | 3 | 1.6145 | no | 216.2297 | 16.1815 | 4033.6662 | 2031 |
| 10 | 11 | conversion_efficiency | 1.1760 | 1.1760 | conv_eff_high (34.0841) | conv_eff_low (35.2601) | none | 3.360 | 3 | no | 3 | 1.1760 | no | 393.2056 | 1108.5660 | 9029.4677 | 2025 |
| 11 | 12 | peak | 0.9333 | 0.9333 | peak_suppressed (34.6917) | peak_high (35.6250) | none | 2.667 | 3 | yes | 3 | 0.9333 | no | 123.4667 | 1.5555 | 18697.2049 | 2026 |
| 12 | 10 | gas_retirement | 0.7448 | 1.2478 | gas_retire_none (34.2507) | gas_retire_vintage_30 (34.9955) | gas_retire_proxy | 2.128 | 5 | yes | 5 | 1.2478 | no | 186.8642 | 7.6861 | 43131.7868 | 2031 |
| 13 | 13 | gas_deliverability_drift | 0.1288 | 0.1288 | baseline (34.9955) | downside (35.1244) | none | 0.368 | 4 | yes | 2 | 0.1288 | no | 3428.9488 | 51.7325 | 0.0000 | 2046 |

## Families whose arms produce an identical plan (rescaling; not ranked)

| family | span (bn) | low arm | high arm | % central | arms | in F-5 | max d solar MW / gas thermal draw 2025-54 TWh_th / storage MWh | first year |
|---|---|---|---|---|---|---|---|---|
| gas_deliverability_gmp | 0.0000 | baseline (34.9955) | baseline (34.9955) | 0.000 | 4 | yes | 0.0000 / 0.0000 / 0.0000 | none |
| dscr | 0.0000 | dscr_central (34.9955) | dscr_central (34.9955) | 0.000 | 3 | no | 0.0000 / 0.0000 / 0.0000 | none |

## Not comparable: arms differ in dispatch model (not ranked; TASK 83 A.2)

| family | span (bn) | low arm | high arm | arms | max d solar MW / gas thermal draw 2025-54 TWh_th / storage MWh | why it is not ranked |
|---|---|---|---|---|---|---|
| new_gas_deliverability | 6.1963 | constrained (34.9955) | exempt (41.1918) | 2 | 2052.1623 / 1322.6704 / 1586.1584 | the exempt arm cannot run vintage dispatch (the builder raises if it is on), so it is a single-stream arm compared with a vintage-dispatch constrained arm: the span is the dispatch-model difference, not the value of deliverability (capped, exempt equals the BLENDED constrained objective to the cent; L138, L145) |

## Bounds and conventions (do not enter the ranking)

Bound (`bound: True`), not a lever:

| family | span (bn) | low arm | high arm | % central | arms | plans | first year |
|---|---|---|---|---|---|---|---|
| hydro | 23.3577 | hydro_ndc (11.6378) | hydro_flat (34.9955) | 66.745 | 3 | 3 | 2025 |

## Declared but not ranked (missing member)

none

## Joint arms (differ from central in two or more fields; excluded)

none

## Values the files carry that no declared family covers (reported, not adopted)

| field | value | files | where |
|---|---|---|---|
| dscr_case | dscr_regression | 0 | a scenario-function key |
| gas_availability_case | gas_avail_central | 0 | a scenario-function key |
| gas_deliverability_case | flat_downside | 0 | a scenario-function key |
| gas_deliverability_case | flat_shock_recovery | 0 | a scenario-function key |
| gas_deliverability_case | flat_upside | 0 | a scenario-function key |
| peak_case | peak_nirp | 0 | a scenario-function key |
| tariff_case (stored; no declared family) | tariff_central | 42 | stored in files |

## Live-arm rule (column 3): every arm the rule could have touched

An arm is UNASSERTED when it is unsourced, represents no state or bound the thesis asserts, and is retained only so that superseded results remain attributable. Only UNASSERTED arms are excluded from column 3; the span with each other candidate excluded is shown for information and is not applied.

| family | arm | status | span all arms (bn) | span with this arm excluded (bn) | reason (declared in LEVER_FAMILIES) |
|---|---|---|---|---|---|
| gas_availability | `gas_avail_nameplate` | live | 10.3358 | undefined (fewer than two arms left) | LIVE: the central arm and the anchor of every result (BASELINE_CENTRAL_ARM); its 'retained' docstring wording does not make it unasserted |
| conversion | `conv_none` | live | 2.1030 | 1.9692 | LIVE: an attainable state (no conversion) and the cap builder's own baseline (Ch3 s3.7.6); 'regression arm' wording does not make it unasserted |
| gas_retirement | `gas_retire_proxy` | UNASSERTED | 1.2478 | 0.7448 | pre-2.11 'brownfield retirement proxy': unsourced [PENDING]; retained only so pre-2.11 results stay attributable; claims no state or bound |
| gas_retirement | `gas_retire_none` | live | 1.2478 | 1.2341 | LIVE: a declared upper bound on the surviving fleet; a bound is a claim |
| peak | `peak_suppressed` | live | 0.9333 | 0.6295 | LIVE: a declared floor (L2) |
| hydro | `hydro_flat` | live | 23.3577 | 20.5435 | LIVE: the central arm and anchor; 'regression arm' wording does not make it unasserted |
| gas_build | `gas_none` | live | 7.5550 | 1.8376 | LIVE: an attainable policy state; the span to it measures the value of being permitted to build new gas |
| gas_capex | `gas_capex_low` | live | 1.6145 | 0.2580 | LIVE: a DECLARED LOWER BOUND; a bound is a claim; attribution wording alone never makes an arm unasserted |
| om | `om_none` | live | 8.3687 | 0.8891 | NOT unasserted: SYNTHETIC and unattainable; the family is already outside the ranked table by the build-response rule |

## Column-2 exclusions

* `gas_deliverability_gmp`: excluded baseline, gas_deliv_flat, gas_deliv_gmp_partial, gas_deliv_gmp_full
* `dscr`: excluded dscr_central, dscr_low, dscr_high

## Per-arm cap binding under `ndc3_conditional` (TASK 82 B.4)

`cap_binding` is True where the annual cap's shadow price (`carbon_shadow_yr_usd_per_tco2_by_year`) is non-zero (|x| > 1e-6 USD/tCO2) in at least one year. `unc>cap yrs` counts the years in which the UNCAPPED counterpart's total emissions exceed the cap: if it is 0 the uncapped plan already meets the cap in every year. The capped runs also carry a public-capital envelope the uncapped run does not (`capital_case` tight/moderate), so `capped - uncapped` is cap plus envelope, not cap alone.

| family | member | capped (bn) | uncapped (bn) | capped - uncapped (bn) | cap_binding | binding yrs | first-last | unc>cap yrs | emissions Mt | leakage share | leakage flagged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gas_availability | gas_avail_nameplate | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| gas_availability | gas_avail_observed | 45.3313 | 31.8127 | 13.5187 | **yes** | 26 | 2029-2054 | 18 | 343.7 | 0.1434 | True |
| conversion | conv_none | 35.5980 | 25.3036 | 10.2944 | **yes** | 30 | 2025-2054 | 29 | 336.5 | 0.0943 | False |
| conversion | conv_low | 35.4642 | 24.5536 | 10.9106 | **yes** | 29 | 2026-2054 | 29 | 335.7 | 0.0962 | False |
| conversion | conv_central | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| conversion | conv_high | 33.4950 | 22.2129 | 11.2821 | **yes** | 26 | 2029-2054 | 26 | 328.5 | 0.1101 | True |
| conversion_efficiency | conv_eff_low | 35.2601 | 23.9542 | 11.3059 | **yes** | 29 | 2026-2054 | 27 | 335.6 | 0.0963 | False |
| conversion_efficiency | conv_central | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| conversion_efficiency | conv_eff_high | 34.0841 | 22.8020 | 11.2821 | **yes** | 26 | 2029-2054 | 26 | 328.5 | 0.1101 | True |
| gas_retirement | gas_retire_none | 34.2507 | 22.2846 | 11.9661 | **yes** | 29 | 2026-2054 | 27 | 335.8 | 0.0974 | False |
| gas_retirement | gas_retire_vintage_30 | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| gas_retirement | gas_retire_vintage_40 | 34.2644 | 22.6851 | 11.5793 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| gas_retirement | gas_retire_vintage_50 | 34.4636 | 22.8144 | 11.6492 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| gas_retirement | gas_retire_proxy | 35.4985 | 24.2368 | 11.2616 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| capital_recovery | cr_central | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| capital_recovery | cr_ida21 | 34.9958 | 23.3232 | 11.6726 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| capital_recovery | cr_comm_low | 33.0264 | 21.8392 | 11.1871 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| capital_recovery | cr_comm_high | 36.6046 | 24.5144 | 12.0903 | **yes** | 29 | 2026-2054 | 27 | 335.8 | 0.0974 | False |
| gas_deliverability_drift | baseline | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| gas_deliverability_drift | upside | 34.9955 | 22.2141 | 12.7814 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| gas_deliverability_drift | downside | 35.1244 | 26.0098 | 9.1146 | **yes** | 20 | 2026-2045 | 18 | 325.8 | 0.0991 | False |
| gas_deliverability_drift | shock_recovery | 34.9955 | 23.6752 | 11.3203 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| gas_deliverability_gmp | baseline | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| gas_deliverability_gmp | gas_deliv_flat | 34.9955 | 23.7532 | 11.2423 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| gas_deliverability_gmp | gas_deliv_gmp_partial | 34.9955 | 22.1210 | 12.8746 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| gas_deliverability_gmp | gas_deliv_gmp_full | 34.9955 | 21.7718 | 13.2237 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| concessional_envelope | env_none | 35.3038 | 23.5449 | 11.7588 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| concessional_envelope | env_dares | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| concessional_envelope | env_5x | 33.9886 | 22.6055 | 11.3830 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| concessional_envelope | env_10x | 32.9603 | 21.9326 | 11.0277 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| concessional_envelope | env_dares_plus | 34.8722 | 23.2388 | 11.6334 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| peak | peak_suppressed | 34.6917 | 22.8557 | 11.8360 | **yes** | 29 | 2026-2054 | 27 | 335.5 | 0.0967 | False |
| peak | peak_unsuppressed | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| peak | peak_high | 35.6250 | 23.9612 | 11.6638 | **yes** | 29 | 2026-2054 | 27 | 335.5 | 0.0966 | False |
| hydro | hydro_flat | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| hydro | hydro_committed | 32.1813 | 22.5841 | 9.5972 | **yes** | 28 | 2027-2054 | 26 | 325.4 | 0.0735 | False |
| hydro | hydro_ndc | 11.6378 | 10.8847 | 0.7532 | **yes** | 15 | 2040-2054 | 15 | 209.3 | 0.0000 | False |
| new_gas_deliverability | constrained | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| new_gas_deliverability | exempt | 41.1918 | 23.6014 | 17.5904 | **yes** | 26 | 2029-2054 | 27 | 350.5 | 0.1670 | True |
| storage_cc | storage_cc_low | 36.2516 | 24.4898 | 11.7618 | **yes** | 29 | 2026-2054 | 27 | 335.0 | 0.0954 | False |
| storage_cc | storage_cc_central | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| storage_cc | storage_cc_high | 34.5683 | 22.9251 | 11.6433 | **yes** | 29 | 2026-2054 | 27 | 335.8 | 0.0974 | False |
| gas_build | gas_unconstrained | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| gas_build | gas_ndc_17gw | 36.8331 | 23.3231 | 13.5101 | **yes** | 27 | 2028-2054 | 27 | 331.9 | 0.1120 | True |
| gas_build | gas_none | 42.5505 | 25.4694 | 17.0812 | **yes** | 20 | 2028-2047 | 20 | 350.5 | 0.1963 | True |
| gas_capex | gas_capex_low | 33.6391 | 22.5578 | 11.0813 | **yes** | 29 | 2026-2054 | 27 | 334.8 | 0.0948 | False |
| gas_capex | gas_capex_central | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| gas_capex | gas_capex_high | 35.2535 | 23.4730 | 11.7805 | **yes** | 29 | 2026-2054 | 27 | 335.8 | 0.0974 | False |
| ccgt_efficiency | ccgt_eff_central | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| ccgt_efficiency | ccgt_eff_iso | 32.5914 | 23.1186 | 9.4729 | **yes** | 29 | 2026-2054 | 27 | 325.6 | 0.0693 | False |
| ccgt_efficiency | ccgt_eff_low | 36.5945 | 23.4495 | 13.1450 | **yes** | 29 | 2026-2054 | 27 | 342.0 | 0.1138 | True |
| dscr | dscr_central | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| dscr | dscr_low | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| dscr | dscr_high | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| om | om_none | 27.1668 | 16.7464 | 10.4204 | **yes** | 29 | 2026-2054 | 28 | 335.3 | 0.0962 | False |
| om | om_low | 34.6464 | 23.0394 | 11.6069 | **yes** | 29 | 2026-2054 | 27 | 335.8 | 0.0974 | False |
| om | om_central | 34.9955 | 23.3231 | 11.6725 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |
| om | om_high | 35.5355 | 23.7399 | 11.7956 | **yes** | 29 | 2026-2054 | 27 | 335.3 | 0.0962 | False |

Arms (family, member rows) where the cap does not bind: 0 of 58.

## Solves that did not return an optimal plan

none: every driver invocation exited 0 (02_ raises on a non-optimal status).

## Uncapped rank against capped rank (TASK 82 B.7), column 3 (live arms)

Uncapped: `results/lever_ranking/arms_661b03c1/lever_ranking.csv`. Capped: this table's `ndc3_conditional` ranking. A family is ranked only where its arms produce different plans; otherwise its role is shown.

| family | uncapped rank | uncapped span (bn) | capped rank | capped span (bn) | capped role |
|---|---|---|---|---|---|
| gas_availability | 1 | 8.4896 | 1 | 10.3358 | ranked |
| conversion | 4 | 3.0907 | 7 | 2.1030 | ranked |
| conversion_efficiency | 10 | 1.1521 | 10 | 1.1760 | ranked |
| gas_retirement | 12 | 1.0385 | 12 | 0.7448 | ranked |
| capital_recovery | 5 | 2.6751 | 5 | 3.5783 | ranked |
| gas_deliverability_drift | 3 | 3.7957 | 13 | 0.1288 | ranked |
| gas_deliverability_gmp | 7 | 1.9814 | - | 0.0000 | no_build_response |
| concessional_envelope | 8 | 1.6123 | 6 | 2.3435 | ranked |
| peak | 11 | 1.1055 | 11 | 0.9333 | ranked |
| hydro | - | - | - | 23.3577 | bound |
| new_gas_deliverability | - | - | - | 6.1963 | not_comparable |
| storage_cc | 9 | 1.5648 | 8 | 1.6833 | ranked |
| gas_build | 6 | 2.1463 | 3 | 7.5550 | ranked |
| gas_capex | 13 | 0.9153 | 9 | 1.6145 | ranked |
| ccgt_efficiency | 14 | 0.3309 | 4 | 4.0030 | ranked |
| dscr | - | - | - | 0.0000 | no_build_response |
| om | 2 | 6.9935 | 2 | 8.3687 | ranked |
