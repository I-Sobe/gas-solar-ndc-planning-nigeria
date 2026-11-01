# Lever ranking -- post-epoch, generated 2026-10-23T22:10:24 at HEAD `dabd2a7` (src/scripts have uncommitted changes) by `scripts/build_lever_ranking.py`

Central `60585e1f` objective 26,672,293,112.76 USD. Files read 42, admitted 42. A verification artefact, not a finding.

## Ranked levers (arms produce different build decisions)

Ordered by column 3 (live arms). `rank (col 1)` is the rank by all-arms span.

| rank (col 3) | rank (col 1) | family | span col 3 live (bn) | span col 1 all arms (bn) | col 3 low arm | col 3 high arm | unasserted (excluded) | % central (col 3) | arms | in F-5 | plans | col 2 (bn) | col 2 differs | max d solar 2054 MW | max d gas thermal draw 2025-54 TWh_th | max d storage 2054 MWh | first year |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | gas_availability | 9.2774 | 9.2774 | gas_avail_nameplate (26.6723) | gas_avail_observed (35.9497) | none | 34.783 | 2 | yes | 2 | 9.2774 | no | 2675.3571 | 1093.2322 | 17077.0920 | 2025 |
| 2 | 2 | gas_build | 8.9587 | 8.9587 | gas_unconstrained (26.6723) | gas_none (35.6310) | none | 33.588 | 3 | no | 3 | 8.9587 | no | 9953.1775 | 1196.8029 | 16054.8638 | 2025 |
| 3 | 3 | om | 8.2924 | 8.2924 | om_none (18.8718) | om_high (27.1642) | none | 31.090 | 4 | no | 3 | 8.2924 | no | 64.1957 | 119.9215 | 444.5450 | 2032 |
| 4 | 4 | capital_recovery | 3.3370 | 3.3370 | cr_comm_low (24.8348) | cr_comm_high (28.1718) | none | 12.511 | 4 | yes | 3 | 3.3370 | no | 174.2077 | 31.1014 | 0.0000 | 2035 |
| 5 | 5 | conversion | 2.9041 | 2.9041 | conv_high (24.7011) | conv_none (27.6052) | none | 10.888 | 4 | yes | 4 | 2.9041 | no | 248.0274 | 1375.8163 | 7470.1711 | 2025 |
| 6 | 6 | concessional_envelope | 2.2569 | 2.2569 | env_10x (24.6851) | env_none (26.9420) | none | 8.462 | 5 | yes | 2 | 2.2569 | no | 0.0000 | 9.4421 | 0.0000 | 2031 |
| 7 | 7 | ccgt_efficiency | 1.7996 | 1.7996 | ccgt_eff_iso (25.8582) | ccgt_eff_low (27.6578) | none | 6.747 | 3 | no | 3 | 1.7996 | no | 1016.4796 | 313.7722 | 2596.3616 | 2025 |
| 8 | 8 | conversion_efficiency | 1.7062 | 1.7062 | conv_eff_high (25.2902) | conv_eff_low (26.9964) | none | 6.397 | 3 | no | 3 | 1.7062 | no | 116.3853 | 1124.9232 | 7116.1993 | 2025 |
| 9 | 9 | storage_cc | 1.6628 | 1.6628 | storage_cc_high (26.2561) | storage_cc_low (27.9189) | none | 6.234 | 3 | no | 3 | 1.6628 | no | 324.7119 | 41.6082 | 43286.6391 | 2032 |
| 10 | 10 | gas_capex | 1.5965 | 1.5965 | gas_capex_low (25.3372) | gas_capex_high (26.9337) | none | 5.985 | 3 | no | 3 | 1.5965 | no | 135.4487 | 31.8300 | 1164.4990 | 2031 |
| 11 | 12 | peak | 1.0029 | 1.0029 | peak_suppressed (26.3577) | peak_high (27.3606) | none | 3.760 | 3 | yes | 3 | 1.0029 | no | 86.2145 | 21.5043 | 17950.5785 | 2025 |
| 12 | 11 | gas_retirement | 0.7448 | 1.2677 | gas_retire_none (25.9275) | gas_retire_vintage_30 (26.6723) | gas_retire_proxy | 2.792 | 5 | yes | 5 | 1.2677 | no | 206.3147 | 89.3108 | 41655.7875 | 2025 |
| 13 | 13 | gas_deliverability_drift | 0.6582 | 0.6582 | baseline (26.6723) | downside (27.3305) | none | 2.468 | 4 | yes | 2 | 0.6582 | no | 7570.2959 | 201.6563 | 0.0000 | 2038 |

## Families whose arms produce an identical plan (rescaling; not ranked)

| family | span (bn) | low arm | high arm | % central | arms | in F-5 | max d solar MW / gas thermal draw 2025-54 TWh_th / storage MWh | first year |
|---|---|---|---|---|---|---|---|---|
| gas_deliverability_gmp | 0.0000 | baseline (26.6723) | baseline (26.6723) | 0.000 | 4 | yes | 0.0000 / 0.0000 / 0.0000 | none |
| dscr | 0.0000 | dscr_central (26.6723) | dscr_central (26.6723) | 0.000 | 3 | no | 0.0000 / 0.0000 / 0.0000 | none |

## Not comparable: arms differ in dispatch model (not ranked; TASK 83 A.2)

| family | span (bn) | low arm | high arm | arms | max d solar MW / gas thermal draw 2025-54 TWh_th / storage MWh | why it is not ranked |
|---|---|---|---|---|---|---|
| new_gas_deliverability | 7.2874 | constrained (26.6723) | exempt (33.9597) | 2 | 3130.2858 / 1354.0702 / 1251.3021 | the exempt arm cannot run vintage dispatch (the builder raises if it is on), so it is a single-stream arm compared with a vintage-dispatch constrained arm: the span is the dispatch-model difference, not the value of deliverability (capped, exempt equals the BLENDED constrained objective to the cent; L138, L145) |

## Bounds and conventions (do not enter the ranking)

Bound (`bound: True`), not a lever:

| family | span (bn) | low arm | high arm | % central | arms | plans | first year |
|---|---|---|---|---|---|---|---|
| hydro | 15.3490 | hydro_ndc (11.3233) | hydro_flat (26.6723) | 57.547 | 3 | 3 | 2025 |

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
| gas_availability | `gas_avail_nameplate` | live | 9.2774 | undefined (fewer than two arms left) | LIVE: the central arm and the anchor of every result (BASELINE_CENTRAL_ARM); its 'retained' docstring wording does not make it unasserted |
| conversion | `conv_none` | live | 2.9041 | 2.5114 | LIVE: an attainable state (no conversion) and the cap builder's own baseline (Ch3 s3.7.6); 'regression arm' wording does not make it unasserted |
| gas_retirement | `gas_retire_proxy` | UNASSERTED | 1.2677 | 0.7448 | pre-2.11 'brownfield retirement proxy': unsourced [PENDING]; retained only so pre-2.11 results stay attributable; claims no state or bound |
| gas_retirement | `gas_retire_none` | live | 1.2677 | 1.2519 | LIVE: a declared upper bound on the surviving fleet; a bound is a claim |
| peak | `peak_suppressed` | live | 1.0029 | 0.6883 | LIVE: a declared floor (L2) |
| hydro | `hydro_flat` | live | 15.3490 | 13.8515 | LIVE: the central arm and anchor; 'regression arm' wording does not make it unasserted |
| gas_build | `gas_none` | live | 8.9587 | 0.5686 | LIVE: an attainable policy state; the span to it measures the value of being permitted to build new gas |
| gas_capex | `gas_capex_low` | live | 1.5965 | 0.2614 | LIVE: a DECLARED LOWER BOUND; a bound is a claim; attribution wording alone never makes an arm unasserted |
| om | `om_none` | live | 8.2924 | 0.8077 | NOT unasserted: SYNTHETIC and unattainable; the family is already outside the ranked table by the build-response rule |

## Column-2 exclusions

* `gas_deliverability_gmp`: excluded baseline, gas_deliv_flat, gas_deliv_gmp_partial, gas_deliv_gmp_full
* `dscr`: excluded dscr_central, dscr_low, dscr_high

## Per-arm cap binding under `ndc3_unconditional` (TASK 82 B.4)

`cap_binding` is True where the annual cap's shadow price (`carbon_shadow_yr_usd_per_tco2_by_year`) is non-zero (|x| > 1e-6 USD/tCO2) in at least one year. `unc>cap yrs` counts the years in which the UNCAPPED counterpart's total emissions exceed the cap: if it is 0 the uncapped plan already meets the cap in every year. The capped runs also carry a public-capital envelope the uncapped run does not (`capital_case` tight/moderate), so `capped - uncapped` is cap plus envelope, not cap alone.

| family | member | capped (bn) | uncapped (bn) | capped - uncapped (bn) | cap_binding | binding yrs | first-last | unc>cap yrs | emissions Mt | leakage share | leakage flagged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gas_availability | gas_avail_nameplate | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| gas_availability | gas_avail_observed | 35.9497 | 31.8127 | 4.1370 | **yes** | 18 | 2030-2047 | 14 | 361.2 | 0.0252 | False |
| conversion | conv_none | 27.6052 | 25.3036 | 2.3016 | **yes** | 30 | 2025-2054 | 29 | 367.0 | 0.0000 | False |
| conversion | conv_low | 27.2125 | 24.5536 | 2.6589 | **yes** | 29 | 2026-2054 | 29 | 365.8 | 0.0005 | False |
| conversion | conv_central | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| conversion | conv_high | 24.7011 | 22.2129 | 2.4882 | **yes** | 26 | 2029-2054 | 26 | 353.9 | 0.0029 | False |
| conversion_efficiency | conv_eff_low | 26.9964 | 23.9542 | 3.0422 | **yes** | 29 | 2026-2054 | 27 | 365.7 | 0.0005 | False |
| conversion_efficiency | conv_central | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| conversion_efficiency | conv_eff_high | 25.2902 | 22.8020 | 2.4882 | **yes** | 26 | 2029-2054 | 26 | 353.9 | 0.0029 | False |
| gas_retirement | gas_retire_none | 25.9275 | 22.2846 | 3.6429 | **yes** | 28 | 2027-2054 | 26 | 365.0 | 0.0022 | False |
| gas_retirement | gas_retire_vintage_30 | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| gas_retirement | gas_retire_vintage_40 | 25.9433 | 22.6851 | 3.2582 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| gas_retirement | gas_retire_vintage_50 | 26.1419 | 22.8144 | 3.3275 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| gas_retirement | gas_retire_proxy | 27.1952 | 24.2368 | 2.9584 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| capital_recovery | cr_central | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| capital_recovery | cr_ida21 | 26.6725 | 23.3232 | 3.3493 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| capital_recovery | cr_comm_low | 24.8348 | 21.8392 | 2.9956 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| capital_recovery | cr_comm_high | 28.1718 | 24.5144 | 3.6574 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| gas_deliverability_drift | baseline | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| gas_deliverability_drift | upside | 26.6723 | 22.2141 | 4.4582 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| gas_deliverability_drift | downside | 27.3305 | 26.0098 | 1.3207 | **yes** | 11 | 2027-2037 | 9 | 324.0 | 0.0011 | False |
| gas_deliverability_drift | shock_recovery | 26.6723 | 23.6752 | 2.9970 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| gas_deliverability_gmp | baseline | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| gas_deliverability_gmp | gas_deliv_flat | 26.6723 | 23.7532 | 2.9191 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| gas_deliverability_gmp | gas_deliv_gmp_partial | 26.6723 | 22.1210 | 4.5513 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| gas_deliverability_gmp | gas_deliv_gmp_full | 26.6723 | 21.7718 | 4.9005 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| concessional_envelope | env_none | 26.9420 | 23.5449 | 3.3970 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| concessional_envelope | env_dares | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| concessional_envelope | env_5x | 25.6957 | 22.6055 | 3.0901 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| concessional_envelope | env_10x | 24.6851 | 21.9326 | 2.7524 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| concessional_envelope | env_dares_plus | 26.5644 | 23.2388 | 3.3256 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| peak | peak_suppressed | 26.3577 | 22.8557 | 3.5020 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| peak | peak_unsuppressed | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| peak | peak_high | 27.3606 | 23.9612 | 3.3994 | **yes** | 28 | 2027-2054 | 26 | 365.3 | 0.0007 | False |
| hydro | hydro_flat | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| hydro | hydro_committed | 25.1748 | 22.5841 | 2.5907 | **yes** | 27 | 2028-2054 | 26 | 358.5 | 0.0006 | False |
| hydro | hydro_ndc | 11.3233 | 10.8847 | 0.4386 | **yes** | 10 | 2045-2054 | 11 | 237.8 | 0.0000 | False |
| new_gas_deliverability | constrained | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| new_gas_deliverability | exempt | 33.9597 | 23.6014 | 10.3583 | **yes** | 26 | 2029-2054 | 26 | 380.2 | 0.0737 | False |
| storage_cc | storage_cc_low | 27.9189 | 24.4898 | 3.4290 | **yes** | 28 | 2027-2054 | 26 | 365.3 | 0.0006 | False |
| storage_cc | storage_cc_central | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| storage_cc | storage_cc_high | 26.2561 | 22.9251 | 3.3310 | **yes** | 28 | 2027-2054 | 26 | 365.7 | 0.0016 | False |
| gas_build | gas_unconstrained | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| gas_build | gas_ndc_17gw | 27.2409 | 23.3231 | 3.9178 | **yes** | 26 | 2029-2054 | 26 | 356.4 | 0.0023 | False |
| gas_build | gas_none | 35.6310 | 25.4694 | 10.1616 | **yes** | 17 | 2029-2045 | 17 | 361.1 | 0.1101 | True |
| gas_capex | gas_capex_low | 25.3372 | 22.5578 | 2.7794 | **yes** | 28 | 2027-2054 | 26 | 365.3 | 0.0004 | False |
| gas_capex | gas_capex_central | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| gas_capex | gas_capex_high | 26.9337 | 23.4730 | 3.4606 | **yes** | 28 | 2027-2054 | 26 | 365.7 | 0.0016 | False |
| ccgt_efficiency | ccgt_eff_central | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| ccgt_efficiency | ccgt_eff_iso | 25.8582 | 23.1186 | 2.7397 | **yes** | 28 | 2027-2054 | 26 | 362.6 | 0.0000 | False |
| ccgt_efficiency | ccgt_eff_low | 27.6578 | 23.4495 | 4.2083 | **yes** | 28 | 2027-2054 | 26 | 368.2 | 0.0084 | False |
| dscr | dscr_central | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| dscr | dscr_low | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| dscr | dscr_high | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| om | om_none | 18.8718 | 16.7464 | 2.1254 | **yes** | 28 | 2027-2054 | 28 | 365.3 | 0.0006 | False |
| om | om_low | 26.3565 | 23.0394 | 3.3171 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| om | om_central | 26.6723 | 23.3231 | 3.3492 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |
| om | om_high | 27.1642 | 23.7399 | 3.4244 | **yes** | 28 | 2027-2054 | 26 | 365.5 | 0.0010 | False |

Arms (family, member rows) where the cap does not bind: 0 of 58.

## Solves that did not return an optimal plan

none: every driver invocation exited 0 (02_ raises on a non-optimal status).

## Uncapped rank against capped rank (TASK 82 B.7), column 3 (live arms)

Uncapped: `results/lever_ranking/arms_661b03c1/lever_ranking.csv`. Capped: this table's `ndc3_unconditional` ranking. A family is ranked only where its arms produce different plans; otherwise its role is shown.

| family | uncapped rank | uncapped span (bn) | capped rank | capped span (bn) | capped role |
|---|---|---|---|---|---|
| gas_availability | 1 | 8.4896 | 1 | 9.2774 | ranked |
| conversion | 4 | 3.0907 | 5 | 2.9041 | ranked |
| conversion_efficiency | 10 | 1.1521 | 8 | 1.7062 | ranked |
| gas_retirement | 12 | 1.0385 | 12 | 0.7448 | ranked |
| capital_recovery | 5 | 2.6751 | 4 | 3.3370 | ranked |
| gas_deliverability_drift | 3 | 3.7957 | 13 | 0.6582 | ranked |
| gas_deliverability_gmp | 7 | 1.9814 | - | 0.0000 | no_build_response |
| concessional_envelope | 8 | 1.6123 | 6 | 2.2569 | ranked |
| peak | 11 | 1.1055 | 11 | 1.0029 | ranked |
| hydro | - | - | - | 15.3490 | bound |
| new_gas_deliverability | - | - | - | 7.2874 | not_comparable |
| storage_cc | 9 | 1.5648 | 9 | 1.6628 | ranked |
| gas_build | 6 | 2.1463 | 2 | 8.9587 | ranked |
| gas_capex | 13 | 0.9153 | 10 | 1.5965 | ranked |
| ccgt_efficiency | 14 | 0.3309 | 7 | 1.7996 | ranked |
| dscr | - | - | - | 0.0000 | no_build_response |
| om | 2 | 6.9935 | 3 | 8.2924 | ranked |
