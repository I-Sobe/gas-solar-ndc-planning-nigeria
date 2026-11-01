# Lever ranking -- post-epoch, generated 2026-10-23T21:50:48 at HEAD `17f8436` (src/scripts have uncommitted changes) by `scripts/build_lever_ranking.py`

Central `3fbc75ac` objective 36,140,487,294.76 USD. Files read 57, admitted 42. A verification artefact, not a finding.

## Ranked levers (arms produce different build decisions)

Ordered by column 3 (live arms). `rank (col 1)` is the rank by all-arms span.

| rank (col 3) | rank (col 1) | family | span col 3 live (bn) | span col 1 all arms (bn) | col 3 low arm | col 3 high arm | unasserted (excluded) | % central (col 3) | arms | in F-5 | plans | col 2 (bn) | col 2 differs | max d solar 2054 MW | max d gas thermal draw 2025-54 TWh_th | max d storage 2054 MWh | first year |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | gas_build | 21.3464 | 21.3464 | gas_unconstrained (36.1405) | gas_none (57.4869) | none | 59.065 | 3 | no | 3 | 21.3464 | no | 18386.4672 | 1302.7428 | 58032.3305 | 2025 |
| 2 | 2 | gas_availability | 20.7626 | 20.7626 | gas_avail_nameplate (36.1405) | gas_avail_observed (56.9030) | none | 57.450 | 2 | yes | 2 | 20.7626 | no | 7402.5121 | 214.5498 | 12822.9742 | 2025 |
| 3 | 3 | new_gas_deliverability | 9.7426 | 9.7426 | constrained (36.1405) | exempt (45.8831) | none | 26.958 | 2 | no | 2 | 9.7426 | no | 6156.7715 | 1429.3979 | 1049.4108 | 2025 |
| 4 | 4 | om | 8.2500 | 8.2500 | om_none (28.2629) | om_high (36.5129) | none | 22.828 | 4 | no | 4 | 8.2500 | no | 1196.0358 | 25.7201 | 2770.1272 | 2030 |
| 5 | 5 | ccgt_efficiency | 4.5931 | 4.5931 | ccgt_eff_iso (33.3819) | ccgt_eff_low (37.9750) | none | 12.709 | 3 | no | 3 | 4.5931 | no | 1749.7355 | 13.3783 | 5573.2845 | 2030 |
| 6 | 6 | capital_recovery | 3.9607 | 3.9607 | cr_comm_low (33.9577) | cr_comm_high (37.9184) | none | 10.959 | 4 | yes | 3 | 3.9607 | no | 830.5631 | 15.3007 | 4701.7263 | 2030 |
| 7 | 7 | gas_capex | 2.5168 | 2.5168 | gas_capex_low (34.0288) | gas_capex_high (36.5456) | none | 6.964 | 3 | no | 3 | 2.5168 | no | 2108.6414 | 23.9683 | 181.0243 | 2030 |
| 8 | 9 | concessional_envelope | 2.3434 | 2.3434 | env_10x (34.1053) | env_none (36.4487) | none | 6.484 | 5 | yes | 3 | 2.3434 | no | 0.0000 | 0.0000 | 0.0000 | 2052 |
| 9 | 10 | conversion | 1.6465 | 1.6465 | conv_high (35.0575) | conv_none (36.7040) | none | 4.556 | 4 | yes | 4 | 1.6465 | no | 230.3816 | 920.4758 | 1909.8053 | 2025 |
| 10 | 11 | peak | 1.5675 | 1.5675 | peak_suppressed (35.5636) | peak_high (37.1311) | none | 4.337 | 3 | yes | 3 | 1.5675 | no | 331.0009 | 58.9661 | 969.0341 | 2026 |
| 11 | 8 | gas_retirement | 1.4548 | 2.4120 | gas_retire_none (34.6857) | gas_retire_vintage_30 (36.1405) | gas_retire_proxy | 4.025 | 5 | yes | 5 | 2.4120 | no | 1336.8060 | 103.8095 | 2843.2599 | 2030 |
| 12 | 12 | storage_cc | 0.8723 | 0.8723 | storage_cc_high (35.7594) | storage_cc_low (36.6317) | none | 2.414 | 3 | no | 3 | 0.8723 | no | 2662.4233 | 14.6729 | 824.6468 | 2030 |
| 13 | 13 | gas_deliverability_drift | 0.7053 | 0.7053 | baseline (36.1405) | downside (36.8458) | none | 1.952 | 4 | yes | 2 | 0.7053 | no | 844.2642 | 27.8248 | 51.6055 | 2030 |
| 14 | 14 | conversion_efficiency | 0.6859 | 0.6859 | conv_eff_high (35.6256) | conv_eff_low (36.3115) | none | 1.898 | 3 | no | 3 | 0.6859 | no | 169.1173 | 548.8880 | 1593.4159 | 2025 |

## Families whose arms produce an identical plan (rescaling; not ranked)

| family | span (bn) | low arm | high arm | % central | arms | in F-5 | max d solar MW / gas thermal draw 2025-54 TWh_th / storage MWh | first year |
|---|---|---|---|---|---|---|---|---|
| gas_deliverability_gmp | 0.0000 | baseline (36.1405) | baseline (36.1405) | 0.000 | 4 | yes | 0.0000 / 0.0000 / 0.0000 | none |
| dscr | 0.0000 | dscr_central (36.1405) | dscr_central (36.1405) | 0.000 | 3 | no | 0.0000 / 0.0000 / 0.0000 | none |

## Bounds and conventions (do not enter the ranking)

Bound (`bound: True`), not a lever:

| family | span (bn) | low arm | high arm | % central | arms | plans | first year |
|---|---|---|---|---|---|---|---|
| hydro | 24.5026 | hydro_ndc (11.6378) | hydro_flat (36.1405) | 67.798 | 3 | 3 | 2025 |

## Declared but not ranked (missing member)

none

## Joint arms (differ from central in two or more fields; excluded)

none

## Values the files carry that no declared family covers (reported, not adopted)

| field | value | files | where |
|---|---|---|---|
| dscr_case | dscr_regression | 5 | a scenario-function key |
| gas_availability_case | gas_avail_central | 0 | a scenario-function key |
| gas_deliverability_case | flat_downside | 0 | a scenario-function key |
| gas_deliverability_case | flat_shock_recovery | 0 | a scenario-function key |
| gas_deliverability_case | flat_upside | 0 | a scenario-function key |
| peak_case | peak_nirp | 0 | a scenario-function key |
| tariff_case (stored; no declared family) | tariff_central | 51 | stored in files |

## Live-arm rule (column 3): every arm the rule could have touched

An arm is UNASSERTED when it is unsourced, represents no state or bound the thesis asserts, and is retained only so that superseded results remain attributable. Only UNASSERTED arms are excluded from column 3; the span with each other candidate excluded is shown for information and is not applied.

| family | arm | status | span all arms (bn) | span with this arm excluded (bn) | reason (declared in LEVER_FAMILIES) |
|---|---|---|---|---|---|
| gas_availability | `gas_avail_nameplate` | live | 20.7626 | undefined (fewer than two arms left) | LIVE: the central arm and the anchor of every result (BASELINE_CENTRAL_ARM); its 'retained' docstring wording does not make it unasserted |
| conversion | `conv_none` | live | 1.6465 | 1.4672 | LIVE: an attainable state (no conversion) and the cap builder's own baseline (Ch3 s3.7.6); 'regression arm' wording does not make it unasserted |
| gas_retirement | `gas_retire_proxy` | UNASSERTED | 2.4120 | 1.4548 | pre-2.11 'brownfield retirement proxy': unsourced [PENDING]; retained only so pre-2.11 results stay attributable; claims no state or bound |
| gas_retirement | `gas_retire_none` | live | 2.4120 | 1.9593 | LIVE: a declared upper bound on the surviving fleet; a bound is a claim |
| peak | `peak_suppressed` | live | 1.5675 | 0.9906 | LIVE: a declared floor (L2) |
| hydro | `hydro_flat` | live | 24.5026 | 21.3907 | LIVE: the central arm and anchor; 'regression arm' wording does not make it unasserted |
| gas_build | `gas_none` | live | 21.3464 | 2.0762 | LIVE: an attainable policy state; the span to it measures the value of being permitted to build new gas |
| gas_capex | `gas_capex_low` | live | 2.5168 | 0.4051 | LIVE: a DECLARED LOWER BOUND; a bound is a claim; attribution wording alone never makes an arm unasserted |
| om | `om_none` | live | 8.2500 | 0.6240 | NOT unasserted: SYNTHETIC and unattainable; the family is already outside the ranked table by the build-response rule |

## Column-2 exclusions

* `gas_deliverability_gmp`: excluded baseline, gas_deliv_flat, gas_deliv_gmp_partial, gas_deliv_gmp_full
* `dscr`: excluded dscr_central, dscr_low, dscr_high

## Per-arm cap binding under `ndc3_conditional` (TASK 82 B.4)

`cap_binding` is True where the annual cap's shadow price (`carbon_shadow_yr_usd_per_tco2_by_year`) is non-zero (|x| > 1e-6 USD/tCO2) in at least one year. `unc>cap yrs` counts the years in which the UNCAPPED counterpart's total emissions exceed the cap: if it is 0 the uncapped plan already meets the cap in every year. The capped runs also carry a public-capital envelope the uncapped run does not (`capital_case` tight/moderate), so `capped - uncapped` is cap plus envelope, not cap alone.

| family | member | capped (bn) | uncapped (bn) | capped - uncapped (bn) | cap_binding | binding yrs | first-last | unc>cap yrs | emissions Mt | leakage share | leakage flagged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gas_availability | gas_avail_nameplate | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| gas_availability | gas_avail_observed | 56.9030 | 31.8127 | 25.0904 | **yes** | 25 | 2030-2054 | 18 | 356.2 | 0.1889 | True |
| conversion | conv_none | 36.7040 | 25.3036 | 11.4004 | **yes** | 30 | 2025-2054 | 29 | 339.8 | 0.1031 | True |
| conversion | conv_low | 36.5247 | 24.5536 | 11.9712 | **yes** | 29 | 2026-2054 | 29 | 338.6 | 0.1039 | True |
| conversion | conv_central | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| conversion | conv_high | 35.0575 | 22.2129 | 12.8446 | **yes** | 26 | 2029-2054 | 26 | 330.9 | 0.1165 | True |
| conversion_efficiency | conv_eff_low | 36.3115 | 23.9542 | 12.3573 | **yes** | 29 | 2026-2054 | 27 | 338.5 | 0.1039 | True |
| conversion_efficiency | conv_central | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| conversion_efficiency | conv_eff_high | 35.6256 | 22.8020 | 12.8235 | **yes** | 27 | 2028-2054 | 26 | 334.9 | 0.1121 | True |
| gas_retirement | gas_retire_none | 34.6857 | 22.2846 | 12.4011 | **yes** | 29 | 2026-2054 | 27 | 338.9 | 0.1058 | True |
| gas_retirement | gas_retire_vintage_30 | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| gas_retirement | gas_retire_vintage_40 | 35.1976 | 22.6851 | 12.5126 | **yes** | 29 | 2026-2054 | 27 | 337.6 | 0.1023 | True |
| gas_retirement | gas_retire_vintage_50 | 35.1383 | 22.8144 | 12.3239 | **yes** | 29 | 2026-2054 | 27 | 336.8 | 0.1001 | True |
| gas_retirement | gas_retire_proxy | 37.0977 | 24.2368 | 12.8608 | **yes** | 29 | 2026-2054 | 27 | 338.5 | 0.1047 | True |
| capital_recovery | cr_central | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| capital_recovery | cr_ida21 | 36.1408 | 23.3232 | 12.8175 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| capital_recovery | cr_comm_low | 33.9577 | 21.8392 | 12.1184 | **yes** | 29 | 2026-2054 | 27 | 337.1 | 0.1011 | True |
| capital_recovery | cr_comm_high | 37.9184 | 24.5144 | 13.4040 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| gas_deliverability_drift | baseline | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| gas_deliverability_drift | upside | 36.1405 | 22.2141 | 13.9264 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| gas_deliverability_drift | downside | 36.8458 | 26.0098 | 10.8360 | **yes** | 20 | 2026-2045 | 18 | 336.5 | 0.1278 | True |
| gas_deliverability_drift | shock_recovery | 36.1405 | 23.6752 | 12.4652 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| gas_deliverability_gmp | baseline | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| gas_deliverability_gmp | gas_deliv_flat | 36.1405 | 23.7532 | 12.3873 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| gas_deliverability_gmp | gas_deliv_gmp_partial | 36.1405 | 22.1210 | 14.0195 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| gas_deliverability_gmp | gas_deliv_gmp_full | 36.1405 | 21.7718 | 14.3687 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| concessional_envelope | env_none | 36.4487 | 23.5449 | 12.9038 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| concessional_envelope | env_dares | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| concessional_envelope | env_5x | 35.1335 | 22.6055 | 12.5280 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| concessional_envelope | env_10x | 34.1053 | 21.9326 | 12.1727 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| concessional_envelope | env_dares_plus | 36.0172 | 23.2388 | 12.7783 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| peak | peak_suppressed | 35.5636 | 22.8557 | 12.7079 | **yes** | 29 | 2026-2054 | 27 | 338.3 | 0.1041 | True |
| peak | peak_unsuppressed | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| peak | peak_high | 37.1311 | 23.9612 | 13.1699 | **yes** | 29 | 2026-2054 | 27 | 338.5 | 0.1048 | True |
| hydro | hydro_flat | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| hydro | hydro_committed | 33.0285 | 22.5841 | 10.4444 | **yes** | 28 | 2027-2054 | 26 | 325.4 | 0.0735 | False |
| hydro | hydro_ndc | 11.6378 | 10.8847 | 0.7532 | **yes** | 15 | 2040-2054 | 15 | 209.3 | 0.0000 | False |
| new_gas_deliverability | constrained | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| new_gas_deliverability | exempt | 45.8831 | 23.6014 | 22.2817 | **yes** | 26 | 2029-2054 | 27 | 397.8 | 0.2668 | True |
| storage_cc | storage_cc_low | 36.6317 | 24.4898 | 12.1419 | **yes** | 29 | 2026-2054 | 27 | 335.0 | 0.0953 | False |
| storage_cc | storage_cc_central | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| storage_cc | storage_cc_high | 35.7594 | 22.9251 | 12.8343 | **yes** | 29 | 2026-2054 | 27 | 338.5 | 0.1047 | True |
| gas_build | gas_unconstrained | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| gas_build | gas_ndc_17gw | 38.2167 | 23.3231 | 14.8937 | **yes** | 27 | 2028-2054 | 27 | 336.8 | 0.1251 | True |
| gas_build | gas_none | 57.4869 | 25.4694 | 32.0175 | **yes** | 20 | 2028-2047 | 20 | 566.3 | 0.4834 | True |
| gas_capex | gas_capex_low | 34.0288 | 22.5578 | 11.4710 | **yes** | 29 | 2026-2054 | 27 | 335.4 | 0.0965 | False |
| gas_capex | gas_capex_central | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| gas_capex | gas_capex_high | 36.5456 | 23.4730 | 13.0725 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| ccgt_efficiency | ccgt_eff_central | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| ccgt_efficiency | ccgt_eff_iso | 33.3819 | 23.1186 | 10.2633 | **yes** | 29 | 2026-2054 | 27 | 325.5 | 0.0689 | False |
| ccgt_efficiency | ccgt_eff_low | 37.9750 | 23.4495 | 14.5255 | **yes** | 29 | 2026-2054 | 27 | 346.4 | 0.1252 | True |
| dscr | dscr_central | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| dscr | dscr_low | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| dscr | dscr_high | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| om | om_none | 28.2629 | 16.7464 | 11.5164 | **yes** | 29 | 2026-2054 | 28 | 338.2 | 0.1040 | True |
| om | om_low | 35.8888 | 23.0394 | 12.8494 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| om | om_central | 36.1405 | 23.3231 | 12.8174 | **yes** | 29 | 2026-2054 | 27 | 338.6 | 0.1049 | True |
| om | om_high | 36.5129 | 23.7399 | 12.7730 | **yes** | 29 | 2026-2054 | 27 | 337.0 | 0.1007 | True |

Arms (family, member rows) where the cap does not bind: 0 of 58.

## Solves that did not return an optimal plan

none: every driver invocation exited 0 (02_ raises on a non-optimal status).

## Uncapped rank against capped rank (TASK 82 B.7), column 3 (live arms)

Uncapped: `results/lever_ranking/arms_661b03c1/lever_ranking.csv`. Capped: this table's `ndc3_conditional` ranking. A family is ranked only where its arms produce different plans; otherwise its role is shown.

| family | uncapped rank | uncapped span (bn) | capped rank | capped span (bn) | capped role |
|---|---|---|---|---|---|
| gas_availability | 1 | 8.4896 | 2 | 20.7626 | ranked |
| conversion | 4 | 3.0907 | 9 | 1.6465 | ranked |
| conversion_efficiency | 10 | 1.1521 | 14 | 0.6859 | ranked |
| gas_retirement | 12 | 1.0385 | 11 | 1.4548 | ranked |
| capital_recovery | 5 | 2.6751 | 6 | 3.9607 | ranked |
| gas_deliverability_drift | 3 | 3.7957 | 13 | 0.7053 | ranked |
| gas_deliverability_gmp | 7 | 1.9814 | - | 0.0000 | no_build_response |
| concessional_envelope | 8 | 1.6123 | 8 | 2.3434 | ranked |
| peak | 11 | 1.1055 | 10 | 1.5675 | ranked |
| hydro | - | - | - | 24.5026 | bound |
| new_gas_deliverability | 15 | 0.2783 | 3 | 9.7426 | ranked |
| storage_cc | 9 | 1.5648 | 12 | 0.8723 | ranked |
| gas_build | 6 | 2.1463 | 1 | 21.3464 | ranked |
| gas_capex | 13 | 0.9153 | 7 | 2.5168 | ranked |
| ccgt_efficiency | 14 | 0.3309 | 5 | 4.5931 | ranked |
| dscr | - | - | - | 0.0000 | no_build_response |
| om | 2 | 6.9935 | 4 | 8.2500 | ranked |
