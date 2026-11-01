# Lever ranking -- post-epoch, generated 2026-10-23T21:50:50 at HEAD `17f8436` (src/scripts have uncommitted changes) by `scripts/build_lever_ranking.py`

Central `ae31b29c` objective 35,561,545,184.67 USD. Files read 57, admitted 42. A verification artefact, not a finding.

## Ranked levers (arms produce different build decisions)

Ordered by column 3 (live arms). `rank (col 1)` is the rank by all-arms span.

| rank (col 3) | rank (col 1) | family | span col 3 live (bn) | span col 1 all arms (bn) | col 3 low arm | col 3 high arm | unasserted (excluded) | % central (col 3) | arms | in F-5 | plans | col 2 (bn) | col 2 differs | max d solar 2054 MW | max d gas thermal draw 2025-54 TWh_th | max d storage 2054 MWh | first year |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | gas_build | 44.4640 | 44.4640 | gas_unconstrained (35.5615) | gas_none (80.0255) | none | 125.034 | 3 | no | 3 | 44.4640 | no | 17116.1588 | 1416.2265 | 74423.6843 | 2025 |
| 2 | 2 | gas_availability | 24.3087 | 24.3087 | gas_avail_nameplate (35.5615) | gas_avail_observed (59.8703) | none | 68.357 | 2 | yes | 2 | 24.3087 | no | 153.4980 | 265.2474 | 0.0000 | 2025 |
| 3 | 3 | new_gas_deliverability | 17.5400 | 17.5400 | constrained (35.5615) | exempt (53.1015) | none | 49.323 | 2 | no | 2 | 17.5400 | no | 5148.8369 | 1457.8860 | 1348.6540 | 2025 |
| 4 | 4 | om | 7.9815 | 7.9815 | om_none (27.7890) | om_high (35.7704) | none | 22.444 | 4 | no | 2 | 7.9815 | no | 24.3938 | 3.0769 | 0.0000 | 2032 |
| 5 | 5 | gas_deliverability_drift | 7.2514 | 7.2514 | baseline (35.5615) | downside (42.8130) | none | 20.391 | 4 | yes | 2 | 7.2514 | no | 1875.1070 | 227.2299 | 0.0000 | 2038 |
| 6 | 6 | ccgt_efficiency | 5.9456 | 5.9456 | ccgt_eff_iso (32.0668) | ccgt_eff_low (38.0124) | none | 16.719 | 3 | no | 3 | 5.9456 | no | 3576.2846 | 12.8248 | 0.0000 | 2030 |
| 7 | 7 | gas_capex | 3.0475 | 3.0475 | gas_capex_low (33.0199) | gas_capex_high (36.0674) | none | 8.570 | 3 | no | 2 | 3.0475 | no | 4.1637 | 0.3278 | 0.0000 | 2031 |
| 8 | 9 | concessional_envelope | 2.0517 | 2.0517 | env_10x (33.7318) | env_none (35.7834) | none | 5.769 | 5 | yes | 2 | 2.0517 | no | 0.0000 | 0.0000 | 0.0000 | 2048 |
| 9 | 10 | conversion | 1.9169 | 1.9169 | conv_high (34.3425) | conv_none (36.2594) | none | 5.390 | 4 | yes | 4 | 1.9169 | no | 390.9040 | 672.2476 | 0.0000 | 2025 |
| 10 | 11 | peak | 1.6586 | 1.6586 | peak_suppressed (34.9518) | peak_high (36.6104) | none | 4.664 | 3 | yes | 3 | 1.6586 | no | 216.9303 | 23.2212 | 0.0000 | 2025 |
| 11 | 8 | gas_retirement | 1.4605 | 2.3831 | gas_retire_none (34.1011) | gas_retire_vintage_30 (35.5615) | gas_retire_proxy | 4.107 | 5 | yes | 5 | 2.3831 | no | 110.5670 | 30.3144 | 0.0000 | 2040 |
| 12 | 12 | conversion_efficiency | 0.8197 | 0.8197 | conv_eff_high (34.9317) | conv_eff_low (35.7514) | none | 2.305 | 3 | no | 3 | 0.8197 | no | 230.8802 | 541.5153 | 0.0000 | 2025 |

## Families whose arms produce an identical plan (rescaling; not ranked)

| family | span (bn) | low arm | high arm | % central | arms | in F-5 | max d solar MW / gas thermal draw 2025-54 TWh_th / storage MWh | first year |
|---|---|---|---|---|---|---|---|---|
| capital_recovery | 3.5152 | cr_comm_low (33.5619) | cr_comm_high (37.0771) | 9.885 | 4 | yes | 0.0000 / 0.0000 / 0.0000 | none |
| storage_cc | 0.0000 | storage_cc_high (35.5615) | storage_cc_low (35.5615) | 0.000 | 3 | no | 0.0000 / 0.0000 / 0.0000 | none |
| gas_deliverability_gmp | 0.0000 | baseline (35.5615) | baseline (35.5615) | 0.000 | 4 | yes | 0.0000 / 0.0000 / 0.0000 | none |
| dscr | 0.0000 | dscr_central (35.5615) | dscr_central (35.5615) | 0.000 | 3 | no | 0.0000 / 0.0000 / 0.0000 | none |

## Bounds and conventions (do not enter the ranking)

Bound (`bound: True`), not a lever:

| family | span (bn) | low arm | high arm | % central | arms | plans | first year |
|---|---|---|---|---|---|---|---|
| hydro | 24.2382 | hydro_ndc (11.3233) | hydro_flat (35.5615) | 68.159 | 3 | 3 | 2025 |

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
| gas_availability | `gas_avail_nameplate` | live | 24.3087 | undefined (fewer than two arms left) | LIVE: the central arm and the anchor of every result (BASELINE_CENTRAL_ARM); its 'retained' docstring wording does not make it unasserted |
| conversion | `conv_none` | live | 1.9169 | 1.6237 | LIVE: an attainable state (no conversion) and the cap builder's own baseline (Ch3 s3.7.6); 'regression arm' wording does not make it unasserted |
| gas_retirement | `gas_retire_proxy` | UNASSERTED | 2.3831 | 1.4605 | pre-2.11 'brownfield retirement proxy': unsourced [PENDING]; retained only so pre-2.11 results stay attributable; claims no state or bound |
| gas_retirement | `gas_retire_none` | live | 2.3831 | 2.0231 | LIVE: a declared upper bound on the surviving fleet; a bound is a claim |
| peak | `peak_suppressed` | live | 1.6586 | 1.0489 | LIVE: a declared floor (L2) |
| hydro | `hydro_flat` | live | 24.2382 | 20.9415 | LIVE: the central arm and anchor; 'regression arm' wording does not make it unasserted |
| gas_build | `gas_none` | live | 44.4640 | 1.6824 | LIVE: an attainable policy state; the span to it measures the value of being permitted to build new gas |
| gas_capex | `gas_capex_low` | live | 3.0475 | 0.5058 | LIVE: a DECLARED LOWER BOUND; a bound is a claim; attribution wording alone never makes an arm unasserted |
| om | `om_none` | live | 7.9815 | 0.3555 | NOT unasserted: SYNTHETIC and unattainable; the family is already outside the ranked table by the build-response rule |

## Column-2 exclusions

* `capital_recovery`: excluded cr_central, cr_ida21, cr_comm_low, cr_comm_high
* `gas_deliverability_gmp`: excluded baseline, gas_deliv_flat, gas_deliv_gmp_partial, gas_deliv_gmp_full
* `storage_cc`: excluded storage_cc_low, storage_cc_central, storage_cc_high
* `dscr`: excluded dscr_central, dscr_low, dscr_high

## Per-arm cap binding under `ndc3_unconditional` (TASK 82 B.4)

`cap_binding` is True where the annual cap's shadow price (`carbon_shadow_yr_usd_per_tco2_by_year`) is non-zero (|x| > 1e-6 USD/tCO2) in at least one year. `unc>cap yrs` counts the years in which the UNCAPPED counterpart's total emissions exceed the cap: if it is 0 the uncapped plan already meets the cap in every year. The capped runs also carry a public-capital envelope the uncapped run does not (`capital_case` tight/moderate), so `capped - uncapped` is cap plus envelope, not cap alone.

| family | member | capped (bn) | uncapped (bn) | capped - uncapped (bn) | cap_binding | binding yrs | first-last | unc>cap yrs | emissions Mt | leakage share | leakage flagged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gas_availability | gas_avail_nameplate | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| gas_availability | gas_avail_observed | 59.8703 | 31.8127 | 28.0576 | **yes** | 24 | 2031-2054 | 14 | 445.4 | 0.2186 | True |
| conversion | conv_none | 36.2594 | 25.3036 | 10.9559 | **yes** | 30 | 2025-2054 | 29 | 464.5 | 0.2099 | True |
| conversion | conv_low | 35.9662 | 24.5536 | 11.4126 | **yes** | 29 | 2026-2054 | 29 | 464.0 | 0.2120 | True |
| conversion | conv_central | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| conversion | conv_high | 34.3425 | 22.2129 | 12.1297 | **yes** | 26 | 2029-2054 | 26 | 455.8 | 0.2260 | True |
| conversion_efficiency | conv_eff_low | 35.7514 | 23.9542 | 11.7972 | **yes** | 29 | 2026-2054 | 27 | 463.9 | 0.2121 | True |
| conversion_efficiency | conv_central | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| conversion_efficiency | conv_eff_high | 34.9317 | 22.8020 | 12.1297 | **yes** | 26 | 2029-2054 | 26 | 455.8 | 0.2260 | True |
| gas_retirement | gas_retire_none | 34.1011 | 22.2846 | 11.8165 | **yes** | 28 | 2027-2054 | 26 | 465.6 | 0.2159 | True |
| gas_retirement | gas_retire_vintage_30 | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| gas_retirement | gas_retire_vintage_40 | 34.6246 | 22.6851 | 11.9396 | **yes** | 28 | 2027-2054 | 26 | 464.2 | 0.2135 | True |
| gas_retirement | gas_retire_vintage_50 | 34.4610 | 22.8144 | 11.6466 | **yes** | 28 | 2027-2054 | 26 | 464.1 | 0.2133 | True |
| gas_retirement | gas_retire_proxy | 36.4841 | 24.2368 | 12.2473 | **yes** | 28 | 2027-2054 | 26 | 463.6 | 0.2124 | True |
| capital_recovery | cr_central | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| capital_recovery | cr_ida21 | 35.5617 | 23.3232 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| capital_recovery | cr_comm_low | 33.5619 | 21.8392 | 11.7226 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| capital_recovery | cr_comm_high | 37.0771 | 24.5144 | 12.5627 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| gas_deliverability_drift | baseline | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| gas_deliverability_drift | upside | 35.5615 | 22.2141 | 13.3474 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| gas_deliverability_drift | downside | 42.8130 | 26.0098 | 16.8032 | **yes** | 11 | 2027-2037 | 9 | 528.7 | 0.3878 | True |
| gas_deliverability_drift | shock_recovery | 35.5615 | 23.6752 | 11.8863 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| gas_deliverability_gmp | baseline | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| gas_deliverability_gmp | gas_deliv_flat | 35.5615 | 23.7532 | 11.8083 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| gas_deliverability_gmp | gas_deliv_gmp_partial | 35.5615 | 22.1210 | 13.4406 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| gas_deliverability_gmp | gas_deliv_gmp_full | 35.5615 | 21.7718 | 13.7897 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| concessional_envelope | env_none | 35.7834 | 23.5449 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| concessional_envelope | env_dares | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| concessional_envelope | env_5x | 34.6750 | 22.6055 | 12.0694 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| concessional_envelope | env_10x | 33.7318 | 21.9326 | 11.7991 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| concessional_envelope | env_dares_plus | 35.4728 | 23.2388 | 12.2340 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| peak | peak_suppressed | 34.9518 | 22.8557 | 12.0961 | **yes** | 28 | 2027-2054 | 26 | 465.6 | 0.2159 | True |
| peak | peak_unsuppressed | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| peak | peak_high | 36.6104 | 23.9612 | 12.6492 | **yes** | 28 | 2027-2054 | 26 | 463.4 | 0.2129 | True |
| hydro | hydro_flat | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| hydro | hydro_committed | 32.2649 | 22.5841 | 9.6808 | **yes** | 28 | 2027-2054 | 26 | 432.0 | 0.1589 | True |
| hydro | hydro_ndc | 11.3233 | 10.8847 | 0.4386 | **yes** | 10 | 2045-2054 | 11 | 237.8 | 0.0000 | False |
| new_gas_deliverability | constrained | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| new_gas_deliverability | exempt | 53.1015 | 23.6014 | 29.5001 | **yes** | 26 | 2029-2054 | 26 | 612.4 | 0.4249 | True |
| storage_cc | storage_cc_low | 35.5615 | 24.4898 | 11.0717 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| storage_cc | storage_cc_central | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| storage_cc | storage_cc_high | 35.5615 | 22.9251 | 12.6365 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| gas_build | gas_unconstrained | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| gas_build | gas_ndc_17gw | 37.2440 | 23.3231 | 13.9209 | **yes** | 26 | 2029-2054 | 26 | 460.0 | 0.2270 | True |
| gas_build | gas_none | 80.0255 | 25.4694 | 54.5561 | **yes** | 17 | 2029-2045 | 17 | 929.5 | 0.6395 | True |
| gas_capex | gas_capex_low | 33.0199 | 22.5578 | 10.4621 | **yes** | 28 | 2027-2054 | 26 | 463.9 | 0.2130 | True |
| gas_capex | gas_capex_central | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| gas_capex | gas_capex_high | 36.0674 | 23.4730 | 12.5943 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| ccgt_efficiency | ccgt_eff_central | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| ccgt_efficiency | ccgt_eff_iso | 32.0668 | 23.1186 | 8.9482 | **yes** | 28 | 2027-2054 | 26 | 425.6 | 0.1421 | True |
| ccgt_efficiency | ccgt_eff_low | 38.0124 | 23.4495 | 14.5629 | **yes** | 28 | 2027-2054 | 26 | 488.1 | 0.2521 | True |
| dscr | dscr_central | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| dscr | dscr_low | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| dscr | dscr_high | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| om | om_none | 27.7890 | 16.7464 | 11.0425 | **yes** | 28 | 2027-2054 | 28 | 463.7 | 0.2126 | True |
| om | om_low | 35.4149 | 23.0394 | 12.3755 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| om | om_central | 35.5615 | 23.3231 | 12.2385 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |
| om | om_high | 35.7704 | 23.7399 | 12.0306 | **yes** | 28 | 2027-2054 | 26 | 464.0 | 0.2131 | True |

Arms (family, member rows) where the cap does not bind: 0 of 58.

## Solves that did not return an optimal plan

none: every driver invocation exited 0 (02_ raises on a non-optimal status).

## Uncapped rank against capped rank (TASK 82 B.7), column 3 (live arms)

Uncapped: `results/lever_ranking/arms_661b03c1/lever_ranking.csv`. Capped: this table's `ndc3_unconditional` ranking. A family is ranked only where its arms produce different plans; otherwise its role is shown.

| family | uncapped rank | uncapped span (bn) | capped rank | capped span (bn) | capped role |
|---|---|---|---|---|---|
| gas_availability | 1 | 8.4896 | 2 | 24.3087 | ranked |
| conversion | 4 | 3.0907 | 9 | 1.9169 | ranked |
| conversion_efficiency | 10 | 1.1521 | 12 | 0.8197 | ranked |
| gas_retirement | 12 | 1.0385 | 11 | 1.4605 | ranked |
| capital_recovery | 5 | 2.6751 | - | 3.5152 | no_build_response |
| gas_deliverability_drift | 3 | 3.7957 | 5 | 7.2514 | ranked |
| gas_deliverability_gmp | 7 | 1.9814 | - | 0.0000 | no_build_response |
| concessional_envelope | 8 | 1.6123 | 8 | 2.0517 | ranked |
| peak | 11 | 1.1055 | 10 | 1.6586 | ranked |
| hydro | - | - | - | 24.2382 | bound |
| new_gas_deliverability | 15 | 0.2783 | 3 | 17.5400 | ranked |
| storage_cc | 9 | 1.5648 | - | 0.0000 | no_build_response |
| gas_build | 6 | 2.1463 | 1 | 44.4640 | ranked |
| gas_capex | 13 | 0.9153 | 7 | 3.0475 | ranked |
| ccgt_efficiency | 14 | 0.3309 | 6 | 5.9456 | ranked |
| dscr | - | - | - | 0.0000 | no_build_response |
| om | 2 | 6.9935 | 4 | 7.9815 | ranked |
