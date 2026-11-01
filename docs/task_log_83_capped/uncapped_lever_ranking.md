# Lever ranking -- post-epoch, generated 2026-10-23T22:06:08 at HEAD `dabd2a7` (src/scripts have uncommitted changes) by `scripts/build_lever_ranking.py`

Central `661b03c1` objective 23,323,068,340.46 USD. Files read 177, admitted 42. A verification artefact, not a finding.

## Ranked levers (arms produce different build decisions)

Ordered by column 3 (live arms). `rank (col 1)` is the rank by all-arms span.

| rank (col 3) | rank (col 1) | family | span col 3 live (bn) | span col 1 all arms (bn) | col 3 low arm | col 3 high arm | unasserted (excluded) | % central (col 3) | arms | in F-5 | plans | col 2 (bn) | col 2 differs | max d solar 2054 MW | max d gas 2054 MW | max d storage 2054 MWh | first year |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | gas_availability | 8.4896 | 8.4896 | gas_avail_nameplate (23.3231) | gas_avail_observed (31.8127) | none | 36.400 | 2 | yes | 2 | 8.4896 | no | 20986.4996 | 4387.5856 | 26899.8693 | 2025 |
| 2 | 2 | om | 6.9935 | 6.9935 | om_none (16.7464) | om_high (23.7399) | none | 29.985 | 4 | no | 4 | 6.9935 | no | 1153.2862 | 1977.3004 | 5434.2869 | 2029 |
| 3 | 3 | gas_deliverability_drift | 3.7957 | 3.7957 | upside (22.2141) | downside (26.0098) | none | 16.274 | 4 | yes | 4 | 3.7957 | no | 35662.0658 | 6128.6174 | 24514.4694 | 2030 |
| 4 | 4 | conversion | 3.0907 | 3.0907 | conv_high (22.2129) | conv_none (25.3036) | none | 13.252 | 4 | yes | 4 | 3.0907 | no | 1231.4770 | 2953.1840 | 7672.4049 | 2026 |
| 5 | 5 | capital_recovery | 2.6751 | 2.6751 | cr_comm_low (21.8392) | cr_comm_high (24.5144) | none | 11.470 | 4 | yes | 3 | 2.6751 | no | 143.9106 | 30.0869 | 120.3480 | 2035 |
| 6 | 6 | gas_build | 2.1463 | 2.1463 | gas_unconstrained (23.3231) | gas_none (25.4694) | none | 9.202 | 3 | no | 3 | 2.1463 | no | 21357.6558 | 4507.5856 | 18133.8043 | 2032 |
| 7 | 7 | gas_deliverability_gmp | 1.9814 | 1.9814 | gas_deliv_gmp_full (21.7718) | gas_deliv_flat (23.7532) | none | 8.495 | 4 | yes | 4 | 1.9814 | no | 33214.9790 | 7462.2142 | 29848.8567 | 2025 |
| 8 | 9 | concessional_envelope | 1.6123 | 1.6123 | env_10x (21.9326) | env_none (23.5449) | none | 6.913 | 5 | yes | 5 | 1.6123 | no | 0.0001 | 0.0000 | 0.0000 | 2033 |
| 9 | 10 | storage_cc | 1.5648 | 1.5648 | storage_cc_high (22.9251) | storage_cc_low (24.4898) | none | 6.709 | 3 | no | 3 | 1.5648 | no | 427.1057 | 446.0523 | 42177.1686 | 2025 |
| 10 | 11 | conversion_efficiency | 1.1521 | 1.1521 | conv_eff_high (22.8020) | conv_eff_low (23.9542) | none | 4.940 | 3 | no | 3 | 1.1521 | no | 2405.9300 | 503.0008 | 2012.0029 | 2031 |
| 11 | 12 | peak | 1.1055 | 1.1055 | peak_suppressed (22.8557) | peak_high (23.9612) | none | 4.740 | 3 | yes | 3 | 1.1055 | no | 155.3115 | 149.7015 | 18098.3991 | 2035 |
| 12 | 8 | gas_retirement | 1.0385 | 1.9522 | gas_retire_none (22.2846) | gas_retire_vintage_30 (23.3231) | gas_retire_proxy | 4.453 | 5 | yes | 5 | 1.9522 | no | 698.3801 | 4348.3219 | 26394.7125 | 2025 |
| 13 | 13 | gas_capex | 0.9153 | 0.9153 | gas_capex_low (22.5578) | gas_capex_high (23.4730) | none | 3.924 | 3 | no | 3 | 0.9153 | no | 143.9106 | 30.0869 | 120.3479 | 2025 |
| 14 | 14 | ccgt_efficiency | 0.3309 | 0.3309 | ccgt_eff_iso (23.1186) | ccgt_eff_low (23.4495) | none | 1.419 | 3 | no | 3 | 0.3309 | no | 1725.9887 | 360.8474 | 1443.3897 | 2032 |

## Families whose arms produce an identical plan (rescaling; not ranked)

| family | span (bn) | low arm | high arm | % central | arms | in F-5 | max d solar MW / gas 2054 MW / storage MWh | first year |
|---|---|---|---|---|---|---|---|---|
| dscr | 0.0000 | dscr_central (23.3231) | dscr_central (23.3231) | 0.000 | 3 | no | 0.0000 / 0.0000 / 0.0000 | none |

## Not comparable: arms differ in dispatch model (not ranked; TASK 83 A.2)

| family | span (bn) | low arm | high arm | arms | max d solar MW / gas 2054 MW / storage MWh | why it is not ranked |
|---|---|---|---|---|---|---|
| new_gas_deliverability | 0.2783 | constrained (23.3231) | exempt (23.6014) | 2 | 29840.7479 / 6809.5450 / 27238.1799 | the exempt arm cannot run vintage dispatch (the builder raises if it is on), so it is a single-stream arm compared with a vintage-dispatch constrained arm: the span is the dispatch-model difference, not the value of deliverability (capped, exempt equals the BLENDED constrained objective to the cent; L138, L145) |

## Bounds and conventions (do not enter the ranking)

Bound (`bound: True`), not a lever:

| family | span (bn) | low arm | high arm | % central | arms | plans | first year |
|---|---|---|---|---|---|---|---|
| hydro | 12.4384 | hydro_ndc (10.8847) | hydro_flat (23.3231) | 53.331 | 3 | 3 | 2032 |

## Declared but not ranked (missing member)

none

## Joint arms (differ from central in two or more fields; excluded)

none

## Values the files carry that no declared family covers (reported, not adopted)

| field | value | files | where |
|---|---|---|---|
| dscr_case | dscr_regression | 27 | a scenario-function key |
| gas_availability_case | gas_avail_central | 0 | a scenario-function key |
| gas_deliverability_case | flat_downside | 0 | a scenario-function key |
| gas_deliverability_case | flat_shock_recovery | 0 | a scenario-function key |
| gas_deliverability_case | flat_upside | 0 | a scenario-function key |
| peak_case | peak_nirp | 0 | a scenario-function key |
| tariff_case (stored; no declared family) | tariff_central | 135 | stored in files |
| tariff_case (stored; no declared family) | tariff_legacy_65m | 4 | stored in files |
| capital_window_case (tag token only; not stored; no declared family) | arrears | 95 | in arm_tag |
| storage_capex_case (tag token only; not stored; no declared family) | declining_nrel | 95 | in arm_tag |

## Live-arm rule (column 3): every arm the rule could have touched

An arm is UNASSERTED when it is unsourced, represents no state or bound the thesis asserts, and is retained only so that superseded results remain attributable. Only UNASSERTED arms are excluded from column 3; the span with each other candidate excluded is shown for information and is not applied.

| family | arm | status | span all arms (bn) | span with this arm excluded (bn) | reason (declared in LEVER_FAMILIES) |
|---|---|---|---|---|---|
| gas_availability | `gas_avail_nameplate` | live | 8.4896 | undefined (fewer than two arms left) | LIVE: the central arm and the anchor of every result (BASELINE_CENTRAL_ARM); its 'retained' docstring wording does not make it unasserted |
| conversion | `conv_none` | live | 3.0907 | 2.3407 | LIVE: an attainable state (no conversion) and the cap builder's own baseline (Ch3 s3.7.6); 'regression arm' wording does not make it unasserted |
| gas_retirement | `gas_retire_proxy` | UNASSERTED | 1.9522 | 1.0385 | pre-2.11 'brownfield retirement proxy': unsourced [PENDING]; retained only so pre-2.11 results stay attributable; claims no state or bound |
| gas_retirement | `gas_retire_none` | live | 1.9522 | 1.5518 | LIVE: a declared upper bound on the surviving fleet; a bound is a claim |
| peak | `peak_suppressed` | live | 1.1055 | 0.6382 | LIVE: a declared floor (L2) |
| hydro | `hydro_flat` | live | 12.4384 | 11.6994 | LIVE: the central arm and anchor; 'regression arm' wording does not make it unasserted |
| gas_build | `gas_none` | live | 2.1463 | 0.0000 | LIVE: an attainable policy state; the span to it measures the value of being permitted to build new gas |
| gas_capex | `gas_capex_low` | live | 0.9153 | 0.1500 | LIVE: a DECLARED LOWER BOUND; a bound is a claim; attribution wording alone never makes an arm unasserted |
| om | `om_none` | live | 6.9935 | 0.7004 | NOT unasserted: SYNTHETIC and unattainable; the family is already outside the ranked table by the build-response rule |

## Column-2 exclusions

* `dscr`: excluded dscr_central, dscr_low, dscr_high
