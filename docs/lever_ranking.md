# Lever ranking -- PROVISIONAL

> **SUPERSEDED (TASK 59 Part 4.4, 2026-10-22).** This is the TASK 48 provisional table, built on the pre-epoch generation. It is kept as a record and is not to be quoted or reconciled against. Its successor is `results/lever_ranking/arms_6046f759/lever_ranking.csv` (with `lever_ranking_summary.json` and `lever_ranking.md` beside it), produced by `scripts/build_lever_ranking.py`, which produced both. `results/` is gitignored; the committed record of the successor is `docs/task_log_59.md` Part 3.

> **PROVISIONAL: the central arm's objective is read from `results/baseline_sliced/summary__6bf84218.json` (om_case om_central, 2026-10-03T18:47:55) because `results/baseline_sliced/summary__02c17f55.json` is not a valid om_central file at this time. That file differs from the central arm in dscr_case: dscr_low (file) vs dscr_central (central); its objective equalling the central one (27,317,670,155.72, as recorded in findings_frozen.md and in the pre-Phase-C central file) is an observation, not a proof that the central arm would reproduce it.**
> Regenerate after Group 4. This table is a verification artefact, not a finding.

Generated 2026-10-18T19:17:21 at HEAD `9d8f979` (src/scripts/results have uncommitted changes) by `scripts/build_lever_ranking.py`. Input: CURRENT rows of `docs/results_inventory.csv` only; the generator refuses any LEGACY or UNCLASSIFIABLE row, any md5 mismatch, and any file whose recorded `om_case` or `arm_tag` differs from the declaration.

**Span rule.** A lever's magnitude is max minus min of the objective (`npv_total_cost_usd`, full-horizon 2025-2054 NPV) across every arm in its declared family (`src/scenarios.py: LEVER_FAMILIES`), the central arm included. One rule for every family.

| rank | policy rank | lever | span (bn USD) | low arm | high arm | arms | in F-5 list | note |
|---|---|---|---|---|---|---|---|---|
| 1 | -- | hydro trajectory (flat -> ndc) | 14.7150 | hydro_ndc (12.6027) | hydro_flat (27.3177) | 3 | no | **bound, not a policy lever** [VERIFY: Part C] |
| 2 | 1 | gas fleet availability (nameplate -> observed) | 8.5028 | gas_avail_nameplate (27.3177) | gas_avail_observed (35.8204) | 2 | yes |  |
| 3 | 2 | conversion (none -> high) | 5.8446 | conv_high (26.1412) | conv_none (31.9858) | 4 | yes |  |
| 4 | 3 | gas retirement (none / vintage 30-50 / proxy) | 3.8710 | gas_retire_none (24.5777) | gas_retire_proxy (28.4487) | 5 | yes |  |
| 5 | 4 | capital recovery span (financing) | 3.8109 | cr_comm_low (25.1945) | cr_comm_high (29.0054) | 4 | yes |  |
| 6 | 5 | deliverability, drift family | 2.9620 | upside (26.4242) | downside (29.3863) | 4 | yes |  |
| 7 | 6 | deliverability, GMP family | 2.3837 | gas_deliv_gmp_full (25.2950) | gas_deliv_flat (27.6786) | 4 | yes |  |
| 8 | 7 | concessional envelope | 2.3001 | env_10x (25.3089) | env_none (27.6089) | 5 | yes |  |
| 9 | 8 | demand shape (suppressed -> high) | 2.1496 | peak_suppressed (26.3468) | peak_high (28.4964) | 3 | yes |  |
| 10 | 9 | new-gas deliverability (constrained / exempt) | 2.0059 | exempt (25.3118) | constrained (27.3177) | 2 | no |  |
| 11 | 10 | storage capacity credit | 1.9817 | storage_cc_high (26.3738) | storage_cc_low (28.3555) | 3 | no |  |
| 12 | 11 | gas capex | 0.9375 | gas_capex_low (26.4409) | gas_capex_high (27.3784) | 3 | no |  |
| 13 | 12 | gas build (unconstrained / ndc_17gw / none) | 0.8068 | gas_unconstrained (27.3177) | gas_none (28.1245) | 3 | no |  |
| 14 | 13 | CCGT efficiency | 0.0000 | all arms equal | all arms equal | 3 | no |  |
| 15 | 14 | DSCR | 0.0000 | all arms equal | all arms equal | 3 | no |  |

`rank` counts every row; `policy rank` excludes rows marked as bounds.

**Declared but not ranked (arms missing from the CURRENT inventory):**

* `om` (O&M band (none / low / central / high)): om_none (expected results/baseline_sliced/summary__ba2f0fd3.json: not in inventory / not on disk); om_low (expected results/baseline_sliced/summary__9137a106.json: not in inventory / not on disk); om_high (expected results/baseline_sliced/summary__b4106a3e.json: not in inventory / not on disk)

## Family declarations

* `gas_availability` -- field `gas_availability_case`: `gas_avail_nameplate`, `gas_avail_observed`
* `conversion` -- field `conversion_case`: `conv_none`, `conv_low`, `conv_central`, `conv_high`
* `gas_retirement` -- field `gas_retirement_case`: `gas_retire_none`, `gas_retire_vintage_30`, `gas_retire_vintage_40`, `gas_retire_vintage_50`, `gas_retire_proxy`
* `capital_recovery` -- field `capital_recovery_case`: `cr_central`, `cr_ida21`, `cr_comm_low`, `cr_comm_high`
* `gas_deliverability_drift` -- field `gas_deliverability_case`: `baseline`, `upside`, `downside`, `shock_recovery`
* `gas_deliverability_gmp` -- field `gas_deliverability_case`: `baseline`, `gas_deliv_flat`, `gas_deliv_gmp_partial`, `gas_deliv_gmp_full`
* `concessional_envelope` -- field `concessional_envelope_case`: `env_none`, `env_dares`, `env_5x`, `env_10x`, `env_dares_plus`
* `peak` -- field `peak_case`: `peak_suppressed`, `peak_unsuppressed`, `peak_high`
* `hydro` -- field `hydro_case`: `hydro_flat`, `hydro_committed`, `hydro_ndc` **[bound]**
* `new_gas_deliverability` -- field `new_gas_deliverability`: `constrained`, `exempt`
* `storage_cc` -- field `storage_cc_case`: `storage_cc_low`, `storage_cc_central`, `storage_cc_high`
* `gas_build` -- field `gas_build_case`: `gas_unconstrained`, `gas_ndc_17gw`, `gas_none`
* `gas_capex` -- field `gas_capex_case`: `gas_capex_low`, `gas_capex_central`, `gas_capex_high`
* `ccgt_efficiency` -- field `ccgt_efficiency_case`: `ccgt_eff_central`, `ccgt_eff_iso`, `ccgt_eff_low`
* `dscr` -- field `dscr_case`: `dscr_central`, `dscr_low`, `dscr_high`
* `om` -- field `om_case`: `om_none`, `om_low`, `om_central`, `om_high`

## Source file per arm

| family | arm | file | objective (USD) | mtime | md5 |
|---|---|---|---|---|---|
| gas_availability | gas_avail_nameplate | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
| gas_availability | gas_avail_observed | `results/baseline_sliced/summary__fe552b51.json` | 35,820,429,760.24 | 2026-10-03T18:47:58 | `a581ae93893096d37fd2093cb375d1fd` |
| conversion | conv_none | `results/baseline_sliced/summary__062eba52.json` | 31,985,807,874.57 | 2026-10-03T18:47:30 | `dd9f3c2ac2f1db1a25b77ff86f952359` |
| conversion | conv_low | `results/baseline_sliced/summary__abbb2e9e.json` | 29,577,596,240.05 | 2026-10-03T18:47:31 | `d3299a1d29fef44cbb69f862ea995af3` |
| conversion | conv_central | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
| conversion | conv_high | `results/baseline_sliced/summary__138c45af.json` | 26,141,199,602.21 | 2026-10-03T18:47:32 | `88238aa43dedb516a3cc0b52b5cb456a` |
| gas_retirement | gas_retire_none | `results/baseline_sliced/summary__bf2805fc.json` | 24,577,737,606.14 | 2026-10-03T18:48:00 | `3b55b7b0a59f46b315a767a38057f25e` |
| gas_retirement | gas_retire_vintage_30 | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
| gas_retirement | gas_retire_vintage_40 | `results/baseline_sliced/summary__977b534a.json` | 27,048,662,190.33 | 2026-10-03T18:48:02 | `6a64a76bc5cd6c46197b09acde6a2d78` |
| gas_retirement | gas_retire_vintage_50 | `results/baseline_sliced/summary__fb66cb38.json` | 26,545,437,000.87 | 2026-10-03T18:48:03 | `be63bf72240f2b86f71df81475453789` |
| gas_retirement | gas_retire_proxy | `results/baseline_sliced/summary__2f4b84cc.json` | 28,448,716,712.25 | 2026-10-03T18:48:12 | `9561479b9b4b054161a7d75ba113d995` |
| capital_recovery | cr_central | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
| capital_recovery | cr_ida21 | `results/baseline_sliced/summary__20bf87d2.json` | 27,317,922,491.05 | 2026-10-03T18:47:26 | `1e4e61bd0dd103a89d69dfb0ac7514f2` |
| capital_recovery | cr_comm_low | `results/baseline_sliced/summary__c52b926f.json` | 25,194,520,235.16 | 2026-10-03T18:47:27 | `d230b5a06fe54d04a4472617acb2ec65` |
| capital_recovery | cr_comm_high | `results/baseline_sliced/summary__5bc7cc5a.json` | 29,005,440,056.55 | 2026-10-03T18:47:28 | `3637c9c6b89cbf587d8b1e8932441cd9` |
| gas_deliverability_drift | baseline | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
| gas_deliverability_drift | upside | `results/baseline_sliced/summary__be72250f.json` | 26,424,249,456.06 | 2026-10-03T18:47:35 | `19e8e09246be5a98d3c7d60a2500913e` |
| gas_deliverability_drift | downside | `results/baseline_sliced/summary__8dae8de3.json` | 29,386,251,626.76 | 2026-10-03T18:47:34 | `670f49cd1c36ec3fe4b151196d09944e` |
| gas_deliverability_drift | shock_recovery | `results/baseline_sliced/summary__d8abaa7e.json` | 27,924,844,628.33 | 2026-10-03T18:47:37 | `c78789448c9706a6d525d489eb4c5540` |
| gas_deliverability_gmp | baseline | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
| gas_deliverability_gmp | gas_deliv_flat | `results/baseline_sliced/summary__5fbb0421.json` | 27,678,637,278.66 | 2026-10-03T18:47:38 | `15ea4ab069dd9b58ec51f0d5a670f0b0` |
| gas_deliverability_gmp | gas_deliv_gmp_partial | `results/baseline_sliced/summary__81890b55.json` | 26,057,380,045.86 | 2026-10-03T18:47:40 | `a5e3bed00f8c52d0e201a27430a4aa91` |
| gas_deliverability_gmp | gas_deliv_gmp_full | `results/baseline_sliced/summary__7537e513.json` | 25,294,954,835.01 | 2026-10-03T18:47:41 | `84d5449a96953e2f87251b6654682304` |
| concessional_envelope | env_none | `results/baseline_sliced/summary__950458b0.json` | 27,608,922,551.52 | 2026-10-03T18:47:46 | `fe271ee3558939176c6b08587b50a273` |
| concessional_envelope | env_dares | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
| concessional_envelope | env_5x | `results/baseline_sliced/summary__d8237c02.json` | 26,306,524,895.20 | 2026-10-03T18:47:51 | `42769bcb322622bd932a9c2ecfa7fc02` |
| concessional_envelope | env_10x | `results/baseline_sliced/summary__2e0d019f.json` | 25,308,868,756.89 | 2026-10-03T18:47:52 | `d5672718f7cee8cbc54528b7cd3396b1` |
| concessional_envelope | env_dares_plus | `results/baseline_sliced/summary__0524a6e2.json` | 27,207,244,026.71 | 2026-10-03T18:47:49 | `6f7820867f42653988e40a09e72e94d6` |
| peak | peak_suppressed | `results/baseline_sliced/summary__2aa7e999.json` | 26,346,768,844.18 | 2026-10-03T18:48:07 | `0ba77ce5e0412e2877743f8d660a98f3` |
| peak | peak_unsuppressed | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
| peak | peak_high | `results/baseline_sliced/summary__15987301.json` | 28,496,350,206.94 | 2026-10-03T18:48:05 | `91d7b007ba4c1ac8c2242f713a4e1f47` |
| hydro | hydro_flat | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
| hydro | hydro_committed | `results/baseline_sliced/summary__33cbd361.json` | 26,389,359,344.35 | 2026-10-03T18:47:42 | `f7f9b4103f3d012201cb5b2c2b55c830` |
| hydro | hydro_ndc | `results/baseline_sliced/summary__9221de7b.json` | 12,602,684,561.59 | 2026-10-03T18:47:44 | `4676843d995205d621e52a5b3dfeebbd` |
| new_gas_deliverability | constrained | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
| new_gas_deliverability | exempt | `results/baseline_sliced/summary__6f3780de.json` | 25,311,786,185.37 | 2026-10-03T18:48:16 | `1cb792b2977d12ebe921d1a3757630df` |
| storage_cc | storage_cc_low | `results/baseline_sliced/summary__bbe7d77a.json` | 28,355,467,443.08 | 2026-10-03T18:48:13 | `fc8b78580bef96a820246248ee5ae580` |
| storage_cc | storage_cc_central | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
| storage_cc | storage_cc_high | `results/baseline_sliced/summary__857701d1.json` | 26,373,796,116.59 | 2026-10-03T18:48:15 | `5835edeab8d83a30f6ff7601ada1f855` |
| gas_build | gas_unconstrained | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
| gas_build | gas_ndc_17gw | `results/baseline_sliced/summary__024fdf9f.json` | 27,317,670,161.50 | 2026-10-03T18:48:08 | `85f4d556c680692a6269c443591b4ace` |
| gas_build | gas_none | `results/baseline_sliced/summary__376ed5e0.json` | 28,124,506,790.81 | 2026-10-03T18:48:10 | `b26ec73a82d65f6a3d27ce779bd18393` |
| gas_capex | gas_capex_low | `results/baseline_sliced/summary__d3e67c1a.json` | 26,440,889,855.27 | 2026-10-03T18:48:18 | `326aa78fbf98fa4ad9028cd86dace953` |
| gas_capex | gas_capex_central | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
| gas_capex | gas_capex_high | `results/baseline_sliced/summary__35bc64f7.json` | 27,378,416,469.15 | 2026-10-03T18:48:19 | `61daf3050b44f10f396d3b601ff10f8c` |
| ccgt_efficiency | ccgt_eff_central | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
| ccgt_efficiency | ccgt_eff_iso | `results/baseline_sliced/summary__6434340a.json` | 27,317,670,155.72 | 2026-10-03T18:48:21 | `1519782c99983a21d73212a45bd02d5e` |
| ccgt_efficiency | ccgt_eff_low | `results/baseline_sliced/summary__d8681e68.json` | 27,317,670,155.72 | 2026-10-03T18:48:22 | `6a81d483a9b5b886065a448716c27bf4` |
| dscr | dscr_central | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
| dscr | dscr_low | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
| dscr | dscr_high | `results/baseline_sliced/summary__fbe4d5b2.json` | 27,317,670,155.72 | 2026-10-03T18:47:56 | `904769f5baa843cfb4fbd3bccedf042e` |
| om | om_central | `results/baseline_sliced/summary__6bf84218.json` | 27,317,670,155.72 | 2026-10-03T18:47:55 | `e3f29e1b5041b34894700f182db6c9d6` |
