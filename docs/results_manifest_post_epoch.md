# Post-epoch results manifest

Generated 2026-10-20 03:20 at HEAD `0af709f` (tracked tree clean) by a throwaway script from the `results/` tree as it stands.
`results/` is gitignored, so this file is the committed record of what the post-epoch tree held.

**31 `arms_<hash8>` directories.** One row each. Columns: path; hash8; runner (`ARMS.json` invocation[0]); `ARMS.json` mtime; newest file mtime in the directory; full arm tag (`ARMS.json`).
Headline figures are read from every JSON/CSV in the directory that carries `npv_total_cost_usd`, `storage_capacity_2045_mwh`, `storage_capacity_2054_mwh` or `solar_cumulative_built_2045_mw` (any nesting; a CSV column counts). One value is printed as is; several as `min .. max (n=count)`. Each file is followed by the first 12 hex digits of its sha256. A runner whose outputs carry none of the four (REL, GAS, DEM etc. report other quantities) shows no figures.

**Not covered by the `arms_` convention (so not rows below):** baseline_sliced (summary__<hash8>.json): 127 files; ndc (<case>_<hash8>/): 62 files; ndc_eaas (<case>_<hash8>/): 20 files; f3: 21 files. These runners name their own files by hash8 (`complete_arm_tag`), not by an `arms_` directory. The central anchor is `results/baseline_sliced/summary__6046f759.json`.

Pre-epoch and post-epoch files share the `results/` tree; the `arms_` layout exists only since TASK 53 §2.5, so every row here is post-epoch by construction. `*_gate` rows are Phase C Group 0 gate runs, not Phase B outputs.

| path | hash8 | runner | ARMS.json mtime | newest file | files | arm tag |
|---|---|---|---|---|---|---|
| `results/capital_frontier/arms_92390a63` | `92390a63` | `28_run_capital_frontier.py` | 2026-10-20 00:44 | 2026-10-20 00:45 | 7 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_deliverability_case=baseline__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/carbon_price_sweep/arms_92390a63` | `92390a63` | `19_run_carbon_price_sweep.py` | 2026-10-19 23:48 | 2026-10-19 23:49 | 6 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_deliverability_case=baseline__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/cf1/arms_92390a63` | `92390a63` | `20_run_cf1_solar_yield_sensitivity.py` | 2026-10-19 23:48 | 2026-10-19 23:48 | 2 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_deliverability_case=baseline__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/dem2/arms_31d60098` | `31d60098` | `13_run_dem2_growth_gas_interaction.py` | 2026-10-19 23:56 | 2026-10-19 23:57 | 6 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__us_inflation=0.02` |
| `results/emissions_baseline_agreement/arms_b4a83985` | `b4a83985` | `57_emissions_baseline_builder_agreement.py` | 2026-10-19 23:36 | 2026-10-19 23:36 | 6 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_none__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_deliverability_case=baseline__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/fin2/arms_5e1e4f68` | `5e1e4f68` | `04_run_fin2_tariff_sweep.py` | 2026-10-19 23:52 | 2026-10-19 23:56 | 4 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/fin2_gate/arms_5e1e4f68` | `5e1e4f68` | `04_run_fin2_tariff_sweep.py` | 2026-10-20 02:57 | 2026-10-20 03:00 | 4 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/fin3/arms_92390a63` | `92390a63` | `05_run_fin3_capital_sweep.py` | 2026-10-20 01:17 | 2026-10-20 01:18 | 3 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_deliverability_case=baseline__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/fin3_gate/arms_92390a63` | `92390a63` | `05_run_fin3_capital_sweep.py` | 2026-10-20 03:01 | 2026-10-20 03:01 | 3 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_deliverability_case=baseline__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/gas1/arms_be9472ef` | `be9472ef` | `06_run_gas1_shadow_benchmarks.py` | 2026-10-19 23:50 | 2026-10-19 23:50 | 5 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/gas1_gate/arms_be9472ef` | `be9472ef` | `06_run_gas1_shadow_benchmarks.py` | 2026-10-20 03:01 | 2026-10-20 03:01 | 5 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/gas2/arms_be9472ef` | `be9472ef` | `07_run_gas2_eaas_gas_relief.py` | 2026-10-19 23:50 | 2026-10-19 23:51 | 3 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/gas2_gate/arms_be9472ef` | `be9472ef` | `07_run_gas2_eaas_gas_relief.py` | 2026-10-20 03:01 | 2026-10-20 03:02 | 3 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/gas3/arms_02e16d75` | `02e16d75` | `08_run_gas3_regime_ndc_feasibility.py` | 2026-10-19 23:51 | 2026-10-19 23:52 | 6 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/gas3_gate/arms_02e16d75` | `02e16d75` | `08_run_gas3_regime_ndc_feasibility.py` | 2026-10-20 03:02 | 2026-10-20 03:02 | 6 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/genset_arm_sensitivity/arms_5e1e4f68` | `5e1e4f68` | `99c_run_genset_arm_sensitivity.py` | 2026-10-19 23:50 | 2026-10-19 23:50 | 2 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/hydro_minflow/arms_1b623521` | `1b623521` | `27_run_hydro_minflow_sensitivity.py` | 2026-10-20 02:47 | 2026-10-20 02:47 | 6 | `capital_case=unconstrained__capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_deliverability_case=baseline__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__hydro_min_flow_values=0.30_0.50_0.70__new_gas_deliverability=constrained__om_case=om_none__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__solar_build_case=deployment_unconstrained__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/hydro_minflow/arms_77bf0ff4` | `77bf0ff4` | `27_run_hydro_minflow_sensitivity.py` | 2026-10-20 02:47 | 2026-10-20 02:47 | 6 | `capital_case=unconstrained__capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_deliverability_case=baseline__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__hydro_min_flow_values=0.30_0.50_0.70__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__solar_build_case=deployment_unconstrained__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/leakage_arm/arms_5e1e4f68` | `5e1e4f68` | `24_run_leakage_arm.py` | 2026-10-19 23:49 | 2026-10-19 23:49 | 2 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/monte_carlo/arms_be9472ef` | `be9472ef` | `17_run_monte_carlo.py` | 2026-10-20 00:25 | 2026-10-20 00:34 | 7 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/nirp_cap_check/arms_92390a63` | `92390a63` | `58_nirp_cap_binding_check.py` | 2026-10-20 01:22 | 2026-10-20 01:22 | 2 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_deliverability_case=baseline__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/obligation_trajectory/arms_5e1e4f68` | `5e1e4f68` | `25_run_obligation_trajectory.py` | 2026-10-19 23:49 | 2026-10-19 23:50 | 3 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/parameter_uncertainty/arms_ce5387ec` | `ce5387ec` | `29_run_parameter_uncertainty.py` | 2026-10-20 00:45 | 2026-10-20 01:17 | 3 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_deliverability_case=baseline__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__n_base=256__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__seed=42__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/pareto_frontier/arms_92390a63` | `92390a63` | `18_run_pareto_frontier.py` | 2026-10-20 00:42 | 2026-10-20 00:44 | 6 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_deliverability_case=baseline__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/pol1/arms_92390a63` | `92390a63` | `15_run_pol1_ndc_comparison.py` | 2026-10-19 23:47 | 2026-10-19 23:48 | 6 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_deliverability_case=baseline__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__new_gas_deliverability=constrained__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/rel2/arms_02e16d75` | `02e16d75` | `10_run_rel2_marginal_cost.py` | 2026-10-20 00:04 | 2026-10-20 00:05 | 5 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/rel2/arms_999c6367` | `999c6367` | `10_run_rel2_marginal_cost.py` | 2026-10-20 00:05 | 2026-10-20 00:06 | 5 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_observed__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/rel2_gate/arms_02e16d75` | `02e16d75` | `10_run_rel2_marginal_cost.py` | 2026-10-20 03:02 | 2026-10-20 03:03 | 5 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__tariff_case=tariff_central__us_inflation=0.02` |
| `results/rel3/arms_31d60098` | `31d60098` | `11_run_rel3_financing_frontier.py` | 2026-10-20 00:06 | 2026-10-20 00:14 | 5 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__us_inflation=0.02` |
| `results/rel3/arms_5037b4db` | `5037b4db` | `11_run_rel3_financing_frontier.py` | 2026-10-20 00:14 | 2026-10-20 00:23 | 5 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_observed__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__us_inflation=0.02` |
| `results/rel3_gate/arms_31d60098` | `31d60098` | `11_run_rel3_financing_frontier.py` | 2026-10-20 03:03 | 2026-10-20 03:11 | 5 | `capital_recovery_case=cr_central__capital_treatment_case=f1_tranche__capital_window_case=arrears__ccgt_efficiency_case=ccgt_eff_central__concessional_envelope_case=env_dares__conversion_case=conv_central__dscr_case=dscr_central__gas_availability_case=gas_avail_nameplate__gas_build_case=gas_unconstrained__gas_capex_case=gas_capex_central__gas_retirement_case=gas_retire_vintage_30__hydro_case=hydro_flat__om_case=om_central__peak_multiple_case=peak_unsuppressed__reserve_margin=nirp_spinning__storage_capex_case=declining_nrel__storage_cc_case=storage_cc_central__us_inflation=0.02` |

## Headline figures by directory

### `results/capital_frontier/arms_92390a63` (`92390a63`, 28_run_capital_frontier.py)

No file in this directory carries any of the four headline quantities.

### `results/carbon_price_sweep/arms_92390a63` (`92390a63`, 19_run_carbon_price_sweep.py)

No file in this directory carries any of the four headline quantities.

### `results/cf1/arms_92390a63` (`92390a63`, 20_run_cf1_solar_yield_sensitivity.py)

No file in this directory carries any of the four headline quantities.

### `results/dem2/arms_31d60098` (`31d60098`, 13_run_dem2_growth_gas_interaction.py)

* `dem2_results.csv` (sha256 `ba6b28bba034`): npv_total_cost_usd = 20,699,525,585.52 .. 113,413,668,162.03 (n=72)

### `results/emissions_baseline_agreement/arms_b4a83985` (`b4a83985`, 57_emissions_baseline_builder_agreement.py)

No file in this directory carries any of the four headline quantities.

### `results/fin2/arms_5e1e4f68` (`5e1e4f68`, 04_run_fin2_tariff_sweep.py)

No file in this directory carries any of the four headline quantities.

### `results/fin2_gate/arms_5e1e4f68` (`5e1e4f68`, 04_run_fin2_tariff_sweep.py)

No file in this directory carries any of the four headline quantities.

### `results/fin3/arms_92390a63` (`92390a63`, 05_run_fin3_capital_sweep.py)

* `capital_sweep_results.csv` (sha256 `720f5f278b5b`): npv_total_cost_usd = 33,959,713,169.84 .. 61,324,395,024.06 (n=15)

### `results/fin3_gate/arms_92390a63` (`92390a63`, 05_run_fin3_capital_sweep.py)

* `capital_sweep_results.csv` (sha256 `720f5f278b5b`): npv_total_cost_usd = 33,959,713,169.84 .. 61,324,395,024.06 (n=15)

### `results/gas1/arms_be9472ef` (`be9472ef`, 06_run_gas1_shadow_benchmarks.py)

* `gas_shadow_matrix.csv` (sha256 `4c8d3b28b823`): npv_total_cost_usd = 24,371,953,576.58 .. 38,700,007,347.40 (n=8)

### `results/gas1_gate/arms_be9472ef` (`be9472ef`, 06_run_gas1_shadow_benchmarks.py)

* `gas_shadow_matrix.csv` (sha256 `4c8d3b28b823`): npv_total_cost_usd = 24,371,953,576.58 .. 38,700,007,347.40 (n=8)

### `results/gas2/arms_be9472ef` (`be9472ef`, 07_run_gas2_eaas_gas_relief.py)

No file in this directory carries any of the four headline quantities.

### `results/gas2_gate/arms_be9472ef` (`be9472ef`, 07_run_gas2_eaas_gas_relief.py)

No file in this directory carries any of the four headline quantities.

### `results/gas3/arms_02e16d75` (`02e16d75`, 08_run_gas3_regime_ndc_feasibility.py)

* `regime_ndc_matrix.csv` (sha256 `1f0eb7a9b191`): npv_total_cost_usd = 31,187,879,120.47 .. 46,590,667,173.83 (n=28)

### `results/gas3_gate/arms_02e16d75` (`02e16d75`, 08_run_gas3_regime_ndc_feasibility.py)

* `regime_ndc_matrix.csv` (sha256 `1f0eb7a9b191`): npv_total_cost_usd = 31,187,879,120.47 .. 46,590,667,173.83 (n=28)

### `results/genset_arm_sensitivity/arms_5e1e4f68` (`5e1e4f68`, 99c_run_genset_arm_sensitivity.py)

No file in this directory carries any of the four headline quantities.

### `results/hydro_minflow/arms_1b623521` (`1b623521`, 27_run_hydro_minflow_sensitivity.py)

* `hydro_minflow_0.30.json` (sha256 `a525157288e2`): npv_total_cost_usd = 18,452,055,664.69
* `hydro_minflow_0.50.json` (sha256 `8513f5e5474b`): npv_total_cost_usd = 18,466,184,917.99
* `hydro_minflow_0.70.json` (sha256 `c1e11255a802`): npv_total_cost_usd = 18,480,870,053.51
* `hydro_minflow_sweep_0.30_0.50_0.70.csv` (sha256 `9def1d0cf4c5`): npv_total_cost_usd = 18,452,055,664.69 .. 18,480,870,053.51 (n=54)

### `results/hydro_minflow/arms_77bf0ff4` (`77bf0ff4`, 27_run_hydro_minflow_sensitivity.py)

* `hydro_minflow_0.30.json` (sha256 `419e70f28d2f`): npv_total_cost_usd = 24,970,088,257.74
* `hydro_minflow_0.50.json` (sha256 `7e1f86518228`): npv_total_cost_usd = 24,986,620,949.18
* `hydro_minflow_0.70.json` (sha256 `0b1baebacd2c`): npv_total_cost_usd = 25,003,848,458.53
* `hydro_minflow_sweep_0.30_0.50_0.70.csv` (sha256 `4c7ae1322227`): npv_total_cost_usd = 24,970,088,257.74 .. 25,003,848,458.53 (n=54)

### `results/leakage_arm/arms_5e1e4f68` (`5e1e4f68`, 24_run_leakage_arm.py)

No file in this directory carries any of the four headline quantities.

### `results/monte_carlo/arms_be9472ef` (`be9472ef`, 17_run_monte_carlo.py)

No file in this directory carries any of the four headline quantities.

### `results/nirp_cap_check/arms_92390a63` (`92390a63`, 58_nirp_cap_binding_check.py)

No file in this directory carries any of the four headline quantities.

### `results/obligation_trajectory/arms_5e1e4f68` (`5e1e4f68`, 25_run_obligation_trajectory.py)

No file in this directory carries any of the four headline quantities.

### `results/parameter_uncertainty/arms_ce5387ec` (`ce5387ec`, 29_run_parameter_uncertainty.py)

No file in this directory carries any of the four headline quantities.

### `results/pareto_frontier/arms_92390a63` (`92390a63`, 18_run_pareto_frontier.py)

No file in this directory carries any of the four headline quantities.

### `results/pol1/arms_92390a63` (`92390a63`, 15_run_pol1_ndc_comparison.py)

* `pol1_results.csv` (sha256 `57a646a68bc9`): npv_total_cost_usd = 26,396,097,238.14 .. 52,792,376,799.39 (n=8)
* `trajectory_comparison.csv` (sha256 `aa07bfdff407`): npv_total_cost_usd = 26,396,097,238.14 .. 52,792,376,799.39 (n=8)

### `results/rel2/arms_02e16d75` (`02e16d75`, 10_run_rel2_marginal_cost.py)

No file in this directory carries any of the four headline quantities.

### `results/rel2/arms_999c6367` (`999c6367`, 10_run_rel2_marginal_cost.py)

No file in this directory carries any of the four headline quantities.

### `results/rel2_gate/arms_02e16d75` (`02e16d75`, 10_run_rel2_marginal_cost.py)

No file in this directory carries any of the four headline quantities.

### `results/rel3/arms_31d60098` (`31d60098`, 11_run_rel3_financing_frontier.py)

No file in this directory carries any of the four headline quantities.

### `results/rel3/arms_5037b4db` (`5037b4db`, 11_run_rel3_financing_frontier.py)

No file in this directory carries any of the four headline quantities.

### `results/rel3_gate/arms_31d60098` (`31d60098`, 11_run_rel3_financing_frontier.py)

No file in this directory carries any of the four headline quantities.
