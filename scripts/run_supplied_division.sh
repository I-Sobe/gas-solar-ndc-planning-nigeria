#!/usr/bin/env bash
# TASK 98 driver: the Question 2 measurement (value of determining the concessional-commercial division).
# Solves, at the central arm set of scripts/run_phase_b.sh (C_* block) and each positive envelope size, the supplied-division arms
# of scripts/01b_run_baseline_sliced.py --conc-share-rule; results go to results/supplied_division/. The endogenous partners are the
# stored baseline_sliced files named below. Also runs the two regression arms (A.4). 18 solves, about 5 s each.
set -u
cd "$(dirname "$0")/.." || exit 1
eval "$(grep -E '^C_[A-Z_]+="[^"]*"$' scripts/run_phase_b.sh)"
run () {  # run ENV_CASE [01b args...]
  local envc="$1"; shift
  python scripts/01b_run_baseline_sliced.py --peak-case "$C_PEAK" --storage-cc-case "$C_STORAGE_CC" --gas-build-case "$C_GAS_BUILD" \
    --new-gas-deliverability "$C_NEW_GAS_DELIV" --gas-availability-case "$C_GAS_AVAIL" --gas-retirement-case "$C_GAS_RETIRE" \
    --gas-capex-case "$C_GAS_CAPEX" --ccgt-efficiency-case "$C_CCGT" --capital-recovery-case "$C_CR" --dscr-case "$C_DSCR" \
    --us-inflation "$C_INFL" --capital-treatment-case "$C_TREAT" --concessional-envelope-case "$envc" \
    --gas-deliverability-case "$C_GAS_DELIV" --conversion-case "$C_CONV" --hydro-case "$C_HYDRO" --tariff-case "$C_TARIFF" --om-case "$C_OM" "$@"
}
# A.4 regression arms (both must reproduce the endogenous / env_none objectives; see docs/task_log_98.md)
run env_dares --conc-share-rule split_central_realised
run env_dares --conc-share-rule split_zero
# B: endogenous partner (env case, stored summary hash8, first vintage in which the partner builds eligible capital)
for row in "env_dares 661b03c1 2035" "env_dares_plus 4dad4483 2035" "env_5x 29d7d360 2034" "env_10x 9f6bab89 2033"; do
  set -- $row
  run "$1" --conc-share-rule split_prorata --conc-share-reference-summary "results/baseline_sliced/summary__$2.json"
  run "$1" --conc-share-rule split_frontload                              # first model year (2025)
  run "$1" --conc-share-rule split_frontload --conc-share-first-year "$3"  # supplementary: first vintage the partner builds
done
