#!/usr/bin/env bash
# ============================================================================
# Central-configuration sweep
# ============================================================================
#
# PURPOSE
#   Put every reported number on ONE model state. Findings F-1..F-29 were
#   produced across five different states (pre-conversion, pre-hydro-arms,
#   pre-tariff-arm, at dscr_regression, at the 65M silent-default tariff).
#   This sweep re-runs the headline set at a single declared central
#   configuration, with one-at-a-time arms so every delta is attributable.
#
# METHOD
#   Section A : central configuration, all four live runners.
#   Section B : one-at-a-time arms on 01b -- exactly one dimension differs
#               from centre per run (correction plan Sec 7.2).
#   Section C : deliberate two-way cells where interaction is already
#               suspected. One-at-a-time from an uncapped centre would have
#               missed F-12 entirely (deliverability worth 3.16 bn uncapped,
#               exactly zero under a binding cap), so these are run
#               explicitly rather than inferred.
#
# CENTRAL CONFIGURATION
#   conversion_case            conv_central
#   hydro_case                 hydro_flat        (only arm reproducing the
#                                                 2024 backcast at -3.4%)
#   gas_deliverability_case    baseline          (historical-drift family;
#                                                 the GMP family encodes a
#                                                 policy target -- register
#                                                 L13)
#   capital_recovery_case      cr_central
#   dscr_case                  dscr_central      (1.30, sourced; the 1.10
#                                                 regression arm is not a
#                                                 candidate central case)
#   concessional_envelope_case env_dares
#   tariff_case                tariff_central    (95 USD/MWh -- the MIDPOINT
#                                                 of Nigeria's 2016 NBET
#                                                 negotiation positions, 75
#                                                 and 115. A CONSTRUCTION,
#                                                 not an independently
#                                                 observed price. Reported
#                                                 with 75/115 as the band.)
#   capital_treatment_case     f1_tranche
#
#   Physical arms held at their established values throughout:
#   peak_unsuppressed / storage_cc_central / gas_unconstrained /
#   constrained / gas_avail_nameplate / gas_retire_vintage_30 /
#   gas_capex_central / ccgt_eff_central / us_inflation 0.020
#
# SCOPED CONDITIONS carried by every result (see the register):
#   L5  storage capacity credit is flat 1.0 -- storage magnitude is a
#       residual once gas reaches its floor, not a cost optimum
#   L16 EaaS and public solar are physically identical -- financing
#       comparison only, says nothing about siting
#   L17 bankability discounts at the retired pre-F1 blend over a horizon
#       the objective contradicts; all EaaS results are conditional on it
#   P2-9 gas_avail_nameplate is 0.80 against a measured 30.6% (29.5% before TASK 73)
#   Sliced shape factors season_solar_factor and hydro_season_factor are
#       [SOURCE NEEDED]; no sliced result is final until they are sourced
#
# EPOCH
#   Anchor values in this file (Section A's regression check, the
#   completion trailer) are as of the rate-coherence task (2026-09-25:
#   organic_central 4.0%->3.9%, register L54; cr_central concessional
#   rate AND tenor to IDA21, register L53) -- update these values again
#   if either the demand-growth arm or capital_recovery_scenarios()'s
#   central case changes. Previously datable to the O&M implementation
#   (task_om_implementation): --om-case is now a required arm, and
#   wiring it raised the central configuration's objective by roughly
#   31% against the pre-O&M objective.
#
# USAGE
#   bash scripts/run_central_sweep.sh            # everything
#   bash scripts/run_central_sweep.sh A          # central only
#   bash scripts/run_central_sweep.sh B          # one-at-a-time only
#   bash scripts/run_central_sweep.sh C          # interaction cells only
#
# Logs to logs/batch3_26/. Nothing is deleted; re-running overwrites by name.
# ============================================================================

set -uo pipefail          # NOT -e: one failed arm must not kill the sweep
cd "$(dirname "$0")/.."   # repo root
mkdir -p logs/batch3_26

SECTION="${1:-ALL}"

# ---- central configuration -------------------------------------------------
C_PEAK="peak_unsuppressed"
C_STORAGE_CC="storage_cc_central"
C_GAS_BUILD="gas_unconstrained"
C_NEW_GAS_DELIV="constrained"
C_GAS_AVAIL="gas_avail_nameplate"
C_GAS_RETIRE="gas_retire_vintage_30"
C_GAS_CAPEX="gas_capex_central"
C_CCGT="ccgt_eff_central"
C_GAS_DELIV="baseline"
C_CONV="conv_central"
C_HYDRO="hydro_flat"
C_TARIFF="tariff_central"
C_TREAT="f1_tranche"
C_DSCR="dscr_central"
C_CR="cr_central"
C_ENV="env_dares"
C_INFL="0.020"
# task_om_implementation: required=True on 01_run_baseline.py,
# 01b_run_baseline_sliced.py, 02_run_eaas.py, 02_run_ndc_caps.py,
# 03_run_ndc_eaas.py only. om_central is the new central O&M case.
C_OM="om_central"

FAILED=0

# run <runner> <label> [key=value ...]   -- any key overrides the centre
run () {
  local runner="$1"; shift
  local label="$1";  shift

  local peak=$C_PEAK storage_cc=$C_STORAGE_CC gas_build=$C_GAS_BUILD
  local new_gas_deliv=$C_NEW_GAS_DELIV gas_avail=$C_GAS_AVAIL
  local gas_retire=$C_GAS_RETIRE gas_capex=$C_GAS_CAPEX ccgt=$C_CCGT
  local gas_deliv=$C_GAS_DELIV conv=$C_CONV hydro=$C_HYDRO tariff=$C_TARIFF
  local treat=$C_TREAT dscr=$C_DSCR cr=$C_CR env=$C_ENV infl=$C_INFL
  local om=$C_OM
  local extra=""

  for kv in "$@"; do
    case "${kv%%=*}" in
      peak)          peak="${kv#*=}" ;;
      storage_cc)    storage_cc="${kv#*=}" ;;
      gas_build)     gas_build="${kv#*=}" ;;
      gas_deliv)     gas_deliv="${kv#*=}" ;;
      conv)          conv="${kv#*=}" ;;
      hydro)         hydro="${kv#*=}" ;;
      tariff)        tariff="${kv#*=}" ;;
      treat)         treat="${kv#*=}" ;;
      dscr)          dscr="${kv#*=}" ;;
      cr)            cr="${kv#*=}" ;;
      env)           env="${kv#*=}" ;;
      gas_avail)     gas_avail="${kv#*=}" ;;
      gas_retire)    gas_retire="${kv#*=}" ;;
      om)            om="${kv#*=}" ;;
      extra)         extra="${kv#*=}" ;;
      *) echo "  !! unknown override: $kv"; return 1 ;;
    esac
  done

  # task_om_implementation: --om-case is NOT a flag on every runner this
  # function invokes (26_run_f3_identification.py does not expose it --
  # it resolves om_case internally). Gate the flag on the runner, not
  # add it unconditionally, or 26's invocations below break on an
  # unrecognised argument.
  local om_flag=()
  case "$runner" in
    01_run_baseline.py|01b_run_baseline_sliced.py|02_run_eaas.py|02_run_ndc_caps.py|03_run_ndc_eaas.py)
      om_flag=(--om-case "$om")
      ;;
  esac

  local log="logs/batch3_26/${label}.log"
  printf '%-46s ' "$label"

  # shellcheck disable=SC2086
  python3 "scripts/${runner}" \
    --peak-case "$peak" --storage-cc-case "$storage_cc" \
    --gas-build-case "$gas_build" --new-gas-deliverability "$new_gas_deliv" \
    --gas-availability-case "$gas_avail" \
    --gas-retirement-case "$gas_retire" \
    --gas-capex-case "$gas_capex" --ccgt-efficiency-case "$ccgt" \
    --gas-deliverability-case "$gas_deliv" --conversion-case "$conv" \
    --hydro-case "$hydro" --tariff-case "$tariff" \
    --capital-treatment-case "$treat" --dscr-case "$dscr" \
    --capital-recovery-case "$cr" --us-inflation "$infl" \
    --concessional-envelope-case "$env" \
    "${om_flag[@]}" \
    $extra \
    > "$log" 2>&1

  if [ $? -ne 0 ]; then
    echo "FAILED  -> $log"
    FAILED=$((FAILED+1))
    return 1
  fi

  local obj
  obj=$(grep -oE 'objective +: +[0-9.]+ bn USD' "$log" | tail -1 | grep -oE '[0-9.]+' | head -1)
  if [ -n "$obj" ]; then
    echo "obj ${obj} bn"
  else
    # multi-case runners (02, 03) print per case; report the count instead
    local n
    n=$(grep -cE '^--- .* saved ---|^saved -> ' "$log")
    echo "ok (${n} solves)"
  fi
}

banner () { echo; echo "════════════════════════════════════════════════════"; echo "$1"; echo "════════════════════════════════════════════════════"; }
run 26_run_f3_identification.py          A_central_f3_moderate \
    extra="--capital-case moderate --tol 1e-6"
run 26_run_f3_identification.py B_f3_adequacy \
    extra="--capital-case adequacy --tol 1e-6"
run 26_run_f3_identification.py B_f3_expansion \
    extra="--capital-case expansion --tol 1e-6"
run 26_run_f3_identification.py C_f3_tight \
    extra="--capital-case tight --tol 1e-6"
run 26_run_f3_identification.py C_f3_unconstrained \
    extra="--capital-case unconstrained --tol 1e-6"
run 26_run_f3_identification.py C_f3_tariff_low \
    tariff=tariff_low extra="--capital-case moderate --tol 1e-6"
run 26_run_f3_identification.py C_f3_tariff_high \
    tariff=tariff_high extra="--capital-case moderate --tol 1e-6"
run 26_run_f3_identification.py C_gas_avail_obs_pre_f1 \
    gas_avail=gas_avail_observed \
    extra="--capital-case moderate --tol 1e-6"
echo "failed runs: $FAILED"
