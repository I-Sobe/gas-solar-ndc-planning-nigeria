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
# Logs to logs/sweep/. Nothing is deleted; re-running overwrites by name.
# ============================================================================

set -uo pipefail          # NOT -e: one failed arm must not kill the sweep
cd "$(dirname "$0")/.."   # repo root
mkdir -p logs/sweep

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

  local log="logs/sweep/${label}.log"
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

# ============================================================================
if [ "$SECTION" = "ALL" ] || [ "$SECTION" = "A" ]; then
banner "A -- CENTRAL CONFIGURATION, all four live runners"

# task_om_implementation: A_REGRESSION_ANCHOR below runs --om-case
# om_none; every other run in this file (this section and B/C) runs
# om_central. The two are NOT comparable to each other -- the anchor's
# job is reproducing the pre-O&M value exactly, not reporting a cost
# result under the central O&M assumption.

run 01b_run_baseline_sliced.py           A_central_baseline
run 02_run_ndc_caps.py                   A_central_ndc_public
run 03_run_ndc_eaas.py                   A_central_ndc_eaas
run 26_run_f3_identification.py          A_central_f3_moderate \
    extra="--capital-case moderate --tol 1e-6"

# The regression anchor. NOT central -- this is the attribution arm that
# must reproduce 21.328 bn USD / 43,539.97566 MWh (re-baselined
# 2026-09-25; was 21,811,042,472.821114 USD / 45,502.9754 MWh -- see
# register L58). If it does not, the sweep is invalid and nothing below
# it should be read.
#
# The two commits that moved this pair (and A_central_baseline below,
# now 27.318 bn / 63,056.31077 MWh, was 27.954 bn / 65,612.985 MWh):
#   6d5a076 -- organic_central demand-growth arm, 4.0% -> 3.9% (L54)
#   30561ce -- cr_central concessional RATE, IDA20 2.80% -> IDA21 2.12% (L53)
# plus an L_conc tenor correction (30yr -> 25yr, closing the rate/tenor
# hybrid the rate-only commit above left behind) applied in the working
# tree alongside this re-run, not yet its own separate commit at the
# time this comment was written -- see docs/task_log_rate_coherence.md.
run 01b_run_baseline_sliced.py           A_REGRESSION_ANCHOR \
    conv=conv_none treat=pre_f1_overnight dscr=dscr_regression \
    tariff=tariff_legacy_65m om=om_none
fi

# ============================================================================
if [ "$SECTION" = "ALL" ] || [ "$SECTION" = "B" ]; then
banner "B -- ONE-AT-A-TIME ARMS (01b), one dimension off centre each"

echo "-- capital recovery (rate x tenor)"
run 01b_run_baseline_sliced.py  B_cr_ida21          cr=cr_ida21
run 01b_run_baseline_sliced.py  B_cr_comm_low       cr=cr_comm_low
run 01b_run_baseline_sliced.py  B_cr_comm_high      cr=cr_comm_high

echo "-- conversion"
run 01b_run_baseline_sliced.py  B_conv_none         conv=conv_none
run 01b_run_baseline_sliced.py  B_conv_low          conv=conv_low
run 01b_run_baseline_sliced.py  B_conv_high         conv=conv_high

echo "-- gas deliverability, historical-drift family"
run 01b_run_baseline_sliced.py  B_deliv_downside    gas_deliv=downside
run 01b_run_baseline_sliced.py  B_deliv_upside      gas_deliv=upside
run 01b_run_baseline_sliced.py  B_deliv_shock       gas_deliv=shock_recovery

echo "-- gas deliverability, GMP policy-target family (register L13:"
echo "   NOT comparable to the drift family without saying which)"
run 01b_run_baseline_sliced.py  B_deliv_gmp_flat    gas_deliv=gas_deliv_flat
run 01b_run_baseline_sliced.py  B_deliv_gmp_partial gas_deliv=gas_deliv_gmp_partial
run 01b_run_baseline_sliced.py  B_deliv_gmp_full    gas_deliv=gas_deliv_gmp_full

echo "-- hydro (only hydro_flat reproduces the backcast; others depart"
echo "   from the 2024 observation by construction)"
run 01b_run_baseline_sliced.py  B_hydro_committed   hydro=hydro_committed
run 01b_run_baseline_sliced.py  B_hydro_ndc         hydro=hydro_ndc

echo "-- concessional envelope"
run 01b_run_baseline_sliced.py  B_env_none          env=env_none
run 01b_run_baseline_sliced.py  B_env_dares_plus    env=env_dares_plus
run 01b_run_baseline_sliced.py  B_env_5x            env=env_5x
run 01b_run_baseline_sliced.py  B_env_10x           env=env_10x

echo "-- DSCR coverage cushion"
run 01b_run_baseline_sliced.py  B_dscr_low          dscr=dscr_low
run 01b_run_baseline_sliced.py  B_dscr_high         dscr=dscr_high

echo "-- gas fleet availability (P2-9: 0.80 assumed vs 30.6% measured)"
run 01b_run_baseline_sliced.py  B_gas_avail_obs     gas_avail=gas_avail_observed
echo "-- gas retirement (register L8)"
run 01b_run_baseline_sliced.py  B_retire_none       gas_retire=gas_retire_none
run 01b_run_baseline_sliced.py  B_retire_v40        gas_retire=gas_retire_vintage_40
run 01b_run_baseline_sliced.py  B_retire_v50        gas_retire=gas_retire_vintage_50

echo "-- demand shape"
run 01b_run_baseline_sliced.py  B_peak_high         peak=peak_high
run 01b_run_baseline_sliced.py  B_peak_suppressed   peak=peak_suppressed

# TASK 15, Part B.2 (docs/task_log_15b_unexercised_arms.md): twelve of
# TASK 14's nineteen usable-unexercised values are addable here without
# any new CLI flag -- seven more (demand_case x2, land_case x2,
# solar_build_case x1, carbon_case x2) are blocked because no runner in
# this sweep exposes a flag for those dimensions at all (reported, not
# wired, per B.1's own treatment of the same category of gap for
# bank_tenor_case/blend_case -- which turned out not to need wiring at
# all, since both are already fully exercised by
# 04_run_fin2_tariff_sweep.py's own internal sweep over every
# blended_finance_scenarios() entry, pinned to bank_tenor_legacy).
echo "-- gas fleet build policy"
run 01b_run_baseline_sliced.py  B_gas_build_ndc     gas_build=gas_ndc_17gw
run 01b_run_baseline_sliced.py  B_gas_build_none    gas_build=gas_none

echo "-- gas fleet retirement, pre-2.11 proxy"
run 01b_run_baseline_sliced.py  B_gas_retire_proxy  gas_retire=gas_retire_proxy

echo "-- storage firm-capacity credit"
run 01b_run_baseline_sliced.py  B_storage_cc_low    storage_cc=storage_cc_low
run 01b_run_baseline_sliced.py  B_storage_cc_high   storage_cc=storage_cc_high

echo "-- new gas capacity fuel-deliverability treatment"
run 01b_run_baseline_sliced.py  B_new_gas_exempt    extra="--new-gas-deliverability exempt"

echo "-- new-build gas unit cost"
run 01b_run_baseline_sliced.py  B_gas_capex_low     extra="--gas-capex-case gas_capex_low"
run 01b_run_baseline_sliced.py  B_gas_capex_high    extra="--gas-capex-case gas_capex_high"

echo "-- new-build CCGT efficiency"
run 01b_run_baseline_sliced.py  B_ccgt_eff_iso      extra="--ccgt-efficiency-case ccgt_eff_iso"
run 01b_run_baseline_sliced.py  B_ccgt_eff_low      extra="--ccgt-efficiency-case ccgt_eff_low"

echo "-- public capital envelope, extremes not covered by the F3 cells"
run 26_run_f3_identification.py B_f3_adequacy \
    extra="--capital-case adequacy --tol 1e-6"
run 26_run_f3_identification.py B_f3_expansion \
    extra="--capital-case expansion --tol 1e-6"
fi

# ============================================================================
if [ "$SECTION" = "ALL" ] || [ "$SECTION" = "C" ]; then
banner "C -- INTERACTION CELLS (suspected, not inferable one-at-a-time)"

echo "-- deliverability x cap (F-12: worth 3.16 bn uncapped, ZERO capped)"
run 02_run_ndc_caps.py  C_cap_x_deliv_flat     gas_deliv=gas_deliv_flat
run 02_run_ndc_caps.py  C_cap_x_deliv_partial  gas_deliv=gas_deliv_gmp_partial

echo "-- conversion x cap (F-16: worth 33-36 bn under caps vs 3-4 bn free)"
run 02_run_ndc_caps.py  C_cap_x_conv_none      conv=conv_none

echo "-- EaaS x cap (F-27: the 8.895 bn financing-channel result)"
run 03_run_ndc_eaas.py  C_eaas_x_cap_conv_none conv=conv_none

echo "-- envelope x public budget (F-26/F-28: the identification interval"
echo "   is bounded by the budget below and bankability above)"
run 26_run_f3_identification.py C_f3_tight \
    extra="--capital-case tight --tol 1e-6"
run 26_run_f3_identification.py C_f3_unconstrained \
    extra="--capital-case unconstrained --tol 1e-6"

echo "-- tariff band at the identification test (F-29: F-27 is robust"
echo "   to the tariff; confirm at the sourced endpoints)"
run 26_run_f3_identification.py C_f3_tariff_low \
    tariff=tariff_low extra="--capital-case moderate --tol 1e-6"
run 26_run_f3_identification.py C_f3_tariff_high \
    tariff=tariff_high extra="--capital-case moderate --tol 1e-6"

# TASK 15, Part B.3 (docs/task_log_15b_unexercised_arms.md): patterned on
# C_f3_tariff_low/C_f3_tariff_high above -- one arm crossed against 26_'s
# own financing-identification machinery (capital-case held at its
# central "moderate", the F1 per-tranche treatment 26_ runs under by
# default) rather than the one-at-a-time Section B cells, since the
# question is specifically whether gas_avail_observed interacts with the
# concessional-commercial split (F-26/F-28's identification interval),
# not just its own marginal effect.
echo "-- gas availability x financing treatment (does the availability"
echo "   assumption interact with the concessional-commercial split?)"
run 26_run_f3_identification.py C_gas_avail_obs_pre_f1 \
    gas_avail=gas_avail_observed \
    extra="--capital-case moderate --tol 1e-6"
fi

# ============================================================================
banner "SWEEP COMPLETE"
echo "failed runs: $FAILED"
echo "logs in logs/sweep/"
echo
echo "CHECK FIRST: A_REGRESSION_ANCHOR must read 21.328 bn (re-baselined"
echo "2026-09-25, register L58; was 21.811 bn -- see docs/task_log_rate_coherence.md)."
echo "If it does not, every other number in this sweep is unattributable."
