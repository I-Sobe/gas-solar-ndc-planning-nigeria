#!/usr/bin/env bash
set -u
export PYTHONIOENCODING=utf-8
# Snapshot target. A bare name is resolved under ./snapshots/ -- Git Bash
# maps /tmp to %TEMP%, but python is a native Windows binary and reads /tmp
# literally as C:\tmp, so /tmp snapshots are unreadable from Python. Keep
# snapshots repo-relative.
SNAPSHOT_ARG="${1:-}"
SNAPSHOT_DIR=""
if [ -n "$SNAPSHOT_ARG" ]; then
    case "$SNAPSHOT_ARG" in
        /*|[A-Za-z]:*) SNAPSHOT_DIR="$SNAPSHOT_ARG" ;;
        *)             SNAPSHOT_DIR="snapshots/$SNAPSHOT_ARG"; mkdir -p snapshots ;;
    esac
fi
LOG_DIR="/tmp/sweep_logs"
mkdir -p "$LOG_DIR"; rm -f "$LOG_DIR"/*.log
RUNNERS="
scripts/01_run_baseline.py
scripts/01b_run_baseline_sliced.py
scripts/02_run_eaas.py
scripts/02_run_ndc_caps.py
scripts/03_run_ndc_eaas.py
scripts/05_run_fin3_capital_sweep.py
scripts/06_run_gas1_shadow_benchmarks.py
scripts/07_run_gas2_eaas_gas_relief.py
scripts/08_run_gas3_regime_ndc_feasibility.py
scripts/09_run_rel1_feasibility.py
scripts/09b_run_rel1_mode_sensitivity.py
scripts/09b_run_rel1_retirement_sensitivity.py
scripts/10_run_rel2_marginal_cost.py
scripts/11_run_rel3_financing_frontier.py
scripts/13_run_dem2_growth_gas_interaction.py
scripts/14_run_str1_storage_role.py
scripts/15_run_pol1_ndc_comparison.py
scripts/18_run_pareto_frontier.py
scripts/19_run_carbon_price_sweep.py
scripts/20_run_cf1_solar_yield_sensitivity.py
scripts/99c_run_voll_sensitivity.py
"
MISSING=""
for f in $RUNNERS; do [ -f "$f" ] || MISSING="$MISSING $f"; done
if [ -n "$MISSING" ]; then echo "ABORT -- not found:$MISSING"; exit 1; fi
FAILED=""; N=0
for f in $RUNNERS; do
  name=$(basename "$f" .py); N=$((N+1))
  python "$f" > "$LOG_DIR/$name.log" 2>&1
  code=$?; [ $code -ne 0 ] && FAILED="$FAILED $name"
  printf "%-44s exit=%d\n" "$name" "$code"
done
echo; echo "ran $N runners"; echo "FAILED:${FAILED:- none}"
echo; echo "--- tracebacks ---"; grep -il "traceback" "$LOG_DIR"/*.log || echo "  none"
echo; echo "--- plan 2.5 guard ---"
grep -h "plan 2.5" "$LOG_DIR"/*.log 2>/dev/null | sed 's/^ *//' | sort | uniq -c || echo "  none"
if [ -n "$SNAPSHOT_DIR" ]; then
  if [ -n "$FAILED" ]; then echo; echo "REFUSING snapshot: runners failed."; exit 1; fi
  rm -rf "$SNAPSHOT_DIR"; cp -r results "$SNAPSHOT_DIR"; echo; echo "snapshot -> $SNAPSHOT_DIR"
fi
[ -n "$FAILED" ] && exit 1; exit 0
