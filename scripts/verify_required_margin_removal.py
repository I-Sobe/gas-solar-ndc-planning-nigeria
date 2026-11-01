"""
TASK 29 regression harness (copy of the scratchpad tool; see
docs/task_log_29_1_required_margin_change.md). NOT a runner.

Runs a MINIMAL public+EaaS subset of one affected runner from a given source
tree, with constants monkeypatched, so a pre-change tree (e.g. `git worktree
add <dir> <commit-before>`) and the current tree can be compared cell by
cell. Each tree writes only to ITS OWN results/. Needs SCRATCH set to a
directory containing a 2-draw copy of results/stochastic/draws__<hash>.csv
named draws2.csv (for the MC subset), and <tree>/data symlinked to data/.

    SCRATCH=... python scripts/verify_required_margin_removal.py <tree> {rel3|dem2|mc|pareto|carbon|genset}
"""
import sys, os, importlib.util
tree, runner = sys.argv[1], sys.argv[2]
os.chdir(tree); sys.path.insert(0, tree)
STD = ["--peak-case","peak_unsuppressed","--reserve-margin","nirp_spinning","--storage-cc-case","storage_cc_central",
 "--gas-build-case","gas_unconstrained","--new-gas-deliverability","constrained","--gas-availability-case","gas_avail_nameplate",
 "--gas-retirement-case","gas_retire_vintage_30","--gas-capex-case","gas_capex_central","--ccgt-efficiency-case","ccgt_eff_central",
 "--capital-recovery-case","cr_central","--dscr-case","dscr_central","--us-inflation","0.020","--capital-treatment-case","f1_tranche",
 "--concessional-envelope-case","env_dares","--conversion-case","conv_central","--hydro-case","hydro_flat","--om-case","om_central"]
def flags(drop=(), add=()):
    out=[];i=0
    while i<len(STD):
        if STD[i] in drop: i+=2; continue
        out+=STD[i:i+2]; i+=2
    return out+list(add)
files={"rel3":"11_run_rel3_financing_frontier.py","dem2":"13_run_dem2_growth_gas_interaction.py","mc":"17_run_monte_carlo.py",
       "pareto":"18_run_pareto_frontier.py","carbon":"19_run_carbon_price_sweep.py","genset":"99c_run_genset_arm_sensitivity.py"}
path=os.path.join(tree,"scripts",files[runner])
spec=importlib.util.spec_from_file_location("runner_"+runner,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
S=os.environ["SCRATCH"]
if runner=="rel3":
    m.GAS_CASES=["downside"]; m.EPS_LEVELS=[0.25,0.20,0.15,0.10]
    m.POLICY_CONFIGS=[c for c in m.POLICY_CONFIGS if c["policy_label"]=="ndc3_unconditional"]
    argv=flags(drop=("--new-gas-deliverability","--tariff-case"))
elif runner=="dem2":
    m.GAS_CASES=["downside"]; m.DEMAND_CASES=[m.MID_CASE]
    m.POLICY_CONFIGS=[c for c in m.POLICY_CONFIGS if c["policy_label"]=="ndc3_unconditional"]
    argv=flags(drop=("--new-gas-deliverability","--tariff-case"))
elif runner=="mc":
    m.CASES=[c for c in m.CASES if c["label"] in ("NDC3 uncond (public)","NDC3 uncond (EaaS)","NDC3 cond (EaaS)","Matched to 30_ central (no NDC)")]
    argv=flags(add=("--tariff-case","tariff_central","--draws-file",S+"/draws2.csv"))
elif runner=="pareto":
    m.N_POINTS=2; m.TIGHT_END_EXTENSION=[0.08]
    m.FINANCING_ARMS=[a for a in m.FINANCING_ARMS if a["label"] in ("public_only_tight","eaas_tight")]
    argv=flags(add=("--gas-deliverability-case","baseline","--tariff-case","tariff_central","--baseline-diagnostics","/home/zeus/thesis-code/results/baseline_sliced/diagnostics__02c17f55.json"))
elif runner=="carbon":
    m.CARBON_PRICES=[0,50]
    m.FINANCING_ARMS=[a for a in m.FINANCING_ARMS if a["label"] in ("public_tight","eaas_tight")]
    argv=flags(add=("--gas-deliverability-case","baseline","--tariff-case","tariff_central"))
elif runner=="genset":
    m.EPS_LEVELS=[0.10]
    argv=flags(drop=("--reserve-margin",),add=("--tariff-case","tariff_central"))
sys.argv=[path]+argv
try:
    m.main()
except SystemExit as e:
    print("SystemExit",e)
