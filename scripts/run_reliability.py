import matplotlib.pyplot as plt
import sys

from src.optimize_experiments import run_bottleneck_sensitivity
from pathlib import Path
from src.scenarios import load_scenario
from src.optimize_experiments import (
    run_reliability_sweep,
    reliability_results_to_df
)
from src.io import load_econ


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
# ------------------------------------------------------------
# PLOTTING FUNCTIONS
# ------------------------------------------------------------

def plot_reliability_frontier(df):

    df2 = df[
        (df["status"] == "feasible") &
        (df["eps"].notnull())
    ].copy()
    df2 = df2.sort_values("reliability")

    print("\n=== FEASIBILITY SUMMARY ===")
    print(df[["eps", "reliability", "status"]])
    print("\n=== FULL RESULTS ===")
    print(df)
    plt.figure()
    plt.semilogx(1 - df2["reliability"], df2["npv_cost"])

    plt.xlabel("Unserved Fraction (log scale)")
    plt.ylabel("System Cost (USD NPV)")
    plt.title("Reliability–Cost Frontier")

    plt.grid()
    plt.show()


def plot_marginal_cost(df):

    df2 = df[
        (df["dual_reliability"].notnull()) &
        (df["status"] == "feasible") &
        (df["eps"].notnull())
    ].copy()
    df2 = df2.sort_values("reliability")

    plt.figure()
    plt.plot(df2["reliability"], df2["dual_reliability"])

    plt.xlabel("Reliability (fraction served)")
    plt.ylabel("Marginal Cost (USD per TWh)")
    plt.title("Marginal Cost of Reliability")

    plt.grid()
    plt.show()


# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------

def main():

    scenario = load_scenario(
        demand_level_case="served",
        demand_case="baseline",
        gas_deliverability_case="baseline",
        solar_build_case="aggressive",
        carbon_case="no_policy",
        start_year=2025,
        end_year=2045,
    )

    econ = load_econ()

    # ---- RUN SWEEP
    results = run_reliability_sweep(scenario, econ)

    # ---- CONVERT TO DF
    df = reliability_results_to_df(results)

    gas_cases = ["baseline", "upside", "downside", "shock_recovery"]

    all_results = {}

    for g in gas_cases:

        print(f"\n=== GAS SCENARIO: {g} ===")

        scenario_mod = load_scenario(
            demand_level_case="served",
            demand_case="baseline",
            gas_deliverability_case=g,
            solar_build_case="aggressive",
            carbon_case="no_policy",
            start_year=2025,
            end_year=2045,
        )

        baseline_results = run_reliability_sweep(scenario_mod, econ)
        baseline_df = reliability_results_to_df(baseline_results)

        all_results[g] = df
        baseline_df.to_csv(f"results/reliability_frontier_{g}.csv", index=False)
        print(df[["eps", "reliability", "status"]])
        print(df)

    # ---- SAVE CSV (important for thesis)
    df.to_csv("results/reliability_frontier.csv", index=False)

    # ---- PLOTS
    plot_reliability_frontier(df)
    plot_marginal_cost(df)

    sens = run_bottleneck_sensitivity(scenario, econ)
    print("\n=== BOTTLENECK SENSITIVITY ===")
    print(sens)

if __name__ == "__main__":
    main()
