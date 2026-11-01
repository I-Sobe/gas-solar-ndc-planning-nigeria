"""
I/O Utilities

Low-level helpers for loading configuration files and
persisting numerical model outputs.

This module does not perform plotting or reporting.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def load_yaml(filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    filepath : str
        Path to YAML file

    Returns
    -------
    dict
        Parsed YAML contents
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"YAML file not found: {filepath}")

    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def save_results(output, filepath):
    """
    Save numerical model results to disk.

    Parameters
    ----------
    output : dict or array-like
        Model results to persist
    filepath : str
        Output file path ('.npz' or '.npy')

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if isinstance(output, dict):
        np.savez(filepath, **output)
    else:
        np.save(filepath, output)


def load_solar_capex_by_year(scenario_name="solar_low", start_year=2025, end_year=2045):
    """
    Returns a dict {year: capex_usd_per_mw} from the NREL solar CAPEX CSV.
    Handles UTF-8 BOM, dollar signs, commas, and trailing spaces.

    SOLAR CAPEX BASES (scenario_name; data/cost/processed/solar_capex.csv,
    2025-2050, one row per year per scenario):
      solar_low / solar_mid / solar_high -- a US COST BASIS: NREL ATB 2024,
        Utility-Scale PV, Class 5, Market financial case (low = ATB
        Advanced, mid = Moderate, high = Conservative). Every runner
        defaults to solar_low; this docstring changes no default.
      solar_irena_africa -- an AFRICAN cost basis: IRENA Renewable Power
        Generation Costs (2025), Africa utility-scale PV total installed
        cost USD 837/kW, with ATB solar_low's year-on-year decline shape
        re-anchored to it: 837000 * solar_low[t] / solar_low[2025],
        scale = 837000 / 1456000 = 0.5748626. Sourcing:
        docs/sourcing_pack_02_ledger.md (secondary citation, not verified
        against a primary document; see that file's header).

    DOLLAR YEAR. ATB figures are real 2022 USD (raw workbook
    data/cost/raw/solar_capex/2024_v3_Workbook.xlsx, sheet "Solar -
    Utility PV", Base Year 2022). The IRENA anchor is in IRENA's own
    real-dollar year. The 1.74x ratio (ATB solar_low 2025 = USD 1,456/kW
    against IRENA Africa USD 837/kW) mixes dollar years and understates
    the like-for-like gap. Because ATB is in 2022 USD and IRENA in a later
    dollar year, 1.74x is a LOWER BOUND on the like-for-like basis gap;
    re-expressed in common dollars the ATB figure would be higher still.
    The deflator is an OPEN ITEM (none is sourced in the repository). The
    construction above is a ratio, so it cancels ATB's dollar year from
    the solar_irena_africa SHAPE; the arm's LEVEL carries IRENA's dollar
    year. For scale: ATB solar_low 2025 is also ~2.1x IRENA's global 2024
    figure of USD 691/kW (pack 02; unverified). Neither basis is a
    Nigeria figure; see register entry L47.
    """
    solar_df = pd.read_csv(
        ROOT / "data/cost/processed/solar_capex.csv", 
        thousands=",",
        encoding="utf-8-sig" # strips BOM automatically
        )
    df = solar_df[solar_df["Scenario"] == scenario_name].copy()
    df = df[df["Year"] >= start_year].sort_values("Year")

    # Gaps INSIDE the source coverage are still an error. Gaps PAST it are the
    # end-effect buffer and are filled by _extend_capex_series.
    found_years = set(int(row["Year"]) for _, row in df.iterrows())
    if not found_years:
        raise ValueError(f"solar_capex.csv has no rows for scenario '{scenario_name}'.")
    last_sourced = max(found_years)
    missing_inside = set(range(start_year, min(end_year, last_sourced) + 1)) - found_years
    if missing_inside:
        raise ValueError(
            f"solar_capex.csv missing years INSIDE its coverage for scenario "
            f"'{scenario_name}': {sorted(missing_inside)}. Check your NREL ATB data."
        )

    result = {}
    for _, row in df.iterrows():
        raw = str(row["Solar_capex_usd_per_mw"]).replace("$", "").replace(",", "").strip()
        result[int(row["Year"])] = float(raw)
    result = {y: v for y, v in result.items() if y <= end_year}
    return _extend_capex_series(result, end_year)


def _extend_capex_series(result, end_year):
    """
    Extend a CAPEX series past its source coverage into the end-effect buffer.

    NREL ATB publishes to 2050; the model runs to scenarios.MODEL_END_YEAR
    (2055) so terminal-year salvage effects fall outside the reporting window.

    Buffer years are HELD FLAT at the final sourced value. Flat is the
    conservative choice: continuing the ATB decline would make buffer-year
    solar progressively cheaper and could pull build forward into the
    reporting window -- precisely the distortion the buffer exists to prevent.
    Buffer-year CAPEX is never reported.
    """
    if not result:
        return result
    last_sourced = max(result)
    if last_sourced >= end_year:
        return result
    terminal = result[last_sourced]
    for y in range(last_sourced + 1, end_year + 1):
        result[y] = terminal
    return result


def load_storage_capex_by_year(scenario_name="Storage_low", start_year=2025, end_year=2045):
    """
    Returns a dict {year: capex_usd_per_mwh} from the NREL storage CAPEX CSV.
    Handles commas and trailing spaces.
    """
    df = pd.read_csv(
        ROOT / "data/cost/processed/storage_capex.csv",
        thousands=",",
    )
    df = df[df["Scenario"] == scenario_name].copy()
    df = df[df["Year"] >= start_year].sort_values("Year")

    # Gaps INSIDE the source coverage are still an error. Gaps PAST it are the
    # end-effect buffer and are filled by _extend_capex_series.
    found_years = set(int(row["Year"]) for _, row in df.iterrows())
    if not found_years:
        raise ValueError(f"storage_capex.csv has no rows for scenario '{scenario_name}'.")
    last_sourced = max(found_years)
    missing_inside = set(range(start_year, min(end_year, last_sourced) + 1)) - found_years
    if missing_inside:
        raise ValueError(
            f"storage_capex.csv missing years INSIDE its coverage for scenario "
            f"'{scenario_name}': {sorted(missing_inside)}."
        )

    result = {}
    for _, row in df.iterrows():
        raw = str(row["Storage_capex_usd_per_mwh"]).replace("$", "").replace(",", "").strip()
        result[int(row["Year"])] = float(raw)
    result = {y: v for y, v in result.items() if y <= end_year}
    return _extend_capex_series(result, end_year)


_OM_TECHNOLOGIES = ("solar_utility", "storage", "gas_ct", "gas_cc")


def load_om_cost_by_year(om_case, start_year=2025, end_year=2050):
    """
    Returns {technology: {year: {'fom_usd_per_kw_yr': float, 'vom_usd_per_mwh': float}}}
    from data/cost/processed/om_cost.csv (NREL ATB), for
    om_case in {'om_none', 'om_low', 'om_central', 'om_high'} -- see
    scenarios.om_scenarios(). Technologies: solar_utility, storage,
    gas_ct, gas_cc (task_om_implementation, Part 3.1).

    'om_none' is SYNTHETIC -- it is not a row in the CSV. It returns
    every technology's rates as 0.0 for every year: the REGRESSION arm.
    With it selected, both builders' O&M term evaluates to exactly 0.0
    and the pre-O&M anchors must reproduce to the digit.

    Buffer years past the CSV's 2050 coverage are held flat at the 2050
    value, via _extend_capex_series -- the same helper and convention
    load_solar_capex_by_year / load_storage_capex_by_year use. Do not
    write a second extension convention.
    """
    if om_case == "om_none":
        zero = {"fom_usd_per_kw_yr": 0.0, "vom_usd_per_mwh": 0.0}
        return {
            tech: _extend_capex_series(
                {y: zero for y in range(start_year, 2051)}, end_year
            )
            for tech in _OM_TECHNOLOGIES
        }

    om_df = pd.read_csv(ROOT / "data/cost/processed/om_cost.csv")
    case_df = om_df[om_df["om_case"] == om_case]
    if case_df.empty:
        raise ValueError(f"om_cost.csv has no rows for om_case '{om_case}'.")

    result = {}
    for tech in _OM_TECHNOLOGIES:
        tech_df = case_df[case_df["technology"] == tech].copy()
        tech_df = tech_df[tech_df["year"] >= start_year].sort_values("year")

        found_years = set(int(y) for y in tech_df["year"])
        if not found_years:
            raise ValueError(
                f"om_cost.csv has no rows for technology '{tech}', "
                f"om_case '{om_case}'."
            )
        last_sourced = max(found_years)
        missing_inside = set(range(start_year, min(end_year, last_sourced) + 1)) - found_years
        if missing_inside:
            raise ValueError(
                f"om_cost.csv missing years INSIDE its coverage for "
                f"technology '{tech}', om_case '{om_case}': "
                f"{sorted(missing_inside)}."
            )

        by_year = {}
        for _, row in tech_df.iterrows():
            by_year[int(row["year"])] = {
                "fom_usd_per_kw_yr": float(row["fom_usd_per_kw_yr"]),
                "vom_usd_per_mwh": float(row["vom_usd_per_mwh"]),
            }
        by_year = {y: v for y, v in by_year.items() if y <= end_year}
        result[tech] = _extend_capex_series(by_year, end_year)

    return result


def compute_om_cost_npv(
    T,
    years,
    DF,
    om_cost_by_year,
    storage_duration_hours,
    solar_capacity_mw,
    storage_capacity_mwh,
    gas_unconverted_capacity_mw,
    gas_converted_capacity_mw,
    gas_unconverted_generation_twh,
    gas_converted_generation_twh,
    hydro_capacity_mw=None,
    hydro_fom_usd_per_kw_yr=0.0,
):
    """
    Shared fixed + variable O&M NPV term for BOTH LP builders
    (task_om_implementation, Part 3.3; plan 2.5: "the two must share a
    cost definition or the annual-vs-sliced premium compares different
    things"). Discounted by the same DF[t] every other operating/capital
    term in the objective uses.

    Units and conversions
    ----------------------
    - FOM (fom_usd_per_kw_yr) is USD/kW-yr on POWER capacity. Model
      capacities are in MW, so FOM x 1000 converts kW-yr to MW-yr.
    - Storage is the exception: the model tracks storage in MWh of
      ENERGY; ATB's storage FOM is per kW of POWER. Converted here as
          USD/MWh-yr = fom_usd_per_kw_yr x 1000 / storage_duration_hours
      which makes storage O&M DEPEND on storage_duration_hours --
      register entry L21 declares that value ASSUMED and unswept.
      storage_duration_hours is always read from the caller (the
      builder's own scenario-derived local), never hardcoded here.
    - VOM (vom_usd_per_mwh) is on GENERATION (MWh electrical). The gas
      generation streams passed in are TWh_e; x 1e6 converts to MWh.
    - Solar and storage carry zero VOM in every ATB case sourced for
      this model (om_cost.csv) -- no VOM term is built for them.
    - HYDRO (TASK 69 B.1, register L95): `hydro_capacity_mw` is None by
      default and then NO hydro term exists, exactly as before (hydro carries
      capacity and no O&M cost). When a caller passes it, a fixed O&M term
      DF[t] x hydro_capacity_mw[t] x rate x 1000 is added, with `rate` the
      hydro FOM in USD/kW-yr, either a float or a {calendar year: float}
      dict. The null value is 0.0, which adds a zero term. No non-zero rate
      is set anywhere; the candidate source (NREL ATB hydropower FOM) has no
      row in om_cost.csv or in docs/parameter_provenance.md yet.
    - gas_converted_* covers OCGT->CCGT retrofit (Phase 2.8), modelled
      only in the sliced builder. Callers that never build converted
      capacity (the annual builder; the sliced builder under conv_none)
      pass zero for both converted arguments, for every t.

    Parameters are indexable by t in T (dict or Pyomo Expression/Param/Var);
    `years` maps t -> calendar year for the rate lookup.

    Returns an unwrapped expression -- callers wrap it in their own
    pyo.Expression (kept out of this module so io.py stays pyomo-free).
    """
    solar_om = om_cost_by_year["solar_utility"]
    storage_om = om_cost_by_year["storage"]
    gas_ct_om = om_cost_by_year["gas_ct"]
    gas_cc_om = om_cost_by_year["gas_cc"]

    fom_solar = sum(
        DF[t] * solar_capacity_mw[t]
        * solar_om[int(years[t])]["fom_usd_per_kw_yr"] * 1000.0
        for t in T
    )
    fom_storage = sum(
        DF[t] * storage_capacity_mwh[t]
        * (storage_om[int(years[t])]["fom_usd_per_kw_yr"] * 1000.0
           / storage_duration_hours)
        for t in T
    )
    fom_gas = sum(
        DF[t] * (
            gas_unconverted_capacity_mw[t]
            * gas_ct_om[int(years[t])]["fom_usd_per_kw_yr"] * 1000.0
            + gas_converted_capacity_mw[t]
            * gas_cc_om[int(years[t])]["fom_usd_per_kw_yr"] * 1000.0
        )
        for t in T
    )
    vom_gas = sum(
        DF[t] * (
            gas_unconverted_generation_twh[t] * 1e6
            * gas_ct_om[int(years[t])]["vom_usd_per_mwh"]
            + gas_converted_generation_twh[t] * 1e6
            * gas_cc_om[int(years[t])]["vom_usd_per_mwh"]
        )
        for t in T
    )
    base = fom_solar + fom_storage + fom_gas + vom_gas
    if hydro_capacity_mw is None:
        return base
    if isinstance(hydro_fom_usd_per_kw_yr, dict):
        def _hydro_rate(t):
            return float(hydro_fom_usd_per_kw_yr[int(years[t])])
    else:
        def _hydro_rate(t):
            return float(hydro_fom_usd_per_kw_yr)
    if any(_hydro_rate(t) < 0.0 for t in T):
        raise ValueError("hydro_fom_usd_per_kw_yr must be >= 0")
    fom_hydro = sum(
        DF[t] * hydro_capacity_mw[t] * _hydro_rate(t) * 1000.0
        for t in T
    )
    return base + fom_hydro


_GENSET_REQUIRED = object()  # sentinel: distinguishes "not passed" from "None"


def load_econ(voll_case="voll_low", gas_price_case="gas_low", genset_case=_GENSET_REQUIRED):
    """
    Parameters
    ----------
    voll_case : str
        One of 'voll_low', 'voll_mid', 'voll_high'.
    gas_price_case : str
        One of 'gas_low', 'gas_mid', 'gas_high'.
        Controls the fuel commodity + transport cost assumption.
        NOTE: gas deliverability volume is controlled separately via
        gas_deliverability_case in load_scenario(). These are independent.
        gas_low (no transport tariff): 7.27 M USD/TWh_th
        gas_mid (domestic tariff):    11.01 M USD/TWh_th
        gas_high (export-parity):     14.87 M USD/TWh_th
    
    genset_case : str or None -- REQUIRED, no default (correction plan 2.6)
        One of 'genset_srmc', 'genset_central', 'genset_high', or an explicit
        None to run without the self-generation backstop.

        There is deliberately NO DEFAULT. Omitting it raises.

        Why: with genset_case=None as a silent default, forgetting the argument
        removes the backstop and the run reverts to pre-2.6 behaviour without
        any error, warning, or visible signature in the output. That happened
        twice -- in 02_run_ndc_caps.py and 03_run_ndc_eaas.py, the two headline
        NDC runners -- and both produced clean-looking tables that were wrong.
        The tell was only found by noticing that total emissions equalled
        emissions_grid_only exactly.

        Same reasoning as the gas_eta guard in optimize_model.py: a silent
        revert to the old value is more dangerous than a loud failure.

        Pass None EXPLICITLY where the backstop is genuinely not wanted --
        the regression harness (verify_2_6_commit_a.sh) and the 'voll' arm of
        23_run_selfgen_frontier.py both rely on it.
    """
    if genset_case is _GENSET_REQUIRED:
        raise TypeError(
            "load_econ() requires genset_case (correction plan 2.6). Pass one "
            "of 'genset_srmc' / 'genset_central' / 'genset_high', or an "
            "explicit None to run WITHOUT the self-generation backstop.\n"
            "There is no default: omitting it silently reverted runs to "
            "pre-2.6 behaviour with no error and no visible signature in the "
            "output."
        )

    def to_float(x):
        return float(str(x).replace(",", "").replace("$","").strip())

    econ = {}

    gas_df = pd.read_csv(ROOT/"data/cost/processed/gas_cost.csv", thousands=",")
    gas_row = gas_df[gas_df["Scenario"] == gas_price_case].iloc[0]
    econ["GAS_COST_PER_TWH_TH"] = to_float(gas_row["total_usd_per_twh_th"])

    solar_df = pd.read_csv(
        ROOT/"data/cost/processed/solar_capex.csv", 
        thousands=",", 
        encoding="utf-8-sig")
    solar_row = solar_df[
        (solar_df["Scenario"]=="solar_low") &
        (solar_df["Year"]==2025)
    ].iloc[0]

    # SOLAR_CAPEX_PER_MW: 2025 anchor value (solar_low NREL ATB).
    # Used ONLY as fallback when solar_capex_by_year is not passed to build_model().
    # All runner scripts must pass solar_capex_by_year. Do not use this scalar
    # in any post-solve cost diagnostic when time-varying CAPEX is active.
    econ["SOLAR_CAPEX_PER_MW"] = to_float(solar_row["Solar_capex_usd_per_mw"])

    storage_df = pd.read_csv(ROOT/"data/cost/processed/storage_capex.csv", thousands=",")
    storage_row = storage_df[
        (storage_df["Scenario"]=="Storage_low") &
        (storage_df["Year"]==2025)
    ].iloc[0]

    econ["STORAGE_COST_PER_MWH"] = to_float(storage_row["Storage_capex_usd_per_mwh"])

    # STORAGE_OM_PER_MWH_YR (2,000 USD/MWh-yr, a numerical-conditioning
    # constant with no cost provenance) is RETIRED here (task_om_implementation,
    # Part 3.1). It never entered either builder's objective -- see
    # docs/task_storage_om.md and the comment in
    # scripts/14_run_str1_storage_role.py. Real, sourced O&M (NREL ATB,
    # data/cost/processed/om_cost.csv) is now wired via the REQUIRED
    # om_case arm -- see scenarios.om_scenarios() / io.load_om_cost_by_year().

    voll_df = pd.read_csv(ROOT/"data/cost/processed/unserved_energy_penalty.csv", thousands=",")
    voll_row = voll_df[
        (voll_df["scenario"]==voll_case) &
        (voll_df["year"]==2025)
    ].iloc[0]

    econ["UNSERVED_ENERGY_PENALTY"] = to_float(voll_row["voll_usd_per_twh"])

    # Thermal-basis emission factor (tCO2 per MWh_th), HHV basis to match the
    # HHV calorific value used in the gas-to-power volume->energy conversion.
    # IPCC default 56.1 kgCO2/GJ (NCV) = 0.2020 tCO2/MWh_th (LHV), converted to
    # HHV basis (/1.108) = 0.1823. EF per MWh_e is DERIVED in build_model as
    # EF_th / gas_eta, so efficiency and emissions can never drift apart.
    # [SOURCE: IPCC default, HHV-adjusted. I'd Replace with Nigeria NIR factor if found.]
    econ["EF_TCO2_PER_MWH_TH"] = 0.1823
    # ------------------------------------------------------------
    # Private self-generation backstop (correction plan 2.6)
    # ------------------------------------------------------------
    #  genset_case has NO DEFAULT -- see the docstring. When an EXPLICIT None is
    # passed, build_model() falls back to pricing selfgen at VoLL with a zero
    # emission factor, which makes selfgen an exact duplicate of unserved and
    # leaves the objective unchanged. That fallback IS the commit-A regression
    # test -- do not remove it.
    #
    # As with the gas fleet (correction plan 1.2), the ELECTRICAL emission
    # factor is never stored. Only EF_th (HHV basis) and eta are loaded here;
    # EF_e = EF_th / eta is derived in build_model so that sweeping eta moves
    # cost and emissions together and they can never drift apart.
    if genset_case is not None:
        g_df = pd.read_csv(
            ROOT / "data/cost/processed/genset_cost.csv", thousands=","
        )
        g_rows = g_df[g_df["Scenario"] == genset_case]
        if g_rows.empty:
            raise ValueError(
                f"genset_cost.csv has no row for scenario '{genset_case}'. "
                f"Available: {sorted(g_df['Scenario'].unique())}"
            )
        g_row = g_rows.iloc[0]
        econ["GENSET_COST_PER_TWH_E"] = to_float(g_row["total_usd_per_twh_e"])
        econ["GENSET_ETA_HHV"] = to_float(g_row["genset_eta_hhv"])
        econ["GENSET_EF_TH_HHV"] = to_float(g_row["ef_th_hhv_tco2_per_mwh_th"])

    # ALWAYS set, even when genset_case is explicitly None (2.6, F0 Step 2,
    # phase_2_4_inversion). Previously this key was set only inside the
    # branch above, so "key absent" and "genset_case=None on purpose" were
    # indistinguishable from econ alone -- an econ dict built by hand,
    # bypassing load_econ() entirely, looked identical to a deliberate
    # opt-out. build_model()/build_model_sliced() now guard on "GENSET_CASE"
    # in econ (key presence, not truthiness) precisely because None here is
    # a legitimate, considered value (the commit-A regression test), not a
    # missing one -- only the key's absence indicates load_econ() was
    # bypassed.
    econ["GENSET_CASE"] = genset_case

    return econ

