"""
Scenario Definitions Module (Configuration Only)

Scope
-----
Defines deterministic planning scenarios as structured parameter dictionaries
for use in system evaluation and optimization studies.

This module provides scenario DEFINITIONS ONLY.
It does NOT execute models, run experiments, or aggregate results.

Demand framing (examiner-safe)
------------------------------
- Demand is treated as an EXOGENOUS planning input, not statistically forecast
  from historical time series.
- The base-year demand level is Tier 1: observed gross generation at busbar.
  Reconstructed latent-demand bases (latent_low/latent_high) are DEPRECATED
  and raise in load_scenario; they are superseded by the tiered demand
  architecture (Tier 2 self-generation, Tier 3 access-adjusted), Phase 2.
- Growth-rate scenarios remain separate from demand-level scenarios.

Base-year demand anchor (2024) — GROSS GENERATION at busbar
------------------------------------------------------------
Demand is metered at gross generation sent out. This is the natural boundary
for a generation capacity-expansion model: capacity serves generation, and the
NDC emissions caps account against actual grid output.

  [SOURCE: NERC 2024 — gross generation 37,093.70 GWh; hydro share 30.92%]
    grid gross generation   37.09 TWh   <- Tier 1, the model's demand base
      of which gas          25.62 TWh
      of which hydro        11.47 TWh

SELF-GENERATION IS EXCLUDED FROM THIS DEMAND BASE BY CONSTRUCTION. Tier 1
(37.09 TWh grid generation) does not include electricity produced by
captive/backup generators; in the model self-generation is a costed
backstop the LP chooses (m.selfgen), not a demand input (Tier 2 tiered
demand architecture, plan 2.7 `[OPEN]`, docs/merged_correction_plan_v2.md;
NIRP 2024 self-generation audit, Tables 2 and 15, Section 4.2, lambda =
0.51-0.57, Tier 2 ~65-73 TWh for 2024). Observed Nigerian self-generation
is ~18 TWh/yr from ~9 GW (IEA 2019, for 2018) and backup generators cover
~40% of Nigeria's electricity needs (IEA, Jan 2024) -- docs/
sourcing_pack_02_ledger.md, section 2 (secondary citations, unverified).
So the model represents roughly 60% of national consumption (the IEA ~40%
figure) to 67% (37.09 against 37.09 + 18 TWh); the two IEA figures are
from different years and are not mutually checkable without a stated
base. At the central configuration the model's own self-generation output
is zero, which is therefore not comparable with the observed ~18 TWh/yr.

Loss ladder (NERC 2024) — for reference, NOT applied in the energy balance:
    gross generation        37.09 TWh
      -7% TLF            -> 34.50 TWh   delivered to DisCos
      -36.4% ATC&C       -> 21.97 TWh   collected (billed and paid)

The previous base (23.08 TWh — NBS Q1 2024 served energy of 5.770 TWh,
annualised as Q1 x 4) sat at the bottom of that ladder: it was COLLECTED
energy, excluding electricity consumed but stolen, unmetered, or unpaid — all
of which is real demand of real users. The Q1 x 4 annualisation is separately
unsound, ignoring seasonality, and is not used anywhere in the current model.

UNITS CONVENTION (read before editing any value)
------------------------------------------------
- Demand, hydro, solar/gas generation entering the electricity balance are in
  TWh_e (electrical energy) per year.
- Gas *deliverability* (gas_scenario data) is in TWh_th (thermal); it is
  converted to electrical via gas_eta inside build_model. Hydro is ALREADY
  electrical and must NOT be multiplied by gas_eta.
"""

from __future__ import annotations

import csv
import os
import warnings

import numpy as np

# 2.6, F0 Step 2.4: PAR and the peak-slice demand share are the SAME
# assumption. Importing the model's own slice-hours/weighting functions (and
# the hydro seasonal-share helper, reused for capacity_credit['hydro'] in
# Step 2.3) means the consistency check below tests the LITERAL computation
# build_model_sliced() performs, not a shadow copy that could drift out of
# sync with it.
from src.demand import project_baseline_demand
from src.optimize_model_sliced import (
    SEASON_DAYS,
    DEFAULT_SEASON_DEMAND_FACTOR,
    slice_hours,
    _normalised_demand_weights,
    _hydro_season_energy_share,
)
from src.io import load_om_cost_by_year, load_storage_capex_by_year


# ============================================================
# BLENDED CONCESSIONAL FINANCE (single source of truth)
# ============================================================

def blended_finance_scenarios():
    """
    Blended concessional finance capital-structure scenarios for EaaS solar.

    The blended WACC fed to the bankability test is DERIVED from these tuples,
    not asserted. 'blended_central' is the adopted working case.

    Tuple order: (concessional_share, r_concessional, r_commercial)

    r_concessional is SOURCED; r_commercial and concessional_share remain
    PLACEHOLDERS pending primary-source verification:
      - r_concessional : SOURCED (2026-09-20 sourcing pack). IDA21 Blend
                         (Fixed Rate) USD interest charge, 2.12% effective
                         2026-01-01 (2.13% effective 2026-07-01; band
                         0.0212-0.0213); service charge set to zero on the
                         blend term; 25-yr maturity, 5-yr grace. Nigeria is
                         an IDA blend country. Floating alternative: IBRD
                         Flexible Loan Group A minus 250 bps. Replaces the
                         placeholder 0.06. SCOPE: this is the SOVEREIGN
                         on-lending rate; it is the right rate here because
                         the model's concessional envelope (env_dares,
                         $750M) is the World Bank DARES IDA credit, i.e.
                         sovereign-backed. Project-level concessional
                         finance (GCF/AfDB to an IPP) would carry a higher
                         rate. Figures as supplied by Zeus; not
                         independently checked by Code; currency to be
                         confirmed. Source: docs/sourcing_pack_01_rates.md,
                         section 1 (secondary citation; see that file's
                         header). DIRECTION OF BIAS of the old 0.06: it
                         OVERSTATED concessional cost ~3x. The sign of its
                         effect on the EaaS advantage is NOT established:
                         a lower blended WACC raises bankable revenue but
                         also shrinks the financing gap the subsidy may
                         cover (register L17: two opposing effects).
      - r_commercial   : STILL A PLACEHOLDER (0.18). Nigerian commercial
                         power-sector WACC. Anchor: IEA / CCSI 'Africa
                         power-sector WACC > 18% (2023)'. Unresolved: the
                         F1 capital-recovery arms carry a DIFFERENT
                         commercial rate (r_comm 0.11, capital_recovery_
                         scenarios()); see task log row for the basis
                         question. Not edited.
                         EVIDENCE (docs/sourcing_pack_01_rates.md, section
                         2; secondary, unverified): Climate Policy
                         Initiative (CPI, 2023), Cost of Capital for
                         Renewable Energy Investments in Developing
                         Economies -- cost of DEBT for climate projects by
                         sovereign rating, CAPM adjusted for country
                         climate-investment risk: Jamaica (S&P B+) 18.5%,
                         Bangladesh (BB-) 19.0%. Nigeria rates at or below
                         B, so its cost of debt should be >= 18.5%; 0.18 is
                         therefore likely conservative-LOW, and understating
                         commercial cost flatters EaaS (opposite in
                         direction to the r_concessional correction; the two
                         do not cancel and are reported separately).
                         UNRESOLVED: a WACC cannot lie below its own cost of
                         debt, yet the F1 objective's commercial rate is
                         0.11 (cr_central, capital_recovery_scenarios()) while
                         this bankability blend uses 0.18. No value changed
                         here; awaiting author decision.
      - concessional_share : from an actual blend / mobilisation ratio.
                         SOURCE. Still a placeholder.

    NOTE: blended_central here is the SINGLE definition of the adopted capital
    structure. load_scenario() pulls its defaults from this entry so the whole
    codebase has one source of truth for the central blend. Do not hard-code
    these three numbers anywhere else.
    """
    return {
        # label:              (conc_share, r_conc, r_comm)   blended WACC
        "commercial_only":    (0.00, 0.0212, 0.18),   # 18.00% (unblended benchmark)
        "blended_central":    (0.30, 0.0212, 0.18),   # 13.24% (ADOPTED working case; was 14.4% at r_conc 0.06)
        "blended_optimistic": (0.50, 0.0212, 0.18),   # 10.06% (deeper concessional; was 12.0%)
        "concessional_heavy": (0.75, 0.0212, 0.18),   #  6.09% (near-intervention; was 9.0%)
    }


# ============================================================
# PEAK DEMAND SHAPE ARMS -- PAR IS DERIVED, NOT AN INPUT (2.6, F0 Step 2.4)
# ============================================================
# INVERTED from the original design (2.6, F0 phase_2_4_inversion). The
# original design supplied a scalar PAR per arm and CHECKED it against the
# 6-slice demand shape (season_demand_factor x period_demand_factor),
# raising on >1% disagreement. That was the wrong direction: h_peak (the
# combined dry_peak+wet_peak hours) is FIXED calendar arithmetic --
# 724 + 736 = 1460 h/yr, independent of any scenario input -- and
# 8760 / 1460 = 6 EXACTLY, so PAR = f_peak * 8760 / h_peak = 6 * f_peak
# IDENTICALLY. PAR is a restatement of the peak block's demand-energy share
# (f_peak), not an independent quantity that could genuinely disagree with
# it. Checking a derived quantity against the very thing it was derived
# from is structurally either redundant (numbers agree) or a bug (they
# don't) -- never informative. See docs/phase_2_4_inversion.md.
#
# Each arm below supplies the SHAPE itself:
#   period_demand_factor : REQUIRED per arm. None = SOURCE: [PENDING];
#                           selecting an arm before this is filled in raises.
#   season_demand_factor  : OPTIONAL per arm -- only set it if this arm
#                           varies season shape from the default. None here
#                           means "use DEFAULT_SEASON_DEMAND_FACTOR"
#                           (imported above from optimize_model_sliced.py),
#                           NOT a pending/blocking gap the way
#                           period_demand_factor's None is.
# load_scenario() computes PAR from whichever arm is selected (via
# _normalised_demand_weights()/slice_hours(), imported above -- so this is
# the model's own computation, not a shadow copy) and reports it as
# par_block on the built scenario. There is no consistency guard any more:
# with one source of truth, disagreement is unreachable by construction.
#
# par_block IS A 4-HOUR BLOCK-AVERAGE RATIO, NOT AN INSTANTANEOUS PEAK
# RATIO. The peak slice (18:00-22:00) is a 4-hour block; par_block is the
# ratio of THAT BLOCK'S AVERAGE power to the annual average -- not the
# instantaneous system peak that can occur at a single moment inside that
# 4-hour window, which the block average necessarily smooths over. See
# within_block_peaking_factor (2.6, F0 Step 2.3, read inside the builders --
# optimize_model.py / optimize_model_sliced.py), the SEPARATE, SOURCE:
# [PENDING] correction for that gap. Do not conflate the two: par_block is
# derived and reported; within_block_peaking_factor is a distinct,
# independently-sourced multiplier applied on top of it.
#
# SOURCE STATUS PER ARM (corrects earlier text that said "[PENDING] for all
# three arms", which no longer describes the file):
#   peak_suppressed   -- FILLED (served 2016 Oluwole series; a declared floor).
#   peak_unsuppressed -- FILLED (2016 Oluwole reconstruction; CENTRAL ARM).
#   peak_nirp         -- [PENDING]: None; raises the moment it is selected,
#                        by design (mirrors the genset_case discipline in
#                        load_econ() -- see src/io.py). NIRP gives a load
#                        factor but no daily profile.
#   peak_high         -- FILLED but CONSTRUCTED, not measured.
# A None arm is never to be shipped as anything but None.
#
# NERC/TCN SUPPRESSED ANCHOR (docs/sourcing_pack_02_ledger.md, section 3;
# secondary citation, not verified against a primary document): all-time
# peak 5,801.84 MW (4 Mar 2025) with same-day energy 128,370.75 MWh, i.e. a
# 5,348.8 MW average and a multiple of 1.0847. Grid load factor 92.26%
# (2026/Q1; 82.45% in 2025/Q4), 94% (Apr 2026), 90% (Jan 2026). Installed
# 13,625 MW; average available 4,286 MW (PAF 31%, Apr 2026); estimated
# demand exceeds 20,000 MW against ~4,000 MW delivered.
# WARNING -- THESE ARE RATIONING ARTEFACTS, NOT DEMAND OBSERVATIONS. Supply
# binds, so dispatch runs flat while DisCos shed load. They are recorded
# here as the SUPPRESSED anchor only and MUST NEVER ANCHOR AN UNSUPPRESSED
# ARM (same discipline as register L2). An unsuppressed multiple must come
# from load research or a peer-system analogue, never from Nigerian grid
# data. NOT a corroboration of the central arm: NERC's 1.0847 (single-day
# peak / that day's average, supply-constrained, 2025) lies within 0.3% of
# the peak_unsuppressed arm's par_block, which load_scenario() derives at
# HEAD as 1.0877 (4-hour block average / annual average, reconstructed
# unsuppressed, 2016 shape; 0.9834 suppressed, 1.2000 high). They are
# different ratios on different bases -- a numerical near-coincidence, not
# evidence. (Other comments in this file, register L2 and the physical-layer
# methodology text state the block ratio as 1.0847 / 0.9807; that is the
# stale figure recorded as C-4 in docs/methodology_claims_ledger.md.)

# SOURCE: Oluwole (2023), Nigeria National Electricity Demand Timeseries,
# DOI 10.17632/pxvdm26rn7.2 -- 8,784 hourly observations, calendar 2016.
# Period partition: night 22:00-05:00 (7h), day 05:00-18:00 (13h),
# peak 18:00-22:00 (4h). Season: dry Nov-Apr, wet May-Oct.
#
# VINTAGE: 2016 shape against a 2024 planning start. Only the SHAPE is
# taken; levels come from NERC 2024. See docs/limitations_register.md L1.

# SUPPRESSED (served) series. DECLARED FLOOR, NOT a candidate central case:
# this is a SUPPLY shape, not a demand shape. Rationing flattens the evening
# specifically -- peak factor falls 1.088 -> 0.983 while night RISES
# 0.906 -> 1.022. Suppression ratio 0.4192 (only 41.9% of demand served).
# 20.21 TWh, mean 2,300 MW, max 3,886 MW, LF 0.5920. See L2.
PEAK_SUPPRESSED_PERIOD_FACTOR = {"night": 1.0215, "day": 0.9935, "peak": 0.9834}
PEAK_SUPPRESSED_SEASON_FACTOR = {"dry": 1.0315, "wet": 0.9689}

# UNSUPPRESSED reconstruction. CENTRAL ARM.
# Derived: par_block 1.0847, par_inst 1.4588, LF 0.6855.
# 48.20 TWh, mean 5,487 MW, max 8,005 MW.
PEAK_UNSUPPRESSED_PERIOD_FACTOR = {"night": 0.9056, "day": 1.0239, "peak": 1.0877}
PEAK_UNSUPPRESSED_SEASON_FACTOR = {"dry": 1.0224, "wet": 0.9778}

# SOURCE: [PENDING]. NIRP 2024 stipulates a load factor of 0.76 (Table 14
# prints 71/76 and is internally inconsistent; the Table 15 pairs give
# 0.756 / 0.759 / 0.760, so 0.76 is the applied value), implying par_inst
# 1.3158. But NIRP states NO daily load profile anywhere in the document --
# no typical-day curve, no peak timing, no temporal resolution -- so block
# factors CANNOT be derived from it. Raises if selected, by design.
PEAK_NIRP_PERIOD_FACTOR = None   # SOURCE: [PENDING]
PEAK_NIRP_SEASON_FACTOR = None

# CONSTRUCTED upper bound, not measured. Evening factor pinned at 1.20 (the
# model's own former unsourced default); night and day renormalised to an
# hours-weighted mean of 1.0. Rationale: the 2016 shape predates DARES/REA
# distributed solar, 2023 subsidy removal and 2024 Band A tariff reform, all
# of which shift daytime load behind the meter and concentrate grid draw in
# the evening. Drift since 2016 is expected TOWARD greater evening peaking,
# so this arm sweeps the DIRECTION of the vintage risk, not both directions.
PEAK_HIGH_PERIOD_FACTOR = {"night": 0.8849, "day": 1.0005, "peak": 1.2000}
PEAK_HIGH_SEASON_FACTOR = {"dry": 1.0224, "wet": 0.9778}


def peak_multiple_scenarios():
    """
    Peak-demand SHAPE arms: {period_demand_factor, season_demand_factor}.
    PAR is DERIVED from whichever arm is selected in load_scenario() (never
    supplied here directly -- see the module comment above). SOURCE STATUS:
    peak_suppressed, peak_unsuppressed and peak_high are filled (the last is
    constructed, not measured); peak_nirp is [PENDING] (None, raises if
    selected). season_demand_factor is None (= use
    DEFAULT_SEASON_DEMAND_FACTOR) unless an arm's own constant above is
    filled in. The NERC/TCN 1.0847 peak-to-average observation is a
    SUPPRESSED anchor only and backs no arm here (docs/
    sourcing_pack_02_ledger.md, section 3). Keys are what
    peak_multiple_case accepts in load_scenario().
    """	
    return {
        "peak_suppressed": {
            "period_demand_factor": PEAK_SUPPRESSED_PERIOD_FACTOR,
            "season_demand_factor": PEAK_SUPPRESSED_SEASON_FACTOR,
        },
        "peak_unsuppressed": {
            "period_demand_factor": PEAK_UNSUPPRESSED_PERIOD_FACTOR,
            "season_demand_factor": PEAK_UNSUPPRESSED_SEASON_FACTOR,
        },
        "peak_nirp": {
            "period_demand_factor": PEAK_NIRP_PERIOD_FACTOR,
            "season_demand_factor": PEAK_NIRP_SEASON_FACTOR,
        },
        "peak_high": {
            "period_demand_factor": PEAK_HIGH_PERIOD_FACTOR,
            "season_demand_factor": PEAK_HIGH_SEASON_FACTOR,
        },
    }


# ============================================================
# RESERVE MARGIN -- SPINNING RESERVE (2.6, F0 Step 2.3)
# ============================================================
# SOURCE: NIRP 2024 Section 2.3, p.19 -- spinning reserve requirement.
# Unlike the PAR arms above, this IS a cited value, not a placeholder.
#
# load_scenario()'s reserve_margin argument accepts either a plain fraction
# (used as-is, unchanged) or the literal string 'nirp_spinning', which
# derives a PER-YEAR fraction as RESERVE_MARGIN_SPINNING_MW / peak_mw[t]:
# a FIXED absolute MW requirement implies a SHRINKING fractional margin as
# peak_mw[t] grows over the horizon. Do not average this into a single
# scalar -- report and use the per-year series.
RESERVE_MARGIN_SPINNING_MW = 900.0  # NIRP 2024 Section 2.3, p.19 -- spinning reserve requirement


# ============================================================
# PLANNING HORIZON
# ============================================================
# ============================================================
# HORIZON — model vs reporting (plan 2.5 step 4, end-effect control)
# ============================================================
# The model runs ten years beyond the reporting window so that terminal-year
# effects fall outside the results.
#
# WHY: salvage credits a final-year vintage at (life-1)/life of capex --
# 96.7% for 30-year solar -- discounted at the same factor the capex is paid
# at, so net cost is ~3.3% of discounted capex, about $11.5/MWh. Below gas
# SRMC. With no build-rate cap the model exploited this: 2045 solar additions
# jumped to 17,861 MW (from 1,635 MW in 2044) and 2045 gas generation fell
# from 24.67 to 0.00 TWh_e. Cumulative solar rose 25.3 -> 39.3 GW while the
# objective FELL 0.7%. A horizon artefact, not an economic result.
#
# It was invisible while an unsourced 2,000 MW/yr build cap was active -- the
# same lesson as the min-build floor, at the other end of the horizon.
#
# THE FIX IS THE HORIZON, NOT THE SALVAGE TERM. Salvage is unchanged: same
# formula, every vintage. With the edge at 2055, a 2045 vintage serves 11 of
# 30 years, so its salvage is 63% rather than 97% and net cost is ~37% of
# capex -- no longer cheaper than gas. 2045 becomes an ordinary interior year.
# The arbitrage relocates to 2055, outside the reporting window.
#
# The edge cannot be removed entirely (a 2045 vintage is not fully depreciated
# until 2075); it is moved out of the results. Horizon extension is the
# standard treatment for end-of-horizon effects in capacity-expansion models.
#
# 2046-2055 series are EXTRAPOLATIONS, not sourced -- no published Nigerian
# demand or gas deliverability projection extends past 2045. Acceptable
# precisely because those years are never reported: their only function is to
# hold the terminal edge away from the results window.
MODEL_START_YEAR = 2025
MODEL_END_YEAR   = 2054
REPORT_END_YEAR  = 2045

# SOURCE: NERC Quarterly Reports -- 23 gas plants tracked for availability,
# Dec 2024 and 2026/Q1, both totalling 10,947 MW. AES_1 (294 MW) and
# Gbarain_1 (120 MW) are licensed (NERC 2024 Table 4.3) but carry no
# availability data and are excluded. Named here (2.12, Step 1) so
# gas_vintage_retirement_series_mw()'s guard checks against the SAME
# number load_scenario() uses for "gas_baseline_mw" below, rather than a
# second, driftable copy of 10947.
GAS_BASELINE_MW = 10947


def planning_horizon(start_year: int = MODEL_START_YEAR,
                     end_year: int = MODEL_END_YEAR) -> np.ndarray:
    """Define the planning horizon (annual resolution)."""
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")
    return np.arange(start_year, end_year + 1)


# ============================================================
# SCENARIO REGISTRIES
# ============================================================

def demand_level_scenarios() -> dict[str, float]:
    """
    Base-year demand level scenarios (TWh/year).

    - served     : Tier 1 base — observed GROSS GENERATION at busbar, 2024.
                   [SOURCE: NERC 2024 — 37,093.70 GWh]. Not a bound; measured.
                   NOTE: the key name is a misnomer pending rename (plan 1.6
                   Step 5). Served demand is 21.97 TWh; see loss ladder above.
    - latent_low : DEPRECATED (raises in load_scenario) — was lambda = 0.60
                   applied to the collected base.
    - latent_high: DEPRECATED (raises in load_scenario) — was lambda = 0.30.

    Traceable to: NERC 2024 annual gross generation (37,093.70 GWh).
    NOTE: data/demand/demand_base_annualized_2024.csv.xlsx is STALE — it holds the
    superseded 23.08 collected-energy base and is not read by this module.
    """
    return {
        # WARNING: latent_low/latent_high are retained for provenance only and
        # raise in load_scenario(). Do not iterate this dict without filtering.
        # Tier 1 (calibration anchor): observed GROSS GENERATION at busbar, 2024.
        # Corrected from 23.08, which was COLLECTED energy (a revenue quantity),
        # not demand. See docstring for the loss ladder.
        "served": 37.09,
        # DEPRECATED — wrong base (built as multiples of collected energy).
        # Rebuilt as the tiered demand architecture in Phase 2. DEM-1 suspended.
        "latent_low": 38.47,
        "latent_high": 76.93,
    }


def demand_growth_scenarios() -> dict[str, float]:
    """
    Annual demand growth arms (fraction/year), applied to the Tier 1 base
    (37.09 TWh gross generation at busbar, NERC 2024).

    SOURCING: correction plan Phase 1.6, Steps 1-4. Every value below is
    triangulated from at least two independent routes. Workbook and full
    provenance: data/demand/step1_per_capita_diagnostic.xlsx and
    data/demand/README_step1_per_capita.md.

    WHY NOT HISTORICAL GENERATION CAGR
    ----------------------------------
    Nigerian grid generation is a SUPPLY series, not a demand series. Over
    2010-2025 it grew 3.14%/yr while population grew 2.39%/yr — i.e. 76% of
    observed growth is demographic. Per CONNECTED person, generation FELL
    1.66%/yr (327 -> 258 kWh, 2010-2024). Calibrating demand growth to this
    series would assume the supply constraint persists and then report modest
    capacity needs because demand was assumed not to grow. Same error class as
    the superseded 23.08 TWh base. NIRP 2024 (Table 14) reaches the same
    conclusion and forecasts bottom-up from drivers instead.

    THE THREE SOURCING ROUTES
    -------------------------
    Route A - driver decomposition.  g = eps*g_GDP + (1-eps)*g_pop
        eps    = 0.80, long-run income elasticity of electricity demand.
                 Liddle & Hasanov (2023), Environmental and Ecological
                 Statistics, p.827/p.829/p.843: economy-wide electricity
                 demand (not total energy), 31-country middle-income
                 panel, 1990-2019, income elasticity averaged 0.8 (stated
                 range 0.7-0.9). Traced against the primary (register
                 L54); supersedes an earlier 0.76-0.90 range with a 0.83
                 midpoint, neither of which traced to this or any source.
                 The model excludes SSA-estimated elasticities (0.60-0.76,
                 unsourced here) as rationing-contaminated. Note the cited
                 panel itself includes South Africa and Cote d'Ivoire (2
                 of 31), so this exclusion is partial.
        g_pop  = 1.77%/yr, 2024-2045 average [UN WPP: 232.68m -> 336.66m].
                 NOTE: g is INSENSITIVE to this. The population coefficient is
                 (1-eps) = 0.20, so a 0.3pp change in g_pop moves g by about
                 0.06pp. g is approximately eps x GDP growth. The GDP
                 assumption is the load-bearing choice, not the population
                 assumption.
        g_GDP  = declared arms, not forecast. See table below.

    Route B - per-capita convergence.  Nigeria 2024 = 161 kWh/cap.
        Nigeria is ~15 years behind Kenya (Kenya crossed 161 kWh/cap in 2009).
        Ghana is unusable as a target: Nigeria's 2024 level is BELOW Ghana's
        worst year since 2000, and converging to Ghana's 2024 level (705
        kWh/cap) would require 9.24%/yr.

    Route C - NIRP 2024, Nigeria's official Integrated Resource Plan.
        Total demand (grid + off-grid) 59.6 TWh (2022) -> 328.6 (2045) = 7.70%.
        This is TIER 2 SCOPE, not Tier 1: the base already includes
        self-generation. NIRP's grid-only rate (11.0%) is organic growth PLUS
        absorption of self-generation and must NOT be applied here — in this
        architecture suppression closure is the Tier1->Tier2 gap, not a growth
        rate. Applying it would double-count.
        NIRP's implied elasticity is 1.19 (= (7.7-2.1)/(6.8-2.1)), ~35% above
        the credible literature; its GDP path (6.8%/yr to 2045) is roughly the
        pre-2015 oil-boom rate. Aggressive on both terms simultaneously.

    THE ARMS
    --------
    constrained_continuation = 0.0314
        Ember 2010-2025 grid generation CAGR. This is a CONSTRAINT-PERSISTENCE
        arm ("what if the constraints hold"), NOT organic demand growth, and
        must never be presented as such. Window range 1.53%-3.22%.
        Clears the floor test (>1.77% population growth) and implies 1.35%/yr
        per-capita growth — above the 0.73% actually observed, so even the
        pessimistic arm is mildly optimistic against the record.

    organic_central = 0.039
        Route A: eps 0.80 (Liddle & Hasanov 2023, p.829/p.843, economy-wide
        electricity, 31-country middle-income panel 1990-2019), GDP 4.5%
        (declared assumption) -> 3.95%.
        Route B: converge to Kenya's 2024 level by 2045 -> 3.67%.
        Adopted: 3.9%. GDP 4.5% is a DECLARED ASSUMPTION: above the
        post-2015 average (two recessions, weak diversification),
        materially below the oil-boom rate. It does NOT assume a commodity
        boom returns. Verify the two historical averages against WDI
        NY.GDP.MKTP.KD; IMF WEO covers only ~5 years, so beyond 2030 this
        is an assumption, not a forecast.
        SUPERSEDES 4.0% (Route A at an untraced eps=0.83): see register
        L54 for the full citation history.

    organic_high = 0.060
        Route A: eps 0.90, GDP 6.8% -> 6.30%.
        Route B: converge to Ghana's 2015 level by 2045 -> 6.40%.
        Route C: NIRP low case -> 5.9%.

    NOT AN ARM: NIRP's 7.7% base case. See demand_growth_benchmark().

    KNOWN LIMITATIONS (must be disclosed in the thesis)
    ---------------------------------------------------
    1. AMPLIFICATION. Gas is deliverability-bound and hydro is exogenous, so
       solar is an accounting residual: solar_TWh(2045) ~ D(2045) - 39.9. The
       amplification factor D/(D-39.9) is ~2x. A 1pp error in g moves the
       headline solar build by roughly 20%.
    2. CONSTANT g. NIRP's trajectory is front-loaded (10.8%/yr to 2030, then
       6.9%). A constant g reaching the same D(2045) understates near-term
       build pressure and therefore near-term financing stress. Piecewise g
       (break at 2030) requires changing demand.project_baseline_demand, which
       takes a scalar. Deferred deliberately — one structural change at a time.
    3. CONSTRAINT-RELIEF ASYMMETRY. These arms are sourced from UNCONSTRAINED
       drivers while gas deliverability is anchored to OBSERVED CONSTRAINED
       throughput (89.27 TWh_th, a commercial not a geological constraint).
       That asymmetry is what generates the solar residual. It is a scenario
       design choice and must be declared, not inherited. See the demand-arm x
       gas-regime coherence cross-tab (plan 1.6 Step 8).
    4. GDP IS EXOGENOUS. Electricity supply constrains GDP in reality; that
       feedback is not modelled. High-GDP arms paired with high unserved energy
       are internally incoherent. Excluding those cells is CONSERVATIVE (it
       removes demand, not adds it). NIRP has the identical omission and does
       not flag it.
    """
    return {
        "constrained_continuation": 0.0314,
        "organic_central":          0.039,
        "organic_high":             0.060,
    }


def demand_growth_benchmark() -> dict[str, float]:
    """
    External demand-growth benchmarks from other models. NOT scenario arms.

    Deliberately kept OUT of demand_growth_scenarios() so that consumers
    iterating that registry (optimize_experiments.run_all_deterministic_
    scenarios, scripts/13_) are unaffected, and so the three-arm structure
    stays intact.

    nirp_2024_base = 0.077
        NIRP 2024 total-demand CAGR, 59.6 TWh (2022) -> 328.6 TWh (2045).
        Tier 2 scope. Rests on eps = 1.19 and GDP 6.8%/yr, both above the
        credible ranges used for the arms above. Used only for the NIRP
        benchmark comparison run.
    """
    return {"nirp_2024_base": 0.077}


def demand_growth_prior() -> dict[str, float]:
    """
    Prior over annual demand growth for Monte Carlo sampling (scripts/17_).

    SOURCED — correction plan Phase 1.6 Step 5b. Supersedes the previous
    N(0.04, 0.01), in which BOTH parameters were unsourced: the mean was
    inherited from a placeholder of unknown provenance, and sigma was assumed.

    MEAN = demand_growth_scenarios()["organic_central"] = 0.039
        Triangulated from three independent routes; see
        demand_growth_scenarios() for the full derivation and sources. The key
        is referenced, not hardcoded, so the prior cannot drift from the arms.
        (organic_central moved 0.040 -> 0.039 when the elasticity was traced
        to its primary source, register L54; SIGMA below was derived from the
        superseded 0.040 arm set and has NOT been recomputed — see the STALE
        note after it.)

    SIGMA = 0.0120
        Derived from the SPREAD BETWEEN THE ARMS rather than assumed. It is the
        equal-weighted standard deviation of the three sourced arms:

            arms          : 0.0314, 0.0400, 0.0600   [SUPERSEDED, see STALE note]
            arm mean      : 0.0438
            deviations    : -0.0124, -0.0038, +0.0162
            sigma         : sqrt(mean of squared deviations) = 0.01198 -> 0.0120

        This is the defensible construction: the dispersion now reflects the
        observed disagreement between the driver decomposition, the per-capita
        convergence path and the NIRP forecast, rather than an analyst's guess
        about how uncertain the parameter feels.

        STALE (declared, not fixed): organic_central changed to 0.039
        (register L54); this SIGMA and the worked arm-mean/deviations above
        were computed against the superseded 0.0400 and are not recomputed
        here — recomputing SIGMA is a second value change this task did not
        authorize (only organic_central moved). SIGMA remains 0.0120 pending
        that decision; see docs/task_log_elasticity_sourcing.md.

    KNOWN LIMITATION — THE PRIOR IS SYMMETRIC BUT THE ARMS ARE NOT
    ---------------------------------------------------------------
    The arms are right-skewed around organic_central (figures below predate
    the 0.040 -> 0.039 move; see STALE note above):

        constrained_continuation  0.0314   = central - 0.86 pp
        organic_central           0.0400   = central
        organic_high              0.0600   = central + 2.00 pp

    The upside arm is 2.3x further from centre than the downside arm, because
    the downside is bounded by the observed historical record (a
    constraint-persistence scenario cannot fall far below what actually
    happened) while the upside is bounded only by the GDP assumption, which is
    genuinely open. The equal-weighted arm mean is 0.0438, ABOVE
    organic_central.

    A symmetric normal therefore:
      - understates the probability of high-growth outcomes, and
      - overstates the probability of growth below the historical record.

    Because of the ~2x amplification into the solar build (see
    demand_growth_scenarios(), limitation 1), this is not cosmetic: it biases
    the cost and infeasibility distributions DOWNWARD. MC results are
    conservative in this respect, which is the safer direction, but the
    asymmetry must be disclosed.

    Candidate replacements, deferred to plan 5.1 rather than fixed here:
      (a) an explicit three-point discrete prior over the arms with declared
          weights — simplest, fully transparent, and matches how the arms were
          actually constructed;
      (b) a lognormal calibrated to the arm quantiles.
    Option (a) is preferred: it makes the prior an exact restatement of the
    sourcing work rather than a smooth approximation to it.

    ALSO UNRESOLVED (plan 1.6 Step 8): scripts/17_ samples this prior
    INDEPENDENTLY of the gas regime. Both are driven by the same underlying
    reform dynamic, so independent sampling generates incoherent worlds and
    inflates both tails. No MC tail statistic — including the headline
    infeasibility rate — is reportable until the correlation structure is
    imposed.
    """
    return {
        "mean":  demand_growth_scenarios()["organic_central"],
        "sigma": 0.0120,
    }


def solar_build_scenarios() -> dict[str, float | None]:
    """
    Solar deployment-capability arms, as MULTIPLIERS on the NIRP 2024
    cumulative capacity trajectory (see nirp_solar_cumulative_mw()).

    The previous flat 2,000 MW/yr annual cap is REMOVED (plan 2.5 step 4). It
    was unsourced and wrong in shape at both ends: it permitted 2,000 MW in
    2026, when NIRP judges ~240 MW achievable, and capped 2042-45 at 2,000 when
    NIRP builds 6.0-8.3 GW/yr. A constant rate against compounding demand must
    eventually bind at any level, so the terminal build sitting exactly on
    2,000.0 was an artefact of the constraint's SHAPE, not a deployment limit.

    nirp_trajectory        1.0x  -- NIRP's own schedule. Binds 2027-2037
                                    against the current solution (cumulative
                                    shortfall peaks at 4,505 MW in 2032).
                                    REPORT UNSERVED ENERGY AND FEASIBILITY
                                    FROM THIS ARM, NOT NPV, until plan 2.6:
                                    the shortfall is priced at VoLL and the
                                    objective measures the penalty parameter.
    nirp_accelerated       2.0x  -- deployment reform succeeds. The coherent
                                    pairing for constrained-gas regimes, where
                                    solar must compensate for gas that does
                                    not recover.
    deployment_unconstrained None -- no cap. The counterfactual that isolates
                                    what the deployment limit itself costs.
                                    The gap to nirp_accelerated is the
                                    deployment-capability cost -- a quantity
                                    not reported anywhere in the SSA
                                    power-planning literature.

    [SOURCE NEEDED] for the 2.0x accelerated multiplier: currently a declared
    assumption. Candidate anchors are comparator national ramps (Vietnam
    2019-20, Egypt Benban, South Africa REIPPPP rounds).

    The legacy case names conservative/baseline/aggressive are NOT aliased.
    load_scenario raises on unknown keys, so every call site fails loudly
    rather than silently inheriting a different constraint.
    """
    return {
        "nirp_trajectory":          1.0,
        "nirp_accelerated":         2.0,
        "deployment_unconstrained": None,
    }


def nirp_solar_cumulative_mw() -> dict[int, float]:
    """
    NIRP 2024 cumulative solar PV capacity by year (MW).

    [SOURCE: NIRP 2024, Annex F Table 19 solar PV generation (TWh), converted
    to capacity using the Table 8 capacity factor, 19% in 2025 declining
    linearly to 18% in 2045.]

    Validation: the series reaches 61,898 MW at 2045 against NIRP's published
    ~61 GW headline, confirming the CF basis.

    NIRP's own annual build rate implied by this series: essentially nothing to
    2032 (3 MW in 2030), 1.3-1.7 GW/yr 2033-2036, then 2.4-8.3 GW/yr to 2045.
    Peak 8,302 MW in 2043; mean 2,948 MW/yr over the horizon.

    APPLIED TO CUMULATIVE ADDITIONS, NOT TOTAL CAPACITY. NIRP's 2025 figure
    (180 MW) is below this model's existing solar baseline (500 MW), so a
    total-capacity cap would be infeasible in year one. Applying it to
    additions is mildly generous -- NIRP's total includes existing plant -- and
    must be declared as such.

    NOTE ON COHERENCE (see docs/scenario_coherence.md): NIRP builds almost no
    solar before 2032 BECAUSE it assumes gas recovers to 88 TWh_e by 2035. Its
    solar schedule is downstream of its gas assumption. Pairing this arm with a
    constrained-gas regime is the pessimistic stress corner, not a central
    case, and it will produce large unserved energy through the 2030s.
    """
    # 2025-2045: the NIRP source series covers exactly these years. Buffer
    # years past 2045 are extrapolated in _solar_cumulative_cap, not here.
    years = list(range(2025, 2046))
    years = list(range(2025, 2046))
    cum = [180, 422, 664, 848, 1093, 1096, 1099, 1163, 2455, 4123, 5739,
           7425, 9802, 13748, 19275, 25333, 31361, 37360, 45662, 53947, 61898]
    return dict(zip(years, cum))


def asset_lifetimes() -> dict[str, float]:
    """
    Technical SERVICE life by asset class (years). Used for the end-of-horizon
    salvage credit (correction plan 2.5).

    DEFINITION: technical service life -- how long the asset physically
    operates. NOT the capital recovery period (a financial assumption used in
    LCOE, typically shorter) and NOT the performance warranty.

    solar   : 30 yr. [SOURCE: IRENA (2026), Renewable Power Generation Costs in
              2025.] Cross-checks against NREL ATB, the source of
              SOLAR_CAPEX_PER_MW.
    storage : 15 yr, UNAUGMENTED. [SOURCE: NREL ATB, utility-scale battery
              storage.] ATB's 30-year PV-plus-battery life assumes cell
              replacement at year 15; this model carries no augmentation capex,
              so the unaugmented figure is the correct one.
              WARNING: 15 < the 21-year horizon. Any storage vintage built
              before 2031 retires WITHIN the horizon, and the model neither
              charges replacement capex nor stops dispatching it. Inert as at
              Phase 2.5 (first storage build is 2038), but Phases 2.6/2.7 may
              shift the optimum earlier. solve_model() carries a post-solve
              guard that warns if this occurs -- do not ignore it.
    gas     : 30 yr. [SOURCE: NREL ATB, Gas-CC technical life.] Inert while
              gas_add is fixed at 0; required for the plan 5.3 experiment.

    DECLARED SIMPLIFICATIONS -- all bias salvage UPWARD, so the corrected NPV
    is a LOWER bound on system cost:
      1. Straight-line residual, no degradation (~0.5%/yr for solar).
      2. No decommissioning or disposal cost.
      3. Assumes a liquid residual value. The real economic content is
         continued service beyond 2045, which is a different claim.
      4. Credited at DF[T_last] rather than DF[T_last + 1], following the
         model's convention that year t occurs at time t. Marginally
         understates discounting, i.e. overstates salvage.

    SALVAGE ACCOUNTING DECISIONS (methods chapter):
      - Salvage enters the OBJECTIVE ONLY, never the capital budget
        constraint. The envelope constrains cash out the door; an investor
        cannot fund a 2045 build with residual value realised in 2045.
        Netting it inside the constraint would inflate the envelope ~25% for
        free and soften the finding that public capital binds.
      - Salvage is computed on RAW capex, never on EaaS-marked-up capex.
        required_margin is a financing premium, not asset value -- a panel
        bought with expensive capital is not worth more at resale.
      - Residual value accrues SYSTEM-WIDE. Coherent for a social-planner
        objective, but it blurs ownership: in reality the EaaS investor holds
        it. Revisit at 3.2 when the tranche formulation makes ownership
        explicit.
    """
    return {
        "solar":   30.0,
        "storage": 15.0,
        "gas":     30.0,
    }


def asset_lifetime_sweep() -> dict[str, list[float]]:
    """
    Sensitivity bands on asset life. Life enters the headline NPV directly
    through the salvage term, so both are swept.

    [SOURCE NEEDED] for the band WIDTHS -- these are declared assumptions, not
    sourced ranges.

    Storage matters more than solar despite the narrower absolute band: at a
    15-year life, +/-3 years is a 20% proportional change against 17% for
    +/-5 on 30, and it lands directly on the 28.2 GWh sliced-model result
    that is being reported as a methods finding.
    """
    return {
        "solar":   [25.0, 30.0, 35.0],
        "storage": [12.0, 15.0, 18.0],
    }


def solar_min_build_default() -> float:
    """
    Minimum annual solar build (MW/yr) applied by runners.

    STATUS: [UNSOURCED] -- an arbitrary floor, plan 2.5. Its docstring in
    optimize_model.py claims it prevents "pathological all-delay solutions that
    leave 2025-2030 with no solar". That justification is being tested: the
    2025 backcast is gas 24.57 + hydro 11.47 + baseline solar 0.88 = 36.9
    against demand 37.09, so essentially NO new solar is needed in 2025. If
    removing the floor gives 2025 = 0 MW, that is the CORRECT answer -- a model
    that builds capacity it does not need is the distorted one.

    Consolidated here (Step 3a) so the value lives in one place. Set to 0.0 in
    Step 3b, after byte-identity has verified the consolidation.

    2.11, Step 1: as of this phase, no runner calls this function any more --
    every scenario["solar_min_build_mw_per_year"] = solar_min_build_default()
    override has been deleted (they forced this [UNSOURCED] floor over
    load_scenario()'s own sourced 0.0 default). The firm-capacity constraint
    (2.6, F0 Step 2.3) now performs the role this floor was invented for: it
    already forces enough capacity to be built to cover peak demand with a
    reserve margin, which is a sourced, dimensioned requirement -- unlike this
    arbitrary MW/yr floor. Left in place, unchanged, rather than deleted, so
    the STATUS: [UNSOURCED] history above stays attributable. See
    docs/phase_2_11_minbuild_retirement.md.
    """
    return 0.0


def carbon_policy_scenarios() -> dict[str, dict[str, float | bool]]:
    """Carbon policy stances (deterministic). Price in USD/tCO2."""
    return {
        "no_policy": {
            "carbon_active": False,
            "carbon_price": 0.0,
        },
        "moderate_policy": {
            "carbon_active": True,
            "carbon_price": 50.0,
        },
        "stringent_policy": {
            "carbon_active": True,
            "carbon_price": 75.0,
        },
    }


def gas_deliverability_scenarios() -> dict[str, str]:
    """
    Gas deliverability-to-power scenario labels.

    Must match the 'scenario' column in:
    data/gas/processed/gas_available_power_annual_twh_th.csv

    Structural (all RQs): downside, baseline, upside, shock_recovery
    Level-equivalent flat controls (GAS-3 only): flat_downside, flat_upside,
      flat_shock_recovery -- identical cumulative supply, no shape variation,
      to isolate SHAPE vs LEVEL effects.
    """
    return {
        "downside":            "downside",
        "baseline":            "baseline",
        "upside":              "upside",
        "shock_recovery":      "shock_recovery",
        "flat_downside":       "flat_downside",
        "flat_upside":         "flat_upside",
        "flat_shock_recovery": "flat_shock_recovery",
        # GMP 2026 policy-target family (distinct from the historical-drift
        # arms above -- see data/gas/processed/gas_deliverability_cases.yaml).
        # I applied to the planned increment: 0.0 / 0.50 / 1.0.
        "gas_deliv_flat":        "gas_deliv_flat",
        "gas_deliv_gmp_partial": "gas_deliv_gmp_partial",
        "gas_deliv_gmp_full":    "gas_deliv_gmp_full",
    }


def land_scenarios():
    """Land availability envelopes (km2)."""
    return {
        "tight": 1000,
        "moderate": 3000,
        "loose": 6000,
    }


def capital_envelope_scenarios():
    """
    Public capital envelopes (NPV over full planning horizon, USD).

    B* is the public clean-energy capital requirement: the realised public
    spend (public solar CAPEX + storage CAPEX + EaaS subsidy) at the point
    where the public capital envelope ceases to bind, under a named
    reference plan.

    Derived value: 10,273,343,610 USD (NPV). Source:
    `28_run_capital_frontier.py`, run 2026-09-20, uncapped central
    configuration (carbon_case="no_policy", no emissions_cap_by_year,
    solar_build_case="deployment_unconstrained"), sliced builder,
    `om_central`, traditional financing regime; arms per
    `results/capital_frontier_run.log`; the traditional slack point at
    envelope 10,924,800,000 realises 10,273,343,610 and carries a dual of
    zero. Superseded the previous undocumented constant 9,104,000,000, which
    understated the requirement by 12.84% and had no derivation anywhere in
    the repository (see L45).

    Scope condition. The requirement is plan-dependent: the envelope charges
    only public solar CAPEX, storage CAPEX and the EaaS subsidy -- gas CAPEX
    is not charged to it -- and envelope consumption is near zero under
    no-policy and large under an emissions cap. B* is therefore the
    requirement under the reference plan named above, not a universal
    adequacy figure. The tier names (`tight` 0.50, `moderate` 0.85,
    `adequacy` 1.00, `expansion` 1.20) are fractions of that reference
    requirement. Coded values (int() truncation): tight 5,136,671,805;
    moderate 8,732,342,068; adequacy 10,273,343,610; expansion
    12,328,012,332.

    Anchors: `01_run_baseline.py` and `01b_run_baseline_sliced.py` run at
    capital_case="unconstrained" (no envelope constraint), so the regression
    and central anchors are unaffected by this constant. Every runner using
    tight/moderate/adequacy/expansion IS affected: results solved under the
    previous constant carry envelopes 0.50/0.85/1.00/1.20 x 9,104,000,000.
    """
    B_star = 10_273_343_610

    return {
        "tight": int(0.50 * B_star),
        "moderate": int(0.85 * B_star),
        "adequacy": int(1.00 * B_star),
        "expansion": int(1.20 * B_star),
        "unconstrained": None,
    }


def tariff_scenarios() -> dict[str, float]:
    """
    EaaS service tariff arms (USD per TWh; 1 USD/MWh = 1e6 USD/TWh).

    Replaces solar_tariff_scenarios() and its provisional 45/65/85M grid,
    retired with the defaulted solar_tariff_case (docs/tariff_arm.md,
    Steps 1-3). tariff_case is now a REQUIRED parameter of load_scenario().

    SOURCED, natively USD -- no FX conversion, so the naira-basis
    problem that affects r_comm does not arise here.
    The anchors are the two positions in Nigeria's 2016 NBET solar
    IPP negotiation:
    tariff_high 115 USD/MWh -- the rate NBET agreed with 14
    developers, PPAs initialled July 2016, down from an earlier
    23 cents/kWh. The same rate appears as the guaranteed tariff
    for the Katsina 75 MW plant.
    tariff_low 75 USD/MWh -- the Federal Government's counter-
    position; some investors agreed to proceed at this level.
    The finance ministry cited competitive procurements in South
    Africa, Zambia, Egypt and Ethiopia at 5-7 cents/kWh.
    tariff_central 95 USD/MWh -- the MIDPOINT of the two positions.
    A CONSTRUCTION, not an independently sourced figure. It
    coincides with the incumbent bare literal.
    CORROBORATION: BNEF found C&I solar generable at 0.10-0.14
    USD/kWh in Kenya, Nigeria and Ghana, declining toward
    0.05-0.07 by 2030.
    CAVEATS: (1) these are utility-scale grid-connected IPP rates,
    not C&I PPA rates -- the right match for this model, where the
    EaaS channel is grid-connected and physically identical to
    public solar (register L16), but state it rather than imply it.
    (2) 2016-2018 vintage; solar costs have fallen since, so a flat
    tariff against declining capex makes bankability progressively
    easier through the horizon. (3) BNEF puts Nigerian C&I capex at
    1,100-1,600 USD/kW against this model's 905 -- register L16
    (DIST-5) restated from the other side.
    The two legacy arms are RETAINED FOR ATTRIBUTION ONLY, both
    UNSOURCED: 65M was load_scenario()'s silent default, 95M the
    runner override literal. Neither is a candidate central case.
    """
    return {
        "tariff_low":         75_000_000.0,
        "tariff_central":     95_000_000.0,
        "tariff_high":       115_000_000.0,
        "tariff_legacy_95m":  95_000_000.0,
        "tariff_legacy_65m":  65_000_000.0,
    }


def om_scenarios() -> tuple[str, ...]:
    """
    Names of the fixed + variable O&M cost arms (task_om_implementation).
    REQUIRED, no default -- see the _OM_REQUIRED sentinel above
    load_scenario().

    'om_low' / 'om_central' / 'om_high' select the matching om_case in
    data/cost/processed/om_cost.csv (NREL ATB 2024: Advanced / Moderate /
    Conservative cost cases), per technology (solar_utility, storage,
    gas_ct, gas_cc), year (2025-2050). 'om_central' is the new central
    case (replacing the retired STORAGE_OM_PER_MWH_YR, which never
    reached the objective for any technology -- see
    docs/task_storage_om.md).

    'om_none' is SYNTHETIC: not a row in the CSV, it forces every O&M
    rate to 0.0. This is the REGRESSION arm -- with it selected, both
    builders' O&M term is identically 0.0 and the pre-O&M anchors
    (docs/task_om_implementation.md) must reproduce to the digit.
    """
    return ("om_none", "om_low", "om_central", "om_high")


# Tariff sweep grid for FIN-2 bankability analysis (USD per TWh).
# STALE-DOC: fine resolution was placed around the old 44-46M crossing. Under
# the blended-finance rates the crossing moves (higher, and blend-dependent).
# Widen and re-centre this grid in the final pass once rates are frozen.
TARIFF_SWEEP_GRID = [
    30_000_000,
    35_000_000,
    40_000_000,
    44_000_000,
    46_000_000,
    50_000_000,
    60_000_000,
    75_000_000,
    95_000_000,
    110_000_000,
]


def gas_probability_weights():
    """
    Prior over gas deliverability regimes for Monte Carlo robustness diagnostics.

    Not a forecast -- a weighted sensitivity prior. Report headline results under
    symmetric weights (0.25 each) as a robustness check as well.

    SOURCING NOTE: the justifications reference NGC Decade of Gas, IEA Nigeria
    Outlook 2023, Nigerian Gas Master Plan / PIA 2021. Confirm each citation
    against a primary source before submission (standing verification rule).
    """
    return {
        "baseline":       0.50,
        "downside":       0.25,
        "upside":         0.20,
        "shock_recovery": 0.05,
    }


def storage_cc_scenarios() -> dict[str, float]:
    """
    Storage's firm-capacity credit (fraction of storage_power_mw counted
    toward the firm-capacity constraint). Named levels low/central/high
    (2.6, phase_2_3b Step 2).

    SOURCE: the effective load-carrying capability (ELCC) literature for
    4-hour battery storage, all US systems -- no Nigerian or African ELCC
    study was found, so applying these figures here is a DECLARED
    TRANSFER, not a like-for-like measurement:
      - Frazier et al. / NREL, "Average and Marginal Capacity Credit
        Values of Generation and Storage Technologies in the Standard
        Scenarios" (NREL/TP-6A20-89587): average 4-hour battery capacity
        credit 66-100% across 2036-2050; marginal 67-100%.
      - Southwest Power Pool, "2022 ELCC ESR Study Report" (Feb 2023),
        Figure 3 and accompanying table: 4-hour Energy Storage Resource
        accreditation 100% / 96% / 84% at 1,000 / 3,000 / 5,000 MW
        installed.
      - LBNL, "A Simple and Fast Algorithm for Estimating the Capacity
        Credit of Solar and Energy Storage": 4-hour marginal credit
        declining from 85-95% to 40-60% as storage reaches 15% of peak
        demand.
    The literature's central finding is that ELCC DECLINES with
    penetration; the model instead applies one FIXED credit throughout
    the horizon, so the three arms below are read as a declared band
    across the literature's decline curve, not as a low/central/high
    confidence interval around one point estimate. At 63,056 MWh (2045,
    peak_unsuppressed arm) the model sits well past the first-tranche,
    high-credit end of that curve. Cross-reference §3.3.5.

    Also note, without applying it: NREL finds battery marginal capacity
    credits do NOT decline over time the way solar's and wind's do, and
    that added solar can RAISE storage ELCC by widening the net-peak
    window -- a second-order effect running counter to the penetration
    decline above in a solar-heavy plan like this model's. Not built
    into any arm; recorded here as a declared, unmodelled tension.

    storage_cc_high (1.00) = the first-tranche case (SPP at 1,000 MW
    installed; also reproduces the old flat duration/peak_hours figure,
    min(1.0, duration/peak_hours) = 1.0 at the model's 4-hour storage
    duration and 4-hour peak slice) -- an explicit, named UPPER bound,
    not a default.
    storage_cc_central (0.80) = a moderate-penetration reading, inside
    every source's measured range (NREL average/marginal, SPP at
    3,000-5,000 MW, LBNL's upper band) -- the central case.
    storage_cc_low (0.50) = the deep-penetration end of LBNL's declining
    curve (40-60% as storage approaches 15% of peak demand) -- brackets
    the low end if penetration effects are more severe than the central
    case assumes.

    See docs/parameter_provenance.md row 14 for the full six-field
    provenance record.
    """
    return {
        "storage_cc_low":     0.50,
        "storage_cc_central": 0.80,
        "storage_cc_high":    1.00,
    }


def gas_availability_scenarios() -> dict[str, float | None]:
    """
    Gas fleet availability arm (2.10, Step 2). Mirrors genset_case/
    peak_multiple_case/storage_cc_case discipline: REQUIRED, no default --
    see the _GAS_AVAILABILITY_REQUIRED sentinel above load_scenario().

    Feeds BOTH gas_availability_factor (annual energy ceiling on gas
    generation, in build_model()/build_model_sliced()) and
    capacity_credit['gas'] (peak-hour firm-capacity credit, in the firm
    capacity constraint) -- two DIFFERENT quantities that were a single
    hard-coded parameter (0.80) before 2.10. Both are now independently
    settable dict entries in load_scenario(), initialized equal from this
    SAME arm value only because no separately sourced peak-hour figure
    exists yet (2.10, Step 3) -- see the comments at both dict-entry sites
    in load_scenario().

    This must be FUEL-AGNOSTIC availability: mechanical + atmospheric
    availability only (NERC Availability Factor clauses (i) and (ii)). It
    must EXCLUDE clause (iii) feedstock availability, because gas-supply
    scarcity is already carried separately by m.gas_balance / the
    deliverability scenarios -- using NERC's headline Availability Factor
    (which includes clause iii) here would double-count fuel scarcity.

        gas_avail_observed   (0.3061): NERC 2024 (3,351.16 / 10,947 MW, the
            same 23 plants, Tbl 7.1). DECLARED LOWER BOUND -- double-counts fuel, because NERC's
            PAF includes fuel unavailability while the model constrains
            fuel separately in m.gas_balance. Not a candidate central case.
        gas_avail_central    (None): SOURCE: [PENDING]. Must represent
            MECHANICAL availability only, i.e. availability given fuel is
            allocated. Raises if selected until sourced.
        gas_avail_nameplate  (0.80): the pre-2.10 unsourced assumption,
            retained as an explicit UPPER BOUND so every pre-2.10 result
            stays attributable rather than orphaned.
    """
    return {
        "gas_avail_observed":  0.3061,
        "gas_avail_central":   None,
        "gas_avail_nameplate": 0.80,
    }


# ============================================================
# Gas plant vintage data (2.12, Step 1)
# ============================================================
# Companion to gas_retirement_scenarios()'s gas_retire_vintage_30/40/50
# arms -- see docs/gas_retirement_provenance.md for full provenance,
# per-plant confidence flags, and the Afam IV-V blended-vintage caveat.
# SOURCE: ASSUMED. Commissioning years in the CSV below are an analyst
# estimate from general knowledge of Nigerian power-sector project
# histories, NOT traced to NERC/NBET/NDPHC/TCN or company filings. See
# docs/limitations_register.md L8.
_GAS_VINTAGE_CSV_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "data", "capacity", "raw", "gas_plant_vintages.csv",
)

# Plants present in the vintage CSV (25 plants, 11,361 MW) but excluded
# from gas_baseline_mw / GAS_BASELINE_MW (10,947 MW) -- the SAME exclusion
# already documented at GAS_BASELINE_MW's definition (2.10 finding): AES_1
# and Gbarain_1 are licensed but carry no NERC availability data, so they
# are not part of the 23-plant fleet gas_baseline_mw represents.
# 11,361 - 294 (AES_1) - 120 (Gbarain_1) = 10,947 -- exact match. This is
# an EXCLUSION, not a scaling factor: the task is explicit that a mismatch
# here must be reconciled by removing the two plants that do not belong to
# the tracked fleet, not by rescaling every plant's MW to force a total.
_GAS_VINTAGE_EXCLUDED_PLANTS = frozenset({"AES_1", "Gbarain_1"})


def _load_gas_plant_vintages(csv_path=None):
    """
    Load per-plant installed MW and commissioning year from
    data/capacity/raw/gas_plant_vintages.csv (2.12, Step 1).

    SOURCE: ASSUMED -- see the module comment above and
    docs/gas_retirement_provenance.md for full provenance.

    Excludes AES_1 and Gbarain_1 (see _GAS_VINTAGE_EXCLUDED_PLANTS) so the
    surviving fleet is the SAME 23-plant, GAS_BASELINE_MW fleet the rest of
    the model retires from -- not the CSV's superset of it. Raises if the
    surviving total does not come out to exactly GAS_BASELINE_MW: a silent
    mismatch here would let the vintage retirement series and the baseline
    it retires from drift apart without warning.

    Returns
    -------
    list[dict]
        One dict per surviving plant:
        {"name": str, "installed_mw": float, "commissioning_year": int,
         "confidence": str}.
    """
    path = csv_path if csv_path is not None else _GAS_VINTAGE_CSV_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gas plant vintage file not found: {path}")

    plants = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"plant_nerc_name", "installed_mw", "commissioning_year", "confidence"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must contain columns {sorted(required)}; found {reader.fieldnames}"
            )
        for row in reader:
            name = row["plant_nerc_name"]
            if name in _GAS_VINTAGE_EXCLUDED_PLANTS:
                continue
            plants.append({
                "name": name,
                "installed_mw": float(row["installed_mw"]),
                "commissioning_year": int(row["commissioning_year"]),
                "confidence": row["confidence"],
            })

    total_mw = sum(p["installed_mw"] for p in plants)
    if abs(total_mw - GAS_BASELINE_MW) > 1e-6:
        raise ValueError(
            f"Gas plant vintages: after excluding "
            f"{sorted(_GAS_VINTAGE_EXCLUDED_PLANTS)}, the surviving fleet "
            f"totals {total_mw} MW, not GAS_BASELINE_MW ({GAS_BASELINE_MW} "
            "MW). The CSV must represent the SAME tracked fleet "
            "gas_baseline_mw counts (2.10) -- if the CSV changed, this is "
            "not a rounding difference to silently absorb; investigate "
            "which plants moved."
        )
    return plants


# ============================================================
# OCGT-to-CCGT conversion tiers (Phase 2.8, Step 1)
# ============================================================
# Per-plant availability for the 16 convertible open-cycle plants, from
# data/capacity/raw/gas_plant_availability.csv (NERC Table 7.1 2024;
# NERC 2026/Q1 Table 1 for 2025/Q4 and 2026/Q1). AES_1 and Gbarain_1 are
# NOT in the CSV -- licensed but absent from NERC's availability-tracked
# fleet, excluded per docs/limitations_register.md L8 (same exclusion as
# _load_gas_plant_vintages).
_GAS_CONVERSION_CSV_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "data", "capacity", "raw", "gas_plant_availability.csv",
)

# Merit-order banding on each plant's mean PAF across the three reported
# periods. These three predicates are the ONLY literals in the tier
# derivation -- membership, nameplate and availability-weighted PAF all
# fall out of the CSV. Bands partition [0, 1]: A is PAF > 0.50, B is
# 0.25 <= PAF <= 0.50, C is PAF < 0.25 (0.50 and 0.25 fall in B).
_CONVERSION_TIER_BANDS = (
    ("A", lambda paf: paf > 0.50),
    ("B", lambda paf: 0.25 <= paf <= 0.50),
    ("C", lambda paf: paf < 0.25),
)
_CONVERSION_TIER_KEYS = ("A", "B", "C")
# Structural guard target: the 16 convertible plants' nameplate MUST sum
# to exactly this (step 0 confirmed 571 + 3,215 + 2,696). A mismatch
# means the CSV changed -- investigate which plants moved, do not absorb.
_CONVERSION_NAMEPLATE_TOTAL_MW = 6482.0


def gas_conversion_tiers(csv_path=None):
    """
    Derive the three OCGT-to-CCGT conversion tiers from
    data/capacity/raw/gas_plant_availability.csv (Phase 2.8, Step 1).

    Per plant: mean PAF = mean(paf_2024_pct, paf_2025q4_pct,
    paf_2026q1_pct) / 100. Plants are banded by _CONVERSION_TIER_BANDS.

    Returns
    -------
    dict[str, dict]  keyed "A" / "B" / "C", each:
        {"plants": list[str],
         "nameplate_mw": float,        # sum of installed_mw in the tier
         "available_mw": float,        # sum of installed_mw * mean_paf
         "paf": float}                 # available_mw / nameplate_mw
                                       # (nameplate-weighted mean PAF)

    Raises if the CSV is missing, a required column is absent, tier
    nameplate does not sum to _CONVERSION_NAMEPLATE_TOTAL_MW, or a plant
    lands in zero or more than one tier.
    """
    path = csv_path if csv_path is not None else _GAS_CONVERSION_CSV_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gas plant availability file not found: {path}")

    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "plant_nerc_name", "installed_mw", "turbine_type",
            "paf_2024_pct", "paf_2025q4_pct", "paf_2026q1_pct",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must contain columns {sorted(required)}; found "
                f"{reader.fieldnames}"
            )
        for row in reader:
            mean_paf = (
                float(row["paf_2024_pct"])
                + float(row["paf_2025q4_pct"])
                + float(row["paf_2026q1_pct"])
            ) / 300.0   # mean of three percentages -> fraction in [0, 1]
            rows.append({
                "name": row["plant_nerc_name"],
                "installed_mw": float(row["installed_mw"]),
                "mean_paf": mean_paf,
            })

    tiers = {}
    for tier, predicate in _CONVERSION_TIER_BANDS:
        members = [r for r in rows if predicate(r["mean_paf"])]
        nameplate = sum(r["installed_mw"] for r in members)
        available = sum(r["installed_mw"] * r["mean_paf"] for r in members)
        tiers[tier] = {
            "plants": [r["name"] for r in members],
            "nameplate_mw": nameplate,
            "available_mw": available,
            "paf": (available / nameplate) if nameplate > 0 else 0.0,
        }

    total_nameplate = sum(t["nameplate_mw"] for t in tiers.values())
    if abs(total_nameplate - _CONVERSION_NAMEPLATE_TOTAL_MW) > 1.0:
        raise ValueError(
            f"Conversion tiers: nameplate sums to {total_nameplate} MW, not "
            f"{_CONVERSION_NAMEPLATE_TOTAL_MW} MW. The CSV must represent the "
            "16 convertible plants (step 0 confirmed 571 / 3,215 / 2,696). "
            "If the CSV changed, investigate which plants moved -- this is "
            "not a rounding difference to silently absorb."
        )
    n_assigned = sum(len(t["plants"]) for t in tiers.values())
    if n_assigned != len(rows):
        raise ValueError(
            f"Conversion tiers: {len(rows)} plants in the CSV but "
            f"{n_assigned} tier assignments -- a plant matched zero or "
            "multiple bands. _CONVERSION_TIER_BANDS must partition [0, 1]."
        )
    return tiers


def conversion_scenarios() -> dict[str, dict]:
    """
    OCGT-to-CCGT conversion arm (Phase 2.8, Step 2). REQUIRED, no default
    -- see the _CONVERSION_REQUIRED sentinel above load_scenario(). Mirrors
    the gas_capex_case / ccgt_efficiency_case discipline: capex, capacity
    uplift and retrofit efficiency are declared judgement calls, not
    measured constants.

    conv_none reproduces pre-2.8 results exactly: build_model_sliced()
    builds NO conversion components when conversion_available is False, and
    build_model() (annual) raises on any available arm -- conversion
    dispatch is slice-resolved (Decision 3) and has no annual form.

    Conversion capex: DERIVED, not SOURCED. No public OCGT-to-CCGT
    retrofit capex figure exists. Method: greenfield CC-minus-SC
    increment (Gas Turbine World 2024: F-class 237MW simple cycle
    $713/kW installed; H-class combined cycle $950-1,084/kW) plus a
    brownfield premium. Ipieca (2022) states OCGT-to-CCGT retrofit
    poses brownfield challenges of complexity, cost and deferred
    production, and that such upgrades are UNCOMMON industry-wide --
    the qualitative source for the premium and a declared limitation.
    Retrofit efficiency 0.46 HHV: an OCGT not specified for
    bottoming-cycle exhaust conditions will not reach greenfield CCGT
    efficiency (ccgt_eff_central 0.5173). ASSUMED, bracketed 0.42-0.50,
    same 5% ambient derate discipline as the new-build arm.
    Capacity uplift 0.50: steam turbine contributes ~1/3 of combined
    cycle output. ASSUMED, bracketed 0.40-0.50.

    Fields per arm:
      available            : bool. False = conv_none regression arm.
      capex_per_kw          : USD per kW of OCGT NAMEPLATE converted
                             (Decision 2 -- the HRSG is sized for the GT
                             exhaust, so capex is on nameplate; the
                             benefit accrues only on available capacity).
      uplift               : bottoming-cycle capacity as a fraction of
                             converted OCGT nameplate (Decision 1).
      retrofit_efficiency  : HHV electrical efficiency of the converted
                             plant (Decision 3 -- distinct from gas_eta).
    """
    return {
        "conv_none": {
            "available": False,
            "capex_per_kw": None,
            "uplift": None,
            "retrofit_efficiency": None,
        },
        "conv_central": {
            "available": True,
            "capex_per_kw": 600.0,
            "uplift": 0.50,
            "retrofit_efficiency": 0.46,
        },
        "conv_low": {
            "available": True,
            "capex_per_kw": 850.0,
            "uplift": 0.40,
            "retrofit_efficiency": 0.42,
        },
        "conv_high": {
            "available": True,
            "capex_per_kw": 450.0,
            "uplift": 0.50,
            "retrofit_efficiency": 0.50,
        },
        # TASK 76 C.3: an EFFICIENCY-ONLY family. conv_low / conv_central / conv_high move capex, uplift AND
        # efficiency together (register L129), so their span is a conversion PROGRAMME's worth, not the
        # unsourced 0.46's. These two vary retrofit_efficiency alone, at the conv_central capex (600) and
        # uplift (0.50). The family is conv_eff_low / conv_central / conv_eff_high.
        "conv_eff_low": {
            "available": True,
            "capex_per_kw": 600.0,
            "uplift": 0.50,
            "retrofit_efficiency": 0.42,
        },
        "conv_eff_high": {
            "available": True,
            "capex_per_kw": 600.0,
            "uplift": 0.50,
            "retrofit_efficiency": 0.50,
        },
    }


def gas_vintage_retirement_series_mw(years, technical_life_years, csv_path=None):
    """
    Cumulative retired MW per model year, from a flat technical life
    applied plant-by-plant to data/capacity/raw/gas_plant_vintages.csv
    (2.12, Step 1). Consumed by gas_cap_rule (build_model() /
    build_model_sliced()) via scenario['gas_baseline_retirement_series_mw'].

    A plant retires in commissioning_year + technical_life_years. This
    schedule is LUMPY, not linear -- retirement mass concentrates in years
    several plants share (e.g. 2013's four NIPP plants), with zero-
    retirement years elsewhere -- which the pre-2.12 rate *
    (year - start + 1) form cannot represent. See
    docs/phase_2_12_vintage_retirement.md, Step 0.

    A retirement event that falls BEFORE `years[0]` is NOT counted: the
    plant is observably still operating (it is part of GAS_BASELINE_MW,
    itself measured from live 2024/2026 NERC data), so an assumed life
    that has already elapsed before the model horizon even starts is not
    evidence the plant is gone -- it is evidence the flat-life assumption
    undershoots that plant. This can still surface WITHIN the horizon at
    its most extreme: at technical_life_years=40, Delta_1 (commissioned
    1985) retires in year 2025 and Egbin_1 (commissioned 1986) in 2026 --
    both plants are currently operating. See
    docs/gas_retirement_provenance.md Section 4 for why this is reported
    as a limitation of the flat-life assumption, not silently patched
    over: rehabilitation (extends true life beyond nameplate vintage) and
    NIPP under-utilisation (extends calendar life because running hours,
    not years, drive thermal degradation) run in opposite directions and
    partly cancel, which is the stated reason gas_retire_vintage_30 is
    taken as central rather than a "corrected" life figure.

    Parameters
    ----------
    years : sequence of int
        Model year index (e.g. scenario['years'] / range(MODEL_START_YEAR,
        MODEL_END_YEAR+1)). The returned series is aligned to this,
        element-for-element.
    technical_life_years : int
        Assumed plant technical life (30 / 40 / 50 in the current arms).

    Returns
    -------
    list[float]
        Cumulative retired MW, aligned to `years`. Non-decreasing, capped
        at GAS_BASELINE_MW (guard: total retired can never exceed the
        fleet it retires from -- mirrors the existing
        min(retired, gas_baseline) cap in gas_cap_rule).
    """
    plants = _load_gas_plant_vintages(csv_path)

    retirement_by_year = {}
    for p in plants:
        retire_year = p["commissioning_year"] + technical_life_years
        retirement_by_year[retire_year] = (
            retirement_by_year.get(retire_year, 0.0) + p["installed_mw"]
        )

    series = []
    cumulative = 0.0
    for y in years:
        cumulative += retirement_by_year.get(int(y), 0.0)
        cumulative = min(cumulative, GAS_BASELINE_MW)
        series.append(cumulative)
    return series


def gas_retirement_scenarios() -> dict[str, dict]:
    """
    Gas baseline fleet retirement arm (2.11, Step 2; renamed/extended
    2.12, Step 2). Mirrors genset_case/peak_multiple_case/storage_cc_case/
    gas_availability_case discipline: REQUIRED, no default -- see the
    _GAS_RETIREMENT_REQUIRED sentinel above load_scenario().

    Two arm SHAPES:
      - FLAT    {"mw_per_year": float, "start_year": int | None} -- a
                constant annual retirement rate from start_year (2.11,
                unchanged). Feeds gas_baseline_retirement_mw_per_year /
                gas_baseline_retirement_start_year; gas_cap_rule applies
                its existing rate * (year - start + 1) formula.
      - VINTAGE {"technical_life_years": int} -- 2.12, Step 2. Feeds
                gas_baseline_retirement_series_mw (a per-model-year
                CUMULATIVE retired-MW list, via
                gas_vintage_retirement_series_mw() and
                data/capacity/raw/gas_plant_vintages.csv), which
                gas_cap_rule consumes directly instead of the flat
                formula -- see load_scenario() for how the two shapes are
                distinguished and resolved.

        gas_retire_none       (FLAT, 0.0, None): no baseline retirement --
            a DECLARED UPPER BOUND on the surviving fleet, not a forecast.
            start_year is None because there is no retirement event to
            date from. UNCHANGED from 2.11.
        gas_retire_proxy      (FLAT, 680.0, 2035): the pre-2.11
            "brownfield retirement proxy" -- unsourced, retained as an
            explicit named arm so every pre-2.11 result stays attributable
            rather than orphaned. SOURCE: [PENDING]. UNCHANGED from 2.11.
        gas_retire_vintage_30 (VINTAGE, 30) [CENTRAL]: SOURCE: ASSUMED --
            analyst vintage estimate, not traced to a Nigerian source. See
            docs/gas_retirement_provenance.md and
            docs/limitations_register.md L8. Central not because 30 years
            is the correct technical life, but because two opposite
            errors partly cancel at that figure: rehabilitation (Delta,
            Egbin) extends true life beyond nameplate vintage, while NIPP
            under-utilisation extends calendar life (running hours, not
            years, drive thermal degradation) -- see the provenance memo
            Section 4.
        gas_retire_vintage_40 (VINTAGE, 40): SOURCE: ASSUMED -- analyst
            vintage estimate, not traced to a Nigerian source. See
            docs/gas_retirement_provenance.md and
            docs/limitations_register.md L8. Retires Delta_1 in 2025 and
            Egbin_1 in 2026 -- BOTH CURRENTLY OPERATING -- because a flat
            life measured from original commissioning misrepresents a
            rehabilitated fleet. Reported as a limitation, not corrected
            here: see gas_vintage_retirement_series_mw()'s docstring and
            docs/gas_retirement_provenance.md Section 4.
        gas_retire_vintage_50 (VINTAGE, 50): SOURCE: ASSUMED -- analyst
            vintage estimate, not traced to a Nigerian source. See
            docs/gas_retirement_provenance.md and
            docs/limitations_register.md L8. Upper bound on plausible
            fleet survival (short of gas_retire_none itself).

    gas_retire_sourced was REMOVED in 2.12, Step 2 (not deprecated-and-
    kept, not aliased) -- it promised a schedule "traced to" commissioning
    dates that was never sourced, and the replacement is explicitly an
    ANALYST ESTIMATE, not a traced one. A parameter whose name asserts
    provenance it does not have is the exact failure mode this project has
    already corrected twice (gas_availability_factor, 2.10; the retirement
    proxy itself, 2.11). load_scenario() raises a specific, named error if
    'gas_retire_sourced' is passed, rather than letting it fall through to
    the generic "Unknown gas_retirement_case" message -- see
    _GAS_RETIRE_SOURCED_REMOVED / load_scenario().
    """
    return {
        "gas_retire_none":       {"mw_per_year": 0.0, "start_year": None},
        "gas_retire_proxy":      {"mw_per_year": 680.0, "start_year": 2035},
        "gas_retire_vintage_30": {"technical_life_years": 30},
        "gas_retire_vintage_40": {"technical_life_years": 40},
        "gas_retire_vintage_50": {"technical_life_years": 50},
    }


# gas_build_case (2.9, Step 2). Mirrors genset_case/peak_multiple_case/
# storage_cc_case discipline: REQUIRED, no default -- see the
# _GAS_BUILD_REQUIRED sentinel below.
#
#   gas_ndc_17gw      : cumulative gas capacity capped on a trajectory
#                       reaching 17 GW by 2035 [Nigeria NDC 3.0, Sept 2025].
#                       UNBLOCKED (Phase 2.9 addendum): the trajectory is
#                       the NIRP-derived NDC path (finding P2-6), on the
#                       nameplate basis m.gas_capacity_mw uses -- see
#                       gas_ndc_17gw_trajectory_mw() and
#                       gas_new_capacity_cap_constraint in both builders.
#                       Intended CENTRAL. DERIVED, not stated by NIRP.
#   gas_unconstrained : no cap; the LP builds whatever is cost-optimal.
#   gas_none          : gas_add fixed at 0 in both builders. Reproduces
#                       every pre-2.9 result; kept as an explicit, named arm
#                       so those runs stay attributable rather than orphaned.
GAS_BUILD_CASE_CHOICES = ("gas_ndc_17gw", "gas_unconstrained", "gas_none")


def gas_ndc_17gw_trajectory_mw(years) -> list[float]:
    """
    Cumulative gas capacity cap (MW, NAMEPLATE), aligned to the model's
    year index, for the gas_ndc_17gw arm (2.9, Step 2).

    SOURCE: DERIVED from NIRP 2024 Tables 19 and 22 as
    generation / (8760 x capacity factor) -- NIRP states generation and
    capacity factor, not installed MW. Same method as gas_capex_central
    (P2-7). Anchor years, derated basis (~0.77 of nameplate):
    2030 9,188 / 2035 17,027 / 2041 20,945 / 2045 18,408 MW.
    CROSS-VALIDATION: the 2035 value lands on NDC 3.0's stated 17 GW
    gas target to three significant figures, so the NDC commitment and
    NIRP's modelled path are the same number, not two independent
    sources (finding P2-6).
    INTERPOLATION: linear between anchors, held at gas_baseline_mw
    before 2030, held flat after 2045. DECLARED, not sourced -- NIRP
    states the anchor years only.

    BASIS (Phase 2.9 addendum): gas_new_capacity_cap_constraint in BOTH
    builders caps m.gas_capacity_mw, which is NAMEPLATE -- gas_baseline_mw
    is NERC's installed-MW figure (10,947), and gas_availability_factor
    derates it SEPARATELY for the energy ceiling. The task's anchors are
    on a derated basis (P2-6, "~0.77 of nameplate"), so they are divided
    by 0.77 here to put the cap on m.gas_capacity_mw's own basis; the
    "hold at gas_baseline_mw before 2030" instruction is itself a
    nameplate value, confirming the cap's basis. Nameplate anchors:
    2030 11,932 / 2035 22,113 / 2041 27,201 / 2045 23,906 MW -- 2035 is
    ~2x the 10,947 MW baseline fleet, matching NDC 3.0's own "roughly
    double current levels" description of its gas target (P2-5), which
    the un-converted 17,027 (1.55x) does not. The 0.77 derate is
    DECLARED (P2-6), not separately sourced.
    """
    _DERATE = 0.77
    _DERATED_ANCHORS_MW = {
        2030: 9188.0,
        2035: 17027.0,
        2041: 20945.0,
        2045: 18408.0,
    }
    anchors = {y: v / _DERATE for y, v in _DERATED_ANCHORS_MW.items()}
    y_lo = min(anchors)   # 2030
    y_hi = max(anchors)   # 2045

    out = []
    for y in years:
        y = int(y)
        if y < y_lo:
            out.append(float(GAS_BASELINE_MW))
        elif y >= y_hi:
            out.append(anchors[y_hi])
        elif y in anchors:
            out.append(anchors[y])
        else:
            lo = max(a for a in anchors if a < y)
            hi = min(a for a in anchors if a > y)
            frac = (y - lo) / (hi - lo)
            out.append(anchors[lo] + frac * (anchors[hi] - anchors[lo]))
    return out


# ============================================================
# Hydro trajectory arms (correction plan 1.5)
# ============================================================
# Pinned NERC 2024 anchor pair. NOT changed by any hydro_case arm --
# capacity_credit["hydro"] is derived from these two in load_scenario()
# and every arm moves capacity proportionally with energy so that
# fraction stays invariant to the energy level (see phase_1_5_hydro.md).
_HYDRO_ANCHOR_TWH = 11.47      # NERC 2024: 30.92% of 37,093.70 GWh
_HYDRO_ANCHOR_MW = 2678.0      # NERC: Kainji+Jebba+Shiroro+Zungeru+Dadin-Kowa
# Implied capacity factor, held fixed across every arm and year:
# 11.47e6 / (2678 * 8760) = 0.4889.
_HYDRO_CF = _HYDRO_ANCHOR_TWH * 1e6 / (_HYDRO_ANCHOR_MW * 8760.0)

# Exploitable-hydro resource ceiling (Phase 1.5 validity guards). Hydro is
# EXOGENOUS, so this is a VALIDITY GUARD raising at load_scenario() time --
# a resource ceiling as an LP constraint would do nothing.
# SOURCE: Energy Commission of Nigeria -- 14,120 MW exploitable
# large-scale hydro + ~3,500 MW small hydro = 17,620 MW.
# NOT the same as NIRP's ~21 GW, which is a MODELLING ceiling
# (1,978 existing + 700 committed Zungeru + 18,600 candidate),
# built from site-specific plus GENERIC hydro capped at 15 GW
# nationally, with zonal shares from 24 sites in JICA (2019).
# NIRP's own 2045 build of 20.6 GW is 98.1% of its ceiling and
# 116.9% of the ECN estimate (finding F-22).
# Competing published figures exist (~14,750 MW academic;
# UNIDO 2022 gives 3,500 MW theoretical small-hydro of which only
# 734.3 MW economically feasible undeveloped). ECN is used as the
# central bound; the disagreement is itself declared.
# `hydro_ndc` at 10,400 MW is 59.0% of this ceiling and passes.
_HYDRO_EXPLOITABLE_MW = 17_620.0

# Build-rate WARN threshold (Phase 1.5, step 2). DECLARED, not sourced.
# One Zungeru-equivalent per year: Zungeru is 700 MW and took ~7 yr
# FID-to-commissioning; large hydro generally runs 8-12 yr. A sustained
# national build faster than one Zungeru a year has no precedent and
# nothing at that scale is currently under construction. WARN, never
# raise -- the arm is a declared counterfactual and must stay runnable,
# and the model has NO lead-time constraint for ANY technology, so the
# same limitation applies to solar and gas (finding F-23).
_HYDRO_BUILD_RATE_WARN_MW_PER_YR = 700.0
_ZUNGERU_MW = 700.0


def hydro_case_scenarios() -> tuple[str, ...]:
    """
    Names of the exogenous hydro trajectory arms (correction plan 1.5).
    REQUIRED, no default -- see the _HYDRO_REQUIRED sentinel above
    load_scenario(). All three keep hydro OUT of the optimiser.
    """
    return ("hydro_flat", "hydro_committed", "hydro_ndc")


def hydro_trajectory(hydro_case, years):
    """
    Exogenous annual hydro ENERGY series (TWh_e/year) and the matching
    installed-CAPACITY series (MW), both aligned to the model year index,
    for the hydro_case arm (correction plan 1.5). Hydro is NEVER a
    decision variable -- both series are fixed data.

    CAPACITY DERIVED FROM ENERGY at the fixed anchor capacity factor
    _HYDRO_CF (= 11.47 TWh / (2,678 MW x 8760 h) = 0.489):
        hydro_capacity_mw[t] = 2,678 x hydro_energy_twh[t] / 11.47
    so capacity and energy never drift and capacity_credit["hydro"]
    (a dry-season-output-per-installed-MW fraction) stays invariant to
    the energy level -- load_scenario() does not re-derive it per arm.
    This DELIBERATELY conflates capacity growth with UTILISATION growth:
    under hydro_committed / hydro_ndc the extra energy is partly Zungeru
    ramping up on already-installed plant (its 700 MW is already inside
    the 2,678 MW anchor), not purely new capacity. The fixed-CF
    derivation is the price of keeping capacity_credit["hydro"] coherent
    without touching it. DECLARED.

    FIXED-CF DIRECTION OF BIAS (Phase 1.5 validity guards, step 3):
    uprating or better-utilising EXISTING plant raises installed MW
    without raising available flow, so a hydro system's real capacity
    factor FALLS as its capacity grows. Holding CF fixed at the
    2,678 MW anchor's 0.4889 therefore OVERSTATES energy per MW at
    higher capacity -- it biases the high-hydro arms (hydro_ndc
    especially) toward MORE energy than the resource actually supports.
    The direction is stated, NOT corrected: a declining-CF derivation
    would make capacity no longer proportional to energy and break
    capacity_credit["hydro"]'s invariance to the energy level (see
    step 1 of docs/phase_1_5_hydro.md).

    VALIDITY GUARDS (Phase 1.5): raises if any year's derived
    hydro_capacity_mw exceeds _HYDRO_EXPLOITABLE_MW (ECN's 17,620 MW
    exploitable-hydro estimate); WARNS (never raises) if the per-year
    capacity build rate exceeds _HYDRO_BUILD_RATE_WARN_MW_PER_YR
    (one Zungeru-equivalent, 700 MW/yr).

    Arms:
      hydro_flat      SOURCED level 11.47 TWh_e (NERC 2024, 30.92% of
                      37,093.70 GWh), held flat the whole horizon.
                      REGRESSION ARM -- reproduces every pre-1.5 result
                      exactly (energy 11.47, capacity 2,678 in every year).
      hydro_committed SOURCED 2025 figure ~12.8 TWh_e (Zungeru's 700 MW
                      plus minor committed additions, correction plan
                      1.5), held flat thereafter. "12.8" is SOURCED;
                      "held flat after 2025" is DECLARED.
      hydro_ndc       Toward the NDC's SOURCED 10,400 MW by 2030
                      (correction plan 1.5) -- its energy equivalent at
                      _HYDRO_CF, ~44.5 TWh_e -- linear from the 2025
                      committed level, then held flat 2031 onward. The
                      10,400 MW 2030 target is SOURCED; the linear
                      interpolation SHAPE and the post-2030 flat treatment
                      are DECLARED, not sourced.
    """
    yrs = [int(y) for y in years]

    if hydro_case == "hydro_flat":
        energy = {y: _HYDRO_ANCHOR_TWH for y in yrs}
    elif hydro_case == "hydro_committed":
        energy = {y: 12.8 for y in yrs}
    elif hydro_case == "hydro_ndc":
        _e_2025 = 12.8                                    # committed 2025 level
        # 10,400 MW at _HYDRO_CF, algebraically 11.47 * 10400 / 2678:
        _e_2030 = _HYDRO_ANCHOR_TWH * 10400.0 / _HYDRO_ANCHOR_MW
        energy = {}
        for y in yrs:
            if y <= 2025:
                energy[y] = _e_2025
            elif y >= 2030:
                energy[y] = _e_2030
            else:
                frac = (y - 2025) / (2030 - 2025)
                energy[y] = _e_2025 + frac * (_e_2030 - _e_2025)
    else:
        raise ValueError(f"Unknown hydro_case: {hydro_case}")

    energy_series = [float(energy[y]) for y in yrs]
    capacity_series = [
        _HYDRO_ANCHOR_MW * e / _HYDRO_ANCHOR_TWH for e in energy_series
    ]

    # --- Validity guard 1: resource ceiling (RAISE) ---
    _peak_mw = max(capacity_series)
    if _peak_mw > _HYDRO_EXPLOITABLE_MW:
        _peak_yr = yrs[capacity_series.index(_peak_mw)]
        raise ValueError(
            f"hydro_case='{hydro_case}': derived installed hydro reaches "
            f"{_peak_mw:,.0f} MW in {_peak_yr}, exceeding the exploitable-hydro "
            f"ceiling of {_HYDRO_EXPLOITABLE_MW:,.0f} MW (Energy Commission of "
            f"Nigeria: 14,120 MW large + ~3,500 MW small). The trajectory "
            f"posits more hydro than the resource supports."
        )

    # --- Validity guard 2: build-rate diagnostic (WARN, never raise) ---
    _deltas = [
        capacity_series[i] - capacity_series[i - 1]
        for i in range(1, len(capacity_series))
    ]
    _peak_rate = max(_deltas) if _deltas else 0.0
    if _peak_rate > _HYDRO_BUILD_RATE_WARN_MW_PER_YR:
        warnings.warn(
            f"hydro_case='{hydro_case}': peak capacity build rate "
            f"{_peak_rate:,.0f} MW/yr = {_peak_rate / _ZUNGERU_MW:.1f} "
            f"Zungeru-equivalents/yr (Zungeru: 700 MW, ~7 yr build; large "
            f"hydro 8-12 yr FID-to-commissioning). Nothing at that scale is "
            f"currently under construction. This is a DECLARED counterfactual "
            f"-- the model has no lead-time constraint for any technology "
            f"(finding F-23), so it stays runnable.",
            stacklevel=2,
        )

    return energy_series, capacity_series


def gas_capex_scenarios() -> dict[str, float]:
    """
    New-build gas capacity unit cost (USD/MW) arm (2.13, Step 1). Mirrors
    genset_case/peak_multiple_case/storage_cc_case/gas_availability_case/
    gas_retirement_case discipline: REQUIRED, no default -- see the
    _GAS_CAPEX_REQUIRED sentinel above load_scenario(). Feeds
    scenario['gas_capex_per_mw'], read by build_model()/
    build_model_sliced() at every m.gas_add build/salvage/replacement-capex
    site (the replacement-capex mechanism's FORM is unchanged by this
    phase -- only the source of the numeric value it reads).

        gas_capex_low     (900,000 USD/MW): the pre-2.13 bare literal
            (2.9, Step 0.1 finding) -- never sourced, no CCGT/OCGT
            technology attribution. Retained as a DECLARED LOWER BOUND so
            every pre-2.13 result stays attributable rather than orphaned.
            Not a candidate central case.
        gas_capex_central (1,317,000 USD/MW) [CENTRAL]: DERIVED from NIRP
            2024 Tables 19, 22 and 23 -- NOT stated by NIRP as a per-MW
            figure. Derivation, reproducible from these numbers alone:
                1. Capacity implied per NIRP gas-build year = generation
                   (TWh) / (8760 h x capacity factor). NIRP states
                   generation and capacity factor; it does not state
                   installed MW directly.
                2. Capex per MW, per year = that year's stated gas
                   investment (USD) / that year's implied capacity
                   addition (MW). Year-by-year cluster across NIRP's
                   2030-2045 gas-build years: 877 / 1,092 / 1,152 / 1,162 /
                   1,207 / 1,393 USD/kW.
                3. gas_capex_central = cumulative gas investment,
                   2030-2045 (17.0 bn USD) / cumulative capacity added
                   over the same window (12,905 MW) = 1,317,000 USD/MW.
            Method CROSS-VALIDATED on solar: applying the identical
            generation/(8760 x CF) -> capex/MW method to NIRP's solar
            investment and generation returns 905 USD/kW (55 bn USD /
            60,756 MW) -- consistent with independent solar cost data,
            which is the check that the method itself is sound before
            trusting its gas output. NIRP does NOT state OCGT vs CCGT for
            its gas build; this figure is technology-blind by necessity of
            the source, not by omission here.
        gas_capex_high    (1,400,000 USD/MW): the upper value of the same
            877-1,393 USD/kW year-by-year cluster (Step 2 above), taken as
            an explicit upper bound rather than the derived central value.

    All three values are given, not computed here from raw NIRP tables --
    this function records the arm and the reproducible arithmetic behind
    gas_capex_central/gas_capex_high; it does not re-derive them from a
    NIRP source file (none is checked into this repo).
    """
    return {
        "gas_capex_low":     900_000,
        "gas_capex_central": 1_317_000,
        "gas_capex_high":    1_400_000,
    }


def ccgt_efficiency_scenarios() -> dict[str, float]:
    """
    New-build CCGT electrical efficiency (HHV basis) arm (2.13, Step 2).
    Mirrors genset_case/peak_multiple_case/storage_cc_case/
    gas_availability_case/gas_retirement_case/gas_capex_case discipline:
    REQUIRED, no default -- see the _CCGT_EFFICIENCY_REQUIRED sentinel
    above load_scenario(). Sets scenario['gas_new_efficiency'] directly
    (2.13, Step 2 -- gas_new_efficiency is no longer a free
    load_scenario() kwarg; see load_scenario()'s docstring), consumed at
    its one functional use, m.gas_exempt_fuel_allowance_twh_th
    (new_gas_deliverability='exempt' only) in both builders. NOT used for
    dispatch OPEX or emissions -- those remain on the shared, blended
    fleet gas_eta (0.287) throughout (2.9 does not split dispatch by
    vintage).

    COMMON BASIS (TASK 74): all labelled HHV (no LHV->HHV conversion). Only
    these values are documented as net of auxiliary load at 60 F / 60% RH
    (S&L Table 5-1); gas_eta (realised fleet, assumed 1,036 Btu/scf) and the
    converted stream are NOT established on any common plant-gate or ambient
    basis, so cross-ratios are bounds, not points: docs/efficiency_common_
    basis.md, parameter_provenance rows 23-26.

        ccgt_eff_iso     (0.5445): net HHV efficiency at 60 F / 60% RH, no
            further derate. Upper bound. SOURCE: EIA / Sargent & Lundy,
            SL-018001 Rev A (6 Dec 2023), Table 5-1, Case 5 (CC 2x2x1,
            H-class, 1,227 MW net): net heat rate 6,266 Btu/kWh HHV; NOT
            the NREL ATB. efficiency = 3,412 / 6,266 = 0.5445 (3,412
            Btu/kWh = 1 kWh in HHV-consistent units).
        ccgt_eff_central (0.5173) [CENTRAL]: S&L efficiency with a 5%
            Nigerian ambient derate. SOURCE: S&L as above for the base
            figure; the 5% derate is a DECLARED ASSUMPTION, NOT from S&L
            -- S&L quotes 60 F / 60% RH at a generic US site (not named
            ISO), against a Nigerian ambient of roughly 27-35 C.
            Gas turbine output and efficiency both fall with rising inlet
            temperature.
        ccgt_eff_low     (0.5010): S&L efficiency with a high (8%)
            ambient derate. Lower bound. SOURCE: as ccgt_eff_central;
            derate is a declared assumption, not from S&L.

    The ambient derate is SEPARATE from and ADDITIONAL to the HHV/LHV
    question above -- one is a unit-convention check, the other a
    site-condition adjustment; they do not offset each other.

    LEVER SIZE: ccgt_eff_central / gas_eta = 0.5173 / 0.287 = 1.80 -- one
    new CCGT MWh consumes gas_eta / ccgt_eff_central = 55.5% as much fuel
    as the existing fleet average, i.e. roughly 44% LESS gas per MWh. This
    is a large lever specifically because m.gas_balance (the fuel budget)
    is a binding deliverability constraint in most scenarios: new CCGT
    capacity does not just add MW, it lets the SAME scarce fuel envelope
    produce materially more electricity.
    """
    return {
        "ccgt_eff_iso":     0.5445,
        "ccgt_eff_central": 0.5173,
        "ccgt_eff_low":     0.5010,
    }


def capital_recovery_scenarios() -> dict[str, dict[str, float]]:
    """
    Per-tranche capital recovery terms (F1, Step 2) -- rate AND tenor,
    separately for the concessional and commercial tranche. REQUIRED, no
    default -- see the _CAPITAL_RECOVERY_REQUIRED sentinel above
    load_scenario(). Feeds the CRF() capital charge in build_model()/
    build_model_sliced(): capital_recovery_period is a NEW parameter,
    DISTINCT from asset_lifetimes() (technical service life) -- do not
    conflate the two.

    RATES BELOW ARE NOMINAL USD. Deflated ONCE, at the point where each
    rate enters CRF(), using scenario['us_inflation'] -- NOT here. This
    function returns the nominal figures unchanged; see build_model()'s
    CRF call site for the deflation step and its comment.

    r_conc / L_conc : SOURCED. World Bank IDA Terms, USD single-currency
      blend. `cr_central` uses IDA21 THROUGHOUT (rate-coherence task,
      Parts 2 and 3; register L53, superseding this arm's earlier IDA20
      rate AND tenor): 2.12% effective 2026-01-01 (2.13% effective
      2026-07-01, band 0.0212-0.0213), 25yr maturity, service charge
      set to zero on the blend term. DARES is modelled at current IDA
      terms (IDA21, 2.12%/25yr) rather than its IDA20 origination
      vintage (2.80%/30yr), because the model evaluates forward
      pathways rather than repricing an existing commitment. The IDA20
      rate AND tenor are retained in the register as the superseded
      values. Part 2 moved the rate only, leaving a rate/tenor hybrid
      (IDA21 rate on IDA20's 30yr maturity) that priced concessional
      capital CHEAPER than either real vintage -- CRF at 0.1176% real
      is 0.033943 over 30yr (-9.6% against the IDA20 baseline) but
      0.040616 over 25yr (+8.2%), so the hybrid could not stand and
      L_conc was corrected to 25yr in the same task. `cr_ida21` below
      is now NEARLY IDENTICAL to `cr_central` -- both 2.12-2.13%/25yr,
      differing only by which of IDA21's two 2026 effective dates each
      was set from (a 0.127% relative difference in CRF); NOT deleted,
      since a same-vintage/different-effective-date pair is a real,
      if narrow, distinction, not true duplication.
    r_comm : SOURCED BASE + ASSUMED UPLIFT. IEA Cost of Capital
      Observatory reports 8.5-9% WACC for USD-denominated clean power in
      Kenya and Senegal. Nigeria is NOT covered. The 11% central applies
      a 2-2.5pp uplift for worse offtaker credit (NBET as sole
      state-owned offtaker). THE UPLIFT IS AN ANALYST ESTIMATE, NOT
      SOURCED. The 9% and 14% arms carry it.
    L_comm : SOURCED. Afrigreen Debt Impact Fund (C&I solar, Nigeria
      focus) lends 8-10yr. Nigerian commercial bank loans 3-7yr absent
      credit enhancement; no project-financed renewable has closed in
      Nigeria. Continental average tenor has fallen 12yr -> 5yr.

        cr_central   [CENTRAL, F1 task]: r_conc 2.12% (IDA21; was 2.80%
            IDA20, rate-coherence task Part 2, register L53),
            L_conc 25 yr (IDA21; was 30 yr IDA20-legacy tenor,
            corrected in the same task -- see the r_conc/L_conc note
            above); r_comm 11%, L_comm 10 yr.
        cr_ida21: r_conc 2.13%, L_conc 25 yr; r_comm 11%, L_comm 10 yr.
        cr_comm_low: r_conc 2.80% (IDA20, unchanged by Part 2 -- only
            cr_central's rate moved), L_conc 30 yr; r_comm 9%,
            L_comm 15 yr.
        cr_comm_high: r_conc 2.80% (IDA20, unchanged by Part 2), L_conc
            30 yr; r_comm 14%, L_comm 7 yr.
    """
    return {
        "cr_central":   {"r_conc": 0.0212, "L_conc": 25, "r_comm": 0.11, "L_comm": 10},
        "cr_ida21":     {"r_conc": 0.0213, "L_conc": 25, "r_comm": 0.11, "L_comm": 10},
        "cr_comm_low":  {"r_conc": 0.0280, "L_conc": 30, "r_comm": 0.09, "L_comm": 15},
        "cr_comm_high": {"r_conc": 0.0280, "L_conc": 30, "r_comm": 0.14, "L_comm":  7},
    }


def storage_capex_scenarios() -> dict[str, dict]:
    """
    Storage capital-cost trajectory (TASK 52 section 2.1; register L90).

      flat            the 2025 Storage_low value (252,000 USD/MWh) at every
                      vintage and rebuild; every result before TASK 53. Now a
                      declared sensitivity, not the default.
      declining_nrel  THE DEFAULT since TASK 53. The Storage_low series in
                      data/cost/processed/storage_capex.csv (252,000 in 2025
                      to 146,000 in 2045, held flat after 2045 by
                      io._extend_capex_series), indexed by the year of the
                      build or rebuild as solar capex is.
    """
    return {"flat": {}, "declining_nrel": {}}


def capital_window_scenarios() -> dict[str, dict]:
    """
    Timing convention of the F1 per-tranche capital charge (TASK 51 M1).

    The charge on a vintage v is  CRF(r, L_eff) x W(v) x I , with
    W(v) = sum of m.DF[t] over the vintage's recovery window and CRF the
    ordinary-annuity factor r(1+r)^L / ((1+r)^L - 1). TASK 50 V1 found the
    window starts in the build year (t = v ... v+L_eff-1) against an
    ordinary-annuity CRF, so PV/I = 1 + r_s = 1.08 at r_f = r_s. Two
    conventions are internally consistent; the code matches neither:

      advance  the behaviour of every result produced before TASK 53 and no
               longer the default. window_offset 0, CRF as now. Reproduces
               every pre-epoch result bit-identically (kept as an arm).
      arrears  THE DEFAULT since TASK 53 (decided on principle: an annuity
               sized in arrears must be paid in arrears). Standard loan
               amortisation: first instalment one year after the build.
               window_offset 1 (t = v+1 ... v+L_eff), CRF as now.
               Removes the (1 + r_s) = 8% factor from ALL capital uniformly.
      due_crf  annuity-due: window as 'advance', CRF multiplied by
               1/(1 + r) at the rate the CRF was computed at (real
               concessional / real commercial). Removes (1 + r_f): about
               0.12% on concessional, 8.8% on commercial, so it moves the
               concessional-to-commercial ratio.

    window_offset: shift of the first instalment from the build year.
    crf_due: divide each CRF by (1 + its own real rate).

    Applies to _df_window_conc, _df_window_comm, the conversion charge
    (_df_window_comm["gas"][0]) and the rebuild charges that reuse them.
    Nothing else changes. This function does not migrate any runner.
    """
    return {
        "advance": {"window_offset": 0, "crf_due": False},
        "arrears": {"window_offset": 1, "crf_due": False},
        "due_crf": {"window_offset": 0, "crf_due": True},
    }


def dscr_scenarios() -> dict[str, float]:
    """
    DSCR coverage cushion arm (F1, Step 3). REPLACES the pre-F1
    required_margin fork entirely (builder 1.0 fallback, scenarios.py
    unconditional 1.10, run_gas_regime_ndc_matrix's 1.10 kwarg -- all
    three removed) -- this is a REPLACEMENT, not a reconciliation of the
    old values. See docs/required_margin_inventory.md for the fork this
    supersedes.

    A SINGLE cushion, constant across financing regimes: the pre-F1
    1.05 (concessional) / 1.10 (commercial) split is RETIRED as a
    double-count now that the concessional/commercial difference is
    carried explicitly by the CRF rates (capital_recovery_case). Feeds
    scenario['required_margin'], which stays a revenue divisor in the
    eaas_bankability / eaas_subsidy_limit constraints and does NOT enter
    the objective (unchanged mechanism -- see
    docs/required_margin_inventory.md Section 2).

    required_margin is a debt-SIZING ratio -- this model's analogue of
    Yescombe's minimum Annual Debt-Service Cover Ratio (ADSCR), which
    divides net operating cash flow to size how much debt (here,
    revenue-financed capacity) a project can support. It is NOT a
    lock-up (Distribution-Stop) or default-covenant ratio -- Yescombe
    treats those as separate, lower thresholds applied off the same
    sizing figure after financial close (worked example: 1.35 sizing
    -> 1.2 lock-up -> 1.1 default; Yescombe 2014, Sec.14.4.3 p.376,
    Sec.14.12 p.397). This model carries only the sizing role; it has
    no separate lock-up/default mechanism.

    Source: Yescombe, E.R. (2014). Principles of Project Finance, 2nd
    edn. Waltham, MA: Academic Press (Elsevier). ISBN
    978-0-12-415755-2, Sec.12.3.1 p.323 -- sector-differentiated
    minimum-ADSCR table: 1.20 (Accommodation-based Contract) / 1.25
    (process-plant project with an Offtake Contract) / 1.50 (natural
    resources, no offtake) / 1.75 (transport Concession) / 2.00
    (merchant power, no offtake). No renewable-specific figure is
    given; by the book's own taxonomy (Sec.2.5.1 p.14) a solar IPP on
    a long-term PPA is a "process-plant project with an Offtake
    Contract" -- the 1.25 row. See docs/parameter_provenance.md row 13
    and docs/task_log_sourcing_A1.md for the full trace, including the
    retraction of an earlier (slide-deck-sourced) claim that
    DFIs/MDFIs accept lower cover ratios -- checked directly against
    this book (Sec.15.8, Chapter 16) and NOT found there.

        dscr_regression : 1.10 -- reproduces the pre-F1 commercial-arm
            value exactly, for the F1 regression check (paired with
            capital_treatment_case='pre_f1_overnight'). Numerically
            coincides with Yescombe's own Default Ratio worked example
            (1.1:1, Sec.14.12 p.397) but is NOT derived from it --
            flagged as coincidence, not design. Not a sourced central
            case in its own right.
        dscr_central    : 1.30 -- 0.05 ABOVE the book's own 1.25
            process-plant/Offtake-Contract figure (the correct sector
            match for a PPA-backed solar IPP). Justified only by the
            book's unquantified qualifier that "higher cover levels
            would be required for... a country with a poor credit
            risk"; the 0.05 premium is a declared Nigeria adjustment,
            not itself a sourced figure.
        dscr_low        : 1.20 -- matches the book's
            Accommodation-based-Contract minimum, NOT the applicable
            process-plant figure (1.25); a declared conservative lower
            bound, not a sector match.
        dscr_high       : 1.40 -- a further stress case 0.10 above
            dscr_central; ASSUMED, no source beyond internal
            sensitivity design.
    """
    return {
        "dscr_regression": 1.10,
        "dscr_central":    1.30,
        "dscr_low":        1.20,
        "dscr_high":       1.40,
    }


def concessional_envelope_scenarios() -> dict[str, float]:
    """
    F1's own concessional envelope arm (envelope resolution; see
    docs/f1_implementation.md, "Envelope resolution"). Undiscounted
    commitment-stock ceiling on
    sum_v(solar_public_conc_vol[v] + solar_eaas_conc_vol[v]
          + storage_conc_vol[v])
    in build_model()/build_model_sliced(). Gas is EXCLUDED from both the
    envelope sum and concessional eligibility: DARES is a
    distributed-renewables facility, and IDA credit for a CCGT is not a
    defensible claim (gas_conc_vol is fixed to 0.0 at the build site).

    env_dares / env_dares_plus -- classification (TASK 57; scheme SOURCED /
    DERIVED / ASSUMED, docs/parameter_provenance.md row 22): the 750M
    AMOUNT is SOURCED to the World Bank DARES PAD (PAD00037, PDF pp.1, 8,
    24); its USE here as an unconditional, fully capital-eligible
    concessional debt stock at IDA21 terms is ASSUMED; 1,050M, 5x and 10x
    are ASSUMED. Earlier wording, retained for history: STIPULATED, not
    SOURCED, within this
    repository: the figures come from the SAME bare dollar amounts already
    used by the pre-existing (now-retired) Level-2 scarcity constraint in
    scripts/04_run_fin2_tariff_sweep.py. External corroboration for the
    750M figure: World Bank DARES (Distributed Access through Renewable
    Energy Scale-up), $750M IDA credit approved December 2023. The
    1,050M figure has no external corroboration (ASSUMED).
    In that script these are labelled ONLY
    "750M USD facility" / "1050M USD facility" (module docstring) -- no
    URL, no programme document, no further citation exists anywhere in
    this repository. src.blended_finance.concessional_envelope_npv(),
    despite its name and its own docstring calling the input "NPV, USD",
    applies NO discounting -- it is a bare
    `total_facility_usd * domestic_deployable_fraction` (identity at
    fraction=1.0). There is no separate, more-discounted "NPV" figure and
    no separate, less-discounted "principal" figure recorded anywhere:
    the bare dollar amount IS the only figure that exists, so it is used
    here directly as the undiscounted principal (envelope resolution
    Decision 3) -- not backed out by inverting a discount factor, because
    none was ever applied to it.

    env_none -- 0.0. All-commercial floor, a DECLARED bound: not a claim
    about DARES, a deliberate zero-concessional-capital counterfactual.

    env_5x / env_10x -- DECLARED COUNTERFACTUALS, NOT sourced values. The
    DARES envelope (env_dares) is small relative to system capex, so it
    is expected to bind early and leave the marginal dollar commercial
    for most of the horizon. These multiplier arms establish what
    envelope scale would be required to change the pathway -- they do
    NOT claim DARES itself is 5x or 10x its stated size.
    """
    _env_dares = 750_000_000.0
    _env_dares_plus = 1_050_000_000.0
    return {
        "env_dares":      _env_dares,
        "env_dares_plus": _env_dares_plus,
        "env_none":       0.0,
        "env_5x":         5.0 * _env_dares,
        "env_10x":        10.0 * _env_dares,
    }


# capital_treatment_case (F1, Step 6). NOT one of the task's three
# "new required keys" (us_inflation / capital_recovery_case / dscr_case)
# -- added because Step 6 ("the live default objective must contain
# EXACTLY ONE of {F1, salvage, replacement}") has no way to be satisfied
# without an explicit, named selector between the two objective forms.
# Mirrors the SAME required/no-default discipline as every other arm in
# this file, for the same reason: an unstated choice here would silently
# pick one of two materially different objective functions. Judgement
# call, not part of the task's literal Step 1-3 list -- flagged in
# docs/f1_implementation.md.
CAPITAL_TREATMENT_CASE_CHOICES = ("pre_f1_overnight", "f1_tranche")

# TASK 51 M1: timing convention of the per-tranche capital charge. TASK 53:
# the default of load_scenario() is now 'arrears'; 'advance' (every result
# before TASK 53) and 'due_crf' remain available as arms. See
# capital_window_scenarios().
CAPITAL_WINDOW_CASE_CHOICES = ("advance", "arrears", "due_crf")
STORAGE_CAPEX_CASE_CHOICES = ("flat", "declining_nrel")

# bank_tenor_case (rate-coherence task, Part 1). Selects the basis for the
# bankability discount factor, m.remaining_npv_factor -- see
# resolve_remaining_npv_factor() (src/blended_finance.py) and
# docs/task_log_rate_coherence.md.
#
#   bank_tenor_comm   [NEW, DEFAULT]: the objective's OWN commercial
#       tranche (r_comm_nominal / L_comm, capital_recovery_scenarios()),
#       deflated at the build site by us_inflation. F_t truncates at
#       min(L_comm, T_last-t+1) -- a real tenor, not the model horizon.
#       Adopted because the bankability test and the objective were
#       pricing the same capital differently: objective 11% nominal /
#       8.824% real over 10 yr (cr_central); bankability 18%, flagged in
#       its own source comments as naira-basis, over the full horizon
#       with no tenor at all (register L17).
#   bank_tenor_legacy: the pre-correction blended-WACC factor
#       (resolve_private_rate() -- concessional/commercial blend from
#       blended_finance_scenarios()[blend_case]), summed to the model
#       horizon, undiscriminated by any tenor. Every result produced
#       before this task used this basis by construction, whether or not
#       it said so. Retained, not deleted -- see resolve_private_rate(),
#       blended_wacc(), blended_finance_scenarios() (all still live).
#
# Plain default, like new_gas_deliverability -- NOT one of the task's
# sentinel-guarded REQUIRED arms. Unlike that one, the default here is
# the NEWLY CORRECTED behaviour, not the pre-existing one: an unstated
# caller now gets the coherent basis, and must name bank_tenor_legacy
# explicitly to reproduce a pre-correction result.
BANK_TENOR_CASE_CHOICES = ("bank_tenor_comm", "bank_tenor_legacy")


def _solar_cumulative_cap(solar_build_case: str, years) -> list[float] | None:
    """
    Cumulative solar ADDITIONS cap (MW) aligned to the model's year index.
    Returns None when the arm is unconstrained.

    Years outside the NIRP series carry the nearest endpoint value; the model
    horizon should not extend beyond 2045 without extending the source series.
    """
    mult = solar_build_scenarios()[solar_build_case]
    if mult is None:
        return None
    nirp = nirp_solar_cumulative_mw()
    lo, hi = min(nirp), max(nirp)
        # Past the NIRP series, continue at its terminal growth rate. Clamping flat
    # would freeze deployment capability in the buffer years and could force
    # unserved energy there, distorting decisions inside the reporting window.
    g_term = nirp[hi] / nirp[hi - 1] - 1.0
    out = []
    for y in years:
        y = int(y)
        if y < lo:
            out.append(mult * nirp[lo])
        elif y <= hi:
            out.append(mult * nirp[y])
        else:
            out.append(mult * nirp[hi] * (1.0 + g_term) ** (y - hi))
    return out


# ============================================================
# SCENARIO CONSTRUCTOR
# ============================================================

# Sentinels distinguishing "not passed" from a real value -- same mechanism
# as _GENSET_REQUIRED in src/io.py's load_econ(). A real default (or None)
# cannot be used here: silently omitting these two would build a firm-
# capacity constraint against an unstated peak/reserve assumption with no
# error, warning, or visible signature in the output -- the exact failure
# mode genset_case was introduced to close off (see src/io.py docstring).
_PEAK_MULTIPLE_REQUIRED = object()
_RESERVE_MARGIN_REQUIRED = object()
# Same mechanism, same reason (2.6, phase_2_3b Step 2): storage's
# firm-capacity credit was a bare, unsourced 1.0 sitting above gas's 0.80 --
# a silent default here would keep crediting storage as the model's best
# firm-capacity resource on an undeclared assumption.
_STORAGE_CC_REQUIRED = object()
# Same mechanism, same reason (2.9, Step 2): m.gas_add moves from a Param
# fixed at 0 to a free Var in this phase. A silent default here would let a
# scenario build new gas capacity -- NDC 3.0's single largest capacity
# commitment -- without a named, traceable choice of how much is allowed.
_GAS_BUILD_REQUIRED = object()
# Same mechanism, same reason (2.10, Step 2): the gas fleet's availability
# figure feeds BOTH the annual energy ceiling (gas_availability_factor) and
# the peak-hour firm-capacity credit (capacity_credit['gas']) -- a silent
# default here (the old bare, unsourced 0.80) would keep both roles pinned
# to an unstated, ~2.7x-above-observed assumption with no error, warning,
# or visible signature in the output. See gas_availability_scenarios().
_GAS_AVAILABILITY_REQUIRED = object()
# Same mechanism, same reason (2.11, Step 2): gas_baseline_retirement_mw_per_year
# was a bare, unsourced 680.0 MW/yr ("brownfield retirement proxy") that drives
# the entire baseline gas fleet to zero by 2051 and now dominates the
# late-horizon result -- a silent default here would keep doing that on an
# unstated assumption. See gas_retirement_scenarios().
_GAS_RETIREMENT_REQUIRED = object()
# gas_retire_sourced was REMOVED in 2.12, Step 2 -- not deprecated-and-
# kept (unlike _DEPRECATED_DEMAND_LEVELS above, which stays selectable
# only to raise a specific message), simply gone from
# gas_retirement_scenarios()'s dict. Named here so load_scenario() can
# raise a specific "this name no longer exists, here is its replacement"
# error instead of the generic "Unknown gas_retirement_case" message.
_GAS_RETIRE_SOURCED_REMOVED = "gas_retire_sourced"
# Same mechanism, same reason (2.13, Step 1): gas_capex_per_mw was a bare,
# unsourced 900,000 USD/MW literal (2.9, Step 0.1 finding) that blocked
# gas_unconstrained and gas_ndc_17gw outright via an unconditional raise --
# a silent default here would either resurrect that block by omission or
# silently spend an unstated capex figure once the raise is removed. See
# gas_capex_scenarios().
_GAS_CAPEX_REQUIRED = object()
# Same mechanism, same reason (2.13, Step 2): gas_new_efficiency was a
# free load_scenario() kwarg with no sourced value behind it (2.9, Step
# 1/0.2) -- a silent default here would let the exempt fuel-equivalent
# allowance (new_gas_deliverability='exempt') size itself off an unstated
# CCGT efficiency. See ccgt_efficiency_scenarios().
_CCGT_EFFICIENCY_REQUIRED = object()
# Same mechanism, same reason (F1, Step 2): capital_recovery_case's rate
# AND tenor are new, declared judgement calls (see
# capital_recovery_scenarios()) -- a silent default would price every
# tranche's CRF() charge on an unstated recovery period.
_CAPITAL_RECOVERY_REQUIRED = object()
# Same mechanism, same reason (F1, Step 3): dscr_case REPLACES the old
# required_margin fork (builder 1.0 fallback, scenarios.py's unconditional
# 1.10, run_gas_regime_ndc_matrix's 1.10 kwarg) -- a silent default here
# would resurrect exactly the fork this arm was introduced to retire. See
# dscr_scenarios().
_DSCR_REQUIRED = object()
# Same mechanism, same reason (F1, Step 6): capital_treatment_case decides
# which of two materially different objective functions build_model()
# constructs -- a silent default would make that choice invisibly.
_CAPITAL_TREATMENT_REQUIRED = object()
# F1, Step 1: us_inflation has no named-arm alternatives in the task (a
# single required scalar, not a "_case" -- "Central 0.020", no
# us_inflation_low/high given). REQUIRED, no default anyway: a silent
# default would deflate every CRF rate on an unstated inflation
# assumption. See load_scenario()'s us_inflation parameter below.
_US_INFLATION_REQUIRED = object()
# Envelope resolution (docs/f1_implementation.md): concessional_envelope_case
# REPLACES the None-default concessional_envelope_usd left unconstrained by
# Steps 1-7. A silent default here would leave comm_vol[t] == 0 everywhere
# and 100% of every channel's build funded concessionally with nothing
# limiting it -- structurally impossible in reality and the exact
# degenerate solution the envelope exists to rule out. See
# concessional_envelope_scenarios().
_CONCESSIONAL_ENVELOPE_REQUIRED = object()
# Phase 2.8, Step 2: conversion_case decides whether OCGT-to-CCGT retrofit
# is in the choice set, and at what capex / uplift / efficiency. A silent
# default would either omit the only fuel-free route to more generation
# under a binding gas deliverability constraint (understating what is
# cost-optimal) or spend an unstated retrofit capex. See
# conversion_scenarios().
_CONVERSION_REQUIRED = object()
# Correction plan 1.5: hydro_case selects the exogenous hydro trajectory
# (flat / committed-only / NDC-aligned). A silent default would hold
# hydro flat at 11.47 TWh, leaving Zungeru's ramp unrepresented and hydro
# under-delivery untested -- and the NIRP 20.6 GW / 103.9 TWh divergence
# uncarried as a sensitivity. See hydro_case_scenarios() / hydro_trajectory().
_HYDRO_REQUIRED = object()
# task_om_implementation: om_case selects the fixed + variable O&M cost
# arm (NREL ATB 2024, data/cost/processed/om_cost.csv). A silent default
# would leave O&M unset with no error -- the exact failure mode that let
# STORAGE_OM_PER_MWH_YR sit in io.py for the life of the project without
# ever reaching either builder's objective (docs/task_storage_om.md).
# See om_scenarios().
_OM_REQUIRED = object()
# Revised tariff task (docs/tariff_arm.md, Steps 1-3): tariff_case REPLACES
# the defaulted solar_tariff_case, which wrote
# solar_service_tariff_usd_per_twh = 65,000,000 into every scenario dict
# this project has ever built -- a value never selected by any runner,
# never present in an output tag, and carrying a STALE-DOC warning on its
# own grid. Third instance of the silent-default failure mode after
# gas_capex (900,000 USD/MW) and the retirement proxy (680 MW/yr).
# REQUIRED, no default: the tariff parameterises m.eaas_bankability, which
# gates finding F-27 (EaaS worth 8.895 bn under NDC 3.0 conditional). See
# tariff_scenarios().
_TARIFF_REQUIRED = object()


def load_scenario(
    demand_level_case: str = "served",
    demand_case: str = "organic_central",
    land_case: str = "moderate",
    capital_case: str = "moderate",
    gas_deliverability_case: str = "baseline",
    solar_build_case: str = "nirp_trajectory",
    tariff_case=_TARIFF_REQUIRED,
    carbon_case: str = "no_policy",
    blend_case: str = "blended_central",
    peak_multiple_case=_PEAK_MULTIPLE_REQUIRED,
    reserve_margin=_RESERVE_MARGIN_REQUIRED,
    storage_cc_case=_STORAGE_CC_REQUIRED,
    gas_build_case=_GAS_BUILD_REQUIRED,
    gas_availability_case=_GAS_AVAILABILITY_REQUIRED,
    gas_retirement_case=_GAS_RETIREMENT_REQUIRED,
    gas_capex_case=_GAS_CAPEX_REQUIRED,
    ccgt_efficiency_case=_CCGT_EFFICIENCY_REQUIRED,
    capital_recovery_case=_CAPITAL_RECOVERY_REQUIRED,
    dscr_case=_DSCR_REQUIRED,
    us_inflation=_US_INFLATION_REQUIRED,
    capital_treatment_case=_CAPITAL_TREATMENT_REQUIRED,
    concessional_envelope_case=_CONCESSIONAL_ENVELOPE_REQUIRED,
    conversion_case=_CONVERSION_REQUIRED,
    hydro_case=_HYDRO_REQUIRED,
    om_case=_OM_REQUIRED,
    capital_window_case: str = "arrears",
    storage_capex_case: str = "declining_nrel",
    new_gas_deliverability: str = "constrained",
    gas_vintage_dispatch: bool = True,
    bank_tenor_case: str = "bank_tenor_comm",
    start_year: int = MODEL_START_YEAR,
    end_year: int = MODEL_END_YEAR,
) -> dict:
    """
    Build a scenario parameter dictionary consumed by the optimization model.

    Parameters
    ----------
    demand_level_case : {"served"} — the only accepted value. "latent_low" and
                        "latent_high" are retained in the registry for
                        provenance but raise (see validation below).
    demand_case       : key of demand_growth_scenarios() —
                        {"constrained_continuation", "organic_central",
                         "organic_high"}. NIRP's 7.7% benchmark is NOT here;
                        see demand_growth_benchmark().
    land_case         : key of land_scenarios()
    capital_case      : key of capital_envelope_scenarios()
    gas_deliverability_case : key of gas_deliverability_scenarios()
    solar_build_case  : key of solar_build_scenarios()
    tariff_case       : key of tariff_scenarios() -- REQUIRED, no default
                        (docs/tariff_arm.md). Sets
                        scenario['solar_service_tariff_usd_per_twh'], read
                        by m.eaas_bankability / m.eaas_subsidy_limit in
                        both builders. Replaces the defaulted
                        solar_tariff_case.
    carbon_case       : key of carbon_policy_scenarios()
    blend_case        : key of blended_finance_scenarios() (default central).
                        Sets the EaaS capital structure -> blended private rate.
    peak_multiple_case : key of peak_multiple_scenarios() -- REQUIRED, no
                        default (2.6, F0 Step 2.4, inverted design). One of
                        'peak_suppressed' / 'peak_unsuppressed' / 'peak_nirp' / 'peak_high'. Selects
                        a period_demand_factor (required) / season_demand_factor
                        (optional) SHAPE arm; PAR (par_block on the returned
                        scenario) is DERIVED from that shape, not supplied
                        independently -- there is no consistency check any
                        more because there is nothing left to check against.
                        par_block is a 4-hour BLOCK-AVERAGE ratio, not an
                        instantaneous peak ratio -- see
                        within_block_peaking_factor, read inside the builders.
    reserve_margin    : float or 'nirp_spinning' -- REQUIRED, no default
                        (2.6, F0 Step 2.3). A numeric fraction is used as-is;
                        the literal string 'nirp_spinning' derives a PER-YEAR
                        fraction as RESERVE_MARGIN_SPINNING_MW / peak_mw[t]
                        (NIRP 2024 Section 2.3, p.19 -- spinning reserve
                        requirement).
    storage_cc_case   : key of storage_cc_scenarios() -- REQUIRED, no
                        default (2.6, phase_2_3b Step 2, mirrors
                        genset_case/peak_multiple_case discipline). One of
                        'storage_cc_low' / 'storage_cc_central' /
                        'storage_cc_high'. Storage's firm-capacity credit
                        (capacity_credit['storage']) is a DECLARED
                        JUDGEMENT, not a measured constant -- see
                        storage_cc_scenarios().
    gas_build_case    : key of GAS_BUILD_CASE_CHOICES -- REQUIRED, no
                        default (2.9, Step 2, mirrors genset_case/
                        peak_multiple_case/storage_cc_case discipline). One
                        of 'gas_ndc_17gw' / 'gas_unconstrained' /
                        'gas_none'. Governs whether/how much new gas
                        capacity (m.gas_add, now a free Var) the LP may
                        build. 'gas_ndc_17gw' caps cumulative gas capacity
                        on the NIRP-derived NDC 3.0 trajectory (finding
                        P2-6), nameplate basis -- see
                        gas_ndc_17gw_trajectory_mw(); UNBLOCKED in the
                        Phase 2.9 addendum. gas_capex_per_mw / CCGT
                        efficiency are supplied by gas_capex_case /
                        ccgt_efficiency_case (2.13).
    gas_availability_case : key of gas_availability_scenarios() -- REQUIRED,
                        no default (2.10, Step 2, mirrors genset_case/
                        peak_multiple_case/storage_cc_case discipline). One
                        of 'gas_avail_observed' / 'gas_avail_central' /
                        'gas_avail_nameplate'. Feeds BOTH
                        gas_availability_factor (annual energy ceiling) and
                        capacity_credit['gas'] (peak-hour firm-capacity
                        credit) -- independently settable dict entries
                        (2.10, Step 3) initialized equal from this SAME arm
                        value. 'gas_avail_central' currently always raises:
                        SOURCE: [PENDING] -- see gas_availability_scenarios().
    gas_retirement_case : key of gas_retirement_scenarios() -- REQUIRED, no
                        default (2.11, Step 2, mirrors genset_case/
                        peak_multiple_case/storage_cc_case/
                        gas_availability_case discipline). One of
                        'gas_retire_none' / 'gas_retire_proxy' /
                        'gas_retire_vintage_30' ([CENTRAL]) /
                        'gas_retire_vintage_40' / 'gas_retire_vintage_50'
                        (2.12, Step 2). The first two are FLAT arms: set
                        gas_baseline_retirement_mw_per_year and
                        gas_baseline_retirement_start_year (start_year
                        settable PER ARM rather than fixed at 2035). The
                        three vintage arms are per-plant SCHEDULES: set
                        gas_baseline_retirement_series_mw (a per-model-year
                        cumulative MW list from
                        data/capacity/raw/gas_plant_vintages.csv) instead
                        -- SOURCE: ASSUMED, an analyst vintage estimate,
                        not traced to a Nigerian source (see
                        docs/gas_retirement_provenance.md and
                        docs/limitations_register.md L8).
                        'gas_retire_sourced' was REMOVED in 2.12 and raises
                        a specific error naming its replacement if passed
                        -- see gas_retirement_scenarios().
    gas_capex_case    : key of gas_capex_scenarios() -- REQUIRED, no
                        default (2.13, Step 1, mirrors genset_case/
                        peak_multiple_case/storage_cc_case/
                        gas_availability_case/gas_retirement_case
                        discipline). One of 'gas_capex_low' /
                        'gas_capex_central' ([CENTRAL]) / 'gas_capex_high'.
                        Sets scenario['gas_capex_per_mw'], read at every
                        m.gas_add build/salvage/replacement-capex site in
                        both builders. Replaces the 2.9 unconditional
                        raise that blocked gas_unconstrained/gas_ndc_17gw
                        on unsourced capex -- see gas_capex_scenarios()
                        for the full NIRP-derived provenance.
    ccgt_efficiency_case : key of ccgt_efficiency_scenarios() -- REQUIRED,
                        no default (2.13, Step 2, same discipline). One of
                        'ccgt_eff_iso' / 'ccgt_eff_central' ([CENTRAL]) /
                        'ccgt_eff_low'. Sets scenario['gas_new_efficiency']
                        directly -- gas_new_efficiency is NO LONGER a free
                        load_scenario() kwarg (2.13, Step 2); its one
                        functional use is sizing
                        m.gas_exempt_fuel_allowance_twh_th when
                        new_gas_deliverability='exempt'. See
                        ccgt_efficiency_scenarios() for the HHV-basis and
                        ambient-derate provenance.
    capital_recovery_case : key of capital_recovery_scenarios() -- REQUIRED,
                        no default (F1, Step 2). One of 'cr_central'
                        ([CENTRAL, this task]) / 'cr_ida21' / 'cr_comm_low'
                        / 'cr_comm_high'. Sets the concessional/commercial
                        tranche's rate AND tenor for the CRF() capital
                        charge -- NOMINAL USD, deflated once inside
                        build_model()/build_model_sliced() using
                        us_inflation, not here. Distinct from
                        asset_lifetimes() (technical service life) -- see
                        capital_recovery_scenarios().
    dscr_case         : key of dscr_scenarios() -- REQUIRED, no default
                        (F1, Step 3). One of 'dscr_regression' /
                        'dscr_central' / 'dscr_low' / 'dscr_high'. Sets
                        scenario['required_margin'] -- REPLACES the pre-F1
                        required_margin fork entirely (see
                        docs/required_margin_inventory.md); a single
                        cushion, constant across financing regimes.
    us_inflation      : float -- REQUIRED, no default (F1, Step 1). Central
                        0.020 (US Federal Reserve long-run target). Used
                        ONLY to deflate capital_recovery_case's nominal
                        rates at the point they enter CRF() -- does NOT
                        deflate social_discount_rate (0.08 is treated as
                        already real; see docs/f1_implementation.md Step
                        0b / Decision 4).
    capital_treatment_case : one of 'pre_f1_overnight' / 'f1_tranche' --
                        REQUIRED, no default. NOT one of the task's three
                        named "new required keys" -- added so the
                        objective's F1-vs-pre-F1 form is a traceable,
                        named choice rather than an implicit one (see
                        CAPITAL_TREATMENT_CASE_CHOICES and
                        docs/f1_implementation.md for the rationale).
                        'pre_f1_overnight' reproduces the pre-F1 objective
                        exactly (four overnight capex terms + salvage +
                        replacement); 'f1_tranche' replaces all six with
                        the per-tranche CRF annuity.
    gas_vintage_dispatch : bool, default True (TASK 74 Part D, register L123).
                        ON: capacity added in the model converts at
                        scenario['gas_new_efficiency'] (ccgt_eff_central
                        0.5173) and pre-existing capacity at gas_eta (0.287);
                        emissions follow the thermal draw. Effective only at
                        new_gas_deliverability='constrained' (the sliced
                        builder raises otherwise), so it is forced False
                        under 'exempt'. False reproduces the pre-TASK-74
                        blended dispatch exactly. It is a key arm: runner
                        output tags carry it (src/cli_arms.py).
    new_gas_deliverability : 'constrained' (default) or 'exempt' (2.9, Step
                        3). 'constrained': new gas capacity draws from the
                        SAME shared fuel budget (m.gas_balance) as the
                        existing fleet -- the Nigerian baseline, capacity
                        without fuel, not relaxed by assumption.  'exempt':
                        new capacity is assumed sited at gas supply or on
                        dedicated commercial contracts -- its own
                        capacity-and-efficiency-sized fuel-equivalent
                        allowance is added to the shared budget, so it is
                        not competing with the existing fleet for scarce
                        gas. Both are implemented; this default does not
                        choose between them for reporting purposes -- see
                        docs/phase_2_9_gas_addition.md.
    bank_tenor_case   : 'bank_tenor_comm' (default) or 'bank_tenor_legacy'
                        (rate-coherence task, Part 1). Selects the basis
                        for m.remaining_npv_factor, the bankability
                        discount factor -- see BANK_TENOR_CASE_CHOICES
                        and resolve_remaining_npv_factor()
                        (src/blended_finance.py). 'bank_tenor_comm':
                        the objective's own commercial tranche
                        (r_comm_nominal/L_comm), truncated at a real
                        tenor. 'bank_tenor_legacy': the pre-correction
                        blended-WACC factor (resolve_private_rate()),
                        undiscriminated by any tenor -- every
                        pre-correction result used this basis by
                        construction. Unlike new_gas_deliverability
                        above, the default here IS the corrected
                        behaviour, not the historical one.
    start_year, end_year : planning horizon bounds.
    """

    # ---- Validate labels
    if demand_level_case not in demand_level_scenarios():
        raise ValueError(f"Unknown demand_level_case: {demand_level_case}")
    _DEPRECATED_DEMAND_LEVELS = {"latent_low", "latent_high"}
    if demand_level_case in _DEPRECATED_DEMAND_LEVELS:
        raise ValueError(
            f"demand_level_case='{demand_level_case}' is deprecated and must "
            "not be used. These values were constructed as multiples of "
            "COLLECTED energy (23.08 TWh, a revenue quantity), not demand. "
            "Superseded by the tiered demand architecture (correction plan "
            "2.7). DEM-1 is suspended pending that rebuild."
        )
    if demand_case not in demand_growth_scenarios():
        raise ValueError(f"Unknown demand_case: {demand_case}")
    if carbon_case not in carbon_policy_scenarios():
        raise ValueError(f"Unknown carbon_case: {carbon_case}")
    if gas_deliverability_case not in gas_deliverability_scenarios():
        raise ValueError(f"Unknown gas_deliverability_case: {gas_deliverability_case}")
    if land_case not in land_scenarios():
        raise ValueError(f"Unknown land_case: {land_case}")
    if capital_case not in capital_envelope_scenarios():
        raise ValueError(f"Unknown capital_case: {capital_case}")
    if solar_build_case not in solar_build_scenarios():
        raise ValueError(f"Unknown solar_build_case: {solar_build_case}")
    if tariff_case is _TARIFF_REQUIRED:
        raise TypeError(
            "load_scenario() requires tariff_case (docs/tariff_arm.md, "
            "Steps 1-3). Pass one of 'tariff_low' / 'tariff_central' / "
            "'tariff_high' / 'tariff_legacy_95m' / 'tariff_legacy_65m' (see "
            "tariff_scenarios()). There is no default: the retired "
            "solar_tariff_case defaulted to 65,000,000 USD/TWh and wrote it "
            "into every scenario dict silently, with no error, warning, or "
            "visible signature in the output."
        )
    if tariff_case not in tariff_scenarios():
        raise ValueError(f"Unknown tariff_case: {tariff_case}")
    if blend_case not in blended_finance_scenarios():
        raise ValueError(f"Unknown blend_case: {blend_case}")

    # ---- 2.6, F0 Step 2.4: peak_multiple_case (required, no default --
    # mirrors genset_case discipline in load_econ()). Inverted design: the
    # arm supplies a SHAPE (period_demand_factor / season_demand_factor),
    # not a PAR scalar -- see peak_multiple_scenarios() and the module
    # comment above it.
    if peak_multiple_case is _PEAK_MULTIPLE_REQUIRED:
        raise TypeError(
            "load_scenario() requires peak_multiple_case (2.6, F0 Step 2.4). "
            "Pass one of 'peak_observed' / 'peak_nirp' / 'peak_high' (see "
            "peak_multiple_scenarios()). There is no default: omitting it "
            "would build a firm-capacity constraint against an unstated "
            "peak-demand shape with no error, warning, or visible signature "
            "in the output."
        )
    if peak_multiple_case not in peak_multiple_scenarios():
        raise ValueError(f"Unknown peak_multiple_case: {peak_multiple_case}")
    _peak_arm = peak_multiple_scenarios()[peak_multiple_case]
    _arm_period_demand_factor = _peak_arm["period_demand_factor"]
    if _arm_period_demand_factor is None:
        raise ValueError(
            f"peak_multiple_scenarios()['{peak_multiple_case}']"
            "['period_demand_factor'] is None -- SOURCE: [PENDING]. Supply "
            "a sourced period_demand_factor for this arm (module-level "
            "PEAK_*_PERIOD_FACTOR constants, top of scenarios.py) before "
            "building a scenario with a firm-capacity constraint."
        )
    # season_demand_factor is OPTIONAL per arm: None means "use the shared
    # default shape" -- not a pending/blocking gap the way
    # period_demand_factor's None is (see the module comment above).
    _arm_season_demand_factor = (
        _peak_arm["season_demand_factor"]
        if _peak_arm["season_demand_factor"] is not None
        else DEFAULT_SEASON_DEMAND_FACTOR
    )

    # ---- 2.6, F0 Step 2.3: reserve_margin (required, no default). Either a
    # numeric fraction (used as-is) or the literal string 'nirp_spinning'
    # (derives a per-year fraction from RESERVE_MARGIN_SPINNING_MW /
    # peak_mw[t], resolved further down once peak_mw is known).
    if reserve_margin is _RESERVE_MARGIN_REQUIRED:
        raise TypeError(
            "load_scenario() requires reserve_margin (2.6, F0 Step 2.3). "
            "Pass a fractional reserve margin (e.g. 0.15), or the literal "
            "'nirp_spinning' to derive it from RESERVE_MARGIN_SPINNING_MW / "
            "peak_mw[t]. There is no default: omitting it would build a "
            "firm-capacity constraint against an unstated reserve "
            "assumption with no error, warning, or visible signature in "
            "the output."
        )
    if isinstance(reserve_margin, str):
        if reserve_margin != "nirp_spinning":
            raise ValueError(
                f"Unknown reserve_margin mode: '{reserve_margin}'. Pass a "
                "numeric fraction, or the literal 'nirp_spinning'."
            )
    elif reserve_margin < 0:
        raise ValueError(f"reserve_margin must be >= 0, got {reserve_margin}")

    # ---- storage_cc_case (required, no default; 2.6, phase_2_3b Step 2).
    # Mirrors genset_case/peak_multiple_case discipline: storage's
    # firm-capacity credit is a declared judgement call (see
    # storage_cc_scenarios()), not a measured constant, so it must be named
    # explicitly rather than silently defaulting to the old flat 1.0.
    if storage_cc_case is _STORAGE_CC_REQUIRED:
        raise TypeError(
            "load_scenario() requires storage_cc_case (2.6, phase_2_3b Step "
            "2). Pass one of 'storage_cc_low' / 'storage_cc_central' / "
            "'storage_cc_high' (see storage_cc_scenarios()). There is no "
            "default: omitting it would build a firm-capacity constraint "
            "crediting storage at an unstated, unsourced fraction."
        )
    if storage_cc_case not in storage_cc_scenarios():
        raise ValueError(f"Unknown storage_cc_case: {storage_cc_case}")
    storage_cc = storage_cc_scenarios()[storage_cc_case]

    # ---- gas_availability_case (required, no default; 2.10, Step 2).
    # Mirrors genset_case/peak_multiple_case/storage_cc_case discipline: the
    # gas fleet's availability figure is a declared judgement call (see
    # gas_availability_scenarios()), not a measured constant, so it must be
    # named explicitly rather than silently defaulting to the old bare 0.80.
    if gas_availability_case is _GAS_AVAILABILITY_REQUIRED:
        raise TypeError(
            "load_scenario() requires gas_availability_case (2.10, Step "
            "2). Pass one of 'gas_avail_observed' / 'gas_avail_central' / "
            "'gas_avail_nameplate' (see gas_availability_scenarios()). "
            "There is no default: omitting it would build both the gas "
            "annual-energy ceiling and the gas firm-capacity credit "
            "against an unstated availability assumption, with no error, "
            "warning, or visible signature in the output."
        )
    if gas_availability_case not in gas_availability_scenarios():
        raise ValueError(f"Unknown gas_availability_case: {gas_availability_case}")
    _gas_availability_factor = gas_availability_scenarios()[gas_availability_case]
    if _gas_availability_factor is None:
        raise ValueError(
            f"gas_availability_case='{gas_availability_case}' is SOURCE: "
            "[PENDING] (2.10, Step 2) -- it must represent MECHANICAL "
            "availability only (availability given fuel is allocated), and "
            "has not been sourced. Select 'gas_avail_observed' (declared "
            "lower bound) or 'gas_avail_nameplate' (declared upper bound) "
            "until sourced -- see gas_availability_scenarios()."
        )

    # ---- gas_retirement_case (required, no default; 2.11, Step 2; renamed/
    # extended 2.12, Step 2). Mirrors genset_case/peak_multiple_case/
    # storage_cc_case/gas_availability_case discipline: the gas fleet's
    # retirement schedule is a declared judgement call (see
    # gas_retirement_scenarios()), not a measured constant, so it must be
    # named explicitly rather than silently defaulting to the old bare
    # 680.0 MW/yr from 2035.
    if gas_retirement_case is _GAS_RETIREMENT_REQUIRED:
        raise TypeError(
            "load_scenario() requires gas_retirement_case (2.11, Step 2). "
            "Pass one of 'gas_retire_none' / 'gas_retire_proxy' / "
            "'gas_retire_vintage_30' / 'gas_retire_vintage_40' / "
            "'gas_retire_vintage_50' (see gas_retirement_scenarios()). "
            "There is no default: omitting it would retire the gas "
            "baseline fleet on an unstated, unsourced schedule with no "
            "error, warning, or visible signature in the output."
        )
    # gas_retire_sourced was REMOVED in 2.12, Step 2 -- raise a specific,
    # named error rather than letting it fall through to the generic
    # "Unknown gas_retirement_case" message below, so anyone still naming
    # it is pointed straight at the replacement (2.12, Step 2: a parameter
    # whose name asserts provenance it does not have is the exact failure
    # mode this project has corrected twice already).
    if gas_retirement_case == _GAS_RETIRE_SOURCED_REMOVED:
        raise ValueError(
            "gas_retirement_case='gas_retire_sourced' was REMOVED in 2.12 "
            "and no longer exists. Its replacement is an ANALYST "
            "ESTIMATE, not a traced source -- pass 'gas_retire_vintage_30' "
            "([CENTRAL]), 'gas_retire_vintage_40', or "
            "'gas_retire_vintage_50' instead. See "
            "docs/gas_retirement_provenance.md and "
            "docs/limitations_register.md L8."
        )
    if gas_retirement_case not in gas_retirement_scenarios():
        raise ValueError(f"Unknown gas_retirement_case: {gas_retirement_case}")
    _gas_retirement_arm = gas_retirement_scenarios()[gas_retirement_case]
    if "technical_life_years" in _gas_retirement_arm:
        # VINTAGE arm (2.12, Step 2): the actual per-year series is
        # resolved below, once `years` is known --
        # gas_vintage_retirement_series_mw() needs the model's year index
        # to build an aligned series. gas_baseline_retirement_mw_per_year/
        # _start_year stay None: gas_cap_rule (both builders) checks the
        # series first and only falls back to the flat formula when it is
        # None, so these two are simply unused for a vintage arm.
        _gas_retirement_technical_life = _gas_retirement_arm["technical_life_years"]
        _gas_retirement_mw_per_year = None
        _gas_retirement_start_year = None
    else:
        # FLAT arm (2.11, unchanged): gas_retire_none / gas_retire_proxy.
        _gas_retirement_technical_life = None
        _gas_retirement_mw_per_year = _gas_retirement_arm["mw_per_year"]
        _gas_retirement_start_year = _gas_retirement_arm["start_year"]

    years = planning_horizon(start_year, end_year)

    # (2.12, Step 1) Resolve the vintage arm's per-year cumulative
    # retirement series now that `years` is known. None for the two FLAT
    # arms -- build_model()/build_model_sliced() fall back to the
    # existing flat-rate formula whenever this is None, so
    # gas_retire_none/gas_retire_proxy are byte-for-byte unchanged from
    # 2.11 (per the task's explicit "do not touch" instruction).
    if _gas_retirement_technical_life is not None:
        _gas_retirement_series_mw = gas_vintage_retirement_series_mw(
            years, _gas_retirement_technical_life
        )
    else:
        _gas_retirement_series_mw = None

    # ---- gas_build_case (required, no default; 2.9, Step 2). Mirrors
    # genset_case/peak_multiple_case/storage_cc_case discipline: whether/how
    # much new gas capacity the LP may build is a declared, named choice,
    # not a silent default -- NDC 3.0's single largest capacity commitment.
    if gas_build_case is _GAS_BUILD_REQUIRED:
        raise TypeError(
            "load_scenario() requires gas_build_case (2.9, Step 2). Pass "
            "one of 'gas_ndc_17gw' / 'gas_unconstrained' / 'gas_none' (see "
            "GAS_BUILD_CASE_CHOICES). There is no default: omitting it "
            "would let new gas capacity be built (or not) on an unstated "
            "assumption, with no error, warning, or visible signature in "
            "the output."
        )
    if gas_build_case not in GAS_BUILD_CASE_CHOICES:
        raise ValueError(f"Unknown gas_build_case: {gas_build_case}")

    gas_new_capacity_cap_mw = None
    if gas_build_case == "gas_ndc_17gw":
        # Phase 2.9 addendum: gas_ndc_17gw is UNBLOCKED. The trajectory is
        # the NIRP-derived NDC 3.0 path (finding P2-6), converted to
        # m.gas_capacity_mw's nameplate basis -- see
        # gas_ndc_17gw_trajectory_mw(). The prior "trajectory is
        # SOURCE: [PENDING], select gas_unconstrained/gas_none" raise here
        # is removed; the function no longer returns None.
        gas_new_capacity_cap_mw = gas_ndc_17gw_trajectory_mw(years)
    # gas_build_case's pre-2.13 unconditional block on
    # gas_unconstrained/gas_ndc_17gw (both the gas_new_efficiency-unset
    # raise and the gas_capex_per_mw-unsourced raise) is REMOVED (2.13,
    # Steps 1-2): gas_capex_case and ccgt_efficiency_case, validated
    # below, now always supply both values.

    # ---- gas_capex_case (required, no default; 2.13, Step 1). Mirrors
    # genset_case/peak_multiple_case/storage_cc_case/gas_availability_case/
    # gas_retirement_case discipline: new-build gas unit cost is a
    # declared judgement call (see gas_capex_scenarios()), not a measured
    # constant, so it must be named explicitly rather than silently
    # defaulting to the old bare 900,000 USD/MW literal.
    if gas_capex_case is _GAS_CAPEX_REQUIRED:
        raise TypeError(
            "load_scenario() requires gas_capex_case (2.13, Step 1). Pass "
            "one of 'gas_capex_low' / 'gas_capex_central' / "
            "'gas_capex_high' (see gas_capex_scenarios()). There is no "
            "default: omitting it would price every m.gas_add MW on an "
            "unstated capex assumption, with no error, warning, or "
            "visible signature in the output."
        )
    if gas_capex_case not in gas_capex_scenarios():
        raise ValueError(f"Unknown gas_capex_case: {gas_capex_case}")
    _gas_capex_per_mw = gas_capex_scenarios()[gas_capex_case]

    # ---- ccgt_efficiency_case (required, no default; 2.13, Step 2).
    # Mirrors the same discipline: new-build CCGT efficiency is a declared
    # judgement call (see ccgt_efficiency_scenarios()), not a measured
    # constant. Replaces gas_new_efficiency's pre-2.13 free-kwarg form --
    # the arm now ALWAYS supplies a value, so the old raise-if-None checks
    # on gas_new_efficiency (here and at new_gas_deliverability='exempt'
    # below, and in both builders) are removed; only arm-membership
    # validation remains.
    if ccgt_efficiency_case is _CCGT_EFFICIENCY_REQUIRED:
        raise TypeError(
            "load_scenario() requires ccgt_efficiency_case (2.13, Step "
            "2). Pass one of 'ccgt_eff_iso' / 'ccgt_eff_central' / "
            "'ccgt_eff_low' (see ccgt_efficiency_scenarios()). There is no "
            "default: omitting it would size the exempt fuel-equivalent "
            "allowance (new_gas_deliverability='exempt') on an unstated "
            "CCGT efficiency, with no error, warning, or visible "
            "signature in the output."
        )
    if ccgt_efficiency_case not in ccgt_efficiency_scenarios():
        raise ValueError(f"Unknown ccgt_efficiency_case: {ccgt_efficiency_case}")
    _gas_new_efficiency = ccgt_efficiency_scenarios()[ccgt_efficiency_case]

    # ---- capital_recovery_case (required, no default; F1, Step 2).
    # Mirrors the same discipline: the per-tranche rate/tenor pair is a
    # declared judgement call (see capital_recovery_scenarios()), not a
    # measured constant. Rates are NOMINAL USD, deflation pending 0b
    # ruling -- see the CRF() call site in build_model()/
    # build_model_sliced() for the deflation step.
    if capital_recovery_case is _CAPITAL_RECOVERY_REQUIRED:
        raise TypeError(
            "load_scenario() requires capital_recovery_case (F1, Step 2). "
            "Pass one of 'cr_central' / 'cr_ida21' / 'cr_comm_low' / "
            "'cr_comm_high' (see capital_recovery_scenarios()). There is "
            "no default: omitting it would price every tranche's capital "
            "charge on an unstated recovery period, with no error, "
            "warning, or visible signature in the output."
        )
    if capital_recovery_case not in capital_recovery_scenarios():
        raise ValueError(f"Unknown capital_recovery_case: {capital_recovery_case}")
    _cr_terms = capital_recovery_scenarios()[capital_recovery_case]

    # ---- dscr_case (required, no default; F1, Step 3). REPLACES the
    # pre-F1 required_margin fork entirely -- see dscr_scenarios() and
    # docs/required_margin_inventory.md.
    if dscr_case is _DSCR_REQUIRED:
        raise TypeError(
            "load_scenario() requires dscr_case (F1, Step 3). Pass one of "
            "'dscr_regression' / 'dscr_central' / 'dscr_low' / "
            "'dscr_high' (see dscr_scenarios()). There is no default: "
            "omitting it would size the EaaS bankability coverage cushion "
            "on an unstated value, with no error, warning, or visible "
            "signature in the output."
        )
    if dscr_case not in dscr_scenarios():
        raise ValueError(f"Unknown dscr_case: {dscr_case}")
    _dscr_value = dscr_scenarios()[dscr_case]

    # ---- us_inflation (required, no default; F1, Step 1). No named-arm
    # alternatives (a single required scalar, not a "_case") -- see the
    # _US_INFLATION_REQUIRED sentinel above. Used ONLY to deflate
    # capital_recovery_case's nominal rates; does NOT touch
    # social_discount_rate (Decision 4, docs/f1_implementation.md).
    if us_inflation is _US_INFLATION_REQUIRED:
        raise TypeError(
            "load_scenario() requires us_inflation (F1, Step 1). Pass a "
            "fractional inflation rate (e.g. 0.020, the US Federal "
            "Reserve long-run target). There is no default: omitting it "
            "would deflate every capital_recovery_case rate on an "
            "unstated inflation assumption, with no error, warning, or "
            "visible signature in the output."
        )
    if not (-1.0 < float(us_inflation) < 1.0):
        raise ValueError(
            f"us_inflation={us_inflation} is outside a plausible annual "
            "fractional range (-1, 1) -- pass a fraction (e.g. 0.020), "
            "not a percentage (e.g. 2.0)."
        )

    # ---- capital_treatment_case (required, no default; F1, Step 6).
    # NOT one of the task's three named "new required keys" -- see the
    # _CAPITAL_TREATMENT_REQUIRED sentinel above for why it exists anyway.
    if capital_treatment_case is _CAPITAL_TREATMENT_REQUIRED:
        raise TypeError(
            "load_scenario() requires capital_treatment_case (F1, Step "
            "6). Pass 'pre_f1_overnight' or 'f1_tranche' (see "
            "CAPITAL_TREATMENT_CASE_CHOICES). There is no default: "
            "omitting it would silently pick one of two materially "
            "different objective functions."
        )
    if capital_treatment_case not in CAPITAL_TREATMENT_CASE_CHOICES:
        raise ValueError(
            f"Unknown capital_treatment_case: {capital_treatment_case}"
        )

    # ---- concessional_envelope_case (required, no default; envelope
    # resolution). See concessional_envelope_scenarios(): with no bound at
    # all (the prior None default), CRF(r_conc) < CRF(r_comm) always, so
    # comm_vol[t] == 0 everywhere is optimal and every channel is funded
    # 100% concessionally with nothing to stop it -- structurally
    # impossible in reality.
    if concessional_envelope_case is _CONCESSIONAL_ENVELOPE_REQUIRED:
        raise TypeError(
            "load_scenario() requires concessional_envelope_case (F1 "
            "envelope resolution). Pass 'env_dares', 'env_dares_plus', "
            "'env_none', 'env_5x', or 'env_10x' (see "
            "concessional_envelope_scenarios()). There is no default: "
            "omitting it would leave concessional volume unconstrained "
            "and every channel would be funded 100% concessionally, with "
            "no error, warning, or visible signature in the output."
        )
    if concessional_envelope_case not in concessional_envelope_scenarios():
        raise ValueError(
            f"Unknown concessional_envelope_case: {concessional_envelope_case}"
        )
    _concessional_envelope_value = concessional_envelope_scenarios()[concessional_envelope_case]

    # ---- conversion_case (required, no default; Phase 2.8, Step 2).
    # Mirrors the gas_capex_case / ccgt_efficiency_case discipline:
    # OCGT-to-CCGT retrofit capex / uplift / efficiency are declared
    # judgement calls (see conversion_scenarios()), not measured constants.
    if conversion_case is _CONVERSION_REQUIRED:
        raise TypeError(
            "load_scenario() requires conversion_case (Phase 2.8, Step 2). "
            "Pass 'conv_none' / 'conv_central' / 'conv_low' / 'conv_high' / "
            "'conv_eff_low' / 'conv_eff_high' (see conversion_scenarios()). There is no default: omitting it "
            "would drop OCGT-to-CCGT retrofit -- the only fuel-free route "
            "to more generation under a binding gas deliverability "
            "constraint -- from the choice set with no visible signature."
        )
    if conversion_case not in conversion_scenarios():
        raise ValueError(f"Unknown conversion_case: {conversion_case}")
    _conversion_arm = conversion_scenarios()[conversion_case]
    _conversion_tiers = gas_conversion_tiers()
    # conversion capex is wired ONLY into the F1 per-tranche capital charge
    # (build_model_sliced, f1_tranche branch); the pre_f1_overnight branch
    # is deliberately not modified (Phase 2.8, Do NOT). An available
    # conversion arm under pre_f1_overnight would build converted plant
    # whose capex never enters the objective -- refuse it here.
    if _conversion_arm["available"] and capital_treatment_case == "pre_f1_overnight":
        raise ValueError(
            f"conversion_case='{conversion_case}' (an available arm) requires "
            "capital_treatment_case='f1_tranche'. Phase 2.8 wires conversion "
            "capex into the F1 per-tranche capital charge only and does not "
            "modify the pre_f1_overnight branch. Pass conv_none with "
            "pre_f1_overnight."
        )

    # ---- hydro_case (required, no default; correction plan 1.5).
    # Mirrors the gas_capex_case / conversion_case discipline: which hydro
    # trajectory the horizon carries is a declared, named choice. Hydro is
    # EXOGENOUS in all three arms -- a fixed annual energy series plus the
    # matching installed-capacity series, never a decision variable.
    if hydro_case is _HYDRO_REQUIRED:
        raise TypeError(
            "load_scenario() requires hydro_case (correction plan 1.5). "
            "Pass 'hydro_flat' / 'hydro_committed' / 'hydro_ndc' (see "
            "hydro_case_scenarios()). There is no default: omitting it "
            "would hold hydro flat at 11.47 TWh on an unstated assumption, "
            "leaving Zungeru's ramp and hydro under-delivery untested."
        )
    if hydro_case not in hydro_case_scenarios():
        raise ValueError(f"Unknown hydro_case: {hydro_case}")
    _hydro_energy_series, _hydro_capacity_series = hydro_trajectory(
        hydro_case, years
    )

    # ---- om_case (required, no default; task_om_implementation).
    # Fixed + variable O&M rates are per-technology, per-year, and depend
    # on the selected ATB cost case -- there is no single scalar to
    # default to. STORAGE_OM_PER_MWH_YR's history (defined, unsourced,
    # never read -- docs/task_storage_om.md) is exactly the outcome a
    # silent default here would risk repeating.
    if om_case is _OM_REQUIRED:
        raise TypeError(
            "load_scenario() requires om_case (task_om_implementation). "
            "Pass 'om_none' / 'om_low' / 'om_central' / 'om_high' (see "
            "om_scenarios()). There is no default: omitting it would leave "
            "fixed and variable O&M either unset (a builder KeyError) or "
            "silently reintroduce an unsourced constant, the same failure "
            "mode as the retired STORAGE_OM_PER_MWH_YR."
        )
    if om_case not in om_scenarios():
        raise ValueError(f"Unknown om_case: {om_case}")
    _om_cost_by_year = load_om_cost_by_year(
        om_case, start_year=start_year, end_year=end_year
    )

    # ---- capital_window_case (TASK 51 M1; default 'arrears' since TASK 53,
    # 'advance' = every result produced before). The
    # builder reads scenario["capital_window_case"] with NO fallback, so a
    # hand-built scenario that lacks the key fails loudly.
    if capital_window_case not in capital_window_scenarios():
        raise ValueError(f"Unknown capital_window_case: {capital_window_case}")

    # ---- storage_capex_case (TASK 52 section 2.1). Default 'declining_nrel'
    # since TASK 53 (restores the series 72cad20 wired and 96b50cf removed);
    # 'flat' = every result before, kept as the declared sensitivity.
    # Builder reads the key with NO fallback.
    if storage_capex_case not in storage_capex_scenarios():
        raise ValueError(f"Unknown storage_capex_case: {storage_capex_case}")
    _storage_capex_by_year = (
        load_storage_capex_by_year("Storage_low", start_year=start_year,
                                   end_year=end_year)
        if storage_capex_case == "declining_nrel" else None)

    # ---- new_gas_deliverability (2.9, Step 3). Real default ('constrained'
    # -- the Nigerian baseline, capacity without fuel, not relaxed by
    # assumption), unlike the REQUIRED-no-default arms above: both
    # treatments are implemented (see build_model()/build_model_sliced()),
    # this default only governs what an unstated caller gets.
    if new_gas_deliverability not in ("constrained", "exempt"):
        raise ValueError(
            f"Unknown new_gas_deliverability: {new_gas_deliverability}. "
            "Pass 'constrained' or 'exempt'."
        )

    # ---- gas_vintage_dispatch (TASK 74 Part D): on by default, forced off where the builder does not
    # support it (exempt deliverability); the annual builder raises if it is on.
    _gas_vintage_dispatch = bool(gas_vintage_dispatch) and new_gas_deliverability == "constrained"

    # ---- bank_tenor_case (rate-coherence task, Part 1). Real default
    # ('bank_tenor_comm' -- the objective's own commercial tranche), UNLIKE
    # new_gas_deliverability above: that default reproduces the pre-existing
    # behaviour; this one is the newly corrected behaviour. Both bases are
    # implemented (resolve_remaining_npv_factor(), src/blended_finance.py);
    # this default only governs what an unstated caller gets.
    if bank_tenor_case not in BANK_TENOR_CASE_CHOICES:
        raise ValueError(
            f"Unknown bank_tenor_case: {bank_tenor_case}. "
            f"Pass one of {BANK_TENOR_CASE_CHOICES}."
        )

    # ---- Blended finance: single source of truth (no hard-coded duplicates)
    conc_share, r_conc, r_comm = blended_finance_scenarios()[blend_case]

    # ============================================================
    # Firm-capacity / PAR inputs (2.6, F0 Step 2.3/2.4)
    # ============================================================
    # Extracted to local variables BEFORE the scenario dict is built so the
    # SAME numbers feed both the dict entries below and the derivations here
    # -- avoiding a second, driftable copy of gas_availability_factor,
    # hydro_baseline_twh, hydro_capacity_mw.
    # _gas_availability_factor is NOT redefined here (2.10, Step 2) -- it
    # was already resolved above from gas_availability_case, before
    # gas_build_case's validation. Redefining it here would silently
    # shadow the sourced/raising arm value with a bare literal again.
    _base_demand_twh = demand_level_scenarios()[demand_level_case]
    _demand_growth = demand_growth_scenarios()[demand_case]
    _hydro_baseline_twh = 11.47       # SOURCE: see dict entry below
    _hydro_capacity_mw = 2678         # SOURCE: see dict entry below
    # Storage discharge duration (hours). Previously an IMPLICIT default
    # (scenario.get("storage_duration_hours", 4.0)) inside both builders;
    # made explicit here because capacity_credit['storage'] below needs the
    # same number the builders will actually use, not a second copy of 4.0
    # that could silently drift from the builders' fallback.
    _storage_duration_hours = 4.0
    # season_demand_factor/period_demand_factor now come from the SELECTED
    # peak_multiple_case arm (2.6, F0 Step 2.4 inversion) -- there is no
    # independent local default any more.
    _season_demand_factor = _arm_season_demand_factor
    _period_demand_factor = _arm_period_demand_factor

    # par_block: DERIVED from the arm's shape via the model's own
    # weighting function (_normalised_demand_weights/slice_hours, imported
    # from optimize_model_sliced.py), never set independently (2.6, F0 Step
    # 2.4 inversion). PAR = f_peak * 8760 / h_peak = 6 * f_peak identically
    # (h_peak = 1460 h/yr is fixed calendar arithmetic) -- this IS the peak
    # block's demand-energy share, restated, not a second, independently
    # checkable number. Reported-only: stored on the scenario dict for
    # visibility, never fed back into a guard.
    #
    # par_block IS A 4-HOUR BLOCK-AVERAGE RATIO, NOT AN INSTANTANEOUS PEAK
    # RATIO -- see the module comment above peak_multiple_scenarios() and
    # within_block_peaking_factor (read inside the builders, NOT set here).
    _implied_weights = _normalised_demand_weights({
        "season_demand_factor": _season_demand_factor,
        "period_demand_factor": _period_demand_factor,
    })
    _slice_h = slice_hours()
    _f_peak = sum(_implied_weights[(s, "peak")] for s in ("dry", "wet"))
    _h_peak = sum(_slice_h[(s, "peak")] for s in ("dry", "wet"))
    par_block = _f_peak * 8760 / _h_peak

    # peak_mw[t] = avg_mw[t] * par_block, aligned to the model's year index
    # (same convention as solar_cumulative_cap_mw / _solar_cumulative_cap
    # above).
    _demand_proj = project_baseline_demand(
        base_demand=_base_demand_twh,
        growth_rate=_demand_growth,
        start_year=years[0],
        end_year=years[-1],
    )["demand"]
    peak_mw = [
        float(_demand_proj[t]) * 1e6 / 8760.0 * par_block
        for t in range(len(years))
    ]

    # Resolve reserve_margin now that peak_mw is known (2.6, F0 Step 2.3).
    # 'nirp_spinning': a FIXED absolute MW requirement implies a SHRINKING
    # per-year fraction as peak_mw[t] grows -- both builders accept either a
    # flat scalar or a per-year sequence here (same pattern already used for
    # max_offgrid_fraction).
    if reserve_margin == "nirp_spinning":
        reserve_margin_resolved = [
            RESERVE_MARGIN_SPINNING_MW / mw for mw in peak_mw
        ]
    else:
        reserve_margin_resolved = reserve_margin

    # Capacity credits (2.6, F0 Step 2.3). ALL FOUR are declared assumptions
    # pending sourcing -- SOURCE: [PENDING], except storage (see below).
    #   solar   : 0.0 -- no credit at system peak (evening peak, no solar
    #             yield in the peak slice; see SOLAR_PERIODS in
    #             optimize_model_sliced.py). SOURCE: [PENDING] as a judgement,
    #             not a citation.
    #   gas     : (2.10, Step 3) an INDEPENDENTLY SETTABLE peak-hour
    #             firm-capacity credit -- NOT the same object as
    #             gas_availability_factor (an annual energy ceiling) below
    #             any more, though both are currently initialized equal
    #             from the SAME gas_availability_case arm (see
    #             _gas_capacity_credit just below), because an annual
    #             energy availability factor and a peak-hour firm-capacity
    #             credit are different quantities -- before 2.10 they were
    #             a single hard-coded parameter (0.80). SOURCE: [PENDING].
    #   storage : storage_cc_case (2.6, phase_2_3b Step 2) -- REQUIRED, no
    #             default. Previously a bare min(1, duration/peak_hours) =
    #             1.0, unsourced, sitting above gas's 0.80. Replaced by a
    #             swept arm; see storage_cc_scenarios() for the SOURCE:
    #             DECLARED JUDGEMENT rationale (storage_cc_high=1.00
    #             reproduces the old flat figure as an explicit upper
    #             bound, not a default).
    #   hydro   : seasonal-dry -- average dry-season hydro output (the
    #             tighter of the two seasons) as a fraction of installed
    #             hydro capacity, using the same seasonal-share helper the
    #             sliced model itself uses. SOURCED (see the
    #             "hydro_season_factor" scenario-dict entry below for full
    #             provenance) -- this is a SEPARATE literal copy of the
    #             same two numbers, needed here because capacity_credit is
    #             computed before the scenario dict exists. KEEP IN SYNC.
    _hydro_share = _hydro_season_energy_share({
        "hydro_season_factor": {"dry": 1.045, "wet": 0.956},
    })
    _dry_hours_per_year = SEASON_DAYS["dry"] * 24
    _hydro_dry_avg_mw = (
        _hydro_baseline_twh * _hydro_share["dry"] * 1e6 / _dry_hours_per_year
    )
    # (2.10, Step 3): capacity_credit['gas'] is now a value DISTINCT from
    # gas_availability_factor (set on the scenario dict below), even though
    # both are initialized equal here. An annual energy availability factor
    # and a peak-hour firm-capacity credit are different quantities; before
    # 2.10 they were a single parameter (_gas_availability_factor, reused
    # directly). Not changing either NUMBER in this step -- only making
    # them independently settable so a later phase can source them
    # separately. See the matching comment on the scenario-dict
    # "gas_availability_factor" entry below.
    _gas_capacity_credit = _gas_availability_factor
    capacity_credit = {
        "solar": 0.0,                                                  # SOURCE: [PENDING]
        "gas": _gas_capacity_credit,                                    # SOURCE: [PENDING] -- peak-hour firm-capacity credit; see the 2.10, Step 3 comment above
        "storage": storage_cc,                          # SOURCE: DECLARED JUDGEMENT -- see storage_cc_scenarios()
        "hydro": _hydro_dry_avg_mw / _hydro_capacity_mw,                # SOURCE: [PENDING]
    }

    scenario = {
        # ---- Temporal
        "years": years,

        # ---- Demand (TWh_e, annual)
        "base_demand_twh": demand_level_scenarios()[demand_level_case],
        "demand_growth": demand_growth_scenarios()[demand_case],

        # ---- Gas deliverability to power (TWh_th; converted by gas_eta)
        "gas_scenario": gas_deliverability_scenarios()[gas_deliverability_case],
        "gas_eta": 0.287,

        # ---- Gas capacity baseline (gas-only fleet, MW) -- see
        # GAS_BASELINE_MW (module level, top of file) for SOURCE.
        "gas_baseline_mw": GAS_BASELINE_MW,
        # Brownfield retirement schedule: SOURCE: gas_retirement_case arm
        # (2.11, Step 2; extended 2.12, Step 2) -- see
        # gas_retirement_scenarios() for the FLAT-vs-VINTAGE breakdown per
        # case. For a FLAT arm (gas_retire_none/gas_retire_proxy),
        # gas_baseline_retirement_start_year/_mw_per_year are set and
        # gas_baseline_retirement_series_mw is None; for a VINTAGE arm
        # (gas_retire_vintage_30/40/50) it is the reverse -- gas_cap_rule
        # (both builders) checks the series first, falling back to the
        # flat formula only when it is None.
        "gas_retirement_case": gas_retirement_case,
        "gas_baseline_retirement_start_year": _gas_retirement_start_year,
        "gas_baseline_retirement_mw_per_year": _gas_retirement_mw_per_year,
        "gas_baseline_retirement_series_mw": _gas_retirement_series_mw,
        # SOURCE: gas_capex_case arm (2.13, Step 1) -- see
        # gas_capex_scenarios() for the NIRP-derived provenance behind
        # gas_capex_central/gas_capex_high. Was a bare, unsourced 900,000
        # USD/MW literal pre-2.13 (2.9, Step 0.1 finding); that value is
        # retained as the explicit gas_capex_low lower bound, not silently
        # dropped.
        "gas_capex_case": gas_capex_case,
        "gas_capex_per_mw": _gas_capex_per_mw,

        # ---- Gas fleet availability ceiling (fraction of installed capacity) ---
        # This scales INSTALLED gas capacity to a realistic ANNUAL ENERGY
        # ceiling (gas_capacity_constraint in both builders). It is a capacity
        # ceiling, not a dispatch prediction: the optimiser sets actual
        # utilisation below it, subject to fuel (gas_balance) and demand.
        #
        # (2.10, Step 3): this ANNUAL ENERGY CEILING is a DIFFERENT QUANTITY
        # from capacity_credit['gas'] (a peak-hour firm-capacity credit,
        # set above) -- before 2.10 they were a single parameter reused in
        # both roles. Both are now independently settable, initialized
        # equal from the SAME gas_availability_case arm only because no
        # separately sourced peak-hour figure exists yet.
        #
        # CRITICAL -- this must be FUEL-AGNOSTIC availability: mechanical +
        # atmospheric availability only (NERC Availability Factor clauses (i) and
        # (ii)). It must EXCLUDE clause (iii) feedstock availability, because gas-
        # supply scarcity is already carried by gas_balance / the deliverability
        # scenarios. Using NERC's headline Availability Factor here would double-
        # count fuel scarcity and contaminate the gas shadow prices.
        #
        # Source: gas_availability_case arm (2.10, Step 2) -- see
        # gas_availability_scenarios() for the sourced/pending breakdown per
        # case.
        "gas_availability_case": gas_availability_case,
        "gas_availability_factor": _gas_availability_factor,   # <-- set from fuel-agnostic availability

        # ---- Hydro (exogenous must-run, TWh_e/year, already electrical)
        # -------------------------------------------------------------------
        # Correction plan 1.5: the flat 11.47 TWh is now the hydro_flat arm
        # of a REQUIRED hydro_case (hydro_case_scenarios() / hydro_trajectory()).
        # Both builders read the by-year series below; hydro_baseline_twh /
        # hydro_growth / hydro_capacity_mw (scalars) are the year-2025 values
        # of the selected arm, kept for backward compatibility and for the
        # capacity_credit["hydro"] derivation (which uses the pinned
        # _HYDRO_ANCHOR pair, unchanged by any arm).
        "hydro_case": hydro_case,
        "hydro_energy_twh_by_year": _hydro_energy_series,
        "hydro_capacity_mw_by_year": _hydro_capacity_series,
        "hydro_baseline_twh": _hydro_energy_series[0],    "hydro_growth": 0.0,
        # Installed hydro capacity (MW), year-2025 value of the selected arm.
        # Kainji + Jebba + Shiroro + Zungeru + Dadin-Kowa.
        # [SOURCE: NERC — sum of installed hydro plant ratings]
        "hydro_capacity_mw": _hydro_capacity_series[0],

        # ---- Land policy
        "land_available_km2": land_scenarios()[land_case],
        "land_intensity_solar_km2_per_mw": 0.025,
        "land_intensity_gas_km2_per_mw": 0.001,
        "land_intensity_storage_km2_per_mwh": 0.00001,
        "storage_baseline_mwh": 0.0,

        # ---- Solar
        # SOURCED. Global Solar Atlas annual PVOUT / 8760, ground-mounted
        # large scale at each site's own optimum fixed tilt:
        #   Kano (Kumbotso) 11.887N  0.1968  <- national maximum measured
        #   FCT             9.165N   0.1743  (run at 10deg vs optimum 14deg,
        #                                     so slightly understated)
        #   Anambra         6.620N   0.1587  <- southern
        # The retired central 0.20 was the NATIONAL MAXIMUM: it assumed
        # every plant is built at the best site in the country. NIRP 2024
        # itself uses 0.19 falling to 0.18 (Annex F Table 19 with Table 8
        # CF), so the model sat ABOVE the national plan's own assumption.
        # The retired high arm 0.22 is ABOVE EVERY MEASURED NIGERIAN SITE
        # and is unattainable; it is retired rather than kept as an
        # unreachable bound. 0.16 (Anambra) replaces it as a reachable low.
        # DECLARED: this model has no transmission and no siting, so the CF
        # stands for a national average deployment. Best-resource land is
        # northern and overlaps insecurity zones; demand is southern. The
        # model cannot represent that trade-off (see the limitations
        # register), and the 0.16-0.20 band is the honest span.
        # Sweep {0.16, 0.19, 0.20} -- swept by scripts/20_run_cf1_solar_yield_sensitivity.py.
        "solar_cf": 0.19,   # fixed-tilt CF, central.
        "solar_baseline_mw": 500,
                "solar_cumulative_cap_mw": _solar_cumulative_cap(solar_build_case, years),
        # Min annual build floor: set to 100 MW/yr in runners when time-varying
        # CAPEX is active (prevents pathological all-delay). 0 = disabled here.
        "solar_min_build_mw_per_year": 0.0,
        "solar_capex_scenario": "solar_low",

        "asset_lifetimes": asset_lifetimes(),   # service life (yrs) by asset class, used for salvage credit
        # ---- Intra-annual shape parameters (sliced model only) --------------
        # NOTE: as of 2.6, F0 Step 2.4 (inversion), these are SOURCED FROM
        # peak_multiple_case, not set independently -- see
        # peak_multiple_scenarios() and the derivation above. Editing these
        # here directly has no effect; edit the selected arm's
        # PEAK_*_PERIOD_FACTOR / PEAK_*_SEASON_FACTOR constants instead.
        # Relative average power by season (ratio only; normalised internally).
        # Dry (Nov-Apr) demand higher: heat, peak Jan-Apr per cited load research.
        # Supplied by the selected peak_multiple_case arm's
        # PEAK_*_SEASON_FACTOR constant (~lines 176-202; registry in
        # peak_multiple_scenarios(), ~lines 215-232). peak_suppressed /
        # peak_unsuppressed trace to Oluwole (2023), Nigeria National
        # Electricity Demand Timeseries, DOI 10.17632/pxvdm26rn7.2 (2016
        # hourly data, ~line 163); peak_high is a DECLARED construction
        # built on that same shape (evening pinned at 1.20, ~line 194).
        # season_demand_factor is the one OPTIONAL per-arm value (~line
        # 2274): an arm whose PEAK_*_SEASON_FACTOR is None falls back to
        # DEFAULT_SEASON_DEMAND_FACTOR (optimize_model_sliced.py, still
        # UNSOURCED) instead. Only peak_nirp's is None today, and
        # peak_nirp is itself blocked (its PEAK_NIRP_PERIOD_FACTOR is
        # also None, which raises before this fallback is ever reached,
        # per period_demand_factor's note below) -- so every currently
        # reachable arm is fully sourced with no fallback taken.
        "season_demand_factor": _season_demand_factor,
        # Relative average power by period. Flatter than residential: national
        # aggregate load has C&I daytime offset; night is the genuine trough.
        # Observed grid data CANNOT settle this (shedding truncates the peak).
        # Supplied by the selected peak_multiple_case arm's
        # PEAK_*_PERIOD_FACTOR constant (~lines 176-202; registry in
        # peak_multiple_scenarios(), ~lines 215-232); same Oluwole (2023)
        # / declared-construction provenance as season_demand_factor
        # above. Unlike season_demand_factor, this has NO per-arm
        # fallback: a None constant (peak_nirp only) raises immediately
        # inside load_scenario() (~line 2263) rather than silently
        # substituting a default -- sourced or the call fails loudly.
        "period_demand_factor": _period_demand_factor,
        # Relative solar yield by season, day slices only. Dry = clearer skies
        # (offset by harmattan dust); wet = monsoon cloud.
        # SOURCED. Global Solar Atlas (World Bank / Solargis, retrieved
        # 2026-09-05), monthly PVOUT for three Nigerian ground-mounted
        # large-scale sites, each at its own optimum fixed tilt:
        #   Kumbotso, Kano             11.887N  1,723.7 kWh/kWp  tilt 16deg  ratio 1.146
        #   Federal Capital Territory   9.165N  1,526.9          tilt 10deg  ratio 1.236
        #   Anambra                     6.620N  1,390.3          tilt  8deg  ratio 1.106
        # Dry = Nov-Apr, wet = May-Oct, day-weighted, normalised so the
        # day-weighted annual mean is 1.0 (181 dry days, 184 wet).
        # Site mean ratio 1.1624 -> dry 1.0758 / wet 0.9255.
        # The retired 1.15/0.85 (ratio 1.353) overstated the seasonal
        # contrast at EVERY site measured, by 9.5% against the strongest
        # (FCT) and 22% against the weakest (Anambra).
        # HARMATTAN: the dry season is not uniformly better. At FCT, output
        # PEAKS in November (5,003 kWh/day) and declines monotonically to
        # April (4,241) -- a 15% fall WITHIN the dry season as dust loads.
        # The two-block seasonal factor cannot represent that; it is
        # averaged into the dry figure. DECLARED simplification.
        "season_solar_factor": {"dry": 1.076, "wet": 0.925},
        # Hydro seasonal energy split.
        # SOURCED, and the sign of the retired value was WRONG.
        # Two independent NERC quarterly datasets, both excluding Zungeru:
        #   2023, PRE-ZUNGERU (Zungeru absent from the 2023 tables), total
        #   generation GWh for Shiroro + Jebba + Kainji + Dadin Kowa,
        #   converted to average power: Q1 1,097.6 MW / Q2 823.6 /
        #   Q3 1,000.2 / Q4 1,224.0. ratio 1.118
        #   Clean check, Q1 (all dry) vs Q3 (all wet): 1.097
        #   2025, Zungeru excluded from the aggregate, average hourly
        #   generation basis: ratio 1.070
        # Two-year mean ratio 1.0938 -> dry 1.0452 / wet 0.9556.
        # Quarters straddle the dry/wet boundary (Q1 all dry, Q3 all wet,
        # Q2 and Q4 mixed), so quarters are month-weighted onto the split.
        # WHY THE RETIRED VALUE WAS BACKWARDS: 0.90/1.10 assumed
        # run-of-river hydrology -- low flow in the dry season. Kainji and
        # Jebba are RESERVOIR schemes on the Niger (Kainji regulates,
        # Jebba re-regulates downstream), so output follows DEMAND, not
        # inflow, and Nigerian demand peaks in the dry season. Operators
        # draw down through the dry months. The seasonal signal is an
        # operating decision, not a hydrological one.
        # KEEP IN SYNC with the identical literal feeding
        # capacity_credit['hydro'] above (this function, ~30 lines up) and
        # with DEFAULT_HYDRO_SEASON_FACTOR / DEFAULT_SEASON_SOLAR_FACTOR
        # (src/optimize_model_sliced.py) -- four copies of these two
        # numbers exist; see docs/shape_factors_and_dist5.md.
        "hydro_season_factor": {"dry": 1.045, "wet": 0.956},
        # Minimum hydro per slice as a fraction of proportional share
        # (environmental flow, irrigation, downstream obligations).
        # Classification: ASSUMED (analyst estimate), NOT SOURCED.
        # A documented search found no Nigerian minimum-flow obligation
        # expressed as a fraction. Nigerian dam-operation sources give
        # ABSOLUTE DISCHARGE RATES, not fractions.
        # Nearest documented analogue: JICA, Nigeria National Water
        # Resources Master Plan, Vol. 5, SR6, p. SR6-7, Table SR6-9,
        # "Assumed Operation Rules for Release from Shiroro Dam" -- 315 /
        # 100 / 50 m3/s at reservoir storage of 20% / 10% / below 10%.
        # JICA itself labels these ASSUMED operating rules, not a
        # mandated environmental-flow obligation.
        # That rule is NOT convertible into this parameter, for two
        # independent reasons: (i) it is an absolute discharge at a
        # single plant, while this parameter is a fleet-wide fraction of
        # an hours-proportional per-slice energy share; (ii) it is
        # STORAGE-STATE DEPENDENT, and this model carries no reservoir
        # state variable (verified: the only hydro decision variable is
        # m.hydro_gen[t,(s,p)], a per-slice dispatch quantity, and all
        # three hydro constraints in optimize_model_sliced.py are
        # contemporaneous -- see docs/task_hydro_gate_bucketA.md, 1A), so
        # there is nothing for such a rule to key off even in principle.
        # Turbine discharge at 95% reliability (Kainji 440, Jebba 500,
        # Shiroro 120 m3/s) is PLANT CAPABILITY, not release obligation,
        # and must not be read as a minimum flow. An earlier reading of
        # the 500 m3/s figure as an obligation was incorrect and is
        # recorded here so it is not repeated.
        # PROHIBITION: do not derive, estimate, back-calculate or
        # otherwise produce a numeric fraction from any discharge figure
        # above. Conversion would require head, turbine efficiency,
        # obligation coverage, fleet weighting and a slice-mapping --
        # five assumptions whose product would be an analyst construction
        # wearing a citation. The value stays 0.50 and stays ASSUMED.
        "hydro_min_flow_fraction": 0.50,
        # ---- Public capital constraint (NPV, USD)
        "public_solar_budget_npv": capital_envelope_scenarios()[capital_case],

        # ---- Storage reduced-form (annual)
        "storage_deployable_hours_per_year": 700.0,
        # storage_solar_surplus_frac: coupling cap intentionally DROPPED.
        # Storage is governed by power + throughput limits only. If reinstated,
        # it must be sourced from an 8760 solar-vs-load surplus calculation, not
        # a bare 0.20 default. Left commented as an explicit record of the choice.
        # "storage_solar_surplus_frac": 0.20,
        "storage_round_trip_eff": 0.90,
        # Was an IMPLICIT builder-side default (scenario.get(..., 4.0)) before
        # 2.6, F0 Step 2.3; now explicit here because capacity_credit['storage']
        # is derived from this exact number (see above).
        "storage_duration_hours": _storage_duration_hours,

        # ---- Fixed + variable O&M (task_om_implementation). SOURCED: NREL
        # ATB 2024, data/cost/processed/om_cost.csv. Both builders convert
        # storage's FOM (USD/kW-yr, power basis) to USD/MWh-yr using
        # storage_duration_hours above -- see io.compute_om_cost_npv.
        "om_case": om_case,
        "om_cost_by_year": _om_cost_by_year,
        "capital_window_case": capital_window_case,
        "storage_capex_case": storage_capex_case,
        "storage_capex_by_year": _storage_capex_by_year,

        # ---- Financing regime
        "financing_regime": "traditional",  # or "eaas"

        # ---- EaaS service tariff (USD per TWh)
        "solar_service_tariff_usd_per_twh":
            tariff_scenarios()[tariff_case],

        # ---- Discounting
        "social_discount_rate": 0.08,   # objective DF (society's time preference)

        # ---- EaaS capital structure (blended concessional finance)
        # Pulled from blended_finance_scenarios()[blend_case] above -- single
        # source of truth. Consumed by resolve_private_rate() ONLY under
        # bank_tenor_case='bank_tenor_legacy' now (rate-coherence task, Part
        # 1) -- see resolve_remaining_npv_factor(), src/blended_finance.py.
        # Under the default 'bank_tenor_comm' these three keys are set but
        # do not reach m.remaining_npv_factor; they still gate
        # private_discount_rate's fallback status below and remain read by
        # run_fin2_blended_bankability_sweep()'s own analytics regardless of
        # bank_tenor_case.
        "blend_case": blend_case,
        "concessional_share": conc_share,
        "r_concessional": r_conc,
        "r_commercial": r_comm,
        "private_discount_rate": r_comm,   # fallback == unblended commercial rate

        # ---- Bankability discount-factor basis (rate-coherence task, Part
        # 1). See BANK_TENOR_CASE_CHOICES above and
        # resolve_remaining_npv_factor() (src/blended_finance.py).
        "bank_tenor_case": bank_tenor_case,

        # Envelope resolution: the pre-existing Level-2 scarcity mechanism
        # (concessional_envelope_npv scenario key, NPV-discounted,
        # EaaS-solar-only) is RETIRED -- see docs/f1_implementation.md,
        # "Envelope resolution", Decision 2. The F1 envelope below is the
        # only concessional envelope this scenario dict carries now.
        #
        # F1's OWN concessional envelope: commitment STOCK, UNDISCOUNTED,
        # spanning solar (public + EaaS) and storage -- gas is
        # COMMERCIAL-ONLY and excluded (Decision 1). REQUIRED, no default
        # -- see concessional_envelope_case above.
        "concessional_envelope_case": concessional_envelope_case,
        "concessional_envelope_usd": _concessional_envelope_value,

        # ---- Required NPV margin (bankability coverage cushion, distinct from WACC)
        # SOURCE: dscr_case arm (F1, Step 3) -- see dscr_scenarios(). REPLACES
        # the pre-F1 required_margin fork (builder 1.0 fallback, this
        # unconditional 1.10 literal, run_gas_regime_ndc_matrix's 1.10 kwarg --
        # all three removed). Single cushion, constant across financing
        # regimes -- the old 1.05/1.10 split is retired as a double-count now
        # that the CRF rates (capital_recovery_case) carry that difference.
        "dscr_case": dscr_case,
        "required_margin": _dscr_value,

        # ---- F1 per-tranche capital recovery (Step 2). NOMINAL USD -- see
        # capital_recovery_scenarios() for full provenance. Deflated ONCE,
        # inside build_model()/build_model_sliced() at the CRF() call site,
        # using us_inflation below -- NOT deflated here.
        "capital_recovery_case": capital_recovery_case,
        "r_conc_nominal": _cr_terms["r_conc"],
        "L_conc": _cr_terms["L_conc"],
        "r_comm_nominal": _cr_terms["r_comm"],
        "L_comm": _cr_terms["L_comm"],
        # NOMINAL USD, deflation pending 0b ruling -- RULED (Decision 4,
        # docs/f1_implementation.md): model is real-USD, so this rate DOES
        # deflate capital_recovery_case (does NOT deflate
        # social_discount_rate above, which is already real).
        "us_inflation": float(us_inflation),

        # ---- F1 objective-form selector (Step 6). 'pre_f1_overnight'
        # reproduces the pre-F1 objective exactly (four overnight capex terms
        # + salvage + replacement); 'f1_tranche' replaces all six with the
        # per-tranche CRF annuity. See CAPITAL_TREATMENT_CASE_CHOICES.
        "capital_treatment_case": capital_treatment_case,

        # ---- Diagnostic-only inputs (consumed by specific runners, not the LP)
        # Confirm these are read where intended; they do not affect the core solve.
        "disco_collection_rate": 1.0,   # used by FIN-3 runner
        # NOTE: this is a PRE-EXISTING, DIFFERENT quantity from
        # peak_multiple_case/peak_mw below. It feeds only the post-solve
        # heuristic in optimize_experiments.py (peak_adequacy_by_year), not
        # any LP constraint, and is not derived from
        # season_demand_factor/period_demand_factor the way par_block now
        # is. Left as-is (2.6, F0 Step 2.3/2.4 did not touch it); flagged
        # here so the two are not confused.
        "peak_demand_multiple": 2.5,    # used by peak-adequacy diagnostic

        # ============================================================
        # Firm-capacity constraint inputs (2.6, F0 Step 2.3/2.4, inverted)
        # ============================================================
        # peak_mw[t] = avg_mw[t] * par_block. par_block is DERIVED from the
        # peak_multiple_case arm's season_demand_factor/period_demand_factor
        # above -- reported here for visibility, never fed back into a
        # guard (there is no consistency check any more; see
        # peak_multiple_scenarios()). par_block is a 4-hour BLOCK-AVERAGE
        # ratio, not an instantaneous peak ratio -- see
        # within_block_peaking_factor, read inside the builders (NOT set
        # here; it was the selfgen_busbar_factor pattern until TASK 69 B.2
        # made that factor an explicit key below -- a real, non-blocking
        # default read via scenario.get() at the builder, not forced into
        # this dict).
        "peak_multiple_case": peak_multiple_case,
        "par_block": par_block,
        "peak_mw": peak_mw,
        "reserve_margin": reserve_margin_resolved,
        "capacity_credit": capacity_credit,
        "storage_cc_case": storage_cc_case,
        # TASK 69 B.2 (register L96): the busbar-equivalence factor of self-generation, now an EXPLICIT
        # scenario value instead of a builder-side scenario.get default. 1.0 = no credit for the avoided
        # transmission loss (the value every stored result used); 1/(1-0.07) credits a 7% TLF. The builders
        # read it directly, so a hand-built scenario without it fails loudly.
        "selfgen_busbar_factor": 1.0,

        # ============================================================
        # New gas capacity inputs (2.9, Steps 1-3; 2.13, Steps 1-2)
        # ============================================================
        "gas_build_case": gas_build_case,
        "gas_new_capacity_cap_mw": gas_new_capacity_cap_mw,
        # SOURCE: ccgt_efficiency_case arm (2.13, Step 2) -- see
        # ccgt_efficiency_scenarios() for the HHV-basis/ambient-derate
        # provenance. No longer a free load_scenario() kwarg.
        "ccgt_efficiency_case": ccgt_efficiency_case,
        "gas_new_efficiency": _gas_new_efficiency,
        "new_gas_deliverability": new_gas_deliverability,
        "gas_vintage_dispatch": _gas_vintage_dispatch,

        # ---- OCGT-to-CCGT conversion (Phase 2.8). conversion_available
        # gates the whole mechanism in build_model_sliced(); conv_none
        # sets it False and the sliced builder builds no conversion
        # components (regression arm). build_model() (annual) raises on
        # an available arm -- dispatch is slice-resolved, no annual form.
        # conversion_tiers is always derived from the CSV (harmless when
        # unavailable); the builder reads nameplate_mw / paf per tier.
        "conversion_case": conversion_case,
        "conversion_available": _conversion_arm["available"],
        "conversion_capex_per_kw": _conversion_arm["capex_per_kw"],
        "conversion_uplift": _conversion_arm["uplift"],
        "conversion_retrofit_efficiency": _conversion_arm["retrofit_efficiency"],
        "conversion_tiers": _conversion_tiers,

        # ---- Carbon policy (spreads carbon_active / carbon_price)
        **carbon_policy_scenarios()[carbon_case],

        # ---- Labels (reporting)
        "labels": {
            "demand_level": demand_level_case,
            "demand": demand_case,
            "carbon": carbon_case,
            "gas_deliverability": gas_deliverability_case,
            "land": land_case,
            "solar_build": solar_build_case,
            "blend": blend_case,
            "peak_multiple": peak_multiple_case,
            "storage_cc": storage_cc_case,
            "gas_build": gas_build_case,
            "new_gas_deliverability": new_gas_deliverability,
            "gas_availability": gas_availability_case,
            "gas_retirement": gas_retirement_case,
            "gas_capex": gas_capex_case,
            "ccgt_efficiency": ccgt_efficiency_case,
            "capital_recovery": capital_recovery_case,
            "dscr": dscr_case,
            "capital_treatment": capital_treatment_case,
            "concessional_envelope": concessional_envelope_case,
            "conversion": conversion_case,
            "hydro": hydro_case,
            "bank_tenor": bank_tenor_case,
        },
    }

    return scenario


# ============================================================
# Lever families (TASK 48 Part F.1)
# ============================================================
# Which arms form one "lever" for the lever ranking, declared as DATA so that
# docs/lever_ranking.md is not assembled by inference (TASK 47 C1 recorded the
# deliverability and GMP memberships as inferred).
#
# SPAN RULE (F.2): a lever's magnitude is max minus min of the objective
# (npv_total_cost_usd) across EVERY arm in its family -- the central arm
# included, since the central arm is a member of every family. Applied
# uniformly; there is no family-specific definition.
#
# Each family varies ONE field of the 01b_run_baseline_sliced.py arm tag
# (BASELINE_ARM_TAG_FIELDS, in tag order); every other field is held at
# BASELINE_CENTRAL_ARM. A member is the value that field takes, so the arm's
# tag -- and therefore its hash8 (src/output_naming.py) -- is DERIVED, never
# typed. Source arms are results/baseline_sliced/summary__<hash8>.json.
#
# om_case is NOT a tag field: it is hashed only when non-central
# (output_naming.om_tag_suffix, TASK 48 B.1), so the `om` family is the one
# whose members differ by tag suffix; see "om_family" handling in
# scripts/build_lever_ranking.py.
#
# `bound`: the lever is an UPPER BOUND on a counterfactual, not a policy
# lever (hydro: exogenous, costless, no build-rate constraint; TASK 48 Part C).
# `in_f5_ranking`: the family is one of the eight levers F-5 / TASK 47 C1
# ranked; the others are swept families the ranking did not list.
#
# LIVE-ARM RULE (TASK 70): `unasserted` {arm: reason} marks a member that is unsourced and
# represents no state or bound the thesis asserts, retained only so that superseded results
# remain attributable. The generator reports a third span over the family's other (live) arms and
# lists the unasserted arm with its reason. This declaration is the authority; the generator never
# parses comments. Attribution wording alone never makes an arm unasserted: `candidate_notes`
# {arm: reason} records, for every arm a comment search turned up that is NOT unasserted, why it
# is live. The generator emits the span-with-arm-excluded for every arm in either dict.
BASELINE_ARM_TAG_FIELDS = (
    "peak_case", "storage_cc_case", "gas_build_case", "new_gas_deliverability",
    "gas_availability_case", "gas_retirement_case", "gas_capex_case",
    "ccgt_efficiency_case", "capital_recovery_case", "dscr_case",
    "capital_treatment_case", "concessional_envelope_case",
    "gas_deliverability_case", "conversion_case", "hydro_case",
)

BASELINE_CENTRAL_ARM = {
    "peak_case": "peak_unsuppressed",
    "storage_cc_case": "storage_cc_central",
    "gas_build_case": "gas_unconstrained",
    "new_gas_deliverability": "constrained",
    "gas_availability_case": "gas_avail_nameplate",
    "gas_retirement_case": "gas_retire_vintage_30",
    "gas_capex_case": "gas_capex_central",
    "ccgt_efficiency_case": "ccgt_eff_central",
    "capital_recovery_case": "cr_central",
    "dscr_case": "dscr_central",
    "capital_treatment_case": "f1_tranche",
    "concessional_envelope_case": "env_dares",
    "gas_deliverability_case": "baseline",
    "conversion_case": "conv_central",
    "hydro_case": "hydro_flat",
}
BASELINE_CENTRAL_OM_CASE = "om_central"

LEVER_FAMILIES = (
    {"id": "gas_availability", "label": "gas fleet availability (nameplate -> observed)",
     "field": "gas_availability_case",
     "members": ("gas_avail_nameplate", "gas_avail_observed"),
     "candidate_notes": {
         "gas_avail_nameplate": "LIVE: the central arm and the anchor of every result (BASELINE_CENTRAL_ARM); "
                                "its 'retained' docstring wording does not make it unasserted"},
     "bound": False, "in_f5_ranking": True},
    {"id": "conversion", "label": "conversion (none -> high)",
     "field": "conversion_case",
     "members": ("conv_none", "conv_low", "conv_central", "conv_high"),
     "candidate_notes": {
         "conv_none": "LIVE: an attainable state (no conversion) and the cap builder's own baseline (Ch3 s3.7.6); "
                      "'regression arm' wording does not make it unasserted"},
     "bound": False, "in_f5_ranking": True},
    # TASK 76 C.3: efficiency ONLY (capex 600, uplift 0.50 fixed at conv_central). The `conversion` family above is
    # COMPOUND: it moves retrofit capex (850/600/450), uplift (0.40/0.50/0.50) and retrofit efficiency
    # (0.42/0.46/0.50) together, so its span is a conversion programme's worth (register L129).
    {"id": "conversion_efficiency", "label": "conversion retrofit efficiency only (0.42 / 0.46 / 0.50)",
     "field": "conversion_case",
     "members": ("conv_eff_low", "conv_central", "conv_eff_high"),
     "bound": False, "in_f5_ranking": False},
    {"id": "gas_retirement", "label": "gas retirement (none / vintage 30-50 / proxy)",
     "field": "gas_retirement_case",
     "members": ("gas_retire_none", "gas_retire_vintage_30", "gas_retire_vintage_40",
                 "gas_retire_vintage_50", "gas_retire_proxy"),
     # UNASSERTED: unsourced, represents no state or bound the thesis asserts, retained only so
     # that pre-2.11 results stay attributable (src/scenarios.py gas_retire_proxy docstring).
     "unasserted": {
         "gas_retire_proxy": "pre-2.11 'brownfield retirement proxy': unsourced [PENDING]; retained only so "
                             "pre-2.11 results stay attributable; claims no state or bound"},
     "candidate_notes": {
         "gas_retire_none": "LIVE: a declared upper bound on the surviving fleet; a bound is a claim"},
     "bound": False, "in_f5_ranking": True},
    {"id": "capital_recovery", "label": "capital recovery span (financing)",
     "field": "capital_recovery_case",
     "members": ("cr_central", "cr_ida21", "cr_comm_low", "cr_comm_high"),
     "bound": False, "in_f5_ranking": True},
    {"id": "gas_deliverability_drift", "label": "deliverability, drift family",
     "field": "gas_deliverability_case",
     "members": ("baseline", "upside", "downside", "shock_recovery"),
     "bound": False, "in_f5_ranking": True},
    {"id": "gas_deliverability_gmp", "label": "deliverability, GMP family",
     "field": "gas_deliverability_case",
     "members": ("baseline", "gas_deliv_flat", "gas_deliv_gmp_partial", "gas_deliv_gmp_full"),
     "bound": False, "in_f5_ranking": True},
    {"id": "concessional_envelope", "label": "concessional envelope",
     "field": "concessional_envelope_case",
     "members": ("env_none", "env_dares", "env_5x", "env_10x", "env_dares_plus"),
     "bound": False, "in_f5_ranking": True},
    {"id": "peak", "label": "demand shape (suppressed -> high)",
     "field": "peak_case",
     "members": ("peak_suppressed", "peak_unsuppressed", "peak_high"),
     "candidate_notes": {"peak_suppressed": "LIVE: a declared floor (L2)"},
     "bound": False, "in_f5_ranking": True},
    {"id": "hydro", "label": "hydro trajectory (flat -> ndc)",
     "field": "hydro_case",
     "members": ("hydro_flat", "hydro_committed", "hydro_ndc"),
     "candidate_notes": {
         "hydro_flat": "LIVE: the central arm and anchor; 'regression arm' wording does not make it unasserted"},
     "bound": True, "in_f5_ranking": False},
    {"id": "new_gas_deliverability", "label": "new-gas deliverability (constrained / exempt)",
     "field": "new_gas_deliverability",
     "members": ("constrained", "exempt"),
     "bound": False, "in_f5_ranking": False},
    {"id": "storage_cc", "label": "storage capacity credit",
     "field": "storage_cc_case",
     "members": ("storage_cc_low", "storage_cc_central", "storage_cc_high"),
     "bound": False, "in_f5_ranking": False},
    {"id": "gas_build", "label": "gas build (unconstrained / ndc_17gw / none)",
     "field": "gas_build_case",
     "members": ("gas_unconstrained", "gas_ndc_17gw", "gas_none"),
     "candidate_notes": {
         "gas_none": "LIVE: an attainable policy state; the span to it measures the value of being permitted to build new gas"},
     "bound": False, "in_f5_ranking": False},
    {"id": "gas_capex", "label": "gas capex",
     "field": "gas_capex_case",
     "members": ("gas_capex_low", "gas_capex_central", "gas_capex_high"),
     "candidate_notes": {
         "gas_capex_low": "LIVE: a DECLARED LOWER BOUND; a bound is a claim; attribution wording alone never makes an arm unasserted"},
     "bound": False, "in_f5_ranking": False},
    {"id": "ccgt_efficiency", "label": "CCGT efficiency",
     "field": "ccgt_efficiency_case",
     "members": ("ccgt_eff_central", "ccgt_eff_iso", "ccgt_eff_low"),
     "bound": False, "in_f5_ranking": False},
    {"id": "dscr", "label": "DSCR",
     "field": "dscr_case",
     "members": ("dscr_central", "dscr_low", "dscr_high"),
     "bound": False, "in_f5_ranking": False},
    # Not a tag field: the members differ by output_naming.om_tag_suffix.
    {"id": "om", "label": "O&M band (none / low / central / high)",
     "field": "om_case",
     "members": ("om_none", "om_low", "om_central", "om_high"),
     "candidate_notes": {
         "om_none": "NOT unasserted: SYNTHETIC and unattainable; the family is already outside the ranked table by the build-response rule"},
     "bound": False, "in_f5_ranking": False},
)
