"""
00_build_emissions_cap.py  —  NDC 3.0 (proportional top-down) + NDC 2.0 (BAU-relative)
========================================================================================

TWO NDC FORMULATIONS
---------------------
This file generates annual power-sector emissions caps under two distinct
NDC apportionment methodologies:

    NDC 2.0 (BAU-relative):
        Cap is a percentage of the model's own baseline power-sector emissions.
        Unconditional: 20% below BAU 2030.  Conditional: 47% below BAU 2030.

    NDC 3.0 (proportional top-down):
        Cap is derived from NDC 3.0's energy-sector abatement target, apportioned
        to gas-fired power generation by its share of energy-sector emissions.

NDC 3.0 PROPORTIONAL METHODOLOGY
----------------------------------
NDC 3.0 (September 2025) defines economy-wide absolute reduction targets:
    - 168.2 MtCO2e by 2030  (29.3% of 2018 baseline of 573.5 MtCO2e)
    - 184.9 MtCO2e by 2035  (32.2% of 2018 baseline)
    Source: NDC 3.0, Section 3, p.14

The energy sector's share of economy-wide abatement (NDC 3.0, p.15):
    - 2030: 22.0%  ->  37.0 MtCO2e energy-sector abatement
    - 2035: 26.3%  ->  48.6 MtCO2e energy-sector abatement
NDC 3.0's own definition of "Energy" (p.38, ICTU item 3(b)) -- Fuel
Combustion (Energy industries, Manufacturing, Transport, Residential,
Agriculture/Forestry/Fishing) + Fugitive emissions (Solid fuels, Oil and
Gas) -- is structurally identical to BTR1 Table 2.10's Sector 1 "Energy"
(below), and matches it numerically to the sub-category. The p.15 share
is stated as covering "Energy sector (including buildings and
transport)" -- exactly this scope, not a narrower one. The pairing
below is correct because the two documents' own sector definitions
match, not merely convenient (docs/task_log_btr1_denominator.md Part 2
item 2).

Gas-fired power's share of energy-sector emissions -- INVENTORY-DERIVED,
not model-derived (docs/task_log_btr1_denominator.md):
    POWER_SHARE_OF_ENERGY_2018 = 16,234.0 / 193,346.72 = 8.3963%
    Numerator: National Inventory Document (NID, 24 Dec 2024) Table 3.9,
      "1.A.1.a Public electricity and heat production", 2018, p.36.
    Denominator: BTR1 (26 Dec 2024) Table 2.10, Sector 1 "Energy", 2018,
      p.33.
    Both figures: 2018, the same inventory vintage (submissions two days
    apart), AR5 GWP, 2006 IPCC Guidelines + 2019 Refinement -- confirmed
    independently in both source documents, not assumed from one.

THIS REPLACES A CONSTRUCTION THAT DIVIDED THE WRONG THING. The retired
GAS_POWER_SHARE = E_base / 217.0 divided the MODEL's own baseline
gas-to-power emissions (a model quantity) by an UNSOURCED "~38% of
573.5, round conservative estimate" (217.0 MtCO2e) that traced to no
page or table anywhere in NDC 3.0, BTR1 or the NID -- checked directly
(docs/task_log_ndc_extraction.md Part 4, docs/task_log_btr1_denominator.md).
A ratio of a MODEL figure to an UNSOURCED figure expresses no actual
share of anything; the ~7.24% it produced was not measuring gas-power's
share of anything real. The new share is INVENTORY-OVER-INVENTORY --
both terms are 2018 BTR1/NID figures on the same basis -- and is a
genuine share: gas-fired power's own emissions as a fraction of the
Energy sector's own emissions, both from the same national inventory.

Gas-power abatement = POWER_SHARE_OF_ENERGY_2018 x energy_sector_abatement
Power-sector cap    = E_base - gas_power_abatement.

DECLARED ASSUMPTION, not an NDC figure, sensitivity-testable: this
apportions abatement to gas-fired power in EQUAL PROPORTION to its
share of Energy-sector EMISSIONS -- i.e. the power sector is asked to
abate at the same intensity (abatement / emissions) as the Energy
sector average, no more, no less. NDC 3.0 never states a power-sector-
specific abatement share anywhere found (checked,
docs/task_log_ndc_extraction.md); this equal-proportional-effort rule is
this project's own construction, not a quoted NDC figure. A PLAUSIBLE
ALTERNATIVE: power abates MORE than proportionally, since grid
electricity is generally the cheapest sector to decarbonise (solar and
storage, against liquid-fuel transport or industrial process heat) --
under that alternative the power-sector cap would be TIGHTER than this
construction gives. Register L60 (docs/limitations_register.md),
docs/parameter_provenance.md row 11.

This method remains scope-appropriate in the sense the prior text
claimed: it asks gas-fired power to carry only its OWN measured share
of energy-sector emissions/abatement, not a share attributable to
hydro, DSM, or turbine efficiency (outside this model's scope) -- that
reasoning is unchanged; what changed is how the share itself is
computed.


UNCONDITIONAL vs CONDITIONAL (NDC 3.0, p.14)
---------------------------------------------
"Nigeria maintains its 20% unconditional objective and views the conditional
 reduction at 80% of the 29% and 32% reduction."

READING ADOPTED (Zeus's ruling, docs/task_log_btr1_denominator.md Part 2
item 4): "20% unconditional" is 20% OF THE 2018 BASE (573.5 MtCO2e), not
20% of the target itself.
    Unconditional: 20% of 573.5 = 114.7 MtCO2e economy-wide reduction.
    Conditional:   Full 168.2 / 184.9 MtCO2e targets (requires int'l finance).

The conditional-to-unconditional ratio scales the gas-power abatement:
    Conditional multiplier 2030 = 168.2 / 114.7 ≈ 1.467
    Conditional multiplier 2035 = 184.9 / 114.7 ≈ 1.612

WHY NOT THE ALTERNATIVE READING ("20% of the target itself"): that
reading sets unconditional = 0.20 x target, so the multiplier is
target / (0.20 x target) = 1 / 0.20 = 5.0 EXACTLY, in every year,
regardless of the target's value -- it cannot vary with the thing it is
supposedly measuring, and therefore measures nothing about the NDC; it
is an algebraic restatement of "20%" alone. The adopted reading varies
by year (1.467 vs 1.612) because it is anchored to a FIXED base while
the targets themselves change -- that variation is what makes it an
actual measurement of the conditional/unconditional gap.

DECLARED, UNRESOLVED: 573.5 MtCO2e -- the 2018 base the adopted reading
(and the whole conditional arm) is anchored to -- appears NOWHERE in
BTR1 or the NID (checked directly,
docs/task_log_btr1_denominator.md Part 1 item 2: zero matches for
"573.5" or "573,5" in either document's full text). NDC 3.0 itself
cites 614 Mt CO2e (BTR1's own 2018 inventory figure, Table 2.10) on
p.13, then states its own targets against 573.5 on p.14, two pages
later, with no stated adjustment reconciling the two. The
UNCONDITIONAL caps built by this script are now fully sourced --
POWER_SHARE_OF_ENERGY_2018 is inventory-over-inventory, and
168.2/184.9/22.0%/26.3% all trace to NDC 3.0 with page locators. The
CONDITIONAL caps retain exactly ONE unsourced anchor: 573.5 itself.
Declared, not fixed -- resolving it would require NDC 3.0 (or a future
BTR2/NID2) to state where 573.5 comes from, which no document checked
in this task does.

RESEARCH QUESTIONS ADDRESSED
    RQ-POL-1: What is the investment trajectory and cost differential between
                unconditional and conditional NDC 3.0 targets in the power sector?"
    RQ-POL-2: What is the implied MAC of moving from unconditional to conditional
              NDC 3.0 target in the power sector — and how does EaaS change that?


REQUIRED INPUT: a conv_none BASELINE diagnostics.json (TASK 32 rewrite)
------------------------------------------------------------------------------------
BASELINE_DIAG_PATH must point at a diagnostics.json produced by
scripts/01_run_baseline.py under the conv_none conversion arm. The builder
no longer matters for the value: since the A5 port (c549bba, 2026-10-01)
01_run_baseline.py builds with build_model_sliced(), and TASK 32 tested the
equivalence that this script's earlier wording ("an ANNUAL ... not a
sliced run") assumed could not be relied on: a sliced conv_none baseline
at HEAD gives E_base = 15.7450032 MtCO2/yr against the stored annual-
builder value 15.7450033 (fleet gas 86.3686406 vs 86.368641 TWh_th in
2025), and every one of the 150 cap values in emissions_cap.csv agrees to
a relative 8e-9 (largest absolute difference 0.073 tCO2 on caps of order
1e7). Comparison files: results/emissions_cap_check/ (untracked).

What DOES matter is conv_none, not the builder: the field read below,
`fleet_gas_to_power_twh_th_by_year`, is FLEET-ONLY by construction
(task_conversion_diagnostics, Part 1D and Part 2C rename) -- it reads
m.gas_to_power alone and never m.convert_to_power, the separate thermal-
draw variable Phase 2.8 gives converted OCGT-to-CCGT plant. Under conv_none
there is no conversion in either builder, so "fleet" and "total" are the
same thing and this script's assumption holds exactly. (The annual builder
cannot represent conversion at all -- build_model() raises on any available
conversion arm -- which is why the earlier wording said "annual"; the
requirement was really "no conversion".) Point the script at a conversion-
active baseline (conv_central/conv_low/conv_high), annual or sliced, and it
silently computes Ebase from a fraction of true gas thermal input,
understating every downstream NDC cap -- with no error raised, because
fleet_gas_to_power_twh_th_by_year is always present and always a
plausible-looking number.

Numeric illustration (task_conversion_diagnostics background, central
sliced run 02c17f55/conv_central, 2025): fleet thermal input alone is
10.5443 TWh_th; converted plant's OWN thermal draw is not in that field
at all, but is recoverable as converted electrical output / retrofit
efficiency = 21.7598 / 0.46 ~= 47.30 TWh_th_th. True total gas thermal
input for that run is therefore ~57.85 TWh_th, not 10.54 -- fleet-only
understates by a factor of ~5.5x here, not the "~86 TWh_th / ~8x" figure
an earlier draft of this note asserted; that figure could not be
reproduced from the same session's own anchor numbers and is not used
here. Either way the direction and order of magnitude are the same:
rebuilding NDC caps from a sliced, conversion-active baseline would
produce caps several times too tight, silently.

Provenance of the CURRENTLY COMMITTED data/cost/processed/
emissions_cap.csv (git commit 9700f3f, 2026-09-14, "existing four
unchanged"): computed by this script from the bare, untracked (see
.gitignore: results/) results/baseline/diagnostics.json, filesystem
mtime 2026-08-23 13:44 -- independently confirmed this session, not
merely repeated from an earlier note. That bare file's own
`fleet_gas_to_power_twh_th_by_year` (nee gas_to_power_twh_th_by_year)
values trace to the fuel-constrained annual baseline dispatch (gas is
fully subscribed there -- see optimize_experiments.py's own comment on
`gas_to_power == gas_avail` when the fuel constraint binds), which is in
turn set by the gas deliverability CSV (data/gas/processed/
gas_available_power_annual_twh_th.csv, "baseline" scenario column):
its values were last substantively set 2026-08-11 (commit 986f467,
"re-anchor physics ... gas deliverability to 89.27 TWh_th") and extended
to 2054 on 2026-08-18 (commit 74d34f2). The one commit since that
touched this CSV (5b811f2, 2026-08-28, "GMP 2026 policy-target family")
only ADDED new scenario columns and dropped out-of-horizon 2055 rows --
diff-verified this session to leave the "baseline" column's own values
untouched. So: the bare baseline diagnostics.json (Aug 23) was generated
from gas-deliverability data that remains, for the "baseline" scenario,
identical today. NOT "source CSVs dated 2026-08-15" as an earlier draft
of this note claimed -- no commit near that date touches gas data at
all; corrected here to the actual, checkable commits above.

Net: the caps in emissions_cap.csv are STALE IN PROVENANCE (last
regenerated from an Aug-23 baseline that itself sits behind several
since-superseded commits elsewhere in the model) but NOT STALE IN VALUE
for the reason above -- the one input that actually determines Ebase has
not changed. This does not extend to a hypothetical sliced,
conversion-active rebuild: see the understatement illustration above.

HOW TO REGENERATE (TASK 32; tested, see below). From the repository root:

  1. Run the baseline at conv_none with the standard arm block, i.e. the
     ITEM 1 invocation of scripts/run_phase_b.sh with `--conversion-case
     conv_none` in place of conv_central:
         python3 scripts/01_run_baseline.py <standard arms> --conversion-case
             conv_none --hydro-case hydro_flat --tariff-case tariff_central
             --om-case om_central
     It writes results/baseline/diagnostics__<tag>.json, where <tag> ends
     `__baseline__conv_none__hydro_flat`. NOTE this OVERWRITES the existing
     conv_none file of that name (results/ is untracked): copy it aside first
     if you want the old one.
  2. Copy that file to the bare path this script reads:
         cp results/baseline/diagnostics__<...>__conv_none__hydro_flat.json
            results/baseline/diagnostics.json
     (01_run_baseline.py cannot write the bare path itself; see
     task_epsilon_constraint.md, Part 4B. The bare file is unstamped and
     untracked -- register L61 -- so record which tag you copied.)
  3. python3 scripts/00_build_emissions_cap.py
     This OVERWRITES data/cost/processed/emissions_cap.csv, which is
     committed: review `git diff` before committing, and do not rebuild it
     for any reason except a changed input.

Tested 2026-10-08 in an isolated copy of the tree (so the committed cap and
results/baseline/ were not touched): steps 1-3 reproduce E_base 15.7450032
and all caps to 8e-9 relative (above). This supersedes the earlier text,
which told the reader to re-run 01_run_baseline.py "under conv_none" and
insisted the result be "an annual conv_none run" -- unsatisfiable after A5,
because 01_run_baseline.py builds sliced (register L75).
"""
import json
import sys
import pandas as pd
from pathlib import Path
import sys
from pathlib import Path
# ── Repo root ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.scenarios import MODEL_START_YEAR, MODEL_END_YEAR, REPORT_END_YEAR

#import sys, os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Emission factor (consistent with optimize_model.py) ───────────────────────
EF_TCO2_PER_TWH_TH = 182_300.0   # tCO2 per TWh_th (IPCC default, HHV basis = 0.1823 tCO2/MWh_th)
# ── Time horizon ──────────────────────────────────────────────────────────────
START_YEAR    = 2025
TARGET_YEAR   = 2030
TARGET_YEAR_2 = 2035
END_YEAR      = MODEL_END_YEAR

# ── NDC 3.0 economy-wide numbers (NDC 3.0 Section 3, pp.2, 14) ───────────────
NDC3_BASELINE_2018_MTCO2E          = 573.5
NDC3_ECONOMY_REDUCTION_2030_MTCO2E = 168.2   # 29.3% of baseline
NDC3_ECONOMY_REDUCTION_2035_MTCO2E = 184.9   # 32.2% of baseline

# ── Energy sector share of economy-wide abatement (NDC 3.0, p.15) ─────────────
ENERGY_SECTOR_SHARE_2030 = 0.220   # 22.0%
ENERGY_SECTOR_SHARE_2035 = 0.263   # 26.3%

# ── Gas-fired power's share of energy-sector emissions, 2018 ──────────────────
# INVENTORY-DERIVED (docs/task_log_btr1_denominator.md), not model-derived.
# Numerator: NID Table 3.9, "1.A.1.a Public electricity and heat
# production", 2018, p.36. Denominator: BTR1 Table 2.10, Sector 1
# "Energy", 2018, p.33. Both 2018, AR5 GWP, 2006 IPCC Guidelines + 2019
# Refinement, same inventory vintage (24/26 Dec 2024). Replaces the
# retired ENERGY_SECTOR_TOTAL_EMISSIONS_2018_MTCO2E = 217.0 (an
# unsourced "~38% of 573.5, round conservative estimate" tracing to no
# NDC 3.0/BTR1/NID page or table) and the GAS_POWER_SHARE = E_base /
# 217.0 construction, which divided a MODEL quantity by an UNSOURCED
# figure and therefore expressed no share of anything real. See the
# module docstring for the full account.
POWER_SHARE_OF_ENERGY_2018 = 16_234.0 / 193_346.72   # = 0.083963

# ── Unconditional reduction (NDC 3.0, p.14) ───────────────────────────────────
UNCONDITIONAL_ABS_REDUCTION_MTCO2E = 0.20 * NDC3_BASELINE_2018_MTCO2E  # 114.7

# ── Conditional-to-unconditional scaling ──────────────────────────────────────
CONDITIONAL_MULTIPLIER_2030 = NDC3_ECONOMY_REDUCTION_2030_MTCO2E / UNCONDITIONAL_ABS_REDUCTION_MTCO2E
CONDITIONAL_MULTIPLIER_2035 = NDC3_ECONOMY_REDUCTION_2035_MTCO2E / UNCONDITIONAL_ABS_REDUCTION_MTCO2E

# ── File paths ────────────────────────────────────────────────────────────────
BASELINE_DIAG_PATH = ROOT / "results" / "baseline" / "diagnostics.json"
OUT_PATH           = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"


# ── Helper: linear interpolation ─────────────────────────────────────────────
def _lerp(y, y0, v0, y1, v1):
    if y1 == y0:
        return v1
    frac = (y - y0) / (y1 - y0)
    return (1.0 - frac) * v0 + frac * v1


def main():
    # ── 1. Load baseline diagnostics ──────────────────────────────────────────
    if not BASELINE_DIAG_PATH.exists():
        raise FileNotFoundError(
            f"Missing baseline diagnostics at {BASELINE_DIAG_PATH}. "
            f"Run scripts/01_run_baseline.py first."
        )

    with open(BASELINE_DIAG_PATH, "r") as f:
        diag = json.load(f)

    # RENAMED from gas_to_power_twh_th_by_year (task_conversion_diagnostics,
    # Part 2C) -- see the module docstring section "REQUIRED INPUT" for why
    # this script's correctness depends on that field being fleet-only AND
    # on the source run being annual/conv_none.
    #
    # task_audit_fixes_p2 (audit finding A18): the RENAME above made this a
    # bare KeyError with no context the moment BASELINE_DIAG_PATH's file
    # predates it -- exactly the state the bare, untracked file is in right
    # now (results/ is gitignored; the copy on disk was generated before
    # this rename and still carries the OLD key). A `src/` rename being
    # invisible to a `scripts/` consumer reading a stale on-disk file is
    # the defect; this guard makes the NEXT such mismatch loud and
    # specific, naming both the missing key and the file, rather than
    # letting Python's own bare KeyError name only the key.
    if "fleet_gas_to_power_twh_th_by_year" not in diag:
        raise KeyError(
            f"'fleet_gas_to_power_twh_th_by_year' not found in "
            f"{BASELINE_DIAG_PATH}. This file was likely generated before "
            f"the task_conversion_diagnostics rename (it may still carry "
            f"the old key, 'gas_to_power_twh_th_by_year') -- regenerate it "
            f"with scripts/01_run_baseline.py under conv_none and place the "
            f"output at this bare path (see this module's docstring, "
            f"'REQUIRED INPUT' and 'HOW TO REGENERATE': it must be a "
            f"conv_none run; the builder no longer matters)."
        )
    gas_to_power = diag["fleet_gas_to_power_twh_th_by_year"]

    def _get(year):
        key_str = str(year)
        if key_str in gas_to_power:
            return float(gas_to_power[key_str])
        if year in gas_to_power:
            return float(gas_to_power[year])
        raise KeyError(f"Year {year} not found in fleet_gas_to_power_twh_th_by_year")

    # ── 2. Modelled power-sector baseline emissions (tCO2/year) ───────────────
    Ebase_2025 = _get(START_YEAR)    * EF_TCO2_PER_TWH_TH
    Ebase_2030 = _get(TARGET_YEAR)   * EF_TCO2_PER_TWH_TH
    Ebase_2035 = _get(TARGET_YEAR_2) * EF_TCO2_PER_TWH_TH

    E_BASE_MTCO2 = Ebase_2025 / 1e6   # annual baseline in MtCO2

    # ── 3. Gas-power share of energy-sector emissions -- a fixed,
    # inventory-derived constant. Does NOT depend on E_base (unlike the
    # retired construction); see module docstring and
    # POWER_SHARE_OF_ENERGY_2018's own comment above. ─────────────────
    GAS_POWER_SHARE = POWER_SHARE_OF_ENERGY_2018

    # ── 4. NDC 3.0 energy-sector abatement targets ───────────────────────────
    energy_abatement_2030 = ENERGY_SECTOR_SHARE_2030 * NDC3_ECONOMY_REDUCTION_2030_MTCO2E
    energy_abatement_2035 = ENERGY_SECTOR_SHARE_2035 * NDC3_ECONOMY_REDUCTION_2035_MTCO2E

    # ── 5. Gas-power abatement (unconditional) ────────────────────────────────
    gas_abatement_uncond_2030 = GAS_POWER_SHARE * energy_abatement_2030
    gas_abatement_uncond_2035 = GAS_POWER_SHARE * energy_abatement_2035

    # ── 6. Gas-power abatement (conditional — scaled by ambition ratio) ───────
    gas_abatement_cond_2030 = gas_abatement_uncond_2030 * CONDITIONAL_MULTIPLIER_2030
    gas_abatement_cond_2035 = gas_abatement_uncond_2035 * CONDITIONAL_MULTIPLIER_2035

    # ── 7. Power-sector caps ──────────────────────────────────────────────────
    ndc3_uncond_cap_2030 = max(E_BASE_MTCO2 - gas_abatement_uncond_2030, 0.0) * 1e6
    ndc3_uncond_cap_2035 = max(E_BASE_MTCO2 - gas_abatement_uncond_2035, 0.0) * 1e6
    ndc3_cond_cap_2030   = max(E_BASE_MTCO2 - gas_abatement_cond_2030, 0.0) * 1e6
    ndc3_cond_cap_2035   = max(E_BASE_MTCO2 - gas_abatement_cond_2035, 0.0) * 1e6

    # ── 8. Build scenario definitions ─────────────────────────────────────────
    SCENARIOS = {
        # ── NDC 3.0 (proportional top-down) ────────────────────────────────
        "ndc3_unconditional": {
            "type":        "ndc3_proportional",
            "cap_2030":    ndc3_uncond_cap_2030,
            "cap_2035":    ndc3_uncond_cap_2035,
            "description": (
                f"NDC 3.0 proportional: gas-power share ({GAS_POWER_SHARE*100:.2f}%) "
                f"of energy-sector abatement (self-funded)"
            ),
        },
        "ndc3_conditional": {
            "type":        "ndc3_proportional",
            "cap_2030":    ndc3_cond_cap_2030,
            "cap_2035":    ndc3_cond_cap_2035,
            "description": (
                f"NDC 3.0 proportional: gas-power share ({GAS_POWER_SHARE*100:.2f}%) "
                f"of energy-sector abatement (conditional, int'l finance)"
            ),
        },
        # ── NDC 2.0 (BAU-relative) ─────────────────────────────────────────
        "ndc2_unconditional": {
            "type":          "bau_relative",
            "bau_mult_2030": 0.80,
            "description":   "NDC 2.0 unconditional: -20% vs projected 2030 power-sector BAU",
        },
        "ndc2_conditional": {
            "type":          "bau_relative",
            "bau_mult_2030": 0.53,
            "description":   "NDC 2.0 conditional: -47% vs projected 2030 power-sector BAU",
        },
        # ── Plan 4.2: cap-shape sensitivity, additive arm ────────────────────
        # 2025-2035 IDENTICAL to ndc3_conditional (same cap_2030/cap_2035
        # anchors, referenced live from that scenario's own values below,
        # never re-derived). From 2036 the cap declines LINEARLY from the
        # 2035 anchor toward zero at 2060.
        #
        # The 2060 endpoint is NOT an analyst choice: net-zero-2060 is
        # Nigeria's own stated national target, so the endpoint is
        # policy-anchored. The LINEAR INTERPOLATION between the 2035 NDC
        # anchor and that endpoint IS an analyst construction -- in the same
        # sense plan 4.1 documents the proportional-apportionment rule as
        # "your methodological construction, not something in NDC 3.0,"
        # this trajectory's shape between two real anchors is ours, not
        # NDC 3.0's. Reported as a sensitivity (plan 4.2), not adopted as
        # central, three days before the freeze.
        #
        # Buffer years 2046-2054 CONTINUE the decline rather than holding
        # flat. This deliberately departs from the flat-hold buffer
        # convention every other scenario in this file uses (see the D10
        # comment in the two branches below): a non-binding buffer under a
        # DECLINING in-window cap would reopen exactly the arbitrage sink
        # D10 closed (selfgen routed into the buffer where CO2 is free),
        # and a flat hold at the 2045 value would contradict this
        # trajectory's own declining premise.
        "ndc3_conditional_decline": {
            "type":        "ndc3_declining",
            "cap_2030":    ndc3_cond_cap_2030,
            "cap_2035":    ndc3_cond_cap_2035,
            "description": (
                "NDC 3.0 proportional (conditional) through 2035, identical "
                "to ndc3_conditional; linear decline from the 2035 cap "
                "toward Nigeria's own net-zero-2060 target thereafter "
                "(plan 4.2 cap-shape sensitivity, buffer continues the "
                "decline)"
            ),
        },
    }

    # ── 9. Build annual cap trajectories ──────────────────────────────────────
    rows = []

    for scen_name, scen_cfg in SCENARIOS.items():
        scen_type = scen_cfg["type"]

        # ── NDC 2.0: BAU-relative ─────────────────────────────────────────
        if scen_type == "bau_relative":
            bau_mult = scen_cfg["bau_mult_2030"]
            cap_2030 = bau_mult * Ebase_2030

            for y in range(START_YEAR, END_YEAR + 1):
                if y <= START_YEAR:
                    cap_y = Ebase_2025
                elif y <= TARGET_YEAR:
                    cap_y = _lerp(y, START_YEAR, Ebase_2025, TARGET_YEAR, cap_2030)
                else:
                    cap_y = cap_2030
                # Buffer years (post-reporting-window) are HELD FLAT at the
                # terminal in-window cap, NOT released to 1e18.
                #
                # The original design released them, on the reasoning that a
                # binding cap outside the reporting window would distort
                # investment inside it. That was correct until Phase 2.6 added
                # private self-generation. With a priced backstop, an UNCAPPED
                # buffer is itself an arbitrage window: the optimiser routes
                # essentially all selfgen into 2046-2054 where its CO2 is free,
                # and pays VoLL to shed load in the capped years instead.
                # Observed: ~358 TWh of selfgen against ~360 TWh of permissible
                # buffer-window shortfall. That is not a coincidence.
                #
                # Flat continuation matches the treatment already applied to
                # CAPEX in io.py::_extend_capex_series, for the same reason:
                # buffer years must be neither cheaper nor freer than the last
                # reported year, or they pull behaviour out of the window.
                # Buffer-year caps are never reported.

                rows.append({
                    "year":     y,
                    "scenario": scen_name,
                    "cap_tco2": max(cap_y, 0.0),
                })
            continue

        # ── NDC 3.0: proportional top-down ────────────────────────────────
        if scen_type == "ndc3_proportional":
            cap_2030 = scen_cfg["cap_2030"]
            cap_2035 = scen_cfg["cap_2035"]

            for y in range(START_YEAR, END_YEAR + 1):
                if y <= START_YEAR:
                    cap_y = Ebase_2025   # unconstrained in year 0
                elif y <= TARGET_YEAR:
                    cap_y = _lerp(y, START_YEAR, Ebase_2025, TARGET_YEAR, cap_2030)
                elif y <= TARGET_YEAR_2:
                    cap_y = _lerp(y, TARGET_YEAR, cap_2030, TARGET_YEAR_2, cap_2035)
                else:
                    cap_y = cap_2035     # held flat after 2035

                # Buffer years (post-reporting-window) are HELD FLAT at the
                # terminal in-window cap, NOT released to 1e18. See the
                # equivalent block in the bau_relative branch above for the
                # full rationale (correction plan 2.6, D10): an uncapped
                # buffer is an arbitrage sink once selfgen is priced.

                rows.append({
                    "year":     y,
                    "scenario": scen_name,
                    "cap_tco2": max(cap_y, 0.0),
                })
            continue

        # ── Plan 4.2: NDC 3.0 conditional, declining beyond 2035 ───────────
        if scen_type == "ndc3_declining":
            cap_2030 = scen_cfg["cap_2030"]
            cap_2035 = scen_cfg["cap_2035"]
            # Nigeria's own stated national target (policy-anchored, not
            # analyst-chosen -- see the SCENARIOS comment above).
            NET_ZERO_YEAR = 2060

            for y in range(START_YEAR, END_YEAR + 1):
                if y <= START_YEAR:
                    cap_y = Ebase_2025   # unconstrained in year 0
                elif y <= TARGET_YEAR:
                    cap_y = _lerp(y, START_YEAR, Ebase_2025, TARGET_YEAR, cap_2030)
                elif y <= TARGET_YEAR_2:
                    cap_y = _lerp(y, TARGET_YEAR, cap_2030, TARGET_YEAR_2, cap_2035)
                else:
                    # Linear decline from the 2035 anchor toward zero at
                    # 2060. UNLIKE the two branches above, this is NOT held
                    # flat past 2045 -- buffer years 2046-2054 continue the
                    # same decline. See the SCENARIOS comment above for why:
                    # a flat buffer under a declining in-window cap would
                    # reopen the D10 arbitrage sink (uncapped-relative-to-
                    # trend years absorbing selfgen at near-zero CO2 cost).
                    cap_y = cap_2035 * (
                        1.0 - (y - TARGET_YEAR_2) / (NET_ZERO_YEAR - TARGET_YEAR_2)
                    )

                rows.append({
                    "year":     y,
                    "scenario": scen_name,
                    "cap_tco2": max(cap_y, 0.0),
                })
            continue

    # ── 10. Write output ──────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    # ── 11. Console summary ───────────────────────────────────────────────────
    print("=" * 70)
    print("  Emissions cap written — NDC 3.0 (proportional) + NDC 2.0 (BAU-relative)")
    print("=" * 70)

    print(f"\nNDC 3.0 economy-wide 2018 baseline:    {NDC3_BASELINE_2018_MTCO2E:.1f} MtCO2e")
    print(f"Economy-wide reduction target 2030:     {NDC3_ECONOMY_REDUCTION_2030_MTCO2E:.1f} MtCO2e "
          f"({NDC3_ECONOMY_REDUCTION_2030_MTCO2E/NDC3_BASELINE_2018_MTCO2E*100:.1f}%)")
    print(f"Economy-wide reduction target 2035:     {NDC3_ECONOMY_REDUCTION_2035_MTCO2E:.1f} MtCO2e "
          f"({NDC3_ECONOMY_REDUCTION_2035_MTCO2E/NDC3_BASELINE_2018_MTCO2E*100:.1f}%)")

    print(f"\nEnergy sector abatement share (NDC 3.0 p.15):")
    print(f"  2030: {ENERGY_SECTOR_SHARE_2030*100:.1f}% -> {energy_abatement_2030:.1f} MtCO2e")
    print(f"  2035: {ENERGY_SECTOR_SHARE_2035*100:.1f}% -> {energy_abatement_2035:.1f} MtCO2e")

    print(f"\nGas-power share of energy-sector emissions (inventory-derived, 2018):")
    print(f"  NID 1.A.1.a Public electricity and heat production = 16,234.0 kt CO2e")
    print(f"  BTR1 Table 2.10 Sector 1 Energy                    = 193,346.72 kt CO2e")
    print(f"  Gas-power share = {GAS_POWER_SHARE*100:.4f}%")
    print(f"  (E_base, for reference -- no longer part of this ratio: {E_BASE_MTCO2:.3f} MtCO2/yr)")

    print(f"\nGas-power abatement targets:")
    print(f"  Unconditional 2030: {gas_abatement_uncond_2030:.3f} MtCO2e "
          f"-> cap = {ndc3_uncond_cap_2030/1e6:.3f} MtCO2/yr")
    print(f"  Unconditional 2035: {gas_abatement_uncond_2035:.3f} MtCO2e "
          f"-> cap = {ndc3_uncond_cap_2035/1e6:.3f} MtCO2/yr")
    print(f"  Conditional 2030:   {gas_abatement_cond_2030:.3f} MtCO2e "
          f"-> cap = {ndc3_cond_cap_2030/1e6:.3f} MtCO2/yr")
    print(f"  Conditional 2035:   {gas_abatement_cond_2035:.3f} MtCO2e "
          f"-> cap = {ndc3_cond_cap_2035/1e6:.3f} MtCO2/yr")

    print(f"\nConditional multipliers (ambition scaling):")
    print(f"  2030: {CONDITIONAL_MULTIPLIER_2030:.3f}×  "
          f"(168.2 / 114.7)")
    print(f"  2035: {CONDITIONAL_MULTIPLIER_2035:.3f}×  "
          f"(184.9 / 114.7)")

    print(f"\nModelled power-sector baseline (gas-fired generation):")
    print(f"  Ebase 2025: {Ebase_2025/1e6:.3f} MtCO2/yr")
    print(f"  Ebase 2030: {Ebase_2030/1e6:.3f} MtCO2/yr")
    print(f"  Ebase 2035: {Ebase_2035/1e6:.3f} MtCO2/yr")

    print()
    for scen_name, scen_cfg in SCENARIOS.items():
        subset = df[df["scenario"] == scen_name]
        cap_2030 = subset.loc[subset["year"] == TARGET_YEAR,   "cap_tco2"].values[0]
        cap_2035 = subset.loc[subset["year"] == TARGET_YEAR_2, "cap_tco2"].values[0]
        abate_2030 = Ebase_2030 - cap_2030
        abate_2035 = Ebase_2035 - cap_2035
        print(f"  Scenario: {scen_name}")
        print(f"    {scen_cfg['description']}")
        print(f"    Power cap 2030: {cap_2030/1e6:.3f} MtCO2/yr  "
              f"(abatement: {abate_2030/1e6:.3f} MtCO2/yr)")
        print(f"    Power cap 2035: {cap_2035/1e6:.3f} MtCO2/yr  "
              f"(abatement: {abate_2035/1e6:.3f} MtCO2/yr)")
        print()

    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    # TASK 53 section 4: the two paths can be overridden so the cap trajectory can be rebuilt from ANOTHER baseline
    # diagnostics file into ANOTHER output (the builder-agreement test, scripts/57_). With no flag the behaviour is
    # unchanged: read results/baseline/diagnostics.json, overwrite the committed data/cost/processed/emissions_cap.csv.
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--baseline-diagnostics", type=Path, default=None)
    _ap.add_argument("--out", type=Path, default=None)
    _a = _ap.parse_args()
    if _a.baseline_diagnostics is not None:
        BASELINE_DIAG_PATH = _a.baseline_diagnostics
    if _a.out is not None:
        OUT_PATH = _a.out
    main()
