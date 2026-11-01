"""
Single source of truth for the GAS-1 / GAS-2 gas-price benchmarks.

Before this module the benchmark dictionary and MMBTU_PER_TWH_TH were
hand-typed at four sites (06_run_gas1_shadow_benchmarks.py,
optimize_experiments.run_gas_regime_ndc_matrix, and twice in
07_run_gas2_eaas_gas_relief.py); register L35 item 1. Every site now
imports from here. Nothing else may define either name.

Units: the model's gas shadow price is USD per TWh_th at the domestic
delivery point. Benchmarks are entered in USD/MMBtu and converted with
MMBTU_PER_TWH_TH.

Sourcing classification per entry: SOURCED (a named primary source),
ASSUMED (analyst choice, direction of bias stated), PLACEHOLDER (a
number typed so the code can run, with NO source behind it -- Zeus to
source before anything downstream is cited).
"""

# 1 MMBtu = 293.071 kWh_th  =>  1 TWh_th = 1e9 / 293.071 MMBtu = 3,412,142 MMBtu
MMBTU_PER_TWH_TH = 1e9 / 293.071


# ------------------------------------------------------------------
# LNG netback derivation (USD/MMBtu)
#
#   netback = delivered_price - liquefaction - shipping [- regas]
#
# The model's gas shadow price is a marginal value at the DOMESTIC
# delivery point; a delivered LNG price in Asia/Europe includes
# liquefaction, shipping and (on some price bases) regas that an export
# project must pay before the gas is worth anything at the Nigerian
# wellhead. Comparing the shadow price to the delivered price directly
# overstates the export side by that whole wedge (register L-GAS1,
# declaration 3).
#
# BASIS. Whether regas is deducted depends on where the delivered price
# is assessed, so the basis is a REQUIRED argument -- there is no default
# and no regas constant of zero (an unsourced zero that silently selected
# one answer was the A47/L35 shape):
#   "DES" -- delivered ex-ship, assessed BEFORE regasification (e.g.
#            JKM). Regas is not deducted.
#   "HUB" -- in-grid hub price, assessed AFTER regasification (e.g. TTF).
#            Regas/terminalling is deducted.
#
# SOURCING. The three cost components below are SOURCED but each is
# marked "vintage unverified, charter rates volatile, Zeus to confirm
# currency". The delivered price is SOURCED (interim), see below.
# PACK COVERAGE (docs/sourcing_pack_01_rates.md; secondary citations,
# vintage unverified): delivered price = section 4; NMDPRA domestic-price
# series = section 7; GECF liquefaction and World Bank shipping/regas
# components = section 8.
# ------------------------------------------------------------------

# Delivered reference price. SOURCED (interim) -- Japan landed LNG spot,
# ~10.05 USD/MMBtu in January 2024 (US LNG export prices were 6.41 in the
# same month); DES basis (assessed before regasification). Supersede with
# a JKM or TTF assessment over a stated window. A single month is one
# data point, not a band, and Jan 2024 is a single-window value: no
# low/high delivered band is asserted. Source figure as supplied by Zeus
# (docs/sourcing_pack_01_rates.md, section 4; secondary citation, NOT
# independently checked by Code).
# Because the anchor is a DES assessment the DES netbacks are the
# operative ones; the six-key DES/HUB design is retained until Zeus
# retires the HUB keys.
LNG_DELIVERED_USD_MMBTU = 10.05  # SOURCED (interim) -- Japan landed LNG spot, Jan 2024, DES basis; supersede with a JKM/TTF assessment over a stated window

# Liquefaction, USD/MMBtu. SOURCED -- GECF, "Rising Marginal Costs in LNG
# Liquefaction" (2026), capacity-weighted average across EXISTING
# projects; NLNG is a legacy facility so the existing-project range
# applies, not the proposed-project range (> 4). Central 2.86, band
# 1.0-3.5. Vintage unverified, Zeus to confirm currency.
LNG_LIQUEFACTION_LOW_USD_MMBTU = 1.0
LNG_LIQUEFACTION_CENTRAL_USD_MMBTU = 2.86
LNG_LIQUEFACTION_HIGH_USD_MMBTU = 3.5

# Shipping, Nigeria -> NW Europe, USD/MMBtu. SOURCED -- World Bank, "LNG
# for Emerging Markets" (Nigeria -> Ghana ~0.20, about 0.50 less than to
# Zeebrugge, hence ~0.70). Nigeria is short-haul to Europe. Charter rates
# are volatile; vintage unverified, Zeus to confirm currency. Single
# value: no band is asserted (the earlier 1.50-4.00 range and 2.75
# midpoint were wrong for this route and are deleted).
LNG_SHIPPING_USD_MMBTU = 0.70

# Regas / terminalling, USD/MMBtu, HUB basis ONLY. SOURCED -- World Bank,
# "LNG for Emerging Markets": Zeebrugge terminalling plus European grid
# access, ~0.70. Vintage unverified, Zeus to confirm currency.
LNG_REGAS_HUB_USD_MMBTU = 0.70

_BASES = ("DES", "HUB")


def lng_netback_usd_mmbtu(delivered, liquefaction, shipping, *, basis):
    """
    netback = delivered - liquefaction - shipping   (basis "DES")
    netback = delivered - liquefaction - shipping - regas   (basis "HUB")

    `basis` is REQUIRED ("DES" or "HUB"); anything else, or omission,
    raises. Regas is taken from LNG_REGAS_HUB_USD_MMBTU on HUB only.
    """
    if basis not in _BASES:
        raise ValueError(
            f"lng_netback_usd_mmbtu: basis must be one of {_BASES}, got {basis!r}")
    regas = LNG_REGAS_HUB_USD_MMBTU if basis == "HUB" else 0.0
    return delivered - liquefaction - shipping - regas


def _netbacks(basis):
    # The LOW netback pairs with the HIGH liquefaction cost and vice versa.
    return {
        "low":     lng_netback_usd_mmbtu(LNG_DELIVERED_USD_MMBTU, LNG_LIQUEFACTION_HIGH_USD_MMBTU,
                                         LNG_SHIPPING_USD_MMBTU, basis=basis),
        "central": lng_netback_usd_mmbtu(LNG_DELIVERED_USD_MMBTU, LNG_LIQUEFACTION_CENTRAL_USD_MMBTU,
                                         LNG_SHIPPING_USD_MMBTU, basis=basis),
        "high":    lng_netback_usd_mmbtu(LNG_DELIVERED_USD_MMBTU, LNG_LIQUEFACTION_LOW_USD_MMBTU,
                                         LNG_SHIPPING_USD_MMBTU, basis=basis),
    }


LNG_NETBACKS_DES_USD_MMBTU = _netbacks("DES")
LNG_NETBACKS_HUB_USD_MMBTU = _netbacks("HUB")


BENCHMARKS = {
    # SOURCED -- NMDPRA 2024 Domestic Base Price, power sector, $2.42,
    # rounded. Stale-high: the regulator cut to $2.13 in 2025 and $2.18 in
    # 2026. Bias: conservative (fewer exceedances).
    "dom_power_2p4":      2.4  * MMBTU_PER_TWH_TH,
    # SOURCED -- NMDPRA 2024 DBP, commercial sector, $2.92, rounded. Same
    # staleness and bias.
    "dom_commercial_2p9": 2.9  * MMBTU_PER_TWH_TH,
    # ASSUMED -- no regulated tier matches. The gas-based-industry price
    # is floored at $0.90 and capped at the DBP, so 3.3 sits above the
    # regulated ceiling. Unregulated industrial / willing-buyer proxy.
    # Bias: high, conservative.
    "dom_high_3p3":       3.3  * MMBTU_PER_TWH_TH,
    # ASSUMED -- opportunity-cost ladder, no single source. Bias: high,
    # conservative.
    "opp_3":              3.0  * MMBTU_PER_TWH_TH,
    "opp_5":              5.0  * MMBTU_PER_TWH_TH,
    "opp_7":              7.0  * MMBTU_PER_TWH_TH,
    # ASSUMED -- a DELIVERED LNG price, NOT a netback. Dimensionally not
    # comparable to a wellhead shadow price (see register L-GAS1,
    # declaration 3): it overstates the export side by the full
    # liquefaction + shipping + regas wedge. Retained for reference and
    # for continuity with prior runs; the comparable quantities are the
    # six netbacks below (DES and HUB basis). Reporting exceedance
    # against both the delivered price and the netbacks makes the wedge
    # visible.
    "lng_export_10":     10.0  * MMBTU_PER_TWH_TH,
    # DERIVED -- delivered - liquefaction - shipping [- regas], from the
    # SOURCED cost components above and the SOURCED (interim) delivered price.
    # Reported under BOTH bases (DES: no regas; HUB: regas deducted)
    # (the delivered anchor is DES, so the DES keys are the operative ones;
    # HUB retained for now). Interim until a JKM/TTF assessment supersedes.
    "lng_netback_des_low":     LNG_NETBACKS_DES_USD_MMBTU["low"]     * MMBTU_PER_TWH_TH,
    "lng_netback_des_central": LNG_NETBACKS_DES_USD_MMBTU["central"] * MMBTU_PER_TWH_TH,
    "lng_netback_des_high":    LNG_NETBACKS_DES_USD_MMBTU["high"]    * MMBTU_PER_TWH_TH,
    "lng_netback_hub_low":     LNG_NETBACKS_HUB_USD_MMBTU["low"]     * MMBTU_PER_TWH_TH,
    "lng_netback_hub_central": LNG_NETBACKS_HUB_USD_MMBTU["central"] * MMBTU_PER_TWH_TH,
    "lng_netback_hub_high":    LNG_NETBACKS_HUB_USD_MMBTU["high"]    * MMBTU_PER_TWH_TH,
}
