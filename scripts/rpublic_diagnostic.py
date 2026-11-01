#!/usr/bin/env python3
"""
r_public stance diagnostic — standalone, no repo or solver dependency.

PURPOSE
-------
r_public's identity is a MODELLING STANCE, not a fact, and no script can
settle a stance. What this script settles is whether the stance is
LOAD-BEARING: it shows how far the F1 capital charge moves across the
candidate interpretations of r_public.

  Stance A (tranche cost):      r_public = actual cost of the IDA money
                                to the sovereign. ~2.25% nominal USD.
  Stance B (opportunity cost):  r_public = what Nigeria must otherwise pay
                                for capital. Eurobond territory, ~9-11%.
  Stance C (handover placeholder): 6% "concessional nominal USD" —
                                included because it is what the project has
                                been carrying, and it may be an IBRD
                                (market) rate mislabelled as concessional.

If the spread in capital charge is small, declare one stance and move on.
If it is large, r_public needs an explicit sensitivity arm and the choice
must be argued in the text, not asserted.

WHAT IT DOES NOT DO
-------------------
Does not touch the repo. Does not solve. Does not know what the model's
DF_social is. Reports the OWNER-SIDE annuity only (the CRF leg of F1);
the social-discount leg is orthogonal and is deliberately excluded so the
two are not conflated.

SOURCES FOR THE RATES (record in the register, not here)
--------------------------------------------------------
  r_public Stance A : World Bank IDA Terms, SINGLE-CURRENCY USD blend rate.
                      IDA20 FY24Q3 (DARES vintage, Dec 2023): 2.80% all-in
                      (1.39% service + 1.41% interest), 30yr/5yr grace.
                      IDA21 FY27Q1 (current): 2.13% interest, service set to
                      zero, 25yr/5yr grace.
                      NOTE: the 150bp+75bp figures in the general notes are
                      SDR terms, NOT USD. Do not use them for a USD model.
  r_conc / r_comm   : project inputs, sourced separately by Zeus.
  US inflation      : Fisher deflator input, stated below, ASSUMED.
"""

# ----------------------------------------------------------------------
# INPUTS -- edit these, everything below is derived
# ----------------------------------------------------------------------

# Fisher deflator: real = (1 + nominal) / (1 + inflation) - 1
# ASSUMED. The model is real-USD, so every nominal rate must be deflated.
US_INFLATION = 0.023

# Candidate r_public interpretations, NOMINAL USD
R_PUBLIC_NOMINAL = {
    # SOURCED: World Bank IDA Terms, single-currency USD blend rates.
    # DARES credit is Dec-2023 -> IDA20 blend terms, FY24 Q3 table:
    #   service 1.39% + interest 1.41% = 2.80% USD, 30yr maturity / 5yr grace.
    "A_ida20_dec2023_dares":    0.0280,
    # IDA21 terms eff. 01-Jul-2026, FY27 Q1: blend interest 2.13% USD,
    # blend service charge SET TO ZERO. 25yr maturity / 5yr grace.
    # What a NEW concessional credit prices at today.
    "A_ida21_current":          0.0213,
    "C_handover_placeholder":   0.0600,   # possibly an IBRD/market rate
    "B_opportunity_eurobond":   0.0950,   # sovereign external cost of funds
    "B_opportunity_high":       0.1100,   # upper Eurobond territory
}

# For context: the private tranches F1 prices alongside r_public.
# NOMINAL. Basis (naira vs USD) is the open coherence question -- the 18%
# commercial anchor is almost certainly nominal NAIRA and must not be
# deflated by US inflation without an explicit FX/Fisher argument.
R_PRIVATE_NOMINAL = {
    "r_conc_placeholder": 0.0600,
    "r_comm_placeholder": 0.1800,   # BASIS SUSPECT -- see handover decision 4
}

# Asset lifetimes (years). SOURCING STATUS OPEN -- handover decision 5
# asks whether asset_lifetimes() values are sourced. The CRF depends on
# them, so an unsourced L propagates straight into the objective.
LIFETIMES = {"solar": 25, "gas": 30, "storage": 15}

# Representative unit capex, USD per MW.
# solar 905 USD/kW and gas 1317 USD/kW are the DERIVED NIRP figures.
CAPEX_USD_PER_MW = {"solar": 905_000, "gas": 1_317_000, "storage": 300_000}


# ----------------------------------------------------------------------
# CORE
# ----------------------------------------------------------------------

def crf(r: float, L: int) -> float:
    """Capital recovery factor. r=0 degenerates to straight-line 1/L."""
    if r <= 0:
        return 1.0 / L
    return r * (1 + r) ** L / ((1 + r) ** L - 1)


def to_real(nominal: float, inflation: float = US_INFLATION) -> float:
    """Fisher. Model is real-USD; nominal rates must be deflated."""
    return (1 + nominal) / (1 + inflation) - 1


def undiscounted_annuity_total(r: float, L: int, capex: float) -> float:
    """Total nominal charge over the full service window = CRF * capex * L.
    Not discounted -- the social-discount leg is deliberately excluded."""
    return crf(r, L) * capex * L


def banner(text: str) -> None:
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def main() -> None:
    banner("1. CRF SENSITIVITY TO r_public  (real rates, Fisher-deflated)")
    print(f"US inflation assumption: {US_INFLATION:.1%}  (ASSUMED)\n")
    print(f"{'stance':<28} {'nominal':>8} {'real':>8}", end="")
    for tech in LIFETIMES:
        print(f" {'CRF_' + tech:>10}", end="")
    print()
    print("-" * 74)

    for label, nom in R_PUBLIC_NOMINAL.items():
        real = to_real(nom)
        print(f"{label:<28} {nom:>7.2%} {real:>7.2%}", end="")
        for tech, L in LIFETIMES.items():
            print(f" {crf(real, L):>10.4f}", end="")
        print()

    banner("2. ANNUAL CAPITAL CHARGE PER MW  (USD/MW/yr, real)")
    print("How much the objective is charged each service year, per MW.\n")
    print(f"{'stance':<28}", end="")
    for tech in LIFETIMES:
        print(f" {tech:>14}", end="")
    print()
    print("-" * 74)

    for label, nom in R_PUBLIC_NOMINAL.items():
        real = to_real(nom)
        print(f"{label:<28}", end="")
        for tech, L in LIFETIMES.items():
            print(f" {crf(real, L) * CAPEX_USD_PER_MW[tech]:>14,.0f}", end="")
        print()

    banner("3. IS THE STANCE LOAD-BEARING?  (the decision-relevant number)")
    print("Ratio of lifetime undiscounted capital charge, each stance")
    print("against Stance A (IDA20 blend, Dec-2023 DARES vintage) as the base.\n")

    base_real = to_real(R_PUBLIC_NOMINAL["A_ida20_dec2023_dares"])
    print(f"{'stance':<28}", end="")
    for tech in LIFETIMES:
        print(f" {tech + ' x':>14}", end="")
    print()
    print("-" * 74)

    for label, nom in R_PUBLIC_NOMINAL.items():
        real = to_real(nom)
        print(f"{label:<28}", end="")
        for tech, L in LIFETIMES.items():
            base = undiscounted_annuity_total(base_real, L, CAPEX_USD_PER_MW[tech])
            this = undiscounted_annuity_total(real, L, CAPEX_USD_PER_MW[tech])
            print(f" {this / base:>13.3f}x", end="")
        print()

    banner("4. r_public AGAINST THE PRIVATE TRANCHES")
    print("The F1 spread that drives the concessional-first draw order.")
    print("WARNING: r_comm 18% is BASIS-SUSPECT (likely nominal naira).")
    print("Deflating it by US inflation is shown but is NOT defensible")
    print("without an explicit FX + Fisher argument -- handover decision 4.\n")

    L = LIFETIMES["solar"]
    print(f"Solar, L = {L} yr, capex {CAPEX_USD_PER_MW['solar']:,} USD/MW\n")
    print(f"{'rate':<28} {'nominal':>8} {'real':>8} {'CRF':>9} {'USD/MW/yr':>13}")
    print("-" * 74)

    combined = {"r_public (Stance A)": R_PUBLIC_NOMINAL["A_ida20_dec2023_dares"]}
    combined.update(R_PRIVATE_NOMINAL)
    for label, nom in combined.items():
        real = to_real(nom)
        print(f"{label:<28} {nom:>7.2%} {real:>7.2%} {crf(real, L):>9.4f}"
              f" {crf(real, L) * CAPEX_USD_PER_MW['solar']:>13,.0f}")

    banner("5. LIFETIME SENSITIVITY  (handover decision 5)")
    print("If asset_lifetimes() is unsourced, this is the exposure.")
    print("Solar CRF at Stance A real rate, across plausible L.\n")

    real = to_real(R_PUBLIC_NOMINAL["A_ida20_dec2023_dares"])
    print(f"{'L (yr)':>8} {'CRF':>10} {'USD/MW/yr':>14} {'vs L=25':>10}")
    print("-" * 74)
    ref = crf(real, 25)
    for L in (20, 25, 30, 35, 40):
        c = crf(real, L)
        print(f"{L:>8} {c:>10.4f} {c * CAPEX_USD_PER_MW['solar']:>14,.0f}"
              f" {c / ref:>9.3f}x")

    banner("READING THIS")
    print("""
Section 3 is the decision. If the spread across stances is small, pick
Stance A, declare it, move on. If it is large, r_public is load-bearing:
it needs its own sensitivity arm and the stance must be argued in the
text rather than asserted.

Section 5 is the quiet risk. An unsourced asset lifetime propagates into
every CRF and therefore into the objective, and it does so silently --
exactly the self-cancelling-error shape that has bitten this project
before. If L is ASSUMED, say so in the register with a direction.
""")


if __name__ == "__main__":
    main()
