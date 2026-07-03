"""
Blended Concessional Finance Module (EaaS Financing Structure)

Scope
-----
Represents the capital structure behind EaaS-financed distributed solar as a
BLEND of two tranches:

    (1) a CONCESSIONAL tranche  — DFI / climate-fund / CBN-intervention capital
                                  at a below-market rate, available only up to a
                                  finite envelope (concessional capital is scarce
                                  and rationed);
    (2) a COMMERCIAL tranche    — private debt + equity at the prevailing
                                  Nigerian market cost of capital.

The blended WACC that a private EaaS provider actually faces is the
capital-weighted average of the two tranche rates. That blended rate is what
should enter the bankability test (private_discount_rate in optimize_model.py).

Why a MODULE and not just a lower number
----------------------------------------
Setting private_discount_rate = 0.12 "because blended finance" is an assumption
smuggled in as a parameter. This module makes the assumption auditable:
the blended rate is DERIVED from (concessional share, concessional rate,
commercial rate), each of which is separately sourced and separately swept.

Scarcity (important)
--------------------
Concessional capital is not unlimited. The honest way to model a blend is to
cap the concessional tranche at a finite NPV envelope. Below that envelope the
provider borrows concessionally; above it, marginal capital reverts to the
commercial rate. This module exposes BOTH:

    - blended_wacc()          : the simple weighted-average rate (planning proxy,
                                used as private_discount_rate when the concessional
                                envelope is assumed to cover the deployed capital);
    - concessional_envelope_* : helpers to construct a finite concessional-capital
                                constraint in the LP, so the shadow price on that
                                envelope prices the marginal value of concessional
                                finance (dual-variable methodology, on-thesis).

CITATION DISCIPLINE
-------------------
Every rate below is a PLACEHOLDER pending a primary source the author has
personally verified. Do NOT ship these defaults.
  - r_commercial : Nigerian commercial power-sector WACC. anchored to the IEA/CCSI 
                    "Africa power-sector WACC >18% (2023)" 
                    statement, representing unblended Nigerian commercial capital. 
                    Defensible as the high-risk end.
  - r_concessional : DFI / CBN-intervention concessional rate. If citing the
                   CBN "7.5-10%" figure, VERIFY the URL and note whether it is a
                   market blend or a concessional-dominant intervention rate
                   (arithmetic shows 7.5-10% blended at an 18% commercial rate
                   implies a 75-100%+ concessional share — i.e. NOT a balanced
                   blend but a near-fully-concessional window).
  - concessional_share / envelope : from the actual facility being invoked
                   (e.g. a DFI programme size), not assumed.
"""

from __future__ import annotations


# ============================================================
# CORE: BLENDED WACC
# ============================================================

def blended_wacc(
    concessional_share: float,
    r_concessional: float,
    r_commercial: float,
) -> float:
    """
    Capital-weighted average of the concessional and commercial tranche rates.

    blended = share_conc * r_conc + (1 - share_conc) * r_comm

    Parameters
    ----------
    concessional_share : float in [0, 1]
        Fraction of the EaaS capital stack funded by concessional capital.
    r_concessional : float
        Concessional tranche cost of capital (fraction, e.g. 0.08).
    r_commercial : float
        Commercial tranche cost of capital (fraction, e.g. 0.18).

    Returns
    -------
    float
        Blended WACC (fraction).

    Notes
    -----
    This is the rate to feed private_discount_rate in build_model() WHEN you are
    assuming the concessional envelope covers the deployed EaaS capital (i.e. the
    scarcity constraint is slack). If concessional capital is binding, use the
    envelope formulation instead so the blend degrades to r_commercial at the
    margin.
    """
    if not (0.0 <= concessional_share <= 1.0):
        raise ValueError("concessional_share must be in [0, 1]")
    if r_concessional < 0 or r_commercial < 0:
        raise ValueError("rates must be non-negative")
    if r_concessional > r_commercial:
        # Not fatal, but almost certainly a mistake: concessional should be cheaper.
        raise ValueError(
            "r_concessional > r_commercial: concessional capital should be "
            "cheaper than commercial. Check inputs."
        )

    return (
        concessional_share * r_concessional
        + (1.0 - concessional_share) * r_commercial
    )


def implied_concessional_share(
    target_blended: float,
    r_concessional: float,
    r_commercial: float,
) -> float:
    """
    Inverse of blended_wacc: what concessional share is needed to hit a target
    blended rate. Use this to STRESS-TEST a quoted blended figure.

    share = (r_comm - target) / (r_comm - r_conc)

    A result > 1.0 means the target is UNREACHABLE even at 100% concessional
    capital — a strong signal the quoted "blended" rate is not a genuine blend.
    """
    if r_commercial <= r_concessional:
        raise ValueError("r_commercial must exceed r_concessional")
    return (r_commercial - target_blended) / (r_commercial - r_concessional)


# ============================================================
# SCARCITY: CONCESSIONAL CAPITAL ENVELOPE
# ============================================================

def concessional_envelope_npv(
    total_facility_usd: float,
    domestic_deployable_fraction: float = 1.0,
) -> float:
    """
    NPV size of the concessional capital available to this system's EaaS solar.

    Parameters
    ----------
    total_facility_usd : float
        Headline size of the concessional facility (NPV, USD). SOURCE THIS to a
        real programme (e.g. a named DFI / climate-fund facility for Nigerian
        grid-scale or distributed solar). Do not assume.
    domestic_deployable_fraction : float in [0, 1]
        Fraction of the facility realistically deployable to the grid-connected
        EaaS solar this model plans (facilities often span mini-grid, C&I, etc.).

    Returns
    -------
    float
        Concessional envelope (NPV, USD) usable as an LP constraint RHS.
    """
    if total_facility_usd < 0:
        raise ValueError("total_facility_usd must be non-negative")
    if not (0.0 <= domestic_deployable_fraction <= 1.0):
        raise ValueError("domestic_deployable_fraction must be in [0, 1]")
    return total_facility_usd * domestic_deployable_fraction


def resolve_private_rate(scenario: dict) -> float:
    """
    Single entry point for build_model(): returns the private_discount_rate to
    use in the bankability test, derived from the blended-finance parameters if
    present, else falling back to a directly-specified private rate.

    Precedence:
      1. If scenario has blended-finance keys, compute the blended WACC.
      2. Else use scenario['private_discount_rate'] as-is.

    This keeps optimize_model.py agnostic: it just calls resolve_private_rate().
    """
    keys = ("concessional_share", "r_concessional", "r_commercial")
    if all(k in scenario and scenario[k] is not None for k in keys):
        return blended_wacc(
            concessional_share=scenario["concessional_share"],
            r_concessional=scenario["r_concessional"],
            r_commercial=scenario["r_commercial"],
        )
    return float(scenario.get("private_discount_rate", 0.18))
