# Currency and rate-basis convention

**Correction plan item 3.1. Pulled forward from Phase 3 because genset fuel prices
are natively naira-denominated and cannot be converted without it.**

Status: `[DONE]` for the currency convention. `[OPEN]` for the deflator base year
and the Fisher conversion of the financing rates — see §5 and §6.

---

## 1. The declared convention

**The model is real US dollars throughout, at a single declared base year.**
Every cost parameter entering `load_econ()` or any `*_cost.csv` is a real-USD
quantity. No naira value appears anywhere in `src/`.

Two rules follow, and both are load-bearing:

1. **Naira series are converted month-by-month at the matched-month exchange
   rate, never end-to-end.** Converting a period mean at a spot rate, or a
   current naira price at a current rate to represent a historical average,
   silently attributes exchange-rate movement to the commodity.
2. **Every rate used in a discount factor must be real USD.** A nominal-naira
   rate and a nominal-USD rate cannot be blended into one WACC and applied to
   real-USD cashflows. See §6.

## 2. Why real USD is the right basis for Nigerian fuel — evidenced, not assumed

The original concern was that declaring real USD would impose a foreign unit on
a naira economy. The monthly conversion shows the opposite: the **USD price is
the stable quantity and the naira price is the noisy one.**

Over December 2024 – February 2026 (15 months), converting each month's NBS
national average at that month's mean CBN NFEM closing rate:

| Fuel | Mean USD/l | Std dev | CV | Min | Max |
|---|---|---|---|---|---|
| AGO (diesel) | **1.0157** | 0.0839 | **8.3%** | 0.8528 | 1.1696 |
| PMS (petrol) | **0.7311** | 0.0635 | **8.7%** | 0.6437 | 0.8283 |

Across that window the naira moved from 1,553.7 to 1,356.3 per USD (−12.7%) and
both naira pump prices swung by roughly ±20%, yet both USD prices held within
±9%. Nigerian refined product is priced off international parity; the naira price
is that parity multiplied by an exchange rate.

**Corroboration from market structure.** On 13 July 2026 Dangote Refinery
suspended naira-denominated sales and moved to USD pricing — petrol $0.779/l,
diesel $1.087/l — citing insufficient crude through the naira-for-crude
programme. The dominant domestic refiner now denominates in dollars. Its diesel
gate price sits within 7% of the independently derived clean-window mean of
$1.0157. Two different methods, two different sources, the same number.

**This converts the currency convention from a modelling assumption into an
observed property of the market, and it belongs in the methods chapter.**

## 3. The anchor window, and what is excluded

**Anchor: December 2024 – February 2026.** Fifteen months, post-deregulation,
post-float, pre-shock.

**Excluded from the anchor: March – May 2026.** AGO rose from $1.193 to $1.818 to
$2.392/l; PMS from $0.933 to $1.126 to $1.165. This was a crude shock (Brent
approximately $112–120/bbl on Middle East disruption), not an FX effect — the
naira was *appreciating* throughout. By mid-June 2026 diesel was back to
approximately $1.21/l and by 10 August 2026 to approximately NGN 1,661/l
(approximately $1.22 at the August NFEM mean of 1,360.11).

A trailing-twelve-month mean would give AGO $1.209 with CV 35.4%. That is a
contaminated statistic: it prices a transient shock as if it were the level.

**The excluded months are retained as the high arm.** An observed crude-shock
outcome with a documented cause is a stronger high arm than an assumed
percentage uplift.

## 4. Data gaps declared

- **FX coverage begins December 2024.** NBS fuel prices run from January 2023
  (AGO) and October 2023 (PMS). Those earlier months cannot be converted without
  I&E/NFEM rates back to at least mid-2023. They are not needed for the anchor
  and are carried in the raw workbook unconverted.
- **PMS June and July 2026 are not yet collected.** The series terminates at
  May 2026 for PMS and May 2026 for AGO.
- **Monthly FX means are computed from 12–23 daily closing observations per
  month.** August 2026 has only 12 (partial month) and is not used in the anchor.

## 5. Real base year — `[OPEN]`

The model's other cost parameters (NREL ATB solar and storage CAPEX) are real USD
at the ATB vintage's base year. The fuel anchor above is a nominal-USD average
over Dec-2024 to Feb-2026, whose midpoint is roughly mid-2025.

**Action required:** confirm the ATB vintage's stated base year, then deflate the
fuel anchor to it using the US GDP implicit price deflator (or CPI, declared).
The adjustment is small — one to two years of US inflation — but it must be
stated, not skipped, because the whole point of this memo is that units are
declared rather than assumed.

## 6. Financing rates — `[OPEN]`, blocks 3.2

The 18% commercial anchor and 6% concessional rate currently in
`blended_finance_scenarios()` have not been placed on a basis.

- 18% is almost certainly **nominal naira**. Nigerian inflation has run 20–35%;
  18% *real USD* would be an extreme figure for any asset class.
- 6% is presumably **nominal USD** DFI lending.

Blending these into one WACC and applying it to real-USD cashflows is
dimensionally incoherent, and the bankability frontier that results is an
artefact of unit mixing rather than a finding.

**Two admissible routes, pick one and declare it:**

1. **Fisher conversion.** Convert the naira rate to real terms using a declared
   Nigerian inflation expectation, then to USD using a declared long-run real
   exchange-rate assumption. Requires two assumptions, both of which must be
   swept.
2. **Source USD-denominated Nigerian power financing directly** — DFI term
   sheets, Nigerian IPP financial closes, the Nigeria Infrastructure Debt Fund,
   InfraCredit-guaranteed issuances. One assumption instead of two, and it is
   observed rather than inferred. **Preferred.**

Note that §2 makes route 2 more attractive than it first appears: if refined
product is already dollar-priced and USD prices are stable across a 12.7% naira
move, then USD-denominated financing is the natural instrument for this sector
and should be findable.

## 7. Files

| File | Contents |
|---|---|
| `data/cost/raw/fuel_price_monthly_usd.csv` | Monthly NBS AGO and PMS in NGN, monthly mean NFEM rate with daily observation count, and the matched-month USD conversion. `in_clean_window` flags the anchor. |
| `data/cost/raw/genset_lcoe_buildup.xlsx` | Sheet `FX_and_fuel` reproduces the above with live formulas and computes the anchor and high-arm values. |

---

*Sources: NBS Premium Motor Spirit Price Watch and Automotive Gas Oil Price Watch
(monthly national averages, collected across 774 LGAs). CBN NFEM daily closing
rates. Dangote Refinery USD pricing announcement, 13 July 2026.*
