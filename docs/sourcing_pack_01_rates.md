> **Provenance header (added 2026-09-20).** Compiled 2026-09-20. Figures
> in this file are **secondary citations transcribed from web sources**;
> **no primary document has been verified against the repo.** Treat every
> number as "as supplied, unchecked" until a primary is read.

# §3.5 SOURCING PACK — paste to Claude Code

Sourced 2026-09-20 by Chat. Every figure below carries its source and a
direction of bias. **Two of the three adopted placeholders are wrong by
a material margin**, and one has a definitional ambiguity that must be
resolved before it can be sourced at all.

Apply these in `src/` as instructed, log each in
`docs/task_log_bstar_gas1.md`, and **run nothing** — a single re-run
pass happens after the whole task closes.

---

## 1. r_concessional — placeholder 0.06 is ~3× too high

**Nigeria is an IDA *blend* country**, so IDA Blend terms apply.

| | value | source |
|---|---|---|
| IDA21 Blend (Fixed Rate), USD, interest charge | **2.12%** | IDA Terms effective 2026-01-01 (FY26Q3) |
| Same, effective 2026-07-01 | **2.13%** | IDA Terms effective 2026-07-01 |
| Service charge, blend term | **0** (set at zero) | same |
| Maturity / grace | 25 yr / 5 yr | IDA21 repayment schedule |
| Blend floating alternative | IBRD Flexible Loan Group A **minus 250 bps** | same |

**Apply:** `r_concessional = 0.0212` (central), band `0.0212–0.0213`.
Comment must state: IDA21 Blend fixed-rate USD interest charge,
Nigeria as an IDA blend country, effective 2026-01-01, service charge
zero, 25-yr/5-yr grace; floating alternative is IBRD Group A − 250 bps.

**Direction of bias of the old placeholder:** 0.06 *overstated*
concessional cost by ~3×, which understated the value of concessional
finance and therefore **understated the EaaS/blended advantage**.
Correcting it will make EaaS look *better*. Declare that plainly — it
moves in the thesis's favour and a referee will check it.

**Scope caveat to record:** IDA21 blend is the *sovereign* on-lending
rate. If the concessional envelope in this model represents
project-level concessional finance (GCF/AfDB to an IPP) rather than
sovereign-backed lending, the applicable rate is higher. **State which
the model means.** The `env_dares` arm name suggests the World Bank
DARES programme, i.e. sovereign-backed IDA — in which case IDA21 blend
is exactly right.

---

## 2. r_commercial — placeholder 0.18 is defensible but slightly LOW

Climate Policy Initiative's emerging-market cost-of-capital study
derives cost of debt for climate projects by sovereign rating using
CAPM adjusted for country climate-investment risk:

| country | S&P | cost of debt (climate project) |
|---|---|---|
| Jamaica | B+ | **18.5%** |
| Bangladesh | BB− | **19.0%** |

Nigeria rates **at or below B**, i.e. below both comparators, so its
cost of debt should be **≥ 18.5%**.

Corroborating context: there is no recorded project-financed renewable
energy project in Nigeria and commercial bank loans are typically
short-term, three to seven years, except where credit enhancement is
provided (as in the Azura-Edo deal) — consistent with a high risk
premium and a short tenor.

**Apply:** keep `r_commercial = 0.18` as central, add
`0.185` (CPI Jamaica) and `0.19` (CPI Bangladesh) as the high band.
Comment must cite CPI (2023), *Cost of Capital for Renewable Energy
Investments in Developing Economies*, CAPM adjusted for country climate
investment risk premium, and note that Nigeria rates below both
comparators so 0.18 is likely conservative-low.

**Direction of bias:** understating commercial cost makes private
financing look cheaper and therefore **flatters EaaS** — the opposite
direction to the concessional error, and the one that needs declaring
most. The two errors partly offset in the blend, so **do not claim they
cancel**; report each separately.

**Basis check required (Code):** `bankability_tenor.md:191-192` computes
`r_comm_real = (1 + r_comm_nom) / (1 + us_inflation) − 1 = 0.088235…`,
which implies `r_comm_nom = 0.11`, **not 0.18**. Establish whether 0.18
enters directly or through a blend, and report which figure is actually
nominal-USD commercial. Do not change anything until that is resolved.

---

## 3. Collection rate — a DEFINITIONAL problem, not just an unsourced one

The model's arms are `no_friction` 1.00, `reform_target` 0.60,
`current` 0.35. **None of these matches DisCo collection efficiency**,
which NERC reports at 74–81%. They are measuring something else, and
the model does not say what.

NERC quarterly data (authoritative, current):

| quarter | collection efficiency | ATC&C loss | end-to-end retention (1 − ATC&C) |
|---|---|---|---|
| 2025/Q1 | 74.39% | 39.61% | 60.39% |
| 2025/Q2 | 76.07% | 37.92% | 62.08% |
| 2025/Q3 | 80.70% | 33.27% | 66.73% |
| 2025/Q4 | 79.36% | 34.90% | 65.10% |
| 2026/Q1 | 78.95% | 37.44% | 62.56% |

MYTO targets: **20.54%** ATC&C (2025) → 79.46% retention; **16.92%**
(2026) → 83.08% retention. A twelve-month aggregation of NERC's
quarterly reports for April 2025 to March 2026 gives about 35.9% ATC&C
loss, i.e. roughly 64 naira of every 100 of supplied electricity
ultimately collected.

**The three candidate definitions give very different numbers:**

| definition | current value | matches which arm? |
|---|---|---|
| (a) DisCo collection efficiency (cash / billed) | ~79% | none |
| (b) End-to-end retention (1 − ATC&C) | ~62.6% | `reform_target` 0.60 |
| (c) GenCo settlement / remittance rate | ~39% (2025) | `current` 0.35 |

`reform_target` 0.60 maps to (b); `current` 0.35 maps to (c). **Those
are different quantities**, so the arm set may be mixing two
definitions. That is a referee-visible defect.

**Do NOT change the values yet.** Instead:

- **Code:** report exactly how `disco_collection_rate` is used — what it
  scales, and whether it multiplies revenue, billed energy, or the
  budget. That determines which definition is correct. Log it. **STOP.**
- **Once the definition is known**, the sourced values are:
  - if (b): current **0.626** (2026/Q1), reform target **0.831**
    (MYTO 2026), no-friction 1.00
  - if (a): current **0.790**, reform target ~0.95, no-friction 1.00
  - if (c): current **~0.39**, keep the reform target as an author
    assumption and declare it as such

Cite NERC 2025/Q1–Q4 and 2026/Q1 quarterly reports, per the standing
instruction to use NNPC monthly reports and NERC quarterlies rather
than the press.

---

## 4. Delivered LNG price (closes the last `gas_benchmarks` placeholder)

Interim sourced anchor: Japan's landed LNG spot price reached
approximately **$10.05/MMBtu in January 2024**, while US LNG export
prices were **$6.41/MMBtu**.

**Apply as an interim** `delivered_des = 10.05`, marked
`SOURCED (interim) — Japan landed LNG spot, Jan 2024; supersede with a
JKM or TTF assessment over a stated window`. Note that this is a DES
Japan assessment, so on the **DES basis** — regas not deducted.

With the sourced components this gives:
`10.05 − 2.86 (liquefaction) − 0.70 (shipping) = **6.49 USD/MMBtu**`
DES netback central.

**Reminder of the component ranking already logged:** the liquefaction
band (1.0–3.5) is 3.57× the DES/HUB spread (0.70). Liquefaction is the
component whose sourcing matters most, and it is now sourced.

---

## 5. What this does NOT close

§3.5 has three blockers. This pack addresses blocker 3 (placeholder
rates) and blocker 1 was addressed by the B\* replacement. **Blocker 2
remains open and is not touched here:** the tariff grid has not been
recomputed against T\*(t, blend). `04_` computes one analytical T\* per
blend, explicitly "same for all rows within a blend", not a surface;
plan item 3.4 is `[OPEN]`; L17 is the live declaration and records that
F-26–F-29 and F3's 47,528 MW ceiling rest on it.

**So §3.5 cannot be closed by this pack.** Two of three blockers can be
marked addressed-pending-re-run; blocker 2 needs its own work item.

Also still open on the plan's sourcing ledger and NOT covered here:
solar cost adder, guarantee headroom, peak multiple, self-gen energy,
and the concessional envelope size (as distinct from its rate).

---

## 6. Instruction block for Code

> Apply the §3.5 sourcing pack:
>
> 1. **`r_concessional`**: 0.06 → **0.0212**, band 0.0212–0.0213. Cite
>    IDA21 Blend (Fixed Rate) USD interest charge, effective
>    2026-01-01; blend service charge zero; 25-yr maturity, 5-yr grace;
>    Nigeria an IDA blend country. Record the direction of bias: the old
>    value overstated concessional cost ~3×, understating the EaaS
>    advantage. Record the scope caveat: sovereign on-lending rate;
>    state whether the envelope is sovereign-backed (the `env_dares`
>    name suggests it is).
> 2. **`r_commercial`**: keep 0.18 central; add 0.185 and 0.19 as band,
>    cited to CPI (2023) EM cost-of-capital (Jamaica B+ 18.5%,
>    Bangladesh BB− 19.0%; Nigeria rates below both). Record that 0.18
>    is conservative-low and flatters EaaS — opposite direction to (1);
>    do not claim the two cancel. **Before editing, resolve the basis
>    question**: `bankability_tenor.md:191-192` implies
>    `r_comm_nom = 0.11`, not 0.18. Report which figure is actually the
>    nominal-USD commercial rate and where 0.11 comes from. **STOP on
>    this point if unresolved.**
> 3. **Collection rate**: change no values. Report how
>    `disco_collection_rate` is used — what it multiplies — so the
>    definition (collection efficiency vs 1−ATC&C vs GenCo settlement)
>    can be fixed. Log the NERC series (2025/Q1–2026/Q1 above) as the
>    sourced candidates. **STOP.**
> 4. **`src/gas_benchmarks.py`**: delivered price 10.0 PLACEHOLDER →
>    **10.05**, marked `SOURCED (interim) — Japan landed LNG spot,
>    Jan 2024, DES basis; supersede with a JKM/TTF assessment over a
>    stated window`. Recompute and log the six netback keys.
> 5. **Update `docs/merged_correction_plan_v2.md`'s §3.5 wording only
>    where it is factually wrong**: the item says "reconcile B\*
>    (6.13bn docstring vs 9.104bn coded)"; 6.13 was never a coded value
>    (L45 clause 2), so restate as *derive and document B\**. **Do not
>    change the `[PARTIAL]` `[MUST]` status** — blocker 2 (tariff grid
>    vs T\*(t, blend)) is untouched and the gate stands.
>
> Run nothing. Log every step. STOP at points 2 and 3 as marked.

---

## 7. NMDPRA Domestic Base Price series (added 2026-09-20)

Added on Zeus's instruction so that `src/gas_benchmarks.py` and register
L48 can cite it. Secondary citation, as supplied; not verified against
an NMDPRA primary document (see the header of this file).

| year | power sector (USD/MMBtu) | commercial sector (USD/MMBtu) |
|---|---|---|
| 2024 | 2.42 | 2.92 |
| 2025 | 2.13 | 2.63 |
| 2026 | 2.18 | 2.68 |

The model's `dom_power_2p4` and `dom_commercial_2p9` benchmarks round the
2024 values, so both are stale-high against the 2025-2026 series; the
bias is conservative (fewer exceedances).

---

## 8. LNG netback components (added 2026-09-20)

Added on Zeus's instruction to close the coverage gap: these components
feed `src/gas_benchmarks.py` and were previously traceable only to the
task instructions. Secondary citations, as supplied; **vintage
unverified** and not checked against a primary document (see the header
of this file). Charter and terminalling rates are volatile; Zeus to
confirm currency.

| component | value (USD/MMBtu) | source |
|---|---|---|
| liquefaction, central | 2.86 (band 1.0-3.5) | GECF, *Rising Marginal Costs in LNG Liquefaction* (2026); capacity-weighted average across existing projects |
| shipping, Nigeria -> NW Europe | 0.70 | World Bank, *LNG for Emerging Markets*; Nigeria -> Ghana ~0.20, about 0.50 below Zeebrugge |
| regas / terminalling, HUB basis only | 0.70 | World Bank, *LNG for Emerging Markets*; Zeebrugge terminalling plus European grid access |

NLNG is a legacy facility, so the existing-project liquefaction range
applies, not the proposed-project range (> 4).
