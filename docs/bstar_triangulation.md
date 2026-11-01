# B\* triangulation — task log

Role separation per TASK.md: Claude Code extracts, computes and
reports. No solves, no runners, no git commits, no model value changed.
Logged after each part.

Source PDF: `~/thesis-code/literature/NIRP_2024.pdf` (gitignored, not
committed). Extracted with `pdftotext -layout` (succeeded; no fallback
extractor needed). All page numbers below are the PRINTED page numbers
in the document's own footer/table-of-contents, cross-checked against
the document's own Table of Contents page listings (which independently
confirm every page number used below — see the cross-check note at the
end of this part).

---

## PART A — NIRP's investment costs, extracted verbatim

### Locating the section

The document has no single section literally headed "Investment costs
2024-45 – NIRP scenario" as a body-text heading with a table directly
under it. That exact string is the caption of **Figure 9** (a chart,
page 32): *"Figure 9: Investment costs 2024-45 – Draft NIRP (M11)
scenario"*. The figure itself is a chart (no printed per-year values
inside the image); the **prose immediately preceding it, on page 32**,
states the headline figures in words, and two separate numbered tables
elsewhere in the document — **Table 10** (page 36) and **Table 23**
(page 59) — give the same investment-cost data as printed numbers, by
technology and by year respectively. All three (prose, Table 10, Table
23) are reported below, cross-checked against each other.

### 1. Solar investment cost to 2045, and storage separately

**Prose, page 32** (immediately before Figure 9's caption), quoted in
full:

> "Figure 9 presents annual investment costs¹² required for the NIRP
> scenario to 2045. Investments pick up in the early years rising to
> $2 bn by 2030. From then onwards they stay within a range of $4 bn to
> $8 bn per year until 2040. Most of the investment is required in the
> later years, increasing to $14 bn-$15 bn in 2044 and 2045.
>
> By 2045 a total of **$122 bn** will be required, of which the largest
> expenditure will be on **solar PV at $56 bn**, hydro at $39 bn, gas at
> $16 bn, Gas+CCUS at $6 bn, and **storage at $3 bn**. Detailed year by
> year results are provided in Annex F."

**Table 10: Investment costs 2024-2045 by technology and zone (bn$),
page 36**, quoted in full (zone columns retained for completeness even
though only the Total column is needed here):

| bn$ | Zone 1 | Zone 2 | Zone 3 | Zone 4 | **Total** | % |
|---|---|---|---|---|---|---|
| Gas | 6 | 4 | 2 | 5 | 16 | 13% |
| Hydro | 7 | 3 | 26 | 3 | 39 | 32% |
| Gas+CCUS | – | – | – | 6 | 6 | 5% |
| BECCS | – | – | – | – | – | – |
| Hydrogen | 0.4 | – | – | – | 0.4 | – |
| Nuclear | – | – | – | – | – | – |
| **Solar PV** | 8 | 22 | 19 | 7 | **56** | **46%** |
| Solar CSP | – | – | – | – | – | – |
| Wind | 0.1 | 0.3 | 1 | – | 1 | 1% |
| DSM | 0.2 | 0.1 | 0.1 | 0.1 | 0.1 | 0% |
| **Storage** | 0.3 | 3 | – | 0.2 | **3** | **2%** |
| **Total** | 22 | 32 | 47 | 20 | **122** | |
| % | 18% | 27% | 39% | 17% | | |

**Confirmed: Solar PV = $56 bn, Storage = $3 bn, Total = $122 bn**,
identical across the prose (page 32) and Table 10 (page 36) — two
independent statements in the same document agree exactly.

**Table 23 cross-check (Annex F, page 59)** — the year-by-year table
the prose above points to. Summing the printed annual **Solar PV** row
(2024-2045, bn$; only non-zero years shown): 2033:1, 2034:2, 2035:1,
2036:1, 2037:2, 2038:3, 2039:5, 2040:6, 2041:5, 2042:5, 2043:8, 2044:8,
2045:8 → **sums to 55**, against the stated cumulative 56 — a 1 bn gap
fully consistent with each annual entry being independently rounded to
the nearest whole (or one-decimal) billion across 22 years. Summing the
**Storage** row (2041:0.1, 2042:0.4, 2043:0.3, 2044:2, 2045:1, all
other years zero) → **sums to 3.8**, against the stated cumulative
3 — here the rounding gap (0.8 bn) is larger and does not close as
cleanly; reported as-is, not resolved. Summing the **Total** row across
all 22 years → **sums to ≈120.2**, against the stated cumulative
122 — again consistent with rounding, not exact.

### 2. Basis

**Units:** `bn$` (billions of US dollars) throughout Table 10, Table
23, and the page-32 prose — not USD millions.

**Real/nominal and base year:** The phrase **"Note: $ are USD in 2022
prices"** appears three times in the document (page 7 — an unlabelled
Executive Summary table duplicating the data below; page 29, directly
under **Table 6: Summary of results – NIRP scenario**; and again near
**Table 11**'s sensitivity comparison). It is **not restated directly
adjacent to Table 10, Table 23, or the page-32 prose** — checked
directly, no such note appears within several lines of any of those
three. Table 6 (page 29) and Table 10 (page 36) draw on the same
underlying NIRP capex stream (see the NPV cross-check below), and the
document's `$` notation is used consistently throughout with no
alternative currency-basis statement anywhere — but per the standing
instruction not to infer a basis the document does not state at the
specific locator in question, **this report does not assert that the
2022-prices note covers Table 10/23/the page-32 prose**; it reports
that the note exists, where it is actually printed, and that it is not
printed a second time at the investment-cost tables specifically.

**Discounted or undiscounted:** **Table 23's raw annual figures sum to
the same total ($122 bn ≈ 120.2 bn, rounding) as Table 10's printed
Total and the page-32 prose**, with no NPV language anywhere near
either table — this is an **undiscounted cumulative sum** of nominal
annual capex. This is corroborated by a *different*, explicitly-labelled
figure elsewhere in the same document: **Table 6 (page 29) states "NPV
of capex bn$ 40" (63% of "NPV of total costs bn$ 63")**, for the same
NIRP central scenario, same 2024-45 window. **$40 bn (NPV) against
$122 bn (undiscounted cumulative) for what appears to be the same
capex stream** — reported as two separate, explicitly-sourced figures,
not reconciled into one number here.

**NIRP's own discount rate is not stated anywhere in the document as a
number.** Exhaustive search (`grep -i "discount"` over the full
extracted text) returns exactly one hit: page 35, in the sensitivities
discussion — *"a higher social rate of discount"* — a qualitative
mention (one of the sensitivity scenarios uses a higher rate) with no
numeric value given anywhere nearby or elsewhere in the document. **Per
FORBIDDEN: no rate is inferred or back-solved from the $40bn/$122bn
ratio.**

Footnote 12 (page 32, attached to "annual investment costs¹²"), quoted
in full: *"The investment costs refer to consist of build cost (capex),
which are assumed to be incurred in the year in which the project is
commissioned. In practice, investment costs for large projects are
spread over several years. The details of financing arrangements are
to be incorporated in the SIP."* This describes WHEN each year's capex
is recognised (at commissioning), not whether the $122bn total is
discounted — it does not resolve the discounting question, only the
timing-within-the-annual-series question.

### 3. Capacity the costs finance

**Table 6 / Executive Summary duplicate (pages 29 and 7), 2045 column**,
quoted in full:

| | Unit | 2045 |
|---|---|---|
| Installed capacity (incl. storage) | GW | 111 |
| RES capacity | GW | 83 |
| Storage capacity (incl. solar CSP storage) | GW | 3 |
| Storage energy capacity (incl. solar CSP storage) | GWh | 11 |
| Average storage duration | hrs | 3 |

"RES capacity" (83 GW) is renewables broadly (solar + hydro + wind +
BECCS etc., not solar alone) — no solar-PV-only MW figure is given in
this table.

**Table 21: Total installed capacity by technology (GW), page 57**, the
2045 column, quoted (note: the table's own printed column header reads
"(%)" while the table's title reads "(GW)" and the values are plainly
capacity, not percentages — an apparent labelling inconsistency in the
source document, quoted exactly as printed, not corrected):

| technology | 2045 |
|---|---|
| Gas | 18.5 |
| Hydro | 20.6 |
| Gas+CCUS | 2.3 |
| Hydrogen | 0.7 |
| **Solar PV** | **61.1** |
| Wind | 0.9 |
| DSM | 3.9 |
| **Storage** | **3.4** |
| **Total** | **111** |

**Solar PV = 61.1 GW, Storage = 3.4 GW at 2045** (Table 21) — the
Storage figure here (3.4 GW) is more precise than, but consistent with,
Table 6/Executive Summary's rounded 3 GW. Table 21 gives no separate
storage MWh column; the MWh figure (11 GWh) is only in Table 6, at the
rounded 3 GW storage-power basis, implying the "3 hrs" duration figure
already stated.

### 4. Public vs private financing split

**Not stated.** Exhaustive search for "public"/"private" financing
language near any investment-cost figure returns no split. The document
discusses IPP (independent power producer) financing **qualitatively**
in two places — the "14 Solar IPP projects" saga (page 40: PPAs signed
2016, none reached financial close, "the main one [barrier] being the
provision of government guarantees to mitigate developers' risk") and
a passing mention of "Solar IPP projects (with PPAs)" (page 23) — but
neither attaches a number, a share, or any quantitative split to the
$122bn / $56bn investment totals. **Per FORBIDDEN: no split is
inferred.** NIRP does not state what fraction of its investment
programme is publicly vs privately financed.

### 5. Annual investment requirement (cross-check against the cumulative figure)

Already used above (§1, §2) as the primary basis-determination evidence:

- **Prose, page 32**: "$2 bn by 2030... $4 bn to $8 bn per year until
  2040... $14 bn-$15 bn in 2044 and 2045" (total-investment, all
  technologies, not solar/storage-specific).
- **Table 23, page 59** (Annex F): full year-by-year breakdown by
  technology, 2024-2045, reproduced in relevant part in §1 above. This
  is a **printed table with numeric values**, not a chart — the annual
  figures are NOT read-off-chart; they are typeset numbers.
- Cross-checked against Table 10's cumulative Total (§1): sums are
  consistent with rounding, as reported.

No numeric annual investment figure appears ONLY in chart form; Figure
9 itself (the chart referenced by the page-32 prose) has no printed
axis values in the extracted text, but every figure the prose or a
table states is a typeset number, not a chart read-off. **Nothing in
this report is marked read-off-chart.**

### Locator cross-check

Every page number above was independently verified two ways: (1)
counting page-footer markers in the full `pdftotext -layout` dump
between successive printed page numbers, and (2) confirmed against the
document's own Table of Contents, which independently states: "Figure
9: Investment costs 2024-45 – Draft NIRP (M11) scenario ... 32",
"Table 10: Investment costs 2024-2045 by technology and zone (bn$) ...
36", "Table 21: Total installed capacity by technology (GW) ... 57",
"Table 23: Investment costs 2024 – 2045 (bn$) ... 59", "Table 6:
Summary of results – NIRP scenario ... 29". All five match the
footer-counting method exactly.

---

**STOP — Part A complete.** The basis (§2) is partially, not fully,
established: units, real-vs-nominal currency year, and
discounted-vs-undiscounted are all determined (undiscounted, 2022 USD
bn, most likely — see the caveat on the currency note's exact
locator); NIRP's own discount rate is confirmed absent as a number
anywhere in the document. Per TASK.md: *"If the basis in item 2 cannot
be established, the comparison cannot be made and Part C changes
shape"* — the basis here IS establishable (undiscounted, page-32/Table-10
figures, cross-checked three ways), so Part C can proceed on that
basis without a shape change, with the currency-year caveat carried
forward explicitly rather than silently assumed. Nothing was run,
solved, or committed. Awaiting Zeus's review of Part A before Part B.

---

## Zeus's ruling on Part A, and three amendments before Part B

Part A accepted. Three amendments applied throughout Part B/C below:

1. Use both NIRP bases (NPV $40bn, Table 6 p.29; cumulative $122bn,
   Table 10 p.36), not one. Report the model against both. Report
   NIRP's own implied NPV/cumulative ratio (40/122 = 0.328) beside the
   model's own NPV/undiscounted ratio at 8% — as a diagnostic
   comparison only, not to back-solve NIRP's discount rate.
2. Lead with unit costs, since the two plans are not the same plan in
   scale (NIRP solar 61.1 GW vs model 39,030 MW, about 57% less; NIRP
   storage 3.4 GW/11 GWh vs model 63,056 MWh, roughly 5.7× the storage
   energy). State plainly that the unit-cost comparison is the valid
   one and the total comparison is indicative only.
3. Carry Part A's two caveats through: the "$ in 2022 prices" note is
   printed at Table 6/Executive Summary, not at Table 10 or Table 23;
   and Table 23's Storage row sums to 3.8 against a stated 3 bn — the
   weaker of NIRP's two cross-checks.

---

## PART B — bottom-up from the reference plan

### Method note, read first: how the solar/storage split below was obtained

`results/capital_frontier/capital_frontier_raw.csv` gives, per swept
point, only `solar_total_mw` (final 2054 stock), `storage_mwh` (final
2054 stock), `realised_public_spend_usd` (NPV, combined across every
technology the public budget charges), and `npv_cost` (total system
cost) — no per-technology or per-vintage breakdown, and no separate
undiscounted total, for any point. `28_run_capital_frontier.py` calls
`extract_planning_diagnostics()`, which DOES compute a full breakdown
internally, but the script only pulls out the four fields above before
discarding the rest — confirmed by reading the script directly
(`scripts/28_run_capital_frontier.py:345-399`); no other saved file in
`results/capital_frontier/` carries a finer breakdown.

**A verified, not assumed, cross-reference closes this gap for the
traditional slack point specifically.** `logs/sweep/A_central_baseline.log`
(from the unrelated rate-coherence task, same day, a different runner —
`01b_run_baseline_sliced.py` via `scripts/run_central_sweep.sh`) prints
a full year-by-year `Solar addition (MW/year)` dict and a final storage
figure. Checked directly: that log's **final storage capacity,
63056.310770000004 MWh, matches `capital_frontier_raw.csv`'s traditional
slack-point `storage_mwh` to 12 significant figures**, and the SUM of
its printed solar-addition dict, **39030.06509999999 MW, matches
`solar_total_mw` at the same point to the same precision.** Two
independent totals matching to 12 significant figures is not
coincidence — these are the same optimal solution, reached by two
different driver scripts pointed at what is, at this specific point,
the same underlying scenario (the NIRP-trajectory solar cap the default
runner uses does not bind below the sweep's own unconstrained optimum,
39.03 GW, so the two configurations converge). **This is a read of an
already-solved, already-saved log file, not a new computation or a
solve** — no runner was invoked for this task.

The formula used below to split `realised_public_spend_usd` into
per-technology NPV is not an assumption either — it is the literal
source of that field, read directly from
`src/optimize_experiments.py:2804-2812`:

```python
realised_public_spend = sum(
    DF[t] * (
        solar_public_add[t] * solar_capex_param[t]
        + storage_add[t] * econ["STORAGE_COST_PER_MWH"]
        + eaas_subsidy[t]
    )
    for t in T
)
```

`DF[t] = 1/(1+r_s)^t`, `r_s = 0.08` (the model's `social_discount_rate`,
a flat, unconditional literal in `src/scenarios.py`, confirmed
elsewhere this session). `solar_capex_param[t]` is confirmed, by
reading `scripts/28_run_capital_frontier.py:306-310` directly, to be
loaded as `load_solar_capex_by_year(scenario_name="solar_low", ...)` —
the ATB series. `econ["STORAGE_COST_PER_MWH"]` is confirmed, by reading
`src/io.py:408-414`, to be a **flat, run-independent constant**: the
`Storage_low` row at `Year=2025` in `data/cost/processed/storage_capex.csv`
— **$252,000/MWh**, not time-varying, and not selectable by any
scenario arm. Years beyond ATB's own 2050 coverage (this build profile
runs to 2054) are held flat at the 2050 rate, per the model's own
documented convention (`src/io.py:128-149`,
`_extend_capex_series`: *"Buffer years are HELD FLAT at the final
sourced value... the conservative choice"*) — not a rate I chose.

This gives EaaS solar CAPEX NPV and EaaS subsidy NPV exactly (both $0,
structural — `financing_regime != "eaas"` fixes `solar_eaas_add[t] ==
0` for every t, `src/optimize_model.py:778-780`), and gives solar
CAPEX NPV exactly (verified build profile × verified capex series ×
verified discount convention). It gives **storage CAPEX NPV by
subtraction** (combined total, exact from the CSV, minus solar NPV,
exact) — itself exact, since both terms it is built from are exact. It
does **not** give an independently-verified storage UNDISCOUNTED total:
that would need `storage_add[t]` (annual additions) year by year, which
was not in the log; using final stock × the flat rate instead is a
**lower bound**, understating true spend by however much
replacement/rebuild capex the horizon required — and the same log
confirms at least one such rebuild occurred (*"storage vintage 2037
retires within the horizon (life 15yr<18yr served) and is rebuilt at
2052, with replacement capex charged there"*). Flagged at every figure
below that depends on it.

### 1. At the two named points

| | envelope | envelope $ | solar MW | storage MWh | `realised_public_spend_usd` | `npv_cost` |
|---|---|---|---|---|---|---|
| Traditional, first slack point | 1.00×B\* | 10,273,343,610 | 39,030.065 | 63,056.31077 | **9,761,349,093.39** | 27,317,670,150.78 |
| EaaS, point reaching the optimum | 0.40×B\* | 4,109,337,444 | 39,030.065 | 63,056.31077 | **4,109,337,438.57** | 27,317,670,153.94 |

Both points read directly from `capital_frontier_raw.csv`
(`financing_regime="traditional"`, `envelope_fraction=1.0`;
`financing_regime="eaas"`, `envelope_fraction=0.4`). Solar MW and
storage MWh are identical between the two points (confirms
`docs/methodology.md` §4.4's existing statement that above saturation
the physical plan does not change, only who pays). Neither runner was
re-invoked; both rows are already-saved output.

`28_run_capital_frontier.py` does not wire `bank_tenor_case` into its
CLI (checked: no match in `src/cli_arms.py` or the script itself), so
the EaaS point above was solved under `load_scenario()`'s own default,
**`bank_tenor_comm`** — the corrected commercial-tenor bankability
basis from the rate-coherence task, not the legacy blended-WACC. Noted
for completeness; it does not change either figure reported here.

### 2. Decomposition at the traditional slack point

| component | NPV (USD) | basis |
|---|---|---|
| Public solar CAPEX NPV | **6,258,256,576.40** | exact — verified build profile × ATB `solar_low` × `DF[t]` at r_s=8% |
| EaaS solar CAPEX NPV | **0** | exact — structural, `financing_regime≠"eaas"` fixes `solar_eaas_add≡0` |
| Storage CAPEX NPV | **3,503,092,516.99** | exact, by subtraction (combined total − solar NPV, both exact) |
| EaaS subsidy NPV | **0** | exact — structural, no EaaS channel active at this point |
| **Total (= `realised_public_spend_usd`)** | **9,761,349,093.39** | exact, from CSV |

**What the envelope charges**: public solar CAPEX + storage CAPEX +
EaaS subsidy (all three present in the formula above) — confirmed
directly from source, not inferred. **What it does not charge**: EaaS
solar CAPEX (here $0 anyway, since EaaS is inactive under this
regime) and gas CAPEX (structurally excluded from the formula
entirely — no gas term appears in it).

**Total solar + storage capital, including any EaaS-financed portion —
the figure comparable to NIRP**: at THIS reference plan, the
EaaS-financed portion is exactly $0 (structural), so **the total is
identical to `realised_public_spend_usd` itself: $9,761,349,093.39
NPV.** This is a clean structural finding, not an approximation: under
the reference (traditional) plan B\* is derived from, "public capital
required" and "total clean-energy capital required" are the SAME
number, because nothing in this specific plan is privately financed.
The public/private distinction only becomes live at the EaaS point
(§1 above), where it is a different plan.

### 3. Recompute on `solar_irena_africa`, without solving

`solar_irena_africa`'s own CSV notes (`data/cost/processed/solar_capex.csv`)
state it is *"ATB solar_low decline shape re-anchored to [IRENA's 2025]
level (scale = 837000 / ATB solar_low 2025)"* — checked directly against
every year 2025-2050: the ratio `solar_irena_africa[y] / solar_low[y]`
is **constant** at 837,000/1,456,000 = **0.574863** in every single
year (not merely at 2025) — a flat rescaling of the ATB shape, not an
independently-shaped curve. Rescaling "year by year" and rescaling the
total by this one constant are therefore identical operations here;
both were computed, and agree:

| | ATB (`solar_low`) | IRENA (`solar_irena_africa`) | ratio |
|---|---|---|---|
| Solar CAPEX, UNDISCOUNTED | 24,136,751,471.20 | 13,875,316,608.10 | 0.574863 |
| Solar CAPEX, NPV @ 8% | 6,258,256,576.40 | 3,597,637,880.80 | 0.574863 |
| **Total (+ storage NPV 3,503,092,516.99, unchanged by capex basis)** | **9,761,349,093.39** | **7,100,730,397.79** | 0.7274 |

The programme's total-capital scale is **robust to the capex basis
by construction here**: because `solar_irena_africa` is defined as a
constant multiple of `solar_low`, moving from ATB to IRENA pricing
does not change WHICH plan is optimal (this is a re-pricing exercise
per the task's own instruction, not a re-solve — a genuine re-solve
under IRENA prices could build MORE solar, since it would be cheaper;
that is not evaluated here, per FORBIDDEN).

### 4. Undiscounted vs NPV, and the ratio

| | UNDISCOUNTED | NPV @ 8% | ratio (NPV/undiscounted) |
|---|---|---|---|
| Solar (ATB) | 24,136,751,471.20 | 6,258,256,576.40 | 0.2593 |
| Storage | ≥15,890,190,314.04 (lower bound) | 3,503,092,516.99 (exact) | ≤0.2205 |
| **Combined (ATB)** | **≥40,026,941,785.24** | **9,761,349,093.39** | **≤0.2439** |
| Solar (IRENA) | 13,875,316,608.10 | 3,597,637,880.80 | 0.2593 (same, constant rescale) |
| **Combined (IRENA)** | **≥29,765,506,922.14** | **7,100,730,397.79** | **≤0.2385** |

**Model's own NPV/undiscounted ratio (ATB, combined): ≤ 0.2439** — an
upper bound, not a point estimate, because the undiscounted denominator
is itself a lower bound (storage rebuild capex omitted, per the Method
note above). The true ratio is somewhat lower than 0.2439.

**Diagnostic only, per instruction — not a discount-rate inference**:
NIRP's own stated NPV/cumulative ratio (Table 6's $40bn / Table 10's
$122bn) = **0.328**. The model's bound (≤0.244) is LOWER than NIRP's
ratio, meaning the model's spending profile is MORE back-loaded
relative to its own undiscounted total than NIRP's — consistent with
the model's own solar build profile (verified above: essentially zero
solar built before 2032, the bulk after 2040) and with storage
building even later. This is a structural observation about the two
profiles' shapes, not a claim about either one's discount rate.

### 5. Implied unit costs, undiscounted

| | USD/kW (solar) | USD/kWh (storage) |
|---|---|---|
| **NIRP** (Table 10 $56bn / Table 21 61.1 GW; $3bn / 11 GWh) | **916.53** | **272.73** |
| **Model, ATB basis** (undiscounted total / final MW or MWh stock — a blended, build-weighted average, not a single rate) | **618.41** | **252.00 (exact, flat, not blended — see below)** |
| **Model, IRENA basis** | **355.50** | 252.00 (storage unaffected by solar's capex basis) |
| **IRENA Africa 2025** (stated reference point, not model-derived) | 837.00 | — |

Storage's model-side figure, **$252.00/kWh, is not "implied" from a
total — it is the direct, flat, run-independent input rate**
(`econ["STORAGE_COST_PER_MWH"]`, `Storage_low`/2025), unchanged at
every point in every run in this codebase; there is nothing to imply.
Solar's model-side figures ARE genuinely implied (total ATB-priced
spend ÷ total MW), since ATB's own per-MW rate varies by year
(1,456,000 in 2025 falling to 563,000 by 2050) and the model's build is
concentrated in the cheaper, later years — the $618.41/kW blended
figure sits inside ATB's own stated range but below both the IRENA
Africa reference point and NIRP's own implied rate, for that reason.

### 6. Route 1 bracket, restated

From `capital_frontier_raw.csv`, traditional regime, last two points
either side of the transition (sweep grid does not resolve the exact
crossover more finely — confirmed, no intermediate `envelope_fraction`
between 0.9 and 1.0 exists in the swept set):

| envelope_fraction | state | dual | `realised_public_spend_usd` |
|---|---|---|---|
| 0.9 | binding | 0.10102134 | 9,246,009,182.24 |
| 1.0 | slack | 0.0 | 9,761,349,093.39 |

**Bracket: [9,246,009,182.24, 9,761,349,093.39].** Restated from
`docs/limitations_register.md` L45's update and
`docs/task_log_rate_coherence.md`'s B\* re-derivation report — unchanged
here, since Part B ran no solve and the underlying CSV is the same file
already read for that report.

---

## PART C — the comparison, and a recommendation

### The three routes, side by side

| route | quantity | value | basis | independence |
|---|---|---|---|---|
| **1. Model dual** | Public clean-energy capital at the slack point | **[9,246,009,182 – 9,761,349,093]** | NPV @ 8%, public only, solar+storage+EaaS subsidy (EaaS subsidy = $0 here) | Model-internal |
| **2. Bottom-up (ATB)** | Total solar + storage capital (public, since EaaS-financed = $0 at the reference plan) | NPV **9,761,349,093**; undiscounted **≥40,026,941,785** | NPV @ 8% AND undiscounted, both stated; model quantities, external (ATB) prices | Model quantities + external prices |
| **2. Bottom-up (IRENA)** | Same, re-priced | NPV **7,100,730,398**; undiscounted **≥29,765,506,922** | Same as above, IRENA-Africa-anchored solar price | Model quantities + external prices |
| **3. NIRP 2024** | Solar + storage investment 2024-45 (Table 10, solar+storage rows only) | undiscounted **59 bn** ($56bn + $3bn); all-technology NPV **40 bn** (not solar+storage-specific) | Undiscounted, confirmed; NPV figure is ALL-technology, not solar+storage-isolated — see caveat below | External (NIRP 2024) |

**Basis mismatches, stated explicitly, not smoothed over:**

- Route 1 and Route 2 are the SAME basis by construction (Route 2 is
  Route 1's own components, re-derived from the capex series as a
  cross-check — they agree to the extent the reconstruction is exact,
  i.e., exactly at the combined-NPV level, since that figure is shared;
  Route 2 adds the undiscounted and per-technology split Route 1 alone
  does not give).
- Route 3's $59bn (solar+storage) is undiscounted only — NIRP never
  states a solar+storage-specific NPV, only an ALL-technology NPV
  ($40bn, covering gas/hydro/Gas+CCUS/DSM too, not comparable at the
  technology-matched level).
- Route 3's $40bn (all-technology NPV) is therefore basis-comparable to
  Routes 1/2's NPV figures (both are "NPV @ [a discount rate]") but
  NOT technology-matched (NIRP's $40bn includes hydro and gas; the
  model's $9.76bn is solar+storage only).
- Route 3's $59bn is technology-matched to Routes 1/2 (solar+storage
  only, both sides) but NOT basis-matched to the model's own NPV figure
  (undiscounted vs NPV) — it IS basis-matched to Route 2's own
  undiscounted total.
- **The one truly apples-to-apples pairing available: Route 2's
  undiscounted total (≥40.03bn ATB / ≥29.77bn IRENA, solar+storage) against
  Route 3's $59bn (solar+storage, undiscounted).** Both technology-
  matched AND basis-matched. Every other pairing above differs in at
  least one dimension, stated in the row it applies to.
- Currency-year caveat (Part A, carried forward): NIRP's "$ in 2022
  prices" note is printed at Table 6, not at Table 10 — the $59bn
  figure's currency vintage relative to the model's own (nominal 2025+
  USD, uninflated further within the model) is not independently
  confirmed.
- Storage caveat (Part A, carried forward): NIRP's own Table 23 sums to
  $3.8bn against a stated $3bn for storage — the weaker of NIRP's two
  self-consistency checks; the $59bn / $56bn+$3bn figures used above
  are NIRP's OWN headline (Table 10), not the annex sum.

### The one clean external agreement in the triangulation: storage unit cost

**$252.00/kWh (model, flat, exact) against $272.73/kWh (NIRP, implied
from Table 10's $3bn / Table 21's 11 GWh — reading Table 6's rounded
figures, or the more precise Table 21 3.4 GW at Table 6's 3hr duration
gives an equivalent figure) — 7.6% apart.** Recorded prominently, per
instruction, because it is the one place in this entire triangulation
where two independently-sourced figures — a US-basis NREL ATB storage
cost the model applies flat and unmodified (`Storage_low`, 2025, no
Nigeria or Africa adjustment of any kind — the same absence of local
evidence L47 already flags for solar) and NIRP's own bottom-up zonal
plan's implied rate — land within 8% of each other, with NO rescaling,
NO basis adjustment, and NO forced reconciliation applied to produce
the agreement. Both routes 1/2 and route 3 arrive at storage's per-unit
cost independently and agree closely. This does not corroborate the
storage PROGRAMME SIZE (§ below: the two plans disagree by ≥5.3× on
dollar volume) — only the PRICE the two independent sources put on a
megawatt-hour of storage, which is a materially different, narrower
claim, and the only unit-cost pairing in this report that does not
need a level-vs-blend or basis caveat attached to it.

### Do routes 2 and 3 agree on programme scale?

**Reframed, per Zeus's ruling: the disagreement is COMPOSITIONAL, not a
uniform scale difference.** The total ratio (0.68 ATB / 0.50 IRENA,
below) is the NET of two OPPOSING per-technology ratios that mostly, but
not fully, cancel:

| | model (undiscounted) | NIRP (Table 10) | ratio (model/NIRP) |
|---|---|---|---|
| Solar | 24,136,751,471.20 (ATB) / 13,875,316,608.10 (IRENA) | 56,000,000,000 | **0.43× (ATB)** / 0.25× (IRENA) |
| Storage | ≥15,890,190,314.04 (lower bound) | 3,000,000,000 | **≥5.3×** |
| **Total** | ≥40,026,941,785.24 (ATB) / ≥29,765,506,922.14 (IRENA) | 59,000,000,000 | 0.68× (ATB) / 0.50× (IRENA) |

**The model builds 0.43× NIRP's solar dollar-value and at least 5.3×
NIRP's storage dollar-value** — a composition difference, not a scale
difference the two plans merely disagree on the size of. This is not
incidental: `docs/chapters/chapter3_3_draft.md` §3.3 states the
mechanism directly — *"Solar produces nothing in the model's evening
peak block by construction, so it carries zero capacity credit; storage
and gas carry positive credits. This distinction drives the composition
of every plan the model builds."* Solar meets ENERGY demand only in
this model; it earns no credit toward the firm-capacity constraint that
drives the plan's ADEQUACY requirement, because Nigeria's system peak
falls in the evening block, after sunset. §3.3.5 states the consequence
for storage explicitly: *"storage absorbs the entire firm-capacity
residual: its quantity is determined by the constraint rather than
chosen... Storage composition — that it is displaced by gas, and at
what ratio — is a model result. Storage magnitude is not."* **The
model's storage build is therefore adequacy-driven, not an independent
economic choice the two plans can be compared on equal terms** — it is
sized to cover exactly the firm-capacity gap solar cannot, at whatever
credit `storage_cc_case` assigns it (a declared, swept, ASSUMED
parameter per §3.3.5, not sourced).

**NIRP's own storage sizing basis differs in a way this explains
further.** NIRP's average storage duration is stated as **3 hours**
(Table 6, page 29/7), against **4.0 hours** in this model
(`storage_duration_hours`, itself reached through a default rather than
a required key — §3.3.5/§3.3.7's own flagged inconsistency). A shorter
duration at NIRP's own 3.4 GW power rating caps NIRP's storage ENERGY
well below what a 4-hour-duration fleet of comparable power would
reach; the two plans are not sizing storage on the same power-to-energy
ratio, independently of how much peak-adequacy responsibility each
model's own formulation assigns to it. NIRP's own zonal generation
model may or may not carry an explicit adequacy/capacity-credit
mechanism for solar — Part A found no statement either way — so
whether NIRP's much smaller storage fleet reflects a different
capacity-credit treatment of solar, a different reserve-margin
methodology, or simply a different view of the technology mix cannot be
determined from what NIRP states.

- Undiscounted, technology-matched total: model **≥40.03bn (ATB)** /
  **≥29.77bn (IRENA)** against NIRP's **59bn** — ratio **0.68 (ATB)** /
  **0.50 (IRENA)**. Both model figures are LOWER BOUNDS (storage
  rebuild capex omitted, Part B Method note), so the true storage ratio
  is somewhat higher than 5.3×, and the true total ratio somewhat
  higher than 0.68/0.50 — the composition finding (solar down, storage
  up) is not weakened by this, since both corrections push in the same
  direction (more storage dollars, not fewer).
- **Demand scope is a second, independent contributor, not addressed by
  the composition finding above.** This model's reference plan runs at
  `organic_central` demand growth (3.9%, register L54), against NIRP's
  own stated 7.7%/yr base-case growth (Part A) — roughly half. A
  smaller demand forecast feeding a capacity-expansion model would
  produce a smaller PHYSICAL programme independently of the
  composition effect above; the two are not separated here (a formal
  demand-driven decomposition was not performed — out of scope; FORBIDDEN
  also excludes computing a blended/triangulated figure). **Both the
  composition effect (capacity-credit-driven, evidenced above) and the
  demand-scope effect (evidenced, not decomposed) are named as
  contributing causes; neither is claimed to be the sole explanation,
  and the routes are not forced to agree.**

### What share of the programme is public, and is it plausible against NIRP?

**At the reference (traditional) plan: 100%.** Confirmed structurally
(§B2): EaaS solar CAPEX and EaaS subsidy are both exactly $0 at this
plan, because `financing_regime≠"eaas"` disables the EaaS channel
entirely. This is not a computed share so much as a definitional one —
the reference plan B\* is derived from IS the all-public plan by
construction.

The share becomes genuinely informative only away from the reference
plan: at the EaaS optimum point (§B1), the right comparison is
`realised_public_spend_usd` ($4.11bn) against that SAME plan's total
solar+storage capital (public + EaaS-financed together) — NOT against
`npv_cost` (total system cost), which includes gas and every other
technology the public/EaaS split has nothing to do with. Computing that
ratio would need the same per-technology split done at the EaaS point
specifically, which (per the Method note in Part B) would need a
verified build-profile cross-reference this report does not have for
that point. **Not computed, not estimated** — flagged as a gap rather
than approximated.

**Is 100%-public-at-the-reference-plan plausible against NIRP?** NIRP
states no public/private split at all (Part A §4) — there is nothing
in NIRP to compare this figure against. The question cannot be
answered from NIRP; it can only be noted that NIRP's own silence on
financing structure means this model's reference-plan share is
unconstrained by anything NIRP says, in either direction.

### Does the model's implied solar unit cost sit between IRENA's 837 and ATB's basis, and where does NIRP's sit?

**Corrected framing, per Zeus's ruling: the $618/kW figure and IRENA's
$837/kW are not comparable as stated — one is a 2025–2045 build-weighted
blend, the other a single 2025 level.** Conflating them was the error
in this report's own earlier draft of this section. There are two
valid, separately-stated pairings, not one:

**Pairing 1 — LEVEL vs LEVEL, both 2025.** ATB `solar_low` 2025 =
$1,456,000/MW against IRENA's 2025 Africa reference = $837,000/MW:
ratio **1,456/837 = 1.74×**. This is register **L47's existing
finding, UNCHANGED** — this report adds nothing new to it, and
confirms it independently via the same two source figures.

**Pairing 2 — BLEND vs BLEND, both build-weighted over the horizon.**
Model's own realised solar spend ÷ MW built (ATB basis) = **$618.41/kW**
against NIRP's own realised solar spend ÷ MW built = **$916.53/kW**:
ratio **618.41 / 916.53 = 0.67×**. This is a NEW comparison this report
computes; it is not L47 and does not supersede it — L47 is a pure
2025-vintage price comparison (ATB vs IRENA, no model quantities
involved), while this pairing compares two DIFFERENT capacity-expansion
plans' own realised, quantity-weighted spending.

**Why the two ratios point in opposite directions, and neither is
wrong.** ATB's `solar_low` series is steeply DECLINING ($1,456,000/MW
in 2025 to $563,000/MW by 2050 and held flat, `src/io.py`'s own
documented end-effect-buffer convention). The model's own solar build
is heavily BACK-LOADED (Part B's Method note: essentially zero MW built
before 2032, the bulk after 2040) — it realises capacity almost
entirely in the cheap, late-horizon years of a declining series. **The
model's low blended cost (below even IRENA's flat 2025 reference point)
reflects WHEN it builds on a declining price series, not that its price
BASIS is cheap** — the basis (ATB `solar_low`) is the same 1.74×-above-
IRENA-2025 series L47 already flags as expensive; the blend result
looks cheap only because most of the model's dollars are spent at the
series' 2040s-2050s prices, not its 2025 price. NIRP's own build
profile is evidently less back-loaded relative to ITS OWN price
assumptions (not independently sourced here — NIRP states no
year-by-year solar unit-cost series, only the aggregate $56bn/61.1GW),
since its blended rate ($916.53/kW) sits close to its OWN early-2030s
range on the ATB curve (ATB `solar_low`[2032] = $915,000/MW,
coincidentally close) rather than near ATB's late-horizon low.

**Where NIRP's rate sits**: $916.53/kW is the HIGHEST of the three
figures compared (1,456 2025-ATB-level > 916.53 NIRP-blend > 837
IRENA-2025-level > 618.41 model-blend) — inside ATB's own stated
2025-2050 range ($563-$1,456/kW), near its early-2030s vintage, and
above the IRENA reference point.

### Recommendation

**(a) Retain 10,273,343,610 as a declared normalisation constant, with
its derivation date and state, reporting the current bracket
[9,246,009,182 – 9,761,349,093] alongside it — recommended.**

Reasons for (a):

- The self-consistency property's failure is small in relative terms
  (the bracket's own width is [9.246bn, 9.761bn] — a 5.6% span around
  the old constant, itself about 5% below the old constant) — this is
  a re-baselining-scale drift (matching register L54/L58's own
  demand-and-rate corrections), not evidence the original B\* derivation
  method was wrong.
- The bracket cannot be resolved further by the swept grid (Part B §6,
  restated) — replacing B\* with either bracket endpoint would still be
  choosing a point inside an unresolved range, not a genuinely sharper
  figure than "retain and declare the bracket."
- NIRP corroborates programme SCALE (Part C above: same order of
  magnitude on the total, 0.68×/0.50×, a COMPOSITIONAL rather than
  uniform-scale disagreement — solar 0.43×, storage ≥5.3×, traced to the
  model's zero-capacity-credit treatment of solar and its
  adequacy-driven storage sizing, `docs/chapters/chapter3_3_draft.md`
  §3.3/§3.3.5 — not an order-of-magnitude disagreement) without
  corroborating a specific dollar figure NIRP never computes on a
  comparable basis (Part A: no NIRP figure is both NPV-basis and
  solar+storage-isolated) — so NIRP cannot supply a REPLACEMENT constant
  even if one were wanted; it can only corroborate that the existing
  constant is not absurd in scale, and that the storage UNIT cost
  specifically (as opposed to volume) independently agrees to within
  7.6%.
- Envelope-grid stability: retaining B\* means every existing
  `tight`/`moderate`/`adequacy`/`expansion` result stays interpretable
  as "a stated fraction of a declared 2026-09-20 reference," with no
  re-run required — the FORBIDDEN clause already prohibits changing
  `B_star` in this task regardless, but (a) is also the option that
  does not create pressure to do so later purely for its own sake.

Reasons AGAINST my own recommendation, stated per instruction:

- (a) leaves a KNOWN-STALE constant in place indefinitely unless
  someone commits to periodically re-deriving and re-declaring it — the
  same drift that broke self-consistency once (organic_central + the
  concessional rate/tenor correction) will recur after the next
  parameter correction, and there is no process in this repo that
  automatically flags it (this task itself was Zeus's manual
  initiative, not a triggered check).
- (a) is a documentation fix to a MODEL-INTERNAL inconsistency; it does
  not resolve WHETHER 10,273,343,610 was ever the "right" figure by any
  standard other than "what a specific run produced on a specific
  date" — Route 3 (NIRP) suggests the true programme COMPOSITION
  question is genuinely open (solar 0.43×, storage ≥5.3× NIRP's own
  dollar allocation), which (a) does not engage with at all, only (b)
  or (c) would, and neither fully would either, since the composition
  gap is a modelling-choice question (capacity credit, storage duration
  basis) that no B\* value alone resolves.
- A reader unfamiliar with the derivation history could reasonably ask
  why the envelope tiers are fractions of a number that is, by this
  report's own finding, no longer self-consistent — "declared, not
  exact" is a real answer but a less satisfying one than either a
  corrected figure (b) or an externally-anchored one (c).

(b) and (c) were both considered and are available if Zeus weighs the
above differently: (b) replace with (for example) the bracket's slack
endpoint, 9,761,349,093, accepting the full envelope-grid re-run and
the new fixed point's own likely non-termination (a corrected B\* would
itself move again at the next parameter correction, for the same
reason this one did); (c) re-pose against NIRP's corroborated scale
specifically — Part A/B support a NUMBER for this (NIRP's $59bn
undiscounted solar+storage, or some IRENA/ATB-adjusted variant of it)
but Part C's own finding is that this number reflects a DIFFERENT
composition (NIRP's own solar/storage split, under NIRP's own capacity-
credit and duration assumptions, neither stated by NIRP and both
different from this model's) rather than a directly comparable
programme, so re-posing against it would import NIRP's own,
undocumented compositional assumptions into the model's own
normalisation constant rather than resolve it — the weakest of the
three options on this report's own evidence, though not ruled out if
Zeus judges the demand-scope contribution (separate from the
composition effect, per the finding above) more decisive than this
report treats it.

**STOP — Part C complete.** Nothing was run, solved, or committed.
`B_star` (`src/scenarios.py:933`, still `10_273_343_610`) is unchanged.
No figure from `literature/` was committed. Recommendation given with
reasons against it, per instruction; decision remains Zeus's.
