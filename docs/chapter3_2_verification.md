# Chapter 3.2 — verification (demand construction)

**Read-only except item 4** (the citation correction, which changes no
value). No chapter 3.2 draft exists in this repo/session — this
document verifies the underlying algebra and sourcing directly against
`src/scenarios.py`, `src/demand.py`, `src/optimize_model_sliced.py` and
the methodology documents, as the basis a (3.24)–(3.27) sequence would
need to match, not as a diff against an existing draft.

---

## Job 1 — Equations (3.25), (3.26), and the peak-demand equation

### (3.25) — `project_baseline_demand` (`src/demand.py:34-77`)

Rule body, quoted verbatim:

```python
years = np.arange(start_year, end_year + 1)
t = years - start_year

demand = base_demand * (1 + growth_rate) ** t
```

**MATCHES.** This is exactly the constant-exponential-growth form the
methodology text already describes (`docs/methodology_demand_horizon.md`
§1.3's "driver decomposition" and every arm table entry): $D_t =
D_0(1+g)^{t-t_0}$, with `base_demand` = $D_0$ (the Tier 1 base, 37.09
TWh), `growth_rate` = $g$ (one of the three arms, or the NIRP
benchmark), and `t - start_year` = $t-t_0$. No slicing, no elasticity
term, and no piecewise growth appear inside this function itself — the
function's own docstring states "Constant exponential growth" and the
module docstring states "No price elasticity or sectoral
disaggregation" as a declared non-scope; elasticity enters only
upstream, in the choice of `growth_rate` via
`demand_growth_scenarios()` (Job 2). The known limitation already
declared in the module docstring — a piecewise/front-loaded $g$ would
require changing this function's signature, "deferred deliberately" —
is worth carrying into the chapter text as the reason (3.25) is a
constant-$g$ form despite NIRP's own trajectory being front-loaded.

Guard behaviour, for completeness: `base_demand` is checked
non-negative (`assert_non_negative`); `growth_rate < -1.0` raises;
`end_year < start_year` raises; the returned `demand` array is also
checked non-negative before return.

### (3.26) — `_normalised_demand_weights` (`src/optimize_model_sliced.py:150-172`)

Rule body, quoted verbatim:

```python
h = slice_hours()
season_f = scenario["season_demand_factor"]
period_f = scenario["period_demand_factor"]

raw = {
    (s, p): h[(s, p)] * season_f[s] * period_f[p]
    for (s, p) in SLICES
}
total = sum(raw.values())
return {k: v / total for k, v in raw.items()}
```

**MATCHES.** In the notation already used for the sliced builder
(`docs/formulation_inventory_sliced.md` §1): $w_s = w_{(\sigma,p)} =
\dfrac{H_{(\sigma,p)}\cdot f^{season}_\sigma\cdot f^{period}_p}
{\sum_{(\sigma',p')} H_{(\sigma',p')}\cdot f^{season}_{\sigma'}\cdot
f^{period}_{p'}}$, i.e. hours-in-slice weighted by the product of a
season factor and a period factor, then renormalised over all six
slices so $\sum_s w_s = 1$ by construction (the division by `total`
guarantees this algebraically, not merely numerically). `season_f` and
`period_f` are read as bare `scenario[...]` subscripts, not
`.get(...)` — both are required keys, supplied by the selected
`peak_multiple_case` arm (confirmed against `docs/formulation_inventory_sliced.md`
§2.3: `season_demand_factor` and `period_demand_factor`, both `REQUIRED
(subscript)`, lines 164-165 inside this same function).

### The peak-demand equation (not yet numbered — material for it)

`peak_mw[t]` is computed inside `load_scenario()`
(`src/scenarios.py:2853-2874`), quoted verbatim:

```python
_implied_weights = _normalised_demand_weights({
    "season_demand_factor": _season_demand_factor,
    "period_demand_factor": _period_demand_factor,
})
_slice_h = slice_hours()
_f_peak = sum(_implied_weights[(s, "peak")] for s in ("dry", "wet"))
_h_peak = sum(_slice_h[(s, "peak")] for s in ("dry", "wet"))
par_block = _f_peak * 8760 / _h_peak

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
```

In symbols, using (3.25)'s $D_t$ and (3.26)'s $w_s$: the peak-block
demand share is $f_{peak} = \sum_{\sigma} w_{(\sigma,\text{peak})}$
(summed over both seasons), and the peak-block hours are
$h_{peak}=\sum_\sigma H_{(\sigma,\text{peak})}$ (1,460 h/yr, fixed
calendar arithmetic: two 4-hour peak periods × ~365 days worth of
season-days). The **block-average peaking ratio** is

$$
\text{par\_block} = f_{peak}\cdot\frac{8760}{h_{peak}},
$$

and peak demand in MW is

$$
P_t = \frac{D_t\cdot 10^{6}}{8760}\cdot\text{par\_block},
$$

i.e. annual demand converted to an average-MW figure ($D_t\cdot
10^6/8760$), scaled up by the block-average peaking ratio. **This is
not an independent measurement**: the code comment at the definition
site (`scenarios.py:2841-2848`) states par_block is "DERIVED from the
arm's shape via the model's own weighting function... never set
independently... this IS the peak block's demand-energy share,
restated, not a second, independently checkable number." `par_block`
itself is stored on the scenario dict "Reported-only... never fed back
into a guard." Also confirmed directly: `par_block` is a **4-hour
block-average** ratio (peak period = 4 h), not the instantaneous peak
ratio `within_block_peaking_factor` applies inside the firm-capacity
constraint (3.21 in `chapter3_1_v2.md`) — the two are read from the
same file region but are structurally distinct quantities, exactly as
`methodology_physical_layer.md` §2's "within-block peaking factor"
discussion (already in Chapter 3.1) describes.

---

## Job 2 — Elasticity source

### Every attribution site of ε = 0.76–0.90 or ε = 0.83, with file and line

| file:line | text |
|---|---|
| `src/scenarios.py:454-459` | `eps = 0.76-0.90, long-run income elasticity of electricity demand, middle-income NON-SSA panel [Liddle, Smyth & Zhang, Energy Economics]. SSA-estimated elasticities (0.60-0.76) are rejected as a central source... Retained as a lower bound only.` |
| `src/scenarios.py:495` | `Route A: eps 0.83, GDP 4.5% -> 4.04%.` (inside `organic_central`'s derivation) |
| `src/scenarios.py:504` | `Route A: eps 0.90, GDP 6.8% -> 6.30%.` (inside `organic_high`'s derivation) |
| `docs/methodology_demand_horizon.md:122-124` | `ε = **0.76–0.90** [SOURCED: Liddle, Smyth & Zhang, *Energy Economics*, middle-income non-SSA panel]. Non-SSA deliberately, for the contamination reason in §1.2. Central working value 0.83 [ASSUMED: midpoint].` |
| `docs/merged_correction_plan_v2.md:239` | `ε = **0.76–0.90** from Liddle, Smyth & Zhang (*Energy Economics*), middle-income...` (historical correction-plan document; not in scope for item 4's correction) |
| `docs/merged_correction_plan_v2.md:1610` | table row: `Income elasticity ε \| Liddle, Smyth & Zhang, *Energy Economics* (middle-income, non-SSA) \| ✅ 0.76–0.90` (same document) |
| `docs/findings_provisional_pre_sweep.md:46` | `ε = 0.76–0.90 from Liddle, Smyth & Zhang (*Energy Economics*), middle-income...` (superseded findings document; not in scope for item 4) |
| `docs/methodology_claims_ledger.md:392` (row 205) | `ε = 0.76–0.90 [Liddle, Smyth & Zhang]; central working value 0.83 [ASSUMED: midpoint]` — verdicted VERIFIED against the (mis-)attributed source at the time (Phase 1 ledger; a historical audit record, not corrected here — see note at the end of this section) |

Only `src/scenarios.py` (three sites) and `docs/methodology_demand_horizon.md`
§1.3 (one site) carry a live, currently-read attribution; the other
three are historical/superseded documents.

### Whether the attributed paper supports the range

**It does not, and the repo's own figures identify the correct paper.**
Liddle, Smyth & Zhang (2020), *Energy Economics* 86, estimates **total
energy demand** income elasticity at **0.6–0.8** — a different
dependent variable (total energy, not electricity specifically) and a
different range. The range actually carried in this repo, **0.76–0.90**
with a central **0.83**, matches **Liddle & Hasanov (2023)**,
*Environmental and Ecological Statistics* 30(4), 827–849, which
estimates **economy-wide electricity** income elasticity at **0.7–0.9,
stable around 0.8** — the right dependent variable (electricity) and
a range that contains the repo's figures. The mis-citation is
internally diagnosable from the repo's own numbers without needing the
papers themselves: an electricity-specific model should cite an
electricity-specific estimate, and 0.83 sits inside Liddle & Hasanov's
"stable around 0.8," not naturally inside a 0.6–0.8 total-energy range
whose upper bound (0.8) is *below* the repo's own lower-bound-adjacent
central value (0.83).

### Whether any file traces the lower bound 0.76 to a specific table or estimator

**No.** Every site above cites the paper by title/journal only — none
names a table number, an equation number, a specific panel estimator
(e.g. FMOLS, DOLS, PMG, or any other cointegration/panel technique
common to this literature), a sub-sample, or a specification. `grep -n
"Table" docs/methodology_demand_horizon.md` finds three unrelated table
citations elsewhere in the document (NIRP 2024 Table 14 for the growth
methodology; NIRP Annex F Table 19/Table 8 for solar; NIRP Tables 2 and
15 for something else) — none attached to the elasticity figure. The
0.76–0.90 range and the 0.83 central value are carried as bare numbers
with a journal-level citation only.

### Whether the source's panel excludes sub-Saharan Africa "deliberately"

**Confirmed, and the reasoning is stated, not merely asserted.**
`src/scenarios.py:454-459`: "middle-income NON-SSA panel... SSA-estimated
elasticities (0.60-0.76) are rejected as a central source: they are
estimated on supply-rationed systems and inherit the same contamination
as the Nigerian series. Retained as a lower bound only."
`docs/methodology_demand_horizon.md:122-123`: "Non-SSA deliberately, for
the contamination reason in §1.2" — §1.2 is the section (already read
for Job 1) arguing that consumption response to income cannot be
observed where supply is rationed, so SSA/Nigerian estimates are
biased downward. This reasoning is sound in principle and independent
of which of the two candidate papers is the correct source — but it is
worth flagging that **Liddle & Hasanov (2023)'s own panel composition
has not been independently checked against the repo's "non-SSA"
description**; this document reports the repo's claim, not a
verification of the actual Liddle & Hasanov panel membership (that
would require reading the paper itself, which is not in the repo).

**Note on the ledger row (methodology_claims_ledger.md:392).** That row
verdicted the (misattributed) citation VERIFIED at Phase 1, against
Liddle, Smyth & Zhang — i.e. it confirmed the number matched what the
repo *said* it was citing, not that the citation was the correct paper
for that number. This is not corrected here: the claims ledger is a
point-in-time Phase 1 audit record (see its own header, "Started:
2026-09-20T16:48:41Z, HEAD: 8fc5ccb") and item 4 below does not extend
to it — flagging for awareness, not action.

---

## Job 3 — Data vintages

**Method:** searched `docs/methodology_demand_horizon.md`,
`src/scenarios.py`, `data/demand/*` (including embedded file metadata
for the one `.xlsx` raw file present), `docs/limitations_register.md`,
`docs/methodology_claims_ledger.md`, and `docs/sourcing_pack_02_ledger.md`.
No date is invented; where none is recorded, that is stated plainly.

| source | release/access date recorded in the repo |
|---|---|
| **NERC 2024** | Cited throughout as **"NERC 2024"** (e.g. `scenarios.py:28,405,412,433`; `docs/methodology_demand_horizon.md:37`) with the specific figure 37,093.70 GWh, and elsewhere by table number for related-but-different figures (`scenarios.py:380` "NERC 2024 Table 4.3", licensed capacity; `scenarios.py:1188-1189` "NERC Table 7.1 2024" and "NERC 2026/Q1 Table 1", gas availability). **No month, quarter, or access/retrieval date is recorded for the 37.09 TWh demand-base figure itself** — only the report year "2024." |
| **World Bank WDI** | Cited by indicator code only: `SP.POP.TOTL` (population) and `EG.ELC.ACCS.ZS` (electricity access) — `docs/methodology_demand_horizon.md:74-75`. **No pull/access date recorded.** The claims ledger's own Phase 1 audit (`methodology_claims_ledger.md:385`, row 198) independently found **`SP.POP.TOTL` is not stored anywhere in the repo as a data file** — the population figures used in the growth-arm derivation exist only as inline numbers in prose/comments, not as a retrievable dataset with a vintage. |
| **UN World Population Prospects** | Cited only as the two endpoint figures, "232.68m → 336.66m" (`scenarios.py:460`; `docs/methodology_demand_horizon.md:125-126`), used to derive the 1.77%/yr population growth figure. **No WPP revision year (e.g. "2024 Revision") or access date is recorded anywhere** — not even which WPP release the two figures come from. No underlying data file is stored in the repo for this source either. |
| **Ember** | Cited as "Ember 2010-2025 grid generation CAGR" (`scenarios.py:487`; `docs/methodology_demand_horizon.md:115`), 3.14%. **No access date recorded.** The stored data file is `data/demand/electricity_generation_owid.csv` — named for Our World in Data, not Ember directly (OWID commonly re-hosts Ember's Yearly Electricity Data, but the repo does not state this re-hosting relationship or an OWID pull date anywhere). More materially: the claims ledger's own prior audit (`methodology_claims_ledger.md:386`, row 199) found the **stored file's own 2014-2024 CAGR (2.20%) does not match the 1.53%/yr figure quoted in the methodology text for that same window**, and concluded "Could be a different Ember vintage; not reproducible from the repo." So not only is there no recorded vintage/access date for Ember — the one check that was possible (recomputing from the stored file) did not reproduce a quoted figure, which the ledger already flagged and which this task does not re-litigate. |
| **Oluwole (2023)** | Cited with a full DOI, including a version suffix: **DOI 10.17632/pxvdm26rn7.2** (Mendeley Data; the ".2" is a dataset-version identifier, e.g. `scenarios.py:249`, `docs/methodology_physical_layer.md:75`, register L1). No separate "accessed on" date is recorded in any `.md` file. **Data-file-header evidence** (the one place a date does exist): the repo's own copy, `data/demand/raw/Nigeria_National_Demand_Timeseries.xlsx`, carries embedded OOXML metadata — `dc:creator: "Damilola"`, `dcterms:created: 2022-12-02T14:50:15Z`, `dcterms:modified: 2022-12-02T14:53:49Z` — i.e. the file itself was authored/last-touched 2022-12-02, which is *prior to* the paper's own 2023 publication year, consistent with it being a pre-print/working version of the dataset rather than evidence against the citation. This is embedded file metadata, not a documented "accessed on" note. |

**Summary:** of the five sources, **none has a documented access or
retrieval date** in the sense of "pulled/downloaded on YYYY-MM-DD" —
only publication years, and in Oluwole's case a dataset-version DOI
suffix. One data-file-header date exists (Oluwole's raw `.xlsx`,
embedded creation timestamp 2022-12-02). Two of the five (WDI's
`SP.POP.TOTL`, UN WPP) have **no underlying data file in the repo at
all** — the figures are inline prose numbers. Ember's stored file does
not reproduce a figure quoted from it elsewhere in the methodology text
(pre-existing finding, claims ledger row 199, not re-litigated here).

### The two IEA self-generation citations (`docs/sourcing_pack_02_ledger.md`)

Both quoted verbatim from that file (§2, "Self-generation energy — SOURCED",
lines 45-51):

> **Primary (IEA, 2019):** 40 TWh generated from 40 GW of backup
> capacity across sub-Saharan Africa in 2018, of which Nigeria
> accounted for almost half — about 18 TWh from roughly 9 GW of backup
> generation. That is the citable energy figure.

> **Share (IEA, reported January 2024):** backup generators cover about
> 40% of Nigeria's electricity needs.

Neither carries a report title, more specific publication date (beyond
"2019" and "reported January 2024"), or URL. The file's own provenance
header (`sourcing_pack_02_ledger.md:1-4`) applies to both: **"Figures
in this file are secondary citations transcribed from web sources; no
primary document has been verified against the repo. Treat every
number as 'as supplied, unchecked' until a primary is read."**

---

## Job 4 — Attribution correction (the one write in this task)

**Changed:** citation only. **No numeric value changed anywhere.**

- `docs/methodology_demand_horizon.md` §1.3 (line 122): `Liddle, Smyth &
  Zhang` → `Liddle & Hasanov`.
- `src/scenarios.py:454-459` (the only code-comment site, inside
  `demand_growth_scenarios()`'s docstring): `Liddle, Smyth & Zhang` →
  `Liddle & Hasanov`.

`docs/merged_correction_plan_v2.md` and `docs/findings_provisional_pre_sweep.md`
are historical/superseded documents and were **not** edited — they are
outside item 4's scope ("§1.3 and every code comment"), and per this
project's own stratigraphy convention (`docs/methodology_protocols.md`
§3) a superseded document is retained, not retroactively rewritten.
`docs/methodology_claims_ledger.md` row 205 likewise not edited, per
the note in Job 2 above.

**Log: mis-citation corrected.** The elasticity range ε = 0.76–0.90
(central 0.83) used throughout `demand_growth_scenarios()` was
attributed to Liddle, Smyth & Zhang (2020), *Energy Economics* 86 — a
paper estimating **total energy** demand elasticity at 0.6–0.8, not
electricity-specific and not matching the repo's range. The correct
attribution is Liddle & Hasanov (2023), *Environmental and Ecological
Statistics* 30(4), 827–849, which estimates **economy-wide electricity**
income elasticity at 0.7–0.9, stable around 0.8. **Range discrepancy
noted and not resolved**: the new source gives 0.7–0.9; the code and
methodology text carry 0.76–0.90. The repo's 0.76 lower bound in
particular does not trace to a specific table or estimator in either
paper (Job 2) and does not match Liddle & Hasanov's stated 0.7 lower
bound. This is now a declared open item, not a silently-inherited one.

---

## Job 5 — Recompute the §3.2.3 decomposition table from stored files only

`docs/chapters/chapter3_2_draft.md` §3.2.3 carries three rows. Stored
files checked, in full: `data/demand/electricity_generation_owid.csv`
(year, generation TWh, **2000–2024 inclusive — no 2025 row**) and
`data/demand/electricity_access_worldbank.csv` (year, access %,
**1990–2023 inclusive — no 2024 row**). `find data -iname "*pop*"`
(excluding the `.claude/worktrees` mirror): **no result** — there is no
population file of any kind, total or connected, anywhere in the repo.
This matches the claims ledger's own prior finding (row 198: "there is
no population series in the repo").

### Row 1 — "total population, 2010–2025": generation 3.14%/yr, denominator 2.39%/yr, residual 0.73%/yr

- **Generation: NOT COMPUTABLE for this window.** The stored file ends
  at 2024; there is no 2025 row to close a 2010–2025 CAGR. The nearest
  computable window, 2010–2024 (14 years, `26.13 → 40.13` TWh), gives
  **3.11%/yr** — close to but not identical to the drafted 3.14%, and
  over a different (shorter) window, so it neither confirms nor
  contradicts the drafted figure; it is simply a different quantity.
- **Denominator ("total population," 2010–2025 or 2010–2024): NOT
  COMPUTABLE.** No population file exists in the repo, at any vintage,
  for any window.
- **Residual: NOT COMPUTABLE**, since it requires both terms above via
  $(1+g^{gen})=(1+g^{pop})(1+g^{pc})$ and neither the exact-window
  generation figure nor the population figure exists in a stored file.

**Does 3.14% for 2010–2025 reproduce? No — not confirmed and not
refuted. The window itself cannot be evaluated from any file in the
repo**, because the generation series does not extend to 2025. The
closest available window (2010–2024) gives 3.11%, not 3.14%, but this
is not a like-for-like comparison since the window lengths differ (14
vs 15 years) — it is offered as the nearest available evidence, not as
a reproduction or a refutation.

### Row 2 — "total population, 2014–2024": generation 1.53%/yr [flagged DOES NOT REPRODUCE], denominator 2.27%/yr, residual −0.73%/yr

- **Generation: COMPUTABLE, and confirmed not to reproduce.**
  `data/demand/electricity_generation_owid.csv`, years 2014 (32.28 TWh)
  and 2024 (40.13 TWh), 10-year window: $(40.13/32.28)^{1/10}-1 =
  $ **2.20%/yr**, not 1.53%/yr. This exactly reproduces the claims
  ledger's own prior finding (row 199: "gives 2.20%... not 1.53%"). The
  draft's own `[DOES NOT REPRODUCE]` flag on this row is confirmed
  correct.
- **Denominator ("total population," 2014–2024): NOT COMPUTABLE.** No
  population file exists in the repo.
- **Residual: NOT COMPUTABLE**, for the same reason as Row 1. (The
  draft's own text already notes that *if* the generation figure alone
  is substituted — 2.20% for 1.53% — the residual implied by the
  *stated* 2.27%/yr population figure would be −0.07%, not −0.73%; that
  substitution is arithmetic on the drafted population number, not a
  recomputation from a stored population file, since none exists.)

### Row 3 — "connected population, 2010–2024": generation 2.63%/yr, denominator 4.36%/yr, residual −1.66%/yr

- **Generation: COMPUTABLE, and a second, previously unflagged
  discrepancy found.** `data/demand/electricity_generation_owid.csv`,
  years 2010 (26.13 TWh) and 2024 (40.13 TWh), 14-year window:
  $(40.13/26.13)^{1/14}-1 = $ **3.11%/yr**, not the drafted 2.63%/yr —
  the same figure computed for Row 1's nearest window above, since both
  rows share the same 2010–2024 generation endpoints. **This row is not
  flagged `[DOES NOT REPRODUCE]` in the draft, but it does not
  reproduce from the stored file either.** Tracing the drafted 2.63%
  figure: it matches `data/demand/README_step1_per_capita.md`
  ("Result 2 — connected-population basis," the `2010–2024` row: `2.63%
  | +1.90% | +4.36% | −1.66%`) — i.e. the draft's Row 3 was transcribed
  from that README table, not recomputed from the CSV now checked into
  the repo. The README and the stored CSV disagree on 2010–2024
  generation growth (2.63% vs 3.11%) by more than the Row 2 discrepancy
  in relative terms, which suggests the README's underlying pull
  predates, or differs from, the currently stored Ember/OWID file —
  consistent with the claims ledger's "could be a different Ember
  vintage" conclusion, now shown to extend beyond the one row it had
  flagged.
- **Access-rate component (part of the "connected population"
  denominator): partially computable, does not fully reproduce
  either.** `data/demand/electricity_access_worldbank.csv` ends at 2023,
  not 2024, so the exact 2010–2024 window cannot be closed; the nearest
  computable window, 2010–2023 (13 years, `48.0% → 61.2%`), gives
  **1.89%/yr**, close to the README's quoted "+1.90%/yr" for its
  (nominally 2010–2024) window — the access component is the one piece
  of this row that comes close to reproducing, on a slightly shorter
  window.
- **Denominator ("connected population" = total population × access
  rate) as a whole: NOT COMPUTABLE.** Even with the access-rate
  component available, connected population requires total population
  as the other factor, and no population file exists in the repo.
- **Residual: NOT COMPUTABLE**, for the same reason.

### Summary

**None of the three residual figures in the draft's decomposition table
can be recomputed from files actually stored in the repo**, because no
population series — total or the total-population component underlying
"connected population" — exists anywhere in the repository. Of the
three generation-growth figures, only two windows are computable at
all (Rows 2 and 3; Row 1's exact window cannot be evaluated since the
stored file stops at 2024), and **both computable rows disagree with
the drafted figures**: Row 2 by a factor of 1.4x (2.20% vs 1.53%,
already flagged and now independently reconfirmed), and **Row 3 by a
newly-identified discrepancy (3.11% vs 2.63%), not previously flagged
in the draft**. The access-rate sub-component of Row 3's denominator is
the one figure that comes close to reproducing. **Confirms the draft's
own "Reproducibility hold" note**: this table cannot go into the
chapter until every row is recomputed from a stored, versioned file —
and that currently means acquiring and checking in a population series
before any residual in this table can be stated at all.

No value was changed in `src/` to produce this section — it is a
read-only recomputation from the two files that do exist.

---

## Job 6 — Correcting my own over-attribution to Liddle & Hasanov

**What was wrong.** When Job 4 (above) corrected the citation from
Liddle, Smyth & Zhang to Liddle & Hasanov, the surrounding text in both
`src/scenarios.py:454-459` and `docs/methodology_demand_horizon.md`
§1.3 was left reading "middle-income **NON-SSA** panel [Liddle &
Hasanov...]" — i.e. it kept asserting that Liddle & Hasanov's own panel
excludes sub-Saharan Africa. **That was never checked.** The "non-SSA"
claim was true of whatever the model's original SSA-exclusion reasoning
was built on, but middle-income economies include Nigeria, Ghana and
Kenya, and nothing in this repo establishes which countries Liddle &
Hasanov's (2023) panel actually covers. Correcting the author names
while leaving an unverified claim attached to them was an error — the
citation now supported a claim about the paper's own sample that no one
had verified.

**Corrected.** Both sites rewritten so the citation supports only the
elasticity *range*, and the sub-Saharan exclusion is stated as the
model's own methodological choice, not an attribute of the source:

- `src/scenarios.py:454-459` (inside `demand_growth_scenarios()`'s
  docstring): now reads "Liddle & Hasanov (2023), Environmental and
  Ecological Statistics, find 0.7-0.9 for middle-income economy-wide
  electricity demand. The model's narrower 0.76-0.90 excludes a
  sub-Saharan range (0.60-0.76) rejected as rationing-contaminated;
  that range is unsourced here."
- `docs/methodology_demand_horizon.md` §1.3: same correction, adapted
  to the surrounding markdown prose.

**No value changed** in either file — this is the second citation-only
correction to this same passage in this task. The excluded sub-Saharan
range (0.60–0.76) itself remains unsourced in the repo, as both
corrected passages now state explicitly rather than implying it traces
to the same (now-corrected) citation.

---

## Job 7 — Verify `data/demand/step1_per_capita_diagnostic.xlsx` before trusting it

**No `openpyxl` in this environment; read directly** via `zipfile` +
`xml.etree.ElementTree` against the OOXML parts
(`xl/worksheets/sheet*.xml`, `xl/sharedStrings.xml`,
`xl/workbook.xml`/`.rels` for sheet-name-to-file mapping). Five sheets:
`README` (sheet1), `Raw_NERC` (sheet2), `Raw_Ember_WDI` (sheet3),
`Per_Capita` (sheet4), `Decomposition` (sheet5).

### `Raw_Ember_WDI` — column headers, year range, columns B/C/D in full

Header row (row 3): **A = "Year", B = "Ember gen (TWh)", C = "WDI
population", D = "Access rate (fraction)", E = "Source"**. Data rows
4–19, **year range 2010–2025** (16 years). Column E is constant on
every row: `"Ember / WDI SP.POP.TOTL / WDI EG.ELC.ACCS.ZS"`.

| year | B: Ember gen (TWh) | C: WDI population | D: access (fraction) |
|---|---|---|---|
| 2010 | 26.13 | 166,642,886 | 0.48 |
| 2011 | 27.04 | 171,379,598 | 0.559 |
| 2012 | 28.74 | 176,200,625 | 0.53 |
| 2013 | 28.93 | 181,049,443 | 0.556 |
| 2014 | 32.28 | 185,896,915 | 0.542 |
| 2015 | 33.17 | 190,671,878 | 0.525 |
| 2016 | 36.54 | 195,443,700 | 0.593 |
| 2017 | 32.24 | 200,254,579 | 0.544 |
| 2018 | 36.71 | 204,938,755 | 0.565 |
| 2019 | 36.46 | 209,485,641 | 0.554 |
| 2020 | 38.0 | 213,996,181 | 0.554 |
| 2021 | 39.2 | 218,529,286 | 0.595 |
| 2022 | 38.0 | 223,150,896 | 0.605 |
| 2023 | 40.92 | 227,882,945 | 0.612 |
| 2024 | 37.56 | 232,679,478 | 0.625 |
| 2025 | 41.54 | 237,527,782 | **(empty — no access-rate value recorded for 2025)** |

### Check against the expected values

| check | expected | found | match? |
|---|---|---|---|
| generation 2010 | 26.13 | 26.13 | ✓ |
| generation 2014 | 32.28 | 32.28 | ✓ |
| generation 2024 | 37.56 | 37.56 | ✓ |
| generation 2025 | 41.54 | 41.54 | ✓ |
| population 2010 | 166,642,886 | 166,642,886 | ✓ |
| population 2024 | 232,679,478 | 232,679,478 | ✓ |
| population 2025 | 237,527,782 | 237,527,782 | ✓ |
| access 2010 | 48.0% | 48.0% | ✓ |
| access 2023 | 61.2% | 61.2% | ✓ |
| access 2024 | 62.5% | 62.5% | ✓ |

**All ten checks match exactly. No mismatch. Not stopping.**

### `Raw_NERC` — contents, and whether it holds 37,093.70 GWh

Header row (row 3): **A = "Year", B = "Q1 (MWh)", C = "Q2 (MWh)", D =
"Q3 (MWh)", E = "Q4 (MWh)", F = "Annual (GWh)", G = "Total (TWh)", H =
"Note / integrity flag"**. Data rows 4–13, years **2017–2026**, quoted
in full:

| year | Q1 | Q2 | Q3 | Q4 | Annual (GWh) | Total (TWh) | note |
|---|---|---|---|---|---|---|---|
| 2017 | — | — | 7,399,765.1 | 8,705,606 | — | — | "Q1,Q2 missing — annual total not computable" |
| 2018 | 8,511,481 | — | — | 8,878,473 | — | — | — |
| 2019 | 8,878,473 | 8,451,428 | 7,984,685 | 8,101,193 | — | 33.41578 | — |
| 2020 | — | — | — | — | — | — | "not gathered" |
| 2021 | — | — | — | — | — | — | "not gathered" |
| 2022 | — | — | — | — | — | 34.93488 | "[F2] source gives 3,988 MWh/h — see G8 formula" |
| 2023 | — | — | — | — | 36,710.38 | 36.71038 | "annual total, NERC" |
| **2024** | — | — | — | — | **37,093.70** | **37.09370** | **"annual total, NERC — model Tier 1 anchor"** |
| 2025 | — | — | — | — | 39,193.93 | 39.19393 | "annual total, NERC" |
| 2026 | — | — | — | — | — | — | "[F5] Q1 only (8,883.47 GWh) — partial year, excluded" |

Source note (row 15): *"Source: NERC quarterly reports. Units as
printed in source — MWh, GWh and MWh/h all appear; see [F2]."*

**Yes — `Raw_NERC` holds the 37,093.70 GWh base-year figure exactly**,
at 2024 (F11 = 37093.70, G11 = 37.0937), explicitly flagged in the
sheet's own H column as **"annual total, NERC — model Tier 1 anchor."**
This is the demand-base figure used throughout `src/scenarios.py` and
`docs/methodology_demand_horizon.md`.

**Read from the `README` sheet, since it bears directly on trusting the
rest of the workbook:** three named sources (NERC — "AUTHORITATIVE for
level and metering point," but "incomplete: only 2019 has four
quarters, and 2023–2025 have annual totals"; Ember — "Used for SHAPE
only, not level"; WDI `SP.POP.TOTL`), and five numbered flags, the most
consequential being **[F3]: "The NERC-to-Ember level gap is NOT a
stable definitional offset: +8.8% (2022), +11.5% (2023), +1.3% (2024),
+6.0% (2025). A stable offset would permit clean index-splicing; this
does not. The two series are therefore reported side by side and never
merged."** This is the same discipline this task's item 2 (below)
applies to the Ember-vs-OWID conflict.

---

## Job 8 — Extraction to versioned CSVs

Three files written to `data/demand/`, one per series in
`Raw_Ember_WDI`, none overwriting an existing file:

- `data/demand/electricity_generation_ember_diagnostic.csv` — Ember
  generation (TWh), 2010–2025.
- `data/demand/population_wdi_diagnostic.csv` — WDI `SP.POP.TOTL`,
  2010–2025.
- `data/demand/electricity_access_wdi_diagnostic.csv` — WDI
  `EG.ELC.ACCS.ZS`, 2010–2024 (2025 omitted — the source cell is
  empty).

Each carries a `#`-prefixed header block recording: the named source
(Ember / WDI, with indicator code where applicable), the year range,
an explicit statement that it was extracted from
`step1_per_capita_diagnostic.xlsx` rather than pulled from the primary,
and the extraction date (2026-09-24). `data/demand/electricity_generation_owid.csv`
and `data/demand/electricity_access_worldbank.csv` were **not**
touched — `git status --short` on both after this extraction shows no
change. The generation conflict (Ember 37.56/40.92 TWh for 2024/2023 vs
OWID 40.13/42.51 TWh for the same years) and the access-series overlap
(matching WDI figures for 2010–2023, extended one year by the
diagnostic) now both exist as separate, named files rather than one
overwriting the other's evidence.

---

## Job 9 — §3.2.3 table recomputed on both series, side by side

**Population and access figures come from the one population series
that exists (WDI, just extracted) in both columns below — Ember and
OWID differ only in the generation numerator.** All three windows
matched exactly against the workbook's own `Decomposition` sheet
formulas (cross-checked to 10+ significant figures) before being
restated here.

| row | window | generation CAGR — Ember | generation CAGR — OWID | population/connected-pop CAGR | residual — Ember | residual — OWID |
|---|---|---|---|---|---|---|
| 1: total population | 2010–2025 | **3.14%** (26.13→41.54, 15 yr) | **not computable** — `electricity_generation_owid.csv` has no 2025 row | 2.39% (166,642,886→237,527,782, 15 yr) | **0.73%** | — |
| 2: total population | 2014–2024 | **1.53%** (32.28→37.56, 10 yr) | 2.20% (32.28→40.13, 10 yr) | 2.27% (185,896,915→232,679,478, 10 yr) | **−0.73%** | −0.07% |
| 3: connected population | 2010–2024 | **2.63%** (26.13→37.56, 14 yr) | 3.11% (26.13→40.13, 14 yr) | 4.36% (connected pop 79,988,585→145,424,674, i.e. total pop × access rate 48.0%→62.5%, 14 yr) | **−1.66%** | −1.20% |

**On the Ember series, all three rows of the draft's decomposition
table reproduce exactly** (3.14%/2.39%/0.73%; 1.53%/2.27%/−0.73%;
2.63%/4.36%/−1.66% — matching the draft to two decimal places in every
cell, and matching `step1_per_capita_diagnostic.xlsx`'s own
`Decomposition` sheet to 10+ significant figures, e.g. row 5's
generation CAGR `3.1387361947803782E-2`). **This corrects Job 5,
above, in this same document**: Job 5 checked the drafted figures
against `electricity_generation_owid.csv` (the only generation file
identified as stored at the time) and correctly found two of three
rows did not reproduce against *that* file — but the figures were never
computed from OWID in the first place. They are computed from the
Ember series in this workbook, which had not yet been located or
extracted. Against the correct source, the reproducibility question
resolves the other way: nothing in the draft's decomposition table
needs correcting; the *table* was right, and Job 5's "does not
reproduce" verdict was a comparison against the wrong file, now
superseded by this job. The `merged_correction_plan_v2.md:1606`
citation (`` `constrained_continuation` growth | Ember 2010–2025 CAGR;
`data/demand/step1_per_capita_diagnostic.xlsx` | ✅ 0.0314 ``), found
during Job 10 below, already named the correct source; it was not
cross-referenced during Job 5.

### Does `constrained_continuation` = 3.14% reproduce from the Ember series over 2010–2025?

**Yes, exactly.** $(41.54/26.13)^{1/15}-1 = 3.1387\%$, rounding to
3.14%, computed from `data/demand/electricity_generation_ember_diagnostic.csv`
(equivalently, `Raw_Ember_WDI` rows 4 and 19) — and this matches
`step1_per_capita_diagnostic.xlsx`'s own `Decomposition!D5`
(`3.1387361947803782E-2`), which that same workbook's
`Decomposition!D26` row labels **"CARRIED INTO scenarios.py —
constrained_continuation (central)."** The workbook is not merely
consistent with the code value; it is the recorded derivation of it.

**On the OWID series, 2010–2025 cannot be evaluated at all** — the
stored `electricity_generation_owid.csv` has no 2025 row, exactly as
found in Job 5.

---

## Job 10 — Every docstring/methodology line naming Ember, OWID or WDI

**Report only — nothing changed.** Grouped by file; live code and
methodology sites first, then the data-pipeline script that actually
produces the OWID file, then historical/planning documents found by
the same search.

### `src/scenarios.py` (live docstrings)

| line | text |
|---|---|
| 488 | `Ember 2010-2025 grid generation CAGR. This is a CONSTRAINT-PERSISTENCE` (inside `demand_growth_scenarios()`, deriving `constrained_continuation`) |
| 501 | `historical averages against WDI NY.GDP.MKTP.KD; IMF WEO covers only` (a different WDI indicator — GDP, `NY.GDP.MKTP.KD` — not the population/access indicators this task's items 1–3 concern; noted for completeness) |

**`src/scenarios.py` names no data file at either site** — no
`electricity_generation_owid.csv`, no reference to the diagnostic
workbook, and no `SP.POP.TOTL`/`EG.ELC.ACCS.ZS` indicator codes for the
population/access series used in the Route A decomposition. The
derivation is stated in prose only.

### `docs/methodology_demand_horizon.md` (live methodology)

| line | text |
|---|---|
| 74–75 | `observed generation growth against population [SOURCED: World Bank WDI` / `` `SP.POP.TOTL`] and access [SOURCED: WDI `EG.ELC.ACCS.ZS`]: `` |
| 91 | `**Direction of bias in the connected-population figure:** WDI access counts` |
| 115 | table row: `` `constrained_continuation` \| **3.14%** \| [DERIVED] Ember 2010–2025 grid generation CAGR `` |
| 150 | `behind Kenya** [DERIVED from Ember generation and WDI population]. Ghana cannot` |

**No file path is named at any of these four sites either** — the
methodology text cites the indicator codes (`SP.POP.TOTL`,
`EG.ELC.ACCS.ZS`) and "Ember," but not
`data/demand/electricity_generation_owid.csv`,
`data/demand/step1_per_capita_diagnostic.xlsx`, or any other specific
file.

### `docs/chapters/chapter3_2_draft.md` (the draft itself, for completeness — not "methodology" proper but the nearest thing to it currently in the repo)

| line | text |
|---|---|
| 69 | `` `SP.POP.TOTL`, `EG.ELC.ACCS.ZS`) and Ember generation data (Ember, `` |
| 79–80 | `` (`data/demand/electricity_generation_owid.csv`) gives **2.20%** for `` — the one site in the whole repo that names the OWID file explicitly in a derivation context |
| 180 | table row: `` `constrained_continuation` \| 3.14% \| 2010–2025 grid-generation growth (Ember) `` |
| 379–380 | `` Ember — electricity generation data. Stored in the repo as `electricity_generation_owid.csv` (Our World in Data re-hosting, not `` |

### `data/make_data.py` (the actual data-pipeline script — not a "methodology" document, but this is the literal provenance of the OWID file and belongs in this report)

| line | text |
|---|---|
| 24–26 | `OWID_ENERGY_URL = ("https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv")` |
| 52–63 | `def fetch_owid_demand(): df = safe_read_csv(OWID_ENERGY_URL, "OWID energy data") ... out.to_csv(BASE_DIR / "demand" / "electricity_generation_owid.csv", ...)` |

This confirms `electricity_generation_owid.csv` is fetched
programmatically from **OWID's own compiled `owid-energy-data.csv`**
(OWID's global energy dataset, which blends multiple underlying
sources by country and year), **not** a direct pull of Ember's own
Electricity Data Explorer — a materially different pipeline from the
hand-transcribed `Raw_Ember_WDI` sheet in the diagnostic workbook, and
a plausible structural explanation for why the two series diverge in
2023–2025 despite matching in 2010 and 2014.

### Historical / planning documents (found by the same search, not "methodology" proper, reported for completeness)

| file:line | text |
|---|---|
| `docs/merged_correction_plan_v2.md:234` | `` `constrained_continuation` \| **0.0314** \| Ember 2010–2025 generation CAGR. Constraint-persistence, **not** organic demand. Window range 1.53–3.22% `` |
| `docs/merged_correction_plan_v2.md:1606` | `` `constrained_continuation` growth \| Ember 2010–2025 CAGR; `data/demand/step1_per_capita_diagnostic.xlsx` \| ✅ 0.0314 `` — **the one pre-existing site in the repo that already names the diagnostic workbook as the source** |
| `docs/merged_correction_plan_v2.md:190` | `notably WDI access includes off-grid, so the connected-basis decline is` |
| `docs/merged_correction_plan_v2.md:1614` | `` **GDP growth arms** \| WDI `NY.GDP.MKTP.KD` (pre/post-2015 averages) + IMF WEO \| ⚠ 4.5% central declared, history not yet pulled `` |
| `docs/findings_provisional_pre_sweep.md:67` | `Caveat: WDI access includes off-grid/DRE, so the connected-basis decline is` |
| `docs/scenario_coherence.md:55` | `` `constrained_continuation` \| 3.14% \| Ember 2010–2025 grid generation CAGR. A **constraint-persistence** arm, not organic demand growth `` |
| `docs/methodology_claims_ledger.md:385` (row 198) | cites `data/demand/electricity_generation_owid.csv` explicitly as the file checked against the 3.14% claim — the Phase-1 audit that did not have the diagnostic workbook's Ember column available as a comparison at the time |
| `docs/methodology_claims_ledger.md:386` (row 199) | same file, checked against the 1.53% claim; concludes "Could be a different Ember vintage; not reproducible from the repo" — now resolved by Job 9 above: it is a different vintage (the diagnostic workbook's Ember column, not the OWID file), and it does reproduce against that vintage |
| `docs/methodology_claims_ledger.md:399` (row 212) | `needs Ember generation + WDI population for Kenya/Ghana; not in the repo` |
| `data/demand/README_step1_per_capita.md:94,108–110,123,134,142,153,155` | the diagnostic workbook's own companion README (already read in full under Job 7); names Ember, WDI `SP.POP.TOTL`, WDI `EG.ELC.ACCS.ZS`, and the NERC-vs-Ember gap explicitly |

**Summary for provenance correction.** Once a canonical source is
chosen (Ember-via-diagnostic-workbook vs OWID-via-`make_data.py`), the
sites that will need a citation change are: `src/scenarios.py:488`
(names "Ember" with no file), `docs/methodology_demand_horizon.md:115,150`
(same), and `docs/chapters/chapter3_2_draft.md:79-80,180,379-380`
(the draft, not yet submitted). `docs/merged_correction_plan_v2.md:1606`
already names the correct file and needs no change if the diagnostic
workbook is chosen as canonical.
