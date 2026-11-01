# Chapter 4 — figure schedule

Eleven figures, mapped to sections. Each entry gives what it shows, the
file and fields it is built from, where it is placed, and whether a
plotting script already exists.

**Standing requirements.** Every figure is built from a vintage result
file (`gas_vintage_dispatch=True`) and its caption names that file.
Every figure reports to 2045 unless it is explicitly a full-horizon
quantity, per §4.1.3. No figure duplicates a table that is already in
the chapter — Figures 4.2 and 4.5 replace tables, they do not accompany
them. No figure is captioned "leakage" unless it plots an increment over
an uncapped counterfactual.

---

## §4.1 — Basis of the reported results

### Figure 4.1 — Capacity trajectory of the central solution, 2025–2045

Stacked area or grouped bar: cumulative solar (MW), gas (MW) and
storage (MWh, second axis) by year. Shade 2046–2054 or omit it; if
shown, label it as the end-effect buffer.

- **Source:** `results/baseline_sliced/summary__661b03c1.json`,
  `decision_variables.solar_add_mw_by_year`, `gas_add_mw_by_year`,
  `storage_add_mwh_by_year`, cumulated.
- **Placement:** after §4.1.2's second paragraph.
- **Script:** none directly; `plot_system_diagnostics.py` is the nearest
  starting point.
- **Caption must state** that stocks are reported and annual additions
  carry the horizon-sensitivity bound of §4.1.3.

### Figure 4.2 — Decomposition of system cost

Single stacked bar, or a bar per term: capital charge ⟦40.00⟧ per cent
split into its five lines, fuel ⟦32.25⟧, operation and maintenance
⟦27.75⟧. Avoid a pie chart; the five capital lines are the point and a
pie hides them.

- **Source:** `results/baseline_sliced/diagnostics__661b03c1.json`,
  `cost_decomposition.components_unrounded`; objective from the summary.
- **Placement:** replacing the prose list in §4.1.2's third paragraph,
  which then reduces to one sentence and a cross-reference.
- **Script:** `scripts/16_plot_cost_decomposition.py` exists and reads
  this block. Confirm it reads the vintage file.

---

## §4.2 — Determinants of system cost

### Figure 4.3 — Lever ranking, unconstrained

Horizontal bar chart (tornado), fourteen ranked families, longest at
top. Mark the three unranked families in a separate panel or with
distinct shading, with the reason in the caption: hydro as a bound,
new-gas deliverability as not comparable, and the coverage ratio with no
build response. The Master Plan deliverability family is ranked.

- **Source:** `results/lever_ranking/arms_661b03c1/lever_ranking.csv`,
  the live-arms column.
- **Placement:** replacing the table in §4.2.3 (§4.2.2 before the RQ
  patch). Keep the table only if the style guide requires tabulated
  figures; otherwise the chart carries it.
- **Script:** none. The CSV is three columns and plots directly.

### Figure 4.4 — Build response against span

Scatter: span on the x-axis, a build-response measure on the y-axis —
the number of distinct plans, or the maximum terminal capacity
difference. This is the figure that makes the ranking rule visible, and
it shows at a glance why two families with large spans are not ranked.

- **Source:** the same CSV, build-response columns.
- **Placement:** §4.2.2, after the rule is stated.
- **Script:** none.

### Figure 4.5 — Unconstrained against capped ranking

Paired horizontal bars or a slope chart: each family's unconstrained
rank on the left, its rank under the conditional cap on the right, with
lines connecting. The three substantive moves — gas build sixth to
third, combined-cycle efficiency fourteenth to fourth, the
deliverability families collapsing — are visible in one image and take
a paragraph in prose.

- **Source:** the uncapped CSV and
  `results/lever_ranking_capped/.../lever_ranking.csv` at
  `capital_case=unconstrained`.
- **Placement:** replacing the second table in §4.2.6.
- **Script:** none.

### Figure 4.6 — First-order Sobol indices

Bar chart, five parameters, first-order and total-effect indices side
by side on system cost. One bar at ⟦0.9466⟧ beside four near zero is
the whole argument of §4.2.7's third paragraph.

- **Source:** the Sobol output at `arms_09bd6dbc` (the keyed path, not
  the unkeyed file — see L-entry on the stale draws file).
- **Placement:** §4.2.7.
- **Script:** `29_run_sobol` writes the indices; no plotter.

---

## §4.3 — Endogenous capital structure

### Figure 4.7 — The conditionality of the service-financed channel

Scatter, one point per configuration (thirty-six). X axis: the public
capital budget dual, taken from the `public_only` row. Y axis: the value
of the service-financed channel, public-only objective minus
service-financed objective. Eighteen configurations where the dual is
zero, in which the value does not exceed ⟦−0.61⟧ dollars, against
eighteen where it is positive, in which the value is never below
⟦653,600,175⟧ dollars; nothing between. This is the thesis's cleanest
result and prose understates it.

The figure does not show the value of determining the
concessional–commercial division, which is a different quantity and is
not measured.

- **Source:** `results/dem2/arms_1fa34539/dem2_results.csv` (the
  vintage run cited in F-38's re-derived block; F-38's original `Source`
  line names the fixed-name `results/dem2/eaas_value.csv`).
- **Placement:** §4.3.2, after the second paragraph.
- **Script:** `scripts/fig_4_7_budget_dual_separation.py` (TASK 95).
- **Note:** the zero-dual cells sit at ⟦−0.61⟧ dollars, which a log axis
  cannot show. The figure uses a broken y axis with the break labelled
  (upper panel in million dollars, lower panel in dollars).

### Figure 4.8 — Composition of the capital programme

Grouped bars: the five capital lines under the preceding specification
against the corrected one. Public solar falling from ⟦22.2⟧ to ⟦13.1⟧
per cent of system cost while the gas capital charge rises by a factor of
⟦3.75⟧, from ⟦0.635⟧ to ⟦2.383⟧ billion dollars, is the substance of
§4.3.4.

- **Source:** `cost_decomposition` from the vintage and blended central
  diagnostics.
- **Placement:** §4.3.4.
- **Caption must state** that the comparison is across model
  specifications, not across scenarios.

---

## §4.4 — Emissions outside the grid boundary

### Figure 4.9 — Out-of-boundary share and policy-induced increment

Grouped bars by capital tier, one panel per collection rate: the capped
out-of-boundary share, the uncapped share, and the increment between
them, with the ⟦10⟧ per cent threshold as a horizontal line. The cells
without an uncapped counterpart (`unconstrained` at ⟦60⟧ and ⟦35⟧ per
cent collection) are marked.

- **Source:** the vintage FIN-3 files `results/fin3/arms_1d68be49`
  (capped) and `results/fin3/arms_7f91f752` (uncapped, `--no-cap`).
- **Placement:** §4.4.2, after the third paragraph.
- **Script:** `scripts/fig_4_9_outofboundary_and_increment.py`.
- **Caption must state** that the field named `leakage_share` is the
  out-of-boundary level, that the increment equals the level on this
  generation because uncapped self-generation is zero, and that this is
  a measurement rather than a definition (register L150).

### Figure 4.10 — Cost–emissions frontier

Scatter: system cost on the x-axis, national emissions on the y-axis,
one series per financing arm, points ordered by cap fraction and joined.
Mark non-dominated points. The backward bend in the two public arms and
its absence in the service-financed arms is the figure's purpose.

- **Source:** `18_run_pareto_frontier` output, vintage generation.
- **Placement:** §4.4.5.
- **Script:** `18_run_pareto_frontier.py` may already plot; confirm.
- **Caption must state** that the tight end is unlocated — every point
  to cap fraction ⟦0.01⟧ remained feasible — and that the turning point
  at ⟦0.625⟧ is the lowest sampled point on a ⟦0.0375⟧ grid.

---

## §4.5 — Adequacy across demand realisations

### Figure 4.11 — Distribution of per-draw cost

Two cumulative distribution functions or overlaid histograms: the
committed plan against wait-and-see, with medians and means marked. The
figure shows a cost tail: the committed leg (`19490865`, gas free) is
⟦100⟧ of ⟦100⟧ feasible. The ⟦50⟧ of ⟦100⟧ infeasibility result of
§4.5.2 comes from a different leg, the gas-fixed leg
`summary__857ac191`, and is not shown here. The cost tail drives the
bound in §4.5.4.

- **Source:** the Monte Carlo per-draw outputs behind F-42, vintage
  generation.
- **Placement:** §4.5.4, with a cross-reference from §4.5.2 stating that
  the infeasibility result there is from a different leg.
- **Script:** none.
- **Caption must state** that both legs run at
  `capital_case=unconstrained`.

---

## Figures deliberately not proposed

**A gas balance shadow price series.** It would be informative, but the
dual carries two bases differing by a factor reaching ⟦9.3⟧ (§4.1.4),
and a reader who takes the wrong basis off a chart has no way to know.
If it is wanted, it needs both series on one axis with the basis in the
legend, not the caption.

**An emissions path against cap trajectories.** The caps barely move
between generations and the emissions path is fuel-bound, so the figure
would show two near-flat relationships and say less than the sentence it
replaces.

**Anything from §4.1.5.** The re-derivation tally is process, not
result, and a figure would give it more prominence than it should have
in a results chapter.

---

## Order of work

1. Figures 4.3, 4.5 and 4.7 first. All three read a CSV or a single
   JSON block, all three replace prose or a table, and all three carry
   a headline result.
2. Figures 4.1, 4.2 and 4.10, which have existing scripts or
   near-scripts.
3. The remainder.

Every caption names its source file and its generation. Every figure
that reports a boundary-resolved or grid-located quantity states the
grid in its caption, per §4.1.4.
