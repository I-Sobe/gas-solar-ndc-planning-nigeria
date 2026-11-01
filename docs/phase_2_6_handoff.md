# Phase 2.6 — CLOSED. Handoff to the next session.

Branch `phase-2.6-genset`. Baseline tagged `pre-2.6`.

**Standing rule: results produced during correction are VERIFICATION
instruments, not thesis findings. Nothing is written up until the full six-phase
correction closes.**

---

## 1. What 2.6 delivered

`selfgen` added to both builders as a priced backstop, replacing an uncosted
VoLL cliff with a diesel-priced slope. Parameters in
`data/cost/processed/genset_cost.csv`; derivation in
`data/cost/raw/genset_lcoe_buildup.xlsx`.

| arm | eta_HHV | USD/MWh | EF_e tCO2/MWh |
|---|---|---|---|
| `genset_srmc` | 0.3474 (75% load) | 277 | 0.729 |
| `genset_central` | 0.2809 (25% load) | 445 | 0.902 |
| `genset_high` | 0.0511 (A2EI metered) | 3,323 | 4.638 |

eta from four Caterpillar PRIME datasheets. Fuel anchored Dec-2024..Feb-2026,
NBS AGO/PMS at matched monthly CBN NFEM rates: AGO 1.0157 USD/l (CV 8.3%), PMS
0.7311 (CV 8.7%). Mar-May 2026 excluded as a crude shock, retained as the high
arm. `EF_e` and `SFC` DERIVED from eta at build time, never stored.

Quotable: `genset_central` eta 0.2809 against the gas fleet's 0.287. A
part-loaded Nigerian C&I diesel converts fuel about as efficiently as the grid;
the entire cost gap is the fuel price.

## 2. Findings carried forward

- **Leakage.** Under the correct NDC boundary the model complies while emitting
  **1.24x / 1.54x / 7.89x** the entire cumulative power-sector budget in
  uncounted diesel. Widening the boundary is INFEASIBLE in all three arms.
  All four NDC cases flag at 53-67%.
- **Carbon price is a transfer when capital binds.** `public_tight` emissions
  constant at 903.1 Mt across 0-150 USD/tCO2 while cost rises 78.4 -> 104.3 bn.
  `eaas_tight` responds sharply (389.5 -> 167.3 Mt between 15 and 20).
- **Cost decomposition reconciles exactly.** EaaS spends MORE public money
  (16.3 vs 10.5 bn), Nigerians spend FAR LESS private diesel (31.5 vs 75.6 bn).
- **Residual finding SURVIVES.** Zero MW moved against a 12.67%/14.73% cost
  swing. The backstop is a supply channel, not a capital channel.
- **D9 obligation trajectory — front-loading is CHEAPER.** At constant 0.45
  average: front 90.45 bn / 25,964 MW / 48.7% early; flat 92.54 / 29,265 / 28.0%;
  back 94.69 / 32,566 / 12.3%. **RETRACTS** the earlier 2.6E claim. Deferring
  build means building against larger demand.

## 3. Declared conventions

- Busbar factor **1.0 central**, 1.0753 sensitivity (cuts selfgen 9.5%).
- Obligation renamed (F0): `max_offgrid_fraction` / `grid_service_mode` /
  `m.grid_service_obligation`. Bounds `selfgen + unserved`, not adequacy.
- Mode **`annual`**, never `total` — `total` is slack under a rising deficit.
- Builders accept scalar **or per-year list** for `max_offgrid_fraction`.
- `genset_case` is **REQUIRED** in `load_econ` (no default).
- `unserved` is ~0 BY CONSTRUCTION. Quote `selfgen + unserved`.

## 4. OPEN — filed to their proper phases

| # | Item | Phase |
|---|---|---|
| D17 | EaaS arm has no binding constraint at the 65M tariff; every "EaaS saves X" is constrained-vs-unconstrained | FIN-2 |
| D25 | `04_run_fin2_tariff_sweep`, `12_run_dem1_demand_sensitivity` are `NotImplementedError` stubs | FIN-2 / DEM-1 |
| D21 | `11_run_rel3` threshold logic disagrees with its own `status` column | REL cleanup |
| D23 | `eps_levels` dropped at `optimize_experiments.py:774`; sweep silently uses `reliability_levels_log()` | REL cleanup |
| D24 | Six REL scripts sweep four different eps axes. Needs one documented rationale | REL cleanup |
| D13 | `19_run_carbon_price_sweep` arms too weak; the tipping point IS the result | carbon policy |
| D14 | APERC reporting layer — selfgen TWh/CO2/spend/shortfall over existing runs | reporting |

**D14 motivation, recorded:** the model has **no instrument** pricing diesel's
health, accessibility or acceptability externalities. `carbon_case="no_policy"`
is zero; the cap excludes diesel by design. Those dimensions must be REPORTED,
not pretended into the objective.

## 5. Verification assets

- `scripts/smoke_all.sh` — whole-suite executability. **Current: 29 pass, 2 fail**
  (both D25 stubs). Run after any `src/` change.
- `scripts/verify_2_6_commit_a.sh` — worktree regression vs `pre-2.6`.
- `scripts/probe_2_6_structure.py` — structural probes (balance coefficient,
  objective price, emissions boundary, public-budget isolation).
- Leakage audit:
  `grep -L "leakage_diagnostic" $(grep -ln "emissions_cap_by_year=caps\\|emissions_cap_by_year=cap_by_year" scripts/*.py)`

## 6. Traps that bit us — do not repeat

- An artefact reproducing identically across arms is one assumed parameter, not
  N measurements (ZE-Gen's 2.000 L/kWh across five segments and two fuels).
- A large objective delta is usually the VoLL parameter, not the mechanism under
  test. Happened twice (-94.6%, then 98.7%).
- **Rename when meaning changes.** Silent reverts cost more than loud failures.
  `genset_case` defaulting to None removed the backstop twice in the headline
  NDC runners and produced clean-looking wrong tables.
- **Inventory before editing.** Enumerate every consumer of a changed quantity
  first; do not discover them one traceback at a time.
- Aggregate vs per-year: a horizon-aggregate test reads "slack" while six
  individual years sit pinned at the bound.
- Stale `results/*.json` will silently produce pre-fix numbers. Re-run producers
  before consumers.
- CSV runners need FLAT fields; JSON runners nest.

---

## 7. WORKFLOW — Claude Code is now installed

Installed via npm-global, signed in on Pro, `ANTHROPIC_API_KEY` unset (so it
draws on the subscription, not per-token billing). **Usage is a SHARED pool with
this chat** — a heavy agent run consumes the same budget.

**Split by task type:**

| Use Claude Code for | Use chat for |
|---|---|
| Inventory passes ("find every consumer of X") | Design decisions and trade-offs |
| NameErrors, stale kwargs, missing constants | Parameter sourcing and provenance |
| Applying edits against real line numbers | What a result MEANS for the thesis |
| Reading tracebacks, running the smoke suite | Deciding which of several routes to take |

**Rules that made it work:**
- Launch from the repo root, with `conda activate nigeria-energy` first.
- Bound every task. "Run the smoke suite and diagnose failures" is cheap;
  "fix everything broken" is not.
- Say **"show me the diff before applying"** — that is the checkpoint.
- Approve individual commands (option 1), not "don't ask again" (option 2),
  until you trust a command class.
- Run plain shell things (`bash scripts/smoke_all.sh`, `python scripts/NN_*.py`)
  **in Git Bash directly** — costs no tokens. Save Claude Code for diagnosis.
- Let long runs finish before starting another task; several runners write to
  `results/` and a concurrent read gets half-written data.

**Cost control in chat:** start fresh chats often (this one carried the entire
2.6 history and every turn re-sent it); send `sed -n 'A,Bp' file` slices, not
whole files.

`CLAUDE.md` in the repo root carries the standing rules and is read
automatically each session.

## 8. Not yet endogenous

Cost of capital remains exogenous. `blended_finance.py` supplies
`r_concessional`, `r_commercial`, `concessional_share` as scenario inputs;
nothing feeds back from the solve. **That is Phase 3, and 2.6 is what makes it
meaningful** — without a priced backstop the counterfactual to grid investment
was VoLL, and any rate looked bankable.

Also open from 3.1: the 18% commercial anchor is almost certainly nominal naira
and the 6% concessional nominal USD. Blending them and applying to real-USD
cashflows is dimensionally incoherent. See `docs/currency_and_rate_basis.md` §6.
Note the supporting evidence found during 2.6: Nigerian fuel prices are stable
in USD across a 12.7% naira move, and Dangote now dollar-prices refined product
outright — which makes USD-denominated financing the natural instrument and
strengthens the case for sourcing rates directly in USD rather than converting.

## 9. Next

**2.7 — tiered demand**, which depends on 2.6. Note the rule established here:
`selfgen` and the Tier 2 self-generation wedge are **the same variable at
different tiers**. Never add both. This is also the mechanism that finally makes
λ endogenous, resolving λ(t) = λ(0) structurally rather than by a growth-rate
assumption.
