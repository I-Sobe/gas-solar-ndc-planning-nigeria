# Handover — session close, September 2026

**Read this first in a new chat.** It states where the project is,
what is true, what is open, and how the work is run. Where it
disagrees with an older document, this one wins.

---

## 1. Model state

**Objective frozen at HEAD `2a9226b`.** Every commit since has been
diagnostics, reporting fields, audit remediation, or runner revival —
all verified by anchor reproduction.

    regression   conv_none + pre_f1_overnight + dscr_regression
                 + tariff_legacy_65m + om_none
                 21,811,042,472.821114 USD  /  45,502.9754 MWh

    central      conv_central + f1_tranche + dscr_central + cr_central
                 + env_dares + baseline deliv + hydro_flat
                 + tariff_central + om_central
                 27.954 bn USD  /  65,612.985 MWh  /  41,031.4 MW solar
                 capital mobilised 2025-2045: 12,772,100,150 USD
                 backcast -3.2% against observed 2024 gas generation

**Run both after ANY change to `src/`.** If either moves, nothing
below is attributable.

Canonical documents: `docs/frozen_state_2026_09_10.md`,
`docs/findings_frozen.md`, `docs/limitations_register.md` (L1-L45),
`docs/forensic_audit.md` (A1-A59), five methodology documents.

---

## 2. THE CORRECTION THAT MATTERS MOST

**The freeze is provenance discipline, not a content filter.**

It means every reported number comes from a known, stated model state.
It does **not** mean only the 38 swept arms may be cited. Any result
run at the frozen state is reportable regardless of when it was
produced.

Already reportable and produced post-freeze: `om_low`/`om_high`, both
ε-constraint frontiers, the 1,792-solve Sobol decomposition, the
regenerated selfgen frontier.

**Do not let "not in the frozen sweep" become a reason to exclude a
result.** That reasoning was applied wrongly during this session and
corrected by the author. The standard is rigour, not tidiness — and
unfreezing is acceptable if a result needs it.

---

## 3. Work method

**Three roles.** Chat designs and writes bounded TASK.md files and
prose. Claude Code executes them — edits against real line numbers,
inventories, audits. **Zeus runs every solve, every runner, every
commit.** Chat and Code run nothing.

**TASK.md conventions that work:**
- An exhaustive whitelist of permitted actions. Everything else
  forbidden, explicitly.
- State what is already COMPLETE so the agent does not re-verify.
- **A STOP after any inventory step whose findings could change the
  task's shape.** This has caught real errors every time.
- Require incremental reporting — two early runs produced edits across
  sixteen files and no report.
- Never trust a list Chat supplies. **Every "established facts" list
  given in this session contained at least one error** — the agent
  should establish the facts itself.
- For audits: read bound wide, write bound narrow. Finding something
  and leaving it broken is correct behaviour.

---

## 4. What was done this session

**O&M implementation** — no technology carried any O&M cost. Wired
from NREL ATB 2024, three sourced cases. **+6.602 bn on central,
+30.9%** — the largest single correction in the project. Sourced band
measured: 27.531 / 27.954 / 28.542 bn, span 1.011 bn.

**Capital mobilised** — `npv_gross_capex_report_window_usd` was named
in two authoritative documents and written by no code. Now produced:
**12.772 bn** at central. Gives F-1 and F-2 a proper denominator (the
capital-recovery span is 31.1% of capital, against 14.2% of system
cost).

**Two Pareto frontiers by ε-constraint**, both ported to the sliced
builder: cost-emissions (`18_`) and cost-capital (`28_`), with duals,
binding checks and non-dominated verification.

**Sobol parameter uncertainty** (`29_`) — 1,792 solves, 0 failures.
**Gas availability explains 90% of variance** in both cost and
capital. First-order = total-order everywhere (no interactions).

**F-21** — tightening a grid-only cap by 40% RAISES national emissions
by 37.5 Mt and costs 10.24 bn. Robust to supply-chain CO2e; reverses
only at a 49% uplift.

**Three-phase forensic audit** — 59 findings, 16 fixed, **zero ever
LIVE-REPORTED**. Register grew 28 → 45 entries.

**Nine dead runners revived** (`0012f16`): `02_run_eaas`, `06_`, `07_`,
`08_gas3`, `09_`, `10_`, `11_`, `13_`, `probe_2_6_structure`.

---

## 5. IMMEDIATE NEXT STEPS

**5.1 — Run the nine revived runners.** Commands are in
`docs/task_runner_revival.md` under `## PART 2A`, but **every one
omits `--reserve-margin nirp_spinning`**, which became required after
they were drafted. Add it.

**All nine use the ANNUAL builder** and must run `conv_none`. The
annual builder cannot represent conversion (5.91 bn, the second-largest
lever) and storage behaves differently there (78,882 MWh built, never
discharged, against 65,613 dispatched in sliced central). **That is a
scope condition on every result they produce.** Whether to port the six
shared functions in `optimize_experiments.py` to the sliced builder is
an open decision — run them first, then judge.

**Two cautions:**
- **`13_` must not be run for EaaS-arm figures** until A56 is closed.
  Its 36 EaaS rows carry a hardcoded `required_margin = 1.10`
  regardless of `--dscr-case`. Wrong output, not an error. The
  `public_only` half is fine.
- **`11_` carries A45's `required_margin` double-count shape**, found
  during revival and correctly left alone.

**5.2 — GAS-1 is the priority.** `06_run_gas1_shadow_benchmarks.py`
gives the gas shadow price against export and domestic benchmarks —
the opportunity-cost question (a molecule in a turbine versus on a
ship). Most departmental result available and never yet run. Connects
directly to the NIRP defence.

**5.3 — Still dead, out of scope so far:** two SUPERSEDED runners, two
STUBs, `17_run_monte_carlo`, anything BACKEND-BLOCKED. See
`docs/task_runner_revival.md` Part 1.

**5.4 — Then:** a script-to-question map across all of `scripts/`, to
find duplicates and retire overlap. The author asked for this
explicitly. Part 1's inventory already records each docstring question.

**5.5 — Then the figures layer.** A drafted task exists
(`TASK_figures_layer_deferred.md`): scripts reading frozen results,
**no solves**, one per thesis figure, with a shared guard module.
`16_plot_cost_decomposition.py` is the template. Deferred until the
result set is complete.

---

## 6. UPDATES NEEDED — methodology and register

### 6.1 — methodology: one claim is now WRONG

**`methodology_protocols.md` §6 states the model is deterministic and
that "the thesis does not claim otherwise."** That is still true about
stochasticity — but the document should now also state that the model
performs **bi-objective optimisation by the ε-constraint method** on
two objective pairs (cost-emissions, cost-capital), with binding
checks, trade-off duals and non-dominated verification.

ε-constraint **is** a recognised multi-objective method. Chat argued
otherwise earlier in the session and was wrong. The only thing to avoid
claiming is a simultaneous three-objective Pareto surface.

### 6.2 — methodology: additions

- **The annual/sliced builder split** and what it means for any result
  produced by an annual-builder runner. Currently undocumented and it
  now affects nine runners.
- **The freeze's meaning** (§2 above) — provenance, not a content
  filter.
- **The post-hoc sensitivity method** — why methane is applied
  post-hoc and O&M was wired. Currently only inferable from worked
  examples.

### 6.3 — register: new, undeclared

- **A56** — `13_`'s hardcoded `required_margin` in EaaS arms; root
  cause inside `optimize_experiments.py`.
- **A47's shape recurring** in `07_` and `08_gas3` (stale capital
  tiers) — third and fourth instances.
- **A45's shape** in `11_` (`required_margin` double-count).
- **`02_run_eaas` dead since 2026-08-29** (`f43a54c`); two later
  commits touched the call site and missed it. Its only output
  predates the break, is read by nothing, cited nowhere.
- **The annual-builder scope condition** as a register entry, since it
  bounds nine runners' results.

---

## 7. Writing assets already drafted

- `docs/nirp_defence_sourced.md` — the NIRP gas-ceiling defence, fully
  sourced. **Line 3 is the strongest argument**: OB3 + AKK at full
  delivery add ~55 TWh_th/yr against a 102-217 TWh_th gap to NIRP's
  2035 target, so both flagship pipelines close 25-54% of it at best.
  AKK is six years past a 24-month schedule; OB3 took ten years. 2025
  GenCo settlement rate 39.25%; ~₦200bn accumulating monthly.
  **Cite NNPC monthly reports and NERC quarterlies, not the press.**
- Abstract, two versions (emissions-first recommended; financing-first
  alternative).
- Capital-denominator additions to F-1, F-2, F-6.
- M-4 finding and the protocols §6 text, both applied.

---

## 8. Standing cautions

**Treat anything Chat asserts about the codebase as a hypothesis until
a grep confirms it.** During this session Chat was wrong about: the
shape-factor sourcing direction, a 748 USD/kW figure that never
existed, three magnitude predictions, four "established facts" lists,
`02_run_eaas`'s health (asserted three times), `grid_service_mode`
being removed, and `17_`'s mechanism. **Every one was caught by
checking code or a stored file.**

**Uploads sometimes arrive empty in long chats.** If a document tag
appears with no content, ask for a re-send or a direct paste.

**Register drafts written inside task reports are frequently never
applied**, because task whitelists exclude the register. L28, L29 and
L29-L33 all stalled this way. **Check `grep -c "^## L"` after every
task that drafts an entry.**
