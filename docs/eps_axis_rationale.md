# The eps axis: one rationale for the REL suite's sweep range

**Status:** closes the open item recorded in
`docs/methodology_genself_backstop.md` §7 ("Several runners currently
sweep different ranges; a single documented rationale is outstanding")
and register **D21**. Assembled from three existing sources — the
correction plan, `methodology_genself_backstop.md` §5/§7, and
`23_run_selfgen_frontier.py`'s own already-reasoned calibration — not
invented here. No eps grid is changed by this document; see
`docs/task_defect_closure.md` Part 5 for the audit this rests on.

---

## 1. What eps is

`eps` (the model's `max_offgrid_fraction` / `reliability_max_unserved_
fraction` argument, bounding `selfgen + unserved`) is the **grid
service obligation**: the maximum fraction of demand that may go
unserved by the grid — met instead by self-generation or genuinely
unmet — in a given year. `methodology_genself_backstop.md` §5.1 is
explicit that this is **not** a reliability/adequacy standard (LOLE,
EENS): a household running its own generator was not served by the
grid, but no generation capacity was short. It is a **policy
commitment** — an accessibility target, the model's only direct,
price-independent lever on how much demand the grid itself must reach.
Electrification targets have exactly this shape.

Consequently: eps bounds `selfgen + unserved`, never `unserved` alone.
Once a priced self-generation backstop exists, `unserved` alone is ~0
by construction (register-documented elsewhere in this project) and a
constraint written on it can never bind — every REL script that still
reports `unserved` in isolation must be read alongside `selfgen`, not
instead of it.

## 2. Why eps has no external measurement

Unlike a physical parameter (a capex figure, a capacity factor, a
fuel price), **eps has nothing to be measured against.** No published
Nigerian document states "the grid service obligation should be X%."
`methodology_genself_backstop.md` §7 states this plainly: *"The
fraction bounded by the obligation is a policy instrument, not a
physical parameter — there is no measurement of a correct service
standard."* This is a structural fact about the quantity, not a gap
this project has simply not yet filled — there is no analogous NERC or
NIRP table to source it from the way `gas_avail_observed` or the NERC
2024 demand split are sourced.

## 3. Why the axis must therefore bracket the model's own feasibility boundary

Because eps cannot be anchored externally, it must be justified
**methodologically**: the swept range is correct if and only if it
places the model's own feasibility threshold somewhere inside it —
loose enough at one end that the system is trivially feasible, tight
enough at the other that it is not. An axis that never crosses the
threshold (entirely feasible, or entirely infeasible, throughout) does
not locate anything; it is not wrong, exactly, but it is uninformative
by construction, no matter how many points it has.

This is precisely what went wrong with `23_run_selfgen_frontier.py`'s
predecessor iteration (documented in its own header): `reliability_
levels_log()`'s default range, `[0.20, 0.15, 0.10, 0.08, 0.06, 0.05]`,
is calibrated to a **grid-adequacy-style** shortfall band (5-20%
unserved) — a reasonable range for REL-1/REL-2's questions, where the
public-only baseline is close to fully served and the interesting
threshold sits in that band. It is "infeasible almost everywhere" for
a question whose own supply/demand arithmetic puts the shortfall
fraction at order 0.4-0.5 (Tier 1 demand reaching ~81 TWh by 2045
against a gas supply bounded near 40 TWh) — the default axis simply
never reaches the region where that frontier lives.

**The general rule this project should follow:** before adopting an
eps axis for a new REL-style sweep, locate the model's own
feasibility boundary for that specific research question first (a
short scoping run, or a structural argument like `23_`'s own
supply/demand arithmetic), then choose an axis that brackets it — not
the other way around.

## 4. What the lambda anchor does, and does not, justify

The plan's original framing (`merged_correction_plan_v2.md`) cites
"Nigeria's observed lambda 0.51-0.57" as "an indicative anchor" for
the eps axis. **This needs the correction `23_run_selfgen_frontier.
py`'s own docstring already made, not the plan's original wording
verbatim** — reused here, not silently dropped:

λ (`methodology_demand_horizon.md`, DERIVED from NIRP 2024 Tables 2
and 15) is **grid-served demand ÷ TOTAL (Tier 2, grid + self-
generation) demand** — a national off-grid-share statistic, base
0.51-0.57, i.e. roughly 43-49% of Tier 2 demand is currently self-
generated or unmet nationally. **eps is shortfall ÷ Tier 1 (grid-only)
demand** — a different, narrower denominator. `23_`'s own docstring:
*"That comparison was WRONG — lambda is grid / (grid + selfgen) on a
Tier 2 base, while epsilon is shortfall / Tier 1 demand. Different
denominators."*

So: **λ does NOT justify a specific numeric eps value**, and no REL
script should set an eps bound directly equal to (1 − λ) or similar —
that would silently convert a Tier-2-denominator national statistic
into a Tier-1-denominator model constraint, which is exactly the
mistake `23_` caught and retracted in its own history.

**What λ DOES justify:** it is evidence that a large, policy-relevant
off-grid share is a real, observed feature of the Nigerian system —
supporting the general judgement that eps axes in this project should
be wide (bracketing shortfall fractions considerably above the
5-20% adequacy-style band), not evidence for exactly how wide any
particular script's axis should be. Read it as directional
corroboration that "the true feasibility boundary for demand-facing
REL questions likely sits well above `reliability_levels_log()`'s
default range," not as a numeric target.

## 5. Which axis each REL script should use, and why they may legitimately differ

Confirmed by reading all six scripts' actual eps grids (not assumed):
**three distinct axes**, not four — TASK.md's framing ("four different
eps axes across six scripts") overstates the count by one; corrected
here, reported in `docs/task_defect_closure.md`.

| Axis | Values | Used by |
|---|---|---|
| `reliability_levels_log()` default | `[0.20, 0.15, 0.10, 0.08, 0.06, 0.05]` | `09_run_rel1_feasibility.py`, `10_run_rel2_marginal_cost.py`, `retired/09b_run_rel1_mode_sensitivity.py`, `retired/09b_run_rel1_retirement_sensitivity.py` |
| REL-3's own | `[0.45, 0.40, 0.35, 0.30, 0.25, 0.20]` | `11_run_rel3_financing_frontier.py` |
| `23_`'s calibrated grid | `[0.05, 0.10, ..., 0.60]` (12 points) | `23_run_selfgen_frontier.py` |

**These may legitimately differ, because they answer different
questions with different feasibility boundaries — the rule in §3
applied, not violated, by three different answers:**

- **REL-1 / REL-2 / both retired 09b_ scripts** use the tightest,
  adequacy-style default (5-20% unserved). These questions are
  structured around the SUPPLY-CONSTRAINED, non-EaaS, moderate-capital
  baseline, where the interesting feasibility transition (gas regime
  driving the threshold; the marginal cost of tightening reliability)
  plausibly sits inside that tighter band. This project has not
  re-verified that boundary placement as part of this task (no solve
  performed here — see Part 5 close), so this is a documented
  judgement to revisit, not a re-confirmed fact.
- **REL-3** widens to 20-45% because it is explicitly comparing
  public-only against EaaS financing, including scenarios where public
  capital alone is expected to struggle — a wider net is needed to
  observe the threshold SHIFT between two arms, not just whether either
  arm alone is feasible at one tight band.
- **`23_` (self-generation frontier)** widens furthest, to 5-60%,
  because its research question is explicitly about locating where the
  shortfall fraction saturates under a fixed, supply-bounded gas
  regime with NO EaaS or extra financing arm to relieve it — the
  widest structural gap of the three, requiring the widest bracket.

**This document does not change any of the three axes.** Changing a
swept axis changes results, and per TASK.md Part 5, that requires
agreement first. What this document settles is that the three axes
are each a considered response to a different feasibility boundary,
not an unexamined inconsistency — and that any FUTURE REL script
should pick its axis the same way: locate the boundary for its own
question, then bracket it, informed but not dictated by λ.
