# Methodology — protocols

**Current as of HEAD `8fc5ccb`.** Last substantive model changes:
`5beaef0` (storage-CAPEX realised-spend diagnostic) and `85f2598` (B\*
replacement, concessional rate sourcing, consolidated gas benchmarks,
IRENA solar arm). The operating-and-maintenance implementation
(`d7538ff`) was the last change before those two.

This is a living document describing a model under active correction.
Results produced under it are provisional. The sections below cover
**how this project audits itself** — discipline that cuts across every
model layer rather than belonging to one of them.

**Sections covered here:** required-key discipline (why load-bearing
parameters are declared with no default, and fail loudly if one is
omitted); the do-not-report protocol (screening invalidated results
before they reach a chapter); epoch stamping and the stratigraphy (why
every figure belongs to a dated model state); sourcing classification and
the negative search (SOURCED/DERIVED/ASSUMED, and why an absence of a
source is itself a finding); binding versus materiality, and the scope of
a correction as a modelling decision in its own right.

**Sections NOT covered here, written elsewhere:** model architecture and
the financing formulation (`docs/methodology.md`); the physical layer —
demand shape, firm capacity, storage, the gas fleet
(`docs/methodology_physical_layer.md`); demand growth, end-of-horizon
accounting, and emissions cap construction
(`docs/methodology_demand_horizon.md`); self-generation, the
emissions-cap boundary, and the grid service obligation
(`docs/methodology_genself_backstop.md`).

---

## 1. Required-key discipline

**SETTLED**

A required-key discipline means a parameter that materially affects a
model's output is declared with no default value at the point it enters
the model, so that every code path that constructs a scenario must state
the value explicitly or fail loudly with an error naming the parameter.
This model enforces the discipline through a sentinel-object pattern:
each such parameter's keyword default in `load_scenario()` is a unique
module-level marker object (e.g. `_TARIFF_REQUIRED = object()`), not a
plausible-looking literal. If a caller does not supply the parameter,
`load_scenario()` raises a `TypeError` that names the parameter, states
what value it needs, and explains why an unstated value is unsafe; if the
caller supplies a value outside the recognised set, it raises a
`ValueError`. **Seventeen** required parameters in `load_scenario()`
(plus one more, `genset_case`, in `load_econ` — 18 total) are governed
this way — `tariff_case`, `peak_multiple_case`, `reserve_margin`,
`dscr_case`, `hydro_case`, and others — each following the identical
raise-with-explanation pattern rather than a bespoke one. No evidence in
the repository that a count of twenty was ever reached; that figure did
not describe any inventoried state.

The discipline exists to prevent a specific, previously-realised defect
class: a load-bearing parameter reached through `dict.get(key, DEFAULT)`
inside a builder, where `DEFAULT` was chosen once, undocumented, never
exposed to any runner's command line, and therefore invisible in every
output filename or arm tag. The canonical instance is the retired
`solar_tariff_case`, which silently defaulted to 65,000,000 USD/TWh —
every run before that defect was found used this value with no error,
warning, or visible signature that it had been used at all, and no
result could be distinguished from one run under a different tariff
assumption. Register **L19** documents a live recurrence of the same
class: `26_run_f3_identification.py` resolves two required keys
(`dist5_case`, `solar_cf_case`) internally rather than exposing them as
flags, so those results cannot be swept or attributed from the command
line even though the values used are correct. A required-key promotion
closes exactly this gap: it does not change what value is used, only
whether the code can silently proceed without one being stated.

When such a promotion is carried out, the value carried through the
promotion is deliberately frozen — the same number that every call site
already resolved to under the old default or the old local convention is
threaded through explicitly, and nothing else changes. This is what
makes the promotion its own regression test: because no computed quantity
can change (every call site receives the identical value it already
had), any discrepancy between the pre- and post-promotion output is
proof that the promotion was not, in fact, value-preserving, and can be
caught by exact reproduction of a frozen reference run rather than by
new test-writing. The promotion is a pure access-and-visibility change,
not a modelling change, and its correctness criterion — byte-identical
output — is verifiable without rerunning the model's judgement about
what the value *should* be.

Making a parameter's value explicit at every call site is a distinct
decision from making it *selectable* — i.e. exposed as a CLI flag, swept
across an arm, or written into the output filename. `storage_duration_hours`
is a candidate for the first but not, at present, the second: an
inventory found far more call sites threading into `load_scenario()` than
an earlier estimate assumed (**24 files / 35 invocations** in
`scripts/*.py` + `src/*.py` — 27 files / 38 invocations including
`scripts/retired` and `scripts/deferred` — against a stated 22), which
is itself a reason to promote carefully rather than in one
motion — a required-key promotion done at the wrong scale risks silently
missing a caller. Separately, even where a promotion is mechanically
safe, exposing a new dimension as selectable renames every stored
result's arm tag, which breaks the provenance chain linking
already-verified results to the inputs that produced them (the same
reasoning recorded in register **L9** for `reserve_margin` and
`genset_case`, which are required keys satisfied by a hardcoded constant
rather than a swept arm, by deliberate choice). Explicitness and
selectability therefore answer two different questions — "is this value
visible and accounted for" versus "can this value be varied without a
code change" — and a correction task should not conflate them.

---

## 2. The do-not-report protocol

**SETTLED**

This project maintains a do-not-report list alongside its results: a
table, kept in the same document as the correction plan, naming every
figure, runner, or claim that is known to be invalid and must not enter a
chapter, slide, or briefing until its defect is fixed. The list exists
because a model under active correction produces two kinds of output that
look identical from the outside — numbers generated by the system
working, and numbers generated by the system failing — and only the
person who fixed the defect can currently tell them apart. Without a
written record, that knowledge lives in one person's memory and decays
with time; a chapter drafted eight weeks after a defect was found and
forgotten will cite the broken figure with the same confidence as a
correct one, because nothing on the page distinguishes them.

Four broad classes of defect put a figure on this list. The first is a
VoLL-dominated objective: when a scenario's demand cannot physically be
met by the technologies the model is allowed to build, the shortfall is
priced at the value of lost load, a number chosen to represent the cost
of an outage, not the cost of a plan — a cost dominated by that penalty
says something about infeasibility, not about the policy being
evaluated. The second is a defective metric denominator or numerator: a
percentage or a levelized cost computed with a term double-counted,
omitted, or wrongly signed produces a number that is precise and wrong in
the same breath, which is more dangerous than a number that is obviously
absent. The third is a superseded epoch: this model has been
recalibrated and re-anchored several times as errors were found and data
was sourced, and each recalibration moved every downstream objective —
sometimes by single digits, once by an order of magnitude. A figure
computed correctly under an earlier configuration is still wrong once
quoted against a later one, without a single line of code changing. The
fourth is a dead runner: a script that no longer executes, whether
because an interface it depends on changed underneath it or because a
bug corrupted its output silently, can leave behind results that look
complete while having no defensible claim to correctness going forward.

Entries are lifted when their condition is satisfied, but they are moved
to a record, never deleted. This matters for two reasons. First, an
entry that disappears leaves no trace that the underlying number was ever
suspect, which invites a reader — or a future version of the author — to
re-derive the same false confidence from scratch. Second, the record of
what was once unreportable, and exactly what resolved it, is itself part
of the audit trail a thesis under active correction needs: it
demonstrates that "resolved" was a judgment made and dated against a
specific fix, not an assumption that time alone made a number safe. A
screening list that only ever grows, and is checked before every claim,
is slower to work with than trusting memory — but the alternative is a
chapter that cannot distinguish a result from an artefact, which is the
more expensive failure by far.

---

## 3. Epoch stamping and the stratigraphy

**SETTLED**

Every reported figure in this project belongs to a specific model state,
not to the model in the abstract. This project's history is a sequence
of recalibrations — moments where a defect was found and fixed, or a
parameter moved from an assumed value to a sourced one — and each
recalibration moved every downstream objective, sometimes by single
digits, once by an order of magnitude. A figure that was correct under
one state can be wrong when quoted against a later one, with no code
change between the two: the state moved, not the number's arithmetic.
Treating a model's output as timeless is therefore not a convenience but
an error waiting to surface the first time two documents from different
states are compared.

The practice this project follows is threefold. First, every methods
document states the commit (`HEAD`) it is current as of, near its head,
so a reader can check whether a cited figure postdates or predates a
given correction. Second, a superseded document is retained and renamed
rather than overwritten or deleted — `handover_pre_om.md` and
`findings_provisional_pre_om.md` are earlier states of documents now
superseded by later ones, kept precisely so a reader tracing a figure's
history has somewhere to look. Third, a register entry that reports a
measured magnitude carries the arm set and the epoch it was measured
under, at the moment it is written, not reconstructed afterward.

The cost of skipping this is demonstrated twice in this project's own
record. Register L8's magnitude table was written without an epoch stamp
or a recorded run tag, and became undatable by anything but band-matching
its objective values against a separate document's numbers — an
inference, not a record. (**L8 now carries an epoch and configuration
stamp** — the historical omission described here is not itself
checkable any further back, but the current entry is stamped, and this
paragraph is a record of the failure mode, not a live defect in L8
today.) And `handover_pre_om.md` states two anchor
values (20.9899 bn and 20.856 bn) that are not reproducible on any commit
in this repository's history; the cause could not be established after
the fact, and the entry records this as unexplained rather than inventing
a rationalisation to fit. Both cases show the same failure: without a
stamp fixed at the moment of writing, a number's epoch becomes a research
question in its own right, one that is not always answerable.

The rule that follows is stated plainly so it is enforceable: every
magnitude table entering the register carries its arm set and its epoch
at the moment it is written. A number without a dated state attached to
it is not yet a finding — it is a number waiting to be misquoted.

**"The freeze" is this same discipline applied to the model as a
whole, and it is a provenance convention, not a content filter.** It
means every reported number comes from a known, stated model state
("the frozen state" names a dated model state, not a permanent one) —
not that only a pre-enumerated sweep of arms may be cited. Any result
run at a stated, reproducible state is reportable, regardless of when
it was produced.

`2a9226b` was the model state most recently called "the frozen state."
It is now **behind HEAD and superseded for capital-envelope and
EaaS-bankability results**: `5beaef0` (storage-CAPEX realised-spend
diagnostic) and `85f2598` (B\* replacement, concessional rate,
consolidated gas benchmarks, IRENA solar arm) both changed `src/` after
it. Citing `2a9226b` as the current frozen state for those two result
families is therefore stale.

The two anchors most often quoted reproduce at HEAD regardless: the
regression anchor (`conv_none` + `pre_f1_overnight` + `dscr_regression`
+ `tariff_legacy_65m` + `om_none`) reproduces 21,811,042,472.821114 USD
/ 45,502.9754 MWh of 2054 storage capacity (`results/anchor_regression_postbstar.log`, run
2026-09-20 after both `5beaef0` and `85f2598`), and the central anchor
(`conv_central` + `f1_tranche` + `dscr_central` + `cr_central` +
`env_dares` + `baseline` deliverability + `hydro_flat` +
`tariff_central` + `om_central`) reproduces 27.954 bn USD / 65,613 MWh of 2054 storage capacity
(`results/anchor_central_postbstar.log`, same run). Both logs post-date
the two `src/`-changing commits above.
**[Historical, TASK 43, 2026-10-10: these reproductions are the 2026-09-20 run, before the rate-coherence re-baseline (L58). Current anchors, unchanged across Phase B: regression 21,328,493,384.73 USD / 43,539.97566 MWh and central 27,317,670,155.72 USD / 63,056.31077 MWh (2054 storage capacity), `results/baseline_sliced/summary__19c81f7c.json` and `summary__02c17f55.json`.]**

---

## 4. Sourcing classification and the negative search

**SETTLED**

Every parameter in this model carries one of three classifications,
recorded at its definition site rather than inferred from how confident
it looks. SOURCED means the value is stated by a named, retrievable
document — a citation a reader could go and check. DERIVED means the
value is computed from sourced quantities by an arithmetic method that is
shown, not merely asserted, so a reader can redo the computation from the
same inputs and reach the same number. ASSUMED means the value is an
analyst estimate: defensible, but not read off any external document, and
required to carry a stated direction of bias so its effect on the
model's conclusions can be reasoned about even though its exact value
cannot be checked.

The classification is a property of the parameter's provenance, not of
its plausibility. A value that looks precise, or that carries a name
suggesting it was measured, is not thereby SOURCED — absence of a
citation at the definition site is itself the finding, and the correct
response is to record the parameter as ASSUMED rather than to infer a
source it does not have. This project's own history contains the failure
mode this guards against: a function once named as though it discounted a
cash flow, and did not; a variable named as though it were sourced, and
was an analyst's estimate. A name may not assert a provenance the value
does not carry.

Where a search for a source is conducted and returns nothing, that
documented absence is itself citable, and is a stronger position than an
invented figure would be. The worked example is register **L20**, on the
minimum-flow fraction governing hydro dispatch: a documented search
established that no Nigerian minimum-flow obligation exists expressed as
the fraction this model's parameter requires. What exists instead is a
table of absolute discharge rates in a Nigerian water-resources master
plan, and that source itself labels those rates ASSUMED operating rules,
not a mandated obligation. The two quantities are not convertible into
one another for two independent reasons — one is an absolute discharge at
a single plant, the other a fleet-wide energy-share fraction; and one is
dependent on reservoir storage state, which this model does not carry as
a variable at all. Recording that non-convertibility, and why, is worth
more than picking a number that happens to satisfy the model's units.

Where even a defensible range cannot be sourced, a different move is
available: report the break-even value the finding would need to reverse
at, rather than a point estimate. The worked example is the supply-chain
methane and flaring sensitivity in the frozen-state findings (F-21): the
assumed uplift range is stated as ASSUMED and not narrowed further, but
the finding under test is shown to reverse only once the uplift reaches
49% — a threshold that is falsifiable by anyone who later sources the
true range, without this project having had to pin that range down
first.

---

## 5. Binding is not material; the scope of a correction can flip its sign

**SETTLED**

A constraint can bind exactly — to several decimal places, in multiple
time slices, across the full horizon — and still carry negligible shadow
value. The worked example is `hydro_min_flow_fraction` (register **L20**;
`findings_frozen.md` M-3): the floor is shown to bind exactly in specific
slices in specific years, verified independently of any solve, and yet
sweeping it across a wide band moves the reported objective by 0.141%
and terminal storage by 0.000%. Binding and material are therefore
distinct properties of a constraint, and neither implies the other; a
materiality threshold — how large a movement counts as meaningful — is
fixed before the sensitivity result is read, not chosen afterward to fit
whatever number appears.

The scope over which a correction is applied is itself a modelling
decision, and leaving it unstated can silently determine which direction
a finding points. The worked example is the same supply-chain sensitivity
referenced in §4: applied to grid gas alone, the emissions uplift
narrows the leakage gap the model reports; applied to grid gas and
self-generated diesel at comparable rates, it widens the same gap. Both
scopes are individually defensible — one asks what happens to the
grid-side commitment, the other to national emissions as a whole — but
they are different questions, and answering one is not answering the
other.

The discipline this implies is the same one materiality thresholds
require: the scope of a correction is stated before the result under it
is read. A scope chosen after seeing which answer it produces is not a
specification, it is the conclusion wearing a specification's clothes.

## 6 — Parameter uncertainty, and why it is not scenario uncertainty

This model is **deterministic**. It solves a single linear program
under perfect foresight and produces one optimal plan per
configuration. Nothing in it is stochastic, and the thesis does not
claim otherwise. Uncertainty is handled in two distinct ways, and
keeping them distinct is a methodological commitment rather than a
convenience.

**Scenario uncertainty — which world happens — is handled by the arm
sweep.** **`run_central_sweep.sh` runs 39 invocations across three
sections** (A 5, including the regression anchor; B 26; C 8), not 38 —
verified directly by counting `run` lines in the script, unchanged since
`2a9226b` (the run lines themselves have not drifted). **Section B is
the one-at-a-time set**, varying exactly one of **nine dimensions** from
the central configuration: capital recovery (`cr`), conversion,
gas deliverability (including the GMP arms), hydro, the concessional
envelope, DSCR, gas availability, gas retirement, and peak shape.
**Section C's 8 invocations are deliberate interaction cells, not
one-at-a-time — but only four of the eight are two-dimensional.** The
`cap x deliverability` (×2) and `cap x conversion` cells cross one
explicit arm override against the cap/EaaS regime implied by the
runner choice (`02_`/`03_` vs `01b_`) — genuinely two dimensions. The
four F3 cells do not: `tight`/`unconstrained` vary `capital-case` only,
and `tariff_low`/`tariff_high` vary `tariff` only (holding
`capital-case=moderate`, unchanged from Section A's own central point
for that runner) — each is one dimension against `A_central_f3_moderate`,
not two (checked directly against `run_central_sweep.sh`, 2026-09-26;
see `docs/task_log_ch36_inputs.md` item A). "Each varying one
dimension" still does not hold for Section C as a whole, only for
Section B — that conclusion is unaffected — but "two-dimension" is not
a uniform description of all eight cells. "Financing regime" and
"cap stringency" are not swept as one-at-a-time B dimensions: the
financing-regime contrast lives in Section A (`02_`/`03_` vs `01b_`),
and cap stringency only appears crossed with another dimension in the
four Section-C cells where that actually holds. These are not samples
from a distribution. They are named, internally coherent states of the
world, and the constraint-relief
cross-tab rules out combinations that would be incoherent — a world
where demand booms while the gas that would serve it collapses, for
instance.

**That coherence is exactly what sampling would destroy.** An earlier,
deferred implementation of this analysis drew demand growth and gas
regime **independently despite their sharing a driver**, generating
worlds that cannot occur and inflating both tails of the resulting
distribution. The arm sweep is the stronger treatment of scenario
uncertainty precisely because a human chose which combinations are
possible.

**Parameter uncertainty — the true value of a fixed quantity being
unknown — is handled by Sobol sampling at the central scenario.**
Five continuous parameters: gas fleet availability, fleet thermal
efficiency, solar capacity factor, O&M level, and conversion capex.
Each has a range with stated provenance; all priors are uniform,
because none has a sourced distributional shape and a uniform prior
over sourced bounds is a statement of ignorance rather than a claim
about likelihood.

**The same failure exists within a single parameter, and was found
here.** Gas availability is stored at two scenario keys — one read by
the annual energy ceiling, one by the firm-capacity constraint — which
the scenario loader only ever initialises equal. A sampler mutating
one would silently decouple energy accounting from capacity
accounting: the shared-driver failure reproduced inside one variable.
The implementation sets both and asserts their equality before every
build, because the failure mode is silent by nature and the guard must
not be.

### What a variance decomposition gives that a sensitivity does not

A one-at-a-time sensitivity establishes which input moves the output
most. A Sobol decomposition establishes **how much of the output's
total variance each input explains**, which is a different and more
actionable quantity: it ranks inputs against each other on a common
scale that sums to one.

It also separates two effects. The **first-order index** is the
variance explained by a parameter alone. The **total-order index**
adds everything it explains through interactions with other
parameters. **A large gap between them means the parameter matters
mainly in combination**; equality means it acts independently. Both
are computed here by Saltelli's estimator via `scipy.stats.sobol_indices`,
with bootstrap confidence intervals, at a cost of `n(d+2)` model
solves.

### Reporting discipline

Three rules govern how these results may be used.

**The prior's width is part of the result.** Where a parameter
dominates a decomposition because its prior is wide, the finding is
about the cost of not knowing it — not a discovery that the model is
sensitive to it. This must be stated wherever the indices are quoted.

**The central configuration's position in the distribution must be
stated.** Where a central arm sits at the edge of a sampled range
rather than its middle — as gas availability's does — the sampled mean
is not a better estimate of the central case. It is a different
quantity.

**And feasibility counts are only meaningful where infeasibility is
possible.** This model prices self-generation as an unbounded variable,
but **`unserved` is not priced at all under any priced genset case — it
is fixed at zero by construction** (`optimize_model_sliced.py:591-592,
637-640`; `optimize_model.py:455-456`;
`docs/methodology_genself_backstop.md` §1.4, which states the rule
plainly: `cum_unserved = 0` must never be quoted as evidence that
demand is met). The model never fails to serve load not because it
prices the alternative, but because self-generation absorbs the
shortfall with `unserved` structurally excluded from the objective. A
zero failure count across 1,792 draws is a property of the formulation,
not evidence of a robust system — and specifically not evidence that
`unserved` was priced and simply never bound.