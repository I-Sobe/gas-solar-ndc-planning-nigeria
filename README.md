# Gas deliverability and the financing of power-sector decarbonisation in Nigeria

A capacity-expansion model of Nigeria's grid-connected power system to
2054, with the division of investment capital between concessional and
commercial sources determined within the model rather than supplied to
it.

This repository holds the model, the data, the run scripts and the
submitted thesis. MSc, petroleum and energy resource engineering.

**Tag `thesis-submitted`** marks the state the thesis was submitted in.
467 tracked files, 113 MiB; a full clone fetches about 130 MB, most of
it third-party PDFs and workbooks under `data/` rather than code.

---

## What the model is

A deterministic linear programme. Each solve minimises discounted
system cost over 2025–2054 under perfect foresight, with fixed
coefficients. Results are reported over **2025–2045**; the final nine
years are an end-effect buffer and are not reported.

The analysis built on those solves is neither single-objective nor
deterministic. Bi-objective trade-offs are recovered by the
**ε-constraint** method over two objective pairs. Uncertainty is
treated at three levels — scenario, parameter and realisation — and the
value of the recourse formulation the study does *not* solve is
bounded above from two evaluations it does perform.

**There is no weighted-sum objective and no stochastic programming with
recourse.** Neither is claimed anywhere in the thesis.

### Determined within the model

Solar additions, public and service-financed, by vintage; new gas
capacity and open-cycle to combined-cycle conversion; battery storage;
dispatch across six intra-annual slices; self-generation as a priced
backstop and unserved energy; and **the split of each vintage's capital
between a concessional and a commercial tranche**, bounded by a finite
concessional envelope and disciplined by a coverage-ratio condition on
privately financed capacity.

### Supplied as scenarios

Gas deliverability, gas fleet availability and retirement, demand
growth, solar and storage capital cost trajectories, hydro energy, the
emissions cap trajectory, the carbon price, and the financing terms.
Each is exposed as a named arm; each arm value is classified SOURCED,
DERIVED or ASSUMED; and where no defensible value exists the arm raises
rather than substituting a placeholder.

### Not represented

Single node, so no transmission constraint. No unit commitment,
minimum stable generation, ramping or start-up cost; capacity is
continuous. Demand is exogenous and price-inelastic. Policy is
exogenous with no feedback. No reservoir or field-level gas production
modelling — gas enters as a delivered-energy constraint on the power
sector, not as subsurface mechanics. Chapter 3 §3.7 states each
limitation with its direction of bias.

---

## Research questions

1. What limits the least-cost expansion pathway to 2045, and what is
   the system-cost value of relieving each limit?
2. When the division of each vintage's capital between concessional and
   commercial sources is determined within the model, what does that
   change, and under what conditions does it change nothing?
3. When an emissions constraint is applied to the grid-supplied
   quantity rather than to the national inventory, how does tightening
   it affect national emissions, and does the financing channel alter
   the relationship?
4. Does a plan optimised against a single deterministic future retain
   capacity adequacy across demand realisations, and does the
   specification of the service standard change what is observed?

Chapter 2 reports the systematic review, conducted to a registered
protocol over seventy-five studies, that establishes the gaps these
answer. Results are in Chapter 4; their interpretation in Chapter 5.

---

## The model — `src/`

```
optimize_model_sliced.py   the sliced builder; every reported result comes from it
optimize_model.py          the annual builder, plus the shared solve_model used
                           by every runner; retained and live
optimize_experiments.py    experiment orchestration
scenarios.py               arm definitions and scenario construction
cli_arms.py                shared CLI arm-flag helper

demand.py                  electricity demand projection
solar.py                   PV generation and capacity factors
storage.py                 battery storage, planning-level energy abstraction
dispatch.py                deterministic energy-balance dispatch
gas_supply.py              deliverability series construction
gas_benchmarks.py          single source for the gas-price benchmarks

economics.py               economic evaluation utilities
blended_finance.py         blended concessional finance structure
supplied_division.py       the supplied-division counterfactual (Question 2)

stochastic.py              sampling for the realisation treatment
pareto_plot.py             Pareto dominance and the two-panel frontier figure
summary_years.py           year-labelled summary quantities
output_naming.py           deterministic arm-keyed output paths
io.py  utils.py            loading, serialisation, helpers

EaaS.py  gas_balance.py  postprocess.py    dead; no import anywhere
```

## The runs — `scripts/`

143 files. They fall into six groups, and the distinction matters for
anyone running them.

| group | count | what they are |
|---|---|---|
| drivers | 20 | shell scripts that sequence runners; `run_phase_b.sh` is the main one |
| baseline and cap construction | 7 | must run before the cap-dependent runners |
| runners | 33 | produce reported results |
| verification and diagnostics | 31 | read-only or regression; `52_invariants.py` is the suite |
| figure scripts | 18 | reproduce the Chapter 4 figures |
| one-off records | 32 | ran once; **not part of any pipeline** |

**The one-off records are not re-runnable.** `task85_apply.py` and
`task86_apply.py` edit chapter text and `task84_apply.py` appends to
the findings file; running them again would rewrite the thesis. They
are retained because they record how a correction was applied, not
because they are tools.

Two runners are suspended and their headers say so:
`12_run_dem1_demand_sensitivity.py` and `14_run_str1_storage_role.py`.
Two more are in `scripts/retired/`, moved rather than deleted. Eight
further scripts have no live caller and are retained for attribution.

A duplicate pair exists for figures 4.7 and 4.10 — a v1 and a v2 — and
which produced the committed figure is not determinable from the files.

## The thesis — `docs/chapters/`

```
ABSTRACT.md
CHAPTER_1.md   Introduction          CHAPTER_4.md   Results
CHAPTER_2.md   Literature review     CHAPTER_5.md   Discussion
CHAPTER_3_assembled.md  Methodology  CHAPTER_6.md   Conclusions

appendix_a.md  parameter provenance — every value classified and sourced
appendix_b.md  limitations register, live entries
appendix_c.md  limitations register, entries resolved during the study
appendix_d.md  reproducibility record

docx/                 the submitted Word copies, generated from the markdown
media_CHAPTER_4/      figures embedded in the Chapter 4 document
```

## The inputs — `data/`

74 files across the domains below. Provenance for every value is in
Appendix A, not here.

| directory | domain |
|---|---|
| `capacity/raw/` | gas plant availability and commissioning vintages |
| `cost/processed/` | gas, genset, O&M, solar, storage, VoLL, emissions cap series |
| `cost/raw/` | fuel prices, genset build-up, and the third-party sources below |
| `cost/raw/{carbon,gas_price,solar_capex,storage_capex,unserved_energy_penalty}/` | IEA, IRENA, NREL, NERC and regulatory publications |
| `demand/` | grid generation, access, population, per-capita diagnostics |
| `gas/assumptions/` | availability, heating value, efficiency notes |
| `gas/processed/` | gas-to-power, thermal generation, netback, deliverability cases |
| `gas/raw/NERC_report/` | NERC 2025 quarterly reports, Q1–Q4 |
| `review/` | the frozen systematic-review evidence base |
| `stochastic/` | carbon price reference |

## The outputs — `results/`

Not tracked. Regenerated from the scripts; archives of the submitted
state are held outside the repository.

Every result directory is **keyed by its arm set**:
`results/<runner>/arms_<hash8>/`, where `hash8` is the first eight hex
characters of the SHA-1 of the full arm tag. Inside each sits an
`ARMS.json` recording the arm tag and the exact invocation, alongside
the runner's output. An `INDEX.md` in each runner folder maps hash to
arm tag. The same configuration always writes the same directory.

`_pre_*` directories are timestamped snapshots taken before a re-run.
`DO_NOT_CITE.txt` marks a smoke-test tree that is not citable.

---

## Reproducing the results

```bash
conda env create -f environment_linux.yml
conda activate nigeria-energy

python scripts/build_gas_deliverability.py   # deliverability series
bash  scripts/run_emissions_baseline.sh      # baseline, then the cap trajectory
bash  scripts/run_phase_b.sh                 # the reported result set
```

Roughly 4,600 solves, four to five hours of wall clock.

Four things matter and are easy to get wrong.

**The generation is set by the code default, not by the driver.** The
reported results are the *vintage* dispatch generation. At the tagged
code `run_phase_b.sh` selects it by default and writes to the same
`arms_<hash8>` directories, so the command above reproduces the
reported generation. The driver that originally produced it was
`run_vintage_rerun.sh`, which is the same sequence differing in its log
directory and in two items; `run_phase_b.sh` produced the earlier
blended generation that Appendix D.2 records.

**Some results were produced outside the drivers.** The baseline, NDC
and F3 arms came from batch drivers, and at least one directory
(`fin3/arms_7f91f752`) has no driver log. The vintage logs were also
written from a working tree whose commits are not in the published
history, so the exact generating commit cannot be shown from this
repository. Appendix D records the reproducibility position in full.

**`environment_linux.yml` is canonical.** `environment.yml` omits the
solvers and a package the code imports, and will not reproduce the
results.

**The solver is CBC 2.10.13**, with Pyomo 6.10.1. HiGHS is available
through the same interface but did not produce the reported results.
One reported result rests on a degenerate optimal face where solver
choice determines which vertex is returned, so the version is part of
the record rather than an implementation detail.

---

## Verification

The formulation is verified by an executable suite of **88 assertions**
in `scripts/52_invariants.py`. Most derive from properties the
formulation must satisfy mathematically — accounting identities,
coefficient constructions, and the agreement of each reported aggregate
with an independent recomputation — rather than from any claim made
about the model. A smaller number test consistency between the model as
built and statements made about it in the thesis.

Of the 88, **81 pass and 7 fail**. Each failure is recorded in the
limitations register reproduced as Appendices B and C.

The suite found defects in the model's own claimed contribution,
including an error in the annuitisation of capital and a cost series
silently removed by an unrelated refactor. Chapter 3 §3.6.10 sets out
the practices, what each caught, and three classes of defect the method
cannot detect.

---

## Data and third-party sources

Every parameter's provenance is in Appendix A, classified and sourced,
with unsourced values declared rather than filled.

Third-party publications are included under `data/` for
reproducibility. They remain the property of their publishers — IRENA,
the IEA, NREL, NERC and others — and are subject to those publishers'
terms, not to this repository's licence.

---

## Status

The thesis is submitted. Open items at submission — citation locators,
sub-claims without a post-correction measurement, and one block of runs
not executed — are recorded in Appendix B.

---

## Author

Nwangene Sobe-Olisa Andrew
MSc Petroleum and Energy Resource Engineering
