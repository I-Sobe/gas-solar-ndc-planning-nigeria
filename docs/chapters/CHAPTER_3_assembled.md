# Chapter 3 — Methodology

## 3.1 Model formulation

### 3.1.1 Model class and solution method

The model is a deterministic linear-programming capacity-expansion model of the Nigerian power system, choosing investment, dispatch and financing jointly over a multi-decade horizon to minimise discounted system cost. It belongs to the class of generation-expansion planning models reviewed by Oree et al. (2017) and to the bottom-up optimisation tools surveyed by Ringkjøb et al. (2018): a single-node, perfect-foresight formulation in which investment, dispatch and financing are co-optimised rather than solved in sequence.

The formulation is implemented in Python with Pyomo (Hart et al., 2011; Bynum et al., 2021). Every result reported in this thesis was produced with the COIN-OR branch-and-cut solver, version 2.10.13 (Forrest and Lougee-Heimer, 2005); the HiGHS solver (Huangfu and Hall, 2018) is available through the same interface but did not produce the reported results. The distinction is recorded because one reported result rests on a degenerate optimal face, where the choice of solver determines which vertex is returned (§3.6.10). Because the formulation is linear, every solution is a global optimum and every constraint carries a dual value, which Chapter 4 uses directly as a shadow price.

The model runs over 2025 to 2054 and results are reported to 2045. The reporting horizon is that of both NDC 3.0 and NIRP 2024 (Federal Ministry of Power, 2024), and the nine years beyond it serve as an end-effect buffer. Terminal-condition effects are a recognised difficulty in long-horizon investment models, and the usual remedies are a salvage term, a terminal constraint, or a set of years excluded from reporting (Grubb et al., 1995). This model uses the third.

The buffer’s adequacy is measured rather than assumed. Solving at terminal years 2045, 2050, 2051, 2052, 2053 and 2054, and at storage technical lives of 15 and 30 years, cumulative capacity at 2045 is identical in all eight cells to within 2 × 10⁻⁴ MW of solar, 0.0000 MW of gas and 1 × 10⁻⁵ MWh of storage, measured against the 2045-horizon solve at the same storage life. The annual addition series are not identical: every cell extending beyond 2045 first differs in 2035, where 47.7 MW of solar and 25.5 MW of gas are displaced in time. Summed over the reporting window, those displacements amount to 1.48 to 1.52 per cent of the solar addition series, 1.86 to 1.88 per cent of the gas series and 0.33 to 0.34 per cent of the storage series. The quantities reported as results in this thesis are therefore invariant to the position of the horizon end; the timing of individual additions within the window is not, and annual addition series are accordingly reported as illustration carrying that bound rather than as results. Storage technical life changes the objective and not the plan. One in-horizon replacement enters as the horizon lengthens, the 293.0 MWh 2039 storage vintage rebuilt at terminal year 2054, and it does not move the reported stocks. The invariance of the reported stocks is measured of the model as reported and is not structural.

- Metering point and currency: Demand is metered as gross generation at the busbar, so transmission and distribution losses fall outside the model boundary rather than being carried as an undeclared wedge. All monetary quantities are expressed in real United States dollars of 2025.

### 3.1.2 Discounting

Every cost stream in the objective — fuel, operation and maintenance, self-generation, and the annual capital payments defined in §3.4 — is discounted to the base year 2025 at a social discount rate $r_{s} = 8\%$ real:

$\delta_{t} = (1 + r_{s})^{- (t - t_{0})}.$ (3.1)

Public economics distinguishes a social rate of time preference, which values consumption across time, from a social opportunity cost of capital, which reflects the return capital would earn in its next-best use; multilateral development banks typically apply the latter at 10 to 12% real in developing countries (Zhuang et al., 2007). The distinction determines what $r_{s}$ may represent here. Because the model charges the opportunity cost of capital explicitly, through the financing rates and capital recovery factors set out in §3.4.1, discounting at an opportunity-cost rate would charge that cost a second time. The social discount rate is therefore interpreted as a rate of social time preference, and the two forces distinguished in Fisher’s theory of interest — impatience and investment opportunity (Fisher, 1930) — are carried by $r_{s}$ and by the financing rates respectively.

- Sensitivity of the plan to the social discount rate**:** The physical plan is not conditional on the discount rate. At social discount rates of ⟦4⟧, ⟦6⟧, ⟦8⟧, ⟦10⟧ and ⟦12⟧ per cent, cumulative capacity at 2045 agrees to within ⟦2 × 10⁻⁴⟧ MW of solar, ⟦0.0000⟧ MW of gas and ⟦2 × 10⁻⁴⟧ MWh of storage; cumulative solar additions to 2054 are ⟦30,938.8⟧ MW and storage at 2054 is ⟦56,289.9⟧ MWh at every rate, with spreads of ⟦0.0001⟧ MW and ⟦0.0002⟧ MWh. The objective moves as expected, from ⟦40.536⟧ billion dollars at ⟦4⟧ per cent to ⟦15.162⟧ billion at ⟦12⟧ per cent, a ratio of ⟦2.67⟧. What responds to the rate is the timing of additions within the window rather than their total. The largest displacement is at the lowest rate, where ⟦119.7⟧ MW of solar moves from 2039 to 2037 and ⟦73.2⟧ MW of gas moves between years, amounting to ⟦6.1⟧ per cent of the gas addition series against the central rate. A low rate therefore brings solar construction forward rather than deferring it, which is the expected response to a declining capital cost series (§3.3.4) when future costs are discounted lightly; gas additions move in both directions and storage additions move later. The solutions at ⟦10⟧ and ⟦12⟧ per cent are identical to each other and differ from the central rate in timing alone.

  ----------------------------------------------------------------------------------------------------------------------------------------------------------
  $$\mathbf{r}_{\mathbf{s}}$$   **objective (bn USD)**   **solar to 2045 (MW)**   **solar to 2054 (MW)**   **storage 2045 (MWh)**   **storage 2054 (MWh)**
  ----------------------------- ------------------------ ------------------------ ------------------------ ------------------------ ------------------------
  4%                            ⟦40.536⟧                 ⟦11,787.7⟧               ⟦30,938.8⟧               ⟦24,697.7⟧               ⟦56,289.9⟧

  6%                            ⟦30.248⟧                 ⟦11,787.7⟧               ⟦30,938.8⟧               ⟦24,697.7⟧               ⟦56,289.9⟧

  8% (central)                  ⟦23.323⟧                 ⟦11,787.7⟧               ⟦30,938.8⟧               ⟦24,697.7⟧               ⟦56,289.9⟧

  10%                           ⟦18.544⟧                 ⟦11,787.7⟧               ⟦30,938.8⟧               ⟦24,697.7⟧               ⟦56,289.9⟧

  12%                           ⟦15.162⟧                 ⟦11,787.7⟧               ⟦30,938.8⟧               ⟦24,697.7⟧               ⟦56,289.9⟧
  ----------------------------------------------------------------------------------------------------------------------------------------------------------

The objective ranges from ⟦40.536⟧ billion USD at 4% to ⟦15.162⟧ billion at 12%, a factor of ⟦2.67⟧, reflecting present-value scale rather than any change in what is in place by 2045. Every monetary result reported in this thesis is conditional on $r_{s} = 8\%$. No quantity reported as a result is conditional on it at any rate tested; the timing of additions within the window is, at every rate.

- Position of the central rate against sourced benchmarks: The 8% value lies between two benchmarks and coincides with neither. The World Bank’s prescriptive Ramsey-type mapping — a 3% per-capita growth rate corresponding to a 6% discount rate, with per-capita growth of 1 to 5% yielding 2 to 10% (World Bank, 2016) — applied to this model’s own demand-growth parameters (§3.2.3: GDP growth 4.5% and population growth 1.77% per year, giving per-capita growth of approximately 2.7%) implies a Ramsey-consistent rate of roughly 5.4 to 6%. The administrative convention applied by multilateral development banks runs materially higher, at 10 to 12% real (Zhuang et al., 2007). The central value sits above the figure the model’s own growth assumptions imply and below the administrative convention. It is classified ASSUMED, since no source states 8% itself, but it is bounded by two sourced figures rather than unanchored.

- Direction of bias: OECD (2021) distinguishes the social discount rate from the financial hurdle rate and finds that setting the social rate too high in energy-sector modelling systematically undervalues capital-intensive, low-running-cost technologies such as renewables, biasing against decarbonisation rather than toward it — contrary to the expectation that a higher rate favours such a technology by discounting away more of its future costs. The sweep reported above shows the mechanism inoperative over the 6 to 12% range, within which the plan does not move at all.

- Scenario specification: Unlike `dscr_case`, `capital_recovery_case` or `storage_cc_case` (§3.4), `social_discount_rate` is not a sentinel-required scenario key. It is set as a literal in the scenario dictionary and separately defaults to the identical value of 0.08 at the point of use in both builders, so a scenario dictionary omitting the key falls back silently to the same rate rather than raising an error. It is one of three parameters found to sit outside the required-key discipline, with `within_block_peaking_factor` and `selfgen_busbar_factor` (Appendix A, row 7; Appendix B). The general property — that every arm a runner sets reaches the built model, and that every default is declared — is tested in §3.6.10, and its results are reported there.

### 3.1.3 Temporal structure

Each year is divided into six time slices: two seasons by three intra-day periods. Representing intra-annual variation through a small set of representative slices is standard in long-term planning, and the consequences of the chosen resolution for investment decisions are well documented (Poncelet et al., 2016; Kotzur et al., 2018).

  ----------------------------------------------------------------------------------------------------------------------------------
  **slice set**            **members**                                        **hours**
  ------------------------ -------------------------------------------------- ------------------------------------------------------
  season $\sigma$          dry (181 days), wet (184 days)                     

  period $p$               night (7 h), day (13 h), peak (4 h, 18:00–22:00)   

  slice $s = (\sigma,p)$   six combinations                                   $H_{s} = D_{\sigma}h_{p}$, $\sum_{s}^{}H_{s} = 8760$
  ----------------------------------------------------------------------------------------------------------------------------------

$H_{\sigma} = \sum_{p}^{}H_{(\sigma,p)}$ denotes the total hours in season $\sigma$. The peak block lies after sunset, so solar contributes nothing to it by construction; this is what makes the firm-capacity constraint (3.21) bind.

An annual model cannot value storage so this makes slicing is necessary. With one time step per year, storage is bounded by

$d_{t} \leq \eta^{rt}q_{t},\quad\quad 0 < \eta^{rt} \leq 1,$ (3.4)

`[code: optimize_model.py, build_model: storage_balance]` and enters the annual balance as $d_{t} - q_{t}$, so its net contribution satisfies

$d_{t} - q_{t} \leq (\eta^{rt} - 1)\, q_{t} \leq 0.$ (3.5)

Storage can only consume energy, never supply it, so any storage investment is strictly dominated and optimal build is zero for any storage cost. No parameter sweep can change this. Slicing breaks the dominance by letting storage shift energy between slices within a season, and gives it capacity value through (3.21). §3.1.8 states what else follows from the difference between the two formulations.

### 3.1.4 Notation

**Sets and indices**

  -------------------------------------------------------------------------------------------------------------
  **symbol**                           **meaning**
  ------------------------------------ ------------------------------------------------------------------------
  $$t\mathcal{\in T}$$                 model years, 2025–2054

  $$\sigma \in \Sigma$$                seasons, $\{\text{dry},\text{wet}\}$

  $$p\mathcal{\in P}$$                 periods, $\{\text{night},\text{day},\text{peak}\}$

  $$s = (\sigma,p\mathcal{) \in S}$$   time slices, $\mathcal{S} = \Sigma \times \mathcal{P}$

  $$k\mathcal{\in K}$$                 conversion tiers $\{ A,B,C\}$, present only when conversion is enabled

  $$c\mathcal{\in C}$$                 buildable capital channels (§3.4.2)

  $$v$$                                capital vintage year (§3.4.2)
  -------------------------------------------------------------------------------------------------------------

**Parameters**

  ------------------------------------------------------------------------------------------------------------------------
  **Symbol**                                    **meaning**                                                **unit**
  --------------------------------------------- ---------------------------------------------------------- ---------------
  $$H_{s},\ H_{\sigma}$$                        hours in slice $s$; in season $\sigma$                     h

  $$\delta_{t}$$                                social discount factor (3.1)                               —

  $$D_{t},\ D_{t,s}$$                           annual busbar demand; $D_{t,s} = D_{t}w_{s}$               TWh

  $$w_{s}$$                                     normalised slice demand weight                             —

  $${\bar{K}}_{0}^{g},\ {\bar{K}}_{0}^{sol}$$   baseline gas and solar capacity                            MW

  $$R_{t}$$                                     cumulative retired baseline gas capacity (3.6a)            MW

  $$K_{t}^{h},\ E_{t}^{h}$$                     hydro capacity and annual energy (exogenous)               MW, TWh

  $$\theta_{\sigma}$$                           seasonal share of hydro energy                             —

  $$\mu$$                                       hydro minimum-flow fraction (0.50)                         —

  $$A_{t}$$                                     gas deliverable to the power sector                        TWh$_{th}$

  $$\eta^{g},\ \eta^{c}$$                       gas and converted-plant efficiency                         —

  $$\alpha$$                                    gas fleet availability factor                              —

  $${\bar{Y}}_{k}$$                             convertible open-cycle capacity in tier $k$                MW

  $$\upsilon$$                                  output uplift from a bottoming cycle (0.50)                —

  $$\phi_{s}$$                                  solar energy per MW in slice $s$                           TWh/MW

  $$\eta^{rt},\ \Delta$$                        storage round-trip efficiency; duration                    —, h

  $$\kappa^{j}$$                                capacity credit of technology $j$                          —

  $$P_{t},\ \omega,\ \rho_{t}$$                 peak demand; within-block peaking factor; reserve margin   MW, —, —

  $$\underline{x},\ {\bar{X}}_{t}$$             solar minimum-build floor; cumulative build cap            MW

  $$\lambda^{j},\ L$$                           land use per unit of technology $j$; land available        km²/MW, km²

  $$\beta$$                                     self-generation busbar factor                              —

  $$c^{gas},\ c^{self},\ c^{VoLL}$$             fuel cost; self-generation cost; value of lost load        USD/TWh

  $$\pi^{CO_{2}}$$                              carbon price (zero unless a carbon arm is active)          USD/tCO$_{2}$

  $$\varepsilon_{t}$$                           grid-service obligation (maximum off-grid share)           —

  $${\bar{E}}_{t}$$                             emissions cap                                              tCO$_{2}$
  ------------------------------------------------------------------------------------------------------------------------

Financing parameters — the concessional and commercial rates and tenors, the concessional envelope $B$, the public capital budget $B^{pub}$, the service tariff $\tau$ and the coverage margin $m$ — are defined in §3.4 with the constraints they enter.

$c^{self}$ is the priced cost of self-generation whenever a genset case is active. Where none is supplied it falls back to $c^{VoLL}$, which is a silent substitution of one parameter for another and is listed among the exceptions to the arm system in §3.6.3.

**Decision variables** (all non-negative)

  -----------------------------------------------------------------------------------------------------------
  **Symbol**                            **Meaning**                                  **index**   **unit**
  ------------------------------------- -------------------------------------------- ----------- ------------
  $$x_{t}^{pub},\ x_{t}^{eaas}$$        solar additions, public / service-financed   $$t$$       MW

  $$x_{t}^{g}$$                         new gas capacity                             $$t$$       MW

  $$x_{t}^{st}$$                        new storage energy capacity                  $$t$$       MWh

  $$\zeta_{t}$$                         service-channel subsidy                      $$t$$       USD

  $$V_{c,v}^{conc},\ V_{c,v}^{comm}$$   concessional / commercial tranche volume     $$c,v$$     USD

  $$f_{t,s}$$                           gas burned for power                         $$t,s$$     TWh$_{th}$

  $$\gamma_{t}$$                        gas curtailment (see §3.1.7)                 $$t$$       TWh$_{th}$

  $$h_{t,s}$$                           hydro generation                             $$t,s$$     TWh

  $$q_{t,s},\ d_{t,s}$$                 storage charge / discharge                   $$t,s$$     TWh

  $$g_{t,s}$$                           self-generation (backstop)                   $$t,s$$     TWh

  $$u_{t,s}$$                           unserved energy                              $$t,s$$     TWh

  $$y_{k}$$                             open-cycle capacity converted in tier $k$    $$k$$       MW

  $$f_{k,t,s}^{c}$$                     gas burned in converted plant                $$k,t,s$$   TWh$_{th}$
  -----------------------------------------------------------------------------------------------------------

Whenever self-generation is priced, $u_{t,s} = 0$ for all $t,s$ by construction `[code: optimize_model_sliced.py, build_model_sliced: unserved_fixed]`, and the grid-service shortfall is carried entirely by $g_{t,s}$. A reported unserved-energy total of zero is therefore not evidence that demand is met, and is never quoted as such. For the same reason, the value of lost load (Schröder and Kuckshinrichs, 2015) enters the objective only under the regression configuration.

### 3.1.5 Derived quantities

$K_{t}^{g} = {\bar{K}}_{0}^{g} - R_{t} + \sum_{\tau \leq t}^{}x_{\tau}^{g}$ (3.6)

where retirement follows a supplied cumulative schedule where one exists, and otherwise a flat rate $\bar{r}$ from a start year $t^{R}$:

$R_{t} = min\left( {\bar{K}}_{0}^{g},\ \ R_{t}^{sched}\ \text{ or }\ \bar{r}\, max(0,\, t - t^{R} + 1) \right)$ (3.6a)

`[code: optimize_model_sliced.py, build_model_sliced: gas_retirement]` The two forms correspond to the vintage schedule and the superseded flat proxy described in §3.3.1.

$K_{t}^{sol} = {\bar{K}}_{0}^{sol} + \sum_{\tau \leq t}^{}\left( x_{\tau}^{pub} + x_{\tau}^{eaas} \right)$ (3.7)

$E_{t}^{st} = \sum_{\tau \leq t}^{}x_{\tau}^{st},\quad\quad P_{t}^{st} = E_{t}^{st}/\Delta$ (3.8)

$G_{t,s}^{g} = \eta^{g}f_{t,s},\quad G_{k,t,s}^{c} = \eta^{c}f_{k,t,s}^{c},\quad G_{t,s}^{sol} = K_{t}^{sol}\,\phi_{s}$ (3.9)

`[code: optimize_model_sliced.py, build_model_sliced: capacity_accounting, generation_expressions]`

Equations (3.7) and (3.8) carry capacity forward without retirement within the horizon, which is exact for solar at a 30-year life and not for storage at 15: storage vintages reaching the end of their life are replaced, and the replacement capital is charged through the same treatment as the original (§3.4.2).

$E_{t}$ denotes total annual emissions, grid and self-generation combined. $E_{t}^{cap}$ denotes emissions inside the capped boundary, which is grid-only by default and total only if self-generation is explicitly placed inside the cap `[code: optimize_model_sliced.py: emissions_by_year, emissions_capped_by_year]`. §3.5.1 sets out the three quantities and the instruments they attach to.

### 3.1.6 Objective

At the central configuration, where capital is charged by tranche, the programme minimises

$Z = C^{fuel} + C^{cap} + C^{self} + C^{uns} + C^{CO_{2}} + C^{O\& M}$ (3.10)

`[code: optimize_model_sliced.py, build_model_sliced: system_cost_npv]` with

$C^{fuel} = \sum_{t}^{}\delta_{t}\sum_{s}^{}c^{gas}\left( f_{t,s} + \sum_{k}^{}f_{k,t,s}^{c} \right)$ (3.11)

$C^{self} = \sum_{t}^{}\delta_{t}\sum_{s}^{}c^{self}\, g_{t,s},\quad\quad C^{uns} = \sum_{t}^{}\delta_{t}\sum_{s}^{}c^{VoLL}\, u_{t,s}$ (3.12)

$C^{CO_{2}} = \sum_{t}^{}\delta_{t}\,\pi^{CO_{2}}\, E_{t}$ (3.13)

`[code: optimize_model_sliced.py, build_model_sliced: cost_fuel_npv, cost_selfgen_npv, cost_unserved_npv, cost_carbon_npv]`

The unserved-energy term prices involuntary curtailment at a value of lost load, the standard device for admitting reliability into a least-cost formulation (Schröder and Kuckshinrichs, 2015); its value is discussed with its provenance in §3.7.4. The self-generation term prices the diesel and petrol backstop described in §3.3.7. The carbon term is active only where a carbon-price arm is supplied; otherwise emissions enter through the constraint (3.23b) rather than the objective.

$C^{cap}$ is the discounted per-tranche capital charge on every vintage of public solar, service-based solar, gas and storage, defined in §3.4.2 at equation (3.35), together with the replacement capital those assets require where their technical life ends inside the horizon. Capital for open-cycle to combined-cycle conversion is charged separately, as a commercial-only, single-vintage expenditure, and is specified with the conversion tiers in §3.3.3 rather than with the tranche split. $C^{O\& M}$ is fixed and variable operation and maintenance on the installed fleet, taken from the NREL Annual Technology Baseline (Mirletz et al., 2024) and computed by `compute_om_cost_npv` `[code: io.py: compute_om_cost_npv]`.

Under the regression configuration, which retains an overnight capital treatment for comparison, $C^{cap}$ is replaced by overnight capital expenditure plus replacement cost less an end-of-horizon salvage credit. The two treatments are mutually exclusive and no reported result mixes them.

$Z$ is a financial cost borne by the Nigerian power system rather than a social resource cost, and §3.4.1 sets out what follows from that perspective for the treatment of concessional finance and of transfers inside the system boundary.

There is one objective and it is always cost-minimising. Bi-objective trade-offs are constructed by the ε-constraint method, not by weighting the objective (§3.6.3).

### 3.1.7 Constraints

Financing constraints are stated in §3.4. Those below describe the physical system.

- **Energy balance**, for every $t,s$:

$G_{t,s}^{g} + \sum_{k}^{}G_{k,t,s}^{c} + G_{t,s}^{sol} + h_{t,s} + d_{t,s} - q_{t,s} + \beta\, g_{t,s} + u_{t,s}\mspace{6mu} \geq \mspace{6mu} D_{t,s}$ (3.14)

`[code: optimize_model_sliced.py, build_model_sliced: energy_balance]` The inequality admits free disposal of surplus generation; the conversion term is absent when no conversion arm is active. The factor $\beta$ converts self-generated energy to a busbar-equivalent basis and is unity in every reported configuration (§3.7.3).

- **Gas capacity**, for every $t,s$:

$G_{t,s}^{g} \leq \left( K_{t}^{g} - \sum_{k}^{}y_{k} \right)H_{s}\,\alpha/10^{6}$ (3.15)

`[code: optimize_model_sliced.py, build_model_sliced: gas_capacity_constraint]` Converted capacity is removed from the simple-cycle fleet here and re-enters through the conversion generation terms, so no megawatt is counted twice.

- **Gas deliverability**, for every $t$:

$\sum_{s}^{}f_{t,s} + \sum_{k,s}^{}f_{k,t,s}^{c} + \gamma_{t} \leq A_{t}$ (3.16)

`[code: optimize_model_sliced.py, build_model_sliced: gas_balance]` This is a fuel-supply bound on a power system rather than a bound on generating capacity, and it is the distinguishing physical feature of the Nigerian case: deliverability, not installed capacity, is the binding gas constraint in most years (§3.3.1).

The variable $\gamma_{t}$ carries no cost in the objective and appears in no other constraint. In a less-than-or-equal constraint it can therefore only tighten the bound, and it takes the value zero in every reported solution; forcing it to zero reproduces every solve exactly. It is retained because it is read by a diagnostic, and it is stated here so that it is not mistaken for a priced curtailment decision.

Under the *exempt* deliverability arm, new gas capacity carries its own fuel allowance and the right-hand side becomes $A_{t} + \left( \sum_{\tau \leq t}^{}x_{\tau}^{g} \right)\, 8760\,\alpha/\eta^{new}/10^{6}$ `[code: optimize_model_sliced.py, build_model_sliced: gas_balance]` — a full-year allowance, which makes that arm an upper bound rather than a realistic alternative.

- **Conversion**, for every tier $k$:

$0 \leq y_{k} \leq {\bar{Y}}_{k}$ (3.17)

`[code: optimize_model_sliced.py, build_model_sliced: conversion_bounds]`

- **Storage**, for every $t$ and season $\sigma$:

$\sum_{p}^{}d_{t,(\sigma,p)} \leq \eta^{rt}\sum_{p}^{}q_{t,(\sigma,p)}$ (3.18)

and for every $t,s$:

$q_{t,s} \leq P_{t}^{st}H_{s}/10^{6},\quad\quad d_{t,s} \leq P_{t}^{st}H_{s}/10^{6}$ (3.19)

`[code: optimize_model_sliced.py, build_model_sliced: storage_season_balance, storage_power_limits]` The balance holds within each season and not across them: storage shifts energy between the periods of a season but cannot carry it from wet to dry. Two properties of this representation are declared. It is an energy balance over a season rather than a chronological state of charge, so discharge in one period is not required to follow charging in an earlier one; the formulation therefore does not guarantee that a dispatch is chronologically feasible within a season, and the direction of the resulting bias favours storage. And the round-trip efficiency $\eta^{rt}$ is the only loss represented; no degradation, self-discharge or cycle limit is imposed.

- **Hydro**, for every $t$, season $\sigma$ and slice $s$:

$\sum_{p}^{}h_{t,(\sigma,p)} = E_{t}^{h}\,\theta_{\sigma}$ (3.20a)

$h_{t,s} \leq K_{t}^{h}H_{s}/10^{6}$ (3.20b)

$h_{t,s} \geq \mu\, E_{t}^{h}\,\theta_{\sigma}\,\frac{H_{s}}{H_{\sigma}}$ (3.20c)

`[code: optimize_model_sliced.py, build_model_sliced: hydro_season_budget, hydro_power_limit, hydro_min_flow]` The seasonal budget (3.20a) is an equality: hydro energy is exogenous and must all be dispatched within its season, so the model cannot spill. The minimum-flow constraint (3.20c) requires each slice to carry at least a fraction $\mu$ of its hours-proportional share of that season’s energy, representing the downstream release obligations of the Kainji and Jebba schemes (§3.3.5). The plan’s insensitivity to $\mu$ is reported in §3.3.5.

$\kappa^{sol}K_{t}^{sol} + \kappa^{g}K_{t}^{g} + \kappa^{st}P_{t}^{st} + \kappa^{h}K_{t}^{h} + \sum_{k}^{}\kappa_{k}^{c}\,\upsilon\, y_{k}\mspace{6mu} \geq \mspace{6mu} P_{t}\,\omega\,(1 + \rho_{t})$ (3.21)

`[code: optimize_model_sliced.py, build_model_sliced: firm_capacity_constraint]` The capacity credit of a technology is its contribution to generation adequacy, conventionally estimated as an effective load-carrying capability against a loss-of-load probability target (Keane et al., 2011). This model uses fixed credits instead, which is a simplification of that method. For solar the fixed credit of zero is exact by construction, since the peak block follows sunset; for storage and gas the fixed values are the simplification.

In the conversion term, $\upsilon = 0.50$ is the output uplift a bottoming cycle adds to a converted open-cycle unit. Only that increment is credited here: the converted unit’s original nameplate is already inside $K_{t}^{g}$, so conversion contributes to adequacy once.

The reserve margin $\rho_{t} = 900/P_{t}$ is derived from NIRP 2024’s 900 MW spinning-reserve requirement (Federal Ministry of Power, 2024, §2.3, p.19). Two consequences follow and are declared here rather than left implicit. The first is categorical: a spinning reserve is an operating quantity, conventionally sized to the largest single contingency, whereas the requirement in (3.21) is a planning reserve margin, and the two are not the same object. The second is arithmetical: because the requirement is fixed in megawatts while peak demand grows across the horizon, the implied fractional margin falls from ⟦19.5%⟧ in 2025 to ⟦6.4%⟧ by 2054. A reader who takes $\rho_{t}$ for a conventional planning margin of 15 to 20% would misread the adequacy standard this model imposes in later years. The direction of the resulting bias is toward finding adequacy cheaper to meet, and §3.7.4 records it with the other weakly identified inputs to this constraint.

The within-block factor $\omega$ corrects for sizing against the four-hour block average rather than the instantaneous peak (§3.2.5).

- **Solar build and land**, for every $t$:

$x_{t}^{pub} + x_{t}^{eaas} \geq \underline{x},\quad\quad\sum_{\tau \leq t}^{}\left( x_{\tau}^{pub} + x_{\tau}^{eaas} \right) \leq {\bar{X}}_{t}$ (3.22a)

$\lambda^{sol}K_{t}^{sol} + \lambda^{g}K_{t}^{g} + \lambda^{st}E_{t}^{st} \leq L$ (3.22b)

`[code: optimize_model_sliced.py, build_model_sliced: solar_min_build, solar_cumulative_cap, land_constraint]` The minimum-build floor $\underline{x}$ is a scalar set to zero in every standard configuration, so that half of (3.22a) is inactive in practice. The cumulative cap is constructed only when a deployment series is supplied. The land constraint (3.22b) is slack in every reported solution, using ⟦16.4%⟧ of its allowance at the central configuration; it is retained so that deployment-constrained arms cannot return physically unsited capacity, and it does not shape any reported result.

- **Policy**, for every $t$, each constructed only when supplied:

$\sum_{s}^{}\left( g_{t,s} + u_{t,s} \right) \leq \varepsilon_{t}\, D_{t}$ (3.23a)

$E_{t}^{cap} \leq {\bar{E}}_{t}$ (3.23b)

`[code: optimize_model_sliced.py, build_model_sliced: grid_service_obligation, emissions_cap_constraint]` The grid-service obligation (3.23a) bounds self-generation and unserved energy jointly, so the two routes by which demand may go unmet from the grid are limited together rather than separately. It is applied year by year; an alternative mode sums it over the horizon, which permits self-generation to be front- or back-loaded, and §3.5.4 reports what that difference does to a reliability standard. The emissions cap (3.23b) is applied year by year; where only a scalar is supplied it applies cumulatively. Because the cap’s default boundary is grid-only, abatement can be displaced across that boundary into self-generation, which is examined in §3.5.6 and in Chapter 4.

### 3.1.8 The two builders

The model has two builders. They differ structurally, not in temporal resolution, and that difference is what justifies the dominance argument of §3.1.3.

The annual builder represents each year as a single period. The sliced builder represents each year as six slices — dry and wet seasons crossed with night, day and peak periods, weighted 1,267, 2,353, 724, 1,288, 2,392 and 736 hours to sum to 8,760. Capacity additions, the entire financing layer, land, build caps, the annual gas budget, emissions and the reliability constraint are annual in both. Dispatch, curtailment, unserved energy, self-generation, storage flows and hydro generation are sliced in the second. Solar generates in day slices only, at the same annual capacity factor.

Three consequences follow, each structural rather than a matter of resolution.

- Conversion is not implemented in the annual builder. It raises on any conversion arm other than `conv_none`, so no scenario in which conversion is available can be solved there at all.

- Hydro is exogenous in the annual builder. It enters as a fixed annual energy series with no dispatch and no seasonality. The sliced builder dispatches it against a seasonal energy budget, a per-slice power limit and a minimum-flow floor.

- The annual builder cannot value storage’s energy-arbitrage contribution, only its firm-capacity credit. Within a single annual period, discharge is bounded by round-trip efficiency times charge, so net energy is non-positive by construction: storage can only consume energy and has no mechanism by which to earn from shifting it. A throughput parameter does exist in that builder, capping storage flows at power times 700 deployable hours per year, but a cap on throughput is an additional limit rather than a mechanism for earning from it, and it does not repair the formulation; the parameter is inert in the sliced builder, which reads it nowhere. The firm-capacity credit term is identical in both builders, so storage can still be built for adequacy — it simply cannot be built for energy. In the sliced builder, charge and discharge are assigned to distinct slices of the same season, and throughput follows from that structure.

Every result reported in this thesis comes from the sliced builder. Each shared experiment function takes a required builder argument with no default, and all eight live runners pass the sliced builder. The annual builder survives in three deliberate two-leg comparison instruments — the salvage sensitivity, the self-generation frontier and a structural probe — which solve both formulations by design in order to measure the difference between them, and in dead code reachable from no live caller.

Two exceptions are declared rather than left implicit.

Two results reported in Chapter 4 retain annual-builder figures and are labelled in place where they appear: one set of superseded figures since re-measured on the sliced builder, and three components of a second result, marked annual-only in that result’s own staleness table.

And the emissions cap’s baseline quantity $E_{\text{base}}$ is derived under `conv_none`, because the diagnostic field the cap construction reads is fleet-only and would understate the baseline on a conversion-active run. That inheritance was tested rather than argued: regenerating the baseline under the same no-conversion arm on each builder reproduces $E_{\text{base}}$ to a relative ⟦4.6×10⁻⁹⟧ and every cap value in the resulting trajectory to within ⟦8×10⁻⁹⟧, which is agreement at every printed digit. The cap is therefore builder-invariant in value, and no capped result depends on which builder produced its baseline.

## 3.2 Demand

### 3.2.1 Demand as a service obligation

In a capacity-expansion model, exogenous demand is the service obligation the system is required to meet, not a forecast of what the system will deliver. The distinction governs every choice in this section. A demand series held down by the very constraints the model is built to evaluate — gas supply, network capacity, the sector’s payment chain — would, if adopted as the obligation, ask the model only for the cheapest way to keep the grid the size it already is.

A single audit question is therefore applied to every demand input: was this number produced by the system working, or by the system failing? Three inputs in this section fail that test and are rejected or bounded as a result: the collected-energy base (§3.2.2), historical generation growth (§3.2.3), and the served load shape (§3.2.5).

### 3.2.2 Base year and metering point

Base-year demand is 37.09 TWh in 2024, measured as gross generation at the busbar (NERC, 2025, NERC annual reports; 37,093.70 GWh). Gross generation is the natural boundary for a generation capacity-expansion model: capacity serves generation, and the NDC emissions caps account against grid output.

An earlier version of the model used 23.08 TWh, an annualisation of first-quarter 2024 *collected* energy. That figure sits near the bottom of the sector’s loss ladder:

  -----------------------------------------------------------------------------------------
  **Stage**                                                             **TWh**
  --------------------------------------------------------------------- -------------------
  gross generation                                                      37.09

  after transmission losses (7%)                                        34.50

  after aggregate technical, commercial and collection losses (36.4%)   21.97
  -----------------------------------------------------------------------------------------

Collected energy excludes electricity that is consumed but stolen, unmetered or unpaid — all of it real demand by real users — and so understates the system’s scale by roughly 38%. It is a revenue quantity, not a demand quantity.

### 3.2.3 Demand growth

#### Why historical generation cannot supply the growth rate

The obvious calibration, Nigeria’s historical grid-generation growth, is rejected as circular. Grid generation is a supply series, flat because gas supply, wheeling capacity and the payment chain held it flat. Calibrating demand to it assumes those constraints persist, and the model would then report modest capacity requirements because demand was assumed not to grow.

The rejection rests on decomposition rather than on argument alone. Since $(1 + g^{gen}) = (1 + g^{pop})(1 + g^{pc})$ exactly, observed generation growth splits into a demographic part and a per-capita residual, using World Bank population and access series (World Bank; `SP.POP.TOTL`, `EG.ELC.ACCS.ZS`) and Ember generation data (Ember, yearly electricity data):

  ------------------------------------------------------------------------------------
  Basis                             generation     denominator   per-capita residual
  --------------------------------- -------------- ------------- ---------------------
  total population, 2010–2025       +3.14%/yr      +2.39%/yr     +0.73%/yr

  total population, 2014–2024       +1.53%/yr      +2.27%/yr     −0.73%/yr

  connected population, 2010–2024   +2.63%/yr      +4.36%/yr     −1.66%/yr
  ------------------------------------------------------------------------------------

All three rows are recomputed from the versioned Ember extract held in the repository — 26.13 TWh in 2010, 32.28 in 2014, 37.56 in 2024, 41.54 in 2025 — giving 3.1387%, 1.5265% and 2.6257% per year respectively, and matching the table to the rounding shown.

Two 2024 figures appear in this section, from different compilations and used for different purposes. The model’s demand level is NERC’s 2024 gross generation of 37,093.70 GWh, and that figure alone sets the base year. Ember’s 37.56 TWh enters only the derivation of the growth rates above, never as a level. Ember exceeds NERC by 1.26% in 2024 and by more in adjacent years — 8.8% in 2022, 11.5% in 2023 and 6.0% in 2025 — and neither series is a stable rescaling of the other (Appendix B, L55). The 2022 gap is the least reliable of the four, since NERC’s 2022 entry is an average-power figure and whether it is sent-out or generated is unverified. The growth rates are therefore taken from Ember’s internally consistent series and the level from NERC’s, rather than mixing the two. The 3.14% of the first row is also the constrained_continuation arm's value.

Between two-thirds and all of observed generation growth is demographic. Per connected person, grid generation fell from about 327 to 258 kWh over 2010 to 2024 and is negative on every window tested. The series carries almost no information about consumption intensity.

The connected-population row overstates the decline in magnitude, though not in sign: the access series counts off-grid and distributed supply as well as grid connections, so it overstates grid-connected population, and increasingly so over time. The true intensity trend lies between the two rows, at or below zero either way.

The same reasoning bounds income elasticities estimated on Nigerian or sub-Saharan African data: where supply is rationed the consumption response to income cannot be observed, so such estimates are biased downward and are used here only as a lower bound. NIRP 2024 takes the same position, deriving its demand forecast bottom-up from population, GDP, electrification and losses rather than from historical generation (Federal Ministry of Power, 2024, Table 14).

#### Three routes to a growth rate

Each adopted arm is supported by at least two independent routes.

1.  Route A — driver decomposition: If per-capita demand grows with elasticity $\varepsilon$ on per-capita income, total demand grows at

$g = \varepsilon\, g^{GDP} + (1 - \varepsilon)\, g^{pop}.$ (3.24)

The long-run income elasticity of economy-wide electricity demand in middle-income countries averages between 0.7 and 0.9 and is stable around 0.8 (Liddle and Hasanov, 2023). The adopted value is $\varepsilon = 0.80$, the paper’s own stated central tendency (p.827, abstract; p.829; p.843, §6). Population growth averages $g^{pop} = 1.77\%$ over 2024 to 2045, from 232.68 to 336.66 million (United Nations, Department of Economic and Social Affairs, Population Division). GDP growth of $g^{GDP} = 4.5\%$ is a declared assumption and not a forecast: above the post-2015 average, which contains two recessions, and well below the oil-boom rate. Beyond about 2030 no credible source forecasts Nigerian GDP.

At the central values (3.24) gives 3.95%; across the source’s range $\varepsilon \in \lbrack 0.7,0.9\rbrack$ it gives 3.68 to 4.23%.

A previously carried range of 0.76 to 0.90 does not survive tracing: 0.76 is a different paper's result cited in Liddle and Hasanov's own review, 0.90 is their individual-country midpoint rather than the range, and 0.83 appears in neither. The panel includes two sub-Saharan countries, so the earlier range's implicit exclusion of the region was an uncited judgement (Appendix C, L54).

Equation (3.24) makes the load-bearing assumption explicit. The coefficient on population growth is $(1 - \varepsilon) = 0.20$ at the central value, so a 0.3 percentage-point change in population growth moves $g$ by about 0.06 points, and GDP growth is roughly four times more load-bearing than population growth. That matters because Nigeria’s last census was in 2006 and every population figure is a contested projection: the demand arm is far more sensitive to the GDP assumption, which is declared, than to the population figure, which is uncertain.

Consistency with NIRP 2024. NIRP's figures satisfy (3.24) at an implied elasticity of 1.19, with population growth of 2.1% and GDP growth of 6.8%, returning its stated 7.70%. The elasticity is obtained by solving (3.24) using NIRP's own three figures, so this is arithmetic consistency and not independent validation. An elasticity above unity makes the population coefficient negative, which is not an interpretable decomposition: NIRP's growth is not generated by the mechanism (3.24) represents.

2.  Route B — per-capita convergence. Nigeria’s 2024 consumption is about 161 kWh per person. Kenya crossed that level in about 2009, so Nigeria is roughly fifteen years behind; convergence to Kenya’s trajectory by 2045 gives 3.67%. Ghana cannot serve as the comparator: Nigeria’s 2024 level is below Ghana’s lowest since 2000, and converging on Ghana’s present level would require 9.24% a year.

3.  Route C — NIRP 2024, as a benchmark only. NIRP’s total demand rises from 59.6 to 328.6 TWh over 2022 to 2045, or 7.70% a year. That base already includes self-generation, so it is a wider-scope quantity (§3.2.4) and cannot be applied to this model’s grid-only base. NIRP’s grid-only rate of 11.0% is worse still for this purpose: it is organic growth plus the absorption of self-generation onto the grid, which in this architecture is the gap between scopes rather than a growth rate. Route C is held outside the arms as a benchmark.

#### The arms

  ----------------------------------------------------------------------------------------------------------------------------------------------------
  **Arm**                      **Rate**   **Basis**
  ---------------------------- ---------- ------------------------------------------------------------------------------------------------------------
  `constrained_continuation`   3.14%      2010–2025 grid-generation growth

  `organic_central`            3.9%       Route A 3.95%, Route B 3.67%

  `organic_high`               6.0%       Route A 6.30% \[CONFIRM: the GDP growth assumption behind this figure\], Route B 6.40%, NIRP low case 5.9%
  ----------------------------------------------------------------------------------------------------------------------------------------------------

Annual demand is projected as

$D_{t} = D_{2024}\,(1 + g)^{\, t - 2024}$ (3.25)

`[code: demand.py: project_baseline_demand]`

`constrained_continuation` is a constraint-persistence arm rather than organic demand growth, and is never presented as such: it is the counterfactual in which the sector’s constraints hold. It clears the floor test that demand grow at least as fast as population, and implies per-capita growth of 1.35% a year, above the 0.73% actually observed, so even the pessimistic arm is mildly optimistic against the record.

### 3.2.4 Scope of the demand series

The base of 37.09 TWh is grid generation only. Electricity produced by households’ and firms’ own generators lies outside it by construction, and that is a large quantity: backup generation across sub-Saharan Africa was about 40 TWh from 40 GW in 2018, some 8% of the region’s generation, of which Nigeria accounted for almost half — about 18 TWh from roughly 9 GW (IEA, 2019). A figure of about 40% of Nigerian electricity needs met by backup generators has also been reported for 2024.

The model therefore represents roughly 60 to 67% of national electricity consumption. The two figures bound the share from different years — 40% of needs implies 60% represented, while 18 TWh against 37.09 TWh implies 67% — and neither is exact. This is a deliberate choice of a grid-planning boundary rather than an omission, but it bounds every claim the thesis makes: conclusions are about the grid’s service obligation, not about national demand. Self-generation enters the model only as the priced backstop that meets unserved grid demand (§3.3.7), and the absorption of existing self-generation onto the grid is outside this formulation.

### 3.2.5 Intra-annual shape

Annual demand is allocated across the six slices by a period factor $f_{p}$ and a season factor $f_{\sigma}$, each normalised so that its hours-weighted mean is one:

$D_{t,s} = D_{t}\, w_{s},\quad\quad w_{s} = \frac{f_{\sigma}\, f_{p}\, H_{s}}{\sum_{s'}^{}f_{\sigma'}\, f_{p'}\, H_{s'}}$ (3.26)

`[code: optimize_model_sliced.py: _normalised_demand_weights]`

The peak-to-average ratio is derived, never set. For a fixed slice partition, the block peak-to-average ratio is an identity of the peak block’s energy share $\varphi_{peak}$:

$\text{PAR}_{block} = \varphi_{peak}\,\frac{8760}{h_{peak}}$ (3.27)

where $h_{peak}$ is the annual hours in the peak block. It is therefore a restatement of the shape rather than an independent quantity. The specification was inverted: each arm supplies the shape, and the ratio is derived and reported.

Peak demand is annual demand expressed as an average load and scaled by that ratio:

$P_{t} = \frac{D_{t} \cdot 10^{6}}{8760}\mspace{6mu}\text{PAR}_{block},\quad\quad\varphi_{peak} = \sum_{\sigma}^{}w_{(\sigma,\,\text{peak})},\quad h_{peak} = \sum_{\sigma}^{}H_{(\sigma,\,\text{peak})} = 1460$ (3.28)

`[code: scenarios.py: peak demand construction]` Because both factor sets are normalised to an hours-weighted mean of one, the season factors cancel in (3.27) and $\text{PAR}_{block}$ reduces to the peak period factor $f_{peak}$ to four decimal places — 1.087668 against a declared 1.0877, the residual arising from rounding in the published factor sets. The derived ratios in the table below are therefore the peak factors themselves.

$P_{t}$ is a four-hour block-average peak. The firm-capacity constraint (3.21) lifts it to an instantaneous peak through the within-block factor $\omega$, defined as the ratio of the instantaneous to the block peak-to-average ratio in the source data.

The shape is aggregated onto the six-slice partition from an hourly dataset for 2016 carrying both an unsuppressed demand reconstruction and the suppressed series actually served (Oluwole, 2023; 8,784 observations).

  ---------------------------------------------------------------------------------------
  **Arm**                         **night / day / peak factor**   **derived block PAR**
  ------------------------------- ------------------------------- -----------------------
  `peak_suppressed`               1.0215 / 0.9935 / 0.9834        0.9834

  `peak_unsuppressed` (central)   0.9056 / 1.0239 / 1.0877        1.0877

  `peak_high`                     0.8849 / 1.0005 / 1.2000        1.2000
  ---------------------------------------------------------------------------------------

`peak_high` is constructed rather than measured: its evening factor is pinned at 1.20, with night and day derived by preserving the unsuppressed night-to-day ratio and solving for an hours-weighted mean of one.

The served shape is a floor, not a candidate. The suppressed series is a supply shape, and the data shows it directly. Rationing flattens the evening specifically: the peak factor falls from 1.088 to 0.983 while the night factor rises from 0.906 to 1.022. Only 41.9% of demand was served. It is retained only as a declared lower bound, so that the direction of the bias is visible.

The same pattern appears in current grid data. Nigeria’s all-time peak of 5,801.84 MW on 4 March 2025, against that day’s average of 5,348.8 MW, gives a ratio of 1.085, and grid load factors of 90 to 94% were recorded in early 2026. Figures of that kind are artefacts of rationing — supply binds, so dispatch runs flat while distribution companies shed load — and are never used as a demand observation.

NIRP 2024 contains no daily load profile, no peak-hour timing and no statement of its temporal resolution. Its load factor of 0.76 is stipulated and, as printed, is internally inconsistent: Table 14 gives both 71% and 76%, while the energy–capacity pairs in Table 15 return 0.756, 0.759 and 0.760. A NIRP shape arm is therefore defined but raises if selected, since the block factors it would require cannot be derived from the document.

The shape is a 2016 vintage applied from 2025. Only the shape is taken — levels come from NERC and the growth arms — and load shapes are more persistent than levels, being driven by near-constant equatorial daylight, working and school hours, and sectoral mix. Three developments since 2016 plausibly moved it, all in the same direction: distributed solar under rural electrification programmes moves daytime load behind the meter; the 2023 removal of the fuel subsidy changed the economics of self-generation; and the 2024 Band A tariff reform reallocated served supply. Each concentrates grid draw in the evening. The true 2024 shape is therefore expected to be more evening-peaked than 2016’s, so the central arm understates firm-capacity requirements and the cost of a solar-heavy pathway. `peak_high` sweeps that direction; nothing sweeps the other, deliberately.

### 3.2.6 Independence of the cap from the demand arm

The NDC caps are built from the model’s own baseline gas-to-power trajectory. Rebuilding them from the `constrained_continuation` baseline yields caps identical to those built from `organic_central`, because baseline gas reaches its deliverability ceiling by 2026 and the baseline emissions level is then set by gas supply rather than by demand.

The caps are therefore robust to any revision of the demand arm, which decouples the cap specification from the demand specification. The corollary is uncomfortable and is taken up in Chapter 5: because the cap is anchored to a gas-constrained baseline, the emissions trajectory is met substantially by the system remaining supply-constrained.

## 3.3 Supply representation

The model represents four supply resources — gas, solar, hydro and storage — together with a priced self-generation backstop. Three principles govern the section and are stated once here rather than repeated.

The binding constraint on gas is fuel, not capacity. Nigeria’s installed gas fleet exceeds what the gas supply system delivers to it, so a capacity-expansion model that constrained only megawatts would misrepresent the system entirely.

A resource’s value depends on when it is available and not only on how much energy it produces. Solar generates nothing in the model’s evening peak block by construction and therefore carries zero capacity credit, while storage and gas carry positive credits. This distinction governs the composition of every plan the model builds.

Several parameters in this layer are declared assumptions rather than sourced values, and each is stated with a direction of bias. The gas availability factor and the fleet’s commissioning vintages are the two most consequential.

### 3.3.1 The gas fleet

**Baseline.** The modelled gas fleet is 10,947 MW, comprising the 23 plants for which NERC publishes availability data, identical in the December 2024 and 2026/Q1 tables (NERC, Annual Report 2025). NERC’s licensed total is 11,361 MW; the difference is AES (294 MW) and Gbarain (120 MW), which appear in the licensed list but carry no availability data. The reconciliation is by exclusion and never by scaling, and a structural guard raises if the two totals diverge.

The model reproduces the observed 2024 position closely: against NERC’s reported gas generation of 25.62 TWh, the 2025 solve produces ⟦24.79⟧ TWh, a deviation of ⟦−3.2%⟧, with no unserved energy and no self-generation in that year.

**Composition, and what it implies about efficiency.** Of the 10,947 MW fleet, 59.2% is open cycle (6,482 MW), 21.7% combined cycle (2,380 MW) and 19.0% gas-steam (2,085 MW). A capacity-weighted ISO-condition efficiency across that mix is approximately 0.3375 on a higher heating value basis, against a measured fleet efficiency of 0.287. A substantial part of the fleet’s low efficiency is therefore composition rather than degradation — and the composition is itself a rational response to the constraint this thesis examines, since open-cycle plant is cheap to build and tolerant of interrupted fuel supply. The weighting is capacity-based and indicative; a generation-weighted figure would differ and has not been computed.

**Availability.** This is the largest open parameter in the supply layer. NERC measures gas plant availability at 30.6% in 2024, 29.0% in 2025/Q4 and 26.2% in 2026/Q1 against a stable fleet. The model’s inherited assumption is 0.80, roughly 2.6 times the observed figure, and worth about 5,700 MW of credited firm capacity in a constraint that binds in ⟦20 of 30⟧ years at the central configuration.

Neither figure can simply replace the other, and the reason is an identification problem rather than a choice of value. NERC’s availability factor conflates mechanical, atmospheric and fuel-related causes of unavailability, whereas this model constrains fuel separately through the gas balance (3.16); adopting the measured figure would charge the gas shortage twice. The model’s 0.80 was intended as a fuel-agnostic mechanical availability, which is a different quantity from the one NERC reports (Institute of Electrical and Electronics Engineers \[IEEE\], 2023).

The measured distribution is bimodal, and its low end is not a fuel story: Azura at 93.7 to 98.5% and Ikeja at 88.1 to 100%, against Alaoji at 0.01%, Sapele Steam at 2.7% and Omotosho II at 1.7%. Availability tracks gas-contract quality at the top of the distribution and non-operation at the bottom. Excluding every plant below 20% availability lifts the survivors only to 41.4%, and discards 5,101 MW of nameplate to do it: the shortfall is fleet-wide and not a matter of a few stranded assets.

The specification therefore carries arms rather than a value.

  ----------------------------------------------------------------------------------------------------------------------
  **Arm**                 $$\mathbf{\alpha}$$            **Status**
  ----------------------- ------------------------------ ---------------------------------------------------------------
  `gas_avail_observed`    0.3061                         declared lower bound; double-counts fuel

  `gas_avail_central`     not set — raises if selected   intended as mechanical availability only

  `gas_avail_nameplate`   0.80                           the inherited assumption, retained as an explicit upper bound
  ----------------------------------------------------------------------------------------------------------------------

The central arm is unsourced and raises if selected, and results are therefore reported under the upper bound. The direction of bias is stated plainly: `gas_avail_nameplate` credits the fleet with firm capacity it demonstrably does not deliver, and so understates both the adequacy problem and the cost of solving it.

**Retirement.** A flat 680 MW a year from 2035, described in its own source comment as a brownfield proxy, removes the entire fleet by 2051 and hands the firm-capacity burden to storage. It is replaced by a vintage schedule: per-plant commissioning years and capacities, retiring at commissioning plus an assumed technical life, with arms at 30 years (central), 40 and 50, entering as the cumulative series \$R_t\$ of (3.6a).

The commissioning years are ASSUMED. They are an analyst estimate from general knowledge of Nigerian project histories, not traced to NERC, NDPHC, TCN or company filings, none of which was obtainable within the project timeframe. Confidence is flagged per plant: 35.8% of capacity high, 47.2% medium, 17.0% low. The largest single weakness is Afam IV–V (726 MW), where one entry blends a 1970s and an approximately 2002 vintage that should be separate rows. The arms are named `gas_retire_vintage_*` and never `_sourced`.

A flat technical life is itself imperfect for a rehabilitated fleet: the 40-year case retires Delta in 2025 and Egbin in 2026, both operating. Two effects run in opposite directions — rehabilitation extends life beyond nameplate vintage, while under-utilisation extends calendar life because running hours rather than years drive degradation — and 30 years is taken as central on the grounds that these partly cancel, not on the grounds that 30 is correct.

Direction of bias: every vintage arm retires substantially less than the proxy, with means of 359, 277 and 152 MW per year over 2035 to 2054 against 680, so results computed under the proxy overstate retirement, the storage requirement and system cost.

**New build.** New gas capacity is a decision variable under ‘gas_build_case’: ‘gas_none’ (fixed at zero, retained so that pre-correction results stay attributable), ‘gas_unconstrained’, and gas_ndc_17gw, capped on an NDC-consistent trajectory derived from NIRP Tables 19 and 22 (9,188 MW by 2030; 17,027 by 2035; 20,945 by 2041; 18,408 by 2045, nameplate, linearly interpolated) and implemented in both builders. Holding it at zero was not defensible in a model evaluating a commitment targeting roughly 17 GW of gas-fired capacity by 2035: with solar credited at zero and hydro exogenous, storage became the only resource able to supply an incremental firm megawatt, which determined the storage result mechanically. New gas is declared combined cycle, which places the burden of restraint on the deliverability constraint rather than on an unfavourable technology assumption.

New gas is now a decision variable under `gas_build_case`: `gas_none` (fixed at zero, retained so that pre-correction results stay attributable), `gas_unconstrained` (no cap), and `gas_ndc_17gw`, capped on an NDC-consistent trajectory derived from NIRP Tables 19 and 22 (9,188 MW by 2030; 17,027 by 2035; 20,945 by 2041; 18,408 by 2045, on a nameplate basis, linearly interpolated) and implemented in both builders `[code: scenarios.py: gas_build_case; optimize_model_sliced.py, build_model_sliced: gas_cap_rule]`. New gas is declared combined cycle, which biases toward gas appearing attractive and therefore places the burden of restraint on the deliverability constraint rather than on an unfavourable technology assumption.

### 3.3.2 Gas deliverability

Deliverability, not capacity, is the binding gas constraint in most years. Constraint (3.16) limits total gas burned to the quantity $A_{t}$ deliverable to the power sector, and the dual of that constraint is what Chapter 4 reports as the gas shadow price.

The anchor rests on one measurement. Every arm starts from $A_{2025} = 89.27$ TWh$_{th}$, derived from NERC’s 2024 generation of 37,093.70 GWh, of which 30.92% is hydro, giving 25.62 TWh$_{e}$ of gas generation, divided by an assumed fleet efficiency of 0.287. The implied volume of approximately 805 MMscf/d at a higher heating value of 1,036 Btu/scf is that same figure expressed in volume terms rather than an independent observation, and no gas-volume series for 2024 was obtained. The anchor therefore rests on a single measured quantity, NERC’s generation, together with two assumed parameters. Section 3.7 records the consequence for every result that the fuel envelope binds.

The seven arms answer two different questions and are distinct epistemic objects.

Extrapolative arms project historical drift from the anchor:

  ---------------------------------------------------------------------------------------------------------------
  **Arm**            **Form**                                                              **2035**   **2054**
  ------------------ --------------------------------------------------------------------- ---------- -----------
  `D``ownside`       2-year lag, then −3%/yr                                               69.96      39.22

  `B``aseline`       drift +0.5%/yr, no lag                                                93.84      103.16

  `U``pside`         2-year lag, then +2%/yr                                               104.59     152.37

  `shock_recovery`   flat to 2029, −3%/yr 2030–33, ramp 2034–38 to 5% above anchor, held   84.91      93.73
  ---------------------------------------------------------------------------------------------------------------

All four are analyst constructions. No NNPC or NGC citation supports any decline rate, drift rate, shock timing or recovery ramp. They are functional-form choices and are declared as such.

Policy-target arms discount the Gas Master Plan rather than extrapolating history. The sourced chain (NNPC Limited, 2026, p.3; NUPRC, 2025) runs national production from 7.45 to 10 to 12 Bcf/d across 2025, 2027 and 2030, monetisation from 0.60 to 0.75 to 0.80, a domestic share of commercialised gas of 0.43 and a power share of domestic gas of 0.481, giving a chain ratio of 2.148 against 2025.

The level is deliberately not taken from that chain, and the reason is central to this thesis. The Gas Master Plan’s 1.0 Bcf/d power figure is *demand*; the anchor is observed *delivery*. The roughly 24% gap between them is the deliverability shortfall the model exists to represent, so adopting the chain’s level would assume away the constraint under study. The chain supplies shape only, as a multiplier on the observed anchor.

An implementation factor $I$ scales the planned increment rather than the level — following the 2008 National Gas Master Plan retrospective’s own basis for reserve growth, 23.16/70 = 33% rather than 203/250 = 81% — so that the uplift is $I \times (2.148 - 1)$:

  -----------------------------------------------------------------------------------------------------------------------------------------------------
  **Arm**                   $$\mathbf{I}$$   **2030 onward**   **status**
  ------------------------- ---------------- ----------------- ----------------------------------------------------------------------------------------
  `gas_deliv_flat`          0                89.27             control arm; no midstream expansion

  `gas_deliv_gmp_partial`   0.50             140.51            central; $I$ sourced to NMDPRA Domestic Supply Obligation compliance of 50.66% in 2020

  `gas_deliv_gmp_full`      1.0              191.75            declared upper bound, not a candidate central case
  -----------------------------------------------------------------------------------------------------------------------------------------------------

Two simplifications are declared in the specification itself. The ramp is linear over 2025 to 2030 while the Gas Master Plan trajectory is front-loaded, so the linear form undershoots 2027, at 1.459 times the anchor against the chain’s 1.678 at $I = 1$. And the series is held flat after 2030, because the plan states no target beyond that year and extrapolating a five-year policy target over a further twenty-four years would be indefensible.

Three further arms — `flat_downside`, `flat_upside` and `flat_shock_recovery` — are not declared cases but level-equivalents computed automatically from their parents, with identical cumulative supply and no shape variation. They exist to separate shape effects from level effects and are used only in the gas-regime analysis.

No relationship between the deliverability series and NIRP’s gas trajectory is asserted anywhere in the specification. NIRP enters the gas layer only through new-build capital cost, a separate parameter answering a separate question.

A further arm governs whether new capacity faces the same fuel constraint as the existing fleet. `constrained` is the default and the Nigerian baseline; `exempt` represents plant sited at gas supply or operating on dedicated commercial contracts, and carries a fuel allowance sized at full-year running, which makes it an upper bound rather than a realistic alternative.

### 3.3.3 Conversion of open-cycle to combined-cycle plant

Conversion belongs with deliverability rather than with capacity, because what it changes is output per unit of a constrained fuel. Open- cycle plant may be converted to combined cycle, raising efficiency from $\eta^{g}$ to $\eta^{c}$ on the converted capacity $y_{k}$ without any additional gas. Its magnitude as a cost lever is reported and ranked in Chapter 4. It exists only in the sliced builder (§3.1.8): dispatch is slice-resolved and has no annual form, so the annual builder raises if conversion is requested.

The three tiers $\mathcal{K}$ are physical, derived from plant availability rather than assumed. Each plant’s mean performance availability factor across NERC’s 2024, 2025/Q4 and 2026/Q1 tables is banded, so tier membership reflects observed operating condition, and each tier is capped at its banded open-cycle nameplate by (3.17).

The economic parameters form a separate arm — `conv_low`, `conv_central`, `conv_high` — and are declared judgement calls rather than measured constants. Retrofit capital cost is DERIVED, not SOURCED: no public open-cycle to combined-cycle retrofit figure exists, so it is constructed as the greenfield combined-cycle minus simple-cycle increment from published turbine pricing (Gas Turbine World, 2024 \[REF: confirm edition and page for the F-class simple-cycle and combined-cycle pricing\]). The `conv_none` arm reproduces pre-conversion results exactly.

### 3.3.4 Solar

Solar generation is $G_{t,s}^{sol} = K_{t}^{sol}\phi_{s}$ (3.9), where $\phi_{s}$ is energy per megawatt in slice $s$. Its capacity credit is zero in (3.21), which is exact rather than assumed: the peak block runs from 18:00 to 22:00, after sunset.

Capital cost follows the NREL Annual Technology Baseline 2024 (NREL, 2024), Utility-Scale PV Class 5, Market case, in three arms — `solar_low`, `solar_mid`, `solar_high`, corresponding to the Baseline’s Advanced, Moderate and Conservative cases — declining from 1,456,000 USD/MW in 2025 to 603,000 by 2045 on the central `solar_low` arm.

This is a United States cost basis. IRENA reports Africa’s 2025 utility-scale installed cost at 837 USD/kW (IRENA, 2025), so the Baseline’s basis is at least 1.74 times the African benchmark. That is a lower bound, since the Baseline is expressed in 2022 dollars and IRENA in a later vintage, and a like-for-like comparison would widen the gap. An alternative arm, `solar_irena_africa`, re-anchors the Baseline’s decline shape to the IRENA 2025 level; it is not the default. Neither basis is a Nigerian figure: the Baseline overstates cost, while IRENA’s African average is dominated by South Africa, Egypt and Morocco and so plausibly understates it (Appendix B, L47). The spread is reported as a sensitivity rather than resolved.

That ratio is a comparison of price levels, both of 2025 vintage, and it is distinct from a second comparison of build-weighted blends. Triangulating the public capital envelope against NIRP 2024 computes what this model’s realised solar spend divides to per megawatt actually built over 2025 to 2045: ⟦618.41⟧ USD/kW on the Baseline’s basis, against NIRP’s own implied rate over its own build programme of ⟦916.53⟧ USD/kW, being NIRP’s 56 billion USD solar total divided by 61.1 GW to 2045 — a ratio of ⟦0.67⟧, in the opposite direction from the level comparison. The two do not contradict each other. The Baseline’s `solar_low` series declines steeply, and this model’s solar build is back-loaded onto its cheap late-horizon years, so the build-weighted blend looks inexpensive because of when the model builds on a declining price series, not because its underlying price basis is low. The 2025 price basis the model actually pays on its first megawatt is the same 1,456,000 USD/MW from which the 1.74 figure is computed.

Deployment is bounded above by a cumulative cap ${\bar{X}}_{t}$ where a deployment series is supplied (3.22a), and by the land constraint (3.22b), which is slack in every reported solution (§3.1.7). Public and service-financed additions are separate decision variables because they draw on different capital, not because they differ physically; the financing treatment is §3.4.

### 3.3.5 Hydro

Hydro capacity and annual energy are exogenous: the model dispatches hydro but never builds it, and both series are fixed data. Annual energy $E_{t}^{h}$ is allocated across seasons by $\theta_{\sigma}$ and must be fully dispatched within its season, by the equality (3.20a); generation is bounded by capacity (3.20b) and floored in every slice by a minimum-flow requirement (3.20c).

Capacity is derived from energy, at a fixed anchor capacity factor of 0.489 = 11.47 TWh / (2,678 MW × 8,760 h):

$K_{t}^{h} = 2,678 \times \frac{E_{t}^{h}}{11.47}$ (3.29)

so that capacity and energy cannot drift apart, and hydro’s capacity credit — a fraction expressing dry-season output per installed megawatt — stays invariant to the energy level. This deliberately conflates capacity growth with utilisation growth, which is a declared simplification: an arm that raises hydro energy represents more hydro output, and does not distinguish new dams from better utilisation of existing ones. Three arms govern the trajectory: `hydro_flat` (central), `hydro_committed` and `hydro_ndc`.

The seasonal split is sourced. $\theta_{\sigma}$ is derived from two independent NERC quarterly datasets, both excluding Zungeru: 2023 pre-Zungeru generation for Shiroro, Jebba, Kainji and Dadin Kowa gives quarterly average power of 1,097.6, 823.6, 1,000.2 and 1,224.0 MW, a dry-to-wet ratio of 1.118, with a Q1-against-Q3 check of 1.097. Nigerian hydro output is higher in the dry season, because the major reservoirs release stored water when run-of-river inflow is lowest.

The minimum-flow fraction is ASSUMED, at $\mu = 0.50$, and the search behind it is recorded. No Nigerian minimum-flow obligation expressed as a fraction was found, because Nigerian dam-operation sources state absolute discharge rates. The nearest documented analogue is JICA’s *Nigeria National Water Resources Master Plan* (Vol. 5, SR6, p. SR6-7, Table SR6-9) \[VINTAGE: year\], which gives assumed release rules for Shiroro Dam of 315, 100 and 50 m³/s at reservoir storage of 20%, 10% and below 10% respectively. The fraction is an analyst estimate and not a translation of that table. The plan is insensitive to it over the range tested: varying $\mu$ across 0.30, 0.50 and 0.70 moves the objective by ⟦0.150%⟧ and storage in place at 2045 by ⟦0.022%⟧, against pre-set materiality thresholds of 0.5% and 2%.

Hydro carries neither a capital charge nor an operation and maintenance cost**.** The 2,678 MW of existing hydro capacity appears in no fixed or variable operating cost term, which is an omission rather than a modelling choice; its capital is sunk and is properly excluded. Because that capacity is constant across every central arm, the operating omission is a constant offset that cancels in every comparison between arms and affects only the absolute level of cost. Neither omission is neutral where hydro capacity varies. Under `hydro_committed` and `hydro_ndc`, expansion enters the energy and capacity balances but no term in the objective, so it is costless in both capital and operation, and the benefit attributed to hydro expansion in those arms is an upper bound (Appendix B, L95). The model carries an optional fixed operating cost term for hydro, held at zero and inactive in every result reported here (§3.7.3).

### 3.3.6 Storage

Storage was structurally dominated in the annual formulation, not merely uneconomic. By (3.4) and (3.5), storage could only subtract energy while costing capital, so optimal build was identically zero at any cost. No parameter sweep could have changed that, which is why the storage-cost experiment was suspended rather than recalibrated. The distinction matters as method: sweeping a cost against a formulation in which the resource is dominated produces a null result that resembles an economic finding and is not one.

The sliced formulation removes the dominance — storage can move energy from a slack slice to a binding one within a season, by (3.18) — but without an adequacy constraint it still had no reason to build. The firm-capacity constraint (3.21) supplied the missing value channel, and storage built for the first time. Storage therefore enters the plan through capacity value rather than through energy arbitrage.

Unit capital cost is read as a per-year series from the NREL Annual Technology Baseline's Storage_low case, declining from 252,000 USD/MWh in 2025 to 146,000 by 2045 (Appendix C, L90). Against NIRP's own implied storage rate of 272.73 USD/kWh — 3 billion USD divided by 11 GWh, NIRP 2024 Tables 6 and 10 — the 2025 level agrees to within 7.6%, with no rescaling applied to produce the agreement. It corroborates the unit price only, and not the programme size, on which the two plans disagree sharply.

**Capacity credit is a swept arm:** It was initially $min(1,\Delta/h_{peak}) = 1.0$ at four-hour duration against a four-hour peak block, which placed storage above gas as a firm resource. It is now `storage_cc_case` $\in \{ 0.50,\ 0.80,\ 1.00\}$, ASSUMED, with the high arm reproducing the former value as a declared upper bound. The reasoning is that a four-hour battery covers a four-hour block in principle, but effective load-carrying capability declines steeply with penetration (Keane et al., 2011), and the deployment levels this model reaches are not a low-penetration regime. A flat credit admits no declining capability and therefore no interior optimum.

Under a fixed capacity credit, and with gas at its floor, storage absorbs the entire firm-capacity residual: its quantity is determined by the constraint rather than chosen. Storage composition — that it is displaced by gas, and at what ratio — is a model result; storage magnitude is not.

Three independent measurements bear this out. Storage in place at 2054 is ⟦56,289.9⟧ MWh at every social discount rate from 4 to 12%, with a spread of ⟦0.0002⟧ MWh (§3.1.2); it is identical under all three capital timing conventions (§3.4.2); and it moves by ⟦0.022%⟧ across the minimum-flow sweep (§3.3.5). A quantity that does not respond to the discount rate, to the cost of capital, or to a physical constraint on another resource is not being chosen on price. The contrast with the regression configuration is instructive: there, where the adequacy constraint does not bind in the same years, restoring the declining storage price series raised storage at 2054 by ⟦70.9%⟧, from ⟦43,540⟧ to ⟦74,435⟧ MWh. Where firm capacity binds, storage is set by the constraint; where it does not, storage is set by its price.

`storage_duration_hours` is 4.0 and is reached through a default rather than a required key — an inconsistency with the surrounding discipline, and load-bearing because it sets the capacity credit’s ceiling.

### 3.3.7 Self-generation as a priced backstop

Self-generation $g_{t,s}$ meets grid demand the modelled system cannot, at a priced cost $c^{self}$, entering the energy balance (3.14) scaled by a busbar factor $\beta$. It is deliberately unbounded in the absence of a policy constraint: it is the model’s backstop, and constraining it would make infeasibility, rather than cost, the signal of an inadequate plan.

Unserved energy $u_{t,s}$ is fixed at zero whenever a genset case is priced (§3.1.4). The grid-service shortfall is therefore carried entirely by $g_{t,s}$, and a reported unserved-energy total of zero is never quoted as evidence that demand is met.

Where a grid-service obligation is imposed, (3.23a) caps self-generation and unserved energy jointly, year by year. Because self-generation emissions fall outside the default grid-only cap boundary, this backstop is also the channel through which abatement can be displaced across that boundary, which §3.5.6 examines.

### 3.3.8 Parameters reached through defaults rather than required keys

The required-key discipline (§3.6) exists so that no parameter enters a run without an explicit choice recorded in the output. Five parameters in the supply layer bypass it, entering through a `scenario.get(key, default)` fallback:

  --------------------------------------------------------------------------
  **Parameter**                **Default**             **Where**
  ---------------------------- ----------------------- ---------------------
  `new_gas_deliverability`     `"constrained"`         both builders

  `conversion_available`       `False`                 sliced builder

  `hydro_season_factor`        dry 1.045 / wet 0.956   sliced builder

  `hydro_min_flow_fraction`    0.50                    sliced builder

  `storage_duration_hours`     4.0                     `load_scenario`
  --------------------------------------------------------------------------

Each is set on every standard path, so none is exercised in the reported results. They are recorded because the same pattern at `within_block_peaking_factor` (§3.1.7) produced a live parameter with no provenance and no signature in any output, and because a default that is correct today becomes a silent assumption the moment a new runner omits the key. The `hydro_season_factor` default is also one of four copies of the same two numbers, alongside a sibling solar shape factor; the duplication is declared rather than resolved. A systematic test of this property across the whole model, rather than in this layer alone, is reported in §3.6.10.

## 3.4 Financing

### 3.4.1 The financing layer

The preceding sections specify a physical investment problem: a fleet with heat rates, availability factors and retirement dates; a fuel supply with a deliverability limit; an adequacy requirement met by credited firm capacity. This section adds the terms that determine how that investment is paid for, and it is written in the same terms. A capital recovery factor is an annuity. A concessional envelope is a resource limit on a scarce input. A bankability test is a coverage ratio applied to a revenue stream. Each enters the programme as a constraint or as an objective coefficient on the same physical decision variables, and none of them requires the model to leave the linear class.

The layer consists of three objects, set out in §3.4.2 to §3.4.5: a per-vintage split of capital between a concessional and a commercial tranche, with the split as a decision variable; a limit on the total concessional volume available over the horizon; and a coverage-ratio test that privately financed capacity must pass before it can be built.

1.  **Financing rates.** Each investment is annualised over its financing tenor through the capital recovery factor

$\text{CRF}(r,L) = \frac{r(1 + r)^{L}}{(1 + r)^{L} - 1},\quad\quad\text{CRF}: = \frac{1}{L}\ \text{ if }r \leq 0,$ (3.30)

the standard annuity formulation for energy-technology investment appraisal (Short et al., 1995, pp. 19, 51). The floor applied at non-positive deflated rates is declared rather than silent: it biases against concessionality, and it makes the model insensitive to variation in the concessional rate below the rate of inflation.

The factor is applied at a concessional rate $r_{c}$ or a commercial rate $r_{m}$. Nominal rates are deflated to real terms once, by the exact Fisher relation $(1 + i) = (1 + r)(1 + \pi)$ (Fisher, 1930), with United States inflation $\pi = 2.0\%$. At the central configuration the concessional rate is 2.12% nominal, or 0.1176% real, over a 25-year tenor, reflecting IDA21 terms (World Bank, 2026); the commercial rate is 11.0% nominal, or 8.824% real, over 10 years. Gas and conversion capital are commercial-only.

2.  What financing does to present cost. The present value of an investment’s capital payments, relative to its overnight cost $I$, is

$\frac{PV}{I} = \text{CRF}(r_{f},L) \cdot \sum_{\tau = 1}^{L}(1 + r_{s})^{- \tau},$ (3.31)

the product of an annuity factor at the financing rate and an annuity factor at the social discount rate, which equals unity when $r_{f} = r_{s}$. Financing raises the present cost of an asset above its overnight cost only where the financing rate exceeds the social rate, and lowers it where the financing rate falls below. At the central parameters a solar vintage recovering over its full concessional window carries a present value of 0.434 of its overnight cost, an effective concessional subsidy of 57%, while commercial capital carries a premium of 3.7%, because 8.824% real lies close to $r_{s}$. This wedge is the mechanism through which concessional finance, and the service-based channel that draws on it, lower modelled system cost.

3.  The accounting perspective of the objective. The quantity minimised is the discounted financial cost borne by the Nigerian power system, not the social resource cost of the capacity it builds. The distinction matters because concessional finance does not make a solar plant cheaper to construct; it transfers part of that cost to the concessional lender, and (3.31) prices the transfer rather than the resource. The same principle governs the service subsidy defined in §3.4.4, which is excluded from the objective: it is a transfer between parties inside the system boundary and nets to zero there, whereas the concessional wedge crosses that boundary and does not. Both treatments follow from one accounting perspective, and results stated as cost savings are therefore savings to Nigeria, a part of which are transfers rather than reductions in resources consumed.

### 3.4.2 The per-tranche capital charge

The formulation set out in this section is not adapted from an existing capacity-expansion model in sub-Saharan Africa. Endogenous financing in such models is ordinarily represented by a single weighted average cost of capital, applied uniformly and exogenously (§2.5.3.). The per-vintage split between two tranches at different rates and tenors, with the split as a decision variable constrained by a concessional envelope, is novel to this thesis. The coverage-ratio test conditions privately financed capacity rather than the tranche split, and is set out in §3.4.5.

- Splitting each vintage’s capital. For each buildable channel $c \in \{$public solar, service-based solar, gas, storage$\}$ and each vintage $v$, the capital raised is split between a concessional and a commercial tranche, both non-negative decision variables:

$V_{c,v}^{conc} + V_{c,v}^{comm}\mspace{6mu} = \mspace{6mu} I_{c,v}\, a_{c,v}$ (3.32)

`[code: optimize_model_sliced.py, build_model_sliced: *_split_identity]`

where $I_{c,v}$ is unit capital cost and $a_{c,v}$ the capacity added in that vintage. The split is the endogenous financing decision: nothing else in the model determines it, and it is the variable on which the concessional envelope of §3.4.3 acts. Replacement capital for assets whose technical life ends inside the horizon is charged through the same identity and the same windows, taking its rebuild year as its vintage.

- Rates, tenors and the recovery window. Each tranche is annualised by (3.30) at its own rate. Taking the concessional arm at its central values illustrates why the rate and the tenor cannot be varied independently: IDA21 terms give $\text{CRF}(0.001176,\, 25) = 0.040616$, against $\text{CRF}(0.007843,\, 30) = 0.037539$ under the superseded IDA20 terms of 2.80% over 30 years (Appendix C, L53). Replacing IDA20 with IDA21 therefore raises the concessional capital charge per dollar of concessional volume by 8.2%, rather than lowering it as a rate-only reading would suggest, because shortening the recovery window from 30 to 25 years raises the annuity factor by more than the fall in the rate lowers it.

The recovery window is technology-specific, being the shorter of the financing tenor and the asset’s technical life:

$L_{c}^{eff} = min\left( L,\ \text{life}_{c} \right)$ (3.33)

`[code: optimize_model_sliced.py, build_model_sliced: _L_eff_comm, _L_eff_conc]` following the standard treatment in investment appraisal, in which an asset cannot be amortised beyond its service life.

- Discounting the instalments inside the horizon. Each vintage’s annuity is weighted by the sum of discount factors over its own recovery window. The first instalment falls in the year following the vintage, as an annuity computed in arrears requires, and the window is truncated at the horizon end:

$W_{c,v}\mspace{6mu} = \mspace{6mu}\sum_{t = v + 1}^{min(v + L_{c}^{eff},\ T_{last} + 1)}\delta_{t}$ (3.34)

`[code: optimize_model_sliced.py, build_model_sliced: _df_window_conc, _df_window_comm]` giving the capital charge

$C^{cap}\mspace{6mu} = \mspace{6mu}\sum_{c}^{}{\sum_{v}^{}\left( \text{CRF}_{c}^{conc}\, W_{c,v}^{conc}\, V_{c,v}^{conc}\mspace{6mu} + \mspace{6mu}\text{CRF}_{c}^{comm}\, W_{c,v}^{comm}\, V_{c,v}^{comm} \right)}$ (3.35)

`[code: optimize_model_sliced.py, build_model_sliced: cost_*_capital_charge_npv]` Equation (3.34) adapts the conventional annuity-weighting treatment by truncating each vintage’s window at the horizon end rather than carrying it to term; (3.35) is specific to this formulation, having no counterpart in a model with a single capital cost.

Both capital recovery factors and every weight $W$ are precomputed constants rather than Pyomo expressions, so every objective coefficient is exogenous data and the programme remains linear. This is what makes an endogenous capital structure tractable within a linear programme: the financing decision is a volume split at fixed prices, not an optimisation over rates.

- The timing convention, and a correction to it. Equations (3.30) and (3.34) must agree on when the first instalment falls. A capital recovery factor converts a principal at the start of a period into $L$ payments at the end of each subsequent period; placing those payments one year earlier, while leaving the factor unchanged, charges the same annuity over a window shifted toward the present.

The correction was measured rather than asserted. Holding the model otherwise fixed and varying only the summation window, the objective falls by ⟦3.39%⟧, from ⟦25,862,214,166.78⟧ to ⟦24,986,620,949.18⟧ USD, while the physical plan is unchanged. Because the correction is a uniform deflation of every capital charge, and because the plan does not move, the resulting cost difference must equal $\left( 1 - (1 + r_{s})^{- 1} \right)C^{cap}$ exactly. Against a capital charge of ⟦11,820,508,401⟧ USD under the superseded convention this predicts ⟦875,593,215⟧ USD, and the measured difference is ⟦875,593,218⟧ USD. The defect was therefore an error in the level of cost and not in the composition of the plan.

An alternative correction, retaining the window and dividing the annuity by $(1 + r_{f})$ in the manner of an annuity due, was also measured and was rejected. It deflates the concessional and commercial tranches by different factors and so alters the tranche mix: public solar capital falls to exactly ⟦1/1.08⟧ of its superseded value under the arrears convention, but to ⟦0.92158⟧ under the annuity-due reading, which is neither financing rate’s factor.

- What the truncation in (3.34) does. Vintages built late in the horizon carry only those annuity years falling inside it. The truncation is not confined to the buffer years: with a 25-year concessional tenor against a 30-year solar life, concessional solar windows truncate from vintage 2031, and with a 15-year storage life, concessional storage windows truncate from 2041. Commercial windows, at a 10-year tenor, truncate only from 2045.

Such truncation is often described as standing in for an end-of-horizon salvage value. On this model that description holds in cost and not in the capacity in place by 2045. Solving at terminal years from 2045 through 2054 and at storage lives of 15 and 30 years, cumulative capacity at 2045 is identical in all eight cells (§3.1.1), while cost is not: at a 2054 terminal year, the difference between a 15-year and a 30-year storage life is ⟦657,277⟧ USD, which reproduces the model’s own reported replacement capital charge of ⟦657,279⟧ USD. The in-horizon replacement is charged, and it does not change what is in place by 2045. Truncation is therefore an approximation in the level of the objective, not a distortion of the capacity in place over the reported window; the timing of additions within the window does move with the horizon end (§3.1.1).

- The capital charge in the central solution. $C^{cap}$ is ⟦9,329,045,793⟧ USD, or ⟦40.0%⟧ of the objective, with conversion capital a further ⟦10.1%⟧. It decomposes into ⟦3,061,123,411⟧ USD of public solar, ⟦2,382,982,003⟧ of gas, ⟦1,527,593,314⟧ of storage and ⟦657,279⟧ of replacement, with service-based solar at zero under the central financing regime. Capital is thus a little over half of total system cost once conversion is included, which is why a timing convention in its annuitisation reaches the headline figure at all.

- A property of the declared terms. Decomposing the gap between the concessional and the commercial capital charge by a Shapley attribution over rate and tenor (Shapley, 1953) assigns ⟦50.1%⟧ of the concessional advantage to the longer tenor and ⟦49.9%⟧ to the lower rate. This is arithmetic on $\text{CRF}(0.001176,\, 25)$ against $\text{CRF}(0.08824,\, 10)$, computed in closed form from the declared terms. It is not a decomposition of any solved quantity, and no result file stands behind it. It is reported because the near-equality is informative about where the value of concessionality lies, and it is identified as closed-form arithmetic so that it is not read as a model output.

### 3.4.3 The concessional envelope

Concessional capital is cheaper than commercial capital at every configuration, so $\text{CRF}(r_{c}) < \text{CRF}(r_{m})$ always. Without a binding constraint the optimiser funds the whole of capacity concessionally and the endogenous capital structure becomes structurally impossible. The envelope is therefore not a detail of the specification; it is what makes the financing split a decision.

$\sum_{c \in \mathcal{C}^{elig}}^{}\ \sum_{v}^{}V_{c,v}^{conc}\mspace{6mu} \leq \mspace{6mu} B$ (3.36)

`[code: optimize_model_sliced.py, build_model_sliced: f1_concessional_envelope_constraint]`

Concessional envelopes of this kind are the defining instrument of blended development finance, in which a limited quantity of below-market capital is deployed to mobilise commercial capital alongside it. Representing it as a resource limit rather than as a rate concession is what allows the model to ask where a scarce concessional dollar is best placed.

Two design choices are declared. Eligibility is restricted to solar and storage, with gas commercial-only, implemented by fixing $V_{gas,v}^{conc} = 0$: DARES is a distributed-renewables facility, and claiming concessional credit for a combined-cycle gas turbine would not be defensible. And the envelope is an undiscounted commitment stock rather than a present value — the quantity a lender commits, which is what a facility size is.

The central envelope is the DARES facility’s domestic deployable portion. At 750 million USD against system capital in the billions, the envelope binds early and the marginal dollar is commercial for most of the horizon, which is the empirically interesting case. Counterfactual arms at five and ten times the facility size establish what scale of concessional finance would change the pathway rather than merely subsidise it.

### 3.4.4 The public capital budget

The envelope of §3.4.3 limits how much capital may be *concessional*. A second and distinct constraint limits how much may be *public*, regardless of the terms on which it is raised:

$\sum_{t}^{}\delta_{t}\left( I_{t}^{pub}\, x_{t}^{pub} + I_{t}^{st}\, a_{t}^{st} + \zeta_{t} \right)\mspace{6mu} \leq \mspace{6mu} B^{pub}$ (3.37)

`[code: optimize_model_sliced.py, build_model_sliced: public_budget_constraint]`

The two constraints are separate objects and are often conflated. The envelope is an undiscounted stock of concessional commitment; the budget is a present-value limit on what the public balance sheet spends. A concessional dollar deployed to public solar consumes both; a commercial dollar deployed to public solar consumes the budget only; and capital raised through the service-based channel consumes neither, which is the whole of that channel’s mechanism (§3.4.6). These relationships are asserted directly against the constraint’s coefficients rather than inferred from solutions, and they hold.

$B^{pub}$ is expressed as a fraction of a reference quantity $B^{*}$, and `capital_case` supplies that fraction: `tight` 0.50, `moderate` 0.85, `adequacy` 1.00, `expansion` 1.20, and `unconstrained`, which imposes no budget. The constraint’s dual is reported as a present-value cost per present-value budget dollar, and Chapter 4 uses it to identify where the public balance sheet, rather than the physical system, is what limits the plan.

$B^{*}$ is a fixed constant of ⟦10,273,343,610⟧ USD, not a quantity the model recomputes. It is the realised public spend, in present value, at the first point where the public-budget dual reaches zero on a capital-frontier sweep of the traditional financing regime, solved under no emissions cap. No arm varies it. Two properties of that construction condition every result expressed in its units, and both are stated here rather than left to be inferred.

The tier names are grid labels, not economic characterisations. In units of the measured unconstrained requirement, `tight` is ⟦0.931⟧, `moderate` ⟦1.582⟧, `adequacy` ⟦1.862⟧ and `expansion` ⟦2.234⟧. The tier named `moderate` therefore funds more than half again the requirement rather than sitting at it, and the tier named `adequacy` funds nearly twice it. Where a result turns on whether the budget binds, the dual is the quantity to read, not the tier’s name. The tier named `adequacy` nonetheless constrains the plan where revenue is short: its budget dual is ⟦0⟧ at full collection, ⟦0.3135⟧ at a ⟦60⟧ per cent collection rate and ⟦0.8294⟧ at ⟦35⟧ per cent.

Every result in this thesis concerning the public capital requirement, and every comparison between financing regimes, rests on this constraint rather than on the envelope of §3.4.3.

### 3.4.5 The bankability constraint

Privately financed capacity is not built merely because it is economic; it is built when its revenue covers its debt service with a margin. For each vintage $t$, service-financed solar additions are limited by the revenue they can raise:

$I_{t}\, x_{t}^{eaas}\mspace{6mu} \leq \mspace{6mu}\frac{\tau\,\phi\, F_{t}}{m}\, x_{t}^{eaas}\mspace{6mu} + \mspace{6mu}\zeta_{t}$ (3.38)

`[code: optimize_model_sliced.py, build_model_sliced: eaas_bankability_rule]` where $\tau$ is the service tariff, $\phi$ solar energy per megawatt, $m$ the required coverage margin, $\zeta_{t}$ the public subsidy, and

$F_{t}\mspace{6mu} = \mspace{6mu}\sum_{j = 0}^{L^{eff} - 1}(1 + r^{priv})^{- j}$ (3.39)

the remaining-revenue annuity factor in year-$t$ terms. The subsidy is separately capped at the financing gap `[code: optimize_model_sliced.py, build_model_sliced: subsidy_limit_rule]`.

Both sides of (3.38) are linear in $x_{t}^{eaas}$ apart from the subsidy, so the constraint is a per-megawatt test rather than a test of project size: where the unit revenue covers the unit cost it never binds, and where it does not, deployment is limited to what the subsidy will carry. Real bankability assessments are project-specific and non-linear; this is the linear-programming representation of the same test.

The coverage margin is an arm, not a literal. `dscr_case` takes 1.10 (regression), 1.20, 1.30 (central) or 1.40. It is a revenue divisor in the bankability and subsidy-limit constraints and never enters the objective: applying it to objective-side capital would double-count against the bankability constraint’s own discounting, since a coverage cushion and a cost of capital are distinct objects. The model carries that rationale as a guard comment in both builders. One consequence follows and is declared in §3.6.10: because $m$ does not enter the objective, no verification test built on the objective can confirm that it is doing what the specification says, and the only quantity that responds to it is one the model does not identify.

Six runners bypassed that arm and have been corrected. Until commit 4ec43b4, the service-financed configurations in six runners forced \$m\$ to 1.10, or 1.05 under the conditional-NDC arms, rather than taking the dscr_case value. Because \$m\$ enters only the bankability and subsidy-limit constraints, the public arms carried the literal inertly while the service-financed arms took a laxer version of a test only they face, biasing toward finding privately financed capacity bankable where the correct cushion would not. The override was removed and the affected results re-measured against a saved copy of the pre-correction outputs. The cost effect was nil or below ⟦1%⟧ in most cells and material only where a binding cap meets a tight capital envelope. Plans moved where costs did not, but the quantities that moved are not identified by the model and are not reported as results (Appendix C, L34 and L43).

$m$ is a debt-sizing ratio, not a lock-up or default covenant. Yescombe (2014, §12.3.1, p.322) calls the numerator it divides *net operating cash flow* — not “CFADS”, a term absent from that source — and treats the sizing annual debt-service cover ratio as distinct from, and higher than, the lock-up and default thresholds applied off the same figure after financial close, with a worked example of 1.35 sizing, 1.2 lock-up and 1.1 default (§14.4.3, p.376; §14.12, p.397). This model carries only the sizing role and has no separate lock-up or default mechanism. Appendix A, row 13, records the arm's per-value sourcing, including that the source gives no renewable-specific figure and that an earlier claim about development finance institutions accepting lower cover ratios is withdrawn.

The subsidy has no objective cost, because it is a transfer inside the system boundary rather than a resource cost (§3.4.1). It draws on the public capital budget instead. This has a consequence that must be declared wherever service-financed results are reported: whenever the public budget is slack, the optimiser subsidises to the financing gap at zero objective cost, so the bankability constraint never binds and the service-financed arm becomes cost-flat in that region.

The capital-requirement comparison that follows from this rests on the public budget constraint of §3.4.4 and is reported in Chapter 4.

### 3.4.6 What the service-based channel represents in this model

Energy-as-a-Service is a contracting structure in which a provider finances, installs and operates generation assets and the customer pays for delivered service. In this model its causal content is precise and narrower than the commercial concept: service-financed solar capital does not consume the public capital budget, and is instead subject to the bankability test of §3.4.5. The superscript $eaas$ on the decision variables of §3.1.4 and in the constraint names denotes this channel.

That is the whole mechanism. Service-financed and public solar are priced at the same unit capital cost, being the same physical assets, and the private cost of capital is represented through the bankability constraint’s discounting rather than as a capital-cost premium.

Two boundaries are stated plainly. The service-payment structure is not in the optimisation: a module implementing contract mechanics — capital recovery factors, tenor, service payments, and on- or off-balance-sheet treatment — exists in the codebase but is never called by the linear programme (Appendix B). What the programme represents is budget decoupling plus a bankability test.

The mechanism is therefore generic private capital mobilisation. Nothing in the formulation is specific to a service contract as opposed to any other structure that moves solar capital off the public balance sheet and subjects it to a revenue-adequacy test. This is a legitimate object of study and Chapter 4’s results follow from it, but the claim is scoped to what is modelled.

### Coherence of the financing layer: two resolved, two open

Four items, each consequential enough to state in the methodology rather than only in the register.

1.  The bankability test’s commercial basis — resolved. The test now prices at the objective's own commercial basis: 11% nominal, 8.824% real, over a 10-year tenor truncated at the horizon, in place of a blended 13.24% whose dominant leg was flagged in its own source comments as naira-basis. The remaining-revenue factor \$F_0\$ falls from 8.349709 to 7.038499, a reduction of 15.7%, so service-financed capacity is harder to self-finance than every pre-correction result showed. The correction runs against this thesis's own conclusion. The legacy basis is retained as bank_tenor_legacy (Appendix C, L17 and L52).

2.  The concessional rate — resolved. The objective and the bankability blend both price concessional capital at IDA21 terms, 2.12% over 25 years. DARES is modelled at current terms rather than at its IDA20 origination vintage, because the model evaluates forward pathways rather than repricing an existing commitment (Appendix C, L53).

3.  The blend’s commercial leg — open. The blended-finance scenario constructor still carries a commercial rate of 18% on a naira basis. After the first correction above it reaches exactly one live consumer, the tariff sweep of §3.4.5, which is pinned to `bank_tenor_legacy` so that its analytics and its solves stay on a single basis. Every result from that sweep is therefore still priced through a basis-suspect rate, and correcting it requires its own re-run. (Appendix B, L17.)

4.  The tariff threshold is a scalar, not a surface — open. The bankability test is year-specific, but the reported break-even tariff is computed once, from 2025 capital cost over a full-horizon annuity. Late vintages are therefore judged unbankable by a test the model does not impose on them.

Consequently. The objective and the bankability test now price the same capital on the same basis for every path except the tariff sweep. Results turning on the level of that sweep’s threshold remain provisional. Results turning on budget decoupling — the capital-requirement comparison in Chapter 4 — were never affected, since they run through the public budget constraint and the capital charge.

## 3.5 Policy representation

Policy enters the model through three instruments, and the relationship between them is the substance of this section. An emissions cap (3.23b) bounds emissions inside a stated boundary. A grid service obligation (3.23a) bounds the quantity of demand met off-grid. A carbon price (3.13) is available but is zero in the central case.

The first two are not independent. The cap is drawn at the grid boundary, so it can in principle be satisfied by moving emissions across that boundary rather than abating them. The obligation is the only instrument that prevents this. The architecture is deliberate and its weak point is declared rather than concealed.

### 3.5.1 Three emissions quantities

A single switch deciding whether self-generation CO₂ counts would conflate two questions — what the national commitment covers, and what is physically emitted — which attach to different instruments. The model carries three quantities:

  ------------------------------------------------------------------------------------------------
  **Quantity**            **Contents**                **attaches to**
  ----------------------- --------------------------- --------------------------------------------
  $$E_{t}^{grid}$$        gas-fired grid generation   the cap (3.23b)

  $$E_{t}^{self}$$        private self-generation     reported; the out-of-boundary measure

  $$E_{t}$$               both — physical emissions   the carbon price (3.13), and all reporting
  ------------------------------------------------------------------------------------------------

`[code: optimize_model_sliced.py: emissions_grid_only, emissions_selfgen, emissions_by_year]`

Only $E_{t}^{grid}$ is capped. $E_{t}$ is what the country actually emits, and is what every emissions figure in Chapter 4 reports unless stated otherwise.

### 3.5.2 Constructing the cap

The cap is not a published series — NDC 3.0 states no power-sector emissions ceiling — so it must be constructed. The construction is a four-step chain.

#### The chain

${\bar{E}}_{y}\mspace{6mu} = \mspace{6mu} E^{base}\mspace{6mu} - \mspace{6mu}\underset{\text{NDC 3.0}}{\underbrace{A_{y}}} \times \underset{\text{NDC 3.0}}{\underbrace{\sigma_{y}^{energy}}} \times \underset{\text{BTR1 / NID}}{\underbrace{s^{power}}}$ (3.40)

  -----------------------------------------------------------------------------------------------------
  **Step**                                      **Quantity**   **2030**   **2035**   **Source**
  --------------------------------------------- -------------- ---------- ---------- ------------------
  economy-wide abatement $A_{y}$                MtCO₂e         168.2      184.9      NDC 3.0, p.14

  × energy-sector share $\sigma_{y}^{energy}$   —              22.0%      26.3%      NDC 3.0, p.15

  = energy-sector abatement                     MtCO₂e         37.00      48.63      derived

  × power share of energy $s^{power}$           —              8.3963%    8.3963%    BTR1 / NID, 2018

  = power-sector abatement                      MtCO₂e         3.11       4.08       derived

  $E^{base}$ (model, `conv_none`)               MtCO₂/yr       ⟦15.745⟧   ⟦15.745⟧   model

  = cap, unconditional                          MtCO₂/yr       ⟦12.638⟧   ⟦11.662⟧   
  -----------------------------------------------------------------------------------------------------

`[code: 00_build_emissions_cap.py]`

$E^{base}$ is read from a no-conversion run on the sliced builder. The equivalence across builders was tested rather than assumed: regenerating the baseline under the same arm on each builder reproduces $E^{base}$ to a relative ⟦$4.6 \times 10^{- 9}$⟧ and every cap value in the resulting trajectory to within a relative ⟦$8 \times 10^{- 9}$⟧, so the cap is builder-invariant in value. Between the anchor years the cap is interpolated linearly; before 2030 it runs from $E^{base}$ at 2025.

#### The power share is inventory-derived

$s^{power}\mspace{6mu} = \mspace{6mu}\frac{\text{1.A.1.a Public electricity and heat production, 2018}}{\text{Sector 1 Energy, 2018}}\mspace{6mu} = \mspace{6mu}\frac{16,234}{193,347}\mspace{6mu} = \mspace{6mu} 8.3963\%$ (3.41)

The numerator is Nigeria’s *2024 National Inventory Document*, Table 3.9, p.36; the denominator the *First Biennial Transparency Report*, Table 2.10, p.33. Both are 2018 figures from the same inventory vintage, prepared two days apart in December 2024, on AR5 global warming potentials and the 2006 IPCC Guidelines with the 2019 Refinement. The nesting arithmetic closes exactly: 16,234 for public electricity and heat, plus 990 for refining, plus 12,640 for solid-fuel manufacture, gives 29,864, which is the Transparency Report’s 1.A.1 energy industries.

The scope of numerator and denominator matches NDC 3.0’s own share, and that match is established from the document’s internal structure rather than from a stated definition. NDC 3.0’s p.15 figures are qualified “Energy sector (including buildings and transport)” — 22.0% of the 2030 reduction and 26.3% of the 2035 one. That phrase is the only place the document says what its Energy category contains. The scope match therefore follows from a structural check: p.2 gives a six-category mitigation-potential breakdown, with the energy sector including fugitive emissions and solid-fuel transformation at 31.2 MtCO₂e and the fuel-combustion sector at 103.4 MtCO₂e, alongside industrial processes at 14.0, agriculture at 2.1, land use at 347.9 and waste at 12.5; while p.15 and Figure 4 report only five categories, summing to 100% of the same 511.1 Mt total. Figure 4’s Energy bar is 134.6 Mt, which is 31.2 plus 103.4 exactly, and 134.6/511.1 is 26.3%, matching the stated 2035 share to the digit. The five-sector Energy is therefore the six-line list’s energy and fuel-combustion sectors combined — fugitive emissions plus solid-fuel transformation plus all fuel combustion — which is exactly the inventory’s Sector 1 Energy. The two documents use “energy sector” in two different senses, and the ambiguity is real but resolved by the document’s own arithmetic rather than asserted.

#### A superseded construction divided the model's own \$E\^{base}\$ by an assumed 217.0 MtCO₂e energy-sector total, justified by a claim about NDC 3.0 that appears nowhere in the document. The quantity was malformed: dividing a model output by an inventory estimate expresses no share, and it yielded a plausible 7.24% only because the denominator was broader in scope than the numerator. Inventory over inventory, as in (3.41), is the form that behaves (Appendix B, L60).

#### Two alternatives tested and rejected

1.  NDC 3.0’s Table 2 electricity measures. The document states electricity-generation mitigation at measure level (p.16): 12.220 MtCO₂e for a 52% renewable share across a mix including 17,000 MW of gas, and 5.018 MtCO₂e for replacing single-cycle steam turbines with combined cycle. Those are the document’s most directly relevant numbers, and the model builds exactly those levers. Subtracting them from $E^{base}$ nonetheless drives the cap negative — by 49% on the full 23.467 Mt, taking all three electricity-generation measures on that page, and by 9.48% on the narrowest reading of 17.238 Mt. The reason is a category difference: Table 2 reports avoided emissions against a counterfactual capacity mix, whereas $E^{base}$ is baseline emissions. The two are not subtractable.

2.  The Transparency Report’s Energy total as the base. That report gives 2018 total net emissions of 614.17 MtCO₂e, which NDC 3.0 itself cites (p.13). But NDC 3.0 states its targets against 573.5 MtCO₂e (p.14), a figure appearing nowhere in either inventory document, with no stated reconciliation between the two. Equation (3.40) avoids the problem because it uses the absolute abatement figures, 168.2 and 184.9 Mt, so the base they were derived from never enters the arithmetic.

#### What remains an assumption, and what remains unsourced

- The assumption: equal proportional effort. Equation (3.41) allocates abatement to power in proportion to its share of sector emissions. NDC 3.0 states no power-sector share of its target, so some allocation rule is unavoidable; this is the neutral one. It is testable, and the plausible alternative runs one way: electricity is generally the cheapest sector to decarbonise, so a least-cost allocation would assign power more than proportional effort and tighten the cap further. The rule is therefore conservative.

- The unsourced anchor: the conditional arm only**.** NDC 3.0 states that Nigeria “maintains its 20% unconditional objective and views the conditional reduction at 80% of the 29% and 32%” (p.14). Two readings are possible: 20% of the 2018 base, giving 114.7 Mt and multipliers of 1.466 and 1.612, or 20% of the target itself, giving exactly 5.0 in both years. The second is rejected because a multiplier fixed at 5.0 regardless of the underlying targets measures nothing about the NDC. The first is adopted, and it is the one place the 573.5 Mt base re-enters, so the conditional arm carries one unsourced anchor while the unconditional caps are sourced end to end.

#### Everything after 2035

NDC 3.0 gives no target beyond 2035, so every cap value from 2036 onward — ten of the twenty-one reported years, plus the 2046 to 2054 buffer — is a second construction on top of the first.

The default holds the 2035 cap flat. Flat is not neutral: it assumes ambition stops accumulating in 2035, which understates what a country committed to net zero by 2060 would need across the back two-thirds of the horizon. The `ndc3_conditional_decline` arm therefore adds a declared alternative rather than replacing the default — a linear decline from the 2035 anchor toward zero at 2060, Nigeria’s own stated net-zero year (NDC 3.0, pp. vii, viii, 14), so that the endpoint is policy-anchored. The straight line between those points is this project’s construction; no document specifies the path’s shape. It is reported as a sensitivity for that reason: it establishes that a range exists, not that this range is correct (Appendix B, L24).

Buffer-year caps are held flat at the terminal in-window value in every arm but one, and are never reported. Releasing them to an effectively infinite value, on the reasoning that a binding cap outside the reporting window would distort investment inside it, held only until self-generation was priced: with a backstop available an uncapped buffer became an arbitrage sink, and the optimizer routed essentially all self-generation into the uncapped years while paying the shedding penalty in the capped ones. Buffer years must therefore be neither cheaper nor freer than the last reported year.

The declining arm is the one exception. `ndc3_conditional_decline` does not hold its buffer flat: its cap continues the same linear decline through 2046 to 2054, falling from ⟦5.50⟧ MtCO₂ at 2045 to ⟦2.20⟧ at 2054, because a flat buffer under a declining in-window cap would reopen the arbitrage sink, and a flat hold at the 2045 value would contradict the trajectory’s own premise. Its buffer years are therefore tighter than its last reported year rather than equal to it, so this arm’s buffer is not comparable with any other arm’s.

#### The cap is insensitive to the demand arm

The construction is independent of which demand arm is in force, for the reason set out in §3.2.6: baseline gas reaches its deliverability ceiling by 2026, so the baseline emissions level is set by gas supply rather than by demand, and the caps rebuild to identical values from either arm.

### 3.5.3 The cap’s boundary

The cap is derived as the gas-power share of Nigeria’s energy-sector abatement target, which is what the NDC commits for this sector. Private self-generation sits inside the national inventory but outside that slice.

Capping total emissions against a budget sized for the gas-power slice would be an accounting error. It would also charge diesel at a shadow price that is an artefact of how narrowly the slice was drawn rather than of any policy.

This is the right accounting choice and it creates an obvious risk: the model could satisfy the NDC by pushing load onto self-generation, meeting the target on paper while national emissions rise. Self-generation emission factors exceed the grid’s, so the substitution is always emissions-negative in reality.

### 3.5.4 The grid service obligation

The instrument that disciplines self-generation is a quantity constraint, not a price. For each year $t$:

$\sum_{s}^{}\left( g_{t,s} + u_{t,s} \right)\mspace{6mu} \leq \mspace{6mu}\varepsilon_{t}\, D_{t}$ (3.23a, repeated)

It was previously a reliability constraint bounding $u_{t,s}$ alone, and both the form and the name were wrong once a backstop existed.

Bounding unserved energy alone stopped working because, with a cheaper backstop priced, $u_{t,s}$ is zero by construction (§3.1.4) and a constraint written on it can never bind. It would have remained in the model, silently inert, while several sensitivity runners depended on it.

Calling it reliability also overstated the term. Reliability in power-system planning is adequacy — loss-of-load expectation, expected energy not served. A household running its own generator was not served by the grid, but no capacity was short. The constraint is a service obligation: a policy commitment that at least a stated fraction of demand is met by the grid. That is a real instrument, and electrification targets have exactly this shape.

The renaming matters beyond terminology. As a service obligation this is the model’s only direct lever on accessibility: it constrains self-generation by quantity, independent of price, and therefore does work the cost objective cannot.

It is enforced year by year, on measured grounds. A horizon-total form exists but is not used. Under a monotonically rising deficit a horizon-total bound is satisfied by early-year slack: it concentrates the shortfall in the final years and understates what the service standard does to the plan. The difference is observable at the measured availability arm and not at the nameplate arm: at the measured arm the horizon form leaves ⟦2⟧ years above the ten per cent standard, the worst at ⟦0.1551⟧, expending ⟦20.8⟧ per cent of the horizon budget, where the annual form leaves none. A second comparison, of which determinant each form understates, cannot be made on the current model: every cell resolves at the tightest grid value in both forms, the determinant is indeterminate, and the magnitudes measured under the superseded specification are withdrawn rather than restated.

### 3.5.5 The carbon price

A carbon price on total emissions $E_{t}$ (3.13) would price self-generation correctly, since it attaches to the physical quantity rather than the capped slice. It is nonetheless zero in the central case.

Nigeria has no carbon price. Introducing one to suppress a modelling artefact would assume the conclusion: the model would report a compliant, low-leakage plan because it had been given an instrument the country does not have. The carbon price is retained as a sensitivity, where it answers a legitimate and separate question — what price would be required to price self-generation out of the plan.

The question the sweep was posed to answer does not arise on the current dispatch representation. Where the public capital budget binds at its tightest tier, the configuration builds no self-generation at a zero-carbon price, so there is no displacement for a price to achieve. What the sweep does establish is that a carbon price abates by other means: at that tier, raising the price to ⟦150⟧ USD/tCO₂ reduces cumulative emissions by ⟦64.1⟧ MtCO₂, or ⟦12.7⟧ per cent, with the increase in system cost exceeding the carbon payment collected. Abatement is therefore obtained by changing the generation mix inside the boundary rather than by pricing demand back across it, and Chapter 4 reports the magnitudes. Whether a price outside the swept range would displace self-generation in a configuration that builds some is not answered here: the grid stops at 150.

### 3.5.6 The out-of-boundary diagnostic

Because the obligation is the only defence and does not bind in every configuration, a post-solve check reports the share of physical emissions falling outside the cap boundary, cumulated over the whole solved horizon:

$\lambda_{t}\mspace{6mu} = \mspace{6mu}\frac{E_{t}^{self}}{E_{t}}$ (3.42)

and flags the run where $\lambda$ exceeds a threshold of 0.10. Chapter 4 recomputes the same quantity over the reporting window rather than the solved horizon. The two bases differ materially and each figure states which it uses.

- What $\lambda$ is, and what it is not. It is a share, not a displacement. A positive $\lambda$ means some emissions fall outside the commitment boundary; it does not by itself establish that any emission was displaced there by the policy. The quantity that corresponds to leakage in the usual sense — emissions relocated across a policy boundary in response to the policy — is the increment of $\lambda$ over the same configuration solved without the cap. Chapter 4 reports both wherever it makes a leakage claim (§4.4.3).

> On this model the distinction favors the finding rather than qualifying it. Solved without the cap, the configurations reported in Chapter 4 produce no self-generation at all, so the out-of-boundary share is zero and there is no baseline share to net off. The increment is therefore equal to the level, and a statement about the level is a statement about what the cap caused.
>
> The threshold is a declared judgement, not a derived quantity. The claim it supports is narrow and is stated in full: if more than a stated fraction of physical emissions falls outside the commitment boundary, the plan’s compliance is an artefact of the boundary, and the run is not reported as an NDC-compliant plan.

- The diagnostic is not applied to every capped run. Several scripts passing a real emissions cap never call it (Appendix B, L51), so a non-compliant plan from those runners can be reported without the flag. The count has grown since that entry was written, because runners reaching a cap through shared experiment functions inherit the same gap. At the current commit nine scripts are affected: three pass the cap directly — the leakage arm, the obligation trajectory and the identification test — and six reach it through shared functions.

> Two readings follow from the uncapped column. Where capital is not scarce, the entire out-of-boundary share is cap-induced: the share is ⟦7.37%⟧ under the cap and zero without it, in every configuration from `adequacy` upward. And where capital is scarce, a second mechanism operates independently of the cap — at a tight envelope the uncapped share is already ⟦8.28%⟧ at full collection and ⟦20.62%⟧ at the current rate, so revenue friction pushes load off the grid whether or not an emissions cap is in force. The two mechanisms compound rather than substitute.

Chapter 4 reports the surface across capital tier and collection rate, and the inversion it produces on two instruments: a discrete comparison between a declining cap trajectory and a flat hold, and a continuous sweep of a flat cap as a fraction of the reference, which finds the same inversion as a turning point rather than as a single comparison. The two instruments agree on the mechanism and differ on where it bites, and any percentage figure attached to a cap tightening belongs to the instrument that produced it rather than to cap tightening generally. That result is this section’s architecture turned into a finding. A cap drawn at a boundary disciplines what is inside it; abatement displaced across the boundary is not abatement.

## 3.6 Scenario design and verification

### 3.6.1 Scope

Sections 3.1 to 3.5 state the model: its formulation, its demand and supply representation, its financing layer and its policy constraints. This section states how that model is used — how scenarios are constructed and controlled, how trade-offs between competing objectives are recovered, how uncertainty is treated, and how results are verified before they are reported.

The distinction matters because the solver and the method are different objects. Every solve in this study is a deterministic, single-objective linear programme with fixed coefficients and perfect foresight. The analysis built on those solves is neither single-objective nor deterministic: trade-offs are recovered by the ε-constraint method across two objective pairs, and uncertainty is addressed at three distinct levels. “A deterministic linear programme” is accurate about the solver and misleading as a description of the method.

### 3.6.2 The arm system

Every model input a scenario can vary is exposed as a named arm: a key taking one of a small set of named values, each value a fully specified parameter set rather than a free numeric input. A scenario is a tuple of arm values, and any reported result can be reproduced by naming the arms that produced it. The model carries 23 such arms.

Two properties are worth stating explicitly:

- Arms are named, not numeric. Values are selected by name and the underlying numbers live in one module, changed there once for every call site. This prevents the same quantity from taking different values in different runners, a failure mode this study encountered and corrected during development.

- Arm values carry provenance. Each value is classified SOURCED, DERIVED or ASSUMED, and no value enters the model without that classification. Where no defensible value exists, the arm is left unset and raises when selected rather than substituting a placeholder. Two arms are in that state: the central gas availability factor and a fourth peak-multiple candidate. §3.7.5 treats both. The design choice is deliberate: an arm that cannot be run is a declared gap, whereas an arm silently carrying an invented number is not visible at all.

### 3.6.3 Scenario specification

Scenario construction requires every arm to be supplied explicitly. There is no default for a scenario-relevant key, and omitting one raises. The motivation is a specific defect encountered during development, in which a silent default wrote a tariff value into results that no invocation had requested.

The discipline is not universal, and its exceptions are declared here rather than left implicit. §3.3.8 enumerates those in the supply layer; the remainder are:

1.  The social discount rate, the within-block peaking factor and the self-generation busbar factor are supplied outside the arm system.
2.  One runner constructs its scenario once and then overwrites the carbon-price fields directly for each sweep point, bypassing the arm mechanism for the swept dimension.
3.  The fixed-plan re-dispatch runner selects its first-stage gas deliverability arm independently of the distribution from which its second-stage realisations are drawn; nothing enforces that the frozen plan is optimised for the regime the sampling distribution centres on.
4.  The reliability feasibility runner exposes its mode as a runner-local argument rather than through the shared interface, because registering it centrally would alter every runner’s command line; one further script hardcodes the gas availability arm with no corresponding flag.
5.  Where no self-generation case is supplied, the self-generation price falls back to the value of lost load rather than to a generation cost. That path is inert in every runner reported here, but it is a silent substitution of one parameter for another and belongs in this list.

A set of hardcoded constants and in-function literals sits outside the arm system entirely. These are not scenario dimensions and are not intended to be varied; they are enumerated in Appendix D so that a reader can see what is fixed by construction rather than by choice. §3.6.10 reports a systematic test of whether every arm a runner sets reaches the built model, and what that test found.

The central configuration. Results described as central are solved with the sliced builder (§3.1.8) at the following arm set, which is defined once in the Phase B driver and taken from there rather than restated per runner: conversion at its central case; hydro flat; gas deliverability on the baseline trajectory; demand on the central organic growth arm; the capital-recovery and coverage-ratio parameters at their central values; the concessional envelope at the scale set in §3.4.3; the service tariff at ⟦95⟧ United States dollars per megawatt-hour; and operation and maintenance costs at the moderate case of the sourced series (§3.3). Gas fleet availability is held at the nameplate value, which §3.7.5 records as the declared upper bound of a bracket whose lower bound is the measured figure; no central value exists for that parameter and none is asserted. Every other configuration in this thesis is described as a deviation from this set, naming the arms that differ.

### 3.6.4 The anchors

Two scenarios are designated anchors and re-solved whenever the model changes structurally.

The central anchor is the configuration against which substantive results are read: ⟦23.323⟧ billion USD of discounted system cost, with ⟦24,697.69⟧ MWh of storage capacity in place at 2045 and ⟦56,289.88⟧ MWh at the terminal year. Both years are quoted because the terminal figure lies inside the end-effect buffer and is a reproducibility signature rather than a reported result (§3.1.1). The regression anchor is a deliberately reduced configuration whose purpose is to reproduce a known pre-change state exactly: ⟦19.853⟧ billion USD and ⟦74,435.22⟧ MWh at the terminal year.

Both anchors are solved without the cumulative deployment trajectory applied to solar build. The distinction is immaterial for the central anchor, whose plan clears that trajectory with slack in every year, and material for the regression anchor, whose plan exceeds it in eleven years. Results taken from runners that inherit the capped default are identified as such where they appear.

An anchor’s function is to make structural change visible. Any change that alters an anchor has changed model behaviour, whether or not that was intended; any change that leaves it bit-identical has not. Every structural change made in this study was accompanied by an arm reproducing pre-change behaviour exactly, and was accepted only once that reproduction was demonstrated.

The anchors held unchanged across one substantial re-run and then moved deliberately at a later one, and both facts are informative. Across the first, fourteen runners were ported to the sliced builder and four parameters were re-sourced, and neither anchor moved by a reported digit, which bounds the aggregate effect of those changes on the headline figures at zero. At the later correction epoch both moved, by amounts attributable to two named changes: the central objective fell ⟦3.39%⟧ on the capital-charge timing correction of §3.4.2 and a further ⟦5.11%⟧ on the restoration of the storage capital series (§3.3.6), while the regression anchor was unaffected by the first — it runs an overnight capital treatment that the timing convention does not touch — and fell ⟦6.92%⟧ on the second, with its storage rising ⟦70.9%⟧. The pre-epoch values are retained in the record, and Appendix C carries the corrections.

That the regression anchor was insensitive to the first correction is a property of its configuration rather than evidence about the change, and it exposed a gap in the design: a single anchor running an overnight capital treatment cannot detect any change to the per-tranche financing layer. A second signature, exercising that layer, was added for that reason. §3.6.10 states what the gap was and what remains uncovered.

### 3.6.5 The central sweep

A single-deviation sweep tests the model’s response to each arm value in isolation, organised in three parts: the anchors; one-at-a-time deviations from the central scenario; and paired deviations exposing interactions that single deviations cannot. The sweep comprises 52 invocations — 5 anchor invocations, 38 single deviations and 9 paired cells.

The sweep was extended during this study to cover every arm value that no prior invocation had exercised. That enumeration found 23 usable unexercised values across 13 arms, rather than the eight a prior inventory had recorded, plus four values that raise by design. The distinction matters: an arm never run is a gap in the analysis, whereas an arm that cannot be run because its value is unsourced is a declared limitation, treated as such in §3.7.

### 3.6.6 Bi-objective trade-offs by the ε-constraint method

Two objective pairs are examined: cost against emissions, and cost against capital. In each the frontier is recovered by the ε-constraint method (Haimes et al., 1971; Cohon, 1978; Chankong and Haimes, 1983), in which one objective is minimised while the other is imposed as a parametric constraint, and that constraint is swept.

This is the standard multi-objective construction. The frontier is the multi-objective result; it is assembled from single-objective solves. Each point carries the dual on its ε-constraint, a check of whether that constraint binds, and a non-dominance verification against its neighbours. That verification is not incidental. An optimal solution to the plain ε-constraint formulation is not guaranteed to be efficient and may be only weakly efficient, a weakness addressed in the literature by modified formulations; checking non-dominance at each point is the alternative adopted here (Ehrgott and Ruzika, 2008; Mavrotas, 2009).

No weighted-sum scalarization is used. The deletion is methodologically motivated: a weighted sum cannot reach non-convex regions of a frontier (Das and Dennis, 1997), whereas the ε-constraint method can. No result in this thesis derives from a weighted-sum objective.

The cost–emissions frontier sweeps the emissions cap as a fraction of the reference cap across four financing arms. The cost–capital frontier sweeps the available capital as a fraction across two financing regimes, 19 points each.

Both frontiers carry an endpoint condition established during verification. A frontier whose extreme points lie outside its own grid is a middle segment rather than a frontier, and both were initially in that state: the cost–emissions grid bottomed out at a tenth of the reference cap with no infeasible point located, and its loose end had no uncapped optimum to anchor against. Both ends were subsequently located and the tight end extended. §3.6.9 describes the discipline this produced, and §3.7.7 records what remains boundary-resolved.

### 3.6.7 Three treatments of uncertainty

Uncertainty enters at three levels. They answer different questions, they are not alternatives, and none subsumes another.

- Scenario uncertainty is handled by the arm system and the sweep: discrete, named alternatives for quantities whose uncertainty is structural rather than distributional — which gas deliverability regime obtains, which demand trajectory is realised, which policy is in force. These carry no probabilities.

- Parameter uncertainty is handled by Sobol variance decomposition (Sobol’, 2001; Saltelli et al., 2010; implemented in SciPy, Virtanen et al., 2020) over five continuous parameters, at $N = 256$, giving 1,792 deterministic solves. First- and total-order indices are reported for system cost and for capital. One caveat travel with every statement of these indices. The gas availability factor carries a first-order index of ⟦0.9466⟧ on system cost and ⟦0.9545⟧ on capital mobilized, and its prior spans the entire distance between the observed and the assumed value of a parameter that has no sourced central figure. The dominance is a property of that prior width rather than a discovery about the system, and a narrower prior, if one could be justified, would reduce the share correspondingly.

- Realization uncertainty is handled by fixed-plan re-dispatch. The investment decision is frozen at its deterministic optimum and the system is re-dispatched under 100 Monte Carlo realisations of demand growth and gas regime, yielding the distribution of realised cost, value at risk, conditional value at risk (Rockafellar and Uryasev, 2000) and — uniquely among the three treatments — the frequency with which the committed plan is infeasible. That last quantity is why this treatment is not redundant with the variance decomposition. The Sobol analysis re-optimises at every draw and therefore cannot observe a plan failing; it measures how the optimum moves, not whether a commitment survives.

What is not claimed. This study does not perform stochastic programming with recourse. The optimizer never hedges against a distribution inside a single programme. The treatment above is two-stage evaluation, not two-stage optimisation, and the implementation’s own documentation states the limitation.

### 3.6.8 Bounding the value of the stochastic solution

The absence of a recourse formulation can nonetheless be bounded, using only the two evaluations the study does perform (Madansky, 1960; Birge and Louveaux, 2011).

Let the here-and-now decision be the solar and storage build, with gas investment and dispatch both treated as recourse. Then $WS$, the wait-and-see value, re-optimises the full problem under each realisation with perfect foresight of it; $ECP$, the expected cost of the committed plan, freezes solar and storage at the deterministic plan and re-optimises gas and dispatch under each realisation; and $RP$, the recourse problem, is not solved. Since $WS \leq RP \leq ECP$, the value of the stochastic solution for this stage boundary satisfies $VSS \in \lbrack 0,\mspace{6mu} ECP - WS\rbrack$.

Four conditions must be stated whenever that bound is quoted.

1.  The upper bound contains both $VSS$ and the expected value of perfect information, since $ECP - WS = (ECP - RP) + (RP - WS)$, and the two cannot be separated without solving $RP$. A small bound is therefore informative, because it proves the value of the stochastic solution is small, while a large one is not, because the excess may be entirely the value of perfect information.

2.  The quantity is not the expected result of the expected-value solution. That construction requires the first-stage plan to solve the mean-parameter instance. One uncertain dimension here is a categorical regime with no mean, so no expected-value instance exists for it. The relation $RP\mathbb{\leq E\lbrack}\text{cost of any fixed first stage}\rbrack$ holds for any feasible plan, so the bound survives the distinction while the label does not.

3.  The configuration the bound is computed in travels with it. Both legs are solved with the public capital budget unconstrained, so the frozen first-stage plan is the unconstrained anchor’s plan. The bound is a property of the configuration rather than of the tier’s name, and is quoted with the configuration for that reason.

4.  The stage boundary is a modelling choice and travels with the result. Gas is treated as recourse because it is the shorter-lead-time and shorter-tenor asset; solar and storage are committed because they carry the long-tenor concessional capital this study concerns. The bound is the value of the stochastic solution for that commitment, not for the full investment problem.

Both legs are computed on an identical realised draw set, shared as a file rather than regenerated from a common seed, because the two implementations sample differently and the same seed would not produce the same draws. Five automated guards refuse to emit a bound unless the draw-set hash matches across legs, the recourse configuration is correct, the first-stage regime matches the sampling distribution’s mode, and both legs are feasible on every draw. Those guards compare the recourse configuration but not the full arm set, and that gap is recorded (Appendix B, L93).

Measured: $ECP$ = ⟦36.343⟧ billion USD and $WS$ = ⟦24.253⟧ billion, giving an upper bound of ⟦12.089⟧ billion, or ⟦33.3⟧ per cent of $ECP$. The mean is reported alongside its median, ⟦25.71⟧ billion, because the expectation is the quantity the bound is defined on while the gap between the two locates the result in the distribution’s right tail.

### 3.6.9 Threshold detection and grid resolution

Several results in this study take the form of a threshold: the tightest reliability standard a configuration can meet, the point at which a constraint begins to bind, the price at which a technology leaves the solution. Each is detected by solving across a bounded grid and locating the transition.

A threshold detected at a grid boundary has not been located. It is a statement about the grid rather than about the system, and reported without qualification it reads as a substantive negative result.

This study encountered that failure four times — in all three reliability analyses and in the cost–emissions frontier — and in each case the conclusion changed when the grid was extended. In one financing comparison, an advantage appeared in ⟦2 of 20⟧ cells at the original resolution and in ⟦5 of 20⟧ at the extended one, with the maximum effect rising from ⟦0.05⟧ to ⟦0.15⟧ in the units of the standard. In another, a determination reported as indeterminate became definite.\]

Every threshold-detecting runner therefore reports, alongside its threshold: the number of cells resolving at a grid boundary, whether that is a majority, and an explicit statement that boundary-resolved measurements are lower bounds on the true effect rather than confirmed magnitudes. Where a threshold remains boundary-resolved after extension, §3.7.7 records it. The failure mode is general — any bounded-grid threshold search is exposed to it — and the discipline transfers.

### 3.6.10 Verification

A model of this size cannot be verified by inspection. What follows sets out the five practices that govern what is reported, what each of them caught, and — in the final part, which matters most — what none of them can detect.

1.  Regression signatures. Every structural change carries an arm reproducing pre-change behaviour exactly, and the change is accepted only when byte-identity is demonstrated on a case where the change would be visible. A regression demonstrated where old and new behaviour coincide proves nothing; this study rejected one such demonstration and repeated it on a discriminating case. Where removing a defective coefficient breaks bit-identity, the discriminating test is to vary that coefficient: if the result does not move with it, the difference is vertex selection in a degenerate programme rather than changed model behavior.

> The practice has a limit that this study discovered in its own arrangements. For most of the project a single regression signature was in use, and that signature runs under the overnight capital treatment — so it never executes the per-tranche financing path at all. Every structural change to the formulation this thesis claims as its contribution was therefore accepted on a test that could not have failed. A second signature, exercising the tranche split, the concessional envelope, the public capital budget and the bankability constraint, was added once this was found. Even with both, the emissions cap, the carbon price, the grid-service obligation, the self-generation backstop and the land constraint have no signature covering them, and that gap is recorded rather than closed (Appendix B, L104).

2.  Executable invariants. Regression signatures detect change; they do not detect a formulation that has been wrong from the start. For that, a suite of ⟦87⟧ assertions were written, each derived from a property the formulation must satisfy mathematically rather than from any claim made about it, and each returning a pass or a failure against a model the suite builds and solves itself. The assertions cover the capital chain, the two capital constraints, the bankability test, gas deliverability and the gas stock, the emissions cap and its boundary, the grid-service obligation, the operating-cost layer, the horizon and its buffer, the physical dispatch constraints, the analysis layer above the programme, and a dimensional audit of every unit conversion in both builders.

> On first execution ⟦70⟧ assertions passed and ⟦14⟧ failed. The failures included the capital-charge timing error corrected in §3.4.2, the storage capital series described in §3.3.6, the shadow-price basis discussed below, the horizon dependence recorded in §3.1.1, and the absence of hydro operating and capital cost noted in §3.3.5. After those corrections, ⟦81⟧ of ⟦88⟧ assertions pass and ⟦7⟧ fail. The seven remaining failures are each declared in the limitations register and none is silently carried.
>
> The design matters as much as the result. An assertion derived from the formulation’s mathematics can fail on a property nobody ever claimed, which is what the capital-charge timing error was: the capital recovery factor, the discount factor series and the summation window were each individually correct, and the defect existed only in their product. No amount of reading the three expressions separately would have found it. A single assertion — that the present value of a vintage’s payments equals its overnight cost when the financing rate equals the social discount rate — fails on it immediately.

3.  Provenance before implementation. No parameter value enters the model before it is classified and documented. Where a value cannot be sourced, the arm is declared unsourced and raises rather than substituting a placeholder, as the central gas-availability arm does (§3.3.1).

> The discipline failed once; in a mode it was not designed against. A capital-cost series for storage was implemented, documented and verified; a later refactor addressing a different parameter removed the call that read it, and the model reverted silently to a scalar, while three project documents and a claims-ledger entry continued to describe the series as live. A verification record must therefore name the call path and the value the built model holds, not the function in which a value is computed.

4.  Separation of statement from verification. Every quantitative claim reported here was written first as a claim and then checked against the result file that produced it, by a separate pass reporting failures rather than correcting them. Of the claims checked in the principal such pass, eleven failed, including one in which a mechanism was attributed to a constraint the relevant runs did not impose. Two further defects were identified by that pass outside its own scope, both of which had previously been noticed and not acted on.

5.  Result provenance. Each reported figure traces to a named result file, produced by a named runner, with the results tree described by a manifest tied to a repository tag. This too was found to have been incomplete. Most runners wrote to fixed filenames carrying no record of the arms that produced them, so running one at a second configuration would silently overwrite its own earlier output; the practice had been safe only because each was driven at a single configuration. Every runner now writes into a directory keyed by a hash of its full arm set, alongside a record of the arms and the invocation.

What this method cannot detect. Three classes of defect survive all five practices, and they are stated here rather than left for a reader to infer.

The first is a parameter whose stringency the objective does not register. The debt service coverage ratio of §3.4.5 is the clearest instance: within the range tested it changes no objective value, and the only quantity that responds is the volume of service-financed capacity, which is itself unidentified wherever the capital budget does not bind. No regression signature or invariant built on the objective can therefore verify it, and none does.

The second is a quantity that is dimensionally consistent and semantically wrong. The dimensional audit checks that every unit conversion closes and that the two sides of every constraint carry the same units, and it passes on all of them. It did not detect that shadow prices were being reported as present values while compared against benchmarks expressed in the currency of their own year — an error of roughly threefold at the end of the horizon, invisible to a unit check because both quantities are prices per unit of the same physical thing. A third audit class, requiring every quantity derived from a dual or a discounted total to state its basis, was written for that reason and is part of the suite.

The third is anything resting on the identification of a solution that is not unique. Where the optimal face is flat, the solver returns one vertex among many, and which vertex is returned depends on the solver and its tolerances. Every result in this thesis was produced with a single solver, and five of the invariants that assert exact reproduction have been verified under that solver alone. The composition result most exposed to this is identified in §3.7.4 and the direction of its uncertainty stated there.

### 3.6.11 Summary of method

Every solve is a deterministic, single-objective linear programme with fixed coefficients and perfect foresight. On that core the analysis builds three layers, and the claims of this thesis rest on all three. Bi-objective trade-offs are recovered by the ε-constraint method over two objective pairs, with non-dominance verified at each point. Uncertainty is treated at three levels — scenario, parameter and realisation — which answer different questions and do not substitute for one another. And the value of the recourse formulation the study does not solve is bounded above, from the two evaluations it does perform, with the conditions of §3.6.8 attached.

There is no weighted-sum objective and no stochastic programming with recourse. Neither is claimed anywhere in this thesis.

## 3.7 Limitations

### 3.7.1 Scope and organisation

A limitation is recorded here when it conditions how a result should be read. The project’s limitations register, reproduced as Appendix B at the state of the frozen results, carries every entry including those too narrow to warrant discussion; entries resolved during the study are reproduced separately in Appendix C, since a defect found and engineered away is evidence about process rather than a caveat on a result. This section treats the subset a reader needs in order to interpret Chapters 4 and 5 correctly.

Three classes are kept separate, because conflating them makes a model’s weaknesses look either worse or better than they are: **method boundaries**, what the approach does not attempt; **structural simplifications**, what the formulation represents approximately or not at all; and **provenance gaps**, parameters whose values are not sourced to a primary, with the direction of the resulting bias stated where it is known.

### 3.7.2 Method boundaries

**No stochastic programming with recourse.** The optimiser never hedges against a distribution inside a single programme. §3.6.7 sets out the three uncertainty treatments performed and §3.6.8 bounds the value of the formulation that is not. That bound is an upper bound containing both the value of the stochastic solution and the expected value of perfect information, for a first stage comprising solar and storage only.

**No chance constraints and no integer decisions.** Reliability is imposed as a deterministic bound on shortfall rather than as a probabilistic service standard. The formulation is kept linear so that duals remain available, and every shadow price reported in Chapter 4 depends on that choice.

**A single solver.** Every reported result was produced with one solver at one tolerance setting. Where the optimal face is flat the returned vertex is solver-dependent, and §3.6.10 identifies which class of result is exposed to that.

### 3.7.3 Structural simplifications

The limitations in §3.7.5 are gaps in what parameters are known. Those below are properties of the model’s form — things the formulation cannot represent whatever its parameters are. They are recorded because a reader who knows only the equations in §3.1 would not infer all of them, and because several bias results in a known direction.

**Spatial resolution.** Nigeria is a single node. Neither builder contains a locational variable, a transmission-capacity constraint or a zonal balance; generation, storage and demand sum to one national balance each period. A plan optimal nationally may require transmission reinforcement the model neither prices nor constrains against. The direction cannot be signed without knowing whether Nigeria’s network bottlenecks bear more heavily on distributed solar or on concentrated gas and hydro, and it is not asserted here.

**Perfect foresight.** One programme spans the whole horizon and is solved once, so every year’s decision sees every other year’s data. There is no rolling, receding or myopic formulation. This understates the cost of uncertainty and overstates how efficiently a real planner could sequence investment: a planner commits capital without knowing future demand, deliverability or costs, and would carry insurance a perfect-foresight plan has no reason to buy. §3.6.7’s fixed-plan re-dispatch and §3.6.8’s bound probe this gap without closing it.

**No integer structure.** Every decision variable in both builders is continuous. Four absent structures follow, three biasing the same way:

- No unit commitment, which understates the cost and emissions of keeping thermal plant synchronised and cycling, and overstates how freely gas can follow a residual-demand shape net of solar.
- No minimum stable generation, which understates gas plants’ inflexibility at low output — the same direction, compounding.
- No ramping constraint, which understates the cost of steep net-load swings around the daytime solar dip and the evening peak. The sliced structure exposes the shape of that swing; it does not price the transition between slices.
- No start-up cost and no lumpy build. Capacity is added in arbitrary continuous megawatts, which biases total system cost downward relative to a world in which plant comes in discrete unit sizes.

**Storage is an energy balance, not a state of charge.** The seasonal constraint (3.18) requires discharge within a season not to exceed round-trip efficiency times charge within that season, but it imposes no ordering between the periods of the season. The model may therefore discharge in a period preceding the one in which it charges, and a reported dispatch is not guaranteed chronologically feasible within a season. The direction is unambiguous: the representation is more permissive than a chronological one, so it favours storage. No degradation, self-discharge or cycle limit is represented either.

**Surplus generation is freely disposable.** The energy balance (3.14) is an inequality, so generation in excess of demand is dumped at no cost. There is no curtailment price and no minimum-generation floor on any resource, so the model cannot represent a situation in which surplus solar imposes a cost on the system rather than merely producing no benefit.

**Outage treatment.** Firm capacity is derated by a fixed exogenous credit per technology. There is no forced-outage rate, no outage-event distribution and no outage timing anywhere. The model therefore cannot show a short, sharp outage forcing unserved energy or self-generation within an otherwise-adequate year. The direction cannot be signed: a flat annual derate may overstate or understate real firmness depending on whether gas outages cluster — around maintenance windows or fuel shortages — more or less than an average implies. §3.7.4 treats the parameterisation separately, and the two problems compound.

**Demand is exogenous and price-inelastic.** Demand follows a fixed path entering the electricity balance as a hard right-hand side. Tariff appears only in the financing layer, never as a term demand responds to. The growth arms were calibrated using a long-run income elasticity of electricity demand with respect to GDP, but that is a one-time external input fixing a growth rate before the solve, not a live mechanism inside the programme. The consequence is specific: a high-tariff scenario that would in reality suppress some load instead appears entirely as a capacity and cost consequence, overstating the capacity Nigeria would need under such a regime and understating demand’s own capacity to adjust. That is the regime this study’s own service-based channel implies, so the bias falls where the central contribution is measured.

**System boundary.** Cross-border trade is absent from the boundary entirely — there is no import or export dimension set to zero. Self-generation is modelled, but as an unbounded backstop outside the emissions cap boundary, which is the mechanism §3.5.6 measures. On network losses, the model carries one named concept, a busbar factor recognising that a generator at the point of use displaces more than one unit of busbar generation; it defaults to unity, so those losses are not credited. That default understates how much self-generation displaces, by roughly the transmission loss factor — a conservative bias against the leakage finding rather than one inflating it. No general loss term applies elsewhere: solar, gas, storage and hydro all meet demand at busbar.

**Hydro carries no cost term of any kind.** No fixed or variable operating cost term applies to hydro capacity, and hydro expansion enters no capital charge (§3.3.5). The operating omission is a constant offset across the central arms; neither omission is neutral where hydro capacity varies, so the benefit attributed to hydro expansion is an upper bound, and the hydro trajectory is reported in Chapter 4 as a bound rather than as a lever (Appendix B, L95). The model carries an optional fixed operating cost term for hydro, held at zero and inactive in every result reported here.

**Technology learning is exogenous.** Solar and storage capital costs are read as per-year series which decline over the horizon, consistent with an assumed learning curve. No expression ties a cost in any year to anything the model has built. Deployment-linked learning does not exist, so the model can capture neither the cost declines a large domestic programme might induce nor the supply-chain pressure it might create. Recorded as an omission rather than a biased one.

**Policy is exogenous throughout.** The cap trajectory and the carbon price are constructed before the solve and passed in; neither is computed within the programme nor fed back into a later one. There is no rule of the form “if leakage exceeds a threshold, tighten further.” The consequence for this study’s central policy result is specific: every leakage finding is a comparative-statics result across pre-specified cap paths, never a statement about how policy would evolve in response to the leakage it discovers. A policymaker observing leakage at the measured scale would do something; this model cannot say what.

**Capacity credits are assumptions, not results.** Each technology’s contribution to the firm-capacity constraint is a fixed exogenous derate, none estimated from any loss-of-load or reliability calculation internal to the model. Firm capacity is fed to the programme and is never something the programme could revise: the model cannot discover that a mix it otherwise favours would need a different derate to meet the assumed standard. §3.7.4 treats the most consequential instances.

**Two narrower omissions.** Gas price is resolved once per run and is constant across the horizon, so no single solve can represent a mid-horizon price shock — distinct from, and narrower than, varying the gas regime across solves as §3.6.7 describes. And hydro’s annual energy and seasonal shares are exogenous named scenarios rather than a hydrological process, treating historical hydrology as stationary across the horizon.

**No single direction summarises these.** Some push cost and capacity down: no lumpy build, no ramping or start-up cost, free disposal of surplus, the permissive storage representation. Some push up or cannot be signed: single-node resolution, outage timing, network losses net of the conservative busbar default. Demand exogeneity specifically overstates capacity need under high-tariff scenarios. They are stated individually rather than netted into one claim the formulation does not support.

### 3.7.4 Weakly identified inputs to the firm-capacity constraint

The adequacy constraint requires credited firm capacity to meet peak demand scaled by a within-block peaking factor and a reserve margin. Four of its inputs are more weakly identified than the rest of the parameter set, and because they enter the same constraint their weaknesses compound rather than offset.

**The gas capacity credit is not independently parameterised.** The model sets the firm-capacity credit for gas equal to the gas availability factor. These are different quantities — one an energy ceiling on annual fleet output, the other the fraction of nameplate capacity relied upon at the system peak instant — and the implementation makes them independently settable while binding them to the same value, pending a sourcing exercise that was not completed.

The consequence is directional. Fuel scarcity enters the model through the gas balance, which constrains the annual fuel budget. It does not enter the firm-capacity constraint at all: no gas deliverability regime derates credited capacity at peak. A fleet unable to obtain gas is therefore represented as unable to generate energy while still able to cover peak. Since the availability factor is specified as fuel-agnostic — mechanical and atmospheric availability only, excluding feedstock, so as not to double-count what the gas balance already constrains — the credit inherits that fuel-agnosticism into a constraint where fuel availability is precisely what should bite. The firm-capacity constraint is easier to satisfy in this model than in the system it represents, and adequacy results should be read as optimistic by an unquantified margin.

This is observable rather than merely argued. Under the fuel-agnostic availability arm the cost-optimal unconstrained plan serves the entire demand projection with no unserved energy and no self-generation across every gas deliverability regime except the downside case. Under the declared lower-bound arm the same regimes show realised shortfall of ⟦0.47⟧ and ⟦1.07⟧ per cent, absorbed entirely as self-generation.

**The reserve margin is an operating quantity used as a planning standard.** §3.1.7 sets out the two defects: a spinning reserve is sized against the largest single contingency and is not a planning reserve margin, and because the requirement is fixed in megawatts while peak demand grows, the implied fractional margin falls from ⟦19.5⟧ to ⟦6.4⟧ per cent across the horizon. Both run in the same direction as the capacity-credit problem above — adequacy is cheaper to meet in this model than in the system it represents — so the two compound rather than offset.

**The peak shape has a single source.** The peak multiple arm offers a declared floor, a central case and a constructed upper bound, of which the central case is used throughout. All three derive from one hourly dataset. A fourth candidate, intended to be sourced independently from the national planning document, could not be constructed: that document publishes an implied instantaneous peak ratio but no daily load profile, and the block-level seasonal and period factors the model requires cannot be recovered from the ratio alone. The declared central case is not wrong, but it is uncorroborated.

**Solar’s capacity credit is zero, and the justification given elsewhere does not hold in general.** The credit is a hardcoded literal of 0.0 with no arm, no validation and a pending provenance tag; the builders raise only if the key is absent, so the value is taken silently. In the sliced builder a zero credit is defensible — solar generates in day slices and the adequacy constraint is written against peak — but the firm-capacity constraint is annual in both builders rather than evaluated per slice, so zero is a modelling choice and not a consequence of the slice structure. Any claim that it is exact by construction should be read with that qualification. The direction of the resulting bias is conservative: a zero credit denies solar any contribution to adequacy, so the model procures more firm capacity from gas and storage than a positive credit would require, and the cost of a solar-heavy plan is overstated rather than flattered.

**Taken together.** The constraint’s demand side is single-sourced and scaled by a margin that is the wrong class of quantity and shrinking; one supply-side credit is a transferred benchmark doing double duty; and another is unsourced. None is an error, and all are declared here rather than discovered later. But the adequacy results in Chapter 4 report the margin this parameterisation implies, and the most defensible improvements — separating the capacity credit from the availability factor, sourcing a planning reserve margin as such, sourcing the solar credit, and obtaining an independent load shape — would each be expected to tighten it.

### 3.7.5 Parameter provenance gaps

**Gas fleet availability.** No sourced central value exists. The arm offers a declared lower bound of 0.3061, from the regulator’s published plant availability factor, and an upper bound of 0.80; the central value raises rather than substituting a figure. The lower bound is a blended measurement including feedstock unavailability, so using it as the model’s fuel-agnostic factor would double-count the gas shortage the gas balance already represents. Two attempts to construct a Nigerian fuel-agnostic estimate failed: a dedicated-supply plant subsample, for want of documentary basis for the classification, and the system operator’s constraint decomposition, for want of a stable primary source.

A technology-weighted benchmark was constructed instead, applying published per-class equivalent availability factors to the national fleet’s own technology composition, giving 0.8307 — placing the model’s assumed 0.80 some 3.07 percentage points, or 3.83 per cent, below international practice for the same technology mix, on the conservative side. That benchmark is corroboration, not derivation. The 0.80 was set before any benchmark existed and remains classified ASSUMED; the gap between the two is a residual, not an estimate of a Nigeria-specific derate, and is not presented as one. Maintenance regime, parts supply, ambient and dust conditions and plant age are each reasons a Nigerian figure would be expected to fall below an international benchmark, which supports 0.80 being conservative without quantifying by how much.

The exposure of this gap is bounded and was measured. Results turning on feasibility or on frontier position are invariant to the choice between the two availability arms, because under unconstrained gas build a lower availability is answered by building more capacity. Results turning on duals — the marginal cost of reliability, the gas scarcity rent — are not. Chapter 4 states which of its results fall in which class.

**Value of lost load.** Read from a data file carrying no source column and no classification, at 5, 10 and 30 USD per kWh across its three arms, flat across every year. It prices unserved energy in every solve and is the backstop against which every reliability result is measured. Its ratio to the model’s own self-generation price is recorded at 22.45 at the central arm, so that a later derivation has something to argue from; the self-generation price it is measured against is itself only partially sourced.

**Capital cost bases are not Nigerian.** Solar and storage unit capital costs are United States series, with no Nigerian or African adjustment applied to either. §3.3.4 reports the solar gap against the nearest African benchmark and §3.3.6 the storage agreement with the national plan’s own implied rate; neither resolves the absence of a local series, and the two run in opposite directions.

**Monte Carlo demand dispersion.** The standard deviation applied to the demand growth rate in the realisation-uncertainty treatment is a model-internal calibration with no external source, and a separate parameter characterises the same uncertain quantity differently for the variance-decomposition treatment. The two are unreconciled. The bound in §3.6.8 is unaffected, because both legs consume the same realised draws, but the value at risk and conditional value at risk reported in Chapter 4 depend on the former, and their right tail is amplified by compounding a growth-rate perturbation across a thirty-year horizon.

**Self-generation cost and emission factors.** The three arms are not an ordered sensitivity on one quantity. They are three market segments: large commercial and industrial diesel at short-run marginal cost, part-loaded commercial and industrial diesel at full levelised cost, and a small petrol generator as metered, at 34.7, 28.1 and 5.1 per cent higher-heating-value efficiency, giving electrical emission factors of 0.729, 0.902 and 4.638 tCO₂ per MWh. The model applies one segment to the entire self-generation volume, whereas the actual mix spans all three. Chapter 4’s central leakage figure is the commercial and industrial diesel case, with the petrol case reported as a bounded upper estimate rather than as a high scenario of the same quantity. The cost dimension of these arms moves the objective and nothing else: self-generation volume is pinned by the shortfall constraint rather than chosen against price, and was measured invariant across a twelvefold cost range.

### 3.7.6 Boundary conditions on the policy representation

**The emissions cap binds a grid boundary, not a system boundary.** Self-generation lies outside the capped quantity and is unbounded in the formulation. A cap tightening on grid emissions can therefore be met in part by load leaving the grid, and total emissions can rise while the capped quantity falls. This is represented and measured rather than assumed away, but it means every cap-compliance statement in this thesis is a statement about the grid boundary and must be read as one.

A consequence for the cost–emissions frontier follows directly: because self-generation is unbounded, the frontier need have no infeasible end — load can always leave the grid rather than the problem becoming infeasible — and under public-only financing the frontier turns. On the public-moderate arm ⟦12 of 32⟧ points are non-dominated on total emissions, and every capped point tighter than `cap_fraction` ⟦0.625⟧ is dominated; on the public-tight arm ⟦7 of 32⟧ points are non-dominated, down to `cap_fraction` ⟦0.8125⟧. On the capped quantity all 32 points of every arm remain non-dominated. Chapter 4 reports both axes for that reason.

**The cap’s baseline is inherited from a no-conversion run.** The baseline emissions quantity against which the cap trajectory is constructed derives from a run under the no-conversion arm, because the diagnostic field the construction reads is fleet-only and would understate the baseline on a conversion-active run. The inheritance was argued to be builder-invariant — baseline gas sits at its deliverability ceiling and the no-conversion arm reproduces across builders — and was then tested rather than left as an argument: regenerating the baseline on each builder reproduces $E_{\text{base}}$ to a relative ⟦$4.6 \times 10^{- 9}$⟧ and every cap value in the resulting trajectory to within ⟦$8 \times 10^{- 9}$⟧, so the cap is builder-invariant in value.

### 3.7.7 Results bounded by their own grids

Where a result is a threshold detected across a bounded grid and the threshold resolves at a grid boundary, the reported magnitude is a lower bound on the true effect rather than a confirmed measurement. §3.6.9 describes the discipline; the instances remaining after extension are eight, with the boundary-resolved counts each runner reports:

1.  **Reliability feasibility.** ⟦8 of 8⟧ cells resolve at the 0.001 floor, identically at each of the four invocations (two service-obligation forms by two availability arms), a majority; the gas-regime and solar-arm ranges are lower bounds.
2.  **Reliability marginal cost.** The onset of the left shift is undefined in ⟦2 of 4⟧ gas regimes at nameplate availability, where no onset exists, and in none at observed availability; the analysis is not degenerate by its own criterion.
3.  **Financing frontier.** ⟦12 of 20⟧ cells resolve at the 0.05 floor, a majority, distributed across three policy settings — four under no policy, four under the NDC 2.0 unconditional cap and four under the NDC 3.0 unconditional cap — at both availability arms; the reported shifts are lower bounds.
4.  **Cost–emissions frontier.** The tight end is unlocated in all ⟦4 of 4⟧ arms: every cap down to the 0.01 floor is feasible. The loose end lies above the grid.
5.  **Capital frontier.** No infeasible end is found in either financing regime, since both are feasible at a zero envelope. The binding-to-slack boundary is resolved only to the grid step: ⟦11 binding, 7 slack and 1 absent⟧ point under public-only financing; ⟦9, 9 and 1⟧ under service-based financing.
6.  **Capital-budget sweep.** The capital level at which the unserved energy bottleneck disappears is the tightest case swept, at ⟦3 of 3⟧ collection-rate scenarios, so the boundary lies below the grid.
7.  **Tariff sweep.** In ⟦16 of 20⟧ policy-by-blend cells the subsidy has no zero-crossing within the grid, whose upper end of 110 million USD per TWh lies below the analytical break-even for the commercial-only (174.7) and central blend (136.2) and just below the optimistic blend (110.2). The ⟦4⟧ cells that do cross are all no-policy cells.
8.  **Carbon-price sweep.** Neither the 50 nor the 90 per cent self-generation displacement is reached up to 150 USD per tCO₂ in the one arm that has any self-generation to displace, where it falls by ⟦1.8⟧ per cent; displacement is undefined in the other two arms, whose baseline is zero. The price at which solar passes half of 2035 generation is not reached in the tight public arm and is located only to one grid step, between 20 and 30, in the other two.

One class is worth distinguishing. Several reliability cells resolve at a shortfall standard of a tenth of one per cent, tighter than any practical service standard. Those are not untested gaps; they are the finding that the configurations concerned achieve essentially perfect reliability, and are reported as such.

### 3.7.8 Two builders

The model’s two builders differ structurally rather than in temporal resolution, and every result reported in this thesis comes from the sliced builder. §3.1.8 states the differences and the two declared exceptions; they are not repeated here.

### 3.7.9 The register

The limitations register records every defect, gap and scope condition identified during the study, whether or not it was subsequently closed. It is the authoritative list; this section is a reading of the subset that conditions Chapters 4 and 5. Appendix B reproduces it at the commit the frozen results were produced from, and Appendix C separately the entries resolved during the study, which are retained because the record of what was found and fixed is itself evidence about how far the reported results can be relied upon.

## References for Chapter 3

- Birge, J.R. and Louveaux, F. (2011). *Introduction to Stochastic Programming*, 2nd ed. New York: Springer.

- Bynum, M.L., Hackebeil, G.A., Hart, W.E., Laird, C.D., Nicholson, B.L., Siirola, J.D., Watson, J.-P. and Woodruff, D.L. (2021). *Pyomo — Optimization Modeling in Python*, 3rd ed. Springer Optimization and Its Applications, vol. 67. Cham: Springer. <https://doi.org/10.1007/978-3-030-68928-5>

- Chankong, V. and Haimes, Y.Y. (1983). *Multiobjective Decision Making: Theory and Methodology*. North-Holland Series in System Science and Engineering, vol. 8. New York: North-Holland.

- Gautam, K., Purkayastha, D., & Widge, V. (2023, June 21). Cost of capital for renewable energy investments in developing economies. Climate Policy Initiative. <https://www.climatepolicyinitiative.org/publication/cost-of-capital-for-renewable-energy-investments-in-developing-economies/>

- Cohon, J.L. (1978). *Multiobjective Programming and Planning*. Mathematics in Science and Engineering, vol. 140. New York: Academic Press.

- Das, I. and Dennis, J.E. (1997). A closer look at drawbacks of minimizing weighted sums of objectives for Pareto set generation in multicriteria optimization problems. *Structural Optimization*, 14(1), 63–69.

- Ehrgott, M. and Ruzika, S. (2008). Improved ε-constraint method for multiobjective programming. *Journal of Optimization Theory and Applications*, 138(3), 375–396. <https://doi.org/10.1007/s10957-008-9394-2>

- Ember. (n.d.). Yearly electricity data. <https://ember-energy.org/data/yearly-electricity-data/>

- Federal Ministry of Power. (2024). Nigeria integrated resource plan 2024 (NIRP 2024).

- Federal Republic of Nigeria (2024). *2024 National Inventory Document (NID)*, 24 December 2024. National Council on Climate Change. Table 3.9 (1.A.1.a Public electricity and heat production, 2018: 16,234 kt CO₂e), p.36.

- Federal Republic of Nigeria (2024). *First Biennial Transparency Report (BTR1)*, 26 December 2024. National Council on Climate Change. Table 2.10 (Sector 1 Energy, 2018: 193,346.72 kt CO₂e), p.33.

- Federal Republic of Nigeria (2025). *Nigeria’s Third Nationally Determined Contribution (NDC 3.0)*, September 2025. National Council on Climate Change, The Presidency. 168.2/184.9 MtCO₂e targets and 573.5 MtCO₂e 2018 base, p.14; energy-sector share (22.0%/26.3%), p.15; Table 2 electricity-generation measures, p.16; net-zero-2060, pp. vii, viii, 14.

- Fisher, I. (1930). *The Theory of Interest: As Determined by Impatience to Spend Income and Opportunity to Invest It*. New York: Macmillan.

- Forrest, J. and Lougee-Heimer, R. (2005). CBC user guide. In Greenberg, H.J. and Smith, J.C. (eds), *Emerging Theory, Methods, and Applications*, INFORMS Tutorials in Operations Research, pp. 257–277. Catonsville, MD: INFORMS. <https://doi.org/10.1287/educ.1053.0020>

- Gas Turbine World. (2024, July 12). How much does it cost to build a gas power plant? <https://gasturbineworld.com/capital-costs/>

- Haimes, Y.Y., Lasdon, L.S. and Wismer, D.A. (1971). On the bicriterion formulation of the problems of integrated system identification and system optimization. *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-1, 296–297. <https://doi.org/10.1109/TSMC.1971.4308298>

- Hart, W.E., Watson, J.-P. and Woodruff, D.L. (2011). Pyomo: modeling and solving mathematical programs in Python. *Mathematical Programming Computation*, 3(3), 219–260.

- Huangfu, Q. and Hall, J.A.J. (2018). Parallelizing the dual revised simplex method. *Mathematical Programming Computation*, 10(1), 119–142. <https://doi.org/10.1007/s12532-017-0130-5>

- IEA (2019). *Africa Energy Outlook 2019*. World Energy Outlook Special Report. Paris: International Energy Agency. [https://www.iea.org/reports/africa-energy-outlook-2019 (CC BY 4.0)](https://www.iea.org/reports/africa-energy-outlook-2019%20(CC%20BY%204.0))

- Institute of Electrical and Electronics Engineers. (2023). *IEEE standard definitions for use in reporting electric generating unit reliability, availability, and productivity* (IEEE Std 762-2023). [https://standards.ieee.org/ieee/762/6856/](https://standards.ieee.org/ieee/762/6856/?utm_source=chatgpt.com)

- International Energy Agency. (2025, February 7). Weighted average cost of capital in Kenya, Senegal and South Africa, 2022. <https://www.iea.org/data-and-statistics/charts/weighted-average-cost-of-capital-in-kenya-senegal-and-south-africa-2022>

- International Development Association. (n.d.). IDA lending terms. World Bank. <https://ida.worldbank.org/en/financing/ida-lending-terms>

- International Renewable Energy Agency. (2026). Renewable power generation costs in 2025. <https://www.irena.org/Publications/2026/Jul/Renewable-Power-Generation-Costs-in-2025>

- Japan International Cooperation Agency. (2014). The project for review and update of Nigeria national water resources master plan: Volume 5, supporting report.

- Keane, A., Milligan, M., Dent, C.J., Hasche, B., D’Annunzio, C., Dragoon, K., Holttinen, H., Samaan, N., Söder, L. and O’Malley, M. (2011). Capacity value of wind power. *IEEE Transactions on Power Systems*, 26(2), 564–572. <https://doi.org/10.1109/TPWRS.2010.2062543>

- Kotzur, L., Markewitz, P., Robinius, M. and Stolten, D. (2018). Impact of different time series aggregation methods on optimal energy system design. *Renewable Energy*, 117, 474–487.

- Liddle, B. and Hasanov, F.J. (2023). Are the income and price elasticities of economy-wide electricity demand in middle-income countries time-varying? Evidence from panels and individual countries. *Environmental and Ecological Statistics*, 30(4), 827–849. <https://doi.org/10.1007/s10651-023-00585-4>

- Madansky, A. (1960). Inequalities for stochastic linear programming problems. *Management Science*, 6(2), 197–204. <https://doi.org/10.1287/mnsc.6.2.197>

- Mavrotas, G. (2009). Effective implementation of the ε-constraint method in multi-objective mathematical programming problems. *Applied Mathematics and Computation*, 213(2), 455–465. <https://doi.org/10.1016/j.amc.2009.03.037>

- Mirletz, B., Vimmerstedt, L., Stehly, T. et al. (2024). *2024 Annual Technology Baseline (ATB) Cost and Performance Data for Electricity Generation Technologies*. Golden, CO: National Renewable Energy Laboratory. <https://doi.org/10.25984/2377191>

- Nigerian Electricity Regulatory Commission. (2025). 2024 annual report. <https://nerc.gov.ng/wp-content/uploads/2025/07/2024-Annual-Report.pdf>

- Nigerian Electricity Regulatory Commission. (2025). First quarter 2025 report.

- Nigerian Electricity Regulatory Commission. (2025). Second quarter 2025 report.

- Nigerian Electricity Regulatory Commission. (2025). Third quarter 2025 report.

- Nigerian Electricity Regulatory Commission. (2026). Fourth quarter 2025 report.

- Nigerian National Petroleum Company Limited. (2026). NNPC Limited Gas Master Plan 2026. <https://nnpcgroup.com/insights/nnpc-gas-master-plan-2026-report>

- NREL (2024). *2024 Annual Technology Baseline*. Golden, CO: National Renewable Energy Laboratory. <https://atb.nrel.gov/>

- Nigerian Upstream Petroleum Regulatory Commission. (2025). Nigeria’s gas production hits daily average of 7.59 billion scf as NUPRC releases July production. <https://www.nuprc.gov.ng/nigerias-gas-production-hits-daily-average-of-7-59billion-scf-as-nuprc-releases-july-production/>

- Coleman, B. (2021). Assessment of a social discount rate and financial hurdle rates for energy system modelling in Viet Nam (OECD Environment Working Papers No. 181). OECD Publishing. <https://doi.org/10.1787/a4f9aff3-en>

- Oluwole (2023). *Nigeria National Electricity Demand Timeseries*. Mendeley Data, v2. <https://doi.org/10.17632/pxvdm26rn7.2>

- Oree, V., Sayed Hassen, S.Z. and Fleming, P.J. (2017). Generation expansion planning optimisation with renewable energy integration: A review. *Renewable and Sustainable Energy Reviews*, 69, 790–803. <https://doi.org/10.1016/j.rser.2016.11.120>

- Poncelet, K., Delarue, E., Six, D., Duerinck, J. and D’haeseleer, W. (2016). Impact of the level of temporal and operational detail in energy-system planning models. *Applied Energy*, 162, 631–643. <https://doi.org/10.1016/j.apenergy.2015.10.100>

- Ringkjøb, H.-K., Haugan, P.M. and Solbrekke, I.M. (2018). A review of modelling tools for energy and electricity systems with large shares of variable renewables. *Renewable and Sustainable Energy Reviews*, 96, 440–459. <https://doi.org/10.1016/j.rser.2018.08.002>

- Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. The Journal of Risk, 2(3), 21–41. <https://doi.org/10.21314/JOR.2000.038>

- Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M. and Tarantola, S. (2010). Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index. *Computer Physics Communications*, 181(2), 259–270. <https://doi.org/10.1016/j.cpc.2009.09.018>

- Schröder, T. and Kuckshinrichs, W. (2015). Value of lost load: An efficient economic indicator for power supply security? A literature review. *Frontiers in Energy Research*, 3, 55. <https://doi.org/10.3389/fenrg.2015.00055>

- Shapley, L. S. (1953). A value for n-person games. In H. W. Kuhn & A. W. Tucker (Eds.), Contributions to the theory of games (Vol. II, pp. 307–317). Princeton University Press. <https://doi.org/10.1515/9781400881970-018>

- Short, W., Packey, D. J., & Holt, T. (1995). A manual for the economic evaluation of energy efficiency and renewable energy technologies (NREL/TP-462-5173). National Renewable Energy Laboratory. <https://doi.org/10.2172/35391>

- Sobol’, I.M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. *Mathematics and Computers in Simulation*, 55(1–3), 271–280. <https://doi.org/10.1016/S0378-4754(00)00270-6>

- United Nations, Department of Economic and Social Affairs, Population Division — *World Population Prospects*, as the upstream of the World Bank population series (workbook README: “UN World Population Prospects plus national statistical office data”).\]

- Institute of Electrical and Electronics Engineers. (2023). *IEEE standard definitions for use in reporting electric generating unit reliability, availability, and productivity* (IEEE Std 762-2023). IEEE.

- European Commission. (n.d.). *Carbon leakage*` `

- `Grubb, M., Edmonds, J., ten Brink, P., & Morrison, M. (1993). The costs of limiting fossil-fuel CO₂ emissions: A survey and analysis. Annual Review of Energy and the Environment, 18, 397–478.`

- Bopp, A. E., Kannan, V. R., Palocsay, S. W., & Stevens, S. W. (1996). An optimization model for planning natural gas purchases, transportation, storage and deliverability. *Computers & Operations Research, 24*(5), 511–522. <https://doi.org/10.1016/S0305-0483(96)00025-4>

- Organisation for Economic Co-operation and Development. (2018). *OECD DAC blended finance principles: For unlocking commercial finance for the Sustainable Development Goals*. OECD Publishing. [https://doi.org/10.1787/dc66bd9c-en](https://doi.org/10.1787/dc66bd9c-en?utm_source=chatgpt.com) [OECD](https://www.oecd.org/en/publications/oecd-dac-blended-finance-principles_dc66bd9c-en.html?utm_source=chatgpt.com)

- Virtanen, P., Gommers, R., Oliphant, T.E. et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2

- World Bank. (n.d.). World Development Indicators. <https://databank.worldbank.org/source/world-development-indicators>

- Yescombe, E.R. (2014). *Principles of Project Finance*, 2nd ed. `Academic Press. Sections cited in §3.4.5: §12.3.1 p.322, §14.4.3 p.376, §14.12 p.397, §2.5.1, §15.8, Ch.16.]`

- Zhuang, J., Liang, Z., Lin, T. and De Guzman, F. (2007). *Theory and Practice in the Choice of Social Discount Rate for Cost–Benefit Analysis: A Survey*. ERD Working Paper Series No. 94. Manila: Asian Development Bank.
