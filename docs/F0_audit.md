# F0 Audit — Cost of Capital, Adequacy, and Interaction Effects

Read-only audit. Scope: `src/optimize_model.py`, `src/optimize_model_sliced.py`,
`src/blended_finance.py`, `src/storage.py`. No edits were made to the model
code; nothing was executed. Line numbers refer to these files as committed on
branch `phase-fin2-d17` at the time of this audit. Where something is absent
from the audited files, this is stated plainly rather than inferred.

---

## A. Cost of capital

### A1 — Objective expressions for EaaS vs. public solar capex; overnight lump or CRF?

Both channels are priced identically and as an **overnight lump**, discounted
by the social discount factor `m.DF[t]`. No capital recovery factor (CRF) or
annuity term is applied to either.

`optimize_model.py:381-402`:
```python
public_solar_capex_npv = pyo.quicksum(
    m.DF[t] * m.solar_public_add[t]
    * m.solar_capex_param[t]
    for t in T
)
eaas_solar_capex_npv = pyo.quicksum(
    m.DF[t] * m.solar_eaas_add[t]
    * m.solar_capex_param[t]
    for t in T
)
```

Identical structure in `optimize_model_sliced.py:576-581`:
```python
public_solar_capex_npv = pyo.quicksum(
    m.DF[t] * m.solar_public_add[t] * m.solar_capex_param[t] for t in T
)
eaas_solar_capex_npv = pyo.quicksum(
    m.DF[t] * m.solar_eaas_add[t] * m.solar_capex_param[t] for t in T
)
```

The accompanying comment (`optimize_model.py:389-397`) states this as a
deliberate modelling choice: EaaS and public solar are "priced at the SAME
unit CAPEX," and the private cost of capital is represented "NOT by an
objective-side CAPEX premium but by the bankability constraint" — i.e., cost
of capital is kept out of the objective by design and confined to a
constraint (see A3).

### A2 — What rate builds the discount factors?

A **single social discount rate**, not channel-specific, builds the
`m.DF[t]` series that discounts every term in `system_cost_npv` (gas
opex/capex, public solar capex, EaaS solar capex, storage capex, selfgen,
unserved, carbon, salvage).

`optimize_model.py:123-125`:
```python
r_social = float(scenario.get("social_discount_rate", 0.08))
# t=0 at start year
df = {t: 1.0 / ((1.0 + r_social) ** t) for t in T}
```

Identical in `optimize_model_sliced.py:238-239`:
```python
r_social = float(scenario.get("social_discount_rate", 0.08))
df = {t: 1.0 / ((1.0 + r_social) ** t) for t in T}
```

A second, private/blended rate exists (`r_private = resolve_private_rate(scenario)`,
`optimize_model.py:126`; `optimize_model_sliced.py:240`) but it does **not**
feed `m.DF`. It builds a separate parameter, `m.remaining_npv_factor`
(`optimize_model.py:127-132`; `optimize_model_sliced.py:241-246`), which —
per A3 below — reaches only the bankability-related constraints, never the
objective's discounting.

### A3 — Consumers of `blended_wacc`, `r_concessional`, `r_commercial`, `concessional_share`

`r_concessional` and `r_commercial` as identifiers do **not appear** in
either builder file — grep across `optimize_model.py` and
`optimize_model_sliced.py` for `r_concessional|r_commercial|blended_wacc|concessional_share`
returns zero literal matches for `r_concessional`/`r_commercial`/`blended_wacc`
in either builder. They exist only inside `blended_finance.py` and reach the
builders indirectly, as string keys read out of the `scenario` dict inside
`resolve_private_rate()`.

- **`blended_finance.py:67-113`** — `blended_wacc(concessional_share, r_concessional, r_commercial)`:
  pure function, `blended = share*r_conc + (1-share)*r_comm`
  (`blended_finance.py:110-113`):
  ```python
  return (
      concessional_share * r_concessional
      + (1.0 - concessional_share) * r_commercial
  )
  ```
  No side effects; not called anywhere else in the audited files except via
  `resolve_private_rate`.

- **`blended_finance.py:168-187`** — `resolve_private_rate(scenario)`: the
  sole consumer of `r_concessional`/`r_commercial`/`concessional_share` as
  scenario keys (via `blended_wacc`), with fallback to
  `scenario['private_discount_rate']`:
  ```python
  def resolve_private_rate(scenario: dict) -> float:
      keys = ("concessional_share", "r_concessional", "r_commercial")
      if all(k in scenario and scenario[k] is not None for k in keys):
          return blended_wacc(
              concessional_share=scenario["concessional_share"],
              r_concessional=scenario["r_concessional"],
              r_commercial=scenario["r_commercial"],
          )
      return float(scenario.get("private_discount_rate", 0.18))
  ```
  This is the single bridge into the builders.

- **`optimize_model.py:126`** / **`optimize_model_sliced.py:240`** —
  `r_private = resolve_private_rate(scenario)`. Consumer: feeds
  `remaining_npv_priv` / `m.remaining_npv_factor`
  (`optimize_model.py:127-132`; `optimize_model_sliced.py:241-246`), used
  only in `eaas_bankability_rule` (`optimize_model.py:450-459`;
  `optimize_model_sliced.py:596-605`) and `subsidy_limit_rule`
  (`optimize_model.py:490-497`; `optimize_model_sliced.py:613-619`) — both
  **constraints**, not the objective.

- **`concessional_share`** is *also* re-read directly from `scenario` a
  second time, independent of `resolve_private_rate`
  (`optimize_model.py:134`: `concessional_share = scenario.get("concessional_share", 0.0)`;
  `optimize_model_sliced.py:249`: same), and consumed at
  `optimize_model.py:418-427` inside `m.concessional_drawdown_npv`:
  ```python
  if (concessional_envelope_npv is not None
          and concessional_share > 0.0
          and financing_regime == "eaas"):

      m.concessional_drawdown_npv = pyo.Expression(
          expr=concessional_share * pyo.quicksum(
              m.DF[t] * m.solar_eaas_add[t] * m.solar_capex_param[t]
              for t in T
          )
      )
  ```
  which feeds `m.concessional_envelope_constraint` — again a **constraint**
  (see A4), not an objective term. Identical structure at
  `optimize_model_sliced.py:583-593`.

**Net finding: none of `blended_wacc`, `r_concessional`, `r_commercial`,
`concessional_share` reach the objective directly.** They reach two
constraints (`eaas_bankability`/`eaas_subsidy_limit` via the derived private
rate; `concessional_envelope_constraint` via the share). The objective's
only path from these quantities is indirect, through what those constraints
force `eaas_subsidy` and `solar_eaas_add` to be.

(Outside the declared 4-file scope: `src/scenarios.py:62-92` is the single
source of truth populating these scenario keys —
`blended_central = (0.30, 0.06, 0.18)` etc. — noted for traceability, not
audited here since it is not one of the four files in scope.)

### A4 — Concessional envelope constraint: `<=` or `==`?

**`<=`**, in both builders. Quote, `optimize_model.py:429-431`:
```python
m.concessional_envelope_constraint = pyo.Constraint(
    expr=m.concessional_drawdown_npv <= concessional_envelope_npv
)
```

Identical in `optimize_model_sliced.py:591-593`:
```python
m.concessional_envelope_constraint = pyo.Constraint(
    expr=m.concessional_drawdown_npv <= concessional_envelope_npv
)
```

The algebra is a strict inequality; nothing in either file converts it to an
equality. The reported empirical pattern (utilisation exactly 1.0 on 150/150
rows) is **not explained by the constraint's form** — a `<=` constraint can
still bind at equality in every solved instance if it is economically
advantageous to draw the full envelope in each case, but that would be a
solve-time result, not something visible from static inspection. This audit
ran nothing, so the 150/150 observation is neither confirmed nor
contradicted here — only the constraint's algebraic form (`<=`, not `==`) is
established from the code.

### A5 — Consumers of `required_margin`

Defined once per builder, default 1.0:
- `optimize_model.py:177`: `required_margin = scenario.get("required_margin", 1.0)`
- `optimize_model_sliced.py:273`: `required_margin = scenario.get("required_margin", 1.0)`

Two consumers in each file, both constraints, both dividing a
bankable-revenue term (larger `required_margin` shrinks bankable revenue,
tightening the constraint):

1. **`eaas_bankability_rule`** (`optimize_model.py:450-459`):
   ```python
   def eaas_bankability_rule(m, t):

       capex = m.solar_capex_param[t] * m.solar_eaas_add[t]

       effective_private_cost = capex - m.eaas_subsidy[t]
       bankable_revenue = (tariff * solar_energy_per_mw * m.remaining_npv_factor[t] * m.solar_eaas_add[t]) / required_margin

       return effective_private_cost <= bankable_revenue

   m.eaas_bankability = pyo.Constraint(T, rule=eaas_bankability_rule)
   ```
   Sliced counterpart, `optimize_model_sliced.py:596-605`:
   ```python
   def eaas_bankability_rule(m, t):
       capex = m.solar_capex_param[t] * m.solar_eaas_add[t]
       effective_private_cost = capex - m.eaas_subsidy[t]
       bankable_revenue = (
           tariff * solar_energy_per_mw_annual
           * m.remaining_npv_factor[t] * m.solar_eaas_add[t]
       ) / required_margin
       return effective_private_cost <= bankable_revenue

   m.eaas_bankability = pyo.Constraint(T, rule=eaas_bankability_rule)
   ```
   Represents the coverage cushion a lender requires on top of raw
   discounted revenue before financing an EaaS project.

2. **`subsidy_limit_rule`** (`optimize_model.py:490-497`):
   ```python
   def subsidy_limit_rule(m, t):
       max_bankable_at_t = tariff * solar_energy_per_mw * m.remaining_npv_factor[t] / required_margin
       gap_at_t = capex_init[t] - max_bankable_at_t
       gap_at_t = max(0.0, gap_at_t)
       return m.eaas_subsidy[t] <= gap_at_t * m.solar_eaas_add[t]
   ```
   Sliced counterpart, `optimize_model_sliced.py:613-619`:
   ```python
   def subsidy_limit_rule(m, t):
       max_bankable_at_t = (
           tariff * solar_energy_per_mw_annual
           * m.remaining_npv_factor[t] / required_margin
       )
       gap_at_t = max(0.0, capex_init[t] - max_bankable_at_t)
       return m.eaas_subsidy[t] <= gap_at_t * m.solar_eaas_add[t]
   ```
   Same cushion, used to size the maximum public subsidy per MW as the gap
   between raw capex and the margin-adjusted bankable revenue.

`required_margin` does not appear in `system_cost_npv`, in salvage, or
anywhere in `blended_finance.py` or `storage.py` (confirmed by inspection of
all four audited files). In both consumers it plays the identical role: a
divisor shrinking the revenue side of a bankability-style test. It is not a
rate — it is dimensionless. (Outside the declared 4-file scope, the comment
at `src/scenarios.py:912` labels it "bankability coverage cushion, distinct
from WACC," default 1.10 — noted for traceability, not audited here.)

---

## B. Adequacy and storage

### B6 — Firm-capacity / peak / reserve-margin constraint?

**Absent in both files.** Grep for
`reserve|firm.capacity|capacity_credit|planning_reserve|peak_demand|reserve_margin`
across `optimize_model.py` and `optimize_model_sliced.py` returns no matches
except unrelated substrings ("preserved," "annualized power capacity").
Neither builder defines a capacity-credit table or a reserve-margin
requirement of any kind. The only adequacy-adjacent constraints present are:
`gas_capacity_constraint` (fuel/availability-factor-bounded gas output),
`land_constraint`, `solar_cumulative_cap`, `storage_charge_power_limit` /
`storage_power_limit` (or their sliced per-slice counterparts:
`storage_charge_power_limit` / `storage_discharge_power_limit`), and
`grid_service_obligation` (bounds `selfgen + unserved`). The code itself
states this constraint is explicitly not an LOLE/EENS-style reliability
standard (`optimize_model.py:41-50`):
```python
# ------------------------------------------------------------------
# Grid service obligation (renamed from "reliability" in 2.6 F0).
#
# This constraint bounds (selfgen + unserved) -- the GRID SERVICE
# SHORTFALL -- not unserved alone. A household running its own genset was
# not served by the grid, but no capacity was short, so this is a service
# obligation, NOT an adequacy/reliability standard: LOLE and EENS do not
# describe it. The old names are retained as deprecated aliases so the
# pre-2.6 REL and VoLL runners keep working unchanged.
# ------------------------------------------------------------------
```
None of the constraints present is a peak/firm-capacity constraint, and no
capacity credits are enumerated anywhere in either file because no such
mechanism exists in either file.

### B7 — Channels through which storage reduces the objective; SOC linkage in the sliced builder

**Channels (both builders):**

1. `storage_discharge[t]` (or `[t,(s,p)]`) enters the electricity balance
   with a `+` sign and `storage_charge` with a `-` sign. `optimize_model.py:624-635`:
   ```python
   m.balance = pyo.Constraint(
       T,
       rule=lambda m, t:
           m.gas_generation[t]
           + m.solar_generation[t]
           + hydro[t]
           + m.storage_discharge[t]
           - m.storage_charge[t]
           + m.selfgen[t] * selfgen_busbar_factor
           + m.unserved[t]
           >= demand[t],
   )
   ```
   Sliced, `optimize_model_sliced.py:555-566`:
   ```python
   m.balance = pyo.Constraint(
       T, m.S,
       rule=lambda m, t, s, p:
           m.gas_generation[t, (s, p)]
           + m.solar_generation[t, (s, p)]
           + m.hydro_gen[t, (s, p)]
           + m.storage_discharge[t, (s, p)]
           - m.storage_charge[t, (s, p)]
           + m.selfgen[t, (s, p)] * selfgen_busbar_factor
           + m.unserved[t, (s, p)]
           >= demand[(t, s, p)]
   )
   ```
   This is the only channel through which storage affects system operation:
   by shifting generation timing it can displace `gas_to_power` (lower
   `gas_opex_npv`), `selfgen` (lower `selfgen_npv`), or `unserved` (lower
   `unserved_npv`/VoLL) in the periods where it discharges.

2. `m.salvage_npv` credits a straight-line residual on storage capex at the
   horizon end (`optimize_model.py:742-761`, quoted in full under C9),
   reducing `system_cost_npv` by a fraction of the capex already charged —
   this is a partial refund of storage's own cost, not an independent value
   channel.

There is **no** other channel in either file: no cycling cost or opex term
for storage, no arbitrage/price-spread revenue term, no capacity payment.
The `storage_capacity_mwh[t] * l_sto` term in `land_constraint`
(`optimize_model.py:514-521`; `optimize_model_sliced.py:632-639`) is a
cost-side (land-consuming) term, not a reduction channel. Under a
public-budget constraint, `storage_add` competes with public solar and EaaS
subsidy for the same envelope (`optimize_model.py:897-906`;
`optimize_model_sliced.py:878-888`) — a constraint interaction, not an
objective-reduction channel.

**SOC linkage, sliced builder:** There is **no explicit state-of-charge
variable** in either builder — grep and inspection of both files confirm no
`soc` variable exists anywhere in `optimize_model.py` or
`optimize_model_sliced.py`. The sliced model enforces a **season-level
throughput inequality**, not per-slice or per-timestep SOC continuity.

`optimize_model_sliced.py:530-536`:
```python
m.storage_season_balance = pyo.Constraint(
    T, ["dry", "wet"],
    rule=lambda m, t, season:
        pyo.quicksum(m.storage_discharge[t, (s, p)] for (s, p) in S if s == season)
        <= storage_round_trip_eff
        * pyo.quicksum(m.storage_charge[t, (s, p)] for (s, p) in S if s == season)
)
```
Discharge across all three slices of a season (night/day/peak) is bounded by
RTE times charge summed across the same three slices of the same season — an
aggregate seasonal energy budget, not a chronological or slice-by-slice
link. It "resets" at season and year boundaries: nothing couples dry-season
storage flows to wet-season flows, and nothing couples year `t`'s flows to
year `t+1`'s flows (each `t` is a fresh, independent index into the
`pyo.quicksum` above).

The annual builder's analogous constraint, `optimize_model.py:562-567`
(`m.storage_energy_balance`), is the same form collapsed to one bucket per
year:
```python
m.storage_energy_balance = pyo.Constraint(
    T,
    rule=lambda m, t:
        m.storage_discharge[t]
        <= storage_round_trip_eff * m.storage_charge[t],
)
```
independent across years — no cross-year SOC carry-over exists here either.

### B8 — Does the 6-slice structure carry inter-slice chronology or slice-to-slice SOC continuity?

**No, for both.** The slice set is defined at `optimize_model_sliced.py:63`:
```python
SLICES = [(s, p) for s in ("dry", "wet") for p in ("night", "day", "peak")]
```
and used purely as an unordered index for summation everywhere it appears
(balance, gas fuel budget `m.gas_balance` at `:483-488`, hydro season budget
`m.hydro_season_budget` at `:496-501`, storage season balance at `:530-536`).
No constraint in `optimize_model_sliced.py` references slice order or
enforces that energy charged in one slice is available only to
chronologically later slices within the season. The seasonal storage budget
(B7) permits, algebraically, discharge recorded in `dry_night` to be "backed"
by charge recorded in `dry_peak` of the same season with no causal/temporal
check — the abstraction is an annual/seasonal accounting budget, not a
timeline.

The module's own docstring self-describes this scope, `optimize_model_sliced.py:36-48`:
```
WHAT IS AND IS NOT SLICED
-------------------------
Sliced [T,S]      : gas_to_power, gas_curtail, unserved, storage_charge,
                    storage_discharge, hydro_gen, and the electricity balance.
Annual [T]        : all capacity additions, all financial constraints
                    (bankability, subsidy, concessional envelope, public
                    budget), land, and build caps. Capacity is built once a
                    year, not per slice.
Annual, summed    : gas fuel budget, emissions, reliability.
```
(Note: `gas_curtail` is declared `pyo.Var(T, ...)` at `optimize_model_sliced.py:374`
— i.e. annual, not per-slice — despite being listed under "Sliced [T,S]" in
this docstring block; the code and the docstring disagree on this one
variable. This is stated as an observed discrepancy, not resolved here.)
Chronology across slices is not claimed anywhere in the module's own
description, and no code introduces it.

---

## C. Interactions

### C9 — Salvage computation; double-counting risk under CRF-annualised capex

Salvage is a **straight-line residual value** on raw (un-marked-up) capex,
credited once at the final horizon year, per vintage.

`optimize_model.py:735-761`:
```python
def _salvage_fraction(t, life):
    """Straight-line residual fraction at end of horizon for vintage t.
    Vintage t operates years t..T_last inclusive = (T_last - t + 1) years.
    Returns 0.0 for vintages that retire within the horizon."""
    served = T_last - t + 1
    return max(0.0, float(life) - served) / float(life)

m.salvage_npv = pyo.Expression(
    expr=m.DF[T_last] * (
        pyo.quicksum(
            (m.solar_public_add[t] + m.solar_eaas_add[t])
            * m.solar_capex_param[t]
            * _salvage_fraction(t, lives["solar"])
            for t in T
        )
        + pyo.quicksum(
            m.storage_add[t] * econ["STORAGE_COST_PER_MWH"]
            * _salvage_fraction(t, lives["storage"])
            for t in T
        )
        + pyo.quicksum(
            m.gas_add[t] * scenario["gas_capex_per_mw"]
            * _salvage_fraction(t, lives["gas"])
            for t in T
        )
    )
)
```
Identical in `optimize_model_sliced.py:762-781`.

The comment at `optimize_model.py:708-718` states explicitly:
```
# Salvage is computed on RAW capex for BOTH arms: required_margin is a
# financing premium, not asset value. It is subtracted from the objective
# ONLY -- the public budget constraint below continues to see gross capex,
# because the envelope constrains cash out the door.
```

As currently written, capex is charged as an **overnight lump** (A1) and
salvage credits back the un-depreciated fraction of that same lump at the
horizon end — there is no double count in the code as it stands, since
nothing else in either file amortises or recovers the capex a second time.

**Would it double-count under CRF annualisation?** This cannot be answered
from the audited files as a description of existing behaviour, since neither
builder annuitises capex — grep for `capital_recovery|CRF|annuity_factor`
within `src/optimize_model.py`, `src/optimize_model_sliced.py`,
`src/blended_finance.py`, and `src/storage.py` returns zero matches. As a
structural observation about the mechanism (not a description of code that
exists in these four files): a CRF-style annuity converts a lump capex into
a level payment stream that, summed over the asset's economic life, returns
the full principal plus financing cost. If capex were annualised that way
*and* this same straight-line salvage term were kept unchanged, the salvage
credit would be crediting residual value against a capital sum that the CRF
payment stream had already fully recovered by construction — i.e., the two
mechanisms would double-count the same capital recovery, once through the
annuity's principal component and once through the salvage credit. This is
stated as a structural inference about a hypothetical combination, not a
finding about the current code, which does not perform CRF annuitisation of
capex anywhere in the four audited files. (Outside the declared scope: the
only CRF implementation in the repository, `_capital_recovery_factor()` in
`src/EaaS.py:43-47`, was located by the same grep but is not part of this
audit's four files and was not inspected further here.)

### C10 — Does the EaaS channel carry a storage component, or is it solar-only?

**Solar-only.** `solar_eaas_add` is a distinct decision variable from
`solar_public_add`; storage has exactly one variable, `m.storage_add`,
shared across all financing regimes with no regime-specific split.

Evidence:

- `eaas_disable_rule`, active whenever `financing_regime != "eaas"`
  (`optimize_model.py:473-476`):
  ```python
  if financing_regime != "eaas":
      def eaas_disable_rule(m, t):
          return m.solar_eaas_add[t] == 0
      m.eaas_disable = pyo.Constraint(T, rule=eaas_disable_rule)
  ```
  zeroes only `m.solar_eaas_add[t]`; it does not reference or constrain
  `m.storage_add` in any way. Sliced counterpart,
  `optimize_model_sliced.py:607-610`:
  ```python
  if financing_regime != "eaas":
      m.eaas_disable = pyo.Constraint(
          T, rule=lambda m, t: m.solar_eaas_add[t] <= 0.0
      )
  ```
  same: storage untouched.

- `m.eaas_bankability` and `m.eaas_subsidy_limit`
  (`optimize_model.py:449-502`; `optimize_model_sliced.py:595-624`)
  reference only `solar_eaas_add`, `solar_capex_param`, and `eaas_subsidy` —
  no storage term appears in either constraint in either file.

- `storage_capex_npv` (`optimize_model.py:702-706`):
  ```python
  storage_capex_npv = pyo.quicksum(
  m.DF[t] * m.storage_add[t] * econ["STORAGE_COST_PER_MWH"]
  for t in T
  )
  ```
  is a single, regime-agnostic cost term. When a public budget constraint is
  active, `storage_add` is charged against it alongside public solar and the
  EaaS subsidy (`optimize_model.py:897-906`):
  ```python
  m.public_budget_constraint = pyo.Constraint(
      expr=pyo.quicksum(
          m.DF[t] * (
              m.solar_public_add[t] * m.solar_capex_param[t]
              + m.storage_add[t] * econ["STORAGE_COST_PER_MWH"]
              + m.eaas_subsidy[t]
          )
          for t in T
      ) <= public_budget_npv
  )
  ```
  i.e. storage is always treated as publicly-financed infrastructure, never
  as part of the EaaS private-financing mechanism, regardless of
  `financing_regime`. Identical structure at
  `optimize_model_sliced.py:878-888`.

Same structure, same variable names, confirmed identically in
`optimize_model_sliced.py`.
