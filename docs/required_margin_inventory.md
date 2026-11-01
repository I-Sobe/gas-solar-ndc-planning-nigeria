# `required_margin` inventory — locate and quote only

Per `TASK.md`: this is a location-and-quotation exercise for the F1
(per-tranche CRF) rewrite. **No file was changed.** Values are not
reconciled here, and no claim is made about what `required_margin`
*should* mean — only what each site mechanically does with it. Where a
site's role is not mechanically obvious from the code alone, it is marked
**ambiguous** and quoted rather than guessed.

Scope searched: `src/*.py` (all four files that reference it) and every
runner script under `scripts/` that references it (13 files, found via
`grep -rln "required_margin" scripts/*.py`).

---

## 1. Definitions and defaults

### 1a. Builder default — `1.0`

**`src/optimize_model.py:332`** (inside `build_model`):
```python
330	    tariff = scenario.get("solar_service_tariff_usd_per_twh", None)
331	    financing_regime = scenario.get("financing_regime", "traditional")
332	    required_margin = scenario.get("required_margin", 1.0)
```

**`src/optimize_model_sliced.py:408`** (inside `build_model_sliced`,
identical form):
```python
406	    tariff = scenario.get("solar_service_tariff_usd_per_twh", None)
407	    financing_regime = scenario.get("financing_regime", "traditional")
408	    required_margin = scenario.get("required_margin", 1.0)
```

This `1.0` is a **local Python-level fallback**, read only if the
`scenario` dict handed to the builder has no `"required_margin"` key at
all. It is never itself written into the scenario dict.

### 1b. `scenarios.py` default — `1.10`

**`src/scenarios.py:2121-2122`** (inside `load_scenario()`'s returned
dict, unconditional — not gated by any `if`, not tied to any
`load_scenario()` parameter):
```python
2121	        # ---- Required NPV margin (bankability coverage cushion, distinct from WACC)
2122	        "required_margin": 1.10,
```

**Confirmed: `required_margin` is not a parameter of `load_scenario()`.**
Its full signature (`src/scenarios.py:1389-1410`) has no
`required_margin` argument — every caller of `load_scenario()` receives
`1.10` in the returned dict and must overwrite it afterward
(`scenario["required_margin"] = ...`) to get any other value.

### 1c. A third, independent default — `1.10` (again), in a library function

**`src/optimize_experiments.py:2652-2660`** (function signature of
`run_gas_regime_ndc_matrix`, the GAS-1/GAS-3 shared runner):
```python
2652	def run_gas_regime_ndc_matrix(
2653	    econ,
2654	    ndc_cap_scenario,
2655	    gas_cases,
2656	    financing_regime="traditional",
2657	    required_margin=1.10,
2658	    capital_case="moderate",
2659	    cap_path="data/cost/processed/emissions_cap.csv",
2660	):
```
This `1.10` is a **separate, independently-typed default** — a Python
keyword-argument default on this specific function, not inherited from or
tied to `scenarios.py`'s `1.10`. It happens to share the same numeral.
Consumption:
```python
2707	        scenario["financing_regime"] = financing_regime
2708	        if financing_regime == "eaas":
2709	            scenario["required_margin"] = required_margin
```
Only overwrites `scenario["required_margin"]` when `financing_regime ==
"eaas"`. Under the function's own default (`financing_regime="traditional"`),
this parameter is **never applied** — the scenario dict keeps whatever
`load_scenario()` already put there (`1.10`), coincidentally the same
number.

### 1d. A required (no-default) local parameter in a runner script

**`scripts/07_run_gas2_eaas_gas_relief.py:92`** (function signature):
```python
92	def run_no_policy_gas_matrix_eaas(econ, gas_cases, required_margin, cap_path):
```
No default — the caller must supply a value every time. Consumption,
unconditional (the function is EaaS-only by construction):
```python
132	        scenario["financing_regime"] = "eaas"
133	        scenario["required_margin"] = required_margin
```

---

## 2. LP read sites — what each mechanically does

Both builders (`optimize_model.py`, `optimize_model_sliced.py`) contain
**exactly two** live consumers of `required_margin` each, both inside the
EaaS bankability machinery, both gated by `if tariff is not None:` —
i.e. these constraints are **not built at all** unless
`scenario["solar_service_tariff_usd_per_twh"]` is set to something other
than `None`, independent of `financing_regime`'s value (see §5 for which
runners actually trigger this).

### 2a. `eaas_bankability` constraint

**`src/optimize_model.py:689-699`:**
```python
689	    if tariff is not None:
690	        def eaas_bankability_rule(m, t):
691	
692	            capex = m.solar_capex_param[t] * m.solar_eaas_add[t]
693	
694	            effective_private_cost = capex - m.eaas_subsidy[t]
695	            bankable_revenue = (tariff * solar_energy_per_mw * m.remaining_npv_factor[t] * m.solar_eaas_add[t]) / required_margin
696	
697	            return effective_private_cost <= bankable_revenue
698	
699	        m.eaas_bankability = pyo.Constraint(T, rule=eaas_bankability_rule)
```
**`src/optimize_model_sliced.py:818-828`** (identical form):
```python
818	    if tariff is not None:
819	        def eaas_bankability_rule(m, t):
820	            capex = m.solar_capex_param[t] * m.solar_eaas_add[t]
821	            effective_private_cost = capex - m.eaas_subsidy[t]
822	            bankable_revenue = (
823	                tariff * solar_energy_per_mw_annual
824	                * m.remaining_npv_factor[t] * m.solar_eaas_add[t]
825	            ) / required_margin
826	            return effective_private_cost <= bankable_revenue
827	
828	        m.eaas_bankability = pyo.Constraint(T, rule=eaas_bankability_rule)
```

**One-line role**: `required_margin` **divides the revenue side**
(`tariff × energy × remaining_npv_factor × solar_eaas_add`) of a
`cost <= revenue` constraint, before the comparison. Mechanically, a
larger `required_margin` shrinks the revenue counted as "bankable,"
tightening the constraint — the same *shape* as a debt-service-coverage
ratio applied to projected revenue. It does not appear as a rate inside
any discounting/compounding expression (that role is played by
`m.remaining_npv_factor[t]`, computed elsewhere from `r_private`, a
distinct object).

### 2b. `eaas_subsidy_limit` constraint (via `subsidy_limit_rule`)

**`src/optimize_model.py:729-737`:**
```python
729	    if tariff is not None:
730	        def subsidy_limit_rule(m, t):
731	            # remaining_npv_factor is a fixed Param, so this arithmetic resolves
732	            # to a numeric constant; capex_init[t] is read as a plain float to
733	            # avoid any dependence on Param evaluation inside the rule.
734	            max_bankable_at_t = tariff * solar_energy_per_mw * m.remaining_npv_factor[t] / required_margin
735	            gap_at_t = capex_init[t] - max_bankable_at_t
736	            gap_at_t = max(0.0, gap_at_t)
737	            return m.eaas_subsidy[t] <= gap_at_t * m.solar_eaas_add[t]
```
**`src/optimize_model_sliced.py:835-842`** (identical form):
```python
835	    if tariff is not None:
836	        def subsidy_limit_rule(m, t):
837	            max_bankable_at_t = (
838	                tariff * solar_energy_per_mw_annual
839	                * m.remaining_npv_factor[t] / required_margin
840	            )
841	            gap_at_t = max(0.0, capex_init[t] - max_bankable_at_t)
842	            return m.eaas_subsidy[t] <= gap_at_t * m.solar_eaas_add[t]
```

**One-line role**: `required_margin` **divides the same revenue-side
expression** to compute `max_bankable_at_t` (a per-MW ceiling), which is
then subtracted from CAPEX to get the subsidy gap — bounding
`m.eaas_subsidy[t]` above. Mechanically identical division to §2a, feeding
a different constraint's bound rather than a `cost <= revenue` comparison
directly.

**Both LP sites divide a revenue-side term; neither multiplies a cost or
return term directly, and neither enters `m.DF[t]` or any other
discounting expression.**

---

## 3. Diagnostic / post-solve read sites (not read by the LP itself)

These are in `src/optimize_experiments.py`, used for reporting or
analytical cross-checks against the LP's own solved values — they do not
themselves constrain or price anything in `build_model`/`build_model_sliced`.

### 3a. Analytical break-even tariff threshold

**`src/optimize_experiments.py:870-889`** (inside
`run_fin2_blended_bankability_sweep`):
```python
870	    solar_cf = base_scenario["solar_cf"]
871	    solar_energy_per_mw = solar_cf * 8760 / 1_000_000
872	    required_margin = base_scenario["required_margin"]
873	    # Use the 2025 time-varying CAPEX value for the analytical threshold, to
874	    # match what the LP prices in the first build year (fixes stale-scalar bug).
875	    capex_2025 = float(solar_capex_tv[years[0]])
876	
877	    results = []
878	
879	    for blend_label, (share, r_conc, r_comm) in blend_scenarios.items():
880	
881	        r_private = blended_wacc(share, r_conc, r_comm)
882	
883	        # Analytical threshold at THIS blended rate
884	        npv_energy = sum(
885	            solar_energy_per_mw / (1.0 + r_private) ** t
886	            for t in range(len(years))
887	        )
888	        threshold_usd_per_twh = capex_2025 * required_margin / (npv_energy)
889	        # (threshold in USD/TWh: capex[USD/MW]*margin / (npv_energy[TWh/MW]))
```
**One-line role**: here `required_margin` **multiplies `capex_2025`**
(the numerator), not a revenue term — the mirror-image rearrangement of
§2a's `cost <= revenue/required_margin` solved for the break-even tariff
(`capex × required_margin <= tariff × npv_energy` ⟺ `tariff_threshold =
capex × required_margin / npv_energy`). Algebraically equivalent to §2a's
LP constraint, but the code multiplies a cost term rather than dividing a
revenue term.

Two more sites in the same function, both matching §2b's division-of-revenue
form:
```python
906	            max_bankable = tariff * npv_energy / required_margin
```
```python
969	                built_with_open_gap = any(
970	                    float(pyo.value(m.solar_eaas_add[t])) > MW_EPS
971	                    and max(
972	                        0.0,
973	                        float(pyo.value(m.solar_capex_param[t]))
974	                        - tariff * solar_energy_per_mw
975	                        * float(pyo.value(m.remaining_npv_factor[t])) / required_margin,
976	                    ) > GAP_EPS_USD_PER_MW
977	                    for t in range(len(years))
978	                )
```
(line 975 is inside a post-solve reconstruction of *why* `eaas_subsidy_npv`
came out at or near zero — comment at lines 949-960 cites the exact LP
constraint it is reproducing.)

### 3b. `extract_planning_diagnostics` — two reported quantities

**`src/optimize_experiments.py:2299-2306`:**
```python
2299	        # Reported at the PRIVATE/blended rate (consistent with the LP's
2300	        # bankability constraint). Units: USD/TWh.
2301	        "bankability_threshold":
2302	            scenario["required_margin"] *
2303	            (
2304	                float(pyo.value(m.solar_capex_param[0])) /
2305	                npv_energy_priv
2306	            ),
```
**One-line role**: multiplies `capex/npv_energy` — same multiply-cost
form as §3a.

**`src/optimize_experiments.py:2338-2342`:**
```python
2338	        # Bankable CAPEX headroom at the PRIVATE/blended rate (matches the LP).
2339	        "bankable_capex_per_mw":
2340	            scenario["solar_service_tariff_usd_per_twh"] *
2341	            npv_energy_priv
2342	            / scenario["required_margin"],
```
**One-line role**: divides `tariff × npv_energy` — same divide-revenue
form as §2b, explicitly commented "matches the LP."

### 3c. Pure pass-through / traceability (no arithmetic)

Read but not used in any calculation — copied into an output row or
docstring for traceability:

- `src/optimize_experiments.py:1293`: `"required_margin": scenario.get("required_margin"),`
- `src/optimize_experiments.py:444-445`, `733`, `1129`, `1349`, `1536`:
  docstring `Parameters` listings naming `required_margin` as part of a
  passed-in `scenario`/`fin_cfg` dict, not code.

---

## 4. Comments stating an intended *meaning* (quoted, not adjudicated)

These do not execute any arithmetic themselves but are the clearest
statements in the codebase of what someone believed `required_margin`
*was for* — directly relevant to the F1 meaning question, reproduced
verbatim without resolving the conflict between them.

**`src/scenarios.py:2121`** (label on the definition site):
```python
        # ---- Required NPV margin (bankability coverage cushion, distinct from WACC)
```
— asserts **coverage cushion**, explicitly distinguished from WACC/return.

**`src/optimize_model.py:628-637`** (why it is *not* applied to EaaS solar CAPEX in the objective):
```python
628	    # EaaS (private) solar CAPEX.
629	    # MODELLING CHOICE (state explicitly in the thesis): EaaS-financed and
630	    # publicly-financed solar are priced at the SAME unit CAPEX -- they are the
631	    # same physical assets. The private cost of capital is represented NOT by an
632	    # objective-side CAPEX premium but by the bankability constraint, which
633	    # discounts EaaS revenue at the private/blended rate via remaining_npv_factor.
634	    # Applying a required_margin multiplier here would double-count: required_margin
635	    # is the bankability coverage cushion, a distinct object from the cost of
636	    # capital. The EaaS mechanism's effect in this model is therefore structural
637	    # (budget decoupling + a harder bankability test), not a cheaper sticker price.
```
— again **coverage cushion, distinct from cost of capital**; explicitly
states that applying it as a capex multiplier would double-count against
`remaining_npv_factor` (which already carries the rate).

**`src/optimize_model.py:1022-1025`** / **`src/optimize_model_sliced.py:1018-1021`**
(identical, salvage section):
```python
        # Salvage is computed on RAW capex for BOTH arms: required_margin is a
        # financing premium, not asset value. It is subtracted from the objective
        # ONLY -- the public budget constraint below continues to see gross capex,
        # because the envelope constrains cash out the door.
```
— calls it a **"financing premium"** (return-adjacent language) while
using it to justify *excluding* it from asset valuation.

**`src/scenarios.py:662-664`** (`asset_lifetimes()` docstring, same claim
restated):
```python
        required_margin is a financing premium, not asset value -- a panel
        bought with expensive capital is not worth more at resale.
```

**`scripts/03_run_ndc_eaas.py:258-268`** (a runner's own stated understanding — this one uses "hurdle rate" language directly):
```python
258	    #   required_margin — private investor NPV hurdle rate proxy.
259	    #     unconditional: 1.10  (commercial rate, ~10% hurdle)
260	    #     conditional:   1.05  (concessional blended finance, ~5% hurdle)
261	    #     Rationale: NDC 3.0 conditional finance is provided at
262	    #     concessional terms (DFI/MDB rates ~3-5% vs commercial ~8-12%).
263	    #     A 4-5 pp reduction in hurdle rate is consistent with typical
264	    #     blended finance structures. required_margin feeds directly
265	    #     into the bankability constraint:
266	    #       bankable_revenue = tariff * remaining_npv_factor / required_margin
267	    #     Lower required_margin -> higher bankable revenue -> more EaaS
268	    #     solar is deployable without gap-funding -> lower subsidy need.
```
**AMBIGUOUS, flagged as such**: this comment treats `1.10`/`1.05` as if
they were themselves rates ("~10% hurdle," "~5% hurdle," "a 4-5 pp
reduction"), i.e. reads `required_margin` as a **return requirement**
directly comparable to a percentage interest rate. This is in direct
tension with the `scenarios.py`/`optimize_model.py` comments above, which
insist it is a **coverage cushion "distinct from" and not double-counted
with** the cost of capital (`r_private`, carried separately in
`remaining_npv_factor`). The code's mechanical operation (§2a/§2b: a
divisor applied to revenue, structurally a coverage ratio, not a rate
compounding/discounting term) matches the `scenarios.py` framing more than
this comment's framing — but this report does not resolve that; both are
quoted as found.

---

## 5. Full assignment/override map — every runner, every value, and whether it is *live*

`required_margin` is only ever consumed by the LP (§2) inside a block
gated by `if tariff is not None:` — **independent of `financing_regime`'s
string value**. A runner that sets `required_margin` but never sets
`scenario["solar_service_tariff_usd_per_twh"]` writes a value the LP never
reads. Checked directly (`grep -n "solar_service_tariff_usd_per_twh"` per
file) for every one of the 13 files below.

| Script | Values set | Assignment site(s) | Tariff ever set? | LP-live? |
|---|---|---|---|---|
| `scripts/02_run_eaas.py` | `1.10` | L157 | Yes (L156, `95_000_000.0`) | **Live** |
| `scripts/03_run_ndc_eaas.py` | `1.10` / `1.05` (NDC_CASES, L279/284/289/294) | L325 | Yes (L321, `95_000_000.0`) | **Live** |
| `scripts/04_run_fin2_tariff_sweep.py` | `1.10` / `1.05` (NDC_CASES, L228/233/239/244/249) | L316 | Yes — swept (`TARIFF_SWEEP_GRID`) | **Live** |
| `scripts/07_run_gas2_eaas_gas_relief.py` | `1.10` / `1.05` (NDC_CASES, L80/82/84/86/88) | L133 (via `run_no_policy_gas_matrix_eaas`) | Yes (L134, `95_000_000`) | **Live** |
| `scripts/15_run_pol1_ndc_comparison.py` | `1.10` / `1.05` (NDC_CONFIGS, L111/119/127/135, key `eaas_margin`) | L193, only `if fin_label == "eaas"` | Yes (L194, `CANONICAL_TARIFF`), same branch | **Live** |
| `scripts/11_run_rel3_financing_frontier.py` | `1.10` / `1.05` (POLICY_CONFIGS, L133/141/149/157/165, key `eaas_margin`) | Delegated — see §1c/`optimize_experiments.py:810` | Yes (`eaas_tariff` in same POLICY_CONFIGS dicts, applied at `optimize_experiments.py:811`) | **Live** |
| `scripts/13_run_dem2_growth_gas_interaction.py` | `1.10` (FINANCING_CONFIGS, L125, `eaas` arm only) | Delegated — `optimize_experiments.py:1407` | Yes (L126, `CANONICAL_TARIFF`, same dict) | **Live** |
| `scripts/19_run_carbon_price_sweep.py` | `1.10` (all 3 arms, L64/70/76) | L111 | **No** — `grep` for `solar_service_tariff_usd_per_twh` in this file: no matches | **Inert** (`eaas_bankability`/`subsidy_limit_rule`'s tariff-gated branch never builds) |
| `scripts/18_run_pareto_frontier.py` | `1.10` / `1.05` (FINANCING_ARMS, L72/78/84/90) | L146 | **No** — no matches | **Inert** |
| `scripts/24_run_leakage_arm.py` | `1.10` | L206 (`financing_regime="public"`, L205) | **No** — no matches | **Inert** |
| `scripts/25_run_obligation_trajectory.py` | `1.10` | L203 (`financing_regime="public"`, L202) | **No** — no matches | **Inert** |
| `scripts/99c_run_genset_arm_sensitivity.py` | `1.10` (both arms, L180/182) | L230 | **No** — no matches | **Inert** |
| `scripts/17_run_monte_carlo.py` | `1.10` / `1.05` (CASES, L92/94/96/98) | L171 | **No** — no matches | **Inert** |

**6 of 13 runners write `required_margin` into a scenario that never sets
a tariff** — in every one of those, both LP constraints that read
`required_margin` (§2a, §2b) are skipped entirely (`if tariff is not
None:` is false), so the write is inert as far as the LP is concerned.
This is true regardless of `financing_regime` (three of the six —
`19`, `18`, `99c` — do set `financing_regime="eaas"` for at least one arm
and still never set a tariff).

### Values used, tallied

- **`1.0`**: two builder-level fallback defaults only (§1a) — never
  appears in any scenario dict actually built, since `load_scenario()`
  always supplies `1.10` and no runner ever writes exactly `1.0`.
- **`1.10`**: `scenarios.py`'s unconditional default (§1b); also the
  independent `run_gas_regime_ndc_matrix` function default (§1c); also
  the majority "commercial"/"unconditional" value across every runner in
  §5.
- **`1.05`**: used in 7 of the 13 runners (`03`, `04`, `07`, `15`, `11`,
  `18`, `17`) for "conditional"/concessional-finance arms — **a third
  value, confirmed present**, always paired with a comment invoking
  concessional or blended finance.
- No runner uses any value other than `1.0` / `1.10` / `1.05`.

### Which wins, under the current central config

Every runner calls `load_scenario()` first (which always sets `1.10`)
and then, in every script that touches `required_margin` at all,
immediately overwrites it via `scenario["required_margin"] = <value>`
(direct literal `1.10` in most non-EaaS/diagnostic call sites, or a
per-arm-config value of `1.10`/`1.05` in EaaS-comparison sweeps). **The
builder's own `1.0` fallback (§1a) is therefore only reachable by a
hand-built scenario dict that skips both `load_scenario()` and any
runner-level override** — no file in `scripts/` does this; every runner
inventoried here sets `required_margin` explicitly (whether or not that
write ends up LP-live, per §5) or leaves `load_scenario()`'s `1.10` in
place untouched.
