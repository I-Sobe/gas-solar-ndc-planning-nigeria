# d6_storage_guard.md — D6: storage in-horizon retirement guard is stale

## Step 0 — Confirm before changing (locate and quote, NO edit)

### (a) The guard in `solve_model()`, in full

`src/optimize_model.py:1859-1893` (as it stood before this task):

```python
    # ---- Post-solve guard: in-horizon retirement (plan 2.5)
    # Storage life (15 yr) is shorter than the horizon (21 yr). Any vintage
    # built early enough to retire before the horizon ends is still dispatched
    # by the model, and no replacement capex is charged. Inert as at Phase 2.5
    # (first storage build 2038) but Phases 2.6/2.7 may shift the optimum
    # earlier. This warns rather than raises so that exploratory runs proceed.
    try:
        _lives = scenario.get("asset_lifetimes", {})
        _T = list(range(len(scenario["years"])))
        _T_last = _T[-1]
        _checks = [
            ("storage", _lives.get("storage"), lambda t: pyo.value(model.storage_add[t])),
            ("solar",   _lives.get("solar"),
             lambda t: pyo.value(model.solar_public_add[t]) + pyo.value(model.solar_eaas_add[t])),
            ("gas",     _lives.get("gas"),   lambda t: pyo.value(model.gas_add[t])),
        ]
        for _name, _life, _getter in _checks:
            if not _life:
                continue
            for t in _T:
                if (_T_last - t + 1) > float(_life) and _getter(t) > 1e-6:
                    print(
                        f"  WARNING [plan 2.5]: {_name} vintage {scenario['years'][t]} "
                        f"retires within the horizon (life {_life:.0f} yr < "
                        f"{_T_last - t + 1} yr served). No replacement capex is "
                        f"charged and the asset is still dispatched. Salvage for "
                        f"this vintage is 0. Replacement capex is now required."
                    )
                    break
    except AttributeError as _e:
        print(f"  NOTE [plan 2.5]: retirement check skipped -- scenario not passed to "
              f"solve_model, or model lacks an expected variable ({_e})")
    except Exception as _e:
        print(f"  WARNING [plan 2.5]: retirement check FAILED unexpectedly ({_e}). "
              f"This is a bug, not a skip.")
```

Note the guard is NOT storage-only, despite D6's title and the comment
above it: `_checks` loops over `storage`, `solar`, AND `gas` with the
identical condition and message shape. The fix in step 1 below preserves
this three-channel structure.

### (b) Confirmation that BOTH objective branches charge storage replacement

**`pre_f1_overnight`** — `src/optimize_model_sliced.py:1423-1441` (annual
builder mirror: `src/optimize_model.py:1194-1212`):

```python
m.cost_replacement_capex_npv = pyo.Expression(
    expr=(
        pyo.quicksum(
            m.DF[rebuild_t]
            * (m.solar_public_add[build_t] + m.solar_eaas_add[build_t])
            * m.solar_capex_param[rebuild_t]
            for build_t in T
            for rebuild_t in _rebuild_years(build_t, lives["solar"])
        )
        + pyo.quicksum(
            m.DF[rebuild_t] * m.storage_add[build_t] * econ["STORAGE_COST_PER_MWH"]
            for build_t in T
            for rebuild_t in _rebuild_years(build_t, lives["storage"])
        )
        + pyo.quicksum(
            m.DF[rebuild_t] * m.gas_add[build_t] * scenario["gas_capex_per_mw"]
            for build_t in T
            for rebuild_t in _rebuild_years(build_t, lives["gas"])
        )
    )
)
```

Confirmed entering the objective: `optimize_model_sliced.py:1846`
(annual: `optimize_model.py:1613`), inside the `pre_f1_overnight`
`system_cost_npv` sum:

```python
    if capital_treatment_case == "pre_f1_overnight":
        system_cost_npv = (
            gas_opex_npv
            + public_solar_capex_npv
            + eaas_solar_capex_npv
            + gas_capex_npv
            + storage_capex_npv
            + m.cost_replacement_capex_npv
            + selfgen_npv
            ...
```

**`f1_tranche`** — `src/optimize_model_sliced.py:1694-1716` (annual
builder mirror: `src/optimize_model.py:1480-1502`):

```python
m.cost_rebuild_capital_charge_npv = pyo.Expression(
    expr=(
        pyo.quicksum(
            _crf_comm["solar"] * _df_window_comm["solar"][rebuild_t]
            * (m.solar_public_add[build_t] + m.solar_eaas_add[build_t])
            * m.solar_capex_param[rebuild_t]
            for build_t in T
            for rebuild_t in _rebuild_years(build_t, _lives["solar"])
        )
        + pyo.quicksum(
            _crf_comm["storage"] * _df_window_comm["storage"][rebuild_t]
            * m.storage_add[build_t] * econ["STORAGE_COST_PER_MWH"]
            for build_t in T
            for rebuild_t in _rebuild_years(build_t, _lives["storage"])
        )
        + pyo.quicksum(
            _crf_comm["gas"] * _df_window_comm["gas"][rebuild_t]
            * m.gas_add[build_t] * scenario["gas_capex_per_mw"]
            for build_t in T
            for rebuild_t in _rebuild_years(build_t, _lives["gas"])
        )
    )
)
```

Confirmed entering the objective: `optimize_model_sliced.py:1726-1732`
(annual: `optimize_model.py:1505-1511`), inside
`_capital_charge_total`:

```python
        _capital_charge_total = (
            m.cost_public_solar_capital_charge_npv
            + m.cost_eaas_solar_capital_charge_npv
            + m.cost_gas_capital_charge_npv
            + m.cost_storage_capital_charge_npv
            + m.cost_rebuild_capital_charge_npv
        )
```

which is (elsewhere, unquoted here) summed into
`m.capital_charge_total_npv`, which feeds `system_cost_npv` under
`f1_tranche`.

**Confirmed for BOTH builders.** `optimize_model.py` (annual) carries its
own byte-identical `_rebuild_years` / `cost_replacement_capex_npv` /
`cost_rebuild_capital_charge_npv` trio (lines 1137, 1194, 1331, 1480) —
`grep -n "_rebuild_years\|cost_replacement_capex_npv\|
cost_rebuild_capital_charge_npv" src/optimize_model.py` confirms this.
`solve_model()` is builder-agnostic (it only reads `model.storage_add[t]`
etc., which exist under either builder), so both mechanisms are in scope
for the guard.

### (c) Arithmetic: is there a vintage where the guard fires but no rebuild is generated?

`_rebuild_years(build_t, life)` (identical in all four copies —
`optimize_model.py:1137,1331`; `optimize_model_sliced.py:1366,1552`):

```python
def _rebuild_years(build_t, life):
    life_int = int(round(float(life)))
    k = 1
    while build_t + k * life_int <= T_last:
        yield build_t + k * life_int
        k += 1
```

Its first (`k=1`) term is yielded **iff** `build_t + life_int <= T_last`.

The guard's fire condition is `(T_last - t + 1) > life` (the RAW,
unrounded `life`, not `life_int`).

**Rearranging the guard condition** (`T_last`, `t` integers):
`T_last - t + 1 > life` ⟺ `T_last - t + 1 >= life + 1` (only if `life` is
an integer — see below) ⟺ `T_last - t >= life` ⟺ `t + life <= T_last`.

**Rearranging the generator's k=1 condition:**
`t + life_int <= T_last`.

**These two conditions are IDENTICAL whenever `life_int == life`, i.e.
whenever `life` is a whole number.** Checked every lifetime value this
project actually supplies anywhere in `src/` or `scripts/`
(`asset_lifetimes()`: `{"solar": 30.0, "storage": 15.0, "gas": 30.0}`;
`asset_lifetime_sweep()`: `solar [25.0, 30.0, 35.0]`, `storage [12.0,
15.0, 18.0]`; `scripts/22_salvage_sensitivity.py`'s
`lifetime_arms()`: `1.0`/`25.0`/`35.0`/`12.0`/`18.0`) — **every single
one is a whole-number float.** No fractional life exists anywhere in the
codebase today (`grep` for `asset_lifetimes` assignments confirms this;
none is hand-constructed with a fractional value in any script).

Given `life` is always integer-valued in every reachable path, `t + life
<= T_last` (the guard's rearranged condition) and `t + life_int <=
T_last` (the generator's k=1 condition) are the exact same test.
**Therefore: whenever the guard's condition is true, `_rebuild_years`'s
first term is DEFINITELY yielded — the generator can never yield nothing
while the guard fires, for any vintage in any scenario this project
actually runs.**

**Answer to (c): NO — no such vintage exists** in the reachable codebase.
The guard's condition, as currently written, is logically equivalent to
"a rebuild is generated" (not merely correlated with it) given the
lifetimes this project uses.

**Caveat, stated for completeness (not a live defect).** The two
conditions are provably identical ONLY because `life` is always an
integer here. If a scenario ever supplied a NON-integer life (nothing in
`src/`/`scripts/` does; `asset_lifetimes` is a plain dict read via
`.get()`, so nothing prevents a hand-built one), `round()` could push
`life_int` above the raw `life` (e.g. `life = 15.5` rounds to `16`, by
Python's round-half-to-even), opening a one-year window where the guard
fires (`X > 15.5`, i.e. `X >= 16`) but the generator's first term needs
`X >= 17`. This is a genuine theoretical gap in the guard's arithmetic,
but it is NOT reachable by anything in this codebase today, so it does
not satisfy step 0(c)'s "found any such vintage" — nothing here trips it.
The rewritten guard in step 1 closes this gap anyway (by replicating
`_rebuild_years`'s own rounding rather than re-deriving an algebraically
equivalent-only-for-integers condition), so the caveat is moot after
this task regardless.

**Not stopping.** Per TASK.md, step 0 only requires a STOP if (c) finds
an actual vintage with the guard's condition true and no rebuild
generated. None was found. Proceeding to step 1.

---

## Step 1 — Rewrite

Implemented in `src/optimize_model.py`. Summary (full diff in the
working tree):

- The guard now computes the SAME rebuild years the builder's
  `_rebuild_years(build_t, life)` would (replicated inline, since that
  helper is a closure local to `build_model`/`build_model_sliced` and not
  reachable from `solve_model()`; comment states this and names all four
  source copies to keep in sync with).
- If the rebuild list is non-empty (the case for every vintage this
  project can currently produce, per step 0c): prints an INFORMATIONAL
  note (not a warning) stating the vintage retires at year Y and is
  rebuilt at year(s) Z, with replacement capex charged in whichever
  branch is active. Message no longer says capex is uncharged.
- If the rebuild list is EMPTY despite the guard's condition being true
  (not reachable today, per step 0c, but now actually tested rather than
  assumed): this is the real D6-shaped defect, and now raises
  `RuntimeError` after the diagnostic try/except block (not swallowed by
  its catch-all), rather than a printed warning that "warns rather than
  raises so exploratory runs proceed" — an uncompensated in-horizon
  retirement is exactly the kind of silent-wrong-answer risk this
  project's standing correction discipline does not let pass with a mere
  print.
- Kept as three-channel (storage/solar/gas), matching the existing
  `_checks` structure — not narrowed to storage-only, since TASK.md's
  step 1 preferred fix says "for each channel."

## Step 2 — Register updates

`docs/limitations_register.md` — new entry **L18** added (the register
uses `Lnn` IDs; `D6` is `merged_correction_plan_v2.md`'s docket numbering
and has no prior register entry — L18's "What" cross-references D6's
original text). Status `RESOLVED, retained for the record`. Records:
charged in the annual branch since Phase 2.6, in the tranche branch since
F2, guard message stale for two phases (written for plan 2.5, before the
rebuild mechanism existed; never updated after 2.6 or F2 added it), and
states plainly that the buffer-year concern D6 described no longer
applies. Also amended L14 (`_rebuild_years` duplication) to note the
guard rewrite added a fifth copy of that helper's arithmetic to keep in
sync.

**Correction to TASK.md's file citation.** TASK.md named
`docs/phase_2_8_conversion.md` as carrying the "still fires under
f1_tranche" note to correct. Checked — that file contains no reference
to the plan-2.5 guard, replacement capex, or storage life anywhere
(`grep` for all of those terms returns nothing). The actual note is in
**`docs/f1_implementation.md`**, in two places:

1. The STEP 0b implementation note (pre-F1, before the rebuild mechanism
   existed under `f1_tranche`): "Whether it still fires under F1, and
   what its message should say once salvage/replacement are gone, is
   deferred ... Flagged for the F1 implementation turn."
2. A dedicated section, "## Storage post-solve guard under `f1_tranche`"
   (written during the F1 implementation turn once `f1_tranche` existed):
   "The guard still fires under `f1_tranche`, unchanged, for the same
   reason it fires under `pre_f1_overnight`. Not fixed, per the task's
   instruction to report only."

Both are accurate as far as they go (the guard DOES still fire) but
neither records that the guard's MESSAGE is wrong — exactly the gap
TASK.md described, just in the other file. Both corrected in
`docs/f1_implementation.md`, with a pointer to L18.

## Verification

`python3 -m py_compile src/optimize_model.py` — passes. No solver, no
runner, no build call, no scratchpad script was run, per TASK.md. Zeus
runs both regression arms.

## Files touched

- `src/optimize_model.py` — `solve_model()`'s guard rewritten.
- `docs/limitations_register.md` — L18 added; L14 amended.
- `docs/f1_implementation.md` — both stale notes corrected (see above;
  TASK.md's `docs/phase_2_8_conversion.md` citation was itself
  incorrect — that file was not touched, as there was nothing in it to
  correct).
- `docs/d6_storage_guard.md` — this report.
