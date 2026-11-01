# output_naming.md — lifting the hash8 output-naming scheme into a shared helper

Per TASK.md, `docs/f3_identification.md`'s "Path-length fix" section was
read and NOT re-inventoried; site line numbers below were re-confirmed
directly against the current files where they matter for an edit (some
had drifted since that doc was written, from unrelated intervening
tasks).

---

## 1. The shared helper

**New module: `src/output_naming.py`.** Not added to `src/utils.py`
(the existing general-helpers module), because that module's own
declared docstring scope explicitly excludes this: `utils.py`'s "Scope"
list is numerical validation / interpolation / unit conversion / index
helpers, and its "Non-scope (by design)" list names "High-level I/O
orchestration" — which is exactly what an `INDEX.md` appender is. A new,
narrowly-scoped module keeps that discipline intact rather than
widening `utils.py`'s stated scope to fit this task.

**API** (semantics copied exactly from `26_run_f3_identification.py`'s
original implementation — same hash, same length, same index format):

```python
def hash8_of(arm_tag: str) -> str:
    """First 8 hex chars of SHA-1(arm_tag)."""
    return hashlib.sha1(arm_tag.encode("utf-8")).hexdigest()[:8]

def short_tag(arm_tag: str, prefix: str) -> str:
    """f"{prefix}_{hash8}" -- deterministic, for a DIRECTORY component."""
    return f"{prefix}_{hash8_of(arm_tag)}"

def append_index(results_dir, hash8: str, arm_tag: str) -> None:
    """Append one "- `hash8` -> `arm_tag`" line to results_dir/INDEX.md.
    Append-only, deduplicated."""
    ...
```

`hash8_of` is exposed separately from `short_tag` because Pattern B
(filename component) needs the bare hash, not the `prefix_hash8` shape
`short_tag` returns (Pattern B's filename shape is `<prefix>__<hash8>.
<ext>`, double-underscore, matching the existing `__`-joined tag
convention already used in every runner touched — not `short_tag`'s
single-underscore `<prefix>_<hash8>`, which is Pattern A's directory
shape).

**Verification (bounded, per TASK.md):**

```python
from src.output_naming import short_tag
t1 = short_tag('some_arm_tag_a', 'prefix')       # -> 'prefix_680c1a9e'
t2 = short_tag('some_arm_tag_b', 'prefix')       # -> 'prefix_018b93bc'
t1_again = short_tag('some_arm_tag_a', 'prefix') # -> 'prefix_680c1a9e'
assert t1 != t2
assert t1 == t1_again
```

Passed. Also cross-checked `hash8_of()` against a manual
`hashlib.sha1(...).hexdigest()[:8]` computation for the same string —
identical, confirming the helper reproduces the exact algorithm
`26` used before refactoring (see below).

`python3 -m py_compile src/output_naming.py` — passes.

### Can `26` be refactored onto the helper without changing any existing hash?

**Yes — done.** `26`'s original inline code was:

```python
hash8 = hashlib.sha1(arm_tag.encode("utf-8")).hexdigest()[:8]
short_name = f"{NDC_CAP_SCENARIO}_{args.capital_case}_{hash8}"
case_dir = RESULTS_DIR / short_name
...
index_path = RESULTS_DIR / "INDEX.md"
index_line = f"- `{hash8}` -> `{arm_tag}`\n"
existing = index_path.read_text() if index_path.exists() else ""
if index_line not in existing:
    with open(index_path, "a") as f:
        f.write(index_line)
```

This is `short_tag(arm_tag, f"{NDC_CAP_SCENARIO}_{args.capital_case}")`
and `append_index(RESULTS_DIR, hash8, arm_tag)` verbatim — identical
SHA-1 call, identical `[:8]` slice, identical `f"{prefix}_{hash8}"`
shape, identical index-line string. Confirmed by direct comparison for a
sample arm tag:

```
old: ndc3_conditional_moderate_33e46b00
new: ndc3_conditional_moderate_33e46b00   (MATCH)
```

Refactored `scripts/26_run_f3_identification.py`: `import hashlib`
removed (its only use was this computation); `from src.output_naming
import short_tag, hash8_of, append_index` added; the six-line inline
block replaced with the three-call equivalent. No hash, filename, or
`INDEX.md` line format changes as a result.

---

## 2. Applied to the three failing runners

All three write the FULL tag back into every output file (JSON:
`"arm_tag"` key; CSV: `# arm_tag: <tag>` as the file's first line, written
before the `to_csv` body) and append to `RESULTS_DIR/INDEX.md`.

### `01b_run_baseline_sliced.py` — Pattern B (filename component)

`RESULTS_DIR = results/baseline_sliced/`. Tag built once at the top of
`main()`; `hash8 = hash8_of(tag)` and `append_index(RESULTS_DIR, hash8,
tag)` called immediately after.

| Old filename | New filename | Full-tag write-back |
|---|---|---|
| `summary__{tag}.json` | `summary__{hash8}.json` | `summary["arm_tag"] = tag` |
| `diagnostics__{tag}.json` | `diagnostics__{hash8}.json` | `diag_out["arm_tag"] = tag` |
| `timeseries__{tag}.csv` | `timeseries__{hash8}.csv` | `# arm_tag: {tag}` header line |
| `slice_detail__{tag}.csv` | `slice_detail__{hash8}.csv` | `# arm_tag: {tag}` header line |

This was the confirmed failure site (`:530` in TASK.md's report, on the
`diagnostics__{tag}.json` write, after a successful solve that had
already printed 21.475 bn / 44,377 MWh).

### `02_run_ndc_caps.py` — Pattern A (directory component)

`RESULTS_DIR = results/ndc/`. Per NDC-case/voll-case iteration: `arm_tag`
built (same content as the old inline `case_dir` expression, now also
kept as a named variable); `hash8 = hash8_of(arm_tag)`;
`append_index(RESULTS_DIR, hash8, arm_tag)`; `case_dir = RESULTS_DIR /
short_tag(arm_tag, c)` (`c` = the NDC case label, e.g.
`ndc2_unconditional` — the "case_or_prefix" TASK.md's Pattern A
description names).

| Old directory | New directory | Full-tag write-back |
|---|---|---|
| `{c}_{voll_case}_..._{hydro_case}/` | `{c}_{hash8}/` | — |

Files inside `case_dir` keep their existing plain names
(`summary.json`, `diagnostics.json`, `timeseries.csv` — already short,
Pattern A's length problem is the directory, not these), but each now
also carries the tag:

| File | Full-tag write-back |
|---|---|
| `summary.json` | `"arm_tag": arm_tag` added to the dict |
| `diagnostics.json` | `{"arm_tag": arm_tag, **json_safe(out["diagnostics"])}` |
| `timeseries.csv` | `# arm_tag: {arm_tag}` header line |

This runner "survived this sweep only because its tag is marginally
shorter" (TASK.md) — not touched by `tariff_case` at all (the arm is
inert here, per `docs/tariff_arm.md` step 2, so it was never added to
this tag), but would fail on the next arm added regardless.

### `03_run_ndc_eaas.py` — Pattern A (directory component)

`RESULTS_DIR = results/ndc_eaas/`. Same shape as `02`, one difference:
this runner's `arm_tag` DOES include `{args.tariff_case}` (EaaS is
active here, so the arm is not inert — per `docs/tariff_arm.md`, this is
one of the two runners the tariff arm was originally wired into). This
was the confirmed failure site (`case_dir.mkdir()`, after solving all
four NDC cases).

| Old directory | New directory | Full-tag write-back |
|---|---|---|
| `{c}_{voll_case}_..._{hydro_case}_{tariff_case}/` | `{c}_{hash8}/` | — |

| File | Full-tag write-back |
|---|---|
| `summary.json` | `"arm_tag": arm_tag` added to the dict |
| `diagnostics.json` | `{"arm_tag": arm_tag, **json_safe(out["diagnostics"])}` |
| `timeseries.csv` | `# arm_tag: {arm_tag}` header line |

### Files touched, step 1+2

`src/output_naming.py` (new), `scripts/26_run_f3_identification.py`,
`scripts/01b_run_baseline_sliced.py`, `scripts/02_run_ndc_caps.py`,
`scripts/03_run_ndc_eaas.py`. `python3 -m py_compile` passes on all
five.

---

## 3. The other Pattern B runners — live/dead classification (report only, NOT fixed)

`load_scenario()` currently has sixteen required (no-default) keys:
`tariff_case`, `peak_multiple_case`, `reserve_margin`, `storage_cc_case`,
`gas_build_case`, `gas_availability_case`, `gas_retirement_case`,
`gas_capex_case`, `ccgt_efficiency_case`, `capital_recovery_case`,
`dscr_case`, `us_inflation`, `capital_treatment_case`,
`concessional_envelope_case`, `conversion_case`, `hydro_case`. A runner
is LIVE iff its `load_scenario()` call supplies all sixteen. Checked
directly against each file's actual call (not assumed from a prior
task's notes):

| Runner | Live? | Missing required keys |
|---|---|---|
| `01_run_baseline.py` | **LIVE** | none |
| `02_run_eaas.py` | **DEAD** | `conversion_case`, `hydro_case` |
| `22_salvage_sensitivity.py` | **LIVE** | none |
| `23_run_selfgen_frontier.py` | **LIVE** | none |
| `24_run_leakage_arm.py` | **DEAD** | `tariff_case`, `conversion_case`, `hydro_case` |
| `25_run_obligation_trajectory.py` | **DEAD** | `tariff_case`, `conversion_case`, `hydro_case` |
| `99c_run_genset_arm_sensitivity.py` | **DEAD** | `tariff_case`, `conversion_case`, `hydro_case` |

**Live (a follow-up should wire these onto the helper next):**
`01_run_baseline.py`, `22_salvage_sensitivity.py`,
`23_run_selfgen_frontier.py`. All three build a filename directly from a
`__`-joined tag of fifteen or sixteen arms (`01`/`22`: sixteen, tag
excludes `tariff_case` since the arm is inert there per
`docs/tariff_arm.md`; `23`: sixteen, same reason) — the same failure
mode as `01b`, not yet tripped only because these three haven't been run
since the `tariff_case` migration completed the required-arg list.

**Dead on other grounds (not a path-length follow-up target until
revived independently):** `02_run_eaas.py`, `24_run_leakage_arm.py`,
`25_run_obligation_trajectory.py`, `99c_run_genset_arm_sensitivity.py`.
All four were already dead before this task, on required-arg omissions
unrelated to path length — `02_run_eaas.py` per
`docs/tariff_arm.md`'s Step 2 table; `24`/`25`/`99c` per the required-arg
inventory in `docs/numerical_hygiene_0_4.md`'s Item C equivalence check
and the tariff-arm repair task. Wiring the output-naming helper into a
runner that cannot currently reach its output-write code at all would
be dead code on dead code; per TASK.md, not done here.

Per TASK.md, none of the seven above were wired onto the helper in this
task — `01b`, `02_run_ndc_caps.py` and `03` were the sweep's actual
blockers and are what this task fixes.

---

## Do-NOT checklist

- No arm, constraint, or cost term changed.
- `26`'s existing hashes unchanged (verified by direct comparison, see
  above) — the refactor is a pure code-motion.
- No arm shortened or dropped from any tag — every tag string is
  unchanged in content; only its role (embedded in a path vs. hashed for
  one) changed. The full tag is still written into every output file.
