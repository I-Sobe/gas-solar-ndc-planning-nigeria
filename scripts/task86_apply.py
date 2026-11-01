"""TASK 86 Part B source edits. Exact-match replacements, each old string asserted to occur the stated number of
times. Writes docs/task_log_86_B_changes.md with exact old and new text."""
import re
from pathlib import Path
D = Path(__file__).resolve().parent.parent / "docs"
log = []

def sub(fname, old, new, count=1, why=""):
    p = D / fname
    t = p.read_text(encoding="utf-8")
    n = t.count(old)
    assert n == count, (fname, n, old[:80])
    p.write_text(t.replace(old, new), encoding="utf-8")
    log.append((fname, why, old, new, count))

PP, REG, INV = "parameter_provenance.md", "limitations_register.md", "model_capability_inventory.md"

# ---- B.1 row 10 B*
sub(PP, "Current bracket (re-run, 2026-09-25): [9,246,009,182.24 – 9,761,349,093.39] | Model dual",
    "Bracket, blended generation (re-run, 2026-09-25): [9,246,009,182.24 – 9,761,349,093.39]; bracket, vintage generation (`gas_vintage_dispatch=True`, TASK 84): [5,136,671,781 – 5,518,409,480], from `results/capital_frontier/arms_1d68be49/capital_frontier_raw.csv` (traditional regime: last positive dual at 0.5 x B*, first zero dual at 0.6 x B*), so B* is 86.2% above the vintage requirement 5,518,409,480 | Model dual",
    why="B.1 B*: vintage bracket added beside the blended one; source: capital_frontier_raw.csv rows envelope_fraction 0.5 (realised_public_spend_pv_usd 5136671781.10, dual 0.39919) and 0.6 (5518409479.67, dual 0.0), matching docs/findings_frozen.md TASK 84 block")
sub(PP, "`results/capital_frontier/capital_frontier_raw.csv` (current bracket, `envelope_fraction` 0.9/1.0 rows)",
    "`results/capital_frontier/capital_frontier_raw.csv` (blended-generation bracket, `envelope_fraction` 0.9/1.0 rows; vintage-generation bracket in `results/capital_frontier/arms_1d68be49/capital_frontier_raw.csv`, 0.5/0.6 rows)",
    why="B.1 B* locator")
sub(PP, "[9,246,009,182.24 – 9,761,349,093.39] — the current bracket the swept grid cannot resolve further, NOT a source-stated confidence interval",
    "Blended generation [9,246,009,182.24 – 9,761,349,093.39]; vintage generation [5,136,671,781 – 5,518,409,480] — in each generation the bracket the swept grid cannot resolve further, NOT a source-stated confidence interval",
    why="B.1 B* range")
# ---- row 14 storage at 2054/2045
sub(PP, "at 63,056 MWh (2054 storage capacity; 32,742 MWh at 2045)",
    "at 56,289.88 MWh (2054 storage capacity) and 24,697.69 MWh at 2045 (vintage generation, `results/baseline_sliced/timeseries__661b03c1.csv`, column `storage_capacity_mwh`; this row previously read 63,056 MWh at 2054 and 32,742 MWh at 2045, figures of earlier generations)",
    why="B.1 storage: source results/baseline_sliced/timeseries__661b03c1.csv (2045 24697.686, 2054 56289.880)")
# ---- row 15 SDR sweep
sub(PP, "(cumulative solar to 2054, 39,030.065 MW) is identical to 0.0003 MW",
    "(cumulative solar to 2054, 39,030.065 MW [VERIFY: blended generation, `results/_sdr_sweep_v3/`; no social-discount-rate arm with `gas_vintage_dispatch=True` exists (`docs/task_log_85.md`), so this figure has no vintage counterpart; the vintage central solar build is 30,938.84 MW, `docs/findings_frozen.md` TASK 84 block]) is identical to 0.0003 MW",
    why="B.1 SDR sweep: not confirmable against a vintage file; marked [VERIFY], not replaced")
# ---- register L59 table
sub(REG, "| 12% | 17.416 | 39,030.065 | 19,240.8 | 63,056 |\n",
    "| 12% | 17.416 | 39,030.065 | 19,240.8 | 63,056 |\n\n**[VERIFY: this table is a pre-epoch, blended-dispatch state (27.318 bn at 8% is the advance-convention, flat-storage objective of L88). No social-discount-rate arm with `gas_vintage_dispatch=True` exists (`docs/task_log_85.md`), so the solar-to-2045 column (18,841.7 to 19,240.8 MW) and the 63,056 MWh storage figure are not confirmed against a vintage file.]**\n",
    why="B.1 SDR sweep 19,240.8 MW: no vintage counterpart; marked [VERIFY], not replaced")

# ---- B.3 draft annotations
note = {"chapter3_1_v2": "§3.1", "chapter3_2_draft": "§3.2", "chapter3_3_draft": "§3.3"}
for f in (REG, PP, INV):
    p = D / f
    t = p.read_text(encoding="utf-8")
    for stem, sec in note.items():
        pat = re.compile(r"`(docs/)?chapters/" + stem + r"\.md(:[\d,]+)?`")
        def rep(m):
            return m.group(0) + f" [removed `009d230`; held at `a993f15`; successor `CHAPTER_3_assembled.md` {sec}]"
        ms = list(pat.finditer(t))
        for m in ms:
            log.append((f, "B.3 removed draft: annotated, not repointed (the cited text exists only in the draft)", m.group(0), rep(m), 1))
        t = pat.sub(rep, t)
    # revived runner
    pat = re.compile(r"`scripts/deferred/17_run_monte_carlo\.py`")
    for m in pat.finditer(t):
        log.append((f, "B.3 revived runner: annotated", m.group(0), m.group(0) + " [revived as `scripts/17_run_monte_carlo.py`, commit `36ddf55`, TASK 06]", 1))
    t = pat.sub(lambda m: m.group(0) + " [revived as `scripts/17_run_monte_carlo.py`, commit `36ddf55`, TASK 06]", t)
    p.write_text(t, encoding="utf-8")

# ---- B.4 header notes
GEN = ("**Result generations (TASK 86 B.4).** Two generations of stored results coexist (L142). *Blended*: dispatch without the vintage split, "
       "`gas_vintage_dispatch` absent from the arm tag or `False`; central objective 24,986,620,949.18 USD, hash `6046f759`. *Vintage*: the arm tag carries "
       "`gas_vintage_dispatch=True`; central objective 23,323,068,340.46 USD, hash `661b03c1`. A figure in this document describes the generation of the result "
       "file it is taken from. Where an entry cites a hash-keyed result file, a **Generation** line at its end says which generation those files belong to, checked "
       "against each file's arm record by `scripts/task86_generation_tags.py`; fixed-name and pattern citations are not assessed, because a fixed name is overwritten by later runs. "
       "An entry with no Generation line cites no hash-keyed file and describes structure or provenance, or its generation is not established by a file.\n\n")
sub(REG, "Status values: `LIVE` (stands, goes into the thesis)", GEN + "Status values: `LIVE` (stands, goes into the thesis)",
    why="B.4 generation marker, register header")
GENP = ("**Result generations (TASK 86 B.4).** A parameter value is generation-independent. A figure quoted from a stored result is not: two generations coexist "
        "(register L142), *blended* (`gas_vintage_dispatch` absent or `False`; central objective 24,986,620,949.18 USD) and *vintage* "
        "(`gas_vintage_dispatch=True`; 23,323,068,340.46 USD; row 29). Each result-derived figure in a row below is labelled with its generation where one is stated; "
        "an unlabelled result-derived figure predates the vintage generation (blended dispatch).\n\n")
p = D / PP
t = p.read_text(encoding="utf-8")
assert t.startswith("# Parameter provenance protocol\n\n")
p.write_text(t.replace("# Parameter provenance protocol\n\n", "# Parameter provenance protocol\n\n" + GENP, 1), encoding="utf-8")
log.append((PP, "B.4 generation marker, provenance header", "# Parameter provenance protocol\n\n", "# Parameter provenance protocol\n\n" + GENP, 1))

out = ["# TASK 86 Part B — exact changes to the source documents\n", "Generated by `scripts/task86_apply.py` (and the Generation lines by `scripts/task86_generation_tags.py --write`; see the end).\n"]
for i, (f, why, old, new, c) in enumerate(log, 1):
    out.append(f"\n### {i}. `docs/{f}`\n\n*{why}*\n\n**old:** `{old}`\n\n**new:** `{new}`\n")
(D / "task_log_86_B_changes.md").write_text("\n".join(out), encoding="utf-8")
print(len(log), "changes")
