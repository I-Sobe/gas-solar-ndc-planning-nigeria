#!/usr/bin/env python3
"""Generate thesis appendices A-D (docs/chapters/appendix_{a,b,c,d}.md).

TASK 38. Deterministic: no timestamps and no HEAD hash in the output (it
records the git blob hash of each source instead), so re-running on an
unchanged source reproduces the output byte for byte and a diff after a
register change shows only what changed.  No solve; reads documents only.

    python scripts/build_appendices.py [--check]

  A  parameter provenance          <- docs/parameter_provenance.md (verbatim)
  B  register: entries carrying live content (LIVE + SPLIT)
                                   <- docs/limitations_register.md (verbatim entries)
  C  register: wholly closed entries (CLOSED)
                                   <- docs/limitations_register.md (verbatim entries)
  D  reproducibility record        <- docs/model_capability_inventory.md (non-arm
                                      section, verbatim), scripts/run_phase_b.sh,
                                      environment_linux.yml, git tags, the results
                                      manifest header, logs/phase_b/ headers

Register classification
-----------------------
The register's own Status lines are not uniform (twelve begin with a word
outside LIVE/RESOLVED/PENDING), so the class of each entry is taken from the
TASK 27 triage table (docs/task_log_27d_register_triage.md, L1-L73) with the
documented amendments below, and entries added after the triage are
classified by AMENDMENTS or, failing that, by the first word of their Status
line.  Anything classified by the fallback is listed in the appendix header
as AUTO-CLASSIFIED so a human looks at it.  B and C partition the register:
every entry is in exactly one, so len(B) + len(C) == total.  SPLIT entries
(closed and live parts) are placed in B, the live side, and marked.
"""
import hashlib, importlib.metadata as md, re, subprocess, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "chapters"

REGISTER = DOCS / "limitations_register.md"
TRIAGE = DOCS / "task_log_27d_register_triage.md"
PROVENANCE = DOCS / "parameter_provenance.md"
INVENTORY = DOCS / "model_capability_inventory.md"
MANIFEST = DOCS / "results_manifest_phase_b_start.md"
DRIVER = ROOT / "scripts" / "run_phase_b.sh"
ENVFILE = ROOT / "environment_linux.yml"

# Changes to the TASK 27 triage classes, each with its source.
AMENDMENTS = {
    "L10": ("CLOSED", "TASK 28 Part E.1"),
    "L53": ("CLOSED", "TASK 28 Part E.1"),
    "L45": ("SPLIT", "TASK 28 Part E.1"),
    "L64": ("SPLIT", "TASK 28 Part E.1"),
    "L34": ("CLOSED", "TASK 29"),
    "L43": ("CLOSED", "TASK 29"),
    "L74": ("LIVE", "TASK 30"),
    "L75": ("CLOSED", "TASK 32"),
    "L76": ("LIVE", "TASK 36"),
    "L92": ("SPLIT", "TASK 86 close-out"),
    "L86": ("SPLIT", "TASK 86 close-out"),
    "L104": ("SPLIT", "TASK 86 close-out"),
    "L103": ("CLOSED", "TASK 86 close-out"),
}
CLOSED_WORDS = {"RESOLVED", "CLOSED", "FIXED", "REPAIRED", "DECLARED", "CORRECTED"}


def blob(path):
    return subprocess.check_output(["git", "hash-object", str(path)], cwd=ROOT, text=True).strip()[:12]


def parse_register():
    lines = REGISTER.read_text().split("\n")
    heads = [i for i, l in enumerate(lines) if re.match(r"^## L\d+\b", l)]
    entries = []
    for k, i in enumerate(heads):
        j = heads[k + 1] if k + 1 < len(heads) else len(lines)
        # entries end at the next entry heading; strip the trailing template
        # heading ("## Lnn") block if it falls inside this span
        block = lines[i:j]
        # the register's "## Entry template" section (with a fenced "## Lnn"
        # example) sits inside the span of L7; cut it, and the rule above it
        for t, l in enumerate(block):
            if l.startswith("## Entry template"):
                block = block[:t]
                break
        while block and (not block[-1].strip() or block[-1].strip() == "---"):
            block.pop()
        m = re.match(r"^## (L\d+)\s*[—-]\s*(.*)", block[0])
        status = ""
        for l in block:
            sm = re.match(r"^\s*>?\s*\*\*Status:?\*\*:?\s*(.*)$", l)
            if sm:
                status = sm.group(1).strip()
                break
        entries.append({"id": m.group(1), "title": m.group(2).strip(), "status": status, "lines": block})
    return entries


def triage_classes():
    cls = {}
    for l in TRIAGE.read_text().split("\n"):
        m = re.match(r"^\| (L\d+) \|.*\| \*\*(LIVE|CLOSED|SPLIT|STALE)\*\* \|", l)
        if m:
            cls[m.group(1)] = m.group(2)
    return cls


def classify(entries):
    tri = triage_classes()
    out, auto = {}, []
    for e in entries:
        eid = e["id"]
        if eid in AMENDMENTS:
            out[eid] = AMENDMENTS[eid][0]
        elif eid in tri and tri[eid] != "STALE":
            out[eid] = tri[eid]
        else:
            w = re.sub(r"[^A-Za-z]", "", (e["status"].replace("~~", "").split() or [""])[0]).upper()
            out[eid] = "CLOSED" if w in CLOSED_WORDS else "LIVE"
            auto.append(eid)
    return out, auto


def demote(lines, n=1):
    return ["#" * n + l if l.startswith("#") else l for l in lines]


def header(title, blurb, src):
    return [f"# {title}", "", blurb, "", f"*Generated by `scripts/build_appendices.py` from `{src}` "
            f"(git blob `{blob(ROOT / src)}`). Regenerate with `python scripts/build_appendices.py`; do not edit by hand.*", ""]


def appendix_a():
    src = PROVENANCE.relative_to(ROOT)
    body = PROVENANCE.read_text().rstrip("\n").split("\n")
    out = header("Appendix A — Parameter provenance",
                 "The provenance ledger: for every parameter whose value the model takes from outside the code, its source, "
                 "locator, range and status (SOURCED / DERIVED / ASSUMED / PENDING), with the rule that produced the ledger. "
                 "Reproduced verbatim; headings are demoted one level.", str(src))
    return "\n".join(out + demote(body) + [""])


def register_appendix(which, entries, cls, auto):
    total = len(entries)
    counts = Counter(cls.values())
    if which == "B":
        sel = [e for e in entries if cls[e["id"]] in ("LIVE", "SPLIT")]
        title = "Appendix B — The limitations register: entries carrying live content"
        blurb = ("Every register entry that stands as a limitation of the reported results (class LIVE), plus the entries that "
                 "carry both a closed and a live part (class SPLIT, marked; their closed parts are described within the entry, "
                 "and the entry is not repeated in Appendix C). Entries wholly closed are in Appendix C; the two appendices "
                 "partition the register.")
    else:
        sel = [e for e in entries if cls[e["id"]] == "CLOSED"]
        title = "Appendix C — Limitations resolved during the study"
        blurb = ("The register entries wholly closed during the study (class CLOSED): defects found and engineered away, "
                 "retained as evidence about process rather than as caveats on a result. Struck-through text inside an "
                 "entry is the superseded original, kept deliberately.")
    out = header(title, blurb, str(REGISTER.relative_to(ROOT)))
    out += [f"**Entries in this appendix: {len(sel)} of {total} in the register.**  "
            f"Register classes: LIVE {counts['LIVE']}, SPLIT {counts['SPLIT']}, CLOSED {counts['CLOSED']} "
            f"(total {total}); B = LIVE + SPLIT = {counts['LIVE'] + counts['SPLIT']}, C = CLOSED = {counts['CLOSED']}.",
            "", "Classification: the TASK 27 triage (`docs/task_log_27d_register_triage.md`) as amended: "
            + "; ".join(f"{k} -> {v[0]} ({v[1]})" for k, v in AMENDMENTS.items()) + ".", ""]
    if auto:
        out += [f"**AUTO-CLASSIFIED (not in the triage or the amendment list; classed from the first word of the Status line, "
                f"check these): {', '.join(auto)}.**", ""]
    out += ["| id | class | title |", "|---|---|---|"]
    for e in sel:
        out.append(f"| {e['id']} | {cls[e['id']]} | {e['title'].replace('|', '/')} |")
    out += ["", "---", ""]
    for e in sel:
        block = list(e["lines"])
        block[0] = "#" + block[0]  # '## L..' -> '### L..'
        if cls[e["id"]] == "SPLIT":
            block.insert(1, "")
            block.insert(2, "**[SPLIT: this entry carries a closed part and a live part; see its Status.]**")
        out += block + ["", "---", ""]
    return "\n".join(out)


def section(text, start_pat, end_pat):
    lines = text.split("\n")
    s = next(i for i, l in enumerate(lines) if re.match(start_pat, l))
    e = next(i for i in range(s + 1, len(lines)) if re.match(end_pat, lines[i]))
    return lines[s:e]


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def appendix_d():
    out = header("Appendix D — Reproducibility record",
                 "How the reported results were produced: the tagged states, the Phase B run sequence with each item's runner, solve "
                 "count, builder and log header, the standard arm set, the software environment, the results manifest, and the model's "
                 "constants and defaults that sit outside the arm system.", "scripts/run_phase_b.sh")
    # D.1 tags
    out += ["## D.1 Tagged states", "", "| tag | commit | date | subject |", "|---|---|---|---|"]
    for t in git("tag", "--sort=creatordate").split("\n"):
        if not t:
            continue
        sha = git("rev-list", "-n", "1", t)[:7]
        d = git("log", "-1", "--format=%ad", "--date=short", t)
        s = git("log", "-1", "--format=%s", t).replace("|", "/")[:90]
        out.append(f"| `{t}` | `{sha}` | {d} | {s} |")
    out += ["", "`phase-b-start` marks the state before the Phase B re-runs. The commit recorded per item in D.2 is the HEAD of that item's first "
            "surviving log. One driver pass on 2026-10-19 overwrote the earlier per-item logs; `logs/` is not tracked, so the earlier headers are "
            "unrecoverable. D.2 therefore records the blended generation only: the vintage generation of 2026-10-23 has no `logs/phase_b` entries.", ""]
    # D.2 Phase B sequence
    drv = DRIVER.read_text()
    items = re.findall(r'banner "ITEM (\d+) -- ([^"]*)"', drv)
    out += ["## D.2 The Phase B run sequence", "",
            "Driver: `scripts/run_phase_b.sh` (`--only N`, `--from/--to`, `--dry-run`). Builder: `sliced` if the runner builds with "
            "`build_model_sliced` or passes `builder=\"sliced\"` to a shared function; `none` for a runner that builds no model.", "",
            "| item | description (driver banner) | runner builder | log files | commit at first log | exit |", "|---|---|---|---|---|---|"]
    for n, banner in items:
        runners = re.findall(r"\b(\d+[a-z]?_[A-Za-z0-9_]+\.py)\b", banner)
        bld = []
        for r in runners:
            p = ROOT / "scripts" / r
            if p.exists():
                s = p.read_text()
                bld.append("sliced" if ('build_model_sliced' in s or 'builder="sliced"' in s) else "none")
        builder = "/".join(sorted(set(bld))) if bld else "n/a (shell)"
        logs = sorted((ROOT / "logs" / "phase_b").glob(f"{int(n):02d}_*.log"))
        head = exitc = "-"
        if logs:
            txt = logs[0].read_text(errors="replace")
            m = re.search(r"^HEAD\s*:\s*([0-9a-f]{7})", txt, re.M)
            head = f"`{m.group(1)}`" if m else "-"
            ex = re.findall(r"exit status: (\d+)", "".join(l.read_text(errors="replace") for l in logs))
            exitc = ",".join(sorted(set(ex))) if ex else "-"
        out.append(f"| {n} | {banner.replace('|', '/')} | {builder} | {len(logs)} | {head} | {exitc} |")
    out += [""]
    # D.3 standard arm set
    arms = re.findall(r'^(C_[A-Z_]+)="([^"]*)"', drv, re.M)
    out += ["## D.3 The standard arm set", "", "Every item runs the central configuration below unless its row says otherwise "
            "(`scripts/run_phase_b.sh`, `C_*`).", "", "| driver variable | value |", "|---|---|"]
    out += [f"| `{k}` | `{v}` |" for k, v in arms]
    out += [""]
    # D.4 environment
    out += ["## D.4 Software environment", "", "The environment the results were produced in matches `environment_linux.yml` "
            "(register L76: `environment.yml` is not the reproducing file). Verbatim:", "", "```yaml"]
    out += ENVFILE.read_text().rstrip("\n").split("\n") + ["```", ""]
    vers = []
    for pk in ("numpy", "pandas", "scipy", "pyomo", "matplotlib", "pyyaml"):
        try:
            vers.append(f"{pk} {md.version(pk)}")
        except md.PackageNotFoundError:
            vers.append(f"{pk} (not found)")
    out += ["Installed at generation: " + ", ".join(vers) + f"; Python {sys.version.split()[0]}.", ""]
    # D.5 manifest
    mt = MANIFEST.read_text().split("\n")
    meta = [l for l in mt[:8] if l.startswith("- ")]
    rows = [l for l in mt if l.startswith("| ") and not l.startswith("| path") and not l.startswith("|---")]
    top = Counter(l.split("|")[1].strip().split("/")[0] if "/" in l.split("|")[1] else "(top level)" for l in rows)
    out += ["## D.5 Results manifest", "", f"`docs/results_manifest_phase_b_start.md` (git blob `{blob(MANIFEST)}`): SHA-256, size and "
            "modification time of every file under `results/`.", ""] + meta + [""]
    out += ["**This manifest pre-dates the Phase B re-runs** (generated 2026-10-02, before items 5, 8, 13, 16, 18, 20 and 23 were re-run on "
            "2026-10-03 to 2026-10-08), so the hashes of re-run outputs differ from the current files. It is the record of the starting "
            "state, not of the reported results; `scripts/make_results_manifest.py` regenerates one. Files per top-level directory:", "",
            "| directory | files |", "|---|---|"]
    out += [f"| `{d}` | {c} |" for d, c in sorted(top.items())]
    out += [""]
    # D.6 runner inventory note
    out += ["## D.6 Runner inventory", "",
            "The Phase B table in D.2 is the current record of which runner produced which result. `docs/runner_inventory.md` is a "
            "separate, earlier document: a read-only inventory of what every script asked and whether it ran, **written at the "
            "2026-09-07 freeze (`2a9226b`)**. It reports many runners as failing or dead that were repaired afterwards (the A5 port "
            "and the argument repairs), so it must not be read as the present state; it is retained in the repository for the record.", ""]
    # D.7 non-arm section (verbatim)
    sec = section(INVENTORY.read_text(), r"^### Non-arm modelling choices", r"^---$|^## PART 4")
    out += ["## D.7 Constants and defaults outside the arm system", "",
            f"Verbatim from `docs/model_capability_inventory.md` (git blob `{blob(INVENTORY)}`), Part 3, section 'Non-arm modelling "
            "choices'. The inventory is dated 2026-10-01; re-verify values against HEAD before relying on them.", ""]
    out += demote(sec, 2) + [""]
    return "\n".join(out)


def build():
    entries = parse_register()
    cls, auto = classify(entries)
    files = {
        "appendix_a.md": appendix_a(),
        "appendix_b.md": register_appendix("B", entries, cls, auto),
        "appendix_c.md": register_appendix("C", entries, cls, auto),
        "appendix_d.md": appendix_d(),
    }
    counts = Counter(cls.values())
    b = counts["LIVE"] + counts["SPLIT"]
    c = counts["CLOSED"]
    summary = {"total": len(entries), "LIVE": counts["LIVE"], "SPLIT": counts["SPLIT"], "CLOSED": counts["CLOSED"],
               "B": b, "C": c, "B+C==total": (b + c) == len(entries), "auto": auto}
    return files, summary


if __name__ == "__main__":
    files, summary = build()
    if "--check" in sys.argv:
        bad = [n for n, t in files.items() if (OUT / n).read_text() != t]
        print("stale:", bad or "none"); sys.exit(1 if bad else 0)
    for n, t in files.items():
        (OUT / n).write_text(t)
        print(f"wrote {OUT / n}  ({len(t.splitlines())} lines)")
    print(summary)
