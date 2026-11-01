#!/usr/bin/env python3
"""
55_findings_reconcile.py -- TASK 55 Part 0: map every finding in docs/findings_frozen.md to its
post-epoch counterpart. READ-ONLY: proposes, edits nothing, solves nothing.

What it does
  1. Parses docs/findings_frozen.md into entries keyed by `## F-n` / `## M-n` (an entry runs to the next
     such heading or the next `#`/`##` heading that is not the one-line continuation of a title). For each
     entry it captures the Source line(s), the pre-epoch results paths those name, and every distinct
     numeric literal with the sentence it sits in (struck-through `~~...~~` text is counted separately).
  2. Walks results/ for `arms_<hash8>/ARMS.json` and every `INDEX.md` (hash8 -> full arm tag).
  3. For each finding proposes a post-epoch counterpart and states the BASIS:
        EXACT-TAG   the post-epoch arm tag is the pre-epoch tag plus the arms TASK 53 2.5 completed
                    (a prefix match; the only match that identifies one arm)
        RUNNER-ONLY the pre-epoch output carried no arm tag (fixed-name runner); the candidates are all
                    arms_ directories of the same family, each shown with how its arm set differs
                    from the central arm set (parsed from run_phase_b.sh's C_* lines)
        NONE        nothing found
  4. Classifies every candidate file OLD / NEW / NEITHER by key presence (the TASK 54 Phase 1 rule:
     final_storage_capacity_mwh, solar_total_built_mw, cumulative_unserved_twh  vs
     storage_capacity_{2045,2054}_mwh and the labelled forms), and REFUSES an OLD-schema file as a
     post-epoch counterpart -- with one stated refinement: runners that still write unlabelled keys
     (register L111: the NDC/EaaS/F3 runners) are accepted only when their arm TAG carries the epoch
     completions (capital_window_case=arrears, storage_capex_case=declining_nrel), which a pre-epoch run
     cannot. Such a candidate is shown as "OLD keys, POST tag" so the exception is visible.
  5. Ends with a block of everything it could not parse, resolve or attribute.

Usage
    python scripts/55_findings_reconcile.py                       # markdown to stdout
    python scripts/55_findings_reconcile.py --out FILE            # also write the markdown to FILE
    python scripts/55_findings_reconcile.py --entry F-12          # print what the parser captured for one entry
"""
import argparse
import glob
import itertools
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINDINGS = os.path.join(ROOT, "docs", "findings_frozen.md")
RESULTS = os.path.join(ROOT, "results")
PHASE_B = os.path.join(ROOT, "scripts", "run_phase_b.sh")

OLD_KEYS = {"final_storage_capacity_mwh", "solar_total_built_mw", "cumulative_unserved_twh"}
NEW_KEYS = {"storage_capacity_2045_mwh", "storage_capacity_2054_mwh", "cumulative_unserved_2025_2045_twh",
            "cumulative_unserved_2025_2054_twh", "solar_cumulative_built_2045_mw", "solar_cumulative_built_2054_mw"}
# arms TASK 53 2.5 appended to tags that omitted them: a post-epoch tag that is a pre-epoch tag plus ONLY these
# is the same arm set; anything else (e.g. social_discount_rate=) is a different arm set and is not an EXACT match
COMPLETION_KEYS = {"reserve_margin", "tariff_case", "us_inflation", "capital_window_case", "storage_capex_case",
                   "om_case", "capital_case", "solar_build_case"}
EPOCH_TAG_MARKERS = ("capital_window_case=arrears", "storage_capex_case=declining_nrel")

# results family -> runner. Families come from the Source lines (Phase 1) and the manifest.
FAMILY_RUNNER = {
    "baseline_sliced": "01b_run_baseline_sliced.py", "baseline": "01_run_baseline.py", "eaas": "02_run_eaas.py",
    "ndc": "02_run_ndc_caps.py", "ndc_eaas": "03_run_ndc_eaas.py", "fin2": "04_run_fin2_tariff_sweep.py",
    "fin3": "05_run_fin3_capital_sweep.py", "gas1": "06_run_gas1_shadow_benchmarks.py",
    "gas2": "07_run_gas2_eaas_gas_relief.py", "gas3": "08_run_gas3_regime_ndc_feasibility.py",
    "rel1": "09_run_rel1_feasibility.py", "rel2": "10_run_rel2_marginal_cost.py",
    "rel3": "11_run_rel3_financing_frontier.py", "dem1": "12_run_dem1_demand_sensitivity.py",
    "dem2": "13_run_dem2_growth_gas_interaction.py", "pol1": "15_run_pol1_ndc_comparison.py",
    "monte_carlo": "17_run_monte_carlo.py", "pareto_frontier": "18_run_pareto_frontier.py",
    "carbon_price_sweep": "19_run_carbon_price_sweep.py", "cf1": "20_run_cf1_solar_yield_sensitivity.py",
    "salvage_sensitivity": "22_salvage_sensitivity.py", "selfgen_frontier": "23_run_selfgen_frontier.py",
    "leakage_arm": "24_run_leakage_arm.py", "obligation_trajectory": "25_run_obligation_trajectory.py",
    "f3": "26_run_f3_identification.py", "hydro_minflow": "27_run_hydro_minflow_sensitivity.py",
    "capital_frontier": "28_run_capital_frontier.py", "parameter_uncertainty": "29_run_parameter_uncertainty.py",
    "stochastic_redispatch": "30_run_stochastic_redispatch.py", "vss_bound": "31_compute_vss_bound.py",
    "stochastic": "17_/30_ (draws files)", "genset_arm_sensitivity": "99c_run_genset_arm_sensitivity.py",
    "_attrib": "(derived attribution tables)",
}
# hash-named families: INDEX.md maps hash8 -> tag and the hash is in the file/directory name
HASH_FAMILIES = ("baseline_sliced", "baseline", "eaas", "ndc", "ndc_eaas", "f3",
                 "stochastic_redispatch", "stochastic", "vss_bound")

# the seven changes of TASK 55 section 0, by keyword (a PROPOSAL, shown as such)
CHANGE_KEYWORDS = OrderedDict([
    (1, ("bn", "usd", "npv", "objective", "cost", "capital charge", "window")),
    (2, ("storage", "mwh")),
    (3, ("shadow", "dual", "exceed", "binding", "price")),
    (4, ("final_storage", "solar_total", "cumulative_unserved", "storage_capacity")),
    (5, ("results/",)),
    (6, ("reliab", "unserved", "shortfall", "adequacy")),
    (7, ("sobol", "draws", "append")),
])

DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUM_RE = re.compile(r"(?<![A-Za-z0-9_.])(?<![A-Za-z_][-/])[-\u2212+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?(?:%|x(?![A-Za-z0-9_]))?(?![A-Za-z0-9_]|\.\d)")
HEX8 = re.compile(r"(?<![0-9a-zA-Z])[0-9a-f]{8}(?![0-9a-zA-Z])")
PATH_RE = re.compile(r"results/[A-Za-z0-9_./*{}<>,\- ]*[A-Za-z0-9_*}>/]")


# --------------------------------------------------------------------------- parsing
def parse_findings(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    heading_re = re.compile(r"^## ((?:F|M)-\d+)\b\s*(.*)$")
    entries, order, cur = {}, [], None
    unparsed = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = heading_re.match(ln)
        if m:
            cur = m.group(1)
            if cur in entries:
                unparsed.append(f"line {i+1}: duplicate heading {cur}")
            entries[cur] = dict(id=cur, line=i + 1, title=m.group(2), body=[], lines=[])
            order.append(cur)
            # a `## ` line directly after a heading is the title's continuation (M-4), not a new section
            if i + 1 < len(lines) and lines[i + 1].startswith("## ") and not heading_re.match(lines[i + 1]):
                entries[cur]["title"] += " " + lines[i + 1][3:]
                i += 1
            i += 1
            continue
        if ln.startswith("# ") or ln.startswith("## "):
            cur = None                       # a section heading or a non-entry level-2 heading ends the entry
        if cur is not None:
            entries[cur]["body"].append(ln)
            entries[cur]["lines"].append(i + 1)
        i += 1
    for e in entries.values():
        e["text"] = "\n".join(e["body"])
    return entries, order, unparsed


def sentences(text):
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        out += [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z*`(\[~])", para) if s]
    return out


# digits that name something rather than quantify it: task, register, finding, phase, item, group, section references
REF_RE = re.compile(r"(?:\bTASK\s+\d+(?:\.\d+)?|\b[LFMC]-?\d+\b|\bPhase\s+[A-C0-9]+\b|\bitem\s+\d+\b|\bGroup\s+\d+\b"
                    r"|\b(?:section|Section|sections)\s+[\d.]+|\u00a7\s*[\d.]+|\bPart\s+[A-Z0-9]+\b|\bB\\?\*)")


def numeric_literals(text):
    """[(literal, sentence, struck)] for every numeric literal in the CLAIM text. Excluded: Source paragraphs
    (provenance, not claims), dates, and digits that are references (TASK n, L-n, F-n, Phase, item, Group, section)."""
    text = "\n".join(re.sub(r"^\s*>\s?", "", ln) for ln in text.split("\n"))
    text = "\n\n".join(para for para in re.split(r"\n\s*\n", text) if not re.match(r"^\s*\*\*Source", para))
    res = []
    for sent in sentences(text):
        clean = REF_RE.sub(" ", DATE_RE.sub(" ", sent))
        spans = [(m.start(), m.end()) for m in re.finditer(r"~~.*?~~", clean)]
        for m in NUM_RE.finditer(clean):
            lit = m.group(0)
            if re.fullmatch(r"[-−+]?\d{1,2}", lit) and False:
                continue
            struck = any(a <= m.start() < b for a, b in spans)
            res.append((lit.replace("−", "-"), sent[:220], struck))
    return res


def expand_braces(p):
    m = re.search(r"\{([^{}]*)\}", p)
    if not m:
        return [p]
    inner = m.group(1)
    if HEX8.search(inner):                       # {02c17f55 central, c52b926f cr_comm_low, ...}: the hashes
        alts = HEX8.findall(inner)
    else:
        alts = [a.strip() for a in inner.split(",")]
    out = []
    for a in alts:
        out += expand_braces(p[:m.start()] + a + p[m.end():])
    return out


def source_paths(entry_text):
    """(source_lines, [expanded results paths]) from the entry's Source paragraph(s)."""
    paras = re.split(r"\n\s*\n", entry_text)
    src = [" ".join(p.split()) for p in paras if re.match(r"^\s*\*\*Source", p)]
    pre = [s for s in src if "TASK 54 Phase 1" in s]
    paths = []
    for s in src:
        for raw in PATH_RE.findall(s):
            raw = raw.strip().rstrip(",.;)")
            paths += expand_braces(raw)
    return src, pre, sorted(set(paths))


# --------------------------------------------------------------------------- results tree
def schema_of_json(path):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    ks = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                ks.add(k)
                walk(v)
    walk(d)
    old, new = bool(ks & OLD_KEYS), bool(ks & NEW_KEYS)
    return "BOTH" if old and new else "OLD" if old else "NEW" if new else "NEITHER"


def read_index(path):
    out = {}
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            m = re.match(r"^- `([0-9a-f]{8})` -> `(.*)`\s*$", ln)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def central_arms():
    """kwarg -> value from run_phase_b.sh's ^C_NAME="value" lines (the central arm set)."""
    names = dict(C_PEAK="peak_multiple_case", C_RESERVE="reserve_margin", C_STORAGE_CC="storage_cc_case",
                 C_GAS_BUILD="gas_build_case", C_NEW_GAS_DELIV="new_gas_deliverability",
                 C_GAS_AVAIL="gas_availability_case", C_GAS_RETIRE="gas_retirement_case",
                 C_GAS_CAPEX="gas_capex_case", C_CCGT="ccgt_efficiency_case", C_CR="capital_recovery_case",
                 C_DSCR="dscr_case", C_INFL="us_inflation", C_TREAT="capital_treatment_case",
                 C_ENV="concessional_envelope_case", C_GAS_DELIV="gas_deliverability_case",
                 C_CONV="conversion_case", C_HYDRO="hydro_case", C_TARIFF="tariff_case", C_OM="om_case")
    out = {}
    for ln in open(PHASE_B, encoding="utf-8"):
        m = re.match(r'^(C_[A-Z_]+)="([^"]*)"$', ln.strip())
        if m and m.group(1) in names:
            out[names[m.group(1)]] = m.group(2)
    return out


def parse_kv_tag(tag):
    kv = {}
    for part in tag.split("__"):
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k] = v
    return kv


def scan_tree():
    """All post-epoch-capable records: arms_ dirs (ARMS.json) and hash-named files/dirs (INDEX.md + file)."""
    recs = []
    for armsjson in sorted(glob.glob(os.path.join(RESULTS, "**", "arms_*", "ARMS.json"), recursive=True)):
        d = os.path.dirname(armsjson)
        try:
            a = json.load(open(armsjson))
        except Exception:
            recs.append(dict(kind="arms", path=d, tag=None, unparsed=True))
            continue
        rel = os.path.relpath(d, RESULTS)
        schemas = set()
        for jf in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
            if jf.endswith("ARMS.json"):
                continue
            s = schema_of_json(jf)
            if s:
                schemas.add(s)
        recs.append(dict(kind="arms", path=d, rel=rel, family=rel.split(os.sep)[0], hash8=a.get("hash8"),
                         tag=a.get("arm_tag"), schemas=schemas, mtime=os.path.getmtime(armsjson)))
    for fam in HASH_FAMILIES:
        idx = read_index(os.path.join(RESULTS, fam, "INDEX.md"))
        for h, tag in idx.items():
            cands = glob.glob(os.path.join(RESULTS, fam, f"summary__{h}.json")) + \
                    glob.glob(os.path.join(RESULTS, fam, f"*_{h}", "summary.json")) + \
                    glob.glob(os.path.join(RESULTS, fam, f"vss_bound__{h}.json")) + \
                    (glob.glob(os.path.join(RESULTS, fam, f"draws__{h}_meta.json")) if fam == "stochastic" else [])
            for f in cands:
                recs.append(dict(kind="hash", path=f, rel=os.path.relpath(f, RESULTS), family=fam, hash8=h, tag=tag,
                                 schemas={schema_of_json(f) or "NEITHER"}, mtime=os.path.getmtime(f)))
    return recs


EPOCH_START = [0.0]          # set in main(): mtime of the earliest NEW-schema summary in the tree


def epoch_status(rec):
    if rec.get("forced_post"):
        return "POST (mtime >= epoch start; overwritten in place)"
    """POST (keys NEW), POST-TAG (OLD keys but the tag carries the epoch completions), PRE (refused), UNKNOWN."""
    tag = rec.get("tag") or ""
    m1 = re.match(r"stage1=summary__([0-9a-f]{8})__", tag)
    if m1:                                   # vss_bound: the epoch is that of the 30_ run it consumes
        tag = STAGE1_INDEX.get(m1.group(1), tag)
    tagged = all(m in tag for m in EPOCH_TAG_MARKERS)
    sch = rec.get("schemas") or set()
    if "NEW" in sch and "OLD" not in sch:
        return "POST (NEW keys)"
    if tagged and ("OLD" in sch or "BOTH" in sch):
        return "POST-TAG (OLD keys, epoch tag)"
    if tagged:
        return "POST (epoch tag)"
    if "OLD" in sch:
        return "PRE (OLD keys, no epoch tag) -- REFUSED"
    return "UNKNOWN (no epoch marker, no schema keys)"


def usable(status):
    return status.startswith("POST")


# --------------------------------------------------------------------------- matching
def family_of(path):
    parts = path.split("/")
    return parts[1] if len(parts) > 1 else None


def only_completions(suffix, allow_om=False):
    """True iff `suffix` (the part of a post-epoch tag after the pre-epoch tag) is nothing but completed arms."""
    keys = "|".join(sorted(COMPLETION_KEYS, key=len, reverse=True))
    parts = [x.strip("_") for x in re.split(r"(?:__|_)(?=(?:%s)=)" % keys, suffix) if x.strip("_")]
    # 01b's `om_tag_suffix` puts a bare `__om_none|om_low|om_high` in the tag; pre-epoch regression-family tags
    # (19c81f7c) lacked it, the post-epoch tag has it -- the same arm, now explicit.
    return bool(parts) and all(re.match(r"^(?:%s)=" % keys, part) or (allow_om and part in ("om_none", "om_low", "om_high")) for part in parts)


def kv_diff(pre_tag, post_tag):
    """Human-readable arm differences between two kv tags, ignoring the completed arms."""
    if "=" not in pre_tag or "=" not in post_tag:
        return ["(positional tags; no per-arm diff)"]
    a, b = parse_kv_tag(pre_tag), parse_kv_tag(post_tag)
    for k in COMPLETION_KEYS:
        a.pop(k, None); b.pop(k, None)
    d = [f"{k}: {a[k]} -> {b[k]}" for k in sorted(set(a) & set(b)) if a[k] != b[k]]
    d += [f"{k} only pre ({a[k]})" for k in sorted(set(a) - set(b))]
    d += [f"{k} only post ({b[k]})" for k in sorted(set(b) - set(a))]
    return d


STAGE1_INDEX = {}            # stochastic_redispatch hash8 -> tag, set in main()


def equal_modulo_completions(pre_tag, post_tag):
    """Both are `k=v__k=v` tags: equal after dropping the arms TASK 53 2.5 completed. A vss_bound tag's
    `stage1=summary__<hash8>` points at a 30_ run; two such tags are equal if those two 30_ tags are."""
    m1, m2 = re.match(r"stage1=summary__([0-9a-f]{8})__(.*)$", pre_tag), re.match(r"stage1=summary__([0-9a-f]{8})__(.*)$", post_tag)
    if m1 and m2:
        t1, t2 = STAGE1_INDEX.get(m1.group(1)), STAGE1_INDEX.get(m2.group(1))
        return bool(t1 and t2 and m1.group(2) == m2.group(2) and equal_modulo_completions(t1, t2))
    if "=" not in pre_tag or "=" not in post_tag or pre_tag.count("=") < 5:
        return False
    a, b = parse_kv_tag(pre_tag), parse_kv_tag(post_tag)
    for k in COMPLETION_KEYS:
        a.pop(k, None); b.pop(k, None)
    return a == b and bool(a)


def match_finding(paths, recs, pre_indexes, central, text_hashes=(), _sub=False):
    """[(pre_path, basis, [candidate dicts])]"""
    out = []
    by_fam = defaultdict(list)
    for r in recs:
        by_fam[r["family"]].append(r)
    for p in paths:
        fam = family_of(p)
        hs = HEX8.findall(p)
        if "/#" in p:
            hs = [p.rsplit("/#", 1)[1]]
            p = p.rsplit("/#", 1)[0] + f" [pre hash {hs[0]}]"
        if fam is None:
            out.append((p, "UNRESOLVED", [])); continue
        if fam.endswith(".log") or fam.endswith(".md") or fam.endswith(".csv") or fam.endswith(".json"):
            out.append((p, "NOT A RESULT FILE (log/loose file at results/ top level)", [])); continue
        if fam.startswith("_pre") or fam.startswith("_superseded") or fam.startswith("_recovered"):
            out.append((p, "PRE-EPOCH ARCHIVE directory (by name); no counterpart sought", [])); continue
        if fam.endswith("_uncapped") or fam.endswith("_gate"):
            base = fam.rsplit("_", 1)[0]
            out.append((p, f"NONE: {fam} is a Phase C {'Group 3' if fam.endswith('_uncapped') else 'Group 0'} design directory"
                           f" ({'Group 3 has not run' if fam.endswith('_uncapped') else 'gate run, not Phase B output'})", []))
            continue
        if fam == "_attrib":
            out.append((p, "NONE: derived attribution table (not a runner output)", [])); continue
        if fam in HASH_FAMILIES and not hs:
            known = [h for h in text_hashes if h in pre_indexes.get(fam, {})]
            if known:
                hs = known
        if fam in HASH_FAMILIES and hs:
            if not _sub and (len(hs) > 1 or not HEX8.search(p)):
                # several pre-epoch hashes named in the Source text for one directory path: match each
                sub = []
                for h in hs:
                    sub += match_finding([p.rstrip("/") + f"/#{h}"], recs, pre_indexes, central, (), _sub=True)
                out += sub
                continue
            pre_tag = pre_indexes.get(fam, {}).get(hs[-1])
            if pre_tag is None:
                out.append((p, "UNRESOLVED (hash not in INDEX.md)", [])); continue
            def _cands(allow_om):
                return [r for r in by_fam[fam] if r.get("tag") and r["tag"] != pre_tag and not r["rel"].startswith("_")
                        and ((r["tag"].startswith(pre_tag) and only_completions(r["tag"][len(pre_tag):], allow_om))
                             or equal_modulo_completions(pre_tag, r["tag"]))]
            # strict first; a bare __om_* suffix is accepted only when nothing else matches (pre-epoch regression-family tags
            # lacked it), so a pre-epoch om_central tag can never be matched to an om_low / om_high / om_none file
            cands = _cands(False) or _cands(True)
            if not cands:
                near = [r for r in by_fam[fam] if r.get("tag") and not r["rel"].startswith("_") and epoch_status(r).startswith("POST")]
                best = min(near, key=lambda r: len(kv_diff(pre_tag, r["tag"]))) if near else None
                why = ("; nearest post-epoch tag " + best["rel"] + " differs in: " + "; ".join(kv_diff(pre_tag, best["tag"])[:6])) if best else ""
                out.append((p, "NONE (no post-epoch tag equals the pre-epoch tag modulo completed arms)" + why, [])); continue
            out.append((p, "EXACT-TAG", cands)); continue
        cands = [r for r in by_fam.get(fam, [])]
        if not cands:
            live = os.path.join(ROOT, p.rstrip("/"))
            hits = glob.glob(live) if "*" in live else ([live] if os.path.exists(live) else [])
            files = []
            for h in hits:
                files += [h] if os.path.isfile(h) else [os.path.join(r, f) for r, _, fs in os.walk(h) for f in fs]
            fresh = [f for f in files if os.path.getmtime(f) >= EPOCH_START[0]]
            if files and len(fresh) == len(files):
                out.append((p, "SAME-PATH (file(s) overwritten in place after the epoch; no arm tag recorded)",
                            [dict(kind="samepath", path=p, rel=p.replace("results/", "", 1), family=fam, tag=None,
                                  schemas={"NEW"} if any(schema_of_json(f) == "NEW" for f in files if f.endswith(".json")) else set(),
                                  mtime=max(os.path.getmtime(f) for f in files), forced_post=True)]))
                continue
            if files and fresh:
                out.append((p, f"PARTIAL SAME-PATH ({len(fresh)} of {len(files)} files newer than the epoch)", []))
                continue
        if not cands and not glob.glob(os.path.join(ROOT, p.rstrip("/"))) and fam in FAMILY_RUNNER:
            out.append((p, "PATH DOES NOT EXIST in the live tree as quoted", [])); continue
        if not cands and fam not in FAMILY_RUNNER:
            out.append((p, "UNRESOLVED (unknown results family)", [])); continue
        out.append((p, "RUNNER-ONLY" if cands else "NONE", cands))
    return out


def arm_diff(tag, central):
    if "=" not in (tag or "") or tag.count("=") < 5:
        # "_"-separated positional tag (02_/03_/26_): report only whether every central value is present
        missing = [f"{k}={v}" for k, v in central.items() if v not in (tag or "") and k not in ("reserve_margin", "us_inflation")]
        return ["central arm values all present"] if not missing else ["differs from central in: " + ", ".join(missing)]
    kv = parse_kv_tag(tag or "")
    diffs = []
    for k, v in central.items():
        if k in kv and kv[k] != v:
            try:
                if float(kv[k]) == float(v):
                    continue
            except ValueError:
                pass
            diffs.append(f"{k}={kv[k]} (central {v})")
    extra = sorted(set(kv) - set(central) - {"capital_window_case", "storage_capex_case"})
    if extra:
        diffs.append("own arms: " + ", ".join(f"{k}={kv[k]}" for k in extra))
    return diffs


# --------------------------------------------------------------------------- output
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--entry", default=None, help="print what the parser captured for one entry (e.g. F-12) and exit")
    args = ap.parse_args()

    entries, order, unparsed = parse_findings(FINDINGS)
    raw_headings = len(re.findall(r"^## F-\d+", open(FINDINGS, encoding="utf-8").read(), re.M))
    raw_m = len(re.findall(r"^## M-\d+", open(FINDINGS, encoding="utf-8").read(), re.M))

    if args.entry:
        e = entries[args.entry]
        src, pre, paths = source_paths(e["text"])
        print(f"{e['id']}  line {e['line']}  body lines {e['lines'][0]}-{e['lines'][-1]}  title: {e['title']}")
        print("Source paragraphs:", len(src), " (Phase 1 lines:", len(pre), ")")
        for s in src:
            print("  -", s[:300])
        print("paths:", paths)
        lits = numeric_literals(e["text"])
        live = [l for l in lits if not l[2]]
        print(f"numeric literals: {len(lits)} ({len(live)} live, {len(lits)-len(live)} struck); distinct live: {len(set(l[0] for l in live))}")
        for lit, sent, st in lits[:15]:
            print(f"  {'~' if st else ' '} {lit:>14}  {sent[:110]}")
        return

    recs = scan_tree()
    new_mt = [r["mtime"] for r in recs if r.get("schemas") and "NEW" in r["schemas"]]
    EPOCH_START[0] = min(new_mt) - 60 if new_mt else 0.0
    central = central_arms()
    pre_indexes = {fam: read_index(os.path.join(RESULTS, fam, "INDEX.md")) for fam in HASH_FAMILIES}
    STAGE1_INDEX.update(pre_indexes.get("stochastic_redispatch", {}))

    L = []
    L.append(f"# Findings reconciliation (generated by scripts/55_findings_reconcile.py)")
    L.append("")
    L.append(f"Entries parsed: **{len(entries)}** ({sum(1 for k in entries if k.startswith('F-'))} F, "
             f"{sum(1 for k in entries if k.startswith('M-'))} M); `## F-` headings in the file: {raw_headings}; `## M-`: {raw_m}.")
    if len(entries) != raw_headings + raw_m or sum(1 for k in entries if k.startswith('F-')) != raw_headings:
        L.append("**PARSER DEFECT: entry count disagrees with the file's heading count.**")
    L.append(f"Records scanned in results/: {len(recs)} (arms_ dirs: {sum(1 for r in recs if r['kind']=='arms')}; "
             f"hash-named files: {sum(1 for r in recs if r['kind']=='hash')}).")
    L.append("")
    L.append("| finding | pre-epoch source | post-epoch counterpart | how matched | epoch status | changes (proposed) | live numeric literals |")
    L.append("|---|---|---|---|---|---|---|")

    could_not = dict(entries=list(unparsed), sources=[], literals=[], noattr=[])
    mapped_dirs = set()
    table_rows = []
    for fid in order:
        e = entries[fid]
        src, pre, paths = source_paths(e["text"])
        lits = numeric_literals(e["text"])
        live = sorted(set(l[0] for l in lits if not l[2]))
        low = e["text"].lower()
        changes = [str(k) for k, kws in CHANGE_KEYWORDS.items() if any(w in low for w in kws)]
        if not src:
            could_not["sources"].append(f"{fid}: no Source paragraph at all")
        elif not pre:
            could_not.setdefault("existing_only", []).append(fid)
        if not paths:
            m_rows = [("(no results path in Source)", "NONE", [])]
            could_not["sources"].append(f"{fid}: Source names no results/ path (closed form, input data, or unparsed)")
        else:
            m_rows = match_finding(paths, recs, pre_indexes, central, tuple(sorted(set(HEX8.findall(e["text"])))))
        if not lits:
            could_not["noattr"].append(f"{fid}: no numeric literal found in the body")
        seen = set()
        for p, basis, cands in m_rows:
            usable_c = [c for c in cands if usable(epoch_status(c))]
            refused = [c for c in cands if not usable(epoch_status(c))]
            if basis.startswith("UNRESOLVED"):
                could_not["sources"].append(f"{fid}: {p} -> {basis}")
            key = (basis, tuple(sorted(c["rel"] for c in usable_c)), p if not usable_c else "")
            if key in seen:
                continue
            seen.add(key)
            for c in usable_c:
                mapped_dirs.add(c["path"])
            if usable_c:
                shown = "; ".join(sorted(f"`{c['rel']}`" for c in usable_c)[:3]) + (f" (+{len(usable_c)-3} more)" if len(usable_c) > 3 else "")
                stat = "; ".join(sorted(set(epoch_status(c) for c in usable_c)))
                if basis == "RUNNER-ONLY":
                    diffs = sorted(set("; ".join(arm_diff(c["tag"], central)) or "central arm set" for c in usable_c))
                    how = f"RUNNER-ONLY ({len(usable_c)} dirs); arm set vs central: " + (" | ".join(d[:100] for d in diffs[:2]) + (f" | (+{len(diffs)-2} more variants)" if len(diffs) > 2 else ""))
                else:
                    how = basis
            else:
                shown = "NONE" + (f" ({len(refused)} OLD/pre files refused)" if refused else "")
                stat = "-"
                how = basis
            table_rows.append((fid, p, shown, how, stat, ",".join(changes), len(live)))
    merged, none_groups = [], OrderedDict()
    for row in table_rows:
        fid, p, shown, how, stat, ch, n = row
        if "[pre hash" in p and shown == "NONE":
            base = p.split(" [pre hash")[0]
            g = none_groups.setdefault((fid, base), dict(h=[], diffs=set(), row=row))
            g["h"].append(re.search(r"\[pre hash (\w+)\]", p).group(1))
            g["diffs"].update(x.strip() for x in how.split("differs in:")[-1].split(";"))
        else:
            merged.append(row)
    for (fid, base), g in none_groups.items():
        r = g["row"]
        merged.append((fid, f"{base} [pre hashes {', '.join(g['h'])}]", "NONE",
                       f"NONE x{len(g['h'])}: no post-epoch tag equals these pre-epoch tags modulo completed arms; differences from the nearest post tag include: "
                       + "; ".join(sorted(g["diffs"])[:6]), "-", r[5], r[6]))
    table_rows = sorted(merged, key=lambda r: (order.index(r[0]), r[1]))
    for fid, p, shown, how, stat, ch, n in table_rows:
        L.append(f"| {fid} | `{p}` | {shown} | {how} | {stat} | {ch} | {n} |")

    # closing block
    arms_unmapped = [r for r in recs if r["kind"] == "arms" and r["path"] not in mapped_dirs]
    L += ["", "## Could not parse / resolve / attribute", ""]
    L.append(f"**Entries not parsed:** {len(could_not['entries'])}" + ("" if not could_not["entries"] else ": " + "; ".join(could_not["entries"])))
    L.append("")
    L.append(f"**Source lines not resolved ({len(could_not['sources'])}):**")
    L += [f"* {x}" for x in sorted(set(could_not["sources"]))] or ["* none"]
    L.append("")
    eo = could_not.get("existing_only", [])
    L.append(f"**Entries with an existing Source line but no 'TASK 54 Phase 1' line ({len(eo)}; Phase 1 left their Source lines as they stood):** " + ", ".join(eo))
    L.append("")
    L.append(f"**Numeric literals not attributable to any sentence/entry ({len(could_not['noattr'])}):**")
    L += [f"* {x}" for x in could_not["noattr"]] or ["* none"]
    L.append("")
    L.append(f"**`arms_` directories no finding maps to ({len(arms_unmapped)} of {sum(1 for r in recs if r['kind']=='arms')}):**")
    L += [f"* `{r['rel']}` ({r.get('tag','')[:70]}...)" for r in arms_unmapped] or ["* none"]
    text = "\n".join(L)
    print(text)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
