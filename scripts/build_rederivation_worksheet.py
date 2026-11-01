"""TASK 83 Part C.1/C.3: for every F-/M- entry of docs/findings_frozen.md, the files its Source paragraphs name and whether a
vintage-generation counterpart exists (docs/results_manifest.json rules C4, C5, S3, S4). Reads only; writes docs/task_log_83_partC.csv.

A 'Source paragraph' is the run of non-blank lines starting at a line that contains the word Source. Paths are the backtick tokens in it
that contain a '/' or end in .csv/.json/.py/.log/.png/.pdf/.md/.jsonl/.txt; {a,b} is expanded. Counterpart test: a vintage manifest entry
under results/<top>/ with the same basename. Status values:
  exists            a vintage file of that basename is in the manifest (first one shown)
  exists (twin)     the cited blended file's arm tag, with the vintage token added, hashes to a path that exists (shown)
  dir-has-vintage   the directory has vintage files but none of that basename
  not re-run        results/<top> exists, nothing in it is vintage; newest file date shown; REGISTERED BLENDED if L142 names the runner
  absent            results/<top> does not exist
  out of scope      not under results/ (script, log, document, data, source code)
"""
import csv, glob, json, os, re, itertools, datetime as dt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
V = {"C4", "C5", "S3", "S4"}
man = json.load(open("docs/results_manifest.json"))["files"]
vdirs = {os.path.dirname(e["path"]) for e in man if e["epoch_rule"] in V}
# a manifest entry is vintage if its rule is in V, or it is an S2 sibling (same directory, written within 30 min) of one that is
vint = [e["path"] for e in man if e["epoch_rule"] in V or (e["epoch_rule"] == "S2" and os.path.dirname(e["path"]) in vdirs)]
bytop = {}
for p in vint: bytop.setdefault(p.split("/")[1], []).append(p)
BLENDED = {"salvage_sensitivity": "22_", "selfgen_frontier": "23_", "capital_window": "51_", "horizon_rebuild": "53_",
           "storage_capex": "54_", "capital_regime": "56_", "nirp_cap_check": "58_", "dem1": "12_ (suspended stub)", "str1": "14_ (suspended stub)"}
L = open("docs/findings_frozen.md").read().split("\n")
heads = [(i, l) for i, l in enumerate(L) if re.match(r"## (F|M)-\d+", l)]; heads.append((len(L), "END"))
EXT = (".csv", ".json", ".py", ".log", ".png", ".pdf", ".md", ".jsonl", ".txt", ".sh")

def expand(t):
    m = re.search(r"\{([^{}]*)\}", t)
    if not m: return [t]
    return [x for alt in m.group(1).split(",") for x in expand(t[:m.start()] + alt + t[m.end():])]

def newest(top):
    ps = [p for p in glob.glob(f"results/{top}/**/*", recursive=True) if os.path.isfile(p)]
    return max((os.path.getmtime(p) for p in ps), default=None)

import sys; sys.path.insert(0, ROOT)
from src.output_naming import hash8_of
_idx = {}
def index_tags(top):
    """hash8 -> arm_tag from results/<top>/INDEX.md (and ARMS.json files below it)."""
    if top in _idx: return _idx[top]
    d = {}
    ip = f"results/{top}/INDEX.md"
    if os.path.exists(ip):
        for ln in open(ip, errors="replace"):
            m = re.search(r"`([0-9a-f]{8})`\s*(?:->|→|:)\s*`(.*)`", ln)
            if m: d[m.group(1)] = m.group(2)
    for a in glob.glob(f"results/{top}/**/ARMS.json", recursive=True):
        try: j = json.load(open(a)); d.setdefault(j["hash8"], j["arm_tag"])
        except Exception: pass
    _idx[top] = d; return d

def twin(path, top):
    """Vintage twin of a cited blended file: same arm tag with the vintage token added, hash replaced in the path. -> path or None."""
    for h in re.findall(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])", path):
        tag = index_tags(top).get(h)
        if not tag or "gas_vintage_dispatch" in tag or "capital_window_case=" not in tag: continue
        for sep in ("__", "_"):
            k = sep + "storage_capex_case="
            if k in tag:
                t2 = tag.replace(k, sep + "gas_vintage_dispatch=True" + k, 1); break
        else: continue
        h2 = hash8_of(t2)
        cand = path.replace(h, h2)
        if h2 in index_tags(top) or os.path.exists(cand):
            if os.path.exists(cand) or os.path.isdir(cand.rstrip("/")): return cand
    return None

def status(path):
    if not path.startswith("results/"): return "out of scope", ""
    parts = path.split("/"); top = parts[1] if len(parts) > 2 else ""
    if not top or not os.path.isdir(f"results/{top}"): return "absent", f"results/{top} does not exist"
    base = os.path.basename(path)
    tw = twin(path, top) if os.path.exists(path) else None
    if tw: return "exists (twin)", tw
    if top in bytop:
        hit = [p for p in bytop[top] if os.path.basename(p) == base or (("*" in base) and glob.fnmatch.fnmatch(os.path.basename(p), base))]
        if hit: return "exists", hit[0]
        return "dir-has-vintage", f"{len(bytop[top])} vintage files in results/{top}, none named {base}"
    n = newest(top); d = dt.datetime.fromtimestamp(n).strftime("%Y-%m-%d") if n else "empty"
    return "not re-run", f"newest file {d}" + (f"; REGISTERED BLENDED / not re-run: runner {BLENDED[top]} (L142)" if top in BLENDED else "")

rows = []
for (a, h), (b, _) in zip(heads, heads[1:]):
    key = re.match(r"## ((?:F|M)-\d+)", h).group(1); title = h[3:].split(" — ", 1)[-1][:90]
    seg = L[a:b]; toks = []; i = 0
    while i < len(seg):
        if re.search(r"\bSources?\b", seg[i]):
            j = i
            while j < len(seg) and seg[j].strip(): j += 1
            toks += re.findall(r"`([^`]+)`", " ".join(seg[i:j])); i = j
        else: i += 1
    paths = []
    for t in toks:
        t = t.strip().rstrip(".,;:")
        if " " in t or not ("/" in t or t.endswith(EXT)): continue
        for x in expand(t):
            if x not in paths: paths.append(x)
    if not paths: rows.append(dict(finding=key, title=title, source_file="(no file named on a Source line)", status="", note="")); continue
    for p in paths:
        s, n = status(p); rows.append(dict(finding=key, title=title, source_file=p, status=s, note=n))
with open("docs/task_log_83_partC.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["finding", "title", "source_file", "status", "note"]); w.writeheader(); w.writerows(rows)
print(len({r["finding"] for r in rows}), "entries,", len(rows), "rows")
