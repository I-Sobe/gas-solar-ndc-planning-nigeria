"""Part A.1/A.2 of TASK 48: inventory of every summary/diagnostics file under results/.

One row per file: path, size, mtime, md5, the arm keys the file records, the arm
hash from the filename, and an epoch class (CURRENT / LEGACY / UNCLASSIFIABLE) with
the rule that fired.  Output: docs/results_inventory.csv.

Epoch rules (first match wins; see docs/task_log_48_evidence_spine.md Part A):
  L1  path under a results/_pre_* archive directory           -> LEGACY
  L2  any arm key present with value None                     -> LEGACY (pre-F1 None keys)
  L3  tariff_case == tariff_legacy_65m                        -> LEGACY
  L4  arm_tag contains pre_f1_overnight                       -> LEGACY
  L5  mtime before 2026-10-02 (before Phase B)                -> LEGACY
  (No annual-builder rule: every runner was ported to the sliced builder in c549bba,
   2026-10-01, so L5's date bound already excludes every annual-builder output.)
  L6  (TASK 82 A.2) arm_tag carries capital_window_case=arrears (epoch 1, 2026-10-19/20) but not the
      vintage token: blended-dispatch generation, or an explicit gas_vintage_dispatch=False   -> LEGACY
  C4  (TASK 82 A.2) arm_tag carries gas_vintage_dispatch=True (the vintage-dispatch generation,
      2026-10-23), or gas_vintage_dispatch=False on an exempt-deliverability tag, where the flag is
      forced off by the builder and is part of the arm (L138)   -> CURRENT. Both are tested BEFORE U1.
  C5  (TASK 82 A.2) as C4, the arm_tag read from the ARMS.json of the file's arms_<hash8> directory
      because the runner summary carries none (the evidence is the sidecar, not the artefact).  L6 likewise.
  U1  mtime after 2026-10-10 (after Phase C / HEAD 3f0b382)   -> UNCLASSIFIABLE
  U2  in window, no arm key and no arm_tag recorded           -> UNCLASSIFIABLE
  U3  in window, builder evidence says annual                 -> UNCLASSIFIABLE
  U4  in window, arm keys or an arm_tag recorded but no om_case anywhere in the file
      (neither as a key nor as an om_* token in the tag) -> UNCLASSIFIABLE (the O&M arm
      cannot be confirmed from the artefact; the hash does not carry it either)
  C2  (ADMITTED BY DECISION, 2026-10-18) a U2 file under a declared rule: the DIRECTORY
      PATH identifies the arm (DIR_LOG table below) and a Phase B/C runner log whose
      header records a HEAD has mtime within 15 min of the file's AND whose name carries
      every arm token (nameplate/observed/annual/total) present in the path.  Class CURRENT with
      rule C2 and epoch_log / epoch_head filled.  This is a WEAKER evidence standard
      than arm-keys-in-artefact (C1): the artefact itself proves nothing; the claim
      rests on the path and an external log.  Files that fail the log test stay U2.
  C3  (ADMITTED BY DECISION, 2026-10-18) a U4 file written by scripts/26_run_f3_identification.py
      whose Phase B/C runner log (names the file's directory, mtime within 15 min) records a
      HEAD with a CLEAN working tree, AND `git show <HEAD>:scripts/26_run_f3_identification.py`
      contains the literal om_case="om_central" in its load_scenario call and no
      args.om_case.  The literal is verified AT THAT COMMIT, not at HEAD; the chain is
      recorded in `verification`.  Any U4 file without the chain stays UNCLASSIFIABLE.
  S   a diagnostics file takes the class of its sibling summary (same directory, same
      hash), because the diagnostics carry only arm_tag, which omits om_case
  C1  otherwise (Phase B 10-02..10-04 / 10-08, Phase C 10-10) -> CURRENT
"""
import csv, glob, hashlib, json, os, re, subprocess, datetime as dt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ARM_KEYS = ["om_case", "conversion_case", "capital_case", "tariff_case", "solar_build_case",
            "policy", "peak_case", "storage_cc_case", "gas_build_case", "new_gas_deliverability",
            "gas_availability_case", "gas_retirement_case", "gas_capex_case",
            "ccgt_efficiency_case", "capital_recovery_case", "dscr_case",
            "capital_treatment_case", "concessional_envelope_case", "gas_deliverability_case",
            "hydro_case", "cap_scenario", "reserve_margin"]
WIN_LO = dt.datetime(2026, 10, 2)
WIN_HI = dt.datetime(2026, 10, 11)
HASH_RE = re.compile(r"__([0-9a-f]{8})\.json$|_([0-9a-f]{8})$")

TAG_PATTERNS = {
    "gas_availability_case": r"gas_avail_(?:nameplate|observed)",
    "conversion_case": r"conv_(?:none|low|central|high)",
    "hydro_case": r"hydro_(?:flat|committed|ndc)",
    "tariff_case": r"tariff_(?:legacy_65m|legacy_95m|low|central|high)",
    "om_case": r"om_(?:none|low|central|high)",
}

# directory-name prefix -> substring identifying the runner log (logs/phase_b, logs/phase_c)
DIR_LOG = [
    ("rel1", ["14_"]), ("rel2_gate", ["0_10_"]), ("rel2_uncapped", ["3_10_"]), ("rel2", ["15_"]),
    ("rel3_gate", ["0_11_"]), ("rel3_uncapped", ["3_11_"]), ("rel3", ["16_A", "16_B", "16_rel3"]),
    ("gas1_gate", ["0_06_"]), ("gas1_uncapped", ["3_06_"]), ("gas1", ["09_"]),
    ("gas2_gate", ["0_07_"]), ("gas2_uncapped", ["3_07_"]), ("gas2", ["10_gas2"]),
    ("gas3_gate", ["0_08_"]), ("gas3_uncapped", ["3_08_"]), ("gas3", ["11_gas3"]),
    ("fin2_gate", ["0_04_"]), ("fin2_uncapped", ["3_04_"]), ("fin2", ["12_"]),
    ("fin3_gate", ["0_05_"]), ("fin3_uncapped", ["3_05_"]), ("fin3", ["23_"]),
    ("dem2", ["13_"]), ("pol1", ["03_pol1"]), ("carbon_price_sweep", ["05_"]),
    ("parameter_uncertainty", ["22_"]),
]
LOG_TOL_S = 15 * 60

def find_epoch_log(path, mtime):
    top = path.split(os.sep)[1]
    keys = next((k for pre, k in DIR_LOG if top == pre), None)
    if not keys: return None
    for d in ("logs/phase_b", "logs/phase_c"):
        for lg in sorted(glob.glob(d + "/*.log")):
            if not any(os.path.basename(lg).startswith(k) or k in os.path.basename(lg) for k in keys):
                continue
            if abs(os.stat(lg).st_mtime - mtime) > LOG_TOL_S: continue
            # every arm token in the path must be named in the log (nameplate/observed, annual/total)
            if any(t in os.path.dirname(path) and t not in os.path.basename(lg) for t in ("nameplate", "observed", "annual", "total")):
                continue
            head = ""
            with open(lg, errors="replace") as fh:
                for _ in range(8):
                    ln = fh.readline()
                    if ln.startswith("HEAD"): head = ln.split(":", 1)[1].strip()
            if head: return lg, head
    return None

def verify_26_literal(path, mtime):
    """-> (log, head, verification-string) or None.  See rule C3."""
    for lg in sorted(glob.glob("logs/phase_b/*.log") + glob.glob("logs/phase_c/*.log")):
        if abs(os.stat(lg).st_mtime - mtime) > LOG_TOL_S: continue
        txt = open(lg, errors="replace").read()
        if os.path.dirname(path) not in txt and os.path.basename(os.path.dirname(path)) not in txt: continue
        mh = re.search(r"^HEAD\s*:\s*([0-9a-f]{40})", txt, re.M)
        if not mh or not re.search(r"^Working tree\s*:\s*clean", txt, re.M): continue
        src = subprocess.run(["git", "show", f"{mh.group(1)}:scripts/26_run_f3_identification.py"],
                             capture_output=True, text=True)
        if src.returncode: continue
        lines = src.stdout.splitlines()
        lit = [i + 1 for i, l in enumerate(lines) if re.search(r'^\s*om_case="om_central",\s*$', l)]
        if lit and "args.om_case" not in src.stdout:
            return lg, mh.group(1), (f"log {lg} (clean tree) -> HEAD {mh.group(1)[:7]}; "
                                     f"`git show {mh.group(1)[:7]}:scripts/26_run_f3_identification.py` line {lit[0]}: om_case=\"om_central\", "
                                     f"no args.om_case in the file")
    return None

def arms_json_tag(path):
    """arm_tag recorded in the ARMS.json of the file's arms_<hash8> directory (runner summaries carry no tag of their own), else ''."""
    d = os.path.dirname(path)
    while d.startswith("results") and d != "results":
        a = os.path.join(d, "ARMS.json")
        if os.path.exists(a):
            try: return json.load(open(a)).get("arm_tag", "") or ""
            except Exception: return ""
        d = os.path.dirname(d)
    return ""

def vintage_generation(tag):
    """'vintage' | 'blended' | None from an arm tag. Tokens are matched as substrings: stored tags use '__' between
    arms in summaries and a single '_' in the NDC diagnostics."""
    if not tag: return None
    if re.search(r"gas_vintage_dispatch=True(?![A-Za-z])", tag): return "vintage"
    if re.search(r"gas_vintage_dispatch=False(?![A-Za-z])", tag) and re.search(r"(?<![A-Za-z])exempt(?![A-Za-z])", tag): return "vintage"
    if "capital_window_case=" in tag: return "blended"
    return None

def dir_hash(path):
    for part in reversed(path.split(os.sep)[:-1]):
        m = re.search(r"_([0-9a-f]{8})$", part)
        if m: return m.group(1)
    return ""

def main():
    rows = []
    for p in sorted(glob.glob("results/**/*.json", recursive=True)):
        b = os.path.basename(p)
        if not ("summary" in b or b.startswith("diagnostics")):
            continue
        st = os.stat(p)
        mt = dt.datetime.fromtimestamp(st.st_mtime)
        raw = open(p, "rb").read()
        md5 = hashlib.md5(raw).hexdigest()
        try:
            d = json.loads(raw)
        except Exception:
            d = None
        arm = {}
        if isinstance(d, dict):
            for k in ARM_KEYS:
                if k in d: arm[k] = d[k]
            # arm keys nested one level (some runners nest under a config block)
            for kk in ("arm", "config", "arms"):
                if isinstance(d.get(kk), dict):
                    for k in ARM_KEYS:
                        if k in d[kk]: arm.setdefault(k, d[kk][k])
        tag = d.get("arm_tag", "") if isinstance(d, dict) else ""
        for k, pat in TAG_PATTERNS.items():
            if k not in arm and tag:
                mm = re.search(pat, tag)
                if mm: arm[k] = mm.group(0)
        m = re.search(r"__([0-9a-f]{8})\.json$", b)
        fhash = m.group(1) if m else dir_hash(p)
        sliced = ""
        if isinstance(d, dict):
            cs = str(d.get("cap_scenario", ""))
            def has_slice(x): return "slice_detail_2025" in x or "slice_detail_terminal_year" in x
            sib = None
            if b.startswith("summary"):
                sp = os.path.join(os.path.dirname(p), b.replace("summary", "diagnostics", 1))
                if os.path.exists(sp):
                    try: sib = json.load(open(sp))
                    except Exception: sib = None
            if "SLICED" in cs.upper() or has_slice(d) or (isinstance(sib, dict) and has_slice(sib)):
                sliced = "sliced"
            elif "decision_variables" in d or (isinstance(sib, dict) and "unserved_twh_by_year" in sib):
                sliced = "annual"
        elog = ehead = verif = ""
        vgen = vintage_generation(tag or arms_json_tag(p))
        none_keys = [k for k, v in arm.items() if v is None]
        if "/_pre_" in "/" + p or "/_superseded" in "/" + p: cls, rule = "LEGACY", "L1"
        elif none_keys: cls, rule = "LEGACY", "L2"
        elif arm.get("tariff_case") == "tariff_legacy_65m": cls, rule = "LEGACY", "L3"
        elif "pre_f1_overnight" in tag or "pre_f1_overnight" in b: cls, rule = "LEGACY", "L4"
        elif mt < WIN_LO: cls, rule = "LEGACY", "L5"
        elif vgen == "vintage": cls, rule = "CURRENT", "C4" if tag else "C5"
        elif vgen == "blended": cls, rule = "LEGACY", "L6"
        elif mt >= WIN_HI: cls, rule = "UNCLASSIFIABLE", "U1"
        elif not arm and not tag:
            ev = find_epoch_log(p, st.st_mtime)
            if ev: cls, rule, elog, ehead = "CURRENT", "C2", ev[0], ev[1]
            else: cls, rule = "UNCLASSIFIABLE", "U2"
        elif (arm or tag) and "om_case" not in arm:
            ev = verify_26_literal(p, st.st_mtime) if "/f3" in p and b == "summary.json" else None
            if ev: cls, rule, elog, ehead, verif = "CURRENT", "C3", ev[0], ev[1], ev[2]
            else: cls, rule = "UNCLASSIFIABLE", "U4"
        else: cls, rule = "CURRENT", "C1"
        rows.append(dict(path=p, size=st.st_size, mtime=mt.isoformat(timespec="seconds"), md5=md5,
                         arm_hash=fhash, epoch_log=elog, epoch_head=ehead, verification=verif, builder=sliced, epoch=cls, rule=rule,
                         none_keys="|".join(none_keys), arm_tag=tag,
                         **{k: ("" if k not in arm else json.dumps(arm[k]) if not isinstance(arm[k], str) else arm[k]) for k in ARM_KEYS}))
    bypath = {r["path"]: r for r in rows}
    for r in rows:
        b = os.path.basename(r["path"])
        if b.startswith("diagnostics"):
            sp = os.path.join(os.path.dirname(r["path"]), b.replace("diagnostics", "summary", 1))
            if sp in bypath and bypath[sp]["epoch"] != r["epoch"]:
                r["epoch"], r["rule"] = bypath[sp]["epoch"], "S:" + bypath[sp]["rule"]
            if sp in bypath and not r["om_case"]:
                r["om_case"] = bypath[sp]["om_case"]
    cols = ["path", "size", "mtime", "md5", "arm_hash", "builder", "epoch", "rule", "epoch_log", "epoch_head", "verification", "none_keys"] + ARM_KEYS + ["arm_tag"]
    with open("docs/results_inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(len(rows), "rows")

if __name__ == "__main__":
    main()
