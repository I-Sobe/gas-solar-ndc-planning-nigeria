"""TASK 48 G.1: docs/results_manifest.json -- every CURRENT result file with its md5,
arm configuration, mtime and, where a Phase B/C log records it, the HEAD it was produced at.

Reads docs/results_inventory.csv (CURRENT rows) and logs/phase_b, logs/phase_c.  Commit the
manifest, not the result files (results/ is gitignored).

A file enters the manifest when it is
  - a CURRENT inventory row (evidence standard C1 arm keys in the artefact, S sibling-of-
    summary, or C2 path-plus-log: the WEAKER standard, register L84), or
  - a sibling file in a directory holding a CURRENT inventory row, written within
    SIBLING_TOL_S of that row (standard S2: same run, by timestamp).  Siblings are the
    per-year CSVs and timeseries that findings cite.
Every entry carries `evidence` so a reader can see which standard admitted it.
HEAD is taken from a log header only when the log's text names the file (basename or its
hash8) and the log's mtime is within LOG_TOL_S of the file's; otherwise it is null.
"""
import csv, datetime as dt, glob, hashlib, json, os, re, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
SIBLING_TOL_S = 30 * 60
LOG_TOL_S = 15 * 60
ARM_COLS = ["om_case", "conversion_case", "capital_case", "tariff_case", "solar_build_case", "policy",
            "peak_case", "storage_cc_case", "gas_build_case", "new_gas_deliverability",
            "gas_availability_case", "gas_retirement_case", "gas_capex_case", "ccgt_efficiency_case",
            "capital_recovery_case", "dscr_case", "capital_treatment_case", "concessional_envelope_case",
            "gas_deliverability_case", "hydro_case", "cap_scenario"]

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def load_logs():
    out = []
    for lg in sorted(glob.glob("logs/phase_b/*.log") + glob.glob("logs/phase_c/*.log")):
        txt = open(lg, errors="replace").read()
        m = re.search(r"^HEAD\s*:\s*([0-9a-f]{40})", txt, re.M)
        out.append(dict(path=lg, mtime=os.stat(lg).st_mtime, head=m.group(1) if m else None, text=txt))
    return out

def head_for(path, mtime, logs):
    base = os.path.basename(path)
    h8 = re.findall(r"(?:__|_)([0-9a-f]{8})(?:\.json|\.csv|$)", base) or re.findall(r"_([0-9a-f]{8})/", path + "/")
    keys = {base} | set(h8)
    best = None
    for lg in logs:
        if not lg["head"] or abs(lg["mtime"] - mtime) > LOG_TOL_S: continue
        # a log names a file by the path it prints; require the directory + basename or hash8
        if any(k in lg["text"] for k in keys) and (os.path.dirname(path) in lg["text"] or any(k in lg["text"] for k in h8)):
            d = abs(lg["mtime"] - mtime)
            if best is None or d < best[0]: best = (d, lg)
    return (best[1]["head"], best[1]["path"]) if best else (None, None)

def main():
    inv = [r for r in csv.DictReader(open("docs/results_inventory.csv")) if r["epoch"] == "CURRENT"]
    logs = load_logs()
    entries = {}
    anchors_by_dir = {}
    for r in inv:
        p = r["path"]
        if not os.path.exists(p): continue
        mt = os.stat(p).st_mtime
        hd, lg = head_for(p, mt, logs)
        if r["rule"] in ("C2", "C3") and not hd:
            hd, lg = r["epoch_head"], r["epoch_log"]
        entries[p] = dict(
            path=p, md5=md5(p), size=os.stat(p).st_size,
            mtime=dt.datetime.fromtimestamp(mt).isoformat(timespec="seconds"),
            epoch_rule=r["rule"],
            evidence={"C1": "arm keys recorded in the artefact", "S:C1": "sibling summary carries the arm keys",
                      "C3": "runner log records a clean-tree HEAD at which scripts/26_run_f3_identification.py hard-codes om_case=\"om_central\" (see `verification`); the artefact itself does not record om_case",
                      "C2": "directory path identifies the arm; runner log header fixes the epoch (WEAKER standard, register L84)",
                      "C4": "arm_tag in the artefact carries gas_vintage_dispatch=True (vintage-dispatch generation), or False on an exempt-deliverability tag (flag forced off, part of the arm)",
                      "S:C4": "sibling summary carries the vintage-generation arm_tag",
                      "C5": "arm_tag read from the ARMS.json sidecar of the arms_<hash8> directory (the artefact carries none): WEAKER than C4",
                      "S:C5": "sibling summary classed by the ARMS.json arm_tag"}[r["rule"]],
            arm={c: r[c] for c in ARM_COLS if r.get(c)}, arm_tag=r["arm_tag"] or None, arm_hash=r["arm_hash"] or None,
            verification=r.get("verification") or None,
            head=hd, head_source_log=lg)
        anchors_by_dir.setdefault(os.path.dirname(p), []).append(mt)
    for d, mts in anchors_by_dir.items():
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if p in entries or not os.path.isfile(p) or f == "INDEX.md": continue
            mt = os.stat(p).st_mtime
            if not any(abs(mt - a) <= SIBLING_TOL_S for a in mts): continue
            hd, lg = head_for(p, mt, logs)
            entries[p] = dict(path=p, md5=md5(p), size=os.stat(p).st_size,
                              mtime=dt.datetime.fromtimestamp(mt).isoformat(timespec="seconds"),
                              epoch_rule="S2", evidence="sibling of a CURRENT row in the same directory, written within 30 min of it",
                              arm=None, arm_tag=None, arm_hash=None, head=hd, head_source_log=lg)
    # TASK 82 A.2: a directory whose ARMS.json sidecar records a vintage-generation arm_tag (rule S3) is admitted whole.
    # The inventory classifies only summary*/diagnostics* JSON, so the CSV/PDF/PNG and other JSON outputs of the
    # vintage runners (pareto, capital_frontier, monte_carlo, hydro_minflow, ...) have no CURRENT row to be a sibling of.
    from build_results_inventory import vintage_generation
    for a in sorted(glob.glob("results/**/ARMS.json", recursive=True)):
        try: atag = json.load(open(a)).get("arm_tag", "") or ""
        except Exception: continue
        if vintage_generation(atag) != "vintage": continue
        for root, _, fs in os.walk(os.path.dirname(a)):
            for f in sorted(fs):
                p = os.path.join(root, f)
                if p in entries or f == "INDEX.md": continue
                mt = os.stat(p).st_mtime
                hd, lg = head_for(p, mt, logs)
                entries[p] = dict(path=p, md5=md5(p), size=os.stat(p).st_size,
                                  mtime=dt.datetime.fromtimestamp(mt).isoformat(timespec="seconds"),
                                  epoch_rule="S3", evidence="file in a directory whose ARMS.json arm_tag carries the vintage token (" + a + ")",
                                  arm=None, arm_tag=atag, arm_hash=None, head=hd, head_source_log=lg)
    # TASK 83 B.2 rule S4: Monte Carlo / VSS draws carry the arm tag in their own _meta.json (`draws_tag`). A draws file whose meta
    # carries the vintage token is admitted with its CSV, and a vss_bound JSON whose guards name that CSV as its stage-2 draws file is
    # admitted by that chain (the vss_bound file itself carries no tag).
    vdraws = {}
    for m in sorted(glob.glob("results/stochastic/draws__*_meta.json")):
        try: mt_ = json.load(open(m))
        except Exception: continue
        if vintage_generation(mt_.get("draws_tag", "")) != "vintage": continue
        stem = m[:-len("_meta.json")]
        vdraws[os.path.basename(stem)] = mt_["draws_tag"]
        for p in (m, stem + ".csv"):
            if p not in entries and os.path.exists(p):
                mt = os.stat(p).st_mtime; hd, lg = head_for(p, mt, logs)
                entries[p] = dict(path=p, md5=md5(p), size=os.stat(p).st_size, mtime=dt.datetime.fromtimestamp(mt).isoformat(timespec="seconds"),
                                  epoch_rule="S4", evidence="draws_tag in results/stochastic/*_meta.json carries the vintage token",
                                  arm=None, arm_tag=mt_["draws_tag"], arm_hash=None, head=hd, head_source_log=lg)
    for p in sorted(glob.glob("results/vss_bound/vss_bound__*.json")):
        try: g = json.load(open(p))["guards"]["same_draws_sha256"]["stage2_draws_file"]
        except Exception: continue
        if os.path.basename(g)[:-len(".csv")] in vdraws and p not in entries:
            mt = os.stat(p).st_mtime; hd, lg = head_for(p, mt, logs)
            entries[p] = dict(path=p, md5=md5(p), size=os.stat(p).st_size, mtime=dt.datetime.fromtimestamp(mt).isoformat(timespec="seconds"),
                              epoch_rule="S4", evidence="guards.same_draws_sha256.stage2_draws_file names " + os.path.basename(g) + ", whose meta carries the vintage token",
                              arm=None, arm_tag=None, arm_hash=None, head=hd, head_source_log=lg)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    man = dict(
        generated=dt.datetime.now().isoformat(timespec="seconds"), generator="scripts/build_results_manifest.py",
        repo_head_at_generation=head,
        tag="(none: a tag tying this manifest to the repository is created by hand after commit; see docs/task_log_48_evidence_spine.md G.1)",
        n_files=len(entries),
        evidence_standards={
            "C1": "arm keys recorded in the artefact itself (strongest)",
            "S:C1": "diagnostics file inheriting its sibling summary's class",
            "C2": "WEAKER: path identifies the arm, runner-log header + mtime fix the epoch (register L84)",
            "C4": "arm_tag in the artefact carries the vintage token (TASK 82 A.2)",
            "C5": "WEAKER: arm_tag from the ARMS.json sidecar of the arms_<hash8> directory (TASK 82 A.2)",
            "S4": "draws file whose _meta.json draws_tag carries the vintage token, or a vss_bound file naming such a draws file (TASK 83 B.2)",
            "S3": "file in a directory whose ARMS.json arm_tag carries the vintage token (TASK 82 A.2)",
            "C3": "U4 file admitted by decision: clean-tree log HEAD + the om_central literal verified at that commit (recorded per file in `verification`)",
            "S2": "sibling file written within 30 minutes of a CURRENT row in the same directory"},
        files=sorted(entries.values(), key=lambda e: e["path"]))
    json.dump(man, open("docs/results_manifest.json", "w"), indent=1)
    c = {}
    for e in entries.values(): c[e["epoch_rule"]] = c.get(e["epoch_rule"], 0) + 1
    print(len(entries), "files;", c, "; with HEAD:", sum(1 for e in entries.values() if e["head"]))

if __name__ == "__main__":
    main()
