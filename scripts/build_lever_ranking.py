"""TASK 48 Part F.4, extended by TASK 59 Part 3: the lever ranking over the POST-EPOCH baseline arms.

Reads every results/baseline_sliced/summary__*.json and solves nothing.  Lever families and the
span rule come from src/scenarios.py (LEVER_FAMILIES): a lever's magnitude is max minus min of
npv_total_cost_usd across every arm in its declared family, the central arm included.  All sixteen
declared families are reported; `in_f5_ranking` is emitted as a column and is not a filter.

ADMISSION (epoch filter, TASK 59 gate).  A file is admitted only if its arm_tag carries
capital_window_case=arrears, storage_capex_case=declining_nrel, tariff_case=tariff_central and
us_inflation=0.02.  The docs/results_inventory.csv gate of TASK 48 is no longer read; the
provenance it provided (a file silently overwritten between runs, L85) is recorded instead in
lever_ranking_summary.json for every file read: path, hash8, md5, mtime, npv_total_cost_usd.

MEMBERSHIP comes from LEVER_FAMILIES, not from inference: a member's tag is derived (arm_for) and
the file at its hash8 must carry exactly that arm_tag and the om_case the member requires.

BUILD RESPONSE (measured, not asserted).  For each family the decision_variables blocks of its
arms are compared.  A family whose arms all produce one identical plan rescales the cost of a plan
determined elsewhere; it is listed apart from the ranked table.  Column 2 of the span excludes an
arm only if its plan is identical to the plan of every other arm in the family, so it is empty for
such a family and equal to column 1 otherwise.

LIVE-ARM RULE (TASK 70, column 3).  A family's reported span is taken over its live arms.  A member
that LEVER_FAMILIES declares `unasserted` (unsourced, represents no state or bound the thesis
asserts, retained only so superseded results stay attributable) is excluded from column 3 and listed
with its declared reason.  The declaration is the authority; no comment is parsed.  The ranked table
is ordered by column 3; column 1 (all arms) is kept, with its own rank.  For every arm named in
`unasserted` or `candidate_notes` the generator also emits the family span with that arm excluded,
so the rule's effect is visible on every arm it could have touched.  Build-response classification
is computed over all arms and is not changed by the rule.

NOT RANKED, reported in their own block: `bound` families (hydro) and the social_discount_rate
files (a reporting convention, not a lever).  Joint arms and values no family declares are listed.

CAPPED MODE (TASK 82 B.1, `--capped ndc3_conditional|ndc3_unconditional`).  Reads results/ndc/*/summary.json
(+ diagnostics.json) written by 02_run_ndc_caps.py instead of results/baseline_sliced.  That layout differs: one
directory per (cap case, arm) whose name is a hash8 of a tag in a different format (single '_' separators, cap case first),
so the derived tag cannot be reproduced.  Membership is therefore decided by the STORED case fields: the fifteen arm fields
of S.BASELINE_ARM_TAG_FIELDS, om_case, and cap_scenario must equal the member's; the epoch tokens are read from the stored
tag as substrings.  02_ stores no gas capacity additions or storage additions by year, so "plan" in capped mode is
(solar additions by year, storage capacity 2054, gas thermal draw by year): a gas-build difference that changes none of
those is not seen.  Outputs go to results/lever_ranking_capped/<cap>/.  A per-arm table (capped_arms.csv) carries the
uncapped counterpart, the cap-binding years and the leakage share for every arm, and a rank table puts each family's
uncapped and capped rank side by side.

REFUSES (exit 2) on an arm_tag or om_case mismatch.  A refusal goes to stderr and writes nothing;
it never overwrites docs/lever_ranking.md (TASK 59).  A family with a missing member is not
ranked and is listed under "declared but not ranked".
"""
import argparse, csv, datetime as dt, glob, hashlib, json, os, re, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from src import scenarios as S
from src.output_naming import hash8_of, om_tag_suffix, append_index

BASE = "results/baseline_sliced"
OUTBASE = "results/lever_ranking"
CAPPED = None   # set by --capped
CAP_PATH = "data/cost/processed/emissions_cap.csv"
# the tag tokens that mark the post-epoch generation; the tag of every admitted file carries all five (the fifth marks the vintage-dispatch generation, TASK 81)
EPOCH_TOKENS = ("capital_window_case=arrears", "storage_capex_case=declining_nrel",
                "tariff_case=tariff_central", "us_inflation=0.02", "gas_vintage_dispatch=True")
# order of the suffix inside the stored tag
EPOCH_SUFFIX = "__capital_window_case=arrears__gas_vintage_dispatch=True__storage_capex_case=declining_nrel__tariff_case=tariff_central__us_inflation=0.02"
SDR_RE = re.compile(r"__social_discount_rate=[0-9.]+")
ADD_SERIES = ("solar_add_mw_by_year", "gas_add_mw_by_year", "storage_add_mwh_by_year")
CUM_2054 = ("solar_cumulative_built_2054_mw", "gas_cumulative_built_2054_mw", "storage_capacity_2054_mwh")
# capped mode: the quantities 02_ stores. The gas entry is a THERMAL DRAW (TWh_th), not capacity: renamed so it cannot be read as MW.
CAPPED_ADD_SERIES = ("solar_add_mw_by_year", "gas_to_power_twh_th_by_year")
CAPPED_CUM = ("solar_cumulative_built_2054_mw", "gas_to_power_cumulative_twh_th", "storage_capacity_2054_mwh")
CAPPED_TOL_MW = 1e-3
# TASK 83 A.2: families whose arms differ in DISPATCH MODEL, so their span is not a property of the lever. Reported in their own block.
NOT_COMPARABLE = {"new_gas_deliverability": "the exempt arm cannot run vintage dispatch (the builder raises if it is on), so it is a single-stream arm "
                  "compared with a vintage-dispatch constrained arm: the span is the dispatch-model difference, not the value of deliverability "
                  "(capped, exempt equals the BLENDED constrained objective to the cent; L138, L145)"}
ENVELOPE = "carried"
CAPPED_FIELDS = tuple(S.BASELINE_ARM_TAG_FIELDS) + ("om_case",)


def _has(tag, tok):
    """tok as a whole token of the arm tag: '__'-joined in the baseline summaries, '_'-joined in the capped runners'."""
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(tok) + r"(?![A-Za-z0-9.])", tag or "") is not None


def epoch_tokens_ok(tag, indices):
    """True if the arm tag carries EPOCH_TOKENS[indices]. TASK 82 A.1: the vintage token may read False
    only on an exempt-deliverability tag, where the flag is forced off (L138)."""
    for i in indices:
        t = EPOCH_TOKENS[i]
        if _has(tag, t): continue
        if t == "gas_vintage_dispatch=True" and _has(tag, "gas_vintage_dispatch=False") and re.search(r"(?<![A-Za-z])exempt(?![A-Za-z])", tag or ""): continue
        return False
    return True


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def arm_for(field, value):
    """-> (tag, om_case, hash8) for the family member `field`=`value`, post-epoch tag."""
    arm = dict(S.BASELINE_CENTRAL_ARM); om = S.BASELINE_CENTRAL_OM_CASE
    if field == "om_case": om = value
    else: arm[field] = value
    suffix = EPOCH_SUFFIX
    # TASK 82 A.1 (L138): under new_gas_deliverability=exempt the builder forces gas_vintage_dispatch off
    # (src/optimize_model_sliced.py raises if it is on), so that member's tag states the EFFECTIVE value False.
    if field == "new_gas_deliverability" and value == "exempt":
        suffix = suffix.replace("gas_vintage_dispatch=True", "gas_vintage_dispatch=False")
    tag = "__".join(arm[f] for f in S.BASELINE_ARM_TAG_FIELDS) + om_tag_suffix(om) + suffix
    return tag, om, hash8_of(tag)


def hash_of_path(p):
    return os.path.basename(p)[len("summary__"):-len(".json")]


def plan_same(a, b):
    """Two decision blocks are one plan. Baseline: exact. Capped: solar additions and storage 2054 within CAPPED_TOL_MW (1e-3 MW/MWh,
    a kilowatt: below it two solves of one LP differ by vertex noise), gas thermal draw within 1e-6 TWh_th. TASK 82: with an exact test the
    capped solves put zero-span families (storage_cc, dscr at the unconditional cap) in the ranked table on float noise."""
    if not CAPPED: return a == b
    return all(abs(a["solar_add_mw_by_year"][y] - b["solar_add_mw_by_year"][y]) <= CAPPED_TOL_MW for y in a["solar_add_mw_by_year"]) \
        and abs(a[CUM_2054[2]] - b[CUM_2054[2]]) <= CAPPED_TOL_MW and abs(a[CUM_2054[1]] - b[CUM_2054[1]]) <= 1e-6


def build_response(dvs):
    """dvs: list of decision_variables dicts -> measured build response across them."""
    differ = any(not plan_same(dv, dvs[0]) for dv in dvs[1:])
    mx = {}
    for k in CUM_2054:
        vals = [dv[k] for dv in dvs]
        mx[k] = max(vals) - min(vals)
    first = None
    for ser in ADD_SERIES:
        tol = (1e-6 if ser.startswith("gas_to_power") else CAPPED_TOL_MW) if CAPPED else 0.0   # capped: below the tolerance is vertex noise (see plan_same)
        for y in sorted(dvs[0][ser], key=int):
            vals = [dv[ser][y] for dv in dvs]
            if max(vals) - min(vals) > tol:
                first = int(y) if first is None else min(first, int(y)); break
    groups = []
    for i, dv in enumerate(dvs):
        for g in groups:
            if plan_same(dvs[g[0]], dv): g.append(i); break
        else: groups.append([i])
    return dict(differ=differ, max_diff=mx, first_diff_year=first, groups=groups)


def capped_hash(p):
    """hash8 suffix of the result directory; a pre-hash directory (full tag in its name) has none and is keyed by its name."""
    d = os.path.basename(os.path.dirname(p)); m = re.search(r"_([0-9a-f]{8})$", d)
    return m.group(1) if m else "prehash:" + d


def capped_record(p, cap):
    """-> rec dict for one 02_ summary.json, or None if it is not a `cap` case. The decision block is rebuilt from the
    summary and its sibling diagnostics.json (see CAPPED MODE in the module docstring)."""
    d = json.load(open(p))
    if d.get("cap_scenario") != cap: return None
    if d.get("capital_case") != ("unconstrained" if ENVELOPE == "none" else {"ndc3_unconditional": "tight", "ndc3_conditional": "moderate"}[cap]): return None
    dg = {}
    try: dg = json.load(open(os.path.join(os.path.dirname(p), "diagnostics.json")))
    except Exception: pass
    sol = d["decision_variables"]["solar_add_mw_by_year"]
    gas = dg.get("fleet_gas_to_power_twh_th_by_year", {})
    d["decision_variables"] = {"solar_add_mw_by_year": sol, "gas_to_power_twh_th_by_year": gas,
        CAPPED_CUM[0]: sum(sol.values()), CAPPED_CUM[1]: sum(gas.values()),
        CAPPED_CUM[2]: d["decision_variables"]["storage_capacity_mwh"]}
    d["_diag"] = dg
    return d


def load_caps(cap):
    import pandas as pd
    df = pd.read_csv(CAP_PATH); df = df[df["scenario"] == cap]
    return {int(y): float(c) for y, c in zip(df["year"], df["cap_tco2"])}


def capped_extras(outdir, fam_out, admitted, rows, ranked):
    """Capped mode only: per-arm cap-binding table (capped_arms.csv), failed solves, and the uncapped-against-capped rank table.
    -> list of markdown lines."""
    caps = load_caps(CAPPED); out = ["", f"## Per-arm cap binding under `{CAPPED}` (TASK 82 B.4)", "",
        "`cap_binding` is True where the annual cap's shadow price (`carbon_shadow_yr_usd_per_tco2_by_year`) is non-zero (|x| > 1e-6 USD/tCO2) in at least one year. "
        "`unc>cap yrs` counts the years in which the UNCAPPED counterpart's total emissions exceed the cap: if it is 0 the uncapped plan already meets the cap in every year. "
        "The capped runs also carry a public-capital envelope the uncapped run does not (`capital_case` tight/moderate), so `capped - uncapped` is cap plus envelope, not cap alone.", "",
        "| family | member | capped (bn) | uncapped (bn) | capped - uncapped (bn) | cap_binding | binding yrs | first-last | unc>cap yrs | emissions Mt | leakage share | leakage flagged |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    cols = ["family", "member", "capped_hash", "capped_npv_usd", "uncapped_hash", "uncapped_npv_usd", "capped_minus_uncapped_usd", "cap_binding",
            "n_binding_years", "first_binding_year", "last_binding_year", "n_years_uncapped_total_emissions_above_cap", "capped_emissions_tco2",
            "leakage_share", "leakage_flagged", "capped_path"]
    recs = []
    for fam, arms, inc in fam_out:
        for a in arms:
            r = admitted[a["hash"]]; dg = r["d"].get("_diag", {})
            sh = dg.get("carbon_shadow_yr_usd_per_tco2_by_year", {})
            by = sorted(int(y) for y, v in sh.items() if v is not None and abs(v) > 1e-6)
            _, _, uh = arm_for(fam["field"], a["value"])
            up = f"results/baseline_sliced/summary__{uh}.json"; un = ud = None
            if os.path.exists(up):
                un = json.load(open(up))["npv_total_cost_usd"]
                ud = json.load(open(f"results/baseline_sliced/diagnostics__{uh}.json")).get("emissions_tco2_by_year", {})
            above = None if not ud else sum(1 for y, c in caps.items() if ud[str(y)] > c * (1 + 1e-9))
            lk = r["d"].get("_diag", {}).get("leakage", {})
            rec = dict(family=fam["id"], member=a["value"], capped_hash=a["hash"], capped_npv_usd=a["npv"], uncapped_hash=uh, uncapped_npv_usd=un,
                       capped_minus_uncapped_usd=None if un is None else a["npv"] - un, cap_binding=bool(by), n_binding_years=len(by),
                       first_binding_year=by[0] if by else "", last_binding_year=by[-1] if by else "", n_years_uncapped_total_emissions_above_cap=above,
                       capped_emissions_tco2=r["d"].get("actual_emissions_tco2_total"), leakage_share=lk.get("leakage_share"),
                       leakage_flagged=lk.get("flagged"), capped_path=a["path"])
            recs.append(rec)
            out.append(f"| {fam['id']} | {a['value']} | {a['npv']/1e9:.4f} | {'' if un is None else format(un/1e9, '.4f')} | "
                       f"{'' if un is None else format((a['npv']-un)/1e9, '.4f')} | {'**yes**' if by else '**NO**'} | {len(by)} | "
                       f"{(str(by[0]) + '-' + str(by[-1])) if by else '-'} | {above if above is not None else 'n/a'} | "
                       f"{(rec['capped_emissions_tco2'] or 0)/1e6:.1f} | {'' if rec['leakage_share'] is None else format(rec['leakage_share'], '.4f')} | {rec['leakage_flagged']} |")
    with open(os.path.join(outdir, "capped_arms.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(recs)
    nb = sorted({(r["family"], r["member"]) for r in recs if not r["cap_binding"]})
    out += ["", f"Arms (family, member rows) where the cap does not bind: {len(nb)} of {len(recs)}" + (": " + "; ".join(f"{f}/{m}" for f, m in nb) if nb else "."), ""]
    rj = f"logs/task82/capped{'_unconstrained' if ENVELOPE == 'none' else ''}_{CAPPED}_rows.json"
    out += ["## Solves that did not return an optimal plan", ""]
    bad = [r for r in json.load(open(rj))["rows"] if r["rc"] != 0] if os.path.exists(rj) else None
    out += (["driver rows not found (" + rj + "); no statement"] if bad is None else
            [f"* `{r['label']}` (uncapped arm `{r['uncapped_hash']}`): rc {r['rc']}: {r['err']}" for r in bad] or ["none: every driver invocation exited 0 (02_ raises on a non-optimal status)."])
    # rank table: uncapped col-3 rank (from the uncapped ranking csv) against this ranking
    ucsv = "results/lever_ranking/arms_661b03c1/lever_ranking.csv"
    ur = {}
    if os.path.exists(ucsv):
        rows_u = [r for r in csv.DictReader(open(ucsv)) if r["role"] == "ranked"]
        rows_u.sort(key=lambda r: -float(r["col3_live_span_usd"]))
        ur = {r["id"]: (i, float(r["col3_live_span_usd"])) for i, r in enumerate(rows_u, 1)}
    cr = {r["id"]: (i, r["span3"]) for i, r in enumerate(ranked, 1)}
    out += ["", f"## Uncapped rank against capped rank (TASK 82 B.7), column 3 (live arms)", "",
            f"Uncapped: `{ucsv}`. Capped: this table's `{CAPPED}` ranking. A family is ranked only where its arms produce different plans; otherwise its role is shown.", "",
            "| family | uncapped rank | uncapped span (bn) | capped rank | capped span (bn) | capped role |", "|---|---|---|---|---|---|"]
    for r in rows:
        u = ur.get(r["id"]); c = cr.get(r["id"])
        out.append(f"| {r['id']} | {u[0] if u else '-'} | {format(u[1]/1e9, '.4f') if u else '-'} | {c[0] if c else '-'} | "
                   f"{format(c[1]/1e9, '.4f') if c else (format(r['span']/1e9, '.4f') if 'span' in r else '-')} | {r['role']} |")
    return out


def main():
    global CAPPED, BASE, OUTBASE, ADD_SERIES, CUM_2054
    global ENVELOPE
    ap = argparse.ArgumentParser(); ap.add_argument("--capped", choices=("ndc3_conditional", "ndc3_unconditional"))
    ap.add_argument("--envelope", choices=("carried", "none"), default="carried",
                    help="capped mode: read the solves that carry the cap's public-capital envelope (tight/moderate) or the ones with capital_case=unconstrained (TASK 83 A.1)")
    _a = ap.parse_args(); CAPPED = _a.capped; ENVELOPE = _a.envelope
    if CAPPED:
        BASE = "results/ndc"; OUTBASE = f"results/lever_ranking_capped/{CAPPED}" + ("_noenvelope" if ENVELOPE == "none" else "")
        ADD_SERIES, CUM_2054 = CAPPED_ADD_SERIES, CAPPED_CUM
    problems = []
    GASHDR = "max d gas thermal draw 2025-54 TWh_th" if CAPPED else "max d gas 2054 MW"
    # ---- read every summary; admit by the epoch tokens -------------------------------------------
    files = {}
    for p in sorted(glob.glob(f"{BASE}/ndc*/summary.json") if CAPPED else glob.glob(f"{BASE}/summary__*.json")):
        d = capped_record(p, CAPPED) if CAPPED else json.load(open(p))
        if d is None: continue   # another cap case: not part of this ranking
        tag = d.get("arm_tag")
        rec = dict(path=p, hash8=(capped_hash(p) if CAPPED else hash_of_path(p)), md5=md5(p),
                   mtime=dt.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds"),
                   npv=d["npv_total_cost_usd"], d=d)
        toks = tag.split("__") if tag else []
        if tag is None: rec["excluded"] = "no arm_tag key (pre-hash legacy file)"
        elif not epoch_tokens_ok(tag, (0, 1, 3, 4)):
            rec["excluded"] = "arm_tag lacks the post-epoch tokens (pre-epoch generation)"
        elif not _has(tag, "tariff_case=tariff_central"):
            rec["excluded"] = "tariff_case token is not tariff_central (" + next(t for t in toks if t.startswith("tariff_case=")) + ")"
        else: rec["excluded"] = None
        files[rec["hash8"]] = rec
    admitted = {h: r for h, r in files.items() if r["excluded"] is None}
    def find(field, value):
        """-> (record or None, expected tag or None, om, hash8, problems[]) for the family member field=value."""
        arm = dict(S.BASELINE_CENTRAL_ARM); om = S.BASELINE_CENTRAL_OM_CASE
        if field == "om_case": om = value
        else: arm[field] = value
        if not CAPPED:
            tag, _, h = arm_for(field, value); return admitted.get(h), tag, om, h, []
        want = dict(arm, om_case=om)
        hit = [r for r in admitted.values() if all(r["d"].get(f) == want[f] for f in CAPPED_FIELDS)]
        if len(hit) > 1:
            return None, None, om, "", [f"{value}: {len(hit)} admitted files carry the same stored case fields: " + ", ".join(r["path"] for r in hit)]
        return (hit[0] if hit else None), None, om, (hit[0]["hash8"] if hit else ""), []
    _cr, ctag, com, chash, _cp = find("conversion_case", "conv_central")
    if _cp or _cr is None:
        print(f"REFUSED: the central arm {chash or '(none)'} is not an admitted file. {' '.join(_cp)}", file=sys.stderr); sys.exit(2)
    if CAPPED: ctag = _cr["d"]["arm_tag"]
    cnpv = admitted[chash]["npv"]
    # ---- families --------------------------------------------------------------------------------
    fam_out, used = [], {chash}
    for fam in S.LEVER_FAMILIES:
        arms, incomplete = [], []
        for v in fam["members"]:
            r, tag, om, h, pr = find(fam["field"], v)
            problems.extend(pr)
            if r is None:
                incomplete.append(f"{v} (expected " + (f"stored fields of {CAPPED} in {BASE}/*/summary.json" if CAPPED else f"{BASE}/summary__{h}.json") + ": not an admitted file)"); continue
            if r["d"].get("om_case") != om:
                problems.append(f"{r['path']}: records om_case={r['d'].get('om_case')}, family member {v} requires {om}."); continue
            if not CAPPED and r["d"].get("arm_tag") != tag:
                problems.append(f"{r['path']}: arm_tag in file differs from the tag derived from LEVER_FAMILIES."); continue
            used.add(h)
            arms.append(dict(value=v, hash=h, path=r["path"], npv=r["npv"], dv=r["d"]["decision_variables"]))
        fam_out.append((fam, arms, incomplete))
    if problems:
        print("REFUSED:\n" + "\n".join(f"* {m}" for m in dict.fromkeys(problems)) +
              "\nNo output was written; docs/lever_ranking.md is untouched.", file=sys.stderr); sys.exit(2)
    # ---- per-family span, build response, two columns --------------------------------------------
    rows = []
    for fam, arms, inc in fam_out:
        row = dict(id=fam["id"], label=fam["label"], field=fam["field"], in_f5_ranking=fam["in_f5_ranking"],
                   bound=fam["bound"], n_declared=len(fam["members"]), n_present=len(arms))
        if inc or len(arms) < len(fam["members"]):
            row.update(role="declared_not_ranked", missing="; ".join(inc)); rows.append(row); continue
        lo = min(arms, key=lambda x: x["npv"]); hi = max(arms, key=lambda x: x["npv"])
        br = build_response([a["dv"] for a in arms])
        # column 2: drop an arm whose plan equals the plan of every other arm in the family
        keep = [a for a, dv in zip(arms, [a["dv"] for a in arms]) if not all(plan_same(dv, o["dv"]) for o in arms if o is not a)]
        if len(keep) >= 2:
            lo2 = min(keep, key=lambda x: x["npv"]); hi2 = max(keep, key=lambda x: x["npv"]); span2 = hi2["npv"] - lo2["npv"]
        else:
            lo2 = hi2 = None; span2 = None
        role = "bound" if fam["bound"] else ("ranked" if br["differ"] else "no_build_response")
        if fam["id"] in NOT_COMPARABLE: role = "not_comparable"
        # column 3: live arms only (LEVER_FAMILIES `unasserted` is the authority)
        unas = fam.get("unasserted", {}); notes = fam.get("candidate_notes", {})
        assert set(unas) <= set(fam["members"]) and set(notes) <= set(fam["members"]) and not (set(unas) & set(notes)), fam["id"]
        def span_without(drop):
            k = [a for a in arms if a["value"] not in drop]
            if len(k) < 2: return None, None, None
            l = min(k, key=lambda x: x["npv"]); h = max(k, key=lambda x: x["npv"]); return h["npv"] - l["npv"], l, h
        span3, lo3, hi3 = span_without(set(unas))
        cands = [dict(arm=a, status="UNASSERTED" if a in unas else "live", reason=(unas.get(a) or notes[a]),
                      span_without=span_without({a})[0]) for a in list(unas) + list(notes)]
        row.update(role=role, span=hi["npv"] - lo["npv"], lo=lo, hi=hi, span_pct=100 * (hi["npv"] - lo["npv"]) / cnpv,
                   br=br, arms=arms, span2=span2, lo2=lo2, hi2=hi2,
                   excluded2=[a["value"] for a in arms if a not in keep],
                   span3=span3, lo3=lo3, hi3=hi3, unasserted=dict(unas), cands=cands)
        rows.append(row)
    ranked = sorted([r for r in rows if r["role"] == "ranked"], key=lambda r: -r["span3"])
    rank1 = {r["id"]: i for i, r in enumerate(sorted([r for r in rows if r["role"] == "ranked"], key=lambda r: -r["span"]), 1)}
    # ---- bounds and conventions block ------------------------------------------------------------
    sdr = []
    for h, r in admitted.items():
        t = r["d"]["arm_tag"]
        if "social_discount_rate=" in t and SDR_RE.sub("", t) == ctag:
            sdr.append(dict(rate=float(re.search(r"social_discount_rate=([0-9.]+)", t).group(1)), hash=h, path=r["path"],
                            npv=r["npv"], dv=r["d"]["decision_variables"]))
    sdr.sort(key=lambda x: x["rate"])
    for s in sdr: used.add(s["hash"])
    sdr_block = None
    if sdr:
        lo = min(sdr, key=lambda x: x["npv"]); hi = max(sdr, key=lambda x: x["npv"])
        sdr_block = dict(arms=sdr, span=hi["npv"] - lo["npv"], lo=lo, hi=hi, span_pct=100 * (hi["npv"] - lo["npv"]) / cnpv,
                         br=build_response([s["dv"] for s in sdr]))
    # ---- joint arms and undeclared values --------------------------------------------------------
    cd = files[chash]["d"]; cmp_fields = list(S.BASELINE_ARM_TAG_FIELDS) + ["om_case", "tariff_case"]
    joint, undeclared_single, unaccounted = [], [], []
    for h, r in files.items():
        t = r["d"].get("arm_tag") or ""
        if not epoch_tokens_ok(t, (0, 1, 3, 4)): continue
        if "social_discount_rate=" in t and r["excluded"] is None and h in used: continue
        diff = {f: (cd.get(f), r["d"].get(f)) for f in cmp_fields if r["d"].get(f) != cd.get(f)}
        if len(diff) >= 2: joint.append((h, r, diff))
        elif r["excluded"] is None and h not in used: unaccounted.append((h, r, diff))
        elif len(diff) == 1 and r["excluded"] is not None: undeclared_single.append((h, r, diff))
    declared = {}
    for fam in S.LEVER_FAMILIES: declared.setdefault(fam["field"], set()).update(fam["members"])
    fn_keys = {"peak_case": S.peak_multiple_scenarios, "gas_availability_case": S.gas_availability_scenarios,
               "dscr_case": S.dscr_scenarios, "gas_deliverability_case": S.gas_deliverability_scenarios}
    undecl = []
    for fld, mem in sorted(declared.items()):
        extra = set(fn_keys[fld]()) - mem if fld in fn_keys else set()
        seen = {}
        for r in files.values():
            v = r["d"].get(fld)
            if v is not None and v not in mem: seen[v] = seen.get(v, 0) + 1
        for v in sorted(set(seen) | extra):
            undecl.append((fld, v, seen.get(v, 0), "a scenario-function key" if v in extra else "stored in files"))
    tv = {}
    for r in files.values():
        v = r["d"].get("tariff_case")
        if v is not None: tv[v] = tv.get(v, 0) + 1
    for v, n in sorted(tv.items()): undecl.append(("tariff_case (stored; no declared family)", v, n, "stored in files"))
    for tok in (() if CAPPED else ("capital_window_case", "storage_capex_case")):
        c = {}
        for r in files.values():
            m = re.search(tok + r"=([a-z_]+?)(?:__|$)", r["d"].get("arm_tag") or "")
            if m: c[m.group(1)] = c.get(m.group(1), 0) + 1
        for v, n in sorted(c.items()): undecl.append((tok + " (tag token only; not stored; no declared family)", v, n, "in arm_tag"))
    # ---- outputs ---------------------------------------------------------------------------------
    outdir = os.path.join(OUTBASE, f"arms_{chash}"); os.makedirs(outdir, exist_ok=True)
    json.dump({"arm_tag": ctag, "hash8": chash, "invocation": sys.argv}, open(os.path.join(outdir, "ARMS.json"), "w"), indent=2)
    append_index(OUTBASE, chash, ctag)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"], capture_output=True, text=True).stdout.strip())
    cols = ["id", "role", "field", "in_f5_ranking", "bound", "n_declared", "n_present", "span_usd", "span_bn", "span_pct_of_central",
            "low_arm", "low_npv", "high_arm", "high_npv", "build_differs", "n_plan_groups", "max_abs_diff_solar_cum_2054_mw",
            ("max_abs_diff_gas_to_power_cum_twh_th" if CAPPED else "max_abs_diff_gas_cum_2054_mw"), "max_abs_diff_storage_2054_mwh", "first_year_additions_differ",
            "col2_span_usd", "col2_low_arm", "col2_high_arm", "col2_excluded_arms", "col2_differs_from_col1",
            "col3_live_span_usd", "col3_live_span_bn", "col3_low_arm", "col3_high_arm", "col3_unasserted_arms", "col3_unasserted_reasons",
            "candidate_span_without_bn"]
    with open(os.path.join(outdir, "lever_ranking.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in rows:
            if r["role"] == "declared_not_ranked":
                w.writerow([r["id"], r["role"], r["field"], r["in_f5_ranking"], r["bound"], r["n_declared"], r["n_present"]] + [""] * 25); continue
            br = r["br"]
            w.writerow([r["id"], r["role"], r["field"], r["in_f5_ranking"], r["bound"], r["n_declared"], r["n_present"],
                        repr(r["span"]), f"{r['span']/1e9:.6f}", f"{r['span_pct']:.4f}", r["lo"]["value"], repr(r["lo"]["npv"]),
                        r["hi"]["value"], repr(r["hi"]["npv"]), br["differ"], len(br["groups"]),
                        repr(br["max_diff"][CUM_2054[0]]), repr(br["max_diff"][CUM_2054[1]]), repr(br["max_diff"][CUM_2054[2]]),
                        br["first_diff_year"] if br["first_diff_year"] is not None else "none",
                        "" if r["span2"] is None else repr(r["span2"]), r["lo2"]["value"] if r["lo2"] else "",
                        r["hi2"]["value"] if r["hi2"] else "", "; ".join(r["excluded2"]),
                        "n/a (family has one plan)" if r["span2"] is None else (r["span2"] != r["span"]),
                        "" if r["span3"] is None else repr(r["span3"]), "" if r["span3"] is None else f"{r['span3']/1e9:.6f}",
                        r["lo3"]["value"] if r["lo3"] else "", r["hi3"]["value"] if r["hi3"] else "",
                        "; ".join(r["unasserted"]), " | ".join(r["unasserted"].values()),
                        "; ".join(f"{c['arm']}={'undefined' if c['span_without'] is None else format(c['span_without']/1e9, '.6f')}" for c in r["cands"])])
    allfiles = [dict(path=r["path"], hash8=r["hash8"], md5=r["md5"], mtime=r["mtime"], npv_total_cost_usd=r["npv"],
                     admitted=r["excluded"] is None, excluded_reason=r["excluded"]) for r in files.values()]
    summary = dict(
        generated=dt.datetime.now().isoformat(timespec="seconds"), head=head, src_scripts_dirty=dirty,
        generator="scripts/build_lever_ranking.py", central_hash8=chash, central_arm_tag=ctag, central_objective_usd=cnpv,
        input_file_count=len(files), admitted_file_count=len(admitted),
        excluded_by_reason={k: sum(1 for r in files.values() if r["excluded"] and r["excluded"].split(" (")[0] == k)
                            for k in sorted({r["excluded"].split(" (")[0] for r in files.values() if r["excluded"]})},
        excluded_joint_arms=[dict(path=r["path"], hash8=h, differs={f: {"central": a, "file": b} for f, (a, b) in diff.items()}) for h, r, diff in joint],
        undeclared_values=[dict(field=f, value=v, files=n, where=wh) for f, v, n, wh in undecl],
        admitted_not_in_any_family_or_convention=[r["path"] for h, r, diff in unaccounted],
        declared_not_ranked=[dict(id=r["id"], missing=r["missing"]) for r in rows if r["role"] == "declared_not_ranked"],
        live_arm_rule=[dict(id=r["id"], span_all_usd=r["span"], span_live_usd=r["span3"], unasserted=r["unasserted"], candidates=r["cands"])
                       for r in rows if r.get("cands")],
        families_no_build_response=[r["id"] for r in rows if r["role"] == "no_build_response"],
        rule=("Admission: arm_tag carries " + ", ".join(EPOCH_TOKENS) + ". Membership: LEVER_FAMILIES (derived tag, om_case and arm_tag must match). "
              "Span: max minus min of npv_total_cost_usd over every declared member of the family including the central arm. "
              "Build response: decision_variables compared exactly across the family's arms; a family whose arms all share one plan is a "
              "rescaling of a plan determined elsewhere and is listed apart from the ranked table. Column 2 excludes an arm only if its plan "
              "equals the plan of every other arm in the family. Column 3 (live-arm rule, TASK 70) is the span over the family's members "
              "other than those LEVER_FAMILIES declares `unasserted`; the ranked table is ordered by column 3. Bound families and the social_discount_rate files are reported in their own "
              "block and do not enter the ranking."),
        files_read=allfiles)
    json.dump(summary, open(os.path.join(outdir, "lever_ranking_summary.json"), "w"), indent=2, default=str)
    # ---- the printed table -----------------------------------------------------------------------
    L = [f"# Lever ranking -- post-epoch, generated {summary['generated']} at HEAD `{head}`"
         f"{' (src/scripts have uncommitted changes)' if dirty else ''} by `scripts/build_lever_ranking.py`",
         "", f"Central `{chash}` objective {cnpv:,.2f} USD. Files read {len(files)}, admitted {len(admitted)}. A verification artefact, not a finding.", "",
         "## Ranked levers (arms produce different build decisions)", "",
         "Ordered by column 3 (live arms). `rank (col 1)` is the rank by all-arms span.", "",
         "| rank (col 3) | rank (col 1) | family | span col 3 live (bn) | span col 1 all arms (bn) | col 3 low arm | col 3 high arm | unasserted (excluded) | % central (col 3) | arms | in F-5 | plans | col 2 (bn) | col 2 differs | max d solar 2054 MW | " + GASHDR + " | max d storage 2054 MWh | first year |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def fmt(r, rank):
        br = r["br"]
        c2 = "n/a" if r["span2"] is None else f"{r['span2']/1e9:.4f}"
        c2d = "n/a" if r["span2"] is None else ("yes, by %.4f bn" % ((r["span"] - r["span2"]) / 1e9) if r["span2"] != r["span"] else "no")
        return (f"| {rank} | {rank1[r['id']]} | {r['id']} | {r['span3']/1e9:.4f} | {r['span']/1e9:.4f} | {r['lo3']['value']} ({r['lo3']['npv']/1e9:.4f}) | "
                f"{r['hi3']['value']} ({r['hi3']['npv']/1e9:.4f}) | {', '.join(r['unasserted']) or 'none'} | "
                f"{100*r['span3']/cnpv:.3f} | {r['n_present']} | {'yes' if r['in_f5_ranking'] else 'no'} | {len(br['groups'])} | {c2} | {c2d} | "
                f"{br['max_diff'][CUM_2054[0]]:.4f} | {br['max_diff'][CUM_2054[1]]:.4f} | {br['max_diff'][CUM_2054[2]]:.4f} | "
                f"{br['first_diff_year'] if br['first_diff_year'] is not None else 'none'} |")
    for i, r in enumerate(ranked, 1): L.append(fmt(r, i))
    nb = [r for r in rows if r["role"] == "no_build_response"]
    nc = [r for r in rows if r["role"] == "not_comparable"]
    L += ["", "## Families whose arms produce an identical plan (rescaling; not ranked)", "",
          "| family | span (bn) | low arm | high arm | % central | arms | in F-5 | max d solar MW / " + GASHDR[6:] + " / storage MWh | first year |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(nb, key=lambda r: -r["span"]):
        br = r["br"]
        L.append(f"| {r['id']} | {r['span']/1e9:.4f} | {r['lo']['value']} ({r['lo']['npv']/1e9:.4f}) | {r['hi']['value']} ({r['hi']['npv']/1e9:.4f}) | {r['span_pct']:.3f} | "
                 f"{r['n_present']} | {'yes' if r['in_f5_ranking'] else 'no'} | {br['max_diff'][CUM_2054[0]]:.4f} / {br['max_diff'][CUM_2054[1]]:.4f} / {br['max_diff'][CUM_2054[2]]:.4f} | none |")
    L += ["", "## Not comparable: arms differ in dispatch model (not ranked; TASK 83 A.2)", "",
          "| family | span (bn) | low arm | high arm | arms | max d solar MW / " + GASHDR[6:] + " / storage MWh | why it is not ranked |", "|---|---|---|---|---|---|---|"]
    for r in nc:
        br = r["br"]
        L.append(f"| {r['id']} | {r['span']/1e9:.4f} | {r['lo']['value']} ({r['lo']['npv']/1e9:.4f}) | {r['hi']['value']} ({r['hi']['npv']/1e9:.4f}) | {r['n_present']} | "
                 f"{br['max_diff'][CUM_2054[0]]:.4f} / {br['max_diff'][CUM_2054[1]]:.4f} / {br['max_diff'][CUM_2054[2]]:.4f} | {NOT_COMPARABLE[r['id']]} |")
    if not nc: L.append("| none | | | | | | |")
    L += ["", "## Bounds and conventions (do not enter the ranking)", ""]
    for r in rows:
        if r["role"] == "bound":
            L += ["Bound (`bound: True`), not a lever:", "", "| family | span (bn) | low arm | high arm | % central | arms | plans | first year |", "|---|---|---|---|---|---|---|---|",
                  f"| {r['id']} | {r['span']/1e9:.4f} | {r['lo']['value']} ({r['lo']['npv']/1e9:.4f}) | {r['hi']['value']} ({r['hi']['npv']/1e9:.4f}) | {r['span_pct']:.3f} | {r['n_present']} | {len(r['br']['groups'])} | {r['br']['first_diff_year'] if r['br']['first_diff_year'] is not None else 'none'} |", ""]
    if sdr_block:
        L += ["Reporting convention (`social_discount_rate`), not a property of the system:", "", "| rate | file | objective (USD) |", "|---|---|---|"]
        for s in sdr: L.append(f"| {s['rate']} | `{s['path']}` | {s['npv']:,.2f} |")
        L += ["", f"Span {sdr_block['span']/1e9:.4f} bn ({sdr_block['span_pct']:.3f}% of central); low rate {sdr_block['lo']['rate']}, high rate {sdr_block['hi']['rate']}; "
              f"{len(sdr_block['br']['groups'])} distinct plan(s); first year additions differ: "
              f"{sdr_block['br']['first_diff_year'] if sdr_block['br']['first_diff_year'] is not None else 'none'}.", ""]
    L += ["## Declared but not ranked (missing member)", ""] + ([f"* `{r['id']}`: {r['missing']}" for r in rows if r['role'] == 'declared_not_ranked'] or ["none"])
    L += ["", "## Joint arms (differ from central in two or more fields; excluded)", ""]
    L += [f"* `{r['path']}`: " + "; ".join(f"{f}: {b} (file) vs {a} (central)" for f, (a, b) in diff.items()) for h, r, diff in joint] or ["none"]
    L += ["", "## Values the files carry that no declared family covers (reported, not adopted)", "", "| field | value | files | where |", "|---|---|---|---|"]
    L += [f"| {f} | {v} | {n} | {wh} |" for f, v, n, wh in undecl]
    L += ["", "## Live-arm rule (column 3): every arm the rule could have touched", "",
          "An arm is UNASSERTED when it is unsourced, represents no state or bound the thesis asserts, and is retained only so that "
          "superseded results remain attributable. Only UNASSERTED arms are excluded from column 3; the span with each other candidate "
          "excluded is shown for information and is not applied.", "",
          "| family | arm | status | span all arms (bn) | span with this arm excluded (bn) | reason (declared in LEVER_FAMILIES) |", "|---|---|---|---|---|---|"]
    for r in rows:
        for c in r.get("cands", []):
            sw = "undefined (fewer than two arms left)" if c["span_without"] is None else f"{c['span_without']/1e9:.4f}"
            L.append(f"| {r['id']} | `{c['arm']}` | {c['status']} | {r['span']/1e9:.4f} | {sw} | {c['reason']} |")
    L += ["", "## Column-2 exclusions", ""] + [f"* `{r['id']}`: excluded {', '.join(r['excluded2'])}" for r in rows if r.get("excluded2")] or ["none"]
    if CAPPED: L += capped_extras(outdir, fam_out, admitted, rows, ranked)
    open(os.path.join(outdir, "lever_ranking.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(f"\nWrote {outdir}/")


if __name__ == "__main__":
    main()
