"""TASK 58 §2: curated relocation inventory. Ranges were chosen by reading the
chapter; this script only measures them. Read-only."""
import re
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from task58_measure import load, classify, words, CH, EQ_IN_PARA, CITE_RE, MKR

# id, section, [(first_line, last_line), ...], quantity, note
A = [
 ("A1", "3.1.1", [(48, 53)], "terminal-year (end-effect) dependence of the plan", ""),
 ("A2", "3.1.5", [(301, 302)], "gas retirement schedule (flat proxy)", "one sentence; sits under a [code:] anchor"),
 ("A3", "3.1.8", [(682, 685)], "which two Chapter 4 results still carry annual-builder figures", "status of Chapter 4 results, not a Chapter 3 parameter"),
 ("A4", "3.2.2", [(729, 742)], "base-year demand level (23.08 vs 37.09 TWh)", "includes the loss-ladder table"),
 ("A5", "3.2.3", [(787, 792)], "2014-2024 generation growth (2.20% vs 3.14%/1.53%)", "earlier finding compared against the wrong file"),
 ("A6", "3.2.3", [(839, 849)], "income elasticity range (0.76-0.90 vs 0.7-0.9)", ""),
 ("A7", "3.2.3", [(916, 921)], "organic_central growth rate (4.0% to 3.9%)", ""),
 ("A8", "3.2.5", [(974, 978)], "peak-to-average ratio: specification inverted (set to derived)", ""),
 ("A9", "3.2.5", [(1002, 1006)], "within-block factor omega (block ratio 1.0847 vs 1.0877)", "same quantity as A38"),
 ("A10", "3.3.1", [(1174, 1177)], "gas retirement (flat 680 MW/yr proxy replaced by vintage schedule)", "last line begins the current specification"),
 ("A11", "3.3.1", [(1199, 1202)], "gas retirement, direction of bias against the proxy (359/277/152 vs 680 MW/yr)", "values-versus-values"),
 ("A12", "3.3.1", [(1204, 1211)], "new gas build (fixed at zero, now a decision variable)", "mixed: also carries the NDC 17 GW justification"),
 ("A13", "3.3.2", [(1244, 1246)], "gas deliverability anchor A_2025 (40.66 vs 89.27 TWh_th; eta 0.43 vs 0.287)", ""),
 ("A14", "3.3.5", [(1424, 1425), (1430, 1432)], "hydro seasonal split theta (sign of dry-wet difference)", "sourced values between the two ranges stay"),
 ("A15", "3.3.6", [(1479, 1482)], "storage capital cost series (flat 2025 value vs declining series)", "L90; story repeated at A37"),
 ("A16", "3.3.6", [(1490, 1493)], "storage capacity credit (1.0 vs swept 0.50/0.80/1.00)", ""),
 ("A17", "3.3.6", [(1514, 1518)], "storage at 2054 under the regression configuration (43,540 vs 74,435 MWh)", "also carries the argument 'where firm capacity binds storage is set by the constraint'"),
 ("A18", "3.4.2", [(1693, 1701)], "concessional rate and tenor (IDA20 2.80%/30y vs IDA21 2.12%/25y)", "contains the CRF arithmetic"),
 ("A19", "3.4.2", [(1753, 1759)], "capital-charge timing convention", "L88"),
 ("A20", "3.4.2", [(1761, 1774)], "objective under the timing correction (25.86 to 23.32 bn; 3.39%)", "the measurement that verifies the correction"),
 ("A21", "3.4.2", [(1776, 1783)], "timing correction: annuity-due alternative, rejected", ""),
 ("A22", "3.4.3", [(1860, 1863)], "concessional envelope parameter (concessional_envelope_npv, undiscounted)", ""),
 ("A23", "3.4.4", [(1915, 1920)], "B* (10.27 bn measured on an earlier state vs 5.52 bn)", "LIVE caveat that conditions every result in B* units"),
 ("A24", "3.4.5", [(1982, 2003)], "coverage margin m forced to 1.10 in six runners", "commit 4ec43b4; L34, L43"),
 ("A25", "3.4.5", [(2016, 2019)], "DFI cover-ratio claim (withdrawn)", ""),
 ("A26", "3.4.5", [(2031, 2035)], "capital-requirement comparison figures (previously carried here)", "a ⟦ ⟧ note-to-self, to be resolved in Chapter 4 not moved"),
 ("A27", "3.4.7", [(2075, 2095)], "bankability commercial basis (13.24% WACC to 11%/8.824%); concessional rate unification", "the two 'resolved' items; the two 'open' items (2097-2116) are live"),
 ("A28", "3.5.1", [(2138, 2142)], "self-generation CO2 treatment (single switch vs three quantities)", ""),
 ("A29", "3.5.2", [(2234, 2249)], "power share of energy (7.24% vs 8.3963%)", "L60 is cited here as Appendix C"),
 ("A30", "3.5.2", [(2314, 2324)], "buffer-year caps (effectively infinite vs held flat)", "mixed: also the reason for the present design"),
 ("A31", "3.5.4", [(2375, 2394)], "grid-service obligation (previously a reliability constraint on u alone)", "mixed: rationale for the present form and name"),
 ("A32", "3.6.4", [(2655, 2668)], "anchor values before and after the correction epoch", ""),
 ("A33", "3.6.6", [(2712, 2717)], "weighted-sum scalarisation (existed, deleted)", ""),
 ("A34", "3.6.8", [(2816, 2819)], "moderate tier's binding status on an earlier state", "mixed: the configuration caveat is live"),
 ("A35", "3.6.10", [(2890, 2898)], "regression signatures (single signature, second added)", "L104 cited at 2901"),
 ("A36", "3.6.10", [(2915, 2922)], "invariant suite counts (70/14 first run; 80/6/1 after)", "counts currently stale against the 88-assertion suite"),
 ("A37", "3.6.10", [(2940, 2950)], "storage capital series reverted silently", "same story as A15"),
 ("A38", "3.1.7", [(570, 573)], "omega (1.3448 default vs 1.3411 derived)", "LIVE declared limitation; value-versus-value"),
]
B = [
 ("B1", "3.6.2", [(2574, 2575)], "arm values: same quantity different values in different runners", "process anecdote"),
 ("B2", "3.6.3", [(2590, 2592)], "tariff value written by a silent default", "process anecdote"),
 ("B3", "3.6.6", [(2724, 2731)], "frontier endpoints (grids extended)", ""),
 ("B4", "3.6.9", [(2856, 2862)], "threshold detection (conclusion changed on grid extension: 2 of 20 to 5 of 20)", ""),
 ("B5", "3.6.10", [(2952, 2960)], "claim check (eleven failed)", ""),
 ("B6", "3.6.10", [(2962, 2973)], "result provenance (fixed filenames, now hash-keyed)", ""),
]


def measure(lines, cls, ranges):
    p = t = e = 0
    txt = []
    for a, b in ranges:
        for i in range(a - 1, b):
            k = cls[i][0]
            w = words(lines[i])
            if k == "table":
                t += w
            elif k == "eq":
                e += w
            elif k in ("prose", "code"):
                p += w
            txt.append(lines[i])
    return p, t, e, " ".join(txt)


def main():
    L = load(CH)
    cls = classify(L)
    for name, rows in (("TIER A (marker or value-pair hit, read and judged superseded-state narration)", A),
                       ("TIER B (same genre, no marker hit)", B)):
        print(f"\n### {name}\n")
        print("| id | section | lines | prose w | table w | quantity | math/eq | cite | reg | ⟦⟧ | [VERIFY…] | note |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        tp = tt = 0
        for id_, sec, rng, q, note in rows:
            p, t, e, txt = measure(L, cls, rng)
            tp += p
            tt += t
            r = "; ".join(f"{a}-{b}" for a, b in rng)
            eq = "Y" if (EQ_IN_PARA.search(txt) or e) else ""
            ci = "Y" if CITE_RE.search(txt) else ""
            rg = "Y" if re.search(r"Appendix [A-D]|\bL\d{1,3}\b|commit `", txt) else ""
            va = "Y" if "⟦" in txt else ""
            mk = "Y" if MKR.search(txt) else ""
            print(f"| {id_} | {sec} | {r} | {p} | {t} | {q} | {eq} | {ci} | {rg} | {va} | {mk} | {note} |")
        print(f"\nTotal: prose {tp}, table {tt}, sum {tp+tt}")
        free = 0
        for id_, sec, rng, q, note in rows:
            p_, t_, e_, txt = measure(L, cls, rng)
            flags = (EQ_IN_PARA.search(txt) or e_ or CITE_RE.search(txt) or "⟦" in txt or MKR.search(txt)
                     or re.search(r"Appendix [A-D]|\bL\d{1,3}\b|commit `", txt))
            if not flags:
                free += p_ + t_
        print(f"Words in rows carrying none of math/eq, cite, reg, ⟦⟧, marker: {free}")


if __name__ == "__main__":
    main()
