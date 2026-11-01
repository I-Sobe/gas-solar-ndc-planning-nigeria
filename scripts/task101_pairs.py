"""TASK 101 Part 1/2: curated overlap pairs.  Ranges were chosen by reading both
passages; this script only measures them (words, shared 8-word and 12-word
runs, shared numeric tokens).  Read-only.  Run with CH2=<extracted ch2 text>.
Usage: python scripts/task101_pairs.py [md|csv]
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task101_overlap as T

F = {k: T.FILES[k].read_text(encoding="utf-8").split("\n") for k in T.FILES}

# (id, ch3 section, ch3 ranges, other key, other location, other ranges, class, note)
P = [
 ("P01", "3.1.1", [(26, 35)], "ch4", "4.1.3", [(81, 86)], "S", "reporting window 2025-2054 solved, 2045 reported, final nine years a buffer; Ch1 1.6.3 [409-414] states it a third time"),
 ("P02", "3.1.1", [(37, 55)], "ch4", "4.1.3", [(88, 103)], "R", "stock invariance to the horizon end and the annual-series bound; same measurement at different rounding (2e-4 MW against 0.1 MW; 47.7/25.5 against 48/26 MW; 1.48-1.88% against 1.9%); no 12-word run"),
 ("P03", "3.1.1", [(57, 63)], "ch4", "4.1.5", [(153, 158)], "M", "the two correction epochs told as history: timing convention and storage capital series"),
 ("P04", "3.4.2", [(1815, 1827)], "ch4", "4.1.3", [(83, 86)], "S", "truncation at 2045 would need a terminal salvage treatment the per-vintage formulation avoids; Ch1 1.6.3 [414-417] states the same sentence"),
 ("P05", "3.4.2", [(1782, 1805)], "ch4", "4.1.5", [(153, 158)], "M", "the capital-charge timing correction (blended, VERIFY-marked) against its one-sentence summary in Ch4"),
 ("P06", "3.4.2", [(1829, 1836)], "ch4", "4.1.2", [(63, 70)], "R", "capital charge and its four lines: five dollar values identical to the digit; Ch1 1.3 [170-176] and Ch4 4.3.6 [497-505] repeat the percentages"),
 ("P07", "3.4.2", [(1838, 1849)], "ch1", "1.5.3", [(373, 376)], "R", "Shapley split 50.1% to tenor; Ch1 reports it as closed-form arithmetic with the same caveat"),
 ("P08", "3.4.1", [(1665, 1677)], "ch1", "1.6.1", [(384, 393)], "S", "accounting perspective of the objective; 14-, 18- and 11-word runs; Ch3 3.1.6 also carries an 8-word run of it"),
 ("P09", "3.4.2", [(1766, 1770)], "ch1", "1.5.1", [(316, 322)], "M", "both capital recovery factors precomputed, financing a volume split at fixed prices, programme linear"),
 ("P10", "3.4.2", [(1686, 1693)], "ch1", "1.5.2", [(326, 332)], "S", "novelty claim and its scope"),
 ("P11", "3.4.3", [(1878, 1887)], "ch1", "1.5.2", [(334, 343)], "S", "concessional eligibility restricted to solar and storage as a declared assumption; Ch1 1.4 Q2 [257-259], Ch5 5.2.2 [124-132] and Ch4 4.3.6 [509-515] repeat it"),
 ("P12", "3.4.4", [(1931, 1944)], "ch4", "4.1.4", [(136, 143)], "R", "B* against the measured requirement: 5,518,409,480 and 86.2% identical"),
 ("P13", "3.4.4", [(1946, 1955)], "ch4", "4.1.4", [(140, 143)], "R", "tier multiples 0.931, 1.582, 1.862, 2.234 identical; Ch1 1.6.5 [436-441] restates the tiers with the superseded values"),
 ("P14", "3.4.6", [(2080, 2094)], "ch1", "1.5.2", [(343, 348)], "S", "the channel is generic private capital mobilisation; 11-word run; Ch5 5.3.2 [265-272] repeats it (11-word run)"),
 ("P15", "3.5.2", [(2301, 2306)], "ch1", "1.2.1", [(79, 84)], "S", "no power-sector share in the NDC, allocation by share of sector emissions (11-word run)"),
 ("P16", "3.3.1", [(1219, 1227)], "ch1", "1.2.1", [(84, 89)], "S", "the 17 GW NDC gas capacity figure; Ch1 adds the derating arithmetic"),
 ("P17", "3.5.3", [(2371, 2389)], "ch1", "1.2.1", [(91, 98)], "S", "the cap binds the grid-supplied quantity and leaves self-generation outside; Ch5 5.1 [21-24] repeats it (13-word run)"),
 ("P18", "3.5.4", [(2420, 2435)], "ch4", "4.5.3", [(664, 681)], "R", "annual against horizon form of the reliability standard; Ch3's magnitudes (38%, 49-fold, three years, 20.2%) are the ones Ch4 withdraws"),
 ("P19", "3.5.5", [(2453, 2465)], "ch4", "4.4.4", [(601, 610)], "R", "the carbon-price sweep reported twice: Ch3 self-generation falls 2.3% (blended); Ch4 (vintage) the tight arm builds no self-generation"),
 ("P20", "3.5.5", [(2439, 2465)], "ch1", "1.6.7", [(461, 463)], "S", "the swept range does not reach a displacing carbon price; Ch5 5.7 [727-729] states it again"),
 ("P21", "3.5.6", [(2482, 2499)], "ch4", "4.4.1", [(523, 534)], "M", "the out-of-boundary share is a level, leakage is the increment over the uncapped solve; 8-word run"),
 ("P22", "3.5.6", [(2516, 2550)], "ch4", "4.4.2", [(544, 577)], "R", "the out-of-boundary surface across capital tier and collection rate; different generation and basis, see Part 2 Q1"),
 ("P23", "3.5.6", [(2552, 2559)], "ch4", "4.4.3", [(581, 597)], "X", "Ch3 describes what Ch4 reports (two instruments); no shared words"),
 ("P24", "3.6.1", [(2578, 2585)], "ch1", "1.6.4", [(421, 432)], "S", "every solve is a deterministic single-objective LP; 12-word run; same paragraph also overlaps 3.6.11 and 3.7.2"),
 ("P25", "3.6.11", [(3036, 3045)], "ch1", "1.6.4", [(421, 432)], "S", "the summary of method, near-verbatim (24-, 15-, 14- and 11-word runs)"),
 ("P26", "3.7.2", [(3075, 3081)], "ch1", "1.6.4", [(426, 432)], "S", "the bound contains both the VSS and the EVPI, first stage solar and storage only (14- and 9-word runs); Ch4 4.5.4 [691-696] and Ch5 5.5.3 [580-585] repeat the sentence (13-word runs)"),
 ("P27", "3.6.4", [(2657, 2665)], "ch4", "4.1.2", [(53, 61)], "R", "central anchor 23.323 bn, 2045 storage 24,697.69, terminal 56,289.88 (Ch4 4.1.3 [88-91] gives 24,698 and 56,290)"),
 ("P28", "3.6.4", [(2681, 2694)], "ch4", "4.1.5", [(153, 166)], "M", "the correction epochs as they moved the anchors; Ch3's percentages are blended"),
 ("P29", "3.6.7", [(2776, 2785)], "ch4", "4.2.7", [(368, 375)], "R", "Sobol first-order index of gas availability: Ch3 87.3 and 90.3 per cent, Ch4 0.9466 and 0.9545: same quantity, different generation, they disagree"),
 ("P30", "3.6.8", [(2864, 2868)], "ch4", "4.5.4", [(685, 702)], "R", "ECP 36.343, WS 24.253, bound 12.089 bn, 33.3%, median 25.71: identical; Ch5 5.5.3 [578-604] repeats the bound without figures"),
 ("P31", "3.6.8", [(2840, 2846)], "ch4", "4.5.4", [(706, 707)], "M", "both legs solved with the public capital budget unconstrained (8-word run); Ch4's two lines are a repeat of its own 702-704"),
 ("P32", "3.6.10", [(2929, 2948)], "ch1", "1.6.8", [(472, 489)], "R", "suite counts: Ch3 87 / 70-14 / 80-6-1, Ch1 88 / 81-7-0; they disagree"),
 ("P33", "3.7.2", [(3089, 3092)], "ch1", "1.6.8", [(491, 494)], "S", "single solver at one tolerance, flat optimal face solver-dependent (14-word run); Ch4 4.1.1 [44-49] and Ch5 5.8.1 [809-813] repeat it"),
 ("P34", "3.7.4", [(3250, 3269)], "ch5", "5.2.3", [(180, 186)], "L", "the gas capacity credit equals the availability factor (14-word run)"),
 ("P35", "3.7.4", [(3278, 3285)], "ch5", "5.5.4", [(613, 619)], "L", "reserve margin an operating quantity used as a planning standard, 19.5 to 6.4 per cent (14-, 9- and 9-word runs)"),
 ("P36", "3.7.4", [(3312, 3321)], "ch5", "5.8.3", [(935, 939)], "L", "three of four weakly identified inputs make adequacy easier; Ch1 1.4 [283-286] has a 9-word run"),
 ("P37", "3.7.5", [(3325, 3355)], "ch5", "5.1", [(15, 21)], "L", "feasibility and frontier results are invariant to the availability arm, dual-based results are not (13- and 15-word runs); Ch4 4.1.4 [111-115] repeats it (13- and 14-word runs)"),
 ("P38", "3.7.5", [(3325, 3335)], "ch5", "5.8.3", [(889, 895)], "L", "no sourced central value; declared lower bound (15-word run)"),
 ("P39", "3.7.5", [(3357, 3364)], "ch5", "5.8.3", [(913, 916)], "L", "value of lost load: 22-word run, 97% of the Ch5 passage inside shared runs"),
 ("P40", "3.7.5", [(3366, 3371)], "ch5", "5.8.3", [(918, 920)], "L", "capital cost bases are not Nigerian (18-word run)"),
 ("P41", "3.7.5", [(3373, 3381)], "ch5", "5.8.3", [(922, 927)], "L", "demand dispersion unsourced and unreconciled (26-word run); Ch5 5.5.3 [596-604] repeats the tail-amplification clause (13-word run)"),
 ("P42", "3.7.5", [(3383, 3397)], "ch5", "5.8.3", [(929, 933)], "L", "self-generation arms are market segments; Ch1 1.1.4 [58-61] has a 22-word run of the same clause"),
 ("P43", "3.7.6", [(3401, 3407)], "ch5", "5.1", [(21, 24)], "L", "every cap-compliance statement is a statement about the grid boundary (13-word run)"),
 ("P44", "3.7.6", [(3409, 3418)], "ch5", "5.4.4", [(477, 485)], "L", "self-generation unbounded so the frontier need have no infeasible end (23-word run)"),
 ("P45", "3.7.6", [(3409, 3418)], "ch4", "4.4.5", [(623, 630)], "R", "frontier non-dominated counts 12 of 32 and 7 of 32, turning points 0.625 and 0.8125: identical"),
 ("P46", "3.7.7", [(3432, 3481)], "ch5", "5.8.4", [(943, 950)], "L", "boundary-resolved results are lower bounds (12-word run); Ch4 4.1.4 [117-123] repeats 8 of 8 and 12 of 20"),
 ("P47", "3.7.7", [(3453, 3455)], "ch4", "4.4.5", [(632, 635)], "R", "item 4: the tight end of the cost-emissions frontier is unlocated in all four arms, every cap down to 0.01 feasible"),
 ("P48", "3.3.3", [(1340, 1340)], "ch4", "4.2.3", [(216, 231)], "R", "Ch3 calls conversion the second-largest cost lever; Ch4's ranking puts it fourth (Ch5 5.2.2 [92-97] says so too): a contradiction, not a duplicate"),
 ("P49", "3.3.7", [(1549, 1554)], "ch1", "1.1.4", [(51, 57)], "S", "self-generation priced and unbounded, infeasibility not cost the signal of an inadequate plan (14-word run)"),
 ("P50", "3.3.3", [(1347, 1356)], "ch1", "1.1.3", [(39, 47)], "M", "convertible plants banded in tiers by mean availability across the 2024, 2025/Q4, 2026/Q1 tables (10- and 11-word runs)"),
 ("P51", "3.3.2", [(1249, 1262)], "ch1", "1.1.2", [(21, 35)], "S", "the 89.27 TWh_th anchor rests on one measurement; the implied volume is the same figure restated"),
 ("P52", "3.2.4", [(943, 951)], "ch1", "1.1.1", [(7, 17)], "S", "grid generation only; backup generation 40 TWh / 18 TWh, same IEA citation and [PAGE] marker (12-word run); Ch1 1.6.2 [397-405] restates the demand boundary"),
 ("P53", "3.7.3", [(3176, 3187)], "ch5", "5.8.2", [(860, 866)], "L", "system boundary and busbar factor (19-word run)"),
 ("P54", "3.7.3", [(3189, 3197)], "ch4", "4.2.5", [(291, 295)], "R", "hydro carries no cost term so its span is a bound, not a lever; Ch5 5.8.2 [873-879] is the L pair (8-word run)"),
 ("P55", "3.7.3", [(3207, 3215)], "ch5", "5.4.2", [(419, 425)], "L", "policy exogenous, comparative statics across cap paths (17- and 23-word runs); Ch5 5.8.2 [868-871] repeats it"),
 ("P56", "3.2.2", [(738, 740)], "ch1", "1.1.1", [(7, 9)], "S", "base-year grid generation 37,093.70 GWh in 2024 (NERC); the same figure and citation marker in both"),
]


def words_in(lines, ranges):
    return sum(len(lines[i - 1].split()) for a, b in ranges if a for i in range(a, b + 1))


def text_in(lines, ranges):
    return " ".join(lines[i - 1] for a, b in ranges if a for i in range(a, b + 1))


def stats(a, b):
    ta, tb = T.norm_tokens(a), T.norm_tokens(b)
    r8 = T.shared_runs(ta, tb, 8)
    r12 = T.shared_runs(ta, tb, 12)
    cov = set()
    for i, j, L in r8:
        cov.update(range(j, j + L))
    num = re.compile(r"(?<![\w.])\d[\d,]*\.\d+|(?<![\w.])\d{1,3}(?:,\d{3})+|(?<![\w.])\d{4,}")
    na = {x for x in num.findall(re.sub(r"\[VERIFY[^\]]*\]", " ", a))}
    nb = {x for x in num.findall(re.sub(r"\[VERIFY[^\]]*\]", " ", b))}
    shared = sorted(x for x in na & nb if not re.fullmatch(r"(19|20)\d\d", x) and not re.fullmatch(r"\d\.\d", x))
    return len(r12), max([r[2] for r in r8], default=0), len(cov), shared


def rows():
    out = []
    for pid, sec, r3, ok, osec, ro, cls, note in P:
        l3 = F["ch3"]
        lo = F[ok]
        w3 = words_in(l3, r3)
        wo = words_in(lo, ro)
        n12, mx, cov, shared = stats(text_in(l3, r3), text_in(lo, ro))
        out.append(dict(id=pid, sec=sec, r3="; ".join(f"{a}-{b}" if a != b else str(a) for a, b in r3), w3=w3,
                        other=f"{ok} §{osec}", ro="; ".join(f"{a}-{b}" for a, b in ro), wo=wo, cls=cls,
                        n12=n12, max8=mx, cov8=cov, shared=" ".join(shared), note=note))
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "md"
    R = rows()
    if mode == "csv":
        w = csv.DictWriter(sys.stdout, fieldnames=list(R[0]))
        w.writeheader()
        w.writerows(R)
    else:
        print("| id | Ch3 § | Ch3 lines | Ch3 words | other chapter | other lines | other words | kind | 12-runs | longest run | words in ≥8-runs | shared numbers | what overlaps |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in R:
            print(f"| {r['id']} | {r['sec']} | {r['r3']} | {r['w3']} | {r['other']} | {r['ro']} | {r['wo']} | {r['cls']} | {r['n12']} | {r['max8']} | {r['cov8']} | {r['shared']} | {r['note']} |")
