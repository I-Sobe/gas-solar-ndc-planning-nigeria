"""TASK 84 F.4: lines of Chapter 1 and Chapter 3 that state a figure this task superseded. Token list = the blended-generation (epoch 1) figures struck in docs/task_log_84_entries.json plus the frozen-state figures the first epoch struck. Text search only; a hit is a candidate, not a verdict."""
import re, json, sys
TOK = {  # token -> entry it belongs to
 "24.987":"central blended","24,986,620":"central blended","24.9866":"central blended","27.318":"central frozen","27.954":"central frozen","27,317,670":"central frozen",
 "7.059":"F-5","8.503":"F-5 frozen","5.220":"F-6","5.845":"F-6 frozen","5.91":"F-6 frozen","3.130":"F-1","3.811":"F-1 frozen","3.97 bn":"F-1 frozen","2.147":"F-2","2.300":"F-2",
 "12.53":"F-1","36.0%":"F-1","8.99":"F-1 capital mobilised","8.9917":"F-1","12.772":"F-1 frozen","43.8":"decomposition B.3","30.1%":"decomposition B.3","26.1%":"decomposition B.3","10,944,915":"decomposition B.3",
 "second-largest":"F-6","pays twice":"F-6","37.9":"F-8","43.3%":"F-8","42.4":"F-8","36.5":"F-8","26.4":"F-8","28.9%":"F-8",
 "0.8125":"F-43","0.8":"","7.24":"F-42","18.2%":"F-42","19.1%":"F-42","7.33":"F-42","9.7 to 22.4":"F-10","19.1 ":"F-10","4.7":"F-10",
 "RAISES national":"F-21","raises national":"F-21","fifth of emissions":"F-31","17.6%":"F-31","542.5":"F-39","543.0":"F-39","no abatement":"F-40","does no abatement":"F-40",
 "nine-tenths":"M-4","87.3%":"M-4","0.8729":"M-4","50 of 100":"F-32","3.20":"F-36","57.81":"F-36","6.284":"F-12","48,582":"F-12","41,191":"F-12","60.0%":"F-22","57.1":"F-22","64.7":"F-22","15.09":"F-22","4.98%":"F-22",
 "monotone":"F-20","inversion":"F-20","0.0526":"F-33","0.049":"F-33","7.411":"",
}
TOK = {k: v for k, v in TOK.items() if v}
for path in sys.argv[1:]:
    for n, line in enumerate(open(path, errors="replace"), 1):
        hits = sorted({v for k, v in TOK.items() if k in line})
        if hits: print(f"{path}:{n}\t{'; '.join(hits)}\t{line.strip()[:170]}")
