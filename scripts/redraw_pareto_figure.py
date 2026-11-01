"""Redraw results/pareto_frontier/pareto_frontier.{png,pdf} (two panels, F-43)
from a stored pareto_raw.csv. No solve. TASK 30, Part B.

    python scripts/redraw_pareto_figure.py [pareto_raw.csv] [out_dir]
"""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pareto_plot import flag_dominance, draw_two_panel, dominated_points_table, DOMINATED_NOTE

if len(sys.argv) > 1:
    src = Path(sys.argv[1])
else:
    # TASK 53 2.5: 18_ now writes to results/pareto_frontier/arms_<hash8>/. Accept the default only if exactly one exists.
    _c = sorted((ROOT / "results/pareto_frontier").glob("arms_*/pareto_raw.csv"))
    if len(_c) != 1:
        raise SystemExit(f"pass the pareto_raw.csv path explicitly: {len(_c)} arm-keyed candidates under results/pareto_frontier: {[str(c) for c in _c]}")
    src = _c[0]
out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent
f = flag_dominance(pd.read_csv(src))
for arm, g in f.groupby("arm"):
    print(f"{arm}: {int((~g.dominated_total).sum())} of {len(g)} non-dominated on total; "
          f"{int((~g.dominated_grid_only).sum())} of {len(g)} on grid-only")
dominated_points_table(f).to_csv(out / "pareto_dominated_points.csv", index=False)
draw_two_panel(f, out / "pareto_frontier.png", out / "pareto_frontier.pdf")
print(DOMINATED_NOTE)
print("wrote", out)
