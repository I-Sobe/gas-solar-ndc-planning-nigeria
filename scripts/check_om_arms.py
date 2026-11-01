"""TASK 48 B.2 success check: the three O&M arms sit in three distinct baseline_sliced files."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.output_naming import hash8_of, om_tag_suffix

D = Path("results/baseline_sliced")
IDX = next(l for l in (D / "INDEX.md").read_text().splitlines() if l.startswith("- `02c17f55`"))
BASE = IDX.split("-> `", 1)[1].rstrip("`")          # central tag, no O&M field
ok = True
arms = ("om_central", "om_low", "om_high") + (("om_none",) if "--with-none" in sys.argv else ())
for om in arms:
    h = hash8_of(BASE + om_tag_suffix(om))
    f = D / f"summary__{h}.json"
    if not f.exists():
        print(f"FAIL {om}: {f} missing"); ok = False; continue
    d = json.loads(f.read_text())
    good = d.get("om_case") == om
    ok &= good
    print(f"{'ok  ' if good else 'FAIL'} {om:11s} {h}  om_case={d.get('om_case')}  npv={d['npv_total_cost_usd']/1e9:.6f} bn")
sys.exit(0 if ok else 1)
