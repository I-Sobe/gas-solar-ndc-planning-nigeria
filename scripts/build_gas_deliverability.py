import os
import yaml
import csv

def build_series(years, anchor_year, anchor_value, case_cfg):
    typ = case_cfg["type"]

    if typ == "exponential_decline":
        d = float(case_cfg["decline_rate"])
        out = []
        for y in years:
            out.append(anchor_value * ((1 - d) ** (y - anchor_year)))
        return out

    if typ == "drift":
        g = float(case_cfg["annual_drift"])
        out = []
        for y in years:
            out.append(anchor_value * ((1 + g) ** (y - anchor_year)))
        return out

    if typ == "lag_ramp_uplift":
        L = int(case_cfg["lag_years"])
        R = int(case_cfg["ramp_years"])
        u = float(case_cfg["uplift_fraction"])
        out = []
        for y in years:
            if y <= anchor_year + L:
                f = 1.0
            elif y < anchor_year + L + R:
                f = 1.0 + u * (y - (anchor_year + L)) / R
            else:
                f = 1.0 + u
            out.append(anchor_value * f)
        return out
    
    if typ == "lag_then_drift":
        L = int(case_cfg["lag_years"])
        g = float(case_cfg["annual_drift"])
        out = []
        for y in years:
            if y <= anchor_year + L:
                out.append(anchor_value)
            else:
                growth_years = y - (anchor_year + L)
                out.append(anchor_value * ((1 + g) ** growth_years))
        return out

    if typ == "lag_then_decline":
        L = int(case_cfg["lag_years"])
        d = float(case_cfg["decline_rate"])
        out = []
        for y in years:
            if y <= anchor_year + L:
                out.append(anchor_value)
            else:
                decline_years = y - (anchor_year + L)
                out.append(anchor_value * ((1 - d) ** decline_years))
        return out

    if typ == "shock_recovery":
        baseline_end = int(case_cfg["baseline_end_year"])
        shock_start = int(case_cfg["shock_start_year"])
        shock_end = int(case_cfg["shock_end_year"])
        d = float(case_cfg["shock_decline_rate"])
        rec_start = int(case_cfg["recovery_start_year"])
        R = int(case_cfg["recovery_ramp_years"])
        u = float(case_cfg["recovery_uplift_fraction"])

        out = []
        # level at end of baseline (flat)
        level_baseline_end = anchor_value  # since baseline is flat at anchor

        # level at end of shock (decline applied year-to-year starting shock_start)
        n_shock_years = shock_end - shock_start + 1
        level_shock_end = level_baseline_end * ((1 - d) ** n_shock_years)

        for y in years:
            if y <= baseline_end:
                v = anchor_value

            elif shock_start <= y <= shock_end:
                # decline from baseline_end level
                n = y - shock_start + 1
                v = level_baseline_end * ((1 - d) ** n)

            elif y >= rec_start:
                # ramp from shock_end level up to (1+u)*anchor_value
                target = anchor_value * (1.0 + u)
                if y < rec_start + R:
                    frac = (y - rec_start + 1) / R
                    v = level_shock_end + frac * (target - level_shock_end)
                else:
                    v = target
            else:
                # gap years shouldn't exist if config is consistent; hold shock_end level
                v = level_shock_end

            out.append(v)

        return out


    if typ == "flat_level":
        # Level-equivalent flat scenario.
        # Uses a constant annual value equal to the mean of a reference scenario,
        # so cumulative gas supply matches the reference while removing shape variation.
        # Used by GAS-3 to isolate the shape effect independently of gas level.
        level = float(case_cfg["level_twh_th"])
        return [level for _ in years]

    raise ValueError(f"Unknown case type: {typ}")

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(repo_root, "data", "gas", "processed", "gas_deliverability_cases.yaml")
    out_csv = os.path.join(repo_root, "data", "gas", "processed", "gas_available_power_annual_twh_th.csv")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    anchor_year = int(cfg["anchor"]["year"])
    anchor_value = float(cfg["anchor"]["gas_available_twh_th"])
    cases = cfg["cases"]

    # Planning horizon (keep consistent with your model)
    start_year, end_year = 2025, 2045
    years = list(range(start_year, end_year + 1))

    rows = []
    for case_name, case_cfg in cases.items():
        series = build_series(years, anchor_year, anchor_value, case_cfg)
        for y, v in zip(years, series):
            rows.append({
                "year": y,
                "scenario": case_name,
                "gas_available_twh_th": round(v, 6),  # keep precision in file
            })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["year", "scenario", "gas_available_twh_th"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {out_csv} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
