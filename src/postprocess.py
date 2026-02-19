def export_benchmark_comparison(gas_shadow_price_by_year, netbacks_usd_per_twh_th):
    """
    gas_shadow_price_by_year: {year: dual or None}
    netbacks_usd_per_twh_th: dict like {"low":  X, "mid": Y, "high": Z}
    """
    years = [y for y, v in gas_shadow_price_by_year.items() if v is not None]
    lam = [gas_shadow_price_by_year[y] for y in years]

    out = {}
    for name, p in netbacks_usd_per_twh_th.items():
        gaps = [v - p for v in lam]
        out[name] = {
            "share_years_power_gt_export": sum(v > p for v in lam) / len(lam) if lam else None,
            "min_gap_usd_per_twh_th": min(gaps) if gaps else None,
            "max_gap_usd_per_twh_th": max(gaps) if gaps else None,
            "max_shadow_price_usd_per_twh_th": max(lam) if lam else None,
        }
    return out
