"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Extension 1: Enhanced recommend_tier() considers GPU interruption rate and 1yr vs 3yr reserved.

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        total_days = int(num(j.get("days", DAYS)))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        # ENHANCED: pass gpu_type + total_days for interruption-rate-aware tier selection
        interrupt_rate = pricing.GPU_INTERRUPTION_RATE.get(gtype, pricing.DEFAULT_INTERRUPTION_RATE)
        tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, total_days=total_days)

        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(
                gpu_hours, num(c["spot_hr"]), od,
                interrupt_rate=interrupt_rate,
            )
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            # Use 1yr rate if job < 365 days, else 3yr rate
            if total_days >= 365:
                opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
                tier_label = "reserved-3yr"
            else:
                # 1yr ≈ reserved_3yr_hr / (1 - RESERVED_DISCOUNT_3YR) * (1 - RESERVED_DISCOUNT_1YR)
                # approximated as on_demand * (1 - RESERVED_DISCOUNT_1YR)
                opt_cost = gpu_hours * od * (1.0 - pricing.RESERVED_DISCOUNT_1YR)
                tier_label = "reserved-1yr"
        else:
            opt_cost = on_demand_cost
            tier_label = "on_demand"

        if tier == "spot":
            tier_label = "spot"

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({
            "job_id": j["job_id"], "gpu_type": gtype, "tier": tier, "tier_label": tier_label,
            "interrupt_rate": interrupt_rate, "total_days": total_days,
            "on_demand": round(on_demand_cost), "optimized": round(opt_cost),
        })

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    if verbose:
        print("== M3 Purchasing Strategy (Extension 1: Interruption-Rate-Aware) ==")
        print(f"break-even utilization @ 45% reserved-3yr discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"break-even utilization @ 30% reserved-1yr discount = {pricing.break_even_utilization(0.30):.0%}")
        print()
        print(f"{'job':18}{'gpu':7}{'tier':14}{'interrupt%':>12}{'days':>6}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(
                f"{r['job_id']:18}{r['gpu_type']:7}{r['tier_label']:14}"
                f"{r['interrupt_rate']*100:>11.0f}%{r['total_days']:>6}"
                f"${r['on_demand']:>11,}${r['optimized']:>11,}"
            )
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        print("\nKey insight: High-interruption GPUs (H100 8%, H200 12%) on long jobs may")
        print("prefer reserved over spot to avoid compounding rework costs.")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1)}


if __name__ == "__main__":
    run()
