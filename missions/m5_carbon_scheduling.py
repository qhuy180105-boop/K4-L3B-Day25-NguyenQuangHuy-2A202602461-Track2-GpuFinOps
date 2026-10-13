"""Extension 5 — Carbon-aware Scheduling.

For interruptible workloads, compute carbon and electricity cost across all 5
regions. Recommend the optimal region per criterion (cheapest $, cleanest CO2,
best balanced) and quantify total gCO2e savings vs us-east-1 baseline.

Run: python missions/m5_carbon_scheduling.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30
# GPU power draw (watts) — from price catalog 'watts' column
# Used to compute actual kWh consumed by each job


def run_carbon_scheduling(verbose: bool = True) -> dict:
    """Extension 5: Carbon-aware scheduling for interruptible workloads.

    For each interruptible=1 job, compute:
      - kWh consumed (based on GPU watts × hours × num_gpus)
      - electricity cost ($/kWh by region)
      - carbon footprint (gCO2e by region)

    Then recommend:
      - Cheapest $ region
      - Cleanest CO2 region
      - Best balanced region (Pareto-optimal: low cost AND low carbon)

    Trade-off note: the cleanest region (europe-north1, Norway) may introduce
    100-200 ms extra latency for US users. This is acceptable for batch/offline
    interruptible jobs but NOT for real-time inference with SLA < 200 ms.
    """
    jobs    = load_csv("workloads.csv")
    cat     = catalog_by_type()
    regions = list(sustainability.REGION_CARBON.keys())

    interruptible_jobs = [j for j in jobs if int(num(j["interruptible"])) == 1]

    # --- Per-region totals across all interruptible jobs ---
    region_totals: dict[str, dict] = {
        r: {"kwh": 0.0, "elec_usd": 0.0, "co2_g": 0.0} for r in regions
    }

    job_details = []
    for j in interruptible_jobs:
        gtype  = j["gpu_type"]
        ngpu   = int(num(j["num_gpus"]))
        hpd    = num(j["hours_per_day"])
        days_j = int(num(j.get("days", DAYS)))
        c      = cat[gtype]
        watts  = num(c["watts"])

        # Effective GPU-hours (with spot rework overhead for spot jobs)
        interrupt_rate = pricing.GPU_INTERRUPTION_RATE.get(gtype, pricing.DEFAULT_INTERRUPTION_RATE)
        od_hr = num(c["on_demand_hr"])
        spot_hr = num(c["spot_hr"])
        job_hours = hpd * days_j * ngpu
        spot_sim  = pricing.spot_checkpoint_cost(
            job_hours, spot_hr, od_hr, interrupt_rate=interrupt_rate)
        effective_hours = spot_sim["spot_effective_hours"]

        # kWh consumed (effective hours × watts / 1000)
        kwh = effective_hours * (watts / 1000.0)

        per_job_regions = {}
        for region in regions:
            elec_usd = sustainability.energy_cost_usd(kwh * 1000, region)  # Wh→kWh done inside
            co2_g    = sustainability.carbon_g(kwh * 1000, region)          # Wh input
            per_job_regions[region] = {
                "kwh": round(kwh, 2),
                "elec_usd": round(elec_usd, 2),
                "co2_kg": round(co2_g / 1000, 3),
            }
            region_totals[region]["kwh"]     += kwh
            region_totals[region]["elec_usd"] += elec_usd
            region_totals[region]["co2_g"]   += co2_g

        job_details.append({
            "job_id":       j["job_id"],
            "gpu_type":     gtype,
            "num_gpus":     ngpu,
            "hours_per_day": hpd,
            "days":         days_j,
            "effective_hours": effective_hours,
            "kwh":          round(kwh, 2),
            "regions":      per_job_regions,
        })

    # --- Region comparison table ---
    region_summary = []
    for region in regions:
        t = region_totals[region]
        region_summary.append({
            "region":           region,
            "carbon_intensity": sustainability.REGION_CARBON[region],
            "price_kwh":        sustainability.REGION_PRICE_KWH[region],
            "total_kwh":        round(t["kwh"], 2),
            "total_elec_usd":   round(t["elec_usd"], 2),
            "total_co2_kg":     round(t["co2_g"] / 1000, 3),
        })

    # Sort by different criteria
    cheapest_region  = min(region_summary, key=lambda r: r["total_elec_usd"])
    cleanest_region  = min(region_summary, key=lambda r: r["total_co2_kg"])
    # Balanced: normalize both metrics (0-1) and minimize sum
    max_cost = max(r["total_elec_usd"] for r in region_summary) or 1
    max_co2  = max(r["total_co2_kg"]   for r in region_summary) or 1
    balanced_region = min(
        region_summary,
        key=lambda r: (r["total_elec_usd"] / max_cost) + (r["total_co2_kg"] / max_co2)
    )

    # Savings vs us-east-1 baseline
    baseline_co2_kg  = region_totals["us-east-1"]["co2_g"] / 1000
    baseline_elec_usd = region_totals["us-east-1"]["elec_usd"]
    cleanest_co2_kg  = region_totals[cleanest_region["region"]]["co2_g"] / 1000
    cleanest_elec_usd = region_totals[cleanest_region["region"]]["elec_usd"]
    cheapest_elec_usd = region_totals[cheapest_region["region"]]["elec_usd"]

    co2_savings_kg    = baseline_co2_kg - cleanest_co2_kg
    elec_savings_usd  = baseline_elec_usd - cheapest_elec_usd
    co2_savings_pct   = (co2_savings_kg / baseline_co2_kg * 100) if baseline_co2_kg > 0 else 0

    if verbose:
        print("== Extension 5: Carbon-aware Scheduling ==")
        print(f"Analyzing {len(interruptible_jobs)} interruptible workloads across {len(regions)} regions")
        print()
        print("Region comparison (total for all interruptible jobs):")
        print(f"  {'Region':20} {'gCO2/kWh':>10} {'$/kWh':>8} {'kWh':>10} {'Elec $':>10} {'CO2 (kg)':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
        for rs in sorted(region_summary, key=lambda r: r["total_co2_kg"]):
            print(
                f"  {rs['region']:20} {rs['carbon_intensity']:>10} "
                f"${rs['price_kwh']:>7.3f} {rs['total_kwh']:>10.2f} "
                f"${rs['total_elec_usd']:>9.2f} {rs['total_co2_kg']:>10.3f}"
            )
        print()
        print(f"  Cheapest $    : {cheapest_region['region']}  "
              f"(${cheapest_region['total_elec_usd']:.2f} electricity cost)")
        print(f"  Cleanest CO2  : {cleanest_region['region']}  "
              f"({cleanest_region['total_co2_kg']:.3f} kg CO2)")
        print(f"  Best balanced : {balanced_region['region']}  "
              f"(lowest normalized cost + carbon)")
        print()
        print("Savings vs us-east-1 baseline:")
        print(f"  CO2 reduction (-> {cleanest_region['region']}): "
              f"{co2_savings_kg:.3f} kg CO2e  ({co2_savings_pct:.0f}% reduction)")
        print(f"  Electricity savings (-> {cheapest_region['region']}): "
              f"${elec_savings_usd:.2f}  "
              f"({(elec_savings_usd/baseline_elec_usd*100) if baseline_elec_usd else 0:.0f}% reduction)")
        print()
        print("Trade-off analysis:")
        print("  europe-north1 (Norway, hydro): near-zero carbon but ~100-200 ms extra")
        print("  latency for US users. Suitable for BATCH/offline interruptible jobs only.")
        print("  us-east-wa    (WA state):      90% cleaner than us-east-1, <10 ms added")
        print("  latency from east-coast users. Best for latency-sensitive interruptible workloads.")
        print("  us-west-2     (Oregon, hydro): 68% cleaner, lowest $/kWh in US ($0.07).")
        print()
        print("Optimal strategy:")
        print(f"  [Batch/offline] Run in {cleanest_region['region']} for max carbon reduction.")
        print("  [Spot inference, SLA < 200 ms] Run in us-east-wa or us-west-2.")
        print("  [Non-interruptible prod] Keep in us-east-1 / us-west-2; buy RECs for carbon.")

    return {
        "interruptible_jobs": len(interruptible_jobs),
        "region_summary": region_summary,
        "cheapest_region":   cheapest_region["region"],
        "cleanest_region":   cleanest_region["region"],
        "balanced_region":   balanced_region["region"],
        "baseline_region":   "us-east-1",
        "co2_savings_kg":    round(co2_savings_kg, 3),
        "co2_savings_pct":   round(co2_savings_pct, 1),
        "elec_savings_usd":  round(elec_savings_usd, 2),
        "job_details":       job_details,
    }


if __name__ == "__main__":
    run_carbon_scheduling()
