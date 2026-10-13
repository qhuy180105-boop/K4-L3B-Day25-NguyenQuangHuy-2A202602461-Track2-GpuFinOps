"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Extension 2: Right-sizing by MBU — for memory-bound GPUs, compute $/GB-VRAM and
recommend a cheaper GPU tier with sufficient bandwidth, showing monthly savings.

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


# GPU right-sizing ladder: for memory-bound GPUs, suggest stepping down one tier.
# Key insight: memory-bound decode needs bandwidth (TB/s), not peak FLOPs.
# So $/GB-VRAM and peak_bw_tbs are the right axes, NOT $/GPU-hr alone.
RIGHTSIZE_LADDER = {
    "H100":  "A100",   # H100 80GB → A100 80GB (same VRAM, ~40% cheaper, still 2 TB/s BW)
    "H200":  "H100",   # H200 141GB → H100 80GB
    "A100":  "A10G",   # A100 80GB → A10G 24GB (for small-batch inference)
    "A10G":  "L4",     # A10G 24GB → L4 24GB (same VRAM, lower power)
    "L4":    "L4",     # already at bottom
}


def _dollar_per_gb_vram(cat_row: dict) -> float:
    """$/GB-VRAM = on_demand_hr / hbm_gb. Lower is better (more memory per dollar)."""
    od = num(cat_row["on_demand_hr"])
    hbm = num(cat_row["hbm_gb"])
    return od / hbm if hbm > 0 else float("inf")


def rightsize_memory_bound(summary: list[dict], cat: dict, days: int = 30) -> dict:
    """Extension 2: For each memory-bound GPU (low MBU relative to compute),
    recommend a cheaper right-sized alternative.

    Strategy:
      - MBU < 0.40 AND MFU < 0.25 → GPU is under-utilized on BOTH axes → idle candidate
      - MBU >= 0.40 (memory-bound decode pattern): compare $/GB-VRAM across tiers
      - Select replacement only if it has sufficient peak_bw_tbs for the workload
        and costs meaningfully less per GB-VRAM

    Why not just pick the cheapest $/GPU-hr?
      Because a tiny cheap GPU may lack the HBM bandwidth needed to serve LLM decode
      at acceptable latency. E.g., L4 (0.30 TB/s) cannot replace A100 (2.0 TB/s) for
      large-batch decode — the bottleneck moves from cost to latency SLA.
    """
    recommendations = []
    total_monthly_savings = 0.0

    for s in summary:
        gtype = s["gpu_type"]
        if gtype not in cat or gtype not in RIGHTSIZE_LADDER:
            continue

        cur_row = cat[gtype]
        tgt_type = RIGHTSIZE_LADDER[gtype]
        if tgt_type == gtype:
            continue  # already at bottom
        tgt_row = cat[tgt_type]

        cur_od = num(cur_row["on_demand_hr"])
        tgt_od = num(tgt_row["on_demand_hr"])
        cur_bw = num(cur_row["peak_bw_tbs"])
        tgt_bw = num(tgt_row["peak_bw_tbs"])
        cur_hbm = num(cur_row["hbm_gb"])
        tgt_hbm = num(tgt_row["hbm_gb"])

        cur_dpg = _dollar_per_gb_vram(cur_row)
        tgt_dpg = _dollar_per_gb_vram(tgt_row)

        # Only recommend if:
        # 1. MBU is moderate (memory-bound pattern, not compute-bound)
        # 2. Target GPU has at least 40% of current BW (latency SLA guard)
        # 3. Target is meaningfully cheaper per GPU-hr
        mbu = s["mbu"]
        mfu = s["mfu"]
        is_memory_bound = mbu >= 0.30 and mfu < 0.35
        bw_adequate = tgt_bw >= cur_bw * 0.40
        cost_savings = cur_od - tgt_od

        if is_memory_bound and bw_adequate and cost_savings > 0:
            # Monthly savings: 24h × 30 days × price delta (per GPU)
            monthly_savings = cost_savings * 24 * days
            total_monthly_savings += monthly_savings
            savings_pct = (cost_savings / cur_od * 100) if cur_od > 0 else 0.0

            recommendations.append({
                "gpu_id": s["gpu_id"],
                "current_gpu": gtype,
                "recommended_gpu": tgt_type,
                "current_od_hr": cur_od,
                "recommended_od_hr": tgt_od,
                "savings_pct": round(savings_pct, 1),
                "monthly_savings_usd": round(monthly_savings, 2),
                "current_bw_tbs": cur_bw,
                "recommended_bw_tbs": tgt_bw,
                "current_hbm_gb": cur_hbm,
                "recommended_hbm_gb": tgt_hbm,
                "current_dpg_vram": round(cur_dpg, 4),
                "recommended_dpg_vram": round(tgt_dpg, 4),
                "mbu": mbu,
                "mfu": mfu,
                "reason": (
                    f"Memory-bound (MBU={mbu:.2f}, MFU={mfu:.2f}): "
                    f"{tgt_type} has {tgt_bw:.2f} TB/s BW (adequate) "
                    f"at ${tgt_od:.2f}/hr vs ${cur_od:.2f}/hr"
                ),
            })

    return {
        "recommendations": recommendations,
        "total_monthly_savings": round(total_monthly_savings, 2),
    }


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # --- Extension 2: Right-sizing by MBU ---
    rs = rightsize_memory_bound(summary, cat)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print("\n--- Extension 2: Right-sizing by MBU ---")
        print("Why not just pick cheapest $/GPU-hr?")
        print("  Memory-bound decode needs bandwidth (TB/s), not peak FLOPs.")
        print("  A cheap GPU with low HBM BW creates latency bottlenecks for LLM decode.\n")

        # $/GB-VRAM table for all GPU types in fleet
        fleet_types = sorted({s["gpu_type"] for s in summary})
        print(f"  {'GPU':8} {'$/hr':>8} {'HBM GB':>8} {'$/GB-VRAM':>11} {'BW (TB/s)':>10}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*11} {'-'*10}")
        for gtype in fleet_types:
            row = cat[gtype]
            dpg = _dollar_per_gb_vram(row)
            print(f"  {gtype:8} ${num(row['on_demand_hr']):>7.2f} {num(row['hbm_gb']):>8.0f}"
                  f" ${dpg:>10.4f} {num(row['peak_bw_tbs']):>10.2f}")

        if rs["recommendations"]:
            print(f"\nRight-sizing recommendations (memory-bound GPUs):")
            for rec in rs["recommendations"]:
                print(f"  {rec['gpu_id']}: {rec['current_gpu']} -> {rec['recommended_gpu']}")
                print(f"    MBU={rec['mbu']:.2f} MFU={rec['mfu']:.2f}  "
                      f"${rec['current_od_hr']:.2f}/hr -> ${rec['recommended_od_hr']:.2f}/hr  "
                      f"({rec['savings_pct']:.0f}% cheaper)")
                print(f"    BW: {rec['current_bw_tbs']:.2f} -> {rec['recommended_bw_tbs']:.2f} TB/s  "
                      f"Monthly savings: ${rec['monthly_savings_usd']:,.0f}")
            print(f"\n  Total monthly savings if right-sized: ${rs['total_monthly_savings']:,.0f}")
        else:
            print("  No right-sizing recommendations (all GPUs either at BW limit or already optimal).")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "rightsize": rs,
    }


if __name__ == "__main__":
    run()
