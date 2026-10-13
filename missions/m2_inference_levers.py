"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Extension 2: cache_is_worth_it() is called before applying cache savings to verify
that caching is economically beneficial given the actual read/write ratio in the data.

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def _estimate_reads_per_write(rows: list[dict]) -> float:
    """Estimate how many times each cached prefix is read on average.

    Assumption: rows with cached_input_tokens > 0 represent cache *reads*;
    the number of distinct prefixes ≈ rows with cached > 0 grouped by approximate
    prefix length. We approximate reads_per_write as:
        total cached reads / estimated distinct prefixes written
    Since we don't track write events directly, we assume 1 write per unique
    (team, project, route_tier) combination and count reads per group.
    """
    groups: dict = {}
    for r in rows:
        cached = int(num(r["cached_input_tokens"]))
        if cached > 0:
            key = (r.get("team", ""), r.get("project", ""), r.get("route_tier", ""))
            groups[key] = groups.get(key, 0) + 1
    if not groups:
        return 0.0
    total_reads = sum(groups.values())
    num_groups = len(groups)
    return total_reads / num_groups


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")

    # --- Extension 2: Check cache economics BEFORE applying cache discount ---
    reads_per_write = _estimate_reads_per_write(rows)
    # Use large-model input price for cache check (worst-case: high-value tokens)
    lin_large, _ = MODEL_PRICES["large"]
    # Typical shared prefix length: median cached_input_tokens among cached rows
    cached_rows = [r for r in rows if int(num(r["cached_input_tokens"])) > 0]
    median_prefix = (
        sorted(int(num(r["cached_input_tokens"])) for r in cached_rows)[len(cached_rows) // 2]
        if cached_rows else 1000
    )
    cache_check = pricing.cache_is_worth_it(
        prefix_tokens=median_prefix,
        reads_per_write=reads_per_write,
        price_in_per_m=lin_large,
    )
    # Only apply caching if economically justified
    apply_cache = cache_check["worth_it"]

    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching (only if worth it), batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        effective_cached = cached if apply_cache else 0
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=effective_cached, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    if verbose:
        print("== M2 Inference Cost Levers (Extension 2: Cache Gate) ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print()
        print(f"[Cache Economics Check]")
        print(f"  Median prefix tokens      : {median_prefix:,}")
        print(f"  Estimated reads/write     : {reads_per_write:.1f}")
        print(f"  Break-even reads/write    : {cache_check['break_even_reads']:.2f}")
        print(f"  Cache worth it?           : {cache_check['worth_it']} "
              f"(savings {cache_check['savings_pct']:.1f}% per cycle when reads={reads_per_write:.1f})")
        print(f"  Applying cache discount   : {apply_cache}")
        print()
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + {'caching + ' if apply_cache else ''}batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        if not apply_cache:
            print(f"\nNOTE: Cache not applied — reads/write ({reads_per_write:.1f}) < "
                  f"break-even ({cache_check['break_even_reads']:.2f}). "
                  f"Increase prefix reuse before enabling caching.")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_worth_it": cache_check["worth_it"],
        "cache_break_even_reads": cache_check["break_even_reads"],
        "cache_reads_per_write": round(reads_per_write, 2),
    }


if __name__ == "__main__":
    run()
