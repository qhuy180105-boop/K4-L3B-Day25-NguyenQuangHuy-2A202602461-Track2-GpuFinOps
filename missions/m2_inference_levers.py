"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Extension 3: cache_is_worth_it() is called before applying cache savings to verify
that caching is economically beneficial given the actual read/write ratio in the data.

Extension 4: Reasoning Budget — separate cost & energy for is_reasoning=1 vs 0,
propose routing rules and estimate savings from capping reasoning traffic.

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# Energy per 1k tokens (Wh) — used for sustainability accounting
WH_PER_1K_TOKENS = 0.30


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


def reasoning_budget_analysis(rows: list[dict]) -> dict:
    """Extension 4: Decompose cost and energy for reasoning vs standard traffic.

    Key insight: reasoning queries use extended chain-of-thought, generating
    ~74-86x more output tokens than a standard small-model query. This creates
    a disproportionate energy and cost footprint relative to traffic volume.

    Routing rule proposal: only invoke reasoning mode when task complexity
    warrants it (e.g., multi-step math, code generation > N lines, legal analysis).
    Capping reasoning at 10% of traffic can yield major cost and energy savings.
    """
    reasoning_rows = [r for r in rows if int(num(r["is_reasoning"])) == 1]
    standard_rows  = [r for r in rows if int(num(r["is_reasoning"])) == 0]

    def _cost_and_energy(subset: list[dict]) -> tuple[float, float, int, int]:
        total_cost = 0.0
        total_wh = 0.0
        total_tokens = 0
        n_requests = len(subset)
        for r in subset:
            inp  = int(num(r["input_tokens"]))
            out  = int(num(r["output_tokens"]))
            pin, pout = MODEL_PRICES[r["route_tier"]]
            cached = int(num(r["cached_input_tokens"]))
            is_batch = bool(int(num(r["is_batch"])))
            total_cost += pricing.request_cost(inp, out, pin, pout,
                                               cached_in=cached, batch=is_batch)
            # Energy: reasoning multiplier applied for reasoning rows
            is_r = int(num(r["is_reasoning"])) == 1
            total_wh += sustainability.wh_per_query(inp + out, WH_PER_1K_TOKENS,
                                                     is_reasoning=is_r)
            total_tokens += inp + out
        return total_cost, total_wh, total_tokens, n_requests

    r_cost, r_wh, r_tokens, r_n = _cost_and_energy(reasoning_rows)
    s_cost, s_wh, s_tokens, s_n = _cost_and_energy(standard_rows)
    total_cost = r_cost + s_cost
    total_wh   = r_wh + s_wh
    total_n    = r_n + s_n

    reasoning_traffic_pct = (r_n / total_n * 100) if total_n > 0 else 0.0
    reasoning_cost_pct    = (r_cost / total_cost * 100) if total_cost > 0 else 0.0
    reasoning_energy_pct  = (r_wh / total_wh * 100) if total_wh > 0 else 0.0

    # --- What-if: cap reasoning to 10% of traffic ---
    # If we cap at 10%, some reasoning traffic is rerouted to standard (large) model.
    # We re-run those extra requests as standard large-model calls.
    target_reasoning_pct = 0.10
    if reasoning_traffic_pct / 100 > target_reasoning_pct:
        # Number of reasoning requests to convert to standard large-model calls
        allowed_reasoning = int(total_n * target_reasoning_pct)
        rerouted_n = r_n - allowed_reasoning
        # Estimate cost/energy for the rerouted batch as large-model standard calls
        # Use average tokens per reasoning request
        avg_inp_r = (r_tokens / r_n) * 0.4 if r_n > 0 else 800   # rough input fraction
        avg_out_r = (r_tokens / r_n) * 0.6 if r_n > 0 else 400   # rough output fraction
        lin, lout = MODEL_PRICES["large"]
        rerouted_cost = rerouted_n * pricing.request_cost(
            int(avg_inp_r), int(avg_out_r), lin, lout)
        rerouted_wh = rerouted_n * sustainability.wh_per_query(
            int(avg_inp_r + avg_out_r), WH_PER_1K_TOKENS, is_reasoning=False)

        # Cost of keeping only allowed_reasoning reasoning requests
        kept_r_frac = allowed_reasoning / r_n if r_n > 0 else 0
        kept_r_cost = r_cost * kept_r_frac
        kept_r_wh   = r_wh * kept_r_frac

        cap_total_cost = s_cost + kept_r_cost + rerouted_cost
        cap_total_wh   = s_wh + kept_r_wh + rerouted_wh
        cap_savings_usd = total_cost - cap_total_cost
        cap_savings_wh  = total_wh - cap_total_wh
        cap_savings_pct = (cap_savings_usd / total_cost * 100) if total_cost > 0 else 0.0
    else:
        cap_savings_usd = 0.0
        cap_savings_wh  = 0.0
        cap_savings_pct = 0.0
        rerouted_n = 0

    return {
        "reasoning_requests": r_n,
        "standard_requests":  s_n,
        "total_requests":     total_n,
        "reasoning_traffic_pct":  round(reasoning_traffic_pct, 1),
        "reasoning_cost_usd":     round(r_cost, 4),
        "standard_cost_usd":      round(s_cost, 4),
        "reasoning_cost_pct":     round(reasoning_cost_pct, 1),
        "reasoning_energy_wh":    round(r_wh, 4),
        "standard_energy_wh":     round(s_wh, 4),
        "reasoning_energy_pct":   round(reasoning_energy_pct, 1),
        "cap_10pct_savings_usd":  round(cap_savings_usd, 4),
        "cap_10pct_savings_wh":   round(cap_savings_wh, 4),
        "cap_10pct_savings_pct":  round(cap_savings_pct, 1),
        "rerouted_requests":      rerouted_n,
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")

    # --- Extension 3: Check cache economics BEFORE applying cache discount ---
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
    opt_pm  = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # --- Extension 4: Reasoning Budget ---
    rb = reasoning_budget_analysis(rows)

    if verbose:
        print("== M2 Inference Cost Levers (Extension 3: Cache Gate + Extension 4: Reasoning Budget) ==")
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
            print(f"\nNOTE: Cache not applied -- reads/write ({reads_per_write:.1f}) < "
                  f"break-even ({cache_check['break_even_reads']:.2f}). "
                  f"Increase prefix reuse before enabling caching.")

        print()
        print("--- Extension 4: Reasoning Budget Analysis ---")
        print(f"  Reasoning traffic : {rb['reasoning_requests']:>5} requests  "
              f"({rb['reasoning_traffic_pct']:.1f}% of total)")
        print(f"  Standard traffic  : {rb['standard_requests']:>5} requests  "
              f"({100 - rb['reasoning_traffic_pct']:.1f}% of total)")
        print()
        print(f"  Cost breakdown:")
        print(f"    Reasoning  : ${rb['reasoning_cost_usd']:>10.4f}  ({rb['reasoning_cost_pct']:.1f}% of total cost)")
        print(f"    Standard   : ${rb['standard_cost_usd']:>10.4f}  ({100 - rb['reasoning_cost_pct']:.1f}% of total cost)")
        print()
        print(f"  Energy breakdown:")
        print(f"    Reasoning  : {rb['reasoning_energy_wh']:>10.4f} Wh  ({rb['reasoning_energy_pct']:.1f}% of total energy)")
        print(f"    Standard   : {rb['standard_energy_wh']:>10.4f} Wh  ({100 - rb['reasoning_energy_pct']:.1f}% of total energy)")
        print()
        mult = sustainability.REASONING_ENERGY_MULTIPLIER
        print(f"  Why reasoning costs ~{mult:.0f}x more energy:")
        print(f"    Reasoning uses extended chain-of-thought (CoT), generating far more")
        print(f"    output tokens. Each token requires a full autoregressive forward pass")
        print(f"    on a large model -- so 80x tokens = ~80x compute and ~80x energy.")
        print()
        if rb["cap_10pct_savings_pct"] > 0:
            print(f"  What-if: cap reasoning traffic at 10% (currently {rb['reasoning_traffic_pct']:.1f}%):")
            print(f"    Rerouted {rb['rerouted_requests']} reasoning -> standard large-model calls")
            print(f"    Daily cost savings : ${rb['cap_10pct_savings_usd']:,.4f}  ({rb['cap_10pct_savings_pct']:.1f}%)")
            print(f"    Daily energy saved : {rb['cap_10pct_savings_wh']:,.4f} Wh")
            print()
            print(f"  Routing rule proposal:")
            print(f"    Route to reasoning ONLY when:")
            print(f"      - Task type in {{math_proof, code_debug, legal_analysis}}")
            print(f"      - Estimated task complexity score > 0.7 (classifier gate)")
            print(f"      - User tier == 'premium' (business policy gate)")
            print(f"    Otherwise: route to large standard model (fast, 80x cheaper per query).")
        else:
            print(f"  Reasoning traffic ({rb['reasoning_traffic_pct']:.1f}%) is already at or below 10% -- no cap needed.")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_worth_it": cache_check["worth_it"],
        "cache_break_even_reads": cache_check["break_even_reads"],
        "cache_reads_per_write": round(reads_per_write, 2),
        "reasoning_budget": rb,
    }


if __name__ == "__main__":
    run()
