"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


# GPU spot interruption rates by type (assumed, based on market availability patterns).
# H100/H200 are highest-demand → higher interruption risk; L4/A10G are more available.
GPU_INTERRUPTION_RATE = {
    "H200": 0.12,   # very scarce, high interruption risk
    "H100": 0.08,   # popular training GPU, moderate-high risk
    "A100": 0.05,   # widely deployed, moderate risk
    "A10G": 0.03,   # inference workhorse, lower risk
    "L4":   0.02,   # widely available, low risk
}
DEFAULT_INTERRUPTION_RATE = 0.06  # fallback for unknown GPU types

# Reserved discount tiers (illustrative June-2026 snapshot)
RESERVED_DISCOUNT_1YR = 0.30   # ~30% off on-demand for 1-year commitment
RESERVED_DISCOUNT_3YR = 0.45   # ~45% off on-demand for 3-year commitment
SPOT_DISCOUNT = 0.60            # ~60% off on-demand for spot instances


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = RESERVED_DISCOUNT_3YR,
    gpu_type: str = "A100",
    total_days: int = 90,
) -> str:
    """Pick a purchasing tier from a workload's duty cycle, interruptibility,
    GPU-specific interruption rate, and commitment duration vs break-even.

    ENHANCED policy (Extension 1):
      - interruptible & not 24/7:
          * High interruption rate (>=8%) → spot only for short jobs
          * Low interruption rate (<5%)   → spot is safe for most durations
          * For very long jobs (>180 days) even interruptible may prefer reserved
      - Non-interruptible high duty cycle:
          * Compare 1yr vs 3yr reserved: 3yr only if job spans >365 days
          * Break-even check against actual utilization
      - Otherwise → on_demand

    Args:
        hours_per_day: average GPU hours used per day (0–24)
        interruptible: whether the job supports checkpointing + restart
        reserved_discount: reserved discount fraction (default = 3yr = 0.45)
        gpu_type: GPU model (used to look up interruption risk)
        total_days: estimated total job duration in days (for commitment comparison)
    """
    duty = max(0.0, hours_per_day) / 24.0
    interrupt_rate = GPU_INTERRUPTION_RATE.get(gpu_type, DEFAULT_INTERRUPTION_RATE)

    # Choose reserved commitment tier based on job duration
    if total_days >= 365:
        chosen_reserved_discount = RESERVED_DISCOUNT_3YR
    else:
        chosen_reserved_discount = RESERVED_DISCOUNT_1YR

    be = break_even_utilization(chosen_reserved_discount)

    if interruptible and hours_per_day < 24:
        # High-risk GPU types on very long jobs: interruptions become costly
        # Even if interruptible, for high-demand GPUs running 180+ days,
        # reserved becomes safer (fewer reworks, more predictable cost)
        if interrupt_rate >= 0.08 and total_days >= 180:
            # Check if reserved still beats effective spot cost after interruptions
            effective_spot_discount = SPOT_DISCOUNT * (1 - interrupt_rate * 0.5)
            if chosen_reserved_discount > effective_spot_discount:
                return "reserved"
        return "spot"

    if duty >= be:
        return "reserved"

    return "on_demand"


def cache_is_worth_it(
    prefix_tokens: int,
    reads_per_write: float,
    price_in_per_m: float,
    cache_discount: float = 0.10,
    write_overhead_frac: float = 0.05,
) -> dict:
    """Determine whether prompt caching is economically beneficial.

    Prompt caching saves on *reads* but incurs a small write overhead each time
    the cache is populated (assumed ~write_overhead_frac of token cost).
    The cache only pays off when reads_per_write is above the break-even point.

    Economics (per 1M prefix tokens):
      - Without cache: 1M tokens billed at full price each read → cost = reads * price_in_per_m
      - With cache:    write cost + reads * cache_discount * price_in_per_m + write overhead
      - Break-even:    when savings from reads > write overhead cost

    Args:
        prefix_tokens:      number of tokens in the shared prompt prefix
        reads_per_write:    average number of times a cached prefix is read before expiry
        price_in_per_m:     input token price per 1M tokens (USD)
        cache_discount:     fraction billed for cached reads (default 0.10 = 90% off)
        write_overhead_frac: extra cost fraction for cache write vs normal input (default 0.05)

    Returns:
        dict with keys:
          - worth_it (bool): True if caching is economically beneficial
          - break_even_reads (float): minimum reads/write for cache to pay off
          - cost_without_cache (float): USD cost per write cycle without cache
          - cost_with_cache (float): USD cost per write cycle with cache
          - savings_usd (float): net savings per write cycle
          - savings_pct (float): percentage savings vs no-cache
    """
    token_m = prefix_tokens / 1_000_000.0

    # Cost per write cycle WITHOUT cache: every read pays full price
    cost_without_cache = reads_per_write * token_m * price_in_per_m

    # Cost per write cycle WITH cache:
    #   - 1 write at (1 + write_overhead_frac) × full price
    #   - (reads_per_write) reads at cache_discount × full price
    write_cost = (1.0 + write_overhead_frac) * token_m * price_in_per_m
    read_cost = reads_per_write * token_m * price_in_per_m * cache_discount
    cost_with_cache = write_cost + read_cost

    savings_usd = cost_without_cache - cost_with_cache
    savings_pct = (savings_usd / cost_without_cache * 100.0) if cost_without_cache > 0 else 0.0

    # Break-even: at what reads_per_write does cost_with_cache == cost_without_cache?
    # writes_overhead * price = reads * (1 - cache_discount) * price
    # => break_even = (1 + write_overhead_frac) / (1 - cache_discount)
    break_even_reads = (1.0 + write_overhead_frac) / (1.0 - cache_discount)

    return {
        "worth_it": reads_per_write > break_even_reads,
        "break_even_reads": round(break_even_reads, 3),
        "cost_without_cache": round(cost_without_cache, 6),
        "cost_with_cache": round(cost_with_cache, 6),
        "savings_usd": round(savings_usd, 6),
        "savings_pct": round(savings_pct, 2),
    }


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
