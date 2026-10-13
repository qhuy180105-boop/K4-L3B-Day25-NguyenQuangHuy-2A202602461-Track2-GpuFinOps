# NimbusAI -- GPU Cost Optimization Report

> **Period:** Monthly (June 2026 snapshot)  
> **Data:** Synthetic fleet of 11 GPUs, 2,400 inference requests, 8 training jobs  
> **Figures:** June-2026 illustrative neocloud pricing; re-baseline before acting.

---

## Executive Summary

| Metric | Value |
|---|---|
| Baseline monthly spend | $27,133 |
| Optimized monthly spend | $15,959 |
| Projected savings | $11,174 (**41%**) |
| Inference cost reduction | 83% ($6.49 -> $1.13/1M-token) |
| Purchasing savings | 34% ($25,667 -> $16,960/month) |
| Tag coverage (chargeback gate) | 92% (gate: open) |

---

## Savings by Lever

| Lever | Monthly Savings (USD) | % of Total Savings |
|---|---|---|
| Inference (cascade/cache/batch) | $1,212 | 11% |
| Purchasing (spot/reserved) | $8,707 | 78% |
| Right-size util-lies | $655 | 6% |
| Kill idle GPUs | $600 | 5% |

---

## Lever 1: Inference Cost Optimization (cascade / cache / batch)

### Baseline vs. Optimized

- **Baseline:** All requests served by large model, no caching, no batching
  - Daily cost: $48.87  |  Unit economics: $6.488/1M-token
- **Optimized:** Model cascade + prompt cache + batch API
  - Daily cost: $8.48  |  Unit economics: $1.126/1M-token
- **Savings: 83%** ($1,212/month)

### Three Inference Levers

**1. Model Cascade (routing)**
80% of requests are routed to the small model ($0.20/$0.40 per 1M vs $3.00/$15.00).
Cascade only applies to tasks where a smaller model meets quality SLA.
Wrong routing -- sending complex tasks to small model -- creates false economy:
re-work and hallucinations cost more than the model price difference.

**2. Prompt Caching**
Cache economics check: median prefix = 1.17 break-even reads/write,
actual estimated reads/write = 150x. Cache is **worth it**
(89% savings per cycle at 150x reuse). Teams with large static system
prompts (RAG, assistant) benefit most -- the shared prefix is written once and
read hundreds of times at 10% of the normal input price.

**3. Batch API**
Eval team traffic is fully batchable (nightly runs, no latency SLA).
Batch API discount: 50% off on all tokens. Stacked with caching:
effective multiplier = 0.050x of naive bill (95% off).

### Extension 3: Cache Gate
Before applying cache savings, `cache_is_worth_it()` validates that the
reads/write ratio exceeds break-even (1.17). Only if TRUE does the pipeline
apply the cached-token discount. This prevents negative ROI on low-reuse prefixes.

### Extension 4: Reasoning Budget

| Metric | Value |
|---|---|
| Reasoning traffic | 201 requests (8.4% of total) |
| Reasoning cost share | 16.5% of total cost |
| Reasoning energy share | 94.0% of total energy |
| Standard energy share | 6.0% of total energy |

**Key finding:** Reasoning traffic is only 8.4% of requests but consumes **94% of total
inference energy**. This is because reasoning (chain-of-thought) mode generates ~80x
more output tokens than a standard query. Since each output token requires one full
autoregressive forward pass on a large model, the energy cost scales linearly with
token count: 80x tokens = ~80x GPU compute = ~80x energy and ~80x cost.

**Current status:** Reasoning traffic (8.4%) is already below the 10% cap threshold.
No immediate action needed, but monitoring is critical. If reasoning traffic grows
above 10%, apply the routing rule:

```
Route to REASONING only when ALL of:
  - task_type in {math_proof, code_debug, legal_analysis}
  - complexity_score > 0.7 (classifier gate)
  - user_tier == 'premium'
Otherwise: route to large standard model (fast, ~80x cheaper per query).
```

---

## Lever 2: Purchasing Strategy (spot / reserved)

- **On-demand baseline:** $25,667/month
- **Optimized (mixed tier):** $16,960/month
- **Savings: 34%** ($8,707/month)

### Tier Selection Logic (Extension 1: Interruption-Rate-Aware)

| Job | GPU | Tier | Interrupt Rate | Reasoning |
|---|---|---|---|---|
| job-train-llm | H100 | spot | 8% | Short job (14d), interruptible, spot cheaper despite 8% interrupt rate |
| job-train-embed | A100 | spot | 5% | Short (5d), interruptible, spot wins |
| job-finetune | H100 | spot | 8% | Very short (3d), spot optimal |
| job-infer-chat | A10G | reserved-1yr | 3% | 24h/day, non-interruptible, duty=100% > break-even 70% |
| job-infer-rag | A100 | reserved-1yr | 5% | 24h/day, non-interruptible, reserved wins |
| job-infer-search | L4 | reserved-1yr | 2% | 18h/day, duty=75% > break-even 70% |
| job-dev-sandbox | A10G | spot | 3% | 8h/day, interruptible, spot safe |
| job-batch-eval | H100 | spot | 8% | 3h/day, interruptible, spot wins |

**Break-even utilization:**
- 1-year reserved (30% off): need >= 70% utilization (16.8h/day) to beat on-demand
- 3-year reserved (45% off): need >= 55% utilization (13.2h/day) to beat on-demand

**Enhanced policy:** The `recommend_tier()` function (Extension 1) factors in
GPU-specific interruption rates. H100/H200 have 8-12% spot interruption risk --
on very long jobs (>180 days), compounding rework overhead can erode spot savings.
For those cases, the policy switches to reserved even for interruptible workloads.

---

## Lever 3: Right-sizing GPU-Util Lies

### What is the GPU-Util 'Lie'?

nvidia-smi reports `GPU-Util %` as a **time-active** signal: it measures the fraction
of sampling intervals where the GPU had at least one active kernel. It does NOT measure
how efficiently those kernels used the hardware.

**Mechanism:** A GPU can report 98% util while its Model FLOPs Utilization (MFU) is only
19.4%. This happens because:
1. **Memory stalls:** The SM (streaming multiprocessor) waits for HBM data; it is
   technically 'active' (kernel scheduled) but doing no useful FLOP work.
2. **Kernel launch overhead:** Small, poorly batched CUDA kernels fragment the pipeline.
   The GPU alternates between tiny compute bursts and dispatch gaps -- 'active' but idle.
3. **Memory-bound decode pattern:** LLM autoregressive decode loads the full model weight
   for each token (arithmetic intensity ~1-2 FLOP/byte vs H100 ridge point of ~295).
   The GPU is compute-starved, not compute-saturated.

**Cost implication:** You pay the full on-demand rate ($2.50/hr for H100) for a GPU
doing 19% of the FLOPs you rented. The effective cost per useful FLOP is 5x higher
than on a well-utilized GPU with MFU=0.42.

### Detected Util Lies

| GPU | util% | MFU | MBU | Verdict |
|---|---|---|---|---|
| gpu-h100-4 | 98.2% | 0.194 | 0.207 | LIE: memory stall / kernel inefficiency |
| gpu-a10g-1 | 96.9% | 0.268 | 0.302 | LIE: borderline, low MFU at high util |

### Extension 2: Right-sizing by MBU

**Why not just pick the cheapest $/GPU-hr?**
Memory-bound LLM decode needs HBM bandwidth (TB/s), not peak FLOPs.
Selecting the cheapest GPU by $/GPU-hr ignores whether it has enough memory
bandwidth to meet the decode throughput SLA. Example: L4 ($0.80/hr, 0.30 TB/s)
cannot replace A100 ($1.79/hr, 2.0 TB/s) for large-batch decode -- the bottleneck
shifts from cost to latency, violating the serving SLA.

**$/GB-VRAM is the right metric** for memory-bound inference right-sizing:

| GPU | $/hr | HBM (GB) | $/GB-VRAM | BW (TB/s) |
|---|---|---|---|---|
| A100 | $1.79 | 80 | $0.0224 | 2.00 |
| H100 | $2.50 | 80 | $0.0312 | 3.35 |
| L4   | $0.80 | 24 | $0.0333 | 0.30 |
| A10G | $1.00 | 24 | $0.0417 | 0.60 |

**Recommendation:** gpu-a10g-1 (A10G -> L4): MBU=0.30, MFU=0.27, borderline
memory-bound. L4 has same 24 GB VRAM at 20% lower cost ($0.80 vs $1.00/hr).
BW is halved (0.30 vs 0.60 TB/s) -- acceptable only for low-throughput embed workloads.
**Monthly savings if right-sized: $144**

**Monthly savings from killing util-lies (right-size to next tier down):** ~$655

---

## Lever 4: Kill Idle GPUs

**gpu-h100-5** ($2.50/hr H100): runs at 61% util during business hours but drops
to <5% util for 8 hours overnight (00:00-08:00). Training completed; instance left
running. Cost of overnight idle: 8h x $2.50 = **$20/day = $600/month** burned.

**Fix:** Automated shutdown trigger when GPU util < 10% for > 30 minutes.
Re-launch with preemptible startup script when next training run begins.
Tools: cloud instance scheduler, Slurm idle timeout, or k8s autoscaler TTL.

---

## Sustainability Analysis

### Baseline Energy & Carbon

| Metric | Value |
|---|---|
| Energy per query (median 800 tokens) | 0.24 Wh |
| Carbon per query (us-east-1 grid) | 0.091 gCO2e |
| Cheapest+cleanest region | **europe-north1** |

### Extension 5: Carbon-aware Scheduling

For 5 interruptible workloads running on spot instances, we computed energy and
carbon across all 5 deployment regions:

| Region | gCO2/kWh | $/kWh | Total kWh | Elec. Cost | CO2 (kg) |
|---|---|---|---|---|---|
| europe-north1 (Norway) | 30 | $0.090 | 1,911.7 | $172.05 | 57.4 |
| us-east-wa (WA state) | 90 | $0.055 | 1,911.7 | $105.14 | 172.1 |
| us-west-2 (Oregon) | 120 | $0.070 | 1,911.7 | $133.82 | 229.4 |
| us-east-1 (Virginia) | 380 | $0.120 | 1,911.7 | $229.41 | 726.5 |
| europe-central2 (Poland) | 660 | $0.180 | 1,911.7 | $344.11 | 1,261.7 |

**Savings vs us-east-1 baseline:**
- Moving interruptible batch jobs to **europe-north1** saves **669 kg CO2e** (92% reduction)
- Moving to **us-east-wa** saves **$124/month** in electricity (54% reduction)
- Best balanced region: **us-east-wa** (lowest normalized cost + carbon score)

### Trade-off: Latency vs. Carbon

| Region | CO2 reduction | Latency impact | Suitable for |
|---|---|---|---|
| europe-north1 | 92% | +100-200 ms for US users | Batch training, offline eval only |
| us-east-wa | 76% | <10 ms extra from US east coast | Spot inference, latency SLA >= 50 ms |
| us-west-2 | 68% | 40-70 ms from US east coast | West-coast serving, moderate SLA |

**Which region is 'optimal' depends on company priority:**
- Carbon-first: europe-north1 (near-zero grid intensity, Norwegian hydro)
- Cost-first: us-east-wa (lowest $/kWh at $0.055, 76% cleaner than us-east-1)
- Latency-first with sustainability: us-west-2 (Oregon hydro, $0.07/kWh, 68% cleaner)

**Connection between carbon and actual cost:**
Grid carbon intensity directly correlates with marginal electricity price in many
markets. Poland (660 gCO2/kWh, coal-heavy) costs $0.18/kWh -- 3x the price of
Norwegian hydro ($0.09/kWh). This means carbon-aware scheduling is ALSO cost-aware
scheduling. Companies that optimize for carbon automatically reduce electricity spend.

---

## Prioritized Action Plan

| Priority | Action | Monthly Savings | Effort | ROI |
|---|---|---|---|---|
| 1 (Highest) | Switch training/dev jobs to spot instances | $4,831 | Low (config change) | Very high |
| 2 | Switch non-interruptible inference to 1yr reserved | $3,876 | Medium (commit) | High |
| 3 | Enable model cascade: route 80% traffic to small model | $1,212 | Medium (router logic) | High |
| 4 | Fix idle gpu-h100-5 (auto-shutdown) | $600 | Low (scheduler) | Very high |
| 5 | Right-size gpu-h100-4 / gpu-a10g-1 (util lies) | $655 | Medium (migration) | High |
| 6 | Move batch jobs to europe-north1 / us-east-wa | $124 elec + 669 kg CO2 | Low (region config) | Sustainability |
| 7 | Monitor reasoning traffic; enforce cap if > 10% | Up to 20% cost reduction | Low (routing rule) | Conditional |

**Why this order?**
Purchasing tier changes (P1, P2) have the highest absolute savings and require only
configuration -- no code changes, no model retraining. The compute infrastructure
already supports checkpointing for training jobs. Cascade routing (P3) requires a
classifier or rule-based router, moderate engineering work but high sustained return.
Idle GPU shutdown (P4) is the easiest quick win at $600/month for a one-day fix.

---

## Cost Allocation & Chargeback Readiness

| Team | Daily Spend |
|---|---|
| assistant | $2.59 |
| search | $2.49 |
| eval | $1.79 |
| rag | $1.60 |

- **Tag coverage: 92%** (threshold: 80%) -- chargeback gate is **OPEN**
- ~8% of token_usage rows have empty `project` tag (eval team intermittent tagging)
- FOCUS-format export available at `outputs/focus_export.csv` for multi-cloud billing

**Why 80% coverage threshold?**
Below 80%, untagged cost is spread across teams arbitrarily -- you are billing noise.
A team with 30% of untagged traffic unfairly absorbs charges from another team's
workloads. The 80% gate ensures attribution error is below 20%, acceptable for showback.
For financial chargeback (actual P&L impact), target >= 95% coverage.

---

_Figures are June-2026 illustrative snapshots; re-baseline prices before acting._
_Report generated by NimbusAI GPU FinOps Lab 25 pipeline._