# GPU FinOps Lab 25 — Short Analysis Report

**Student:** Nguyen Quang Huy  
**MSSV:** 2A202602461  
**Repo:** K4-L3B-Day25-NguyenQuangHuy-2A202602461-Track2-GpuFinOps  
**Date:** June 2026 (snapshot)

---

## 1. Baseline vs. Optimized

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| Monthly GPU spend | $27,133 | ~$15,979 | **-41.2%** |
| Inference $/1M-token | $6.488 | $1.126 | **-82.6%** |
| Purchasing cost (monthly) | $25,667 | $16,960 | **-33.9%** |
| Total projected savings | — | $11,154 | **~41%** |

The baseline assumes every inference request runs on the **large model** (no routing), no prompt caching, no batch API, and all GPU capacity purchased at full on-demand rates. The optimized scenario applies cascade routing (small/large), prompt caching, batch API, optimal purchasing tier, right-sizing flagged GPUs, and eliminating idle waste.

---

## 2. Analysis by Savings Lever

| Lever | Savings (USD/month) | Share |
|---|---|---|
| Inference (cascade/cache/batch) | ~$1,212 | ~10.9% |
| Purchasing (spot/reserved) | ~$10,040 | ~90.1% |
| Right-size util-lies | ~$655 | ~5.9% |
| Kill idle GPUs | ~$600 | ~5.4% |

**Purchasing strategy is the dominant lever** (~90% of savings). Switching training workloads to spot and inference workloads to reserved contracts captures the bulk of the savings. This is because GPU compute is expensive at full on-demand rates (~$2–$8/hr for H100/A100), and the volume of GPU-hours (training jobs run 14–30 days with multiple GPUs) means even a 40–60% discount translates to thousands of dollars monthly.

**Inference cost levers** (cascade + cache + batch) are the most *impactful per token*: cutting $/1M-token from $6.49 → $1.13 (82.6% reduction). However, since daily inference spend is ~$48.87 vs. ~$25k/month in GPU provisioning, the absolute dollar savings from inference optimization alone are smaller. Nonetheless, as request volume scales, this becomes the most important lever.

**Right-sizing GPU-Util lies** captures ~$655/month. GPUs flagged as "lies" (`gpu-h100-4`, `gpu-a10g-1`) show high nvidia-smi utilization (98%, 97%) but low MFU (0.19, 0.27), meaning you pay for a full H100/A10G while getting a fraction of the compute. Downsizing to A100/L4 respectively saves the price delta over 24h × 30 days.

**Idle GPU elimination** saves $600/month from gpu-h100-5 sitting at <10% utilization for 8 hours/day. A simple auto-shutdown policy would recover this immediately.

---

## 3. GPU-Util Lie Analysis

**Flagged GPUs:** `gpu-h100-4` (H100, util=98.2%, MFU=0.194) and `gpu-a10g-1` (A10G, util=96.9%, MFU=0.268)

**The lie explained:**  
`nvidia-smi` reports "GPU Utilization %" as a *clock-active fraction* — the fraction of time the GPU's SM clocks were running. This does NOT measure how efficiently those clocks were used. An H100 doing memory-bandwidth-limited decode will spin its clocks reading weights from HBM while the actual tensor cores sit idle. The result: nvidia-smi shows ~98% but MFU is only ~0.19 — you are paying the full H100-hour ($2.00–$3.00/hr) for 19% of the FLOPs you rented.

**Financial impact:**  
- `gpu-h100-4` at H100 on-demand (~$2.00/hr) with only 19% MFU → effectively paying ~$10.50/effective-TFLOP-hr vs. the ~$2.00 you could get at 50% MFU  
- Over 30 days: ~$655 in right-sizing savings by moving workloads to A100 which is better suited for memory-bound inference at lower cost/hr

**Root cause:** These GPUs are likely running memory-bound LLM decode with batch size 1, where the arithmetic intensity (~1–2 FLOP/byte) is far below the H100's ridge point (~295 FLOP/byte). The tensor cores cannot be fed fast enough by HBM bandwidth alone.

**Correct metric:** Use **MBU (Memory Bandwidth Utilization)** for decode and **MFU** for prefill/training. Target MBU ≥ 0.60 for H100 batch-1 decode.

---

## 4. Extension Results

### Extension 1: Interruption-Rate-Aware `recommend_tier()`

**Changes made to [`finops/pricing.py`](../finops/pricing.py):**
- Added `GPU_INTERRUPTION_RATE` dict mapping GPU types to per-hour interruption probability (H200: 12%, H100: 8%, A100: 5%, A10G: 3%, L4: 2%)
- Added `RESERVED_DISCOUNT_1YR = 0.30` alongside existing 3yr = 0.45
- Extended `recommend_tier()` to accept `gpu_type` and `total_days` parameters
- Policy: for high-interruption GPUs (≥8%) on long jobs (≥180 days), compare effective spot ROI vs. reserved; if reserved beats spot after expected rework, recommend reserved

**Results before vs. after:**

| Job | Old Tier | New Tier | Reason |
|---|---|---|---|
| job-infer-chat | reserved (3yr) | reserved-1yr | 30-day job → 1yr commitment more appropriate |
| job-infer-rag | reserved (3yr) | reserved-1yr | 30-day job → 1yr commitment |
| job-infer-search | reserved (3yr) | reserved-1yr | 30-day job → 1yr commitment |
| All training/short jobs | spot | spot | unchanged |

**Impact:** Monthly savings decreased from 39.1% to 33.9% because 1yr reserved discount (30%) is less aggressive than 3yr (45%). However, the model is **more realistic** — committing to a 3yr reserved instance for a 30-day job is not financially rational. The 1yr tier is the appropriate commitment for jobs running under one year.

**Key insight:** GPU type matters for spot decisions. H100 has 8% per-hour interruption probability vs. 2% for L4. For a 14-day H100 training run, expected interruptions = 14×24×0.08 ≈ 26.9 events — each costing 30 min of rework. That's ~13.5 hours of rework on top of checkpoint overhead. Spot still wins for training (large discount), but for long-running inference (>180 days on H100), reserved becomes competitive.

### Extension 2: `cache_is_worth_it()` Economic Gate

**Changes made to [`finops/pricing.py`](../finops/pricing.py) and [`missions/m2_inference_levers.py`](../missions/m2_inference_levers.py):**
- Added `cache_is_worth_it()` function that computes break-even reads/write and net savings
- M2 now estimates actual reads_per_write from the token_usage.csv data before applying cache discount
- Cache is only applied if `reads_per_write > break_even_reads`

**Results from the synthetic dataset:**

| Parameter | Value |
|---|---|
| Median prefix tokens | 412 tokens |
| Estimated reads/write | 150.0 |
| Break-even reads/write | 1.17 |
| Cache worth it? | **True** (savings 89.3% per cycle) |
| Cache applied? | Yes |

The cache is highly beneficial in this dataset: with 150 reads per cached prefix and a break-even of only 1.17 reads, caching saves 89.3% of re-read costs. The write overhead (5%) is completely overwhelmed by the savings.

**Counterfactual — few reads scenario:**  
If reads_per_write were 1.0 (each prefix written once and read once), caching would NOT be worth it:
- Without cache: 1 read × full price = $3.00/M tokens
- With cache: write ($3.15) + 1 read ($0.30) = $3.45/M tokens → **15% MORE expensive**

This validates the gate: cache write overhead must be recovered through enough reads. Below break-even (~1.17 reads), the system correctly disables caching to avoid cost increase.

**Practical rule:** Always measure your prompt cache hit rate and reads-per-unique-prefix before enabling caching. Short-context RAG with unique queries every time has reads/write ≈ 1 → caching loses money. Long system prompts shared across thousands of user turns → reads/write >> 100 → caching wins decisively.

---

## 5. Recommendations for NimbusAI (3 First Actions as FinOps Lead)

### Action 1: Fix GPU Purchasing Immediately (Impact: ~$10K/month)
Switch all interruptible training jobs to **spot** with checkpoint intervals ≤30 min. Switch all steady inference workloads to **reserved-1yr** (don't commit to 3yr until usage patterns stabilize). Use the enhanced `recommend_tier()` with actual GPU type interruption rates to avoid over-recommending spot for high-demand H100s on long inference jobs. This single action captures 90% of the achievable savings.

### Action 2: Deploy Inference Cascade Routing (Impact: $6.49 → $1.13/1M-token)
Implement a router that sends simple requests (greeting, classification, short Q&A) to the small model ($0.20/$0.40 per M) and only escalates complex reasoning to the large model ($3.00/$15.00 per M). Combined with prompt caching for high-reuse prefixes (verify reads/write > 1.17 first) and batch API for async workloads, this reduces per-token cost by 83%. At scale, this is the highest-leverage optimization because it compounds with request volume.

### Action 3: Tag Everything Before Month-End, Then Enforce Chargeback (Impact: Accountability)
Current tag coverage is 92% — above the 80% chargeback gate. Immediately: (a) fix the 8% untagged requests (likely dev/sandbox traffic), (b) share the showback dashboard with team leads (the `eval` team is spending $1.79/day which may surprise them), (c) set per-team monthly budgets in the FOCUS export. Chargeback creates behavioral incentives: teams self-optimize when they see their own GPU bill. Without this, the savings from Actions 1 and 2 will erode as teams provision capacity without accountability.

---

## 6. Sustainability Notes

- **Energy per query:** 0.24 Wh (median 800-token request)
- **Carbon per query:** 0.091 gCO₂e (at us-east-1, 380 gCO₂/kWh)
- **Best region:** `europe-north1` (Norway hydro, 30 gCO₂/kWh) — **12.7× cleaner** than us-east-1
- **Reasoning traffic:** Reasoning queries consume ~80× the energy of standard queries. Even small amounts of reasoning traffic dominate the energy budget. Consider a reasoning budget cap (e.g., ≤10% of requests) routed only when confidence score from the small model is below threshold.

---

_All figures are June-2026 snapshot data generated with seed=25. Re-baseline before any production deployment._
