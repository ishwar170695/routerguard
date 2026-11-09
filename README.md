# RouterGuard

Post-training compression for sparse Mixture-of-Experts LLMs, in two coupled parts:

1. **Post-hoc load balancing** — fit a per-expert router bias on calibration data so routed tokens spread
   more evenly across experts, then freeze it. No weight updates, no retraining. Adapted from
   DeepSeek's Loss-Free Balancing (arXiv:2408.15664) from *training-time* to a single post-hoc pass.
2. **Redundancy-aware expert pruning** — rank routed experts by how *unique* their outputs are (primary)
   and how balanced their utilization is (secondary), then remove greedily with recomputation.

The claimed gap: prior expert-pruning work ranks experts by utilization or reconstruction error, never
under a *rebalanced* router and never using an explicit redundancy signal. Phrased as "to our
knowledge" — **this claim is not yet verified against the literature** (see [References](#references)).

Target model: Qwen1.5-MoE-A2.7B (14.3B total / 2.7B active, 60 routed + 1 shared expert, top-4).
Development model: OLMoE-1B-7B (6.9B total / 1.3B active, 64 experts, top-8).

---

## Status

Code for Steps 1–4 is complete and verified **on CPU with tiny models built from the real
`transformers` classes**. Only Step 1 has been run on a real checkpoint so far.

| Step | Deliverable | Code | Verified how | Run on real model |
|---|---|---|---|---|
| 1 | env pinning, loaders, splits, load stats | ✅ | 12 preflight checks, 1 notebook | ✅ OLMoE, 8-bit, T4 |
| 2 | patched forward, mask sim, removal | ✅ | **bitwise no-op test, 8/8 configs** | ❌ |
| 3 | balancing, baselines, evaluation | ✅ | 78 tests + notebook-02 preflight | ❌ |
| 4 | similarity, greedy pruning, 2×2 | ✅ | 108 tests + notebook-03 preflight | ❌ |
| 5 | Qwen port + write-up | ❌ | — | — |

Test suite: **108 passed, 3 skipped** (the skips are opt-in real-checkpoint tests).
```bash
python -m pytest tests/ -q                        # 108 passed, 3 skipped
python tools/preflight.py        ~/routerguard    # structure, protocol, notebooks compile
python tools/selfcheck.py                         # stats/run bookkeeping, executed
python tools/preflight_step3.py                   # executes notebook 02's call sequence
python tools/preflight_step4.py                   # executes notebook 03's call sequence
```

### The only real-model number so far (Step 1, OLMoE, 8-bit, T4)

| Quantity | Value |
|---|---|
| calib `gini_mean` (200×1024, wikitext train) | **0.2710** |
| eval `gini_mean` (100×1024, wikitext test) | **0.2833** |
| mean KL(calib ‖ eval) per layer | **0.00816** nats (worst layer 15: 0.0160) |
| gate `gini_skip_balance = 0.20` | above it → balancing has headroom |

The small KL is the load-bearing result for the balancing half: the imbalance is a property of the
*model*, not of the sample, so a calibration-fitted bias has a reason to transfer. It also means the
headroom is modest — see [Interpretation limits](#interpretation-limits).

---

## What still needs doing

1. **Run notebook 02** (~25–35 min on a T4): baseline ppl/MMLU/memory/tok-s, fit the biases, check the
   `max_kl_nats` and `min_balance_transfer_ratio` guards, record the ppl cost of the bias, and produce
   the frequency/random accuracy-vs-compression curve. **Gate:** does the bias transfer, and does it cost
   accuracy?
2. **Run notebook 03** (~30–45 min): similarity matrices, then the 2×2 ablation
   (balance on/off × frequency/redundancy ranking) at 25% and 50%. **Gate:**
   `min_gain_over_frequency_pts = 1.0` on the balanced@50% cell. Negative result is a valid deliverable
   and the protocol commits to reporting it.
3. **Step 5 — notebook 04 (Qwen port).** Only after 2–3, because it needs the winning configuration.
   Timing probe first (2 layers, project total, abort over `abort_if_projected_over_seconds`).
   Re-measure everything at 4-bit and never compare against published bf16 numbers.
4. **`README` results section** with the final table: memory saved (headline), ppl delta, MMLU delta,
   tok/s with variance.
5. **Verify the citations** below against the actual PDFs before any of them enter a write-up.
6. **Decide the paper's framing** before seeing Step 4: with baseline Gini 0.27 the load-balancing win
   will be small, so the defensible claim is about the *interaction* (does a rebalanced router change
   which experts are redundant, and therefore which are prunable), not about large balancing gains.

---

## Layout

```
configs/protocol.yaml          frozen protocol: splits, ratios, seeds, gates, budget caps
routerguard/
  env.py                       version pinning, MoE-structure assertions, env dump
  models.py                    OLMoE (bf16/8-bit/4-bit), Qwen (NF4), gate-precision asserts
  data.py                      deterministic wikitext splits, MMLU few-shot prompts
  stats.py                     load counts, Gini, max-violation, KL, hook collector
  moe.py                       patched forward (selection-only bias), mask sim, real removal
  balance.py                   calibration-time LFB loop, KL guard, transfer ratio
  similarity.py                mean-centered cosine per layer, cached/resumable
  prune.py                     baselines + greedy redundancy removal
  eval.py                      perplexity, MMLU, memory, throughput
  run.py                       run ids, resolved config, git sha, metrics files
notebooks/01..03               the runbook (Step 4's notebook is 03; 04 = Qwen, not written)
tests/                         108 tests, tiny models from real transformers classes
tools/                         preflight + self-check scripts
```

---

## Method, precisely

**Selection-only bias.** Selection uses `topk(s + b)`; the mixing weights still come from the *unbiased*
`s`. A plain logit hook cannot express that — it would bias the mixing weights too — so `moe.py` replaces
each block's `forward` with a mirror of `transformers==4.48.3`'s own loop and changes only the selection
line. This is the single most important implementation detail in the repo.

**Bias update** (per layer, per calibration batch, `c` = cumulative load, `c̄` = its mean):

```
b_i ← b_i + u · sign(c̄ − c_i)
```

frozen afterwards; weights never change.

**Removal semantics** (`protocol.yaml: pruning.removal_semantics: reslice`). The pruned model is the
native one: router rows are sliced, so softmax runs over kept experts only and `top_k` clamps to the kept
count. Mask simulation selects the kept router columns with `index_select` so it sees exactly the same
tensor — which is why mask-sim ≡ real removal.

**Redundancy.** Per layer, every expert runs on the *same* tokens (the hidden states entering that block),
outputs are mean-centered across experts per token, and compared by cosine:

```
z_{i,t} = f_i(x_t) − mean_j f_j(x_t)     S_ij = ⟨z_i, z_j⟩ / (‖z_i‖‖z_j‖)
```

Removal is greedy: score each surviving expert by mean similarity to the others (higher = more redundant),
plus a 0.25-weighted utilization term, remove the worst, **recompute**.

---

## Verification, and what it actually guarantees

| Claim | Evidence |
|---|---|
| No-op patch is bitwise identical to the unpatched model | `torch.equal` in 8/8 configs (fp32/bf16 × `norm_topk_prob` on/off × OLMoE/Qwen) |
| Mask simulation ≡ real removal | selection identical exactly; outputs **bitwise in bf16**, ≤3e-08 abs in fp32 (pruning changes the router GEMM shape) |
| Bias is selection-only, not a weighting change | test also proves the naive "bias everything" variant differs |
| Fused v5 expert layout is rejected with an actionable error | `env.assert_moe_structure` on a synthetic fused model |
| A broken dependency environment fails loudly | `env.assert_env` rejects `huggingface-hub>=1.0`, `datasets>=5`, `transformers 5.x` |

---

## Interpretation limits (read before writing anything up)

- **Modest headroom.** Baseline Gini 0.27 vs the 0.20 skip threshold. If LFB's published
  0.707 → 0.035 holds, our starting point is far more balanced, so absolute balancing gains here will be
  small. Do not present this as a large load-balancing win.
- **Sign-update overshoot.** The update has a real instability boundary: measured on a skewed model,
  `u ≤ 0.01` reduces Gini (0.098 → 0.042) while `u ≥ 0.02` *increases* it (0.098 → 0.18), because
  `u × n_updates` grows past the router logit scale. Protocol `u = 0.001` is safely inside; a run that
  lands outside emits a warning (`BalanceResult.warnings()`).
- **Layer 0 similarity is router-invariant** (its inputs are the embeddings), so recomputing similarity
  under a balanced router only matters for deeper layers; notebook 03 skips it when routing KL < 1e-2.
- **Precision.** Qwen is 4-bit only on a T4, so all Qwen numbers are pruned-vs-unpruned *at 4-bit*. Never
  compare them to published bf16 figures.
- **MMLU here is not lm-eval-harness.** Letters are scored as single tokens from the final position with
  a custom 3-shot prompt template, on a 500-question subset with a fixed seed. Comparable internally,
  not comparable to published MMLU numbers.
- **Two seeds** (42, 123) is thin for variance claims; MMLU at 500 questions has roughly ±2 points of noise.
- **Protocol corrections made before any Step 3/4 results existed** (both recorded in the YAML):
  `budget.max_greedy_steps` 16 → 64 (16 cannot express a 50% removal: 30 removals of 60 experts), and
  `design.utilization_weight: 0.25` added because that method parameter was not pinned at all.

---

## How to run

Colab (GPU) is the intended environment; a T4 is enough.

```python
%cd /content
!git clone https://github.com/ishwar170695/routerguard.git
%cd /content/routerguard
!mkdir -p /content/drive/MyDrive/routerguard-results
!ln -sfn /content/drive/MyDrive/routerguard-results results    # /content is wiped on disconnect
```

Then run `notebooks/01 → 02 → 03` top to bottom. The install cell is mandatory once per fresh runtime:

```bash
pip install 'transformers==4.48.3' 'huggingface-hub>=0.24,<1.0' 'datasets>=2.19,<5' \
            accelerate 'bitsandbytes>=0.43.3' pyyaml matplotlib sentencepiece
```

The `huggingface-hub` and `datasets` bounds are **not optional**: a newer `datasets` upgrades
`huggingface-hub` past 1.0 as a side effect, after which `transformers==4.48.3` cannot be imported at
all. `env.assert_env()` enforces the same range.

`transformers` must stay on 4.48.3 (or a logged fallback): v5 fuses experts into 3D tensors, which breaks
per-expert hooks and bitsandbytes expert quantization (transformers issue #1849).

Every run writes `results/<timestamp>_<sha>_<tag>/` with `config.resolved.yaml`, `git_sha.txt`,
`env.txt`, and `metrics/*.json`.

---

## References

⚠️ All of these are cited **second-hand from the project handoff and have not been checked against the
PDFs**. Verify titles, venues, and every number quoted from them before they enter a document.

| Paper | arXiv | Role in this project | Verified |
|---|---|---|---|
| Loss-Free Balancing (DeepSeek) | 2408.15664 | source of the bias update in `balance.py` | ❌ |
| Expert pruning / trimming (TMLR 2025 per handoff) | 2402.14800 | the prior work whose gap this project claims | ❌ |
| DeepSeekMoE | 2401.06066 | background: fine-grained routed experts + always-on shared expert | ❌ |
| OLMoE | 2409.02060 | development model | ❌ |
| Qwen2 technical report | 2407.10671 | target model family | ❌ |
| Switch Transformer | 2101.03961 | background: auxiliary-loss balancing that LFB replaces | ❌ |

Numbers still needing verification: LFB's reported Gini 0.707 → 0.035, and the TMLR paper's claimed 98%
retention.

---

## Possible improvements

**Method**
- **Replace the sign update with a proportional/adaptive one.** The overshoot boundary documented above
  exists because every step has the same magnitude regardless of how far an expert is from target. A
  gap-proportional or normalized update would remove the `u`-sensitivity entirely and make the method
  robust across models without per-model tuning.
- **Refit the biases after pruning.** Removing experts changes the load distribution, so a second
  calibration pass on the pruned model is nearly free and should recover some of the pruned-model
  accuracy. This directly combines the two components and is the most obvious missing experiment.
- **Layer-wise removal budgets** instead of a uniform ratio. Redundancy is not uniform across depth
  (layer-0 is a special case by construction); allocating the budget by measured redundancy should beat
  uniform 25%/50%.
- **Merge instead of remove.** The similarity matrix already identifies redundant *pairs*; averaging or
  SVD-merging them preserves capacity in a way that deletion cannot, and reuses all of the machinery here.
- **Alternative redundancy metrics**: linear CKA, or reconstruction error of one expert from the others,
  which is closer to what the trimming literature optimizes and would make the comparison fairer.

**Evaluation**
- More seeds and confidence intervals; report MMLU noise explicitly instead of a single point estimate.
- Task breadth: GSM8K, HellaSwag, and a domain-specific calibration set (does a bias fitted on code
  transfer to prose?).
- Measure **saved checkpoint bytes on disk**, not only allocator counters, for the headline memory claim.
- Throughput under batching, not just batch-1 tokens/sec — the practical win may differ.
- A bf16 confirmation run for Qwen on a larger GPU, to separate quantization noise from pruning damage.

**Engineering**
- Real-checkpoint versions of the two bitwise tests are written but opt-in
  (`ROUTERGUARD_REAL_MODELS=1`); they should be part of the default GPU run.
- The notebooks repeat model loading; a small CLI (`python -m routerguard.cli …`) would make the runs
  scriptable, cheaper to re-execute, and diffable in git.
