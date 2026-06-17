# Pearl EAGLE real-prompt mining benchmark

Date: 2026-06-17

This note captures the benchmark evidence for Pearl-specific EAGLE speculative
decoding as a mining enabler. The primary comparator is **vanilla no-mining**,
not vanilla mining. Vanilla mining is only a diagnostic that shows the old mining
tax.

The conservative C=32 PR result is that **EAGLE + mining beats vanilla
no-mining on real prompts from ~128 tokens upward**. Under that canonical serving
load, mining becomes *serving-throughput-free* from ~128 tokens onward: the
mining-capable EAGLE path is faster than the no-mining baseline, so the mined
work is not paid for with lower serving throughput. The throughput-positive
bound for enabling the mining-capable path drops from the old ~1024-token
heuristic to ~128 tokens: about an 8x reduction, i.e. the practical "~10x lower
bound" story.

The very-short `<128` bucket is useful but noisier: one run had K2 positive, the
canonical C=32 repeats had K2/K3 negative. Treat `<128` as experimental
follow-up, not the headline claim. Later C sweeps show the bound is
concurrency-sensitive, so production policy should be validated at the target
serving concurrency instead of treated as universal. Across C1/C4/C16/C32, the
best EAGLE+mining configuration is serving-throughput-free or better for the
policy-relevant buckets. At C64, the result is mixed against vanilla no-mining
(positive at ~128–512, roughly near/touching no-mining for the longest bucket,
weaker at 513–1024), but EAGLE remains meaningfully better than vanilla mining
for long prompts.

## Novelty: EAGLE changes where noise is paid

The important mechanism is not merely "lower a prompt threshold". The novelty is
that EAGLE changes **where the noisy mining work is paid** in the serving path.

Vanilla mining pays noisy GEMM as a pure tax in the normal target-model path. In
contrast, Pearl-specific EAGLE puts the mining-capable target work behind the
speculative decode / verification loop: a target pass can validate multiple
drafted tokens, and accepted draft tokens amortize the noisy target work. That is
why `EAGLE + mining` can beat `vanilla no-mining` instead of just beating
`vanilla mining`.

This is **not** a prefill trick and not a change to the GEMM shape threshold. The
existing noisy-GEMM gate remains at `min_m/min_n/min_k = 1024`. The benchmark
shows that, when the noisy target path is used inside speculative decoding, the
noise is paid in a different part of the generation loop and becomes hidden/net
positive in the validated regimes.

## Why real prompts are required

Random-token throughput benchmarks are useful stress tests, but they are a poor
proxy for EAGLE. EAGLE is trained to predict the verifier's natural next-token
distribution. Random prompts/continuations are close to adversarial for the
acceptance metric and materially understate the useful win.

Use real prompts for PR/performance evidence, and treat random-token results as
stress-only. The primary metric in this note is request throughput (`req/s`) at
fixed `max_tokens=96`, greedy decoding, and the stated client concurrency.

## Artifact under test

Scratch Pearl EAGLE3 checkpoint trained on the 200k seq4096 hidden-state run:

```text
/workspace/pearl_eagle3_train_200k_seq4096/output/checkpoints_eagle3_scratch/3
```

The lower-learning-rate checkpoint was also evaluated and is weaker:

```text
/workspace/pearl_eagle3_train_200k_seq4096/output/checkpoints_eagle3_lr5e5/4
```

## Held-out acceptance eval

Dataset: `RedHatAI/speculator_benchmarks`, 9 subsets × 16 prompts, `max_tokens=96`.

| checkpoint | K | acceptance length | draft-token accept | pos0 | pos1 |
|---|---:|---:|---:|---:|---:|
| scratch | 2 | **1.750** | **37.5%** | **49.6%** | **25.4%** |
| lr5e5 | 2 | 1.355 | 17.7% | 29.2% | 6.2% |

The scratch checkpoint is the one worth using for throughput and policy work.

Scratch K=2 by subset:

| subset | acceptance length | draft-token accept |
|---|---:|---:|
| qa | 1.60 | 30.1% |
| writing | 1.77 | 38.7% |
| summarization | 1.90 | 45.2% |
| math_reasoning | 1.90 | 45.0% |
| HumanEval | 1.93 | 46.7% |
| translation | 1.42 | 21.0% |
| question | 1.77 | 38.6% |
| rag | 1.76 | 38.0% |
| tool_call | 1.82 | 40.9% |

## Real-prompt throughput, mixed prompt lengths

Dataset: `RedHatAI/speculator_benchmarks`, 9 subsets × 24 prompts, `max_tokens=96`, concurrency 32.

| case | req/s | output tok/s | total tok/s | acceptance length | draft-token accept |
|---|---:|---:|---:|---:|---:|
| vanilla no-mining | 12.90 | 1238.6 | 4544.7 | — | — |
| vanilla mining | 11.71 | 1124.1 | 4124.4 | — | — |
| scratch EAGLE K2 mining | 14.66 | 1407.6 | 5164.5 | 1.742 | 37.1% |
| scratch EAGLE K3 mining | **14.80** | **1421.1** | **5214.1** | 1.831 | 27.7% |
| scratch EAGLE K4 mining | 14.29 | 1372.3 | 5035.1 | 1.892 | 22.3% |

Relative to vanilla no-mining:

- EAGLE K2 mining: +13.6% req/s.
- EAGLE K3 mining: +14.7% req/s.
- EAGLE K4 mining: +10.8% req/s.

Relative to vanilla mining:

- EAGLE K2 mining: +25.2% req/s.
- EAGLE K3 mining: +26.4% req/s.
- EAGLE K4 mining: +22.1% req/s.

## Length-bucket throughput

Dataset: `RedHatAI/speculator_benchmarks`, prompts bucketed by Pearl tokenizer
input length, `max_tokens=96`, concurrency 32. Buckets used up to 64 prompts each
(the 1025–2048 bucket had 39 prompts available).

| avg prompt len | vanilla no-mining | vanilla mining | EAGLE K2 mining | EAGLE K3 mining |
|---:|---:|---:|---:|---:|
| 44 | 12.83 | 12.76 | **14.07** | 12.88 |
| 182 | 12.82 | 12.55 | 14.58 | **16.15** |
| 351 | 12.26 | 11.86 | **14.67** | 14.61 |
| 699 | 11.75 | 10.98 | 12.75 | **13.10** |
| 1331 | 7.42 | 6.78 | 8.91 | **9.75** |

Relative to vanilla no-mining:

| avg prompt len | EAGLE K2 mining | EAGLE K3 mining |
|---:|---:|---:|
| 44 | **+9.6%** | +0.4% |
| 182 | +13.7% | **+25.9%** |
| 351 | **+19.6%** | +19.1% |
| 699 | +8.5% | **+11.4%** |
| 1331 | +20.1% | **+31.4%** |

Acceptance by bucket:

| avg prompt len | K2 acceptance length | K2 draft accept | K3 acceptance length | K3 draft accept |
|---:|---:|---:|---:|---:|
| 44 | 1.697 | 34.8% | 1.794 | 26.5% |
| 182 | 1.830 | 41.5% | 1.954 | 31.8% |
| 351 | 1.863 | 43.2% | 2.004 | 33.5% |
| 699 | 1.945 | 47.3% | 2.101 | 36.7% |
| 1331 | 1.826 | 41.3% | 1.960 | 32.0% |

### Canonical rerun R1

The first canonical rerun used the harness in this repo with the same buckets,
`max_tokens=96`, greedy decoding, concurrency 32, and the scratch checkpoint. It
confirms the main result from ~128 tokens upward, while showing that `<128` is
noisy and should not be the headline claim yet.

| avg prompt len | vanilla no-mining | vanilla mining | EAGLE K2 mining | EAGLE K3 mining |
|---:|---:|---:|---:|---:|
| 49.8 | 13.03 | 12.80 (-1.8%) | 12.41 (-4.7%) | 11.96 (-8.2%) |
| 186.6 | 12.82 | 13.09 (+2.1%) | 14.05 (+9.6%) | 15.64 (+22.0%) |
| 350.0 | 12.32 | 12.09 (-1.9%) | 14.22 (+15.4%) | 13.17 (+6.9%) |
| 708.3 | 11.68 | 11.02 (-5.6%) | 12.29 (+5.3%) | 12.25 (+4.9%) |
| 1331.3 | 7.38 | 6.92 (-6.2%) | 9.01 (+22.2%) | 8.88 (+20.4%) |

R1 supports the conservative policy bound:

```text
prompt_len >= 128: EAGLE + mining beats vanilla no-mining
prompt_len < 128: keep experimental / validate more before defaulting
```

### Canonical rerun aggregate: R1–R3

Raw outputs on AWS:

```text
/workspace/real_prompt_length_bucket_canonical_r1/combined_summary.csv
/workspace/real_prompt_length_bucket_canonical_r2/combined_summary.csv
/workspace/real_prompt_length_bucket_canonical_r3/combined_summary.csv
```

Three canonical reruns make the conservative bound clearer. The table reports
mean deltas against the primary baseline, `vanilla_no_mining`; ranges in
parentheses are the min/max delta across the three runs.

| avg prompt len | vanilla no-mining req/s | vanilla mining | EAGLE K2 mining | EAGLE K3 mining | best EAGLE |
|---:|---:|---:|---:|---:|---:|
| 49.8 | 13.03 | -2.7% | -4.5% (-7.7..-1.1) | -4.1% (-8.2..-1.0) | -4.1% |
| 186.6 | 12.87 | -2.0% | +21.3% (+9.6..+31.8) | +22.1% (+18.9..+25.3) | +22.1% |
| 350.0 | 12.44 | -4.7% | +14.9% (+12.0..+17.4) | +8.9% (+6.9..+10.5) | +14.9% |
| 708.3 | 11.80 | -6.9% | +4.4% (+1.6..+6.5) | +3.7% (+1.6..+4.9) | +4.4% |
| 1331.3 | 7.45 | -8.7% | +19.2% (+16.8..+22.2) | +21.7% (+20.4..+23.9) | +21.7% |

Mean acceptance across R1–R3 stayed healthy:

| avg prompt len | K2 acceptance length | K2 draft accept | K3 acceptance length | K3 draft accept |
|---:|---:|---:|---:|---:|
| 49.8 | 1.714 | 35.7% | 1.796 | 26.5% |
| 186.6 | 1.854 | 42.7% | 1.999 | 33.3% |
| 350.0 | 1.821 | 41.1% | 1.939 | 31.3% |
| 708.3 | 1.846 | 42.3% | 1.967 | 32.2% |
| 1331.3 | 1.845 | 42.2% | 1.977 | 32.6% |

The robust C=32 claim is therefore:

```text
At the canonical C=32 load, EAGLE lowers the throughput-positive mining bound from ~1024 tokens to ~128 tokens.
```

Equivalently: from ~128 tokens onward, mining is throughput-free in this setup
because `EAGLE + mining` is faster than `vanilla no-mining`. Above ~1024 tokens,
the result is not merely throughput-free: the best EAGLE+mining case is about
+22% vs no-mining and about +34% vs vanilla mining across the R1–R3 repeats.

The `<128` bucket is not a serving win in these canonical repeats despite good
acceptance, so it should not be part of the headline policy.

### Concurrency + K4 validation sweep

Raw outputs on AWS:

```text
/workspace/real_prompt_length_bucket_c16_k4/combined_summary.csv
/workspace/real_prompt_length_bucket_c32_k4/combined_summary.csv
/workspace/real_prompt_length_bucket_c64_k4/combined_summary.csv
```

A follow-up sweep checked C=16/32/64 and added K4. This was a validation sweep,
not the primary PR evidence. It shows two useful things:

1. At C16/C32/C64, K4 is not the best choice in any tested bucket.
2. The throughput win is concurrency-sensitive; C=64 needs separate production
   validation before using the same ~128-bound policy.

| C | avg prompt len | vanilla no-mining req/s | vanilla mining | K2 mining | K3 mining | K4 mining | best EAGLE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 49.8 | 7.40 | +0.1% | +1.8% | +6.9% | +3.5% | K3 +6.9% |
| 16 | 186.6 | 7.47 | -4.8% | +23.8% | +19.9% | +20.5% | K2 +23.8% |
| 16 | 350.0 | 7.31 | -4.7% | +15.3% | +13.5% | +13.0% | K2 +15.3% |
| 16 | 708.3 | 7.12 | -8.6% | +8.0% | +13.7% | +10.3% | K3 +13.7% |
| 16 | 1331.3 | 5.61 | -13.9% | +1.0% | +3.6% | +3.3% | K3 +3.6% |
| 32 | 49.8 | 12.93 | -1.4% | -5.8% | -4.1% | -4.9% | K3 -4.1% |
| 32 | 186.6 | 12.64 | -1.1% | +22.2% | +30.3% | +14.3% | K3 +30.3% |
| 32 | 350.0 | 12.19 | -5.3% | +14.7% | +13.5% | +11.4% | K2 +14.7% |
| 32 | 708.3 | 11.65 | -7.6% | +2.4% | +7.0% | -3.7% | K3 +7.0% |
| 32 | 1331.3 | 7.48 | -9.8% | +17.5% | +24.0% | +14.5% | K3 +24.0% |
| 64 | 49.8 | 23.90 | -6.4% | -19.6% | -14.3% | -22.9% | K3 -14.3% |
| 64 | 186.6 | 24.96 | -12.5% | +5.3% | +1.9% | -0.1% | K2 +5.3% |
| 64 | 350.0 | 23.24 | -13.7% | +6.2% | +5.6% | +0.1% | K2 +6.2% |
| 64 | 708.3 | 20.21 | -11.8% | -10.3% | -8.7% | -10.9% | K3 -8.7% |
| 64 | 1331.3 | 11.75 | -13.6% | -6.1% | -2.9% | -9.1% | K3 -2.9% |

Interpretation of the C16/C32/C64 sweep:

- K4 increases acceptance length but loses enough throughput that it should not
  be the default at C16+.
- C16 is under-saturated enough that EAGLE helps broadly.
- C32 is the stable canonical point for the ~128-bound claim.
- C64 is a different serving regime: vanilla no-mining is much faster. In this
  quick sweep, EAGLE stays positive vs vanilla no-mining in the ~128–512 buckets,
  is close to no-mining in the longest bucket, and is weaker in the 513–1024
  bucket. Do not generalize the C32 policy to C64 without a larger/interleaved
  run at the target production concurrency.
- At C64 long prompts, EAGLE still improves over **vanilla mining**: K3 is +3.5%
  vs vanilla mining in the 513–1024 bucket and +12.3% vs vanilla mining in the
  1025–2048 bucket, even though those buckets are -8.7% and -2.9% vs vanilla
  no-mining respectively. This distinction matters: at high concurrency EAGLE is
  no longer always faster than no-mining, but it still hides a meaningful part of
  the mining tax.

### Very-low-concurrency sweep: C1/C4

Raw outputs on AWS:

```text
/workspace/real_prompt_length_bucket_c1_k4/combined_summary.csv
/workspace/real_prompt_length_bucket_c4_k4/combined_summary.csv
```

Very low concurrency is a different regime. Here EAGLE improves every tested
bucket, including `<128`, and deeper drafting becomes useful. K4 wins at C4;
K3 wins at C1.

| C | avg prompt len | vanilla no-mining req/s | vanilla mining | K2 mining | K3 mining | K4 mining | best EAGLE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49.8 | 0.47 | -0.8% | +37.4% | +43.4% | +39.8% | K3 +43.4% |
| 1 | 186.6 | 0.46 | +2.0% | +50.6% | +62.3% | +54.5% | K3 +62.3% |
| 1 | 350.0 | 0.46 | +3.0% | +49.4% | +61.8% | +52.8% | K3 +61.8% |
| 1 | 708.3 | 0.47 | -0.2% | +51.5% | +63.3% | +57.2% | K3 +63.3% |
| 1 | 1331.3 | 0.47 | -3.8% | +44.4% | +53.7% | +47.5% | K3 +53.7% |
| 4 | 49.8 | 1.89 | -4.9% | +30.4% | +28.3% | +32.7% | K4 +32.7% |
| 4 | 186.6 | 1.90 | -5.0% | +37.3% | +39.3% | +43.8% | K4 +43.8% |
| 4 | 350.0 | 1.89 | -5.2% | +42.0% | +40.7% | +44.5% | K4 +44.5% |
| 4 | 708.3 | 1.90 | -7.4% | +43.9% | +42.2% | +49.0% | K4 +49.0% |
| 4 | 1331.3 | 1.82 | -10.4% | +26.7% | +26.8% | +29.9% | K4 +29.9% |

Interpretation of the low-C sweep:

- If serving is deliberately low-concurrency / latency-oriented, EAGLE+mining can
  be beneficial even below 128 tokens.
- K4 should not be globally dismissed: it wins at C4, but loses at C1 and at
  C16+.
- The production policy should therefore use concurrency/load as part of the
  rollout guard, not only prompt length. Up through C32, the evidence supports
  the "free-or-better than no-mining" framing for the policy-relevant buckets;
  at C64, use the more careful framing: near no-mining in the longest bucket,
  still better than vanilla mining, and mixed elsewhere.

### Mining-path validation

To verify the benchmark was actually mining, a tiny instrumented validation run
wrapped `pearl_gemm_noisy()` in the vLLM worker process and traced calls while
serving one long real prompt with mining enabled. Raw trace:

```text
/workspace/noisy_gemm_trace.jsonl
```

The trace recorded 640 calls into the actual noisy-GEMM mining function across
2 EngineCore worker PIDs. Example shapes:

```text
m=1927, n=4096,  k=4096
m=1927, n=28672, k=4096
m=1927, n=6144,  k=4096
```

`submit_block=false` in the trace is expected: the benchmark sets
`MINER_SKIP_BLOCK_SUBMISSION=1` so it exercises noisy GEMM / noise generation
without submitting found blocks.

## Interpretation

For PR and product decisions, compare against **vanilla no-mining**. That is the
real baseline: if EAGLE+mining beats it, we are not merely improving the mining
path; we are getting mined work as part of a faster serving path. In that
throughput sense, mining is free and comes with a throughput gift. `vanilla_mining`
should remain in the table to show the old problem: mining by itself is slower,
while EAGLE hides that tax and then beats the no-mining path.

The old policy question was:

```text
At what prompt length does mining become worth the throughput cost?
```

The real-prompt EAGLE result changes the policy question because it changes where
the noisy work is paid. This is not "mine in prefill sooner". It is "run the
noisy target path inside a speculative decode/verification loop where accepted
draft tokens amortize the noise". Above ~128 tokens, these runs show:

```text
vanilla mining <= vanilla no-mining
EAGLE + mining > vanilla no-mining
```

So EAGLE is doing three things at once:

1. It changes where the noisy GEMM tax is paid: not as a naked vanilla-mining
   cost, and not as a prefill trick, but inside the speculative decode /
   verification path.
2. It hides the mining overhead: the serving path no longer pays the vanilla
   mining throughput tax from ~128 tokens onward at C32.
3. It provides a serving-throughput win while mining: the mined path is faster
   than vanilla no-mining, and much faster than vanilla mining in the long-prompt
   buckets.

The conservative C=32 conclusion is not "prove no threshold forever". It is:

```text
old throughput-positive mining bound: ~1024 prompt tokens
new throughput-positive EAGLE+mining bound at C=32: ~128 prompt tokens
```

That is the practical ~10x bound reduction for the canonical load. The `<128`
bucket should remain an experimental optimization target until
repeated/interleaved runs make it stable. The C64 sweep also shows the policy
must be validated at the target serving concurrency; if future evidence shows
`<128` and high-concurrency long prompts are reliably positive vs no-mining too,
the policy can be strengthened from "~128 bound at C=32" to a broader dynamic
policy. Existing low-C data already supports a different latency-oriented policy:
K3 at C1 and K4 at C4.

A conservative policy from this benchmark is:

```text
if Pearl-specific EAGLE is available and acceptance is healthy:
    if prompt_len >= 128:
        enable mining with EAGLE
        prefer K3, with K2 as a safer fallback
    else:
        keep the existing no-mining / old policy unless explicitly experimenting
else:
    fall back to the existing mining threshold policy
```

A bolder experimental policy for further validation is:

```text
low-concurrency / latency mode:
    C1: prefer K3 + mining
    C4: prefer K4 + mining
canonical C32 throughput mode:
    prompt_len < 128: keep experimental / default no-mining
    prompt_len >= 128: prefer K2/K3 + mining
high-concurrency C64 mode:
    validate at production load; compare against both no-mining and mining baselines
    expect mixed results vs no-mining, but still better than vanilla mining in long buckets
```

The PR should lead with the conservative C=32 claim: mining becomes
serving-throughput-free from ~128 tokens onward, lowering the throughput-positive
bound from ~1024 to ~128, with large wins over vanilla mining in the long-prompt
buckets. It should also mention the concurrency sweep: C1/C4/C16/C32 are
free-or-better for the relevant buckets; C64 is approximately no-mining-equivalent
in the longest bucket and still better than vanilla mining, but mixed enough to
need separate production-load validation. Do not frame it as an unconditional
no-threshold or all-concurrency claim.

## Methodology and implementation notes

This experiment measures the **net** serving-throughput effect of running
`EAGLE + mining` versus `vanilla no-mining`. It does not decompose EAGLE-only
speedup from mining-only overhead. `vanilla_mining` remains in the tables to show
the non-speculative mining tax, but the headline claim is the net production
comparison against no-mining.

The benchmark sets:

```text
MINER_SKIP_BLOCK_SUBMISSION=1
MINER_NO_GATEWAY=1
```

so it measures the noisy-GEMM / noise-generation compute path without gateway or
block-submission/network overhead. The results should therefore be read as a
compute-path throughput benchmark, not a full end-to-end mining economics
benchmark.

The separate mining-path validation run used the same harness shape plus a
temporary process-local `sitecustomize` wrapper to trace `pearl_gemm_noisy()`.
That trace proves the noisy-GEMM path is reachable and exercised under the
benchmark mining environment; it is not an always-on production instrumentation
and has been removed from the AWS workspace after collecting the trace.

The benchmark does **not** require changing the existing noisy-GEMM shape
threshold in the vLLM miner. The current plugin still gates noisy GEMM with:

```text
miner/vllm-miner/src/vllm_miner/config.yaml
min_m = 1024
min_n = 1024
min_k = 1024
```

and the runtime check remains:

```python
config.should_use_noisy_gemm(m, n, k) and not config.settings.no_mining
```

So the evidence is not "lower the internal GEMM threshold and hope mining pays".
It is stronger: with the existing GEMM threshold, Pearl-specific EAGLE changes
where the noisy work lands in the generation loop. Running the mining-capable
server with EAGLE hides the mining tax and improves throughput on real prompts
from ~128 tokens upward.

There is no in-process prompt-length switch in this plugin today. A production
`prompt_len >= 128` rollout should therefore be treated as a serving/router
policy, or as a separate deployment mode, unless a future change adds dynamic
per-request routing.

## Canonical benchmark harness

The repository includes a reusable harness:

```text
scripts/benchmarks/real_prompt_spec_decode_bench.py
```

Example mixed-prompt run:

```bash
python scripts/benchmarks/real_prompt_spec_decode_bench.py \
  --model pearl-ai/Llama-3.1-8B-Instruct-pearl \
  --eagle-checkpoint /workspace/pearl_eagle3_train_200k_seq4096/output/checkpoints_eagle3_scratch/3 \
  --vllm-bin /workspace/pearl-build/.venv/bin/vllm \
  --output-dir /workspace/aws_real_prompt_throughput_bench \
  --gpu 0 \
  --samples-per-subset 24 \
  --max-tokens 96 \
  --concurrency 32 \
  --cases vanilla_no_mining,vanilla_mining,eagle_k2_mining,eagle_k3_mining,eagle_k4_mining
```

Example length-bucket run:

```bash
python scripts/benchmarks/real_prompt_spec_decode_bench.py \
  --model pearl-ai/Llama-3.1-8B-Instruct-pearl \
  --eagle-checkpoint /workspace/pearl_eagle3_train_200k_seq4096/output/checkpoints_eagle3_scratch/3 \
  --vllm-bin /workspace/pearl-build/.venv/bin/vllm \
  --output-dir /workspace/aws_real_prompt_length_bucket_bench \
  --gpu 0 \
  --bucket-lengths 0:128,129:256,257:512,513:1024,1025:2048 \
  --samples-per-bucket 64 \
  --max-tokens 96 \
  --concurrency 32 \
  --cases vanilla_no_mining,vanilla_mining,eagle_k2_mining,eagle_k3_mining
```

The harness writes:

- `prompt_manifest.json`
- per-case `summary.json`
- per-case `server.log`
- `combined_summary.json`
- `combined_summary.csv`

The combined outputs include explicit deltas against `vanilla_no_mining`:

- `requests_vs_vanilla_no_mining_pct`
- `output_tokens_vs_vanilla_no_mining_pct`
- `total_tokens_vs_vanilla_no_mining_pct`

Those deltas are the headline PR numbers. Deltas against `vanilla_mining` are
secondary diagnostics, not the main claim.

## PR checklist

Before changing serving defaults, use this checklist:

1. Re-run the length-bucket benchmark at least 2–3 times or interleaved to reduce
   warmup/order effects.
2. Keep random-token results out of the primary claim; mention them only as a
   stress test.
3. Add policy behind an experimental flag first:
   - conservative headline: enable EAGLE + mining for `prompt_len >= 128` at
     the validated serving concurrency;
   - experimental follow-up: low-C K3/K4 routing, K2/K3 at C32, and separate
     C64 / production-concurrency validation.
4. Track live acceptance counters:
   - acceptance length,
   - accepted/drafted token rate,
   - per-position acceptance.
5. Fall back to the old threshold if acceptance drops below the tested healthy
   band on production traffic.
