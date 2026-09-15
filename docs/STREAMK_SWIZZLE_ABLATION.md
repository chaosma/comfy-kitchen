# INT8 GEMM scheduling ablation

## Goal

Measure two scheduling changes independently on the MiniMax H3 INT8 GEMMs:

1. output-tile grouping: `g=1` versus `g=4` versus `g=8`;
2. LeanStreamK K-work splitting: enabled versus disabled while keeping its
   specialized kernel path and output-tile raster fixed.

Then measure the four end-to-end dispatch policies obtained by enabling or
disabling the grouped-swizzle rule and the high-wave StreamK override.

Do not attribute config 0 versus config 13 entirely to K splitting. Config 13's
`ThreadblockSwizzleLeanStreamK` also changes the output-tile raster. This branch
adds config 18 as the matched no-split control for config 13.

## Branch and changes

Use branch `perf/kernel-ablation`, based on `perf/kernel-wins`.

The production changes being evaluated are the only two commits on
`perf/kernel-wins` beyond base commit `dae00a1`:

- `9ba977b`: avoid LeanStreamK when the ordinary output grid already has at
  least 140 waves;
- `0c0f28e`: choose a grouped `g=8` raster when the INT8 weight size is at
  least 75% of device L2.

This experiment branch adds:

- config 18, which uses the same `128x256x64` specialized kernel and tile
  raster as config 13 but forces `sk_tiles=0`, so every output tile belongs to
  one CTA and no cross-CTA partial reduction occurs;
- `COMFY_KITCHEN_DISABLE_GROUPED_SWIZZLE_OVERRIDE=1`, which disables only the
  grouped-swizzle dispatch rule;
- `COMFY_KITCHEN_DISABLE_STREAMK_WAVE_OVERRIDE=1`, which disables only the
  high-wave rule that replaces config 13 with config 0;
- the existing `COMFY_KITCHEN_DISABLE_STREAMK_OVERRIDE=1` remains a legacy
  master switch that disables both rules.
- `COMFY_KITCHEN_TRACE_INT8_ABLATION=1` logs each distinct observed M/N/K and
  the two loaded disable flags once per server process, so the synthetic
  shapes and independent policy settings can be checked against actual GEMMs.

Environment flags are read when Python imports the CUDA backend. Restart the
Python process or ComfyUI server after changing them.

## Relevant kernel configurations

All configurations below use an INT8 `128x256x64` threadblock tile. Keeping the
tile shape fixed prevents tile geometry from confounding the scheduling test.

| Config | Kernel scheduling | Purpose |
|---:|---|---|
| 0 | ordinary data parallel, identity raster `g=1` | grouping baseline |
| 13 | LeanStreamK, adaptive short-dimension raster, K splitting enabled | StreamK treatment |
| 14 | ordinary data parallel, grouped raster `g=4` | grouping treatment |
| 15 | ordinary data parallel, grouped raster `g=8` | grouping treatment |
| 18 | LeanStreamK kernel and raster, K splitting disabled | matched control for config 13 |

Use these comparisons:

- **grouping effect:** config 0 versus 14 versus 15;
- **K-splitting effect:** config 18 versus 13;
- **whole scheduling-policy difference:** config 0 versus 13 versus 15;
- **LeanStreamK raster/path difference without splitting:** config 0 versus 18.

Config 0 versus config 13 is not a pure StreamK-splitting comparison.

## Machine and build record

Record the exact commit, GPU, SM count, L2 size, clocks, CUDA version, PyTorch
version, and whether another process is using the GPU. At minimum, capture:

```bash
git rev-parse HEAD
git submodule status
nvidia-smi
python - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", p.name)
print("sm_count", p.multi_processor_count)
print("l2_bytes", getattr(p, "L2_cache_size", None))
PY
```

Build the local checkout for the GPU on the server. Adapt the Python executable
and CUDA architecture to the existing ComfyUI environment:

```bash
python setup.py build_ext --inplace --cuda-archs=89
```

Before benchmarking, verify that Python imports `_C` from this checkout rather
than an installed wheel:

```bash
python - <<'PY'
from comfy_kitchen.backends import cuda
print(cuda.__file__)
print(cuda._C.__file__)
PY
```

## Shapes

Run all four H3 projections at the measured 768p/124-frame value `M=38819`:

| Projection | M | N | K | Output tiles | Waves on 128 SMs |
|---|---:|---:|---:|---:|---:|
| `out_proj` | 38819 | 5376 | 7168 | 6,384 | 49.875 |
| `fc2` | 38819 | 5376 | 14336 | 6,384 | 49.875 |
| `qkv_proj` | 38819 | 21504 | 5376 | 25,536 | 199.5 |
| `fc1` | 38819 | 28672 | 5376 | 34,048 | 266 |

Calculate waves on the actual device as

```text
ceil(M / 128) * ceil(N / 256) / SM_count
```

For the StreamK crossover, also hold `M=38819`, `K=5376` and sweep:

| N | N tiles | Waves on 128 SMs |
|---:|---:|---:|
| 2816 | 11 | 26.125 |
| 5376 | 21 | 49.875 |
| 8192 | 32 | 76 |
| 10752 | 42 | 99.75 |
| 13568 | 53 | 125.875 |
| 16128 | 63 | 149.625 |
| 21504 | 84 | 199.5 |
| 24576 | 96 | 228 |

The actual wave counts must use the server's SM count.

For the resolution and duration comparison, use 864x480 and 1344x768 at
**124 frames (~5 s)** and **345 frames (~15 s)**. Use the same prompt at all
four shapes: the `pf_ref_768_api.json` workflow has **388 text rows** in the
actual ComfyUI run. `M` includes video, audio, **and** text rows. Video rows are
`video_latent_t(frames) * (width/32) * (height/32)`; audio contributes
`2 * round(frames/24 * 40)` rows. This gives `M=15,787` (480p/124f),
`42,848` (480p/345f), `38,098` (768p/124f), and `104,354` (768p/345f).
Confirm all four from the model's projection inputs before treating the
synthetic shape sweep as exact. Run the repeated shape sweep with
`GPU=1 bash samples/run_int8_shape_sweep.sh` on a free GPU 1; it refuses to
start if that card already uses over 200 MiB. The earlier 768p/124f kernel
campaign used `M=38,819` with **another prompt** containing 1,109 text rows,
so those measurements are nearby shapes, not matched to this workflow. An
earlier summary called the 141f residual
of 858 rows “text tokens,” but 470 of those rows were audio.

## Kernel microbenchmark protocol

Use `_C.benchmark_cutlass_int8_dequant_config` so every run forces one explicit
configuration. Its return value is total milliseconds for all iterations;
divide by `iterations` to obtain milliseconds per call.

For every shape:

1. Allocate `A[M,K]` and `B[N,K]` as INT8, row scales `xs[M]` and weight scales
   `ws[N]` as FP32, and output `D[M,N]` as BF16.
2. Run configs 0, 13, 14, 15, and 18 once. A negative benchmark return means
   that configuration cannot implement the shape and must be reported, not
   silently omitted.
3. Verify each result against config 0. Report exact equality and maximum
   absolute difference. Do this before timing.
4. Warm up every configuration for at least 10 calls.
5. Measure at least five rounds of 100 calls. Randomize configuration order in
   each round to reduce clock and temperature bias.
6. Synchronize before and after each measured round. Report the median and
   range across rounds.
7. Compute achieved throughput as

   ```text
   TOPS = 2 * M * N * K / seconds_per_call / 1e12
   ```

Keep tensor allocation, values, output dtype, CUDA stream, clocks, and process
identical across configurations. Record GPU temperature and clocks before and
after the sweep.

The runner `samples/bench_int8_ablation.py` prints raw JSON timing rounds and
correctness per projection. For example, from the checkout with an idle card:

```bash
mkdir -p out/ablation
CUDA_VISIBLE_DEVICES=1 python samples/bench_int8_ablation.py \
  --m 38098 --shape 768p_124f > out/ablation/kernel_768p_124f.jsonl
```

It checks exact BF16 output equality against config 0 before timing. Run the
other three measured M values with the same flags and compare config 0/14/15
for grouping and config 18/13 for K splitting.

LeanStreamK only splits the `sk_tiles` near a partly filled wave, and it does
not split at all if the output tile count is a multiple of the SM count. On
128 SMs, `fc1` at the earlier `M=38,819` has exactly 34,048 output tiles
(266 waves): config 13 and 18 do the same K workload at that shape. At the
current workflow's `M=38,098`, a small remainder can split. A config
13-versus-0 slowdown is **not** evidence that K splitting caused it. Compare
13/18 on smaller test shapes with a higher split-tile fraction too.

Produce one table per shape with at least:

| Config | split K? | raster | ms/call | TOPS | relative to matched control | exact? | max abs diff |
|---:|---|---|---:|---:|---:|---|---:|

Interpret config 13 versus 18 as the K-splitting result. Interpret config 0
versus 14/15 as the grouping result.

## Profiler validation

Timing alone cannot establish the mechanism. Profile one representative call
of configs 0, 14, and 15 for `fc1` and `qkv_proj`, and configs 13 and 18 for at
least one low-wave and one high-wave shape.

Use Nsight Compute metrics available on the installed version for:

- DRAM bytes read;
- L2 read sectors and L2 hit rate;
- SM and Tensor Core utilization;
- kernel duration;
- atomic/reduction traffic for config 13 versus 18, if exposed.

Common metric names include `dram__bytes_read.sum`,
`lts__t_sector_hit_rate.pct`, and
`sm__throughput.avg.pct_of_peak_sustained_elapsed`; query `ncu --query-metrics`
and record substitutions if these names are unavailable.

The grouping hypothesis predicts lower DRAM reads and/or a higher L2 hit rate
for config 15 than config 0 on the large `fc1` and `qkv_proj` shapes. The
StreamK hypothesis predicts better utilization from config 13 only when wave
quantization savings exceed its partial-reduction overhead.

## End-to-end dispatch ablation

Run four fresh server processes. Ensure the legacy master switch
`COMFY_KITCHEN_DISABLE_STREAMK_OVERRIDE` is unset in every case.

| Experiment | Grouped rule | High-wave StreamK override | Environment |
|---|---|---|---|
| A: original heuristic | off | off | both independent disable flags `=1` |
| B: StreamK wave rule only | off | on | grouped disable `=1`, StreamK-wave disable `=0` |
| C: grouping rule only | on | off | grouped disable `=0`, StreamK-wave disable `=1` |
| D: final combined policy | on | on | both independent disable flags `=0` |

Use:

```text
COMFY_KITCHEN_DISABLE_GROUPED_SWIZZLE_OVERRIDE
COMFY_KITCHEN_DISABLE_STREAMK_WAVE_OVERRIDE
```

For each process, confirm the expected `_int8_config_override()` decisions for
the four projection shapes before timing. Run the same warm model, warmup seed,
measured seed, resolution, frame count, denoising steps, and GPU. Capture both
steady per-pass and end-to-end times with at least three repetitions.

To cover all four resolution/duration combinations at manageable cost, use a
fixed four-step **measurement probe without any acceleration LoRA** for the
four-policy matrix. Scheduler choices depend on M/N/K, not denoising step count.
Report its full-graph wall time and label it a short probe, since encode and
decode affect total wall time. Record sampler time separately when measuring
per-pass effects. Validate at least one policy pair again using
the established 20-step 768p/124f workflow before extrapolating a 20-step
end-to-end saving. A 15-second/20-step render is substantially more expensive;
keep the full 20-step comparison separate from the short probe.

From this checkout on gpu4, the client and guarded server launcher run the
same `pf_ref_768_api.json` graph, with no LoRA, on dedicated port 8190. They
render each selected shape once with warmup seed 999 and once with measured seed 1101,
recording client wall time and ComfyUI history in `out/ablation`. Never use
ports 8188 or 8189 or GPUs 0 or 3 for this experiment:

```bash
GPU=1 bash samples/run_h3_policy_matrix.sh
```

The guarded script omits two sets of redundant videos: at 480p/124f, A and B
choose identical GEMM configs because the wave rule never fires; across these
four model shapes, C and D choose identical GEMM configs because the grouping
rule takes precedence on the projections affected by the wave rule. It runs C
only at 768p/124f as a sanity check and labels all other omitted cells as
predicted equality, not measured equality. This saves GPU time without
mistaking repeated identical dispatch for an independent treatment.

For three measured seeds instead of an exploratory single measurement, pass
`--seeds 999 1101 1102 1103` to the script. The same seed sequence must be used
for each policy; repeatedly submitting an identical graph and seed can reuse
ComfyUI's cached output instead of generating a new video. Compare server log
`Prompt executed in ... seconds` against client wall time and inspect the
history status and output file for every measured case. Report the first pass
as exploratory until repetitions and a 20-step reference confirm the result.
Use `[int8-ablation]` lines in each dedicated server log to check that all
four projection shapes used the expected M and independent flags.
The client records `server_wall_s` from ComfyUI's history timestamps and
`wall_s` from its HTTP request and polling. The MP4 attachment appears under
the history node's `images` field even though the file is a video; confirm
the recorded MP4 path exists before comparing decoded streams.

For the matched 20-step 768p/124f check, run only the original and combined
policies after the short matrix finishes. Put its logs in a separate output
directory so the probe logs remain available:

```bash
POLICIES="A D" OUTPUT=/home/ubuntu/chao/h3-lab/out/ablation/full20 GPU=1 \
  bash samples/run_h3_policy_matrix.sh \
  --steps 20 --seeds 999 1101 --shapes 768p_124f
```

Compare decoded video and audio rather than MP4 container bytes. Containers can
differ because of timestamps.

This policy ablation measures product behavior, but it is not a mathematical
factorial inside a single GEMM: a call selects one kernel, so grouped data
parallel and LeanStreamK do not operate simultaneously in the current code.

## Required report

Write `docs/STREAMK_SWIZZLE_ABLATION_RESULTS.md` containing:

1. machine and software record;
2. correctness results;
3. raw and summarized microbenchmark timings;
4. config 0/14/15 grouping comparisons;
5. matched config 18/13 K-splitting comparisons;
6. Nsight Compute evidence;
7. four-policy end-to-end results;
8. a revised attribution of the final speedup, separating what the evidence
   supports from remaining interactions;
9. any failures, unsupported configs, thermal drift, or measurement caveats.

Do not change production thresholds or dispatch behavior from these results.
The experiment is complete when the two mechanisms have matched controls,
correctness is verified, and both kernel-level and end-to-end numbers are
reported.
