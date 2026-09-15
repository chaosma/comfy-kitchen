# H3 INT8 GEMM scheduling ablation results

## Machine and experiment

Measured on gpu4, driver 580.159.03, CUDA 13.0, PyTorch 2.13.0+cu130,
one NVIDIA RTX 4090 with 128 SMs and 72 MiB L2. The CUDA extension was built
from `perf/kernel-ablation` for sm89 and imported directly from this checkout.
Kernel timing used GPU 1 while its memory read 1 MiB before and after each
run. The dedicated video server also used GPU 1 on localhost port 8190; the
operator's two live ComfyUI servers on GPUs 0 and 3 and ports 8188 and 8189
were left running. GPU 2 stayed idle.
During a 768p/345f sample, GPU 1 ran near 2,355 MHz SM clock and its 400 W
power cap, at roughly 65--68 C; GPUs 0 and 3 stayed at 0% GPU utilization.

The two production commits being tested, relative to `dae00a1`, are
`9ba977b` (high-wave config 13 to 0) and `0c0f28e` (grouped config 15).
The short matrix finished at experiment commit `3645fb2`. During its first
policy, the video client gained bookkeeping fixes for history timestamps and
MP4 attachments; the CUDA extension and independent dispatch remained the
same. The later repeat and exact-size checks use `7334305`, which corrected
the guide and made the video runner honor an explicitly requested shape;
this commit also did not change the CUDA extension or production dispatch.
Config 18 and the independent environment switches on this branch are
experimental controls; its default dispatch retains the combined production
policy. The five explicit configs use the same INT8 `128x256x64` tile:

| Config | Output raster | K work split? |
|---:|---|---|
| 0 | identity, M tile advances first, `g=1` | no |
| 13 | adaptive short-dimension raster | only on reserved Stream-K tiles |
| 14 | grouped raster, `g=4` | no |
| 15 | grouped raster, `g=8` | no |
| 18 | **same kernel and raster as 13** | no |

The controlled video graph uses the same prompt, pruned INT8 checkpoint,
SageAttention patch, no acceleration LoRA, 4 steps, and seeds 999 (warmup)
and 1101 (measurement) at all four shapes. The live backend logged these
projection sizes:

| Case | Width x height | Frames | Actual M | Video rows | Audio rows | Text rows |
|---|---:|---:|---:|---:|---:|---:|
| 480p / 5s | 864 x 480 | 124 | 15,787 | 14,985 | 414 | 388 |
| 480p / 15s | 864 x 480 | 345 | 42,848 | 41,310 | 1,150 | 388 |
| 768p / 5s | 1344 x 768 | 124 | 38,098 | 37,296 | 414 | 388 |
| 768p / 15s | 1344 x 768 | 345 | 104,354 | 102,816 | 1,150 | 388 |

The old 768p/124f kernel campaign used `M=38,819`, 721 rows above this live
video graph; its earlier conditioning makeup has not been reverified. Early
shape sweeps at M values 721 above this table are nearby-shape evidence only.
The matching sweep used the actual logged sizes above.

`ffprobe` confirmed the rendered 480p measured videos had 124 and 345 frames,
respectively, at 864 x 480; both contained 32 kHz stereo audio. Each policy
probe uses a different measured seed from its warmup, and the server history
shows the sampler ran rather than returning a cached video.

## What K splitting contributed

The LeanStreamK implementation assigns most output tiles to whole-tile CTAs.
When `output_tiles % 128` is nonzero, at most `128 + remainder` output tiles
use the Stream-K region; it splits no tile when the remainder is zero. At the
actual video sizes, the Stream-K region ranges from 0% to 6.6% of all output
tiles for the four projections. In particular, `fc1` at 768p/345f fills 714
complete waves and splits **zero** tiles.

| Case | `out_proj`/`fc2` in Stream-K region | `qkv_proj` | `fc1` |
|---|---:|---:|---:|
| 480p / 5s | 6.61% | 1.69% | 1.38% |
| 480p / 15s | 3.57% | 0.84% | 0.38% |
| 768p / 5s | 3.87% | 0.80% | 0.67% |
| 768p / 15s | 1.40% | 0.28% | **0%** |

To test genuine splitting on a smaller problem, M=2,048 places 208 of 336
`out_proj`/`fc2` output tiles in the Stream-K region (61.9%). Five randomized
rounds of 100 calls each gave these medians:

| Projection | K | Config 18, no splitting | Config 13, splitting | Config 13 gain |
|---|---:|---:|---:|---:|
| `out_proj` | 7,168 | 0.307 ms | 0.277 ms | 1.11x |
| `fc2` | 14,336 | 0.608 ms | 0.573 ms | 1.06x |
| `qkv_proj` | 5,376 | 0.909 ms | 0.884 ms | 1.03x |

`qkv_proj` reserves only 192 of its 1,344 output tiles (14.3%) at this M;
the opportunity for splitting is smaller than in the first two rows.
The matched control demonstrates that K splitting can help when many tiles
fall into an incomplete wave. At the old `M=38,819`/`fc1` shape, output tiles
are a multiple of 128: config 13 and 18 measured 51.39 and 51.49 ms,
respectively, while config 0 took 30.10 ms. That large config 13 slowdown
cannot be caused by cross-CTA K splitting, because none occurred. Config 13
walks output tile columns fastest for this shape, while config 0 walks tile
rows fastest. A changed raster or another property of the specialized kernel
path must account for their difference. Configs 13 and 18 also stayed close
in the other large-shape matched comparisons.

An illustrative locality calculation for `fc1`, using 128 nearby CTAs and
full-K input slices, makes the raster difference concrete. Config 0's launch
grid walks down a tile column: roughly 128 distinct A slices plus one B slice
occupy 85.3 MiB. Config 13 walks across the shorter, 112-tile-wide N dimension:
about two A slices plus 112 B slices occupy roughly 148.3 MiB. Config 15
visits an approximate `16 x 8` output rectangle: 16 A slices plus eight B
slices occupy 21 MiB. The RTX 4090 has 72 MiB L2. CTA execution order and
other cache users are not guaranteed, so these are locality estimates, not
measured L2 residency.

For `fc1`, a full-K A slice holds 0.656 MiB and a B slice holds 1.313 MiB.
For a nearby 128-CTA group of width `g`, the estimated footprint is
`(128/g) * 0.656 + g * 1.313` MiB. Thus `g=4` gives 26.25 MiB, `g=8` gives
21 MiB, and a hypothetical `g=16` gives 26.25 MiB. The weighted minimum
lies near `g=8`, because each B slice is twice as large as each A slice.
This estimate is for nearby input reuse, not the entire GEMM or a guarantee
that exactly 128 blocks execute at once. When the number of output tile
columns is not divisible by eight, the final group also launches some
out-of-range CTAs: `fc2` has 21 output tile columns, leaving three unused
positions in its last eight-column group per tile row.

## Kernel timings at the four live M values

The explicit-config runner uses the same INT8 `128x256x64` tile for all five
configs and measures five randomized rounds of 100 kernel calls per config.
Each row reports the median milliseconds per call; raw round timings, TOPS,
support status, and BF16 equality are in
`out/ablation/exact/kernel_<shape>.jsonl`. The interrupted 768p/345f sweep
recorded three projections there, and its isolated successful `fc1` retry
is in `kernel_768p_345f_fc1.jsonl`. All 16 rows supported all five configs
and matched config 0 **bit for bit** in BF16 (`max_abs_diff=0`).
Config 0/14/15 isolates `g=1/4/8` on the ordinary kernel; config 18/13
isolates K splitting with the LeanStreamK raster and kernel held fixed.

| Case | Projection | Config 0, g1 | Config 14, g4 | Config 15, g8 | Config 18, no split | Config 13, split eligible |
|---|---|---:|---:|---:|---:|---:|
| 480p/124f | `out_proj` | 2.99 | 2.41 | 2.26 | 2.20 | 2.19 |
| | `fc2` | 5.78 | 4.64 | 4.49 | 4.68 | 4.66 |
| | `qkv_proj` | 8.70 | 7.06 | 6.88 | 10.96 | 10.98 |
| | `fc1` | 11.95 | 9.76 | 9.34 | 21.35 | 21.31 |
| 480p/345f | `out_proj` | 8.31 | 6.63 | 6.23 | 6.08 | 6.11 |
| | `fc2` | 16.34 | 13.20 | 12.40 | 13.24 | 13.42 |
| | `qkv_proj` | 24.95 | 20.10 | 19.18 | 29.56 | 29.56 |
| | `fc1` | 33.38 | 26.90 | 25.62 | 56.79 | 56.85 |
| 768p/124f | `out_proj` | 7.39 | 5.91 | 5.61 | 5.45 | 5.46 |
| | `fc2` | 14.52 | 11.74 | 11.04 | 11.76 | 12.05 |
| | `qkv_proj` | 22.13 | 17.87 | 17.08 | 26.31 | 26.26 |
| | `fc1` | 29.34 | 23.66 | 22.55 | 50.20 | 50.09 |
| 768p/345f | `out_proj` | 20.11 | 16.16 | 15.23 | 14.88 | 14.90 |
| | `fc2` | 39.69 | 32.01 | 29.97 | 32.11 | 32.26 |
| | `qkv_proj` | 60.09 | 48.53 | 46.29 | 71.91 | 71.88 |
| | `fc1` | 81.17 | 65.51 | 62.41 | 147.18 | 147.16 |

For 768p/124f `fc1`, config 0's five rounds spanned 29.31--29.47 ms
and grouped config 15's spanned 22.49--22.58 ms: their 6.79 ms median gap
is much larger than either within-config range. Configs 13 and 18 at
50.09 and 50.20 ms are close compared with their roughly 21 ms gap from
config 0; the high-wave config 13-to-0 gain cannot be attributed mainly to
cross-CTA K splitting. On the smaller M=2,048 stress shape, 13 beat 18 by
11% for `out_proj`, showing that K splitting can still help when many tiles
fall in the reserved region.
The live 768p/345f `fc1` output grid has exactly 714 full waves and
reserves **zero** Stream-K tiles; its 147.16/147.18 ms config 13/18 tie
confirms that this slowdown versus config 0 cannot be caused by K sharing.
That `fc1` row was measured in a separate Python process after the original
shape sweep received a `KeyboardInterrupt` during timing. The initial three
projection rows were preserved, and the isolated retry repeated the full
five-round/100-call protocol with exact output checks. The `fc1` config 0
rounds ranged 76.48--83.40 ms and config 15 ranged 60.16--63.08 ms; their
gap stayed larger than within-config variation. The source of the interrupt
was not determined, and no CUDA correctness failure appeared in either log.

## Dispatch rules and predicted interactions

The A/B/C/D flags disable the *dispatch rules*, not K splitting within one
fixed config. Grouping is checked first, then the high-wave override:

| Policy | Grouping rule | Wave rule | `out_proj` | `fc2` | `qkv_proj` at 768p/5s | `fc1` |
|---|---|---|---:|---:|---:|---:|
| A, old heuristic | off | off | 13 | 13 | 13 | 0 |
| B, wave only | off | on | 13 | 13 | 0 | 0 |
| C, grouping only | on | off | 13 | 15 | 15 | 15 |
| D, combined | on | on | 13 | 15 | 15 | 15 |

At 480p/124f, A and B also agree because the old heuristic chooses config
0 for `qkv_proj`. C and D agree at all four video shapes because grouping
takes precedence on every GEMM the wave rule would affect. The video queue
omits B at 480p/124f and measures C at one representative shape to verify
the otherwise redundant C/D treatment.

## Four-step video matrix

All 24 queued videos completed on the dedicated server, including a seed
999 warmup and a seed 1101 measurement for each selected policy and shape.
The history shows that no `SamplerCustomAdvanced` node was cached. The first
version of the client omitted A's MP4 attachments from its JSON output, but
its status messages contain the execution timestamps and its four measured
MP4 files exist with the expected names. Later B/C/D JSON rows record the
history MP4 attachments directly. Server wall time below runs from history
`execution_start` to `execution_success`; it includes denoising, VAE decoding,
and saving the MP4, and avoids HTTP polling jitter.

| Case | A: old | B: wave only | C: grouping only | D: both | Measured A - D |
|---|---:|---:|---:|---:|---:|
| 480p / 124f | 23.32 s | same dispatch, omitted | same dispatch as D, omitted | 22.21 s | 1.11 s (4.8%) |
| 480p / 345f | 81.55 s | 80.62 s | same dispatch as D, omitted | 77.34 s | 4.21 s (5.2%) |
| 768p / 124f | 61.85 s | 60.99 s | **58.34 s** | **58.35 s** | 3.50 s (5.7%) |
| 768p / 345f | 263.58 s | 261.52 s | same dispatch as D, omitted | 253.99 s | 9.59 s (3.6%) |

The four shapes expose how the workload grows. At fixed 480p, going from
124 to 345 frames increases live M by 2.71x and A's wall time by 3.50x;
at fixed 768p, M grows by 2.74x and A's wall time by 4.26x. At 124 frames,
moving from 480p to 768p increases M by 2.41x and A's wall time by 2.65x;
at 345 frames the corresponding factors are 2.44x and 3.23x. The 768p
A-to-D **absolute** saving rises from 3.50 to 9.59 s as clips grow, but
its **percentage** falls from 5.7% to 3.6%. These full-video ratios also
include VAE and MP4 work; they cannot by themselves separate attention's
quadratic scaling from that overhead.

"Same dispatch" is a predicted config equality; the omitted video times
were **not measured**. C versus D at 768p/124f was measured separately and
agreed within 0.01 s. The high-wave rule alone saved 0.93, 0.86, and 2.06 s
where B was measured; the B-to-D policy changes saved a further 3.28, 2.64,
and 7.53 s. Since C and D select identical GEMMs at these four shapes, the
high-wave rule contributes **no further dispatch change** to the combined
production policy on this specific H3 graph.

The server's measured sampler progress supplies a second view of the same
renders. Its `s/it` values cover a denoise pass, with four passes per video;
they are approximate progress-bar times, not a GPU profiler trace:

| Case | A: seconds/pass | B | C | D |
|---|---:|---:|---:|---:|
| 480p / 124f | 2.69 | omitted | omitted | 2.45 |
| 480p / 345f | 11.47 | 11.26 | omitted | 10.53 |
| 768p / 124f | 9.60 | 9.36 | 8.72 | 8.74 |
| 768p / 345f | 49.84 | 49.42 | omitted | 47.57 |

Multiplying the A-to-D per-pass differences by four gives 0.96, 3.76,
3.44, and 9.08 s, close to the corresponding full-graph savings of 1.11,
4.21, 3.50, and 9.59 s. The video matrix used one measured seed per long
case; those timing differences are exploratory. A/B/D at 768p/124f were
repeated on two additional measured seeds. The matched 20-step A/D run is
reported separately below.

Summing the matching explicit-config medians in the exact-size kernel table
for all four projections gives these policy costs per DiT block. There are
50 blocks per denoise pass; multiplying the A-to-D kernel gap by
`50 * 4 / 1000` predicts seconds saved in the four-step probe:

| Case | A: ms/block | B: ms/block | D: ms/block | Kernel-predicted A - D | Video-measured A - D |
|---|---:|---:|---:|---:|---:|
| 480p / 124f | 27.50 | 27.50 | 22.90 | 0.92 s | 1.11 s |
| 480p / 345f | 82.47 | 77.85 | 63.31 | 3.83 s | 4.21 s |
| 768p / 124f | 73.12 | 68.98 | 56.13 | 3.40 s | 3.50 s |
| 768p / 345f | 200.19 | 188.41 | 153.57 | 9.32 s | 9.59 s |

The predicted video saving lands within 0.38 s of each measured saving.
At 768p/124f the same calculation predicts about 17.0 s for 20 passes,
against the measured 17.75 s reported below. The final 768p/345f `fc1`
kernel median came from a fresh process after the interrupted sweep;
temperature and clocks can differ between processes. These agreements
support GEMM scheduling as the source of the observed video gain, but do
not measure L2 hits or turn a policy comparison into a pure K-splitting
comparison.

## Three paired seeds at 768p / 124 frames

The first matrix measured seed 1101. The second run launched fresh A/B/D
servers, warmed each on seed 999, then measured seeds 1102 and 1103 on the
same 768p/124-frame graph. All nine prompts completed, their MP4 outputs
were recorded, and none reused a cached sampler. Server wall times and
pairwise differences were:

| Seed | A: old | B: wave only | D: both | A - B | B - D | A - D |
|---:|---:|---:|---:|---:|---:|---:|
| 1101 | 61.85 s | 60.99 s | 58.35 s | 0.86 s | 2.64 s | 3.50 s |
| 1102 | 61.54 s | 61.03 s | 58.40 s | 0.51 s | 2.63 s | 3.14 s |
| 1103 | 61.41 s | 60.63 s | 58.35 s | 0.78 s | 2.28 s | 3.06 s |
| **Median paired gain** | | | | **0.78 s** | **2.63 s** | **3.14 s** |

All three paired wave-rule and grouping-rule policy comparisons saved time.
The medians are exploratory estimates from three seeds, not uncertainty
intervals. In the repeat server logs, measured denoising progress stayed
near 9.56 s/pass for A, 9.36 s/pass for B, and 8.74 s/pass for D, while
full-graph times varied somewhat with decoding and saving the MP4. Four
times the per-pass differences imply about 0.80 s for A-to-B and 2.48 s
for B-to-D. These logs are in `out/ablation/repeat/`; the raw seed 1101
histories are in `out/ablation/policy_matrix.jsonl`.

The B-to-D video difference is the effect of enabling the **grouping
dispatch rule**, which can also replace config 13 with 15 for `fc2`. The
config 0/14/15 comparisons in the kernel sweep hold the tile and
ordinary data-parallel path fixed to isolate the `g` value itself.

## Matched 20-step reference at 768p / 124 frames

Fresh A and D servers rendered the same prompt, pruned INT8 checkpoint,
SageAttention patch, no acceleration LoRA, and 20 denoising steps on GPU 1.
Each server warmed on seed 999, then measured seed 1101. History recorded a
completed MP4 and a live sampler for all four renders:

| Policy | Warmup server wall | Measured server wall | Measured sampler s/pass |
|---|---:|---:|---:|
| A: old heuristic | 220.34 s | **215.84 s** | 9.64 |
| D: both production rules | 204.29 s | **198.09 s** | 8.76 |
| **Measured A - D** | | **17.75 s (8.2%)** | 0.88 |

The 0.88 s per-pass gap across 20 passes accounts for approximately 17.60
s of the 17.75 s whole-video difference. The remaining decode/save
overhead was about 23.04 s for A and 22.89 s for D. This is a single
measured-seed reference, while the shorter four-step A/B/D treatment has
three measured seeds. The older production campaign reported similar
20-step totals, but its 768p/124f GEMM table used `M=38,819`; the
matched-prompt check here logged `M=38,098` and should stand on its own.
Raw histories and server logs are in `out/ablation/full20/`.

## Decoded-video and audio correctness

`ffprobe` confirmed the measured D MP4s contained 124 or 345 video frames
at 864x480 or 1344x768, as requested. Each also carried 32 kHz stereo
audio. FFmpeg decoded video to RGB24 and audio to 32 kHz stereo signed
16-bit PCM, then computed SHA256 on the decoded streams rather than on the
MP4 containers. The same seed and shape gave identical decoded streams
in **all 14 comparisons**: A versus D at four four-step video shapes,
A versus B and C versus D at 768p/124f, and A versus D at 20 steps.

| Measured paired video | Decoded video SHA256 prefix | Decoded audio SHA256 prefix |
|---|---|---|
| Four steps, A/D, 480p/124f | `36434be331c8` | `45b42d0c0332` |
| Four steps, A/D, 480p/345f | `3dda16b8d876` | `1be09528a9b7` |
| Four steps, A/B/C/D, 768p/124f | `5aab00108ef7` | `d1eed74b0a91` |
| Four steps, A/D, 768p/345f | `0412562dc7c4` | `3e0cd0215e00` |
| 20 steps, A/D, 768p/124f | `6e832c16a399` | `79e445359376` |

The complete hashes, equality flags, and video/audio metadata are in
`out/ablation/decoded_hashes.jsonl`. The MP4 container bytes need not
match because timestamps and other metadata can differ. The checked seeds
and configs produced bit-identical *decoded* outputs here; K splitting
changes reduction coordination in general, so this is a verified property
of these cases, not a promise for every matrix and output dtype.

## Evidence limits

The kernel runner compared every supported config against config 0 with BF16
exact equality before timing and recorded each raw round in
`out/ablation/kernel_*.jsonl` and the exact-size sweep under
`out/ablation/exact/`. Nsight Compute 2025.3.1 is installed at
`/usr/local/cuda/bin/ncu`, although it is missing from this account's
`PATH`. The driver reports `RmProfilingAdminOnly: 1`, and this account has
no passwordless `sudo`. L2 hit rates and DRAM bytes therefore have not
been measured on the dedicated card. A bounded unprivileged probe on the
first CUTLASS `Kernel2` launch returned `ERR_NVGPUCTRPERM` when requesting
`dram__bytes_read.sum`; its log is `out/ablation/ncu_probe_kernel2.log`.
The probe's own kernel times are not used as performance measurements.
The L2 footprint model motivates grouping and is consistent with timing, but
timings alone do not prove which input was evicted from L2.
