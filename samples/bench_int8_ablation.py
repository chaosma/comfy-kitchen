"""Compare explicit CUTLASS INT8 schedules on one idle CUDA device.

Usage: CUDA_VISIBLE_DEVICES=1 python samples/bench_int8_ablation.py --m 38819
Prints one JSON line per projection, with raw timing rounds and correctness.
"""

import argparse
import json
import random
import statistics

import torch

from comfy_kitchen.backends import cuda


PROJECTIONS = {
    "out_proj": (5376, 7168),
    "fc2": (5376, 14336),
    "qkv_proj": (21504, 5376),
    "fc1": (28672, 5376),
}
CONFIGS = (0, 13, 14, 15, 18)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--shape", default="", help="Label added to each JSON row")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--projections", nargs="+", choices=PROJECTIONS, default=list(PROJECTIONS))
    args = parser.parse_args()

    if min(args.m, args.rounds, args.iterations, args.warmup) <= 0:
        parser.error("m, rounds, iterations, and warmup must be positive")

    device = torch.device("cuda:0")
    stream = torch.cuda.current_stream(device).cuda_stream
    print(json.dumps({"machine": torch.cuda.get_device_name(device),
                      "torch": torch.__version__, "cuda": torch.version.cuda,
                      "extension": cuda._C.__file__, "m": args.m}), flush=True)

    def call(a, b, xs, ws, output, config):
        return cuda._C.cutlass_int8_dequant_config(
            cuda._wrap_for_dlpack(a), cuda._wrap_for_dlpack(b),
            cuda._wrap_for_dlpack(xs), cuda._wrap_for_dlpack(ws),
            cuda._wrap_for_dlpack(output), 2, config, stream)

    def bench(a, b, xs, ws, output, config, iterations):
        return cuda._C.benchmark_cutlass_int8_dequant_config(
            cuda._wrap_for_dlpack(a), cuda._wrap_for_dlpack(b),
            cuda._wrap_for_dlpack(xs), cuda._wrap_for_dlpack(ws),
            cuda._wrap_for_dlpack(output), 2, config, iterations, stream)

    for name in args.projections:
        n, k = PROJECTIONS[name]
        torch.manual_seed(1101)
        a = torch.randint(-8, 8, (args.m, k), dtype=torch.int8, device=device)
        b = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=device)
        xs = torch.full((args.m, 1), 1 / 128, dtype=torch.float32, device=device)
        ws = torch.full((n,), 1 / 128, dtype=torch.float32, device=device)
        baseline = torch.empty((args.m, n), dtype=torch.bfloat16, device=device)
        output = torch.empty_like(baseline)
        if not call(a, b, xs, ws, baseline, 0):
            raise RuntimeError(f"config 0 unsupported for {name}")
        torch.cuda.synchronize(device)

        results = {}
        for config in CONFIGS:
            supported = call(a, b, xs, ws, output, config)
            torch.cuda.synchronize(device)
            if not supported:
                results[config] = {"supported": False}
                continue
            equal = torch.equal(baseline, output)
            maximum = 0.0
            if not equal:
                for start in range(0, args.m, 128):
                    end = min(start + 128, args.m)
                    difference = (baseline[start:end].float() - output[start:end].float()).abs()
                    maximum = max(maximum, difference.max().item())
            results[config] = {"supported": True, "exact": equal, "max_abs_diff": maximum,
                               "round_ms": []}
            if not equal:
                raise RuntimeError(f"{name}: config {config} differs by {maximum}")
            if bench(a, b, xs, ws, output, config, args.warmup) < 0:
                raise RuntimeError(f"{name}: config {config} became unsupported on warmup")

        for round_id in range(args.rounds):
            order = list(CONFIGS)
            random.Random(1101 + round_id).shuffle(order)
            for config in order:
                if not results[config]["supported"]:
                    continue
                torch.cuda.synchronize(device)
                total_ms = bench(a, b, xs, ws, output, config, args.iterations)
                if total_ms < 0:
                    raise RuntimeError(f"{name}: config {config} became unsupported on timing")
                results[config]["round_ms"].append(total_ms / args.iterations)

        for config in CONFIGS:
            entry = results[config]
            if entry["supported"]:
                entry["median_ms"] = statistics.median(entry["round_ms"])
                entry["tops"] = 2 * args.m * n * k / (entry["median_ms"] * 1e9)
        print(json.dumps({"shape": args.shape, "projection": name, "m": args.m,
                          "n": n, "k": k, "rounds": args.rounds,
                          "iterations": args.iterations, "configs": results}), flush=True)
        del a, b, xs, ws, baseline, output
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
