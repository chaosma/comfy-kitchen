#!/usr/bin/env bash
# Run the four projection sweeps on one free GPU; output stays outside git.
set -euo pipefail

ROOT=/home/ubuntu/chao/h3-lab
GPU=${GPU:-1}
OUTPUT=${OUTPUT:-$ROOT/out/ablation}
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
if [ "$USED" -gt 200 ]; then
  echo "GPU $GPU is busy (${USED} MiB); choose an idle card." >&2
  exit 1
fi

mkdir -p "$OUTPUT"
cd "$ROOT/src/comfy-kitchen"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Same 1,109-text-row prompt as the reference 768p/124f workflow.
for pair in 480p_124f:16508 480p_345f:43569 768p_124f:38819 768p_345f:105075; do
  label=${pair%%:*}
  m=${pair##*:}
  "$ROOT/venv/bin/python" samples/bench_int8_ablation.py --m "$m" --shape "$label" \
    > "$OUTPUT/kernel_${label}.jsonl" 2> "$OUTPUT/kernel_${label}.err"
  echo "completed $label (M=$m)"
done
