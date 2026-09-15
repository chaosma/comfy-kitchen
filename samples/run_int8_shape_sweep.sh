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

# Same 388-text-row prompt as the measured ComfyUI workflow.
for pair in 480p_124f:15787 480p_345f:42848 768p_124f:38098 768p_345f:104354; do
  label=${pair%%:*}
  m=${pair##*:}
  "$ROOT/venv/bin/python" samples/bench_int8_ablation.py --m "$m" --shape "$label" \
    > "$OUTPUT/kernel_${label}.jsonl" 2> "$OUTPUT/kernel_${label}.err"
  echo "completed $label (M=$m)"
done
