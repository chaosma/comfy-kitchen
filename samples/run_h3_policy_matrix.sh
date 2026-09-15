#!/usr/bin/env bash
# Run the four dispatch policies through fresh, isolated ComfyUI servers.
set -euo pipefail

ROOT=/home/ubuntu/chao/h3-lab
GPU=${GPU:-1}
PORT=${PORT:-8190}
OUTPUT=${OUTPUT:-$ROOT/out/ablation}
WORKFLOW=$ROOT/workflows/pf_ref_768_api.json
POLICIES=${POLICIES:-"A B C D"}
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
if [ "$USED" -gt 200 ]; then
  echo "GPU $GPU is busy (${USED} MiB); choose an idle card." >&2
  exit 1
fi

mkdir -p "$OUTPUT"
cd "$ROOT"
export PYTHONPATH="$ROOT/src/comfy-kitchen${PYTHONPATH:+:$PYTHONPATH}"
export COMFY_KITCHEN_TRACE_INT8_ABLATION=1
unset COMFY_KITCHEN_DISABLE_STREAMK_OVERRIDE

server_pid=
stop_server() {
  if [ -n "$server_pid" ]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=
  fi
}
trap stop_server EXIT

for policy in $POLICIES; do
  shape_args=()
  case "$policy" in
    A) export COMFY_KITCHEN_DISABLE_GROUPED_SWIZZLE_OVERRIDE=1
       export COMFY_KITCHEN_DISABLE_STREAMK_WAVE_OVERRIDE=1 ;;
    B) export COMFY_KITCHEN_DISABLE_GROUPED_SWIZZLE_OVERRIDE=1
       export COMFY_KITCHEN_DISABLE_STREAMK_WAVE_OVERRIDE=0
       shape_args=(--shapes 480p_345f 768p_124f 768p_345f) ;;
    C) export COMFY_KITCHEN_DISABLE_GROUPED_SWIZZLE_OVERRIDE=0
       export COMFY_KITCHEN_DISABLE_STREAMK_WAVE_OVERRIDE=1
       shape_args=(--shapes 768p_124f) ;;
    D) export COMFY_KITCHEN_DISABLE_GROUPED_SWIZZLE_OVERRIDE=0
       export COMFY_KITCHEN_DISABLE_STREAMK_WAVE_OVERRIDE=0 ;;
  esac

  # start_server.sh refuses busy cards and ports, binds localhost, and execs
  # Python so server_pid is the precise process this script started.
  REQUIRE_IDLE=1 GPU="$GPU" PORT="$PORT" "$ROOT/start_server.sh" \
    > "$OUTPUT/server_${policy}.log" 2>&1 &
  server_pid=$!
  ready=0
  for attempt in {1..90}; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server $policy exited early; inspect $OUTPUT/server_${policy}.log" >&2
      exit 1
    fi
    if "$ROOT/venv/bin/python" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/system_stats', timeout=2)" \
        >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  if [ "$ready" -ne 1 ]; then
    echo "Server $policy did not become ready" >&2
    exit 1
  fi

  echo "policy $policy: grouped_disable=$COMFY_KITCHEN_DISABLE_GROUPED_SWIZZLE_OVERRIDE wave_disable=$COMFY_KITCHEN_DISABLE_STREAMK_WAVE_OVERRIDE" | tee -a "$OUTPUT/matrix.log"
  "$ROOT/venv/bin/python" "$ROOT/src/comfy-kitchen/samples/queue_h3_policy_probe.py" \
    --workflow "$WORKFLOW" --port "$PORT" --policy "$policy" "$@" "${shape_args[@]}" \
    >> "$OUTPUT/policy_matrix.jsonl" 2>> "$OUTPUT/matrix.err"

  stop_server
  for attempt in {1..30}; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    [ "$USED" -le 200 ] && break
    sleep 2
  done
  if [ "$USED" -gt 200 ]; then
    echo "GPU $GPU still holds ${USED} MiB after server $policy" >&2
    exit 1
  fi
done

echo "All four policies complete; results: $OUTPUT/policy_matrix.jsonl"
