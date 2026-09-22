#!/bin/zsh
# Serial queue for every timed Phase -1 measurement. Nothing else may use GPU/ANE meanwhile.
set -u
cd "$(dirname "$0")/.."
step() { echo "=== $(date '+%H:%M:%S') $*"; }
step "latency matrix"; ./scripts/run_bench.sh
step "batched ANE (crossover vs batch)"
for m in laya-typed-decisions laya-multilingual; do
  for B in 4 8; do
    uv run python scripts/bench.py --model $m --spec "{\"backend\":\"ane\",\"units\":\"cpu_ne\",\"batch\":$B}" \
      --lengths 64 128 256 --questions $B --modes forward predict --budget 4 --tag batch --output raw/bench/latency.jsonl 2>&1 | grep -E "fwd p50|Error"
  done
done
step "windowed ANE full body"
uv run python scripts/bench.py --model laya-typed-decisions --spec '{"backend":"ane","units":"cpu_ne","variant":"windowed"}' \
  --lengths 128 256 512 1024 --questions 1 --modes forward --budget 4 --tag windowed --output raw/bench/latency.jsonl 2>&1 | grep -E "fwd p50|Error"
step "component probes"
uv run python scripts/profile_probes.py run --model laya-typed-decisions --lengths 128 256 512 1024 --units cpu_ne \
  --warmup 10 --samples 100 --output raw/profile/probes-laya-typed-decisions.jsonl 2>&1 | grep -vE "Warn|read_temp" | tail -60
step "concurrency"
uv run python scripts/concurrency.py --model laya-typed-decisions --short 128 --long 1024 --seconds 20 --cycles 3 --ane-long \
  --output raw/concurrency/laya-typed-decisions-128-1024.json 2>&1 | grep -vE "Warn|read_temp|concatenate"
uv run python scripts/concurrency.py --model laya-multilingual --short 96 --long 1024 --seconds 20 --cycles 3 \
  --output raw/concurrency/laya-multilingual-96-1024.json 2>&1 | grep -vE "Warn|read_temp|concatenate"
step "traces"
for spec_len in '{"backend":"ane","units":"cpu_ne"}|1024|typed-L1024-ane-cpu_ne' \
                '{"backend":"coreml","units":"cpu_ne"}|128|typed-L128-ordinary-cpu_ne' \
                '{"backend":"coreml","units":"cpu_ne"}|1024|typed-L1024-ordinary-cpu_ne' \
                '{"backend":"ane","units":"all"}|128|typed-L128-ane-all'; do
  spec=${spec_len%%|*}; rest=${spec_len#*|}; L=${rest%%|*}; name=${rest#*|}
  uv run python scripts/trace_ane.py --model laya-typed-decisions --spec "$spec" --length $L --seconds 6 --output raw/trace/$name 2>&1 | grep -E "^ane-hw|^metal-gpu-intervals|returncode"
done
step "done"
