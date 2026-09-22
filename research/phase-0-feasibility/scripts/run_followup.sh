#!/bin/zsh
# Follow-up queue: cold start, crossover refinement, windowed parity. Serial.
set -u
cd "$(dirname "$0")/.."
step() { echo "=== $(date '+%H:%M:%S') $*"; }
step "coldstart"
uv run python scripts/coldstart.py --model laya-typed-decisions --length 128 --runs 3 2>&1 | grep -v Warn
uv run python scripts/coldstart.py --model laya-multilingual --length 128 --runs 2 2>&1 | grep -v Warn
step "refinement conversions"
for L in 160 192 224; do uv run python scripts/convert_ane.py --model laya-typed-decisions --length $L 2>&1 | grep -E '"status"|Error'; done
for L in 320 384 448; do uv run python scripts/convert_ane.py --model laya-multilingual --length $L 2>&1 | grep -E '"status"|Error'; done
step "refinement timing"
uv run python scripts/bench.py --model laya-typed-decisions --spec '{"backend":"ane","units":"cpu_ne"}' --lengths 160 192 224 --questions 1 --modes forward predict --budget 4 --tag refine --output raw/bench/latency.jsonl 2>&1 | grep -E "fwd p50|Error"
uv run python scripts/bench.py --model laya-typed-decisions --spec '{"backend":"mlx","dtype":"float16"}' --lengths 160 192 224 --questions 1 --modes forward predict --budget 4 --tag refine --output raw/bench/latency.jsonl 2>&1 | grep -E "fwd p50|Error"
uv run python scripts/bench.py --model laya-multilingual --spec '{"backend":"ane","units":"cpu_ne"}' --lengths 320 384 448 --questions 1 --modes forward predict --budget 4 --tag refine --output raw/bench/latency.jsonl 2>&1 | grep -E "fwd p50|Error"
uv run python scripts/bench.py --model laya-multilingual --spec '{"backend":"mlx","dtype":"float16"}' --lengths 320 384 448 --questions 1 --modes forward predict --budget 4 --tag refine --output raw/bench/latency.jsonl 2>&1 | grep -E "fwd p50|Error"
step "windowed parity"
uv run python scripts/parity.py --model laya-typed-decisions --spec '{"backend":"ane","units":"cpu_ne","variant":"windowed","buckets":[128,256,512,1024]}' 2>&1 | grep -E "passed=|Error"
step "done"
