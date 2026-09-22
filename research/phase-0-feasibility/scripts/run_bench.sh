#!/bin/zsh
# Formal latency matrix (Experiments 1 and 4). Run only when nothing else uses GPU/ANE.
# Pass A runs backends in the listed order; pass B reverses it (drift control).
set -u
cd "$(dirname "$0")/.."
OUT=raw/bench/latency.jsonl
MODELS=(${=MODELS:-laya-typed-decisions laya-multilingual laya})
typeset -a SPECS
SPECS=(
  '{"backend":"mlx","dtype":"float16"}|1 4 8|forward predict'
  '{"backend":"ane","units":"cpu_ne"}|1 4 8|forward predict'
  '{"backend":"coreml","units":"cpu_ne"}|1 4|forward'
  '{"backend":"coreml","units":"cpu_gpu"}|1 4|forward'
  '{"backend":"torch","device":"mps"}|1 4 8|forward predict'
  '{"backend":"mlx","dtype":"float32"}|1|forward'
  '{"backend":"ane","units":"all"}|1|forward'
  '{"backend":"ane","units":"cpu_gpu"}|1|forward'
  '{"backend":"coreml","units":"all"}|1|forward'
  '{"backend":"coreml","units":"cpu_only"}|1|forward'
  '{"backend":"ane","units":"cpu_only"}|1|forward'
  '{"backend":"torch","device":"cpu"}|1|forward'
)
run() { # tag spec
  local tag=$1 line=$2 spec qs modes
  spec=${line%%|*}; qs=${${line#*|}%%|*}; modes=${line##*|}
  for m in $MODELS; do
    uv run python scripts/bench.py --model $m --spec "$spec" --questions ${=qs} --modes ${=modes} \
      --budget ${BUDGET:-4} --tag $tag --output $OUT 2>&1 | grep -E "fwd p50|Error|Traceback" 
  done
}
for line in $SPECS; do run pass-a "$line"; done
# pass B: reversed order, key configurations only, q=1
for line in ${(Oa)SPECS[1,5]}; do run pass-b "${line%%|*}|1|forward"; done
# enumerated ordinary export (published laya-coreml default) at two lengths, as evidence
for m in $MODELS; do
  uv run python scripts/bench.py --model $m --spec '{"backend":"coreml","units":"cpu_gpu","enumerated":true}' \
    --lengths 128 512 --questions 1 --modes forward --min-samples 20 --budget 3 --tag pass-a --output $OUT 2>&1 | grep -E "fwd p50|Error"
done
