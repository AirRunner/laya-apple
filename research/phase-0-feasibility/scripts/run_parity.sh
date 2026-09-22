#!/bin/zsh
# Full parity matrix, one process per (model, configuration). Untimed; correctness only.
set -u
cd "$(dirname "$0")/.."
for m in ${=MODELS:-laya-typed-decisions laya-multilingual laya}; do
  for spec in \
    '{"backend":"torch","device":"mps"}' \
    '{"backend":"mlx","dtype":"float32"}' \
    '{"backend":"mlx","dtype":"float16"}' \
    '{"backend":"coreml","units":"cpu_gpu","enumerated":true}' \
    '{"backend":"coreml","units":"cpu_gpu"}' \
    '{"backend":"coreml","units":"cpu_ne"}' \
    '{"backend":"coreml","units":"all"}' \
    '{"backend":"coreml","units":"cpu_only"}' \
    '{"backend":"ane","units":"cpu_ne"}' \
    '{"backend":"ane","units":"all"}' \
    '{"backend":"ane","units":"cpu_gpu"}' \
    '{"backend":"ane","units":"cpu_only"}' ; do
    uv run python scripts/parity.py --model $m --spec "$spec" 2>&1 | grep -E "passed=|Error|error" | tail -2
  done
done
