# Hardware report: Apple M4 Pro, macOS 27.0

Matrix row (docs/community-benchmarks.md):

| Mac | MLX | ANE | Auto uses ANE | Heterogeneous |
|---|---|---|---|---|
| Apple M4 Pro (48 GB, macOS 27.0) | ✓ | ✓ | yes | ✓ |

## Environment

| | |
|---|---|
| SoC | Apple M4 Pro (Mac16,11) |
| Memory | 48 GB |
| macOS | 27.0 (26A428) |
| Python | 3.12.13 |
| mlx | 0.32.2 |
| coremltools | 9.0 |
| numpy | 2.1.3 |
| laya-apple | 1.0.2 @ 5eabbafc3cb6 |
| Shipped routing profile matches | no |

Configuration: quick, warmup 3, iters 20, latency measured in-process (one process for every configuration). Wall time 57 s.

## laya-typed-decisions

Revision `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`, weights sha256 `4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e`.

- MLX FP16 parity: passed
- ANE parity: passed
- auto routing (profile: local:<cache>/profiles/Apple_M4_Pro-macos27-coremltools9.0.json):
  - L64 q1: ane (validated_short_single_question_path)
  - L96 q1: ane (validated_short_single_question_path)
  - L128 q1: ane (validated_short_single_question_path)
  - L256 q1: gpu (sequence_exceeds_ane_auto_range)
  - L512 q1: gpu (sequence_exceeds_ane_auto_range)
  - L1024 q1: gpu (sequence_exceeds_ane_auto_range)
  - L64 q4: gpu (multiple_questions)
  - L128 q4: gpu (multiple_questions)
- heterogeneous: GPU-only 13.2 req/s, GPU+ANE 108.5 req/s (8.22x), 0 mismatches

Warm latency, in-process:

| L | q | device | forward P50/P95/P99 ms | predict P50/P95/P99 ms |
|---:|---:|---|---|---|
| 64 | 1 | gpu | 11.84 / 11.92 / 11.97 | 11.95 / 12.12 / 12.15 |
| 128 | 1 | gpu | 18.43 / 18.49 / 18.54 | 18.60 / 18.67 / 18.67 |
| 512 | 1 | gpu | 64.42 / 64.59 / 64.67 | 64.86 / 65.02 / 65.03 |
| 128 | 4 | gpu | 60.86 / 61.50 / 61.54 | 61.33 / 61.89 / 62.12 |
| 64 | 1 | ane | 7.50 / 7.54 / 7.55 | 7.65 / 7.66 / 7.67 |
| 96 | 1 | ane | 8.53 / 8.55 / 8.55 | 8.66 / 8.69 / 8.70 |
| 128 | 1 | ane | 9.32 / 9.33 / 9.33 | 9.45 / 9.46 / 9.47 |
