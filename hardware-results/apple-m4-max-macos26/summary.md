# Hardware report: Apple M4 Max, macOS 26.6.2

Matrix row (docs/community-benchmarks.md):

| Mac | MLX | ANE | Auto uses ANE | Heterogeneous |
|---|---|---|---|---|
| Apple M4 Max (64 GB, macOS 26.6.2) | ✓ | ✓ | yes | ✓ |

## Environment

| | |
|---|---|
| SoC | Apple M4 Max (Mac16,9) |
| Memory | 64 GB |
| macOS | 26.6.2 (25G83) |
| Python | 3.12.14 |
| mlx | 0.32.2 |
| coremltools | 9.0 |
| numpy | 2.1.3 |
| laya-apple | 1.0.1 @ 114530d4e16d |
| Shipped routing profile matches | yes |

Configuration: quick, warmup 3, iters 20, latency measured in-process (one process for every configuration). Wall time 51 s.

## laya-typed-decisions

Revision `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`, weights sha256 `4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e`.

- MLX FP16 parity: passed
- ANE parity: passed
- auto routing (profile: shipped):
  - L64 q1: ane (validated_short_single_question_path)
  - L96 q1: ane (validated_short_single_question_path)
  - L128 q1: ane (validated_short_single_question_path)
  - L256 q1: gpu (sequence_exceeds_ane_auto_range)
  - L512 q1: gpu (sequence_exceeds_ane_auto_range)
  - L1024 q1: gpu (sequence_exceeds_ane_auto_range)
  - L64 q4: gpu (multiple_questions)
  - L128 q4: gpu (multiple_questions)
- heterogeneous: GPU-only 23.9 req/s, GPU+ANE 108.7 req/s (4.54x), 0 mismatches

Warm latency, in-process:

| L | q | device | forward P50/P95/P99 ms | predict P50/P95/P99 ms |
|---:|---:|---|---|---|
| 64 | 1 | gpu | 9.42 / 9.50 / 9.51 | 9.51 / 9.65 / 9.76 |
| 128 | 1 | gpu | 12.03 / 12.28 / 12.30 | 12.27 / 12.61 / 12.64 |
| 512 | 1 | gpu | 35.22 / 35.28 / 35.37 | 35.63 / 35.78 / 35.83 |
| 128 | 4 | gpu | 33.20 / 33.28 / 33.31 | 33.59 / 33.73 / 33.77 |
| 64 | 1 | ane | 8.10 / 8.18 / 8.21 | 8.21 / 8.24 / 8.28 |
| 96 | 1 | ane | 8.98 / 9.03 / 9.03 | 9.08 / 9.15 / 9.15 |
| 128 | 1 | ane | 9.91 / 9.98 / 10.00 | 10.06 / 10.25 / 10.48 |
