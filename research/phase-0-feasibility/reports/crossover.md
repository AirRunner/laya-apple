# GPU / ANE crossover

Forward (model-only) P50, pass A. Ratio > 1 means ANE is faster.

## laya

### 1 question(s), ANE B=1 sequential vs MLX batched — crossover ≈ L227

| L | MLX FP16 | Core ML GPU (fixed) | ANE (CPU_AND_NE) | MLX/ANE | faster |
|---:|---:|---:|---:|---:|---|
| 64 | 9.40 | 8.51 | 8.10 | 1.16 | ANE |
| 96 | 11.59 | 10.73 | 9.00 | 1.29 | ANE |
| 128 | 12.27 | 12.60 | 9.86 | 1.24 | ANE |
| 256 | 19.13 | 20.08 | 20.03 | 0.96 | GPU |
| 512 | 35.22 | 36.19 | 54.51 | 0.65 | GPU |

### 4 question(s), ANE B=1 sequential vs MLX batched — GPU faster at every measured length

| L | MLX FP16 | Core ML GPU (fixed) | ANE (CPU_AND_NE) | MLX/ANE | faster |
|---:|---:|---:|---:|---:|---|
| 64 | 18.85 | 33.72 | 32.41 | 0.58 | GPU |
| 96 | 26.40 | 42.88 | 35.99 | 0.73 | GPU |
| 128 | 33.19 | 50.29 | 39.41 | 0.84 | GPU |
| 256 | 63.48 | 80.25 | 79.63 | 0.80 | GPU |
| 512 | 123.97 | 144.76 | 217.44 | 0.57 | GPU |

### 8 question(s), ANE B=1 sequential vs MLX batched — GPU faster at every measured length

| L | MLX FP16 | Core ML GPU (fixed) | ANE (CPU_AND_NE) | MLX/ANE | faster |
|---:|---:|---:|---:|---:|---|
| 64 | 32.77 | — | 64.82 | 0.51 | GPU |
| 96 | 46.82 | — | 71.94 | 0.65 | GPU |
| 128 | 62.09 | — | 78.91 | 0.79 | GPU |
| 256 | 118.65 | — | 160.21 | 0.74 | GPU |
| 512 | 241.55 | — | 436.15 | 0.55 | GPU |

## laya-multilingual

### 1 question(s), ANE B=1 sequential vs MLX batched — crossover ≈ L264

| L | MLX FP16 | Core ML GPU (fixed) | ANE (CPU_AND_NE) | MLX/ANE | faster |
|---:|---:|---:|---:|---:|---|
| 64 | 5.42 | 4.84 | 3.43 | 1.58 | ANE |
| 96 | 6.36 | 5.32 | 3.85 | 1.65 | ANE |
| 128 | 6.47 | 5.98 | 4.31 | 1.50 | ANE |
| 256 | 8.56 | 9.23 | 8.35 | 1.03 | ANE |
| 320 | 10.06 | — | 11.86 | 0.85 | GPU |
| 384 | 11.69 | — | 13.92 | 0.84 | GPU |
| 448 | 13.40 | — | 18.14 | 0.74 | GPU |
| 512 | 14.97 | 14.96 | 22.75 | 0.66 | GPU |
| 1024 | 29.11 | 29.15 | 82.74 | 0.35 | GPU |

### 4 question(s), ANE B=1 sequential vs MLX batched — GPU faster at every measured length

| L | MLX FP16 | Core ML GPU (fixed) | ANE (CPU_AND_NE) | MLX/ANE | faster |
|---:|---:|---:|---:|---:|---|
| 64 | 8.30 | 20.02 | 13.73 | 0.60 | GPU |
| 96 | 11.06 | 20.98 | 15.56 | 0.71 | GPU |
| 128 | 13.86 | 23.93 | 17.01 | 0.81 | GPU |
| 256 | 24.56 | 36.91 | 33.38 | 0.74 | GPU |
| 512 | 48.42 | 59.92 | 90.71 | 0.53 | GPU |
| 1024 | 105.28 | 116.66 | 331.32 | 0.32 | GPU |

### 8 question(s), ANE B=1 sequential vs MLX batched — GPU faster at every measured length

| L | MLX FP16 | Core ML GPU (fixed) | ANE (CPU_AND_NE) | MLX/ANE | faster |
|---:|---:|---:|---:|---:|---|
| 64 | 13.65 | — | 27.36 | 0.50 | GPU |
| 96 | 19.30 | — | 31.14 | 0.62 | GPU |
| 128 | 23.56 | — | 34.40 | 0.68 | GPU |
| 256 | 45.11 | — | 66.70 | 0.68 | GPU |
| 512 | 92.79 | — | 181.32 | 0.51 | GPU |
| 1024 | 204.70 | — | 662.04 | 0.31 | GPU |

### Batched ANE exports (B questions in one Core ML call) vs MLX with the same question count

| B | L | ANE batched | ANE B=1 sequential | MLX FP16 | MLX/ANE-batched |
|---:|---:|---:|---:|---:|---:|
| 4 | 64 | 7.15 | 13.73 | 8.30 | 1.16 |
| 4 | 128 | 15.23 | 17.01 | 13.86 | 0.91 |
| 4 | 256 | 37.76 | 33.38 | 24.56 | 0.65 |
| 8 | 64 | 14.03 | 27.36 | 13.65 | 0.97 |
| 8 | 128 | 32.48 | 34.40 | 23.56 | 0.73 |
| 8 | 256 | 92.55 | 66.70 | 45.11 | 0.49 |

## laya-typed-decisions

### 1 question(s), ANE B=1 sequential vs MLX batched — crossover ≈ L151

| L | MLX FP16 | Core ML GPU (fixed) | ANE (CPU_AND_NE) | MLX/ANE | faster |
|---:|---:|---:|---:|---:|---|
| 64 | 9.39 | 8.52 | 8.07 | 1.16 | ANE |
| 96 | 11.57 | 10.69 | 9.02 | 1.28 | ANE |
| 128 | 12.23 | 12.65 | 9.93 | 1.23 | ANE |
| 160 | 14.69 | — | 15.73 | 0.93 | GPU |
| 192 | 15.48 | — | 16.76 | 0.92 | GPU |
| 224 | 18.58 | — | 18.39 | 1.01 | ANE |
| 256 | 19.23 | 20.05 | 20.07 | 0.96 | GPU |
| 512 | 35.21 | 36.19 | 54.43 | 0.65 | GPU |
| 1024 | 71.00 | 71.06 | 169.57 | 0.42 | GPU |

### 4 question(s), ANE B=1 sequential vs MLX batched — GPU faster at every measured length

| L | MLX FP16 | Core ML GPU (fixed) | ANE (CPU_AND_NE) | MLX/ANE | faster |
|---:|---:|---:|---:|---:|---|
| 64 | 18.77 | 34.11 | 32.29 | 0.58 | GPU |
| 96 | 26.32 | 42.61 | 36.06 | 0.73 | GPU |
| 128 | 33.16 | 50.56 | 39.67 | 0.84 | GPU |
| 256 | 63.46 | 80.19 | 80.18 | 0.79 | GPU |
| 512 | 123.95 | 144.83 | 217.38 | 0.57 | GPU |
| 1024 | 262.30 | 284.23 | 679.10 | 0.39 | GPU |

### 8 question(s), ANE B=1 sequential vs MLX batched — GPU faster at every measured length

| L | MLX FP16 | Core ML GPU (fixed) | ANE (CPU_AND_NE) | MLX/ANE | faster |
|---:|---:|---:|---:|---:|---|
| 64 | 32.73 | — | 64.60 | 0.51 | GPU |
| 96 | 46.80 | — | 72.08 | 0.65 | GPU |
| 128 | 62.07 | — | 78.93 | 0.79 | GPU |
| 256 | 118.66 | — | 159.90 | 0.74 | GPU |
| 512 | 241.48 | — | 435.84 | 0.55 | GPU |
| 1024 | 518.08 | — | 1357.88 | 0.38 | GPU |

### Batched ANE exports (B questions in one Core ML call) vs MLX with the same question count

| B | L | ANE batched | ANE B=1 sequential | MLX FP16 | MLX/ANE-batched |
|---:|---:|---:|---:|---:|---:|
| 4 | 64 | 18.40 | 32.29 | 18.77 | 1.02 |
| 4 | 128 | 41.59 | 39.67 | 33.16 | 0.80 |
| 4 | 256 | 92.49 | 80.18 | 63.46 | 0.69 |
| 8 | 64 | 40.82 | 64.60 | 32.73 | 0.80 |
| 8 | 128 | 87.49 | 78.93 | 62.07 | 0.71 |
| 8 | 256 | 207.19 | 159.90 | 118.66 | 0.57 |

