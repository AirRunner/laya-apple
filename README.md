# laya-apple

**Apple-native inference runtime for Laya with MLX GPU and Apple Neural Engine acceleration.**

`laya-apple` brings [Laya](https://huggingface.co/convaiinnovations/laya) to Apple Silicon with a unified runtime for local inference.

The project aims to provide:

* MLX / Metal GPU inference
* Core ML / Apple Neural Engine inference
* Automatic backend selection
* Laya checkpoint conversion
* Numerical and decision parity validation
* Reproducible latency, throughput, and energy benchmarks

The goal is simple:

> Run Laya efficiently on Apple Silicon without requiring users to understand MLX, Core ML, or Neural Engine execution details.

---

## Why

Laya is designed for fast contextual and typed decisions without invoking a large generative model for every classification or routing task.

Existing Apple Silicon implementations demonstrate that Laya can run efficiently on both the GPU and Apple Neural Engine, but support is currently fragmented across different ports, checkpoints, and execution paths.

`laya-apple` provides one runtime:

```text
                       Laya API
                          │
                          ▼
                  Runtime Dispatcher
                    /           \
                   /             \
              MLX / GPU      Core ML / ANE
                   \             /
                    \           /
                          ▼
                       Decision
```

Users interact with Laya.

The runtime handles the Apple-specific execution details.

---

## Project Status

Early development.

Initial work focuses on:

1. Reproducing existing MLX and Core ML baselines.
2. Supporting `laya-typed-decisions`.
3. Establishing PyTorch → MLX → Core ML parity.
4. Profiling Neural Engine performance across context lengths.
5. Building a unified GPU / ANE runtime.

---

## Planned Usage

```python
from laya_apple import Laya

model = Laya.from_pretrained(
    "convaiinnovations/laya-typed-decisions"
)

result = model.predict(
    context="""
    The user prefers local inference and wants to minimize
    unnecessary LLM calls.
    """,
    questions=[
        "Is this information useful for future decisions?",
        "Should this information be retained?"
    ],
)

print(result)
```

By default:

```python
device="auto"
```

The runtime selects the appropriate Apple backend.

Explicit backend selection will also be available:

```python
model = Laya.from_pretrained(
    "convaiinnovations/laya-typed-decisions",
    device="gpu",
)
```

or:

```python
model = Laya.from_pretrained(
    "convaiinnovations/laya-typed-decisions",
    device="ane",
)
```

---

## Backends

### MLX / Metal GPU

The MLX backend targets:

* longer contexts
* high-throughput workloads
* development and validation
* workloads where GPU execution is faster than ANE

```text
Laya
  ↓
MLX
  ↓
Metal GPU
```

### Core ML / Apple Neural Engine

The Core ML backend targets:

* low-latency decisions
* energy-efficient inference
* background workloads
* GPU offload
* portable Apple devices

```text
Laya
  ↓
Core ML
  ↓
Apple Neural Engine
```

---

## Automatic Device Selection

GPU and ANE have different performance characteristics.

Short sequences may benefit from Neural Engine execution, while longer sequences can currently favor the GPU.

`laya-apple` will expose a unified:

```python
device="auto"
```

mode using workload characteristics such as:

```text
sequence length
batch size
backend latency
GPU load
ANE availability
```

to select the execution backend.

Future versions will support concurrent GPU + ANE scheduling for mixed workloads.

---

## Long-Context ANE

One of the main technical focuses of the project is understanding why Neural Engine inference scales poorly as Laya context length increases.

The investigation includes:

```text
ModernBERT local attention
global attention
QKV projection
softmax
MLP
tensor layout
transpose / reshape overhead
Core ML graph partitioning
CPU fallback
memory movement
```

The project will first profile the execution graph before introducing model-specific optimizations.

Potential optimization areas include:

```text
ANE-friendly tensor layouts
true windowed local attention
chunked attention
reduced tensor transposes
Core ML graph restructuring
quantization
```

All optimizations must preserve model behavior.

---

## Correctness

Performance results are only accepted when model outputs remain consistent with the original implementation.

Validation will compare:

```text
PyTorch reference
       │
       ▼
      MLX
       │
       ▼
    Core ML
       │
       ▼
      ANE
```

Tests will cover:

* encoder output parity
* decision logits
* selected answers
* typed decisions
* multi-question behavior

---

## Benchmarks

Benchmarks will cover:

```text
L64
L128
L256
L512
L1024
```

Metrics:

```text
P50 latency
P95 latency
P99 latency
throughput
memory usage
energy per request
GPU utilization
ANE utilization
decision parity
```

Workloads will include:

```text
single request
batch inference
mixed sequence lengths
concurrent requests
```

---

## Roadmap

### v0.1 — Baseline

* [ ] Reproduce MLX inference
* [ ] Reproduce Core ML inference
* [ ] Establish reference benchmark suite
* [ ] Establish parity tests

### v0.2 — Typed Decisions

* [ ] Support `convaiinnovations/laya-typed-decisions`
* [ ] MLX backend
* [ ] Core ML backend
* [ ] ANE execution
* [ ] Cross-backend correctness validation

### v0.3 — Unified Runtime

* [ ] `Laya.from_pretrained()`
* [ ] `device="gpu"`
* [ ] `device="ane"`
* [ ] `device="auto"`
* [ ] Common output API

### v0.4 — ANE Optimization

* [ ] Profile long-context bottlenecks
* [ ] Optimize ModernBERT execution graph
* [ ] Improve L256 / L512 / L1024 scaling
* [ ] Evaluate quantization

### v0.5 — Heterogeneous Runtime

* [ ] Runtime cost model
* [ ] Sequence-aware routing
* [ ] Concurrent GPU + ANE execution
* [ ] Mixed-workload scheduler

---

## Repository Structure

```text
laya-apple/
├── laya_apple/
│   ├── backends/
│   │   ├── mlx.py
│   │   └── coreml.py
│   ├── scheduler/
│   ├── conversion/
│   ├── model.py
│   └── runtime.py
│
├── benchmarks/
├── tests/
├── scripts/
├── docs/
├── README.md
├── LICENSE
└── NOTICE
```

---

## Scope

`laya-apple` is not a new Laya model.

It is an Apple Silicon inference runtime for existing Laya checkpoints.

The project focuses on:

> **making Laya practical, efficient, and easy to use on Apple hardware.**

---

## License

License and attribution details will be finalized before the first release.

Upstream model and implementation licenses remain applicable to their respective components.
