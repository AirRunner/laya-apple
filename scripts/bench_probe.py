"""Evidence for the runtime placement probe threshold (PROBE_MAX_RATIO, laya_apple/backends/coreml_ane.py).

For every registered ANE artifact, the fastest of N Core ML predictions of the probe input:
- on CPU_AND_NE, as the runtime loads it (the ANE case);
- on CPU_ONLY, the outcome the probe must catch: Core ML running the graph on the CPU.

The probe refuses a model whose loaded/CPU_ONLY ratio exceeds PROBE_MAX_RATIO; the ANE
ratios (`ane_over_cpu_max`) must sit well below it, and a CPU-resident model is ~1.0. Both
are also reported relative to the bucket's measured ANE service time from routing.json.

    LAYA_APPLE_CACHE=<cache> HF_HUB_OFFLINE=1 \\
        .venv/bin/python scripts/bench_probe.py [--runs 5] [--out benchmarks/v1.0/probe.json]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fastest(model, feats, runs):
    model.predict(feats)
    best = math.inf
    for _ in range(runs):
        t = time.perf_counter()
        model.predict(feats)
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3, help="independent probe measurements per configuration")
    p.add_argument("--out", default=str(ROOT / "benchmarks" / "v1.0" / "probe.json"))
    a = p.parse_args(argv)

    import coremltools as ct

    from laya_apple.artifacts import COMPILED, artifact_dir, load_verified, platform_profile
    from laya_apple.backends.coreml_ane import HostWeights, ane_features
    from laya_apple.hub import checkpoint_path
    from laya_apple.prompt import Tokenizer
    from laya_apple.registry import models, routing_table

    table = routing_table()["models"]
    rows = []
    for name, spec in models().items():
        ckpt = checkpoint_path(spec, local_files_only=True)
        enc = json.loads((ckpt / "encoder/config.json").read_text())
        host = HostWeights(ckpt, int(enc["local_attention"]))
        pad = Tokenizer(ckpt / "tokenizer").pad_token_id
        for b in spec.ane_buckets:
            expected = table[name]["service_ms"]["ane"]["1"][str(b)]
            feats = ane_features(
                [{"ids": [pad] * b, "markers": [1, 2], "qtype": 0}],
                b,
                1,
                host.embedding,
                host.type_embedding,
                host.window(b),
                pad,
            )
            ane_model, _ = load_verified(spec, b)
            cpu_model = ct.models.CompiledMLModel(
                str(artifact_dir(spec, b) / COMPILED), compute_units=ct.ComputeUnit.CPU_ONLY
            )
            ane = [fastest(ane_model, feats, a.runs) for _ in range(a.repeats)]
            cpu = [fastest(cpu_model, feats, a.runs) for _ in range(a.repeats)]
            row = {
                "model": name,
                "bucket": b,
                "expected_ane_service_ms": expected,
                "ane_probe_ms": [round(x, 3) for x in ane],
                "cpu_only_probe_ms": [round(x, 3) for x in cpu],
                "ane_ratio_max": round(max(ane) / expected, 3),
                "cpu_ratio_min": round(min(cpu) / expected, 3),
                "ane_over_cpu_max": round(max(ane) / min(cpu), 3),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "platform": platform_profile(),
                "runs": a.runs,
                "repeats": a.repeats,
                "rows": rows,
                "ane_ratio_max": max(r["ane_ratio_max"] for r in rows),
                "cpu_ratio_min": min(r["cpu_ratio_min"] for r in rows),
                "ane_over_cpu_max": max(r["ane_over_cpu_max"] for r in rows),
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
