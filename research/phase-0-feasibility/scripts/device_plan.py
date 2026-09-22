"""Core ML anticipated device placement (MLComputePlan) summary.

This is Core ML's *plan* — which device each MIL operation is expected to run on and its
estimated relative cost — not a hardware trace. Runtime evidence comes from latency
controls (CPU_ONLY vs CPU_AND_NE) and Instruments (scripts/trace_ane.py).

Adapted from laya-coreml @ 4619e04 benchmarks/common.py::compute_plan (Apache-2.0), extended
with per-operator breakdown and device-transition (partition boundary) counting.
"""

from __future__ import annotations

from collections import Counter, defaultdict

SHORT = {
    "MLCPUComputeDevice": "cpu",
    "MLGPUComputeDevice": "gpu",
    "MLNeuralEngineComputeDevice": "ane",
}


def compute_plan(mlmodel, units: str, keep_ops: bool = False) -> dict:
    import coremltools as ct

    from backends import COMPUTE_UNITS

    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        mlmodel.get_compiled_model_path(), compute_units=getattr(ct.ComputeUnit, COMPUTE_UNITS[units])
    )
    preferred, supported = Counter(), Counter()
    cost = defaultdict(float)
    by_op = defaultdict(Counter)
    cost_by_op = defaultdict(lambda: defaultdict(float))
    sequence = []
    ops = []

    def visit(block):
        for op in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            est = plan.get_estimated_cost_for_mlprogram_operation(op)
            dev = SHORT.get(type(usage.preferred_compute_device).__name__, "?") if usage else "none"
            sup = sorted(SHORT.get(type(d).__name__, "?") for d in usage.supported_compute_devices) if usage else []
            name = op.operator_name
            preferred[dev] += 1
            supported.update(sup)
            by_op[name][dev] += 1
            if est:
                cost[dev] += est.weight
                cost_by_op[name][dev] += est.weight
            if dev != "none" and name != "const":
                sequence.append(dev)
            if keep_ops:
                ops.append({"op": name, "device": dev, "supported": sup, "cost": est.weight if est else None})
            for nested in op.blocks:
                visit(nested)

    for fn in plan.model_structure.program.functions.values():
        visit(fn.block)
    transitions = sum(1 for a, b in zip(sequence, sequence[1:]) if a != b)
    segments = Counter()
    if sequence:
        cur = sequence[0]
        for d in sequence[1:]:
            if d != cur:
                segments[cur] += 1
                cur = d
        segments[cur] += 1
    total_cost = sum(cost.values()) or 1.0
    result = {
        "kind": "MLComputePlan (anticipated placement, not a runtime trace)",
        "compute_units": COMPUTE_UNITS[units],
        "preferred_counts": dict(preferred),
        "supported_counts": dict(supported),
        "estimated_cost_share": {k: v / total_cost for k, v in cost.items()},
        "device_transitions_in_program_order": transitions,
        "contiguous_segments_per_device": dict(segments),
        "by_operator": {k: dict(v) for k, v in sorted(by_op.items())},
        "cost_share_by_operator": {
            k: {d: w / total_cost for d, w in v.items()} for k, v in sorted(cost_by_op.items())
        },
    }
    if keep_ops:
        result["operations"] = ops
    return result


def headline(plan: dict) -> str:
    c = plan["preferred_counts"]
    s = plan["estimated_cost_share"]
    return (
        f"ops ane={c.get('ane', 0)} gpu={c.get('gpu', 0)} cpu={c.get('cpu', 0)} "
        f"| cost ane={s.get('ane', 0):.3f} gpu={s.get('gpu', 0):.3f} cpu={s.get('cpu', 0):.3f} "
        f"| transitions={plan['device_transitions_in_program_order']}"
    )
