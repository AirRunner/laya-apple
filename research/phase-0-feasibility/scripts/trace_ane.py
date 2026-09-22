"""Runtime device evidence: record an Instruments trace while one backend runs a workload.

    uv run python scripts/trace_ane.py --model laya-typed-decisions --spec '{"backend":"ane","units":"cpu_ne"}' \
        --length 128 --seconds 8 --template "Core AI" --output raw/trace/typed-L128-cpu_ne

Starts the workload in a child process, attaches `xctrace record` to its PID, exports the
table of contents and every table whose schema mentions the neural engine / GPU / Core ML,
and writes an allow-listed summary (counts, durations, labels — never the process
environment, which a raw trace contains). Tracing perturbs timing: durations here are
diagnostic, not benchmark results.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

KEYWORDS = ("ane", "neural", "coreml", "core-ml", "coreai", "core-ai", "gpu", "metal", "inference", "model")


def workload(spec, length, seconds, ready):
    from backends import make_backend
    from common import agent_config, checkpoint, make_request
    from laya_coreml.tokenizer import Tokenizer

    b = make_backend({**spec, "length": length} if spec["backend"] in ("ane", "coreml") else spec)
    tok = Tokenizer(checkpoint(spec["model"]) / "tokenizer")
    state, qs, items = make_request(tok, agent_config(spec["model"]), length, 1)
    for _ in range(5):
        b.forward(items)
    Path(ready).write_text("ready")
    end, n = time.time() + seconds, 0
    while time.time() < end:
        b.forward(items)
        n += 1
    print("calls", n, flush=True)


def summarize_table(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    ids = {e.attrib["id"]: e for e in root.iter() if "id" in e.attrib}

    def val(e):
        if e is None:
            return None
        if "ref" in e.attrib:
            e = ids.get(e.attrib["ref"], e)
        return e.attrib.get("fmt", e.text)

    rows = root.findall(".//row")
    cols = [c.findtext("mnemonic") for c in root.findall(".//schema/col")]
    labels, total_ns, durations = {}, 0, []
    for r in rows:
        children = list(r)
        rec = {cols[i] if i < len(cols) else f"c{i}": val(c) for i, c in enumerate(children)}
        key = " | ".join(str(rec.get(k)) for k in rec if k and re.search(r"name|label|device|state|type|model", k))
        labels[key] = labels.get(key, 0) + 1
        d = rec.get("duration")
        if d is not None:
            try:
                ns = int(ids.get(r.find("duration").attrib.get("ref"), r.find("duration")).text)
                durations.append(ns)
                total_ns += ns
            except Exception:
                pass
    return {
        "rows": len(rows),
        "columns": cols,
        "row_kinds": dict(sorted(labels.items(), key=lambda kv: -kv[1])[:25]),
        "duration_total_ms": total_ns / 1e6,
        "duration_median_ms": (sorted(durations)[len(durations) // 2] / 1e6) if durations else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--template", default="Core AI")
    ap.add_argument("--output", required=True, help="directory for the allow-listed summary")
    ap.add_argument("--keep-trace", help="where to keep the full .trace (never committed)")
    args = ap.parse_args()
    if args.spec == "__workload__":
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.keep_trace or f"/tmp/laya-trace-{int(time.time())}")
    scratch.mkdir(parents=True, exist_ok=True)
    ready = scratch / "ready"
    spec = {**json.loads(args.spec), "model": args.model}
    code = (
        "import sys,json; sys.path.insert(0, %r); from trace_ane import workload; "
        "workload(json.loads(%r), %d, %f, %r)"
        % (str(Path(__file__).parent), json.dumps(spec), args.length, args.seconds + 30, str(ready))
    )
    child = subprocess.Popen([sys.executable, "-c", code])
    while not ready.exists():
        if child.poll() is not None:
            raise SystemExit("workload exited before ready")
        time.sleep(0.5)
    trace = scratch / "run.trace"
    rec = subprocess.run(
        ["xcrun", "xctrace", "record", "--template", args.template, "--attach", str(child.pid),
         "--time-limit", f"{int(args.seconds)}s", "--output", str(trace), "--no-prompt"],
        capture_output=True, text=True,
    )
    child.terminate()
    child.wait(timeout=30)
    summary = {
        "spec": spec,
        "length": args.length,
        "template": args.template,
        "record_returncode": rec.returncode,
        "record_stderr_tail": rec.stderr[-1500:],
        "tables": {},
    }
    if trace.exists():
        toc = scratch / "toc.xml"
        subprocess.run(["xcrun", "xctrace", "export", "--input", str(trace), "--toc", "--output", str(toc)], check=True)
        troot = ET.parse(toc).getroot()
        schemas = sorted({t.attrib.get("schema", "") for t in troot.iter("table")})
        summary["schemas"] = schemas
        for sch in schemas:
            if not any(k in sch.lower() for k in KEYWORDS):
                continue
            x = scratch / f"{sch}.xml"
            r = subprocess.run(
                ["xcrun", "xctrace", "export", "--input", str(trace), "--xpath",
                 f'/trace-toc/run[@number="1"]/data/table[@schema="{sch}"]', "--output", str(x)],
                capture_output=True, text=True,
            )
            if r.returncode == 0 and x.exists():
                try:
                    summary["tables"][sch] = summarize_table(x)
                except Exception as e:  # keep going; record what failed
                    summary["tables"][sch] = {"error": repr(e)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "tables"}, indent=1)[:3000])
    for k, v in summary["tables"].items():
        print(k, {kk: v.get(kk) for kk in ("rows", "duration_total_ms", "duration_median_ms")}, list(v.get("row_kinds", {}).items())[:4])
    if not args.keep_trace:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
