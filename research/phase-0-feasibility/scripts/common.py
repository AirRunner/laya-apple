"""Shared helpers for Phase -1 experiments: pinned models, workloads, stats, environment.

Research code only. Nothing here is imported by the `laya_apple` runtime package.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

RESEARCH = Path(__file__).resolve().parents[1]
RAW = RESEARCH / "raw"
REPORTS = RESEARCH / "reports"
# Converted packages are reproducible and large, so they live on the data volume.
ARTIFACTS = Path(
    os.environ.get("LAYA_APPLE_ARTIFACTS", "<research-artifacts>")
)

# Hugging Face revisions. `laya` is pinned to the revision the MLX and Core ML ports
# used; its model.safetensors is byte-identical to the current head 1c5edc17.
MODELS = {
    "laya": {
        "repo": "convaiinnovations/laya",
        "revision": "c5d78730f3493e4fe16d61507ef4b78eef7318cf",
        "weights_sha256": "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
    },
    "laya-multilingual": {
        "repo": "convaiinnovations/laya-multilingual",
        "revision": "052592a15d198d9ad47da779604259b10b47b7aa",
        "weights_sha256": "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204",
    },
    "laya-typed-decisions": {
        "repo": "convaiinnovations/laya-typed-decisions",
        "revision": "f9ab0b228f0fc0f14d873dbc99038f135c2da1b2",
        "weights_sha256": "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e",
    },
}
LENGTHS = (64, 96, 128, 256, 512, 1024)
REFERENCE_REPOS = {
    "NandhaKishorM/laya": "573e5b62696ba441230cd6be71d593331b5d23af",
    "mizorewww/laya-mlx": "0a859518634112655cb97c745dbf04f5191aaf13",
    "mizorewww/laya-coreml": "4619e0483f07adf39068532e85b42ec2347edb83",
}


@functools.cache
def checkpoint(name: str) -> Path:
    from huggingface_hub import snapshot_download

    spec = MODELS[name]
    path = Path(
        snapshot_download(
            spec["repo"],
            revision=spec["revision"],
            allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"],
        )
    )
    return path


def agent_config(name: str) -> dict:
    return json.loads((checkpoint(name) / "rl_agent_config.json").read_text())


def max_len(name: str) -> int:
    return int(agent_config(name)["max_len"])


def valid_lengths(name: str, lengths=LENGTHS) -> list[int]:
    return [n for n in lengths if n <= max_len(name)]


# --------------------------------------------------------------------------- workloads

# Generated question pool (not copied from any fixture). Mixes the three typed primitives.
QUESTION_POOL = [
    {
        "type": "noul",
        "instructions": "Does the customer explicitly ask for a refund?",
    },
    {
        "type": "choice",
        "instructions": "Which team should own this ticket?",
        "criteria": {
            "billing": "charges, invoices, refunds",
            "technical": "errors, outages, broken features",
            "account": "login, profile, permissions",
            "other": "anything else",
        },
    },
    {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["can wait", "should be handled today", "blocking, handle now"],
    },
    {"type": "noul", "instructions": "Is the customer threatening to cancel the service?"},
    {
        "type": "choice",
        "instructions": "What is the overall sentiment of the message?",
        "criteria": ["positive", "neutral", "negative"],
    },
    {"type": "noul", "instructions": "Does the message mention a specific invoice number?"},
    {
        "type": "score",
        "instructions": "How clearly does the customer describe the problem?",
        "criteria": ["unclear", "partly clear", "clear", "very clear with evidence"],
    },
    {"type": "noul", "instructions": "Should this ticket be escalated to a human supervisor?"},
]

_SUBJECTS = ["invoice", "subscription", "account", "order", "payment", "license", "renewal"]
_VERBS = ["was charged", "was billed", "got an error", "cannot log in", "was refused", "saw a delay"]
_DETAILS = [
    "twice this month",
    "after the latest update",
    "on the annual plan",
    "for the second time",
    "despite the earlier ticket",
    "without any notice",
]
_ASKS = [
    "Please refund the duplicate amount.",
    "Can someone look into this today?",
    "I would like an explanation.",
    "Otherwise we will cancel the plan.",
    "Please confirm once it is fixed.",
]


def filler_sentences(seed: int):
    """Endless deterministic stream of generated support-ticket sentences."""
    rng = random.Random(seed)
    while True:
        yield (
            f"The {rng.choice(_SUBJECTS)} {rng.randint(1000, 9999)} {rng.choice(_VERBS)} "
            f"{rng.choice(_DETAILS)}. {rng.choice(_ASKS)}"
        )


def questions_for(count: int, offset: int = 0) -> dict:
    return {f"q{i}": QUESTION_POOL[(offset + i) % len(QUESTION_POOL)] for i in range(count)}


def make_request(tok, cfg: dict, length: int, n_questions: int = 1, seed: int = 0):
    """Return (state, questions, items) whose longest prompt row is exactly `length` tokens.

    The state is grown word by word until the longest row reaches `length`. The prompt is
    built with the upstream-compatible `build_sequence`, so what is measured is exactly what
    the runtime would send. Raises if an exact length cannot be hit (never silently pads).
    """
    from laya_mlx.common import QTYPES, build_sequence

    questions = questions_for(n_questions, offset=seed)
    internal = []
    for q in questions.values():
        crit = q.get("criteria")
        if q["type"] == "choice" and isinstance(crit, list):
            crit = dict.fromkeys(crit)
        internal.append({"t": q["type"], "ins": q["instructions"], "crit": crit})
    mlen, hlen = int(cfg["max_len"]), int(cfg.get("head_max_len", 192))
    if length > mlen:
        raise ValueError(f"length {length} exceeds checkpoint max_len {mlen}")

    def build(words):
        state = " ".join(words)
        items = []
        for q in internal:
            ids, markers = build_sequence(tok, state, q, mlen, hlen)
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
        return state, items

    words: list[str] = []
    stream = filler_sentences(seed)
    sentence: list[str] = []
    for _ in range(20000):
        state, items = build(words)
        longest = max(len(it["ids"]) for it in items)
        if longest == length:
            return state, questions, items
        if longest > length:
            break
        if not sentence:
            sentence = next(stream).split()
        words.append(sentence.pop(0))
    # Overshot by a multi-token word: back off one word and grow with single short tokens.
    words = words[:-1]
    for pad in (["a"] * 64):
        state, items = build(words)
        longest = max(len(it["ids"]) for it in items)
        if longest == length:
            return state, questions, items
        if longest > length:
            break
        words.append(pad)
    raise RuntimeError(f"could not build an exact {length}-token request")


# --------------------------------------------------------------------------- stats / io


def stats(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def save_json(path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)
    return path


def append_jsonl(path, record) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def tree_sha256(path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(path)).encode())
            h.update(bytes.fromhex(sha256_file(f)))
    return h.hexdigest()


def _sh(*cmd) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def conditions() -> dict:
    """Cheap per-run conditions: thermal/power state and background load."""
    load1, load5, load15 = os.getloadavg()
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "loadavg": [load1, load5, load15],
        "thermal": _sh("pmset", "-g", "therm"),
        "power_source": _sh("pmset", "-g", "ps").splitlines()[0:1],
        "low_power_mode": _sh("pmset", "-g").count("lowpowermode 1") > 0,
    }


def environment() -> dict:
    packages = {}
    for name in (
        "torch",
        "numpy",
        "coremltools",
        "transformers",
        "tokenizers",
        "safetensors",
        "huggingface-hub",
        "mlx",
        "mlx-metal",
        "laya",
        "laya-mlx",
        "laya-coreml",
        "psutil",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    git = _sh("git", "-C", str(RESEARCH), "rev-parse", "HEAD")
    dirty = bool(_sh("git", "-C", str(RESEARCH), "status", "--porcelain", "--", "scripts"))
    return {
        "soc": _sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "cpu_cores": {
            "total": int(_sh("sysctl", "-n", "hw.ncpu") or 0),
            "performance": int(_sh("sysctl", "-n", "hw.perflevel0.physicalcpu") or 0),
            "efficiency": int(_sh("sysctl", "-n", "hw.perflevel1.physicalcpu") or 0),
        },
        "memory_bytes": int(_sh("sysctl", "-n", "hw.memsize") or 0),
        "macos": platform.mac_ver()[0],
        "os_build": _sh("sw_vers", "-buildVersion"),
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": packages,
        "research_git_revision": git,
        "research_scripts_dirty": dirty,
        "reference_repos": REFERENCE_REPOS,
        "models": MODELS,
        "conditions": conditions(),
    }


class Timer:
    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter_ns() - self.start) / 1e6


def rss_bytes() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def peak_rss_bytes() -> int:
    import resource

    # macOS reports ru_maxrss in bytes.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
