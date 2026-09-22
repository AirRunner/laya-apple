"""Thin, uniform wrappers over the pinned reference implementations.

Every backend exposes:
  prepare(state, questions) -> items          token ids / markers / qtype (CPU)
  forward(items) -> (logits[B,K], action[B,A])  model-only, synchronous, numpy float32
  predict(state, questions) -> public result    complete end-to-end call
  describe() -> dict                            what actually runs (device, dtype, shape)

Backends:
  torch        upstream NandhaKishorM/laya Agent (PyTorch; cpu or mps)
  mlx          mizorewww/laya-mlx Agent (MLX / Metal GPU)
  coreml       mizorewww/laya-coreml ordinary export (Core ML, one compute-unit setting)
  ane          adapted BC1S ConvBody export (Core ML fixed shape, one compute-unit setting)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from common import ARTIFACTS, checkpoint, sha256_file, tree_sha256

COMPUTE_UNITS = {
    "cpu_only": "CPU_ONLY",
    "cpu_gpu": "CPU_AND_GPU",
    "cpu_ne": "CPU_AND_NE",
    "all": "ALL",
}


_HASHES: dict = {}


def _weights_sha256(path) -> str:
    """Source-weight integrity check, cached per process so it is paid once, not per bucket."""
    key = (str(path), Path(path).stat().st_mtime_ns)
    if key not in _HASHES:
        _HASHES[key] = sha256_file(path)
    return _HASHES[key]


class Backend:
    name = "base"

    def describe(self) -> dict:
        raise NotImplementedError


# ----------------------------------------------------------------------------- torch


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, model: str, device: str = "cpu"):
        import laya
        import torch

        torch.set_grad_enabled(False)
        self.torch = torch
        self.model_name = model
        self.agent = laya.load(str(checkpoint(model)), device=device)
        if self.agent.device.type != device:
            raise RuntimeError(f"upstream fell back to {self.agent.device}, requested {device}")
        self.device = device

    def describe(self):
        return {"backend": "torch", "device": self.device, "dtype": str(self.agent.dtype)}

    def prepare(self, state, questions):
        from laya.common import QTYPES, build_sequence, render_options

        a = self.agent
        items = []
        for qid, qdef in questions.items():
            q = a._to_internal(qdef)
            ids, markers = build_sequence(
                a.tok, state, q, a.cfg.get("max_len", 512), a.cfg.get("head_max_len", 192)
            )
            assert len(markers) == len(render_options(q)), qid
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
        return items

    def forward(self, items):
        from laya.common import collate_items

        b = collate_items([items], self.agent.tok.pad_token_id)
        dev = self.agent.device
        logits, act = self.agent.model(
            b["input_ids"].to(dev),
            b["attention_mask"].to(dev),
            b["marker_pos"].to(dev),
            b["marker_mask"].to(dev),
            b["qtype"].to(dev),
        )
        # .cpu() synchronises MPS; CPU is synchronous already.
        return logits.float().cpu().numpy(), act.float().cpu().numpy()

    def predict(self, state, questions):
        return self.agent.predict(state, questions)

    def memory(self):
        if self.device == "mps":
            return {
                "mps_current_allocated": int(self.torch.mps.current_allocated_memory()),
                "mps_driver_allocated": int(self.torch.mps.driver_allocated_memory()),
            }
        return {}


# ----------------------------------------------------------------------------- mlx


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self, model: str, dtype: str = "float16", compile: bool = False):
        import laya_mlx
        import mlx.core as mx

        self.mx = mx
        self.model_name = model
        self.dtype = dtype
        self.compiled = compile
        # batch_size=64 so every measured request is one forward pass (as the MLX port's
        # own benchmark does); the public default of 16 would chunk 32+ question requests.
        self.agent = laya_mlx.load(str(checkpoint(model)), dtype=dtype, batch_size=64, compile=compile)

    def describe(self):
        return {
            "backend": "mlx",
            "device": str(self.agent.device),
            "dtype": self.dtype,
            "compile": self.compiled,
        }

    def prepare(self, state, questions):
        return self.agent.prepare(state, questions)[0]

    def forward(self, items):
        from laya_mlx.agent import collate_items

        batch = collate_items(items, self.agent.tok.pad_token_id, max_length=self.agent.cfg["max_len"])
        logits, act = self.agent.forward(batch)
        return np.asarray(logits, np.float32), np.asarray(act, np.float32)

    def predict(self, state, questions):
        return self.agent.predict(state, questions)

    def memory(self):
        mx = self.mx
        return {
            "mlx_active": int(mx.get_active_memory()),
            "mlx_peak": int(mx.get_peak_memory()),
            "mlx_cache": int(mx.get_cache_memory()),
        }

    def reset_peak(self):
        self.mx.reset_peak_memory()


# ----------------------------------------------------------------------------- core ml


def load_mlmodel(package, units):
    import coremltools as ct

    return ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, COMPUTE_UNITS[units]))


class CoreMLBackend(Backend):
    """laya-coreml ordinary export (explicit/SDPA attention, B x L x C layout)."""

    name = "coreml"

    def __init__(self, model: str, units: str = "cpu_gpu", fixed: int | None = None):
        from laya_coreml.agent import Agent

        self.model_name = model
        self.units = units
        self.fixed = fixed
        tag = model if fixed is None else f"{model}-fixed{fixed}-sdpa"
        self.export = ARTIFACTS / "coreml-ordinary" / tag
        # Enumerated-shape export: the RangeDim + CPU_AND_GPU guard in laya-coreml does not apply.
        # laya-coreml spells CPU_ONLY as "cpu"; keep the harness key for describe().
        self.agent = Agent(self.export, compute_units={"cpu_only": "cpu"}.get(units, units))
        self.shape = self.agent.shape

    def describe(self):
        return {
            "backend": "coreml",
            "graph": "ordinary-enumerated" if self.fixed is None else "ordinary-fixed",
            "compute_units": COMPUTE_UNITS[self.units],
            "shape": self.shape,
            "precision": self.agent.manifest.get("precision"),
            "attention": self.agent.manifest.get("attention"),
        }

    def prepare(self, state, questions):
        return self.agent.prepare(state, questions)[0]

    def batches(self, items):
        from laya_coreml.inputs import collate_items

        out = []
        for start in range(0, len(items), self.agent.batch_size):
            out.append(
                collate_items(
                    items[start : start + self.agent.batch_size],
                    self.agent.tok.pad_token_id,
                    shape=self.shape,
                    max_length=self.agent.cfg.get("max_len", 512),
                )
            )
        return out

    def forward(self, items):
        logits, acts = [], []
        for batch in self.batches(items):
            lg, ac = self.agent.forward(batch)
            n = min(len(items) - len(logits), lg.shape[0])
            logits.extend(lg[:n])
            acts.extend(ac[:n])
        return np.stack(logits), np.stack(acts)

    def predict(self, state, questions):
        return self.agent.predict(state, questions)

    @property
    def mlmodel(self):
        return self.agent.model


class ANEBackend(Backend):
    """Adapted BC1S ConvBody, one fixed-length package; host embedding lookup + action head.

    Runtime logic adapted from laya-coreml @ 4619e04 laya_coreml/ane.py (Apache-2.0).
    """

    name = "ane"

    def __init__(self, model: str, length: int, units: str = "cpu_ne", batch: int = 1, variant: str = "masked"):
        from laya_coreml.common import read_temperatures
        from laya_coreml.tokenizer import Tokenizer
        from safetensors import safe_open

        self.model_name = model
        self.length = length
        self.units = units
        self.batch = batch
        self.variant = variant
        self.source = checkpoint(model)
        self.package_dir = ane_package_dir(model, length, batch, variant)
        self.manifest = json.loads((self.package_dir / "manifest.json").read_text())
        if self.manifest["source_weights_sha256"] != _weights_sha256(self.source / "model.safetensors"):
            raise ValueError("package was converted from different weights")
        expect = {"batch": batch, "length": length, "options": 32}
        if self.manifest["shape"] != expect or self.manifest["variant"] != variant:
            raise ValueError(f"package manifest {self.manifest['shape']}/{self.manifest['variant']} != requested {expect}/{variant}")
        self.cfg = json.loads((self.source / "rl_agent_config.json").read_text())
        (self.temperature, self.temperature_by_options, *_rest) = read_temperatures(self.cfg)
        self.tok = Tokenizer(self.source / "tokenizer")
        self.model = load_mlmodel(self.package_dir / "model.mlpackage", units)
        enc = json.loads((self.source / "encoder/config.json").read_text())
        self.width = int(enc["hidden_size"])
        with safe_open(str(self.source / "model.safetensors"), framework="numpy") as w:
            self.embedding = w.get_tensor("encoder.embeddings.tok_embeddings.weight")
            self.type_embedding = w.get_tensor("type_emb.weight")
            self.action = {
                k: w.get_tensor("act_head." + k).astype(np.float32)
                for k in ("0.weight", "0.bias", "2.weight", "2.bias")
            }
        pos = np.arange(length)
        self.window = np.abs(pos[:, None] - pos[None, :]) <= int(enc.get("local_attention", 128)) // 2
        self._erf = np.vectorize(math.erf, otypes=[np.float64])
        spec = self.model.get_spec()
        self.output_names = [o.name for o in spec.description.output]

    def describe(self):
        return {
            "backend": "coreml",
            "graph": f"ane-bc1s-{self.variant}",
            "compute_units": COMPUTE_UNITS[self.units],
            "shape": {"batch": self.batch, "length": self.length, "options": 32},
            "precision": "float16",
            "package_sha256": self.manifest.get("package_sha256"),
        }

    @property
    def mlmodel(self):
        return self.model

    def prepare(self, state, questions):
        from laya_coreml.prompt import PromptMixin

        class _P(PromptMixin):
            pass

        p = _P()
        p.tok, p.cfg = self.tok, self.cfg
        return p.prepare(state, questions)[0]

    def model_inputs(self, rows):
        return ane_features(
            rows, self.length, self.batch, self.embedding, self.type_embedding, self.window,
            self.tok.pad_token_id,
        )

    def batches(self, items):
        return [items[s : s + self.batch] for s in range(0, len(items), self.batch)]

    def _tail(self, outputs, rows):
        logits = pooled = None
        for v in outputs.values():
            if v.shape[1] == 1:
                logits = v.reshape(v.shape[0], -1)[: len(rows)].astype(np.float32)
            else:
                pooled = v.reshape(v.shape[0], -1)[: len(rows)].astype(np.float32)
        mm = np.zeros((len(rows), 32), bool)
        for r, it in enumerate(rows):
            mm[r, : len(it["markers"])] = True
        logits = np.where(mm, logits, -1e4)
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        k = np.maximum(mm.sum(-1), 2).astype(np.float32)
        ent = -(p * np.log(np.maximum(p, 1e-9))).sum(-1) / np.log(k)
        top = np.sort(p, -1)[:, -2:]
        feats = np.stack((top[:, 1], top[:, 1] - top[:, 0], ent, k / 255.0), -1)
        h = np.concatenate((pooled, feats), -1) @ self.action["0.weight"].T + self.action["0.bias"]
        h = h * (1 + self._erf(h / math.sqrt(2)).astype(np.float32)) / 2  # exact erf GELU
        act = h @ self.action["2.weight"].T + self.action["2.bias"]
        return logits, act.astype(np.float32)

    def run_features(self, feats):
        return self.model.predict(feats)

    def forward(self, items):
        logits, acts = [], []
        for rows in self.batches(items):
            lg, ac = self._tail(self.model.predict(self.model_inputs(rows)), rows)
            logits.append(lg)
            acts.append(ac)
        return np.concatenate(logits), np.concatenate(acts)

    def predict(self, state, questions):
        from laya_coreml.common import confidence_from_probs, temp_bucket

        items = self.prepare(state, questions)
        logits, act = self.forward(items)
        act = np.exp(act - act.max(-1, keepdims=True))
        act /= act.sum(-1, keepdims=True)
        answers = {}
        for row, (qid, qdef) in enumerate(questions.items()):
            it = items[row]
            k, qt = len(it["markers"]), it["qtype"]
            scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            z = logits[row, :k] / scale
            p = np.exp(z - z.max())
            p /= p.sum()
            answers[qid] = public_answer(qdef, p, k, act[row, 0], confidence_from_probs)
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": sum(len(i["ids"]) for i in items), "output_tokens": 0},
        }


def ane_features(rows, L, B, embedding, type_embedding, window, pad_id):
    """Fixed-shape feature dict for up to B prepared rows. Refuses oversize inputs."""
    n = len(rows)
    if n > B:
        raise ValueError("too many rows for this export")
    ids = np.full((B, L), pad_id, np.int64)
    valid = np.zeros((B, L), bool)
    valid[:, 0] = True  # dummy rows still need one valid key
    qtype = np.zeros(B, np.int64)
    marker_map = np.zeros((B, L, 1, 32), np.float16)
    for r, it in enumerate(rows):
        m = len(it["ids"])
        if m > L:
            raise ValueError(f"input has {m} tokens; this export holds {L} (no truncation)")
        if len(it["markers"]) > 32:
            raise ValueError("more than 32 options")
        ids[r, :m] = it["ids"]
        valid[r, :m] = True
        qtype[r] = it["qtype"]
        marker_map[r, it["markers"], 0, np.arange(len(it["markers"]))] = 1
    emb = embedding[ids].transpose(0, 2, 1)[:, :, None, :]
    full = np.broadcast_to(valid[:, None, :], (B, L, L))
    local = (window[None] | ~valid[:, :, None]) & full
    # attention scores are laid out [B, key, 1, query]
    feats = {
        name: np.where(value.transpose(0, 2, 1)[:, :, None, :], 0, -1e4).astype(np.float16)
        for name, value in (("full_mask", full), ("local_mask", local))
    }
    feats["embeddings"] = np.ascontiguousarray(emb, dtype=np.float16)
    feats["type_vectors"] = np.ascontiguousarray(type_embedding[qtype][:, :, None, None], dtype=np.float16)
    feats["marker_map"] = marker_map
    return feats


def public_answer(qdef, p, k, act0, confidence_from_probs):
    kind = qdef["type"]
    ans = {
        "type": kind,
        "confidence": round(confidence_from_probs(p, k), 4),
        "action": {"act_probability": round(float(act0), 4)},
    }
    crit = qdef.get("criteria")
    if kind == "choice":
        labels = list(crit)
        ans.update(choice=labels[int(p.argmax())], probabilities={l: round(float(v), 4) for l, v in zip(labels, p)})
    elif kind == "score":
        ans.update(
            score=round(float((np.arange(k) * p).sum()), 4),
            legend={str(i): v for i, v in enumerate(crit)},
            probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)},
        )
    else:
        ans.update(noul=round(float(p[1]), 4), confidence=round(max(float(p[1]), 1 - float(p[1])), 4))
    return ans


def ane_package_dir(model: str, length: int, batch: int = 1, variant: str = "masked", block: int = 64) -> Path:
    suffix = "" if variant == "masked" or block == 64 else f"-blk{block}"
    return ARTIFACTS / "ane" / model / f"L{length}-B{batch}-{variant}{suffix}"


def make_backend(spec: dict) -> Backend:
    kind = spec["backend"]
    if kind == "torch":
        return TorchBackend(spec["model"], spec.get("device", "cpu"))
    if kind == "mlx":
        return MLXBackend(spec["model"], spec.get("dtype", "float16"), spec.get("compile", False))
    if kind == "coreml":
        return CoreMLBackend(spec["model"], spec.get("units", "cpu_gpu"), spec.get("length"))
    if kind == "ane":
        return ANEBackend(
            spec["model"], spec["length"], spec.get("units", "cpu_ne"), spec.get("batch", 1), spec.get("variant", "masked")
        )
    raise ValueError(kind)


class Bucketed(Backend):
    """Routes each row to the smallest fixed-length export that holds it (B=1 exports).

    This is how a fixed-shape Core ML runtime must serve variable-length requests: pad up to
    the next bucket, never truncate. Sub-backends load lazily and stay resident.
    """

    name = "bucketed"

    def __init__(self, spec: dict, lengths):
        self.spec = spec
        self.lengths = sorted(lengths)
        self.subs: dict[int, Backend] = {}
        self.load_seconds: dict[int, float] = {}

    def sub(self, L):
        import time

        if L not in self.subs:
            t = time.perf_counter()
            self.subs[L] = make_backend({**self.spec, "length": L})
            self.load_seconds[L] = time.perf_counter() - t
        return self.subs[L]

    def bucket(self, n):
        for L in self.lengths:
            if n <= L:
                return L
        raise ValueError(f"{n} tokens exceeds the largest export {self.lengths[-1]}")

    def describe(self):
        any_sub = self.sub(self.lengths[0])
        return {**any_sub.describe(), "buckets": self.lengths}

    def prepare(self, state, questions):
        return self.sub(self.lengths[0]).prepare(state, questions)

    def forward(self, items):
        lg, ac, buckets = [], [], []
        for it in items:
            L = self.bucket(len(it["ids"]))
            a, b = self.sub(L).forward([it])
            lg.append(np.pad(a[0], (0, 32 - a.shape[1]), constant_values=-1e4) if a.shape[1] < 32 else a[0])
            ac.append(b[0])
            buckets.append(L)
        self.last_buckets = buckets
        return np.stack(lg), np.stack(ac)

    def predict(self, state, questions):
        from laya_coreml.common import confidence_from_probs, temp_bucket

        sub = self.sub(self.lengths[0])
        items = self.prepare(state, questions)
        logits, act = self.forward(items)
        act = np.exp(act - act.max(-1, keepdims=True))
        act /= act.sum(-1, keepdims=True)
        cfg = sub.agent.cfg if hasattr(sub, "agent") else sub.cfg
        from laya_mlx.common import clamp_temperature

        temps = [clamp_temperature(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])]
        by = {k: clamp_temperature(v) for k, v in cfg.get("temperature_by_options", {}).items()}
        answers = {}
        for row, (qid, qdef) in enumerate(questions.items()):
            k, qt = len(items[row]["markers"]), items[row]["qtype"]
            z = logits[row, :k] / by.get(temp_bucket(qt, k), temps[qt])
            p = np.exp(z - z.max())
            p /= p.sum()
            answers[qid] = public_answer(qdef, p, k, act[row, 0], confidence_from_probs)
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": sum(len(i["ids"]) for i in items), "output_tokens": 0},
        }


def package_fingerprint(path) -> str:
    return tree_sha256(path)
