"""Channel-first (B,C,1,L) Laya body for fixed-shape Core ML / ANE export.

Provenance: adapted from mizorewww/laya-coreml @ 4619e0483f07adf39068532e85b42ec2347edb83,
file experiments/ane_engineering/model.py (Apache-2.0). That prototype was written and
validated for laya-multilingual only. Changes here:

* works for any GELU ModernBERT Laya checkpoint (verified shapes for ModernBERT-large:
  28 layers, 16 heads, distinct local/global RoPE bases);
* `local_mode="windowed"` adds an exact block-local attention variant used by the
  long-context profiling experiment (the default "masked" mode is the upstream graph);
* `Probe` wraps sub-graphs (single layer, attention-only, MLP-only, ...) for profiling.

Weights are the original checkpoint parameters; every Linear becomes a 1x1 Conv2d with the
same values. No retraining, no approximation beyond FP16 execution.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def conv_from_weights(weight, bias=None):
    result = nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=bias is not None)
    result.weight = nn.Parameter(weight.detach()[:, :, None, None].clone())
    if bias is not None:
        result.bias = nn.Parameter(bias.detach().clone())
    return result


def conv_from_linear(linear):
    return conv_from_weights(linear.weight, linear.bias)


class ChannelNorm(nn.Module):
    """LayerNorm over channel dim 1 with the original `normalized * weight + bias` order."""

    def __init__(self, source):
        super().__init__()
        self.eps = source.eps
        self.weight = nn.Parameter(source.weight.detach()[None, :, None, None].clone())
        self.bias = None
        if source.bias is not None:
            self.bias = nn.Parameter(source.bias.detach()[None, :, None, None].clone())

    def forward(self, x):
        centered = x - x.mean(dim=1, keepdim=True)
        out = centered * (centered.square().mean(dim=1, keepdim=True) + self.eps).rsqrt()
        out = out * self.weight
        return out if self.bias is None else out + self.bias


class ConvAttention(nn.Module):
    """Per-head attention on B,C,1,L activations. Scores are laid out B,key,1,query."""

    def __init__(self, source, *, rope, length, local_mode="masked", window=None, block=64):
        super().__init__()
        self.rope = rope
        self.heads, self.dim = source.heads, source.dim
        weight = source.Wqkv.weight if rope else source.in_proj_weight
        bias = source.Wqkv.bias if rope else source.in_proj_bias
        self.qkv = conv_from_weights(weight, bias)
        self.out = conv_from_linear(source.Wo if rope else source.out_proj)
        self.local_mode = local_mode
        self.window = window  # radius; only used by windowed local attention
        self.block = block
        self.length = length
        if rope:
            self.register_buffer("cos", source.cos[0, 0, :length].T[None, :, None, :].clone())
            self.register_buffer("sin", source.sin[0, 0, :length].T[None, :, None, :].clone())

    def rotate(self, x):
        left, right = x.chunk(2, dim=1)
        return torch.cat(
            (left * self.cos - right * self.sin, right * self.cos + left * self.sin), dim=1
        )

    def _dense(self, qi, ki, vi, mask):
        scores = torch.einsum("bchq,bkhc->bkhq", qi, ki.transpose(1, 3)) * (self.dim**-0.5)
        probabilities = F.softmax(scores + mask, dim=1)
        return torch.einsum("bkhq,bchk->bchq", probabilities, vi)

    def _windowed(self, qi, ki, vi, mask):
        # Exact sliding-window attention computed on query blocks of `block` tokens; each
        # block only visits the key span [start - radius, end + radius), with the additive mask
        # sliced to the same span. Outputs at valid positions are identical to the dense path;
        # padded query positions (which in the dense path see all valid keys) only see their
        # span here. Those states are never read: padded keys are masked in every later layer
        # and the CLS/marker positions are always valid.
        out = []
        n, blk, r = self.length, self.block, self.window
        for start in range(0, n, blk):
            end = min(n, start + blk)
            k0, k1 = max(0, start - r), min(n, end + r)
            q = qi[..., start:end]
            k, v = ki[..., k0:k1], vi[..., k0:k1]
            m = mask[:, k0:k1, :, start:end]
            scores = torch.einsum("bchq,bkhc->bkhq", q, k.transpose(1, 3)) * (self.dim**-0.5)
            probabilities = F.softmax(scores + m, dim=1)
            out.append(torch.einsum("bkhq,bchk->bchq", probabilities, v))
        return torch.cat(out, dim=3)

    def forward(self, x, mask, windowed=False):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        output = []
        for qi, ki, vi in zip(
            q.split(self.dim, dim=1), k.split(self.dim, dim=1), v.split(self.dim, dim=1)
        ):
            if self.rope:
                qi, ki = self.rotate(qi), self.rotate(ki)
            if windowed:
                output.append(self._windowed(qi, ki, vi, mask))
            else:
                output.append(self._dense(qi, ki, vi, mask))
        return self.out(torch.cat(output, dim=1))


class ConvMLP(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.Wi, self.Wo = conv_from_linear(source.Wi), conv_from_linear(source.Wo)

    def forward(self, x):
        value, gate = self.Wi(x).chunk(2, dim=1)
        return self.Wo(F.gelu(value) * gate)


class ConvEncoderLayer(nn.Module):
    def __init__(self, source, length, local_mode="masked", window=None, block=64):
        super().__init__()
        self.kind = source.kind
        self.attn_norm = (
            nn.Identity()
            if isinstance(source.attn_norm, nn.Identity)
            else ChannelNorm(source.attn_norm)
        )
        self.attn = ConvAttention(
            source.attn, rope=True, length=length, local_mode=local_mode, window=window, block=block
        )
        self.windowed = local_mode == "windowed" and self.kind == "sliding_attention"
        self.mlp_norm = ChannelNorm(source.mlp_norm)
        self.mlp = ConvMLP(source.mlp)

    def forward(self, x, mask):
        x = x + self.attn(self.attn_norm(x), mask, windowed=self.windowed)
        return x + self.mlp(self.mlp_norm(x))


class ConvHeadLayer(nn.Module):
    def __init__(self, source, length):
        super().__init__()
        self.norm1, self.norm2 = ChannelNorm(source.norm1), ChannelNorm(source.norm2)
        self.attn = ConvAttention(source.self_attn, rope=False, length=length)
        self.linear1 = conv_from_linear(source.linear1)
        self.linear2 = conv_from_linear(source.linear2)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))


class ConvBody(nn.Module):
    """Embedding norm, encoder, type addition, decision head and scorer in one graph.

    Inputs: embeddings B,C,1,L (host lookup), additive masks B,key,1,query, type vector
    B,C,1,1, one-hot marker selector B,L,1,K. Outputs: option logits B,1,1,K and CLS B,C,1,1.
    The tiny action head, calibration and formatting stay on the host (as upstream ANE port).
    """

    def __init__(self, source, length, local_mode="masked", block=64):
        super().__init__()
        radius = source.encoder.window // 2
        self.embedding_norm = ChannelNorm(source.encoder.embeddings.norm)
        self.layers = nn.ModuleList(
            [
                ConvEncoderLayer(layer, length, local_mode, radius, block)
                for layer in source.encoder.layers
            ]
        )
        self.final_norm = ChannelNorm(source.encoder.final_norm)
        self.head = nn.ModuleList([ConvHeadLayer(layer, length) for layer in source.head.layers])
        self.scorer = nn.Sequential(
            ChannelNorm(source.scorer[0]),
            conv_from_linear(source.scorer[1]),
            nn.GELU(),
            conv_from_linear(source.scorer[3]),
        )

    def forward(self, embeddings, full_mask, local_mask, type_vectors, marker_map):
        x = self.embedding_norm(embeddings)
        for layer in self.layers:
            x = layer(x, full_mask if layer.kind == "full_attention" else local_mask)
        x = self.final_norm(x) + type_vectors
        for layer in self.head:
            x = layer(x, full_mask)
        markers = torch.einsum("bkhq,bchk->bchq", marker_map, x)
        return self.scorer(markers), x[:, :, :, :1]


# ----------------------------------------------------------------------------- probes


class Probe(nn.Module):
    """Sub-graph used only to attribute ANE time. `kind` selects what is exported."""

    def __init__(self, body: ConvBody, kind: str, layer_index: int = 0, repeat: int = 1):
        super().__init__()
        self.kind = kind
        self.body = body
        self.layer_index = layer_index
        self.repeat = repeat

    def forward(self, x, mask):
        layer = self.body.layers[self.layer_index]
        for _ in range(self.repeat):
            if self.kind == "layer":
                x = layer(x, mask)
            elif self.kind == "attention":
                x = x + layer.attn(layer.attn_norm(x), mask, windowed=layer.windowed)
            elif self.kind == "mlp":
                x = x + layer.mlp(layer.mlp_norm(x))
            elif self.kind == "qkv":
                # projection + head split only (the part of attention that is linear in L)
                q, k, v = layer.attn.qkv(layer.attn_norm(x)).chunk(3, dim=1)
                x = x + layer.attn.out(q + k + v)
            elif self.kind == "norm":
                x = x + layer.mlp_norm(x)
            else:
                raise ValueError(self.kind)
        return x
