"""TT-NN implementation of LFM3-1.7B (tri-hybrid: gated short conv, sliding and full attention).

The module mirrors the checkpoint's own ``modeling_lfm3.py`` math exactly:
per-head QK RMSNorm, partial-RoPE for the global-attention layers, a per-head
sigmoid output gate, sandwich RMSNorm around the operator and FFN residual
adds, and a width-7 causal depthwise convolution with a per-layer recurrent
window.

Two precision modes are available. ``dtype=ttnn.float32`` keeps the residual
stream wide and runs attention with an explicit float32 softmax (the SDPA
kernels are bf16-only); it matches the fp32 HF oracle token for token over a
430-token prompt. ``dtype=ttnn.bfloat16`` is the fast path. Measured TT-Metal
matmul fidelity is ~1e-3 relative even for float32 inputs, so neither mode is
bit-exact against a CPU fp32 reference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
import ttnn


@dataclass(frozen=True)
class Lfm3Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    norm_eps: float
    conv_kernel_size: int
    sliding_window: int
    layer_types: tuple[str, ...]
    rope: dict
    use_sandwich_norm: bool
    tie_word_embeddings: bool

    @classmethod
    def from_json(cls, path: str) -> "Lfm3Config":
        with open(path) as f:
            raw = json.load(f)
        head_dim = (
            raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"]
        )
        return cls(
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=head_dim,
            vocab_size=raw["vocab_size"],
            norm_eps=raw["norm_eps"],
            conv_kernel_size=raw.get("conv_L_cache", 7),
            sliding_window=raw.get("sliding_window", 256),
            layer_types=tuple(raw["layer_types"]),
            rope=raw.get("rope_parameters") or {},
            use_sandwich_norm=raw.get("use_sandwich_norm", True),
            tie_word_embeddings=raw.get("tie_word_embeddings", True),
        )

    def attention_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t != "conv"]

    def conv_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "conv"]


def _inv_freq(cfg: Lfm3Config, layer_type: str) -> torch.Tensor:
    params = cfg.rope[layer_type]
    theta = float(params["rope_theta"])
    partial = float(params.get("partial_rotary_factor", 1.0))
    dim = cfg.head_dim
    angles = int(partial * dim) // 2
    freq = 1.0 / (theta ** (torch.arange(0, 2 * angles, 2, dtype=torch.float32) / dim))
    return torch.cat((freq, torch.zeros(dim // 2 - angles, dtype=torch.float32)))


def rope_table(
    cfg: Lfm3Config, layer_type: str, start: int, length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin for positions [start, start+length), shaped [1, 1, length, head_dim]."""
    pos = torch.arange(start, start + length, dtype=torch.float32)
    emb = pos[:, None] * _inv_freq(cfg, layer_type)[None, :]
    emb = torch.cat((emb, emb), dim=-1)
    return emb.cos()[None, None], emb.sin()[None, None]


class Lfm3Model:
    """Single-mesh LFM3 decoder with recurrent conv windows and KV caches."""

    def __init__(
        self,
        dev: ttnn.MeshDevice,
        cfg: Lfm3Config,
        checkpoint: str,
        max_seq: int,
        dtype: ttnn.DataType = ttnn.bfloat16,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        decode_attention: str = "sdpa",
    ):
        self.dev = dev
        self.cfg = cfg
        self.dtype = dtype
        self.weight_dtype = weight_dtype
        self.torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
        self.weight_torch_dtype = (
            torch.float32 if weight_dtype == ttnn.float32 else torch.bfloat16
        )
        # The SDPA kernels run in bf16 only; with a float32 stream, run attention
        # explicitly in float32. Decode can stay on the fast bf16 SDPA kernel or
        # use the float32 path (slower, keeps the KV cache wide).
        # r[impl tenstorrent.lfm3.decode.attention.default]
        self.attn_fp32 = dtype == ttnn.float32
        self.decode_fp32 = self.attn_fp32 and decode_attention == "fp32"
        self._masks: dict[tuple[str, int], ttnn.Tensor] = {}
        self.max_seq = max_seq
        self._load(checkpoint)

    # ---------------------------------------------------------------- loading
    def _tt(
        self, tensor: torch.Tensor, layout=ttnn.TILE_LAYOUT, dtype=None
    ) -> ttnn.Tensor:
        return ttnn.from_torch(
            tensor.contiguous(),
            dtype=dtype or self.weight_dtype,
            layout=layout,
            device=self.dev,
        )

    def _linear(self, x: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        if x.dtype != weight.dtype:
            # Mixed precision: keep the residual stream wide, run the matmul in the weight dtype.
            y = ttnn.linear(ttnn.typecast(x, weight.dtype), weight)
            return ttnn.typecast(y, x.dtype)
        return ttnn.linear(x, weight)

    def _linear_weight(self, tensor: torch.Tensor) -> ttnn.Tensor:
        # TT-NN computes x @ W, so store the checkpoint's [out, in] transposed.
        return self._tt(tensor.t().to(self.weight_torch_dtype))

    def _load(self, checkpoint: str) -> None:
        from safetensors import safe_open

        cfg = self.cfg
        layers: list[dict] = [{} for _ in range(cfg.num_hidden_layers)]
        with safe_open(checkpoint, framework="pt") as f:
            embed = f.get_tensor("model.embed_tokens.weight").to(self.torch_dtype)
            self.embed_weight = self._tt(
                embed, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
            )
            self.lm_head = self._linear_weight(embed)
            del embed
            self.embedding_norm = self._tt(
                f.get_tensor("model.embedding_norm.weight").to(self.torch_dtype)
            )
            for i in range(cfg.num_hidden_layers):
                p = f"model.layers.{i}."
                layer = layers[i]
                for name in (
                    "operator_norm",
                    "ffn_norm",
                    "post_operator_norm",
                    "post_ffn_norm",
                ):
                    layer[name] = self._tt(
                        f.get_tensor(p + name + ".weight").to(self.torch_dtype)
                    )
                w13 = torch.cat(
                    [
                        f.get_tensor(p + "feed_forward.w1.weight"),
                        f.get_tensor(p + "feed_forward.w3.weight"),
                    ],
                    dim=0,
                )
                layer["w13"] = self._linear_weight(w13)
                layer["w2"] = self._linear_weight(
                    f.get_tensor(p + "feed_forward.w2.weight")
                )
                if cfg.layer_types[i] == "conv":
                    layer["in_proj"] = self._linear_weight(
                        f.get_tensor(p + "conv.in_proj.weight")
                    )
                    layer["out_proj"] = self._linear_weight(
                        f.get_tensor(p + "conv.out_proj.weight")
                    )
                    w = (
                        f
                        .get_tensor(p + "conv.conv.weight")
                        .to(self.torch_dtype)
                        .reshape(cfg.hidden_size, cfg.conv_kernel_size)
                    )
                    # One [1, K, C] tap tensor, row-major to pair with the row-major conv window.
                    layer["conv_taps"] = self._tt(
                        w.t().reshape(1, cfg.conv_kernel_size, cfg.hidden_size),
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        dtype=self.dtype,
                    )
                else:
                    # One fused QKV projection: [q | k | v] rows, split after the matmul.
                    qkv = torch.cat(
                        [
                            f.get_tensor(p + "self_attn." + name + ".weight")
                            for name in ("q_proj", "k_proj", "v_proj")
                        ],
                        dim=0,
                    )
                    layer["qkv"] = self._linear_weight(qkv)
                    layer["gate_proj"] = self._linear_weight(
                        f.get_tensor(p + "self_attn.gate_proj.weight")
                    )
                    layer["o_proj"] = self._linear_weight(
                        f.get_tensor(p + "self_attn.o_proj.weight")
                    )
                    for name in ("q_norm", "k_norm"):
                        layer[name] = self._tt(
                            f.get_tensor(p + "self_attn." + name + ".weight").to(
                                self.torch_dtype
                            )
                        )
        self.layers = layers
        self._alloc_state()

    def _alloc_state(self) -> None:
        cfg = self.cfg
        zeros = torch.zeros(
            1,
            cfg.num_key_value_heads,
            self.max_seq,
            cfg.head_dim,
            dtype=self.torch_dtype,
        )
        self.kv = {}
        for i in cfg.attention_layers():
            self.kv[i] = (
                self._tt(
                    zeros, dtype=ttnn.float32 if self.decode_fp32 else ttnn.bfloat16
                ),
                self._tt(
                    zeros, dtype=ttnn.float32 if self.decode_fp32 else ttnn.bfloat16
                ),
            )
        self.conv_state = {
            # Oldest-first window of the pre-conv operand; one [1, 1, C] row-major tensor per cached token.
            i: [
                self._tt(
                    torch.zeros(1, 1, cfg.hidden_size, dtype=self.torch_dtype),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=self.dtype,
                )
                for _ in range(cfg.conv_kernel_size - 1)
            ]
            for i in cfg.conv_layers()
        }
        self.pos = 0

    def reset(self) -> None:
        for i in self.cfg.attention_layers():
            for cache in self.kv[i]:
                ttnn.multiply(cache, 0.0, output_tensor=cache)
        for i in self.cfg.conv_layers():
            for row in self.conv_state[i]:
                ttnn.multiply(row, 0.0, output_tensor=row)
        self.pos = 0

    # ------------------------------------------------------------- primitives
    def _rms(self, x: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, weight=weight, epsilon=self.cfg.norm_eps)

    def _rope(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        # x: [1, heads, seq, head_dim]; cos/sin: [1, 1, seq, head_dim].
        # HF's rotate-half form; partial-RoPE layers carry cos=1/sin=0 in the tail.
        return ttnn.experimental.rotary_embedding_hf(x, cos, sin)

    def _rope_exact(
        self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor
    ) -> ttnn.Tensor:
        # Same math as the fused op, but in the activation dtype (the fused op is bf16-only).
        heads, seq, hd = x.shape[1], x.shape[2], self.cfg.head_dim
        x1 = ttnn.slice(x, [0, 0, 0, 0], [1, heads, seq, hd // 2])
        x2 = ttnn.slice(x, [0, 0, 0, hd // 2], [1, heads, seq, hd])
        rot = ttnn.concat([ttnn.multiply(x2, -1.0), x1], dim=-1)
        return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rot, sin))

    def _attention_mask(self, kind: str, seq: int) -> ttnn.Tensor:
        key = (kind, seq)
        mask = self._masks.get(key)
        if mask is None:
            rows = torch.arange(seq)
            allowed = rows[:, None] >= rows[None, :]
            if kind == "sliding_attention":
                allowed &= rows[:, None] - rows[None, :] < self.cfg.sliding_window
            additive = torch.where(allowed, 0.0, -1e9).to(self.torch_dtype)
            mask = self._tt(additive.reshape(1, 1, 1, seq, seq), dtype=self.dtype)
            self._masks[key] = mask
        return mask

    def _attention_exact(
        self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, kind: str
    ) -> ttnn.Tensor:
        """GQA causal attention with an explicit float32 softmax; q/k/v are [1, heads|kv, seq, hd]."""
        cfg = self.cfg
        seq, hd = q.shape[2], cfg.head_dim
        group = cfg.num_attention_heads // cfg.num_key_value_heads
        nkv = cfg.num_key_value_heads
        qg = ttnn.reshape(q, (1, nkv, group, seq, hd))
        kg = ttnn.repeat(ttnn.reshape(k, (1, nkv, 1, seq, hd)), (1, 1, group, 1, 1))
        vg = ttnn.repeat(ttnn.reshape(v, (1, nkv, 1, seq, hd)), (1, 1, group, 1, 1))
        kt = ttnn.permute(kg, (0, 1, 2, 4, 3))
        mask = self._attention_mask(kind, seq)
        block = 256
        parts = []
        for start in range(0, seq, block):
            stop = min(start + block, seq)
            qb = ttnn.slice(qg, [0, 0, 0, start, 0], [1, nkv, group, stop, hd])
            scores = ttnn.multiply(ttnn.matmul(qb, kt), hd**-0.5)
            scores = ttnn.add(
                scores, ttnn.slice(mask, [0, 0, 0, start, 0], [1, 1, 1, stop, seq])
            )
            probs = ttnn.softmax(scores, dim=-1)
            parts.append(ttnn.matmul(probs, vg))
        out = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=3)
        return ttnn.reshape(out, (1, cfg.num_attention_heads, seq, hd))

    # r[impl tenstorrent.lfm3.decode.attention.wide]
    def _attention_exact_decode(
        self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, kind: str, layer_idx: int
    ) -> ttnn.Tensor:
        """One-token GQA attention over the float32 KV cache.

        The cache holds positions [0, pos]; rows past the current position are
        masked out, and the slice length is rounded up to a 32-row tile.
        """
        cfg = self.cfg
        nkv, group, hd = (
            cfg.num_key_value_heads,
            cfg.num_attention_heads // cfg.num_key_value_heads,
            cfg.head_dim,
        )
        k_cache, v_cache = self.kv[layer_idx]
        ttnn.update_cache(k_cache, k, self.pos)
        ttnn.update_cache(v_cache, v, self.pos)
        used = min(self.max_seq, ((self.pos // 32) + 1) * 32)
        kc = ttnn.slice(k_cache, [0, 0, 0, 0], [1, nkv, used, hd])
        vc = ttnn.slice(v_cache, [0, 0, 0, 0], [1, nkv, used, hd])
        qg = ttnn.reshape(q, (1, nkv, group, 1, hd))
        kg = ttnn.repeat(ttnn.reshape(kc, (1, nkv, 1, used, hd)), (1, 1, group, 1, 1))
        vg = ttnn.repeat(ttnn.reshape(vc, (1, nkv, 1, used, hd)), (1, 1, group, 1, 1))
        scores = ttnn.multiply(
            ttnn.matmul(qg, ttnn.permute(kg, (0, 1, 2, 4, 3))), hd**-0.5
        )
        rows = torch.arange(used)
        allowed = rows <= self.pos
        if kind == "sliding_attention":
            allowed &= rows > self.pos - cfg.sliding_window
        additive = (
            torch
            .where(allowed, 0.0, -1e9)
            .to(self.torch_dtype)
            .reshape(1, 1, 1, 1, used)
        )
        probs = ttnn.softmax(
            ttnn.add(scores, self._tt(additive, dtype=self.dtype)), dim=-1
        )
        out = ttnn.matmul(probs, vg)
        return ttnn.reshape(out, (1, cfg.num_attention_heads, 1, hd))

    def _conv_prefill(self, h: ttnn.Tensor, layer: dict) -> ttnn.Tensor:
        # h: [1, S, C] row-major. Pads the causal window then accumulates the seven shifted taps.
        seq = h.shape[1]
        C = self.cfg.hidden_size
        padded = ttnn.pad(
            h, [(0, 0), (self.cfg.conv_kernel_size - 1, 0), (0, 0)], value=0.0
        )
        acc = None
        for k in range(self.cfg.conv_kernel_size):
            window = ttnn.slice(padded, [0, k, 0], [1, k + seq, C])
            tap = ttnn.slice(layer["conv_taps"], [0, k, 0], [1, k + 1, C])
            term = ttnn.multiply(window, tap)
            acc = term if acc is None else ttnn.add(acc, term)
        return acc

    def _conv_decode(self, h: ttnn.Tensor, layer: dict, layer_idx: int) -> ttnn.Tensor:
        # h: [1, 1, C] row-major. One concat + multiply + reduce over the six cached rows and the new row.
        window = self.conv_state[layer_idx]
        cat = ttnn.concat(window + [h], dim=1)
        conv = ttnn.sum(ttnn.multiply(cat, layer["conv_taps"]), dim=1, keepdim=True)
        self.conv_state[layer_idx] = window[1:] + [h]
        return conv

    def _mlp(self, x: ttnn.Tensor, layer: dict) -> ttnn.Tensor:
        # Fused gate/up projection, split on the device.
        intermediate = self.cfg.intermediate_size
        wide = self._linear(x, layer["w13"])
        seq = wide.shape[1]
        gate = ttnn.slice(wide, [0, 0, 0], [1, seq, intermediate])
        up = ttnn.slice(wide, [0, 0, intermediate], [1, seq, 2 * intermediate])
        return self._linear(ttnn.multiply(ttnn.silu(gate), up), layer["w2"])

    def _attention(
        self,
        x: ttnn.Tensor,
        layer: dict,
        layer_idx: int,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        decode: bool,
    ) -> ttnn.Tensor:
        cfg = self.cfg
        seq = x.shape[1]
        hd, nh = cfg.head_dim, cfg.num_attention_heads
        nkv = cfg.num_key_value_heads
        q_dim, kv_dim = nh * hd, nkv * hd
        wide = self._linear(x, layer["qkv"])
        q = ttnn.reshape(ttnn.slice(wide, [0, 0, 0], [1, seq, q_dim]), (1, seq, nh, hd))
        k = ttnn.reshape(
            ttnn.slice(wide, [0, 0, q_dim], [1, seq, q_dim + kv_dim]), (1, seq, nkv, hd)
        )
        v = ttnn.reshape(
            ttnn.slice(wide, [0, 0, q_dim + kv_dim], [1, seq, q_dim + 2 * kv_dim]),
            (1, seq, nkv, hd),
        )
        gate = self._linear(x, layer["gate_proj"])
        q = self._rms(q, layer["q_norm"])
        k = self._rms(k, layer["k_norm"])
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))
        kind = cfg.layer_types[layer_idx]
        k_cache, v_cache = self.kv[layer_idx]
        window = cfg.sliding_window if kind == "sliding_attention" else None
        fp32_attn = self.attn_fp32 and (not decode or self.decode_fp32)
        if fp32_attn:
            q = self._rope_exact(q, cos, sin)
            k = self._rope_exact(k, cos, sin)
            if decode:
                # update_cache wants the row tile padded to 32 sequences.
                kp = ttnn.pad(k, [(0, 0), (0, 0), (0, 31), (0, 0)], value=0.0)
                vp = ttnn.pad(v, [(0, 0), (0, 0), (0, 31), (0, 0)], value=0.0)
                out = self._attention_exact_decode(q, kp, vp, kind, layer_idx)
            else:
                ttnn.fill_cache(
                    k_cache, ttnn.typecast(k, k_cache.dtype), 0, update_idx=0
                )
                ttnn.fill_cache(
                    v_cache, ttnn.typecast(v, v_cache.dtype), 0, update_idx=0
                )
                out = self._attention_exact(q, k, v, kind)
        else:
            if self.dtype != ttnn.bfloat16:
                # The SDPA kernel and the fused RoPE op are bf16-only.
                q, k, v = (ttnn.typecast(t, ttnn.bfloat16) for t in (q, k, v))
                cos, sin = (ttnn.typecast(t, ttnn.bfloat16) for t in (cos, sin))
            q = self._rope(q, cos, sin)
            k = self._rope(k, cos, sin)
        if not fp32_attn and decode:
            # update_cache wants [seq, heads, batch padded to 32, head_dim].
            kp = ttnn.pad(k, [(0, 0), (0, 0), (0, 31), (0, 0)], value=0.0)
            vp = ttnn.pad(v, [(0, 0), (0, 0), (0, 31), (0, 0)], value=0.0)
            ttnn.update_cache(k_cache, kp, self.pos)
            ttnn.update_cache(v_cache, vp, self.pos)
            out = ttnn.transformer.scaled_dot_product_attention_decode(
                ttnn.reshape(q, (1, 1, nh, hd)),
                k_cache,
                v_cache,
                cur_pos=[self.pos],
                is_causal=True,
                sliding_window_size=window,
            )
            out = ttnn.reshape(out, (1, nh, 1, hd))
        elif not fp32_attn:
            ttnn.fill_cache(k_cache, k, 0, update_idx=0)
            ttnn.fill_cache(v_cache, v, 0, update_idx=0)
            out = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, sliding_window_size=window
            )
        out = ttnn.permute(out, (0, 2, 1, 3))  # [1, seq, heads, head_dim]
        if self.dtype != ttnn.bfloat16:
            out = ttnn.typecast(out, self.dtype)
        gate = ttnn.reshape(ttnn.sigmoid(gate), (1, seq, nh, 1))
        out = ttnn.multiply(out, gate)
        return self._linear(ttnn.reshape(out, (1, seq, nh * hd)), layer["o_proj"])

    def _layer(
        self,
        h: ttnn.Tensor,
        layer_idx: int,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        decode: bool,
        want_pre: bool = False,
    ):
        cfg, layer = self.cfg, self.layers[layer_idx]
        residual = h
        pre = None
        x = self._rms(h, layer["operator_norm"])
        if cfg.layer_types[layer_idx] == "conv":
            bcx = self._linear(x, layer["in_proj"])
            C = cfg.hidden_size
            b = ttnn.slice(bcx, [0, 0, 0], [1, bcx.shape[1], C])
            c = ttnn.slice(bcx, [0, 0, C], [1, bcx.shape[1], 2 * C])
            xin = ttnn.slice(bcx, [0, 0, 2 * C], [1, bcx.shape[1], 3 * C])
            pre = ttnn.to_layout(ttnn.multiply(b, xin), ttnn.ROW_MAJOR_LAYOUT)
            if decode:
                conv = self._conv_decode(pre, layer, layer_idx)
            else:
                conv = self._conv_prefill(pre, layer)
            out = self._linear(
                ttnn.multiply(c, ttnn.to_layout(conv, ttnn.TILE_LAYOUT)),
                layer["out_proj"],
            )
        else:
            out = self._attention(x, layer, layer_idx, cos, sin, decode)
        if cfg.use_sandwich_norm:
            out = self._rms(out, layer["post_operator_norm"])
        h = ttnn.add(residual, out)

        f = self._mlp(self._rms(h, layer["ffn_norm"]), layer)
        if cfg.use_sandwich_norm:
            f = self._rms(f, layer["post_ffn_norm"])
        h = ttnn.add(h, f)
        return (h, pre) if want_pre else h

    # ------------------------------------------------------------------- calls
    def _capture_conv_state(self, pre: ttnn.Tensor, layer_idx: int, seq: int) -> None:
        # The recurrent window only needs the last K-1 pre-conv rows; take an aligned 32-row slab
        # and split it on the host.
        k = self.cfg.conv_kernel_size - 1
        start = max(0, seq - 32)
        slab = ttnn.slice(pre, [0, start, 0], [1, seq, self.cfg.hidden_size])
        host = ttnn.to_torch(slab).reshape(seq - start, self.cfg.hidden_size)[-k:]
        rows = [
            row.reshape(1, 1, self.cfg.hidden_size) for row in host.to(self.torch_dtype)
        ]
        rows = [torch.zeros(1, 1, self.cfg.hidden_size, dtype=self.torch_dtype)] * (
            k - len(rows)
        ) + rows
        self.conv_state[layer_idx] = [
            self._tt(row, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.dtype)
            for row in rows
        ]

    def prefill(self, ids: list[int], capture_layers: bool = False):
        """Prefill a token sequence; returns last-position logits and, optionally, per-layer hidden states."""
        cfg, dev = self.cfg, self.dev
        seq = len(ids)
        pad = (-seq) % 32
        ids = ids + [0] * pad
        total = len(ids)
        tokens = ttnn.from_torch(
            torch.tensor([ids], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
        )
        h = ttnn.to_layout(
            ttnn.embedding(tokens, self.embed_weight), ttnn.TILE_LAYOUT
        )  # [1, total, C]
        if self.dtype != ttnn.bfloat16:
            h = ttnn.typecast(h, self.dtype)
        rope_dtype = self.dtype if self.attn_fp32 else ttnn.bfloat16
        cos_sin = {
            t: (
                self._tt(
                    rope_table(cfg, t, 0, total)[0].to(
                        self.torch_dtype if self.attn_fp32 else torch.bfloat16
                    ),
                    dtype=rope_dtype,
                ),
                self._tt(
                    rope_table(cfg, t, 0, total)[1].to(
                        self.torch_dtype if self.attn_fp32 else torch.bfloat16
                    ),
                    dtype=rope_dtype,
                ),
            )
            for t in set(cfg.layer_types) - {"conv"}
        }
        captured = [ttnn.to_torch(h)[0, :seq]] if capture_layers else None
        conv_pres = {}
        for i in range(cfg.num_hidden_layers):
            lt = cfg.layer_types[i]
            cos, sin = cos_sin.get(lt, (None, None))
            h, pre = self._layer(h, i, cos, sin, decode=False, want_pre=True)
            if pre is not None:
                conv_pres[i] = pre
            if capture_layers:
                captured.append(ttnn.to_torch(h)[0, :seq])
        # Shorten the recurrent windows to the real tokens (right padding must not enter state).
        for i, pre in conv_pres.items():
            real = ttnn.slice(pre, [0, 0, 0], [1, seq, cfg.hidden_size])
            self._capture_conv_state(real, i, seq)
        h = self._rms(h, self.embedding_norm)
        self.pos = seq
        if capture_layers:
            captured.append(ttnn.to_torch(h)[0, :seq])
            return captured
        start = total - 32
        tail = ttnn.slice(h, [0, start, 0], [1, total, cfg.hidden_size])
        logits = ttnn.to_torch(self._linear(tail, self.lm_head))
        return logits[0, 31 - pad].float()

    def decode(self, token: int) -> torch.Tensor:
        cfg, dev = self.cfg, self.dev
        tokens = ttnn.from_torch(
            torch.tensor([[token]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
        )
        h = ttnn.to_layout(ttnn.embedding(tokens, self.embed_weight), ttnn.TILE_LAYOUT)
        if self.dtype != ttnn.bfloat16:
            h = ttnn.typecast(h, self.dtype)
        cos_sin = {
            t: (
                self._tt(
                    rope_table(cfg, t, self.pos, 1)[0].to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                ),
                self._tt(
                    rope_table(cfg, t, self.pos, 1)[1].to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                ),
            )
            for t in set(cfg.layer_types) - {"conv"}
        }
        for i in range(cfg.num_hidden_layers):
            lt = cfg.layer_types[i]
            cos, sin = cos_sin.get(lt, (None, None))
            h = self._layer(h, i, cos, sin, decode=True)
        h = self._rms(h, self.embedding_norm)
        logits = ttnn.to_torch(self._linear(h, self.lm_head))
        self.pos += 1
        return logits[0, 0].float()
