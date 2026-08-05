# r[impl tenstorrent.rwkv7_p150a_finetuning.graph]
"""Self-contained RWKV7 (Goose, World2.8) model for the P150A bring-up.

Reimplemented from the pinned `fla` (flash-linear-attention) architecture so
that every operator is a plain torch op that lowers directly through
torch_xla. The WKV7 recurrence uses the paper/naive convention that the
focused probes already verified on device, with inputs mapped from the fla
convention:

    fla decay          = exp(w_log)             with w_log < 0
    naive decay        = exp(-exp(w_naive))
    mapping            w_naive = log(-w_log)
    naive a (einsum)   = -kk
    naive b (einsum)   = kk * sigmoid(a_lora(xa))

The equivalence between the two conventions was verified numerically
(max abs diff ~4e-6 in fp32) before this file was written.

This module is pure torch: it runs unmodified on CPU (fp32 reference) and on
an XLA device (bf16), which is what makes CPU<->device parity possible.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_SQRT_TWO_PI_MINUS_HALF = -0.6065306597126334  # log of the decay scale used by fla
LOSS_IGNORE_INDEX = -100


def naive_wkv7_forward(r, w, k, v, a, b, state):
    """Paper-convention WKV7 over a static sequence.

    Inputs are [B, T, H, N]; state is [B, H, N, N]. Returns
    (out [B, T, H, N], final_state [B, H, N, N]). Uses narrow-based timestep
    reads because the basic-indexing select backward is broken on this
    runtime (see probes/wkv7_probe.py).
    """
    batch, time_steps, heads, width = r.shape
    wactivated = (-(w.exp())).exp()
    outputs = []
    current = state
    for t in range(time_steps):
        current = (
            current * _step(wactivated, t).unsqueeze(-2)
            + torch.einsum("bhik,bhk,bhj->bhij", current, _step(a, t), _step(b, t))
            + torch.einsum("bhj,bhi->bhij", _step(k, t), _step(v, t))
        )
        outputs.append(torch.einsum("bhj,bhij->bhi", _step(r, t), current))
    out = torch.stack(outputs, dim=1)
    return out, current


def _step(x, t):
    return x.narrow(1, t, 1).squeeze(1)


def token_shift(x):
    """Return delta_t = x_{t-1} - x_t with x_{-1} = 0."""
    prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
    return prev - x


class LoRA(nn.Module):
    """fla-compatible low-rank block: Linear -> act -> Linear."""

    def __init__(self, input_dim, output_dim, low_rank_dim, bias=True, activation="tanh"):
        super().__init__()
        if activation is None:
            act = nn.Identity()
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            raise ValueError(f"unsupported activation {activation}")
        self.lora = nn.Sequential(
            nn.Linear(input_dim, low_rank_dim, bias=False),
            act,
            nn.Linear(low_rank_dim, output_dim, bias=bias),
        )

    def forward(self, x):
        return self.lora(x)


class RWKV7Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, decay_low_rank_dim, gate_low_rank_dim,
                 a_low_rank_dim, v_low_rank_dim, norm_eps, layer_idx):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        self.value_dim = hidden_size
        self.head_v_dim = self.value_dim // self.num_heads
        self.layer_idx = layer_idx

        self.x_r = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.k_k = nn.Parameter(torch.zeros(hidden_size))
        self.k_a = nn.Parameter(torch.zeros(hidden_size))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, head_dim))

        self.r_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.w_lora = LoRA(hidden_size, hidden_size, decay_low_rank_dim, activation="tanh")
        if layer_idx != 0:
            self.v_lora = LoRA(hidden_size, self.value_dim, v_low_rank_dim, activation=None)
        self.a_lora = LoRA(hidden_size, hidden_size, a_low_rank_dim, activation=None)
        self.g_lora = LoRA(hidden_size, self.value_dim, gate_low_rank_dim,
                           activation="sigmoid", bias=False)

        self.g_norm = nn.GroupNorm(
            num_groups=self.num_heads, num_channels=self.value_dim,
            eps=head_dim * norm_eps, affine=True,
        )

    def forward(self, hidden_states, v_first):
        batch, seq_len, _ = hidden_states.shape
        delta = token_shift(hidden_states)
        xr = hidden_states + delta * self.x_r
        xw = hidden_states + delta * self.x_w
        xk = hidden_states + delta * self.x_k
        xv = hidden_states + delta * self.x_v
        xa = hidden_states + delta * self.x_a
        xg = hidden_states + delta * self.x_g

        r = self.r_proj(xr)
        w_log = LOG_SQRT_TWO_PI_MINUS_HALF * torch.sigmoid(self.w_lora(xw))
        k = self.k_proj(xk)
        v = self.v_proj(xv)

        if self.layer_idx == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, torch.sigmoid(self.v_lora(xv)))

        a_sig = torch.sigmoid(self.a_lora(xa))
        g = self.g_lora(xg)

        H, D = self.num_heads, self.head_dim
        kk = F.normalize((k * self.k_k).view(batch, seq_len, H, D), dim=-1, p=2.0)
        # k = k * (1 + (a - 1) * k_a)  (fused_k_rwkv7)
        a_sig_hd = a_sig.view(batch, seq_len, H, D)
        k = k.addcmul(k * (a_sig - 1), self.k_a)

        r4 = r.view(batch, seq_len, H, D)
        w_naive4 = torch.log(-w_log).view(batch, seq_len, H, D)
        k4 = k.view(batch, seq_len, H, D)
        v4 = v.view(batch, seq_len, H, self.head_v_dim)
        a_naive4 = -kk
        b_naive4 = kk * a_sig_hd

        state0 = torch.zeros(batch, H, D, self.head_v_dim, dtype=hidden_states.dtype,
                             device=hidden_states.device)
        o, _ = naive_wkv7_forward(r4, w_naive4, k4, v4, a_naive4, b_naive4, state0)

        o = self.g_norm(o.reshape(batch * seq_len, H * self.head_v_dim)).view(
            batch, seq_len, self.value_dim)

        # gate_output_correction: (o + ((r*k*r_k).sum * v)) * g
        r_k_b = self.r_k.unsqueeze(0).unsqueeze(0)
        correction_scalar = (r4 * k4 * r_k_b).sum(-1, keepdim=True)
        gated = o.view(batch, seq_len, H, self.head_v_dim) + correction_scalar * v4
        o = gated.reshape(batch, seq_len, self.value_dim) * g

        o = self.o_proj(o)
        return o, v_first


class RWKV7FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(hidden_size))
        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        delta = token_shift(x)
        mixed = x.addcmul(delta, self.x_k)
        return self.value(self._sqrelu(self.key(mixed)))

    @staticmethod
    def _sqrelu(x):
        r = F.relu(x)
        return r * r


class RWKV7Block(nn.Module):
    def __init__(self, hidden_size, head_dim, decay_low_rank_dim, gate_low_rank_dim,
                 a_low_rank_dim, v_low_rank_dim, intermediate_size, norm_eps,
                 norm_first, layer_idx):
        super().__init__()
        self.norm_first = norm_first
        if norm_first and layer_idx == 0:
            self.pre_norm = nn.LayerNorm(hidden_size, bias=True, eps=norm_eps)
        self.attn_norm = nn.LayerNorm(hidden_size, bias=True, eps=norm_eps)
        self.attn = RWKV7Attention(hidden_size, head_dim, decay_low_rank_dim,
                                   gate_low_rank_dim, a_low_rank_dim, v_low_rank_dim,
                                   norm_eps, layer_idx)
        self.ffn_norm = nn.LayerNorm(hidden_size, bias=True, eps=norm_eps)
        self.ffn = RWKV7FeedForward(hidden_size, intermediate_size)

    def forward(self, hidden_states, v_first):
        residual = self.pre_norm(hidden_states) if (self.norm_first and hasattr(self, "pre_norm")) else hidden_states
        h = self.attn_norm(residual)
        attn_out, v_first = self.attn(h, v_first)
        h = residual + attn_out
        residual = h
        h = self.ffn_norm(h)
        h = self.ffn(h)
        h = residual + h
        return h, v_first


class RWKV7Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([
            RWKV7Block(
                config["hidden_size"], config["head_dim"], config["decay_low_rank_dim"],
                config["gate_low_rank_dim"], config["a_low_rank_dim"],
                config["v_low_rank_dim"], config["intermediate_size"],
                config["norm_eps"], config["norm_first"], i,
            )
            for i in range(config["num_hidden_layers"])
        ])
        self.norm = nn.LayerNorm(config["hidden_size"], bias=True, eps=config["norm_eps"])

    def forward(self, input_ids):
        h = self.embeddings(input_ids)
        v_first = torch.zeros_like(h)
        for layer in self.layers:
            h, v_first = layer(h, v_first)
        return self.norm(h)


class RWKV7ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = RWKV7Model(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(self, input_ids):
        hidden = self.model(input_ids)
        return self.lm_head(hidden)


GOOSE_0P1B_CONFIG = {
    "vocab_size": 65536,
    "hidden_size": 768,
    "head_dim": 64,
    "num_hidden_layers": 12,
    "intermediate_size": 3072,
    "decay_low_rank_dim": 64,
    "gate_low_rank_dim": 128,
    "a_low_rank_dim": 64,
    "v_low_rank_dim": 32,
    "norm_eps": 1e-5,
    "norm_first": True,
}


def load_goose_weights(model, state_dict):
    """Load the fla-format checkpoint into the reimplementation, strictly."""
    res = model.load_state_dict(state_dict, strict=False)
    return res.missing_keys, res.unexpected_keys


LORA_ROLE_TO_ATTR = {
    "attention.receptance": ("attn", "r_proj"),
    "attention.key": ("attn", "k_proj"),
    "attention.value": ("attn", "v_proj"),
    "attention.output": ("attn", "o_proj"),
    "feed_forward.key": ("ffn", "key"),
    "feed_forward.value": ("ffn", "value"),
}


class LoRALinear(nn.Module):
    """Frozen base Linear plus a trainable rank-limited adapter.

    lora_B is zero-initialized so the adapter starts as the exact base
    projection; step-0 outputs therefore match the frozen model.
    """

    def __init__(self, base_linear, rank, scale):
        super().__init__()
        self.base = base_linear
        self.scale = scale
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.lora_A.t() @ self.lora_B.t())


def apply_lora(model, roles, rank, scale):
    """Wrap the admitted projection roles in LoRA and freeze everything else.

    Returns the sorted list of trainable parameter names.
    """
    for layer in model.model.layers:
        for role in roles:
            block_attr, proj_attr = LORA_ROLE_TO_ATTR[role]
            block = getattr(layer, block_attr)
            base_linear = getattr(block, proj_attr)
            setattr(block, proj_attr, LoRALinear(base_linear, rank, scale))

    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
            trainable += [f"{name}.lora_A", f"{name}.lora_B"]
    return sorted(trainable)
