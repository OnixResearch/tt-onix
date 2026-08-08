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

import torch
import torch.nn.functional as F
from torch import nn

LOG_SQRT_TWO_PI_MINUS_HALF = -0.6065306597126334  # log of the decay scale used by fla
LOSS_IGNORE_INDEX = -100


def naive_wkv7_forward(r, w, k, v, a, b, state):
    """Paper-convention WKV7 over a static sequence.

    Inputs are [B, T, H, N]; state is [B, H, N, N]. Returns
    (out [B, T, H, N], final_state [B, H, N, N]). Uses narrow-based timestep
    reads because the basic-indexing select backward is broken on this
    runtime (see probes/wkv7_probe.py).
    """
    _, time_steps, _, _ = r.shape
    # fla accumulates the recurrent state in fp32 even for bf16 inputs.
    # Doing the recurrence in the input dtype overflows on the real model,
    # so upcast here and downcast the outputs on return.
    orig_dtype = r.dtype
    r = r.float()
    w = w.float()
    k = k.float()
    v = v.float()
    a = a.float()
    b = b.float()
    state = state.float()
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
    return out.to(orig_dtype), current.to(orig_dtype)


def _step(x, t):
    return x.narrow(1, t, 1).squeeze(1)


def token_shift(x):
    """Return delta_t = x_{t-1} - x_t with x_{-1} = 0."""
    prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
    return prev - x


class PreciseLayerNorm(nn.Module):
    """LayerNorm computed with explicit fp32 reductions.

    The fused nn.LayerNorm lowers to a numerically wrong kernel on this
    runtime (error scales with input magnitude). The explicit mean/variance/
    rsqrt decomposition is exact, so norms use this instead. Keeps the same
    weight/bias parameter names so the checkpoint still loads strictly.
    """

    def __init__(self, dim, bias=True, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        mean = xf.mean(dim=-1, keepdim=True)
        centered = xf - mean
        var = (centered * centered).mean(dim=-1, keepdim=True)
        out = centered * torch.rsqrt(var + self.eps)
        out = out * self.weight.float()
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(x.dtype)


class PreciseGroupNorm(nn.Module):
    """GroupNorm computed with explicit fp32 reductions (same reason)."""

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels)) if affine else None

    def forward(self, x):
        orig_shape = x.shape
        xf = x.float()
        n = orig_shape[0]
        xg = xf.reshape(n, self.num_groups, -1)
        mean = xg.mean(dim=-1, keepdim=True)
        centered = xg - mean
        var = (centered * centered).mean(dim=-1, keepdim=True)
        out = centered * torch.rsqrt(var + self.eps)
        out = out.reshape(orig_shape)
        out = out * self.weight.float()
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(x.dtype)


class LoRA(nn.Module):
    """fla-compatible low-rank block: Linear -> act -> Linear."""

    def __init__(
        self, input_dim, output_dim, low_rank_dim, bias=True, activation="tanh"
    ):
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
    def __init__(
        self,
        hidden_size,
        head_dim,
        decay_low_rank_dim,
        gate_low_rank_dim,
        a_low_rank_dim,
        v_low_rank_dim,
        norm_eps,
        layer_idx,
    ):
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

        self.w_lora = LoRA(
            hidden_size, hidden_size, decay_low_rank_dim, activation="tanh"
        )
        if layer_idx != 0:
            self.v_lora = LoRA(
                hidden_size, self.value_dim, v_low_rank_dim, activation=None
            )
        self.a_lora = LoRA(hidden_size, hidden_size, a_low_rank_dim, activation=None)
        self.g_lora = LoRA(
            hidden_size,
            self.value_dim,
            gate_low_rank_dim,
            activation="sigmoid",
            bias=False,
        )

        self.g_norm = PreciseGroupNorm(
            num_groups=self.num_heads,
            num_channels=self.value_dim,
            eps=head_dim * norm_eps,
            affine=True,
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
        # F.normalize's eps underflows to zero in bf16, so a tiny-norm row
        # divides by ~0 and produces Inf. fla upcasts its l2 norm to fp32;
        # match that here.
        kk = F.normalize(
            (k * self.k_k).view(batch, seq_len, H, D).float(), dim=-1, p=2.0
        ).to(k.dtype)
        # k = k * (1 + (a - 1) * k_a)  (fused_k_rwkv7)
        a_sig_hd = a_sig.view(batch, seq_len, H, D)
        k = k.addcmul(k * (a_sig - 1), self.k_a)

        r4 = r.view(batch, seq_len, H, D)
        w_naive4 = torch.log(-w_log).view(batch, seq_len, H, D)
        k4 = k.view(batch, seq_len, H, D)
        v4 = v.view(batch, seq_len, H, self.head_v_dim)
        a_naive4 = -kk
        b_naive4 = kk * a_sig_hd

        state0 = torch.zeros(
            batch,
            H,
            D,
            self.head_v_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        o, _ = naive_wkv7_forward(r4, w_naive4, k4, v4, a_naive4, b_naive4, state0)

        o = self.g_norm(o.reshape(batch * seq_len, H * self.head_v_dim)).view(
            batch, seq_len, self.value_dim
        )

        # gate_output_correction: (o + ((r*k*r_k).sum * v)) * g
        r_k_b = self.r_k.unsqueeze(0).unsqueeze(0)
        correction_scalar = (r4 * k4 * r_k_b).sum(-1, keepdim=True)
        gated = o.view(batch, seq_len, H, self.head_v_dim) + correction_scalar * v4
        o = gated.reshape(batch, seq_len, self.value_dim) * g

        o = self.o_proj(o)
        return o, v_first


def attention_intermediates(attn, hidden_states):
    """Mirror RWKV7Attention.forward but return every intermediate tensor.

    Used to localize a device-vs-CPU numerical divergence op by op. Only
    valid for layer 0 (v_first = v, no lerp).
    """
    batch, seq_len, _ = hidden_states.shape
    out = {}
    delta = token_shift(hidden_states)
    out["delta"] = delta
    xr = hidden_states + delta * attn.x_r
    xw = hidden_states + delta * attn.x_w
    xk = hidden_states + delta * attn.x_k
    xv = hidden_states + delta * attn.x_v
    xa = hidden_states + delta * attn.x_a
    xg = hidden_states + delta * attn.x_g
    out["xr"] = xr
    out["xw"] = xw
    out["xk"] = xk
    out["xv"] = xv
    out["xa"] = xa
    out["xg"] = xg
    r = attn.r_proj(xr)
    w_log = LOG_SQRT_TWO_PI_MINUS_HALF * torch.sigmoid(attn.w_lora(xw))
    k = attn.k_proj(xk)
    v = attn.v_proj(xv)
    out["r"] = r
    out["w_log"] = w_log
    out["k_pre"] = k
    out["v"] = v
    a_sig = torch.sigmoid(attn.a_lora(xa))
    g = attn.g_lora(xg)
    out["a_sig"] = a_sig
    out["g"] = g
    H, D = attn.num_heads, attn.head_dim
    kk = F.normalize(
        (k * attn.k_k).view(batch, seq_len, H, D).float(), dim=-1, p=2.0
    ).to(k.dtype)
    out["kk"] = kk
    a_sig_hd = a_sig.view(batch, seq_len, H, D)
    k_post = k.addcmul(k * (a_sig - 1), attn.k_a)
    out["k_post"] = k_post
    r4 = r.view(batch, seq_len, H, D)
    w_naive4 = torch.log(-w_log).view(batch, seq_len, H, D)
    k4 = k_post.view(batch, seq_len, H, D)
    v4 = v.view(batch, seq_len, H, attn.head_v_dim)
    a_naive4 = -kk
    b_naive4 = kk * a_sig_hd
    out["w_naive"] = w_naive4
    out["a_naive"] = a_naive4
    out["b_naive"] = b_naive4
    state0 = torch.zeros(
        batch,
        H,
        D,
        attn.head_v_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    o, final_state = naive_wkv7_forward(
        r4, w_naive4, k4, v4, a_naive4, b_naive4, state0
    )
    out["wkv_o"] = o
    out["wkv_state"] = final_state
    o_norm = attn.g_norm(o.reshape(batch * seq_len, H * attn.head_v_dim)).view(
        batch, seq_len, attn.value_dim
    )
    out["o_norm"] = o_norm
    r_k_b = attn.r_k.unsqueeze(0).unsqueeze(0)
    correction_scalar = (r4 * k4 * r_k_b).sum(-1, keepdim=True)
    gated = o_norm.view(batch, seq_len, H, attn.head_v_dim) + correction_scalar * v4
    o_gated = gated.reshape(batch, seq_len, attn.value_dim) * g
    out["o_gated"] = o_gated
    o_final = attn.o_proj(o_gated)
    out["o_final"] = o_final
    return out


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
    def __init__(
        self,
        hidden_size,
        head_dim,
        decay_low_rank_dim,
        gate_low_rank_dim,
        a_low_rank_dim,
        v_low_rank_dim,
        intermediate_size,
        norm_eps,
        norm_first,
        layer_idx,
    ):
        super().__init__()
        self.norm_first = norm_first
        if norm_first and layer_idx == 0:
            self.pre_norm = PreciseLayerNorm(hidden_size, bias=True, eps=norm_eps)
        self.attn_norm = PreciseLayerNorm(hidden_size, bias=True, eps=norm_eps)
        self.attn = RWKV7Attention(
            hidden_size,
            head_dim,
            decay_low_rank_dim,
            gate_low_rank_dim,
            a_low_rank_dim,
            v_low_rank_dim,
            norm_eps,
            layer_idx,
        )
        self.ffn_norm = PreciseLayerNorm(hidden_size, bias=True, eps=norm_eps)
        self.ffn = RWKV7FeedForward(hidden_size, intermediate_size)

    def forward(self, hidden_states, v_first):
        residual = (
            self.pre_norm(hidden_states)
            if (self.norm_first and hasattr(self, "pre_norm"))
            else hidden_states
        )
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
                config["hidden_size"],
                config["head_dim"],
                config["decay_low_rank_dim"],
                config["gate_low_rank_dim"],
                config["a_low_rank_dim"],
                config["v_low_rank_dim"],
                config["intermediate_size"],
                config["norm_eps"],
                config["norm_first"],
                i,
            )
            for i in range(config["num_hidden_layers"])
        ])
        self.norm = PreciseLayerNorm(
            config["hidden_size"], bias=True, eps=config["norm_eps"]
        )

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
        self.lm_head = nn.Linear(
            config["hidden_size"], config["vocab_size"], bias=False
        )

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
