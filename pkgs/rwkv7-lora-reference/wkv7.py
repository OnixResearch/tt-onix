"""Naive WKV7 forward and backward equations from the RWKV-7 paper, Appendix H.

These are bounded, device-free reference equations. The forward path is
differentiable under PyTorch autograd. The manual backward is an independent
oracle. Both run in FP32 on bounded tensors and make no hardware claim.
"""

from __future__ import annotations

import torch

FINITE_TOLERANCE = 1e-5


def wkv7_forward(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paper-matched naive WKV7 forward over a bounded sequence.

    Inputs have shape (B, T, H, N) except state which is (B, H, N, N).
    `w` is the raw pre-activation value; the kernel applies exp(-exp(w)).
    Returns (out, final_state).
    """
    batch, time, heads, width = r.shape
    state_cache = torch.zeros((batch, time + 1, heads, width, width), dtype=r.dtype)
    state_cache[:, 0, :] = state
    wactivated = torch.exp(-torch.exp(w))
    out = torch.zeros((batch, time, heads, width), dtype=r.dtype)
    current = state
    for t in range(time):
        kk = k[:, t, :]
        rr = r[:, t, :]
        vv = v[:, t, :]
        aa = a[:, t, :]
        bb = b[:, t, :]
        current = (
            current * wactivated[:, t, :, None, :]
            + torch.einsum("bhik,bhk,bhj->bhij", current, aa, bb)
            + torch.einsum("bhj,bhi->bhij", kk, vv)
        )
        state_cache[:, t + 1, :] = current
        out[:, t, :] = torch.einsum("bhj,bhij->bhi", rr, current)
    return out, current


def wkv7_backward(
    r: torch.Tensor,
    w0: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gout: torch.Tensor,
    gstate: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Paper-matched naive WKV7 backward oracle.

    Returns (gr, gw, gk, gv, ga, gb, gstate) with each input gradient shaped
    (B, T, H, N).
    """
    batch, time, heads, width = r.shape
    gr = torch.zeros((batch, time, heads, width), dtype=r.dtype)
    gw = torch.zeros((batch, time, heads, width), dtype=r.dtype)
    gk = torch.zeros((batch, time, heads, width), dtype=r.dtype)
    gv = torch.zeros((batch, time, heads, width), dtype=r.dtype)
    ga = torch.zeros((batch, time, heads, width), dtype=r.dtype)
    gb = torch.zeros((batch, time, heads, width), dtype=r.dtype)

    wactivated = torch.exp(-torch.exp(w0))
    state_cache = torch.zeros((batch, time + 1, heads, width, width), dtype=r.dtype)
    state = torch.zeros((batch, heads, width, width), dtype=r.dtype)
    state_cache[:, 0, :] = state
    for t in range(time):
        state = (
            state * wactivated[:, t, :, None, :]
            + torch.einsum("bhik,bhk,bhj->bhij", state, a[:, t, :], b[:, t, :])
            + torch.einsum("bhj,bhi->bhij", k[:, t, :], v[:, t, :])
        )
        state_cache[:, t + 1, :] = state

    for t in range(time - 1, -1, -1):
        gr[:, t, :] = torch.matmul(
            gout[:, t, :, None, :], state_cache[:, t + 1, :]
        ).squeeze(-2)
        gstate = gstate + torch.matmul(gout[:, t, :, :, None], r[:, t, :, None, :])
        gk[:, t, :] = torch.matmul(v[:, t, :, None, :], gstate).squeeze(-2)
        gv[:, t, :] = torch.matmul(gstate, k[:, t, :, :, None]).squeeze(-1)
        ga[:, t, :] = torch.einsum(
            "bhik,bhj,bhij->bhk", state_cache[:, t, :], b[:, t, :], gstate
        )
        gb[:, t, :] = torch.einsum(
            "bhik,bhk,bhij->bhj", state_cache[:, t, :], a[:, t, :], gstate
        )
        gw[:, t, :] = torch.einsum("bhij,bhij->bhj", state_cache[:, t, :], gstate)
        gstate = torch.einsum(
            "bhj,bhij->bhij", wactivated[:, t, :], gstate
        ) + torch.einsum("bhk,bhj,bhij->bhik", a[:, t, :], b[:, t, :], gstate)

    gw = -torch.exp(w0 - torch.exp(w0)) * gw
    return gr, gw, gk, gv, ga, gb, gstate


def autograd_gradients(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
    gout: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Autograd gradients of the forward output dotted with gout."""
    inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (r, w, k, v, a, b)]
    state_input = state.detach().clone()
    out, _ = wkv7_forward(*inputs, state_input)
    torch.autograd.backward(out, gout)
    return tuple(tensor.grad for tensor in inputs)


def max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    return float((left - right).abs().max().item())


def finite_difference_gradient(
    fn,
    tensor: torch.Tensor,
    index: tuple[int, ...],
    epsilon: float = 1e-4,
) -> float:
    """Central finite-difference derivative of scalar fn with respect to one element."""
    original = tensor.detach().clone()
    plus = original.clone()
    minus = original.clone()
    plus[index] = plus[index] + epsilon
    minus[index] = minus[index] - epsilon
    return float((fn(plus) - fn(minus)) / (2 * epsilon))


def require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains a nonfinite value")
