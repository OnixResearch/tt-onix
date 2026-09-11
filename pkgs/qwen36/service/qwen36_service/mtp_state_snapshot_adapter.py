"""Stable, request-owned snapshots for exact speculative rollback."""

import ttnn

from .mtp_rollback_adapter import _device_addresses as tensor_identity


class MtpStateSnapshot:
    """Copy live tensors without replacing any model-owned buffer."""

    def __init__(self, tensors):
        self._tensors = tensors
        self._pairs = []
        self._identities = []
        self._closed = False
        try:
            for tensor in tensors():
                if tensor is None:
                    raise RuntimeError("speculative state tensor is not allocated")
                self._identities.append(tensor_identity(tensor))
                self._pairs.append((tensor, ttnn.clone(tensor)))
            if not self._pairs:
                raise RuntimeError("speculative state snapshot is empty")
        except BaseException:
            self.close()
            raise

    def _require_live(self):
        if self._closed:
            raise RuntimeError("speculative state snapshot is closed")
        observed = [tensor_identity(live) for live in self._tensors()]
        if observed != self._identities:
            raise RuntimeError("speculative state binding changed")

    def saved_tensor(self, tensor):
        """Borrow one request-entry tensor without replacing its allocation."""
        self._require_live()
        identity = tensor_identity(tensor)
        matches = [
            saved for live, saved in self._pairs if tensor_identity(live) == identity
        ]
        if len(matches) != 1:
            raise RuntimeError("speculative snapshot tensor is missing or ambiguous")
        return matches[0]

    def refresh(self):
        self._require_live()
        for live, saved in self._pairs:
            ttnn.copy(live, saved)

    def restore(self):
        self._require_live()
        for live, saved in self._pairs:
            ttnn.copy(saved, live)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for _, saved in self._pairs:
            ttnn.deallocate(saved)
        self._pairs = []


def target_state_tensors(model):
    """Return all target KV and recurrent state, including convolution carry."""
    tensors = []
    for layer in model.layers:
        attention = layer.attention
        if layer.is_full_attention:
            tensors.extend((attention.paged_k, attention.paged_v))
        else:
            if attention.conv_states is None:
                raise RuntimeError("speculative convolution state is not allocated")
            tensors.extend((
                attention.rec_state,
                *attention.conv_states,
                attention.conv_carry,
            ))
    return tuple(tensors)
