# Native decode strip matvec and view consumers

Implement a default-off M=1 native decode specialization using existing storage, checked dataflow ownership and original accumulation order. Identify an actual RWKV view materialization before adding a narrow F32 consumer; never add arbitrary aliasing.

This is a new mechanism, not advancement of the rejected bank-matched traffic-reduction candidate. Historical rejection remains unchanged.
