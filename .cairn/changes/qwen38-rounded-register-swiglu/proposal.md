# Qwen rounded register-resident SwiGLU

Implement a default-off shape-qualified local Qwen epilogue retaining explicit BF16 rounding of SiLU gate and up before their product. Inspect exact 0.77 source; do not equate upstream T16 with local B1/B8.

This is a new mechanism, not advancement of the rejected bank-matched traffic-reduction candidate. Historical rejection remains unchanged.
