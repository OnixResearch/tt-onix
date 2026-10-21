import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[4] / "pkgs/rwkv-layer-harness/reference")
)
from rwkv7_torch_equation_reference import run_reference

model = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    final_hidden, logits, states = run_reference(model)
for name, value in [
    ("final_hidden", final_hidden),
    ("logits", logits),
    ("states", states),
]:
    array = value.detach().cpu().numpy().astype("<f4")
    array.tofile(out / f"{name}.f32")
    print(name, array.shape, float(np.max(np.abs(array))))
