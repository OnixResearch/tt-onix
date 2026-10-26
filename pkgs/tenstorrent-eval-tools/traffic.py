"""Analytic output-stationary traffic; equations follow pinned tensixfuse (MIT).

Source: https://github.com/tanaymihani/tensixfuse/blob/6331e86b2169e7453f0a209b9d7e880dfa0aa38b/src/tensixfuse/traffic.py
This is a model, never a hardware counter or a claim about a particular kernel.
"""

from dataclasses import dataclass

PAYLOAD = {"float32": 4096, "bfloat16": 2048, "bfloat8_b": 1024, "bfloat4_b": 512}
EXPONENT = {name: 64 if name in ("bfloat8_b", "bfloat4_b") else 0 for name in PAYLOAD}


def storage(tiles, dtype):
    if dtype not in PAYLOAD:
        raise ValueError("unsupported dtype: " + str(dtype))
    payload = tiles * PAYLOAD[dtype]
    exponent = tiles * EXPONENT[dtype]
    return {
        "tiles": tiles,
        "payload_bytes": payload,
        "exponent_bytes": exponent,
        "packed_bytes": payload + exponent,
    }


@dataclass(frozen=True)
class Config:
    m: int
    k: int
    n: int
    bm: int = 1
    bn: int = 1
    bk: int = 1
    cols: int = 1
    rows: int = 1
    multicast: bool = False

    def __post_init__(self):
        if any(
            type(x) is not int or x <= 0
            for x in (
                self.m,
                self.k,
                self.n,
                self.bm,
                self.bn,
                self.bk,
                self.cols,
                self.rows,
            )
        ):
            raise ValueError(
                "dimensions, tile blocks and grid dimensions must be positive integers"
            )
        mt, kt, nt = self.tiles
        if kt % self.bk or mt % self.bm or nt % self.bn:
            raise ValueError("tile blocks must divide padded tile dimensions")
        if self.multicast and (mt // self.bm % self.rows or nt // self.bn % self.cols):
            raise ValueError(
                "multicast output blocks must divide evenly over grid rows/columns"
            )

    @property
    def tiles(self):
        return tuple((x + 31) // 32 for x in (self.m, self.k, self.n))


def model(
    cfg,
    a="bfloat16",
    b="bfloat4_b",
    output="bfloat16",
    accumulator="float32",
    buffers=2,
):
    if type(buffers) is not int or buffers < 1:
        raise ValueError("buffer count must be a positive integer")
    for dtype in (a, b, output, accumulator):
        if dtype not in PAYLOAD:
            raise ValueError("unsupported dtype: " + str(dtype))
    mt, kt, nt = cfg.tiles
    mb, nb = mt // cfg.bm, nt // cfg.bn
    ar, br = mt * kt * nb, kt * nt * mb
    if cfg.multicast:
        ar //= cfg.cols
        br //= cfg.rows
    compulsory = {
        "a": storage(mt * kt, a),
        "b": storage(kt * nt, b),
        "c": storage(mt * nt, output),
        "y": storage(mt * nt, output),
    }
    streams = {
        "a": storage(ar, a),
        "b": storage(br, b),
        "c": compulsory["c"],
        "y": compulsory["y"],
    }
    reread = {"a": storage(ar - mt * kt, a), "b": storage(br - kt * nt, b)}
    intermediates = {
        "y1_write_read": storage(2 * mt * nt, output),
        "y2_write_read": storage(2 * mt * nt, output),
    }
    l1 = {
        "a": storage(buffers * cfg.bm * cfg.bk, a),
        "b": storage(buffers * cfg.bk * cfg.bn, b),
        "c": storage(buffers * cfg.bm * cfg.bn, output),
        "accumulator": storage(buffers * cfg.bm * cfg.bn, accumulator),
        "y": storage(buffers * cfg.bm * cfg.bn, output),
    }

    def total(entries):
        return sum(v["packed_bytes"] for v in entries.values())

    no_mcast = (
        storage(mt * kt * nb - ar, a)["packed_bytes"]
        + storage(kt * nt * mb - br, b)["packed_bytes"]
    )
    fused = total(streams)
    return {
        "evidence_kind": "analytic-model-not-hardware-counters",
        "operation": "y=act(a@b+c)",
        "logical_shape_mkn": [cfg.m, cfg.k, cfg.n],
        "padded_shape_mkn": [32 * x for x in cfg.tiles],
        "logical_payload_bytes": {
            "a": cfg.m * cfg.k * PAYLOAD[a] / 1024,
            "b": cfg.k * cfg.n * PAYLOAD[b] / 1024,
            "c": cfg.m * cfg.n * PAYLOAD[output] / 1024,
            "y": cfg.m * cfg.n * PAYLOAD[output] / 1024,
        },
        "payload_semantics": "logical_payload_bytes excludes padding/exponents; all storage entries count padded tile payload plus explicit exponent bytes",
        "block_tiles_mnk": [cfg.bm, cfg.bn, cfg.bk],
        "grid_columns_rows": [cfg.cols, cfg.rows],
        "multicast": cfg.multicast,
        "compulsory": compulsory,
        "fused_streams": streams,
        "rereads": reread,
        "multicast_saved_bytes": no_mcast,
        "unfused_intermediates": intermediates,
        "l1_buffers_per_core": l1,
        "totals": {
            "compulsory_bytes": total(compulsory),
            "reread_bytes": total(reread),
            "fused_bytes": fused,
            "unfused_bytes": fused + total(intermediates),
            "intermediate_saved_bytes": total(intermediates),
            "l1_bytes_per_core": total(l1),
        },
        "limits": "Tile padding explicit; no cache/residency, NoC, collectives, allocator or throughput prediction. L1 is data buffers only, not firmware/reserved capacity. C and Y use output dtype; intermediates materialize in output dtype.",
    }
