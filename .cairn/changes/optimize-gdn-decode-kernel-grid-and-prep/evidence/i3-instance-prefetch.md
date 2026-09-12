# I3: instance-loop reader prefetch (mechanism, guard, and what must be measured)

Recorded 2026-09-12 for change `optimize-gdn-decode-kernel-grid-and-prep`.

## Why this is the retention lever, not the state split

The attribution (`evidence/a1-a4-attribution.md`) shows the admitted decode is
idle at B=1 (24 instances on 110 cores) and **looping** at B>=8 (two instances per
core at B=8, four at B=16). The fused decode spends 26.34 ms of a 54.34 ms step
(0.549 ms per layer) and the per-user retention falls exactly with the loop count:
0.88 at one instance per core, 0.735 at two, 0.634 at four.

A state split cannot help there: instances grow by the split factor while
per-instance work falls by the same factor, so per-core work is unchanged, and the
per-instance *fixed* cost (the `K x K` transpose, reader, CB setup, launch) grows
with the split. So the mechanism that can move the loop case is **overlapping the
instances a core already has**, which is this task.

## The mechanism

`pkgs/tt-metal/gdn-decode-instance-prefetch.patch`, applied only by the GDN grid
variant after the chunking patch:

```cpp
const uint32_t instance_buffers = (heads_per_core > 1) ? 2 : 1;
make_cb(0, key_tiles * instance_buffers);    // q
make_cb(1, key_tiles * instance_buffers);    // k
make_cb(2, val_tiles * instance_buffers);    // v
make_cb(3, instance_buffers);                // beta
make_cb(4, instance_buffers);                // g
make_cb(5, state_tiles * instance_buffers);  // state input
```

With one instance per core the depth is one, so the memory layout is unchanged and
every in-range receipt stays valid. Past the core count the reader-side buffers
hold a second instance, so the reader can reserve, read, and push instance `i+1`
while compute works on instance `i`: the per-instance read latency moves behind
compute instead of sitting in front of it. The reader needed no code change and
the compute kernel is untouched, so the change is purely a depth.

## The guard

The variant derivation refuses a prefetch patch that touches anything but the
factory:

```nix
test "$(grep -c '^--- ' .../gdn-decode-instance-prefetch.patch)" -eq 1
grep -q 'gated_delta_decode_program_factory\.cpp' .../gdn-decode-instance-prefetch.patch
```

That is what makes exact parity the right expectation. A kernel hunk would change
numerics and must fail here, not in a parity screen.

## What must be measured before any claim

The change is a scheduling change, so the order is fixed:

1. **Exact parity first.** Output and recurrent state must be bit-identical to the
   admitted grid-cap kernel at a looping width. B=8 is the cheaper of the two
   looping widths (two instances per core) and B=16 the more demanding (four).
   This is task V6.
2. **Then the step time.** Compare batch step time and per-user retention against
   the same width on the admitted kernel. This is task V7.
3. **Then the disabled state.** With the variant absent, the served route must
   still be the admitted kernel (V8).

No timing may be read before parity passes, and a parity or occupancy failure
keeps the admitted kernel (D2).

## Build cost, recorded

Changing a runtime patch forces a full tt-metal rebuild rather than a cached
package relink: the variant runtime is otherwise reused from the store, so this is
the first change in this line that pays a multi-hour compile under the machine's
current load. That is a property of touching the runtime, and it applies to every
remaining kernel task in this change (I2, I3, I4).

## Status

- patch written and wired into the variant with the factory-only guard;
- variant build running; the device-free gate and the matched screen follow it;
- no timing claim, no parity claim, and no admission movement yet.
