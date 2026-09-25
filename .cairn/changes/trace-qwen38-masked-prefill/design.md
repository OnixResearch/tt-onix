## Context

The service prefills each user through `prefill_masked_bucket`. That path pads
the prompt to a bucket and masks the GDN recurrence past the real length. Both
masks come from `ttnn.from_torch` on the host, and a host write cannot sit inside
a trace. A trace also bakes buffer addresses: a replay writes its scratch into
memory that was free at capture time.

## Decisions

### Decision: Length as device data

**Choice:** `conv_valid_selector` and `seq_valid_mask` build the same host tensors
as before. The FIR conv and the seq kernel accept prebuilt device copies. A
traced replay copies fresh values into persistent buffers. The traced body
passes `valid_len=bucket` to select the masked kernels. The mask then carries the
real length. At full length the mask multiplies by 1.0, which is exact.

**Rationale:** One program set per bucket serves every length. Eager prefill
keeps its own masks, so its numbers do not change.

### Decision: Install inside the trace

**Choice:** Each trace ends by resetting the GDN scratch and computing the
next-token logit. The one-hot row comes from a staged input. A `slot_install`
program (`ttnn.generic_op`, kernel `gdn/kernels/slot_install.cpp`) then copies
the user's state into its decode slot:

- each recurrent state, as a tile slab at `slot * slab_tiles`;
- each conv state and the logits, as row `slot` of one tile row.

The kernel reads the slot index from a staged device buffer. The replays of a
batch are enqueued back to back, and the per-slot logits are read once.

**Rationale:** Host-side clones and `write_slot` cost about 0.37 s per user,
more than the 0.2 s replay. The eager path issues about 1,400 small operations
per user; the install program issues one. Buckets 512 and 1,024 were captured
alongside it, so traced prefill covers every prompt up to the eager chunk size.

### Decision: Full-bucket KV fill

**Choice:** A replay fills KV for the whole bucket through the user's full
page-table row.

**Rationale:** The fill width then no longer depends on the length. Padded rows
land in the user's own blocks past the prompt. Decode rewrites each row before
reading it.

### Decision: Capture once, after every reachable program

**Choice:** At startup the adapter runs decode at every selectable width and
route. It then runs the eager masked buckets up to 1,024, chunked prompts with
tails in every bucket, and a GDN state install into every slot. Only after that
does it capture. Stale traces are released and never recaptured.

**Rationale:** On silicon, any of these orders hung the device:

- capture, then compile a decode width, then replay, then decode;
- capture, replay, compile, then recapture.

Capturing after every compile was exact and stable through a gauntlet of all
routes. A first-time program can allocate buffers that live for the process. If
one lands in trace scratch, the next replay overwrites it. The program-cache
entry count detects such a compile. Releasing is safe; replaying is not.

### Decision: Nothing allocated between replays outlives the next replay

**Choice:**

- Every bucket's inputs are allocated before the first capture.
- Each traced user's logits are read, and its state is installed, before the
  next user runs.
- A batch that mixes traced and eager users runs eagerly.
- Trace reuse (a parked decode trace) runs eagerly.

**Rationale:** Allocating the 256 bucket's inputs after the 128 capture put its
RoPE tables in the 128 trace's scratch. Batches that replayed 128 before 256 then
diverged. Every such input now predates both captures.

### Decision: Full-width prefill page tables

**Choice:** The adapter passes each user's full 32-block row to every prefill
route.

**Rationale:** Request-bounded widths compiled page-table programs per width
after the capture, which released the traces. Eager outputs with full rows equal
the bounded-row outputs on every screened batch.

## Risks

- Startup takes about 220 s longer.
- An unforeseen first-time program releases the traces for the process. The
  service keeps serving eagerly and emits `masked-prefill-traces-released`.
