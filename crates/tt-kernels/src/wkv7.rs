//! One head of the RWKV-7 WKV7 recurrence on the SFPU.
//!
//! The token-at-a-time Goose equation for one head of size 64
//! (`crates/rwkv7/src/lib.rs`, `step`):
//!
//! ```text
//! sa_i    = sum_j a[j] * S[i, j]
//! S'[i,j] = S[i, j] * w[j] + sa_i * b[j] + k[j] * v[i]
//! out_i   = sum_j S'[i, j] * r[j]
//! ```
//!
//! The head's state is four 32x32 tiles (`[64, 64]`), and every vector is two
//! tiles wide. A vector enters as a tile the host has already broadcast: a
//! row-vector tile has every row equal to one 32-value half, and a column
//! vector tile has every column equal to one 32-value half. No arithmetic
//! happens on the host and none is needed on the mover.
//!
//! # Instruction programme
//!
//! `Dst` holds all four state tiles at rows 0, 64, 128 and 192, two vector
//! slots at 256 and 320, and two accumulator tiles at 384 and 448 -- the
//! whole 512 rows. The unpack role serves the vector slots as the math role
//! asks for them, two at a time; the pack role writes the four updated state
//! tiles and the two output tiles.
//!
//! The math role is three phases:
//!
//! 1. **`sa`**: for each row tile `p`, multiply the two state column tiles by
//!    the two halves of `a`, fold the 32 columns of each into its row, add the
//!    two halves and keep the result in accumulator `p`. The fold is the same
//!    in-tile column reduction `sfpu::reduce` uses: four lane groups (the two
//!    column halves of both faces) added, then seven `SFPSHFT2` rotations
//!    across the eight lanes of a row. The result is replicated across the
//!    row's columns, ready to meet `b`.
//! 2. **update**: for each of the four state tiles, `S = S*w + sa*b`, then
//!    `S = k*v + S`. Both are replayed row loops over the tile's 32 quads.
//! 3. **`out`**: for each row tile `p`, multiply the two updated column tiles
//!    by the two halves of `r`, fold and add as in phase 1. The result is a
//!    tile whose every column holds `out_i`.
//!
//! The arithmetic order is not the host reference's: `sa` and `out` are a
//! tree (four products added, then eight lanes), and two terms of the update
//! are `SFPMAD` (one rounding each). Both are within the reduction's stated
//! tree bound and the gate holds the result to the host reference.
//!
//! # Semaphores
//!
//! Two vector slots, each with a `ready` (unpack -> math) and a `free` (math ->
//! unpack) semaphore, so the unpacker fills the next slot while the math reads
//! the other. `state_ready` announces the four state tiles, `state_computed`
//! the updated ones, and `out_computed` the output tiles; all posts on the math
//! side are `post_after(Unit::Sfpu, ..)`, so the consumer cannot see a tile the
//! Vector Unit is still writing.

use tt_isa::backend::Before;
use tt_isa::dm::TILE_SLOT;
use tt_isa::isa::Instruction;
use tt_isa::sync::{self, Semaphore, Unit};

use crate::datapath::{
    clear_unpacker0_adcs, config_program, pack_tile_from_dst, state_id, thread_config,
    tile_unpack_config, unpack_tile_to_dst,
};
use crate::l1::{PlanError, Requirements};
use crate::runtime::SemaphoreInit;
use crate::sfpu::{Format, LReg, Program};

/// State and vector length of one head.
pub const HEAD_SIZE: usize = 64;
/// Tiles one head's state occupies: two row tiles by two column tiles.
pub const STATE_TILES: usize = 4;
/// Tiles one 64-value vector occupies: two halves.
pub const VECTOR_TILES: usize = 2;

/// `Dst` row of each state tile, in `(row tile, column tile)` order.
pub const STATE_ROW: [u32; STATE_TILES] = [0, 64, 128, 192];
/// `Dst` rows of the two vector slots the unpacker fills.
pub const SLOT_ROW: [u32; VECTOR_TILES] = [256, 320];
/// `Dst` rows of the two accumulator tiles.
pub const ACC_ROW: [u32; 2] = [384, 448];

/// The kernel's semaphores, as a plan placed them.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Semaphores {
    /// T0 -> T1: the four state tiles are in `Dst`.
    pub state_ready: Semaphore,
    /// T1 -> T2: the four state tiles have been updated.
    pub state_computed: Semaphore,
    /// T1 -> T2: the two output tiles are in `Dst`.
    pub out_computed: Semaphore,
    /// T0 -> T1 / T1 -> T0 for vector slot 0.
    pub slot0_ready: Semaphore,
    pub slot0_free: Semaphore,
    /// T0 -> T1 / T1 -> T0 for vector slot 1.
    pub slot1_ready: Semaphore,
    pub slot1_free: Semaphore,
}

/// Where one head's operands and results live in L1, and its semaphores.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    /// The four state tiles, `(row tile, column tile)` order.
    pub state_in: [u64; STATE_TILES],
    /// The two tiles of each row-broadcast vector.
    pub a: [u64; VECTOR_TILES],
    pub w: [u64; VECTOR_TILES],
    pub b: [u64; VECTOR_TILES],
    pub k: [u64; VECTOR_TILES],
    pub r: [u64; VECTOR_TILES],
    /// The two column-broadcast tiles of `v`; tile `p` is `v[32p..32p + 32]`
    /// down its rows.
    pub v: [u64; VECTOR_TILES],
    /// The four updated state tiles, `(row tile, column tile)` order.
    pub state_out: [u64; STATE_TILES],
    /// The two output tiles; every column of tile `p` holds `out[32p..32p+32]`.
    pub out: [u64; VECTOR_TILES],
    pub sems: Semaphores,
    /// The values a concurrent run starts the semaphores at.
    pub init: Vec<SemaphoreInit>,
}

/// Plan the slots and semaphores of one head in the data arena.
pub fn plan_layout() -> Result<Layout, PlanError> {
    let mut req = Requirements::new(1);
    let align = tt_isa::dram::ALIGN;
    let mut slot = |name: &'static str| req.scratch(name, TILE_SLOT, align, 0..1);
    let state_in = std::array::from_fn(|_| slot("wkv7 state in"));
    let a = std::array::from_fn(|_| slot("wkv7 a"));
    let w = std::array::from_fn(|_| slot("wkv7 w"));
    let b = std::array::from_fn(|_| slot("wkv7 b"));
    let k = std::array::from_fn(|_| slot("wkv7 k"));
    let r = std::array::from_fn(|_| slot("wkv7 r"));
    let v = std::array::from_fn(|_| slot("wkv7 v"));
    let state_out = std::array::from_fn(|_| slot("wkv7 state out"));
    let out = std::array::from_fn(|_| slot("wkv7 out"));
    let state_ready = req.semaphore("wkv7 state ready", 0, 0..1);
    let state_computed = req.semaphore("wkv7 state computed", 0, 0..1);
    let out_computed = req.semaphore("wkv7 out computed", 0, 0..1);
    let slot0_ready = req.semaphore("wkv7 slot 0 ready", 0, 0..1);
    let slot0_free = req.semaphore("wkv7 slot 0 free", 1, 0..1);
    let slot1_ready = req.semaphore("wkv7 slot 1 ready", 0, 0..1);
    let slot1_free = req.semaphore("wkv7 slot 1 free", 1, 0..1);
    let plan = req.plan(tt_isa::l1::DATA)?;
    let addr = |bufs: [crate::l1::Buf; VECTOR_TILES]| bufs.map(|b| plan.addr(b));
    Ok(Layout {
        state_in: state_in.map(|b| plan.addr(b)),
        a: addr(a),
        w: addr(w),
        b: addr(b),
        k: addr(k),
        r: addr(r),
        v: addr(v),
        state_out: state_out.map(|b| plan.addr(b)),
        out: addr(out),
        sems: Semaphores {
            state_ready: plan.semaphore(state_ready),
            state_computed: plan.semaphore(state_computed),
            out_computed: plan.semaphore(out_computed),
            slot0_ready: plan.semaphore(slot0_ready),
            slot0_free: plan.semaphore(slot0_free),
            slot1_ready: plan.semaphore(slot1_ready),
            slot1_free: plan.semaphore(slot1_free),
        },
        init: plan.semaphore_init(),
    })
}

/// Each vector tile's `(row tile, column tile)` source, in the order the math
/// role asks for them: the two halves of `a`, then for each state tile `w`,
/// `b`, `k` and `v`, then the two halves of `r` twice, once per row tile.
fn vector_sequence(l: &Layout) -> Vec<u64> {
    let mut seq = vec![l.a[0], l.a[1]];
    for p in 0..2 {
        for t in 0..2 {
            seq.extend([l.w[t], l.b[t], l.k[t], l.v[p]]);
        }
    }
    seq.extend([l.r[0], l.r[1], l.r[0], l.r[1]]);
    seq
}

/// The three role programs of one head: unpack, math, pack.
pub fn roles(l: &Layout) -> [Vec<Instruction>; 3] {
    let s = l.sems;
    let sequence = vector_sequence(l);
    let slots = |index: usize| -> (&Semaphore, &Semaphore) {
        if index % 2 == 0 {
            (&s.slot0_ready, &s.slot0_free)
        } else {
            (&s.slot1_ready, &s.slot1_free)
        }
    };

    let mut unpack = thread_config();
    unpack.extend(clear_unpacker0_adcs());
    let mut words = tt_isa::backend::ConfigWords::new();
    tile_unpack_config(&mut words, l.state_in[0]);
    unpack.extend(config_program(&words));
    for (tile, row) in l.state_in.iter().zip(STATE_ROW) {
        unpack.extend(unpack_tile_to_dst(*tile, row));
    }
    unpack.extend(sync::post_after(Unit::Unpacker0, s.state_ready));
    for (index, tile) in sequence.iter().enumerate() {
        let (ready, free) = slots(index);
        let row = SLOT_ROW[index % 2];
        unpack.extend(sync::take(*free, Before::UNPACKER));
        unpack.extend(unpack_tile_to_dst(*tile, row));
        unpack.extend(sync::post_after(Unit::Unpacker0, *ready));
    }
    unpack.push(tt_isa::backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());

    let mut math = vec![state_id()];
    math.extend(sync::take(s.state_ready, Before::SFPU));
    // Both row tiles use the same pair of `a` halves.
    math.extend(sync::take(s.slot0_ready, Before::SFPU));
    math.extend(sync::take(s.slot1_ready, Before::SFPU));
    let mut sa = Program::new();
    fold_pair(
        &mut sa,
        STATE_ROW[0],
        SLOT_ROW[0],
        STATE_ROW[1],
        SLOT_ROW[1],
        ACC_ROW[0],
    );
    fold_pair(
        &mut sa,
        STATE_ROW[2],
        SLOT_ROW[0],
        STATE_ROW[3],
        SLOT_ROW[1],
        ACC_ROW[1],
    );
    math.extend(sa.finish());
    math.extend(sync::post_after(Unit::Sfpu, s.slot0_free));
    math.extend(sync::post_after(Unit::Sfpu, s.slot1_free));
    for (p, t) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
        let state = STATE_ROW[2 * p + t];
        // `S*w + sa*b`, with the two vectors the unpacker brought.
        math.extend(sync::take(s.slot0_ready, Before::SFPU));
        math.extend(sync::take(s.slot1_ready, Before::SFPU));
        let mut stage = Program::new();
        update_ab(&mut stage, state, SLOT_ROW[0], SLOT_ROW[1], ACC_ROW[p]);
        math.extend(stage.finish());
        math.extend(sync::post_after(Unit::Sfpu, s.slot0_free));
        math.extend(sync::post_after(Unit::Sfpu, s.slot1_free));
        // `k*v + S`.
        math.extend(sync::take(s.slot0_ready, Before::SFPU));
        math.extend(sync::take(s.slot1_ready, Before::SFPU));
        let mut stage = Program::new();
        update_kv(&mut stage, state, SLOT_ROW[0], SLOT_ROW[1]);
        math.extend(stage.finish());
        math.extend(sync::post_after(Unit::Sfpu, s.slot0_free));
        math.extend(sync::post_after(Unit::Sfpu, s.slot1_free));
    }
    math.extend(sync::post_after(Unit::Sfpu, s.state_computed));
    for p in 0..2 {
        math.extend(sync::take(s.slot0_ready, Before::SFPU));
        math.extend(sync::take(s.slot1_ready, Before::SFPU));
        let mut fold = Program::new();
        fold_pair(
            &mut fold,
            STATE_ROW[2 * p],
            SLOT_ROW[0],
            STATE_ROW[2 * p + 1],
            SLOT_ROW[1],
            ACC_ROW[p],
        );
        math.extend(fold.finish());
        math.extend(sync::post_after(Unit::Sfpu, s.slot0_free));
        math.extend(sync::post_after(Unit::Sfpu, s.slot1_free));
    }
    math.extend(sync::post_after(Unit::Sfpu, s.out_computed));

    let mut pack = vec![state_id()];
    pack.extend(sync::take(s.state_computed, Before::PACKER));
    for (tile, row) in l.state_out.iter().zip(STATE_ROW) {
        pack.extend(pack_tile_from_dst(*tile + tt_isa::dm::TILE_DATA, row));
    }
    pack.extend(sync::take(s.out_computed, Before::PACKER));
    for (tile, row) in l.out.iter().zip(ACC_ROW) {
        pack.extend(pack_tile_from_dst(*tile + tt_isa::dm::TILE_DATA, row));
    }
    pack.push(tt_isa::backend::wait_for_packer(Before::EVERYTHING).unwrap());

    [unpack, math, pack]
}

/// `out = fold(S0 * V0 + S1 * V1)` for one 32-row tile.
///
/// `S` and `V` are `Dst` tile bases, 64 rows each. A 32x32 tile's four faces
/// put logical rows `0..16` in `Dst` rows `0..32` (faces 0 and 1) and
/// `16..32` in `32..64` (faces 2 and 3); a logical group of four rows is two
/// addresses in each face, `base + q` and `base + q + 2`. The eight groups
/// `q = 0, 4, .., 60` (in the face order `0..16, 32..48`) are written out.
///
/// Within a group the four quads (two column halves of each of the two faces)
/// are added lane-wise -- four products a lane -- then seven lane rotations add
/// the eight lanes of each row. The group's sum is stored back to all four
/// quads, so the output tile holds every row's result in all 32 of its
/// columns.
fn fold_pair(p: &mut Program, s0: u32, v0: u32, s1: u32, v1: u32, out: u32) {
    for group in 0..8u32 {
        let q = if group < 4 {
            4 * group
        } else {
            32 + 4 * (group - 4)
        };
        let quads = [q, q + 2, q + 16, q + 18];
        // S0 * V0 in L0, with L1 the vector and L2 the next tile element.
        p.load(LReg::L0, Format::Fp32, s0 + quads[0]);
        p.load(LReg::L1, Format::Fp32, v0 + quads[0]);
        p.mul(LReg::L0, LReg::L1, LReg::L0);
        for &at in &quads[1..] {
            p.load(LReg::L2, Format::Fp32, s0 + at);
            p.load(LReg::L1, Format::Fp32, v0 + at);
            p.mul(LReg::L2, LReg::L1, LReg::L2);
            p.add(LReg::L0, LReg::L2, LReg::L0);
        }
        // S1 * V1 in L3.
        p.load(LReg::L3, Format::Fp32, s1 + quads[0]);
        p.load(LReg::L1, Format::Fp32, v1 + quads[0]);
        p.mul(LReg::L3, LReg::L1, LReg::L3);
        for &at in &quads[1..] {
            p.load(LReg::L2, Format::Fp32, s1 + at);
            p.load(LReg::L1, Format::Fp32, v1 + at);
            p.mul(LReg::L2, LReg::L1, LReg::L2);
            p.add(LReg::L3, LReg::L2, LReg::L3);
        }
        p.add(LReg::L0, LReg::L3, LReg::L0);
        // Across the eight lanes of each row (`rotate_row` is within a group
        // of eight lanes, one row of four `Dst` rows by eight columns).
        p.mov(LReg::L0, LReg::L2);
        for _ in 0..7 {
            p.rotate_row(LReg::L2, LReg::L2);
            p.add(LReg::L0, LReg::L2, LReg::L0);
        }
        for &at in &quads {
            p.store(LReg::L0, Format::Fp32, out + at);
        }
    }
}

/// `S = S*W + SA*B`, with `W` and `B` in the two vector slots and `SA` in an
/// accumulator tile. One replayed row loop over the tile's 32 quads.
fn update_ab(p: &mut Program, state: u32, w: u32, b: u32, sa: u32) {
    p.for_each_row_group(64, |p, o| {
        p.load(LReg::L0, Format::Fp32, state + o);
        p.load(LReg::L1, Format::Fp32, w + o);
        p.load(LReg::L2, Format::Fp32, b + o);
        p.load(LReg::L3, Format::Fp32, sa + o);
        p.mul(LReg::L0, LReg::L1, LReg::L0);
        p.mad(LReg::L3, LReg::L2, LReg::L0, LReg::L0);
        p.store(LReg::L0, Format::Fp32, state + o);
    });
}

/// `S = K*V + S`, with `K` and `V` in the two vector slots.
fn update_kv(p: &mut Program, state: u32, k: u32, v: u32) {
    p.for_each_row_group(64, |p, o| {
        p.load(LReg::L0, Format::Fp32, state + o);
        p.load(LReg::L1, Format::Fp32, k + o);
        p.load(LReg::L2, Format::Fp32, v + o);
        p.mad(LReg::L1, LReg::L2, LReg::L0, LReg::L0);
        p.store(LReg::L0, Format::Fp32, state + o);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_roles_fit_the_program_slot_and_their_semaphores_balance() {
        let layout = plan_layout().expect("one head fits the data arena");
        let roles = roles(&layout);
        for (role, program) in roles.iter().enumerate() {
            assert!(
                program.len() <= tt_isa::mailbox::PROGRAM_MAX as usize,
                "role {role}: {} instructions past the {} the slot holds",
                program.len(),
                tt_isa::mailbox::PROGRAM_MAX
            );
        }
        // Every post is matched by a take: a producer that runs ahead or a
        // consumer that waits for a hand-off no one makes is the deadlock this
        // checks without a device.
        let mut posts = std::collections::BTreeMap::new();
        let mut gets = std::collections::BTreeMap::new();
        for program in &roles {
            for instruction in program {
                let mask = instruction.operand("SemaphoreMask");
                match instruction.def().mnemonic() {
                    "SEMPOST" => *posts.entry(mask).or_insert(0usize) += 1,
                    "SEMGET" => *gets.entry(mask).or_insert(0usize) += 1,
                    _ => {}
                }
            }
        }
        assert_eq!(posts, gets, "semaphore posts and takes do not balance");
    }
}
