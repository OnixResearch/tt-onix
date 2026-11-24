---- MODULE PrefixCacheLifecycle ----
(*
 * Bounded TLA+ model of the Qwen3.8 hybrid prefix-cache entry lifecycle,
 * slot KV-block allocation, structured-read invalidation, and request cancellation.
 *
 * Source pinned at commit 5fd2ad6f339007560a678bcb5fa3126b84e95b26:
 *   - pkgs/qwen36/service/qwen36_service/continuous_slot_core.py
 *   - pkgs/qwen36/service/qwen36_service/prefix_cache_core.py
 *   - pkgs/qwen36/service/qwen36_service/continuous_slot_service.py
 *
 * See pkgs/qwen36/formal/README.md for the action-to-source-line mapping.
 *)
EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS
    NumSlots,
    MaxEntries,
    NumBlocks,
    NumStateSlots,
    MaxCachedBlocks,
    GuardEvictUnpinnedOnly,
    GuardProtectedBlocks,
    GuardReadRequiresIdle,
    GuardCancelRunningOnly,
    GuardRetireCompleteAtBoundary

VARIABLES
    slots,
    entries,
    readyWriteViolation,
    unpinMissingViolation,
    cancelOverwriteViolation,
    unretiredCompleteViolation

vars == <<
    slots,
    entries,
    readyWriteViolation,
    unpinMissingViolation,
    cancelOverwriteViolation,
    unretiredCompleteViolation
>>

NULL == 0
Slots == 1..NumSlots
EntryIds == 1..MaxEntries
Blocks == 0..(NumBlocks - 1)
StateSlots == 0..(NumStateSlots - 1)

SLOT_FREE == "free"
SLOT_PREFILLING == "prefilling"
SLOT_ACTIVE == "active"
SLOT_COMPLETE == "complete"
SLOT_QUARANTINED == "quarantined"

OccupyingStates == {SLOT_PREFILLING, SLOT_ACTIVE, SLOT_COMPLETE, SLOT_QUARANTINED}
RunningStates == {SLOT_PREFILLING, SLOT_ACTIVE}

FINISH_NONE == "none"
FINISH_STOP == "stop"
FINISH_LENGTH == "length"
FINISH_CANCELLED == "cancelled"
FINISH_FAILED == "failed"

(*
 * Bounded request/prefix shapes exercising every cache transition:
 *   - "root":  2 blocks (shared system/tool root, root=TRUE)
 *   - "leaf1": 3 blocks (extends "root" by 1 block, root=FALSE)
 *   - "leaf2": 4 blocks (extends "leaf1" by 1 block, root=FALSE -> exercises supersede)
 *   - "plain": no digests (forces pure KV eviction pressure)
 *)
DIGEST_NONE == "none"
DIGEST_ROOT == "root"
DIGEST_LEAF1 == "leaf1"
DIGEST_LEAF2 == "leaf2"

DigestBlocks(d) ==
    IF d = DIGEST_ROOT THEN 2
    ELSE IF d = DIGEST_LEAF1 THEN 3
    ELSE IF d = DIGEST_LEAF2 THEN 4
    ELSE 0

(* Request profiles: [blocks, digests (ordered shortest -> longest), rootEligible] *)
RequestProfiles == {
    [blocks |-> 3, digests |-> <<DIGEST_ROOT>>, rootEligible |-> TRUE],
    [blocks |-> 4, digests |-> <<DIGEST_ROOT, DIGEST_LEAF1>>, rootEligible |-> TRUE],
    [blocks |-> 5, digests |-> <<DIGEST_ROOT, DIGEST_LEAF1, DIGEST_LEAF2>>, rootEligible |-> FALSE],
    [blocks |-> 3, digests |-> <<>>, rootEligible |-> FALSE]
}

EmptySlot == [
    state |-> SLOT_FREE,
    ownBlocks |-> {},
    donatedBlocks |-> {},
    prefixEntry |-> NULL,
    prefixBlocks |-> {},
    donationEntry |-> NULL,
    finishReason |-> FINISH_NONE
]

(* Blocks a live slot may still write vs blocks it reads read-only.
 * During prefill the slot writes its entire ownBlocks; once prefill completes
 * (ObservePrefill), its donated prefix blocks become immutable read-only. *)
SlotWritableBlocks(sl) ==
    IF sl.state = SLOT_PREFILLING THEN sl.ownBlocks
    ELSE sl.ownBlocks \ sl.donatedBlocks

SlotReadBlocks(sl) ==
    sl.prefixBlocks \cup (IF sl.state = SLOT_PREFILLING THEN {} ELSE sl.donatedBlocks)

(* Set of all KV blocks currently held by any entry in table `es`. *)
CachedBlocks(es) ==
    UNION { es[e].blocks : e \in DOMAIN es }

(* Set of all KV blocks currently owned by live slots' own runs. *)
SlotOwnBlocks(ss) ==
    UNION {
        IF ss[s].state \in OccupyingStates THEN ss[s].ownBlocks ELSE {}
        : s \in Slots
    }

(* Contiguous run of length `len` starting at `start`. *)
RunSet(start, len) ==
    { b \in Blocks : start <= b /\ b < start + len }

(* Lowest free contiguous run of `len` blocks (continuous_slot_core.py:472-494). *)
FreeRunStarts(ss, es, len) ==
    { start \in Blocks :
        /\ start + len <= NumBlocks
        /\ RunSet(start, len) \cap (SlotOwnBlocks(ss) \cup CachedBlocks(es)) = {}
    }

LowestFreeRunStart(ss, es, len) ==
    CHOOSE start \in FreeRunStarts(ss, es, len) :
        \A other \in FreeRunStarts(ss, es, len) : start <= other

HasFreeRun(ss, es, len) ==
    FreeRunStarts(ss, es, len) /= {}

(* Lookup longest ready matching digest (prefix_cache_core.py:239-248). *)
ReadyEntryForDigest(es, d) ==
    { e \in DOMAIN es : es[e].ready /\ es[e].digest = d }

HasReadyDigest(es, d) ==
    ReadyEntryForDigest(es, d) /= {}

LookupHit(es, digests) ==
    IF Len(digests) >= 3 /\ HasReadyDigest(es, digests[3])
        THEN CHOOSE e \in ReadyEntryForDigest(es, digests[3]) : TRUE
    ELSE IF Len(digests) >= 2 /\ HasReadyDigest(es, digests[2])
        THEN CHOOSE e \in ReadyEntryForDigest(es, digests[2]) : TRUE
    ELSE IF Len(digests) >= 1 /\ HasReadyDigest(es, digests[1])
        THEN CHOOSE e \in ReadyEntryForDigest(es, digests[1]) : TRUE
    ELSE NULL

(* Eviction candidates (prefix_cache_core.py:251-262):
 * ready, unpinned (when GuardEvictUnpinnedOnly is TRUE), not in keep;
 * non-root conversation entries strictly before root. *)
EvictableAll(es, keep) ==
    { e \in DOMAIN es :
        /\ es[e].ready
        /\ (GuardEvictUnpinnedOnly => es[e].pins = 0)
        /\ e \notin keep
    }

EvictableVictims(es, keep) ==
    LET all == EvictableAll(es, keep)
        nonRoot == { e \in all : ~es[e].root }
    IN IF nonRoot /= {} THEN nonRoot ELSE all

(* Free carried-state buffer index (prefix_cache_core.py:265-267). *)
FreeStateIndices(es) ==
    StateSlots \ { es[e].stateIndex : e \in DOMAIN es }

HasFreeState(es) ==
    FreeStateIndices(es) /= {}

LowestFreeState(es) ==
    CHOOSE idx \in FreeStateIndices(es) :
        \A other \in FreeStateIndices(es) : idx <= other

(* Remove a set of keys from function `es`. *)
DropEntries(es, dropSet) ==
    [ e \in (DOMAIN es \ dropSet) |-> es[e] ]

(* Adjust pins on `eid` by `delta` (continuous_slot_core.py:636-653). *)
Repin(es, eid, delta) ==
    IF eid = NULL \/ eid \notin DOMAIN es THEN es
    ELSE [ es EXCEPT ![eid].pins = @ + delta ]

HasRoot(es) ==
    \E e \in DOMAIN es : es[e].root

HasDigest(es, d) ==
    \E e \in DOMAIN es : es[e].digest = d

Occupied(ss) ==
    \E s \in Slots : ss[s].state \in OccupyingStates

HasQuarantined(ss) ==
    \E s \in Slots : ss[s].state = SLOT_QUARANTINED

(* Non-empty contiguous write ranges of size 1 or 2 for structured reads. *)
ReadWriteSets ==
    { {b} : b \in Blocks } \cup { {b, b + 1} : b \in 0..(NumBlocks - 2) }

Init ==
    /\ slots = [ s \in Slots |-> EmptySlot ]
    /\ entries = [ e \in {} |-> [
            digest |-> DIGEST_NONE,
            blocks |-> {},
            stateIndex |-> 0,
            ready |-> FALSE,
            pins |-> 0,
            root |-> FALSE
       ] ]
    /\ readyWriteViolation = FALSE
    /\ unpinMissingViolation = FALSE
    /\ cancelOverwriteViolation = FALSE
    /\ unretiredCompleteViolation = FALSE

(*
 * Action 1: Admit & Place + Optional Donation (_place + _plan_donation + _repin)
 *   continuous_slot_core.py:505-633, 727-814
 *
 * Models a boundary admission choosing a free slot `s`, a request profile `req`,
 * an optional eviction set `placeEvict` from EvictableVictims (up to 1 victim per
 * step needed to open a contiguous free run), optional hit + repin(+1), and
 * optional donation/supersede into a fresh entry id.
 *)
AdmitRequest(s, req) ==
    /\ ~HasQuarantined(slots)
    /\ slots[s].state = SLOT_FREE
    (* plan_boundary always selects the lowest free slot: free[0] (continuous_slot_core.py:728,761). *)
    /\ \A k \in Slots : slots[k].state = SLOT_FREE => s <= k
    /\ LET hit == LookupHit(entries, req.digests)
           prefixLen == IF hit = NULL THEN 0 ELSE Cardinality(entries[hit].blocks)
           ownLen == req.blocks - prefixLen
           keepPlace == IF hit = NULL THEN {} ELSE {hit}
           (* Either place directly if space exists, or evict one valid victim
              (or two if needed to fit within 2-state-slot bound). *)
           evictChoices ==
               IF HasFreeRun(slots, entries, ownLen) THEN { {} }
               ELSE
                   LET v1 == EvictableVictims(entries, keepPlace)
                       singleSufficient == { {e} : e \in { x \in v1 : HasFreeRun(slots, DropEntries(entries, {x}), ownLen) } }
                       allPairs == UNION {
                           { {e1, e2} : e2 \in EvictableVictims(DropEntries(entries, {e1}), keepPlace) }
                           : e1 \in v1
                       }
                       pairSufficient == { p \in allPairs : HasFreeRun(slots, DropEntries(entries, p), ownLen) }
                   IN IF singleSufficient /= {} THEN singleSufficient ELSE pairSufficient
       IN
       \E placeEvict \in evictChoices :
           LET entriesAfterPlace == DropEntries(entries, placeEvict)
               start == LowestFreeRunStart(slots, entriesAfterPlace, ownLen)
               ownRun == RunSet(start, ownLen)
               (* Pin the hit entry (+1) before planning donation (line 765). *)
               entriesPinned == Repin(entriesAfterPlace, hit, 1)
               hitBlocks == IF hit = NULL THEN {} ELSE entriesPinned[hit].blocks
               (* Determine donation target digest and root flag (lines 556-589). *)
               wantRoot ==
                   /\ req.rootEligible
                   /\ ~HasRoot(entriesPinned)
                   /\ prefixLen < 2
                   /\ Len(req.digests) >= 1
               donateDigest ==
                   IF wantRoot THEN DIGEST_ROOT
                   ELSE IF Len(req.digests) > 0 /\ DigestBlocks(req.digests[Len(req.digests)]) > prefixLen
                        THEN req.digests[Len(req.digests)]
                   ELSE DIGEST_NONE
               donateTotal == DigestBlocks(donateDigest)
               donatedOwn == donateTotal - prefixLen
               canDonateBase ==
                   /\ donateDigest /= DIGEST_NONE
                   /\ ~HasDigest(entriesPinned, donateDigest)
                   /\ donatedOwn > 0
                   /\ donatedOwn <= ownLen
                   /\ donateTotal <= MaxCachedBlocks
               newEntryBlocks == hitBlocks \cup RunSet(start, donatedOwn)
               supersede ==
                   /\ hit /= NULL
                   /\ entriesPinned[hit].pins = 1
                   /\ ~entriesPinned[hit].root
               trialBase ==
                   IF supersede THEN DropEntries(entriesPinned, {hit})
                   ELSE entriesPinned
               keepDonate == IF hit = NULL THEN {} ELSE {hit}
               (* next_entry_id is strictly monotonic (continuous_slot_core.py:780),
                  so a fresh donation can never alias a live slot's pending donationEntry
                  or prefixEntry even after invalidation. *)
               liveEntryRefs ==
                   UNION { {slots[k].donationEntry, slots[k].prefixEntry} : k \in Slots }
               freeIds == EntryIds \ (DOMAIN trialBase \cup liveEntryRefs)
               (* Check whether donation fits within MaxCachedBlocks & NumStateSlots
                  directly or by evicting one valid victim from trialBase. *)
               fits(es) ==
                   /\ Cardinality(CachedBlocks(es) \cup newEntryBlocks) <= MaxCachedBlocks
                   /\ (supersede \/ HasFreeState(es))
               donateEvictChoices ==
                   IF ~canDonateBase \/ freeIds = {} THEN { [doDonate |-> FALSE, evict |-> {}] }
                   ELSE IF fits(trialBase) THEN { [doDonate |-> TRUE, evict |-> {}] }
                   ELSE
                       LET dv == EvictableVictims(trialBase, keepDonate)
                           ok1 == { [doDonate |-> TRUE, evict |-> {e}] : e \in { x \in dv : fits(DropEntries(trialBase, {x})) } }
                       IN IF ok1 /= {} THEN ok1 ELSE { [doDonate |-> FALSE, evict |-> {}] }
           IN
           \E donateChoice \in donateEvictChoices :
               LET wroteReady ==
                       \E e \in DOMAIN entriesAfterPlace :
                           entriesAfterPlace[e].ready /\ (entriesAfterPlace[e].blocks \cap ownRun /= {})
               IN
               IF ~donateChoice.doDonate THEN
                   /\ slots' = [ slots EXCEPT ![s] = [
                          state |-> SLOT_PREFILLING,
                          ownBlocks |-> ownRun,
                          donatedBlocks |-> {},
                          prefixEntry |-> hit,
                          prefixBlocks |-> hitBlocks,
                          donationEntry |-> NULL,
                          finishReason |-> FINISH_NONE
                      ] ]
                   /\ entries' = entriesPinned
                   /\ readyWriteViolation' = (readyWriteViolation \/ wroteReady)
                   /\ UNCHANGED <<unpinMissingViolation, cancelOverwriteViolation, unretiredCompleteViolation>>
               ELSE
                   LET trialAfter == DropEntries(trialBase, donateChoice.evict)
                       newId == CHOOSE id \in freeIds : \A o \in freeIds : id <= o
                       stateIdx ==
                           IF supersede THEN entriesPinned[hit].stateIndex
                           ELSE LowestFreeState(trialAfter)
                       newEntry == [
                           digest |-> donateDigest,
                           blocks |-> newEntryBlocks,
                           stateIndex |-> stateIdx,
                           ready |-> FALSE,
                           pins |-> IF supersede THEN 1 ELSE 0,
                           root |-> wantRoot
                       ]
                       entriesFinal == (newId :> newEntry) @@ trialAfter
                       readsEntry == IF supersede THEN newId ELSE hit
                   IN
                   /\ slots' = [ slots EXCEPT ![s] = [
                          state |-> SLOT_PREFILLING,
                          ownBlocks |-> ownRun,
                          donatedBlocks |-> RunSet(start, donatedOwn),
                          prefixEntry |-> readsEntry,
                          prefixBlocks |-> hitBlocks,
                          donationEntry |-> newId,
                          finishReason |-> FINISH_NONE
                      ] ]
                   /\ entries' = entriesFinal
                   /\ readyWriteViolation' = (readyWriteViolation \/ wroteReady)
                   /\ UNCHANGED <<unpinMissingViolation, cancelOverwriteViolation, unretiredCompleteViolation>>

(*
 * Action 2: Observe prefill completion (continuous_slot_core.py:868-885)
 *   Slot moves prefilling -> active (or complete on immediate stop/length).
 *   Planned donation becomes ready = TRUE.
 *)
ObservePrefill(s, stopped) ==
    /\ slots[s].state = SLOT_PREFILLING
    /\ LET dId == slots[s].donationEntry
           newState == IF stopped THEN SLOT_COMPLETE ELSE SLOT_ACTIVE
           newFinish == IF stopped THEN FINISH_STOP ELSE FINISH_NONE
       IN
       /\ slots' = [ slots EXCEPT ![s].state = newState,
                                  ![s].donationEntry = NULL,
                                  ![s].finishReason = newFinish ]
       /\ entries' =
              IF dId /= NULL /\ dId \in DOMAIN entries
              THEN [ entries EXCEPT ![dId].ready = TRUE ]
              ELSE entries
       /\ UNCHANGED <<readyWriteViolation, unpinMissingViolation, cancelOverwriteViolation, unretiredCompleteViolation>>

(*
 * Action 3: Observe decode completion (continuous_slot_core.py:888-894)
 *   Active slot finishes with stop or length.
 *)
FinishDecode(s, reason) ==
    /\ slots[s].state = SLOT_ACTIVE
    /\ reason \in {FINISH_STOP, FINISH_LENGTH}
    /\ slots' = [ slots EXCEPT ![s].state = SLOT_COMPLETE,
                               ![s].finishReason = reason ]
    /\ UNCHANGED <<entries, readyWriteViolation, unpinMissingViolation, cancelOverwriteViolation, unretiredCompleteViolation>>

(*
 * Action 4: Cancel a request in a slot (continuous_slot_core.py:427-451)
 *   With GuardCancelRunningOnly = TRUE, only prefilling/active slots transition
 *   to complete(cancelled); an already-complete slot is unchanged.
 *)
CancelSlot(s) ==
    /\ slots[s].state \in (IF GuardCancelRunningOnly THEN RunningStates ELSE {SLOT_PREFILLING, SLOT_ACTIVE, SLOT_COMPLETE})
    /\ LET prevFinish == slots[s].finishReason
           overwrote == (prevFinish /= FINISH_NONE /\ prevFinish /= FINISH_CANCELLED)
       IN
       /\ slots' = [ slots EXCEPT ![s].state = SLOT_COMPLETE,
                                  ![s].finishReason = FINISH_CANCELLED ]
       /\ cancelOverwriteViolation' = (cancelOverwriteViolation \/ overwrote)
       /\ UNCHANGED <<entries, readyWriteViolation, unpinMissingViolation, unretiredCompleteViolation>>

(*
 * Action 5: Boundary retirement of complete slots (continuous_slot_core.py:679-697)
 *   Retires every SLOT_COMPLETE slot, freeing its KV run and calling _repin(prefix_entry, -1).
 *)
BoundaryRetire ==
    /\ \E s \in Slots : slots[s].state = SLOT_COMPLETE
    /\ IF GuardRetireCompleteAtBoundary THEN
           LET retiring == { s \in Slots : slots[s].state = SLOT_COMPLETE }
               missingUnpin ==
                   \E s \in retiring :
                       slots[s].prefixEntry /= NULL /\ slots[s].prefixEntry \notin DOMAIN entries
               (* Decrement pins by the exact number of retiring slots reading each entry. *)
               entriesAfter ==
                   [ e \in DOMAIN entries |->
                       [ entries[e] EXCEPT !.pins =
                           @ - Cardinality({ s \in retiring : slots[s].prefixEntry = e }) ] ]
           IN
           /\ slots' = [ s \in Slots |-> IF s \in retiring THEN EmptySlot ELSE slots[s] ]
           /\ entries' = entriesAfter
           /\ unpinMissingViolation' = (unpinMissingViolation \/ missingUnpin)
           /\ UNCHANGED <<readyWriteViolation, cancelOverwriteViolation, unretiredCompleteViolation>>
       ELSE
           (* Mutant: skipping retirement of complete slots leaves them holding blocks after boundary. *)
           /\ unretiredCompleteViolation' = TRUE
           /\ UNCHANGED <<slots, entries, readyWriteViolation, unpinMissingViolation, cancelOverwriteViolation>>

(*
 * Action 6: Conservative invalidation (continuous_slot_core.py:980-986)
 *   invalidate_prefix_cache retains only pinned entries (pins > 0).
 *)
InvalidatePrefixCache ==
    /\ DOMAIN entries /= {}
    /\ \E e \in DOMAIN entries : entries[e].pins = 0
    /\ entries' = DropEntries(entries, { e \in DOMAIN entries : entries[e].pins = 0 })
    /\ UNCHANGED <<slots, readyWriteViolation, unpinMissingViolation, cancelOverwriteViolation, unretiredCompleteViolation>>

(*
 * Action 7: Structured read + invalidate_written_prefixes
 *   continuous_slot_service.py:433, 469-491
 *   continuous_slot_core.py:989-1008
 *
 * Nondeterministic write set `written`. When GuardReadRequiresIdle is TRUE,
 * reads run only when no slot is occupied. When GuardProtectedBlocks is TRUE,
 * `written` is disjoint from CachedBlocks(entries).
 * invalidate_written_prefixes drops EVERY intersecting entry, pinned or not.
 *)
StructuredRead(written) ==
    /\ (GuardReadRequiresIdle => ~Occupied(slots))
    /\ ~HasQuarantined(slots)
    /\ (GuardProtectedBlocks => (written \cap CachedBlocks(entries) = {}))
    /\ LET wroteReady ==
               \E e \in DOMAIN entries :
                   entries[e].ready /\ (entries[e].blocks \cap written /= {})
           intersecting ==
               { e \in DOMAIN entries : entries[e].blocks \cap written /= {} }
       IN
       /\ entries' = DropEntries(entries, intersecting)
       /\ readyWriteViolation' = (readyWriteViolation \/ wroteReady)
       /\ UNCHANGED <<slots, unpinMissingViolation, cancelOverwriteViolation, unretiredCompleteViolation>>

(*
 * Action 8: Device failure / quarantine_group (continuous_slot_core.py:940-963)
 *   Quarantines every non-free, non-quarantined slot and drops every cached entry.
 *)
DeviceFailure ==
    /\ (\E s \in Slots : slots[s].state \in {SLOT_PREFILLING, SLOT_ACTIVE, SLOT_COMPLETE})
       \/ DOMAIN entries /= {}
    /\ slots' = [ s \in Slots |->
           IF slots[s].state \in {SLOT_PREFILLING, SLOT_ACTIVE, SLOT_COMPLETE}
           THEN [ slots[s] EXCEPT !.state = SLOT_QUARANTINED, !.finishReason = FINISH_FAILED ]
           ELSE slots[s]
       ]
    /\ entries' = [ e \in {} |-> entries[e] ]
    /\ UNCHANGED <<readyWriteViolation, unpinMissingViolation, cancelOverwriteViolation, unretiredCompleteViolation>>

(*
 * Action 9: Release quarantine (continuous_slot_core.py:966-977)
 *)
ReleaseQuarantine ==
    /\ HasQuarantined(slots)
    /\ slots' = [ s \in Slots |->
           IF slots[s].state = SLOT_QUARANTINED THEN EmptySlot ELSE slots[s]
       ]
    /\ UNCHANGED <<entries, readyWriteViolation, unpinMissingViolation, cancelOverwriteViolation, unretiredCompleteViolation>>

Next ==
    \/ \E s \in Slots, req \in RequestProfiles : AdmitRequest(s, req)
    \/ \E s \in Slots, stopped \in BOOLEAN : ObservePrefill(s, stopped)
    \/ \E s \in Slots, reason \in {FINISH_STOP, FINISH_LENGTH} : FinishDecode(s, reason)
    \/ \E s \in Slots : CancelSlot(s)
    \/ BoundaryRetire
    \/ InvalidatePrefixCache
    \/ \E written \in ReadWriteSets : StructuredRead(written)
    \/ DeviceFailure
    \/ ReleaseQuarantine

Spec == Init /\ [][Next]_vars

(* ======================= INVARIANTS ======================= *)

(* Inv 1: Every block a live slot reads (via its pinned prefix or its own run)
 * is not handed to any other live slot's writable run, and no two live slots
 * share any ownBlocks. *)
LiveBlocksDisjointFromOtherRuns ==
    \A s1, s2 \in Slots :
        (s1 /= s2 /\ slots[s1].state \in OccupyingStates /\ slots[s2].state \in OccupyingStates)
        => /\ (slots[s1].ownBlocks \cap slots[s2].ownBlocks = {})
           /\ ((SlotReadBlocks(slots[s1]) \cup SlotWritableBlocks(slots[s1])) \cap SlotWritableBlocks(slots[s2]) = {})

(* Inv 2: A ready entry's blocks are never written after it becomes ready. *)
ReadyEntryBlocksImmutable ==
    ~readyWriteViolation

(* Inv 3: No two entries share a carried-state buffer index. *)
UniqueCarriedStateSlots ==
    \A e1, e2 \in DOMAIN entries :
        e1 /= e2 => entries[e1].stateIndex /= entries[e2].stateIndex

(* Inv 4: Pins never go negative, match live slot references, and unpinning a
 * missing entry never occurs on protected paths. *)
PinAccountingSound ==
    /\ \A e \in DOMAIN entries :
           /\ entries[e].pins >= 0
           /\ entries[e].pins = Cardinality({
                  s \in Slots :
                      slots[s].state \in {SLOT_PREFILLING, SLOT_ACTIVE, SLOT_COMPLETE}
                      /\ slots[s].prefixEntry = e
              })
    /\ \A s \in Slots :
           (slots[s].state \in {SLOT_PREFILLING, SLOT_ACTIVE, SLOT_COMPLETE}
            /\ slots[s].prefixEntry /= NULL)
           => slots[s].prefixEntry \in DOMAIN entries
    /\ ~unpinMissingViolation

(* Inv 5: At most one root entry is retained. *)
AtMostOneRoot ==
    Cardinality({ e \in DOMAIN entries : entries[e].root }) <= 1

(* Inv 6: Cancellation finality & one-boundary release. *)
CancelPreservesFinishReason ==
    ~cancelOverwriteViolation

CancelledReleasedAtBoundary ==
    ~unretiredCompleteViolation

====
