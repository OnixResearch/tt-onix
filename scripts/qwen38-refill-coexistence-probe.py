"""Bounded P150x2 diagnostic: prefill a new user while a batched decode trace stays resident.

Evidence for the continuous-slot scheduler; diagnostic only, with no performance or
admission claim. It needs both P150 devices, so stop qwen38-p150x2.service first, and run
it with that unit's environment and the qwen38 package's PYTHONPATH and TT_METAL_HOME
(see .cairn/changes/qwen38-continuous-slot-scheduler/evidence/refill-coexistence.md):

    python qwen38-refill-coexistence-probe.py DECODE_PROFILE_JSON RESULT_JSON FILLER_TEXT

Reference: users A-D prefilled together, then N + M greedy steps at width 4.
Refill runs: A-C prefilled into slots 0-2 and a width-4 decode trace captured (slot 3 idle),
N steps, then D prefilled into slot 3 with the decode trace still resident, every decode
input restaged, and M more steps. R2 refills with the masked prefill traces, R1 eagerly.
Pass: A-C match the reference exactly, D matches its reference, the trace binding holds,
and the masked traces stay captured.
"""

import json
import os
import sys
import time

import torch
import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.common import copy_host_to_device
from qwen36_service import server as srv
from qwen36_service.batch_decode_adapter import BatchedTracedDecodeAdapter
from qwen36_service.decode_core import DecodeProfile
from transformers import AutoTokenizer

MODEL = os.environ["HF_MODEL"]
PROFILE, RESULT, FILLER = sys.argv[1:4]
N_BEFORE = 24
M_AFTER = 24
WIDTH = 4


def log(stage, **fields):
    print(json.dumps({"stage": stage, **fields}), flush=True)


def chat(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main():
    os.environ["TT_CACHE_PATH"] = MODEL
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(srv.MESH_ROWS, srv.MESH_COLUMNS),
        l1_small_size=srv.L1_SMALL_SIZE_BYTES,
        num_command_queues=srv.COMMAND_QUEUE_COUNT,
        trace_region_size=srv.TRACE_REGION_SIZE_BYTES,
    )
    results = {}
    try:
        mesh.enable_program_cache()
        model = Qwen36Model.from_pretrained(mesh, max_batch_size=srv.MAXIMUM_BATCH_WIDTH, max_seq_len=65536)
        with open(PROFILE, encoding="utf-8") as profile_file:
            profile = DecodeProfile.from_mapping(json.load(profile_file))
        adapter = BatchedTracedDecodeAdapter(model=model, profile=profile, maximum_width=srv.MAXIMUM_BATCH_WIDTH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        log("model-loaded")

        # Same warm-up as the service: widths 1 and 4, then every decode width and the masked traces.
        warm = tokenizer("Warm the test path.", return_tensors="pt")["input_ids"].to(torch.int32)
        for width in (1, WIDTH):
            first = adapter.prefill_users(tuple(warm for _ in range(width)), generation_tokens=2)
            positions = (int(warm.shape[1]),) * width
            adapter.prepare_width(token_ids=first.first_token_ids, positions=positions)
            adapter.decode_traced(token_ids=first.first_token_ids, positions=positions)
            adapter.reset_all()
        buckets = adapter.prepare_masked_prefill_traces(int(warm[0, 0]), (adapter.fusion_mode,))
        log("warmed", buckets=list(buckets), fusion_mode=adapter.fusion_mode)

        filler = tokenizer(open(FILLER, encoding="utf-8").read(), add_special_tokens=False)["input_ids"]
        texts = {
            "A": chat("Name the three primary colors and one use for each."),
            "B": chat(tokenizer.decode(filler[:1500]) + "\n\nSummarize the passage above in two sentences."),
            "C": chat(tokenizer.decode(filler[2000:5000]) + "\n\nWhat is this passage about? One sentence."),
            "D": chat(tokenizer.decode(filler[6000:8500]) + "\n\nList two technical terms from the passage."),
        }
        ids = {key: tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(torch.int32) for key, text in texts.items()}
        lengths = {key: int(value.shape[1]) for key, value in ids.items()}
        generation = N_BEFORE + M_AFTER + 2
        log("prompts", lengths=lengths)

        def step(tokens, positions):
            return list(adapter.decode_traced(token_ids=tuple(tokens), positions=tuple(positions)))

        # Reference: all four users from the start.
        first = adapter.prefill_users(tuple(ids[key] for key in "ABCD"), generation_tokens=generation)
        tokens = list(first.first_token_ids)
        positions = [lengths[key] for key in "ABCD"]
        reference = {key: [tokens[index]] for index, key in enumerate("ABCD")}
        adapter.prepare_width(token_ids=tuple(tokens), positions=tuple(positions))
        for _ in range(N_BEFORE + M_AFTER):
            winners = step(tokens, positions)
            for index, key in enumerate("ABCD"):
                reference[key].append(winners[index])
                tokens[index] = winners[index]
                positions[index] += 1
        adapter.reset_all()
        adapter._release_traces()
        results["reference"] = {key: tokenizer.decode(value) for key, value in reference.items()}
        log("reference-done")

        for label, masked in (("R2-masked-refill", True), ("R1-eager-refill", False)):
            cache_before = mesh.num_program_cache_entries()
            blocks = [adapter.request_blocks(prompt_tokens=lengths[key], generation_tokens=generation) for key in "ABCD"]
            adapter.page_table = torch.tensor(adapter.kv_pool.assign_group(blocks, width=adapter.maximum_width), dtype=torch.int32)
            adapter.reset_all()
            logits = model.prefill_paged_slots(
                [ids[key] for key in "ABC"],
                adapter.page_table[:3].contiguous(),
                [0, 1, 2],
                valid_lens=[lengths[key] for key in "ABC"],
                masked_traces=True,
            )
            tokens = [int(torch.argmax(row.float().reshape(-1)[: model.args.vocab_size])) for row in logits] + [0]
            positions = [lengths[key] for key in "ABC"] + [0]
            run = {key: [tokens[index]] for index, key in enumerate("ABC")}
            adapter._pending_trace_plan = None
            adapter.prepare_width(token_ids=tuple(tokens), positions=tuple(positions))
            trace = adapter._traces[WIDTH]
            for _ in range(N_BEFORE):
                winners = step(tokens, positions)
                for index, key in enumerate("ABC"):
                    run[key].append(winners[index])
                    tokens[index] = winners[index]
                    positions[index] += 1

            refill_started = time.monotonic()
            logits_d = model.prefill_paged_slots(
                [ids["D"]],
                adapter.page_table[3:4].contiguous(),
                [3],
                valid_lens=[lengths["D"]],
                masked_traces=masked,
            )
            refill_seconds = time.monotonic() - refill_started
            tokens[3] = int(torch.argmax(logits_d[0].float().reshape(-1)[: model.args.vocab_size]))
            positions[3] = lengths["D"]
            run["D"] = [tokens[3]]
            host = model.prepare_decode_inputs_host(
                torch.tensor(tokens, dtype=torch.int32).reshape(WIDTH, 1),
                torch.tensor(positions, dtype=torch.int32),
                page_table=adapter.page_table[:WIDTH].contiguous(),
            )
            copy_host_to_device(host, device_tensors=trace.device_inputs)
            binding_held = adapter._binding_matches(trace)
            traces_intact = bool(model.masked_prefill_trace_buckets()) and not model.masked_prefill_traces_stale()

            for _ in range(M_AFTER):
                winners = step(tokens, positions)
                for index, key in enumerate("ABCD"):
                    run[key].append(winners[index])
                    tokens[index] = winners[index]
                    positions[index] += 1

            matches = {key: run[key] == reference[key][: len(run[key])] for key in "ABCD"}
            first_divergence = {
                key: next((i for i, (a, b) in enumerate(zip(run[key], reference[key])) if a != b), None) for key in "ABCD"
            }
            results[label] = {
                "matches_reference": matches,
                "first_divergence": first_divergence,
                "binding_held": binding_held,
                "masked_traces_intact": traces_intact,
                "program_cache_growth": mesh.num_program_cache_entries() - cache_before,
                "refill_prefill_seconds": round(refill_seconds, 3),
                "text": {key: tokenizer.decode(value) for key, value in run.items()},
            }
            log(label, **{k: v for k, v in results[label].items() if k != "text"})
            adapter.reset_all()
            adapter._release_traces()
    finally:
        with open(RESULT, "w", encoding="utf-8") as result_file:
            json.dump(results, result_file, indent=2)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
