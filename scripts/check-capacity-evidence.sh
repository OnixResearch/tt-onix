#!/usr/bin/env bash
# Device-free evidence-crate test loop for the capacity change.
# Usage: scripts/check-capacity-evidence.sh [cargo args...]
set -euo pipefail
cd "$(dirname "$0")/.."

export MODEL_TARGET_PROFILE_BLAKE3=d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88
export MODEL_TARGET_NAME=rwkv7-goose-world2.8-0.1b-p150x2
export MODEL_TARGET_MODEL_ID=RWKV/RWKV7-Goose-World2.8-0.1B-HF
export MODEL_TARGET_REVISION=d81965cb4e1a9f96696b4f70b84212b8f2e43216
export MODEL_TARGET_ARCHITECTURE_FAMILY=rwkv7
export MODEL_TARGET_STATE_KIND=recurrent_matrix
export MODEL_TARGET_PRECISION=bf16_checkpoint_fp32_reference
export MODEL_TARGET_ARCHITECTURE_ADAPTER=rwkv7-metalium-p150x2-v1
export MODEL_TARGET_EVIDENCE_ADAPTER=rwkv7-p150x2-evidence-v1
export MODEL_TARGET_ARTIFACT_BLAKE3_JSON='{"checkpoint":"905f82048a64b881f9267117a398feb8a8a92bcc5233666bf67904e0d899d0e5","generation_config":"2288838a56ea704a85691828bbc7f0ab2934c949b13a4d8f3af1b13d955ac2fa","model_config":"113edfd55813d327ae7e37987ee9c5ed123c69fa809670b3a0bcc07fd1e9295d","tokenizer_added_tokens":"02893c22a1e92502fdd31ba4d57b6e574692023505be7ba82d68d1e3142ff02f","tokenizer_config":"b2411eb362aefa260493811c9414e8da589a19d6cec44e8456953507e293755e","tokenizer_implementation":"0a2a88e97b455858e03bbbc83bb0228d8f36c2731fcb91cd94f05e2930e2aa24","tokenizer_special_tokens":"751ae3ea4b59073218a85facbee1536739b0aa26d5d5670e11ef6815e5bac870","tokenizer_vocabulary":"3997a74891dd68ced8daadae0d7475274b08988c9263ca042896c8106967aef2"}'
export MODEL_TARGET_TOPOLOGY_ID=p150x2-exact-physical-0-1
export MODEL_TARGET_MESH_SHAPE_JSON='[1,2]'
export MODEL_TARGET_PHYSICAL_DEVICE_IDS_JSON='[0,1]'
export MODEL_TARGET_REPLICATION=full-lockstep
export MODEL_TARGET_REQUIRED_OPERATIONS_JSON='["embedding","layer-norm","layer-norm-unaffined","matmul","add","multiply","subtract","divide","sigmoid","tanh","exp","relu","square","sqrt","sum-last","maximum-scalar","transpose","argmax"]'
export MODEL_TARGET_CHECKPOINT_BLAKE3=905f82048a64b881f9267117a398feb8a8a92bcc5233666bf67904e0d899d0e5
export MODEL_TARGET_TOKENIZER_VOCABULARY_BLAKE3=3997a74891dd68ced8daadae0d7475274b08988c9263ca042896c8106967aef2

cd pkgs/rwkv7-p150x2-evidence
if [[ "${1:-}" == "--clippy" ]]; then
  shift || true
  nix develop ../../. -c cargo clippy --all-targets -- "$@"
else
  nix develop ../../. -c cargo test "$@"
fi