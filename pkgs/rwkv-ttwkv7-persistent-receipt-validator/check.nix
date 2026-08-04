{
  lib,
  runCommand,
  validator,
}:
let
  acceptedReceipt = builtins.path {
    path = ../rwkv-ttwkv7-persistent-passed-evidence/fixtures/ttwkv7-persistent-device-4/artifact/receipt.json;
    name = "rwkv-ttwkv7-persistent-device-4-host-receipt.json";
  };
  validatorSource = builtins.path {
    path = ./src/lib.rs;
    name = "rwkv-ttwkv7-persistent-receipt-validator-core.rs";
  };
  validatorShellSource = builtins.path {
    path = ./src/main.rs;
    name = "rwkv-ttwkv7-persistent-receipt-validator-shell.rs";
  };
  expectedSchemaVersion = 1;
  expectedCallCount = 24;
  expectedContinuityCount = 12;
  wrongCallCount = expectedCallCount - 1;
  wrongContinuityCount = expectedContinuityCount - 1;
  wrongSchemaVersion = expectedSchemaVersion + 1;
  successStatus = 0;
  failureStatus = 1;
  missingReceiptPath = "missing-receipt.json";
in
runCommand "rwkv-ttwkv7-persistent-receipt-validator-check" { } ''
  set -euo pipefail

  validator=${lib.escapeShellArg "${validator}/bin/rwkv-ttwkv7-persistent-receipt-validator"}
  accepted_receipt=${lib.escapeShellArg acceptedReceipt}

  expect_failure() {
    local name="$1"
    local expected_diagnostic="$2"
    shift 2
    if "$@" >"$name.stdout" 2>"$name.stderr"; then
      echo "receipt validator negative fixture passed: $name" >&2
      exit ${toString failureStatus}
    else
      local actual_status="$?"
    fi
    test "$actual_status" -eq ${toString failureStatus}
    grep -F "$expected_diagnostic" "$name.stderr"
  }

  validate_receipt() {
    local receipt="$1"
    "$validator" \
      --receipt "$receipt" \
      --expected-call-count ${toString expectedCallCount} \
      --expected-continuity-count ${toString expectedContinuityCount}
  }

  validate_receipt "$accepted_receipt" >positive-first.log
  validate_receipt "$accepted_receipt" >positive-second.log
  cmp positive-first.log positive-second.log
  grep -F ${lib.escapeShellArg "rwkv persistent receipt validation: PASS calls=${toString expectedCallCount} continuity=${toString expectedContinuityCount}"} \
    positive-first.log

  cp "$accepted_receipt" obsolete-flat-field.json
  substituteInPlace obsolete-flat-field.json \
    --replace-fail '"schema_version": ${toString expectedSchemaVersion},' \
      '"schema_version": ${toString expectedSchemaVersion}, "session_call_count": ${toString expectedCallCount},'
  expect_failure obsolete-flat-field \
    'host receipt contains obsolete session_call_count' \
    validate_receipt obsolete-flat-field.json

  cp "$accepted_receipt" obsolete-null-field.json
  substituteInPlace obsolete-null-field.json \
    --replace-fail '"schema_version": ${toString expectedSchemaVersion},' \
      '"schema_version": ${toString expectedSchemaVersion}, "session_call_count": null,'
  expect_failure obsolete-null-field \
    'host receipt contains obsolete session_call_count' \
    validate_receipt obsolete-null-field.json

  cp "$accepted_receipt" wrong-schema.json
  substituteInPlace wrong-schema.json \
    --replace-fail '"schema_version": ${toString expectedSchemaVersion}' \
      '"schema_version": ${toString wrongSchemaVersion}'
  expect_failure wrong-schema \
    ${lib.escapeShellArg "host schema_version mismatch: expected ${toString expectedSchemaVersion}, found ${toString wrongSchemaVersion}"} \
    validate_receipt wrong-schema.json

  cp "$accepted_receipt" wrong-call-count.json
  substituteInPlace wrong-call-count.json \
    --replace-fail '"physical_wkv_call_count": ${toString expectedCallCount}' \
      '"physical_wkv_call_count": ${toString wrongCallCount}'
  expect_failure wrong-call-count \
    ${lib.escapeShellArg "host physical_wkv_call_count mismatch: expected ${toString expectedCallCount}, found ${toString wrongCallCount}"} \
    validate_receipt wrong-call-count.json

  cp "$accepted_receipt" wrong-continuity-count.json
  substituteInPlace wrong-continuity-count.json \
    --replace-fail '"same_layer_state_continuity_count": ${toString expectedContinuityCount}' \
      '"same_layer_state_continuity_count": ${toString wrongContinuityCount}'
  expect_failure wrong-continuity-count \
    ${lib.escapeShellArg "host same_layer_state_continuity_count mismatch: expected ${toString expectedContinuityCount}, found ${toString wrongContinuityCount}"} \
    validate_receipt wrong-continuity-count.json

  cp "$accepted_receipt" nonzero-child-status.json
  substituteInPlace nonzero-child-status.json \
    --replace-fail '"child_exit_status": ${toString successStatus}' \
      '"child_exit_status": ${toString failureStatus}'
  expect_failure nonzero-child-status \
    ${lib.escapeShellArg "host child_exit_status mismatch: expected ${toString successStatus}, found ${toString failureStatus}"} \
    validate_receipt nonzero-child-status.json

  printf '{\n' >malformed.json
  expect_failure malformed \
    'persistent receipt JSON is invalid' \
    validate_receipt malformed.json

  expect_failure missing-arguments \
    'usage: rwkv-ttwkv7-persistent-receipt-validator' \
    "$validator"
  expect_failure missing-receipt \
    'receipt metadata could not be read' \
    validate_receipt ${lib.escapeShellArg missingReceiptPath}
  expect_failure directory-receipt \
    'receipt path is not a regular file' \
    validate_receipt .
  expect_failure invalid-call-count \
    'expected call count is not an exact nonnegative decimal' \
    "$validator" \
      --receipt "$accepted_receipt" \
      --expected-call-count invalid \
      --expected-continuity-count ${toString expectedContinuityCount}
  expect_failure zero-call-count \
    'expected call count must be positive' \
    "$validator" \
      --receipt "$accepted_receipt" \
      --expected-call-count 0 \
      --expected-continuity-count 0
  expect_failure reordered-arguments \
    'usage: rwkv-ttwkv7-persistent-receipt-validator' \
    "$validator" \
      --expected-call-count ${toString expectedCallCount} \
      --receipt "$accepted_receipt" \
      --expected-continuity-count ${toString expectedContinuityCount}
  expect_failure extra-arguments \
    'usage: rwkv-ttwkv7-persistent-receipt-validator' \
    "$validator" \
      --receipt "$accepted_receipt" \
      --expected-call-count ${toString expectedCallCount} \
      --expected-continuity-count ${toString expectedContinuityCount} \
      unexpected

  grep -F 'serde_json::from_slice(receipt_bytes)' ${lib.escapeShellArg validatorSource}
  grep -F 'validate_persistent_receipt(&receipt_bytes, arguments.expectations)' \
    ${lib.escapeShellArg validatorShellSource}
  if grep -E 'std::process::Command|Command::new|MeshDevice|EnqueueMeshWorkload|/dev/tenstorrent|TT_VISIBLE_DEVICES|tt-smi' \
    ${lib.escapeShellArg validatorSource} ${lib.escapeShellArg validatorShellSource}; then
    echo 'persistent receipt validator contains a process or device execution surface' >&2
    exit ${toString failureStatus}
  fi

  mkdir -p "$out"
  cp positive-first.log "$out/validation.log"
  printf '%s\n' \
    '{' \
    '  "device_initialized": false,' \
    '  "expected_call_count": ${toString expectedCallCount},' \
    '  "expected_continuity_count": ${toString expectedContinuityCount},' \
    '  "hardware_process_started": false,' \
    '  "target": "rwkv_ttwkv7_persistent_receipt_validator"' \
    '}' \
    >"$out/receipt.json"
''
