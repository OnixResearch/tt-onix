{
  lib,
  runCommand,
  writeText,
  b3sum,
  rwkvLayerHarness,
  ttwkv7,
}:
let
  validator = "${ttwkv7}/libexec/ttwkv7/wkv7-decode-composition-validator";
  contract = writeText "ttwkv7-decode-composition-contract.json" (
    builtins.readFile ./decode-composition-contract.json
  );
  abiHeader = "${ttwkv7}/share/ttwkv7/source/ttwkv7-decode-abi.h";
  hostSource = "${ttwkv7}/share/ttwkv7/source/wkv7_runner.cpp";
  validatorSource = "${ttwkv7}/share/ttwkv7/source/decode-composition-validator.cpp";
  readerSource = "${ttwkv7}/share/ttwkv7/kernels/wkv7_decodeL_reader.cpp";
  computeSource = "${ttwkv7}/share/ttwkv7/kernels/wkv7_decodeL_compute.cpp";
  writerSource = "${ttwkv7}/share/ttwkv7/kernels/wkv7_writer.cpp";
  fixture = "${rwkvLayerHarness}/share/rwkv-layer-harness/ttwkv7-boundary.json";
  expectedFixtureByteCount = 420072;
  expectedFixtureBlake3 = "731f44866c869300ca330f703f1adad4c3ae7ee62b832fa881a6bf4ea90211cd";
  validationFailureStatus = 1;
  invalidArgumentStatus = 2;
in
# r[verify onix.tenstorrent.native_runtime.rwkv_lab.ttwkv7_decode_composition]
runCommand "rwkv-ttwkv7-decode-composition-check"
  {
    nativeBuildInputs = [ b3sum ];
  }
  ''
    set -euo pipefail
    mkdir -p "$out"

    for required in \
      ${lib.escapeShellArg validator} \
      ${lib.escapeShellArg contract} \
      ${lib.escapeShellArg abiHeader} \
      ${lib.escapeShellArg hostSource} \
      ${lib.escapeShellArg validatorSource} \
      ${lib.escapeShellArg readerSource} \
      ${lib.escapeShellArg computeSource} \
      ${lib.escapeShellArg writerSource} \
      ${lib.escapeShellArg fixture}; do
      test -e "$required"
    done
    test -x ${lib.escapeShellArg validator}
    test "$(wc -c < ${lib.escapeShellArg fixture})" -eq ${toString expectedFixtureByteCount}
    test "$(b3sum ${lib.escapeShellArg fixture} | cut -d' ' -f1)" = \
      ${lib.escapeShellArg expectedFixtureBlake3}

    # The validator is device-free: no device construction or I/O surface.
    for forbidden in \
      'MeshDevice' 'CreateKernel(' 'EnqueueMeshWorkload' \
      'EnqueueWriteMeshBuffer' 'EnqueueReadMeshBuffer' '/dev/tenstorrent' \
      'MeshCommandQueue' 'tt::tt_metal' 'tt-metalium'; do
      if grep -F "$forbidden" ${lib.escapeShellArg validatorSource}; then
        echo "decode-composition validator contains a device execution surface: $forbidden" >&2
        exit 1
      fi
    done

    first_receipt="$out/receipt.json"
    second_receipt="$out/receipt-second.json"
    ${lib.escapeShellArg validator} \
      ${lib.escapeShellArg contract} \
      ${lib.escapeShellArg abiHeader} \
      ${lib.escapeShellArg hostSource} \
      ${lib.escapeShellArg readerSource} \
      ${lib.escapeShellArg computeSource} \
      ${lib.escapeShellArg writerSource} \
      ${lib.escapeShellArg fixture} >"$first_receipt"
    ${lib.escapeShellArg validator} \
      ${lib.escapeShellArg contract} \
      ${lib.escapeShellArg abiHeader} \
      ${lib.escapeShellArg hostSource} \
      ${lib.escapeShellArg readerSource} \
      ${lib.escapeShellArg computeSource} \
      ${lib.escapeShellArg writerSource} \
      ${lib.escapeShellArg fixture} >"$second_receipt"
    cmp "$first_receipt" "$second_receipt"
    rm "$second_receipt"

    grep -Fq '"device_initialized": false' "$first_receipt"
    grep -Fq '"hardware_execution_claimed": false' "$first_receipt"
    grep -Fq '"staging_index": 21' "$first_receipt"
    grep -Fq '"scratch_index": 22' "$first_receipt"
    grep -Fq '"output_index": 16' "$first_receipt"
    grep -Fq '"reader_staging_tiles": 16' "$first_receipt"
    grep -Fq '"compute_consumed_tiles": 16' "$first_receipt"
    grep -Fq '"compute_output_tiles": 6' "$first_receipt"
    grep -Fq '"writer_consumed_tiles": 6' "$first_receipt"
    grep -Fq '"state_tiles_per_instance": 4' "$first_receipt"
    grep -Fq '"fingerprint": "d3111d60144f2aab"' "$first_receipt"
    grep -Fq '"fingerprint": "4d03e6a109e5505b"' "$first_receipt"
    grep -Fq '"fingerprint": "ce4c56d63ca03382"' "$first_receipt"
    grep -Fq '"fingerprint": "9fa0e9b8368f6885"' "$first_receipt"
    grep -Fq '"fingerprint": "2898af9e2c578078"' "$first_receipt"

    mutation_root="$(mktemp -d)"

    mutable_copy() {
      local source="$1"
      local name="$2"
      local destination="$mutation_root/$name"
      cp "$source" "$destination"
      chmod u+w "$destination"
      printf '%s\n' "$destination"
    }

    expect_rejected() {
      local name="$1"
      local expected_diagnostic="$2"
      local abi="$3"
      local host="$4"
      local reader="$5"
      local compute="$6"
      local writer="$7"
      local log="$mutation_root/$name.log"
      if ${lib.escapeShellArg validator} \
        ${lib.escapeShellArg contract} "$abi" "$host" "$reader" "$compute" \
        "$writer" ${lib.escapeShellArg fixture} >"$log" 2>&1; then
        echo "decode-composition validator accepted negative fixture: $name" >&2
        exit 1
      else
        local status="$?"
      fi
      test "$status" -eq ${toString validationFailureStatus}
      grep -Fq "$expected_diagnostic" "$log"
    }

    changed_cb_index="$(mutable_copy ${lib.escapeShellArg readerSource} changed-cb-index.cpp)"
    substituteInPlace "$changed_cb_index" \
      --replace-fail 'c_natstage = 21' 'c_natstage = 20'
    expect_rejected changed-cb-index \
      'staging CB index drift' \
      ${lib.escapeShellArg abiHeader} \
      ${lib.escapeShellArg hostSource} \
      "$changed_cb_index" \
      ${lib.escapeShellArg computeSource} \
      ${lib.escapeShellArg writerSource}

    changed_page_format="$(mutable_copy ${lib.escapeShellArg hostSource} changed-page-format.cpp)"
    substituteInPlace "$changed_page_format" \
      --replace-fail 'tt::DataFormat::Float16_b' 'tt::DataFormat::Float32'
    expect_rejected changed-page-format \
      'tile page size/format drift' \
      ${lib.escapeShellArg abiHeader} \
      "$changed_page_format" \
      ${lib.escapeShellArg readerSource} \
      ${lib.escapeShellArg computeSource} \
      ${lib.escapeShellArg writerSource}

    under_capacity="$(mutable_copy ${lib.escapeShellArg hostSource} under-capacity.cpp)"
    substituteInPlace "$under_capacity" \
      --replace-fail 'MakeCB(prog, core, CB::c_21, 32u);' \
      'MakeCB(prog, core, CB::c_21, 1u);'
    expect_rejected under-capacity \
      'staging buffer capacity below the contract minimum' \
      ${lib.escapeShellArg abiHeader} \
      "$under_capacity" \
      ${lib.escapeShellArg readerSource} \
      ${lib.escapeShellArg computeSource} \
      ${lib.escapeShellArg writerSource}

    changed_argument_meaning="$(mutable_copy ${lib.escapeShellArg abiHeader} changed-argument-meaning.h)"
    substituteInPlace "$changed_argument_meaning" \
      --replace-fail \
      'arguments.reader[index(ReaderArgument::HeadCount)] = config.head_count;' \
      'arguments.reader[index(ReaderArgument::HeadCount)] = config.token_count;'
    expect_rejected changed-argument-meaning \
      'ABI header assignment meaning drift for reader.HeadCount at index 0' \
      "$changed_argument_meaning" \
      ${lib.escapeShellArg hostSource} \
      ${lib.escapeShellArg readerSource} \
      ${lib.escapeShellArg computeSource} \
      ${lib.escapeShellArg writerSource}

    changed_argument_enum="$(mutable_copy ${lib.escapeShellArg abiHeader} changed-argument-enum.h)"
    substituteInPlace "$changed_argument_enum" \
      --replace-fail '  Input0,' '  Input9,'
    expect_rejected changed-argument-enum \
      'reader argument meaning drift at index 5' \
      "$changed_argument_enum" \
      ${lib.escapeShellArg hostSource} \
      ${lib.escapeShellArg readerSource} \
      ${lib.escapeShellArg computeSource} \
      ${lib.escapeShellArg writerSource}

    tile_count_mismatch="$(mutable_copy ${lib.escapeShellArg computeSource} tile-count-mismatch.cpp)"
    substituteInPlace "$tile_count_mismatch" \
      --replace-fail 'cb_push_back(cb_out, St + NS);' \
      'cb_push_back(cb_out, St);'
    expect_rejected tile-count-mismatch \
      'compute output tile-count mismatch' \
      ${lib.escapeShellArg abiHeader} \
      ${lib.escapeShellArg hostSource} \
      ${lib.escapeShellArg readerSource} \
      "$tile_count_mismatch" \
      ${lib.escapeShellArg writerSource}

    stale_version="$(mutable_copy ${lib.escapeShellArg writerSource} stale-version.cpp)"
    printf '\n// composition revision drift\n' >>"$stale_version"
    expect_rejected stale-version \
      "producer/consumer version is stale for consumer stage 'decode_writer'" \
      ${lib.escapeShellArg abiHeader} \
      ${lib.escapeShellArg hostSource} \
      ${lib.escapeShellArg readerSource} \
      ${lib.escapeShellArg computeSource} \
      "$stale_version"

    # Additional source-bound mutations exercise the same validator paths.
    changed_reader_argument="$(mutable_copy ${lib.escapeShellArg readerSource} changed-reader-argument.cpp)"
    substituteInPlace "$changed_reader_argument" \
      --replace-fail 'uint32_t H  = get_arg_val<uint32_t>(0);' \
      'uint32_t H  = get_arg_val<uint32_t>(1);'
    expect_rejected changed-reader-argument \
      'no longer binds runtime argument 0' \
      ${lib.escapeShellArg abiHeader} \
      ${lib.escapeShellArg hostSource} \
      "$changed_reader_argument" \
      ${lib.escapeShellArg computeSource} \
      ${lib.escapeShellArg writerSource}

    missing_log="$mutation_root/missing.log"
    if ${lib.escapeShellArg validator} >"$missing_log" 2>&1; then
      echo "decode-composition validator accepted a missing argument list" >&2
      exit 1
    else
      missing_status="$?"
    fi
    test "$missing_status" -eq ${toString invalidArgumentStatus}
    grep -Fq 'usage:' "$missing_log"

    suffix_log="$mutation_root/suffix.log"
    if ${lib.escapeShellArg validator} \
      ${lib.escapeShellArg contract} \
      ${lib.escapeShellArg abiHeader} \
      ${lib.escapeShellArg hostSource} \
      ${lib.escapeShellArg readerSource} \
      ${lib.escapeShellArg computeSource} \
      ${lib.escapeShellArg writerSource} \
      ${lib.escapeShellArg fixture} unexpected-suffix >"$suffix_log" 2>&1; then
      echo "decode-composition validator accepted an argument suffix" >&2
      exit 1
    else
      suffix_status="$?"
    fi
    test "$suffix_status" -eq ${toString invalidArgumentStatus}
    grep -Fq 'usage:' "$suffix_log"

    missing_file_log="$mutation_root/missing-file.log"
    if ${lib.escapeShellArg validator} \
      ${lib.escapeShellArg contract} \
      ${lib.escapeShellArg abiHeader} \
      ${lib.escapeShellArg hostSource} \
      ${lib.escapeShellArg readerSource} \
      ${lib.escapeShellArg computeSource} \
      ${lib.escapeShellArg writerSource} \
      "$mutation_root/absent.json" >"$missing_file_log" 2>&1; then
      echo "decode-composition validator accepted a missing fixture" >&2
      exit 1
    else
      missing_file_status="$?"
    fi
    test "$missing_file_status" -eq ${toString invalidArgumentStatus}
    grep -Fq 'boundary fixture must be a readable regular file' "$missing_file_log"

    test -s "$first_receipt"
  ''
