{
  runCommand,
  tt-npe,
}:
# Device-free end-to-end check: simulate the bundled example workload against
# the blackhole model. Nothing here opens a Tenstorrent device node.
runCommand "tt-npe-example-blackhole"
  {
    nativeBuildInputs = [ tt-npe ];
  }
  ''
    export HOME="$TMPDIR"
    log="$(${tt-npe}/bin/tt-npe -d blackhole \
      -w ${tt-npe}/share/tt-npe/examples/example_wl.json 2>&1)"
    printf '%s\n' "$log"
    case "$log" in
      *"tt-npe simulation finished successfully"*) ;;
      *)
        echo "tt-npe did not report a successful blackhole simulation" >&2
        exit 1
        ;;
    esac
    test -f ${tt-npe}/share/tt_npe/device/models/blackhole.yaml
    printf '%s\n' "$log" > "$out"
  ''
