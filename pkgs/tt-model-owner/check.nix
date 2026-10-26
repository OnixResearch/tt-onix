# SPDX-License-Identifier: LGPL-2.1-only
# r[verify tenstorrent.runtime_profile.owner_helper]
# Behaviour check for the owner helper. It runs the built helper against fake
# systemctl, lsof, tt-smi, and record tools, so nothing touches a real unit or
# device, and it proves the helper is the only program that names the devices.
{
  lib,
  pkgs,
  owner,
  ttModel,
}:
pkgs.runCommand "tt-model-owner-check"
  {
    nativeBuildInputs = [ pkgs.coreutils ];
  }
  ''
    set -euo pipefail
    export HOME="$TMPDIR/home"
    mkdir -p "$HOME" "$TMPDIR/state" "$TMPDIR/gc"
    owner="${owner}/bin/tt-model-owner"
    export FAKE_LOG="$TMPDIR/fake.log"
    : >"$FAKE_LOG"

    fail() { echo "install check: $*" >&2; exit 1; }

    expect_refusal() {
      local description="$1" pattern="$2"
      shift 2
      if "$@" >expect.out 2>expect.err; then
        fail "$description: the command was accepted"
      fi
      if ! grep -F -- "$pattern" expect.err; then
        cat expect.err >&2
        fail "$description: expected message not found: $pattern"
      fi
    }

    "$owner" --help >/dev/null 2>&1 || fail "--help must exit zero"

    # The helper is the only program that names the devices, the reset tool, and
    # the unit it owns. The tt-model install check asserts the opposite for its own
    # program text; these two assertions are the other direction of that invariant.
    grep -q -- '/dev/tenstorrent' "$owner" || fail "the helper must own the card device paths"
    grep -q -- 'tt-smi' "$owner" || fail "the helper must own the reset command"
    if grep -q -- 'ttwkv7-owner-control' "$owner"; then
      fail "the helper must not use the legacy owner-control helper"
    fi

    expect_refusal "state-root override without the test flag" \
      "TT_MODEL_STATE_ROOT is overridden" \
      env TT_MODEL_STATE_ROOT="$TMPDIR/state" "$owner" cards
    expect_refusal "gc-root override without the test flag" \
      "TT_MODEL_GC_ROOTS is overridden" \
      env TT_MODEL_GC_ROOTS="$TMPDIR/gc" "$owner" cards

    base_env=(
      TT_MODEL_OWNER_TEST=1
      TT_MODEL_STATE_ROOT="$TMPDIR/state"
      TT_MODEL_GC_ROOTS="$TMPDIR/gc"
    )

    expect_refusal "no installed helper" "no owner helper is installed" \
      env "''${base_env[@]}" "$owner" cards

    ln -s ${ttModel} "$TMPDIR/state/owner-helper"
    expect_refusal "helper store path mismatch" "but the host records" \
      env "''${base_env[@]}" "$owner" cards
    ln -sfn ${owner} "$TMPDIR/state/owner-helper"

    cat >"$TMPDIR/systemctl" <<'SH'
    #!/bin/sh
    printf 'systemctl %s\n' "$*" >>"$FAKE_LOG"
    case "$1" in
      show) printf '0\n' ;;
      is-active) exit 1 ;;
      is-enabled) printf 'disabled\n' ;;
      *) exit 0 ;;
    esac
    SH
    cat >"$TMPDIR/lsof" <<'SH'
    #!/bin/sh
    printf 'lsof %s\n' "$*" >>"$FAKE_LOG"
    printf 'tt_job 4242 brittonr 3u CHR 1,2 0t0 7 %s\n' "$1"
    SH
    cat >"$TMPDIR/tt-smi" <<'SH'
    #!/bin/sh
    printf 'tt-smi %s\n' "$*" >>"$FAKE_LOG"
    SH
    cat >"$TMPDIR/tt-model" <<'SH'
    #!/bin/sh
    printf 'tt-model %s\n' "$*" >>"$FAKE_LOG"
    printf 'state=%s gc=%s\n' "''${TT_MODEL_STATE_ROOT:-}" "''${TT_MODEL_GC_ROOTS:-}" >>"$FAKE_LOG"
    SH
    chmod +x "$TMPDIR/systemctl" "$TMPDIR/lsof" "$TMPDIR/tt-smi" "$TMPDIR/tt-model"

    env_args=(
      "''${base_env[@]}"
      TT_MODEL_OWNER_SYSTEMCTL="$TMPDIR/systemctl"
      TT_MODEL_OWNER_LSOF="$TMPDIR/lsof"
      TT_MODEL_OWNER_TTSMI="$TMPDIR/tt-smi"
      TT_MODEL_OWNER_RECORD_TOOL="$TMPDIR/tt-model"
    )

    env "''${env_args[@]}" "$owner" cards >cards.out 2>cards.err
    grep -F 'owner_name=qwen38-p150x2.service' cards.out
    grep -F 'owner_pid=none' cards.out
    grep -F 'card /dev/tenstorrent/0' cards.out
    grep -F 'holder /dev/tenstorrent/0 4242 - tt_job' cards.out
    grep -F 'lsof /dev/tenstorrent/0' "$FAKE_LOG"

    env "''${env_args[@]}" "$owner" reset >reset.out 2>reset.err
    grep -F 'tt-model-owner: +' reset.err
    grep -F 'tt-smi -r /dev/tenstorrent/0 /dev/tenstorrent/1' reset.err
    grep -F 'reset /dev/tenstorrent/0 /dev/tenstorrent/1' reset.out
    grep -F 'tt-smi -r /dev/tenstorrent/0 /dev/tenstorrent/1' "$FAKE_LOG"

    env "''${env_args[@]}" "$owner" stop-owner >stop.out 2>stop.err
    grep -F 'already-inactive qwen38-p150x2.service' stop.out

    # An active owner is stopped only after the command is printed.
    cat >"$TMPDIR/systemctl-active" <<'SH'
    #!/bin/sh
    printf 'systemctl %s\n' "$*" >>"$FAKE_LOG"
    case "$1" in
      show) printf '4242\n' ;;
      is-active) if [ -e "$FAKE_STOPPED" ]; then exit 1; else exit 0; fi ;;
      is-enabled) printf 'enabled\n' ;;
      stop) : >"$FAKE_STOPPED"; exit 0 ;;
      *) exit 0 ;;
    esac
    SH
    chmod +x "$TMPDIR/systemctl-active"
    rm -f "$TMPDIR/stopped"
    env "''${env_args[@]}" TT_MODEL_OWNER_SYSTEMCTL="$TMPDIR/systemctl-active" \
      FAKE_STOPPED="$TMPDIR/stopped" \
      "$owner" stop-owner >stop-active.out 2>stop-active.err
    grep -F 'tt-model-owner: +' stop-active.err
    grep -F 'stop qwen38-p150x2.service' stop-active.err
    grep -F 'stopped qwen38-p150x2.service' stop-active.out

    # A masked unit is never started by the helper.
    cat >"$TMPDIR/systemctl-masked" <<'SH'
    #!/bin/sh
    printf 'systemctl %s\n' "$*" >>"$FAKE_LOG"
    case "$1" in
      show) printf '0\n' ;;
      is-active) exit 1 ;;
      is-enabled) printf 'masked\n' ;;
      *) exit 0 ;;
    esac
    SH
    chmod +x "$TMPDIR/systemctl-masked"
    expect_refusal "start of a masked unit" "is masked" \
      env "''${env_args[@]}" TT_MODEL_OWNER_SYSTEMCTL="$TMPDIR/systemctl-masked" \
      "$owner" start-owner

    env "''${env_args[@]}" "$owner" record --profile demo --package /nix/store/x >record.out 2>record.err
    grep -F 'record --profile demo --package /nix/store/x' "$FAKE_LOG"
    grep -F "state=$TMPDIR/state gc=$TMPDIR/gc" "$FAKE_LOG"

    mkdir -p "$out"
    echo "tt-model-owner check: PASS" >"$out/result"
  ''
