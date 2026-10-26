# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.runtime_profile.owner_helper]
# The only privileged program in this repository. It owns the five operations a
# switch needs and nothing else: report card holders, stop the declared owner,
# start it, reset the boards, and write the root-owned record.
#
# It is deliberately separate from `tt-model`: that tool is device-free and its
# program text must never name a device node, tt-smi, or the owner helper, which
# the `tt-model` install check asserts.
#
# Every acting subcommand prints its exact argv before it runs, refuses when the
# host has not recorded this exact helper, and refuses any overridden state root,
# GC-root directory, unit name, or command path unless `TT_MODEL_OWNER_TEST=1`.
{
  lib,
  writeShellApplication,
  coreutils,
  systemd,
  lsof,
  tt-smi,
  tt-model,
  commandName,
  ownerUnit,
  cardDevices,
}:
writeShellApplication {
  name = commandName;
  runtimeInputs = [
    coreutils
    lsof
    systemd
    tt-smi
  ];
  text = ''
    readonly command_name=${lib.escapeShellArg commandName}
    readonly default_owner_unit=${lib.escapeShellArg ownerUnit}
    readonly default_state_root="/var/lib/tenstorrent-models"
    readonly default_gc_roots="/nix/var/nix/gcroots/tenstorrent-models"
    readonly default_record_tool=${lib.escapeShellArg "${tt-model}/bin/tt-model"}
    readonly card_devices=(${
      lib.concatMapStringsSep " " (device: lib.escapeShellArg device) cardDevices
    })
    readonly test_flag="TT_MODEL_OWNER_TEST"
    readonly ok_status=0
    readonly failure_status=1
      readonly usage_status=2

    fail() {
      printf '%s: %s\n' "$command_name" "$1" >&2
      exit "$failure_status"
    }

    usage() {
      local status="''${1:-$usage_status}"
      cat >&2 <<USAGE
    usage: $command_name cards|stop-owner|start-owner|reset|record [--] [args...]

      cards        report each card and the processes that hold it
      stop-owner   stop the declared owner unit
      start-owner  start the declared owner unit
      reset        reset the declared card devices
      record       write the root-owned record (arguments go to tt-model record)
    USAGE
      exit "$status"
    }

    # Print the exact command, then run it.
    run() {
      printf '%s: +' "$command_name" >&2
      printf ' %q' "$@" >&2
      printf '\n' >&2
      "$@"
    }

    # An override is a test-only capability. A real host state change must never
    # be one environment variable away from a different directory or unit.
    require_test_flag_if_overridden() {
      local name="$1" value="$2" default="$3"
      if [ "$value" != "$default" ] && [ "''${!test_flag:-0}" != "1" ]; then
        fail "$name is overridden to '$value' but $test_flag is not 1"
      fi
    }

    resolve_owner_unit() {
      local value="''${TT_MODEL_OWNER_UNIT:-$default_owner_unit}"
      require_test_flag_if_overridden TT_MODEL_OWNER_UNIT "$value" "$default_owner_unit"
      [ -n "$value" ] || fail "no owner unit is declared; refusing to guess one"
      printf '%s\n' "$value"
    }

    resolve_command() {
      local name="$1" override="$2" fallback="$3" resolved
      require_test_flag_if_overridden "$name" "$override" ""
      resolved="''${override:-$fallback}"
      [ -n "$resolved" ] || fail "$name is not set and no fallback exists"
      [ -x "$resolved" ] || fail "$name is not executable: $resolved"
      printf '%s\n' "$resolved"
    }

    # The host records which helper may act. A rebuilt helper with a different
    # store path must not inherit the authority of the recorded one.
    require_installed_helper() {
      local pointer="$1" self expected
      self=$(unset CDPATH && cd -- "$(dirname -- "$(readlink -f -- "$0")")/.." && pwd)
      if [ ! -L "$pointer" ] && [ ! -e "$pointer" ]; then
        fail "no owner helper is installed at $pointer; refusing to act"
      fi
      expected=$(readlink -f -- "$pointer") || fail "cannot resolve $pointer"
      if [ "$expected" != "$self" ]; then
        fail "$command_name is $self but the host records $expected"
      fi
    }

    unit_of_pid() {
      local pid="$1" cgroup line component
      cgroup=$(cat "/proc/$pid/cgroup" 2>/dev/null) || { printf -- '-\n'; return; }
      while IFS= read -r line; do
        line="''${line#*:}"
        line="''${line#*:}"
        component="''${line##*/}"
        case "$component" in
          *.service) printf '%s\n' "$component"; return ;;
        esac
      done <<<"$cgroup"
      printf -- '-\n'
    }

    owner_pid() {
      local unit="$1" pid
      pid=$("$systemctl_bin" show -p MainPID --value "$unit" 2>/dev/null) || pid=0
      case "$pid" in
        "" | 0) printf -- 'none\n' ;;
        *[!0-9]*) fail "unexpected MainPID '$pid' for $unit" ;;
        *) printf '%s\n' "$pid" ;;
      esac
    }

    cards() {
      local unit="$1" device status output command pid _user rest holder_unit
      printf 'owner_name=%s\n' "$unit"
      printf 'owner_pid=%s\n' "$(owner_pid "$unit")"
      for device in "''${card_devices[@]}"; do
        printf 'card %s\n' "$device"
        status=0
        output=$("$lsof_bin" "$device" 2>&1) || status=$?
        if [ "$status" -gt "$ok_status" ] && [ "$status" -ne "$failure_status" ]; then
          fail "lsof failed for $device with status $status: $output"
        fi
        if [ "$status" -eq "$ok_status" ]; then
          while read -r command pid _user rest; do
            [ "$command" = "COMMAND" ] && continue
            [ -n "$pid" ] || continue
            holder_unit=$(unit_of_pid "$pid")
            printf 'holder %s %s %s %s\n' "$device" "$pid" "$holder_unit" "$command"
          done <<<"$output"
        fi
      done
    }

    stop_owner() {
      local unit="$1"
      if ! "$systemctl_bin" is-active --quiet "$unit"; then
        printf 'already-inactive %s\n' "$unit"
        return "$ok_status"
      fi
      run "$systemctl_bin" stop "$unit"
      if "$systemctl_bin" is-active --quiet "$unit"; then
        fail "$unit remained active after stop"
      fi
      printf 'stopped %s\n' "$unit"
    }

    start_owner() {
      local unit="$1"
      if [ "$("$systemctl_bin" is-enabled "$unit" 2>/dev/null || true)" = "masked" ]; then
        fail "$unit is masked; enabling it is a host-generation change, not a helper action"
      fi
      run "$systemctl_bin" start "$unit"
      if ! "$systemctl_bin" is-active --quiet "$unit"; then
        fail "$unit did not become active after start"
      fi
      printf 'started %s\n' "$unit"
    }

    reset_cards() {
      [ "''${#card_devices[@]}" -gt 0 ] || fail "no card devices are declared; refusing to reset"
      run "$ttsmi_bin" -r "''${card_devices[@]}"
      printf 'reset %s\n' "''${card_devices[*]}"
    }

    record_active() {
      local state_root="$1" gc_roots="$2" record_tool="$3"
      shift 3
      run env \
        TT_MODEL_STATE_ROOT="$state_root" \
        TT_MODEL_GC_ROOTS="$gc_roots" \
        "$record_tool" record "$@"
    }

    [ "$#" -ge 1 ] || usage
    mode="$1"
    shift

    case "$mode" in
      --help | -h)
        usage "$ok_status"
        ;;
      cards | stop-owner | start-owner | reset | record) ;;
      *)
        usage
        ;;
    esac

    state_root="''${TT_MODEL_STATE_ROOT:-$default_state_root}"
    gc_roots="''${TT_MODEL_GC_ROOTS:-$default_gc_roots}"
    require_test_flag_if_overridden TT_MODEL_STATE_ROOT "$state_root" "$default_state_root"
    require_test_flag_if_overridden TT_MODEL_GC_ROOTS "$gc_roots" "$default_gc_roots"
    require_installed_helper "$state_root/owner-helper"

    systemctl_bin=$(resolve_command TT_MODEL_OWNER_SYSTEMCTL "''${TT_MODEL_OWNER_SYSTEMCTL:-}" "$(command -v systemctl)")
    lsof_bin=$(resolve_command TT_MODEL_OWNER_LSOF "''${TT_MODEL_OWNER_LSOF:-}" "$(command -v lsof)")
    ttsmi_bin=$(resolve_command TT_MODEL_OWNER_TTSMI "''${TT_MODEL_OWNER_TTSMI:-}" "$(command -v tt-smi)")
    record_tool=$(resolve_command TT_MODEL_OWNER_RECORD_TOOL "''${TT_MODEL_OWNER_RECORD_TOOL:-}" "$default_record_tool")
    unit=$(resolve_owner_unit)

    case "$mode" in
      cards)
        cards "$unit"
        ;;
      stop-owner)
        stop_owner "$unit"
        ;;
      start-owner)
        start_owner "$unit"
        ;;
      reset)
        reset_cards
        ;;
      record)
        record_active "$state_root" "$gc_roots" "$record_tool" "$@"
        ;;
    esac
  '';

  meta = {
    description = "The privileged owner helper for the Tenstorrent runtime-profile switch";
    license = lib.licenses.lgpl21Only;
    mainProgram = commandName;
    platforms = [ "x86_64-linux" ];
  };
}
