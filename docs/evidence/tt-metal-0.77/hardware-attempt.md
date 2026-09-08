# Bounded hardware attempt

The user approved a temporary service stop for the GDN test on 2026-09-07. The user stopped `qwen38-p150x2.service` after administrator authentication blocked the agent.

Preflight on `britton-desktop` showed the service inactive with MainPID 0. An unprivileged `fuser` check reported no device users. This check cannot exclude processes hidden by permissions.

Both device sysfs entries reported firmware `19.11.0.0`. UMD reported firmware 19.11.0 and KMD 2.10.0 during discovery.

The test used the same `pkgs/ttsim/gdn-smoke.py` and TT-Metal output as the simulator check:

```text
/nix/store/pj7h4adxvyd96gx5l9gsq50wf608m41g-tt-metal-0.77.0
```

The hardware attempt used slow dispatch, disabled SFPLOADMACRO, and removed simulator environment overrides. Its timeout was 600 seconds with a 30-second termination grace.

UMD discovered devices 0 and 1, then stopped at device locking:

```text
Waiting for lock 'CHIP_IN_USE_0_PCIe' which is currently held by thread TID: 1951525, PID: 1951525
```

A subsequent `ps` query found no process with that PID. The agent killed the probe after approximately 26 seconds rather than wait at the apparent stale lock. No GDN numerical result exists.

The agent did not remove shared mutex files, reset devices, change firmware, or deploy the candidate. The UMD header warns that its implementation relies on clients not deleting the shared mutex files.

The service restart also required interactive administrator authentication. The final observed service state was inactive with MainPID 0. The user must restore it with administrator access. Further testing requires a privileged ownership check and safe lock recovery.

The full local log is `~/.cache/tt-upgrade-review/evidence/gdn077-hardware.log`.

## Recovery and numerical comparison, 2026-09-08

The user ran the guarded recovery script with administrator access. The agent then verified that both chip-in-use lock files were absent and the service remained inactive.

The same bounded GDN test ran on physical hardware with both runtime packages:

- Candidate: `/nix/store/pj7h4adxvyd96gx5l9gsq50wf608m41g-tt-metal-0.77.0`.
- Baseline: `/nix/store/2n6r40bpg5a38aib9i8inxv5hrkyrf24-tt-metal-0.74.0`.

Both runs passed device opening and kernel compilation. Both failed the state comparison and closed the device drivers normally. Neither run reached complete recurrent-state, cache-reuse, or invalid-input acceptance.

Both runs reported the same mismatch summary:

```text
Not equal to tolerance rtol=0.001, atol=0.0001
Mismatched elements: 4890 / 32768 (14.9%)
Max absolute difference among violations: 0.00022145
Max relative difference among violations: 0.03129307
```

This comparison shows that the observed numerical failure also exists with the baseline runtime. It does not establish the cause or complete numerical equivalence between runtimes. No tolerance or kernel code changed.

The local logs are `gdn077-recovered.log` and `gdn074-hardware.log` under `~/.cache/tt-upgrade-review/evidence/`. No passing result JSON exists for either run.

After both runs, the service remained inactive and an unprivileged `fuser` check reported no device users. The startup lock obstruction is resolved for these probes. At that point, service readiness and protection against another OOM kill remained unverified. A restart required the user's administrator access.

## Service restored

The user restarted the existing service at 01:16:16 EDT on 2026-09-08. It passed device initialization, loaded the model, and completed startup warm-up for widths 1 and 4. Systemd reported `active/running` at 01:18:18 EDT.

A subsequent GET to `http://127.0.0.1:8000/readyz` returned `status: ok` and `ready: true` for Qwen3.8-27B. The existing service package remains `/nix/store/cdvkpnr32i9rs6h40i464wwd0p52biy0-qwen36-0.1.0`. The agent did not deploy the 0.77 candidate.

This verifies startup recovery. It does not resolve the separate GDN numerical failure or establish protection against another OOM kill.
