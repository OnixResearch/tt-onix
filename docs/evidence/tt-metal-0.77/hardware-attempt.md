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
