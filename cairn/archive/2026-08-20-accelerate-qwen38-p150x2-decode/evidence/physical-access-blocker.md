# Resolved physical acceptance blocker

Date: 2026-08-20
Resolution date: 2026-08-20

The host exposes two Blackhole P150A devices as UMD chip IDs `0` and `1`.
The root-owned `qwen38-p150x2.service` process owns both devices.
The service listens on `127.0.0.1:8000` and returns a passing health response.

The candidate package passed its device-free package gate.
Physical A/B execution needs an outage for the root-owned service.
The current user cannot authorize that outage.

The bounded stop attempt failed with this exact result:

```text
Failed to stop qwen38-p150x2.service: Access denied as the requested operation requires interactive authentication. However, interactive authentication has not been enabled by the calling program.
```

`sudo -n true` also failed with `sudo: a password is required`.
No physical candidate command ran.
The candidate remains unadmitted, and the production wrapper remains eager.

The owner later authorized a bounded outage and stopped the service.
The matched physical run completed and rejected the candidate because its TTFT ratio exceeded the profile bound.

The final result is in `physical-result.md`.
