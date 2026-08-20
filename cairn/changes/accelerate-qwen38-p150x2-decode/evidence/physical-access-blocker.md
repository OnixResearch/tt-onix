# Physical acceptance blocker

Date: 2026-08-20

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

The next check is a matched eager-versus-candidate run after an owner authorizes a bounded service outage.
The run must restore `qwen38-p150x2.service` before it ends.
