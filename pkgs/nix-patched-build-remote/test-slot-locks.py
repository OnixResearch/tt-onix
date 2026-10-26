#!/usr/bin/env python3
"""Exercise the installed build hook's real slot locks, without a network peer."""

import fcntl
import hashlib
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def integer(value):
    return struct.pack("<Q", value)


def string(value):
    data = value.encode()
    return integer(len(data)) + data + bytes(-len(data) % 8)


def exercise(nix, root, uri, key):
    state = root / "state"
    locks = state / "current-load"
    locks.mkdir(parents=True, exist_ok=True)
    lock = locks / (key + "-0")
    # An occupied slot must yield postpone. Choosing any different key would
    # try connecting instead; an overlong key would abort with ENAMETOOLONG.
    with lock.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = {
            "builders": f"{uri} x86_64-linux - 1 1",
            "max-jobs": "0",
            "plugin-files": "",
        }
        payload = b"".join(
            integer(1) + string(name) + string(value)
            for name, value in settings.items()
        ) + integer(0)
        payload += (
            string("try")
            + integer(0)
            + string("x86_64-linux")
            + string("/nix/store/00000000000000000000000000000000-slot-lock-test.drv")
            + integer(0)
        )
        env = os.environ | {
            "NIX_CONF_DIR": str(root),
            "NIX_STATE_DIR": str(state),
            "NIX_REMOTE": "dummy://",
            "NIX_CONFIG": "",
        }
        result = subprocess.run(
            [nix, "__build-remote", "0"],
            input=payload,
            capture_output=True,
            env=env,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode()
        assert b"# postpone\n" in result.stderr, result.stderr.decode()
    return lock


def main():
    nix = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="nix-slot-locks-") as temp:
        root = Path(temp)
        name_max = os.pathconf(temp, "PC_NAME_MAX")
        prefix = "ssh://builder.invalid?key="
        boundary = prefix + "a" * (name_max - 2 - len(prefix))
        short = exercise(nix, root, boundary, boundary.replace("/", "_"))
        assert len(short.name.encode()) == name_max
        long_uri = boundary + "a"
        digest = hashlib.sha256(long_uri.encode()).hexdigest()
        first = exercise(nix, root, long_uri, digest)
        inode = first.stat().st_ino
        assert exercise(nix, root, long_uri, digest).stat().st_ino == inode
        other_uri = long_uri[:-1] + "b"
        other = exercise(
            nix, root, other_uri, hashlib.sha256(other_uri.encode()).hexdigest()
        )
        assert first.stat().st_ino != other.stat().st_ino
        huge_uri = prefix + "z" * 4096
        exercise(nix, root, huge_uri, hashlib.sha256(huge_uri.encode()).hexdigest())
        print(
            f"slot-lock regression: legacy {name_max}-byte boundary, overflow, deterministic reuse, distinct URIs, 4KiB URI passed"
        )


if __name__ == "__main__":
    main()
