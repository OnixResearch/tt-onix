"""Exercise the actual pinned Blackhole PACR model without building simulator libraries.

Usage: python3 run-packer-zero-flags-regression.py /path/to/patched/ttsim-1.10.6
The source tree is read-only: generated headers and the executable live in a temporary directory.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source.resolve() / "src"
    regression = Path(__file__).with_name("packer-zero-flags-regression.cpp")
    with tempfile.TemporaryDirectory(prefix="ttsim-packer-regression-") as temporary:
        out = Path(temporary)
        for name, generator in (
            ("tensix_decode", "tensix_gen_decode.py"),
            ("tensix_regs", "tensix_gen_regs.py"),
            ("tile_regs", "tile_gen_regs.py"),
        ):
            subprocess.run(
                [
                    sys.executable,
                    str(source / generator),
                    "--chip",
                    "bh",
                    "--out",
                    str(out / f"{name}.h"),
                ],
                cwd=source,
                check=True,
            )
        executable = out / "packer-regression"
        subprocess.run(
            [
                os.environ.get("CXX", "g++"),
                "-std=c++20",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Wno-unused-parameter",
                "-fno-exceptions",
                "-fno-rtti",
                "-ffunction-sections",
                "-fdata-sections",
                "-DTT_ARCH_VERSION=1",
                "-I",
                str(source),
                "-I",
                str(out),
                str(regression),
                str(source / "common.cpp"),
                "-Wl,--gc-sections",
                "-o",
                str(executable),
            ],
            check=True,
        )
        subprocess.run([str(executable)], check=True)
        for case, diagnostic in (
            ("compression", "Disable_zero_compress"),
            ("threshold", "Exp_threshold_en"),
            ("metadata", "set_signals_mode=1"),
        ):
            result = subprocess.run(
                [str(executable), case], text=True, capture_output=True
            )
            if (
                result.returncode != 1
                or "UnsupportedFunctionality" not in result.stdout
                or diagnostic not in result.stdout
            ):
                raise RuntimeError(
                    f"{case}: expected original unsupported guard; got {result.returncode}: "
                    f"{result.stdout}{result.stderr}"
                )
            print(f"PASS: {case} retains UnsupportedFunctionality ({diagnostic})")


if __name__ == "__main__":
    main()
