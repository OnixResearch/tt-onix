"""Command line entry point. Traffic imports no ML dependencies."""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

# The installed entry point uses Python isolated mode; only our own module directory is added.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import retention
from traffic import PAYLOAD, Config, model


def parser():
    p = argparse.ArgumentParser(
        description="Modeled tile traffic and provenance-bound historical retention"
    )
    commands = p.add_subparsers(dest="command", required=True)
    t = commands.add_parser(
        "traffic", help="analytic y=act(A@B+C); never hardware counters"
    )
    for name in ("m", "k", "n"):
        t.add_argument("--" + name, type=int, required=True)
    for name in ("bm", "bn", "bk", "cols", "rows"):
        t.add_argument("--" + name, type=int, default=1)
    t.add_argument("--multicast", action="store_true")
    for name, default in (
        ("a", "bfloat16"),
        ("b", "bfloat4_b"),
        ("output", "bfloat16"),
        ("accumulator", "float32"),
    ):
        t.add_argument("--" + name + "-dtype", choices=PAYLOAD, default=default)
    t.add_argument("--buffers", type=int, default=2)
    t.add_argument("--out")
    r = commands.add_parser("retention")
    sub = r.add_subparsers(dest="action", required=True)
    f = sub.add_parser(
        "freeze", help="local-only actual tokenizer assets, no remote code"
    )
    f.add_argument("--model", required=True)
    f.add_argument(
        "--checkpoint",
        required=True,
        help="resolved checkpoint revision, not a host path",
    )
    f.add_argument("--out", required=True)
    s = sub.add_parser(
        "score", help="CPU full-vocabulary logits; float32 then float64 log-softmax"
    )
    s.add_argument("--corpus", required=True)
    s.add_argument("--logits", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--baseline")
    s.add_argument(
        "--changes", help="JSON file declaring exact changed provenance fields"
    )
    c = sub.add_parser("compare")
    c.add_argument("--baseline", required=True)
    c.add_argument("--treatment", required=True)
    c.add_argument(
        "--original-diagnostic",
        action="store_true",
        help="report original +0.010-nat diagnostic only",
    )
    c.add_argument("--out")
    e = sub.add_parser(
        "collect",
        help="HARDWARE export through supplied package-matched Python environment",
    )
    for name in ("corpus", "model", "package", "runtime", "cache", "out"):
        e.add_argument("--" + name, required=True)
    e.add_argument(
        "--decode-flags",
        help="JSON string mapping decode-only treatment environment flags",
    )
    e.add_argument(
        "--runtime-python",
        help="explicit package-matched interpreter; re-executed in isolated mode",
    )
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "traffic":
            cfg = Config(**{k: getattr(args, k) for k in asdict(Config(32, 32, 32))})
            result = model(
                cfg,
                args.a_dtype,
                args.b_dtype,
                args.output_dtype,
                args.accumulator_dtype,
                args.buffers,
            )
        elif args.action == "freeze":
            frozen = retention.freeze(args.model, args.out, args.checkpoint)
            result = {
                "corpus_sha256": retention.digest(args.out),
                "token_counts": [len(p["input_ids"]) for p in frozen["passages"]],
                "tokenizer_assets_sha256": frozen["tokenizer_assets_sha256"],
            }
        elif args.action == "score":
            result = retention.score(
                args.corpus,
                args.logits,
                args.out,
                args.baseline,
                retention.load(args.changes) if args.changes else None,
            )
        elif args.action == "compare":
            result = retention.compare(
                args.baseline, args.treatment, args.original_diagnostic
            )
        else:
            if args.runtime_python:
                interpreter = Path(args.runtime_python).absolute()
                if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
                    raise ValueError("runtime Python must be an existing executable")
                child = [
                    str(interpreter),
                    "-I",
                    str(Path(__file__).resolve()),
                    "retention",
                    "collect",
                ]
                for name in ("corpus", "model", "package", "runtime", "cache", "out"):
                    child.extend(["--" + name, getattr(args, name)])
                if args.decode_flags is not None:
                    child.extend(["--decode-flags", args.decode_flags])
                os.execv(str(interpreter), child)
            # Only collection admits explicitly supplied immutable runtime modules.
            # Offline traffic/freeze/score/compare retain the isolated packaged environment.
            for identity in (args.package, args.runtime):
                root = Path(identity).resolve(strict=True)
                if root.parent != Path("/nix/store"):
                    raise ValueError(
                        "collector package/runtime must resolve to immutable Nix store roots"
                    )
                modules = (
                    root
                    / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
                )
                if not modules.is_dir():
                    raise ValueError(
                        "package/runtime has no modules matching the collector Python version"
                    )
                sys.path.insert(1, str(modules))
            from collector import collect

            flags = json.loads(args.decode_flags) if args.decode_flags else {}
            if not isinstance(flags, dict) or any(
                not isinstance(k, str)
                or not k.startswith(("QWEN35_", "QWEN36_", "QWEN38_"))
                or not isinstance(v, str)
                for k, v in flags.items()
            ):
                raise ValueError(
                    "decode flags must be a QWEN environment string mapping"
                )
            result = collect(
                args.corpus,
                args.model,
                args.package,
                args.runtime,
                args.cache,
                args.out,
                flags,
            )
        if getattr(args, "out", None) and (
            args.command == "traffic" or args.action == "compare"
        ):
            retention.save(args.out, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, ImportError, RuntimeError) as exc:
        p.exit(2, "error: " + str(exc) + "\n")


if __name__ == "__main__":
    sys.exit(main())
