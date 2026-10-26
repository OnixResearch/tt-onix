"""Read-only access to explicitly selected, pinned official review references."""

import argparse
import json
from pathlib import Path

DOMAINS = {
    "review-core": "common/tt-review-core",
    "precision": "models/tt-precision-review",
    "trace": "models/tt-trace-review",
    "l1": "metal/tt-l1-memory-review",
    "kernel": "ttnn/ttnn-op-kernel-review",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    for name in ("resolve", "read"):
        command = commands.add_parser(name)
        command.add_argument("domains", nargs="+", choices=DOMAINS)
        command.add_argument(
            "--file", default="SKILL.md", help="File relative to each domain"
        )
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.command == "list":
        print(
            json.dumps(
                {
                    "revision": (root / "REVISION").read_text().strip(),
                    "domains": DOMAINS,
                },
                indent=2,
            )
        )
        return
    domains = list(
        dict.fromkeys(domain for domain in args.domains if domain != "review-core")
    )
    if len(domains) > 2:
        parser.error(
            "select at most two domain skills in addition to automatic review-core"
        )
    selections = [("review-core", "SKILL.md")]
    selections.extend((domain, args.file) for domain in domains)
    if not domains and args.file != "SKILL.md":
        selections.append(("review-core", args.file))
    for domain, filename in selections:
        directory = (root / "skills" / DOMAINS[domain]).resolve(strict=True)
        relative = Path(filename)
        path = (directory / relative).resolve()
        if (
            relative.is_absolute()
            or not path.is_relative_to(directory)
            or not path.is_file()
        ):
            parser.error(
                "--file must resolve to an installed file within the selected domain"
            )
        if args.command == "resolve":
            print(path)
        else:
            print(f"# {domain}: {path.relative_to(root)}")
            print(path.read_text())


if __name__ == "__main__":
    main()
