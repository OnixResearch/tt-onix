"""Solve a TT mixed-precision assignment from a sensitivity JSON."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "service"))
from qwen36_service.format_assignment_core import (
    FormatAssignmentError,
    FormatGroup,
    FormatOption,
    solve_assignment,
)


def load_groups(path: Path):
    source = json.loads(path.read_text())
    if set(source) != {"groups"}:
        raise FormatAssignmentError("sensitivity input must hold exactly `groups`")
    groups = []
    for group in source["groups"]:
        if set(group) != {"name", "parameters", "candidates"}:
            raise FormatAssignmentError(
                "a group needs exactly name, parameters and candidates"
            )
        options = []
        for candidate in group["candidates"]:
            if set(candidate) != {"format", "effective_bits", "sensitivity"}:
                raise FormatAssignmentError(
                    "a candidate needs format, effective_bits and sensitivity"
                )
            options.append(
                FormatOption(
                    candidate["format"],
                    candidate["effective_bits"],
                    candidate["sensitivity"],
                )
            )
        groups.append(FormatGroup(group["name"], group["parameters"], tuple(options)))
    return tuple(groups)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sensitivity", type=Path)
    parser.add_argument("--effective-bits", type=float, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        groups = load_groups(args.sensitivity)
        assignment = solve_assignment(
            groups, budget_effective_bits=args.effective_bits
        )
    except (FormatAssignmentError, OSError, json.JSONDecodeError) as error:
        print(f"format assignment: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(
            json.dumps(
                {
                    "formats": {
                        group.name: chosen
                        for group, chosen in zip(groups, assignment.formats)
                    },
                    "effective_bits": assignment.effective_bits,
                    "sensitivity": assignment.sensitivity,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for group, chosen in zip(groups, assignment.formats):
            print(f"{group.name}: {chosen}")
        print(
            "effective bits: "
            f"{assignment.effective_bits:.4f}; sensitivity: "
            f"{assignment.sensitivity:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
