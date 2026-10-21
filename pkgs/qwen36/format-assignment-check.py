"""Device-free checks for the mixed-precision assignment solver."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE_ROOT = Path(__file__).parent / "service"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))
from qwen36_service.format_assignment_core import (  # noqa: E402
    FormatAssignmentError,
    FormatGroup,
    FormatOption,
    solve_assignment,
)


def group(name, parameters, *options):
    return FormatGroup(name, parameters, tuple(FormatOption(*item) for item in options))


class AssignmentTests(unittest.TestCase):
    def test_the_budget_buys_the_sensitivity_reduction_that_pays(self):
        groups = (
            group("attention", 4, ("bfp4", 4.5, 0.9), ("bfp8", 8.0, 0.1)),
            group("experts", 4, ("bfp4", 4.5, 0.2), ("bfp8", 8.0, 0.15)),
        )
        assignment = solve_assignment(groups, budget_effective_bits=6.25)
        self.assertEqual(assignment.formats, ("bfp8", "bfp4"))
        self.assertAlmostEqual(assignment.effective_bits, 6.25)
        self.assertAlmostEqual(assignment.sensitivity, 0.3)

    def test_a_larger_budget_uses_the_sensitive_group_first(self):
        groups = (
            group("a", 1, ("bfp4", 4.5, 0.8), ("bfp8", 8.0, 0.1)),
            group("b", 3, ("bfp4", 4.5, 0.2), ("bfp8", 8.0, 0.1)),
        )
        cheap = solve_assignment(groups, budget_effective_bits=4.5)
        self.assertEqual(cheap.formats, ("bfp4", "bfp4"))
        enough = solve_assignment(groups, budget_effective_bits=5.375)
        self.assertEqual(enough.formats, ("bfp8", "bfp4"))
        self.assertAlmostEqual(enough.sensitivity, 0.3)

    def test_below_the_cheapest_assignment_is_infeasible(self):
        groups = (group("a", 1, ("bfp4", 4.5, 0.8)),)
        with self.assertRaises(FormatAssignmentError):
            solve_assignment(groups, budget_effective_bits=4.0)

    def test_the_frontier_keeps_exact_half_bit_costs(self):
        groups = (
            group("a", 1, ("bfp4", 4.5, 0.3), ("bfp8", 8.0, 0.1)),
            group("b", 1, ("bfp4", 4.5, 0.3), ("bfp8", 8.0, 0.2)),
        )
        assignment = solve_assignment(groups, budget_effective_bits=6.25)
        self.assertEqual(assignment.formats, ("bfp8", "bfp4"))
        self.assertAlmostEqual(assignment.effective_bits, 6.25)

    def test_invalid_inputs_are_refused(self):
        for bad in (
            (),
            (group("a", 0, ("bfp4", 4.5, 0.1)),),
            (group("a", 1),),
            (group("a", 1, ("bfp4", 0.0, 0.1)),),
            (group("a", 1, ("bfp4", 4.5, -0.1)),),
        ):
            with self.subTest(groups=bad):
                with self.assertRaises(FormatAssignmentError):
                    solve_assignment(bad, budget_effective_bits=8.0)

    @unittest.skipUnless(
        Path(__file__).with_name("format-assignment.py").is_file(),
        "the source-tree CLI is not copied into the package check",
    )
    def test_the_cli_emits_the_same_map(self):
        source = {
            "groups": [
                {
                    "name": "attention",
                    "parameters": 4,
                    "candidates": [
                        {"format": "bfp4", "effective_bits": 4.5, "sensitivity": 0.9},
                        {"format": "bfp8", "effective_bits": 8.0, "sensitivity": 0.1},
                    ],
                },
                {
                    "name": "experts",
                    "parameters": 4,
                    "candidates": [
                        {"format": "bfp4", "effective_bits": 4.5, "sensitivity": 0.2},
                        {"format": "bfp8", "effective_bits": 8.0, "sensitivity": 0.15},
                    ],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensitivity.json"
            path.write_text(json.dumps(source))
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / "format-assignment.py"),
                    str(path),
                    "--effective-bits",
                    "6.25",
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        output = json.loads(result.stdout)
        self.assertEqual(output["formats"], {"attention": "bfp8", "experts": "bfp4"})
        self.assertAlmostEqual(output["effective_bits"], 6.25)


if __name__ == "__main__":
    unittest.main()
