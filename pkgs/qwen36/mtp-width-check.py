"""Device-free checks for expected-value MTP width selection."""

import sys
import unittest
from pathlib import Path

SOURCE_ROOT = Path(__file__).parent / "service"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from qwen36_service.mtp_width_core import (  # noqa: E402
    MODE_PLAIN,
    MODE_SPECULATIVE,
    MtpWidthPolicyError,
    WidthMeasurement,
    expected_emitted_tokens,
    select_round_width,
)


class WidthSelectionTests(unittest.TestCase):
    def test_expected_tokens_sum_the_accepted_prefixes(self):
        measurement = WidthMeasurement(3, 0.1, (1.0, 0.5, 0.25))
        self.assertAlmostEqual(expected_emitted_tokens(measurement), 2.625)

    def test_zero_acceptance_still_emits_the_continuation_token(self):
        measurement = WidthMeasurement(3, 0.1, (0.0, 0.0, 0.0))
        self.assertEqual(expected_emitted_tokens(measurement), 1.0)
        plan = select_round_width(
            remaining_tokens=10,
            plain_step_seconds=0.1,
            measurements=(measurement,),
        )
        self.assertEqual(plan.mode, MODE_PLAIN)

    def test_the_wider_round_wins_when_acceptance_pays_for_it(self):
        fast_plain = 0.1
        measurements = (
            WidthMeasurement(2, 0.12, (0.9, 0.9)),
            WidthMeasurement(5, 0.2, (0.9, 0.9, 0.9, 0.9, 0.9)),
        )
        plan = select_round_width(
            remaining_tokens=20,
            plain_step_seconds=fast_plain,
            measurements=measurements,
        )
        self.assertEqual(plan.mode, MODE_SPECULATIVE)
        self.assertEqual(plan.draft_token_count, 5)

    def test_a_cheap_narrow_round_beats_a_slow_wide_one(self):
        measurements = (
            WidthMeasurement(2, 0.05, (0.8, 0.8)),
            WidthMeasurement(4, 0.5, (1.0, 1.0, 1.0, 1.0)),
        )
        plan = select_round_width(
            remaining_tokens=20,
            plain_step_seconds=0.1,
            measurements=measurements,
        )
        self.assertEqual(plan.draft_token_count, 2)

    def test_widths_that_do_not_fit_the_tail_are_ineligible(self):
        measurements = (WidthMeasurement(5, 0.01, (1.0, 1.0, 1.0, 1.0, 1.0)),)
        plan = select_round_width(
            remaining_tokens=5,
            plain_step_seconds=0.1,
            measurements=measurements,
        )
        self.assertEqual(plan.mode, MODE_PLAIN)

    def test_a_plain_step_wins_ties_and_a_missing_catalog_is_plain(self):
        measurement = WidthMeasurement(3, 0.4, (1.0, 1.0, 1.0))
        # Four expected tokens over 0.4 s is exactly one token per plain step.
        plan = select_round_width(
            remaining_tokens=10,
            plain_step_seconds=0.1,
            measurements=(measurement,),
        )
        self.assertEqual(plan.mode, MODE_PLAIN)
        self.assertEqual(
            select_round_width(
                remaining_tokens=10,
                plain_step_seconds=0.1,
                measurements=(),
            ).mode,
            MODE_PLAIN,
        )

    def test_invalid_measurements_are_refused(self):
        for acceptance in ((0.5,), (0.5, 1.5), (0.5, float("nan"))):
            with self.subTest(acceptance=acceptance):
                with self.assertRaises(MtpWidthPolicyError):
                    WidthMeasurement(2, 0.1, acceptance)
        with self.assertRaises(MtpWidthPolicyError):
            WidthMeasurement(2, 0.0, (0.5, 0.5))
        with self.assertRaises(MtpWidthPolicyError):
            select_round_width(
                remaining_tokens=10,
                plain_step_seconds=0.1,
                measurements=(
                    WidthMeasurement(2, 0.1, (0.5, 0.5)),
                    WidthMeasurement(2, 0.2, (0.5, 0.5)),
                ),
            )
        with self.assertRaises(MtpWidthPolicyError):
            select_round_width(
                remaining_tokens=0,
                plain_step_seconds=0.1,
                measurements=(),
            )


if __name__ == "__main__":
    unittest.main()
