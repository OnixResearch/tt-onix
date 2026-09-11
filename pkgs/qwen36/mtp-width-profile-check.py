"""Check diagnostic width profiles without a device or model import."""

import argparse
import json
import sys
import unittest
from pathlib import Path

SOURCE_ROOT = Path(__file__).parent / "service"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from qwen36_service.mtp_drafter_core import (  # noqa: E402
    MAXIMUM_DRAFT_TOKEN_COUNT,
    MINIMUM_DRAFT_TOKEN_COUNT,
    MtpDrafterPolicyError,
    MtpDrafterProfile,
)
from qwen36_service.mtp_trace_core import MtpTracePolicyError, MtpTraceProfile  # noqa: E402

DEFAULT_DRAFT_COUNT = 3
TWO_DRAFT_COUNT = 2
VARIANT_FIELDS = frozenset(("profile_id", "draft_token_count"))


def profile_tests(directory):
    class WidthProfileTests(unittest.TestCase):
        def setUp(self):
            self.sources = {}
            for kind in ("drafter", "trace"):
                for suffix in ("", "-k2"):
                    path = directory / f"mtp-{kind}-profile{suffix}.json"
                    self.sources[kind, suffix] = json.loads(path.read_text())

        def test_two_draft_profiles_preserve_every_other_default(self):
            for kind, loader in (
                ("drafter", MtpDrafterProfile),
                ("trace", MtpTraceProfile),
            ):
                with self.subTest(kind=kind):
                    base = self.sources[kind, ""]
                    candidate = self.sources[kind, "-k2"]
                    self.assertEqual(
                        loader.from_mapping(base).draft_token_count, DEFAULT_DRAFT_COUNT
                    )
                    self.assertEqual(
                        loader.from_mapping(candidate).draft_token_count,
                        TWO_DRAFT_COUNT,
                    )
                    self.assertNotEqual(base["profile_id"], candidate["profile_id"])
                    self.assertEqual(
                        {
                            key: value
                            for key, value in base.items()
                            if key not in VARIANT_FIELDS
                        },
                        {
                            key: value
                            for key, value in candidate.items()
                            if key not in VARIANT_FIELDS
                        },
                    )

        def test_drafter_and_trace_have_matching_width_prompts_and_identity(self):
            drafter = self.sources["drafter", "-k2"]
            trace = self.sources["trace", "-k2"]
            for field in (
                "draft_token_count",
                "prompts",
                "identities",
                "expected_pcie_device_ids",
            ):
                self.assertEqual(drafter[field], trace[field])

        def test_out_of_range_widths_are_rejected_by_both_loaders(self):
            for kind, loader, error in (
                ("drafter", MtpDrafterProfile, MtpDrafterPolicyError),
                ("trace", MtpTraceProfile, MtpTracePolicyError),
            ):
                for width in (
                    MINIMUM_DRAFT_TOKEN_COUNT - 1,
                    MAXIMUM_DRAFT_TOKEN_COUNT + 1,
                    True,
                ):
                    with self.subTest(kind=kind, width=width):
                        source = dict(
                            self.sources[kind, "-k2"], draft_token_count=width
                        )
                        with self.assertRaises(error):
                            loader.from_mapping(source)

    return WidthProfileTests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_directory", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        profile_tests(args.profile_directory)
    )
    result = unittest.TextTestRunner().run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
