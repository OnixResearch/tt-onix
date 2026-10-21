"""Device-free checks for the hybrid prefix checkpoint identity."""

import hashlib
import os
import sys
import unittest
from pathlib import Path

SOURCE_ROOT = Path(__file__).parent / "service"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from qwen36_service.prefix_cache_core import (  # noqa: E402
    PREFIX_IDENTITY_ENVIRONMENT,
    prefix_cache_identity,
    prefix_digests,
)

VARIABLES = (*PREFIX_IDENTITY_ENVIRONMENT,)


class PrefixIdentityTests(unittest.TestCase):
    def setUp(self):
        self.saved = {name: os.environ.get(name) for name in VARIABLES}
        for name in VARIABLES:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_kv_precision_is_part_of_the_checkpoint_identity(self):
        self.assertIn("QWEN_SDPA_BF8", PREFIX_IDENTITY_ENVIRONMENT)
        base = prefix_cache_identity("package", "model")
        os.environ["QWEN_SDPA_BF8"] = "1"
        bf8 = prefix_cache_identity("package", "model")
        self.assertNotEqual(base, bf8)

    def test_each_numeric_or_precision_flag_changes_the_identity(self):
        base = prefix_cache_identity("package", "model")
        for name in VARIABLES:
            with self.subTest(name=name):
                for other in VARIABLES:
                    os.environ.pop(other, None)
                os.environ[name] = "1"
                self.assertNotEqual(prefix_cache_identity("package", "model"), base)

    def test_the_digest_key_binds_the_identity_fingerprint(self):
        tokens = list(range(2048))

        def digest(data):
            return hashlib.sha256(data).digest()

        first = prefix_digests(
            tokens, identity="package|model|", tenant="tenant", blake3_digest=digest
        )
        second = prefix_digests(
            tokens,
            identity="package|model|QWEN_SDPA_BF8=1",
            tenant="tenant",
            blake3_digest=digest,
        )
        self.assertEqual(len(first), 1)
        self.assertNotEqual(first, second)

    def test_identity_is_deterministic_and_binds_package_and_model(self):
        os.environ["QWEN_SDPA_BF8"] = "1"
        first = prefix_cache_identity("package-a", "model-a")
        self.assertEqual(first, prefix_cache_identity("package-a", "model-a"))
        self.assertNotEqual(first, prefix_cache_identity("package-b", "model-a"))
        self.assertNotEqual(first, prefix_cache_identity("package-a", "model-b"))


if __name__ == "__main__":
    unittest.main()
