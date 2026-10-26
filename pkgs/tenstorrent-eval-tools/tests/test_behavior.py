import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import retention as r
import torch
from traffic import Config, model, storage


class TrafficTests(unittest.TestCase):
    def test_qwen_weight_exponents_and_padding(self):
        out = model(Config(1, 5120, 8704))
        self.assertEqual(out["compulsory"]["b"]["packed_bytes"], 25067520)
        self.assertEqual(out["compulsory"]["b"]["exponent_bytes"], 2785280)
        self.assertEqual(out["logical_payload_bytes"]["a"], 10240)
        self.assertEqual(out["compulsory"]["a"]["payload_bytes"], 327680)
        self.assertEqual(storage(1, "bfloat8_b")["packed_bytes"], 1088)

    def test_rereads_and_grid_correspondence(self):
        plain = model(Config(128, 64, 192, cols=3, rows=2))
        cast = model(Config(128, 64, 192, cols=3, rows=2, multicast=True))
        self.assertEqual(plain["fused_streams"]["a"]["tiles"], 48)
        self.assertEqual(plain["fused_streams"]["b"]["tiles"], 48)
        self.assertEqual(cast["fused_streams"]["a"]["tiles"], 16)
        self.assertEqual(cast["fused_streams"]["b"]["tiles"], 24)
        self.assertEqual(cast["rereads"]["a"]["tiles"], 8)
        self.assertEqual(cast["rereads"]["b"]["tiles"], 12)
        self.assertEqual(
            cast["totals"]["fused_bytes"],
            cast["totals"]["compulsory_bytes"] + cast["totals"]["reread_bytes"],
        )
        self.assertEqual(
            cast["multicast_saved_bytes"],
            plain["totals"]["fused_bytes"] - cast["totals"]["fused_bytes"],
        )

    def test_intermediate_and_l1_accounting(self):
        out = model(Config(32, 32, 32), a="bfloat16", b="bfloat16")
        self.assertEqual(out["totals"]["intermediate_saved_bytes"], 8192)
        self.assertEqual(out["totals"]["l1_bytes_per_core"], 24576)
        self.assertEqual(out["totals"]["unfused_bytes"], 16384)

    def test_invalid_boundaries(self):
        for dims in ((0, 32, 32), (True, 32, 32), (1.5, 32, 32)):
            with self.assertRaises(ValueError):
                Config(*dims)
        with self.assertRaises(ValueError):
            Config(32, 32, 32, cols=2, multicast=True)
        for buffers in (0, True, 1.5):
            with self.assertRaises(ValueError):
                model(Config(32, 32, 32), buffers=buffers)
        with self.assertRaisesRegex(ValueError, "unsupported dtype"):
            model(Config(32, 32, 32), b="bf4")


class CheckpointContentTests(unittest.TestCase):
    def test_index_and_every_unique_referenced_shard_are_hashed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            r.save(
                root / "model.safetensors.index.json",
                {
                    "weight_map": {
                        "a": "first.safetensors",
                        "b": "second.safetensors",
                        "c": "second.safetensors",
                    }
                },
            )
            (root / "first.safetensors").write_bytes(b"first weights")
            (root / "second.safetensors").write_bytes(b"second weights")
            manifest = r.weight_manifest(root)
            self.assertEqual(
                set(manifest["shards"]), {"first.safetensors", "second.safetensors"}
            )
            self.assertEqual(
                manifest["index"]["sha256"],
                r.digest(root / "model.safetensors.index.json"),
            )
            self.assertEqual(
                manifest["shards"]["first.safetensors"]["sha256"],
                r.digest(root / "first.safetensors"),
            )
            self.assertEqual(
                manifest["shards"]["second.safetensors"]["sha256"],
                r.digest(root / "second.safetensors"),
            )
            (root / "second.safetensors").unlink()
            with self.assertRaises(FileNotFoundError):
                r.weight_manifest(root)

    def test_index_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            r.save(
                root / "model.safetensors.index.json",
                {"weight_map": {"a": "../outside.safetensors"}},
            )
            with self.assertRaisesRegex(ValueError, "inside the model"):
                r.weight_manifest(root)


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Synthetic CPU logits validate arithmetic, not actual model retention.
        self.frozen = r.load(r.ROOT / "data/qwen38-six-passages.json")
        self.frozen["vocabulary_size"] = 8
        self.frozen["tokenizer_assets_sha256"] = {"synthetic": "CPU-test-not-model"}
        for i, p in enumerate(self.frozen["passages"]):
            p["input_ids"] = [(j + i) % 8 for j in range(14 + i * 3)]
            p["target_indices"] = list(range(13, len(p["input_ids"])))
        self.corpus = self.root / "corpus.json"
        r.save(self.corpus, self.frozen)
        self.prov = {
            "model": "synthetic",
            "checkpoint": "test",
            "runtime": "cpu",
            "package": "test",
            "cache": "none",
            "precision_flags": {},
            "execution": "synthetic-cpu-not-model",
        }
        self.export = {
            "corpus_sha256": r.digest(self.corpus),
            "provenance": self.prov,
            "target_indices": [p["target_indices"] for p in self.frozen["passages"]],
            "rows": [],
        }
        for i, p in enumerate(self.frozen["passages"]):
            logits = torch.zeros((len(p["target_indices"]), 8), dtype=torch.float64)
            for j, target in enumerate(p["input_ids"][13:]):
                logits[j, target] = i + 0.123456789
            self.export["rows"].append(logits)
        self.logits = self.root / "logits.pt"
        self.base = self.root / "base.json"

    def run_score(self, **kwargs):
        torch.save(self.export, self.logits)
        return r.score(self.corpus, self.logits, self.base, **kwargs)

    def test_f32_then_double_equal_passage_weighting(self):
        out = self.run_score()
        expected = [
            math.log(
                math.exp(float(torch.tensor(i + 0.123456789, dtype=torch.float32))) + 7
            )
            - float(torch.tensor(i + 0.123456789, dtype=torch.float32))
            for i in range(6)
        ]
        for actual, want in zip(out["passage_nll"], expected):
            self.assertAlmostEqual(actual, want, places=13)
        self.assertAlmostEqual(out["mean_nll"], sum(expected) / 6, places=13)
        pooled = sum(v * n for v, n in zip(expected, out["scored_token_counts"])) / sum(
            out["scored_token_counts"]
        )
        self.assertGreater(abs(out["mean_nll"] - pooled), 0.1)

    def test_target_offbyone_rejected(self):
        self.export["target_indices"][0] = [12]
        with self.assertRaisesRegex(ValueError, "13..end"):
            self.run_score()

    def test_row_count_and_nonfinite_and_vocabulary_errors(self):
        original = self.export["rows"][0]
        for bad in (
            original[:0],
            torch.full_like(original, float("nan")),
            original[:, :3],
        ):
            self.export["rows"][0] = bad
            with self.assertRaises(ValueError):
                self.run_score()

    def test_corpus_identity_rejected(self):
        self.export["corpus_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_score()

    def test_matched_receipts_and_reference_binding(self):
        self.run_score()
        changes = {"precision_flags": {"QWEN38_ROUNDED_REGISTER_SWIGLU": "1"}}
        self.export["provenance"] = dict(self.prov, **changes)
        torch.save(self.export, self.logits)
        trial = self.root / "trial.json"
        r.score(self.corpus, self.logits, trial, self.base, changes)
        result = r.compare(self.base, trial)
        self.assertEqual(result["mean_nll_delta"], 0)
        self.assertNotIn("original_diagnostic", result)
        self.assertTrue(
            r.compare(self.base, trial, True)["original_diagnostic"]["passes"]
        )
        altered = r.load(self.base)
        altered["provenance"]["runtime"] = "other"
        r.save(self.base, altered)
        with self.assertRaisesRegex(ValueError, "content-bound"):
            r.compare(self.base, trial)

    def test_extra_logit_padding_rejected(self):
        self.export["rows"][0] = torch.zeros((1, 9))
        with self.assertRaisesRegex(ValueError, "exact model full vocabulary"):
            self.run_score()

    def test_declared_package_code_change_allowed_but_execution_fixed(self):
        self.run_score()
        self.export["provenance"] = dict(self.prov, package="candidate-code-identity")
        torch.save(self.export, self.logits)
        trial = self.root / "trial.json"
        r.score(
            self.corpus,
            self.logits,
            trial,
            self.base,
            {"package": "candidate-code-identity"},
        )
        self.assertEqual(r.compare(self.base, trial)["mean_nll_delta"], 0)
        self.export["provenance"]["execution"] = "hardware"
        torch.save(self.export, self.logits)
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            r.score(
                self.corpus,
                self.logits,
                trial,
                self.base,
                {"package": "candidate-code-identity"},
            )

    def test_same_config_tokenizer_but_different_weights_refuses_match(self):
        weights = self.root / "weights"
        weights.mkdir()
        r.save(
            weights / "model.safetensors.index.json",
            {"weight_map": {"a": "first.safetensors", "b": "second.safetensors"}},
        )
        (weights / "first.safetensors").write_bytes(b"same first shard")
        second = weights / "second.safetensors"
        second.write_bytes(b"original second shard")
        original = r.weight_manifest(weights)
        self.prov["model"] = r.weight_identity(original)
        self.prov["execution"] = "hardware:synthetic-test-fixture"
        self.export["weight_manifest"] = original
        self.run_score()
        second.write_bytes(b"different second shard")
        changed = r.weight_manifest(weights)
        self.assertNotEqual(r.weight_identity(original), r.weight_identity(changed))
        self.export["weight_manifest"] = changed
        self.export["provenance"] = dict(self.prov, model=r.weight_identity(changed))
        torch.save(self.export, self.logits)
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            r.score(self.corpus, self.logits, self.root / "trial.json", self.base, {})

    def test_hardware_receipts_without_weight_content_are_rejected(self):
        self.prov["execution"] = "hardware:synthetic-test-fixture"
        with self.assertRaisesRegex(ValueError, "all-shard content"):
            self.run_score()

    def test_unmatched_provenance_rejected(self):
        self.run_score()
        self.export["provenance"] = dict(self.prov, runtime="different")
        torch.save(self.export, self.logits)
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            r.score(self.corpus, self.logits, self.root / "trial.json", self.base, {})


if __name__ == "__main__":
    unittest.main()
