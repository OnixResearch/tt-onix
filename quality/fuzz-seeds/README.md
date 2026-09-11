# Saved parser seeds

The normal quality gate runs deterministic byte-level tests in each parser's owner crate. It does not infer errors from seed names.

| Family | Seed location | Parser under test |
| --- | --- | --- |
| JSON | `quality/fuzz-seeds/json/` | `repository_quality::json_input::parse_json` |
| YAML | `pkgs/rwkv7-p150x2-evidence/fixtures/fuzz-seeds/` | Inspector `yaml_sequence` |
| CSV | `pkgs/rwkv7-p150x2-evidence/fixtures/fuzz-seeds/` | `parse_profiler` |
| Binary frame | `pkgs/rwkv-layer-harness/fixtures/fuzz-seeds/` | Dispatch `decode_request` |
| Vocabulary | `pkgs/rwkv-layer-harness/fixtures/fuzz-seeds/` | `RwkvTokenizer::parse` |
| Receipt | `pkgs/rwkv-ttwkv7-persistent-receipt-validator/fixtures/fuzz-seeds/` | `validate_persistent_receipt` |

Each family has an accepted counterpart and a rejected input. Tests assert the error from the actual parser. The vocabulary tests append a deterministic synthetic suffix to reach the required row count. They do not download a model vocabulary. The frame test pairs a saved truncated prefix with an encoded valid request.

The old five-column CSV seed was not over the production field limit. It is now `no-data.csv`, which matches its actual rejection. The old hexadecimal frame has a wrong magic value. It is now `bad-magic.hex`. A separate saved prefix covers truncation. YAML and vocabulary seeds now use the owning protocol's shape. The receipt pair differs only in the invalid numeric value.

The central `ParserObservation` tests cover policy decisions for supplied counts and flags. They do not count as byte-level parser tests.

These checks do not prove hardware behavior, model correctness, or performance.
