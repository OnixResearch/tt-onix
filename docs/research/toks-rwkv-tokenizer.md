# toks and the RWKV-7 tokenizer

Evaluated `actual-computer/toks` 0.3.2 (Apache-2.0) for the pinned
`RWKV/RWKV7-Goose-World2.8-0.1B-HF` tokenizer.

## Result

toks cannot load the RWKV-7 tokenizer. It is not an exactness failure after load; the file format
it accepts cannot describe this tokenizer.

Probe with the C ABI (`toks_load`):

| Input | Result |
|---|---|
| `/var/tmp/rwkv-model` (RWKV files, no `tokenizer.json`) | `-1`: `model directory without tokenizer.json, tiktoken.model or qwen.tiktoken` |
| `rwkv_vocab_v20230424.txt` | `-1`: `tiktoken.model needs tokenizer_config.json and tokenization_kimi.py beside it` |
| GPT-2 `tokenizer.json` (control) | `0` |

toks loads a standard HF `tokenizer.json` or a tiktoken model. The RWKV checkpoint ships neither:
it uses a custom slow `RwkvTokenizer` with `rwkv_vocab_v20230424.txt`.

## Why no tokenizer.json can represent it exactly

The RWKV tokenizer is greedy longest match over raw UTF-8 bytes, with no normalization,
pre-tokenization, BPE merges, or Unigram scores. The pinned vocabulary has 65,529 rows; 486 tokens
are not valid UTF-8 on their own. Valid text emits them:

| Text | IDs | Invalid token bytes |
|---|---|---|
| `😀` | `[3319, 153, 129]` | `f09f`, `98`, `80` |
| `U+10FFFF` | `[245, 144, 192, 192]` | `f4`, `8f`, `bf`, `bf` |

A `tokenizer.json` added token is a Unicode string; it cannot carry the raw byte `0x80` or the
two-byte prefix `f09f`. Byte-level BPE spellings are matched after normalization/pretokenization,
and the model algorithms (BPE, Unigram, WordPiece) do not reproduce leftmost-longest raw-byte
matching. toks's own exactness rule therefore refuses this tokenizer family.

## Implemented instead

`crates/rwkv7/src/tokenizer.rs` parses the vocabulary and implements the byte trie directly, in
Rust, without a tokenizer library. `rwkv7-tokenizer-gate` pins Python-derived vectors:

| Text | IDs |
|---|---|
| `Hello world` | `[33155, 40213]` |
| `User: Hi\n\nAssistant:` | `[24281, 59, 3880, 261, 5585, 41693, 59]` |
| `你好，世界` | `[10464, 11685, 19137, 10267, 14610]` |
| `😀` | `[3319, 153, 129]` |
| `U+10FFFF` | `[245, 144, 192, 192]` |

All round-trip to the original UTF-8 bytes. toks remains a candidate for standard HF tokenizers
elsewhere in the repository, but not for RWKV-7.
