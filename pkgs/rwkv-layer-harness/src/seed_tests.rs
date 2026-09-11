//! Real tokenizer input tests. The suffix is synthetic, not a downloaded vocabulary.
use super::{RwkvTokenizer, TOKENIZER_FIRST_VOCAB_ID, TOKENIZER_LAST_VOCAB_ID};
use std::fmt::Write;

const BYTE_HEX_WIDTH: usize = 2;
const VALID_ROWS: &str = include_str!("../fixtures/fuzz-seeds/valid-vocabulary-rows.txt");
const DUPLICATE_ROWS: &str = include_str!("../fixtures/fuzz-seeds/duplicate-row.txt");

fn complete_vocabulary(prefix: &str) -> String {
    let mut vocabulary = prefix.to_owned();
    let first_suffix_id = prefix.lines().count() + TOKENIZER_FIRST_VOCAB_ID;
    for id in first_suffix_id..=TOKENIZER_LAST_VOCAB_ID {
        let bytes = u16::try_from(id)
            .expect("fixture ID must fit in two bytes")
            .to_le_bytes();
        writeln!(
            vocabulary,
            "{id} b'\\x{:0width$x}\\x{:0width$x}' {}",
            bytes[0],
            bytes[1],
            bytes.len(),
            width = BYTE_HEX_WIDTH
        )
        .unwrap();
    }
    vocabulary
}

#[test]
fn saved_vocabulary_counterpart_parses_and_round_trips() {
    let vocabulary = complete_vocabulary(VALID_ROWS);
    let tokenizer = RwkvTokenizer::parse(vocabulary.as_bytes()).unwrap();
    assert_eq!(
        tokenizer.decode_bytes(&[TOKENIZER_FIRST_VOCAB_ID]).unwrap(),
        b"a"
    );
    let text = b"abab";
    let tokens = tokenizer.encode_bytes(text).unwrap();
    assert_eq!(tokenizer.decode_bytes(&tokens).unwrap(), text);
}

#[test]
fn saved_duplicate_rows_reach_the_tokenizer_sequence_check() {
    let vocabulary = complete_vocabulary(DUPLICATE_ROWS);
    let error = RwkvTokenizer::parse(vocabulary.as_bytes()).expect_err("duplicate ID must fail");
    assert!(error.contains("must contain ID"), "{error}");

    let duplicate_token = VALID_ROWS.replace("b'b'", "b'a'");
    let vocabulary = complete_vocabulary(&duplicate_token);
    let error = RwkvTokenizer::parse(vocabulary.as_bytes()).expect_err("duplicate token must fail");
    assert!(
        error.contains("duplicates an earlier byte token"),
        "{error}"
    );
}
