//! Tokenizer vectors for the pinned RWKV-7 vocabulary.
//!
//! The expected ids come from the harness's Python `RWKV_TOKENIZER` over the same
//! vocabulary. The byte trie must match greedy longest match, including tokens
//! that are not individually valid UTF-8.

use std::path::Path;

use rwkv7::tokenizer::Tokenizer;

const VECTORS: &[(&str, &[u32])] = &[
    ("Hello world", &[33155, 40213]),
    (
        "User: Hi\n\nAssistant:",
        &[24281, 59, 3880, 261, 5585, 41693, 59],
    ),
    ("你好，世界", &[10464, 11685, 19137, 10267, 14610]),
    ("😀", &[3319, 153, 129]),
    ("\u{10ffff}", &[245, 144, 192, 192]),
];

fn main() -> std::process::ExitCode {
    let mut vocab = None;
    let mut text = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--vocab" => {
                let Some(value) = args.next() else {
                    eprintln!("--vocab needs a vocabulary path");
                    return std::process::ExitCode::FAILURE;
                };
                vocab = Some(value);
            }
            "--text" => {
                let Some(value) = args.next() else {
                    eprintln!("--text needs a string");
                    return std::process::ExitCode::FAILURE;
                };
                text = Some(value);
            }
            "-h" | "--help" => {
                println!("usage: rwkv7-tokenizer-gate --vocab PATH [--text STRING]");
                return std::process::ExitCode::SUCCESS;
            }
            other => {
                eprintln!(
                    "unknown argument {other}\n\nusage: rwkv7-tokenizer-gate --vocab PATH [--text STRING]"
                );
                return std::process::ExitCode::FAILURE;
            }
        }
    }
    let Some(vocab) = vocab else {
        eprintln!("--vocab PATH is required");
        return std::process::ExitCode::FAILURE;
    };
    let tokenizer = match Tokenizer::from_file(Path::new(&vocab)) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            eprintln!("vocab: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    if let Some(text) = text {
        match tokenizer.encode(&text) {
            Ok(ids) => {
                println!("{ids:?}");
                match tokenizer.decode_bytes(&ids) {
                    Ok(bytes) if bytes == text.as_bytes() => {}
                    Ok(_) => {
                        eprintln!("round trip changed the bytes");
                        return std::process::ExitCode::FAILURE;
                    }
                    Err(error) => {
                        eprintln!("decode: {error}");
                        return std::process::ExitCode::FAILURE;
                    }
                }
            }
            Err(error) => {
                eprintln!("encode: {error}");
                return std::process::ExitCode::FAILURE;
            }
        }
        return std::process::ExitCode::SUCCESS;
    }

    for (sample, expected) in VECTORS {
        let ids = match tokenizer.encode(sample) {
            Ok(ids) => ids,
            Err(error) => {
                eprintln!("{sample:?}: {error}");
                return std::process::ExitCode::FAILURE;
            }
        };
        if ids != *expected {
            eprintln!("{sample:?}: got {ids:?}, expected {expected:?}");
            return std::process::ExitCode::FAILURE;
        }
        if tokenizer.decode_bytes(&ids).unwrap() != sample.as_bytes() {
            eprintln!("{sample:?}: round trip changed the bytes");
            return std::process::ExitCode::FAILURE;
        }
        println!("{sample:?}: {} ids", ids.len());
    }
    println!("tokenizer gate PASS");
    std::process::ExitCode::SUCCESS
}
