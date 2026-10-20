//! RWKV-7 byte-trie tokenizer.
//!
//! The checkpoint's vocabulary is 65,529 rows of `id <python literal> <byte length>`.
//! Tokenization is greedy longest match over raw UTF-8 bytes, with no normalization
//! or pre-tokenization. Several tokens are not valid UTF-8 on their own (`b'\x80'`,
//! the middle bytes of an emoji), so this is byte-oriented, not a JSON tokenizer.

use std::path::Path;

#[derive(Default)]
struct Node {
    edges: Vec<(u8, usize)>,
    value: Option<u32>,
}

pub struct Tokenizer {
    nodes: Vec<Node>,
    tokens: Vec<Option<Vec<u8>>>,
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let source = std::fs::read_to_string(path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        Self::parse(&source)
    }

    pub fn parse(source: &str) -> Result<Self, String> {
        let mut entries = Vec::new();
        let mut max_id = 0u32;
        for (line_number, line) in source.lines().enumerate() {
            let line = line.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            let first = line
                .find(' ')
                .ok_or_else(|| format!("line {}: missing id separator", line_number + 1))?;
            let last = line
                .rfind(' ')
                .ok_or_else(|| format!("line {}: missing length separator", line_number + 1))?;
            if first == last {
                return Err(format!("line {}: missing literal", line_number + 1));
            }
            let id: u32 = line[..first]
                .parse()
                .map_err(|_| format!("line {}: invalid id", line_number + 1))?;
            let declared: usize = line[last + 1..]
                .parse()
                .map_err(|_| format!("line {}: invalid length", line_number + 1))?;
            let bytes = parse_literal(&line[first + 1..last])
                .map_err(|error| format!("line {}: {error}", line_number + 1))?;
            if bytes.len() != declared {
                return Err(format!(
                    "line {}: literal is {} bytes, declared {declared}",
                    line_number + 1,
                    bytes.len()
                ));
            }
            max_id = max_id.max(id);
            entries.push((id, bytes));
        }
        let mut tokens: Vec<Option<Vec<u8>>> = vec![None; max_id as usize + 1];
        for (id, bytes) in entries {
            tokens[id as usize] = Some(bytes);
        }
        let mut nodes = vec![Node::default()];
        for (id, bytes) in tokens.iter().enumerate() {
            let Some(bytes) = bytes else { continue };
            let mut node = 0usize;
            for &byte in bytes {
                let next = match nodes[node].edges.iter().find(|(edge, _)| *edge == byte) {
                    Some((_, next)) => *next,
                    None => {
                        let next = nodes.len();
                        nodes.push(Node::default());
                        nodes[node].edges.push((byte, next));
                        next
                    }
                };
                node = next;
            }
            nodes[node].value = Some(id as u32);
        }
        for node in &mut nodes {
            node.edges.sort_unstable_by_key(|(byte, _)| *byte);
        }
        Ok(Self { nodes, tokens })
    }

    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.tokens.get(id as usize)?.as_deref()
    }

    pub fn encode_bytes(&self, source: &[u8]) -> Result<Vec<u32>, String> {
        let mut ids = Vec::new();
        let mut position = 0usize;
        while position < source.len() {
            let start = position;
            let mut node = 0usize;
            let mut value = None;
            while position < source.len() {
                let byte = source[position];
                let Some((_, next)) = self.nodes[node]
                    .edges
                    .iter()
                    .find(|(edge, _)| *edge == byte)
                else {
                    break;
                };
                node = *next;
                position += 1;
                if let Some(id) = self.nodes[node].value {
                    value = Some((id, position));
                }
            }
            let Some((id, end)) = value else {
                return Err(format!(
                    "no vocabulary token at byte {start} (0x{:02x})",
                    source[start]
                ));
            };
            ids.push(id);
            position = end;
        }
        Ok(ids)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode_bytes(text.as_bytes())
    }

    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, String> {
        let mut output = Vec::new();
        for &id in ids {
            let bytes = self
                .token_bytes(id)
                .ok_or_else(|| format!("unknown token id {id}"))?;
            output.extend_from_slice(bytes);
        }
        Ok(output)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        String::from_utf8(self.decode_bytes(ids)?).map_err(|error| error.to_string())
    }
}

fn parse_literal(literal: &str) -> Result<Vec<u8>, String> {
    let (bytes_mode, body) = if let Some(body) = literal.strip_prefix("b'") {
        (true, body.strip_suffix('\''))
    } else if let Some(body) = literal.strip_prefix("b\"") {
        (true, body.strip_suffix('"'))
    } else if let Some(body) = literal.strip_prefix('\'') {
        (false, body.strip_suffix('\''))
    } else if let Some(body) = literal.strip_prefix('"') {
        (false, body.strip_suffix('"'))
    } else {
        (false, None)
    };
    let Some(body) = body else {
        return Err(format!("unsupported literal {literal:?}"));
    };
    let mut output = Vec::new();
    let mut chars = body.chars();
    while let Some(character) = chars.next() {
        if character != '\\' {
            let mut buffer = [0u8; 4];
            output.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
            continue;
        }
        let Some(escape) = chars.next() else {
            return Err("trailing backslash".into());
        };
        match escape {
            'n' => output.push(b'\n'),
            'r' => output.push(b'\r'),
            't' => output.push(b'\t'),
            '\\' => output.push(b'\\'),
            '\'' => output.push(b'\''),
            '"' => output.push(b'"'),
            '0' => output.push(0),
            'a' => output.push(0x07),
            'b' => output.push(0x08),
            'f' => output.push(0x0c),
            'v' => output.push(0x0b),
            'u' | 'U' => {
                if bytes_mode {
                    return Err(format!("\\{escape} in a bytes literal"));
                }
                let digits = if escape == 'u' { 4 } else { 8 };
                let mut value = 0u32;
                for _ in 0..digits {
                    let digit = chars.next().ok_or("short unicode escape")?;
                    value = value * 16
                        + digit
                            .to_digit(16)
                            .ok_or_else(|| format!("invalid unicode escape digit {digit}"))?;
                }
                let character = char::from_u32(value)
                    .ok_or_else(|| format!("invalid unicode scalar U+{value:04X}"))?;
                let mut buffer = [0u8; 4];
                output.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
            }
            'x' => {
                let hi = chars.next().ok_or("short \\x escape")?;
                let lo = chars.next().ok_or("short \\x escape")?;
                let value = u8::from_str_radix(&format!("{hi}{lo}"), 16)
                    .map_err(|_| format!("invalid \\x escape {hi}{lo}"))?;
                if bytes_mode {
                    output.push(value);
                } else {
                    let mut buffer = [0u8; 4];
                    output.extend_from_slice(char::from(value).encode_utf8(&mut buffer).as_bytes());
                }
            }
            other => return Err(format!("unsupported escape \\{other}")),
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bytes_strings_and_escapes() {
        let source = "1 'A' 1\n2 b'\\x80' 1\n3 '\\x80' 2\n4 'é' 2\n";
        let tokenizer = Tokenizer::parse(source).unwrap();
        assert_eq!(tokenizer.token_bytes(1), Some(b"A".as_slice()));
        assert_eq!(tokenizer.token_bytes(2), Some(&[0x80][..]));
        assert_eq!(tokenizer.token_bytes(3), Some(&[0xc2, 0x80][..]));
        assert_eq!(tokenizer.token_bytes(4), Some("é".as_bytes()));
    }

    #[test]
    fn greedy_longest_match_and_round_trip() {
        let source = "1 'a' 1\n2 'ab' 2\n3 b'\\x80' 1\n";
        let tokenizer = Tokenizer::parse(source).unwrap();
        assert_eq!(tokenizer.encode("abab").unwrap(), vec![2, 2]);
        assert_eq!(tokenizer.encode_bytes(b"ab\x80").unwrap(), vec![2, 3]);
        assert_eq!(tokenizer.decode_bytes(&[2, 3]).unwrap(), b"ab\x80".to_vec());
    }
}
