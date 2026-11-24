//! Small, dependency-free RFC 8259 JSON parser for Cargo machine messages.

use std::collections::BTreeMap;
use std::fmt;

/// Maximum container nesting depth accepted by [`parse`].
const MAX_DEPTH: usize = 128;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}

impl Value {
    pub fn as_object(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Object(map) => Some(map),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub offset: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at byte {}", self.message, self.offset)
    }
}

/// Parse one complete RFC 8259 JSON value from `input`.
pub fn parse(input: &str) -> Result<Value, ParseError> {
    let mut parser = Parser {
        input,
        bytes: input.as_bytes(),
        pos: 0,
        depth: 0,
    };
    parser.skip_ws();
    let value = parser.parse_value()?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(parser.err("trailing characters after JSON value"));
    }
    Ok(value)
}

struct Parser<'a> {
    input: &'a str,
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, message: impl Into<String>) -> ParseError {
        ParseError {
            offset: self.pos,
            message: message.into(),
        }
    }

    fn err_at(&self, offset: usize, message: impl Into<String>) -> ParseError {
        ParseError {
            offset,
            message: message.into(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    /// RFC 8259 Section 2 whitespace: space, horizontal tab, LF, CR only.
    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> Result<Value, ParseError> {
        let Some(b) = self.peek() else {
            return Err(self.err("unexpected end of JSON input"));
        };
        match b {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'"' => self.parse_string().map(Value::String),
            b't' => self.parse_literal(b"true", Value::Bool(true)),
            b'f' => self.parse_literal(b"false", Value::Bool(false)),
            b'n' => self.parse_literal(b"null", Value::Null),
            b'-' | b'0'..=b'9' => self.parse_number(),
            _ => Err(self.err(format!("unexpected byte 0x{b:02x}"))),
        }
    }

    fn parse_literal(&mut self, expected: &[u8], value: Value) -> Result<Value, ParseError> {
        if self.bytes.get(self.pos..self.pos + expected.len()) == Some(expected) {
            self.pos += expected.len();
            Ok(value)
        } else {
            Err(self.err(format!(
                "expected `{}`",
                std::str::from_utf8(expected).unwrap_or("?")
            )))
        }
    }

    fn parse_object(&mut self) -> Result<Value, ParseError> {
        self.enter_container()?;
        self.pos += 1; // '{'
        self.skip_ws();
        let mut map = BTreeMap::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(Value::Object(map));
        }
        loop {
            if self.peek() != Some(b'"') {
                return Err(self.err("expected string key in object"));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(self.err("expected `:` after object key"));
            }
            self.pos += 1;
            self.skip_ws();
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                    self.skip_ws();
                }
                Some(b'}') => {
                    self.pos += 1;
                    self.depth -= 1;
                    return Ok(Value::Object(map));
                }
                _ => return Err(self.err("expected `,` or `}` in object")),
            }
        }
    }

    fn parse_array(&mut self) -> Result<Value, ParseError> {
        self.enter_container()?;
        self.pos += 1; // '['
        self.skip_ws();
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(Value::Array(items));
        }
        loop {
            let val = self.parse_value()?;
            items.push(val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                    self.skip_ws();
                }
                Some(b']') => {
                    self.pos += 1;
                    self.depth -= 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.err("expected `,` or `]` in array")),
            }
        }
    }

    fn enter_container(&mut self) -> Result<(), ParseError> {
        if self.depth >= MAX_DEPTH {
            return Err(self.err("JSON nesting depth exceeded"));
        }
        self.depth += 1;
        Ok(())
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        let start_quote = self.pos;
        if self.peek() != Some(b'"') {
            return Err(self.err("expected `\"` to start string"));
        }
        self.pos += 1;
        let mut out = String::new();
        let mut chunk_start = self.pos;
        while let Some(b) = self.peek() {
            match b {
                b'"' => {
                    out.push_str(&self.input[chunk_start..self.pos]);
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    out.push_str(&self.input[chunk_start..self.pos]);
                    let esc_pos = self.pos;
                    self.pos += 1;
                    let Some(esc) = self.peek() else {
                        return Err(self.err_at(esc_pos, "unterminated escape sequence"));
                    };
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let unit = self.parse_hex4()?;
                            if (0xD800..=0xDBFF).contains(&unit) {
                                // High surrogate: must be followed immediately by `\uXXXX` low surrogate.
                                let surr_pos = self.pos;
                                if self.bytes.get(self.pos..self.pos + 2) != Some(b"\\u") {
                                    return Err(self
                                        .err_at(esc_pos, "unpaired high surrogate in \\u escape"));
                                }
                                self.pos += 2;
                                let low = self.parse_hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&low) {
                                    return Err(self.err_at(
                                        surr_pos,
                                        "high surrogate not followed by low surrogate",
                                    ));
                                }
                                let code_point = 0x10000
                                    + (((u32::from(unit) - 0xD800) << 10)
                                        | (u32::from(low) - 0xDC00));
                                let ch = char::from_u32(code_point).ok_or_else(|| {
                                    self.err_at(esc_pos, "invalid surrogate pair code point")
                                })?;
                                out.push(ch);
                            } else if (0xDC00..=0xDFFF).contains(&unit) {
                                return Err(
                                    self.err_at(esc_pos, "unpaired low surrogate in \\u escape")
                                );
                            } else {
                                let ch = char::from_u32(u32::from(unit)).ok_or_else(|| {
                                    self.err_at(esc_pos, "invalid Unicode escape code point")
                                })?;
                                out.push(ch);
                            }
                        }
                        _ => {
                            return Err(self.err_at(
                                esc_pos,
                                format!("invalid escape character `\\{}`", esc as char),
                            ));
                        }
                    }
                    chunk_start = self.pos;
                }
                0x00..=0x1F => {
                    return Err(
                        self.err(format!("unescaped control character 0x{b:02x} in string"))
                    );
                }
                _ => {
                    // `self.input` is valid UTF-8, and multi-byte UTF-8 sequences
                    // only contain bytes >= 0x80, so advancing byte-by-byte until
                    // `"`, `\`, or control chars always lands on UTF-8 char boundaries.
                    self.pos += 1;
                }
            }
        }
        Err(self.err_at(start_quote, "unterminated JSON string"))
    }

    fn parse_hex4(&mut self) -> Result<u16, ParseError> {
        let start = self.pos;
        let Some(slice) = self.bytes.get(self.pos..self.pos + 4) else {
            return Err(self.err_at(start, "truncated \\uXXXX escape"));
        };
        let mut val: u16 = 0;
        for &b in slice {
            let digit = match b {
                b'0'..=b'9' => u16::from(b - b'0'),
                b'a'..=b'f' => u16::from(b - b'a' + 10),
                b'A'..=b'F' => u16::from(b - b'A' + 10),
                _ => return Err(self.err_at(start, "invalid hex digit in \\uXXXX escape")),
            };
            val = (val << 4) | digit;
        }
        self.pos += 4;
        Ok(val)
    }

    fn parse_number(&mut self) -> Result<Value, ParseError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.err("leading zero is not allowed in JSON number"));
                }
            }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(self.err("expected digit in JSON number")),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.err("expected digit after decimal point in JSON number"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.err("expected digit in JSON number exponent"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        Ok(Value::Number(self.input[start..self.pos].to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_all_standard_escapes_and_unicode() {
        let raw = r#"{"s":"quote:\" slash:\\ fwd:\/ bs:\b ff:\f nl:\n cr:\r tab:\t bmp:\u00E9 pair:\uD83D\uDE80 raw:café🦀"}"#;
        let val = parse(raw).unwrap();
        let s = val.as_object().unwrap().get("s").unwrap().as_str().unwrap();
        assert_eq!(
            s,
            "quote:\" slash:\\ fwd:/ bs:\u{0008} ff:\u{000C} nl:\n cr:\r tab:\t bmp:é pair:🚀 raw:café🦀"
        );
    }

    #[test]
    fn handles_arbitrary_whitespace_and_nested_values() {
        let raw =
            " \t\r\n{ \"a\" : [ 1 , -2.5e+3 , true , false , null , { \"k\" : \"v\" } ] }\r\n ";
        let val = parse(raw).unwrap();
        let obj = val.as_object().unwrap();
        assert!(matches!(obj.get("a"), Some(Value::Array(items)) if items.len() == 6));
    }

    #[test]
    fn rejects_invalid_escapes_surrogates_and_control_chars() {
        assert!(parse(r#""\x41""#).is_err());
        assert!(parse(r#""\u123""#).is_err());
        assert!(parse(r#""\u123G""#).is_err());
        assert!(parse(r#""\uD83D""#).is_err());
        assert!(parse(r#""\uD83D\u0041""#).is_err());
        assert!(parse(r#""\uDE80""#).is_err());
        assert!(parse("\"line1\nline2\"").is_err());
        assert!(parse("\"tab\there\"").is_err());
        assert!(parse("\"unterminated").is_err());
    }

    #[test]
    fn rejects_invalid_numbers_and_syntax() {
        assert!(parse("01").is_err());
        assert!(parse("-01").is_err());
        assert!(parse("+1").is_err());
        assert!(parse("1.").is_err());
        assert!(parse("1e").is_err());
        assert!(parse("1e+").is_err());
        assert!(parse("[1,]").is_err());
        assert!(parse(r#"{"a":1,}"#).is_err());
        assert!(parse(r#"{"a":1} trailing"#).is_err());
        assert!(parse("").is_err());
        assert!(parse("   ").is_err());
    }

    #[test]
    fn rejects_excessive_nesting_depth() {
        let deep = format!("{}{}", "[".repeat(MAX_DEPTH + 1), "]".repeat(MAX_DEPTH + 1));
        assert!(parse(&deep).is_err());
    }
}
