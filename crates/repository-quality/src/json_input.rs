//! Bounded JSON decoding at the input boundary. No key can replace an earlier key.
use crate::ParserLimit;
use serde::de::{DeserializeSeed, Error, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};
use std::fmt;

#[derive(Default)]
struct Budget {
    items: usize,
    fields: usize,
}

struct Input<'a> {
    limit: &'a ParserLimit,
    budget: &'a mut Budget,
    depth: usize,
}

pub fn parse_json(bytes: &[u8], limit: &ParserLimit) -> Result<Value, String> {
    if bytes.is_empty() || bytes.len() > limit.max_bytes {
        return Err(format!(
            "JSON input must contain 1 through {} bytes",
            limit.max_bytes
        ));
    }
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = Input {
        limit,
        budget: &mut Budget::default(),
        depth: 1,
    }
    .deserialize(&mut deserializer)
    .map_err(|error| format!("invalid JSON input: {error}"))?;
    deserializer
        .end()
        .map_err(|error| format!("invalid JSON suffix: {error}"))?;
    Ok(value)
}

impl<'de> DeserializeSeed<'de> for Input<'_> {
    type Value = Value;

    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        if self.depth > self.limit.max_depth {
            return Err(D::Error::custom("JSON depth limit exceeded"));
        }
        if self.budget.items >= self.limit.max_items {
            return Err(D::Error::custom("JSON item limit exceeded"));
        }
        self.budget.items += 1;
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Input<'_> {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("bounded JSON without duplicate keys")
    }

    fn visit_bool<E: Error>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E: Error>(self, value: i64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_u64<E: Error>(self, value: u64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_f64<E: Error>(self, value: f64) -> Result<Value, E> {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E: Error>(self, value: &str) -> Result<Value, E> {
        Ok(Value::String(value.to_owned()))
    }

    fn visit_unit<E: Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(Input {
            limit: self.limit,
            budget: self.budget,
            depth: self.depth + 1,
        })? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut values = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(A::Error::custom("duplicate JSON key"));
            }
            if self.budget.fields >= self.limit.max_fields {
                return Err(A::Error::custom("JSON field limit exceeded"));
            }
            self.budget.fields += 1;
            let value = map.next_value_seed(Input {
                limit: self.limit,
                budget: self.budget,
                depth: self.depth + 1,
            })?;
            assert!(values.insert(key, value).is_none());
        }
        Ok(Value::Object(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_policy;

    const OBJECT_WITH_LEAF_DEPTH: usize = 2;
    const OBJECT_WITH_LEAF_ITEMS: usize = 2;

    fn limit() -> ParserLimit {
        load_policy()
            .unwrap()
            .parser_limits
            .into_iter()
            .find(|limit| limit.family == "json")
            .unwrap()
    }

    #[test]
    fn accepts_values_and_round_trips_without_loss() {
        let input = br#"{"identity":"first","values":[null,true,-1,1,1.5]}"#;
        let parsed = parse_json(input, &limit()).unwrap();
        assert_eq!(parsed["identity"], "first");
        let serialized = serde_json::to_vec(&parsed).unwrap();
        assert_eq!(parse_json(&serialized, &limit()).unwrap(), parsed);
    }

    #[test]
    fn rejects_saved_duplicate_and_accepts_repaired_bytes() {
        let seed = include_bytes!("../../../quality/fuzz-seeds/json/duplicate-key.json");
        assert!(
            parse_json(seed, &limit())
                .unwrap_err()
                .contains("duplicate JSON key")
        );
        assert_eq!(
            parse_json(br#"{"identity":"first"}"#, &limit()).unwrap()["identity"],
            "first"
        );
    }

    #[test]
    fn rejects_nested_escaped_duplicate_and_invalid_documents() {
        for input in [
            br#"{"nested":{"valid":false,"valid":true}}"#.as_slice(),
            br#"{"valid":false,"\u0076alid":true}"#,
            br#"{"value":NaN}"#,
            br#"{"value":1e999}"#,
            b"{}{}",
            b"{",
            b"",
            &[u8::MAX],
        ] {
            assert!(parse_json(input, &limit()).is_err(), "accepted {input:?}");
        }
    }

    #[test]
    fn enforces_each_budget_at_the_boundary() {
        let input = br#"{"value":true}"#;
        let mut boundary = limit();
        boundary.max_bytes = input.len();
        boundary.max_depth = OBJECT_WITH_LEAF_DEPTH;
        boundary.max_items = OBJECT_WITH_LEAF_ITEMS;
        boundary.max_fields = 1;
        assert!(parse_json(input, &boundary).is_ok());
        for dimension in ["bytes", "depth", "items", "fields"] {
            let mut rejected = boundary.clone();
            match dimension {
                "bytes" => rejected.max_bytes -= 1,
                "depth" => rejected.max_depth -= 1,
                "items" => rejected.max_items -= 1,
                "fields" => rejected.max_fields -= 1,
                _ => unreachable!(),
            }
            assert!(
                parse_json(input, &rejected).is_err(),
                "accepted excess {dimension}"
            );
        }
    }
}
