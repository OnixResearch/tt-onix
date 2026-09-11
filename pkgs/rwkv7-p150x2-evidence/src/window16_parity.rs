//! Pure boundary and first-divergence contracts for the bounded window-16 diagnosis.
//!
//! r[impl tenstorrent.rwkv7_p150x2.window16_parity_diagnosis]
//! r[impl tenstorrent.rwkv7_p150x2.window16_full_model_diagnosis]
//! r[impl tenstorrent.rwkv7_p150x2.fp32_recurrent_matrix_diagnosis]

const BLAKE3_HEX_LENGTH: usize = 64;
const MAXIMUM_GENERATED_TOKEN_INDEX: usize = 10;
const MAXIMUM_MODEL_STEP_INDEX: usize = 22;
const MAXIMUM_LAYER_INDEX: usize = 11;
const RECURRENT_VECTOR_ELEMENT_COUNT: usize = 12 * 64;
const RECURRENT_MATRIX_ELEMENT_COUNT: usize = 12 * 64 * 64;

pub const CPU_FP32_RECURRENT_MATRIX_BLAKE3: &str =
    "0d904c3766a7cb13aaaeb13266e65623d516056ab6cfa2d697ccb3610dc593d7";

#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryPoint {
    pub generated_token_index: usize,
    pub layer_index: Option<usize>,
    pub operation: String,
    pub value_family: String,
    pub dtype: String,
    pub blake3: String,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatchingBoundary {
    pub generated_token_index: usize,
    pub layer_index: Option<usize>,
    pub operation: String,
    pub value_family: String,
    pub dtype: String,
    pub blake3: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FirstDivergence {
    pub generated_token_index: usize,
    pub layer_index: Option<usize>,
    pub operation: String,
    pub value_family: String,
    pub dtype: String,
    pub expected_blake3: String,
    pub actual_blake3: String,
    pub element_index: usize,
    pub expected: f32,
    pub actual: f32,
    pub maximum_absolute_error: f32,
    pub prior_matching_boundary: Option<MatchingBoundary>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecurrentBoundaryPoint {
    pub model_step_index: usize,
    pub layer_index: usize,
    pub operation: String,
    pub dtype: String,
    pub blake3: String,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecurrentFirstDivergence {
    pub model_step_index: usize,
    pub layer_index: usize,
    pub operation: String,
    pub dtype: String,
    pub element_count: usize,
    pub expected_blake3: String,
    pub actual_blake3: String,
    pub element_index: usize,
    pub expected: f32,
    pub actual: f32,
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComparisonError {
    pub code: &'static str,
    pub detail: String,
}

impl ComparisonError {
    fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for ComparisonError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for ComparisonError {}

fn valid_blake3(value: &str) -> bool {
    value.len() == BLAKE3_HEX_LENGTH
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_point(point: &BoundaryPoint, side: &str) -> Result<(), ComparisonError> {
    if point.generated_token_index > MAXIMUM_GENERATED_TOKEN_INDEX {
        return Err(ComparisonError::new(
            "token_bound_exceeded",
            format!("{side} boundary exceeds generated-token index 10"),
        ));
    }
    if point
        .layer_index
        .is_some_and(|layer| layer > MAXIMUM_LAYER_INDEX)
    {
        return Err(ComparisonError::new(
            "layer_bound_exceeded",
            format!("{side} boundary exceeds layer index 11"),
        ));
    }
    if point.operation.is_empty() {
        return Err(ComparisonError::new(
            "empty_operation",
            format!("{side} boundary has an empty operation"),
        ));
    }
    if point.value_family.is_empty() {
        return Err(ComparisonError::new(
            "empty_value_family",
            format!("{side} boundary has an empty value family"),
        ));
    }
    if point.dtype != "bf16" && point.dtype != "fp32" && point.dtype != "u32" {
        return Err(ComparisonError::new(
            "unsupported_dtype",
            format!("{side} boundary has an unsupported dtype"),
        ));
    }
    if !valid_blake3(&point.blake3) {
        return Err(ComparisonError::new(
            "invalid_blake3",
            format!("{side} boundary has an invalid BLAKE3"),
        ));
    }
    if point.values.is_empty() {
        return Err(ComparisonError::new(
            "empty_boundary",
            format!("{side} {} boundary is empty", point.value_family),
        ));
    }
    if point.values.iter().any(|value| !value.is_finite()) {
        return Err(ComparisonError::new(
            "non_finite_boundary",
            format!("{side} {} boundary is not finite", point.value_family),
        ));
    }
    Ok(())
}

fn matching_identity(left: &BoundaryPoint, right: &BoundaryPoint) -> bool {
    left.generated_token_index == right.generated_token_index
        && left.layer_index == right.layer_index
        && left.operation == right.operation
        && left.value_family == right.value_family
        && left.dtype == right.dtype
}

fn recurrent_operation_element_count(operation: &str) -> Option<usize> {
    match operation {
        "key_rows" | "value_rows" | "state_dot_a" => Some(RECURRENT_VECTOR_ELEMENT_COUNT),
        "initial_state" | "rank_update" | "decay" | "outer_update" | "first_add" | "next_state" => {
            Some(RECURRENT_MATRIX_ELEMENT_COUNT)
        }
        _ => None,
    }
}

fn validate_recurrent_point(
    point: &RecurrentBoundaryPoint,
    side: &str,
) -> Result<(), ComparisonError> {
    if point.model_step_index > MAXIMUM_MODEL_STEP_INDEX {
        return Err(ComparisonError::new(
            "model_step_bound_exceeded",
            format!("{side} recurrent boundary exceeds model-step index 22"),
        ));
    }
    if point.layer_index > MAXIMUM_LAYER_INDEX {
        return Err(ComparisonError::new(
            "layer_bound_exceeded",
            format!("{side} recurrent boundary exceeds layer index 11"),
        ));
    }
    let Some(expected_element_count) = recurrent_operation_element_count(&point.operation) else {
        return Err(ComparisonError::new(
            "unsupported_recurrent_operation",
            format!("{side} recurrent boundary has an unsupported operation"),
        ));
    };
    if point.dtype != "fp32" {
        return Err(ComparisonError::new(
            "unsupported_recurrent_dtype",
            format!("{side} recurrent boundary is not FP32"),
        ));
    }
    if !valid_blake3(&point.blake3) {
        return Err(ComparisonError::new(
            "invalid_blake3",
            format!("{side} recurrent boundary has an invalid BLAKE3"),
        ));
    }
    if point.values.len() != expected_element_count {
        return Err(ComparisonError::new(
            "recurrent_boundary_geometry_mismatch",
            format!(
                "{side} recurrent boundary has {} values, expected {expected_element_count}",
                point.values.len()
            ),
        ));
    }
    if point.values.iter().any(|value| !value.is_finite()) {
        return Err(ComparisonError::new(
            "non_finite_boundary",
            format!("{side} recurrent boundary is not finite"),
        ));
    }
    Ok(())
}

fn matching_recurrent_identity(
    left: &RecurrentBoundaryPoint,
    right: &RecurrentBoundaryPoint,
) -> bool {
    left.model_step_index == right.model_step_index
        && left.layer_index == right.layer_index
        && left.operation == right.operation
        && left.dtype == right.dtype
}

/// Compare ordered CPU and device FP32 recurrence-operation boundaries.
///
/// The shell must normalize each matrix to `[head, value, key]` order. This
/// pure core requires complete per-layer matrices and exact FP32 bit equality.
pub fn first_recurrent_divergence(
    expected: &[RecurrentBoundaryPoint],
    actual: &[RecurrentBoundaryPoint],
) -> Result<Option<RecurrentFirstDivergence>, ComparisonError> {
    if expected.is_empty() || actual.is_empty() {
        return Err(ComparisonError::new(
            "empty_recurrent_comparison",
            "expected and actual recurrent boundary lists must not be empty",
        ));
    }
    if expected.len() != actual.len() {
        return Err(ComparisonError::new(
            "recurrent_boundary_count_mismatch",
            format!(
                "expected {} recurrent boundaries, actual has {}",
                expected.len(),
                actual.len()
            ),
        ));
    }

    for (boundary_index, (expected_point, actual_point)) in expected.iter().zip(actual).enumerate()
    {
        validate_recurrent_point(expected_point, "expected")?;
        validate_recurrent_point(actual_point, "actual")?;
        if !matching_recurrent_identity(expected_point, actual_point) {
            return Err(ComparisonError::new(
                "recurrent_boundary_identity_mismatch",
                format!("recurrent boundary {boundary_index} identifies different values"),
            ));
        }

        let mut first_element = None;
        let mut maximum_absolute_error = 0.0_f32;
        for (element_index, (expected_value, actual_value)) in expected_point
            .values
            .iter()
            .zip(&actual_point.values)
            .enumerate()
        {
            let absolute_error = (*expected_value - *actual_value).abs();
            if !absolute_error.is_finite() {
                return Err(ComparisonError::new(
                    "non_finite_recurrent_error",
                    format!("recurrent boundary {boundary_index} produced a non-finite error"),
                ));
            }
            maximum_absolute_error = maximum_absolute_error.max(absolute_error);
            if first_element.is_none() && expected_value.to_bits() != actual_value.to_bits() {
                first_element = Some((element_index, *expected_value, *actual_value));
            }
        }
        if let Some((element_index, expected_value, actual_value)) = first_element {
            return Ok(Some(RecurrentFirstDivergence {
                model_step_index: expected_point.model_step_index,
                layer_index: expected_point.layer_index,
                operation: expected_point.operation.clone(),
                dtype: expected_point.dtype.clone(),
                element_count: expected_point.values.len(),
                expected_blake3: expected_point.blake3.clone(),
                actual_blake3: actual_point.blake3.clone(),
                element_index,
                expected: expected_value,
                actual: actual_value,
                maximum_absolute_error,
            }));
        }
        if expected_point.blake3 != actual_point.blake3 {
            return Err(ComparisonError::new(
                "recurrent_hash_value_inconsistency",
                format!(
                    "recurrent boundary {boundary_index} has equal values but different hashes"
                ),
            ));
        }
    }
    Ok(None)
}

pub fn exact_fp32_recurrent_matrix_parity(device_blake3: &str) -> Result<(), ComparisonError> {
    if !valid_blake3(device_blake3) {
        return Err(ComparisonError::new(
            "invalid_device_state_blake3",
            "device FP32 recurrent-matrix BLAKE3 is invalid",
        ));
    }
    if device_blake3 != CPU_FP32_RECURRENT_MATRIX_BLAKE3 {
        return Err(ComparisonError::new(
            "fp32_recurrent_matrix_mismatch",
            "device FP32 recurrent-matrix BLAKE3 differs from the pinned CPU authority",
        ));
    }
    Ok(())
}

fn matching_boundary(point: &BoundaryPoint) -> MatchingBoundary {
    MatchingBoundary {
        generated_token_index: point.generated_token_index,
        layer_index: point.layer_index,
        operation: point.operation.clone(),
        value_family: point.value_family.clone(),
        dtype: point.dtype.clone(),
        blake3: point.blake3.clone(),
    }
}

/// Compare ordered CPU and device boundaries with exact FP32 bit equality.
///
/// The shell must order points by generated token, layer, operation, and value
/// family. This pure core stops at the first different value. It also checks
/// that an exact value match has the same deterministic BLAKE3.
pub fn first_divergence(
    expected: &[BoundaryPoint],
    actual: &[BoundaryPoint],
) -> Result<Option<FirstDivergence>, ComparisonError> {
    if expected.is_empty() || actual.is_empty() {
        return Err(ComparisonError::new(
            "empty_comparison",
            "expected and actual boundary lists must not be empty",
        ));
    }
    if expected.len() != actual.len() {
        return Err(ComparisonError::new(
            "boundary_count_mismatch",
            format!(
                "expected {} boundaries, actual has {}",
                expected.len(),
                actual.len()
            ),
        ));
    }

    let mut prior_matching_boundary = None;
    for (boundary_index, (expected_point, actual_point)) in expected.iter().zip(actual).enumerate()
    {
        validate_point(expected_point, "expected")?;
        validate_point(actual_point, "actual")?;
        if !matching_identity(expected_point, actual_point) {
            return Err(ComparisonError::new(
                "boundary_identity_mismatch",
                format!("boundary {boundary_index} identifies different values"),
            ));
        }
        if expected_point.values.len() != actual_point.values.len() {
            return Err(ComparisonError::new(
                "boundary_width_mismatch",
                format!(
                    "boundary {boundary_index} expected {} values, actual has {}",
                    expected_point.values.len(),
                    actual_point.values.len()
                ),
            ));
        }

        let mut first_element = None;
        let mut maximum_absolute_error = 0.0_f32;
        for (element_index, (expected_value, actual_value)) in expected_point
            .values
            .iter()
            .zip(&actual_point.values)
            .enumerate()
        {
            let error = (*expected_value - *actual_value).abs();
            maximum_absolute_error = maximum_absolute_error.max(error);
            if first_element.is_none() && expected_value.to_bits() != actual_value.to_bits() {
                first_element = Some((element_index, *expected_value, *actual_value));
            }
        }
        if let Some((element_index, expected_value, actual_value)) = first_element {
            return Ok(Some(FirstDivergence {
                generated_token_index: expected_point.generated_token_index,
                layer_index: expected_point.layer_index,
                operation: expected_point.operation.clone(),
                value_family: expected_point.value_family.clone(),
                dtype: expected_point.dtype.clone(),
                expected_blake3: expected_point.blake3.clone(),
                actual_blake3: actual_point.blake3.clone(),
                element_index,
                expected: expected_value,
                actual: actual_value,
                maximum_absolute_error,
                prior_matching_boundary,
            }));
        }
        if expected_point.blake3 != actual_point.blake3 {
            return Err(ComparisonError::new(
                "hash_value_inconsistency",
                format!("boundary {boundary_index} has equal values but different hashes"),
            ));
        }
        prior_matching_boundary = Some(matching_boundary(expected_point));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MATCHING_TOKEN_INDEX: usize = 9;
    const DIVERGENT_TOKEN_INDEX: usize = 10;
    const EXPECTED_TOKEN: f32 = 46_170.0;
    const ACTUAL_TOKEN: f32 = 64_390.0;
    const VALID_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";
    const OTHER_HASH: &str = "1111111111111111111111111111111111111111111111111111111111111111";

    fn token_boundary(index: usize, token: f32, hash: &str) -> BoundaryPoint {
        BoundaryPoint {
            generated_token_index: index,
            layer_index: None,
            operation: "argmax".to_owned(),
            value_family: "selected_token".to_owned(),
            dtype: "u32".to_owned(),
            blake3: hash.to_owned(),
            values: vec![token],
        }
    }

    fn recurrent_boundary(operation: &str, value: f32, hash: &str) -> RecurrentBoundaryPoint {
        RecurrentBoundaryPoint {
            model_step_index: 0,
            layer_index: 0,
            operation: operation.to_owned(),
            dtype: "fp32".to_owned(),
            blake3: hash.to_owned(),
            values: vec![
                value;
                recurrent_operation_element_count(operation).expect("known operation")
            ],
        }
    }

    #[test]
    fn accepts_exact_recurrent_operation_boundaries() {
        let boundaries = vec![
            recurrent_boundary("initial_state", 0.0, VALID_HASH),
            recurrent_boundary("outer_update", 1.0, OTHER_HASH),
        ];
        assert_eq!(
            first_recurrent_divergence(&boundaries, &boundaries)
                .expect("valid recurrent comparison"),
            None
        );
    }

    #[test]
    fn reports_the_first_recurrent_operation_divergence() {
        let expected = vec![
            recurrent_boundary("initial_state", 0.0, VALID_HASH),
            recurrent_boundary("outer_update", 1.0, VALID_HASH),
        ];
        let actual = vec![
            recurrent_boundary("initial_state", 0.0, VALID_HASH),
            recurrent_boundary("outer_update", 1.5, OTHER_HASH),
        ];
        let divergence = first_recurrent_divergence(&expected, &actual)
            .expect("valid recurrent comparison")
            .expect("known recurrent divergence");
        assert_eq!(divergence.model_step_index, 0);
        assert_eq!(divergence.layer_index, 0);
        assert_eq!(divergence.operation, "outer_update");
        assert_eq!(divergence.element_index, 0);
        assert_eq!(divergence.maximum_absolute_error, 0.5);
        assert_eq!(divergence.element_count, RECURRENT_MATRIX_ELEMENT_COUNT);
    }

    #[test]
    fn rejects_invalid_recurrent_boundary_geometry_and_identity() {
        let expected = vec![recurrent_boundary("outer_update", 1.0, VALID_HASH)];
        let mut invalid_geometry = expected.clone();
        invalid_geometry[0].values.pop();
        let error = first_recurrent_divergence(&expected, &invalid_geometry)
            .expect_err("invalid recurrent geometry");
        assert_eq!(error.code, "recurrent_boundary_geometry_mismatch");

        let mut invalid_identity = expected.clone();
        invalid_identity[0].model_step_index = 1;
        let error = first_recurrent_divergence(&expected, &invalid_identity)
            .expect_err("invalid recurrent identity");
        assert_eq!(error.code, "recurrent_boundary_identity_mismatch");
    }

    #[test]
    fn rejects_recurrent_error_overflow() {
        let expected = vec![recurrent_boundary("outer_update", f32::MAX, VALID_HASH)];
        let actual = vec![recurrent_boundary("outer_update", -f32::MAX, OTHER_HASH)];
        let error = first_recurrent_divergence(&expected, &actual)
            .expect_err("recurrent error overflow must fail");
        assert_eq!(error.code, "non_finite_recurrent_error");
    }

    #[test]
    fn accepts_the_pinned_cpu_recurrent_matrix_identity() {
        exact_fp32_recurrent_matrix_parity(CPU_FP32_RECURRENT_MATRIX_BLAKE3)
            .expect("pinned CPU matrix identity");
    }

    #[test]
    fn rejects_invalid_or_different_recurrent_matrix_identities() {
        let invalid =
            exact_fp32_recurrent_matrix_parity("not-a-hash").expect_err("invalid matrix identity");
        assert_eq!(invalid.code, "invalid_device_state_blake3");

        let mismatch =
            exact_fp32_recurrent_matrix_parity(OTHER_HASH).expect_err("different matrix identity");
        assert_eq!(mismatch.code, "fp32_recurrent_matrix_mismatch");
    }

    #[test]
    fn equal_boundaries_have_no_divergence() {
        let boundaries = vec![
            token_boundary(MATCHING_TOKEN_INDEX, 5_230.0, VALID_HASH),
            token_boundary(DIVERGENT_TOKEN_INDEX, EXPECTED_TOKEN, OTHER_HASH),
        ];
        assert_eq!(
            first_divergence(&boundaries, &boundaries).expect("valid comparison"),
            None
        );
    }

    #[test]
    fn reports_the_known_first_token_divergence() {
        let expected = vec![
            token_boundary(MATCHING_TOKEN_INDEX, 5_230.0, VALID_HASH),
            token_boundary(DIVERGENT_TOKEN_INDEX, EXPECTED_TOKEN, VALID_HASH),
        ];
        let actual = vec![
            token_boundary(MATCHING_TOKEN_INDEX, 5_230.0, VALID_HASH),
            token_boundary(DIVERGENT_TOKEN_INDEX, ACTUAL_TOKEN, OTHER_HASH),
        ];
        let divergence = first_divergence(&expected, &actual)
            .expect("valid comparison")
            .expect("known divergence");
        assert_eq!(divergence.generated_token_index, DIVERGENT_TOKEN_INDEX);
        assert_eq!(divergence.operation, "argmax");
        assert_eq!(divergence.value_family, "selected_token");
        assert_eq!(divergence.expected, EXPECTED_TOKEN);
        assert_eq!(divergence.actual, ACTUAL_TOKEN);
        assert_eq!(
            divergence
                .prior_matching_boundary
                .expect("matching prefix")
                .generated_token_index,
            MATCHING_TOKEN_INDEX
        );
    }

    #[test]
    fn rejects_different_boundary_identities() {
        let expected = vec![token_boundary(
            DIVERGENT_TOKEN_INDEX,
            EXPECTED_TOKEN,
            VALID_HASH,
        )];
        let mut actual = expected.clone();
        actual[0].operation = "language_model_head".to_owned();
        let error = first_divergence(&expected, &actual).expect_err("identity mismatch");
        assert_eq!(error.code, "boundary_identity_mismatch");
    }

    #[test]
    fn rejects_non_finite_values_and_invalid_hashes() {
        let expected = vec![token_boundary(
            DIVERGENT_TOKEN_INDEX,
            EXPECTED_TOKEN,
            VALID_HASH,
        )];
        let actual = vec![token_boundary(DIVERGENT_TOKEN_INDEX, f32::NAN, VALID_HASH)];
        let error = first_divergence(&expected, &actual).expect_err("non-finite value");
        assert_eq!(error.code, "non_finite_boundary");

        let mut invalid_hash = expected.clone();
        invalid_hash[0].blake3 = "not-a-hash".to_owned();
        let error = first_divergence(&invalid_hash, &expected).expect_err("invalid hash");
        assert_eq!(error.code, "invalid_blake3");
    }

    #[test]
    fn rejects_hashes_that_disagree_with_equal_values() {
        let expected = vec![token_boundary(
            DIVERGENT_TOKEN_INDEX,
            EXPECTED_TOKEN,
            VALID_HASH,
        )];
        let actual = vec![token_boundary(
            DIVERGENT_TOKEN_INDEX,
            EXPECTED_TOKEN,
            OTHER_HASH,
        )];
        let error = first_divergence(&expected, &actual).expect_err("hash mismatch");
        assert_eq!(error.code, "hash_value_inconsistency");
    }
}
