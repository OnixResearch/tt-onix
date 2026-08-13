//! Pure first-divergence comparator for the bounded window-16 diagnosis.
//!
//! r[impl tenstorrent.rwkv7_p150x2.window16_parity_diagnosis]

#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryPoint {
    pub generated_token_index: usize,
    pub layer_index: Option<usize>,
    pub value_family: String,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatchingBoundary {
    pub generated_token_index: usize,
    pub layer_index: Option<usize>,
    pub value_family: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FirstDivergence {
    pub generated_token_index: usize,
    pub layer_index: Option<usize>,
    pub value_family: String,
    pub element_index: usize,
    pub expected: f32,
    pub actual: f32,
    pub maximum_absolute_error: f32,
    pub prior_matching_boundary: Option<MatchingBoundary>,
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

fn validate_point(point: &BoundaryPoint, side: &str) -> Result<(), ComparisonError> {
    if point.value_family.is_empty() {
        return Err(ComparisonError::new(
            "empty_value_family",
            format!("{side} boundary has an empty value family"),
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
        && left.value_family == right.value_family
}

fn matching_boundary(point: &BoundaryPoint) -> MatchingBoundary {
    MatchingBoundary {
        generated_token_index: point.generated_token_index,
        layer_index: point.layer_index,
        value_family: point.value_family.clone(),
    }
}

/// Compare ordered CPU and device boundaries with exact FP32 bit equality.
///
/// The shell must order points by generated token, layer, and value family.
/// This pure core stops at the first different value and reports the prior
/// matching boundary. Length, identity, and finite-value faults reject.
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
                value_family: expected_point.value_family.clone(),
                element_index,
                expected: expected_value,
                actual: actual_value,
                maximum_absolute_error,
                prior_matching_boundary,
            }));
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

    fn token_boundary(index: usize, token: f32) -> BoundaryPoint {
        BoundaryPoint {
            generated_token_index: index,
            layer_index: None,
            value_family: "selected_token".to_owned(),
            values: vec![token],
        }
    }

    #[test]
    fn equal_boundaries_have_no_divergence() {
        let boundaries = vec![
            token_boundary(MATCHING_TOKEN_INDEX, 5_230.0),
            token_boundary(DIVERGENT_TOKEN_INDEX, EXPECTED_TOKEN),
        ];
        assert_eq!(
            first_divergence(&boundaries, &boundaries).expect("valid comparison"),
            None
        );
    }

    #[test]
    fn reports_the_known_first_token_divergence() {
        let expected = vec![
            token_boundary(MATCHING_TOKEN_INDEX, 5_230.0),
            token_boundary(DIVERGENT_TOKEN_INDEX, EXPECTED_TOKEN),
        ];
        let actual = vec![
            token_boundary(MATCHING_TOKEN_INDEX, 5_230.0),
            token_boundary(DIVERGENT_TOKEN_INDEX, ACTUAL_TOKEN),
        ];
        let divergence = first_divergence(&expected, &actual)
            .expect("valid comparison")
            .expect("known divergence");
        assert_eq!(divergence.generated_token_index, DIVERGENT_TOKEN_INDEX);
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
        let expected = vec![token_boundary(DIVERGENT_TOKEN_INDEX, EXPECTED_TOKEN)];
        let mut actual = expected.clone();
        actual[0].layer_index = Some(1);
        let error = first_divergence(&expected, &actual).expect_err("identity mismatch");
        assert_eq!(error.code, "boundary_identity_mismatch");
    }

    #[test]
    fn rejects_non_finite_values() {
        let expected = vec![token_boundary(DIVERGENT_TOKEN_INDEX, EXPECTED_TOKEN)];
        let actual = vec![token_boundary(DIVERGENT_TOKEN_INDEX, f32::NAN)];
        let error = first_divergence(&expected, &actual).expect_err("non-finite value");
        assert_eq!(error.code, "non_finite_boundary");
    }
}
