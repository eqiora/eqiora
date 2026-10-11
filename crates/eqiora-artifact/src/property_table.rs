//! Bounded artifact admission of verified exact real tables.
use crate::{ArtifactDigest, ResolvedArrayDecoderLimits, ResolvedArrayV1, invalid_artifact};
use eqiora_core::{Diagnostic, DimExponents};
use eqiora_schema::{
    kernel::{property_table::MAX_TABLE_POINTS, pure_operator::ExactRational},
    property_table::{AcceptedRealTable, RealTableProfile},
};

/// Decode and authenticate an exact closure member under artifact budgets.
///
/// # Errors
/// Returns `EQ0901` for missing/substituted content, malformed or excessive
/// data, wrong shape/scalar, unordered axes, or unsupported exact arithmetic.
pub fn decode_real_table(
    expected: &ArtifactDigest,
    content: Option<&[u8]>,
    mut limits: ResolvedArrayDecoderLimits,
    axis_dimension: DimExponents,
    value_dimension: DimExponents,
    profile: RealTableProfile,
    validity: [ExactRational; 2],
) -> Result<AcceptedRealTable, Diagnostic> {
    let content =
        content.ok_or_else(|| invalid_artifact("property table closure member is missing"))?;
    limits.array.max_rank = limits.array.max_rank.min(2);
    limits.array.max_values = limits.array.max_values.min(2 * MAX_TABLE_POINTS);
    let array = ResolvedArrayV1::from_json(content, limits)?;
    let digest = array.digest()?;
    if &digest != expected {
        return Err(invalid_artifact(
            "property table resolved-array digest differs from the release reference",
        ));
    }
    let array = array
        .resolved_f64()
        .ok_or_else(|| invalid_artifact("property table requires finite binary64 scalar data"))?
        .clone();
    let table = eqiora_schema::property_table::AcceptedRealTable::from_array(
        array,
        axis_dimension,
        value_dimension,
        profile,
        validity,
    )?;
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn interval() -> [ExactRational; 2] {
        [ExactRational::integer(0), ExactRational::integer(2)]
    }
    const D: DimExponents = DimExponents::DIMENSIONLESS;
    fn accept(array: &ResolvedArrayV1) -> Result<AcceptedRealTable, Diagnostic> {
        decode_real_table(
            &array.digest().unwrap(),
            Some(&array.canonical_json().unwrap()),
            ResolvedArrayDecoderLimits::default(),
            D,
            D,
            RealTableProfile::PiecewiseAffineOpenIntervalsV1,
            interval(),
        )
    }
    #[test]
    fn identity_is_authenticated_through_existing_array_owner() {
        let array = ResolvedArrayV1::from_f64(vec![3, 2], vec![0., 1., 2., 5., 5., 2.]).unwrap();
        let accepted = accept(&array).unwrap();
        assert_eq!(
            accepted.digest().unwrap(),
            array.digest().unwrap().sha256_bytes()
        );
        assert_eq!(
            accepted.profile(),
            RealTableProfile::PiecewiseAffineOpenIntervalsV1,
        );
        let bytes = array.canonical_json().unwrap();
        let mut whitespace = b" \n".to_vec();
        whitespace.extend_from_slice(&bytes);
        assert_eq!(
            decode_real_table(
                &array.digest().unwrap(),
                Some(&whitespace),
                ResolvedArrayDecoderLimits::default(),
                D,
                D,
                RealTableProfile::PiecewiseAffineOpenIntervalsV1,
                interval(),
            )
            .unwrap(),
            accepted
        );
        for content in [None, Some(bytes.as_slice())] {
            assert!(
                decode_real_table(
                    &ArtifactDigest::from_sha256([0; 32]),
                    content,
                    ResolvedArrayDecoderLimits::default(),
                    D,
                    D,
                    RealTableProfile::PiecewiseAffineOpenIntervalsV1,
                    interval(),
                )
                .is_err()
            );
        }
        let changed = ResolvedArrayV1::from_f64(vec![3, 2], vec![0., 1., 2., 6., 5., 2.]).unwrap();
        assert!(
            decode_real_table(
                &array.digest().unwrap(),
                Some(&changed.canonical_json().unwrap()),
                ResolvedArrayDecoderLimits::default(),
                D,
                D,
                RealTableProfile::PiecewiseAffineOpenIntervalsV1,
                interval(),
            )
            .is_err()
        );
        assert_ne!(
            accept(&changed).unwrap().value().digest(),
            accepted.value().digest()
        );
    }
    #[test]
    fn malformed_axes_shapes_and_scalar_domains_reject() {
        for (shape, values) in [
            (vec![4], vec![0., 1., 2., 3.]),
            (vec![1, 2], vec![0., 1.]),
            (vec![2, 2], vec![0., 1., 0., 2.]),
            (vec![2, 2], vec![2., 1., 0., 2.]),
            (vec![2, 2], vec![0., 1., 2., f64::from_bits(1)]),
        ] {
            assert!(accept(&ResolvedArrayV1::from_f64(shape, values).unwrap()).is_err());
        }
        assert!(accept(&ResolvedArrayV1::from_u64(vec![2, 2], vec![0, 1, 2, 3]).unwrap()).is_err());
        let oversized = ResolvedArrayV1::from_f64(vec![129, 2], vec![0.; 258]).unwrap();
        assert!(accept(&oversized).is_err());
    }
    #[test]
    fn missing_values_and_unapproved_preprocessing_are_not_decoded() {
        let array = ResolvedArrayV1::from_f64(vec![2, 2], vec![0., 1., 2., 3.]).unwrap();
        let original = String::from_utf8(array.canonical_json().unwrap()).unwrap();
        for malformed in [
            original.replace("[0.0,1.0,2.0,3.0]", "[0.0,null,2.0,3.0]"),
            original.replacen('{', "{\"preprocessing\":\"fill_zero\",", 1),
        ] {
            assert!(
                decode_real_table(
                    &array.digest().unwrap(),
                    Some(malformed.as_bytes()),
                    ResolvedArrayDecoderLimits::default(),
                    D,
                    D,
                    RealTableProfile::PiecewiseAffineOpenIntervalsV1,
                    interval(),
                )
                .is_err()
            );
        }
    }
}
