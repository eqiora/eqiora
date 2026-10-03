//! Complete typed literal payloads, with a unique compact spelling of zero.

use eqiora_core::{Diagnostic, ScalarDomain, ValueLiteral};
use serde::{Deserialize, Serialize};

use super::{ModelDecoderLimits, require_decoder_count, value_type::WireValueType};
use crate::invalid_artifact;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireValueLiteral {
    value_type: WireValueType,
    pub(super) components: WireComponents,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireComponents {
    Zero,
    Enum { tag: u32 },
    Boolean { value: bool },
    Dense { values: Vec<(f64, f64)> },
    Integer { values: Vec<i64> },
}

impl WireValueLiteral {
    pub(crate) fn encode(value: &ValueLiteral) -> Result<Self, Diagnostic> {
        Ok(Self {
            value_type: WireValueType::encode(value.value_type())?,
            components: if let Some(tag) = value.enum_tag() {
                WireComponents::Enum { tag }
            } else if let Some(value) = value.as_bool() {
                WireComponents::Boolean { value }
            } else if value.is_zero() {
                WireComponents::Zero
            } else if let Some(components) = value.integer_components() {
                WireComponents::Integer {
                    values: components.collect(),
                }
            } else {
                WireComponents::Dense {
                    values: value
                        .components()
                        .ok_or_else(|| invalid_artifact("floating literal payload is absent"))?
                        .collect(),
                }
            },
        })
    }

    pub(crate) fn decode(&self) -> Result<ValueLiteral, Diagnostic> {
        let value_type = self.value_type.decode()?;
        let result = match &self.components {
            WireComponents::Enum { tag } => ValueLiteral::enum_value(value_type, *tag),
            WireComponents::Boolean { value } => {
                let literal = ValueLiteral::boolean(*value);
                if literal.value_type() != &value_type {
                    return Err(invalid_artifact(
                        "Boolean payload requires the exact Boolean scalar type",
                    ));
                }
                return Ok(literal);
            }
            WireComponents::Zero => match value_type.scalar_domain() {
                ScalarDomain::Enum => {
                    return Err(invalid_artifact("enum requires an explicit tag payload"));
                }
                ScalarDomain::Boolean => {
                    return Err(invalid_artifact(
                        "Boolean requires an explicit truth payload",
                    ));
                }
                ScalarDomain::Integer => ValueLiteral::from_integer(value_type, 0),
                _ => ValueLiteral::from_real(value_type, 0.0),
            },
            WireComponents::Integer { values } => {
                if values.iter().all(|value| *value == 0) {
                    return Err(invalid_artifact(
                        "all-zero literal requires compact zero payload",
                    ));
                }
                ValueLiteral::integer(value_type, values.iter().copied())
            }
            WireComponents::Dense { values } => {
                if values.iter().any(|(real, imaginary)| {
                    !real.is_finite()
                        || !imaginary.is_finite()
                        || (*real == 0.0 && real.is_sign_negative())
                        || (*imaginary == 0.0 && imaginary.is_sign_negative())
                }) {
                    return Err(invalid_artifact(
                        "literal components require finite values and canonical positive zero",
                    ));
                }
                if values.iter().all(|component| *component == (0.0, 0.0)) {
                    return Err(invalid_artifact(
                        "all-zero literal requires compact zero payload",
                    ));
                }
                ValueLiteral::new(value_type, values.iter().copied())
            }
        };
        result.map_err(|error| invalid_artifact(error.to_string()))
    }

    pub(crate) fn nominal_references(&self) -> impl Iterator<Item = &super::primitive::WireId> {
        self.value_type.nominal_references()
    }

    pub(crate) fn component_payload_count(&self) -> usize {
        match &self.components {
            WireComponents::Zero => 0,
            WireComponents::Boolean { .. } | WireComponents::Enum { .. } => 1,
            WireComponents::Dense { values } => values.len(),
            WireComponents::Integer { values } => values.len(),
        }
    }

    pub(crate) fn ensure_limits(&self, limits: ModelDecoderLimits) -> Result<(), Diagnostic> {
        self.value_type.ensure_limits(limits)?;
        require_decoder_count(
            "literal component payload",
            self.component_payload_count(),
            limits.max_value_literal_components,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, ScalarDomain, ValueType};

    fn complex_pair() -> ValueType {
        ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
            .expect("valid fixture scalar type")
            .array(2)
            .unwrap()
    }

    #[test]
    fn wire_contains_ordered_real_and_imaginary_channels() {
        let value = ValueLiteral::new(complex_pair(), [(1.0, -2.0), (3.0, 4.0)]).unwrap();
        let wire = WireValueLiteral::encode(&value).unwrap();
        // The expected payload is independently specified by the two input pairs.
        assert_eq!(
            serde_json::to_value(&wire).unwrap()["components"],
            serde_json::json!({"kind":"dense","values":[[1.0,-2.0],[3.0,4.0]]})
        );
        assert_eq!(wire.decode().unwrap(), value);
    }

    #[test]
    fn huge_zero_does_not_expand_and_consumes_no_component_payload() {
        let ty = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
            .expect("valid fixture scalar type")
            .array(1_000_000_000)
            .unwrap();
        let value = ValueLiteral::from_real(ty, 0.0).unwrap();
        let wire = WireValueLiteral::encode(&value).unwrap();
        assert_eq!(wire.component_payload_count(), 0);
        assert_eq!(
            serde_json::to_value(&wire).unwrap()["components"],
            serde_json::json!({"kind":"zero"})
        );
        let limits = ModelDecoderLimits {
            max_value_shape_components: 1_000_000_000,
            max_value_literal_components: 0,
            ..Default::default()
        };
        wire.ensure_limits(limits).unwrap();
        assert_eq!(wire.decode().unwrap(), value);
    }

    #[test]
    fn malformed_payloads_have_no_second_canonical_spelling() {
        let mut wire =
            WireValueLiteral::encode(&ValueLiteral::from_real(complex_pair(), 0.0).unwrap())
                .unwrap();
        for values in [
            vec![],
            vec![(1.0, 2.0)],
            vec![(0.0, 0.0); 2],
            vec![(-0.0, 1.0); 2],
            vec![(1.0, -0.0); 2],
            vec![(f64::INFINITY, 0.0); 2],
            vec![(0.0, f64::NAN); 2],
        ] {
            wire.components = WireComponents::Dense { values };
            assert!(wire.decode().is_err());
        }
        wire.value_type = WireValueType::encode(
            &ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                .expect("valid fixture scalar type")
                .array(2)
                .unwrap(),
        )
        .unwrap();
        wire.components = WireComponents::Dense {
            values: vec![(1.0, 2.0); 2],
        };
        assert!(wire.decode().is_err());
    }
    #[test]
    fn integer_payload_is_exact_beyond_binary64_and_has_one_zero_spelling() {
        let ty = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("valid fixture scalar type")
            .array(3)
            .unwrap();
        let value = ValueLiteral::integer(
            ty.clone(),
            [9_007_199_254_740_992, 9_007_199_254_740_993, i64::MIN],
        )
        .unwrap();
        let wire = WireValueLiteral::encode(&value).unwrap();
        let json = serde_json::to_string(&wire).unwrap();
        // Independently specified decimal integers: adjacent values cannot share binary64.
        assert!(
            json.contains("\"values\":[9007199254740992,9007199254740993,-9223372036854775808]")
        );
        let decoded: WireValueLiteral = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.decode().unwrap(), value);
        assert_eq!(wire.component_payload_count(), 3);
        assert!(
            wire.ensure_limits(ModelDecoderLimits {
                max_value_literal_components: 2,
                ..Default::default()
            })
            .is_err()
        );
        let mut invalid = wire.clone();
        invalid.components = WireComponents::Integer { values: vec![0; 3] };
        assert!(invalid.decode().is_err());
        invalid.components = WireComponents::Dense {
            values: vec![(1.0, 0.0); 3],
        };
        assert!(invalid.decode().is_err());
        invalid = wire;
        invalid.value_type = WireValueType::encode(
            &ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                .expect("valid fixture scalar type")
                .array(3)
                .unwrap(),
        )
        .unwrap();
        assert!(invalid.decode().is_err());
        let huge = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("valid fixture scalar type")
            .array(1_000_000_000)
            .unwrap();
        let zero = ValueLiteral::from_integer(huge, 0).unwrap();
        let wire = WireValueLiteral::encode(&zero).unwrap();
        assert_eq!(wire.component_payload_count(), 0);
        assert_eq!(wire.decode().unwrap(), zero);
    }
    #[test]
    fn nominal_payloads_keep_count_sign_and_index_bounds() {
        let space = eqiora_core::Id::new();
        let set = eqiora_core::Id::new();
        let count = ValueLiteral::integer(ValueType::counts(space, 2).unwrap(), [0, 1]).unwrap();
        let mut wire = WireValueLiteral::encode(&count).unwrap();
        assert_eq!(wire.decode().unwrap(), count);
        wire.components = WireComponents::Integer {
            values: vec![-1, 1],
        };
        assert!(
            wire.decode().is_err(),
            "count payload cannot become signed coordinates"
        );
        let index = ValueLiteral::from_integer(ValueType::index(set, 2).unwrap(), 1).unwrap();
        let mut wire = WireValueLiteral::encode(&index).unwrap();
        assert_eq!(wire.decode().unwrap(), index);
        wire.components = WireComponents::Integer { values: vec![2] };
        assert!(wire.decode().is_err(), "index bound is exclusive");
        wire.components = WireComponents::Zero;
        assert_eq!(wire.decode().unwrap().integer_component(0), Some(0));
        assert_eq!(wire.decode().unwrap().value_type().index_set(), Some(set));
    }
    #[test]
    fn boolean_truth_payload_has_no_numeric_zero_or_integer_alias() {
        for truth in [false, true] {
            let value = ValueLiteral::boolean(truth);
            let wire = WireValueLiteral::encode(&value).unwrap();
            let json = serde_json::to_value(&wire).unwrap();
            assert_eq!(
                json["components"],
                serde_json::json!({"kind":"boolean","value":truth})
            );
            assert_eq!(wire.decode().unwrap(), value);
            assert_eq!(wire.component_payload_count(), 1);
            assert!(
                wire.ensure_limits(ModelDecoderLimits {
                    max_value_literal_components: 0,
                    ..Default::default()
                })
                .is_err()
            );
            for payload in [
                WireComponents::Zero,
                WireComponents::Integer { values: vec![1] },
                WireComponents::Dense {
                    values: vec![(1.0, 0.0)],
                },
            ] {
                let mut malformed = wire.clone();
                malformed.components = payload;
                assert!(malformed.decode().is_err());
            }
            let mut malformed = json;
            malformed["components"]["value"] = serde_json::json!(1);
            assert!(serde_json::from_value::<WireValueLiteral>(malformed).is_err());
        }
        let mut numeric = WireValueLiteral::encode(
            &ValueLiteral::from_integer(
                ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
                    .expect("valid fixture scalar type"),
                1,
            )
            .unwrap(),
        )
        .unwrap();
        numeric.components = WireComponents::Boolean { value: true };
        assert!(numeric.decode().is_err());
    }
}
