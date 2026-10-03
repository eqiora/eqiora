use super::primitive::WireId;
use eqiora_core::{Diagnostic, ScalarDomain, ValueShape, entity::kinds};
use eqiora_core::{FiniteBasis, ValueFrame, ValueType};

use serde::{Deserialize, Serialize};

use super::{
    ModelDecoderLimits,
    vocabulary::{WireValueFrame, WireValueShape},
};
use crate::{dimension::WireDimension, invalid_artifact};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireValueType {
    domain: WireScalarDomain,
    dimension: WireDimension,
    shape: WireValueShape,
    frame: WireValueFrame,
    array_rank: u32,
    basis: Box<WireValueBasis>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireValueBasis {
    Ordinary,
    Enum {
        definition: WireId,
        count: u32,
    },
    Coordinates {
        basis: WireFiniteBasis,
    },
    LinearMap {
        source: WireFiniteBasis,
        target: WireFiniteBasis,
    },
    Counts {
        space: WireId,
    },
    Index {
        set: WireId,
        extent: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireFiniteBasis {
    Atomic {
        space: WireId,
        dual: bool,
    },
    Product {
        factors: [WireFiniteFactor; 2],
        dual: bool,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireFiniteFactor {
    pub(super) space: WireId,
    pub(super) extent: u32,
}
impl WireFiniteBasis {
    fn encode(basis: FiniteBasis) -> Self {
        match basis.factors() {
            Some(factors) => Self::Product {
                factors: factors.map(|factor| WireFiniteFactor {
                    space: WireId::from_raw(factor.space().expect("atomic factor").erase()),
                    extent: factor.extent(),
                }),
                dual: basis.is_dual(),
            },
            None => Self::Atomic {
                space: WireId::from_raw(basis.space().expect("atomic basis").erase()),
                dual: basis.is_dual(),
            },
        }
    }
    fn references(&self) -> [Option<&WireId>; 2] {
        match self {
            Self::Atomic { space, .. } => [Some(space), None],
            Self::Product { factors, .. } => [Some(&factors[0].space), Some(&factors[1].space)],
        }
    }
    fn decode(&self, extent: u32) -> Result<FiniteBasis, Diagnostic> {
        let atomic = |space: &WireId, extent| {
            FiniteBasis::new(space.typed::<kinds::FiniteSpace>()?, extent)
                .map_err(|error| invalid_artifact(error.to_string()))
        };
        let (basis, dual) = match self {
            Self::Atomic { space, dual } => (atomic(space, extent)?, *dual),
            Self::Product {
                factors: [left, right],
                dual,
            } => (
                FiniteBasis::product(
                    atomic(&left.space, left.extent)?,
                    atomic(&right.space, right.extent)?,
                )
                .map_err(|error| invalid_artifact(error.to_string()))?,
                *dual,
            ),
        };
        if basis.extent() != extent {
            return Err(invalid_artifact(
                "finite product extent does not match its factor extents",
            ));
        }
        Ok(if dual { basis.dual() } else { basis })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireScalarDomain {
    Enum,
    Boolean,
    Integer,
    Real,
    Complex,
}

impl WireScalarDomain {
    pub(super) const fn encode(value: ScalarDomain) -> Self {
        match value {
            ScalarDomain::Enum => Self::Enum,
            ScalarDomain::Boolean => Self::Boolean,
            ScalarDomain::Integer => Self::Integer,
            ScalarDomain::Real => Self::Real,
            ScalarDomain::Complex => Self::Complex,
        }
    }

    pub(super) const fn decode(self) -> ScalarDomain {
        match self {
            Self::Enum => ScalarDomain::Enum,
            Self::Boolean => ScalarDomain::Boolean,
            Self::Integer => ScalarDomain::Integer,
            Self::Real => ScalarDomain::Real,
            Self::Complex => ScalarDomain::Complex,
        }
    }
}

impl WireValueType {
    pub(super) fn encode(value: &ValueType) -> Result<Self, Diagnostic> {
        let basis = if let Some(definition) = value.enum_definition() {
            WireValueBasis::Enum {
                definition: WireId::from_raw(definition.erase()),
                count: value.enum_member_count().expect("checked enum type"),
            }
        } else if let Some(set) = value.index_set() {
            WireValueBasis::Index {
                set: WireId::from_raw(set.erase()),
                extent: value.index_extent().expect("checked index type"),
            }
        } else if let Some((source, target)) = value.map_bases() {
            WireValueBasis::LinearMap {
                source: WireFiniteBasis::encode(source),
                target: WireFiniteBasis::encode(target),
            }
        } else if let Some(basis) = value.coordinate_basis() {
            WireValueBasis::Coordinates {
                basis: WireFiniteBasis::encode(basis),
            }
        } else if let Some(space) = value.finite_space() {
            WireValueBasis::Counts {
                space: WireId::from_raw(space.erase()),
            }
        } else {
            WireValueBasis::Ordinary
        };
        Ok(Self {
            basis: Box::new(basis),
            domain: WireScalarDomain::encode(value.scalar_domain()),
            dimension: WireDimension::encode(value.dimension()),
            shape: WireValueShape::encode(value.shape()),
            frame: WireValueFrame::encode(value.frame()),
            array_rank: u32::try_from(value.array_rank())
                .map_err(|_| invalid_artifact("array rank exceeds u32"))?,
        })
    }

    pub(super) fn decode(&self) -> Result<ValueType, Diagnostic> {
        let shape = self.shape.decode()?;
        let rank = usize::try_from(self.array_rank)
            .map_err(|_| invalid_artifact("array rank exceeds usize"))?;
        if rank > shape.rank()
            || (matches!(*self.basis, WireValueBasis::Ordinary)
                && self.frame.decode() == ValueFrame::Invariant
                && rank != shape.rank())
        {
            return Err(invalid_artifact(
                "mathematical type has inconsistent array and spatial axis roles",
            ));
        }
        match self.basis.as_ref() {
            WireValueBasis::Ordinary => {
                if self.domain == WireScalarDomain::Enum {
                    return Err(invalid_artifact(
                        "enum requires its exact nominal declaration",
                    ));
                }
            }
            WireValueBasis::Enum { definition, count } => {
                if self.domain != WireScalarDomain::Enum
                    || !shape.is_scalar()
                    || rank != 0
                    || self.frame.decode() != ValueFrame::Invariant
                    || self.dimension.decode() != eqiora_core::DimExponents::DIMENSIONLESS
                {
                    return Err(invalid_artifact(
                        "enum type requires a dimensionless invariant nominal scalar",
                    ));
                }
                return ValueType::enumeration(definition.typed::<kinds::Enum>()?, *count)
                    .map_err(|error| invalid_artifact(error.to_string()));
            }
            WireValueBasis::Index { set, extent } => {
                if self.domain != WireScalarDomain::Integer
                    || !shape.is_scalar()
                    || rank != 0
                    || self.frame.decode() != ValueFrame::Invariant
                    || self.dimension.decode() != eqiora_core::DimExponents::DIMENSIONLESS
                {
                    return Err(invalid_artifact(
                        "index type requires a dimensionless invariant Integer scalar",
                    ));
                }
                return ValueType::index(set.typed::<kinds::IndexSet>()?, *extent)
                    .map_err(|error| invalid_artifact(error.to_string()));
            }
            WireValueBasis::Counts { space } => {
                if self.domain != WireScalarDomain::Integer
                    || shape.rank() != 1
                    || rank != 0
                    || self.frame.decode() != ValueFrame::Invariant
                    || self.dimension.decode() != eqiora_core::DimExponents::DIMENSIONLESS
                {
                    return Err(invalid_artifact(
                        "counts require a dimensionless invariant Integer basis",
                    ));
                }
                return ValueType::counts(
                    space.typed::<kinds::FiniteSpace>()?,
                    shape.extents()[0].get(),
                )
                .map_err(|error| invalid_artifact(error.to_string()));
            }
            WireValueBasis::Coordinates { basis } => {
                if shape.rank() != 1 || rank != 0 || self.frame.decode() != ValueFrame::Invariant {
                    return Err(invalid_artifact(
                        "coordinates require one explicit finite basis axis",
                    ));
                }
                return ValueType::coordinates(
                    basis.decode(shape.extents()[0].get())?,
                    self.domain.decode(),
                    self.dimension.decode(),
                )
                .map_err(|error| invalid_artifact(error.to_string()));
            }
            WireValueBasis::LinearMap { source, target } => {
                if shape.rank() != 2 || rank != 0 || self.frame.decode() != ValueFrame::Invariant {
                    return Err(invalid_artifact(
                        "linear maps require output and input basis axes",
                    ));
                }
                return ValueType::linear_map(
                    source.decode(shape.extents()[1].get())?,
                    target.decode(shape.extents()[0].get())?,
                    self.domain.decode(),
                    self.dimension.decode(),
                )
                .map_err(|error| invalid_artifact(error.to_string()));
            }
        }
        let (arrays, spatial) = shape.extents().split_at(rank);
        let mut value = ValueType::shaped(
            self.domain.decode(),
            self.dimension.decode(),
            ValueShape::new(spatial.iter().map(|n| n.get()))
                .map_err(|error| invalid_artifact(error.to_string()))?,
            self.frame.decode(),
        )
        .map_err(|error| invalid_artifact(error.to_string()))?;
        for extent in arrays.iter().rev() {
            value = value
                .array(extent.get())
                .map_err(|error| invalid_artifact(error.to_string()))?;
        }
        Ok(value)
    }

    pub(super) fn nominal_references(&self) -> impl Iterator<Item = &WireId> {
        match self.basis.as_ref() {
            WireValueBasis::Ordinary => [None, None, None, None],
            WireValueBasis::Enum { definition, .. } => [Some(definition), None, None, None],
            WireValueBasis::Coordinates { basis } => {
                let [a, b] = basis.references();
                [a, b, None, None]
            }
            WireValueBasis::LinearMap { source, target } => {
                let [a, b] = source.references();
                let [c, d] = target.references();
                [a, b, c, d]
            }
            WireValueBasis::Counts { space } => [Some(space), None, None, None],
            WireValueBasis::Index { set, .. } => [Some(set), None, None, None],
        }
        .into_iter()
        .flatten()
    }

    pub(super) fn ensure_limits(&self, limits: ModelDecoderLimits) -> Result<(), Diagnostic> {
        self.shape.ensure_limits(limits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::WireNode;
    use eqiora_core::{DimExponents, Id};
    use eqiora_schema::kernel::{FieldDef, KernelNode};

    #[test]
    fn finite_types_replay_both_basis_roles_and_reject_erased_meaning() {
        let input = FiniteBasis::new(Id::new(), 2).unwrap();
        let output = FiniteBasis::new(Id::new(), 3).unwrap();
        let dimension = DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap();
        let product = FiniteBasis::product(input, output).unwrap();
        let types = [
            ValueType::coordinates(product, ScalarDomain::Complex, dimension).unwrap(),
            ValueType::linear_map(
                product,
                product.swapped().unwrap().dual(),
                ScalarDomain::Complex,
                dimension,
            )
            .unwrap(),
            ValueType::coordinates(input, ScalarDomain::Real, dimension).unwrap(),
            ValueType::coordinates(input.dual(), ScalarDomain::Complex, dimension).unwrap(),
            ValueType::linear_map(input, output, ScalarDomain::Complex, dimension).unwrap(),
            ValueType::linear_map(
                output.dual(),
                input.dual(),
                ScalarDomain::Complex,
                dimension,
            )
            .unwrap(),
        ];
        for value in types {
            let wire = WireValueType::encode(&value).unwrap();
            let bytes = serde_json::to_vec(&wire).unwrap();
            let replayed: WireValueType = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(replayed.decode().unwrap(), value);
            let references = replayed
                .nominal_references()
                .map(|id| id.typed::<kinds::FiniteSpace>().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                references,
                value
                    .finite_bases()
                    .flat_map(|basis| basis.atoms())
                    .map(|basis| basis.space().unwrap())
                    .collect::<Vec<_>>()
            );
            let mut invalid = wire.clone();
            invalid.array_rank = 1;
            assert!(invalid.decode().is_err());
            let mut invalid = wire.clone();
            invalid.frame = WireValueFrame::SpatialCartesian;
            assert!(invalid.decode().is_err());
            let mut invalid = wire;
            invalid.domain = WireScalarDomain::Boolean;
            assert!(invalid.decode().is_err());
        }
        let map = ValueType::linear_map(input, output, ScalarDomain::Complex, dimension).unwrap();
        let mut json = serde_json::to_value(WireValueType::encode(&map).unwrap()).unwrap();
        json["basis"]["source"]["dual"] = serde_json::Value::Null;
        assert!(serde_json::from_value::<WireValueType>(json).is_err());
    }

    #[test]
    fn field_wire_preserves_role_and_rejects_displaced_initial_payload() {
        let value_type = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
            .expect("valid fixture scalar type")
            .array(2)
            .unwrap();
        for role in [
            eqiora_schema::kernel::FieldRole::Variable,
            eqiora_schema::kernel::FieldRole::State,
        ] {
            let node = KernelNode::from(FieldDef::new(Id::new(), value_type.clone(), role));
            let wire = WireNode::encode(&node).unwrap();
            assert_eq!(wire.decode().unwrap(), node);
            let mut json = serde_json::to_value(&wire).unwrap();
            json["definition"]["initial"] = serde_json::json!(0.0);
            assert!(serde_json::from_value::<WireNode>(json).is_err());
        }
    }

    #[test]
    fn scalar_physical_wire_retains_domains_and_rejects_channel_substitution() {
        use crate::model::{WireNodeDefinition, node::WireDomainKind};
        use eqiora_schema::kernel::DomainDef;
        let across = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
            .expect("valid fixture scalar type");
        let through = ValueType::scalar(
            ScalarDomain::Real,
            DimExponents::from_integers([0, 0, 0, 1, 0, 0, 0]).unwrap(),
        )
        .expect("valid fixture scalar type");
        let node = KernelNode::from(
            DomainDef::scalar_physical(Id::new(), across.clone(), through.clone()).unwrap(),
        );
        let wire = WireNode::encode(&node).unwrap();
        assert_eq!(wire.decode().unwrap(), node);
        for change_across in [true, false] {
            let mut malformed = wire.clone();
            let WireNodeDefinition::Domain {
                domain:
                    WireDomainKind::ScalarPhysical {
                        across_type,
                        through_type,
                    },
            } = &mut malformed.definition
            else {
                panic!("scalar physical domain");
            };
            let target = if change_across {
                across_type
            } else {
                through_type
            };
            *target = WireValueType::encode(&across.clone().array(1).unwrap()).unwrap();
            assert!(malformed.decode().is_err());
        }
        assert!(
            DomainDef::scalar_physical(
                Id::new(),
                across.clone().array(1).unwrap(),
                through.clone()
            )
            .is_err()
        );
        assert!(DomainDef::scalar_physical(Id::new(), across, through.array(1).unwrap()).is_err());
    }

    #[test]
    fn boundary_connector_wire_preserves_component_roles_and_checks_both_types() {
        use crate::model::WireNodeDefinition;
        use crate::model::node::WireDomainKind;
        use eqiora_schema::kernel::{BoundaryPairing, BoundaryPhysicalConnector, DomainDef};
        let vector = ValueType::shaped(
            ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
            ValueShape::new([2]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap();
        let nested = vector.array(2).unwrap();
        let connector = BoundaryPhysicalConnector::new(
            nested.clone(),
            nested.clone(),
            BoundaryPairing::EuclideanBoundaryDuality,
        )
        .unwrap();
        let node = KernelNode::from(DomainDef::boundary_physical(Id::new(), connector));
        let wire = WireNode::encode(&node).unwrap();
        assert_eq!(wire.decode().unwrap(), node);
        for trace in [true, false] {
            let mut malformed = wire.clone();
            let WireNodeDefinition::Domain {
                domain:
                    WireDomainKind::BoundaryPhysical {
                        trace_type,
                        flux_type,
                        ..
                    },
            } = &mut malformed.definition
            else {
                panic!("boundary connector");
            };
            let changed = if trace { trace_type } else { flux_type };
            changed.array_rank = 0;
            assert!(
                malformed.decode().is_err(),
                "equal extents cannot erase array roles on one quantity"
            );

            let mut oversized = wire.clone();
            let WireNodeDefinition::Domain {
                domain:
                    WireDomainKind::BoundaryPhysical {
                        trace_type,
                        flux_type,
                        ..
                    },
            } = &mut oversized.definition
            else {
                panic!("boundary connector");
            };
            let changed = if trace { trace_type } else { flux_type };
            *changed = WireValueType::encode(&nested.clone().array(4097).unwrap()).unwrap();
            assert!(
                oversized
                    .ensure_value_shape_limits(ModelDecoderLimits::default())
                    .is_err()
            );
        }
    }

    #[test]
    fn constant_wire_preserves_types_and_rejects_invalid_literals() {
        use crate::model::expression::{WireExpression, WireExpressionNode};
        use eqiora_core::ValueLiteral;
        use eqiora_schema::kernel::ExprDagBuilder;
        let value_type = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
            .expect("valid fixture scalar type")
            .array(3)
            .unwrap();
        let mut builder = ExprDagBuilder::new();
        let root = builder
            .constant(ValueLiteral::from_real(value_type, 0.0).unwrap())
            .unwrap();
        let expression = builder.finish([root]).unwrap();
        let wire = WireExpression::encode(&expression).unwrap();
        assert_eq!(wire.decode().unwrap(), expression);
        for invalid in [-0.0, f64::INFINITY, f64::NAN] {
            let mut malformed = wire.clone();
            let WireExpressionNode::Constant { value, .. } = &mut malformed.nodes[0] else {
                panic!("constant");
            };
            value.components = crate::model::literal::WireComponents::Dense {
                values: vec![(invalid, 0.0)],
            };
            assert!(malformed.decode().is_err());
        }
    }

    #[test]
    fn relation_and_guard_constant_wire_obey_shape_limits() {
        use eqiora_core::ValueLiteral;
        use eqiora_schema::kernel::{ActivationDef, ActivationKind, ExprDagBuilder, RelationDef};
        let scalar = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("valid fixture scalar type");
        for value_type in [
            scalar.clone().array(4097).unwrap(),
            (0..9).fold(scalar, |value, _| value.array(1).unwrap()),
        ] {
            let mut builder = ExprDagBuilder::new();
            let root = builder
                .constant(ValueLiteral::from_real(value_type.clone(), 0.0).unwrap())
                .unwrap();
            let paired = builder.finish([root, root]).unwrap();
            let mut guard = ExprDagBuilder::new();
            let root = guard
                .constant(ValueLiteral::from_real(value_type, 0.0).unwrap())
                .unwrap();
            let expression = guard.finish([root]).unwrap();
            for node in [
                KernelNode::from(RelationDef::new(Id::new(), paired).unwrap()),
                KernelNode::from(
                    ActivationDef::new(Id::new(), ActivationKind::Guard { guard: expression })
                        .unwrap(),
                ),
            ] {
                assert!(
                    WireNode::encode(&node)
                        .unwrap()
                        .ensure_value_shape_limits(ModelDecoderLimits::default())
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn parameter_wire_rejects_noncanonical_or_invalid_literals() {
        let value_type = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
            .expect("valid fixture scalar type")
            .array(2)
            .unwrap();
        let node = KernelNode::from(eqiora_schema::kernel::ParameterDef::new(
            Id::new(),
            eqiora_core::ValueLiteral::from_real(value_type, 0.0).unwrap(),
        ));
        assert_eq!(WireNode::encode(&node).unwrap().decode().unwrap(), node);
        for invalid in [-0.0, f64::INFINITY, f64::NAN] {
            let mut wire = WireNode::encode(&node).unwrap();
            let crate::model::WireNodeDefinition::Parameter { value, .. } = &mut wire.definition
            else {
                unreachable!()
            };
            value.components = crate::model::literal::WireComponents::Dense {
                values: vec![(invalid, 0.0)],
            };
            assert!(wire.decode().is_err());
        }
    }

    #[test]
    fn parameter_and_signal_wire_obey_the_model_shape_limits() {
        let scalar = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("valid fixture scalar type");
        for value_type in [
            scalar.clone().array(4097).unwrap(),
            (0..9).fold(scalar, |value, _| value.array(1).unwrap()),
        ] {
            for node in [
                KernelNode::from(eqiora_schema::kernel::ParameterDef::new(
                    Id::new(),
                    eqiora_core::ValueLiteral::from_real(value_type.clone(), 0.0).unwrap(),
                )),
                KernelNode::from(eqiora_schema::kernel::PortDef::signal(
                    Id::new(),
                    eqiora_schema::kernel::SignalDirection::Input,
                    value_type,
                )),
            ] {
                let wire = WireNode::encode(&node).unwrap();
                assert!(
                    wire.ensure_value_shape_limits(ModelDecoderLimits::default())
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn field_wire_retains_complete_mathematical_type() {
        let id = Id::new();
        let scalar = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
            .expect("valid fixture scalar type");
        let vector = ValueType::shaped(
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
            ValueShape::new([2]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap();
        let tensor = ValueType::shaped(
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
            ValueShape::new([2, 2]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap();
        let mut encodings = std::collections::BTreeSet::new();
        for value in [
            ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                .expect("valid fixture scalar type"),
            scalar.clone(),
            scalar.array(2).unwrap().array(2).unwrap(),
            vector.array(2).unwrap(),
            tensor,
        ] {
            let node = KernelNode::from(FieldDef::new(
                id,
                value,
                eqiora_schema::kernel::FieldRole::Variable,
            ));
            let bytes = serde_json::to_vec(&WireNode::encode(&node).unwrap()).unwrap();
            assert!(encodings.insert(bytes.clone()));
            let replayed: WireNode = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(replayed.decode().unwrap(), node);
        }
    }

    #[test]
    fn decoder_rejects_inconsistent_axis_roles_and_unknown_scalar_domains() {
        let value = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("valid fixture scalar type")
            .array(2)
            .unwrap();
        let wire = WireValueType::encode(&value).unwrap();
        for rank in [0, 2] {
            let mut invalid = wire.clone();
            invalid.array_rank = rank;
            assert!(invalid.decode().is_err());
        }
        let mut invalid = wire.clone();
        invalid.frame = WireValueFrame::SpatialCartesian;
        assert!(invalid.decode().is_err());
        let mut json = serde_json::to_value(&wire).unwrap();
        json["domain"] = serde_json::json!("floating-point");
        assert!(serde_json::from_value::<WireValueType>(json).is_err());
    }
}
