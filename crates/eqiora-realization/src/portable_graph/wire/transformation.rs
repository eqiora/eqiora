//! Wire representation of numerical transformation nodes.

use eqiora_core::Diagnostic;
use serde::{Deserialize, Serialize};

use super::basic::{
    WireConvectionScheme, WirePositiveScale, WireQuantity, decode_face_history,
    decode_momentum_diagonal, encode_face_history, encode_momentum_diagonal, parse_id,
};
use super::{decode_index, encode_index};
use crate::{FieldRepresentationId, GeometryActionId, TransformationNode};
mod trace;
use trace::WireTraceSource;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireTransformation {
    BackwardEulerDerivative {
        relation_ulid: String,
        state: u64,
        duration: WireQuantity,
    },
    EnergySkewConvection {
        relation_ulid: String,
        velocity: u64,
    },
    CellCenteredConvection {
        relation_ulid: String,
        state: u64,
        scheme: WireConvectionScheme,
    },
    OrthogonalTwoPointDiffusion {
        relation_ulid: String,
        state: u64,
    },
    ImplicitCenteredMomentumConvection {
        relation_ulid: String,
        velocity: u64,
    },
    CartesianCentralNewtonianTraction {
        relation_ulid: String,
        velocity: u64,
        pressure: u64,
    },
    MomentumWeightedLinearExactCoupling {
        momentum_relation_ulid: String,
        incompressibility_relation_ulid: String,
        velocity: u64,
        pressure: u64,
        positive_diagonal: String,
        transient_history: String,
    },
    GclCompatibleAlePullback {
        relation_ulid: String,
        velocity: u64,
        geometry: u64,
    },
    BackwardEulerElimination {
        relation_ulid: String,
        state: u64,
        rate: u64,
        duration: WireQuantity,
        state_scale: WirePositiveScale,
    },
    ConformingTraceQuotient {
        source: WireTraceSource,
        endpoints: [u64; 2],
    },
}

impl WireTransformation {
    pub(super) fn encode(value: TransformationNode) -> Result<Self, Diagnostic> {
        let field = |id: FieldRepresentationId, label| encode_index(id.index(), label);
        Ok(match value {
            TransformationNode::BackwardEulerDerivative {
                relation,
                state,
                duration,
            } => Self::BackwardEulerDerivative {
                relation_ulid: relation.ulid().to_string(),
                state: field(state, "backward-Euler state")?,
                duration: WireQuantity::encode(duration),
            },
            TransformationNode::EnergySkewConvection { relation, velocity } => {
                Self::EnergySkewConvection {
                    relation_ulid: relation.ulid().to_string(),
                    velocity: field(velocity, "energy-skew velocity")?,
                }
            }
            TransformationNode::CellCenteredConvection {
                relation,
                state,
                scheme,
            } => Self::CellCenteredConvection {
                relation_ulid: relation.ulid().to_string(),
                state: field(state, "cell-centered state")?,
                scheme: WireConvectionScheme::encode(scheme),
            },
            TransformationNode::OrthogonalTwoPointDiffusion { relation, state } => {
                Self::OrthogonalTwoPointDiffusion {
                    relation_ulid: relation.ulid().to_string(),
                    state: field(state, "diffusive state")?,
                }
            }
            TransformationNode::ImplicitCenteredMomentumConvection { relation, velocity } => {
                Self::ImplicitCenteredMomentumConvection {
                    relation_ulid: relation.ulid().to_string(),
                    velocity: field(velocity, "momentum velocity")?,
                }
            }
            TransformationNode::CartesianCentralNewtonianTraction {
                relation,
                velocity,
                pressure,
            } => Self::CartesianCentralNewtonianTraction {
                relation_ulid: relation.ulid().to_string(),
                velocity: field(velocity, "traction velocity")?,
                pressure: field(pressure, "traction pressure")?,
            },
            TransformationNode::MomentumWeightedLinearExactCoupling {
                momentum_relation,
                incompressibility_relation,
                velocity,
                pressure,
                positive_diagonal,
                transient_history,
            } => Self::MomentumWeightedLinearExactCoupling {
                momentum_relation_ulid: momentum_relation.ulid().to_string(),
                incompressibility_relation_ulid: incompressibility_relation.ulid().to_string(),
                velocity: field(velocity, "coupling velocity")?,
                pressure: field(pressure, "coupling pressure")?,
                positive_diagonal: encode_momentum_diagonal(positive_diagonal).to_owned(),
                transient_history: encode_face_history(transient_history).to_owned(),
            },
            TransformationNode::GclCompatibleAlePullback {
                relation,
                velocity,
                geometry,
            } => Self::GclCompatibleAlePullback {
                relation_ulid: relation.ulid().to_string(),
                velocity: field(velocity, "ALE velocity")?,
                geometry: encode_index(geometry.index(), "ALE geometry action")?,
            },
            TransformationNode::BackwardEulerElimination {
                relation,
                state,
                rate,
                duration,
                state_scale,
            } => Self::BackwardEulerElimination {
                relation_ulid: relation.ulid().to_string(),
                state: field(state, "eliminated state")?,
                rate: field(rate, "retained rate")?,
                duration: WireQuantity::encode(duration),
                state_scale: WirePositiveScale::encode(state_scale),
            },
            TransformationNode::ConformingTraceQuotient { source, endpoints } => {
                Self::ConformingTraceQuotient {
                    source: WireTraceSource::encode(source),
                    endpoints: [
                        field(endpoints[0], "trace endpoint")?,
                        field(endpoints[1], "trace endpoint")?,
                    ],
                }
            }
        })
    }

    pub(super) fn decode(self) -> Result<TransformationNode, Diagnostic> {
        let field = |index, label| decode_index(index, label).map(FieldRepresentationId::new);
        Ok(match self {
            Self::BackwardEulerDerivative {
                relation_ulid,
                state,
                duration,
            } => TransformationNode::BackwardEulerDerivative {
                relation: parse_id(&relation_ulid)?,
                state: field(state, "backward-Euler state")?,
                duration: duration.decode(),
            },
            Self::EnergySkewConvection {
                relation_ulid,
                velocity,
            } => TransformationNode::EnergySkewConvection {
                relation: parse_id(&relation_ulid)?,
                velocity: field(velocity, "energy-skew velocity")?,
            },
            Self::CellCenteredConvection {
                relation_ulid,
                state,
                scheme,
            } => TransformationNode::CellCenteredConvection {
                relation: parse_id(&relation_ulid)?,
                state: field(state, "cell-centered state")?,
                scheme: scheme.decode(),
            },
            Self::OrthogonalTwoPointDiffusion {
                relation_ulid,
                state,
            } => TransformationNode::OrthogonalTwoPointDiffusion {
                relation: parse_id(&relation_ulid)?,
                state: field(state, "diffusive state")?,
            },
            Self::ImplicitCenteredMomentumConvection {
                relation_ulid,
                velocity,
            } => TransformationNode::ImplicitCenteredMomentumConvection {
                relation: parse_id(&relation_ulid)?,
                velocity: field(velocity, "momentum velocity")?,
            },
            Self::CartesianCentralNewtonianTraction {
                relation_ulid,
                velocity,
                pressure,
            } => TransformationNode::CartesianCentralNewtonianTraction {
                relation: parse_id(&relation_ulid)?,
                velocity: field(velocity, "traction velocity")?,
                pressure: field(pressure, "traction pressure")?,
            },
            Self::MomentumWeightedLinearExactCoupling {
                momentum_relation_ulid,
                incompressibility_relation_ulid,
                velocity,
                pressure,
                positive_diagonal,
                transient_history,
            } => TransformationNode::MomentumWeightedLinearExactCoupling {
                momentum_relation: parse_id(&momentum_relation_ulid)?,
                incompressibility_relation: parse_id(&incompressibility_relation_ulid)?,
                velocity: field(velocity, "coupling velocity")?,
                pressure: field(pressure, "coupling pressure")?,
                positive_diagonal: decode_momentum_diagonal(&positive_diagonal)?,
                transient_history: decode_face_history(&transient_history)?,
            },
            Self::GclCompatibleAlePullback {
                relation_ulid,
                velocity,
                geometry,
            } => TransformationNode::GclCompatibleAlePullback {
                relation: parse_id(&relation_ulid)?,
                velocity: field(velocity, "ALE velocity")?,
                geometry: GeometryActionId::new(decode_index(geometry, "ALE geometry action")?),
            },
            Self::BackwardEulerElimination {
                relation_ulid,
                state,
                rate,
                duration,
                state_scale,
            } => TransformationNode::BackwardEulerElimination {
                relation: parse_id(&relation_ulid)?,
                state: field(state, "eliminated state")?,
                rate: field(rate, "retained rate")?,
                duration: duration.decode(),
                state_scale: state_scale.decode()?,
            },
            Self::ConformingTraceQuotient { source, endpoints } => {
                TransformationNode::ConformingTraceQuotient {
                    source: source.decode()?,
                    endpoints: [
                        field(endpoints[0], "trace endpoint")?,
                        field(endpoints[1], "trace endpoint")?,
                    ],
                }
            }
        })
    }
}
