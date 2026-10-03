//! Bounded pure tensor calculus and semantic support-map proofs.
//!
//! Canonical Kernel expressions remain Model meaning. This module is a
//! lowered, capture-free seam: typed operator definitions expand to exact
//! scalar components, and a deliberately small proof checker classifies only
//! algebra admitted by this version. Numerical transfer, scheduling, and
//! floating-point reassociation do not belong here.

use std::fmt;

use sha2::{Digest, Sha256};

mod application;
mod derivative;
mod expansion;
mod normalization;
mod support_map;

pub use application::{
    OperatorApplicationProof, PureOperatorApplicationProof, StandardPureOperator,
};
pub use eqiora_schema::kernel::pure_operator::{
    CalculusBuilder, CalculusNode, CalculusNodeId, ComponentIndex, ExactRational,
    FormalDimensionMonomial, OperatorDefinitionDigest, PureOperatorDefinition, PureOperatorError,
    PureOperatorInstantiation, PureValueClass,
};
pub use expansion::{OperatorExpansionExt, ScalarCalculus, ScalarCalculusAtom, ScalarCalculusNode};
pub use normalization::{NormalizationProof, NormalizationRuleId};
pub use support_map::{
    SupportMap, SupportMapIntent, SupportMapOrientation, SupportMapPairing, SupportMapViolation,
};

/// Closed pure-calculus construction, typing, expansion, and proof failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalculusError {
    Definition(PureOperatorError),
    NormalizationLimit,
    InvalidFormal(u16),
    InvalidNode,
    UnsupportedDerivative,
    NonPolynomialCalculus,
    DerivativeProjection(String),
    ResultAxisOutOfRange,
    ComponentOutOfRange,
    InvalidExpressionNode,
    ApplicationDefinitionMissing,
    ApplicationDefinitionMismatch,
    ApplicationResultMismatch,
    UnsupportedProofRule(u16),
    ProofSourceMismatch,
    ProofTypeMismatch,
    ProofResultMismatch,
    SupportMap(SupportMapViolation),
}

impl fmt::Display for CalculusError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Definition(error) => error.fmt(formatter),
            Self::NormalizationLimit => {
                formatter.write_str("exact normalization exceeds its term limit")
            }
            Self::InvalidFormal(formal) => {
                write!(formatter, "pure operator formal {formal} is invalid")
            }
            Self::NonPolynomialCalculus => formatter.write_str(
                "exact polynomial calculus cannot expand branches or scalar mathematics",
            ),
            Self::UnsupportedDerivative => formatter.write_str(
                "partial derivatives require real scalar formals/results and order one or two",
            ),
            Self::DerivativeProjection(message) => formatter.write_str(message),
            Self::InvalidNode => {
                formatter.write_str("pure calculus node is invalid or forward-referenced")
            }
            Self::ResultAxisOutOfRange => {
                formatter.write_str("pure calculus result axis is out of range")
            }
            Self::ComponentOutOfRange => {
                formatter.write_str("pure calculus component is outside its exact shape")
            }
            Self::InvalidExpressionNode => formatter
                .write_str("pure operator application references an invalid expression node"),
            Self::ApplicationDefinitionMissing => {
                formatter.write_str("pure operator application has no retained exact definition")
            }
            Self::ApplicationDefinitionMismatch => formatter.write_str(
                "pure operator application digest resolves to a different exact definition",
            ),
            Self::ApplicationResultMismatch => formatter
                .write_str("pure operator expansion result differs from the typed Kernel result"),
            Self::UnsupportedProofRule(rule) => {
                write!(formatter, "normalization proof rule {rule} is unsupported")
            }
            Self::ProofSourceMismatch => {
                formatter.write_str("normalization proof source digest does not match")
            }
            Self::ProofTypeMismatch => {
                formatter.write_str("normalization proof typed context does not match the source")
            }
            Self::ProofResultMismatch => {
                formatter.write_str("normalization proof result does not replay")
            }
            Self::SupportMap(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CalculusError {}

impl From<PureOperatorError> for CalculusError {
    fn from(error: PureOperatorError) -> Self {
        Self::Definition(error)
    }
}

fn calculus_index(id: CalculusNodeId, upper: usize) -> Result<usize, CalculusError> {
    usize::try_from(id.index())
        .ok()
        .filter(|index| *index < upper)
        .ok_or(CalculusError::InvalidNode)
}

fn expr_index(value: eqiora_schema::kernel::ExprId, upper: usize) -> Result<usize, CalculusError> {
    usize::try_from(value.index())
        .ok()
        .filter(|index| *index < upper)
        .ok_or(CalculusError::InvalidExpressionNode)
}

fn push_rational(bytes: &mut Vec<u8>, value: ExactRational) {
    bytes.extend_from_slice(&value.numerator().to_be_bytes());
    bytes.extend_from_slice(&value.denominator().to_be_bytes());
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: usize) {
    let value = u32::try_from(value).expect("bounded calculus count fits u32");
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn hash(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
