//! **eqiora-ir** — backend-independent lowered forms.
//!
//! The Semantic Kernel expression DAG remains canonical meaning. Operator IR
//! resolves repeated symbols to dense slots and lowers expression nodes to a
//! compact scalar SSA program suitable for independent CPU/backend execution.

mod calculus;
mod component;
mod linearization;
mod local_action;
mod scalar;

pub use calculus::{
    CalculusBuilder, CalculusError, CalculusNode, CalculusNodeId, ExactRational,
    FormalDimensionMonomial, NormalizationProof, NormalizationRuleId, OperatorApplicationProof,
    OperatorDefinitionDigest, OperatorExpansionExt, PureOperatorApplicationProof,
    PureOperatorDefinition, PureOperatorError, PureOperatorInstantiation, PureValueClass,
    ResultAxis, ScalarCalculus, ScalarCalculusAtom, ScalarCalculusNode, StandardPureOperator,
    SupportMap, SupportMapIntent, SupportMapOrientation, SupportMapPairing, SupportMapViolation,
};
pub use component::{ComponentScalarRow, ComponentScalarization, ScalarSymbolCoordinate};
pub use linearization::{
    DifferentiationRole, DiscreteStepLinearization, LinearizedOutput, LinearizedRelation,
    RelationCotangent, RelationTangent, ScalarObjectiveLinearization,
};
pub use local_action::LocalLinearActionIr;
pub use scalar::{
    BoundAffineFailure, BoundAffineScalarIr, ConstantSymbolJacobian, ScalarInputOperatorIr,
    ScalarInputSlot, ScalarLinearization, ScalarOperatorIr, SymbolicLinearityFailure,
};
