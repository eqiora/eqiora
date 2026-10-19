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
    CalculusBuilder, CalculusError, CalculusNode, CalculusNodeId, ComponentIndex, ExactRational,
    NormalizationProof, NormalizationRuleId, OperatorApplicationProof, OperatorDefinitionDigest,
    OperatorExpansionExt, PureOperatorApplicationProof, PureOperatorDefinition, PureOperatorError,
    PureValueClass, ScalarCalculus, ScalarCalculusAtom, ScalarCalculusNode, StandardPureOperator,
    SupportMap, SupportMapIntent, SupportMapOrientation, SupportMapPairing, SupportMapViolation,
};
pub use component::{
    ComponentLinearization, ComponentScalarRow, ComponentScalarization, ScalarSymbolCoordinate,
};
pub use linearization::{
    DifferentiationRole, DiscreteStepLinearization, LinearizedOutput, LinearizedRelation,
    RelationCotangent, RelationTangent,
};
mod output;
pub use local_action::LocalLinearActionIr;
pub use output::DenseOutputLinearization;
pub use scalar::{
    BoundAffineFailure, BoundAffineScalarIr, ConstantSymbolJacobian, ScalarAlgebraicNode,
    ScalarExpressionView, ScalarInputSlot, ScalarLinearization, ScalarOperatorIr,
    SymbolicLinearityFailure,
};
