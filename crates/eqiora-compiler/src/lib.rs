//! **eqiora-compiler** — typed lowering from source AST to graph transaction.
//!
//! Parsing and recovery stay in `eqiora-lang`; storage stays in
//! `eqiora-graph`. This crate is the only bridge that resolves source names,
//! checks SI dimensions, constructs inspectable expression DAGs, and emits the
//! same typed transaction available through the handwritten Rust API.

#[doc(hidden)]
pub mod connection_sets;
mod diagnostics;
mod dimensions;
mod external;
pub use external::StaticBindingValue;
pub use property::descriptor::PropertyContractDescriptor;
mod enumeration;
mod external_compile;
mod formulation;
mod hierarchy;
#[doc(hidden)]
pub mod identity;
mod lower;
mod math;
mod nominal;
mod notation;
pub mod projection;
mod property;
#[doc(hidden)]
pub mod provenance;
mod pure_operator;
mod record;
mod resolved;
mod source_compile;
mod source_endpoints;
#[doc(hidden)]
pub mod source_identity;
mod typed_values;
mod units;
mod value_types;

pub use formulation::{
    AuthoredFormExpressionV1, AuthoredFormulationProjection, CompiledAuthoredFormulation,
    check_derived_interval_conservation,
};
pub use lower::{CompiledModel, ModelSymbols, lower_module};
pub use notation::{ModelNotation, QuantityIdentity, QuantityRole, ResolvedNotation};
pub use resolved::{
    AnalyzedResolvedHierarchy, CanonicalDeclarationIdentity, CanonicalDeclarationKind,
    CompilationNamespaceId, ResolvedDependency, ResolvedHierarchyInput, ResolvedSourceUnit,
    ValidatedResolvedHierarchy, analyze_resolved_hierarchy, preflight_resolved_hierarchy,
};
pub use source_compile::compile;

pub use units::InputUnitCatalog;
