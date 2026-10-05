//! **eqiora-core** — the shared vocabulary of the platform (Layer L0).
//!
//! This crate defines the small set of types every other crate speaks:
//!
//! - [`Id`] / [`entity`] — strongly typed identifiers over the Graph
//!   Federation. Raw strings and integers never travel
//!   through public APIs.
//! - [`ontology`] — schema-typed named subgraphs in a namespace separate from
//!   graph nodes.
//! - [`quantity`] — the two-layer unit system: static
//!   [`quantity::Quantity`] checked at compile time, dynamic
//!   [`quantity::DynQuantity`] entering only at external-data boundaries,
//!   promoted exclusively via `checked_cast`.
//! - [`diagnostic`] — engineering-aware errors: stable codes,
//!   graph paths, source spans, machine-applicable suggestions.
//!
//! L0 rule: this crate depends on external crates only — never on another
//! `eqiora-*` crate. Enforced by `cargo xtask check-layers`.

pub mod diagnostic;
pub mod entity;
pub mod id;
pub mod ontology;
pub mod quantity;
pub mod scalar;
mod time_state_coordinate;
pub use time_state_coordinate::TimeStateCoordinate;
pub mod value_literal;
pub mod value_shape;
pub mod value_type;

pub use diagnostic::{Code, Diagnostic, GraphPath, Severity, Span};
pub use entity::{Entity, EntityKind, GraphClass, GraphKind};
pub use id::{Id, RawId};
pub use ontology::{NamedSubgraph, OntologyId, OntologySchema, OntologyView, RawOntologyId};
pub use quantity::{DimExponents, Dimension, DynQuantity, Quantity, Scalar};
pub use scalar::{ScalarDomain, ScalarType};
pub use value_literal::{InvalidValueLiteral, ValueLiteral};
pub use value_shape::{InvalidValueShape, ValueShape};
pub use value_type::{FiniteBasis, InvalidValueType, ValueFrame, ValueType};
