//! **eqiora-sem** — the reference interpreter for the semantic kernel
//! (Layer L2).
//!
//! This crate is the **semantic oracle**: small, slow, obviously correct.
//! The normative meaning of an Eqiora program is "the language spec plus the
//! behavior of this interpreter". Every optimized execution path — compiler,
//! GPU backends, distributed runtime, fixed-step codegen — must match this
//! crate within the numerical contract, never the other way around.
//!
//! Standing rules for contributors:
//!
//! - **No optimizations. Ever.** Clarity beats speed here by design; the
//!   target size is a few thousand lines one person can read end to end.
//! - Kernel concepts only: Domain, Representation, Field, Parameter,
//!   Port, Relation, Activation, Connection, ClockDomain. Standard-ontology
//!   named-subgraph views select and organize kernel nodes but are ignored by
//!   interpretation after that selection; they never acquire node semantics.
//! - When this crate and the spec text disagree, an RFC decides which is
//!   right, and both get fixed.

mod boundary_physical;
mod conserving;
mod evaluate;
mod event;
mod interpreter;
mod program;
mod solver;

pub use boundary_physical::{BoundaryJunctionGeometry, BoundaryJunctionResidual};
pub use conserving::{
    ComposedResidualSystem, PhysicalUnknown, ResidualGroup, ScalarPhysicalSubsystemId,
};
pub use evaluate::{EvaluationInput, EvaluationPoint};
pub use interpreter::{
    EquationAnalysis, ExecutionObserver, ExecutionOutcome, ExecutionProgress, ExecutionSession,
    IncidenceMatching, InitialState, ReferenceConfig,
};
pub use program::KernelProgram;

use eqiora_core::Diagnostic;
use eqiora_core::{DynQuantity, RawId};
use eqiora_schema::kernel::{ExprDag, SymbolRef};

/// Expression-evaluation hook used by conformance backends.
///
/// [`Interpreter::run`] always uses the canonical DAG evaluator. The separate
/// hook lets an independently lowered Operator IR reuse the exact activation
/// calendar and reference numerics while proving expression conformance.
#[doc(hidden)]
pub trait ExpressionBackend {
    /// Evaluate the requested expression roots using complete typed symbol values.
    /// Unreachable nodes and symbols are not evaluated; a discrete assignment
    /// requests its right-hand side without computing a numerical residual.
    fn evaluate(
        &self,
        owner: RawId,
        expression: &ExprDag,
        roots: &[eqiora_schema::kernel::ExprId],
        resolve: &mut dyn FnMut(SymbolRef) -> Option<eqiora_core::ValueLiteral>,
    ) -> Result<Vec<eqiora_core::ValueLiteral>, Diagnostic>;
}

/// One dimensioned observation at a model time and an exact typed coordinate.
/// Field observations use `RawId`; physical observations use `PhysicalUnknown`.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample<Coordinate> {
    time: f64,
    coordinate: Coordinate,
    value: DynQuantity,
}

impl<Coordinate: Copy> Sample<Coordinate> {
    pub(crate) const fn new(time: f64, coordinate: Coordinate, value: DynQuantity) -> Self {
        Self {
            time,
            coordinate,
            value,
        }
    }

    /// Model time in seconds.
    #[must_use]
    pub const fn time(&self) -> f64 {
        self.time
    }

    /// Exact Field identity or physical across/through coordinate.
    #[must_use]
    pub const fn coordinate(&self) -> Coordinate {
        self.coordinate
    }

    /// Accepted coherent-SI value with its physical dimension.
    #[must_use]
    pub const fn value(&self) -> DynQuantity {
        self.value
    }
}

/// Reference trajectory: the ground truth that conformance models
/// compare every optimized execution against.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Trajectory {
    /// Samples in non-decreasing time order.
    samples: Vec<Sample<RawId>>,
    /// Accepted scalar physical algebraic observations.
    physical_samples: Vec<Sample<PhysicalUnknown>>,
}

impl Trajectory {
    pub(crate) const fn new(
        samples: Vec<Sample<RawId>>,
        physical_samples: Vec<Sample<PhysicalUnknown>>,
    ) -> Self {
        Self {
            samples,
            physical_samples,
        }
    }

    /// Samples in non-decreasing model-time order.
    #[must_use]
    pub fn samples(&self) -> &[Sample<RawId>] {
        &self.samples
    }

    /// Scalar physical values at accepted trajectory boundaries.
    #[must_use]
    pub fn physical_samples(&self) -> &[Sample<PhysicalUnknown>] {
        &self.physical_samples
    }

    /// Last recorded value for one Field.
    #[must_use]
    pub fn last_value(&self, field: RawId) -> Option<DynQuantity> {
        self.samples
            .iter()
            .rev()
            .find(|sample| sample.coordinate == field)
            .map(|sample| sample.value)
    }

    /// Last accepted value for one physical across or through slot.
    #[must_use]
    pub fn last_physical_value(&self, unknown: PhysicalUnknown) -> Option<DynQuantity> {
        self.physical_samples
            .iter()
            .rev()
            .find(|sample| sample.coordinate == unknown)
            .map(|sample| sample.value)
    }
}

/// The reference interpreter.
///
/// Construction is trivial on purpose: the interpreter has no configuration
/// that could make "the meaning of a program" ambiguous.
#[derive(Debug, Default)]
pub struct Interpreter {
    _private: (),
}

impl Interpreter {
    /// Create the interpreter.
    #[must_use]
    pub fn new() -> Self {
        Self { _private: () }
    }
}
