use std::fmt::Debug;

use eqiora_core::Diagnostic;

/// Role of one lowered operator input in a particular differentiation.
///
/// Roles are analysis choices rather than properties inferred from Semantic
/// Kernel entity kinds. Dense unknown and parameter coordinates retain the
/// first-occurrence order of their input slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentiationRole {
    /// Coordinate of the implicitly solved value vector `w`.
    Unknown,
    /// Coordinate of the selected design vector `p`.
    Parameter,
    /// Value fixed at the linearization point for this analysis.
    Frozen,
}

/// Selected tangent in the direct sum of unknown and parameter spaces.
#[derive(Debug, Clone, Copy)]
pub enum RelationTangent<'a, S> {
    /// Vary only the implicitly solved coordinates; `dp = 0`.
    Unknown(&'a [S]),
    /// Vary only selected design coordinates; `dw = 0`.
    Parameter(&'a [S]),
    /// Vary both coordinate groups in one JVP.
    Both {
        /// Unknown-space tangent `dw`.
        unknown: &'a [S],
        /// Parameter-space tangent `dp`.
        parameter: &'a [S],
    },
}

/// Requested projection of a relation VJP.
#[derive(Debug)]
pub enum RelationCotangent<'a, S> {
    /// Return only `R_w^T * c`.
    Unknown(&'a mut [S]),
    /// Return only `R_p^T * c`.
    Parameter(&'a mut [S]),
    /// Return both cotangent components in one reverse pass.
    Both {
        /// Unknown-space cotangent output.
        unknown: &'a mut [S],
        /// Parameter-space cotangent output.
        parameter: &'a mut [S],
    },
}

/// One immutable linearization of a residual relation.
///
/// The contract is scalar-representation-parametric. Implementations expose
/// the primal residual, a Jacobian-vector product (JVP), and the paired
/// vector-Jacobian product (VJP) without prescribing symbolic, automatic, or
/// handwritten differentiation.
///
/// Real-coordinate implementations use the Euclidean pairing. Complex Model
/// values lowered through [`crate::ComponentScalarization`] use paired real/imaginary
/// coordinates, equivalent to `Re(sum(conj(a) * b))`; their differential may be
/// real-linear without being complex-linear. A provider must state its scalar
/// representation and pairing before implementing these products. Implementing
/// this trait for real storage does not by itself establish complex semantics.
pub trait LinearizedRelation<S>: Debug + Sync {
    /// Number of implicitly solved coordinates.
    fn unknown_dimension(&self) -> usize;

    /// Number of selected design coordinates.
    fn parameter_dimension(&self) -> usize;

    /// Number of residual equations.
    fn residual_dimension(&self) -> usize;

    /// Evaluate the primal residual at the fixed linearization point.
    ///
    /// # Errors
    /// Returns a structured diagnostic for an output-shape mismatch or an
    /// invalid/non-finite evaluation.
    fn primal(&self, residual: &mut [S]) -> Result<(), Diagnostic>;

    /// Evaluate a selected tangent action, including
    /// `R_w * unknown_tangent + R_p * parameter_tangent`.
    ///
    /// # Errors
    /// Returns a structured diagnostic for a shape mismatch or an
    /// invalid/non-finite evaluation.
    fn jvp(
        &self,
        tangent: RelationTangent<'_, S>,
        residual_tangent: &mut [S],
    ) -> Result<(), Diagnostic>;

    /// Evaluate the paired adjoints for one residual cotangent `c`.
    /// In real coordinates these are `(R_w^T * c, R_p^T * c)`; this is not
    /// an unconjugated complex transpose or an implicit metric adjoint.
    ///
    /// # Errors
    /// Returns a structured diagnostic for a shape mismatch or an
    /// invalid/non-finite evaluation.
    fn vjp(
        &self,
        residual_cotangent: &[S],
        cotangent: RelationCotangent<'_, S>,
    ) -> Result<(), Diagnostic>;
}

/// One immutable linearization of a selected output projection `y = O(w, p)`.
///
/// A relation linearization describes how the accepted implicit state moves;
/// this paired contract describes what an application actually publishes.
/// Keeping the two separate prevents method-native algebraic unknowns from
/// being mistaken for complete semantic Field values. Implementations provide
/// both actions from the same primal projection, including any direct
/// Parameter dependence of eliminated constraints or derived outputs.
pub trait LinearizedOutput<S>: Debug + Sync {
    /// Number of implicit-state coordinates consumed by the projection.
    fn unknown_dimension(&self) -> usize;

    /// Number of selected design coordinates consumed by the projection.
    fn parameter_dimension(&self) -> usize;

    /// Number of published output coordinates.
    fn output_dimension(&self) -> usize;

    /// Evaluate the output at the fixed accepted point.
    ///
    /// # Errors
    /// Returns a structured diagnostic for a shape mismatch or non-finite
    /// output.
    fn primal(&self, output: &mut [S]) -> Result<(), Diagnostic>;

    /// Evaluate `O_w * dw + O_p * dp`.
    ///
    /// # Errors
    /// Returns a structured diagnostic for a shape mismatch or non-finite
    /// action.
    fn jvp(
        &self,
        unknown_tangent: &[S],
        parameter_tangent: &[S],
        output_tangent: &mut [S],
    ) -> Result<(), Diagnostic>;

    /// Evaluate `(O_w^T * c, O_p^T * c)` in one paired reverse action.
    ///
    /// # Errors
    /// Returns a structured diagnostic for a shape mismatch or non-finite
    /// action.
    fn vjp(
        &self,
        output_cotangent: &[S],
        unknown_cotangent: &mut [S],
        parameter_cotangent: &mut [S],
    ) -> Result<(), Diagnostic>;
}

/// One accepted discrete step with an explicit previous-state/model-Parameter
/// direct-sum layout.
///
/// This metadata lets trajectory analyses validate state, time, and design
/// coordinate continuity before composing any adjoint solves. It does not
/// prescribe a time integration method.
pub trait DiscreteStepLinearization: LinearizedRelation<f64> {
    /// Exact source/realization coordinate identity, including spatial support when present.
    type StateCoordinate: Eq;
    /// Exact real coordinate of a selected model Parameter.
    type ModelParameterCoordinate: Eq;
    /// Complete state projection, including eliminated constraints.
    type StateOutput: LinearizedOutput<f64> + ?Sized;

    /// Projection from implicit unknowns and step Parameters to the next state.
    fn state_output(&self) -> &Self::StateOutput;

    /// Canonical state coordinate order shared by previous and next states.
    fn state_coordinates(&self) -> &[Self::StateCoordinate];

    /// Canonical model Parameter coordinates following the previous-state block.
    fn model_parameter_coordinates(&self) -> &[Self::ModelParameterCoordinate];

    /// Revision-local model Parameter point.
    fn model_parameter_values(&self) -> &[f64];

    /// Accepted previous state at this step boundary.
    fn previous_state(&self) -> &[f64];

    /// Accepted next state at this step boundary.
    fn next_state(&self) -> &[f64];

    /// Frozen start time of this discrete relation.
    fn start_time(&self) -> f64;

    /// Frozen end time of this discrete relation.
    fn end_time(&self) -> f64;

    /// Number of leading Parameter coordinates occupied by previous state.
    fn previous_state_parameter_dimension(&self) -> usize;
}
