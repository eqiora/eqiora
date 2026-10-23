//! Backend-neutral time-system action contracts.

use crate::diagnostic::invalid_lowering;
use eqiora_core::Diagnostic;

/// Parameter dependence admitted for a mass matrix during sensitivity solves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MassParameterDependence {
    /// No proof is available, so mass-matrix sensitivities must fail closed.
    Unspecified,
    /// The lowering proves `M_p dp = 0` throughout the admitted run domain.
    Independent,
}

/// Infallible-shape, fallible-value action for one lowered first-order system.
///
/// Implementations bind all immutable model parameters. Every input and output
/// slice has [`Self::dimension`] entries. Methods must overwrite every output
/// entry. An adapter propagates the first diagnostic and rejects non-finite
/// callback results.
pub trait TimeSystem {
    /// Number of scalar state unknowns.
    fn dimension(&self) -> usize;

    /// Evaluate `f(t, y)`.
    ///
    /// # Errors
    /// Returns a stable diagnostic when the lowered action cannot be evaluated.
    fn rhs(&self, time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic>;

    /// Evaluate the state Jacobian action `f_y(t, y) direction`.
    ///
    /// # Errors
    /// Returns a stable diagnostic when the lowered action cannot be evaluated.
    fn rhs_jvp(
        &self,
        time: f64,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic>;

    /// Evaluate `M(t) direction` for a mass-matrix system.
    ///
    /// ODE adapters do not call this method. The default fails closed so an
    /// absent mass action cannot silently become an identity matrix.
    ///
    /// # Errors
    /// Returns a stable diagnostic when no mass action exists or evaluation fails.
    fn mass_action(
        &self,
        _time: f64,
        _direction: &[f64],
        _output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        Err(invalid_lowering(
            "mass-matrix equation class requires an explicit mass action",
        ))
    }
}

/// Residual and tangent actions for `F(t, y, y_dot) = 0`.
///
/// This is intentionally distinct from [`TimeSystem`]. A residual-native
/// adapter receives both state and derivative values, and its Newton/Krylov
/// iteration forms combinations such as `F_y v + alpha F_y_dot v` without
/// pretending that a state-dependent or nonlinear derivative term is a
/// constant mass matrix.
pub trait ImplicitTimeSystem {
    /// Number of state coordinates and residual equations.
    fn dimension(&self) -> usize;

    /// Evaluate the complete implicit residual.
    ///
    /// # Errors
    /// Returns a stable diagnostic when the lowered action cannot be evaluated.
    fn residual(
        &self,
        time: f64,
        state: &[f64],
        derivative: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic>;

    /// Evaluate `F_y state_direction + F_y_dot derivative_direction`.
    ///
    /// # Errors
    /// Returns a stable diagnostic when the lowered tangent action cannot be
    /// evaluated.
    fn residual_jvp(
        &self,
        time: f64,
        state: &[f64],
        derivative: &[f64],
        state_direction: &[f64],
        derivative_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic>;
}

/// Parameter actions required to integrate continuous forward sensitivities.
///
/// This extends the same lowered primal/JVP system rather than introducing a
/// dual-number simulator or a backend-specific derivative model. Parameters
/// are fixed at one linearization point; the two methods provide `f_p dp` and
/// `y0_p dp` actions for arbitrary parameter directions.
pub trait ParametricTimeSystem: TimeSystem {
    /// Number of scalar model parameters at this linearization point.
    fn parameter_dimension(&self) -> usize;

    /// Finite parameter values defining the primal linearization point.
    fn parameters(&self) -> &[f64];

    /// State whether the mass action is proven independent of parameters.
    ///
    /// Explicit ODE sensitivity does not consult this value. The fail-closed
    /// default prevents a missing `M_p y_dot` term from silently becoming
    /// zero for a mass-matrix system.
    #[must_use]
    fn mass_parameter_dependence(&self) -> MassParameterDependence {
        MassParameterDependence::Unspecified
    }

    /// Evaluate `f_p(t, y) parameter_direction`.
    ///
    /// # Errors
    /// Returns a stable diagnostic when the parameter action cannot be evaluated.
    fn rhs_parameter_jvp(
        &self,
        time: f64,
        state: &[f64],
        parameter_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic>;

    /// Evaluate `y0_p parameter_direction`.
    ///
    /// # Errors
    /// Returns a stable diagnostic when the initial-state action cannot be evaluated.
    fn initial_parameter_jvp(
        &self,
        time: f64,
        parameter_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic>;

    /// Evaluate the Parameter tangent of the accepted initial derivative.
    ///
    /// The initial-data owner supplies this alongside `initial_parameter_jvp`;
    /// a time provider must not infer algebraic coordinates from mass diagonals
    /// or solve a second initial-value problem. Unused derivative coordinates
    /// follow the same convention as the supplied initial pair.
    ///
    /// # Errors
    /// Returns a stable diagnostic for an invalid or unproved initial tangent.
    fn initial_derivative_parameter_jvp(
        &self,
        time: f64,
        initial: &crate::ImplicitDaeInitialization,
        parameter_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic>;
}

/// Zero-crossing actions used only to propose candidate event instants.
///
/// A backend may localize the first sign change, but it does not decide event
/// direction, simultaneous grouping, priority, or reset commit semantics.
pub trait RootFunctions {
    /// Number of scalar root functions.
    fn count(&self) -> usize;

    /// Evaluate every root function at one state/time point.
    ///
    /// # Errors
    /// Returns a stable diagnostic when root evaluation fails.
    fn evaluate(&self, time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic>;
}
