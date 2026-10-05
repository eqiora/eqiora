//! Validated problem bindings and initial-condition contracts.

use crate::diagnostic::{invalid_lowering, invalid_sensitivity, time_solve_failed};
use crate::lowering::{DaeVariableKind, MassMatrixRank, TimeEquationClass};
use crate::system::{
    ImplicitTimeSystem, MassParameterDependence, ParametricTimeSystem, TimeSystem,
};
use eqiora_core::Diagnostic;

/// Meaning of the data supplied at the initial model time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InitialConditionPolicy {
    /// The supplied state, and derivative when present, are already accepted.
    Provided,
    /// The supplied state/derivative data are guesses for a consistency solve.
    SolveConsistent,
}

/// Validated, parameter-bound first-order problem.
///
/// This type cannot contain a general implicit DAE: its action vocabulary is
/// intentionally only the admitted `M y_dot = f` projection.
pub struct TimeProblem<'a> {
    system: &'a dyn TimeSystem,
    equation_class: TimeEquationClass,
    initial_condition: InitialConditionPolicy,
    initial_state: Vec<f64>,
}

impl std::fmt::Debug for TimeProblem<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TimeProblem")
            .field("dimension", &self.dimension())
            .field("equation_class", &self.equation_class)
            .field("initial_condition", &self.initial_condition)
            .field("initial_state", &self.initial_state)
            .finish_non_exhaustive()
    }
}

impl<'a> TimeProblem<'a> {
    /// Bind a lowered system to its equation class and initial state/guess.
    ///
    /// # Errors
    /// Returns `EQ0705` for empty/non-finite state, a general residual passed
    /// through the first-order seam, or a contradictory consistency policy.
    pub fn new(
        system: &'a dyn TimeSystem,
        equation_class: TimeEquationClass,
        initial_condition: InitialConditionPolicy,
        initial_state: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        let dimension = system.dimension();
        if dimension == 0 || initial_state.len() != dimension {
            return Err(invalid_lowering(
                "time problem state must have the non-zero system dimension",
            ));
        }
        if initial_state.iter().any(|value| !value.is_finite()) {
            return Err(invalid_lowering(
                "time problem initial state must contain only finite values",
            ));
        }
        match (equation_class, initial_condition) {
            (TimeEquationClass::GeneralImplicitDae, _) => {
                return Err(invalid_lowering(
                    "general implicit DAE cannot enter the mass-matrix time-problem seam",
                ));
            }
            (TimeEquationClass::ExplicitOde, InitialConditionPolicy::SolveConsistent) => {
                return Err(invalid_lowering(
                    "explicit ODE initial state is provided directly; no algebraic consistency solve applies",
                ));
            }
            (
                TimeEquationClass::MassMatrix {
                    rank: MassMatrixRank::RankDeficient,
                },
                InitialConditionPolicy::Provided,
            ) => {
                return Err(invalid_lowering(
                    "rank-deficient mass matrix requires a consistent-initialization solve",
                ));
            }
            _ => {}
        }
        Ok(Self {
            system,
            equation_class,
            initial_condition,
            initial_state,
        })
    }

    /// Lowered callback actions.
    #[must_use]
    pub const fn system(&self) -> &dyn TimeSystem {
        self.system
    }

    /// Scalar state dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.initial_state.len()
    }

    /// Declared equation class.
    #[must_use]
    pub const fn equation_class(&self) -> TimeEquationClass {
        self.equation_class
    }

    /// Declared initial-condition treatment.
    #[must_use]
    pub const fn initial_condition(&self) -> InitialConditionPolicy {
        self.initial_condition
    }

    /// Provided initial state or consistency-solve guess.
    #[must_use]
    pub fn initial_state(&self) -> &[f64] {
        &self.initial_state
    }

    /// Evaluate the physical rate, solving `M y_dot = f` for full-rank mass.
    ///
    /// # Errors
    /// Rejects invalid points, singular mass actions and non-finite arithmetic.
    /// Rank-deficient systems require a residual consistency solve instead.
    pub fn rate(&self, time: f64, state: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        let n = self.dimension();
        if !time.is_finite() || state.len() != n || state.iter().any(|x| !x.is_finite()) {
            return Err(invalid_lowering(
                "rate evaluation requires a finite complete state",
            ));
        }
        if !matches!(
            self.equation_class,
            TimeEquationClass::ExplicitOde
                | TimeEquationClass::MassMatrix {
                    rank: MassMatrixRank::Full
                }
        ) {
            return Err(invalid_lowering(
                "rate evaluation requires a uniquely determined derivative",
            ));
        }
        let mut rhs = vec![0.; n];
        self.system.rhs(time, state, &mut rhs)?;
        if rhs.iter().any(|x| !x.is_finite()) {
            return Err(time_solve_failed("rate right-hand side is non-finite"));
        }
        if self.equation_class == TimeEquationClass::ExplicitOde {
            return Ok(rhs);
        }
        let size = n
            .checked_mul(n)
            .ok_or_else(|| invalid_lowering("mass matrix size overflows"))?;
        let mut matrix = vec![0.; size];
        let mut direction = vec![0.; n];
        let mut column = vec![0.; n];
        for j in 0..n {
            direction[j] = 1.;
            self.system.mass_action(time, &direction, &mut column)?;
            if column.iter().any(|x| !x.is_finite()) {
                return Err(time_solve_failed("rate mass action is non-finite"));
            }
            for i in 0..n {
                matrix[i * n + j] = column[i];
            }
            direction[j] = 0.;
        }
        crate::reference_implicit::solve_dense(matrix, rhs)
    }

    /// Rebind the same lowered system/class after an Eqiora-owned reset.
    ///
    /// The caller, normally the hybrid scheduler, determines atomic reset
    /// meaning and passes only the committed post-event state here.
    ///
    /// # Errors
    /// Returns the same validation diagnostics as [`Self::new`].
    pub fn restart(
        &'a self,
        initial_condition: InitialConditionPolicy,
        initial_state: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        Self::new(
            self.system,
            self.equation_class,
            initial_condition,
            initial_state,
        )
    }
}

/// Validated general residual problem with an explicit variable partition.
///
/// For [`InitialConditionPolicy::SolveConsistent`], a residual-native adapter
/// holds differential state coordinates and algebraic derivative coordinates
/// fixed while solving for algebraic states and differential derivatives.
/// This is the semi-explicit index-one initialization contract; it is not a
/// promise that every general residual has index one.
pub struct ImplicitDaeProblem<'a> {
    system: &'a dyn ImplicitTimeSystem,
    variable_kinds: Vec<DaeVariableKind>,
    initial_condition: InitialConditionPolicy,
    initial_state: Vec<f64>,
    initial_derivative: Vec<f64>,
}

impl std::fmt::Debug for ImplicitDaeProblem<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ImplicitDaeProblem")
            .field("dimension", &self.dimension())
            .field("variable_kinds", &self.variable_kinds)
            .field("initial_condition", &self.initial_condition)
            .field("initial_state", &self.initial_state)
            .field("initial_derivative", &self.initial_derivative)
            .finish_non_exhaustive()
    }
}

impl<'a> ImplicitDaeProblem<'a> {
    /// Bind residual actions to one variable partition and initial pair/guess.
    ///
    /// # Errors
    /// Returns `EQ0705` for empty/mismatched/non-finite vectors or a system
    /// with no differential coordinate. `Provided` means the complete pair is
    /// already consistent; `SolveConsistent` treats both vectors as guesses.
    pub fn new(
        system: &'a dyn ImplicitTimeSystem,
        variable_kinds: Vec<DaeVariableKind>,
        initial_condition: InitialConditionPolicy,
        initial_state: Vec<f64>,
        initial_derivative: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        let dimension = system.dimension();
        if dimension == 0
            || variable_kinds.len() != dimension
            || initial_state.len() != dimension
            || initial_derivative.len() != dimension
        {
            return Err(invalid_lowering(
                "implicit DAE problem requires matching non-empty residual, partition, state, and derivative shapes",
            ));
        }
        if !variable_kinds.contains(&DaeVariableKind::Differential) {
            return Err(invalid_lowering(
                "implicit DAE problem requires at least one differential coordinate",
            ));
        }
        if initial_state
            .iter()
            .chain(&initial_derivative)
            .any(|value| !value.is_finite())
        {
            return Err(invalid_lowering(
                "implicit DAE initial state and derivative must contain only finite values",
            ));
        }
        Ok(Self {
            system,
            variable_kinds,
            initial_condition,
            initial_state,
            initial_derivative,
        })
    }

    /// Residual and JVP action provider.
    #[must_use]
    pub const fn system(&self) -> &dyn ImplicitTimeSystem {
        self.system
    }

    /// Scalar state/residual dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.initial_state.len()
    }

    /// Differential/algebraic role in state coordinate order.
    #[must_use]
    pub fn variable_kinds(&self) -> &[DaeVariableKind] {
        &self.variable_kinds
    }

    /// Initial-condition treatment requested from the adapter.
    #[must_use]
    pub const fn initial_condition(&self) -> InitialConditionPolicy {
        self.initial_condition
    }

    /// Initial state or consistency-solve guess.
    #[must_use]
    pub fn initial_state(&self) -> &[f64] {
        &self.initial_state
    }

    /// Initial derivative or consistency-solve guess.
    #[must_use]
    pub fn initial_derivative(&self) -> &[f64] {
        &self.initial_derivative
    }
}

/// Owned, shape-checked initial pair accepted by a residual-native adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct ImplicitDaeInitialization {
    state: Vec<f64>,
    derivative: Vec<f64>,
}

impl ImplicitDaeInitialization {
    /// Accept one finite, equally shaped, non-empty `(y0, y_dot0)` pair.
    ///
    /// # Errors
    /// Returns `EQ0802` for an invalid shape or non-finite value.
    pub fn accepted(state: Vec<f64>, derivative: Vec<f64>) -> Result<Self, Diagnostic> {
        if state.is_empty()
            || state.len() != derivative.len()
            || state
                .iter()
                .chain(&derivative)
                .any(|value| !value.is_finite())
        {
            return Err(time_solve_failed(
                "implicit DAE backend returned an invalid initial state/derivative pair",
            ));
        }
        Ok(Self { state, derivative })
    }

    /// Accepted initial state.
    #[must_use]
    pub fn state(&self) -> &[f64] {
        &self.state
    }

    /// Accepted initial time derivative.
    #[must_use]
    pub fn derivative(&self) -> &[f64] {
        &self.derivative
    }
}

/// One validated primal problem plus continuous parameter-JVP actions.
pub struct ForwardSensitivityProblem<'a> {
    primal: TimeProblem<'a>,
    system: &'a dyn ParametricTimeSystem,
}

impl std::fmt::Debug for ForwardSensitivityProblem<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ForwardSensitivityProblem")
            .field("primal", &self.primal)
            .field("parameter_dimension", &self.parameter_dimension())
            .field("parameters", &self.parameters())
            .finish_non_exhaustive()
    }
}

impl<'a> ForwardSensitivityProblem<'a> {
    /// Bind a parametric system and validate its primal/parameter point.
    ///
    /// # Errors
    /// Returns the primal lowering diagnostic or `EQ0704` for an empty,
    /// mismatched, or non-finite parameter point.
    pub fn new<S>(
        system: &'a S,
        equation_class: TimeEquationClass,
        initial_condition: InitialConditionPolicy,
        initial_state: Vec<f64>,
    ) -> Result<Self, Diagnostic>
    where
        S: ParametricTimeSystem,
    {
        let parameter_dimension = system.parameter_dimension();
        if parameter_dimension == 0
            || system.parameters().len() != parameter_dimension
            || system.parameters().iter().any(|value| !value.is_finite())
        {
            return Err(invalid_sensitivity(
                "forward time sensitivity requires a non-empty finite parameter point with its declared dimension",
            ));
        }
        if matches!(equation_class, TimeEquationClass::MassMatrix { .. })
            && system.mass_parameter_dependence() != MassParameterDependence::Independent
        {
            return Err(invalid_sensitivity(
                "mass-matrix sensitivity requires an explicit proof that the mass action is parameter independent",
            ));
        }
        let primal = TimeProblem::new(system, equation_class, initial_condition, initial_state)?;
        Ok(Self { primal, system })
    }

    /// Primal lowered time problem.
    #[must_use]
    pub const fn primal(&self) -> &TimeProblem<'a> {
        &self.primal
    }

    /// Parametric action provider.
    #[must_use]
    pub const fn system(&self) -> &dyn ParametricTimeSystem {
        self.system
    }

    /// Parameter dimension.
    #[must_use]
    pub fn parameter_dimension(&self) -> usize {
        self.system.parameter_dimension()
    }

    /// Bound finite parameter point.
    #[must_use]
    pub fn parameters(&self) -> &[f64] {
        self.system.parameters()
    }
}
