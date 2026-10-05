use std::fmt::Debug;

use eqiora_core::Diagnostic;
use eqiora_core::Id;
use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;

/// Accepted-point linearization of one scalar objective functional.
///
/// This is the objective-side counterpart of [`LinearizedRelation`]. The
/// value and both cotangents are produced by one lowering, so adjoint analysis
/// does not receive three unrelated raw values. Accepted-point identity is a
/// provenance/artifact concern; the numerical contract validates finiteness
/// here and relation dimensions at composition time.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarObjectiveLinearization {
    value: f64,
    unknown_cotangent: Vec<f64>,
    parameter_cotangent: Vec<f64>,
}

impl ScalarObjectiveLinearization {
    /// Construct one finite objective value with its `J_w` and direct `J_p`.
    ///
    /// Relation compatibility is checked when the objective is composed with
    /// an accepted relation.
    ///
    /// # Errors
    /// Returns `EQ0704` if any value is non-finite.
    pub fn new(
        value: f64,
        unknown_cotangent: Vec<f64>,
        parameter_cotangent: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        if !value.is_finite()
            || unknown_cotangent
                .iter()
                .chain(&parameter_cotangent)
                .any(|entry| !entry.is_finite())
        {
            return Err(Diagnostic::error(
                codes::INVALID_LINEARIZATION,
                "scalar objective linearization requires finite value and cotangents",
            ));
        }
        Ok(Self {
            value,
            unknown_cotangent,
            parameter_cotangent,
        })
    }

    /// Objective value at the accepted linearization point.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// Objective-state cotangent `J_w` in relation unknown order.
    #[must_use]
    pub fn unknown_cotangent(&self) -> &[f64] {
        &self.unknown_cotangent
    }

    /// Direct objective-design cotangent `J_p` in relation Parameter order.
    #[must_use]
    pub fn parameter_cotangent(&self) -> &[f64] {
        &self.parameter_cotangent
    }
}

impl LinearizedOutput<f64> for ScalarObjectiveLinearization {
    fn unknown_dimension(&self) -> usize {
        self.unknown_cotangent.len()
    }

    fn parameter_dimension(&self) -> usize {
        self.parameter_cotangent.len()
    }

    fn output_dimension(&self) -> usize {
        1
    }

    fn primal(&self, output: &mut [f64]) -> Result<(), Diagnostic> {
        if output.len() != 1 {
            return Err(invalid_linearization(
                "scalar objective primal requires one output value",
            ));
        }
        output[0] = self.value;
        Ok(())
    }

    fn jvp(
        &self,
        unknown_tangent: &[f64],
        parameter_tangent: &[f64],
        output_tangent: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if unknown_tangent.len() != self.unknown_cotangent.len()
            || parameter_tangent.len() != self.parameter_cotangent.len()
            || output_tangent.len() != 1
            || unknown_tangent
                .iter()
                .chain(parameter_tangent)
                .any(|value| !value.is_finite())
        {
            return Err(invalid_linearization(
                "scalar objective JVP has incompatible shape or values",
            ));
        }
        output_tangent[0] = self
            .unknown_cotangent
            .iter()
            .zip(unknown_tangent)
            .chain(self.parameter_cotangent.iter().zip(parameter_tangent))
            .map(|(cotangent, tangent)| cotangent * tangent)
            .sum();
        if output_tangent[0].is_finite() {
            Ok(())
        } else {
            Err(invalid_linearization(
                "scalar objective JVP produced a non-finite value",
            ))
        }
    }

    fn vjp(
        &self,
        output_cotangent: &[f64],
        unknown_cotangent: &mut [f64],
        parameter_cotangent: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if output_cotangent.len() != 1
            || unknown_cotangent.len() != self.unknown_cotangent.len()
            || parameter_cotangent.len() != self.parameter_cotangent.len()
            || !output_cotangent[0].is_finite()
        {
            return Err(invalid_linearization(
                "scalar objective VJP has incompatible shape or values",
            ));
        }
        for (output, derivative) in unknown_cotangent.iter_mut().zip(&self.unknown_cotangent) {
            *output = output_cotangent[0] * derivative;
        }
        for (output, derivative) in parameter_cotangent
            .iter_mut()
            .zip(&self.parameter_cotangent)
        {
            *output = output_cotangent[0] * derivative;
        }
        if unknown_cotangent
            .iter()
            .chain(parameter_cotangent.iter())
            .all(|value| value.is_finite())
        {
            Ok(())
        } else {
            Err(invalid_linearization(
                "scalar objective VJP produced a non-finite value",
            ))
        }
    }
}

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

    /// Evaluate `(R_w^T * c, R_p^T * c)` for one residual cotangent `c`.
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
    /// Canonical state coordinate order shared by previous and next states.
    fn state_coordinates(&self) -> &[(Id<kinds::Field>, u32)];

    /// Canonical model Parameter order following the previous-state block.
    fn model_parameter_fields(&self) -> &[Id<kinds::Parameter>];

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

fn invalid_linearization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_LINEARIZATION, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_objective_is_one_paired_output_projection() {
        let objective = ScalarObjectiveLinearization::new(11.0, vec![2.0, 3.0], vec![5.0])
            .expect("finite objective");
        let mut primal = [0.0];
        objective.primal(&mut primal).unwrap();
        assert_eq!(primal, [11.0]);

        let mut tangent = [0.0];
        objective.jvp(&[7.0, 11.0], &[13.0], &mut tangent).unwrap();
        assert_eq!(tangent, [112.0]);

        let mut state = [0.0; 2];
        let mut parameter = [0.0];
        objective.vjp(&[17.0], &mut state, &mut parameter).unwrap();
        assert_eq!(state, [34.0, 51.0]);
        assert_eq!(parameter, [85.0]);
        assert_eq!(
            tangent[0] * 17.0,
            7.0 * state[0] + 11.0 * state[1] + 13.0 * parameter[0]
        );

        let overflowing =
            ScalarObjectiveLinearization::new(0.0, vec![f64::MAX], Vec::new()).unwrap();
        assert!(overflowing.jvp(&[2.0], &[], &mut [0.0]).is_err());
        assert!(overflowing.vjp(&[2.0], &mut [0.0], &mut []).is_err());
    }
}
