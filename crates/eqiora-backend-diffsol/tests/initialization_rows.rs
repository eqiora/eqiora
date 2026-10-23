#![cfg(feature = "diffsol-runtime")]

use eqiora_backend_diffsol::DiffsolTimeBackend;
use eqiora_core::Diagnostic;
use eqiora_time::{
    ForwardSensitivityPlan, ForwardSensitivityProblem, ImplicitDaeInitialization, MassMatrixRank,
    MassParameterDependence, ParametricTimeSystem, TimeEquationClass, TimeMethod, TimePlan,
    TimeProblem, TimeSystem,
};

struct RowTransformedSystem {
    constrained: bool,
    rows: [[f64; 2]; 2],
}

impl RowTransformedSystem {
    fn transform(&self, values: [f64; 2], output: &mut [f64]) {
        for (row, value) in self.rows.iter().zip(output) {
            *value = row[0] * values[0] + row[1] * values[1];
        }
    }
}

impl TimeSystem for RowTransformedSystem {
    fn dimension(&self) -> usize {
        2
    }

    fn rhs(&self, _time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        let values = if self.constrained {
            [-state[1], state[0] * state[0] - state[1]]
        } else {
            [-state[0], -state[1]]
        };
        self.transform(values, output);
        Ok(())
    }

    fn rhs_jvp(
        &self,
        _time: f64,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        let values = if self.constrained {
            [-direction[1], 2.0 * state[0] * direction[0] - direction[1]]
        } else {
            [-direction[0], -direction[1]]
        };
        self.transform(values, output);
        Ok(())
    }

    fn mass_action(
        &self,
        _time: f64,
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.transform(
            [
                direction[0],
                if self.constrained { 0.0 } else { direction[1] },
            ],
            output,
        );
        Ok(())
    }
}

fn check_rows(constrained: bool, rows: [[f64; 2]; 2]) {
    let system = RowTransformedSystem { constrained, rows };
    let (rank, initial, derivative) = if constrained {
        (MassMatrixRank::RankDeficient, vec![1.0, 1.0], [-1.0, 0.0])
    } else {
        (MassMatrixRank::Full, vec![1.0, 2.0], [-1.0, -2.0])
    };
    // The same exact consistent pair satisfies every original residual row.
    let mut rhs = [0.0; 2];
    let mut mass = [0.0; 2];
    system.rhs(0.0, &initial, &mut rhs).unwrap();
    system.mass_action(0.0, &derivative, &mut mass).unwrap();
    assert_eq!(rhs, mass);
    let problem = TimeProblem::new(
        &system,
        TimeEquationClass::MassMatrix { rank },
        ImplicitDaeInitialization::accepted(initial, derivative.to_vec()).unwrap(),
    )
    .unwrap();
    let plan = TimePlan::new(TimeMethod::Bdf, 0.0, 1e-4, 1e-8, vec![1e-10; 2], vec![0.1]).unwrap();
    let solution = DiffsolTimeBackend::new().solve(&problem, &plan).unwrap();
    let actual = solution.state(0).unwrap();
    // Independently: u'=-u², z=u², or y'=-y in the full-rank case.
    let expected = if constrained {
        [1.0 / 1.1, 1.0 / 1.21]
    } else {
        [(-0.1_f64).exp(), 2.0 * (-0.1_f64).exp()]
    };
    for (value, reference) in actual.iter().zip(expected) {
        assert!((value - reference).abs() < 1e-6);
    }
    if constrained {
        assert!((actual[1] - actual[0] * actual[0]).abs() < 1e-7);
    }
}

#[test]
fn full_rank_initialization_preserves_invertible_residual_row_changes() {
    check_rows(false, [[1.0, 0.0], [0.0, 1.0]]);
    check_rows(false, [[0.0, 1.0], [1.0, 0.0]]);
}

#[test]
fn constrained_initialization_preserves_invertible_residual_row_changes() {
    check_rows(true, [[1.0, 0.0], [0.0, 1.0]]);
    check_rows(true, [[1.0, 1.0], [1.0, -1.0]]);
}

// Parameter p multiplies the differential decay; these callbacks evaluate p=1.
impl ParametricTimeSystem for RowTransformedSystem {
    fn parameter_dimension(&self) -> usize {
        1
    }
    fn parameters(&self) -> &[f64] {
        &[1.0]
    }
    fn mass_parameter_dependence(&self) -> MassParameterDependence {
        MassParameterDependence::Independent
    }
    fn rhs_parameter_jvp(
        &self,
        _time: f64,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        let values = if self.constrained {
            [-state[1] * direction[0], 0.0]
        } else {
            [-state[0] * direction[0], -state[1] * direction[0]]
        };
        self.transform(values, output);
        Ok(())
    }
    fn initial_parameter_jvp(
        &self,
        _time: f64,
        _direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output.fill(0.0);
        Ok(())
    }

    fn initial_derivative_parameter_jvp(
        &self,
        _time: f64,
        initial: &eqiora_time::ImplicitDaeInitialization,
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if self.constrained {
            output[0] = -initial.state()[1] * direction[0];
            output[1] = 0.0;
        } else {
            output[0] = -initial.state()[0] * direction[0];
            output[1] = -initial.state()[1] * direction[0];
        }
        Ok(())
    }
}

fn check_parameter_rows(constrained: bool, rows: [[f64; 2]; 2]) {
    let system = RowTransformedSystem { constrained, rows };
    let (rank, state, derivative) = if constrained {
        (
            MassMatrixRank::RankDeficient,
            vec![1.0, 1.0],
            vec![-1.0, 0.0],
        )
    } else {
        (MassMatrixRank::Full, vec![1.0, 2.0], vec![-1.0, -2.0])
    };
    let problem = ForwardSensitivityProblem::new(
        &system,
        TimeEquationClass::MassMatrix { rank },
        ImplicitDaeInitialization::accepted(state, derivative).unwrap(),
    )
    .unwrap();
    let plan = TimePlan::new(TimeMethod::Bdf, 0.0, 1e-4, 1e-8, vec![1e-10; 2], vec![0.1]).unwrap();
    let controls = ForwardSensitivityPlan::new(1e-8, vec![1e-10; 2]).unwrap();
    let result = DiffsolTimeBackend::new()
        .solve_forward_sensitivities(&problem, &plan, &controls)
        .unwrap();
    // Differentiate 1/(1+p*t), its square, or exp(-p*t)*(1,2) analytically.
    let expected = if constrained {
        [-0.1 / 1.1_f64.powi(2), -0.2 / 1.1_f64.powi(3)]
    } else {
        [-0.1 * (-0.1_f64).exp(), -0.2 * (-0.1_f64).exp()]
    };
    for (actual, reference) in result.sensitivity(0, 0).unwrap().iter().zip(expected) {
        assert!((actual - reference).abs() < 1e-6);
    }
}

#[test]
fn full_rank_initial_parameter_pair_preserves_residual_row_changes() {
    check_parameter_rows(false, [[1.0, 0.0], [0.0, 1.0]]);
    check_parameter_rows(false, [[0.0, 1.0], [1.0, 0.0]]);
}

#[test]
fn constrained_initial_parameter_pair_preserves_residual_row_changes() {
    check_parameter_rows(true, [[1.0, 0.0], [0.0, 1.0]]);
    check_parameter_rows(true, [[1.0, 1.0], [1.0, -1.0]]);
}
