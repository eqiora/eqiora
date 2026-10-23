//! Finite midpoint stages through the shared production Newton and solver owners.
use super::*;
use crate::nonlinear_execution::{NewtonOptions, NewtonTrial, solve_damped_newton, stage_merit};
use eqiora_assembly::{CsrMatrix, LinearSystem};
use eqiora_solver::{
    CanonicalCsrSystemView, LinearOperatorProperties, LinearSolveRequest, LinearSolverBackend,
};
use eqiora_time::{ImplicitMidpointTimeBackend, TimeSolution};

impl CommonTimeRunRequest {
    /// Execute a deterministic Run with its admitted stage solver and time backend.
    /// Midpoint uses the shared production Newton and canonical linear solver;
    /// explicit integration delegates to the supplied time backend callback.
    pub fn run<F>(
        &self,
        backend: &dyn LinearSolverBackend,
        solve: F,
    ) -> Result<crate::CommonTrajectory, Diagnostic>
    where
        F: FnOnce(&TimeProblem<'_>, &TimePlan) -> Result<TimeSolution, Diagnostic>,
    {
        let problem = self.problem()?;
        let plan = self.time_plan()?;
        let solution = if plan.method() == TimeMethod::ImplicitMidpoint {
            let ode = self.ode_plan()?;
            let linear = ode
                .linear
                .as_ref()
                .ok_or_else(|| invalid("midpoint Run omitted its admitted linear policy"))?;
            let checked = linear.checked_backend(backend, None)?;
            let request = LinearSolveRequest::new(&checked, linear.solver);
            let nonlinear = ode.nonlinear;
            ImplicitMidpointTimeBackend::new().solve_with_stages(
                problem.initial_state(),
                plan,
                problem.equation_class(),
                |time, next_time, initial| {
                    let stage = MidpointStage {
                        problem: &problem,
                        time: time + (next_time - time) / 2.,
                        h: next_time - time,
                        initial,
                    };
                    if !stage.time.is_finite() || stage.time <= time || !(1. / stage.h).is_finite()
                    {
                        return Err(invalid(
                            "midpoint requires a finite representable forward stage",
                        ));
                    }
                    if nonlinear.is_none() {
                        ode.program.require_affine_state_dependence(stage.time)?;
                    }
                    let (accepted, _) = solve_damped_newton(
                        initial.to_vec(),
                        NewtonOptions::time_stage(
                            nonlinear,
                            linear.solver,
                            plan.relative_tolerance(),
                            plan.absolute_tolerances(),
                        ),
                        |point| stage.linearize(point),
                        |point| stage_merit(&point.residual, &point.scales, nonlinear.is_some()),
                        |_, point| {
                            let system = CanonicalCsrSystemView::new(
                                &point.system,
                                point.system.rhs(),
                                LinearOperatorProperties::General,
                            )?;
                            let update = request.solve(&system.linear_problem()?)?;
                            Ok((update.values().to_vec(), update.report().clone()))
                        },
                        |point, current| {
                            Ok(Some(NewtonTrial {
                                residual_norm: stage_merit(
                                    &stage.residual(point)?,
                                    &current.scales,
                                    nonlinear.is_some(),
                                )?,
                                linearization: None,
                            }))
                        },
                    )?;
                    Ok(accepted.endpoint)
                },
            )?
        } else {
            solve(&problem, plan)?
        };
        crate::CommonTrajectory::accept_ode(self.clone(), solution)
    }
}

struct MidpointStage<'a, 'b> {
    problem: &'a TimeProblem<'b>,
    time: f64,
    h: f64,
    initial: &'a [f64],
}
struct MidpointPoint {
    endpoint: Vec<f64>,
    residual: Vec<f64>,
    scales: Vec<f64>,
    system: LinearSystem<f64>,
}
impl MidpointStage<'_, '_> {
    fn middle(&self, endpoint: &[f64]) -> Vec<f64> {
        self.initial
            .iter()
            .zip(endpoint)
            .map(|(a, b)| 0.5 * a + 0.5 * b)
            .collect()
    }
    fn mass(&self, direction: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        if self.problem.equation_class() == TimeEquationClass::ExplicitOde {
            output.copy_from_slice(direction);
            Ok(())
        } else {
            self.problem
                .system()
                .mass_action(self.time, direction, output)
        }
    }
    fn residual(&self, endpoint: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        let n = self.initial.len();
        let rate = self
            .initial
            .iter()
            .zip(endpoint)
            .map(|(a, b)| (b - a) / self.h)
            .collect::<Vec<_>>();
        let mut residual = vec![0.; n];
        self.mass(&rate, &mut residual)?;
        let mut rhs = vec![0.; n];
        self.problem
            .system()
            .rhs(self.time, &self.middle(endpoint), &mut rhs)?;
        for (r, f) in residual.iter_mut().zip(rhs) {
            *r -= f;
        }
        Ok(residual)
    }
    fn linearize(&self, endpoint: &[f64]) -> Result<MidpointPoint, Diagnostic> {
        let n = self.initial.len();
        let size = n
            .checked_mul(n)
            .ok_or_else(|| invalid("midpoint stage matrix size overflows"))?;
        let mut matrix = Vec::new();
        matrix
            .try_reserve_exact(size)
            .map_err(|_| invalid("midpoint stage matrix exceeds available storage"))?;
        matrix.resize(size, 0.);
        let middle = self.middle(endpoint);
        let mut basis = vec![0.; n];
        let mut mass = vec![0.; n];
        let mut image = vec![0.; n];
        for column in 0..n {
            basis[column] = 1.;
            self.mass(&basis, &mut mass)?;
            self.problem
                .system()
                .rhs_jvp(self.time, &middle, &basis, &mut image)?;
            for row in 0..n {
                matrix[row * n + column] = mass[row] / self.h - 0.5 * image[row];
            }
            basis[column] = 0.;
        }
        let mut offsets = vec![0];
        let mut columns = Vec::new();
        let mut entries = Vec::new();
        let mut scales = Vec::new();
        for row in matrix.chunks_exact(n) {
            let mut scale = 0_f64;
            for (column, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(invalid("midpoint stage Jacobian is nonfinite"));
                }
                if value != 0. {
                    columns.push(column);
                    entries.push(value);
                }
                scale = scale.max(value.abs());
            }
            if scale == 0. {
                return Err(invalid("midpoint Jacobian requires nonzero equation rows"));
            }
            offsets.push(entries.len());
            scales.push(scale);
        }
        let residual = self.residual(endpoint)?;
        let system = LinearSystem::new(
            CsrMatrix::from_sorted_csr(n, n, offsets, columns, entries)?,
            residual.iter().map(|r| -r).collect(),
        )?;
        Ok(MidpointPoint {
            endpoint: endpoint.to_vec(),
            residual,
            scales,
            system,
        })
    }
}
