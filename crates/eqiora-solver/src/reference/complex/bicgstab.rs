//! Host-serial complex BiCGStab with conjugate-first Euclidean pairing.
use super::super::correction::SolutionCorrection;
use super::{Complex64 as C, Diagnostic, LinearProblem, SolverPlan, norm, solve_failed};
use crate::ConvergenceReason;

pub(super) struct Produced {
    pub values: Vec<C>,
    pub reason: ConvergenceReason,
    pub iterations: usize,
    pub residual: f64,
}

pub(super) fn solve(
    problem: &LinearProblem<'_, C>,
    plan: SolverPlan,
) -> Result<Produced, Diagnostic> {
    let n = problem.operator().rows();
    let zero = C::new(0., 0.);
    let mut x = problem
        .initial_guess()
        .map_or_else(|| vec![zero; n], <[C]>::to_vec);
    let mut r = residual(problem, &x)?;
    let target = plan.residual_target(norm(problem.right_hand_side())?)?;
    let initial = norm(&r)?;
    if initial <= target {
        return Ok(Produced {
            values: x,
            reason: ConvergenceReason::InitialResidualSatisfied,
            iterations: 0,
            residual: initial,
        });
    }
    let mut correction = SolutionCorrection::new(&mut x, target, initial);
    let mut recurrence_target = target;
    let mut shadow = r.clone();
    let mut p = vec![zero; n];
    let mut v = vec![zero; n];
    let mut s = vec![zero; n];
    let mut t = vec![zero; n];
    let mut previous_rho = C::new(1., 0.);
    let mut alpha = C::new(1., 0.);
    let mut omega = C::new(1., 0.);
    let mut fresh = true;
    let mut estimated = initial;
    for iteration in 1..=plan.maximum_iterations().get() {
        let mut rho = dot(&shadow, &r)?;
        if rho == zero {
            // A nonzero residual may be orthogonal to the old shadow even for
            // an invertible operator. Restart; true-residual acceptance remains.
            shadow.clone_from(&r);
            fresh = true;
            rho = dot(&shadow, &r)?;
        }
        nonzero(rho, "shadow pairing")?;
        let started_fresh = fresh;
        if fresh {
            p.copy_from_slice(&r);
            fresh = false;
        } else {
            nonzero(omega, "stabilization coefficient")?;
            let beta = finite((rho / previous_rho) * (alpha / omega))?;
            for j in 0..n {
                p[j] = finite(r[j] + beta * (p[j] - omega * v[j]))?;
            }
        }
        problem.operator().apply(&p, &mut v)?;
        let denominator = dot(&shadow, &v)?;
        if denominator == zero && !started_fresh {
            shadow.clone_from(&r);
            fresh = true;
            continue;
        }
        nonzero(denominator, "step pairing")?;
        alpha = finite(rho / denominator)?;
        for j in 0..n {
            s[j] = finite(r[j] - alpha * v[j])?;
        }
        estimated = norm(&s)?;
        if estimated <= recurrence_target {
            for j in 0..n {
                x[j] = finite(x[j] + alpha * p[j])?;
            }
        } else {
            problem.operator().apply(&s, &mut t)?;
            let denominator = dot(&t, &t)?;
            nonzero(denominator, "stabilization norm")?;
            omega = finite(dot(&t, &s)? / denominator)?;
            for j in 0..n {
                x[j] = finite(x[j] + alpha * p[j] + omega * s[j])?;
                r[j] = finite(s[j] - omega * t[j])?;
            }
            estimated = norm(&r)?;
        }
        if estimated <= recurrence_target {
            let replayed = residual(problem, correction.candidate(&x))?;
            let true_norm = norm(&replayed)?;
            if true_norm <= target {
                return Ok(Produced {
                    values: correction.into_values(),
                    reason: ConvergenceReason::ResidualToleranceSatisfied,
                    iterations: iteration,
                    residual: estimated,
                });
            }
            // Restart from the actual residual; never accept recurrence drift.
            recurrence_target = correction.restart(&mut x, true_norm);
            r = replayed;
            shadow.clone_from(&r);
            fresh = true;
        }
        previous_rho = rho;
    }
    if norm(&residual(problem, correction.candidate(&x))?)? <= target {
        return Ok(Produced {
            values: correction.into_values(),
            reason: ConvergenceReason::ResidualToleranceSatisfied,
            iterations: plan.maximum_iterations().get(),
            residual: estimated,
        });
    }
    Err(solve_failed(format!(
        "complex BiCGStab exceeded {} iterations",
        plan.maximum_iterations()
    )))
}

fn residual(problem: &LinearProblem<'_, C>, x: &[C]) -> Result<Vec<C>, Diagnostic> {
    let mut values = vec![C::new(0., 0.); x.len()];
    problem.operator().apply(x, &mut values)?;
    for (value, rhs) in values.iter_mut().zip(problem.right_hand_side()) {
        *value = finite(*rhs - *value)?;
    }
    Ok(values)
}

// Fixed left-to-right order on complete host-serial complex coordinates.
fn dot(left: &[C], right: &[C]) -> Result<C, Diagnostic> {
    left.iter()
        .zip(right)
        .try_fold(C::new(0., 0.), |sum, (left, right)| {
            finite(sum + left.conj() * right)
        })
}
fn finite(value: C) -> Result<C, Diagnostic> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(solve_failed("complex BiCGStab arithmetic became nonfinite"))
    }
}
fn nonzero(value: C, operation: &str) -> Result<(), Diagnostic> {
    if value == C::new(0., 0.) {
        Err(solve_failed(format!(
            "complex BiCGStab {operation} broke down before convergence"
        )))
    } else {
        Ok(())
    }
}
