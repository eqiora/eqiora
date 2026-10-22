//! Stage matrices use original weak integrals and the admitted solver provider.
use super::*;
use crate::form_compiler::region::FieldPullback;
use crate::region_assembly::mapping::{RegionAlgebraicCoordinates, RegionDofMap};
use crate::spatial_expression::Coefficient;
use eqiora_solver::{CanonicalCsrSystemView, LinearOperatorOrientation};

fn exercise<S: Coefficient + Send + Sync>(
    plan: &CommonSpatialPlan,
    equations: &ExecutableSpatialEquations<S>,
    complex: bool,
    q1: bool,
) where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    let system = SpatialImplicitSystem::new(plan, equations, 0.).unwrap();
    let count = if q1 { 1 } else { 2 };
    let value = |re, im| C::new(re, if complex { im } else { 0. });
    let state = coordinates(&[value(3., 4.), value(-1., 2.)][..count], complex);
    let rate = coordinates(&[value(-2., 1.), value(1., -3.)][..count], complex);
    let direction = [value(-1., 2.), value(2., 0.5)];
    let source = value(1., 2.);
    let t = 0.5;
    let structure = equations.algebraic_structure(None).unwrap();
    let checked = plan
        .admission
        .linear
        .checked_backend(&REFERENCE_LINEAR_SOLVER, Some(&structure))
        .unwrap();
    let request = LinearSolveRequest::new(&checked, plan.linear());
    for (current_weight, rate_weight) in [(0., 1.), (0.5, 16.), (1., 2.)] {
        let assembly = system
            .linearize(
                t,
                &state,
                &rate,
                FieldPullback::Uniform {
                    current: (current_weight != 0.).then_some(current_weight),
                    rate: Some(rate_weight),
                },
            )
            .unwrap();
        // For g(t)=source*t² and forcing 2t*source, Q1 has M_ii=1/3,
        // sum M_if=1/6, K_ii=4; TPFA M=I/2 and K=[[6,-2],[-2,6]].
        // In the complex case storage is conjugated, so its real-coordinate
        // mass has opposite signs on the real and imaginary channels.
        let stored = |v: C| if complex { v.conj() } else { v };
        let y = [value(3., 4.), value(-1., 2.)];
        let v = [value(-2., 1.), value(1., -3.)];
        let residual = (0..count)
            .map(|i| {
                if q1 {
                    stored(v[i]) / 3. + 4. * y[i] - 4. * t * t * source + stored(source) * t / 3.
                        - t * source
                } else {
                    stored(v[i]) / 2. + 6. * y[i] - 2. * y[1 - i] - (4. * t * t + t) * source
                }
            })
            .collect::<Vec<_>>();
        close(&assembly.residuals[0], &coordinates(&residual, complex));
        let rhs = (0..count)
            .map(|i| {
                if q1 {
                    current_weight * 4. * direction[i] + rate_weight * stored(direction[i]) / 3.
                } else {
                    current_weight * (6. * direction[i] - 2. * direction[1 - i])
                        + rate_weight * stored(direction[i]) / 2.
                }
            })
            .collect::<Vec<_>>();
        let rhs = coordinates(&rhs, complex);
        let exact =
            eqiora_assembly::LinearSystem::new(assembly.systems[0].matrix().clone(), rhs.clone())
                .unwrap();
        let canonical =
            CanonicalCsrSystemView::new(&exact, &rhs, LinearOperatorProperties::General).unwrap();
        let solved = request
            .solve_canonical_oriented(&canonical, &rhs, LinearOperatorOrientation::Normal)
            .unwrap();
        close(solved.values(), &coordinates(&direction[..count], complex));
        assert_eq!(
            solved.report().solver_provider(),
            plan.admission.linear.provider
        );
        assert_eq!(solved.report().solver_plan(), plan.linear());
    }
}

#[test]
fn spatial_stage_matrices_solve_with_the_admitted_provider() {
    for complex in [false, true] {
        for spatial in [
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ] {
            let datum = if complex { "math.complex(1,2)" } else { "1" };
            let stored = if complex { "math.conj(u)" } else { "u" };
            let balance =
                format!("1[s/m^2]*derivative({stored})-div(grad(u))=2*{datum}*time()/1[s*m^2];");
            let resolved = resolve(
                complex,
                spatial,
                0.25,
                &balance,
                Some(&format!("{datum}*(time()/1[s])^2")),
            );
            let plan = resolved.as_spatial().unwrap();
            match plan.admission.recognized_model() {
                RecognizedNativeModel::Spatial(equations) => {
                    exercise(plan, equations, complex, spatial == CommonSpatialPolicy::Q1)
                }
                RecognizedNativeModel::ComplexSpatial(equations) => {
                    exercise(plan, equations, complex, spatial == CommonSpatialPolicy::Q1)
                }
                _ => unreachable!(),
            }
        }
    }
}
