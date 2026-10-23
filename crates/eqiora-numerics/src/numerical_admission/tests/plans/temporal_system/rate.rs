//! Independent rates and analytic tangents through the actual admitted linear provider.
use super::*;
use crate::region_assembly::mapping::{RegionAlgebraicCoordinates, RegionDofMap};
use crate::spatial_expression::Coefficient;
use eqiora_time::TimeMethod;

fn check<S: Coefficient + Send + Sync>(
    plan: &CommonSpatialPlan,
    equations: &ExecutableSpatialEquations<S>,
    complex: bool,
    q1: bool,
    profile: usize,
) where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    let system = SpatialImplicitSystem::new(plan, equations, 0.).unwrap();
    let count = if q1 {
        if profile == 0 { 1 } else { 3 }
    } else {
        2
    };
    let value = C::new(1., if complex { 2. } else { 0. });
    let direction = C::new(-2., if complex { 0.5 } else { 0. });
    let state = coordinates(&vec![value; count], complex);
    let tangent = coordinates(&vec![direction; count], complex);
    let before = state.clone();
    let (rate, report) = system
        .explicit_rate(0.5, &state, &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let jvp = system
        .explicit_rate_jvp(0.5, &state, &tangent, &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let (expected, expected_jvp) = match profile {
        // Independently integrated M and K: free Q1 lambda=12, uniform TPFA lambda=8.
        0 => {
            let lambda = if q1 { 12. } else { 8. };
            (-lambda * value, -lambda * direction)
        }
        // d(u²)/dt=2*u*u'=-u², on the regular nonzero branch.
        1 => (-value / 2., -direction / 2.),
        // u*u'=-u differs from d(u²)/dt=-u. Here u'=-1 and its tangent vanishes.
        2 => (C::new(-1., 0.), C::new(0., 0.)),
        // (1+u²)*u'=-u; differentiating the mass is essential.
        3 => {
            let mass = 1. + value * value;
            (
                -value / mass,
                (value * value - 1.) * direction / (mass * mass),
            )
        }
        // conjugate(u')=-u is real-linear, with no holomorphic Jacobian assumption.
        4 => (-value.conj(), -direction.conj()),
        _ => unreachable!(),
    };
    close(&rate, &coordinates(&vec![expected; count], complex));
    close(&jvp, &coordinates(&vec![expected_jvp; count], complex));
    assert_eq!(report.solver_provider(), plan.admission.linear.provider);
    assert_eq!(report.solver_plan(), plan.linear());
    assert_eq!(state, before);
    if profile == 1 || profile == 2 {
        // A compatible zero RHS does not prove uniqueness of the rate.
        let zero = vec![0.; state.len()];
        let error = system
            .explicit_rate(0.5, &zero, &REFERENCE_LINEAR_SOLVER)
            .unwrap_err();
        assert!(
            error.message().contains("invertible rate Jacobian"),
            "{error}"
        );
        assert_eq!(zero, vec![0.; state.len()]);
    }
}

#[test]
fn regular_rate_elimination_retains_state_dependent_and_real_linear_tangents() {
    for complex in [false, true] {
        for spatial in [
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ] {
            for (profile, balance) in [
                "1[s/m^2]*derivative(u)-div(grad(u))=0;",
                "derivative(u*u)=-u*u/1[s];",
                "u*derivative(u)=-u/1[s];",
                "(1+u*u)*derivative(u)=-u/1[s];",
                "derivative(math.conj(u))=-u/1[s];",
            ]
            .into_iter()
            .enumerate()
            .take(if complex { 5 } else { 4 })
            {
                let base = resolve(
                    complex,
                    spatial,
                    0.25,
                    balance,
                    (profile == 0).then_some("0"),
                );
                let plan = policy::with_policy(
                    &base,
                    policy::policy(&base, TimeMethod::Tsitouras45, 0.25),
                )
                .unwrap();
                let plan = plan.as_spatial().unwrap();
                match plan.admission.recognized_model() {
                    RecognizedNativeModel::Spatial(equations) => check(
                        plan,
                        equations,
                        complex,
                        spatial == CommonSpatialPolicy::Q1,
                        profile,
                    ),
                    RecognizedNativeModel::ComplexSpatial(equations) => check(
                        plan,
                        equations,
                        complex,
                        spatial == CommonSpatialPolicy::Q1,
                        profile,
                    ),
                    _ => unreachable!(),
                }
            }
        }
    }
}
