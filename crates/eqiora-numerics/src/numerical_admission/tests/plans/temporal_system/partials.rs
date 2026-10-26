//! Input derivatives include coefficient, trace, and trace-rate contributions.
use super::*;
use crate::region_assembly::mapping::{RegionAlgebraicCoordinates, RegionDofMap};
use crate::spatial_expression::Coefficient;
use eqiora_schema::kernel::KernelNode;

fn exercise<S: Coefficient + Send + Sync>(
    plan: &CommonSpatialPlan,
    equations: &ExecutableSpatialEquations<S>,
    complex: bool,
    q1: bool,
) where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    let parameter = plan
        .kernel()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Parameter(parameter) => Some(parameter.id()),
            _ => None,
        })
        .unwrap();
    let coordinate = eqiora_ir::ScalarSymbolCoordinate::for_value(
        eqiora_schema::kernel::SymbolRef::Parameter(parameter),
        &eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            eqiora_core::DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
    )
    .unwrap()
    .remove(0);
    let system = SpatialImplicitSystem::new(plan, equations, 0.).unwrap();
    let phase = C::new(1., if complex { 2. } else { 0. });
    let count = if q1 { 1 } else { 2 };
    let state = coordinates(&vec![phase * 3.; count], complex);
    let rates = [
        C::new(-2., if complex { 1. } else { 0. }),
        C::new(1., if complex { -3. } else { 0. }),
    ];
    let rate = coordinates(&rates[..count], complex);
    let stored = |value: C| if complex { value.conj() } else { value };
    for t in [0.5, 1.25] {
        let (actual, lift) = system
            .parameter_partial(t, &state, &rate, &coordinate)
            .unwrap();
        // Two half-length cells: Q1 M_ii=1/3, sum M_if=1/6,
        // K_if sum=-4, load=1/2. TPFA M_ii=1/2 and boundary weight=-4.
        // p=2, g=p*phase*t²: both differentiating p*M*g_dot and
        // lifting g_dot,p contribute p*t*conj(phase)/3 for Q1.
        let expected = rates[..count]
            .iter()
            .map(|v| {
                stored(*v) / if q1 { 3. } else { 2. }
                    + if q1 {
                        stored(phase) * (4. * t / 3.)
                    } else {
                        C::new(0., 0.)
                    }
                    - phase * (t / 2. + 4. * t * t)
            })
            .collect::<Vec<_>>();
        close(&actual, &coordinates(&expected, complex));
        let expected_lift = if q1 {
            vec![phase * t * t, C::new(0., 0.), phase * t * t]
        } else {
            vec![C::new(0., 0.); 2]
        };
        close(&lift, &coordinates(&expected_lift, complex));
    }
}

#[test]
fn independent_rate_parameter_action_retains_time_dependent_trace_lifts() {
    for complex in [false, true] {
        for spatial in [
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ] {
            let geometry = cartesian_interval();
            let body = geometry.entity_set("body").unwrap();
            let scalar = if complex { "complex<1>" } else { "1" };
            let phase = if complex { "math.complex(1,2)" } else { "1" };
            let stored = if complex { "math.conj(u)" } else { "u" };
            let source = format!("model Heat(support body:volume(ambient_dimension=1),
                support left:boundary(parent=body),support right:boundary(parent=body)) {{
                parameter p:1=2; state u:{scalar} on body in smooth;
                relation balance on body {{p*1[s/m^2]*derivative({stored})-div(grad(u))=p*{phase}*time()/1[s*m^2];}}
                relation left_value on left {{trace(u)=p*{phase}*(time()/1[s])^2;}}
                relation right_value on right {{trace(u)=p*{phase}*(time()/1[s])^2;}}
            }}");
            let model = compile_model(
                "implicit-input-partials.eqi",
                &source,
                &geometry,
                "Heat",
                &[
                    ("body", body, None),
                    (
                        "left",
                        geometry.entity_set("left").unwrap(),
                        Some(("body", body)),
                    ),
                    (
                        "right",
                        geometry.entity_set("right").unwrap(),
                        Some(("body", body)),
                    ),
                ],
                &[],
            );
            let resolved = ResolvedCommonPlan::resolve(
                &model,
                cartesian_box_resources(&geometry, &[2]),
                spatial,
                CommonSolvePolicy::Linear(exact_reference_linear(
                    LinearSolver::BiConjugateGradientStabilized,
                    1e-12,
                    1e-14,
                    NonZeroUsize::new(100).unwrap(),
                )),
                None,
                Some(CommonTemporalPolicy::backward_euler(1.).unwrap()),
                &REFERENCE_LINEAR_SOLVER,
                None,
                None,
            )
            .unwrap();
            let plan = resolved.as_spatial().unwrap();
            let q1 = spatial == CommonSpatialPolicy::Q1;
            match plan.admission.recognized_model() {
                RecognizedNativeModel::Spatial(equations) => exercise(plan, equations, complex, q1),
                RecognizedNativeModel::ComplexSpatial(equations) => {
                    exercise(plan, equations, complex, q1)
                }
                _ => unreachable!(),
            }
        }
    }
}
