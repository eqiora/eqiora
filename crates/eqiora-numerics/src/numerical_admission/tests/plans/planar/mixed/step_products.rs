//! An admitted mixed step must retain its solver's mathematical prerequisites.
use super::*;
use eqiora_ir::{LinearizedOutput, LinearizedRelation, RelationCotangent};
use eqiora_solver::LinearOperatorOrientation;

#[test]
fn mixed_step_products_preserve_certified_operator_properties() {
    for paired in [true, false] {
        let source = FLOW
            .replace("variable u:", "parameter load:1=1; state u:")
            .replace("variable p:", "state p:")
            .replace(
                "f=0[kg/(m*s^2)]",
                "f=load*(coordinate(0)/1[m]-1)*1[kg/(m*s^2)]",
            )
            .replace(
                "{-div(2[kg/(m*s)]",
                "{1[kg/m^3]*derivative(u)-div(2[kg/(m*s)]",
            )
            .replace(
                "relation constraint on body",
                "initial {u=grad(g);p=f;} relation constraint on body",
            );
        let source = if paired {
            source
        } else {
            source
                .replace("isotropic_lift(p)", "2*isotropic_lift(p)")
                .replace("p=f;", "p=f/2;")
        };
        let geometry = geometry(true);
        let body = geometry.entity_set("body").unwrap();
        let mut supports = vec![("body", body, None)];
        supports.extend(["left", "right", "bottom", "top"].map(|name| {
            (
                name,
                geometry.entity_set(name).unwrap(),
                Some(("body", body)),
            )
        }));
        let model = compile_model(
            "mixed-step-products.eqi",
            &source,
            &geometry,
            "Throughflow",
            &supports,
            &[],
        );
        for permuted in [false, true] {
            let resolved = ResolvedCommonPlan::resolve(
                &model,
                open_resources(&geometry, permuted),
                CommonSpatialPolicy::MiniP1,
                CommonSolvePolicy::Linear(exact_reference_linear(
                    if paired {
                        LinearSolver::MinimumResidual
                    } else {
                        LinearSolver::BiConjugateGradientStabilized
                    },
                    0.,
                    1e-12,
                    NonZeroUsize::new(1000).unwrap(),
                )),
                None,
                Some(CommonTemporalPolicy::backward_euler(0.25).unwrap()),
                &REFERENCE_LINEAR_SOLVER,
                None,
                None,
            )
            .unwrap();
            let plan = resolved.as_spatial().unwrap();
            let initial = plan.initial_state(Some(0.), Vec::new()).unwrap();
            let request =
                CommonTransientRunRequest::from_steps(resolved.clone(), initial, 1, vec![1])
                    .unwrap();
            let std::ops::ControlFlow::Continue(steps) = request
                .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
                .unwrap()
            else {
                panic!("unexpected stop")
            };
            let parameter = plan
                .kernel()
                .nodes()
                .find_map(|node| match node {
                    eqiora_schema::kernel::KernelNode::Parameter(value) => Some(value.id()),
                    _ => None,
                })
                .unwrap();
            let fields = plan.fields().map(|(id, _)| id.erase()).collect::<Vec<_>>();
            let product = plan
                .linearize_step(&steps[0].1, &[parameter], &fields, &Default::default())
                .unwrap();
            let reopened = plan
                .validate_differentiated_step_values(
                    &request,
                    &[parameter],
                    None,
                    &fields,
                    &Default::default(),
                    product.relation().accepted_unknowns(),
                )
                .unwrap();
            assert_eq!(reopened, product);
            assert_eq!(
                product.relation().state_jacobian().properties(),
                if paired {
                    LinearOperatorProperties::SymmetricIndefinite
                } else {
                    LinearOperatorProperties::General
                }
            );

            // The six vertices are (0,0),(1,0),(1,1),(0,1),(1/2,1/2),(1,1/2).
            // For every load, u=(1,0), p=load*(x-1) solves the original equations;
            // fixed committed velocity gives du/dload=0 and dp/dload=x-1.
            assert_eq!(plan.fields().count(), 2);
            // Doubling only the pressure coupling breaks the symmetry certificate
            // and gives p=load*(x-1)/2; it still has a general implicit derivative.
            let pressure =
                [-1., 0., 0., -1., -0.5, 0.].map(|value| value / if paired { 1. } else { 2. });
            let expected = plan
                .fields()
                .flat_map(|(_, ty)| {
                    if ty.shape().is_scalar() {
                        pressure.to_vec()
                    } else {
                        vec![0.; 22]
                    }
                })
                .collect::<Vec<_>>();
            let primal = plan
                .fields()
                .flat_map(|(_, ty)| {
                    if ty.shape().is_scalar() {
                        pressure.to_vec()
                    } else {
                        let mut values = [1., 0.].repeat(6);
                        values.extend([0.; 10]);
                        values
                    }
                })
                .collect::<Vec<_>>();
            for (actual, expected) in product.output_values().iter().zip(&primal) {
                assert!(
                    (actual - expected).abs() < 1e-9,
                    "primal {actual} != {expected}"
                );
            }
            let before = steps[0].1.state().to_bytes().unwrap();
            let relation = product.relation();
            let request = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan.linear());
            let rhs = relation
                .design_jacobian()
                .iter()
                .map(|value| -*value)
                .collect::<Vec<_>>();
            let tangent = request
                .solve_canonical_oriented(
                    relation.state_jacobian(),
                    &rhs,
                    LinearOperatorOrientation::Normal,
                )
                .unwrap();
            assert_eq!(tangent.report().solver_plan(), plan.linear());
            let mut output = vec![0.; product.output_dimension()];
            product.jvp(tangent.values(), &[1.], &mut output).unwrap();
            assert_eq!(output.len(), expected.len());
            for (actual, expected) in output.iter().zip(&expected) {
                assert!(
                    (actual - expected).abs() < 1e-9,
                    "tangent {actual} != {expected}"
                );
            }
            // Use a unit output cotangent: the solver has an absolute residual
            // budget, so arbitrary growth in direction magnitude changes the solve.
            let norm = (1..=output.len())
                .map(|index| (index as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let weights = (1..=output.len())
                .map(|index| index as f64 / norm)
                .collect::<Vec<_>>();
            let mut unknown = vec![0.; relation.unknown_dimension()];
            let mut direct = vec![0.; 1];
            product.vjp(&weights, &mut unknown, &mut direct).unwrap();
            let adjoint = request
                .solve_canonical_oriented(
                    relation.state_jacobian(),
                    &unknown,
                    LinearOperatorOrientation::Transposed,
                )
                .unwrap();
            let mut parameter = vec![0.; 1];
            relation
                .vjp(
                    adjoint.values(),
                    RelationCotangent::Parameter(&mut parameter),
                )
                .unwrap();
            let expected: f64 = expected
                .iter()
                .zip(&weights)
                .map(|(value, weight)| value * weight)
                .sum();
            assert!((direct[0] - parameter[0] - expected).abs() < 1e-9);
            assert_eq!(steps[0].1.state().to_bytes().unwrap(), before);
        }
    }
}
