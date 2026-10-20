//! Independent flux integration fixes the nonlinear point and both parameter columns.
use super::*;
use eqiora_ir::{LinearizedOutput, RelationCotangent};

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-10,
            "{actual:e} != {expected:e}"
        );
    }
}

#[test]
fn nonlinear_material_points_retain_original_residual_and_boundary_chain_products() {
    check_material_points(false);
}

#[test]
fn first_partial_material_contracts_support_nonlinear_parameter_products() {
    check_material_points(true);
}

fn check_material_points(property: bool) {
    let geometry = cartesian_interval();
    let body = geometry.entity_set("body").unwrap();
    let source = "public component Material(
            support body:volume(ambient_dimension=1),
            support left:boundary(parent=body), support right:boundary(parent=body),
            parameter p:1, parameter r:1
        ) {
            variable u:1 on body in smooth;
            relation balance on body {-div((1+p*u)*grad(u))=0;}
            relation lower on left {trace(u)=1;}
            relation upper on right {trace(u)=r;}
        }";
    let source = if property {
        format!(
            "property contract Law(input value:1):1 {{derivatives first_partials;}}
             property release Release:Law {{analytic {{value=1+value;source_unit:1=1;}}
             validity value>=-0.5 and value<=4;outside reject;branch single;
             citation org.example.manufactured;license spdx.CC0_1_0;}}
{}
             public component Material(support body:volume(ambient_dimension=1),
             support left:boundary(parent=body),support right:boundary(parent=body),
             parameter p:1,parameter r:1) {{instance physics:Equation(body=body,
             left=left,right=right,p=p,r=r,response=Release);}}",
            source
                .replace(
                    "public component Material(",
                    "component Equation(property response:Law,"
                )
                .replace("(1+p*u)", "response(value=p*u)")
        )
    } else {
        source.to_owned()
    };
    let model = compile_model(
        "material-sensitivity.eqi",
        &source,
        &geometry,
        "Material",
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
        &[
            ("p", DynQuantity::new(1., DimExponents::DIMENSIONLESS)),
            ("r", DynQuantity::new(2., DimExponents::DIMENSIONLESS)),
        ],
    );
    let resolved = ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[2]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Newton {
            linear: exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-13,
                1e-14,
                NonZeroUsize::new(100).unwrap(),
            ),
            nonlinear: NonlinearSolvePlan::new(0., 1e-11, NonZeroUsize::new(16).unwrap(), 16)
                .unwrap(),
        },
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    let plan = resolved.as_spatial().unwrap();
    let mut parameters = plan
        .kernel()
        .nodes()
        .filter_map(|node| match node {
            eqiora_schema::kernel::KernelNode::Parameter(p) => {
                Some((p.value().component(0).unwrap().0, p.id()))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    parameters.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert_eq!(parameters.len(), 2);
    let selected = parameters.iter().map(|(_, id)| *id).collect::<Vec<_>>();
    let outputs = plan
        .fields()
        .map(|(field, _)| field.erase())
        .collect::<Vec<_>>();
    let initial = plan.scalar_state(None, vec![1., 1.25, 2.]).unwrap();
    let seed_bytes = initial.to_bytes().unwrap();
    let quadratures = std::collections::HashMap::new();
    for (p, r) in [(0., 2.), (1., 2.), (1., 3.)] {
        // q(u)=u+p*u²/2 is linear in x. P1 flux integration is exact, so
        // q(mid)=(q(1)+q(r))/2. This derives the point without the solver.
        let q_mid: f64 = (1. + r) / 2. + p * (1. + r * r) / 4.;
        let u = if p == 0. {
            q_mid
        } else {
            ((1. + 2. * p * q_mid).sqrt() - 1.) / p
        };
        let jacobian = 4. * (1. + p * u);
        let parameter = [2. * u * u - 1. - r * r, -2. * (1. + p * r)];
        let exact = plan
            .validate_differentiated_values(
                Some(&initial),
                &selected,
                Some(&[p, r]),
                &outputs,
                &quadratures,
                &[u],
            )
            .unwrap();
        close(exact.output_values(), &[1., u, r]);
        let relation = exact.relation();
        let mut residual = [0.];
        relation.primal(&mut residual).unwrap();
        close(&residual, &[0.]);
        relation
            .jvp(RelationTangent::Unknown(&[1.]), &mut residual)
            .unwrap();
        close(&residual, &[jacobian]);
        close(relation.design_jacobian(), &parameter);
        let mut transpose = [0.; 2];
        relation
            .vjp(&[1.], RelationCotangent::Parameter(&mut transpose))
            .unwrap();
        close(&transpose, &parameter);
        for column in 0..2 {
            let mut direction = [0.; 2];
            direction[column] = 1.;
            let du = -parameter[column] / jacobian;
            let mut output = [0.; 3];
            exact.jvp(&[du], &direction, &mut output).unwrap();
            close(&output, &[0., du, f64::from(column == 1)]);
        }
        // The endpoint cotangent contributes directly as well as through the
        // implicit state. Test both terms with an arbitrary output covector.
        let mut state = [0.];
        let mut direct = [0.; 2];
        exact.vjp(&[2., 3., 5.], &mut state, &mut direct).unwrap();
        close(&state, &[3.]);
        close(&direct, &[0., 5.]);
        let live = plan
            .differentiate(
                Some(&initial),
                &selected,
                Some(&[p, r]),
                &outputs,
                &quadratures,
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap();
        assert!(live.nonlinear_iterations().unwrap() > 0);
        assert_eq!(
            live.nonlinear_initial_state().unwrap().identity(),
            initial.identity()
        );
        assert!(live.receipt().is_none());
        close(live.linearization().output_values(), &[1., u, r]);
        let reopened = plan
            .validate_differentiated_values(
                Some(&initial),
                &selected,
                Some(&[p, r]),
                &outputs,
                &quadratures,
                live.linearization().relation().accepted_unknowns(),
            )
            .unwrap();
        assert_eq!(&reopened, live.linearization());
        assert!(
            plan.validate_differentiated_values(
                Some(&initial),
                &selected,
                Some(&[p, r]),
                &outputs,
                &quadratures,
                &[u + 0.01]
            )
            .is_err()
        );
    }
    let accepted_seed = plan.scalar_state(None, vec![1., 1.5, 2.]).unwrap();
    let zero = plan
        .differentiate(
            Some(&accepted_seed),
            &selected,
            Some(&[0., 2.]),
            &outputs,
            &quadratures,
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
    assert_eq!(zero.nonlinear_iterations(), Some(0));
    assert!(zero.linear_solves().is_empty());
    assert_eq!(zero.linearization().initial_residual_norm(), Some(0.));
    if property {
        // The manufactured law is strictly positive on [-0.5,4]. The p=0
        // derivative is interior; p=-1 violates the exact operating domain.
        assert!(
            plan.differentiate(
                Some(&initial),
                &selected,
                Some(&[-1., 2.]),
                &outputs,
                &quadratures,
                &REFERENCE_LINEAR_SOLVER
            )
            .is_err()
        );
    }
    assert_eq!(initial.to_bytes().unwrap(), seed_bytes);
    assert!(
        plan.differentiate(
            None,
            &selected,
            None,
            &outputs,
            &quadratures,
            &REFERENCE_LINEAR_SOLVER
        )
        .is_err()
    );
}
