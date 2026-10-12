use super::*;

#[test]
fn fixed_reference_equations_bind_through_shared_simplicial_regions() {
    let geometry = fsi_geometry();
    let model = fsi_model(&geometry);
    let recognized = RecognizedNativeAdmission::recognize(
        &model,
        fsi_resources(&geometry),
        &CommonSpatialRequest::Uniform(CommonSpatialPolicy::P1),
    )
    .unwrap();
    let mut equations =
        crate::numerical_admission::scalar::ExecutableLinearEquations::<f64>::simplicial(
            &recognized.program,
            &recognized.resources,
        )
        .expect("mixed fluid and kinematic solid equations must use the shared Region compiler");
    assert_eq!(
        equations.finite_element_operator_properties(),
        LinearOperatorProperties::SymmetricIndefinite,
    );
    let discretizations = equations
        .resolve_discretizations(
            &CommonSpatialRequest::Scoped(
                equations
                    .regions
                    .iter()
                    .map(|region| {
                        let policy = if region.form.constraint_pairs().next().is_some() {
                            CommonSpatialPolicy::MiniP1
                        } else {
                            CommonSpatialPolicy::P1
                        };
                        CommonScopedSpatialPolicy::new(
                            model.digest().unwrap(),
                            region.domain_id(),
                            policy,
                        )
                    })
                    .collect(),
            ),
            &model,
        )
        .unwrap();
    let NativeMeshResources::AdjacentPartitionSimplicial { mesh, .. } = &recognized.resources
    else {
        panic!("fixture must retain its authenticated adjacent partition");
    };
    let temporal = CommonBackwardEuler::from_seconds(0.05).unwrap();
    for region in &mut equations.regions {
        region.form = region.form.bind_backward_euler(temporal.step()).unwrap();
    }
    let (mapping, forms, _) = equations
        .simplicial_assembly(
            mesh,
            &discretizations,
            &eqiora_realization::SystemScaling::Dimensional,
        )
        .unwrap();
    assert_eq!(forms.len(), 2);
    assert_eq!(mapping.traces().len(), 1);
    assert_eq!(equations.represented_fields().len(), 4);
    assert!(mapping.traces()[0].1.iter().any(|dof| dof.component == 1));
}

#[test]
fn shared_simplicial_interfaces_reject_inexact_boundary_support() {
    use crate::numerical_admission::native::polyhedral::{
        bind_model_support, validate_interface_support,
    };
    let geometry = fsi_geometry();
    let model = fsi_model(&geometry);
    let recognized = RecognizedNativeAdmission::recognize(
        &model,
        fsi_resources(&geometry),
        &CommonSpatialRequest::Uniform(CommonSpatialPolicy::P1),
    )
    .unwrap();
    let supports = bind_model_support(&recognized.program, &recognized.resources).unwrap();
    let equations =
        crate::numerical_admission::scalar::ExecutableLinearEquations::<f64>::simplicial(
            &recognized.program,
            &recognized.resources,
        )
        .unwrap();
    let [interface] = equations.interfaces.as_slice() else {
        panic!("fixture must contain one exact interface");
    };
    let sides = interface
        .sides()
        .each_ref()
        .map(|side| (side.domain(), side.boundary()));
    validate_interface_support(&supports, &recognized.resources, sides).unwrap();

    let mut missing = supports.clone();
    missing
        .get_mut(&sides[0].0)
        .unwrap()
        .facets
        .get_mut(&sides[0].1)
        .unwrap()
        .pop()
        .unwrap();
    assert!(
        validate_interface_support(&missing, &recognized.resources, sides)
            .unwrap_err()
            .message()
            .contains("same nonempty facet set")
    );

    let mut foreign = supports.clone();
    foreign.get_mut(&sides[0].0).unwrap().cells = supports[&sides[1].0].cells.clone();
    assert!(
        validate_interface_support(&foreign, &recognized.resources, sides)
            .unwrap_err()
            .message()
            .contains("orientations are not opposite")
    );

    assert!(
        validate_interface_support(&supports, &recognized.resources, [sides[0], sides[0]])
            .unwrap_err()
            .message()
            .contains("distinct Regions")
    );
    let mut empty = supports.clone();
    for (domain, boundary) in sides {
        empty
            .get_mut(&domain)
            .unwrap()
            .facets
            .get_mut(&boundary)
            .unwrap()
            .clear();
    }
    assert!(
        validate_interface_support(&empty, &recognized.resources, sides)
            .unwrap_err()
            .message()
            .contains("same nonempty facet set")
    );
}

#[test]
fn fixed_reference_equations_execute_through_common_linear_run() {
    let geometry = fsi_geometry();
    let model = fsi_model(&geometry);
    let recognized = RecognizedNativeAdmission::recognize(
        &model,
        fsi_resources(&geometry),
        &CommonSpatialRequest::Uniform(CommonSpatialPolicy::P1),
    )
    .unwrap();
    let RecognizedNativeModel::Linear(equations) = &recognized.recognized else {
        panic!("coupled equations must select the common linear path");
    };
    let scoped = CommonMethodRequest::Scoped(
        equations
            .regions
            .iter()
            .map(|region| {
                CommonScopedSpatialPolicy::new(
                    model.digest().unwrap(),
                    region.domain_id(),
                    if region.form.constraint_pairs().next().is_some() {
                        CommonSpatialPolicy::MiniP1
                    } else {
                        CommonSpatialPolicy::P1
                    },
                )
            })
            .collect(),
    );
    let resolved = ResolvedCommonPlan::resolve(
        &model,
        fsi_resources(&geometry),
        scoped,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::MinimumResidual,
            1e-11,
            1e-13,
            NonZeroUsize::new(20_000).unwrap(),
        )),
        None,
        Some(CommonBackwardEuler::from_seconds(0.05).unwrap()),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
    let plan = resolved.as_linear().unwrap();
    let NativeMeshResources::AdjacentPartitionSimplicial { mesh, .. } = plan.admission.resources()
    else {
        panic!("adjacent fixture");
    };
    let fields = plan
        .fields()
        .map(|(field, value_type)| {
            let entities = plan.field_coefficient_entities(field).unwrap();
            let components = value_type
                .shape()
                .extents()
                .iter()
                .map(|n| n.get() as usize)
                .product::<usize>();
            let block = |dimension| {
                let selected = entities
                    .iter()
                    .filter(|entity| entity.dimension() == dimension)
                    .collect::<Vec<_>>();
                if selected.is_empty() {
                    return None;
                }
                let values = selected
                    .iter()
                    .flat_map(|entity| {
                        (0..components).map(move |component| {
                            if value_type.dimension()
                                == DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap()
                                && dimension == 0
                                && component == 1
                            {
                                0.02 * mesh.mesh().vertices()[entity.index()][0]
                            } else {
                                0.
                            }
                        })
                    })
                    .collect();
                Some(CommonInitialValues::new(value_type.shape().clone(), values).unwrap())
            };
            CommonInitialField::new(model.digest().unwrap(), field, block(0), block(2)).unwrap()
        })
        .collect();
    let initial = plan.initial_state(0., fields).unwrap();
    let run =
        CommonTransientRunRequest::from_steps(resolved.clone(), initial, 2, vec![1, 2]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("shared coupled run must finish");
    };
    let trajectory = crate::CommonTrajectory::accept_spatial_transient(run, outputs).unwrap();
    for (_, step) in trajectory.spatial_steps().unwrap() {
        assert!(step.residual.unwrap().0 <= step.residual.unwrap().1);
        assert!(step.balance().is_ok());
    }
    let first = &trajectory.spatial_steps().unwrap()[0].1;
    let balance = first.balance().unwrap();
    // d=(0,0.02*x) on the unit-area solid: div(d)=0, epsilon_xy=0.01.
    // Integral(mu*epsilon:epsilon + lambda/2*div(d)^2) = 4*2*0.01^2.
    let initial_elastic = balance
        .volume_terms()
        .map(|(_, _, term)| term)
        .filter(|term| term.role() == crate::SpatialBalanceRole::StatePotential)
        .map(|term| term.previous().value())
        .sum::<f64>();
    assert!((initial_elastic - 0.0008).abs() < 1e-12);
    let velocity = plan
        .portable_realization()
        .fields()
        .iter()
        .find(|field| field.space() == Space::simplex_p1_bubble())
        .unwrap()
        .field();
    let entities = plan.field_coefficient_entities(velocity).unwrap();
    assert_eq!(
        entities
            .iter()
            .filter(|entity| entity.dimension() == 0)
            .count(),
        6
    );
    assert_eq!(
        entities
            .iter()
            .filter(|entity| entity.dimension() == 2)
            .count(),
        4
    );
    let offset = plan
        .fields()
        .take_while(|(field, _)| *field != velocity)
        .map(|(field, value_type)| {
            plan.field_coefficient_entities(field).unwrap().len()
                * value_type.shape().component_count().unwrap()
        })
        .sum::<usize>();
    let values = &first.state().linear_values().unwrap()[offset..offset + entities.len() * 2];
    assert!(
        values.iter().any(|value| value.abs() > 1e-8),
        "initial solid shear must drive a nonzero fluid response"
    );
    let bytes = trajectory.to_bytes().unwrap();
    let restored = crate::CommonTrajectory::from_bytes(&bytes, &resolved).unwrap();
    assert_eq!(restored.to_bytes().unwrap(), bytes);
}
