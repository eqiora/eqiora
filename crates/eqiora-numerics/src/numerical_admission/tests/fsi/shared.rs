use super::*;

#[test]
fn fixed_reference_equations_bind_through_shared_simplicial_regions() {
    let geometry = fsi_geometry();
    let model = fsi_model(&geometry);
    let recognized =
        RecognizedNativeAdmission::recognize(&model, fsi_resources(&geometry)).unwrap();
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
    let recognized =
        RecognizedNativeAdmission::recognize(&model, fsi_resources(&geometry)).unwrap();
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
