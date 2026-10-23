//! The existing coupled equations use the same Newton owner as non-FSI Fields.
use super::shared::snapshot;
use super::*;
use crate::common_result::evidence::CommonSolveHistory;

#[test]
fn nonlinear_fixed_mesh_fsi_uses_shared_residuals_interfaces_and_kinematic_history() {
    let geometry = fsi_geometry();
    let source = String::from(
        "public operator dyad(input left:spatial[1],input right:spatial[1]):spatial[2]=component(left,0)*component(right,1);\n",
    ) + &FSI_COMPONENT
        .replace("FixedReferenceFsi2d", "OrdinaryCoupled")
        .replace(
            "fluid_density * derivative(fluid_velocity)",
            "fluid_density * derivative(fluid_velocity) + div(fluid_density * dyad(left=fluid_velocity,right=fluid_velocity))",
        );
    let model = fsi_model_from_source(&geometry, &source, "OrdinaryCoupled");
    let resources = fsi_resources(&geometry);
    let recognized = RecognizedNativeAdmission::recognize(
        &model,
        resources.clone(),
        &CommonSpatialRequest::Uniform(CommonSpatialPolicy::P1),
        None,
        &BTreeMap::new(),
    )
    .unwrap();
    let RecognizedNativeModel::Spatial(equations) = &recognized.recognized else {
        panic!("coupled equations must use common spatial admission");
    };
    let (velocity, _) = equations
        .regions
        .iter()
        .find_map(|region| region.form.constraint_pairs().next())
        .unwrap();
    let pair = equations
        .regions
        .iter()
        .find_map(|region| region.form.kinematics().first())
        .unwrap()
        .0;
    let method = CommonMethodRequest::Scoped(
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
        resources,
        method,
        CommonSolvePolicy::Newton {
            linear: exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                0.,
                1e-12,
                NonZeroUsize::new(20_000).unwrap(),
            ),
            nonlinear: NonlinearSolvePlan::new(0., 1e-12, NonZeroUsize::new(12).unwrap(), 12)
                .unwrap(),
        },
        None,
        Some(CommonTemporalPolicy::backward_euler(0.05).unwrap()),
        &REFERENCE_LINEAR_SOLVER,
        None,
        None,
    )
    .unwrap();
    let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
    let plan = resolved.as_spatial().unwrap();
    let NativeMeshResources::AffineTriangleSimplicial { mesh, .. } = plan.admission.resources()
    else {
        panic!("exact adjacent mesh");
    };
    let fields = plan
        .fields()
        .map(|(field, ty)| {
            let entities = plan.field_coefficient_entities(field).unwrap();
            let width = ty.shape().component_count().unwrap();
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
                        (0..width).map(move |component| {
                            if field == pair.state() && dimension == 0 && component == 1 {
                                0.02 * mesh.mesh().vertices()[entity.index()][0]
                            } else {
                                0.
                            }
                        })
                    })
                    .collect();
                Some(CommonInitialValues::new(ty.shape().clone(), values).unwrap())
            };
            CommonInitialField::new(model.digest().unwrap(), field, block(0), block(2)).unwrap()
        })
        .collect();
    let initial = plan.initial_state(Some(0.), fields).unwrap();
    let original = initial.to_bytes().unwrap();
    let mut previous = snapshot(plan, &initial);
    let run =
        CommonTransientRunRequest::from_steps(resolved.clone(), initial.clone(), 2, vec![1, 2])
            .unwrap();
    let std::ops::ControlFlow::Continue(states) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("uninterrupted common coupled Run");
    };
    assert_eq!(initial.to_bytes().unwrap(), original);
    for (_, accepted) in &states {
        let CommonSolveHistory::Newton {
            iterations,
            linear_solves,
            ..
        } = &accepted.execution.as_ref().unwrap().solve
        else {
            panic!("shared Newton history");
        };
        assert!(*iterations > 0);
        assert_eq!(*iterations, linear_solves.len());
        let current = snapshot(plan, accepted.state());
        assert!(
            current[&velocity]
                .values()
                .flatten()
                .any(|value| value.abs() > 1e-8),
            "initial solid shear must drive a nonzero fluid response"
        );
        // The quotient enforces the same physical velocity at every shared
        // vertex, with no coupling loop or independently accepted substep.
        let shared = current[&velocity]
            .iter()
            .filter_map(|(entity, fluid)| {
                current[&pair.rate().erase()]
                    .get(entity)
                    .map(|solid| (fluid, solid))
            })
            .collect::<Vec<_>>();
        assert_eq!(shared.len(), 3);
        for (fluid, solid) in shared {
            assert_eq!(fluid, solid);
        }
        for (entity, displacement) in &current[&pair.state().erase()] {
            for ((now, old), rate) in displacement
                .iter()
                .zip(&previous[&pair.state().erase()][entity])
                .zip(&current[&pair.rate().erase()][entity])
            {
                assert!((now - old - 0.05 * rate).abs() < 1e-13);
            }
        }
        let replayed =
            CommonState::from_bytes(&accepted.state().to_bytes().unwrap(), &resolved).unwrap();
        assert_eq!(replayed, *accepted.state());
        previous = current;
    }
    let trajectory = crate::CommonTrajectory::accept_spatial_transient(run, states).unwrap();
    let result = crate::CommonResult::accept_trajectory(0., trajectory).unwrap();
    let bytes = result.to_bytes().unwrap();
    assert_eq!(
        crate::CommonResult::from_bytes(&bytes, &resolved)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
}
