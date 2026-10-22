use super::*;
use crate::simplicial_fsi::test_model::{self, ExactFsiTestFields as Fields};
use eqiora_meshing::{MeshEntity, MeshQualityGate, SimplicialMesh};

mod energy;

use super::shared::{Snapshot, snapshot};

fn resources(geometry: &CanonicalGeometryV1, mesh: &SimplicialMesh) -> AuthenticatedCommonMesh {
    // Reuse the scientific fixture's exact cells as a bounded provider observation.
    // This does not claim that a Gmsh binary was executed.
    let mut msh = format!(
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 {0} 1 {0}\n3 1 0 {0}\n",
        mesh.vertices().len()
    );
    for i in 1..=mesh.vertices().len() {
        msh += &format!("{i}\n");
    }
    for p in mesh.vertices() {
        msh += &format!("{} {} {}\n", p[0], p[1], p[2]);
    }
    msh += &format!(
        "$EndNodes\n$Elements\n1 {0} 1 {0}\n3 1 4 {0}\n",
        mesh.cells().len()
    );
    for (i, c) in mesh.cells().iter().enumerate() {
        msh += &format!(
            "{} {} {} {} {}\n",
            i + 1,
            c[0] + 1,
            c[1] + 1,
            c[2] + 1,
            c[3] + 1
        );
    }
    msh += "$EndElements\n";
    AuthenticatedCommonMesh::gmsh_4152(
        geometry.clone(),
        eqiora_artifact::GmshMeshPolicyV1::explicit(1e-12, 0.01, 64, 5.).unwrap(),
        msh.into_bytes(),
    )
    .unwrap()
}

fn model(geometry: &CanonicalGeometryV1) -> ModelEnvelope {
    let mut supports = Vec::new();
    for (body, boundaries) in [
        ("fluid", ["fluid_outer", "fluid_contact"]),
        ("solid", ["solid_outer", "solid_contact"]),
    ] {
        let parent = geometry.entity_set(body).unwrap();
        supports.push((body, parent, None));
        for name in boundaries {
            supports.push((
                name,
                geometry.entity_set(name).unwrap(),
                Some((body, parent)),
            ));
        }
    }
    let parameters = [
        ("fluid_density", 2., [1, -3, 0, 0, 0, 0, 0]),
        ("fluid_viscosity", 0.5, [1, -1, -1, 0, 0, 0, 0]),
        ("solid_density", 3., [1, -3, 0, 0, 0, 0, 0]),
        ("solid_mu", 4., [1, -1, -2, 0, 0, 0, 0]),
        ("solid_lambda", 2., [1, -1, -2, 0, 0, 0, 0]),
        ("zero_pressure", 0., [1, -1, -2, 0, 0, 0, 0]),
    ]
    .map(|(name, value, powers)| {
        (
            name,
            DynQuantity::new(value, DimExponents::from_integers(powers).unwrap()),
        )
    });
    compile_model(
        "tetrahedral-fsi.eqi",
        &test_model::source(3, false),
        geometry,
        "FixtureFsi",
        &supports,
        &parameters,
    )
}

fn difference(new: &Snapshot, old: &Snapshot) -> Snapshot {
    new.iter()
        .map(|(&field, values)| {
            (
                field,
                values
                    .iter()
                    .map(|(&entity, values)| {
                        (
                            entity,
                            values
                                .iter()
                                .zip(&old[&field][&entity])
                                .map(|(a, b)| a - b)
                                .collect(),
                        )
                    })
                    .collect(),
            )
        })
        .collect()
}

#[test]
fn tetrahedral_fsi_uses_common_history_with_independent_energy_balance() {
    let geometry = test_model::polyhedra::adjacent_boxes();
    let original = test_model::mesh_3d::mesh_3d(MeshQualityGate::new(0.1).unwrap());
    let resources = resources(&geometry, &original);
    let mesh = resources.simplicial_mesh().unwrap().mesh().clone();
    let model = model(&geometry);
    let recognized = RecognizedNativeAdmission::recognize(
        &model,
        resources.clone(),
        &CommonSpatialRequest::Uniform(CommonSpatialPolicy::P1),
        None,
    )
    .unwrap();
    let RecognizedNativeModel::Spatial(equations) = &recognized.recognized else {
        panic!("common linear owner");
    };
    let fluid = equations
        .regions
        .iter()
        .find(|r| r.form.constraint_pairs().next().is_some())
        .unwrap();
    let solid = equations
        .regions
        .iter()
        .find(|r| !r.form.kinematics().is_empty())
        .unwrap();
    let (velocity, pressure) = fluid.form.constraint_pairs().next().unwrap();
    let pair = solid.form.kinematics()[0].0;
    let fields = Fields {
        fluid_domain: fluid.domain_id(),
        solid_domain: solid.domain_id(),
        fluid_velocity: velocity.downcast().unwrap(),
        fluid_pressure: pressure.downcast().unwrap(),
        solid_velocity: pair.rate(),
        displacement: pair.state(),
    };
    let method = CommonMethodRequest::Scoped(vec![
        CommonScopedSpatialPolicy::new(
            model.digest().unwrap(),
            fields.fluid_domain,
            CommonSpatialPolicy::MiniP1,
        ),
        CommonScopedSpatialPolicy::new(
            model.digest().unwrap(),
            fields.solid_domain,
            CommonSpatialPolicy::P1,
        ),
    ]);
    let resolved = ResolvedCommonPlan::resolve(
        &model,
        resources,
        method,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::MinimumResidual,
            1e-12,
            1e-14,
            NonZeroUsize::new(20_000).unwrap(),
        )),
        None,
        Some(CommonBackwardEuler::from_seconds(0.05).unwrap().into()),
        &REFERENCE_LINEAR_SOLVER,
        None,
        None,
    )
    .unwrap();
    let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
    let plan = resolved.as_spatial().unwrap();
    let initial_fields = plan
        .fields()
        .map(|(field, ty)| {
            let width = ty.shape().component_count().unwrap();
            let entities = plan.field_coefficient_entities(field).unwrap();
            let vertex_values = entities
                .iter()
                .filter(|e| e.dimension() == 0)
                .flat_map(|e| {
                    if field == fields.displacement && mesh.vertices()[e.index()] == [1., 0.5, 0.5]
                    {
                        vec![0.02, 0.01, -0.005]
                    } else {
                        vec![0.; width]
                    }
                })
                .collect();
            let bubbles = entities.iter().filter(|e| e.dimension() == 3).count();
            CommonInitialField::new(
                model.digest().unwrap(),
                field,
                Some(CommonInitialValues::new(ty.shape().clone(), vertex_values).unwrap()),
                (bubbles > 0).then(|| {
                    CommonInitialValues::new(ty.shape().clone(), vec![0.; bubbles * width]).unwrap()
                }),
            )
            .unwrap()
        })
        .collect();
    let initial = plan.initial_state(Some(0.), initial_fields).unwrap();
    let initial = CommonState::from_bytes(&initial.to_bytes().unwrap(), &resolved).unwrap();
    let request =
        CommonTransientRunRequest::from_steps(resolved.clone(), initial.clone(), 2, vec![1, 2])
            .unwrap();
    let std::ops::ControlFlow::Continue(outputs) = request
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("not cancelled");
    };
    let mut old = snapshot(plan, &initial);
    for (_, step) in &outputs {
        let new = snapshot(plan, step.state());
        for (&entity, velocity) in &new[&fields.solid_velocity.erase()] {
            if let Some(fluid) = new[&fields.fluid_velocity.erase()].get(&entity) {
                assert_eq!(velocity, fluid);
            }
            for (i, v) in velocity.iter().enumerate() {
                assert!(
                    (new[&fields.displacement.erase()][&entity][i]
                        - old[&fields.displacement.erase()][&entity][i]
                        - 0.05 * v)
                        .abs()
                        < 1e-12
                );
            }
        }
        for field in [fields.fluid_velocity, fields.solid_velocity] {
            assert!(
                new[&field.erase()]
                    .values()
                    .flatten()
                    .any(|v| v.abs() > 1e-8),
                "prestrain must drive both Regions"
            );
        }
        let defect = energy::balance(&mesh, fields, &old, &new, 0.05);
        assert!(defect.abs() < 1e-9, "energy defect {defect}");
        // A changed bubble must falsify the independently integrated balance.
        let mut mutant = new.clone();
        mutant
            .get_mut(&fields.fluid_velocity.erase())
            .unwrap()
            .iter_mut()
            .find(|(entity, _)| entity.dimension() == 3)
            .unwrap()
            .1[0] += 1.;
        assert!(energy::balance(&mesh, fields, &old, &mutant, 0.05).abs() > 1e-6);
        old = new;
    }
    let restart =
        CommonState::from_bytes(&outputs[0].1.state().to_bytes().unwrap(), &resolved).unwrap();
    let rerun = plan
        .advance_scalar(&restart, &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    assert_eq!(rerun.state(), outputs[1].1.state());
    let trajectory = crate::CommonTrajectory::accept_spatial_transient(request, outputs).unwrap();
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
