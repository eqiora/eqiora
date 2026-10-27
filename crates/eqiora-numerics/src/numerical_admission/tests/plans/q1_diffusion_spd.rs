//! Independently fixed three-row material energy after trace and essential reduction.
use super::*;
use crate::region_assembly::mapping::{FieldDof, RegionRealCoordinates};
use eqiora_meshing::MeshEntity;

#[test]
fn material_q1_trace_and_essential_reduction_preserve_the_symmetric_energy() {
    let source = super::plural_regions::chain_source(&[0, 1], &["alpha", "beta"])
        .replace(
            "parameter conductivity: 1 = 2;",
            "parameter kl:1=1;parameter kr:1=2;",
        )
        .replace("box(0, 1)", "box(0, 0.5)")
        .replace("box(1, 2)", "box(0.5, 1)")
        .replace("conductivity * grad(alpha)", "kl * grad(alpha)")
        .replace("conductivity * grad(beta)", "kr * grad(beta)")
        .replace(
            "value = alpha, conductivity = conductivity",
            "value = alpha, conductivity = kl",
        )
        .replace(
            "value = beta, conductivity = conductivity",
            "value = beta, conductivity = kr",
        );
    let (model, ids) = super::plural_regions::source_model(&source, &["alpha", "beta"]);
    let geometry = cartesian_interval();
    let owner = cartesian_box_resources(&geometry, &[4]);
    let resolved = ResolvedCommonPlan::resolve(
        &model,
        owner,
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(
            CommonLinearRequest::exact(
                SolverPlan::new(
                    LinearSolver::ConjugateGradient,
                    1e-12,
                    1e-14,
                    NonZeroUsize::new(100).unwrap(),
                )
                .unwrap()
                .with_preconditioner(PreconditionerPolicy::Jacobi),
                REFERENCE_LINEAR_SOLVER.provider(),
            )
            .unwrap(),
        ),
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
        None,
    )
    .unwrap();
    let plan = resolved.as_spatial().unwrap();
    let RecognizedNativeModel::Spatial(equations) = plan.admission.recognized_model() else {
        unreachable!()
    };
    let (mapping, input) = equations
        .static_assembly(
            plan.admission.resources(),
            &plan.admission.discretizations,
            &plan.admission.system_scaling,
            LinearOperatorProperties::General,
        )
        .unwrap();
    let fields = equations
        .q1_diffusion_fields(
            plan.admission.resources(),
            &plan.admission.discretizations,
            &plan.admission.system_scaling,
        )
        .unwrap()
        .unwrap();
    assert!(
        equations
            .q1_diffusion_is_anchored(&mapping, &input, &fields)
            .unwrap()
    );
    let NativeMeshResources::Cartesian { mesh, .. } = plan.admission.resources() else {
        unreachable!()
    };
    let (reduced, _, _) = RegionRealCoordinates::new(&mapping)
        .unwrap()
        .assemble_static(mesh.mesh(), &input)
        .unwrap();
    assert_eq!(reduced.matrix().rows(), 3);
    let index = |field, vertex| {
        mapping
            .free_dof(FieldDof {
                field,
                entity: MeshEntity::new(0, vertex),
                slot: 0,
                component: 0,
            })
            .unwrap()
            .index()
    };
    let indices = [
        index(ids["alpha"], 1),
        index(ids["alpha"], 2),
        index(ids["beta"], 3),
    ];
    // k/h is 4 on the left and 8 on the right. Congruent interface
    // identification adds both diagonal contributions at the middle node.
    let expected = [[8., -4., 0.], [-4., 12., -8.], [0., -8., 16.]];
    let rhs = [0., 0., 8.];
    for column in 0..3 {
        let mut basis = vec![0.; 3];
        basis[indices[column]] = 1.;
        let action = reduced.matrix().multiply(&basis).unwrap();
        for row in 0..3 {
            assert!((action[indices[row]] - expected[row][column]).abs() < 2e-12);
            assert!((reduced.rhs()[indices[row]] - rhs[row]).abs() < 2e-12);
        }
    }
    let mut exact = vec![0.; 3];
    for (index, value) in indices.into_iter().zip([1. / 3., 2. / 3., 5. / 6.]) {
        exact[index] = value;
    }
    for (actual, expected) in reduced
        .matrix()
        .multiply(&exact)
        .unwrap()
        .iter()
        .zip(reduced.rhs())
    {
        assert!((actual - expected).abs() < 2e-12);
    }
}
