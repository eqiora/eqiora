use super::*;

fn independent_tetrahedron_gradients(points: [[f64; 3]; 4]) -> ([[f64; 3]; 4], f64) {
    let edge = |vertex: usize| std::array::from_fn(|axis| points[vertex][axis] - points[0][axis]);
    let first = edge(1);
    let second = edge(2);
    let third = edge(3);
    let determinant = dot_3(first, cross_3(second, third));
    assert!(determinant > 0.0);
    let gradients = [
        [0.0; 3],
        scale_3(cross_3(second, third), 1.0 / determinant),
        scale_3(cross_3(third, first), 1.0 / determinant),
        scale_3(cross_3(first, second), 1.0 / determinant),
    ];
    let first_gradient =
        std::array::from_fn(|axis| -gradients[1][axis] - gradients[2][axis] - gradients[3][axis]);
    (
        [first_gradient, gradients[1], gradients[2], gradients[3]],
        determinant / 6.0,
    )
}

fn cross_3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot_3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn scale_3(vector: [f64; 3], scale: f64) -> [f64; 3] {
    vector.map(|value| scale * value)
}

#[test]
fn skew_tetrahedral_extension_closes_an_independent_barycentric_residual() {
    let (reference, reference_partition) = refined_partition_3d();
    let coordinates = reference
        .vertices()
        .iter()
        .map(|point| {
            vec![
                point[0] + 0.2 * point[1] - 0.1 * point[2],
                0.3 * point[0] + 1.1 * point[1] + 0.15 * point[2],
                -0.2 * point[0] + 0.1 * point[1] + 0.9 * point[2],
            ]
        })
        .collect();
    let mesh = SimplicialMesh::new(
        3,
        coordinates,
        reference.cells().to_vec(),
        MeshQualityGate::new(0.005).expect("valid skew quality gate"),
    )
    .expect("positive affine image remains a conforming tetrahedral mesh");
    let partition = exact_partition(
        &mesh,
        reference_partition
            .domain_cells(fluid_domain())
            .expect("fluid Domain")
            .to_vec(),
        reference_partition
            .domain_cells(solid_domain())
            .expect("solid Domain")
            .to_vec(),
    );
    let motion = seal_motion(&mesh, &partition);
    let displacement = motion
        .apply(solid_displacement(), &solid_field_3d(&mesh, &partition))
        .expect("skew harmonic action applies");

    for interior in motion.fluid_interior_vertices() {
        for (component, _) in displacement[interior.index()].iter().enumerate() {
            let mut residual = 0.0;
            let mut absolute_action = 0.0;
            for cell in partition
                .domain_cells(fluid_domain())
                .expect("fluid Domain")
            {
                let vertices = mesh
                    .entity_vertices(MeshEntity::new(3, cell.index()))
                    .expect("accepted fluid tetrahedron owns vertices");
                let Some(test_basis) = vertices
                    .iter()
                    .position(|vertex| vertex.index() == interior.index())
                else {
                    continue;
                };
                let points: [[f64; 3]; 4] = std::array::from_fn(|local| {
                    mesh.vertices()[vertices[local].index()]
                        .clone()
                        .try_into()
                        .expect("3D test coordinate")
                });
                let (gradients, volume) = independent_tetrahedron_gradients(points);
                let field_gradient: [f64; 3] = std::array::from_fn(|axis| {
                    vertices
                        .iter()
                        .enumerate()
                        .map(|(local, vertex)| {
                            displacement[vertex.index()][component] * gradients[local][axis]
                        })
                        .sum::<f64>()
                });
                let action = volume
                    * gradients[test_basis]
                        .iter()
                        .zip(field_gradient)
                        .map(|(left, right)| left * right)
                        .sum::<f64>();
                residual += action;
                absolute_action += action.abs();
            }
            let tolerance = 8192.0 * f64::EPSILON * (1.0 + absolute_action);
            assert!(residual.abs() <= tolerance, "{residual:e} > {tolerance:e}");
        }
    }
}

#[test]
fn action_rejects_invalid_shape_support_and_finiteness() {
    let (mesh, partition) = refined_partition();
    let motion = seal_motion(&mesh, &partition);
    assert!(
        motion
            .apply(solid_displacement(), &BTreeMap::new())
            .is_err()
    );

    let fluid_only = partition
        .domain_vertices(fluid_domain())
        .expect("fluid Domain")
        .iter()
        .find(|vertex| {
            !partition
                .domain_vertices(solid_domain())
                .expect("solid Domain")
                .contains(vertex)
        })
        .expect("test mesh owns a fluid-only vertex")
        .index();
    let mut unsupported = solid_field(&mesh, &partition, [0.0; 3], [0.0; 3]);
    unsupported.insert(VertexId::new(fluid_only), [1.0, 0.0]);
    assert!(motion.apply(solid_displacement(), &unsupported).is_err());

    let mut non_finite = solid_field(&mesh, &partition, [0.0; 3], [0.0; 3]);
    non_finite
        .get_mut(
            &partition
                .domain_vertices(solid_domain())
                .expect("solid Domain")[0],
        )
        .expect("solid coefficient")[0] = f64::NAN;
    assert!(motion.apply_jvp(solid_displacement(), &non_finite).is_err());
}

#[test]
fn sealed_action_rejects_another_reference_geometry() {
    let (mesh, partition) = refined_partition();
    let motion = seal_motion(&mesh, &partition);
    motion
        .validate_reference(&mesh, &partition)
        .expect("exact root replays");

    let mut vertices = mesh.vertices().to_vec();
    vertices[0][0] -= 0.125;
    let changed = SimplicialMesh::new(
        DIMENSION,
        vertices,
        mesh.cells().to_vec(),
        mesh.quality_gate(),
    )
    .expect("changed geometry remains admissible");
    assert!(motion.validate_reference(&changed, &partition).is_err());
}

#[test]
fn singular_common_problem_fails_closed() {
    let singular = DenseSpdOperator::new(&[1.0, -1.0, -1.0, 1.0], 2)
        .expect("shape and symmetry are admissible before the solve");
    let rhs = [1.0, 0.0];
    let problem = LinearProblem::new(
        &singular,
        &rhs,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .expect("bounded problem has a valid shape");
    assert!(reference_solver().solve(&problem).is_err());
}

#[test]
fn general_bicgstab_policy_is_rejected_before_motion_assembly() {
    let (mesh, partition) = refined_partition();
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(1_000).expect("positive iteration limit"),
    )
    .expect("valid general solver plan");
    let general = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan);
    assert!(
        P1HarmonicMeshMotionAction::<2>::new(
            &mesh,
            &partition,
            motion_policy(general.plan()),
            general
        )
        .is_err()
    );
}
