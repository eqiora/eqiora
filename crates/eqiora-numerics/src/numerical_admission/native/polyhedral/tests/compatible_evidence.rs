//! Independent geometric moments and an explicit cochain gauge consumer.
use super::*;
use crate::discrete_space::DiscreteSpace;
use crate::region_assembly::mapping::RegionDofMap;
use crate::spatial_expression::Coefficient;
use eqiora_assembly::{
    AssemblyMap, CooAssembler, DofId, LinearSystem, LocalContribution, LocalUnknown,
};
use eqiora_core::{Id, entity::kinds};
use eqiora_meshing::{MeshGeometry, ReferenceCell, SimplicialMesh};
use eqiora_solver::CanonicalCsrSystemView;
use num_complex::Complex64 as C;

// Fixed before execution: 27 positive quadrature points, affine maps, at most
// nine contractions. Iterative projection uses the existing 1e-9 solve bound.
fn close<S: Coefficient>(actual: S, expected: C) {
    assert!(
        (C::new(actual.re(), actual.im()) - expected).norm()
            <= 4096. * f64::EPSILON * expected.norm().max(1.)
    );
}
fn policy() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-13,
        1e-14,
        NonZeroUsize::new(2000).unwrap(),
    )
    .unwrap()
}

#[test]
fn compatible_tetrahedral_moments_have_independent_actions_and_constraints() {
    // Ordinary Model -> Plan -> Run -> Result, including replay, succeeds first.
    super::execution::authenticated_polyhedral_equations_execute_real_and_complex_moments();
    for two in [false, true] {
        for permuted in [false, true] {
            for face in [false, true] {
                profile::<f64>(two, permuted, face, false, &REFERENCE_LINEAR_SOLVER);
                profile::<C>(two, permuted, face, true, &REFERENCE_LINEAR_SOLVER);
            }
        }
    }
}

fn profile<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send + std::iter::Sum>(
    two: bool,
    permuted: bool,
    face: bool,
    complex: bool,
    backend: &dyn LinearSolverBackend<S>,
) {
    let space = if face {
        Space::tetrahedral_face()
    } else {
        Space::tetrahedral_edge()
    };
    let (model, program, owner) = fixture_cells(
        &[(0..if two { 6 } else { 4 }).collect()],
        false,
        permuted,
        2.,
        complex,
        two,
        |source| {
            if face {
                source.replace("curl(curl(u))", "-grad(div(u))").replace(
                    "tangential_trace(-curl(u))",
                    "normal(isotropic_lift(div(u)))",
                )
            } else {
                source
            }
        },
    );
    let plan = ResolvedCommonPlan::resolve(
        &model,
        owner.clone(),
        if face {
            CommonSpatialPolicy::TetrahedralFace
        } else {
            CommonSpatialPolicy::TetrahedralEdge
        },
        CommonSolvePolicy::Linear(
            CommonLinearRequest::exact(policy(), REFERENCE_LINEAR_SOLVER.provider()).unwrap(),
        ),
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    let plan = plan.as_linear().unwrap();
    let field = plan.fields().next().unwrap().0;
    let equations = ExecutableLinearEquations::<S>::simplicial(&program, &owner.resources).unwrap();
    let NativeMeshResources::GmshSimplicial { mesh: envelope, .. } = &owner.resources else {
        unreachable!()
    };
    let mesh = envelope.mesh();
    let (mapping, forms, natural) = equations
        .simplicial_assembly(envelope, &equations.discretizations(space, None).unwrap())
        .unwrap();
    assert!(natural.is_empty());
    let mut assembler = CooAssembler::new(mapping.full_count()).unwrap();
    for cell in 0..mesh.cells().len() {
        let local = forms[0]
            .0
            .prepare_cell(
                &mesh.geometry_map(MeshEntity::new(3, cell)).unwrap(),
                &forms[0].1,
            )
            .unwrap()
            .evaluate(&BTreeMap::new())
            .unwrap();
        let signs = mapping.cell_signs(cell).unwrap();
        assembler
            .scatter(
                &mapping.cell_map(cell, false).unwrap(),
                &local.reoriented(signs, signs).unwrap(),
            )
            .unwrap();
    }
    let system = assembler.finish().unwrap();
    let entities = plan.field_coefficient_entities(field).unwrap();
    assert_eq!(
        entities.len(),
        if face {
            if two { 7 } else { 4 }
        } else if two {
            9
        } else {
            6
        }
    );
    assert_eq!(
        mapping.keys().map(|key| key.entity).collect::<Vec<_>>(),
        entities
    );
    let phase = if complex {
        C::new(1., 2.)
    } else {
        C::new(1., 0.)
    };
    let scalar_phase = if complex {
        <S as From<f64>>::from(1.) + S::imaginary_unit().unwrap() * 2.
    } else {
        <S as From<f64>>::from(1.)
    };
    let coefficients = entities
        .iter()
        .map(|entity| {
            let vertices = mesh.entity_vertices(*entity).unwrap();
            let a = &mesh.vertices()[vertices[0].index()];
            let b = &mesh.vertices()[vertices[1].index()];
            let integral = if face {
                let c = &mesh.vertices()[vertices[2].index()];
                let v: [f64; 3] = std::array::from_fn(|i| b[i] - a[i]);
                let w: [f64; 3] = std::array::from_fn(|i| c[i] - a[i]);
                let area = [
                    v[1] * w[2] - v[2] * w[1],
                    v[2] * w[0] - v[0] * w[2],
                    v[0] * w[1] - v[1] * w[0],
                ];
                (0..3)
                    .map(|i| (a[i] + b[i] + c[i]) / 3. * area[i] / 2.)
                    .sum()
            } else {
                -(a[1] + b[1]) / 2. * (b[0] - a[0]) + (a[0] + b[0]) / 2. * (b[1] - a[1])
            };
            scalar_phase * integral
        })
        .collect::<Vec<_>>();
    let expected = if face {
        if two {
            vec![0., 0., 0., 12., 12., -12., 12.]
        } else {
            vec![0., 0., 0., 12.]
        }
    } else if two {
        vec![0., 0., 0., 6., 0., 6., 0., -6., 0.]
    } else {
        vec![0., 0., 0., 6., 0., 0.]
    };
    for (&actual, expected) in coefficients.iter().zip(expected) {
        close(actual, phase * expected);
    }
    let action = system.matrix().multiply(&coefficients).unwrap();
    // Independent affine barycentric gradients and outward face incidences;
    // a correct scalar energy alone would not determine the complete action.
    let expected_action = if face {
        if two {
            vec![-3., 3., -3., 0., 3., -3., 3.]
        } else {
            vec![-3., 3., -3., 3.]
        }
    } else if two {
        vec![
            8. / 3.,
            -8. / 3.,
            0.,
            8. / 3.,
            -8. / 3.,
            8. / 3.,
            8. / 3.,
            -8. / 3.,
            0.,
        ]
    } else {
        vec![8. / 3., -8. / 3., 0., 8. / 3., 0., 0.]
    };
    for (&actual, expected) in action.iter().zip(expected_action) {
        close(actual, phase * expected);
    }
    let energy: C = coefficients
        .iter()
        .zip(&action)
        .map(|(x, a)| C::new(x.re(), x.im()).conj() * C::new(a.re(), a.im()))
        .sum();
    close(
        energy,
        C::new(
            phase.norm_sqr()
                * if face {
                    if two { 108. } else { 36. }
                } else if two {
                    48.
                } else {
                    16.
                },
            0.,
        ),
    );
    let derivative = plan.field_exterior_derivative(field).unwrap();
    if face {
        for (cell, row) in &derivative {
            let integral: S = row
                .iter()
                .map(|(entity, sign)| {
                    coefficients[entity.index()] * <S as From<f64>>::from(f64::from(*sign))
                })
                .sum();
            close(integral, phase * if cell.index() == 0 { 12. } else { 24. });
        }
        if two {
            let shared = MeshEntity::new(2, 3);
            assert_eq!(
                derivative[&MeshEntity::new(3, 0)]
                    .iter()
                    .find(|(e, _)| *e == shared)
                    .unwrap()
                    .1,
                1
            );
            assert_eq!(
                derivative[&MeshEntity::new(3, 1)]
                    .iter()
                    .find(|(e, _)| *e == shared)
                    .unwrap()
                    .1,
                -1
            );
        }
    } else {
        // D*C=0 is an exact integer statement, not a floating-point tolerance.
        for cell in 0..mesh.cells().len() {
            let mut composition = BTreeMap::<_, i32>::new();
            for (face, outer) in mesh.signed_boundary(MeshEntity::new(3, cell)).unwrap() {
                for (edge, inner) in &derivative[&face] {
                    *composition.entry(*edge).or_default() += i32::from(outer) * i32::from(*inner);
                }
            }
            assert!(composition.values().all(|value| *value == 0));
        }
        gauge(plan, field, &system, &coefficients, two, phase, backend);
    }
    reproduce(mesh, &mapping, space, &coefficients, face, phase);
}

fn gauge<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send + std::iter::Sum>(
    plan: &CommonLinearPlan,
    field: Id<kinds::Field>,
    system: &LinearSystem<S>,
    rotation: &[S],
    two: bool,
    phase: C,
    backend: &dyn LinearSolverBackend<S>,
) {
    let modes = plan.field_gradient_modes(field).unwrap();
    let n = rotation.len();
    let g = modes
        .values()
        .map(|column| {
            let mut dense = vec![0.; n];
            for (edge, sign) in column {
                dense[edge.index()] = f64::from(*sign);
            }
            dense
        })
        .collect::<Vec<_>>();
    assert_eq!(g.len(), if two { 4 } else { 3 });
    for column in &g {
        for value in system
            .matrix()
            .multiply(
                &column
                    .iter()
                    .copied()
                    .map(<S as From<f64>>::from)
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        {
            close(value, C::new(0., 0.));
        }
    }
    // Explicit Euclidean cochain gauge G^* c = 0; not a Coulomb/L2 gauge.
    // The returned independent modes make G^*G positive definite. This uses
    // the existing COO/CSR and backend owners, without filtering eigenvalues.
    let m = g.len();
    let gram = (0..m)
        .flat_map(|i| (0..m).map(move |j| (i, j)))
        .map(|(i, j)| {
            <S as From<f64>>::from(g[i].iter().zip(&g[j]).map(|(a, b)| a * b).sum::<f64>())
        })
        .collect();
    let rhs = g
        .iter()
        .map(|column| {
            column
                .iter()
                .zip(rotation)
                .map(|(a, b)| <S as From<f64>>::from(*a) * *b)
                .sum()
        })
        .collect();
    let map = AssemblyMap::new(
        (0..m).map(|i| Some(DofId::new(i))).collect(),
        (0..m).map(|i| LocalUnknown::Free(DofId::new(i))).collect(),
    )
    .unwrap();
    let mut assembler = CooAssembler::new(m).unwrap();
    assembler
        .scatter(&map, &LocalContribution::new(m, m, gram, rhs).unwrap())
        .unwrap();
    let canonical = CanonicalCsrSystemView::new(
        &assembler.finish().unwrap(),
        LinearOperatorProperties::General,
    )
    .unwrap();
    let solution = LinearSolveRequest::new(backend, policy())
        .solve(&canonical.linear_problem().unwrap())
        .unwrap();
    let projected = (0..n)
        .map(|i| {
            rotation[i]
                - g.iter()
                    .zip(solution.values())
                    .map(|(column, q)| <S as From<f64>>::from(column[i]) * *q)
                    .sum::<S>()
        })
        .collect::<Vec<_>>();
    let expected = if two {
        vec![
            12. / 5.,
            -12. / 5.,
            0.,
            6. / 5.,
            -12. / 5.,
            18. / 5.,
            12. / 5.,
            -18. / 5.,
            0.,
        ]
    } else {
        vec![1.5, -1.5, 0., 3., -1.5, 1.5]
    };
    for (&actual, expected) in projected.iter().zip(expected) {
        assert!(
            (C::new(actual.re(), actual.im()) - phase * expected).norm()
                <= 1e-9 * (phase * expected).norm().max(1.)
        );
    }
    for column in &g {
        let residual: S = column
            .iter()
            .zip(&projected)
            .map(|(a, b)| <S as From<f64>>::from(*a) * *b)
            .sum();
        assert!(C::new(residual.re(), residual.im()).norm() <= 1e-9);
    }
    for (before, after) in system
        .matrix()
        .multiply(rotation)
        .unwrap()
        .iter()
        .zip(system.matrix().multiply(&projected).unwrap())
    {
        assert!((C::new(before.re(), before.im()) - C::new(after.re(), after.im())).norm() <= 1e-9);
    }
}

fn reproduce<S: Coefficient + Send + Sync + std::iter::Sum>(
    mesh: &SimplicialMesh,
    mapping: &RegionDofMap<S>,
    space: Space,
    coefficients: &[S],
    face: bool,
    phase: C,
) {
    let element = DiscreteSpace::new(space, ReferenceCell::simplex(3).unwrap()).unwrap();
    let field = mapping.keys().next().unwrap().field;
    let mut traces = Vec::new();
    for cell in 0..mesh.cells().len() {
        let geometry = mesh.geometry_map(MeshEntity::new(3, cell)).unwrap();
        let keys = mapping.cell_field_keys(cell, field).unwrap();
        let signs = mapping.cell_signs(cell).unwrap();
        let table = element.tabulate_on(&geometry, &[0.25; 3]).unwrap();
        let mut point = [0.; 3];
        eqiora_meshing::GeometryMap::map_point(&geometry, &[0.25; 3], &mut point).unwrap();
        for axis in 0..3 {
            let value: S = keys
                .iter()
                .enumerate()
                .map(|(i, key)| {
                    coefficients[key.entity.index()]
                        * <S as From<f64>>::from(
                            f64::from(signs[i]) * table.value(i).unwrap()[axis],
                        )
                })
                .sum();
            close(
                value,
                phase
                    * if face {
                        point[axis]
                    } else {
                        [-point[1], point[0], 0.][axis]
                    },
            );
        }
        if face {
            let div: S = keys
                .iter()
                .enumerate()
                .map(|(i, key)| {
                    coefficients[key.entity.index()]
                        * <S as From<f64>>::from(f64::from(signs[i]) * table.divergence(i).unwrap())
                })
                .sum();
            close(div, phase * 3.);
        } else {
            for axis in 0..3 {
                let curl: S = keys
                    .iter()
                    .enumerate()
                    .map(|(i, key)| {
                        coefficients[key.entity.index()]
                            * <S as From<f64>>::from(
                                f64::from(signs[i]) * table.curl(i).unwrap()[axis],
                            )
                    })
                    .sum();
                close(curl, phase * if axis == 2 { 2. } else { 0. });
            }
        }
        if mesh.cells().len() == 2 {
            let weights = [0., 0.25, 0.25, 0.5, 0.];
            let point: [f64; 3] = std::array::from_fn(|i| weights[mesh.cells()[cell][i + 1]]);
            let table = element.tabulate_on(&geometry, &point).unwrap();
            let value: [f64; 3] = std::array::from_fn(|axis| {
                keys.iter()
                    .enumerate()
                    .map(|(i, key)| {
                        (key.entity.index() + 1) as f64
                            * f64::from(signs[i])
                            * table.value(i).unwrap()[axis]
                    })
                    .sum()
            });
            traces.push(value);
        }
    }
    if traces.len() == 2 {
        let directions = if face {
            vec![[6., 4., 3.]]
        } else {
            vec![[-2., 3., 0.], [-2., 0., 4.]]
        };
        for direction in directions {
            let left: f64 = (0..3).map(|i| direction[i] * traces[0][i]).sum();
            let right: f64 = (0..3).map(|i| direction[i] * traces[1][i]).sum();
            close(left, C::new(right, 0.));
        }
    }
}
