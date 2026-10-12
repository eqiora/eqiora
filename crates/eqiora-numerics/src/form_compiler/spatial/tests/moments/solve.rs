//! Source/compiler to shared solver/recovery; native Plan lineage is not claimed here.
use super::*;
use crate::region_assembly::mapping::RegionDofMap;
use eqiora_meshing::{MeshQualityGate, SimplicialMesh};
use eqiora_solver::{
    LinearSolveRequest, LinearSolver, LinearSolverBackend, REFERENCE_LINEAR_SOLVER, SolverPlan,
};
use std::num::NonZeroUsize;

fn box_mesh(permuted: bool) -> SimplicialMesh {
    let vertices = (0..8)
        .map(|i| {
            vec![
                2.0 * (i & 1) as f64,
                3.0 * ((i >> 1) & 1) as f64,
                4.0 * ((i >> 2) & 1) as f64,
            ]
        })
        .collect();
    let mut cells = vec![
        vec![0, 1, 3, 7],
        vec![0, 3, 2, 7],
        vec![0, 2, 6, 7],
        vec![0, 6, 4, 7],
        vec![0, 4, 5, 7],
        vec![0, 5, 1, 7],
    ];
    if permuted {
        for cell in &mut cells {
            cell.swap(0, 1);
            cell.swap(2, 3);
        }
    }
    SimplicialMesh::new(3, vertices, cells, MeshQualityGate::new(0.01).unwrap()).unwrap()
}

fn execute<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send>(
    complex: bool,
    face: bool,
    backend: &dyn LinearSolverBackend<S>,
) {
    let mut source = super::source(complex, face, None, 6);
    let potential = "2*coordinate(0)+3*coordinate(1)+4*coordinate(2)";
    source = source.replace(
        "variable potential: m^2",
        if complex {
            "variable potential: complex<m>"
        } else {
            "variable potential: m"
        },
    );
    source = source.replace(
        "potential = coordinate(1)^2",
        &if complex {
            format!("potential = math.complex({potential},2*({potential}))")
        } else {
            format!("potential = {potential}")
        },
    );
    let operator = if face {
        "-grad(div(u))"
    } else {
        "curl(curl(u))"
    };
    source = source.replace(
        &format!("a*({operator}) = 0"),
        &format!("a*({operator}) + u = grad(potential)"),
    );
    let form = super::derive::<S>(&source).unwrap();
    let bound = super::bind(&form, face).unwrap();
    let field = form.fields()[0].0;
    let domain = form.domain();
    let policy = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-13,
        1e-14,
        NonZeroUsize::new(2000).unwrap(),
    )
    .unwrap();
    for permuted in [false, true] {
        let mesh = box_mesh(permuted);
        let layouts = BTreeMap::from([(domain, bound.fields().to_vec())]);
        let mapping = RegionDofMap::<S>::new(
            &mesh,
            &layouts,
            ReferenceCell::simplex(3).unwrap(),
            &[domain; 6],
            &[],
            &BTreeMap::new(),
        )
        .unwrap();
        let output = mapping
            .solve(
                &mesh,
                crate::region_assembly::mapping::RegionSolveInput {
                    operator_properties: eqiora_solver::LinearOperatorProperties::General,
                    geometry_action: None,
                    forms: vec![(bound.clone(), simplex_duffy_gauss_legendre(3, 3).unwrap())],
                    natural: vec![],
                    previous: None,
                    prescribed_states: BTreeMap::new(),
                },
                NonZeroUsize::MIN,
                LinearSolveRequest::new(backend, policy),
                |reactions, values| reactions.recover(values),
            )
            .unwrap();
        mapping.validate_physical(&output.fields).unwrap();
        let recovered = &output.fields[&field];
        assert_eq!(
            recovered.space,
            if face {
                Space::tetrahedral_face()
            } else {
                Space::tetrahedral_edge()
            }
        );
        assert_eq!(recovered.coefficients.len(), if face { 18 } else { 19 });
        // Exact constant solution (2,3,4), or (1+2i)*(2,3,4). Its curl and
        // divergence vanish, leaving the mass term and the prescribed gradient.
        // Integrate the constant vector independently on each canonical entity.
        for (key, actual) in &recovered.coefficients {
            let vertices = mesh.entity_vertices(key.entity).unwrap();
            let a = &mesh.vertices()[vertices[0].index()];
            let b = &mesh.vertices()[vertices[1].index()];
            let v: [f64; 3] = std::array::from_fn(|i| b[i] - a[i]);
            let measure = if face {
                let c = &mesh.vertices()[vertices[2].index()];
                let w: [f64; 3] = std::array::from_fn(|i| c[i] - a[i]);
                [
                    v[1] * w[2] - v[2] * w[1],
                    v[2] * w[0] - v[0] * w[2],
                    v[0] * w[1] - v[1] * w[0],
                ]
                .map(|x| x * 0.5)
            } else {
                v
            };
            let moment = (0..3).map(|i| [2., 3., 4.][i] * measure[i]).sum::<f64>();
            let expected = C::new(moment, if complex { 2. * moment } else { 0. });
            // Binary64 iterative solve at 1e-13; reserve 1e-9 relative error for
            // bounded conditioning and coefficient recovery, independent of output.
            assert!(
                (C::new(actual.re(), actual.im()) - expected).norm()
                    <= 1e-9 * expected.norm().max(1.)
            );
        }
        let mut changed = layouts;
        changed.get_mut(&domain).unwrap()[0].scale = 2.;
        let mismatched = RegionDofMap::<S>::new(
            &mesh,
            &changed,
            ReferenceCell::simplex(3).unwrap(),
            &[domain; 6],
            &[],
            &BTreeMap::new(),
        )
        .unwrap();
        let error = mismatched
            .solve(
                &mesh,
                crate::region_assembly::mapping::RegionSolveInput {
                    operator_properties: eqiora_solver::LinearOperatorProperties::General,
                    geometry_action: None,
                    forms: vec![(bound.clone(), simplex_duffy_gauss_legendre(3, 3).unwrap())],
                    natural: vec![],
                    previous: None,
                    prescribed_states: BTreeMap::new(),
                },
                NonZeroUsize::MIN,
                LinearSolveRequest::new(backend, policy),
                |reactions, values| reactions.recover(values),
            )
            .err()
            .unwrap();
        assert!(error.message().contains("exact Field layouts"));
    }
}

#[test]
fn real_and_complex_moment_fields_use_the_existing_global_solver_and_recovery() {
    for face in [false, true] {
        execute::<f64>(false, face, &REFERENCE_LINEAR_SOLVER);
        execute::<C>(true, face, &REFERENCE_LINEAR_SOLVER);
    }
}
