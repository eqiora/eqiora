//! Common mapped solve with a sealed geometry action; source motion binding is separate.
use super::*;
use crate::form_compiler::region::{BoundRegionForm, RegionTimeBinding};
use eqiora_meshing::{
    FixedTopologyGeometryAction, FixedTopologyGeometryState, MeshQualityGate, SimplicialMesh,
    simplex_duffy_gauss_legendre,
};
use eqiora_solver::LinearSolveRequest;

mod source;

fn form() -> BoundRegionForm<f64> {
    let source = "model Inventory() {
        domain body = box(0,1,0,1);
        state density: kg/m^2 on body in h1;
        relation balance on body {
            derivative(density) - div(1[m^2/s]*grad(density)) = 0;
        }
    }";
    let (transaction, model, _) = compile("inventory.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let domain = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Domain(domain) => Some(domain.id().erase()),
            _ => None,
        })
        .unwrap();
    let compiled = CompiledRegionForm::<f64>::derive(&program, domain, 2).unwrap();
    let fields = compiled
        .fields()
        .map(|(field, ty)| RegionFieldBinding {
            field,
            space: Space::continuous_lagrange(NonZeroU16::MIN),
            scale: DynQuantity::new(1., ty.dimension()),
        })
        .collect::<Vec<_>>();
    let rows = compiled
        .rows()
        .map(|(relation, _, ty)| {
            (
                relation,
                DynQuantity::new(
                    1.,
                    ty.dimension()
                        .mul(DimExponents::from_integers([0, 2, 0, 0, 0, 0, 0]).unwrap())
                        .unwrap()
                        .pow(-1, 1)
                        .unwrap(),
                ),
            )
        })
        .collect();
    compiled
        .bind(
            ReferenceCell::simplex(2).unwrap(),
            &fields,
            &rows,
            Some(&RegionTimeBinding {
                step: DynQuantity::new(
                    1.,
                    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap(),
                ),
                states: vec![],
            }),
        )
        .unwrap()
}

fn reference() -> SimplicialMesh {
    SimplicialMesh::new(
        2,
        vec![
            vec![0., 0.],
            vec![1., 0.],
            vec![1., 1.],
            vec![0., 1.],
            vec![0.5, 0.5],
        ],
        vec![vec![0, 1, 4], vec![1, 2, 4], vec![2, 3, 4], vec![3, 0, 4]],
        MeshQualityGate::new(0.05).unwrap(),
    )
    .unwrap()
}

fn execute(
    form: &BoundRegionForm<f64>,
    mesh: &SimplicialMesh,
    action: Option<FixedTopologyGeometryAction<2>>,
    previous: &[f64],
) -> Result<Vec<f64>, Diagnostic> {
    let domain = form.domain();
    let field = form.fields()[0].field;
    let mapping = RegionDofMap::new(
        mesh,
        &BTreeMap::from([(domain, form.fields().to_vec())]),
        form.reference_cell(),
        &[domain; 4],
        &[],
        &BTreeMap::new(),
    )?;
    let history = mapping.recover(previous, &[field])?;
    let policy = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-13,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )?;
    let output = mapping.solve(
        mesh,
        RegionSolveInput {
            forms: vec![(form.clone(), simplex_duffy_gauss_legendre(2, 3).unwrap())],
            natural: vec![],
            previous: Some(history),
            geometry_action: action,
        },
        NonZeroUsize::MIN,
        LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, policy),
        |reactions, values| reactions.recover(values),
    )?;
    Ok(output.fields[&field]
        .coefficients
        .values()
        .copied()
        .collect())
}

#[test]
fn shared_solve_preserves_expanding_material_volume_inventory() {
    let reference = reference();
    let form = form();
    for shift in [0., 0.25] {
        let motion = source::Motion::new(
            &format!("x=(1+0.5[1/s]*time())*xi+{shift}[m/s]*time(),y=(1+0.5[1/s]*time())*eta"),
            false,
        );
        let mut previous = FixedTopologyGeometryState::<2>::reference(&reference).unwrap();
        let mut values = vec![2.; 5];
        // Closed material volume: area=lambda^2 and density=2/lambda^2.
        // Relative advective flux is zero. This is distinct from constant
        // physical density with nonzero mesh-relative boundary transport.
        for (time, scale, expected) in [(1., 1.5, 8. / 9.), (2., 2., 0.5)] {
            let current = motion.state(&reference, time).unwrap();
            for (point, original) in current.coordinates().iter().zip(reference.vertices()) {
                assert_eq!(
                    *point,
                    vec![scale * original[0] + shift * time, scale * original[1]]
                );
            }
            let action =
                FixedTopologyGeometryAction::new(&reference, &previous, &current, 1.).unwrap();
            let mesh = action.current_mesh().clone();
            values = execute(&form, &mesh, Some(action), &values).unwrap();
            for value in &values {
                assert!((value - expected).abs() < 1e-11, "{values:?}");
            }
            // Integrate the recovered P1 field: four equal triangles, each
            // with area lambda^2/4 and average of its three vertex values.
            let inventory: f64 = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
                .iter()
                .map(|cell| scale * scale / 12. * cell.iter().map(|&i| values[i]).sum::<f64>())
                .sum();
            assert!((inventory - 2.).abs() < 1e-11);
            previous = current;
        }
    }
    let previous = FixedTopologyGeometryState::<2>::reference(&reference).unwrap();
    let current = FixedTopologyGeometryState::new(
        &reference,
        reference
            .vertices()
            .iter()
            .map(|p| vec![1.5 * p[0], 1.5 * p[1]])
            .collect(),
    )
    .unwrap();
    let action = FixedTopologyGeometryAction::new(&reference, &previous, &current, 1.).unwrap();
    let error = execute(&form, &reference, Some(action.clone()), &[2.; 5]).unwrap_err();
    assert!(
        error
            .message()
            .contains("current cell topology or coordinates"),
        "{error:?}"
    );
    let wrong_step =
        FixedTopologyGeometryAction::new(&reference, &previous, &current, 0.5).unwrap();
    let error = execute(&form, action.current_mesh(), Some(wrong_step), &[2.; 5]).unwrap_err();
    assert!(
        error.message().contains("exact mesh or time step"),
        "{error:?}"
    );
    let wrong = execute(&form, action.current_mesh(), None, &[2.; 5]).unwrap();
    assert!(wrong.iter().all(|value| (value - 2.).abs() < 1e-11));
    // Fixed specialization uses exactly the established solve path.
    let fixed = FixedTopologyGeometryAction::new(&reference, &previous, &previous, 1.).unwrap();
    let initial = [1., 2., 3., 4., 5.];
    let a = execute(&form, &reference, Some(fixed), &initial).unwrap();
    let b = execute(&form, &reference, None, &initial).unwrap();
    assert_eq!(a, b);
}
