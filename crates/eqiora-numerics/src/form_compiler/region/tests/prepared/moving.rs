use super::*;
use eqiora_meshing::{
    FixedTopologyGeometryAction, FixedTopologyGeometryState, MeshQualityGate, SimplicialMesh,
};

#[test]
fn moving_scalar_inventory_uses_previous_and_current_measures() {
    let mesh = SimplicialMesh::new(
        2,
        vec![vec![0., 0.], vec![1., 0.], vec![0., 1.]],
        vec![vec![0, 1, 2]],
        MeshQualityGate::new(0.05).unwrap(),
    )
    .unwrap();
    let state = |scale: f64, shift: f64| {
        FixedTopologyGeometryState::<2>::new(
            &mesh,
            mesh.vertices()
                .iter()
                .map(|point| vec![scale * point[0] + shift, scale * point[1]])
                .collect(),
        )
        .unwrap()
    };
    let form = heat(1., 3.);
    let field = form.fields()[0].field;
    let rule = simplex_duffy_gauss_legendre(2, 3).unwrap();
    // A material triangle with zero relative flux has area lambda^2/2.
    // Initial density 2 therefore gives conserved inventory 1 and density
    // 2/lambda^2. Constant density has no diffusive flux. This checks the
    // inventory action, not the separate conversion of physical to ALE flux.
    for shift in [0., 0.25] {
        let mut previous = state(1., 0.);
        let mut density = 2.;
        for (scale, expected) in [(1.5, 8. / 9.), (2., 0.5)] {
            let current = state(scale, shift);
            let action = FixedTopologyGeometryAction::new(&mesh, &previous, &current, 1.).unwrap();
            let cell = action.cell(0).unwrap();
            let history = BTreeMap::from([(field, vec![density; 3])]);
            let prepared = form
                .prepare_cell_with_history_geometry(cell.current_map(), cell.previous_map(), &rule)
                .unwrap();
            let local = prepared.evaluate(&history).unwrap();
            for residual in prepared.residual(&history, &[expected; 3]).unwrap() {
                close(residual, 0.);
            }
            // Each basis integrates to old_area/3. This expectation is
            // independent of the implementation's maps and mass matrix.
            for rhs in local.rhs() {
                close(*rhs, 1. / 3.);
            }
            close(expected * scale * scale / 2., 1.);
            let wrong = form.prepare_cell(cell.current_map(), &rule).unwrap();
            assert!(
                wrong
                    .residual(&history, &[expected; 3])
                    .unwrap()
                    .iter()
                    .all(|residual| residual.abs() > 0.1)
            );
            previous = current;
            density = expected;
        }
    }
    // Pure translation and the fixed-map specialization preserve the same
    // mass action, including every off-diagonal entry and nonconstant history.
    let previous = state(1., 0.);
    for shift in [0., 0.25] {
        let current = state(1., shift);
        let action = FixedTopologyGeometryAction::new(&mesh, &previous, &current, 1.).unwrap();
        let cell = action.cell(0).unwrap();
        let history = BTreeMap::from([(field, vec![1., 2., 4.])]);
        let moving = form
            .prepare_cell_with_history_geometry(cell.current_map(), cell.previous_map(), &rule)
            .unwrap()
            .evaluate(&history)
            .unwrap();
        let fixed = form
            .prepare_cell(cell.current_map(), &rule)
            .unwrap()
            .evaluate(&history)
            .unwrap();
        for (a, b) in moving.matrix().iter().zip(fixed.matrix()) {
            close(*a, *b);
        }
        for (a, b) in moving.rhs().iter().zip(fixed.rhs()) {
            close(*a, *b);
        }
    }
}
