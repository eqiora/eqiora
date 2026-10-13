use super::*;
use eqiora_meshing::{
    AffineGeometryLinearization, FixedTopologyGeometryAction, FixedTopologyGeometryState,
    MeshQualityGate, SimplicialMesh,
};

#[test]
fn multiple_pullbacks_select_exact_rows_and_compose_primal_and_shape_actions() {
    let mut source = String::from(
        "public operator dyad(input left: spatial[1], input right: spatial[1]): spatial[2] = component(left, 0) * component(right, 1);\nmodel Mixture() {\n domain body = box(0, 1, 0, 1);\n parameter viscosity: kg/(m*s) = 2;\n",
    );
    for (i, rho) in [3, 7].into_iter().enumerate() {
        source += &format!(
            "parameter rho{i}: kg/m^3 = {rho};
             state v{i}: vector<m/s, 2> on body in smooth;
             variable p{i}: kg/(m*s^2) on body in h1;
             relation momentum{i} on body {{
               rho{i}*derivative(v{i}) + div(rho{i}*dyad(left=v{i}, right=v{i}))
               - div(2*viscosity*symmetric_part(grad(v{i})) - isotropic_lift(p{i})) = 0;
             }}
             relation continuity{i} on body {{ div(v{i}) = 0; }}\n",
        );
    }
    source += "}";
    let compiled = derive(&source).unwrap();
    let (fields, rows, time) = inputs(&compiled, true);
    let form = compiled
        .bind(
            ReferenceCell::simplex(2).unwrap(),
            &fields,
            &rows,
            Some(&time),
        )
        .unwrap();
    let selected = form
        .fields()
        .iter()
        .filter(|f| f.components == 2)
        .map(|f| f.field)
        .collect::<Vec<_>>();
    assert_eq!(selected.len(), 2);
    let mesh = SimplicialMesh::new(
        2,
        vec![vec![0., 0.], vec![1., 0.], vec![0., 1.]],
        vec![vec![0, 1, 2]],
        MeshQualityGate::new(0.01).unwrap(),
    )
    .unwrap();
    let previous = FixedTopologyGeometryState::<2>::reference(&mesh).unwrap();
    let current = FixedTopologyGeometryState::<2>::new(
        &mesh,
        vec![vec![0.01, -0.02], vec![1.04, 0.01], vec![-0.01, 1.02]],
    )
    .unwrap();
    let motion = FixedTopologyGeometryAction::new(&mesh, &previous, &current, 0.25).unwrap();
    let shape = AffineGeometryLinearization::new(
        motion.cell(0).unwrap().current_map().clone(),
        vec![0.02, -0.01],
        vec![0.01, -0.02, 0.03, 0.02],
    )
    .unwrap();
    let rule = simplex_duffy_gauss_legendre(2, 6).unwrap();
    let count = form.fields().iter().map(|f| f.range.len()).sum();
    let point = (0..count)
        .map(|i| 0.02 * i as f64 - 0.1)
        .collect::<Vec<_>>();
    let history = form
        .fields()
        .iter()
        .filter(|f| selected.contains(&f.field))
        .map(|f| (f.field, vec![0.03; f.range.len()]))
        .collect();
    let evaluate = |fields: &[RawId]| {
        let (primal, tangent) = form
            .prepare_gcl_cell(fields, &motion, 0, Some(&shape), &rule)
            .unwrap();
        (
            primal.linearize(&history, &point).unwrap(),
            tangent.unwrap().residual(&history, &point).unwrap(),
        )
    };
    let base = evaluate(&[]);
    let first = evaluate(&selected[..1]);
    let second = evaluate(&selected[1..]);
    let both = evaluate(&selected);
    let reversed = evaluate(&[selected[1], selected[0]]);
    // Each numerical pullback is additive and affects only its exact tested
    // Field. This fails if either action is omitted, duplicated, or misrouted.
    for field in form.fields() {
        let individual = if field.field == selected[0] {
            &first
        } else {
            &second
        };
        for row in field.range.clone() {
            close(
                both.0.residual[row],
                first.0.residual[row] + second.0.residual[row] - base.0.residual[row],
            );
            close(both.0.residual[row], individual.0.residual[row]);
            close(both.1[row], first.1[row] + second.1[row] - base.1[row]);
            close(both.1[row], individual.1[row]);
            close(both.1[row], reversed.1[row]);
            for column in 0..count {
                let i = row * count + column;
                close(
                    both.0.jacobian[i],
                    first.0.jacobian[i] + second.0.jacobian[i] - base.0.jacobian[i],
                );
                close(both.0.jacobian[i], reversed.0.jacobian[i]);
            }
        }
    }
    for action in [&first, &second] {
        assert!(
            action
                .0
                .residual
                .iter()
                .zip(&base.0.residual)
                .any(|(a, b)| (a - b).abs() > 1.0e-6)
        );
        assert!(
            action
                .1
                .iter()
                .zip(&base.1)
                .any(|(a, b)| (a - b).abs() > 1.0e-6)
        );
    }
    assert!(
        form.prepare_gcl_cell(&[selected[0], selected[0]], &motion, 0, None, &rule)
            .unwrap_err()
            .message()
            .contains("repeats")
    );
    let pressure = form
        .fields()
        .iter()
        .find(|f| f.components == 1)
        .unwrap()
        .field;
    assert!(
        form.prepare_gcl_cell(&[pressure], &motion, 0, None, &rule)
            .is_err()
    );
}
