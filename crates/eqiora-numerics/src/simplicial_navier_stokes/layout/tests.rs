use super::*;
use eqiora_assembly::LocalUnknown;
use eqiora_meshing::MeshQualityGate;

#[test]
fn exact_fields_recover_nonuniform_coefficients_with_either_pressure_policy() {
    let mesh = SimplicialMesh::new(
        2,
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![vec![0, 1, 2]],
        MeshQualityGate::new(0.01).unwrap(),
    )
    .unwrap();
    let form = StepForm::reference(4.0, 0.1, 0.2).unwrap();
    let fixed = [Some([1.0, 2.0]), None, None];
    let vertices = [[1.0, 2.0], [11.0, 21.0], [12.0, 22.0]];
    let bubbles = [[100.0, -200.0]];
    let pressure = [1000.0, 1001.0, 1002.0];
    for with_gauge in [false, true] {
        let layout = FlowLayout::new(&mesh, &fixed, with_gauge, &form).unwrap();
        assert_eq!(layout.full_size(), 11 + usize::from(with_gauge));
        assert_eq!(layout.reduced_size(), 9 + usize::from(with_gauge));
        let multiplier = with_gauge.then_some(7.25);
        let point = layout
            .initial_point(&fixed, &vertices, &bubbles, &pressure, multiplier)
            .unwrap();
        let recovered = layout.reconstruct(&point).unwrap();
        assert_eq!(
            recovered,
            (
                vertices.to_vec(),
                bubbles.to_vec(),
                pressure.to_vec(),
                multiplier
            )
        );
        let mapped = layout
            .cell_map(0, true)
            .unwrap()
            .unknowns()
            .iter()
            .map(|unknown| match unknown {
                LocalUnknown::Fixed(value) => *value,
                LocalUnknown::Free(dof) => point[dof.index()],
            })
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        for field in form.form.fields() {
            if field.field == form.vector {
                expected.extend(
                    vertices
                        .iter()
                        .flatten()
                        .chain(bubbles.iter().flatten())
                        .copied(),
                );
            } else {
                expected.extend(pressure);
            }
        }
        assert_eq!(mapped, expected);
        let mut incompatible = vertices;
        incompatible[0][0] += 1e-6;
        assert!(
            layout
                .initial_point(&fixed, &incompatible, &bubbles, &pressure, multiplier)
                .is_err()
        );
        if !with_gauge {
            let vector_constraint = AlgebraicConstraint::ZeroIntegral {
                field: form.vector.downcast().unwrap(),
            };
            assert!(
                layout
                    .mapping
                    .clone()
                    .with_algebraic_constraints(&[vector_constraint])
                    .is_err()
            );
        }
    }
}

#[test]
fn common_cell_actions_keep_physical_history_separate_from_current_essential_values() {
    use crate::region_assembly::residual::mapped_local_point;
    use eqiora_graph::{GraphStore, InMemoryGraphStore};
    use eqiora_meshing::{MeshGeometry, simplex_duffy_gauss_legendre};
    use eqiora_schema::kernel::KernelNode;
    use eqiora_sem::KernelProgram;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let source = "public operator outer_product(input left: spatial[1], input right: spatial[1]): spatial[2] = component(left, 0) * component(right, 1);
model History() {
 domain body = box(0, 1, 0, 1);
 parameter density: kg/m^3 = 4;
 parameter viscosity: kg/(m*s) = 0.1;
 state velocity: vector<m/s, 2> on body;
 variable pressure: kg/(m*s^2) on body;
 relation balance on body { density*derivative(velocity)
   + div(density*outer_product(left = velocity, right = velocity))
   - div(2*viscosity*symmetric_part(grad(velocity)) - isotropic_lift(pressure)) = 0; }
 relation constraint on body { div(velocity) = 0; }
}";
    let mesh = SimplicialMesh::new(
        2,
        vec![vec![0., 0.], vec![1., 0.], vec![0., 1.]],
        vec![vec![0, 1, 2]],
        MeshQualityGate::new(0.01).unwrap(),
    )
    .unwrap();
    let quadrature = simplex_duffy_gauss_legendre(2, 5).unwrap();
    let geometry = mesh.geometry_map(MeshEntity::new(2, 0)).unwrap();
    for source in [
        source.to_owned(),
        source
            .replace("velocity", "renamed_flux")
            .replace("pressure", "renamed_multiplier"),
    ] {
        let (transaction, model, _) = eqiora_compiler::compile("history.eqi", &source)
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
        let form = StepForm::bind(&program, domain, 0.2, [1.0, 2.0, 4.0], [0.0; 2]).unwrap();
        let fixed = [Some([1., 2.]), None, None];
        let layout = FlowLayout::new(&mesh, &fixed, false, &form).unwrap();
        let current = [[1., 2.], [0.5, -0.25], [-0.125, 0.375]];
        let bubbles = [[0.25, -0.125]];
        let pressure = [0.5, -0.25, 0.125];
        let point = layout
            .initial_point(&fixed, &current, &bubbles, &pressure, None)
            .unwrap();
        let point = mapped_local_point(&layout.cell_map(0, true).unwrap(), &point).unwrap();
        let previous = [[-7., 9.], [-1., 0.5], [0.25, -0.5]];
        let fields = layout
            .physical_fields(&form, &previous, &[[0., 0.]], &pressure)
            .unwrap();
        let history = layout.cell_history(0, &form, &fields).unwrap();
        assert_eq!(
            history[&form.vector],
            [-14., 18., -2., 1., 0.5, -1., 0., 0.]
        );
        assert_eq!(history.len(), 1); // Pressure has no time derivative.
        assert_eq!(
            fields[&form.scalar].coefficients[&key(form.scalar, MeshEntity::new(0, 0), 0)],
            2.0
        );
        let calls = AtomicUsize::new(0);
        let load = |[x, y]: [f64; 2]| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok([0.2 + x, -0.1 + y * y])
        };
        let prepared = form.prepare_cell(&geometry, &quadrature, &load).unwrap();
        let action = prepared.linearize(&history, &point).unwrap();
        assert_eq!(
            action.residual,
            prepared.residual(&history, &point).unwrap()
        );
        let mut changed = previous;
        changed[0][0] += 1.0;
        let changed = layout
            .physical_fields(&form, &changed, &[[0., 0.]], &pressure)
            .unwrap();
        let changed = layout.cell_history(0, &form, &changed).unwrap();
        let changed = prepared.residual(&changed, &point).unwrap();
        // Physical history increment is 2. rho/dt=20, weak-row multiplier=1/4.
        // On the unit triangle: int(N0^2)=1/12, int(N0*N1)=1/24,
        // int(N0*27*N0*N1*N2)=3/40. Every other equation/component is unchanged.
        for field in form.form.fields() {
            for (local, row) in field.range.clone().enumerate() {
                let expected = if field.field == form.vector && local % 2 == 0 {
                    -10.0 * [1.0 / 12.0, 1.0 / 24.0, 1.0 / 24.0, 3.0 / 40.0][local / 2]
                } else {
                    0.0
                };
                assert!((changed[row] - action.residual[row] - expected).abs() < 1e-11);
            }
        }
        assert_eq!(calls.load(Ordering::Relaxed), quadrature.points().len());
        assert!(
            layout
                .physical_fields(&form, &previous[..2], &[[0., 0.]], &pressure)
                .is_err()
        );
        assert!(layout.cell_history(1, &form, &fields).is_err());
    }
}
