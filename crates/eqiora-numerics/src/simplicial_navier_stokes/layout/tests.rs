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
        let expected = vertices
            .iter()
            .flatten()
            .chain(bubbles.iter().flatten())
            .chain(&pressure)
            .copied()
            .collect::<Vec<_>>();
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
