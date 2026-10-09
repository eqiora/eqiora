use super::*;

#[test]
fn recovered_moments_keep_space_units_orientation_and_physical_reconstruction() {
    let value = [2.0, 3.0, 4.0];
    let phase = C::new(1.0, 2.0);
    for permuted in [false, true] {
        let mesh = mesh(permuted);
        for (space, power) in [
            (Space::tetrahedral_edge(), 1),
            (Space::tetrahedral_face(), 2),
        ] {
            let (domain, mut layouts) = layout::<C>(space);
            layouts.get_mut(&domain).unwrap()[0].scale = 2.5;
            let mapping = RegionDofMap::<C>::new(
                &mesh,
                &layouts,
                ReferenceCell::simplex(3).unwrap(),
                &[domain; 2],
                &[],
                &BTreeMap::new(),
            )
            .unwrap();
            let field = mapping.keys().next().unwrap().field;
            let mut normalized = vec![C::new(0.0, 0.0); mapping.free_count()];
            let mut moments = BTreeMap::new();
            for key in mapping.keys() {
                let vertices = mesh.entity_vertices(key.entity).unwrap();
                let a = &mesh.vertices()[vertices[0].index()];
                let b = &mesh.vertices()[vertices[1].index()];
                let direction: [f64; 3] = std::array::from_fn(|i| b[i] - a[i]);
                // Independent integrals of a constant vector: displacement along
                // the canonical edge, or half the oriented triangle cross product.
                let measure = if power == 1 {
                    direction
                } else {
                    let c = &mesh.vertices()[vertices[2].index()];
                    let second: [f64; 3] = std::array::from_fn(|i| c[i] - a[i]);
                    [
                        direction[1] * second[2] - direction[2] * second[1],
                        direction[2] * second[0] - direction[0] * second[2],
                        direction[0] * second[1] - direction[1] * second[0],
                    ]
                    .map(|x| x * 0.5)
                };
                let moment = phase * (0..3).map(|i| value[i] * measure[i]).sum::<f64>();
                moments.insert(key, moment);
                normalized[mapping.global_dof(key).unwrap()] = moment / 2.5;
            }
            let recovered = mapping.recover(&normalized, &[field]).unwrap();
            mapping.validate_physical(&recovered).unwrap();
            let physical = &recovered[&field];
            assert_eq!(physical.space, space);
            assert_eq!(
                physical
                    .space
                    .coefficient_dimension(physical.value_type.dimension()),
                eqiora_core::DimExponents::from_integers([0, power, 0, 0, 0, 0, 0])
            );
            for (key, expected) in moments {
                close(physical.coefficients[&key], expected);
            }
            for cell in 0..2 {
                let geometry = mesh.geometry_map(MeshEntity::new(3, cell)).unwrap();
                let basis = DiscreteSpace::new(physical.space, ReferenceCell::simplex(3).unwrap())
                    .unwrap()
                    .tabulate_on(&geometry, &[0.25; 3])
                    .unwrap();
                let keys = mapping.cell_field_keys(cell, field).unwrap();
                let signs = mapping.cell_signs(cell).unwrap();
                for (axis, expected) in value.iter().enumerate() {
                    let actual = keys
                        .iter()
                        .enumerate()
                        .map(|(local, key)| {
                            physical.coefficients[key]
                                * f64::from(signs[local])
                                * basis.value(local).unwrap()[axis]
                        })
                        .sum();
                    close(actual, phase * *expected);
                }
            }
            // Exact same Field, values, keys and shape cannot substitute another
            // coefficient interpretation, even before an attempted reconstruction.
            let mut substituted = recovered;
            substituted.get_mut(&field).unwrap().space = Space::cell_constant();
            assert!(
                mapping
                    .validate_physical(&substituted)
                    .unwrap_err()
                    .message()
                    .contains("Space")
            );
        }
    }
}
