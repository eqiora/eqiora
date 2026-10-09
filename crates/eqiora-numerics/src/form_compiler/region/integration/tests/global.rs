//! Mesh-to-algebra checks, not native Geometry admission or a PDE solve.
use super::*;
use crate::form_compiler::region::RegionFieldLayout;
use crate::region_assembly::mapping::RegionDofMap;
use crate::spatial_expression::Coefficient;

mod traces;
use eqiora_assembly::{CooAssembler, LinearSystem};
use eqiora_core::{RawId, ScalarDomain};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::{MeshEntity, MeshGeometry, MeshQualityGate, MeshTopology, SimplicialMesh};
use eqiora_sem::KernelProgram;
use std::collections::BTreeMap;

fn mesh(permuted: bool) -> SimplicialMesh {
    let cells = if permuted {
        vec![vec![1, 0, 3, 2], vec![2, 1, 4, 3]]
    } else {
        vec![vec![0, 1, 2, 3], vec![1, 2, 3, 4]]
    };
    SimplicialMesh::new(
        3,
        vec![
            vec![0.0, 0.0, 0.0],
            vec![2.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0],
            vec![0.0, 0.0, 4.0],
            vec![2.0, 3.0, 4.0],
        ],
        cells,
        MeshQualityGate::new(0.01).unwrap(),
    )
    .unwrap()
}

fn layout<S: Coefficient>(space: Space) -> (RawId, BTreeMap<RawId, Vec<RegionFieldLayout>>) {
    let scalar = if S::DOMAIN == ScalarDomain::Complex {
        "complex<1>"
    } else {
        "1"
    };
    let source = format!(
        "model M(){{domain body=box(0,2,0,3,0,4); variable u:vector<{scalar},3> on body; relation law on body{{u=u*0;}}}}"
    );
    let (transaction, model, symbols) = eqiora_compiler::compile("oriented-map.eqi", &source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let field = symbols.get("u").unwrap();
    let domain = symbols.get("body").unwrap();
    let expected_power = if space == Space::tetrahedral_edge() {
        1
    } else {
        2
    };
    let scale = eqiora_core::DynQuantity::new(
        1.0,
        eqiora_core::DimExponents::from_integers([0, expected_power, 0, 0, 0, 0, 0]).unwrap(),
    );
    let domains = [eqiora_realization::DomainFieldDiscretization::new(
        domain.downcast().unwrap(),
        [eqiora_realization::FieldSpaceBinding::new(
            field.downcast().unwrap(),
            space,
        )],
        [],
    )
    .unwrap()];
    let layouts = crate::region_assembly::mapping::field_layouts(
        &program,
        &domains,
        ReferenceCell::simplex(3).unwrap(),
        &BTreeMap::from([(field, scale)]),
    )
    .unwrap();
    let wrong_units = BTreeMap::from([(
        field,
        eqiora_core::DynQuantity::new(1.0, eqiora_core::DimExponents::DIMENSIONLESS),
    )]);
    assert!(
        crate::region_assembly::mapping::field_layouts(
            &program,
            &domains,
            ReferenceCell::simplex(3).unwrap(),
            &wrong_units
        )
        .is_err()
    );
    (domain, layouts)
}

fn assemble<S: Coefficient + Send + Sync>(
    mesh: &SimplicialMesh,
    space: Space,
) -> (RegionDofMap<S>, LinearSystem<S>) {
    let (domain, layouts) = layout::<S>(space);
    let reference = ReferenceCell::simplex(3).unwrap();
    let mapping = RegionDofMap::<S>::new(
        mesh,
        &layouts,
        reference,
        &[domain; 2],
        &[],
        &BTreeMap::new(),
    )
    .unwrap();
    let mut assembler = CooAssembler::new(mapping.full_count()).unwrap();
    let rule = eqiora_meshing::simplex_duffy_gauss_legendre(3, 3).unwrap();
    for cell in 0..2 {
        let geometry = mesh.geometry_map(MeshEntity::new(3, cell)).unwrap();
        let terms = if space == Space::tetrahedral_edge() {
            vec![
                IntegralTerm {
                    row: 0,
                    column: 0,
                    pairing: Pairing::Gradient,
                    trial_scale: 2.0,
                },
                IntegralTerm {
                    row: 0,
                    column: 0,
                    pairing: Pairing::SymmetricGradient,
                    trial_scale: -2.0,
                },
            ]
        } else {
            vec![IntegralTerm {
                row: 0,
                column: 0,
                pairing: Pairing::Value,
                trial_scale: 1.0,
            }]
        };
        let local = integrate(
            reference,
            &[(space, 3)],
            &terms,
            &geometry,
            &rule,
            |_, coefficients: &mut [S], _, _| {
                coefficients.fill(<S as From<f64>>::from(1.0));
                Ok(())
            },
        )
        .unwrap();
        let signs = mapping.cell_signs(cell).unwrap();
        assembler
            .scatter(
                &mapping.cell_map(cell, false).unwrap(),
                &local.reoriented(signs, signs).unwrap(),
            )
            .unwrap();
    }
    (mapping, assembler.finish().unwrap())
}

fn edge_moments(mesh: &SimplicialMesh, mapping: &RegionDofMap<f64>) -> (Vec<f64>, Vec<f64>) {
    let mut rotation = vec![0.0; mapping.full_count()];
    let mut gradient = rotation.clone();
    for key in mapping.keys() {
        assert_eq!(key.component, 0);
        assert_eq!(key.entity.dimension(), 1);
        let vertices = mesh.entity_vertices(key.entity).unwrap();
        let (a, b) = (
            &mesh.vertices()[vertices[0].index()],
            &mesh.vertices()[vertices[1].index()],
        );
        let index = mapping.global_dof(key).unwrap();
        rotation[index] =
            -(a[1] + b[1]) * 0.5 * (b[0] - a[0]) + (a[0] + b[0]) * 0.5 * (b[1] - a[1]);
        gradient[index] = 2.0 * (b[0] - a[0]) + 3.0 * (b[1] - a[1]) + 4.0 * (b[2] - a[2]);
    }
    (rotation, gradient)
}

#[test]
fn global_real_and_complex_curl_actions_preserve_orientation_and_gradient_kernel() {
    let mut reference_matrix = None;
    for permuted in [false, true] {
        let mesh = mesh(permuted);
        let (mapping, system) = assemble::<f64>(&mesh, Space::tetrahedral_edge());
        let (_, complex) = assemble::<C>(&mesh, Space::tetrahedral_edge());
        assert_eq!(mapping.full_count(), 9);
        let (rotation, gradient) = edge_moments(&mesh, &mapping);
        let action = system.matrix().multiply(&rotation).unwrap();
        close(
            C::new(rotation.iter().zip(&action).map(|(a, b)| a * b).sum(), 0.0),
            C::new(48.0, 0.0),
        ); // 4 (V1+V2), V1=4, V2=8
        for value in system.matrix().multiply(&gradient).unwrap() {
            close(C::new(value, 0.0), C::new(0.0, 0.0));
        }
        let z = rotation
            .iter()
            .zip(&gradient)
            .map(|(r, g)| C::new(1.0, 1.0) * r + C::new(2.0, -1.0) * g)
            .collect::<Vec<_>>();
        let complex_action = complex.matrix().multiply(&z).unwrap();
        for (actual, real) in complex_action.iter().zip(&action) {
            close(*actual, C::new(1.0, 1.0) * real);
        }
        close(
            z.iter()
                .zip(&complex_action)
                .map(|(z, a)| z.conj() * a)
                .sum(),
            C::new(96.0, 0.0),
        );
        let dense = (0..9)
            .flat_map(|row| (0..9).map(move |column| (row, column)))
            .map(|(row, column)| system.matrix().entry(row, column).unwrap())
            .collect::<Vec<_>>();
        if let Some(previous) = &reference_matrix {
            for (&actual, &expected) in dense.iter().zip(previous) {
                close(C::new(actual, 0.0), C::new(expected, 0.0));
            }
        } else {
            reference_matrix = Some(dense);
        }
    }
}

#[test]
fn global_flux_coefficients_have_exact_interface_cancellation_and_divergence_balance() {
    for permuted in [false, true] {
        let mesh = mesh(permuted);
        let (mapping, _) = assemble::<f64>(&mesh, Space::tetrahedral_face());
        assert_eq!(mapping.full_count(), 7);
        let mut fluxes = BTreeMap::new();
        for key in mapping.keys() {
            let v = mesh.entity_vertices(key.entity).unwrap();
            let [a, b, c] = [0, 1, 2].map(|i| &mesh.vertices()[v[i].index()]);
            let u: [f64; 3] = std::array::from_fn(|i| b[i] - a[i]);
            let w: [f64; 3] = std::array::from_fn(|i| c[i] - a[i]);
            let area = [
                u[1] * w[2] - u[2] * w[1],
                u[2] * w[0] - u[0] * w[2],
                u[0] * w[1] - u[1] * w[0],
            ]
            .map(|x| x * 0.5);
            let flux = (0..3)
                .map(|i| (a[i] + b[i] + c[i]) / 3.0 * area[i])
                .sum::<f64>();
            fluxes.insert(key.entity, flux);
        }
        for (cell, expected) in [(0, 12.0), (1, 24.0)] {
            let integral = mesh
                .signed_boundary(MeshEntity::new(3, cell))
                .unwrap()
                .iter()
                .map(|(face, sign)| f64::from(*sign) * fluxes[face])
                .sum::<f64>();
            close(C::new(integral, 0.0), C::new(expected, 0.0));
        }
        let shared = fluxes
            .keys()
            .copied()
            .find(|face| mesh.incidence(*face, 3).unwrap().len() == 2)
            .unwrap();
        let signs = [0, 1].map(|cell| {
            mesh.signed_boundary(MeshEntity::new(3, cell))
                .unwrap()
                .into_iter()
                .find(|(face, _)| *face == shared)
                .unwrap()
                .1
        });
        assert_eq!(signs, [1, -1]);
        assert_eq!(i32::from(signs[0]) + i32::from(signs[1]), 0);
        assert_eq!(
            f64::from(signs[0]) * fluxes[&shared] + f64::from(signs[1]) * fluxes[&shared],
            0.0
        );
    }
}

#[test]
fn stale_entity_or_orientation_is_rejected_before_global_mapping() {
    struct Stale<'a>(&'a SimplicialMesh, bool);
    impl MeshTopology for Stale<'_> {
        fn topological_dimension(&self) -> usize {
            3
        }
        fn entity_count(&self, d: usize) -> Option<usize> {
            self.0.entity_count(d)
        }
        fn orientation_permutation(
            &self,
            code: eqiora_meshing::OrientationCode,
            arity: usize,
        ) -> Option<eqiora_meshing::VertexPermutation> {
            self.0.orientation_permutation(code, arity)
        }
        fn incidence(
            &self,
            entity: MeshEntity,
            target: usize,
        ) -> Option<Vec<eqiora_meshing::EntityIncidence>> {
            let mut entries = self.0.incidence(entity, target)?;
            if entity == MeshEntity::new(3, 0) && target == 1 {
                if self.1 {
                    entries[0].entity = MeshEntity::new(1, 8);
                } else {
                    entries[0].orientation = eqiora_meshing::OrientationCode::identity();
                }
            }
            Some(entries)
        }
    }
    let mesh = mesh(true);
    let (domain, layouts) = layout::<f64>(Space::tetrahedral_edge());
    for bad_entity in [false, true] {
        assert!(
            RegionDofMap::<f64>::new(
                &Stale(&mesh, bad_entity),
                &layouts,
                ReferenceCell::simplex(3).unwrap(),
                &[domain; 2],
                &[],
                &BTreeMap::new()
            )
            .is_err()
        );
    }
}
