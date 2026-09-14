//! Existing authored-geometry/import/correspondence/production owner composition.
//! MSH fixtures exercise the adapter; they do not attest an external CLI invocation.

use std::collections::BTreeSet;
use std::fmt::Write;

use eqiora::artifact::{
    GeometryDefinitionV1, GeometryMeshCorrespondenceEnvelopeV1, GmshMeshPolicyV1,
    MeshProductionLineageEnvelopeV1, SimplicialMeshEnvelopeV1,
};
use eqiora::geometry::{NamedEntitySet, PlanarFace, PlanarRegion};
use eqiora::io::gmsh::{Msh41Policy, import_msh41};
use eqiora::meshing::{MeshEntity, MeshQualityGate};

fn geometry(vertices: Vec<[f64; 2]>, faces: Vec<PlanarFace>) -> GeometryDefinitionV1 {
    let region = PlanarRegion::new(vertices, faces, vec![], 1e-10).unwrap();
    // Named selections address the geometry owner's canonical enumeration.
    let sets = (0..region.faces().len())
        .map(|index| NamedEntitySet::new(format!("patch-{index}"), 2, vec![index]))
        .chain(
            (0..region.edge_count())
                .map(|index| NamedEntitySet::new(format!("side-{index}"), 1, vec![index])),
        )
        .collect();
    GeometryDefinitionV1::from_region(
        &PlanarRegion::new(
            region.vertices().to_vec(),
            region.faces().to_vec(),
            sets,
            1e-10,
        )
        .unwrap(),
    )
}

fn strips(count: usize) -> GeometryDefinitionV1 {
    let vertices = (0..=count)
        .flat_map(|x| [[x as f64, 0.0], [x as f64, 1.0]])
        .collect();
    let faces = (0..count)
        .map(|x| PlanarFace::new(vec![2 * x, 2 * x + 2, 2 * x + 3, 2 * x + 1], vec![]))
        .collect();
    geometry(vertices, faces)
}

fn msh(vertices: &[[f64; 2]], cells: &[[usize; 3]], tag: usize) -> Vec<u8> {
    let width = vertices
        .iter()
        .map(|point| point[0])
        .fold(0.0_f64, f64::max);
    let mut source = format!(
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
         $Entities\n0 0 1 0\n{tag} 0 0 0 {width} 1 0 0 0\n$EndEntities\n\
         $Nodes\n1 {n} 1 {n}\n2 {tag} 0 {n}\n",
        n = vertices.len()
    );
    for index in 1..=vertices.len() {
        writeln!(source, "{index}").unwrap();
    }
    for [x, y] in vertices {
        writeln!(source, "{x} {y} 0").unwrap();
    }
    writeln!(
        source,
        "$EndNodes\n$Elements\n1 {n} 1 {n}\n2 {tag} 2 {n}",
        n = cells.len()
    )
    .unwrap();
    for (index, [a, b, c]) in cells.iter().enumerate() {
        writeln!(source, "{} {} {} {}", index + 1, a + 1, b + 1, c + 1).unwrap();
    }
    source.push_str("$EndElements\n");
    source.into_bytes()
}

fn grid(nx: usize, ny: usize, width: f64) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
    let vertices = (0..=ny)
        .flat_map(|y| (0..=nx).map(move |x| [width * x as f64 / nx as f64, y as f64 / ny as f64]))
        .collect();
    let cells = (0..ny)
        .flat_map(|y| {
            (0..nx).flat_map(move |x| {
                let a = y * (nx + 1) + x;
                let b = a + 1;
                let c = a + nx + 2;
                let d = a + nx + 1;
                [[a, b, c], [a, c, d]]
            })
        })
        .collect();
    (vertices, cells)
}

fn import(bytes: &[u8]) -> SimplicialMeshEnvelopeV1 {
    let policy = Msh41Policy::mesh(2, MeshQualityGate::new(0.1).unwrap()).unwrap();
    let mesh = import_msh41(bytes, policy, |_, _, _| {
        panic!("labels must not own support")
    })
    .unwrap();
    SimplicialMeshEnvelopeV1::from_mesh(&mesh).unwrap()
}

fn policy() -> GmshMeshPolicyV1 {
    GmshMeshPolicyV1::explicit(1e-10, 0.1, 1024, 0.5).unwrap()
}

#[test]
fn authored_multiple_faces_bind_imported_support_and_production_without_tag_roles() {
    for count in [2, 3, 4] {
        let geometry = strips(count);
        let (vertices, cells) = grid(2 * count, 2, count as f64);
        let mesh = import(&msh(&vertices, &cells, 7));
        assert_eq!(mesh, import(&msh(&vertices, &cells, 991)));
        // The ordinary common Mesh owner rederives these same resources from provider bytes.
        let common = eqiora_numerics::AuthenticatedCommonMesh::gmsh_4152(
            geometry.canonical().clone(),
            policy(),
            msh(&vertices, &cells, 991),
        )
        .unwrap();
        let bytes = common.to_bytes().unwrap();
        assert_eq!(
            eqiora_numerics::AuthenticatedCommonMesh::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes,
        );
        let correspondence =
            GeometryMeshCorrespondenceEnvelopeV1::from_region(&geometry, &mesh).unwrap();
        let mut recovered = BTreeSet::new();
        for region in 0..count {
            let actual = correspondence
                .region_entity_set_entities(&geometry, &format!("patch-{region}"))
                .unwrap();
            // Independent Cartesian construction: four squares/eight triangles per strip.
            let expected = cells
                .iter()
                .enumerate()
                .filter(|(_, cell)| {
                    let x = cell.iter().map(|&v| vertices[v][0]).sum::<f64>() / 3.0;
                    x > region as f64 && x < (region + 1) as f64
                })
                .map(|(index, _)| MeshEntity::new(2, index))
                .collect::<Vec<_>>();
            assert_eq!(actual.len(), 8);
            assert_eq!(actual, expected);
            for entity in actual {
                assert!(recovered.insert(entity));
            }
        }
        assert_eq!(recovered.len(), cells.len());
        for region in 0..count - 1 {
            let right = format!("side-{}", 4 * region + 1);
            let left = format!("side-{}", 4 * (region + 1) + 3);
            assert!(
                geometry
                    .canonical()
                    .selections_form_opposite_parent_interface(
                        &right,
                        &format!("patch-{region}"),
                        &left,
                        &format!("patch-{}", region + 1)
                    )
            );
            let first = correspondence
                .region_entity_set_entities(&geometry, &right)
                .unwrap();
            let second = correspondence
                .region_entity_set_entities(&geometry, &left)
                .unwrap();
            assert_eq!(first, second);
            assert_eq!(first.len(), 2);
        }
        let lineage = MeshProductionLineageEnvelopeV1::from_gmsh_4152_resources(
            policy(),
            geometry.canonical(),
            &mesh,
            &correspondence,
        )
        .unwrap();
        let replay =
            MeshProductionLineageEnvelopeV1::from_json(&lineage.canonical_json().unwrap()).unwrap();
        correspondence
            .validate_against_region(&geometry, &mesh)
            .unwrap();
        replay
            .validate_against_gmsh_4152_resources(
                policy(),
                geometry.canonical(),
                &mesh,
                &correspondence,
            )
            .unwrap();
        let other_policy = GmshMeshPolicyV1::explicit(1e-10, 0.1, 1024, 0.25).unwrap();
        assert!(
            replay
                .validate_against_gmsh_4152_resources(
                    other_policy,
                    geometry.canonical(),
                    &mesh,
                    &correspondence,
                )
                .is_err()
        );
        let foreign = strips(count + 1);
        assert!(
            correspondence
                .region_entity_set_entities(&foreign, "patch-0")
                .is_err()
        );
        assert!(
            replay
                .validate_against_gmsh_4152_resources(
                    policy(),
                    foreign.canonical(),
                    &mesh,
                    &correspondence,
                )
                .is_err()
        );
    }
}

#[test]
fn authored_non_axis_aligned_interface_uses_the_same_correspondence_owner() {
    let geometry = geometry(
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        vec![
            PlanarFace::new(vec![0, 1, 2], vec![]),
            PlanarFace::new(vec![0, 2, 3], vec![]),
        ],
    );
    let (vertices, cells) = grid(2, 2, 1.0);
    let mesh = import(&msh(&vertices, &cells, 37));
    let correspondence =
        GeometryMeshCorrespondenceEnvelopeV1::from_region(&geometry, &mesh).unwrap();
    for (index, face) in geometry.region().unwrap().faces().iter().enumerate() {
        let region = geometry.region().unwrap();
        let lower = face
            .outer()
            .iter()
            .any(|&v| region.vertices()[v] == [1.0, 0.0]);
        let expected = cells
            .iter()
            .enumerate()
            .filter(|(_, cell)| {
                let [x, y] = cell.iter().fold([0.0, 0.0], |[x, y], &v| {
                    [x + vertices[v][0], y + vertices[v][1]]
                });
                (y < x) == lower
            })
            .map(|(index, _)| MeshEntity::new(2, index))
            .collect::<Vec<_>>();
        assert_eq!(
            correspondence
                .region_entity_set_entities(&geometry, &format!("patch-{index}"))
                .unwrap(),
            expected
        );
        assert_eq!(expected.len(), 4);
    }
    let lineage = MeshProductionLineageEnvelopeV1::from_gmsh_4152_resources(
        policy(),
        geometry.canonical(),
        &mesh,
        &correspondence,
    )
    .unwrap();
    let mut wire: serde_json::Value =
        serde_json::from_slice(&correspondence.canonical_json().unwrap()).unwrap();
    let frontiers = wire["frontiers"].as_array().unwrap();
    let interface = frontiers
        .iter()
        .position(|entry| {
            frontiers.iter().any(|other| {
                entry["parent_face"] != other["parent_face"]
                    && entry["facet_indices"] == other["facet_indices"]
            })
        })
        .expect("the diagonal has two distinct parent-relative frontiers");
    let orientation = &mut wire["frontiers"][interface]["parent_outward"][0];
    *orientation = match orientation.as_str().unwrap() {
        "left-of-canonical-facet" => serde_json::json!("right-of-canonical-facet"),
        "right-of-canonical-facet" => serde_json::json!("left-of-canonical-facet"),
        other => panic!("unexpected orientation {other}"),
    };
    let reversed = GeometryMeshCorrespondenceEnvelopeV1::from_json(
        &serde_json::to_vec(&wire).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert!(reversed.validate_against_region(&geometry, &mesh).is_err());
    assert!(
        lineage
            .validate_against_gmsh_4152_resources(policy(), geometry.canonical(), &mesh, &reversed,)
            .is_err()
    );
    lineage
        .validate_against_gmsh_4152_resources(
            policy(),
            geometry.canonical(),
            &mesh,
            &correspondence,
        )
        .unwrap();
}

#[test]
fn authored_production_rejects_incomplete_partition_and_changed_mesh() {
    let geometry = strips(3);
    // x=1 and x=2 cut these triangles, so no exact whole-cell support exists.
    let (vertices, cells) = grid(2, 2, 3.0);
    assert!(
        eqiora_numerics::AuthenticatedCommonMesh::gmsh_4152(
            geometry.canonical().clone(),
            policy(),
            msh(&vertices, &cells, 1),
        )
        .is_err()
    );
    let nonconforming = import(&msh(&vertices, &cells, 1));
    assert!(GeometryMeshCorrespondenceEnvelopeV1::from_region(&geometry, &nonconforming).is_err());
    let (vertices, cells) = grid(6, 2, 3.0);
    let labeled = String::from_utf8(msh(&vertices, &cells, 1))
        .unwrap()
        .replace(" 0 0\n$EndEntities", " 1 999 0\n$EndEntities");
    let import_policy = Msh41Policy::mesh(2, MeshQualityGate::new(0.1).unwrap()).unwrap();
    assert!(import_msh41(labeled.as_bytes(), import_policy, |_, _, _| {}).is_err());
    let mesh = import(&msh(&vertices, &cells, 1));
    let correspondence =
        GeometryMeshCorrespondenceEnvelopeV1::from_region(&geometry, &mesh).unwrap();
    let lineage = MeshProductionLineageEnvelopeV1::from_gmsh_4152_resources(
        policy(),
        geometry.canonical(),
        &mesh,
        &correspondence,
    )
    .unwrap();
    let (vertices, cells) = grid(6, 4, 3.0);
    let changed = import(&msh(&vertices, &cells, 1));
    assert!(
        correspondence
            .validate_against_region(&geometry, &changed)
            .is_err()
    );
    assert!(
        lineage
            .validate_against_gmsh_4152_resources(
                policy(),
                geometry.canonical(),
                &changed,
                &correspondence,
            )
            .is_err()
    );
    assert!(
        correspondence
            .region_entity_set_entities(&geometry, "absent")
            .is_err()
    );
}
