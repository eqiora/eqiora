use std::panic::catch_unwind;

use eqiora_core::diagnostic::codes;
use eqiora_meshing::{MeshQualityGate, MeshTopology};

use super::{
    Decoder, DecoderLimits, Msh41Policy, fallible_map, fallible_set, fallible_vec, import_msh41,
};

#[derive(Clone, Copy)]
enum TestEndian {
    Little,
    Big,
}

const TRIANGLES: &str = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
$Entities\n4 4 1 0\n1 0 0 0 0\n2 1 0 0 0\n3 1 1 0 0\n4 0 1 0 0\n1 0 0 0 1 0 0 0 2 1 -2\n2 1 0 0 1 1 0 0 2 2 -3\n3 0 1 0 1 1 0 0 2 3 -4\n4 0 0 0 0 1 0 0 2 4 -1\n1 0 0 0 1 1 0 0 4 1 2 3 4\n$EndEntities\n\
$Nodes\n2 5 10 50\n1 1 0 2\n10\n20\n0 0 0\n1 0 0\n2 1 0 3\n30\n40\n50\n1 1 0\n0 1 0\n0.5 0.5 0\n$EndNodes\n\
$Elements\n2 8 101 204\n1 1 1 4\n101 10 20\n102 20 30\n103 30 40\n104 40 10\n2 1 2 4\n201 10 20 50\n202 20 30 50\n203 30 40 50\n204 40 10 50\n$EndElements\n";

const TETRAHEDRON: &str = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
$Nodes\n1 4 1 4\n3 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n\
$Elements\n1 1 1 1\n3 1 4 1\n1 1 2 3 4\n$EndElements\n";

fn importer() -> Decoder {
    Decoder::new(
        2,
        MeshQualityGate::new(0.5).unwrap(),
        DecoderLimits::default(),
    )
    .unwrap()
}

fn tetrahedron_importer() -> Decoder {
    Decoder::new(
        3,
        MeshQualityGate::new(0.1).unwrap(),
        DecoderLimits::default(),
    )
    .unwrap()
}

fn binary_tetrahedron(endian: TestEndian, size_t_size: usize, element_type: i32) -> Vec<u8> {
    binary_tetrahedron_with_ignored_points(endian, size_t_size, element_type, 0)
}

fn binary_tetrahedron_with_ignored_points(
    endian: TestEndian,
    size_t_size: usize,
    element_type: i32,
    ignored_points: usize,
) -> Vec<u8> {
    binary_tetrahedron_with_ignored_points_referencing(
        endian,
        size_t_size,
        element_type,
        ignored_points,
        1,
    )
}

fn binary_tetrahedron_with_ignored_points_referencing(
    endian: TestEndian,
    size_t_size: usize,
    element_type: i32,
    ignored_points: usize,
    ignored_node_tag: u64,
) -> Vec<u8> {
    let mut bytes = format!("$MeshFormat\n4.1 1 {size_t_size}\n").into_bytes();
    write_i32(&mut bytes, 1, endian);
    bytes.extend_from_slice(b"\n$EndMeshFormat\n$Nodes\n");
    for value in [1, 4, 1, 4] {
        write_size_t(&mut bytes, value, size_t_size, endian);
    }
    for value in [3, 1, 0] {
        write_i32(&mut bytes, value, endian);
    }
    write_size_t(&mut bytes, 4, size_t_size, endian);
    for tag in 1..=4 {
        write_size_t(&mut bytes, tag, size_t_size, endian);
    }
    for coordinate in [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] {
        write_f64(&mut bytes, coordinate, endian);
    }
    bytes.extend_from_slice(b"\n$EndNodes\n$Elements\n");
    let ignored_points = u64::try_from(ignored_points).unwrap();
    let block_count = if ignored_points == 0 { 1 } else { 2 };
    let element_count = ignored_points + 1;
    for value in [block_count, element_count, 1, element_count] {
        write_size_t(&mut bytes, value, size_t_size, endian);
    }
    if ignored_points != 0 {
        for value in [0, 1, 15] {
            write_i32(&mut bytes, value, endian);
        }
        write_size_t(&mut bytes, ignored_points, size_t_size, endian);
        for tag in 1..=ignored_points {
            write_size_t(&mut bytes, tag, size_t_size, endian);
            write_size_t(&mut bytes, ignored_node_tag, size_t_size, endian);
        }
    }
    for value in [3, 1, element_type] {
        write_i32(&mut bytes, value, endian);
    }
    write_size_t(&mut bytes, 1, size_t_size, endian);
    write_size_t(&mut bytes, element_count, size_t_size, endian);
    for value in 1..=4 {
        write_size_t(&mut bytes, value, size_t_size, endian);
    }
    bytes.extend_from_slice(b"\n$EndElements\n");
    bytes
}

fn write_i32(bytes: &mut Vec<u8>, value: i32, endian: TestEndian) {
    let encoded = match endian {
        TestEndian::Little => value.to_le_bytes(),
        TestEndian::Big => value.to_be_bytes(),
    };
    bytes.extend_from_slice(&encoded);
}

fn write_size_t(bytes: &mut Vec<u8>, value: u64, width: usize, endian: TestEndian) {
    match width {
        4 => {
            let value = value as u32;
            let encoded = match endian {
                TestEndian::Little => value.to_le_bytes(),
                TestEndian::Big => value.to_be_bytes(),
            };
            bytes.extend_from_slice(&encoded);
        }
        8 => {
            let encoded = match endian {
                TestEndian::Little => value.to_le_bytes(),
                TestEndian::Big => value.to_be_bytes(),
            };
            bytes.extend_from_slice(&encoded);
        }
        _ => panic!("test writer requires a four- or eight-byte size_t"),
    }
}

fn overwrite_size_t(bytes: &mut [u8], offset: usize, value: u64, width: usize, endian: TestEndian) {
    let mut encoded = Vec::new();
    write_size_t(&mut encoded, value, width, endian);
    bytes[offset..offset + width].copy_from_slice(&encoded);
}

fn write_f64(bytes: &mut Vec<u8>, value: f64, endian: TestEndian) {
    let encoded = match endian {
        TestEndian::Little => value.to_le_bytes(),
        TestEndian::Big => value.to_be_bytes(),
    };
    bytes.extend_from_slice(&encoded);
}

fn replace_once(source: &[u8], needle: &[u8], replacement: &[u8]) -> Vec<u8> {
    let start = source
        .windows(needle.len())
        .position(|window| window == needle)
        .expect("test fixture contains replacement target");
    let mut result = Vec::with_capacity(source.len() - needle.len() + replacement.len());
    result.extend_from_slice(&source[..start]);
    result.extend_from_slice(replacement);
    result.extend_from_slice(&source[start + needle.len()..]);
    result
}

#[test]
fn imports_sparse_multiblock_triangles_and_ignores_boundary_elements() {
    let mesh = importer().import_bytes(TRIANGLES.as_bytes()).unwrap();
    assert_eq!(mesh.topological_dimension(), 2);
    assert_eq!(mesh.vertices().len(), 5);
    assert_eq!(mesh.vertices()[4], [0.5, 0.5]);
    assert_eq!(mesh.cells().len(), 4);
    assert_eq!(mesh.cells()[3], [3, 0, 4]);
    assert!(mesh.quality_report().minimum_mean_ratio() >= 0.5);
}

#[test]
fn ascii_entity_provenance_retains_block_tags_and_normalized_connectivity() {
    let policy =
        Msh41Policy::ascii_with_entity_assignments(2, MeshQualityGate::new(0.5).unwrap()).unwrap();
    let mut assignments = std::collections::BTreeMap::new();
    let mesh = import_msh41(TRIANGLES.as_bytes(), policy, |dimension, tag, indices| {
        assignments.insert((dimension, tag), indices.to_vec());
    })
    .unwrap();
    assert_eq!(mesh.cells().len(), 4);
    assert_eq!(assignments.len(), 2);
    assert_eq!(assignments[&(1, 1)].len(), 4);
    assert_eq!(assignments[&(2, 1)], [0, 1, 2, 3]);

    let invalid = TRIANGLES.replacen("104 40 10", "104 40 50", 1);
    let mut escaped_assignments = 0;
    assert!(
        import_msh41(invalid.as_bytes(), policy, |_, _, indices| {
            escaped_assignments += indices.len()
        },)
        .is_err()
    );
    assert_eq!(escaped_assignments, 0);

    let binary = binary_tetrahedron(TestEndian::Little, 8, 4);
    let policy =
        Msh41Policy::ascii_with_entity_assignments(3, MeshQualityGate::new(0.1).unwrap()).unwrap();
    assert!(import_msh41(&binary, policy, |_, _, _| {}).is_err());
}

#[test]
fn imports_one_positive_tetrahedron() {
    let mesh = tetrahedron_importer()
        .import_bytes(TETRAHEDRON.as_bytes())
        .unwrap();
    assert_eq!(mesh.topological_dimension(), 3);
    assert_eq!(mesh.cells(), &[vec![0, 1, 2, 3]]);
}

#[test]
fn binary_endianness_and_size_t_width_are_representation_only() {
    let ascii = tetrahedron_importer()
        .import_bytes(TETRAHEDRON.as_bytes())
        .unwrap();
    for endian in [TestEndian::Little, TestEndian::Big] {
        for width in [4, 8] {
            let binary = binary_tetrahedron(endian, width, 4);
            assert_eq!(tetrahedron_importer().import_bytes(&binary).unwrap(), ascii);
        }
    }
}

#[test]
fn every_truncated_binary_representation_prefix_fails_closed() {
    for endian in [TestEndian::Little, TestEndian::Big] {
        for width in [4, 8] {
            let binary = binary_tetrahedron(endian, width, 4);
            for end in 0..binary.len() {
                assert_eq!(
                    tetrahedron_importer()
                        .import_bytes(&binary[..end])
                        .unwrap_err()
                        .code(),
                    codes::INVALID_MESH_IMPORT,
                    "{width}-byte representation prefix ending at byte {end} was unexpectedly admitted",
                );
            }
        }
    }
}

#[test]
fn binary_count_budgets_are_inclusive_and_extreme_limits_do_not_panic() {
    let valid = binary_tetrahedron(TestEndian::Little, 8, 4);
    let exact_limits = DecoderLimits {
        max_bytes: valid.len(),
        max_entities: 1,
        max_entity_references: 1,
        max_node_blocks: 1,
        max_element_blocks: 1,
        max_nodes: 4,
        max_elements: 1,
        max_ignored_elements: 1,
        max_decoded_bytes: usize::MAX,
        max_decoded_work: usize::MAX,
    };
    let exact = Decoder::new(3, MeshQualityGate::new(0.1).unwrap(), exact_limits).unwrap();
    assert_eq!(exact.import_bytes(&valid).unwrap().cells().len(), 1);

    let mut forged = valid;
    let node_header = forged
        .windows(b"$Nodes\n".len())
        .position(|window| window == b"$Nodes\n")
        .unwrap()
        + b"$Nodes\n".len();
    overwrite_size_t(&mut forged, node_header, u64::MAX, 8, TestEndian::Little);
    let extreme_limits = DecoderLimits {
        max_bytes: usize::MAX,
        max_entities: usize::MAX,
        max_entity_references: usize::MAX,
        max_node_blocks: usize::MAX,
        max_element_blocks: usize::MAX,
        max_nodes: usize::MAX,
        max_elements: usize::MAX,
        max_ignored_elements: usize::MAX,
        max_decoded_bytes: usize::MAX,
        max_decoded_work: usize::MAX,
    };
    let extreme = Decoder::new(3, MeshQualityGate::new(0.1).unwrap(), extreme_limits).unwrap();
    let outcome = catch_unwind(|| extreme.import_bytes(&forged));
    assert_eq!(
        outcome
            .expect("forged declaration must not panic")
            .unwrap_err()
            .code(),
        codes::INVALID_MESH_IMPORT,
    );
}

#[test]
fn aggregate_decoded_budgets_and_ignored_elements_fail_before_materialization() {
    let valid = binary_tetrahedron(TestEndian::Little, 8, 4);
    for limits in [
        DecoderLimits {
            max_decoded_bytes: 1,
            ..DecoderLimits::default()
        },
        DecoderLimits {
            max_decoded_work: 1,
            ..DecoderLimits::default()
        },
    ] {
        let bounded = Decoder::new(3, MeshQualityGate::new(0.1).unwrap(), limits).unwrap();
        assert_eq!(
            bounded.import_bytes(&valid).unwrap_err().code(),
            codes::INVALID_MESH_IMPORT,
        );
    }

    let default_limits = DecoderLimits::default();
    let ignored = default_limits.max_ignored_elements + 1;
    let padded = binary_tetrahedron_with_ignored_points(TestEndian::Little, 8, 4, ignored);
    assert!(padded.len() < default_limits.max_bytes);
    assert_eq!(
        tetrahedron_importer()
            .import_bytes(&padded)
            .unwrap_err()
            .code(),
        codes::INVALID_MESH_IMPORT,
    );

    let admitted_ignored = 64;
    let admitted =
        binary_tetrahedron_with_ignored_points(TestEndian::Little, 8, 4, admitted_ignored);
    let explicit = Decoder::new(
        3,
        MeshQualityGate::new(0.1).unwrap(),
        DecoderLimits {
            max_ignored_elements: admitted_ignored,
            ..DecoderLimits::default()
        },
    )
    .unwrap();
    assert_eq!(explicit.import_bytes(&admitted).unwrap().cells().len(), 1);
}

#[test]
fn lower_dimensional_elements_cannot_reference_unknown_nodes() {
    let ascii = TRIANGLES.replacen("101 10 20", "101 10 99", 1);
    assert_eq!(
        importer()
            .import_bytes(ascii.as_bytes())
            .unwrap_err()
            .code(),
        codes::INVALID_MESH_IMPORT,
    );

    let binary =
        binary_tetrahedron_with_ignored_points_referencing(TestEndian::Little, 8, 4, 1, 99);
    assert_eq!(
        tetrahedron_importer()
            .import_bytes(&binary)
            .unwrap_err()
            .code(),
        codes::INVALID_MESH_IMPORT,
    );
}

#[test]
fn ascii_extreme_declarations_never_panic_under_extreme_public_limits() {
    let maximum = usize::MAX.to_string();
    let forged_boundary = TRIANGLES.replacen(
        "1 0 0 0 1 0 0 0 2 1 -2",
        &format!("1 0 0 0 1 0 0 0 {maximum}"),
        1,
    );
    let forged_entities = TRIANGLES.replacen("4 4 1 0", &format!("{maximum} 4 1 0"), 1);
    let forged_nodes = TRIANGLES.replacen("2 5 10 50", &format!("2 {maximum} 10 50"), 1);
    let forged_elements = TRIANGLES.replacen("2 8 101 204", &format!("2 {maximum} 101 204"), 1);
    let importer = Decoder::new(
        2,
        MeshQualityGate::new(0.1).unwrap(),
        DecoderLimits {
            max_bytes: usize::MAX,
            max_entities: usize::MAX,
            max_entity_references: usize::MAX,
            max_node_blocks: usize::MAX,
            max_element_blocks: usize::MAX,
            max_nodes: usize::MAX,
            max_elements: usize::MAX,
            max_ignored_elements: usize::MAX,
            max_decoded_bytes: usize::MAX,
            max_decoded_work: usize::MAX,
        },
    )
    .unwrap();
    for forged in [
        forged_boundary,
        forged_entities,
        forged_nodes,
        forged_elements,
    ] {
        let outcome = catch_unwind(|| importer.import_bytes(forged.as_bytes()));
        assert_eq!(
            outcome
                .expect("extreme ASCII declaration must not panic")
                .unwrap_err()
                .code(),
            codes::INVALID_MESH_IMPORT,
        );
    }
}

#[test]
fn token_dense_ascii_never_allocates_a_token_scratch_vector() {
    let token_count = 100_000_usize;
    let mut dense_header = String::from("$MeshFormat\n4.1 0 8");
    for _ in 0..token_count {
        dense_header.push_str(" 0");
    }
    dense_header.push_str("\n$EndMeshFormat\n");

    let mut dense_entity = String::from("1 0 0 0 1 0 0 0 ");
    dense_entity.push_str(&token_count.to_string());
    for _ in 0..token_count {
        dense_entity.push_str(" 1");
    }
    let dense_entities = TRIANGLES.replacen("1 0 0 0 1 0 0 0 2 1 -2", &dense_entity, 1);

    let limits = DecoderLimits {
        max_decoded_bytes: 64 * 1024,
        ..DecoderLimits::default()
    };
    let importer = Decoder::new(2, MeshQualityGate::new(0.1).unwrap(), limits).unwrap();
    for (dense, expected_budget_rejection) in [(dense_header, false), (dense_entities, true)] {
        assert!(dense.len() <= limits.max_bytes);
        let outcome = catch_unwind(|| importer.import_bytes(dense.as_bytes()));
        let diagnostic = outcome
            .expect("token-dense ASCII must not panic")
            .unwrap_err();
        assert_eq!(diagnostic.code(), codes::INVALID_MESH_IMPORT);
        if expected_budget_rejection {
            assert!(
                diagnostic
                    .message()
                    .contains("aggregate decoded-byte budget")
            );
        }
    }
}

#[test]
fn impossible_importer_owned_reservations_are_diagnostics() {
    assert_eq!(
        fallible_vec::<u8>(usize::MAX, "test vector")
            .unwrap_err()
            .code(),
        codes::INVALID_MESH_IMPORT,
    );
    assert_eq!(
        fallible_set::<u64>(usize::MAX, "test set")
            .unwrap_err()
            .code(),
        codes::INVALID_MESH_IMPORT,
    );
    assert_eq!(
        fallible_map::<u64, usize>(usize::MAX, "test map")
            .unwrap_err()
            .code(),
        codes::INVALID_MESH_IMPORT,
    );
}

#[test]
fn binary_header_counts_sections_and_element_families_fail_closed() {
    let valid = binary_tetrahedron(TestEndian::Little, 8, 4);

    let mut invalid_marker = valid.clone();
    let marker = b"$MeshFormat\n4.1 1 8\n".len();
    invalid_marker[marker..marker + 4].copy_from_slice(&2_i32.to_le_bytes());

    let mut excessive_nodes = valid.clone();
    let node_header = excessive_nodes
        .windows(b"$Nodes\n".len())
        .position(|window| window == b"$Nodes\n")
        .unwrap()
        + b"$Nodes\n".len();
    excessive_nodes[node_header + 8..node_header + 16].copy_from_slice(&u64::MAX.to_le_bytes());

    let mut result_section = valid.clone();
    result_section.extend_from_slice(b"$NodeData\n$EndNodeData\n");

    for rejected in [
        invalid_marker,
        excessive_nodes,
        result_section,
        binary_tetrahedron(TestEndian::Little, 8, 11),
    ] {
        assert_eq!(
            tetrahedron_importer()
                .import_bytes(&rejected)
                .unwrap_err()
                .code(),
            codes::INVALID_MESH_IMPORT,
        );
    }

    for header in ["4.1 1 1", "4.1 1 2", "4.1 1 16"] {
        let rejected = replace_once(&valid, b"4.1 1 8", header.as_bytes());
        assert_eq!(
            tetrahedron_importer()
                .import_bytes(&rejected)
                .unwrap_err()
                .code(),
            codes::INVALID_MESH_IMPORT,
        );
    }
}

#[test]
fn resource_version_and_semantic_boundaries_fail_closed() {
    let mut zero_policy = Msh41Policy::mesh(2, MeshQualityGate::new(0.1).unwrap()).unwrap();
    zero_policy.max_bytes = 0;
    assert!(import_msh41(TRIANGLES.as_bytes(), zero_policy, |_, _, _| {}).is_err());

    let too_small = Decoder::new(
        2,
        MeshQualityGate::new(0.1).unwrap(),
        DecoderLimits {
            max_bytes: 8,
            ..DecoderLimits::default()
        },
    )
    .unwrap();
    assert_eq!(
        too_small
            .import_bytes(TRIANGLES.as_bytes())
            .unwrap_err()
            .code(),
        codes::INVALID_MESH_IMPORT
    );

    for rejected in [
        TRIANGLES.replacen("4.1 0 8", "4.1 1 8", 1),
        TRIANGLES.replacen("4.1 0 8", "2.2 0 8", 1),
        TRIANGLES.replacen("1 1 0 2", "1 1 1 2", 1),
        TRIANGLES.replacen("0.5 0.5 0", "0.5 0.5 0.25", 1),
        TRIANGLES.replacen("2 1 2 4", "2 1 3 4", 1),
    ] {
        assert_eq!(
            importer()
                .import_bytes(rejected.as_bytes())
                .unwrap_err()
                .code(),
            codes::INVALID_MESH_IMPORT
        );
    }
}

#[test]
fn inconsistent_tags_references_orientation_and_quality_are_rejected() {
    let duplicate = TRIANGLES.replacen("20\n0 0 0", "10\n0 0 0", 1);
    assert_eq!(
        importer()
            .import_bytes(duplicate.as_bytes())
            .unwrap_err()
            .code(),
        codes::INVALID_MESH_IMPORT
    );

    let missing = TRIANGLES.replacen("201 10 20 50", "201 10 20 99", 1);
    assert_eq!(
        importer()
            .import_bytes(missing.as_bytes())
            .unwrap_err()
            .code(),
        codes::INVALID_MESH_IMPORT
    );

    let inverted = TRIANGLES.replacen("201 10 20 50", "201 20 10 50", 1);
    assert_eq!(
        importer()
            .import_bytes(inverted.as_bytes())
            .unwrap_err()
            .code(),
        codes::INVALID_MESH
    );

    let strict = Decoder::new(
        2,
        MeshQualityGate::new(0.99).unwrap(),
        DecoderLimits::default(),
    )
    .unwrap();
    assert_eq!(
        strict
            .import_bytes(TRIANGLES.as_bytes())
            .unwrap_err()
            .code(),
        codes::INVALID_MESH
    );
}

#[test]
fn count_limits_and_malformed_sections_are_rejected_before_parsing() {
    let node_limited = Decoder::new(
        2,
        MeshQualityGate::new(0.1).unwrap(),
        DecoderLimits {
            max_nodes: 4,
            ..DecoderLimits::default()
        },
    )
    .unwrap();
    assert_eq!(
        node_limited
            .import_bytes(TRIANGLES.as_bytes())
            .unwrap_err()
            .code(),
        codes::INVALID_MESH_IMPORT
    );
    let malformed = TRIANGLES.replace("$EndNodes", "$EndWrong");
    assert_eq!(
        importer()
            .import_bytes(malformed.as_bytes())
            .unwrap_err()
            .code(),
        codes::INVALID_MESH_IMPORT
    );

    let forged_block_count = TRIANGLES.replacen("1 1 0 2", "1 1 0 999999999", 1);
    assert_eq!(
        importer()
            .import_bytes(forged_block_count.as_bytes())
            .unwrap_err()
            .code(),
        codes::INVALID_MESH_IMPORT
    );
}

#[test]
fn ascii_and_binary_share_node_and_element_admission() {
    // Independently written ASCII specimen; the binary writer emits the same
    // tetrahedron and one ignored point, with two distinct element tags.
    let ascii = TETRAHEDRON.replace(
        "$Elements\n1 1 1 1\n3 1 4 1\n1 1 2 3 4",
        "$Elements\n2 2 1 2\n0 1 15 1\n1 1\n3 1 4 1\n2 1 2 3 4",
    );
    let expected = tetrahedron_importer()
        .import_bytes(ascii.as_bytes())
        .unwrap();
    for endian in [TestEndian::Little, TestEndian::Big] {
        for width in [4, 8] {
            let binary = binary_tetrahedron_with_ignored_points(endian, width, 4, 1);
            assert_eq!(
                tetrahedron_importer().import_bytes(&binary).unwrap(),
                expected
            );
            let section = |name: &[u8]| {
                binary
                    .windows(name.len())
                    .position(|part| part == name)
                    .unwrap()
                    + name.len()
            };
            let nodes = section(b"$Nodes\n");
            let node_block = nodes + 4 * width;
            let node_tags = node_block + 12 + width;
            let elements = section(b"$Elements\n");
            let point_block = elements + 4 * width;
            let point_record = point_block + 12 + width;
            let cell_block = point_record + 2 * width;
            let cell_record = cell_block + 12 + width;
            let reject = |needle: &str, replacement: &str, offset, encoded_width, value| {
                assert!(
                    ascii.contains(needle),
                    "fixture mutation must reach its target"
                );
                let rejected_ascii = ascii.replacen(needle, replacement, 1);
                let mut rejected_binary = binary.clone();
                overwrite_size_t(&mut rejected_binary, offset, value, encoded_width, endian);
                for bytes in [rejected_ascii.as_bytes(), rejected_binary.as_slice()] {
                    assert_eq!(
                        tetrahedron_importer()
                            .import_bytes(bytes)
                            .unwrap_err()
                            .code(),
                        codes::INVALID_MESH_IMPORT,
                        "mutation: {needle} -> {replacement}"
                    );
                }
            };
            reject("1\n2\n3\n4\n", "0\n2\n3\n4\n", node_tags, width, 0);
            reject("1\n2\n3\n4\n", "1\n1\n3\n4\n", node_tags + width, width, 1);
            reject("1 4 1 4\n", "1 3 1 4\n", nodes + width, width, 3);
            reject("1 4 1 4\n", "1 4 0 4\n", nodes + 2 * width, width, 0);
            reject("1 4 1 4\n", "1 4 1 5\n", nodes + 3 * width, width, 5);
            reject("3 1 0 4\n", "4 1 0 4\n", node_block, 4, 4);
            reject("3 1 0 4\n", "3 0 0 4\n", node_block + 4, 4, 0);
            reject("3 1 0 4\n", "3 1 1 4\n", node_block + 8, 4, 1);
            reject("2 2 1 2\n", "2 1 1 2\n", elements + width, width, 1);
            reject("2 2 1 2\n", "2 2 0 2\n", elements + 2 * width, width, 0);
            reject("2 2 1 2\n", "2 2 1 3\n", elements + 3 * width, width, 3);
            reject("2 1 2 3 4\n", "0 1 2 3 4\n", cell_record, width, 0);
            reject("2 1 2 3 4\n", "1 1 2 3 4\n", cell_record, width, 1);
            reject("2 1 2 3 4\n", "2 0 2 3 4\n", cell_record + width, width, 0);
            reject(
                "2 1 2 3 4\n",
                "2 99 2 3 4\n",
                cell_record + width,
                width,
                99,
            );
            reject(
                "1 1\n3 1 4 1",
                "1 99\n3 1 4 1",
                point_record + width,
                width,
                99,
            );
            reject("3 1 4 1\n", "4 1 4 1\n", cell_block, 4, 4);
            reject("3 1 4 1\n", "3 0 4 1\n", cell_block + 4, 4, 0);
            reject("3 1 4 1\n", "3 1 5 1\n", cell_block + 8, 4, 5);
            for limits in [
                DecoderLimits {
                    max_nodes: 3,
                    ..DecoderLimits::default()
                },
                DecoderLimits {
                    max_elements: 1,
                    ..DecoderLimits::default()
                },
                DecoderLimits {
                    max_element_blocks: 1,
                    ..DecoderLimits::default()
                },
                DecoderLimits {
                    max_decoded_bytes: 1,
                    ..DecoderLimits::default()
                },
                DecoderLimits {
                    max_decoded_work: 1,
                    ..DecoderLimits::default()
                },
            ] {
                let decoder = Decoder::new(3, MeshQualityGate::new(0.1).unwrap(), limits).unwrap();
                for bytes in [ascii.as_bytes(), binary.as_slice()] {
                    assert_eq!(
                        decoder.import_bytes(bytes).unwrap_err().code(),
                        codes::INVALID_MESH_IMPORT
                    );
                }
            }
        }
    }
}
