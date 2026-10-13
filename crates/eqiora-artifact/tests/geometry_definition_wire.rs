use std::path::Path;
use std::process::Command;

use eqiora_artifact::{GeometryDefinitionDecoderLimits, GeometryDefinitionV1, JsonDecoderLimits};
use eqiora_geometry::{
    CanonicalGeometryV1, EDGE_DIMENSION, FACE_DIMENSION, NamedEntitySet, PlanarFace, PlanarRegion,
};

const EXPECTED_JSON: &str = r#"{"schema":"eqiora.geometry-definition-envelope/v1","encoding":"eqiora.canonical-json/v1","kind":"straight-edged-planar-v1","length-unit":"metre","tolerance-m":0.0625,"vertices":[[0.0,0.0],[0.0,1.0],[0.25,0.25],[0.25,0.75],[0.75,0.25],[0.75,0.75],[1.0,0.0],[1.0,1.0]],"faces":[{"outer":[0,6,7,1],"holes":[[2,3,5,4]]}],"entity-sets":[{"name":"exterior","dimension":1,"members":[0,1,2,3]},{"name":"hole","dimension":1,"members":[4,5,6,7]},{"name":"fluid","dimension":2,"members":[0]}]}"#;
const EXPECTED_DIGEST: &str = "e6f8e17ac215ef37ca3c9de07b9979e34f13412a5de11dc9240ea1def8130030";

fn fixture_region() -> PlanarRegion {
    PlanarRegion::new(
        vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.25, 0.25],
            [0.75, 0.25],
            [0.75, 0.75],
            [0.25, 0.75],
        ],
        vec![PlanarFace::new(vec![0, 1, 2, 3], vec![vec![4, 5, 6, 7]])],
        vec![
            NamedEntitySet::new("exterior", EDGE_DIMENSION, vec![0, 1, 2, 3]),
            NamedEntitySet::new("hole", EDGE_DIMENSION, vec![4, 5, 6, 7]),
            NamedEntitySet::new("fluid", FACE_DIMENSION, vec![0]),
        ],
        0.0625,
    )
    .expect("oracle geometry")
}

fn limits() -> GeometryDefinitionDecoderLimits {
    GeometryDefinitionDecoderLimits::default()
}

fn rejected(bytes: &[u8], expected: &str) {
    let error = GeometryDefinitionV1::from_json(bytes, limits()).unwrap_err();
    assert!(
        error.message().contains(expected),
        "expected `{expected}` in `{}`",
        error.message()
    );
}

#[test]
fn independent_oracle_and_external_round_trip_are_exact() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../verify/geometry/authored-planar-geometry-artifact/expected/derive_digest.py");
    let output = Command::new("python3")
        .arg(script)
        .output()
        .expect("run independent Python oracle");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains(EXPECTED_DIGEST),
        "independent oracle did not emit the frozen digest"
    );

    let artifact = GeometryDefinitionV1::from_region(&fixture_region());
    assert_eq!(artifact.canonical_json().unwrap(), EXPECTED_JSON.as_bytes());
    assert_eq!(artifact.canonical_json().unwrap().len(), 482);
    assert_eq!(artifact.digest().unwrap().as_str(), EXPECTED_DIGEST);

    let decoded = GeometryDefinitionV1::from_json(EXPECTED_JSON.as_bytes(), limits()).unwrap();
    assert_eq!(decoded.region().unwrap(), fixture_region());
    assert_eq!(decoded.canonical_json().unwrap(), EXPECTED_JSON.as_bytes());
    assert_eq!(decoded.digest().unwrap().as_str(), EXPECTED_DIGEST);
    assert_eq!(
        decoded.canonical().digest_bytes(),
        decoded.digest().unwrap().sha256_bytes()
    );

    let filled = PlanarRegion::new(
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        vec![PlanarFace::new(vec![0, 1, 2, 3], Vec::new())],
        vec![NamedEntitySet::new(
            "exterior",
            EDGE_DIMENSION,
            vec![0, 1, 2, 3],
        )],
        0.0625,
    )
    .unwrap();
    assert_ne!(
        GeometryDefinitionV1::from_region(&filled)
            .digest()
            .unwrap()
            .as_str(),
        EXPECTED_DIGEST
    );
}

#[test]
fn straight_artifact_decoder_rejects_current_circular_wire() {
    let circular = CanonicalGeometryV1::from_circular_hole(
        [[0.0, 2.2], [0.0, 0.41]],
        [0.2, 0.2],
        0.05,
        vec![
            NamedEntitySet::new("inlet", EDGE_DIMENSION, vec![0]),
            NamedEntitySet::new("outlet", EDGE_DIMENSION, vec![1]),
            NamedEntitySet::new("walls", EDGE_DIMENSION, vec![2, 3]),
            NamedEntitySet::new("cylinder", EDGE_DIMENSION, vec![4]),
            NamedEntitySet::new("fluid", FACE_DIMENSION, vec![0]),
        ],
    )
    .unwrap();
    assert_eq!(circular.canonical_bytes().len(), 491);
    assert!(GeometryDefinitionV1::from_json(circular.canonical_bytes(), limits()).is_err());
}

#[test]
fn noncanonical_encodings_do_not_create_second_identities() {
    let with_space = EXPECTED_JSON.replacen("\",\"encoding", "\", \"encoding", 1);
    rejected(with_space.as_bytes(), "not the canonical encoding");

    let negative_zero = EXPECTED_JSON.replacen("[0.0,0.0]", "[-0.0,0.0]", 1);
    rejected(negative_zero.as_bytes(), "not the canonical encoding");
    let negative_zero_region = PlanarRegion::new(
        vec![
            [-0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.25, 0.25],
            [0.75, 0.25],
            [0.75, 0.75],
            [0.25, 0.75],
        ],
        vec![PlanarFace::new(vec![0, 1, 2, 3], vec![vec![4, 5, 6, 7]])],
        fixture_region().entity_sets().to_vec(),
        0.0625,
    )
    .unwrap();
    assert_eq!(
        negative_zero_region.vertices()[0][0].to_bits(),
        0.0_f64.to_bits()
    );
    let negative_zero_artifact = GeometryDefinitionV1::from_region(&negative_zero_region);
    assert_eq!(
        negative_zero_artifact.canonical_json().unwrap(),
        EXPECTED_JSON.as_bytes()
    );
    assert_eq!(
        negative_zero_artifact.digest().unwrap().as_str(),
        EXPECTED_DIGEST
    );

    let rotated_loop = EXPECTED_JSON.replace(
        r#""outer":[0,6,7,1],"holes":[[2,3,5,4]]"#,
        r#""outer":[6,7,1,0],"holes":[[3,5,4,2]]"#,
    );
    rejected(rotated_loop.as_bytes(), "not the canonical encoding");

    let author_order = EXPECTED_JSON
        .replace(
            r#""vertices":[[0.0,0.0],[0.0,1.0],[0.25,0.25],[0.25,0.75],[0.75,0.25],[0.75,0.75],[1.0,0.0],[1.0,1.0]]"#,
            r#""vertices":[[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0],[0.25,0.25],[0.75,0.25],[0.75,0.75],[0.25,0.75]]"#,
        )
        .replace(
            r#""outer":[0,6,7,1],"holes":[[2,3,5,4]]"#,
            r#""outer":[0,1,2,3],"holes":[[4,5,6,7]]"#,
        );
    rejected(author_order.as_bytes(), "not the canonical encoding");

    let canonical_a = GeometryDefinitionV1::from_region(&fixture_region());
    let permuted = PlanarRegion::new(
        vec![
            [0.75, 0.75],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.25, 0.25],
            [1.0, 0.0],
            [0.75, 0.25],
            [0.0, 0.0],
            [0.25, 0.75],
        ],
        vec![PlanarFace::new(vec![2, 4, 6, 1], vec![vec![3, 7, 0, 5]])],
        fixture_region().entity_sets().to_vec(),
        0.0625,
    )
    .unwrap();
    let canonical_b = GeometryDefinitionV1::from_region(&permuted);
    assert_eq!(canonical_a.digest().unwrap(), canonical_b.digest().unwrap());
}

#[test]
fn unreferenced_vertices_remain_identity_bearing_and_externally_admissible() {
    let base = PlanarRegion::new(
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        vec![PlanarFace::new(vec![0, 1, 2, 3], Vec::new())],
        vec![NamedEntitySet::new(
            "exterior",
            EDGE_DIMENSION,
            vec![0, 1, 2, 3],
        )],
        0.0625,
    )
    .unwrap();
    let with_unreferenced = PlanarRegion::new(
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]],
        vec![PlanarFace::new(vec![0, 1, 2, 3], Vec::new())],
        base.entity_sets().to_vec(),
        0.0625,
    )
    .unwrap();
    assert_eq!(with_unreferenced.vertices().len(), 5);
    assert_eq!(with_unreferenced.faces()[0].outer(), &[0, 3, 4, 1]);

    let base_artifact = GeometryDefinitionV1::from_region(&base);
    let unreferenced_artifact = GeometryDefinitionV1::from_region(&with_unreferenced);
    assert_ne!(
        unreferenced_artifact.digest().unwrap(),
        base_artifact.digest().unwrap()
    );
    let bytes = unreferenced_artifact.canonical_json().unwrap();
    let replayed = GeometryDefinitionV1::from_json(&bytes, limits()).unwrap();
    assert_eq!(replayed.region().unwrap(), with_unreferenced);
    assert_eq!(
        replayed.digest().unwrap(),
        unreferenced_artifact.digest().unwrap()
    );
}

#[test]
fn malformed_unknown_and_geometrically_invalid_wire_is_rejected() {
    let vocabulary_mutants = [
        (
            "eqiora.geometry-definition-envelope/v1",
            "eqiora.geometry-definition-envelope/v2",
            "unsupported geometry definition",
        ),
        (
            "eqiora.canonical-json/v1",
            "eqiora.canonical-json/v2",
            "unsupported geometry definition",
        ),
        (
            "straight-edged-planar-v1",
            "curved-planar-v1",
            "unknown variant",
        ),
        (
            r#""length-unit":"metre""#,
            r#""length-unit":"foot""#,
            "unknown variant",
        ),
    ];
    for (from, to, expected) in vocabulary_mutants {
        rejected(EXPECTED_JSON.replacen(from, to, 1).as_bytes(), expected);
    }
    rejected(
        EXPECTED_JSON
            .replacen(r#"{"schema":"#, r#"{"unknown":0,"schema":"#, 1)
            .as_bytes(),
        "unknown field",
    );
    rejected(
        EXPECTED_JSON
            .replacen(
                r#"{"outer":[0,6,7,1]"#,
                r#"{"unknown":0,"outer":[0,6,7,1]"#,
                1,
            )
            .as_bytes(),
        "unknown field",
    );
    rejected(
        EXPECTED_JSON
            .replacen(
                r#"{"name":"exterior""#,
                r#"{"unknown":0,"name":"exterior""#,
                1,
            )
            .as_bytes(),
        "unknown field",
    );
    rejected(
        EXPECTED_JSON
            .replace(r#""outer":[0,6,7,1]"#, r#""outer":[0,6,7,1,0]"#)
            .as_bytes(),
        "repeats a vertex",
    );
    rejected(
        EXPECTED_JSON
            .replace(r#""members":[4,5,6,7]"#, r#""members":[4,5,6,99]"#)
            .as_bytes(),
        "does not exist",
    );
    rejected(
        EXPECTED_JSON
            .replace("[0.25,0.25]", "[0.03125,0.0]")
            .as_bytes(),
        "classification tolerance",
    );
    let self_intersection = br#"{"schema":"eqiora.geometry-definition-envelope/v1","encoding":"eqiora.canonical-json/v1","kind":"straight-edged-planar-v1","length-unit":"metre","tolerance-m":0.0625,"vertices":[[0.0,0.0],[0.0,2.0],[1.0,3.0],[2.0,0.0]],"faces":[{"outer":[0,3,1,2],"holes":[]}],"entity-sets":[{"name":"edge","dimension":1,"members":[0]}]}"#;
    rejected(self_intersection, "intersects itself");
    rejected(
        EXPECTED_JSON
            .replace(r#""name":"fluid""#, r#""name":"hole""#)
            .as_bytes(),
        "names must be unique",
    );
}

#[test]
fn every_decoder_budget_is_enforced() {
    let bytes = EXPECTED_JSON.as_bytes();

    let mut candidate = limits();
    candidate.json.max_bytes = bytes.len() - 1;
    assert!(
        GeometryDefinitionV1::from_json(bytes, candidate)
            .unwrap_err()
            .message()
            .contains("byte decoder limit")
    );

    let mut candidate = limits();
    candidate.json.max_bytes = bytes.len();
    candidate.geometry.max_bytes = bytes.len() - 1;
    assert!(
        GeometryDefinitionV1::from_json(bytes, candidate)
            .unwrap_err()
            .message()
            .contains("geometry definition has")
    );

    let mut candidate = limits();
    candidate.json = JsonDecoderLimits {
        max_bytes: bytes.len(),
        max_nesting_depth: 1,
    };
    assert!(
        GeometryDefinitionV1::from_json(bytes, candidate)
            .unwrap_err()
            .message()
            .contains("nesting")
    );

    type BudgetMutant = (&'static str, fn(&mut GeometryDefinitionDecoderLimits));
    let budget_mutants: [BudgetMutant; 5] = [
        (
            "vertex",
            |candidate: &mut GeometryDefinitionDecoderLimits| {
                candidate.geometry.max_vertices = 7;
            },
        ),
        ("face", |candidate: &mut GeometryDefinitionDecoderLimits| {
            candidate.geometry.max_faces = 0;
        }),
        (
            "loop-index",
            |candidate: &mut GeometryDefinitionDecoderLimits| {
                candidate.geometry.max_loop_indices = 7;
            },
        ),
        (
            "entity-set count",
            |candidate: &mut GeometryDefinitionDecoderLimits| {
                candidate.geometry.max_entity_sets = 2;
            },
        ),
        (
            "entity-set member",
            |candidate: &mut GeometryDefinitionDecoderLimits| {
                candidate.geometry.max_entity_set_members = 8;
            },
        ),
    ];
    for (message, mutate) in budget_mutants {
        let mut candidate = limits();
        mutate(&mut candidate);
        assert!(
            GeometryDefinitionV1::from_json(bytes, candidate)
                .unwrap_err()
                .message()
                .contains(message),
            "budget mutant `{message}`"
        );
    }

    let shipped = limits();
    assert_eq!(shipped.geometry.max_loop_indices, 4_096);
    let hole_indices = 4;
    let boundary_indices = vec!["0"; shipped.geometry.max_loop_indices - hole_indices].join(",");
    let boundary_loop = EXPECTED_JSON.replace(
        r#""outer":[0,6,7,1]"#,
        &format!(r#""outer":[{boundary_indices}]"#),
    );
    let boundary_error =
        GeometryDefinitionV1::from_json(boundary_loop.as_bytes(), shipped).unwrap_err();
    assert!(
        boundary_error.message().contains("repeats a vertex"),
        "exactly 4,096 loop indices must reach topology validation: {}",
        boundary_error.message()
    );

    let excessive_indices =
        vec!["0"; shipped.geometry.max_loop_indices + 1 - hole_indices].join(",");
    let excessive_loop = EXPECTED_JSON.replace(
        r#""outer":[0,6,7,1]"#,
        &format!(r#""outer":[{excessive_indices}]"#),
    );
    let error = GeometryDefinitionV1::from_json(excessive_loop.as_bytes(), shipped).unwrap_err();
    assert!(
        error.message().contains("loop-index count"),
        "the shipped quadratic-work ceiling must bind before topology validation: {}",
        error.message()
    );
}

#[test]
fn canonical_order_covers_multiple_holes_and_member_deduplication() {
    let vertices = vec![
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 4.0],
        [0.0, 4.0],
        [0.5, 0.5],
        [1.5, 0.5],
        [1.5, 1.5],
        [0.5, 1.5],
        [2.5, 2.5],
        [3.5, 2.5],
        [3.5, 3.5],
        [2.5, 3.5],
    ];
    let sets = || {
        vec![NamedEntitySet::new(
            "selected",
            EDGE_DIMENSION,
            vec![11, 8, 11, 8],
        )]
    };
    let first = PlanarRegion::new(
        vertices.clone(),
        vec![PlanarFace::new(
            vec![0, 1, 2, 3],
            vec![vec![8, 9, 10, 11], vec![4, 5, 6, 7]],
        )],
        sets(),
        0.0625,
    )
    .unwrap();
    let permuted = PlanarRegion::new(
        vertices.clone(),
        vec![PlanarFace::new(
            vec![2, 1, 0, 3],
            vec![vec![7, 6, 5, 4], vec![11, 10, 9, 8]],
        )],
        sets(),
        0.0625,
    )
    .unwrap();

    assert_eq!(first, permuted);
    assert_eq!(
        first.faces()[0].holes(),
        &[vec![2, 3, 5, 4], vec![6, 7, 9, 8]]
    );
    assert_eq!(first.entity_set("selected").unwrap().members(), [8, 11]);

    let duplicate = PlanarRegion::new(
        vertices,
        vec![PlanarFace::new(
            vec![0, 1, 2, 3],
            vec![vec![4, 5, 6, 7], vec![7, 6, 5, 4]],
        )],
        sets(),
        0.0625,
    )
    .unwrap_err();
    assert!(duplicate.message().contains("repeats a hole loop"));
}

#[test]
fn canonical_float_rendering_is_the_repository_serializer_spelling() {
    let region = PlanarRegion::new(
        vec![[0.0, 0.0], [1.234_567_890_123e-5, 0.0], [0.0, 1.0]],
        vec![PlanarFace::new(vec![0, 1, 2], Vec::new())],
        vec![NamedEntitySet::new("edge", EDGE_DIMENSION, vec![0])],
        1.0e-9,
    )
    .unwrap();
    let json = String::from_utf8(
        GeometryDefinitionV1::from_region(&region)
            .canonical_json()
            .unwrap(),
    )
    .unwrap();
    assert!(json.contains("0.00001234567890123"));
    assert!(!json.contains("1.234567890123e-5"));
}
