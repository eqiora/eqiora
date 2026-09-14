use eqiora_core::ScalarType;
use std::num::NonZeroUsize;

use eqiora_artifact::{
    DistributedDecoderLimits, DistributedLayoutEnvelopeV1, JsonDecoderLimits,
    LinearSystemEnvelopeV1, PartitionEnvelopeV1,
};
use eqiora_core::diagnostic::codes;
use eqiora_distributed::{GlobalVectorSpace, Partition, PartitionId};
use eqiora_solver::{
    CanonicalCsrSystemView, CompleteCsrStorage, LinearOperator, LinearOperatorProperties,
};
use serde_json::{Value, json};

#[derive(Debug)]
struct Tridiagonal {
    values: Vec<f64>,
    right_hand_side: Vec<f64>,
}

impl CompleteCsrStorage for Tridiagonal {
    fn rows(&self) -> usize {
        3
    }

    fn columns(&self) -> usize {
        3
    }

    fn row_offsets(&self) -> &[usize] {
        &[0, 2, 5, 7]
    }

    fn column_indices(&self) -> &[usize] {
        &[0, 1, 0, 1, 2, 1, 2]
    }

    fn values(&self) -> &[f64] {
        &self.values
    }

    fn right_hand_side(&self) -> &[f64] {
        &self.right_hand_side
    }
}

fn system() -> LinearSystemEnvelopeV1 {
    let storage = Tridiagonal {
        values: vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        right_hand_side: vec![1.0, 0.0, 1.0],
    };
    let complete = CanonicalCsrSystemView::new(
        &storage,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    LinearSystemEnvelopeV1::from_complete(&complete).unwrap()
}

#[test]
fn linear_system_v1_rejects_symmetric_indefinite_without_retagging() {
    let storage = Tridiagonal {
        values: vec![2.0, -1.0, -1.0, -2.0, -1.0, -1.0, 2.0],
        right_hand_side: vec![1.0, 0.0, 1.0],
    };
    let complete =
        CanonicalCsrSystemView::new(&storage, LinearOperatorProperties::SymmetricIndefinite)
            .unwrap();
    let error = LinearSystemEnvelopeV1::from_complete(&complete).unwrap_err();

    assert_eq!(error.code(), codes::INVALID_ARTIFACT);
    assert_eq!(
        error.message(),
        "distributed linear-system artifact v1 cannot encode symmetric-indefinite properties"
    );
}

fn partition_artifact(owners: [usize; 3]) -> PartitionEnvelopeV1 {
    let partition = Partition::new(
        GlobalVectorSpace::new(NonZeroUsize::new(3).unwrap(), ScalarType::F64),
        NonZeroUsize::new(2).unwrap(),
        owners.into_iter().map(PartitionId::new).collect(),
    )
    .unwrap();
    PartitionEnvelopeV1::from_partition(&partition).unwrap()
}

fn artifacts() -> (
    LinearSystemEnvelopeV1,
    PartitionEnvelopeV1,
    DistributedLayoutEnvelopeV1,
) {
    let system = system();
    let partition = partition_artifact([0, 1, 0]);
    let layout = DistributedLayoutEnvelopeV1::derive(&system, &partition).unwrap();
    (system, partition, layout)
}

#[test]
fn fixed_canonical_bytes_and_digests_are_frozen() {
    let (system, partition, layout) = artifacts();
    let expected_system = r#"{"schema":"eqiora.linear-system-envelope/v1","encoding":"eqiora.canonical-json/v1","scalar":"f64","dimension":3,"row_offsets":[0,2,5,7],"column_indices":[0,1,0,1,2,1,2],"values":[2.0,-1.0,-1.0,2.0,-1.0,-1.0,2.0],"right_hand_side":[1.0,0.0,1.0],"properties":"symmetric-positive-definite"}"#.as_bytes();
    let expected_partition = r#"{"schema":"eqiora.partition-envelope/v1","encoding":"eqiora.canonical-json/v1","scalar":"f64","dimension":3,"partition_count":2,"owners":[0,1,0]}"#.as_bytes();
    assert_eq!(system.canonical_json().unwrap(), expected_system);
    assert_eq!(partition.canonical_json().unwrap(), expected_partition);
    assert_eq!(
        system.digest().unwrap().as_str(),
        "327467706aac1a30a838a7443ca9437c3fd1ab264516cca15ddfff39407928b5"
    );
    assert_eq!(
        partition.digest().unwrap().as_str(),
        "a021a8921bc3d0a02cb9ca3d0b3d7c7a1bc6b43b2295a465b0845886a6f80e9c"
    );

    let expected_layout = format!(
        r#"{{"schema":"eqiora.distributed-layout-envelope/v1","encoding":"eqiora.canonical-json/v1","linear_system_sha256":"{}","partition_sha256":"{}","local_layouts":[{{"partition":0,"owned":[0,2],"ghosts":[1]}},{{"partition":1,"owned":[1],"ghosts":[0,2]}}],"halo_exchanges":[{{"owner":0,"receiver":1,"indices":[0,2]}},{{"owner":1,"receiver":0,"indices":[1]}}]}}"#,
        system.digest().unwrap(),
        partition.digest().unwrap(),
    );
    assert_eq!(layout.canonical_json().unwrap(), expected_layout.as_bytes());
    assert_eq!(
        layout.digest().unwrap().as_str(),
        "65986604fba158a061c30a152ca4c65e2fa00fc51e34ecc1fb3e8bc89375a6ab"
    );
}

#[test]
fn all_three_round_trip_and_fresh_derivation_is_exact() {
    let (system, partition, layout) = artifacts();
    let limits = Default::default();
    let decoded_system =
        LinearSystemEnvelopeV1::from_json(&system.canonical_json().unwrap(), limits).unwrap();
    let decoded_partition =
        PartitionEnvelopeV1::from_json(&partition.canonical_json().unwrap(), limits).unwrap();
    let decoded_layout =
        DistributedLayoutEnvelopeV1::from_json(&layout.canonical_json().unwrap(), limits).unwrap();

    let complete = decoded_system.to_complete().unwrap();
    assert_eq!(complete.rows(), 3);
    assert_eq!(
        complete.properties(),
        LinearOperatorProperties::SymmetricPositiveDefinite
    );
    let reconstructed_partition = decoded_partition.to_partition().unwrap();
    assert_eq!(
        reconstructed_partition.owners(),
        partition_owners([0, 1, 0])
    );
    let distributed = decoded_layout
        .validate_against(&decoded_system, &decoded_partition)
        .unwrap();
    assert_eq!(distributed.operator().layouts()[0].owned(), &[0, 2]);
    assert_eq!(distributed.operator().layouts()[1].ghosts(), &[0, 2]);
    assert_eq!(
        decoded_layout.linear_system_digest().unwrap(),
        system.digest().unwrap()
    );
    assert_eq!(
        decoded_layout.partition_digest().unwrap(),
        partition.digest().unwrap()
    );
}

#[test]
fn unusual_finite_bits_action_and_fingerprint_survive_the_artifact_boundary() {
    let storage = Tridiagonal {
        values: vec![
            f64::from_bits(1),
            -f64::MIN_POSITIVE,
            f64::MIN_POSITIVE,
            f64::MAX / 4.0,
            -f64::from_bits(2),
            f64::from_bits(3),
            1.234_567_890_123_456_7e-200,
        ],
        right_hand_side: vec![f64::from_bits(1), -f64::MIN_POSITIVE, f64::MAX / 8.0],
    };
    let source = CanonicalCsrSystemView::new(&storage, LinearOperatorProperties::General).unwrap();
    let envelope = LinearSystemEnvelopeV1::from_complete(&source).unwrap();
    let decoded =
        LinearSystemEnvelopeV1::from_json(&envelope.canonical_json().unwrap(), Default::default())
            .unwrap()
            .to_complete()
            .unwrap();

    assert_eq!(exact_bits(source.values()), exact_bits(decoded.values()));
    assert_eq!(
        exact_bits(source.right_hand_side()),
        exact_bits(decoded.right_hand_side())
    );
    assert_eq!(
        source.agreement_fingerprint(),
        decoded.agreement_fingerprint()
    );
    let input = [1.0, 0.5, -1.0];
    let mut source_action = [0.0; 3];
    let mut decoded_action = [0.0; 3];
    source.apply(&input, &mut source_action).unwrap();
    decoded.apply(&input, &mut decoded_action).unwrap();
    assert_eq!(exact_bits(&source_action), exact_bits(&decoded_action));
}

#[test]
fn source_capture_owns_zero_normalization_and_artifact_projection_is_exact() {
    let negative_zero = Tridiagonal {
        values: vec![2.0, -0.0, -0.0, 2.0, -0.0, -0.0, 2.0],
        right_hand_side: vec![-0.0, 1.0, -0.0],
    };
    let positive_zero = Tridiagonal {
        values: vec![2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0],
        right_hand_side: vec![0.0, 1.0, 0.0],
    };
    let source =
        CanonicalCsrSystemView::new(&negative_zero, LinearOperatorProperties::General).unwrap();
    let positive =
        CanonicalCsrSystemView::new(&positive_zero, LinearOperatorProperties::General).unwrap();
    assert_eq!(exact_bits(source.values()), exact_bits(positive.values()));
    assert_eq!(
        exact_bits(source.right_hand_side()),
        exact_bits(positive.right_hand_side())
    );
    assert_eq!(
        source.agreement_fingerprint(),
        positive.agreement_fingerprint()
    );

    let envelope = LinearSystemEnvelopeV1::from_complete(&source).unwrap();
    let decoded =
        LinearSystemEnvelopeV1::from_json(&envelope.canonical_json().unwrap(), Default::default())
            .unwrap()
            .to_complete()
            .unwrap();
    assert_eq!(exact_bits(source.values()), exact_bits(decoded.values()));
    assert_eq!(
        exact_bits(source.right_hand_side()),
        exact_bits(decoded.right_hand_side())
    );
    assert_eq!(
        source.agreement_fingerprint(),
        decoded.agreement_fingerprint()
    );
    let mut source_action = [0.0; 3];
    let mut decoded_action = [0.0; 3];
    source.apply(&[1.0, 2.0, 3.0], &mut source_action).unwrap();
    decoded
        .apply(&[1.0, 2.0, 3.0], &mut decoded_action)
        .unwrap();
    assert_eq!(exact_bits(&source_action), exact_bits(&decoded_action));
}

#[test]
fn unknown_fields_negative_zero_and_invalid_csr_fail_closed() {
    let (system, partition, layout) = artifacts();
    let limits = Default::default();
    for bytes in [
        with_unknown_field(system.canonical_json().unwrap()),
        with_unknown_field(partition.canonical_json().unwrap()),
        with_unknown_field(layout.canonical_json().unwrap()),
    ] {
        assert!(decode_by_schema(&bytes, limits).is_err());
    }

    let mut negative_zero = parse(system.canonical_json().unwrap());
    negative_zero["right_hand_side"][1] = json!(-0.0);
    assert!(
        LinearSystemEnvelopeV1::from_json(&encode(&negative_zero), limits)
            .unwrap_err()
            .message()
            .contains("positive zero")
    );

    let mut invalid_csr = parse(system.canonical_json().unwrap());
    invalid_csr["column_indices"][1] = json!(0);
    assert!(
        LinearSystemEnvelopeV1::from_json(&encode(&invalid_csr), limits)
            .unwrap_err()
            .message()
            .contains("canonical CSR validation")
    );
}

#[test]
fn every_distributed_decoder_budget_is_independent() {
    let (system, partition, layout) = artifacts();
    let system_bytes = system.canonical_json().unwrap();
    let partition_bytes = partition.canonical_json().unwrap();
    let layout_bytes = layout.canonical_json().unwrap();

    assert_preflight(
        LinearSystemEnvelopeV1::from_json(
            &system_bytes,
            DistributedDecoderLimits {
                max_distributed_dimension: 2,
                ..Default::default()
            },
        ),
        "dimension",
    );
    assert_preflight(
        LinearSystemEnvelopeV1::from_json(
            &system_bytes,
            DistributedDecoderLimits {
                max_distributed_nonzeros: 6,
                ..Default::default()
            },
        ),
        "nonzero",
    );
    assert_preflight(
        PartitionEnvelopeV1::from_json(
            &partition_bytes,
            DistributedDecoderLimits {
                max_distributed_partitions: 1,
                ..Default::default()
            },
        ),
        "partition count",
    );
    assert_preflight(
        PartitionEnvelopeV1::from_json(
            &partition_bytes,
            DistributedDecoderLimits {
                max_distributed_owner_entries: 2,
                ..Default::default()
            },
        ),
        "owner-map",
    );
    assert_preflight(
        DistributedLayoutEnvelopeV1::from_json(
            &layout_bytes,
            DistributedDecoderLimits {
                max_distributed_local_indices: 5,
                ..Default::default()
            },
        ),
        "local index",
    );
    assert_preflight(
        DistributedLayoutEnvelopeV1::from_json(
            &layout_bytes,
            DistributedDecoderLimits {
                max_distributed_halo_records: 1,
                ..Default::default()
            },
        ),
        "halo record",
    );
    assert_preflight(
        DistributedLayoutEnvelopeV1::from_json(
            &layout_bytes,
            DistributedDecoderLimits {
                max_distributed_halo_indices: 2,
                ..Default::default()
            },
        ),
        "halo index",
    );
    assert_preflight(
        LinearSystemEnvelopeV1::from_json(
            &system_bytes,
            DistributedDecoderLimits {
                max_distributed_aggregate_work: 23,
                ..Default::default()
            },
        ),
        "aggregate work",
    );
    assert!(
        LinearSystemEnvelopeV1::from_json(
            &system_bytes,
            DistributedDecoderLimits {
                json: JsonDecoderLimits {
                    max_bytes: system_bytes.len() - 1,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .is_err()
    );
    assert!(
        LinearSystemEnvelopeV1::from_json(
            &system_bytes,
            DistributedDecoderLimits {
                json: JsonDecoderLimits {
                    max_nesting_depth: 1,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .is_err()
    );
}

#[test]
fn preflight_stops_at_limit_plus_one_without_deserializing_the_tail() {
    let huge_tail = format!(r#"{{"ignored":[{}]}}"#, "0,".repeat(4_096) + "0");
    let system = format!(
        r#"{{"schema":"eqiora.linear-system-envelope/v1","encoding":"eqiora.canonical-json/v1","scalar":"f64","dimension":1,"row_offsets":[0,1],"column_indices":[0,0,{huge_tail}],"values":[1.0],"right_hand_side":[1.0],"properties":"general"}}"#
    );
    assert_preflight(
        LinearSystemEnvelopeV1::from_json(
            system.as_bytes(),
            DistributedDecoderLimits {
                max_distributed_nonzeros: 2,
                ..Default::default()
            },
        ),
        "nonzero",
    );

    let partition = format!(
        r#"{{"schema":"eqiora.partition-envelope/v1","encoding":"eqiora.canonical-json/v1","scalar":"f64","dimension":1,"partition_count":1,"owners":[0,{huge_tail}]}}"#
    );
    assert_preflight(
        PartitionEnvelopeV1::from_json(
            partition.as_bytes(),
            DistributedDecoderLimits {
                max_distributed_owner_entries: 1,
                ..Default::default()
            },
        ),
        "owner-map",
    );

    let layout = format!(
        r#"{{"schema":"eqiora.distributed-layout-envelope/v1","encoding":"eqiora.canonical-json/v1","linear_system_sha256":"{}","partition_sha256":"{}","local_layouts":[{{"partition":0,"owned":[0,{huge_tail}],"ghosts":[]}}],"halo_exchanges":[]}}"#,
        "0".repeat(64),
        "1".repeat(64),
    );
    assert_preflight(
        DistributedLayoutEnvelopeV1::from_json(
            layout.as_bytes(),
            DistributedDecoderLimits {
                max_distributed_local_indices: 1,
                ..Default::default()
            },
        ),
        "local index",
    );
}

#[test]
fn allocation_free_preflight_is_field_order_independent() {
    let (system, partition, layout) = artifacts();
    let limits = Default::default();
    let arrays_first_system = br#"{"row_offsets":[0,2,5,7],"column_indices":[0,1,0,1,2,1,2],"values":[2.0,-1.0,-1.0,2.0,-1.0,-1.0,2.0],"right_hand_side":[1.0,0.0,1.0],"properties":"symmetric-positive-definite","dimension":3,"scalar":"f64","encoding":"eqiora.canonical-json/v1","schema":"eqiora.linear-system-envelope/v1"}"#;
    let arrays_first_partition = br#"{"owners":[0,1,0],"partition_count":2,"dimension":3,"scalar":"f64","encoding":"eqiora.canonical-json/v1","schema":"eqiora.partition-envelope/v1"}"#;
    let arrays_first_layout = format!(
        r#"{{"local_layouts":[{{"owned":[0,2],"ghosts":[1],"partition":0}},{{"ghosts":[0,2],"owned":[1],"partition":1}}],"halo_exchanges":[{{"indices":[0,2],"receiver":1,"owner":0}},{{"indices":[1],"owner":1,"receiver":0}}],"linear_system_sha256":"{}","partition_sha256":"{}","encoding":"eqiora.canonical-json/v1","schema":"eqiora.distributed-layout-envelope/v1"}}"#,
        system.digest().unwrap(),
        partition.digest().unwrap(),
    );

    let decoded_system = LinearSystemEnvelopeV1::from_json(arrays_first_system, limits).unwrap();
    let decoded_partition = PartitionEnvelopeV1::from_json(arrays_first_partition, limits).unwrap();
    let decoded_layout =
        DistributedLayoutEnvelopeV1::from_json(arrays_first_layout.as_bytes(), limits).unwrap();
    assert_eq!(
        decoded_system.canonical_json().unwrap(),
        system.canonical_json().unwrap()
    );
    assert_eq!(
        decoded_partition.canonical_json().unwrap(),
        partition.canonical_json().unwrap()
    );
    assert_eq!(
        decoded_layout.canonical_json().unwrap(),
        layout.canonical_json().unwrap()
    );
}

#[test]
fn arrays_before_invalid_last_fields_never_reach_dto_materialization() {
    let limits = Default::default();
    let huge_unknown_value = format!(r#"{{"nested":[{}]}}"#, "0,".repeat(4_096) + "0");
    let unknown_last = format!(
        r#"{{"row_offsets":[0,1],"column_indices":[0],"values":[1.0],"right_hand_side":[1.0],"schema":"eqiora.linear-system-envelope/v1","encoding":"eqiora.canonical-json/v1","scalar":"f64","dimension":1,"properties":"general","unknown":{huge_unknown_value}}}"#
    );
    assert_preflight_message(
        LinearSystemEnvelopeV1::from_json(unknown_last.as_bytes(), limits),
        "unknown field `unknown`",
    );

    let wrong_type_last = br#"{"owners":[0],"schema":"eqiora.partition-envelope/v1","encoding":"eqiora.canonical-json/v1","scalar":"f64","dimension":1,"partition_count":"1"}"#;
    assert_preflight_message(
        PartitionEnvelopeV1::from_json(wrong_type_last, limits),
        "invalid type: string",
    );

    let wrong_value_last = format!(
        r#"{{"local_layouts":[{{"owned":[0],"ghosts":[],"partition":0}}],"halo_exchanges":[],"schema":"eqiora.distributed-layout-envelope/v1","encoding":"eqiora.canonical-json/v1","linear_system_sha256":"{}","partition_sha256":"{}"}}"#,
        "0".repeat(64),
        "A".repeat(64),
    );
    assert_preflight_message(
        DistributedLayoutEnvelopeV1::from_json(wrong_value_last.as_bytes(), limits),
        "partition digest must be 64 lowercase hexadecimal",
    );

    let nested_wrong_type_last = format!(
        r#"{{"local_layouts":[{{"owned":[0],"ghosts":[],"partition":"0"}}],"halo_exchanges":[],"schema":"eqiora.distributed-layout-envelope/v1","encoding":"eqiora.canonical-json/v1","linear_system_sha256":"{}","partition_sha256":"{}"}}"#,
        "0".repeat(64),
        "1".repeat(64),
    );
    assert_preflight_message(
        DistributedLayoutEnvelopeV1::from_json(nested_wrong_type_last.as_bytes(), limits),
        "invalid type: string",
    );
}

#[test]
fn missing_required_scalar_and_nested_fields_fail_in_preflight() {
    let limits = Default::default();
    let missing_dimension = br#"{"row_offsets":[0,1],"column_indices":[0],"values":[1.0],"right_hand_side":[1.0],"schema":"eqiora.linear-system-envelope/v1","encoding":"eqiora.canonical-json/v1","scalar":"f64","properties":"general"}"#;
    assert_preflight_message(
        LinearSystemEnvelopeV1::from_json(missing_dimension, limits),
        "missing field `dimension`",
    );

    let missing_partition_count =
        br#"{"owners":[0],"schema":"eqiora.partition-envelope/v1","encoding":"eqiora.canonical-json/v1","scalar":"f64","dimension":1}"#;
    assert_preflight_message(
        PartitionEnvelopeV1::from_json(missing_partition_count, limits),
        "missing field `partition_count`",
    );

    let missing_partition = format!(
        r#"{{"local_layouts":[{{"owned":[0],"ghosts":[]}}],"halo_exchanges":[],"schema":"eqiora.distributed-layout-envelope/v1","encoding":"eqiora.canonical-json/v1","linear_system_sha256":"{}","partition_sha256":"{}"}}"#,
        "0".repeat(64),
        "1".repeat(64),
    );
    assert_preflight_message(
        DistributedLayoutEnvelopeV1::from_json(missing_partition.as_bytes(), limits),
        "missing field `partition`",
    );

    let missing_indices = format!(
        r#"{{"local_layouts":[{{"owned":[0],"ghosts":[],"partition":0}}],"halo_exchanges":[{{"owner":0,"receiver":1}}],"schema":"eqiora.distributed-layout-envelope/v1","encoding":"eqiora.canonical-json/v1","linear_system_sha256":"{}","partition_sha256":"{}"}}"#,
        "0".repeat(64),
        "1".repeat(64),
    );
    assert_preflight_message(
        DistributedLayoutEnvelopeV1::from_json(missing_indices.as_bytes(), limits),
        "missing field `indices`",
    );
}

#[test]
fn cross_wires_and_forged_derived_records_fail() {
    let (system, partition, layout) = artifacts();
    let other_system = {
        let storage = Tridiagonal {
            values: vec![3.0, -1.0, -1.0, 3.0, -1.0, -1.0, 3.0],
            right_hand_side: vec![1.0, 0.0, 1.0],
        };
        let complete = CanonicalCsrSystemView::new(
            &storage,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        LinearSystemEnvelopeV1::from_complete(&complete).unwrap()
    };
    let other_partition = partition_artifact([0, 0, 1]);
    assert!(layout.validate_against(&other_system, &partition).is_err());
    assert!(layout.validate_against(&system, &other_partition).is_err());

    let mut forged = parse(layout.canonical_json().unwrap());
    forged["local_layouts"][0]["ghosts"] = json!([]);
    let forged =
        DistributedLayoutEnvelopeV1::from_json(&encode(&forged), Default::default()).unwrap();
    assert!(
        forged
            .validate_against(&system, &partition)
            .unwrap_err()
            .message()
            .contains("fresh derivation")
    );
}

#[test]
fn structural_mutations_fail_at_the_narrowest_boundary() {
    let (system, partition, layout) = artifacts();
    let limits = Default::default();

    for mutation in [
        ("row_offsets", json!([])),
        ("row_offsets", json!([0, 2, 5, 8])),
        ("column_indices", json!([0, 1, 0, 1, 2, 1, 3])),
    ] {
        let mut wire = parse(system.canonical_json().unwrap());
        wire[mutation.0] = mutation.1;
        assert!(LinearSystemEnvelopeV1::from_json(&encode(&wire), limits).is_err());
    }

    for mutation in [
        ("owners", json!([])),
        ("owners", json!([0, 2, 0])),
        ("partition_count", json!(3)),
    ] {
        let mut wire = parse(partition.canonical_json().unwrap());
        wire[mutation.0] = mutation.1;
        assert!(PartitionEnvelopeV1::from_json(&encode(&wire), limits).is_err());
    }

    for (path, replacement) in [
        (("local_layouts", 0, "owned"), json!([0, 0])),
        (("local_layouts", 0, "owned"), json!([2, 0])),
        (("halo_exchanges", 0, "indices"), json!([0, 0])),
    ] {
        let mut wire = parse(layout.canonical_json().unwrap());
        wire[path.0][path.1][path.2] = replacement;
        assert!(DistributedLayoutEnvelopeV1::from_json(&encode(&wire), limits).is_err());
    }

    let mut swapped = parse(layout.canonical_json().unwrap());
    swapped["halo_exchanges"].as_array_mut().unwrap().swap(0, 1);
    assert!(DistributedLayoutEnvelopeV1::from_json(&encode(&swapped), limits).is_err());

    for (path, replacement) in [
        (("local_layouts", 0, "owned"), json!([0, 3])),
        (("halo_exchanges", 0, "indices"), json!([0, 3])),
    ] {
        let mut wire = parse(layout.canonical_json().unwrap());
        wire[path.0][path.1][path.2] = replacement;
        let decoded = DistributedLayoutEnvelopeV1::from_json(&encode(&wire), limits).unwrap();
        assert!(decoded.validate_against(&system, &partition).is_err());
    }
}

fn partition_owners<const N: usize>(owners: [usize; N]) -> Vec<PartitionId> {
    owners.into_iter().map(PartitionId::new).collect()
}

fn parse(bytes: Vec<u8>) -> Value {
    serde_json::from_slice(&bytes).unwrap()
}

fn encode(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
}

fn with_unknown_field(bytes: Vec<u8>) -> Vec<u8> {
    let mut value = parse(bytes);
    value["unknown"] = json!(true);
    encode(&value)
}

fn decode_by_schema(bytes: &[u8], limits: DistributedDecoderLimits) -> Result<(), String> {
    let value: Value = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    match value["schema"].as_str() {
        Some("eqiora.linear-system-envelope/v1") => {
            LinearSystemEnvelopeV1::from_json(bytes, limits)
                .map(|_| ())
                .map_err(|error| error.to_string())
        }
        Some("eqiora.partition-envelope/v1") => PartitionEnvelopeV1::from_json(bytes, limits)
            .map(|_| ())
            .map_err(|error| error.to_string()),
        Some("eqiora.distributed-layout-envelope/v1") => {
            DistributedLayoutEnvelopeV1::from_json(bytes, limits)
                .map(|_| ())
                .map_err(|error| error.to_string())
        }
        _ => Err("unknown schema".to_owned()),
    }
}

fn assert_preflight<T: std::fmt::Debug>(
    result: Result<T, eqiora_core::Diagnostic>,
    boundary: &str,
) {
    let message = result.unwrap_err().message().to_owned();
    assert!(
        message.contains("preflight"),
        "unexpected boundary: {message}"
    );
    assert!(
        message.contains(boundary),
        "unexpected diagnostic: {message}"
    );
}

fn assert_preflight_message<T: std::fmt::Debug>(
    result: Result<T, eqiora_core::Diagnostic>,
    expected: &str,
) {
    let message = result.unwrap_err().message().to_owned();
    assert!(
        message.contains("preflight"),
        "DTO materialization was reached: {message}"
    );
    assert!(
        message.contains(expected),
        "unexpected preflight diagnostic: {message}"
    );
}

fn exact_bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}
