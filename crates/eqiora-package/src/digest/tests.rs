use super::*;

#[test]
fn every_digest_preserves_its_bytes_spelling_serde_and_exact_error() {
    const HEX: &str = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
    let bytes: [u8; 32] = std::array::from_fn(|index| u8::try_from(index).unwrap());
    let invalid = [
        HEX.to_uppercase(),
        HEX[..63].to_owned(),
        format!("{HEX}0"),
        "g".repeat(64),
        "é".repeat(32),
        format!("0x{}", "00".repeat(31)),
    ];
    macro_rules! check {
        ($digest:ident, $error:literal) => {
            let digest = $digest::parse(HEX).unwrap();
            assert_eq!(digest.0, bytes);
            assert_eq!(digest.to_hex(), HEX);
            assert_eq!(digest.to_string(), HEX);
            assert_eq!(
                format!("{digest:?}"),
                format!("{}(\"{HEX}\")", stringify!($digest))
            );
            let json = serde_json::to_string(&digest).unwrap();
            assert_eq!(json, format!("\"{HEX}\""));
            assert_eq!(serde_json::from_str::<$digest>(&json).unwrap(), digest);
            assert!(serde_json::from_str::<$digest>("42").is_err());
            for value in &invalid {
                assert_eq!($digest::parse(value).unwrap_err().to_string(), $error);
                assert!(
                    serde_json::from_str::<$digest>(&serde_json::to_string(value).unwrap())
                        .is_err()
                );
            }
        };
    }
    check!(
        PackageSemanticDigest,
        "PackageSemanticDigest must be 64 lowercase hexadecimal characters"
    );
    check!(
        SourceBundleDigest,
        "SourceBundleDigest must be 64 lowercase hexadecimal characters"
    );
    check!(
        ResolutionDigest,
        "ResolutionDigest must be 64 lowercase hexadecimal characters"
    );
    check!(
        PackageCompilationDigest,
        "PackageCompilationDigest must be 64 lowercase hexadecimal characters"
    );
    check!(
        PackageRunBindingDigest,
        "PackageRunBindingDigest must be 64 lowercase hexadecimal characters"
    );
    check!(
        PackageExecutionBindingDigest,
        "PackageExecutionBindingDigest must be 64 lowercase hexadecimal characters"
    );
    check!(
        CanonicalModelDigest,
        "canonical model digest must be 64 lowercase hexadecimal characters"
    );
    check!(
        CanonicalRunDigest,
        "canonical run digest must be 64 lowercase hexadecimal characters"
    );
    check!(
        CanonicalRealizationDigest,
        "canonical Realization digest must be 64 lowercase hexadecimal characters"
    );
}

#[test]
fn internal_hashes_match_independently_calculated_domain_separated_sha256() {
    // Python hashlib.sha256(domain.encode() + b"\0same canonical bytes").hexdigest().
    let bytes = b"same canonical bytes";
    for (actual, expected) in [
        (
            PackageSemanticDigest::compute(bytes).to_hex(),
            "6c8e02e1e51ed69be329f9638dd079479ce2c824cbb00ff8b726360a66c4e172",
        ),
        (
            SourceBundleDigest::compute(bytes).to_hex(),
            "c0a79fe7da60d4ce8ccafe483cec2fb694339d4f12db3f086412422282848532",
        ),
        (
            ResolutionDigest::compute(bytes).to_hex(),
            "a5232b60f04c30215ae1779cd1534c317e5e4c2ebb50d287f82d5becb0d79b04",
        ),
        (
            PackageCompilationDigest::compute(bytes).to_hex(),
            "88e9143be96b62b11e5a58144945983916677456364e76edc4ec81db26c3fddb",
        ),
        (
            PackageRunBindingDigest::compute(bytes).to_hex(),
            "02076ae2eb3eb15fc9dbc86e7025f910dbf706f1e0f80a1730940765a121b390",
        ),
        (
            PackageExecutionBindingDigest::compute(bytes).to_hex(),
            "36014aca8bb5c75b38758b959b25eb5ef73178bf3d1061151722052dd8f383d5",
        ),
    ] {
        assert_eq!(actual, expected);
    }
}

#[test]
fn digest_domains_remain_nominally_distinct() {
    use std::any::TypeId;
    let types = [
        TypeId::of::<PackageSemanticDigest>(),
        TypeId::of::<SourceBundleDigest>(),
        TypeId::of::<ResolutionDigest>(),
        TypeId::of::<PackageCompilationDigest>(),
        TypeId::of::<PackageRunBindingDigest>(),
        TypeId::of::<PackageExecutionBindingDigest>(),
        TypeId::of::<CanonicalModelDigest>(),
        TypeId::of::<CanonicalRunDigest>(),
        TypeId::of::<CanonicalRealizationDigest>(),
    ];
    assert_eq!(
        types
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        9
    );
}

#[test]
fn digest_domains_are_distinct_and_lowercase() {
    let bytes = b"same canonical bytes";
    let semantic = PackageSemanticDigest::compute(bytes);
    let source = SourceBundleDigest::compute(bytes);
    let resolution = ResolutionDigest::compute(bytes);
    let compilation = PackageCompilationDigest::compute(bytes);
    let binding = PackageRunBindingDigest::compute(bytes);
    let execution_binding = PackageExecutionBindingDigest::compute(bytes);

    assert_ne!(semantic.to_hex(), source.to_hex());
    assert_ne!(source.to_hex(), resolution.to_hex());
    assert_ne!(resolution.to_hex(), compilation.to_hex());
    assert_ne!(compilation.to_hex(), binding.to_hex());
    assert_ne!(binding.to_hex(), execution_binding.to_hex());
    assert_eq!(
        PackageSemanticDigest::parse(&semantic.to_hex()),
        Ok(semantic)
    );
    assert!(PackageSemanticDigest::parse(&semantic.to_hex().to_uppercase()).is_err());
}

#[test]
fn external_digests_are_canonical_and_strongly_typed() {
    let value = "4a".repeat(32);
    let model = CanonicalModelDigest::parse(&value).expect("model digest");
    let run = CanonicalRunDigest::parse(&value).expect("run digest");
    let realization = CanonicalRealizationDigest::parse(&value).expect("Realization digest");
    assert_eq!(model.to_hex(), value);
    assert_eq!(run.to_hex(), value);
    assert_eq!(realization.to_hex(), value);
    assert!(CanonicalRunDigest::parse(&"4A".repeat(32)).is_err());
    assert!(CanonicalRealizationDigest::parse(&"4A".repeat(32)).is_err());
    assert!(CanonicalRunDigest::parse("4a").is_err());
}
