use super::*;

fn identity(sample: &[u8], channel: &[u8], lineage: &[u64]) -> SamplingIdentity {
    SamplingIdentity::new(sample, channel, lineage, None).unwrap()
}

#[test]
fn independent_fixed_identity_reference() {
    // Independently derived before Rust implementation using Python hashlib:
    // lp = lambda x: struct.pack('>H', len(x)) + x
    // s = sha256(b'eqiora.parameter-uniform.sha256.v1\0' + bytes(range(32))
    //     + lp(b'coefficient') + b'\0' + lp(b'sample-0001')).digest()
    // n_i = int.from_bytes(sha256(b'eqiora.parameter-uniform.draw.v1\0'
    //     + s + struct.pack('>Q', i)).digest()[:8], 'big') >> 11
    let master = std::array::from_fn(|index| index as u8);
    let request = identity(b"sample-0001", b"coefficient", &[7, 11]);
    let stream = stream_digest(&master, &request);
    assert_eq!(
        stream,
        [
            24, 63, 58, 117, 234, 212, 154, 190, 120, 68, 70, 55, 148, 113, 21, 206, 51, 16, 62, 2,
            53, 186, 65, 230, 33, 250, 71, 116, 204, 178, 35, 85
        ]
    );
    for (index, numerator) in [
        5_806_591_123_252_352u64,
        8_182_165_338_633_557,
        8_500_042_188_450_084,
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(
            uniform(&stream, index as u64),
            numerator as f64 / 9_007_199_254_740_992.0
        );
    }
}

#[test]
fn full_ids_roles_master_and_explicit_coupling_are_not_truncated() {
    let master = [0; 32];
    let base = identity(&[1; 256], b"a", &[0]);
    let original = stream_digest(&master, &base);
    let mut sample = [1; 256];
    sample[255] = 2;
    assert_ne!(
        original,
        stream_digest(&master, &identity(&sample, b"a", &[0]))
    );
    assert_ne!(original, stream_digest(&[1; 32], &base));
    let mut changed_master = master;
    changed_master[31] = 1;
    assert_ne!(original, stream_digest(&changed_master, &base));
    assert_ne!(
        original,
        stream_digest(&master, &identity(&[1; 256], b"b", &[0]))
    );
    assert_ne!(
        stream_digest(&master, &identity(b"ab", b"c", &[])),
        stream_digest(&master, &identity(b"a", b"bc", &[]))
    );
    assert_eq!(
        original,
        stream_digest(&master, &identity(&[1; 256], b"a", &[99, 4]))
    );
    let crn = |sample: &[u8], group: &[u8]| {
        SamplingIdentity::new(sample, b"a", &[4], Some(group)).unwrap()
    };
    assert_eq!(
        stream_digest(&master, &crn(b"one", b"shared")),
        stream_digest(&master, &crn(b"two", b"shared"))
    );
    assert_ne!(
        stream_digest(&master, &crn(b"one", b"shared")),
        stream_digest(&master, &identity(b"shared", b"a", &[4]))
    );
    assert_ne!(
        stream_digest(&master, &crn(b"one", b"shared")),
        stream_digest(&master, &crn(b"one", b"other"))
    );
    assert_ne!(
        stream_digest(&master, &identity(b"one", &[1; 256], &[])),
        stream_digest(&master, &identity(b"one", &sample, &[]))
    );
    assert_ne!(
        stream_digest(&master, &crn(b"one", &[1; 256])),
        stream_digest(&master, &crn(b"one", &sample))
    );
}

#[test]
fn malformed_and_unbounded_identities_reject() {
    for (sample, channel, lineage) in [
        (&b""[..], &b"a"[..], &[][..]),
        (&b"a"[..], &b""[..], &[][..]),
        (&[0; 257][..], &b"a"[..], &[][..]),
        (&b"a"[..], &b"a"[..], &[0; 33][..]),
    ] {
        assert!(SamplingIdentity::new(sample, channel, lineage, None).is_err());
    }
    assert!(SamplingIdentity::new(b"a", b"b", &[], Some(&[])).is_err());
}

#[test]
fn binary64_conversion_respects_negative_adjacent_and_subnormal_bounds() {
    let tiny = f64::from_bits(1);
    for (lower, upper) in [
        (-4.0, -2.0),
        (1.0, 1.0f64.next_up()),
        (-1.0, (-1.0f64).next_up()),
        (0.0, tiny),
        (-tiny, 0.0),
    ] {
        assert_eq!(convert(lower, upper, 0.0), lower);
        for unit in [0.25, 0.5, 1.0 - 1.0 / 9_007_199_254_740_992.0] {
            let value = convert(lower, upper, unit);
            assert!(value >= lower && value < upper);
        }
    }
    assert_eq!(convert(-4.0, -2.0, 0.5), -3.0);
    assert_eq!(convert(1.0, 1.0f64.next_up(), 0.75), 1.0);
    assert_eq!(convert(0.0, tiny, 0.75), 0.0);
}
