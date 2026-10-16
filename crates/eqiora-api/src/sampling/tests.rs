use super::*;

fn stream_digest(master: &[u8; 32], identity: &SamplingIdentity) -> [u8; 32] {
    identity.stream_digest(master, ParameterSampler::GENERATOR)
}

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
