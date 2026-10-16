use super::*;

const GENERATOR: &str = "eqiora.parameter-uniform.sha256.v1";

fn identity(sample: &[u8], channel: &[u8], lineage: &[u64]) -> SamplingIdentity {
    SamplingIdentity::new(sample, channel, lineage, None).unwrap()
}

fn stream_digest(master: &[u8; 32], identity: &SamplingIdentity) -> [u8; 32] {
    identity.stream_digest(master, GENERATOR)
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
fn algorithm_domains_are_separate_but_occurrences_are_not_streams() {
    let sample = identity(b"sample", b"noise", &[1]);
    assert_ne!(
        sample.stream_digest(&[0; 32], GENERATOR),
        sample.stream_digest(&[0; 32], "eqiora.wiener.sha256.v1"),
    );
    assert_eq!(
        sample.stream_digest(&[0; 32], GENERATOR),
        identity(b"sample", b"noise", &[7, 12]).stream_digest(&[0; 32], GENERATOR),
    );
}
