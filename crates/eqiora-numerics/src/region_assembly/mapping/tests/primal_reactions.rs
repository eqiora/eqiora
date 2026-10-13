//! Independent primal fluxes on a chain with arbitrary Domain count.
use super::*;
use crate::region_assembly::{InterfaceReactionProjection, RecoveredDomainReactions};

pub(super) fn check(mapping: &RegionDofMap<f64>, regions: &[(RawId, RawId)]) {
    let projection = InterfaceReactionProjection::new(mapping).unwrap();
    let key = |field, vertex| FieldDof {
        field,
        entity: MeshEntity::new(0, vertex),
        slot: 0,
        component: 0,
    };
    // On interval i, u' = i+1 and q(u') = (u')². The outward
    // primal weak flux is [-q, +q]; a tangent-only J(u')u' is 2q.
    let mut values = BTreeMap::new();
    for (i, &(domain, field)) in regions.iter().enumerate() {
        let q = ((i + 1) as f64).powi(2);
        let mut residual = vec![0.; mapping.full_count()];
        residual[mapping.global_dof(key(field, 2 * i)).unwrap()] = -q;
        residual[mapping.global_dof(key(field, 2 * i + 2)).unwrap()] = q;
        values.insert(domain, residual);
    }
    let recovered = projection
        .recover(&RecoveredDomainReactions {
            values: values.clone(),
        })
        .unwrap();
    for (quotient, _) in mapping.traces() {
        for endpoint in quotient.endpoints() {
            let i = regions
                .iter()
                .position(|&(domain, _)| domain == endpoint.domain().erase())
                .unwrap();
            let other = quotient
                .endpoints()
                .into_iter()
                .find(|other| other.domain() != endpoint.domain())
                .unwrap();
            let j = regions
                .iter()
                .position(|&(domain, _)| domain == other.domain().erase())
                .unwrap();
            let vertex = 2 * i.max(j);
            let expected = if i < j { 1. } else { -1. } * ((i + 1) as f64).powi(2);
            assert_eq!(
                recovered
                    .action(
                        quotient.source().owner(),
                        key(endpoint.field().erase(), vertex)
                    )
                    .unwrap(),
                expected
            );
        }
    }
    let expected_imbalance = (0..regions.len() - 1)
        .map(|i| ((2 * i + 3) as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((recovered.imbalance_norm - expected_imbalance).abs() < 1e-12);
    assert_eq!(recovered.constrained_actions[&key(regions[0].1, 0)], -1.);
    assert_eq!(
        recovered.constrained_actions[&key(regions.last().unwrap().1, 2 * regions.len())],
        (regions.len() as f64).powi(2)
    );

    let mut missing = values.clone();
    missing.remove(&regions[0].0);
    assert!(
        projection
            .recover(&RecoveredDomainReactions { values: missing })
            .is_err()
    );
    let mut foreign = values.clone();
    let residual = foreign.remove(&regions[0].0).unwrap();
    foreign.insert(
        eqiora_core::Id::<eqiora_core::entity::kinds::Domain>::new().erase(),
        residual,
    );
    assert!(
        projection
            .recover(&RecoveredDomainReactions { values: foreign })
            .is_err()
    );
    let mut short = values.clone();
    short.get_mut(&regions[0].0).unwrap().pop();
    assert!(
        projection
            .recover(&RecoveredDomainReactions { values: short })
            .is_err()
    );
    let mut nonfinite = values;
    nonfinite.get_mut(&regions[0].0).unwrap()[0] = f64::INFINITY;
    assert!(
        projection
            .recover(&RecoveredDomainReactions { values: nonfinite })
            .is_err()
    );
}
