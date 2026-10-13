use super::*;

type DriverValues = (Id<kinds::Field>, BTreeMap<VertexId, [f64; 2]>);

struct Fixture {
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<2>,
    policy: P1HarmonicMeshMotionPolicy,
    internal_connection: Id<kinds::Connection>,
}

impl Fixture {
    fn new(split_driven: bool) -> Self {
        let (mesh, pair) = refined_partition();
        let third_domain = Id::new();
        let third_field = Id::new();
        let connections = [Id::new(), Id::new(), Id::new()];
        let (split, other, split_field, other_field) = if split_driven {
            (
                solid_domain(),
                fluid_domain(),
                solid_velocity(),
                fluid_velocity(),
            )
        } else {
            (
                fluid_domain(),
                solid_domain(),
                fluid_velocity(),
                solid_velocity(),
            )
        };
        let (lower, upper) = pair
            .domain_cells(split)
            .unwrap()
            .iter()
            .copied()
            .partition::<Vec<_>, _>(|cell| {
                mesh.cells()[cell.index()]
                    .iter()
                    .all(|&vertex| mesh.vertices()[vertex][1] <= 0.5)
            });
        let endpoint = TraceFieldEndpoint::new;
        let trace = |connection, left, right| {
            ConformingTraceQuotient::new(
                eqiora_realization::ConformingTraceSource::ConservingConnection(connection),
                left,
                right,
            )
            .unwrap()
        };
        let partition = FixedReferenceFsiPartition::new(
            &mesh,
            [
                (split, lower),
                (third_domain, upper),
                (other, pair.domain_cells(other).unwrap().to_vec()),
            ],
            &[
                trace(
                    connections[0],
                    endpoint(split, split_field),
                    endpoint(other, other_field),
                ),
                trace(
                    connections[1],
                    endpoint(third_domain, third_field),
                    endpoint(other, other_field),
                ),
                trace(
                    connections[2],
                    endpoint(split, split_field),
                    endpoint(third_domain, third_field),
                ),
            ],
        )
        .unwrap();
        let (harmonic, drivers) = if split_driven {
            (
                vec![fluid_domain()],
                vec![
                    (solid_domain(), solid_displacement()),
                    (third_domain, Id::new()),
                ],
            )
        } else {
            (
                vec![fluid_domain(), third_domain],
                vec![(solid_domain(), solid_displacement())],
            )
        };
        let policy = P1HarmonicMeshMotionPolicy::new(
            harmonic,
            drivers,
            connections[..2].iter().copied(),
            AleGeometryQualityGate::new(0.01).unwrap(),
            reference_solver().plan(),
        )
        .unwrap();
        Self {
            mesh,
            partition,
            policy,
            internal_connection: connections[2],
        }
    }

    fn action(&self) -> P1HarmonicMeshMotionAction<2> {
        P1HarmonicMeshMotionAction::new(
            &self.mesh,
            &self.partition,
            self.policy.clone(),
            reference_solver(),
        )
        .unwrap()
    }

    fn values(&self) -> Vec<DriverValues> {
        self.policy
            .drivers()
            .iter()
            .map(|&(domain, field)| {
                let values = self
                    .partition
                    .domain_vertices(domain)
                    .unwrap()
                    .iter()
                    .map(|&vertex| {
                        let value = if self.mesh.vertices()[vertex.index()][1] == 0.5 {
                            0.125
                        } else {
                            0.0
                        };
                        (vertex, [value, 0.0])
                    })
                    .collect();
                (field, values)
            })
            .collect()
    }
}

#[test]
fn harmonic_motion_composes_three_domains_without_physical_roles() {
    for split_driven in [false, true] {
        let fixture = Fixture::new(split_driven);
        let action = fixture.action();
        let values = fixture.values();
        let input = || values.iter().map(|(field, values)| (*field, values));
        let displacement = action.apply(input()).unwrap();
        assert_eq!(displacement, action.apply_jvp(input()).unwrap());
        // At (1/2,1/2) the P1 Laplacian on the uniform right-triangle patch
        // has diagonal 4 and four axial neighbor weights -1. Only the right
        // neighbor is nonzero (1/8), hence the independently derived 1/32.
        // 1e-13 bounds this unit-scale solve at the requested 1e-13/1e-15 tolerance.
        let interior = action.harmonic_interior_vertices();
        assert_eq!(interior.len(), 1);
        assert_eq!(fixture.mesh.vertices()[interior[0].index()], [0.5, 0.5]);
        assert!((displacement[interior[0].index()][0] - 1.0 / 32.0).abs() < 1e-13);
        assert_eq!(displacement[interior[0].index()][1], 0.0);
        for (_, field) in &values {
            for (&vertex, &value) in field {
                assert_eq!(displacement[vertex.index()], value);
            }
        }
        for vertex in action.fixed_exterior_vertices() {
            assert_eq!(displacement[vertex.index()], [0.; 2]);
        }
        let reversed = P1HarmonicMeshMotionPolicy::new(
            fixture.policy.harmonic_domains().iter().rev().copied(),
            fixture.policy.drivers().iter().rev().copied(),
            fixture.policy.interfaces().iter().rev().copied(),
            fixture.policy.quality_gate(),
            fixture.policy.solver(),
        )
        .unwrap();
        assert_eq!(fixture.policy, reversed);
        let reversed = P1HarmonicMeshMotionAction::new(
            &fixture.mesh,
            &fixture.partition,
            reversed,
            reference_solver(),
        )
        .unwrap();
        assert_eq!(displacement, reversed.apply(input().rev()).unwrap());
    }
}

#[test]
fn multiple_drivers_require_exact_fields_vertices_and_shared_values() {
    let fixture = Fixture::new(true);
    let action = fixture.action();
    let values = fixture.values();
    let [(first, left), (second, right)] = values.as_slice() else {
        panic!("two fixture drivers")
    };
    assert!(action.apply([(*first, left)]).is_err());
    assert!(action.apply([(*first, left), (Id::new(), right)]).is_err());
    assert!(
        action
            .apply([(*first, left), (*second, right), (*first, left)])
            .is_err()
    );
    let mut missing = left.clone();
    missing.pop_first();
    assert!(
        action
            .apply([(*first, &missing), (*second, right)])
            .is_err()
    );
    let mut foreign = left.clone();
    foreign.insert(VertexId::new(fixture.mesh.vertices().len()), [0.; 2]);
    assert!(
        action
            .apply([(*first, &foreign), (*second, right)])
            .is_err()
    );
    let mut nonfinite = left.clone();
    nonfinite.first_entry().unwrap().get_mut()[0] = f64::NAN;
    assert!(
        action
            .apply_jvp([(*first, &nonfinite), (*second, right)])
            .is_err()
    );
    let shared = *left
        .keys()
        .find(|vertex| right.contains_key(vertex))
        .unwrap();
    let mut disagreement = right.clone();
    disagreement.get_mut(&shared).unwrap()[0] += 0.125;
    assert!(
        action
            .apply([(*first, left), (*second, &disagreement)])
            .is_err()
    );
    assert!(
        action
            .apply_jvp([(*first, left), (*second, &disagreement)])
            .is_err()
    );
}

#[test]
fn harmonic_interfaces_are_exactly_the_cross_group_boundary() {
    let fixture = Fixture::new(true);
    for interfaces in [
        fixture.policy.interfaces()[..1].to_vec(),
        vec![Id::new()],
        fixture
            .policy
            .interfaces()
            .iter()
            .copied()
            .chain([fixture.internal_connection])
            .collect(),
    ] {
        let policy = P1HarmonicMeshMotionPolicy::new(
            fixture.policy.harmonic_domains().iter().copied(),
            fixture.policy.drivers().iter().copied(),
            interfaces,
            fixture.policy.quality_gate(),
            fixture.policy.solver(),
        )
        .unwrap();
        assert!(
            P1HarmonicMeshMotionAction::new(
                &fixture.mesh,
                &fixture.partition,
                policy,
                reference_solver()
            )
            .is_err()
        );
    }
    let policy = P1HarmonicMeshMotionPolicy::new(
        fixture.policy.harmonic_domains().iter().copied(),
        [fixture.policy.drivers()[0]],
        fixture.policy.interfaces().iter().copied(),
        fixture.policy.quality_gate(),
        fixture.policy.solver(),
    )
    .unwrap();
    assert!(
        P1HarmonicMeshMotionAction::new(
            &fixture.mesh,
            &fixture.partition,
            policy,
            reference_solver()
        )
        .is_err()
    );
}

#[test]
fn coincident_interfaces_do_not_hide_a_missing_connection() {
    let fixture = Fixture::new(true);
    let mut quotients = fixture.partition.quotients().collect::<Vec<_>>();
    let first = quotients
        .iter()
        .find(|q| {
            fixture.policy.interfaces().iter().any(|&connection| {
                q.source()
                    == eqiora_realization::ConformingTraceSource::ConservingConnection(connection)
            })
        })
        .unwrap();
    let [left, right] = first.endpoints();
    let extra = Id::new();
    quotients.push(
        ConformingTraceQuotient::new(
            eqiora_realization::ConformingTraceSource::ConservingConnection(extra),
            TraceFieldEndpoint::new(left.domain(), Id::new()),
            TraceFieldEndpoint::new(right.domain(), Id::new()),
        )
        .unwrap(),
    );
    let partition = FixedReferenceFsiPartition::new(
        &fixture.mesh,
        fixture.partition.domains().map(|domain| {
            (
                domain,
                fixture.partition.domain_cells(domain).unwrap().to_vec(),
            )
        }),
        &quotients,
    )
    .unwrap();
    assert!(
        P1HarmonicMeshMotionAction::new(
            &fixture.mesh,
            &partition,
            fixture.policy.clone(),
            reference_solver()
        )
        .is_err()
    );
    let complete = P1HarmonicMeshMotionPolicy::new(
        fixture.policy.harmonic_domains().iter().copied(),
        fixture.policy.drivers().iter().copied(),
        fixture.policy.interfaces().iter().copied().chain([extra]),
        fixture.policy.quality_gate(),
        fixture.policy.solver(),
    )
    .unwrap();
    let action =
        P1HarmonicMeshMotionAction::new(&fixture.mesh, &partition, complete, reference_solver())
            .unwrap();
    let values = fixture.values();
    let input = || values.iter().map(|(field, values)| (*field, values));
    assert_eq!(
        action.apply(input()).unwrap(),
        fixture.action().apply(input()).unwrap()
    );
}
