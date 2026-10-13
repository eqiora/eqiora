use super::*;
use crate::P1HarmonicMeshMotionPolicy;

struct Authoring {
    base: PortableRealizationGraph,
    motion: P1HarmonicMeshMotionPolicy,
    current: Vec<Id<kinds::Domain>>,
    pullbacks: Vec<BackwardEulerRelationStep>,
}

impl Authoring {
    fn new() -> Self {
        let mut graph = geometry::fixture(2);
        let GeometryActionNode::P1HarmonicExtension {
            harmonic_domains,
            drivers,
            interfaces,
            quality_gate,
            solver,
            ..
        } = &graph.geometry_actions[0];
        let motion = P1HarmonicMeshMotionPolicy::new(
            harmonic_domains
                .iter()
                .map(|id| graph.domain(*id).unwrap().domain()),
            drivers.iter().map(|id| {
                let field = graph.field(*id).unwrap();
                (
                    graph.domain(field.domain()).unwrap().domain(),
                    field.field(),
                )
            }),
            interfaces.iter().copied(),
            *quality_gate,
            *solver,
        )
        .unwrap();
        let current = graph
            .domains
            .iter()
            .filter_map(|domain| {
                matches!(
                    domain.configuration,
                    DomainConfiguration::CurrentAleGeometry { .. }
                )
                .then_some(domain.domain)
            })
            .collect();
        let pullbacks = graph
            .transformations
            .iter()
            .filter_map(|node| match node {
                TransformationNode::BackwardEulerDerivative {
                    relation,
                    state,
                    duration,
                } => Some(
                    BackwardEulerRelationStep::new(
                        *relation,
                        graph.field(*state).unwrap().field(),
                        *duration,
                    )
                    .unwrap(),
                ),
                _ => None,
            })
            .collect();
        graph.geometry_actions.clear();
        for domain in &mut graph.domains {
            domain.configuration = DomainConfiguration::FixedGeometry;
        }
        graph
            .transformations
            .retain(|node| matches!(node, TransformationNode::ConformingTraceQuotient { .. }));
        graph.systems[0].transformations = (0..graph.transformations.len())
            .map(TransformationId::new)
            .collect();
        graph.validate().unwrap();
        Self {
            base: graph,
            motion,
            current,
            pullbacks,
        }
    }
}

#[test]
fn geometry_authoring_is_independent_of_selection_order() {
    let input = Authoring::new();
    let ordinary = input
        .base
        .clone()
        .with_harmonic_geometry(
            &input.motion,
            input.current.clone(),
            input.pullbacks.clone(),
        )
        .unwrap();
    let reversed = input
        .base
        .with_harmonic_geometry(
            &input.motion,
            input.current.into_iter().rev(),
            input.pullbacks.into_iter().rev(),
        )
        .unwrap();
    assert_eq!(ordinary.to_bytes().unwrap(), reversed.to_bytes().unwrap());
    assert_eq!(ordinary.digest().unwrap(), reversed.digest().unwrap());
}

#[test]
fn geometry_authoring_rejects_inexact_selection() {
    let input = Authoring::new();
    for variant in 0..8 {
        let mut current = input.current.clone();
        let mut pullbacks = input.pullbacks.clone();
        match variant {
            0 => current.push(current[0]),
            1 => current[0] = Id::new(),
            2 => pullbacks.clear(),
            3 => pullbacks.push(pullbacks[0]),
            4 => {
                pullbacks[0] = BackwardEulerRelationStep::new(
                    pullbacks[0].relation(),
                    Id::new(),
                    pullbacks[0].duration(),
                )
                .unwrap()
            }
            5 => {
                pullbacks[0] = BackwardEulerRelationStep::new(
                    pullbacks[0].relation(),
                    pullbacks[0].state(),
                    DynQuantity::new(0.2, TIME),
                )
                .unwrap()
            }
            6 => current.clear(),
            7 => {
                current.pop();
            }
            _ => unreachable!(),
        }
        assert!(
            input
                .base
                .clone()
                .with_harmonic_geometry(&input.motion, current, pullbacks)
                .is_err(),
            "variant {variant}"
        );
    }
    let accepted = input
        .base
        .with_harmonic_geometry(
            &input.motion,
            input.current.clone(),
            input.pullbacks.clone(),
        )
        .unwrap();
    assert!(
        accepted
            .with_harmonic_geometry(&input.motion, input.current, input.pullbacks)
            .unwrap_err()
            .to_string()
            .contains("already has a current geometry")
    );
}

#[test]
fn geometry_authoring_reuses_an_exact_derivative_but_rejects_a_conflict() {
    let input = Authoring::new();
    let step = input.pullbacks[0];
    let state = FieldRepresentationId::new(
        input
            .base
            .fields
            .iter()
            .position(|field| field.field == step.state())
            .unwrap(),
    );
    for duration in [0.1, 0.2] {
        let mut graph = input.base.clone();
        graph
            .transformations
            .push(TransformationNode::BackwardEulerDerivative {
                relation: step.relation(),
                state,
                duration: DynQuantity::new(duration, TIME),
            });
        graph.systems[0]
            .transformations
            .push(TransformationId::new(graph.transformations.len() - 1));
        graph.validate().unwrap();
        let result = graph.with_harmonic_geometry(
            &input.motion,
            input.current.clone(),
            input.pullbacks.clone(),
        );
        if duration == 0.1 {
            let graph = result.unwrap();
            assert_eq!(graph.transformations.iter().filter(|node| matches!(node, TransformationNode::BackwardEulerDerivative { state: id, .. } if *id == state)).count(), 1);
        } else {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("conflicts with its existing derivative")
            );
        }
    }
}

#[test]
fn graph_replay_requires_physical_time_for_geometry() {
    let graph = geometry::fixture(2);
    let text = String::from_utf8(graph.to_bytes().unwrap()).unwrap();
    let time = "[[0,1],[0,1],[1,1],[0,1],[0,1],[0,1],[0,1]]";
    let length = "[[0,1],[1,1],[0,1],[0,1],[0,1],[0,1],[0,1]]";
    assert!(text.contains(time));
    let changed = text.replace(time, length); // Change both action and derivative to keep them equal.
    assert!(
        PortableRealizationGraph::from_bytes(changed.as_bytes())
            .unwrap_err()
            .to_string()
            .contains("physical time dimension")
    );
}

#[test]
fn triangular_quadrature_checks_basis_product_degree_and_dimension() {
    // P1 products have degree 2, while the cubic triangle bubble squared has
    // degree 6. The Duffy Jacobian adds one degree: Gaussian exactness 2n-1
    // therefore requires respectively n >= 2 and n >= 4 on each axis.
    for (space, minimum) in [
        (Space::continuous_lagrange(NonZeroU16::MIN), Some(2)),
        (Space::simplex_p1_bubble(), Some(4)),
        (Space::tetrahedral_edge(), None),
    ] {
        for points in [1, 2, 3, 4, 5] {
            let admitted = Discretization::new(
                DiscretizationMethod::ContinuousGalerkin,
                MeshPolicy::ImportedSimplicial {
                    artifact: MeshArtifactReference::from_sha256([1; 32]),
                },
                QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension: std::num::NonZeroUsize::new(2).expect("two is nonzero"),
                    points_per_axis: NonZeroUsize::new(points).unwrap(),
                },
            )
            .validate_space(space)
            .is_ok();
            assert_eq!(admitted, minimum.is_some_and(|minimum| points >= minimum));
        }
    }
}

#[test]
fn motion_policy_must_match_the_existing_field_domain_bindings() {
    let input = Authoring::new();
    let (domain, _) = input.motion.drivers()[0];
    let foreign_driver = input.pullbacks[0].state();
    let mismatched = P1HarmonicMeshMotionPolicy::new(
        input.motion.harmonic_domains().iter().copied(),
        [(domain, foreign_driver)],
        input.motion.interfaces().iter().copied(),
        input.motion.quality_gate(),
        input.motion.solver(),
    )
    .unwrap();
    assert!(
        input
            .base
            .with_harmonic_geometry(&mismatched, input.current, input.pullbacks)
            .unwrap_err()
            .to_string()
            .contains("mismatched Field/Domain")
    );
}

#[test]
fn driven_domain_configuration_is_an_explicit_numerical_choice() {
    let mut input = Authoring::new();
    let (domain, field) = input.motion.drivers()[0];
    input.current.push(domain);
    input.pullbacks.push(
        BackwardEulerRelationStep::new(Id::new(), field, DynQuantity::new(0.1, TIME)).unwrap(),
    );
    let graph = input
        .base
        .with_harmonic_geometry(&input.motion, input.current, input.pullbacks)
        .unwrap();
    assert!(graph.domains().iter().all(|domain| matches!(
        domain.configuration(),
        DomainConfiguration::CurrentAleGeometry { .. }
    )));
    assert_eq!(
        PortableRealizationGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(),
        graph
    );
}
