use super::*;

#[test]
fn coupled_collection_replays_three_domains_and_rejects_wire_drift() {
    let fixture = Fixture::new();
    let domain = Id::new();
    let field = Id::new();
    let state = Id::new();
    let relation = Id::new();
    let pair = BackwardEulerStatePair::new(relation, state, field).unwrap();
    let mut pairs = fixture.requirements.eliminated_states().to_vec();
    pairs.push(pair);
    let mut bindings = fixture.plan.time_step().eliminated_states().to_vec();
    bindings.push(BackwardEulerStateBinding::new(
        pair,
        Space::continuous_lagrange(NonZeroU16::MIN),
        scale(length_dimension()),
    ));
    let step =
        BackwardEulerStep::new(fixture.plan.time_step().duration(), bindings.clone()).unwrap();
    bindings.reverse();
    assert_eq!(
        step,
        BackwardEulerStep::new(fixture.plan.time_step().duration(), bindings).unwrap()
    );
    let rate = fixture.requirements.eliminated_states()[0].rate();
    let rate_domain = fixture
        .requirements
        .domains()
        .iter()
        .find(|inventory| inventory.fields().contains(&rate))
        .unwrap()
        .domain();
    let quotient = ConformingTraceQuotient::new(
        eqiora_realization::ConformingTraceSource::ConservingConnection(Id::new()),
        TraceFieldEndpoint::new(rate_domain, rate),
        TraceFieldEndpoint::new(domain, field),
    )
    .unwrap();
    let mut quotients = fixture.requirements.trace_quotients().to_vec();
    quotients.push(quotient);
    let mut inventories = fixture.requirements.domains().to_vec();
    inventories.push(DomainFieldInventory::new(domain, [field, state]).unwrap());
    let mut domains = fixture.plan.spatial().domains().to_vec();
    domains.push(
        DomainFieldDiscretization::new(
            domain,
            [FieldSpaceBinding::new(
                field,
                Space::continuous_lagrange(NonZeroU16::MIN),
            )],
            [],
        )
        .unwrap(),
    );
    let mut scales = fixture.plan.scaling().block_scales().to_vec();
    scales.push(AlgebraicBlockScale::new(
        AlgebraicBlock::Field(field),
        scale(velocity_dimension()),
    ));
    let encode = |quotients: &[ConformingTraceQuotient]| {
        let requirements = CoupledFieldwiseRealizationRequirements::new(
            inventories.clone(),
            quotients,
            pairs.iter().copied(),
            fixture.requirements.execution(),
        )
        .unwrap();
        let spatial = CoupledFieldwiseSpatialDiscretization::new(
            fixture.plan.spatial().coordinate_length_scale(),
            domains.clone(),
            quotients,
            fixture.plan.spatial().discretization(),
        )
        .unwrap();
        let plan = CoupledFieldwiseRealizationPlan::new(
            spatial,
            step.clone(),
            SymmetricCongruenceScaling::new(
                scales.clone(),
                fixture.plan.scaling().weak_functional_scale(),
            )
            .unwrap(),
            fixture.plan.operator_properties(),
            fixture.plan.solver(),
            fixture.plan.target(),
            fixture.plan.schedule(),
        )
        .unwrap();
        let request = CoupledFieldwiseRealizationRequest::explicit(
            fixture.realization.model(),
            fixture.realization.semantic_revision(),
            RealizationRevision::new(23),
            plan,
        );
        let resolved = resolve_coupled_fieldwise(
            &request,
            requirements,
            &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
        )
        .unwrap();
        RealizationEnvelope::new(
            &fixture.model,
            resolved.portable_graph().unwrap(),
            LayoutArtifacts::Replicated,
        )
        .unwrap()
    };
    let envelope = encode(&quotients);
    let bytes = envelope.canonical_json().unwrap();
    quotients.reverse();
    assert_eq!(encode(&quotients).canonical_json().unwrap(), bytes);
    let decoded = RealizationEnvelope::from_json(&bytes, Default::default()).unwrap();
    assert_eq!(decoded.graph(), envelope.graph());
    assert_eq!(decoded.graph().domains().len(), 3);
    let transformations = decoded.graph().transformations();
    assert_eq!(
        transformations
            .iter()
            .filter(|node| matches!(
                node,
                eqiora_realization::TransformationNode::ConformingTraceQuotient { .. }
            ))
            .count(),
        2
    );
    assert_eq!(
        transformations
            .iter()
            .filter(|node| matches!(
                node,
                eqiora_realization::TransformationNode::BackwardEulerElimination { .. }
            ))
            .count(),
        2
    );
    // This synthetic third Domain proves graph inventory replay only. Actual
    // Model/mesh binding is exercised by the two-Domain semantic fixture.
    for changed in [
        json_edit::edit_array(&bytes, "/graph/transformations", |nodes| {
            nodes.pop();
        }),
        json_edit::edit_array(&bytes, "/graph/transformations", |nodes| {
            nodes.push(nodes[0].clone())
        }),
        json_edit::edit_array(&bytes, "/graph/fields", |nodes| nodes.reverse()),
        json_edit::edit_array(&bytes, "/graph/transformations", |nodes| nodes.clear()),
    ] {
        assert!(RealizationEnvelope::from_json(&changed, Default::default()).is_err());
    }
    assert!(
        RealizationEnvelope::from_json(
            &bytes,
            RealizationDecoderLimits {
                max_realization_constraints: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
}
