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
        Id::new(),
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
            fixture.realization.model().unwrap(),
            fixture.realization.semantic_revision(),
            fixture.realization.realization_revision(),
            plan,
        );
        let resolved = resolve_coupled_fieldwise(
            &request,
            requirements,
            &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
        )
        .unwrap();
        RealizationEnvelopeV8::from_resolved(&fixture.model, &resolved, LayoutArtifacts::Replicated)
            .unwrap()
    };
    let envelope = encode(&quotients);
    let bytes = envelope.canonical_json().unwrap();
    quotients.reverse();
    assert_eq!(encode(&quotients).canonical_json().unwrap(), bytes);
    let decoded = RealizationEnvelopeV8::from_json(&bytes, Default::default()).unwrap();
    assert_eq!(decoded.plan().unwrap(), envelope.plan().unwrap());
    assert_eq!(
        decoded.requirements().unwrap(),
        envelope.requirements().unwrap()
    );
    assert_eq!(decoded.plan().unwrap().spatial().domains().len(), 3);
    assert_eq!(decoded.plan().unwrap().spatial().trace_quotients().len(), 2);
    let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        decoded
            .plan()
            .unwrap()
            .time_step()
            .eliminated_states()
            .len(),
        2
    );
    for pointer in ["/requirements", "/plan/time_step"] {
        for mutation in 0..6 {
            let mut bad = original.clone();
            let owner = bad.pointer_mut(pointer).unwrap();
            let entries = owner["eliminated_states"].as_array_mut().unwrap();
            match mutation {
                0 => {
                    entries.pop();
                }
                1 => {
                    entries.push(entries[0].clone());
                }
                2 => {
                    entries.reverse();
                }
                3 => {
                    entries.clear();
                }
                4 => {
                    let single = entries[0].clone();
                    owner.as_object_mut().unwrap().remove("eliminated_states");
                    owner["eliminated_state"] = single;
                }
                _ => {
                    let pair = if pointer == "/requirements" {
                        &mut entries[0]
                    } else {
                        &mut entries[0]["pair"]
                    };
                    pair["relation_ulid"] =
                        serde_json::json!(Id::<kinds::Relation>::new().ulid().to_string());
                }
            }
            assert!(
                RealizationEnvelopeV8::from_json(
                    &serde_json::to_vec(&bad).unwrap(),
                    Default::default()
                )
                .is_err(),
                "{pointer} elimination mutation {mutation}"
            );
        }
    }
    for pointer in ["/requirements", "/plan/spatial"] {
        for mutation in 0..5 {
            let mut bad = original.clone();
            let owner = bad.pointer_mut(pointer).unwrap();
            let entries = owner["trace_quotients"].as_array_mut().unwrap();
            match mutation {
                0 => {
                    entries.pop();
                }
                1 => {
                    entries.push(entries[0].clone());
                }
                2 => {
                    entries.reverse();
                }
                3 => {
                    entries.clear();
                }
                _ => {
                    let single = entries[0].clone();
                    owner.as_object_mut().unwrap().remove("trace_quotients");
                    owner["trace_quotient"] = single;
                }
            }
            assert!(
                RealizationEnvelopeV8::from_json(
                    &serde_json::to_vec(&bad).unwrap(),
                    Default::default()
                )
                .is_err(),
                "{pointer} mutation {mutation}"
            );
        }
    }
    assert!(
        RealizationEnvelopeV8::from_json(
            &bytes,
            RealizationDecoderLimits {
                max_realization_constraints: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
}
