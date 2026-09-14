use super::*;

#[test]
fn three_domains_resolve_all_quotients_independently_of_input_order() {
    let fixture = Fixture::new();
    let original = fixture.plan();
    let third_domain = Id::new();
    let third_field = Id::new();
    let third_state = Id::new();
    let third_relation = Id::new();
    let third_pair = BackwardEulerStatePair::new(third_relation, third_state, third_field).unwrap();
    let third_binding = BackwardEulerStateBinding::new(
        third_pair,
        Space::continuous_lagrange(NonZeroU16::MIN),
        physical_scale(length_dimension()),
    );
    let step = BackwardEulerStep::new(
        original.time_step().duration(),
        [third_binding, fixture.state_binding()],
    )
    .unwrap();
    assert_eq!(
        step,
        BackwardEulerStep::new(
            original.time_step().duration(),
            [fixture.state_binding(), third_binding]
        )
        .unwrap()
    );
    let mut domains = original.spatial().domains().to_vec();
    domains.push(
        DomainFieldDiscretization::new(
            third_domain,
            [FieldSpaceBinding::new(
                third_field,
                Space::continuous_lagrange(NonZeroU16::MIN),
            )],
            [],
        )
        .unwrap(),
    );
    let first = fixture.trace(fixture.connection);
    let second = ConformingTraceQuotient::new(
        Id::new(),
        TraceFieldEndpoint::new(fixture.second_domain, fixture.second_trace),
        TraceFieldEndpoint::new(third_domain, third_field),
    )
    .unwrap();
    let spatial = |quotients: &[ConformingTraceQuotient],
                   domains: Vec<DomainFieldDiscretization>| {
        CoupledFieldwiseSpatialDiscretization::new(
            original.spatial().coordinate_length_scale(),
            domains,
            quotients,
            original.spatial().discretization(),
        )
    };
    let selected = spatial(&[first, second], domains.clone()).unwrap();
    domains.reverse();
    assert_eq!(
        selected,
        spatial(&[second, first], domains.clone()).unwrap()
    );
    for bad in [vec![], vec![first, first], vec![first, second, second]] {
        assert!(
            spatial(&bad, domains.clone())
                .unwrap_err()
                .to_string()
                .contains("duplicate selection")
        );
    }
    let absent = ConformingTraceQuotient::new(
        Id::new(),
        TraceFieldEndpoint::new(fixture.second_domain, third_field),
        TraceFieldEndpoint::new(third_domain, fixture.second_trace),
    )
    .unwrap();
    assert!(spatial(&[first, absent], domains.clone()).is_err());
    let mut wrong_space = domains.clone();
    *wrong_space
        .iter_mut()
        .find(|domain| domain.domain() == third_domain)
        .unwrap() = DomainFieldDiscretization::new(
        third_domain,
        [FieldSpaceBinding::new(
            third_field,
            Space::continuous_lagrange(NonZeroU16::new(2).unwrap()),
        )],
        [],
    )
    .unwrap();
    assert!(
        spatial(&[first, second], wrong_space)
            .unwrap_err()
            .to_string()
            .contains("trace-space signatures")
    );
    let scales = original.scaling().block_scales().to_vec();
    let plan = |scale| {
        let mut scales = scales.clone();
        scales.push(AlgebraicBlockScale::new(
            AlgebraicBlock::Field(third_field),
            physical_scale_value(scale, velocity_dimension()),
        ));
        CoupledFieldwiseRealizationPlan::new(
            selected.clone(),
            step.clone(),
            SymmetricCongruenceScaling::new(scales, physical_scale(functional_dimension()))
                .unwrap(),
            original.operator_properties(),
            original.solver(),
            original.target(),
            original.schedule(),
        )
    };
    assert!(
        plan(2.0)
            .unwrap_err()
            .to_string()
            .contains("exactly equal congruence scales")
    );
    let plan = plan(1.0).unwrap();
    let request = CoupledFieldwiseRealizationRequest::explicit(
        OntologyId::new(),
        SemanticRevision::new(11),
        RealizationRevision::new(3),
        plan,
    );
    let mut inventory = fixture
        .requirements(true, fixture.connection)
        .domains()
        .to_vec();
    inventory.push(DomainFieldInventory::new(third_domain, [third_field, third_state]).unwrap());
    let requirements = |quotients: &[ConformingTraceQuotient]| {
        CoupledFieldwiseRealizationRequirements::new(
            inventory.clone(),
            quotients,
            [third_pair, fixture.state_pair()],
            execution_requirements(),
        )
        .unwrap()
    };
    let capabilities = RealizationCapabilities::symmetric_mixed_simplicial_2d_reference();
    let resolved =
        resolve_coupled_fieldwise(&request, requirements(&[second, first]), &capabilities).unwrap();
    let graph = resolved.portable_graph().unwrap();
    assert_eq!(graph.domains().len(), 3);
    assert_eq!(graph.transformations().len(), 4);
    let represented = resolved.plan().represented_physical_fields().unwrap();
    assert_eq!(represented.len(), 6);
    assert!(
        represented
            .iter()
            .any(|field| field.field() == third_state && field.domain() == third_domain)
    );
    let transformations = graph
        .transformations()
        .iter()
        .filter_map(|transformation| {
            if let crate::TransformationNode::BackwardEulerElimination {
                relation,
                state,
                rate,
                ..
            } = transformation
            {
                Some((
                    *relation,
                    graph.field(*state).unwrap().field(),
                    graph.field(*rate).unwrap().field(),
                ))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert!(transformations.contains(&(third_relation, third_state, third_field)));
    for bad in [
        vec![fixture.state_pair()],
        vec![
            fixture.state_pair(),
            BackwardEulerStatePair::new(Id::new(), third_state, third_field).unwrap(),
        ],
    ] {
        let bad = CoupledFieldwiseRealizationRequirements::new(
            inventory.clone(),
            [first, second],
            bad,
            execution_requirements(),
        )
        .unwrap();
        assert!(resolve_coupled_fieldwise(&request, bad, &capabilities).is_err());
    }
    for bad in [
        vec![],
        vec![fixture.state_binding(), fixture.state_binding()],
        vec![
            fixture.state_binding(),
            BackwardEulerStateBinding::new(
                BackwardEulerStatePair::new(fixture.kinematic_relation, third_state, third_field)
                    .unwrap(),
                third_binding.state_space(),
                third_binding.state_scale(),
            ),
        ],
        vec![
            fixture.state_binding(),
            BackwardEulerStateBinding::new(
                BackwardEulerStatePair::new(third_relation, fixture.second_trace, third_field)
                    .unwrap(),
                third_binding.state_space(),
                third_binding.state_scale(),
            ),
        ],
    ] {
        assert!(BackwardEulerStep::new(step.duration(), bad).is_err());
    }
    assert_eq!(
        crate::PortableRealizationGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(),
        graph
    );
    for bad in [vec![first], vec![first, second, fixture.trace(Id::new())]] {
        assert!(
            resolve_coupled_fieldwise(&request, requirements(&bad), &capabilities)
                .unwrap_err()
                .to_string()
                .contains("trace quotient differs")
        );
    }
    // One Connection may legitimately select more than one distinct Field pair.
    let another_pair = ConformingTraceQuotient::new(
        fixture.connection,
        TraceFieldEndpoint::new(fixture.first_domain, fixture.first_trace),
        TraceFieldEndpoint::new(third_domain, third_field),
    )
    .unwrap();
    assert!(spatial(&[first, another_pair], domains).is_ok());
}
