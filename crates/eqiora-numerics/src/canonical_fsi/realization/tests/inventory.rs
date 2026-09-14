use super::*;
use eqiora_core::Id;
use eqiora_realization::{ConformingTraceQuotient, DomainFieldDiscretization, FieldSpaceBinding};

#[test]
fn replay_graph_requires_exact_field_spaces_and_complete_quotient_inventory() {
    let fixture = Fixture::new(SOURCE);
    let plan = fixed_reference_fsi_plan_2d(
        &fixture.model,
        mesh_reference(),
        DynQuantity::new(0.1, TIME),
        scales(),
        reference_solver(),
    )
    .unwrap();
    let graph = fixture
        .resolve(plan.clone())
        .portable_graph(solid_kinematic_relation(&fixture.model))
        .unwrap();
    assert!(validate::exact_graph_inventory(&plan, &graph));
    let changed = |domains, quotients: Vec<ConformingTraceQuotient>| {
        CoupledFieldwiseRealizationPlan::new(
            CoupledFieldwiseSpatialDiscretization::new(
                plan.spatial().coordinate_length_scale(),
                domains,
                quotients,
                plan.spatial().discretization(),
            )
            .unwrap(),
            plan.time_step(),
            plan.scaling().clone(),
            plan.operator_properties(),
            plan.solver(),
            plan.target(),
            plan.schedule(),
        )
        .unwrap()
    };
    let quotient = plan.spatial().trace_quotients()[0];
    let endpoints = quotient.endpoints();
    let additional = ConformingTraceQuotient::new(Id::new(), endpoints[0], endpoints[1]).unwrap();
    let missing = changed(
        plan.spatial().domains().to_vec(),
        vec![quotient, additional],
    );
    assert!(!validate::exact_graph_inventory(&missing, &graph));
    let stale = changed(
        plan.spatial().domains().to_vec(),
        vec![ConformingTraceQuotient::new(Id::new(), endpoints[0], endpoints[1]).unwrap()],
    );
    assert!(!validate::exact_graph_inventory(&stale, &graph));
    let pressure = fluid_pressure(&fixture.model);
    let domains = plan
        .spatial()
        .domains()
        .iter()
        .map(|domain| {
            DomainFieldDiscretization::new(
                domain.domain(),
                domain.field_spaces().iter().map(|field| {
                    FieldSpaceBinding::new(
                        field.field(),
                        if field.field() == pressure {
                            Space::continuous_lagrange(NonZeroU16::new(2).unwrap())
                        } else {
                            field.space()
                        },
                    )
                }),
                domain.constraints().iter().copied(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let wrong_space = changed(domains, plan.spatial().trace_quotients().to_vec());
    assert!(!validate::exact_graph_inventory(&wrong_space, &graph));
}
