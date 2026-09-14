use super::*;
use eqiora_core::{DimExponents, DynQuantity, Id};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_realization::{
    ConformingTraceQuotient, CoupledFieldwiseSpatialDiscretization, DomainFieldDiscretization,
    FieldSpaceBinding, MeshArtifactReference,
};
use eqiora_solver::{LinearSolver, SolverPlan};

fn fixture() -> (EquationRoles, CoupledFieldwiseRealizationPlan) {
    let source = include_str!(
        "../../../../../../verify/fsi/fixed-reference-monolithic-step-2d/models/direct.eqi"
    );
    let (transaction, model, _) = eqiora_compiler::compile("roles.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = eqiora_sem::KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let model = crate::canonical_fsi::lower_fixed_reference_fsi_cartesian_2d(&program).unwrap();
    let quantity =
        |value, powers| DynQuantity::new(value, DimExponents::from_integers(powers).unwrap());
    let scales = crate::fsi::FixedReferenceFsiScaleProfile2d::new(
        quantity(1.0, [0, 1, 0, 0, 0, 0, 0]),
        quantity(1.0, [0, 1, -1, 0, 0, 0, 0]),
        quantity(1.0, [1, -1, -2, 0, 0, 0, 0]),
    )
    .unwrap();
    let solver = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1e-10,
        1e-12,
        std::num::NonZeroUsize::new(100).unwrap(),
    )
    .unwrap();
    let plan = crate::fsi::fixed_reference_fsi_plan_2d(
        &model,
        MeshArtifactReference::from_sha256([7; 32]),
        quantity(0.1, [0, 0, 1, 0, 0, 0, 0]),
        scales,
        solver,
    )
    .unwrap();
    let equations = EquationRoles::derive(
        &program,
        plan.spatial().domains().iter().map(|d| d.domain().erase()),
    )
    .unwrap();
    (equations, plan)
}

fn spatial(
    plan: &CoupledFieldwiseRealizationPlan,
    domains: Vec<DomainFieldDiscretization>,
    quotients: Vec<ConformingTraceQuotient>,
) -> CoupledFieldwiseRealizationPlan {
    CoupledFieldwiseRealizationPlan::new(
        CoupledFieldwiseSpatialDiscretization::new(
            plan.spatial().coordinate_length_scale(),
            domains,
            quotients,
            plan.spatial().discretization(),
        )
        .unwrap(),
        plan.time_step().clone(),
        plan.scaling().clone(),
        plan.operator_properties(),
        plan.solver(),
        plan.target(),
        plan.schedule(),
    )
    .unwrap()
}

#[test]
fn roles_follow_exact_equations_and_permuted_plan_inventory() {
    let (equations, plan) = fixture();
    let expected = FsiRoles::derive(&equations, &plan).unwrap();
    let mut domains = plan.spatial().domains().to_vec();
    domains.reverse();
    let domains = domains
        .into_iter()
        .map(|d| {
            let mut fields = d.field_spaces().to_vec();
            fields.reverse();
            DomainFieldDiscretization::new(d.domain(), fields, d.constraints().iter().copied())
                .unwrap()
        })
        .collect();
    let reversed = spatial(&plan, domains, plan.spatial().trace_quotients().to_vec());
    assert_eq!(expected, FsiRoles::derive(&equations, &reversed).unwrap());
    assert_eq!(
        expected.solid_velocity,
        plan.time_step().eliminated_states()[0]
            .pair()
            .rate()
            .erase()
    );
    assert_ne!(expected.fluid_velocity, expected.pressure);
}

#[test]
fn roles_reject_stale_missing_equations_layout_and_plural_projection() {
    let (equations, plan) = fixture();
    let roles = FsiRoles::derive(&equations, &plan).unwrap();
    let mut stale = equations.clone();
    stale.fields.get_mut(&roles.fluid_velocity).unwrap().0 =
        Id::<eqiora_core::entity::kinds::Domain>::new().erase();
    assert!(FsiRoles::derive(&stale, &plan).is_err());
    let mut missing = equations.clone();
    missing.fields.remove(&roles.pressure);
    assert!(FsiRoles::derive(&missing, &plan).is_err());
    let mut extra_role = equations.clone();
    let mut extra = extra_role
        .relations
        .values()
        .find(|r| matches!(r.kind, Role::Residual { .. }))
        .unwrap()
        .clone();
    extra.kind = Role::Residual {
        tested: Id::<eqiora_core::entity::kinds::Field>::new().erase(),
    };
    extra_role.relations.insert(
        Id::<eqiora_core::entity::kinds::Relation>::new().erase(),
        extra,
    );
    assert!(FsiRoles::derive(&extra_role, &plan).is_err());
    // A scalar tested residual with the same velocity dependency is insufficient:
    // it must carry the equation owner's actual divergence/multiplier proof.
    let mut foreign_scalar_residual = equations.clone();
    foreign_scalar_residual.constraints.clear();
    assert!(FsiRoles::derive(&foreign_scalar_residual, &plan).is_err());
    let mut missing_rate = equations.clone();
    missing_rate
        .relations
        .retain(|_, r| !matches!(r.kind, Role::Kinematic { .. }));
    assert!(FsiRoles::derive(&missing_rate, &plan).is_err());
    let domains = plan
        .spatial()
        .domains()
        .iter()
        .map(|d| {
            DomainFieldDiscretization::new(
                d.domain(),
                d.field_spaces().iter().map(|f| {
                    FieldSpaceBinding::new(
                        f.field(),
                        if f.field().erase() == roles.pressure {
                            Space::continuous_lagrange(std::num::NonZeroU16::new(2).unwrap())
                        } else {
                            f.space()
                        },
                    )
                }),
                d.constraints().iter().copied(),
            )
            .unwrap()
        })
        .collect();
    assert!(
        FsiRoles::derive(
            &equations,
            &spatial(&plan, domains, plan.spatial().trace_quotients().to_vec())
        )
        .is_err()
    );
    let quotient = plan.spatial().trace_quotients()[0];
    let endpoints = quotient.endpoints();
    let extra = ConformingTraceQuotient::new(Id::new(), endpoints[0], endpoints[1]).unwrap();
    let plural = spatial(
        &plan,
        plan.spatial().domains().to_vec(),
        vec![quotient, extra],
    );
    assert!(FsiRoles::derive(&equations, &plural).is_err());
}
