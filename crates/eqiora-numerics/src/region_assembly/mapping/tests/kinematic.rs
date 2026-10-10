//! Exact kinematic history uses the same mapped solve as ordinary storage.
use super::*;
use crate::form_compiler::region::RegionTimeBinding;
use eqiora_realization::{
    BackwardEulerStateBinding, BackwardEulerStatePair, PositivePhysicalScale,
};
use eqiora_solver::LinearSolveRequest;

#[test]
fn mapped_solve_recovers_kinematic_fields_and_rejects_incomplete_history() {
    let (transaction, model, _) = compile(
        "oscillator.eqi",
        "model Oscillator() {
        domain body=box(0,1,0,1);
        state d:vector<m,2> on body in h1;
        state v:vector<m/s,2> on body in h1;
        relation pair on body {derivative(d)=v;}
        relation balance on body {derivative(v)+1[1/s^2]*d=0;}
    }",
    )
    .unwrap()
    .remove(0)
    .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let domain = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Domain(domain) => Some(domain.id().erase()),
            _ => None,
        })
        .unwrap();
    let roles =
        crate::form_compiler::equation_roles::EquationRoles::derive(&program, [domain]).unwrap();
    let (relation, state, rate) = roles
        .relations
        .iter()
        .find_map(|(relation, role)| match role.kind {
            crate::form_compiler::equation_roles::Role::Kinematic { state, rate } => {
                Some((*relation, state, rate))
            }
            _ => None,
        })
        .unwrap();
    let space = Space::continuous_lagrange(NonZeroU16::MIN);
    let compiled = CompiledRegionForm::<f64>::derive(&program, domain, 2).unwrap();
    let fields = compiled
        .fields()
        .map(|(field, ty)| RegionFieldBinding {
            field,
            space,
            scale: DynQuantity::new(1., ty.dimension()),
        })
        .collect::<Vec<_>>();
    let area = DimExponents::from_integers([0, 2, 0, 0, 0, 0, 0]).unwrap();
    let rows = compiled
        .rows()
        .map(|(relation, _, ty)| {
            (
                relation,
                DynQuantity::new(1., ty.dimension().mul(area).unwrap().pow(-1, 1).unwrap()),
            )
        })
        .collect();
    let second = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    let bound = compiled
        .bind(
            ReferenceCell::hypercube(2).unwrap(),
            &fields,
            &rows,
            Some(&RegionTimeBinding {
                step: DynQuantity::new(0.5, second),
                states: vec![BackwardEulerStateBinding::new(
                    BackwardEulerStatePair::new(
                        relation.downcast().unwrap(),
                        state.downcast().unwrap(),
                        rate.downcast().unwrap(),
                    )
                    .unwrap(),
                    space,
                    PositivePhysicalScale::new(DynQuantity::new(
                        1.,
                        roles.fields[&state].1.dimension(),
                    ))
                    .unwrap(),
                )],
            }),
        )
        .unwrap();
    let mesh = CartesianMesh::from_axes(vec![vec![0., 1.], vec![0., 1.]]).unwrap();
    let mapping = RegionDofMap::new(
        &mesh,
        &BTreeMap::from([(domain, bound.fields().to_vec())]),
        bound.reference_cell(),
        &[domain],
        &[],
        &BTreeMap::new(),
    )
    .unwrap();
    let mut previous = mapping
        .recover(&vec![0.; mapping.free_count()], &[rate])
        .unwrap();
    let mut displacement = previous[&rate].clone();
    displacement.value_type = roles.fields[&state].1.clone();
    displacement.coefficients = displacement
        .coefficients
        .keys()
        .map(|key| {
            (
                FieldDof {
                    field: state,
                    ..*key
                },
                if key.component == 0 { 2. } else { 3. },
            )
        })
        .collect();
    previous.insert(state, displacement);
    let execute = |history| {
        mapping.solve(
            &mesh,
            RegionSolveInput {
                operator_properties: LinearOperatorProperties::General,
                forms: vec![(
                    bound.clone(),
                    QuadratureRule::tensor_product_gauss_legendre(2, 3).unwrap(),
                )],
                natural: vec![],
                previous: history,
                prescribed_states: BTreeMap::new(),
                geometry_action: None,
            },
            NonZeroUsize::MIN,
            LinearSolveRequest::new(
                &REFERENCE_LINEAR_SOLVER,
                SolverPlan::new(
                    LinearSolver::BiConjugateGradientStabilized,
                    1e-13,
                    1e-14,
                    NonZeroUsize::new(100).unwrap(),
                )
                .unwrap(),
            ),
            |reactions, values| reactions.recover(values),
        )
    };
    assert!(
        execute(None)
            .err()
            .unwrap()
            .message()
            .contains("requires history")
    );
    let mut missing = previous.clone();
    missing.remove(&state);
    assert!(
        execute(Some(missing))
            .err()
            .unwrap()
            .message()
            .contains("physical Field inventory")
    );
    let mut wrong = previous.clone();
    wrong.get_mut(&state).unwrap().value_type = roles.fields[&rate].1.clone();
    assert!(execute(Some(wrong)).is_err());
    let output = execute(Some(previous.clone())).unwrap();
    // For d'=v and v'=-d, BE gives d_next=d_old/(1+dt²), v_next=-dt*d_next.
    for (&field, values) in &output.fields {
        for (key, value) in &values.coefficients {
            let initial = if key.component == 0 { 2. } else { 3. };
            let expected = if field == state {
                initial / 1.25
            } else {
                -0.5 * initial / 1.25
            };
            assert!((value - expected).abs() < 1e-12);
        }
    }
    assert_eq!(output.fields.len(), 2);
    assert!(
        previous[&rate]
            .coefficients
            .values()
            .all(|value| *value == 0.)
    );
    let temporal = bound.time_binding().unwrap();
    let step =
        eqiora_realization::BackwardEulerStep::new(temporal.step, temporal.states.clone()).unwrap();
    let key = *previous[&state].coefficients.keys().next().unwrap();
    let rate_key = FieldDof { field: rate, ..key };
    let prescribed = BTreeMap::from([(key, previous[&state].coefficients[&key] + 0.25)]);
    // dt=1/2: a displacement increment 1/4 requires rate 1/2.
    let constrained = RegionDofMap::new(
        &mesh,
        &BTreeMap::from([(domain, bound.fields().to_vec())]),
        bound.reference_cell(),
        &[domain],
        &[],
        &BTreeMap::from([(rate_key, 0.5)]),
    )
    .unwrap();
    let old = previous.clone();
    let recovered = constrained
        .recover_step(
            &vec![0.; constrained.free_count()],
            &previous,
            &step,
            &prescribed,
        )
        .unwrap();
    assert_eq!(recovered[&state].coefficients[&key], prescribed[&key]);
    assert_eq!(recovered[&rate].coefficients[&rate_key], 0.5);
    assert!(
        mapping
            .recover_step(
                &vec![0.; mapping.free_count()],
                &previous,
                &step,
                &prescribed
            )
            .unwrap_err()
            .message()
            .contains("history-dependent rate constraint")
    );
    for bad in [previous[&state].coefficients[&key], f64::NAN] {
        assert!(
            constrained
                .recover_step(
                    &vec![0.; constrained.free_count()],
                    &previous,
                    &step,
                    &BTreeMap::from([(key, bad)])
                )
                .is_err()
        );
    }
    assert!(
        constrained
            .recover_step(
                &vec![0.; constrained.free_count()],
                &previous,
                &step,
                &BTreeMap::from([(rate_key, 0.5)])
            )
            .unwrap_err()
            .message()
            .contains("not an eliminated state")
    );
    assert_eq!(previous, old);
}
