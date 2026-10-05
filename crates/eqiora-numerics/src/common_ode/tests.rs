use eqiora_artifact::ModelEnvelope;
use eqiora_core::{DynQuantity, OntologyId};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::{
    ActivationDef, ExprDagBuilder, FieldDef, ParameterDef, RelationDef, SymbolRef,
};
use eqiora_schema::{Model, ModelView};
use eqiora_sem::KernelProgram;
use eqiora_solver::REFERENCE_LINEAR_SOLVER;
use eqiora_time::TimeBackendIdentity;

use super::*;

const DECAY: &str = r#"
model decay() {
  state x: 1;
  initial { x = 1; }
  parameter rate: 1 / s = 1;
  relation flow {
    derivative(x) + rate * x = 0;
  }
}
"#;

fn fixture() -> (ModelEnvelope, KernelProgram) {
    let compiled = eqiora_compiler::compile("decay.eqi", DECAY)
        .unwrap()
        .pop()
        .unwrap();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let envelope = ModelEnvelope::from_program(&program).unwrap();
    (envelope, program)
}

fn two_state_fixture(
    mass_matrix: bool,
) -> (
    ModelEnvelope,
    KernelProgram,
    Id<kinds::Field>,
    Id<kinds::Field>,
) {
    let inverse_time =
        DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension");
    let decay = Id::<kinds::Field>::new();
    let integral = Id::<kinds::Field>::new();
    let rate = Id::<kinds::Parameter>::new();
    let relation = Id::<kinds::Relation>::new();
    let continuous = Id::<kinds::Activation>::new();
    let model = OntologyId::<Model>::new();

    let mut expression = ExprDagBuilder::new();
    let decay_derivative = expression
        .symbol(SymbolRef::Derivative(decay, std::num::NonZeroU32::MIN))
        .unwrap();
    let integral_derivative = expression
        .symbol(SymbolRef::Derivative(integral, std::num::NonZeroU32::MIN))
        .unwrap();
    let decay_value = expression.symbol(SymbolRef::Field(decay)).unwrap();
    let rate_value = expression.symbol(SymbolRef::Parameter(rate)).unwrap();
    let decay_rate = expression.mul(rate_value, decay_value).unwrap();
    let integral_residual = expression.sub(integral_derivative, decay_rate).unwrap();
    let decay_residual = expression.add(decay_derivative, decay_rate).unwrap();
    let decay_residual = if mass_matrix {
        expression.add(decay_residual, integral_derivative).unwrap()
    } else {
        decay_residual
    };
    let residuals = {
        let equation_zero_0 = expression
            .constant(
                eqiora_core::ValueLiteral::from_real(
                    eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, inverse_time)
                        .expect("numeric scalar type"),
                    0.0,
                )
                .unwrap(),
            )
            .unwrap();
        let equation_zero_1 = expression
            .constant(
                eqiora_core::ValueLiteral::from_real(
                    eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, inverse_time)
                        .expect("numeric scalar type"),
                    0.0,
                )
                .unwrap(),
            )
            .unwrap();
        expression.finish([
            decay_residual,
            equation_zero_0,
            integral_residual,
            equation_zero_1,
        ])
    }
    .unwrap();

    let initial = Id::<kinds::Relation>::new();
    let mut initial_expression = ExprDagBuilder::new();
    let decay_initial = initial_expression.symbol(SymbolRef::Field(decay)).unwrap();
    let decay_value = initial_expression
        .constant(DynQuantity::new(1.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let integral_initial = initial_expression
        .symbol(SymbolRef::Field(integral))
        .unwrap();
    let integral_value = initial_expression
        .constant(DynQuantity::new(0.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let initial_equations = initial_expression
        .finish([decay_initial, decay_value, integral_initial, integral_value])
        .unwrap();
    let nodes = [
        KernelNode::from(RelationDef::initial(initial, initial_equations).unwrap()),
        KernelNode::from(FieldDef::new(
            decay,
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .expect("numeric scalar type"),
            eqiora_schema::kernel::FieldRole::State,
        )),
        KernelNode::from(FieldDef::new(
            integral,
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .expect("numeric scalar type"),
            eqiora_schema::kernel::FieldRole::State,
        )),
        KernelNode::from(ParameterDef::new(
            rate,
            eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, inverse_time)
                    .expect("numeric scalar type"),
                1.0,
            )
            .unwrap(),
        )),
        KernelNode::from(RelationDef::new(relation, residuals).unwrap()),
        KernelNode::from(ActivationDef::continuous(continuous)),
    ];
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("two-state explicit ODE");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for dependency in [decay.erase(), integral.erase(), rate.erase()] {
        transaction.push(Op::Connect {
            from: initial.erase(),
            to: decay.erase(),
            edge: EdgeKind::DependsOn,
        });
        transaction.push(Op::Connect {
            from: initial.erase(),
            to: integral.erase(),
            edge: EdgeKind::DependsOn,
        });
        transaction.push(Op::Connect {
            from: relation.erase(),
            to: dependency,
            edge: EdgeKind::DependsOn,
        });
    }
    transaction
        .push(Op::Connect {
            from: continuous.erase(),
            to: relation.erase(),
            edge: EdgeKind::Activates,
        })
        .push(Op::DefineOntologyView {
            view: ModelView::new(model, members, []).unwrap().into(),
        });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let envelope = ModelEnvelope::from_program(&program).unwrap();
    (envelope, program, decay, integral)
}

#[test]
fn no_mesh_plan_owns_model_initial_state_and_run_only_horizon() {
    let (model, program) = fixture();
    let field = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Field(field) => Some(field.id()),
            _ => None,
        })
        .unwrap();
    let temporal = CommonTsitouras45::new(
        0.01,
        1.0e-9,
        vec![CommonTsitourasTolerance::new((field, 0), 1.0e-11).unwrap()],
    )
    .unwrap();
    let plan = CommonOdePlan::resolve(
        &model,
        &program,
        temporal,
        TimeBackendIdentity::new("eqiora.test.time", "1"),
    )
    .unwrap();
    let resolved = crate::ResolvedCommonPlan::Ode(Box::new(plan.clone()));
    let bytes = resolved.to_bytes().unwrap();
    assert_eq!(
        crate::ResolvedCommonPlan::from_bytes(
            &bytes,
            &REFERENCE_LINEAR_SOLVER,
            TimeBackendIdentity::new("eqiora.test.time", "1"),
        )
        .unwrap(),
        resolved
    );
    let state = plan.initial_state(0.0).unwrap();
    assert_eq!(state.time_s(), 0.0);
    assert_eq!(state.state_coordinates(), &[(field, 0)]);
    assert_eq!(state.values(), &[1.0]);
    let state_bytes = state.to_bytes().unwrap();
    assert_eq!(
        CommonOdeState::from_bytes(&state_bytes, &plan).unwrap(),
        state
    );
    let mut noncanonical_state = state_bytes;
    noncanonical_state.push(b'\n');
    assert!(CommonOdeState::from_bytes(&noncanonical_state, &plan).is_err());

    let request = CommonOdeRunRequest::new(plan.clone(), state.clone(), 0.2, vec![0.1]).unwrap();
    assert_eq!(request.output_times_s(), &[0.1]);
    assert_eq!(request.time_plan().output_times(), &[0.1, 0.2]);
    assert_eq!(request.plan().identity(), plan.identity());
    let output = CommonOdeState::new(&plan, 0.1, vec![0.9], "result").unwrap();
    let history = eqiora_time::AcceptedTimeHistory::accepted(
        1,
        vec![
            eqiora_time::TimeHistoryStep::accepted(0.0, 0.2, vec![1.0], vec![0.9], vec![0.8])
                .unwrap(),
        ],
        vec![],
    )
    .unwrap();
    let trajectory =
        crate::CommonTrajectory::accept_ode_states(request.clone(), vec![output], history).unwrap();
    let trajectory_bytes = trajectory.to_bytes().unwrap();
    assert_eq!(
        crate::CommonTrajectory::from_bytes(&trajectory_bytes, &resolved).unwrap(),
        trajectory
    );
    let result = crate::CommonResult::accept_trajectory(0.25, trajectory.clone()).unwrap();
    let result_bytes = result.to_bytes().unwrap();
    let replayed_result = crate::CommonResult::from_bytes(&result_bytes, &resolved).unwrap();
    assert_eq!(replayed_result, result);
    assert_eq!(replayed_result.to_bytes().unwrap(), result_bytes);
    let mut forged_result: serde_json::Value = serde_json::from_slice(&result_bytes).unwrap();
    forged_result["identity"] = serde_json::Value::String("0".repeat(64));
    assert!(
        crate::CommonResult::from_bytes(&serde_json::to_vec(&forged_result).unwrap(), &resolved,)
            .is_err()
    );
    let mut noncanonical_trajectory = trajectory_bytes;
    noncanonical_trajectory.push(b'\n');
    assert!(crate::CommonTrajectory::from_bytes(&noncanonical_trajectory, &resolved).is_err());
    assert!(CommonOdeRunRequest::new(plan, state, 0.2, vec![0.0]).is_err());
}

#[test]
fn field_tolerances_are_exact_complete_and_positive() {
    let (model, program) = fixture();
    let field = program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Field(field) => Some(field.id()),
            _ => None,
        })
        .unwrap();
    assert!(CommonTsitourasTolerance::new((field, 0), 0.0).is_err());
    assert!(CommonTsitouras45::new(0.01, 1.0e-9, Vec::new()).is_err());
    let foreign = Id::<kinds::Field>::new();
    let temporal = CommonTsitouras45::new(
        0.01,
        1.0e-9,
        vec![CommonTsitourasTolerance::new((foreign, 0), 1.0e-11).unwrap()],
    )
    .unwrap();
    assert!(
        CommonOdePlan::resolve(
            &model,
            &program,
            temporal,
            TimeBackendIdentity::new("eqiora.test.time", "1"),
        )
        .is_err()
    );
}

#[test]
fn field_tolerances_map_to_canonical_first_order_state_coordinates() {
    let (model, program, decay, integral) = two_state_fixture(false);
    let temporal = CommonTsitouras45::new(
        0.01,
        1.0e-9,
        vec![
            CommonTsitourasTolerance::new((integral, 0), 2.0e-11).unwrap(),
            CommonTsitourasTolerance::new((decay, 0), 1.0e-11).unwrap(),
        ],
    )
    .unwrap();
    let plan = CommonOdePlan::resolve(
        &model,
        &program,
        temporal,
        TimeBackendIdentity::new("eqiora.test.time", "1"),
    )
    .unwrap();

    assert_eq!(
        plan.state_coordinates().collect::<Vec<_>>(),
        [(decay, 0), (integral, 0)]
    );
    assert_eq!(plan.ordered_absolute_tolerances, [1.0e-11, 2.0e-11]);
    assert_eq!(
        plan.state_dimensions(),
        [DimExponents::DIMENSIONLESS, DimExponents::DIMENSIONLESS]
    );
    let initial = plan.initial_state(0.0).unwrap();
    assert_eq!(initial.state_coordinates(), [(decay, 0), (integral, 0)]);
    assert_eq!(initial.values(), [1.0, 0.0]);
}

#[test]
fn tsitouras_common_plan_rejects_a_structural_mass_matrix() {
    let (model, program, decay, integral) = two_state_fixture(true);
    let temporal = CommonTsitouras45::new(
        0.01,
        1.0e-9,
        vec![
            CommonTsitourasTolerance::new((decay, 0), 1.0e-11).unwrap(),
            CommonTsitourasTolerance::new((integral, 0), 2.0e-11).unwrap(),
        ],
    )
    .unwrap();
    assert!(
        CommonOdePlan::resolve(
            &model,
            &program,
            temporal,
            TimeBackendIdentity::new("eqiora.test.time", "1"),
        )
        .is_err()
    );
}
