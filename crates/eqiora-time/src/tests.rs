use super::*;
use eqiora_core::TimeStateCoordinate;
use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};

struct ScalarDecay;

impl TimeSystem for ScalarDecay {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(&self, _time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        output[0] = -state[0];
        Ok(())
    }

    fn rhs_jvp(
        &self,
        _time: f64,
        _state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = -direction[0];
        Ok(())
    }
}

struct ProvidedIndexOnePair;

impl ImplicitTimeSystem for ProvidedIndexOnePair {
    fn dimension(&self) -> usize {
        2
    }

    fn residual(
        &self,
        _time: f64,
        state: &[f64],
        derivative: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = derivative[0] + state[0];
        output[1] = state[1] - state[0];
        Ok(())
    }

    fn residual_jvp(
        &self,
        _time: f64,
        _state: &[f64],
        _derivative: &[f64],
        state_direction: &[f64],
        derivative_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = derivative_direction[0] + state_direction[0];
        output[1] = state_direction[1] - state_direction[0];
        Ok(())
    }
}

#[test]
fn residual_native_problem_distinguishes_provided_pair_from_consistency_guess() {
    let problem = ImplicitDaeProblem::new(
        &ProvidedIndexOnePair,
        vec![DaeVariableKind::Differential, DaeVariableKind::Algebraic],
        InitialConditionPolicy::Provided,
        vec![1.0, 1.0],
        vec![-1.0, 0.0],
    )
    .unwrap();
    let plan = TimePlan::new(
        TimeMethod::ImplicitEuler,
        0.0,
        0.1,
        1.0e-8,
        vec![1.0e-10; 2],
        vec![0.1],
    )
    .unwrap();
    let initialized = ReferenceImplicitTimeBackend::new()
        .initialize(&problem, &plan)
        .unwrap();
    assert_eq!(initialized.state(), [1.0, 1.0]);
    assert_eq!(initialized.derivative(), [-1.0, 0.0]);

    let explicit_plan = TimePlan::new(
        TimeMethod::Tsitouras45,
        0.0,
        0.1,
        1.0e-8,
        vec![1.0e-10; 2],
        vec![0.1],
    )
    .unwrap();
    assert_eq!(
        ReferenceImplicitTimeBackend::new()
            .initialize(&problem, &explicit_plan)
            .unwrap_err()
            .code(),
        codes::INVALID_EXECUTION_CONFIG
    );
}

#[test]
fn lowering_proof_derives_equation_class_from_exact_matrix() {
    let relation = Id::<kinds::Relation>::new();
    let differential = Id::<kinds::Field>::new();
    let algebraic = Id::<kinds::Field>::new();
    let matrix = ConstantDerivativeMatrixProof::new(2, vec![-2.0, 0.0, 0.0, 0.0]).unwrap();
    let proof = TimeLoweringProof::new(
        relation,
        vec![
            TimeStateCoordinate::new(differential, 0, 0, false),
            TimeStateCoordinate::new(algebraic, 0, 0, false),
        ],
        matrix,
    )
    .unwrap();

    assert_eq!(proof.relation(), relation);
    assert_eq!(
        proof.state_coordinates(),
        [
            TimeStateCoordinate::new(differential, 0, 0, false),
            TimeStateCoordinate::new(algebraic, 0, 0, false)
        ]
    );
    assert_eq!(
        proof.equation_class(),
        TimeEquationClass::MassMatrix {
            rank: MassMatrixRank::RankDeficient
        }
    );
    assert_eq!(
        proof.initial_condition_policy(),
        InitialConditionPolicy::SolveConsistent
    );
}

#[test]
fn exact_rank_separates_ill_conditioned_full_and_singular_dense_matrices() {
    let relation = Id::<kinds::Relation>::new();
    let first = Id::<kinds::Field>::new();
    let second = Id::<kinds::Field>::new();
    let full =
        ConstantDerivativeMatrixProof::new(2, vec![1.0, 1.0, 1.0, 1.0 + f64::EPSILON]).unwrap();
    assert_eq!(full.exact_rank(), 2);
    assert_eq!(
        TimeLoweringProof::new(
            relation,
            vec![
                TimeStateCoordinate::new(first, 0, 0, false),
                TimeStateCoordinate::new(second, 0, 0, false)
            ],
            full
        )
        .unwrap()
        .equation_class(),
        TimeEquationClass::MassMatrix {
            rank: MassMatrixRank::Full
        }
    );

    let singular = ConstantDerivativeMatrixProof::new(2, vec![1.0, 1.0, 2.0, 2.0]).unwrap();
    assert_eq!(singular.exact_rank(), 1);
    assert_eq!(
        TimeLoweringProof::new(
            relation,
            vec![
                TimeStateCoordinate::new(first, 0, 0, false),
                TimeStateCoordinate::new(second, 0, 0, false)
            ],
            singular
        )
        .unwrap()
        .equation_class(),
        TimeEquationClass::MassMatrix {
            rank: MassMatrixRank::RankDeficient
        }
    );

    let zero = ConstantDerivativeMatrixProof::new(2, vec![0.0; 4]).unwrap();
    assert_eq!(
        TimeLoweringProof::new(
            relation,
            vec![
                TimeStateCoordinate::new(first, 0, 0, false),
                TimeStateCoordinate::new(second, 0, 0, false)
            ],
            zero
        )
        .unwrap_err()
        .code(),
        codes::INVALID_TIME_LOWERING
    );
}

#[test]
fn general_residual_cannot_be_smuggled_through_mass_matrix_contract() {
    let error = TimeProblem::new(
        &ScalarDecay,
        TimeEquationClass::GeneralImplicitDae,
        InitialConditionPolicy::Provided,
        vec![1.0],
    )
    .unwrap_err();
    assert_eq!(error.code(), codes::INVALID_TIME_LOWERING);
}

#[test]
fn rank_deficient_mass_matrix_requires_consistency_policy() {
    let error = TimeProblem::new(
        &ScalarDecay,
        TimeEquationClass::MassMatrix {
            rank: MassMatrixRank::RankDeficient,
        },
        InitialConditionPolicy::Provided,
        vec![1.0],
    )
    .unwrap_err();
    assert!(error.message().contains("consistent"));
}

#[test]
fn plan_separates_adaptive_step_from_requested_samples() {
    let problem = TimeProblem::new(
        &ScalarDecay,
        TimeEquationClass::ExplicitOde,
        InitialConditionPolicy::Provided,
        vec![1.0],
    )
    .unwrap();
    let plan = TimePlan::new(
        TimeMethod::Tsitouras45,
        0.0,
        1.0e-3,
        1.0e-7,
        vec![1.0e-9],
        vec![0.25, 1.0],
    )
    .unwrap();
    plan.validate_for(&problem).unwrap();
    assert_eq!(plan.output_times(), [0.25, 1.0]);
}

#[test]
fn transversal_event_composes_time_reset_and_saltation_derivatives() {
    let gravity = 9.81;
    let restitution = 0.8;
    let impact_time = (2.0_f64 / gravity).sqrt();
    let impact_velocity = -gravity * impact_time;
    let flow = EventFlowLinearization::new(
        vec![impact_velocity, -gravity],
        vec![-restitution * impact_velocity, -gravity],
    )
    .unwrap();
    let guard = EventGuardLinearization::new(vec![1.0, 0.0], vec![0.0, 0.0], 0.0).unwrap();
    let reset = EventResetLinearization::new(
        2,
        2,
        vec![0.0, 0.0, 0.0, -restitution],
        vec![0.0, 0.0, 0.0, -impact_velocity],
        vec![0.0, 0.0],
    )
    .unwrap();
    let event = TransversalEventLinearization::new(flow, guard, reset).unwrap();

    assert_close(event.transversality(), impact_velocity);
    assert_close(event.saltation_matrix()[0], -restitution);
    assert_close(event.saltation_matrix()[1], 0.0);
    assert_close(
        event.saltation_matrix()[2],
        -(1.0 + restitution) * gravity / impact_velocity,
    );
    assert_close(event.saltation_matrix()[3], -restitution);

    let pre_sensitivity = vec![-0.5 * impact_time.powi(2), 0.0, -impact_time, 0.0];
    let propagated = event.propagate_forward(&pre_sensitivity).unwrap();
    assert_close(propagated.event_time()[0], -impact_time / (2.0 * gravity));
    assert_close(propagated.event_time()[1], 0.0);
    assert_close(propagated.post_state()[0], restitution / gravity);
    assert_close(propagated.post_state()[1], 0.0);
    assert_close(
        propagated.post_state()[2],
        (restitution - 1.0) * impact_time / 2.0,
    );
    assert_close(propagated.post_state()[3], -impact_velocity);
}

#[test]
fn grazing_event_fails_closed() {
    let flow = EventFlowLinearization::new(vec![0.0], vec![1.0]).unwrap();
    let guard = EventGuardLinearization::new(vec![1.0], Vec::new(), 0.0).unwrap();
    let reset = EventResetLinearization::new(1, 0, vec![1.0], Vec::new(), vec![0.0]).unwrap();
    let error = TransversalEventLinearization::new(flow, guard, reset).unwrap_err();
    assert_eq!(error.code(), codes::INVALID_LINEARIZATION);
    assert!(error.message().contains("grazing"));
}

struct TwoRoots;

impl RootFunctions for TwoRoots {
    fn count(&self) -> usize {
        2
    }

    fn evaluate(&self, _time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        output.copy_from_slice(&state[..2]);
        Ok(())
    }
}

#[test]
fn root_registration_canonicalizes_groups_and_binds_callback_shape() {
    let first = Id::<kinds::Activation>::new();
    let second = Id::<kinds::Activation>::new();
    let third = Id::<kinds::Activation>::new();
    let proof = RootRegistrationProof::new(vec![
        RootActivationGroup::new(vec![third]).unwrap(),
        RootActivationGroup::new(vec![second, first]).unwrap(),
    ])
    .unwrap();

    assert_eq!(proof.root_count(), 2);
    assert!(
        proof.groups()[0].representative().erase() < proof.groups()[1].representative().erase()
    );
    let registration = RootRegistrationId::from_sha256([11; 32]);
    let registered = RegisteredRootProblem::new(registration, proof.clone(), &TwoRoots).unwrap();
    assert_eq!(registered.registration(), registration);
    assert_eq!(registered.proof(), &proof);
}

#[test]
fn root_registration_rejects_overlap_and_callback_count_mismatch() {
    let activation = Id::<kinds::Activation>::new();
    let group = RootActivationGroup::new(vec![activation]).unwrap();
    assert_eq!(
        RootRegistrationProof::new(vec![group.clone(), group])
            .unwrap_err()
            .code(),
        codes::INVALID_TIME_LOWERING
    );

    let proof =
        RootRegistrationProof::new(vec![RootActivationGroup::new(vec![activation]).unwrap()])
            .unwrap();
    assert_eq!(
        RegisteredRootProblem::new(RootRegistrationId::from_sha256([13; 32]), proof, &TwoRoots,)
            .unwrap_err()
            .code(),
        codes::INVALID_TIME_LOWERING
    );
}

#[test]
fn backend_identity_is_an_atomic_validated_token_pair() {
    let identity = TimeBackendIdentity::new("eqiora.time.reference", "1.2.3-rc.1");
    assert_eq!(identity.id(), "eqiora.time.reference");
    assert_eq!(identity.version(), "1.2.3-rc.1");

    assert!(std::panic::catch_unwind(|| TimeBackendIdentity::new("Eqiora Time", "1.0.0")).is_err());
    assert!(
        std::panic::catch_unwind(|| TimeBackendIdentity::new("eqiora.time", "1.0 beta")).is_err()
    );
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-12,
        "{actual} != {expected}"
    );
}

#[test]
fn local_constant_mass_regularity_distinguishes_hidden_and_coupled_constraints() {
    let hidden = ConstantDerivativeMatrixProof::new(2, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
    let error = hidden
        .require_index_one_regularity(&[0.0, -1.0, 1.0, 0.0])
        .unwrap_err();
    assert!(error.message().contains("rank 2, required 3"));
    for (mass, state) in [
        (vec![1.0, 1.0, 0.0, 0.0], vec![2.0, 0.0, 1.0, -1.0]),
        (vec![1.0, 1.0, 1.0, 1.0], vec![2.0, 0.0, 0.0, 2.0]),
    ] {
        ConstantDerivativeMatrixProof::new(2, mass)
            .unwrap()
            .require_index_one_regularity(&state)
            .unwrap();
    }
    assert!(hidden.require_index_one_regularity(&[1.0]).is_err());
    assert!(hidden.require_index_one_regularity(&[f64::NAN; 4]).is_err());
}

fn exact_column(values: &[f64]) -> Vec<num_rational::BigRational> {
    values
        .iter()
        .map(|value| num_rational::BigRational::from_float(*value).unwrap())
        .collect()
}

#[test]
fn compatibility_projection_is_row_invariant_and_avoids_a_bad_basic_witness() {
    let epsilon = 2.0_f64.powi(-40);
    let delta = 2.0_f64.powi(-30);
    let mass = ConstantDerivativeMatrixProof::new(2, vec![epsilon, 0.0, 1.0, 0.0]).unwrap();
    let rhs = exact_column(&[epsilon + delta, 1.0]);
    let residual = mass.compatibility_residual(&[], &rhs).unwrap();
    // Exact least-squares residual: [delta, -epsilon*delta]/(1+epsilon²).
    assert_eq!(
        residual,
        vec![
            delta / (1.0 + epsilon * epsilon),
            -epsilon * delta / (1.0 + epsilon * epsilon)
        ]
    );
    let permuted = ConstantDerivativeMatrixProof::new(2, vec![1.0, 0.0, epsilon, 0.0]).unwrap();
    assert_eq!(
        permuted
            .compatibility_residual(&[], &exact_column(&[1.0, epsilon + delta]))
            .unwrap(),
        vec![residual[1], residual[0]]
    );
    assert_eq!(
        mass.compatibility_residual(&[exact_column(&[1.0, 0.0])], &rhs)
            .unwrap(),
        vec![0.0, 0.0]
    );
}

#[test]
fn compatibility_projection_handles_zero_rank_and_rejects_invalid_inputs() {
    let mass = ConstantDerivativeMatrixProof::new(2, vec![0.0; 4]).unwrap();
    assert_eq!(
        mass.compatibility_residual(&[], &exact_column(&[2.0, -3.0]))
            .unwrap(),
        vec![2.0, -3.0]
    );
    assert!(
        mass.compatibility_residual(&[], &exact_column(&[1.0]))
            .is_err()
    );
    assert!(
        mass.compatibility_residual(&[exact_column(&[1.0])], &exact_column(&[1.0, 2.0]))
            .is_err()
    );
    assert!(
        mass.compatibility_residual(
            &[vec![
                num_rational::BigRational::new_raw(1.into(), 0.into());
                2
            ]],
            &exact_column(&[1.0, 2.0])
        )
        .is_err()
    );
}

#[test]
fn exact_compatibility_does_not_confuse_underflow_with_column_membership() {
    let epsilon = f64::MIN_POSITIVE;
    let tiny = f64::from_bits(1);
    let mass = ConstantDerivativeMatrixProof::new(2, vec![1.0, 0.0, epsilon, 0.0]).unwrap();
    // Non-proportional columns: both exact residual components round to zero.
    let column = exact_column(&[tiny, 0.0]);
    assert!(!mass.is_compatible(&[], &column).unwrap());
    assert_eq!(
        mass.compatibility_residual(&[], &column).unwrap(),
        vec![0.0, 0.0]
    );
    assert!(
        mass.is_compatible(&[], &exact_column(&[1.0, epsilon]))
            .unwrap()
    );
}

#[test]
fn exact_matrix_proof_preserves_products_and_rejects_invalid_raw_rationals() {
    use num_rational::BigRational;
    let tenth = BigRational::from_float(0.1).unwrap();
    let scale = &tenth * &tenth;
    let mass = ConstantDerivativeMatrixProof::from_exact(
        2,
        vec![
            BigRational::from_integer(1.into()),
            BigRational::from_integer(0.into()),
            scale.clone(),
            BigRational::from_integer(0.into()),
        ],
    )
    .unwrap();
    assert_eq!(mass.exact_rank(), 1);
    assert!(
        mass.is_compatible(&[], &[BigRational::from_integer(1.into()), scale])
            .unwrap()
    );
    let invalid = BigRational::new_raw(1.into(), 0.into());
    assert!(ConstantDerivativeMatrixProof::from_exact(1, vec![invalid.clone()]).is_err());
    let full = ConstantDerivativeMatrixProof::new(1, vec![1.0]).unwrap();
    assert!(
        full.compatibility_residual(&[], std::slice::from_ref(&invalid))
            .is_err()
    );
    assert!(full.is_compatible(&[], &[invalid]).is_err());
}

#[test]
fn time_state_coordinates_require_complete_channels_parts_and_derivative_chains() {
    let field = Id::<kinds::Field>::new();
    let relation = Id::<kinds::Relation>::new();
    let coordinates = (0..2)
        .flat_map(|order| {
            (0..6).flat_map(move |component| {
                [false, true]
                    .map(|imaginary| TimeStateCoordinate::new(field, order, component, imaginary))
            })
        })
        .collect::<Vec<_>>();
    let matrix = |n: usize| {
        ConstantDerivativeMatrixProof::new(
            n,
            (0..n * n)
                .map(|index| f64::from(index / n == index % n))
                .collect(),
        )
        .unwrap()
    };
    let proof = TimeLoweringProof::new(relation, coordinates.clone(), matrix(24)).unwrap();
    assert_eq!(proof.state_coordinates(), coordinates);
    assert_eq!(proof.state_coordinates()[23].field(), field);
    assert_eq!(proof.state_coordinates()[23].derivative_order(), 1);
    assert_eq!(proof.state_coordinates()[23].component(), 5);
    assert!(proof.state_coordinates()[23].is_imaginary());
    for remove in [0, 1, 12, 23] {
        let mut incomplete = coordinates.clone();
        incomplete.remove(remove);
        assert!(TimeLoweringProof::new(relation, incomplete, matrix(23)).is_err());
    }
    let mut repeated = coordinates.clone();
    repeated[23] = repeated[0];
    assert!(TimeLoweringProof::new(relation, repeated, matrix(24)).is_err());
    for impossible in [
        TimeStateCoordinate::new(field, u32::MAX, 0, false),
        TimeStateCoordinate::new(field, 0, usize::MAX, false),
        TimeStateCoordinate::new(field, 0, 0, true),
    ] {
        assert!(TimeLoweringProof::new(relation, vec![impossible], matrix(1)).is_err());
    }
}
