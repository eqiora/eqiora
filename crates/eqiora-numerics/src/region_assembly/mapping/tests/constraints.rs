//! Original-action assessment distinguishes a rejected finite point from invalid data.
use super::*;
use crate::form_compiler::region::BoundRegionForm;

pub(super) fn check_stationary(
    mesh: &CartesianMesh,
    mapping: &RegionDofMap<f64>,
    forms: &[(BoundRegionForm<f64>, QuadratureRule)],
    length: usize,
) {
    let input = || RegionSolveInput {
        operator_properties: LinearOperatorProperties::General,
        forms: forms.to_vec(),
        conservative: None,
        natural: vec![],
        previous: None,
        prescribed_states: BTreeMap::new(),
        geometry_action: None,
    };
    let temporal = input().temporal_structure().unwrap();
    let solver = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap();
    let inventory = mapping.fields.keys().copied().collect::<Vec<_>>();
    // -2u''=0, u(0)=0, u(L)=1 gives u=x/L. Global vertex i has x=i/2;
    // the fixture's physical Field scale 2 gives free coordinates i/(4L).
    let mut point = vec![0.; mapping.free_count()];
    for key in mapping.keys() {
        if let Some(dof) = mapping.free_dof(key) {
            point[dof.index()] = key.entity.index() as f64 / (4 * length) as f64;
        }
    }
    let assess = |fields: &BTreeMap<RawId, RecoveredRegionField<f64>>| {
        mapping.assess_temporal_state(mesh, input(), fields, &temporal, solver, |_| {
            panic!("regular stationary algebraic block needs no initial-rate partial")
        })
    };
    let validate = |fields: &BTreeMap<RawId, RecoveredRegionField<f64>>| {
        mapping.validate_temporal_state(mesh, input(), fields, &temporal, solver, |_| {
            panic!("regular stationary algebraic block needs no initial-rate partial")
        })
    };
    let exact = mapping.recover(&point, &inventory).unwrap();
    let accepted = assess(&exact).unwrap();
    assert!(accepted.accepted());
    accepted.require_accepted().unwrap();
    validate(&exact).unwrap();

    point[0] += 0.1;
    let finite = mapping.recover(&point, &inventory).unwrap();
    let rejected = assess(&finite).expect("a valid finite candidate remains inspectable");
    assert!(!rejected.accepted());
    let diagnostic = rejected.require_accepted().unwrap_err();
    assert!(
        diagnostic
            .message()
            .contains("spatial State algebraic residual")
    );
    let validator = validate(&finite).unwrap_err();
    assert_eq!(validator.code(), diagnostic.code());
    assert_eq!(validator.message(), diagnostic.message());

    let mut invalid = exact;
    let essential = mapping
        .keys()
        .find(|key| mapping.free_dof(*key).is_none())
        .unwrap();
    *invalid
        .get_mut(&essential.field)
        .unwrap()
        .coefficients
        .get_mut(&essential)
        .unwrap() += 1.;
    assert!(
        assess(&invalid).is_err(),
        "invalid essential traces remain errors"
    );
    let free = mapping
        .keys()
        .find(|key| mapping.free_dof(*key).is_some())
        .unwrap();
    let mut nonfinite = finite;
    *nonfinite
        .get_mut(&free.field)
        .unwrap()
        .coefficients
        .get_mut(&free)
        .unwrap() = f64::NAN;
    assert!(
        assess(&nonfinite).is_err(),
        "non-finite evaluations remain errors"
    );
}
