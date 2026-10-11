use super::*;
use crate::form_compiler::equation_roles::Role;
use crate::form_compiler::region::{RegionFieldBinding, RegionTimeBinding};
use eqiora_compiler::compile;
use eqiora_core::DynQuantity;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::{ReferenceCell, simplex_duffy_gauss_legendre};
use eqiora_realization::{
    BackwardEulerStateBinding, BackwardEulerStatePair, PositivePhysicalScale, Space,
};
use std::num::NonZeroU16;

fn fixture(source: &str, bubble: bool, scale: f64) -> (KernelProgram, BoundRegionForm<f64>) {
    let (transaction, model, _) = compile("balance.eqi", source)
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
    let form = super::super::CompiledRegionForm::derive(&program, domain, 2).unwrap();
    let p1 = Space::continuous_lagrange(NonZeroU16::MIN);
    let fields = form
        .fields()
        .map(|(field, ty)| RegionFieldBinding {
            field,
            space: if bubble && !ty.shape().is_scalar() {
                Space::simplex_p1_bubble()
            } else {
                p1
            },
            scale: DynQuantity::new(scale, ty.dimension()),
        })
        .collect::<Vec<_>>();
    let area = DimExponents::from_integers([0, 2, 0, 0, 0, 0, 0]).unwrap();
    let rows = form
        .rows()
        .map(|(relation, _, ty)| {
            (
                relation,
                DynQuantity::new(
                    1. / scale,
                    ty.dimension().mul(area).unwrap().pow(-1, 1).unwrap(),
                ),
            )
        })
        .collect();
    let states = form
        .roles
        .relations
        .iter()
        .filter_map(|(&relation, role)| match role.kind {
            Role::Kinematic { state, rate } => Some(BackwardEulerStateBinding::new(
                BackwardEulerStatePair::new(
                    relation.downcast().unwrap(),
                    state.downcast().unwrap(),
                    rate.downcast().unwrap(),
                )
                .unwrap(),
                p1,
                PositivePhysicalScale::new(DynQuantity::new(
                    7.,
                    form.roles.fields[&state].1.dimension(),
                ))
                .unwrap(),
            )),
            _ => None,
        })
        .collect();
    let time = RegionTimeBinding {
        step: DynQuantity::new(
            0.25,
            DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap(),
        ),
        states,
    };
    let bound = form
        .bind(
            ReferenceCell::simplex(2).unwrap(),
            &fields,
            &rows,
            Some(&time),
        )
        .unwrap();
    (program, bound)
}
fn geometry() -> AffineGeometryMap {
    AffineGeometryMap::new(
        ReferenceCell::simplex(2).unwrap(),
        2,
        vec![0., 0.],
        vec![1., 0., 0., 1.],
    )
    .unwrap()
}
fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 2e-11, "{actual} != {expected}");
}
fn sums(balance: &RegionStepBalance, role: BalanceRole) -> [f64; 4] {
    let mut result = [0.; 4];
    for term in balance
        .contributions
        .iter()
        .filter(|term| term.role == role)
    {
        for (sum, value) in
            result
                .iter_mut()
                .zip([term.previous, term.current, term.increment, term.work])
        {
            *sum += value;
        }
    }
    result
}
const ELASTIC: &str = "model Elastic() {
 domain body=box(0,1,0,1);
 state v: vector<m/s,2> on body;
 state d: vector<m,2> on body in h1;
 relation kinematics on body { derivative(d)-v=0; }
 relation balance on body { 3[kg/m^3]*derivative(v)-div(4[kg/(m*s^2)]*symmetric_part(grad(d))+5[kg/(m*s^2)]*isotropic_lift(div(d)))=0; }
}";
#[test]
fn physical_affine_elastic_balance_ignores_numerical_scales() {
    let mut comparison = None;
    for scale in [1., 9.] {
        let (program, form) = fixture(ELASTIC, false, scale);
        let (&state, &rate) = form.eliminations.iter().next().unwrap();
        // v=(x,0), old d=(x,0), new d=1.25(x,0), dt=1/4.
        let previous = BTreeMap::from([(rate, vec![0.; 6]), (state, vec![0., 0., 1., 0., 0., 0.])]);
        let current = BTreeMap::from([
            (rate, vec![0., 0., 1., 0., 0., 0.]),
            (state, vec![0., 0., 1.25, 0., 0., 0.]),
        ]);
        let balance = form
            .volume_step_balance(
                &program,
                &geometry(),
                &simplex_duffy_gauss_legendre(2, 4).unwrap(),
                &previous,
                &current,
            )
            .unwrap();
        // ∫triangle x²=1/12. Elastic density = (mu+lambda/2)*a²=4.5*a².
        let kinetic = sums(&balance, BalanceRole::Storage);
        close(kinetic[1], 1. / 8.);
        close(kinetic[2], 1. / 8.);
        let elastic = sums(&balance, BalanceRole::StatePotential);
        close(elastic[0], 2.25);
        close(elastic[1], 2.25 * 1.25_f64.powi(2));
        close(elastic[2], 2.25 * 0.25_f64.powi(2));
        let units = DimExponents::from_integers([1, 1, -2, 0, 0, 0, 0]).unwrap();
        assert!(
            balance
                .contributions
                .iter()
                .all(|term| term.dimension == units)
        );
        close(
            balance
                .volume_work()
                .unwrap()
                .iter()
                .find(|(dimension, _)| *dimension == units)
                .unwrap()
                .1,
            1. / 4. + 2.25 * (1.25_f64.powi(2) - 1. + 0.25_f64.powi(2)),
        );
        if let Some(expected) = comparison {
            assert_eq!((kinetic, elastic), expected);
        } else {
            comparison = Some((kinetic, elastic));
        }
    }
}
const MIXED: &str = "model Mixed() {
 domain body=box(0,1,0,1);
 state v: vector<m/s,2> on body in smooth;
 variable p: kg/(m*s^2) on body in h1;
 relation balance on body { 3[kg/m^3]*derivative(v)-div(4[kg/(m*s)]*symmetric_part(grad(v))-isotropic_lift(p))=0; }
 relation constraint on body { div(v)=0; }
}";
#[test]
fn mini_bubble_kinetic_and_affine_viscous_integrals() {
    let (program, form) = fixture(MIXED, true, 3.);
    let (velocity, pressure) = form.form.constraint_pairs().next().unwrap();
    let previous = BTreeMap::from([(velocity, vec![0.; 8]), (pressure, vec![0.; 3])]);
    let mut current = previous.clone();
    // The normalized MINI bubble is 27*x*y*(1-x-y). ∫b²=81/560.
    current.get_mut(&velocity).unwrap()[6] = 1.;
    let rule = simplex_duffy_gauss_legendre(2, 4).unwrap();
    let balance = form
        .volume_step_balance(&program, &geometry(), &rule, &previous, &current)
        .unwrap();
    close(sums(&balance, BalanceRole::Storage)[1], 243. / 1120.);
    // For v=(x,0), ∫2mu eps:eps=4*area=2, dt=1/4.
    current.insert(velocity, vec![0., 0., 1., 0., 0., 0., 0., 0.]);
    let affine = form
        .volume_step_balance(&program, &geometry(), &rule, &previous, &current)
        .unwrap();
    close(sums(&affine, BalanceRole::Dissipation)[3], 0.5);
}
#[test]
fn rejects_missing_nonfinite_history_and_inconsistent_kinematics() {
    let (program, mut form) = fixture(ELASTIC, false, 1.);
    let (&state, &rate) = form.eliminations.iter().next().unwrap();
    let mut values = BTreeMap::from([(state, vec![0.; 6]), (rate, vec![0.; 6])]);
    let rule = simplex_duffy_gauss_legendre(2, 4).unwrap();
    assert!(
        form.volume_step_balance(&program, &geometry(), &rule, &BTreeMap::new(), &values)
            .unwrap_err()
            .message()
            .contains("coverage")
    );
    values.get_mut(&rate).unwrap()[0] = f64::NAN;
    assert!(
        form.volume_step_balance(&program, &geometry(), &rule, &values, &values)
            .unwrap_err()
            .message()
            .contains("nonfinite")
    );
    values.get_mut(&rate).unwrap()[0] = 0.;
    let mut inconsistent = values.clone();
    // A stationary rate cannot produce a nonzero displacement increment.
    inconsistent.get_mut(&state).unwrap()[2] = 1.;
    assert!(
        form.volume_step_balance(&program, &geometry(), &rule, &values, &inconsistent)
            .unwrap_err()
            .message()
            .contains("kinematics")
    );
    form.time = None;
    assert!(
        form.volume_step_balance(&program, &geometry(), &rule, &values, &values)
            .unwrap_err()
            .message()
            .contains("BackwardEuler")
    );
}

const SCALAR: &str = "model Pair() {
 domain body=box(0,1,0,1);
 state a: 1 on body in h1;
 state b: 1 on body in h1;
 relation first on body { 1[s/m^2]*derivative(a)-div(grad(a))=0; }
 relation second on body { 1[s/m^2]*derivative(b)-div(grad(b))=0; }
}";
#[test]
fn reciprocal_storage_retains_cross_terms_and_signed_quadratic_identity() {
    let (program, mut form) = fixture(SCALAR, false, 1.);
    let a = form.form.rows[0].tested;
    let b = form.form.rows[1].tested;
    // Exercise the private evaluator's general reciprocal term inventory.
    // Public scalar admission currently restricts storage to diagonal capacity.
    for (index, (trial, diagonal)) in [(b, 2.), (a, 3.)].into_iter().enumerate() {
        let mass = form.form.rows[index]
            .terms
            .iter_mut()
            .find(|term| term.derivative)
            .unwrap();
        let mut cross = mass.clone();
        mass.coefficient = super::super::Data::constant(2, diagonal);
        cross.trial = trial;
        cross.coefficient = super::super::Data::constant(2, 0.5);
        form.form.rows[index].terms.push(cross);
    }
    let previous = BTreeMap::from([(a, vec![1.; 3]), (b, vec![2.; 3])]);
    let current = BTreeMap::from([(a, vec![3.; 3]), (b, vec![4.; 3])]);
    let rule = simplex_duffy_gauss_legendre(2, 4).unwrap();
    let balance = form
        .volume_step_balance(&program, &geometry(), &rule, &previous, &current)
        .unwrap();
    // Area=1/2, M=[[2,1/2],[1/2,3]], E(q)=1/4 q^T M q.
    let mass = sums(&balance, BalanceRole::Storage);
    close(mass[0], 4.);
    close(mass[1], 19.5);
    close(mass[2], 6.);
    close(balance.volume_work().unwrap()[0].1, 21.5);
    let mut missing = form.clone();
    missing.form.rows[1].terms.pop();
    assert!(
        missing
            .volume_step_balance(&program, &geometry(), &rule, &previous, &current)
            .unwrap_err()
            .message()
            .contains("reciprocal")
    );
    let mut mismatch = form.clone();
    mismatch.form.rows[1].terms.last_mut().unwrap().derivative = false;
    assert!(
        mismatch
            .volume_step_balance(&program, &geometry(), &rule, &previous, &current)
            .unwrap_err()
            .message()
            .contains("reciprocal")
    );
    // Symmetry is not positivity: an indefinite storage block remains signed.
    for row in &mut form.form.rows {
        for term in &mut row.terms {
            if term.derivative {
                term.coefficient = term
                    .coefficient
                    .clone()
                    .multiply(super::super::Data::constant(2, -1.));
            }
        }
    }
    let negative = form
        .volume_step_balance(&program, &geometry(), &rule, &previous, &current)
        .unwrap();
    close(sums(&negative, BalanceRole::Storage)[1], -19.5);
}
