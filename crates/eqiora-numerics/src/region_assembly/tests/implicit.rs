//! Independent two-element weak integrals, including explicit boundary rates.
use super::*;
use crate::constrained_dofs::ConstrainedDofLayout;
use crate::form_compiler::region::coordinates::Real;
use crate::form_compiler::region::{CompiledRegionForm, FieldPullback, RegionTimeBinding};
use eqiora_assembly::AssemblyTarget;
use eqiora_compiler::compile;
use eqiora_core::{DimExponents, DynQuantity};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::ReferenceCell;
use eqiora_realization::Space;
use eqiora_sem::KernelProgram;
use num_complex::Complex64 as C;
use std::num::NonZeroU16;

fn fixture<S: Coefficient + Send + Sync>(
    source: &str,
    fixed: Vec<Option<f64>>,
    step: Option<f64>,
) -> (AssemblyPlan, PreparedRegionAssembly<S, Real>) {
    let (transaction, model, symbols) = compile("implicit-region.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let domain = symbols.get("body").unwrap();
    let form = CompiledRegionForm::<S>::derive(&program, domain, 1).unwrap();
    let fields = form
        .fields()
        .map(
            |(field, value_type)| crate::form_compiler::region::RegionFieldBinding {
                field,
                space: Space::continuous_lagrange(NonZeroU16::new(1).unwrap()),
                scale: DynQuantity::new(3., value_type.dimension()),
            },
        )
        .collect::<Vec<_>>();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let rows = form
        .rows()
        .map(|(relation, _, value_type)| {
            (
                relation,
                DynQuantity::new(
                    1.,
                    value_type
                        .dimension()
                        .mul(length)
                        .unwrap()
                        .pow(-1, 1)
                        .unwrap(),
                ),
            )
        })
        .collect();
    let reference = ReferenceCell::hypercube(1).unwrap();
    let time = step.map(|step| RegionTimeBinding {
        step: DynQuantity::new(
            step,
            DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap(),
        ),
        states: Vec::new(),
    });
    let bound = form.bind(reference, &fields, &rows, time.as_ref()).unwrap();
    let rule =
        QuadratureRule::tensor_product_gauss_legendre(1, bound.minimum_gauss_points()).unwrap();
    let width = fixed.len() / 3;
    let constraints = ConstrainedDofLayout::new(fixed).unwrap();
    let plan = AssemblyPlan::new(vec![
        AssemblyTarget::new(2 * width).unwrap(),
        AssemblyTarget::new(3 * width).unwrap(),
    ])
    .unwrap();
    let cells = (0..2)
        .map(|half| {
            let globals = (half * width..(half + 2) * width).collect::<Vec<_>>();
            RegionAssemblyCell {
                index: half,
                orientation: vec![1; 2 * width],
                geometry: AffineGeometryMap::new(
                    reference,
                    1,
                    vec![0.25 + 0.5 * half as f64],
                    vec![0.25],
                )
                .unwrap(),
                previous_geometry: None,
                previous: if step.is_some() {
                    BTreeMap::from([(fields[0].field, vec![0.; 2 * width])])
                } else {
                    BTreeMap::new()
                },
                mappings: vec![
                    TargetAssemblyMap::new(
                        plan.target_id(0).unwrap(),
                        constraints.reduced_map(&globals).unwrap(),
                    ),
                    TargetAssemblyMap::new(
                        plan.target_id(1).unwrap(),
                        constraints.full_map(&globals).unwrap(),
                    ),
                ]
                .into(),
            }
        })
        .collect();
    let prepared = PreparedRegionAssembly::<S, Real>::new(
        AssemblyPacketSetIdentityV1::from_sha256([74; 32]),
        &plan,
        vec![(bound, rule)],
        &[domain; 2],
        cells,
        Vec::new(),
    )
    .unwrap();
    (plan, prepared)
}

#[test]
fn independent_global_rates_do_not_reuse_fixed_state_values() {
    let (plan, prepared) = fixture::<f64>(
        "model Storage() {
        domain body=box(0,1); state u:1 on body;
        relation r on body {derivative(u*u)+u*u/1[s]=0;}
    }",
        vec![Some(2.), None, None],
        None,
    );
    let target = plan.target_id(0).unwrap();
    let rate_target = plan.target_id(1).unwrap();
    // u=6 throughout. v is linear from 0 to 4 on the first half, then 4.
    // Exact integration gives R=[13,38,21], with the boundary rate explicitly zero.
    let rates = [0., 4. / 3., 4. / 3.];
    let state_matrix = [7., 4., 0., 4., 19., 5., 0., 5., 10.];
    let rate_matrix = [6., 3., 0., 3., 12., 3., 0., 3., 6.];
    for (current, rate) in [(Some(1.), None), (None, Some(1.)), (Some(2.), Some(-3.))] {
        let point = RegionPoint::Independent {
            state: &[2.; 2],
            rate_target,
            rates: &rates,
            pullback: FieldPullback::Uniform { current, rate },
        };
        let action = prepared
            .linearize(&plan, target, point, &REFERENCE_ASSEMBLY_BACKEND)
            .unwrap();
        for (actual, expected) in action.residuals[1].iter().zip([13., 38., 21.]) {
            close(*actual, expected);
        }
        let full = dense(&action.systems[1]);
        for ((actual, state), rate_entry) in full.iter().zip(state_matrix).zip(rate_matrix) {
            close(
                *actual,
                current.unwrap_or(0.) * state + rate.unwrap_or(0.) * rate_entry,
            );
        }
        let reduced = dense(&action.systems[0]);
        for (actual, index) in reduced.iter().zip([4, 5, 7, 8]) {
            close(*actual, full[index]);
        }
        let primal = prepared.residual(&plan, target, point).unwrap();
        for (actual, expected) in primal.iter().zip([38., 21.]) {
            close(*actual, expected);
        }
        // The affine tangent is anchored at the State point; R is retained separately.
        let (rows, remainder) = reduced.as_chunks::<2>();
        assert!(remainder.is_empty());
        for (row, (entries, rhs)) in rows.iter().zip(action.systems[0].rhs()).enumerate() {
            close(*rhs, 2. * entries.iter().sum::<f64>() - primal[row]);
        }
    }
    // Different coordinates select independent State/rate columns; the fixed
    // boundary has no candidate direction, including in the full-target scatter.
    let weights = [[Some(2.), None], [None, Some(-3.)]];
    let point = RegionPoint::Independent {
        state: &[2.; 2],
        rate_target,
        rates: &rates,
        pullback: FieldPullback::Coordinates(&weights),
    };
    let selected = prepared
        .linearize(&plan, target, point, &REFERENCE_ASSEMBLY_BACKEND)
        .unwrap();
    let full = dense(&selected.systems[1]);
    for i in 0..3 {
        close(full[i * 3], 0.);
        close(full[i * 3 + 1], 2. * state_matrix[i * 3 + 1]);
        close(full[i * 3 + 2], -3. * rate_matrix[i * 3 + 2]);
    }
    for (actual, expected) in selected.residuals[1].iter().zip([13., 38., 21.]) {
        close(*actual, expected);
    }
    for invalid_weights in [&weights[..1], &[[None, None], [Some(f64::NAN), None]]] {
        let point = RegionPoint::Independent {
            pullback: FieldPullback::Coordinates(invalid_weights),
            state: &[2.; 2],
            rate_target,
            rates: &rates,
        };
        assert!(
            prepared
                .linearize(&plan, target, point, &REFERENCE_ASSEMBLY_BACKEND)
                .is_err()
        );
    }
    let invalid = RegionPoint::Independent {
        state: &[2.; 2],
        rate_target: target,
        rates: &[0.; 2],
        pullback: FieldPullback::Uniform {
            current: None,
            rate: Some(1.),
        },
    };
    assert!(
        prepared
            .linearize(&plan, target, invalid, &REFERENCE_ASSEMBLY_BACKEND)
            .err()
            .unwrap()
            .message()
            .contains("explicit boundary rates")
    );
    for rates in [&[0.; 2][..], &[0., f64::NAN, 0.][..]] {
        let point = RegionPoint::Independent {
            state: &[2.; 2],
            rate_target,
            rates,
            pullback: FieldPullback::Uniform {
                current: None,
                rate: Some(1.),
            },
        };
        assert!(prepared.residual(&plan, target, point).is_err());
    }
}

#[test]
fn global_rate_partial_does_not_demand_singular_state_partial() {
    let (plan, prepared) = fixture::<f64>(
        "model Storage() {
        domain body=box(0,1); state u:1 on body;
        relation r on body {math.sqrt(u)+1[s]*derivative(u*u)=0;}
    }",
        vec![Some(0.), None, None],
        None,
    );
    let target = plan.target_id(0).unwrap();
    let point = |current, rate| RegionPoint::Independent {
        state: &[0.; 2],
        rate_target: plan.target_id(1).unwrap(),
        rates: &[1.; 3],
        pullback: FieldPullback::Uniform { current, rate },
    };
    let action = prepared
        .linearize(
            &plan,
            target,
            point(None, Some(1.)),
            &REFERENCE_ASSEMBLY_BACKEND,
        )
        .unwrap();
    assert_eq!(action.residuals[0], vec![0.; 2]);
    assert_eq!(dense(&action.systems[0]), vec![0.; 4]);
    for weight in [0., 1.] {
        assert!(
            prepared
                .linearize(
                    &plan,
                    target,
                    point(Some(weight), None),
                    &REFERENCE_ASSEMBLY_BACKEND
                )
                .is_err()
        );
    }
}

#[test]
fn complex_global_rate_coordinates_preserve_conjugation_and_constraints() {
    for conjugate in [false, true] {
        let rate = if conjugate {
            "derivative(math.conj(u))"
        } else {
            "derivative(u)"
        };
        let source=format!("model Storage() {{
            domain body=box(0,1); state u:complex<1> on body;
            relation r on body {{math.complex(2,3)*1[s]*{rate}+math.complex(5,-7)*u=math.complex(11,13);}}
        }}");
        let (plan, prepared) = fixture::<C>(
            &source,
            vec![Some(2. / 3.), Some(1.), None, None, None, None],
            None,
        );
        let target = plan.target_id(0).unwrap();
        let mass = [
            [1. / 6., 1. / 12., 0.],
            [1. / 12., 1. / 3., 1. / 12.],
            [0., 1. / 12., 1. / 6.],
        ];
        let state_matrix = [[5., 7.], [-7., 5.]];
        let rate_matrix = if conjugate {
            [[2., 3.], [3., -2.]]
        } else {
            [[2., -3.], [3., 2.]]
        };
        for (current, rate) in [(Some(1.), None), (None, Some(1.)), (Some(2.), Some(-3.))] {
            let point = RegionPoint::Independent {
                state: &[2. / 3., 1., 2. / 3., 1.],
                rate_target: plan.target_id(1).unwrap(),
                rates: &[4. / 3., -4. / 3., 4. / 3., -4. / 3., 4. / 3., -4. / 3.],
                pullback: FieldPullback::Uniform { current, rate },
            };
            let action = prepared
                .linearize(&plan, target, point, &REFERENCE_ASSEMBLY_BACKEND)
                .unwrap();
            let full = dense(&action.systems[1]);
            for (i, mass_row) in mass.iter().enumerate() {
                for (j, mass) in mass_row.iter().enumerate() {
                    for r in 0..2 {
                        for c in 0..2 {
                            close(
                                full[(2 * i + r) * 6 + 2 * j + c],
                                3. * mass
                                    * (current.unwrap_or(0.) * state_matrix[r][c]
                                        + rate.unwrap_or(0.) * rate_matrix[r][c]),
                            );
                        }
                    }
                }
            }
            let expected = if conjugate {
                [4., 2., 8., 4., 4., 2.]
            } else {
                [10., -2., 20., -4., 10., -2.]
            };
            for (actual, expected) in action.residuals[1].iter().zip(expected) {
                close(*actual, expected);
            }
            let primal = prepared.residual(&plan, target, point).unwrap();
            for (actual, expected) in primal.iter().zip(&expected[2..]) {
                close(*actual, *expected);
            }
        }
    }
}

#[test]
fn zero_newton_residual_does_not_admit_an_empty_complete_equation() {
    let (plan, prepared) = fixture::<f64>(
        "model Degenerate() {domain body=box(0,1); state u:1 on body;
         relation r on body {derivative(u*u)=0;}}",
        vec![Some(0.), None, None],
        Some(0.25),
    );
    let nonlinear =
        eqiora_realization::NonlinearSolvePlan::new(0., 1e-12, std::num::NonZeroUsize::MIN, 0)
            .unwrap();
    let linear = eqiora_solver::SolverPlan::new(
        eqiora_solver::LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        std::num::NonZeroUsize::MIN,
    )
    .unwrap();
    let error = prepared
        .solve_newton(
            &plan,
            plan.target_id(0).unwrap(),
            &[0.; 2],
            nonlinear,
            eqiora_solver::LinearSolveRequest::new(&eqiora_solver::REFERENCE_LINEAR_SOLVER, linear),
            std::num::NonZeroUsize::MIN,
            |_| Ok(()),
        )
        .err()
        .unwrap();
    assert!(
        error.message().contains("complete spatial equation row"),
        "{error:?}"
    );
}
