use std::num::NonZeroU16;

use eqiora_compiler::compile;
use eqiora_core::{DimExponents, DynQuantity};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::{AffineGeometryMap, ReferenceCell, simplex_duffy_gauss_legendre};
use eqiora_realization::{
    BackwardEulerStateBinding, BackwardEulerStatePair, PositivePhysicalScale, Space,
};
use eqiora_schema::kernel::KernelNode;

use super::*;

mod boundary;
mod flux;
mod nonlinear;
mod scalar;
mod tetrahedron;
mod validation;

const MIXED: &str = "model Mixed() {
 domain body = box(0, 1, 0, 1);

 parameter density: kg / m ^ 3 = 3;
 parameter viscosity: kg / (m * s) = 2;
 state v: vector<m / s, 2> on body;
 variable p: kg / (m * s ^ 2) on body;
 relation balance on body {
  density * derivative(v) - div(2 * viscosity * symmetric_part(grad(v)) - isotropic_lift(p)) = 0;
 }
 relation constraint on body { div(v) = 0; }
}";

const ELIMINATED: &str = "model Elastic() {
 domain body = box(0, 1, 0, 1);

 parameter density: kg / m ^ 3 = 3;
 parameter mu: kg / (m * s ^ 2) = 2;
 parameter lambda: kg / (m * s ^ 2) = 5;
 state v: vector<m / s, 2> on body;
 state d: vector<m, 2> on body;
 relation kinematics on body { derivative(d) - v = 0; }
 relation balance on body {
  density * derivative(v) - div(2 * mu * symmetric_part(grad(d)) + lambda * isotropic_lift(div(d))) = 0;
 }
}";

fn derive(source: &str) -> Result<CompiledRegionForm, Diagnostic> {
    derive_dimension(source, 2)
}

fn derive_dimension(source: &str, dimension: usize) -> Result<CompiledRegionForm, Diagnostic> {
    let (transaction, model, _) = compile("region.eqi", source)
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
    let form = CompiledRegionForm::derive(&program, domain, dimension)?;
    assert_eq!(form.domain(), domain);
    assert!(
        form.roles
            .relations
            .values()
            .all(|role| !role.dependencies.is_empty())
    );
    Ok(form)
}

fn p1() -> Space {
    Space::continuous_lagrange(NonZeroU16::new(1).unwrap())
}
fn dim(values: [i32; 7]) -> DimExponents {
    DimExponents::from_integers(values).unwrap()
}
fn geometry() -> AffineGeometryMap {
    AffineGeometryMap::new(
        ReferenceCell::simplex(2).unwrap(),
        2,
        vec![0.0, 0.0],
        vec![1.0, 0.0, 0.0, 1.0],
    )
    .unwrap()
}

fn inputs(
    form: &CompiledRegionForm,
    bubble: bool,
) -> (
    Vec<RegionFieldBinding>,
    BTreeMap<RawId, DynQuantity>,
    RegionTimeBinding,
) {
    let fields = form
        .fields()
        .map(|(field, value_type)| RegionFieldBinding {
            field,
            space: if bubble && !value_type.shape().is_scalar() {
                Space::simplex_p1_bubble()
            } else {
                p1()
            },
            scale: DynQuantity::new(
                if value_type.shape().is_scalar() {
                    7.0
                } else {
                    2.0
                },
                value_type.dimension(),
            ),
        })
        .collect::<Vec<_>>();
    let rows = form
        .rows()
        .map(|(relation, _, value_type)| {
            let units = value_type
                .dimension()
                .mul(dim([0, form.dimension as i32, 0, 0, 0, 0, 0]))
                .unwrap()
                .pow(-1, 1)
                .unwrap();
            (
                relation,
                DynQuantity::new(
                    if value_type.shape().is_scalar() {
                        -11.0
                    } else {
                        13.0
                    },
                    units,
                ),
            )
        })
        .collect();
    let states = form
        .roles
        .relations
        .iter()
        .filter_map(|(relation, role)| match role.kind {
            Role::Kinematic { state, rate } => Some(BackwardEulerStateBinding::new(
                BackwardEulerStatePair::new(
                    relation.downcast().unwrap(),
                    state.downcast().unwrap(),
                    rate.downcast().unwrap(),
                )
                .unwrap(),
                p1(),
                PositivePhysicalScale::new(DynQuantity::new(
                    4.0,
                    form.roles.fields[&state].1.dimension(),
                ))
                .unwrap(),
            )),
            _ => None,
        })
        .collect();
    (
        fields,
        rows,
        RegionTimeBinding {
            step: DynQuantity::new(0.25, dim([0, 0, 1, 0, 0, 0, 0])),
            states,
        },
    )
}

fn bound(form: &CompiledRegionForm, bubble: bool) -> BoundRegionForm {
    let (fields, rows, time) = inputs(form, bubble);
    form.bind(
        ReferenceCell::simplex(2).unwrap(),
        &fields,
        &rows,
        Some(&time),
    )
    .unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 3e-11 * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

const GRADIENT: [[f64; 2]; 3] = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
fn mass(i: usize, j: usize) -> f64 {
    if i == j { 1.0 / 12.0 } else { 1.0 / 24.0 }
}
fn stiffness(i: usize, a: usize, j: usize, b: usize, mu: f64, lambda: f64) -> f64 {
    // Integrate mu*(delta_ab gradNi.gradNj + d_b Ni d_a Nj) + lambda*d_a Ni*d_b Nj.
    0.5 * (mu
        * (if a == b {
            GRADIENT[i][0] * GRADIENT[j][0] + GRADIENT[i][1] * GRADIENT[j][1]
        } else {
            0.0
        })
        + mu * GRADIENT[i][b] * GRADIENT[j][a]
        + lambda * GRADIENT[i][a] * GRADIENT[j][b])
}

#[test]
fn mixed_p1_mass_strain_and_signed_pressure_blocks_are_equation_derived() {
    let form = derive(MIXED).unwrap();
    let bound = bound(&form, false);
    let velocity = bound
        .fields()
        .iter()
        .find(|layout| layout.components == 2)
        .unwrap();
    let pressure = bound
        .fields()
        .iter()
        .find(|layout| layout.components == 1)
        .unwrap();
    let previous = BTreeMap::from([(velocity.field, vec![2.0, -1.0, 2.0, -1.0, 2.0, -1.0])]);
    let local = bound
        .evaluate(
            &geometry(),
            &simplex_duffy_gauss_legendre(2, 4).unwrap(),
            &previous,
        )
        .unwrap();
    let n = 9;
    for (i, gradient) in GRADIENT.iter().enumerate() {
        for (a, component) in gradient.iter().enumerate() {
            let row = velocity.range.start + 2 * i + a;
            close(local.rhs()[row], 13.0 * 12.0 * [2.0, -1.0][a] / 6.0);
            for j in 0..3 {
                for b in 0..2 {
                    let column = velocity.range.start + 2 * j + b;
                    let expected = (if a == b { 12.0 * mass(i, j) } else { 0.0 })
                        + stiffness(i, a, j, b, 2.0, 0.0);
                    close(local.matrix()[row * n + column], 13.0 * 2.0 * expected);
                }
                close(
                    local.matrix()[row * n + pressure.range.start + j],
                    -13.0 * 7.0 * component / 6.0,
                );
                close(
                    local.matrix()[(pressure.range.start + j) * n + row],
                    -11.0 * 2.0 * component / 6.0,
                );
            }
        }
    }
    for i in pressure.range.clone() {
        close(local.rhs()[i], 0.0);
        for j in pressure.range.clone() {
            close(local.matrix()[i * n + j], 0.0);
        }
    }
}

#[test]
fn eliminated_state_gives_dt_stiffness_and_physical_history_rhs() {
    let form = derive(ELIMINATED).unwrap();
    let bound = bound(&form, false);
    let velocity = &bound.fields()[0];
    let state = *bound.eliminations.keys().next().unwrap();
    let old_velocity = vec![1.0, -1.0, 2.0, 0.0, 3.0, 1.0];
    let old_state = vec![0.0, 0.0, 1.0, 0.0, 0.0, 2.0];
    let previous = BTreeMap::from([
        (velocity.field, old_velocity.clone()),
        (state, old_state.clone()),
    ]);
    let local = bound
        .evaluate(
            &geometry(),
            &simplex_duffy_gauss_legendre(2, 4).unwrap(),
            &previous,
        )
        .unwrap();
    for i in 0..3 {
        for a in 0..2 {
            let row = 2 * i + a;
            let mut rhs = 0.0;
            for j in 0..3 {
                for b in 0..2 {
                    let m = if a == b { 12.0 * mass(i, j) } else { 0.0 };
                    let k = stiffness(i, a, j, b, 2.0, 5.0);
                    close(
                        local.matrix()[row * 6 + 2 * j + b],
                        13.0 * 2.0 * (m + 0.25 * k),
                    );
                    rhs += m * old_velocity[2 * j + b] - k * old_state[2 * j + b];
                }
            }
            close(local.rhs()[row], 13.0 * rhs);
        }
    }
}

#[test]
fn mini_bubble_mass_and_mixed_blocks_use_exact_barycentric_integrals() {
    let form = derive(&MIXED.replace("(m * s) = 2", "(m * s) = 0")).unwrap();
    let bound = bound(&form, true);
    let velocity = bound
        .fields()
        .iter()
        .find(|layout| layout.components == 2)
        .unwrap();
    let pressure = bound
        .fields()
        .iter()
        .find(|layout| layout.components == 1)
        .unwrap();
    let previous = BTreeMap::from([(velocity.field, vec![0.0; 8])]);
    let local = bound
        .evaluate(
            &geometry(),
            &simplex_duffy_gauss_legendre(2, 5).unwrap(),
            &previous,
        )
        .unwrap();
    // b=27 lambda0 lambda1 lambda2. Integral(prod lambda_i^a_i)=prod(a_i!)/(2+sum a_i)!.
    for i in 0..4 {
        for j in 0..4 {
            for a in 0..2 {
                for b in 0..2 {
                    let m = match (i == 3, j == 3) {
                        (true, true) => 81.0 / 560.0,
                        (true, false) | (false, true) => 3.0 / 40.0,
                        _ => mass(i, j),
                    };
                    let row = velocity.range.start + 2 * i + a;
                    let column = velocity.range.start + 2 * j + b;
                    close(
                        local.matrix()[row * 11 + column],
                        if a == b { 13.0 * 2.0 * 12.0 * m } else { 0.0 },
                    );
                }
            }
        }
    }
    for (i, gradient_i) in GRADIENT
        .iter()
        .map(Some)
        .chain(std::iter::once(None))
        .enumerate()
    {
        for (a, _) in GRADIENT[0].iter().enumerate() {
            for (j, gradient_j) in GRADIENT.iter().enumerate() {
                let integral = match gradient_i {
                    None => -gradient_j[a] * 9.0 / 40.0,
                    Some(gradient) => gradient[a] / 6.0,
                };
                let row = velocity.range.start + 2 * i + a;
                close(
                    local.matrix()[row * 11 + pressure.range.start + j],
                    -13.0 * 7.0 * integral,
                );
                close(
                    local.matrix()[(pressure.range.start + j) * 11 + row],
                    -11.0 * 2.0 * integral,
                );
            }
        }
    }
}

#[test]
fn derivative_of_eliminated_state_uses_rate_without_unused_previous_coefficients() {
    let form = derive(
        "model Kinematic() {
        domain body = box(0, 1, 0, 1);
        parameter drag: kg / (m ^ 3 * s) = 3;
        state d: vector<m, 2> on body;
        variable v: vector<m / s, 2> on body;
        relation pair on body { derivative(d) - v = 0; }
        relation balance on body { drag * derivative(d) = 0; }
    }",
    )
    .unwrap();
    let bound = bound(&form, false);
    assert!(bound.previous_fields().is_empty());
    let local = bound
        .evaluate(
            &geometry(),
            &simplex_duffy_gauss_legendre(2, 4).unwrap(),
            &BTreeMap::new(),
        )
        .unwrap();
    for i in 0..3 {
        for a in 0..2 {
            for j in 0..3 {
                for b in 0..2 {
                    close(
                        local.matrix()[(2 * i + a) * 6 + 2 * j + b],
                        if a == b {
                            13.0 * 2.0 * 3.0 * mass(i, j)
                        } else {
                            0.0
                        },
                    );
                }
            }
        }
    }
    for rhs in local.rhs() {
        close(*rhs, 0.0);
    }
}
