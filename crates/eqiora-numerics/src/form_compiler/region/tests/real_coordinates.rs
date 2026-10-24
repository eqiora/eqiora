//! Model-derived real-linear component actions use the same physical quadrature.
use super::*;
use crate::form_compiler::region::coordinates::Real;
use crate::parameter_point::ParameterPoint;
use eqiora_ir::ScalarSymbolCoordinate;
use eqiora_meshing::{AffineGeometryLinearization, QuadratureRule};
use eqiora_schema::kernel::SymbolRef;
use num_complex::Complex64 as C;

const SOURCE: &str = "model Tensor() {
 domain body=box(0,2,0,3);
 parameter a:complex<1>=math.complex(2,3);
 parameter b:complex<1>=math.complex(-1,4);
 parameter p:tensor<complex<1>,2,2>=tensor_value(frame=body,components=[
   [math.complex(1,7),math.complex(2,11)],
   [math.complex(3,13),math.complex(5,17)]]);
 variable u:tensor<complex<1>,2,2> on body in h1;
 relation balance on body { a*u+b*transpose(math.conj(u))=p; }
}";

fn close(actual: f64, expected: f64) {
    // Bounded affine tensor contractions and four-point positive quadrature.
    assert!(
        (actual - expected).abs() <= 4096. * f64::EPSILON * expected.abs().max(1.),
        "{actual} != {expected}"
    );
}

#[test]
fn cell_constant_value_basis_does_not_admit_a_weak_gradient() {
    let form = derive(
        "model Cell() {
        domain body=box(0,2,0,3);
        variable u:1 on body in h1;
        relation balance on body { -div(1[m^2]*grad(u))+u=1; }
    }",
    )
    .unwrap();
    let (fields, rows) = form.si_bindings(|_| Ok(Space::cell_constant())).unwrap();
    let error = form
        .bind(ReferenceCell::hypercube(2).unwrap(), &fields, &rows, None)
        .unwrap_err();
    assert!(
        error.message().contains("weak differential pairing"),
        "{error:?}"
    );
}

#[test]
fn mixed_real_and_complex_fields_keep_only_their_independent_coordinates() {
    let source = "model Mixed() {
        domain body=box(0,2,0,3);
        variable u:complex<1> on body;variable v:1 on body;
        relation given on body {2*u=math.complex(6,8);}
        relation projected on body {v=2*math.real(u)-math.imag(u)+1;}
    }";
    let (transaction, model, symbols) = compile("mixed-coordinate-region.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let form = CompiledRegionForm::<C>::derive(&program, symbols.get("body").unwrap(), 2).unwrap();
    let (fields, rows) = form.si_bindings(|_| Ok(Space::cell_constant())).unwrap();
    let reference = ReferenceCell::hypercube(2).unwrap();
    let bound = form.bind(reference, &fields, &rows, None).unwrap();
    let map = AffineGeometryMap::new(reference, 2, vec![1., 1.5], vec![1., 0., 0., 1.5]).unwrap();
    let rule = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
    assert!(
        bound
            .prepare_affine_actions::<Native>(
                &mut Default::default(),
                &map,
                None,
                &rule,
                AffineInput::Current
            )
            .is_err()
    );
    let (local, _) = bound
        .prepare_affine_actions::<Real>(
            &mut Default::default(),
            &map,
            None,
            &rule,
            AffineInput::Current,
        )
        .unwrap();
    let u = symbols.get("u").unwrap();
    let coordinates = bound
        .fields
        .iter()
        .flat_map(|field| {
            if field.field == u {
                vec![0, 1]
            } else {
                vec![2]
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(coordinates.len(), 3);
    assert_eq!(local.rows(), 3);
    // P0 integrates the constant matrix over area six. In (Re u, Im u, v)
    // coordinates: 2u=6+8i and v-2 Re(u)+Im(u)=1. There is no Im(v) unknown.
    let expected = [[2., 0., 0.], [0., 2., 0.], [-2., 1., 1.]];
    let rhs = [6., 8., 1.];
    for (row, &physical_row) in coordinates.iter().enumerate() {
        close(local.rhs()[row], 6. * rhs[physical_row]);
        for (column, &physical_column) in coordinates.iter().enumerate() {
            close(
                local.entry(row, column).unwrap(),
                6. * expected[physical_row][physical_column],
            );
        }
    }
}

fn coefficient(a: C, b: C, conjugate: bool, row: usize, column: usize) -> [[f64; 2]; 2] {
    let mut block = [[0.; 2]; 2];
    if row == column {
        block = [[a.re, -a.im], [a.im, a.re]];
    }
    // The tensor transpose exchanges the two physical indices, leaving the
    // mesh basis axis alone. Conjugation changes the imaginary input sign.
    if row == (column % 2) * 2 + column / 2 {
        let sign = if conjugate { -1. } else { 1. };
        block[0][0] += b.re;
        block[0][1] -= sign * b.im;
        block[1][0] += b.im;
        block[1][1] += sign * b.re;
    }
    block
}

fn check_matrix(
    local: &eqiora_assembly::LocalContribution<f64>,
    nodes: usize,
    a: C,
    b: C,
    conjugate: bool,
) {
    assert_eq!(local.rows(), nodes * 8);
    for i in 0..nodes {
        for j in 0..nodes {
            // On this area-six rectangle Q1 mass is the product of the exact
            // one-dimensional [2,1;1,2] integrals. P0 integrates the constant 1.
            let mass = if nodes == 1 {
                6.
            } else {
                6. / 36. * 2_f64.powi(2 - (i ^ j).count_ones() as i32)
            };
            for row in 0..4 {
                for column in 0..4 {
                    for (r, values) in coefficient(a, b, conjugate, row, column).iter().enumerate()
                    {
                        for (c, value) in values.iter().enumerate() {
                            close(
                                local
                                    .entry(8 * i + 2 * row + r, 8 * j + 2 * column + c)
                                    .unwrap(),
                                mass * value,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn tensor_affine_coordinates_preserve_conjugation_rebinding_and_parameter_partials() {
    for conjugate in [false, true] {
        let source = if conjugate {
            SOURCE.to_owned()
        } else {
            SOURCE.replace("math.conj(u)", "u")
        };
        let (transaction, model, symbols) = compile("real-coordinate-region.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let original =
            CompiledRegionForm::<C>::derive(&program, symbols.get("body").unwrap(), 2).unwrap();
        let ids = ["a", "b"].map(|name| symbols.get(name).unwrap().downcast().unwrap());
        let reference = ReferenceCell::hypercube(2).unwrap();
        let map =
            AffineGeometryMap::new(reference, 2, vec![1., 1.5], vec![1., 0., 0., 1.5]).unwrap();
        let direction =
            AffineGeometryLinearization::new(map.clone(), vec![0.; 2], vec![1., 0., 0., 1.5])
                .unwrap();
        let rule = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
        for (a, b) in [
            (C::new(2., 3.), C::new(-1., 4.)),
            (C::new(-2., 5.), C::new(0., 0.)),
        ] {
            let form = original
                .bind_inputs(
                    &ParameterPoint::bind(&program, &ids, &[a.re, a.im, b.re, b.im]).unwrap(),
                )
                .unwrap();
            for (space, nodes) in [(Space::cell_constant(), 1), (p1(), 4)] {
                let (fields, rows) = form.si_bindings(|_| Ok(space)).unwrap();
                let bound = form.bind(reference, &fields, &rows, None).unwrap();
                let (local, geometry) = bound
                    .prepare_affine_actions::<Real>(
                        &mut Default::default(),
                        &map,
                        Some(&direction),
                        &rule,
                        AffineInput::Current,
                    )
                    .unwrap();
                check_matrix(&local, nodes, a, b, conjugate);
                for i in 0..nodes {
                    for (component, value) in [1., 7., 2., 11., 3., 13., 5., 17.].iter().enumerate()
                    {
                        close(local.rhs()[8 * i + component], 6. / nodes as f64 * value);
                    }
                }
                let geometry = geometry.unwrap();
                for (actual, original) in geometry
                    .matrix()
                    .iter()
                    .zip(local.matrix())
                    .chain(geometry.rhs().iter().zip(local.rhs()))
                {
                    close(*actual, 2. * original);
                }
                if !conjugate {
                    let (native, _) = bound
                        .prepare_affine_actions::<Native>(
                            &mut Default::default(),
                            &map,
                            None,
                            &rule,
                            AffineInput::Current,
                        )
                        .unwrap();
                    for row in 0..native.rows() {
                        for column in 0..native.columns() {
                            let value = native.entry(row, column).unwrap();
                            close(local.entry(2 * row, 2 * column).unwrap(), value.re);
                            close(local.entry(2 * row + 1, 2 * column).unwrap(), value.im);
                            close(local.entry(2 * row, 2 * column + 1).unwrap(), -value.im);
                            close(local.entry(2 * row + 1, 2 * column + 1).unwrap(), value.re);
                        }
                    }
                } else if b.norm() != 0. {
                    assert!(
                        bound
                            .prepare_affine_actions::<Native>(
                                &mut Default::default(),
                                &map,
                                None,
                                &rule,
                                AffineInput::Current
                            )
                            .is_err()
                    );
                }
                for (parameter, id) in ids.iter().enumerate() {
                    let value = program.typed_value(id.erase()).unwrap();
                    for coordinate in ScalarSymbolCoordinate::for_value(
                        SymbolRef::Parameter(*id),
                        value.value_type(),
                    )
                    .unwrap()
                    {
                        let unit = if coordinate.is_imaginary() {
                            C::new(0., 1.)
                        } else {
                            C::new(1., 0.)
                        };
                        let derivative = bound.input_partial(&coordinate).unwrap();
                        let (partial, _) = derivative
                            .prepare_affine_actions::<Real>(
                                &mut Default::default(),
                                &map,
                                None,
                                &rule,
                                AffineInput::Current,
                            )
                            .unwrap();
                        let zero = C::new(0., 0.);
                        check_matrix(
                            &partial,
                            nodes,
                            if parameter == 0 { unit } else { zero },
                            if parameter == 1 { unit } else { zero },
                            conjugate,
                        );
                        for value in partial.rhs() {
                            close(*value, 0.);
                        }
                    }
                }
            }
        }
    }
}
