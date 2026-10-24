//! Independent dense constitutive and physical-basis integrals.
use super::*;
use eqiora_meshing::AffineGeometryLinearization;
use num_complex::Complex64 as C;

const SOURCE: &str = "model Anisotropic() {
    domain body=box(0,2,0,3);
    domain wall=boundary(body,axis=0,side=lower);
    parameter a:tensor<1,2,2,2,2>=tensor_value(frame=body,components=[[[[7,1],[2,0]],[[-1,8],[0,1]]],[[[3,0],[9,2]],[[0,-2],[1,10]]]]);
    parameter g:tensor<1,2,2>=tensor_value(frame=body,components=[[1,2],[3,5]]);
    variable k:tensor<1,2,2,2,2> on body in smooth;
    relation coefficient on body {k=(1+coordinate(0)/1[m])*a;}
    variable u:vector<m,2> on body in smooth;
    relation balance on body {-div(contract(k,grad(u),axes=((2,0),(3,1)))+g)=0;}
    relation law on wall {normal(contract(k,grad(u),axes=((2,0),(3,1)))+g)=0;}
}";

#[test]
fn dense_anisotropic_flux_shares_component_actions_geometry_and_boundary_identity() {
    for complex in [false, true] {
        let source = if complex {
            SOURCE
                .replace("u:vector<m,2>", "u:vector<complex<m>,2>")
                .replace(
                    "contract(k,grad(u),axes=((2,0),(3,1)))+g",
                    "math.complex(2,3)*(contract(k,grad(u),axes=((2,0),(3,1)))+g)",
                )
        } else {
            SOURCE.to_owned()
        };
        if complex {
            check::<C>(&source, C::new(2., 3.));
        } else {
            check::<f64>(&source, 1.);
        }
    }
}

fn check<S: Coefficient>(source: &str, phase: S) {
    let (program, symbols) = fixture(source);
    let form = CompiledRegionForm::<S>::derive(&program, symbols["body"], 2).unwrap();
    assert!(
        form.conservative_system().is_err(),
        "TPFA cannot reinterpret anisotropic cross flux"
    );
    let (fields, rows) = form.si_bindings(|_| Ok(p1())).unwrap();
    let reference = ReferenceCell::simplex(2).unwrap();
    let bound = form.bind(reference, &fields, &rows, None).unwrap();
    let map = AffineGeometryMap::new(reference, 2, vec![0., 0.], vec![2., 0., 0., 3.]).unwrap();
    let direction =
        AffineGeometryLinearization::new(map.clone(), vec![2., -1.], vec![0.; 4]).unwrap();
    let (primal, tangent) = bound
        .prepare_affine_actions::<Native>(
            None,
            &map,
            Some(&direction),
            &simplex_duffy_gauss_legendre(2, 3).unwrap(),
            AffineInput::Current,
        )
        .unwrap();
    let tangent = tangent.unwrap();
    // Triangle area=3, mean x=2/3. Translation dx=2 gives d(1+x)=2.
    // These physical P1 gradients are independent of the integration implementation.
    let gradients = [[-0.5, -1. / 3.], [0.5, 0.], [0., 1. / 3.]];
    let k = [
        [7., 1., 2., 0.],
        [-1., 8., 0., 1.],
        [3., 0., 9., 2.],
        [0., -2., 1., 10.],
    ];
    let g = [[1., 2.], [3., 5.]];
    let close = |actual: S, value: f64| {
        let expected = phase * <S as From<f64>>::from(value);
        assert!(
            (actual.re() - expected.re()).abs() < 1e-11
                && (actual.im() - expected.im()).abs() < 1e-11,
            "{actual:?} != {expected:?}"
        );
    };
    for (i, gi) in gradients.iter().enumerate() {
        for (a, ga) in g.iter().enumerate() {
            close(
                primal.rhs()[2 * i + a],
                -3. * (ga[0] * gi[0] + ga[1] * gi[1]),
            );
            close(tangent.rhs()[2 * i + a], 0.);
            for (j, gj) in gradients.iter().enumerate() {
                for b in 0..2 {
                    let contraction = (0..2)
                        .flat_map(|x| (0..2).map(move |y| gi[x] * k[2 * a + x][2 * b + y] * gj[y]))
                        .sum::<f64>();
                    close(
                        primal.entry(2 * i + a, 2 * j + b).unwrap(),
                        5. * contraction,
                    );
                    close(
                        tangent.entry(2 * i + a, 2 * j + b).unwrap(),
                        6. * contraction,
                    );
                }
            }
        }
    }
    let laws = form
        .boundary_laws(&program, symbols["wall"], symbols["law"], None)
        .unwrap();
    assert_eq!(laws.len(), 1);
}

fn fixture(source: &str) -> (KernelProgram, BTreeMap<&'static str, RawId>) {
    let (transaction, model, symbols) = compile("anisotropic.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let ids = ["body", "wall", "law"]
        .into_iter()
        .map(|name| (name, symbols.get(name).unwrap()))
        .collect();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        ids,
    )
}

#[test]
fn anisotropic_flux_rejects_wrong_axes_and_differentiates_nonlinear_gradients() {
    let (prefix, boundary) = SOURCE.split_once("relation law on wall").unwrap();
    let wrong = format!(
        "{prefix}relation law on wall{}",
        boundary.replace("axes=((2,0),(3,1))", "axes=((2,1),(3,0))")
    );
    let (program, ids) = fixture(&wrong);
    let form = CompiledRegionForm::<f64>::derive(&program, ids["body"], 2).unwrap();
    assert!(
        form.boundary_laws(&program, ids["wall"], ids["law"], None)
            .is_err()
    );
    // The public boundary matcher reports an unmatched law. Inspect the exact
    // flux witness as well so an unrelated admission failure cannot satisfy it.
    let typed = typed_relation(&program, ids["law"]).unwrap();
    let dag = typed.expression();
    let normal = dag
        .nodes()
        .iter()
        .position(|node| {
            matches!(
                node,
                eqiora_schema::kernel::ExprNode::NormalComponent { .. }
            )
        })
        .unwrap();
    let error = form
        .require_boundary_flux(
            &program,
            crate::canonical_boundary::BoundaryRelationBinding::new(ids["wall"], ids["law"]),
            form.fields().next().unwrap().0,
            dag.node_id(normal as u32).unwrap(),
            true,
            None,
        )
        .unwrap_err();
    assert!(error.message().contains("flux differs"), "{error:?}");

    let nonlinear = SOURCE.replace(
        "contract(k,grad(u),axes=((2,0),(3,1)))",
        "contract(grad(u),grad(u),axes=((1,1),))",
    );
    let (program, ids) = fixture(&nonlinear);
    let form = CompiledRegionForm::<f64>::derive(&program, ids["body"], 2).unwrap();
    let (fields, rows) = form.si_bindings(|_| Ok(p1())).unwrap();
    let reference = ReferenceCell::simplex(2).unwrap();
    let bound = form.bind(reference, &fields, &rows, None).unwrap();
    let map = AffineGeometryMap::new(reference, 2, vec![0., 0.], vec![2., 0., 0., 3.]).unwrap();
    let cell = bound
        .prepare_cell(&map, &simplex_duffy_gauss_legendre(2, 3).unwrap())
        .unwrap();
    let action = cell
        .linearize(&BTreeMap::new(), &[0., 0., 2., 0., 0., 3.])
        .unwrap();
    // u=(x,y), so grad(u)grad(u)^T=I on a triangle of area 3.
    let gradients = [[-0.5, -1. / 3.], [0.5, 0.], [0., 1. / 3.]];
    let g = [[1., 2.], [3., 5.]];
    for (a, ga) in gradients.iter().enumerate() {
        for i in 0..2 {
            close(
                action.residual[2 * a + i],
                3. * (ga[i] + g[i][0] * ga[0] + g[i][1] * ga[1]),
            );
            for (b, gb) in gradients.iter().enumerate() {
                for k in 0..2 {
                    let expected = 3.
                        * (if i == k {
                            ga[0] * gb[0] + ga[1] * gb[1]
                        } else {
                            0.
                        } + ga[k] * gb[i]);
                    close(action.jacobian[(2 * a + i) * 6 + 2 * b + k], expected);
                }
            }
        }
    }
    assert!(cell.evaluate(&BTreeMap::new()).is_err());
}

#[test]
fn conjugated_gradient_requires_a_real_linear_realization() {
    let source = SOURCE
        .replace("u:vector<m,2>", "u:vector<complex<m>,2>")
        .replace("grad(u)", "math.conj(grad(u))");
    let (program, ids) = fixture(&source);
    let form = CompiledRegionForm::<C>::derive(&program, ids["body"], 2).unwrap();
    let (fields, rows) = form.si_bindings(|_| Ok(p1())).unwrap();
    let reference = ReferenceCell::simplex(2).unwrap();
    let bound = form.bind(reference, &fields, &rows, None).unwrap();
    let map = AffineGeometryMap::new(reference, 2, vec![0., 0.], vec![2., 0., 0., 3.]).unwrap();
    let error = bound
        .prepare_affine_actions::<Native>(
            None,
            &map,
            None,
            &simplex_duffy_gauss_legendre(2, 3).unwrap(),
            AffineInput::Current,
        )
        .unwrap_err();
    assert!(
        error.message().contains("real-coordinate realization"),
        "{error:?}"
    );
}
