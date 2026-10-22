//! Exact one-cell integrals falsify missing direct material and coefficient products.
use super::*;
use crate::parameter_point::{ParameterPoint, coordinates};
use eqiora_core::{Id, entity::kinds};
use eqiora_meshing::{EntityIncidence, MeshEntity, OrientationCode, QuadratureRule};

fn model(source: &str) -> (CompiledRegionForm<f64>, KernelProgram, Id<kinds::Parameter>) {
    let (transaction, model, symbols) = compile("material-parameter.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let form = CompiledRegionForm::derive(&program, symbols.get("body").unwrap(), 1).unwrap();
    (form, program, symbols.get("p").unwrap().downcast().unwrap())
}

fn bind(form: &CompiledRegionForm<f64>) -> BoundRegionForm<f64> {
    let (fields, rows) = form.si_bindings(|_| Ok(p1())).unwrap();
    form.bind(ReferenceCell::hypercube(1).unwrap(), &fields, &rows, None)
        .unwrap()
}

fn geometry() -> AffineGeometryMap {
    AffineGeometryMap::new(
        ReferenceCell::hypercube(1).unwrap(),
        1,
        vec![0.5],
        vec![0.5],
    )
    .unwrap()
}

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "{actual:e} != {expected:e}"
        );
    }
}

#[test]
fn coefficient_and_material_partials_share_the_physical_weak_action() {
    let (form, program, parameter) = model(
        "model Products() {
            domain body=box(0,1);parameter p:1=1;
            variable u:1 on body in smooth;
            relation r on body {-div(p*(1+p*u)*grad(u))=0;}
        }",
    );
    let coordinate = coordinates(&program, &[parameter]).unwrap().remove(0);
    let rule = QuadratureRule::tensor_product_gauss_legendre(1, 3).unwrap();
    for p in [1., 2.] {
        let point = ParameterPoint::bind(&program, &[parameter], &[p]).unwrap();
        let primal = form.bind_parameter_point(&point).unwrap();
        for partial in [
            primal.input_partial(&coordinate).unwrap(),
            form.input_partial(&coordinate)
                .unwrap()
                .bind_parameter_point(&point)
                .unwrap(),
        ] {
            let bound = bind(&partial);
            let cell = bound.prepare_cell(&geometry(), &rule).unwrap();
            // u=1+2x. The integrated flux is p*(u_R-u_L)*(1+p*(u_L+u_R)/2).
            // Its p partial is (u_R-u_L)*(1+p*(u_L+u_R)).
            let flux_partial = 2. * (1. + 4. * p);
            let expected = [-flux_partial, flux_partial];
            close(
                &cell.residual(&BTreeMap::new(), &[1., 3.]).unwrap(),
                &expected,
            );
            let action = cell.linearize(&BTreeMap::new(), &[1., 3.]).unwrap();
            close(&action.residual, &expected);
            close(
                &action.jacobian,
                &[1. + 2. * p, -1. - 6. * p, -1. - 2. * p, 1. + 6. * p],
            );
        }
        // Differentiating a form must not mutate the primal's parameter point.
        let cell = bind(&primal).prepare_cell(&geometry(), &rule).unwrap();
        let flux = 2. * p * (1. + 2. * p);
        close(
            &cell.residual(&BTreeMap::new(), &[1., 3.]).unwrap(),
            &[-flux, flux],
        );
    }
}

#[test]
fn time_partials_share_volume_products_and_prescribed_boundary_data() {
    use eqiora_ir::ScalarSymbolCoordinate;
    use eqiora_schema::kernel::SymbolRef;

    let source = "model TimedProducts() {
        domain body=box(0,1);
        domain right=boundary(body,axis=0,side=upper);
        variable u:1 on body in smooth;
        relation balance on body {
            -div((time()/1[s])*(1+(time()/1[s])*u)*grad(u))=0;
        }
        relation prescribed on right {trace(u)=(time()/1[s])^2*coordinate(0)/1[m];}
    }";
    let (transaction, model, symbols) = compile("time-partials.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let coordinate = ScalarSymbolCoordinate::for_value(
        SymbolRef::Time,
        &eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            dim([0, 0, 1, 0, 0, 0, 0]),
        )
        .unwrap(),
    )
    .unwrap()
    .remove(0);
    let rule = QuadratureRule::tensor_product_gauss_legendre(1, 3).unwrap();
    for t in [0., 2.] {
        let form = CompiledRegionForm::<f64>::derive_at_time(
            &program,
            symbols.get("body").unwrap(),
            1,
            Some(t),
        )
        .unwrap();
        let bound = bind(&form);
        let partial = bound.input_partial(&coordinate).unwrap();
        let cell = partial.prepare_cell(&geometry(), &rule).unwrap();
        // On [0,1], u=1+2x. The integrated flux is 2t(1+2t).
        // At fixed State its time partial is 2+8t, including at t=0.
        close(
            &cell.residual(&BTreeMap::new(), &[1., 3.]).unwrap(),
            &[-2. - 8. * t, 2. + 8. * t],
        );
        close(
            &cell
                .linearize(&BTreeMap::new(), &[1., 3.])
                .unwrap()
                .jacobian,
            &[1. + 2. * t, -1. - 6. * t, -1. - 2. * t, 1. + 6. * t],
        );
        let mut law = form
            .boundary_laws(
                &program,
                symbols.get("right").unwrap(),
                symbols.get("prescribed").unwrap(),
                Some(t),
            )
            .unwrap()
            .remove(0);
        let primal = law.clone();
        law.input_partial(&coordinate, 1).unwrap();
        close(&law.evaluate(&[1.], &[]).unwrap(), &[2. * t]);
        close(&primal.evaluate(&[1.], &[]).unwrap(), &[t * t]);
    }
}

fn material_source(profile: &str, argument: &str, upper: &str) -> String {
    format!(
        "property contract Response(input value:1):1 {{derivatives {profile};}}
        property release Material:Response {{
            analytic {{value=1+value^2;source_unit:1=1;}}
            validity value>=0 and value<={upper};outside reject;branch single;
            citation org.example.manufactured;license spdx.CC0_1_0;
        }}
        component Equation(
            support body:volume(ambient_dimension=1),property response:Response,parameter p:1
        ) {{
            variable u:1 on body in smooth;
            relation r on body {{-div(response(value={argument})*grad(u))=0;}}
        }}
        model Products() {{
            domain body=box(0,1);parameter p:1=1;
            instance physics:Equation(body=body,response=Material,p=p);
        }}"
    )
}

#[test]
fn direct_weak_material_partial_needs_first_products_without_a_mixed_derivative() {
    for profile in ["first_partials", "value_only"] {
        let (form, program, parameter) = model(&material_source(profile, "p*u", "4"));
        let coordinate = coordinates(&program, &[parameter]).unwrap().remove(0);
        let bound = bind(&form.input_partial(&coordinate).unwrap());
        let cell = bound
            .prepare_cell(
                &geometry(),
                &QuadratureRule::tensor_product_gauss_legendre(1, 3).unwrap(),
            )
            .unwrap();
        let residual = cell.residual(&BTreeMap::new(), &[1., 3.]);
        if profile == "first_partials" {
            // d/dp[(1+p²u²)*u_x] = 2p*u²*u_x;
            // p=1, u=1+2x, integral u² dx = 13/3.
            close(&residual.unwrap(), &[-52. / 3., 52. / 3.]);
        } else {
            assert!(
                residual
                    .unwrap_err()
                    .message()
                    .contains("value-only contract")
            );
        }
        // Linearizing the derivative form would request a mixed material
        // derivative; neither contract grants that extra product.
        assert!(cell.linearize(&BTreeMap::new(), &[1., 3.]).is_err());
    }
}

#[test]
fn parameter_partial_keeps_original_material_domain_on_replaced_natural_flux() {
    for profile in ["first_partials", "value_only"] {
        // p is unconsumed: its residual partial is zero, but the original law
        // remains demanded at volume and natural-boundary points.
        let (form, program, parameter) = model(&material_source(profile, "u", "1.99"));
        let coordinate = coordinates(&program, &[parameter]).unwrap().remove(0);
        let bound = bind(&form.input_partial(&coordinate).unwrap());
        let geometry = geometry();
        let cell = bound
            .prepare_cell(
                &geometry,
                &QuadratureRule::tensor_product_gauss_legendre(1, 3).unwrap(),
            )
            .unwrap();
        close(
            &cell.residual(&BTreeMap::new(), &[1., 2.]).unwrap(),
            &[0., 0.],
        );
        let field = bound.fields()[0].field;
        for (side, x) in [(0, 0.), (1, 1.)] {
            let facet = AffineGeometryMap::new(ReferenceCell::point(), 1, vec![x], vec![]).unwrap();
            let prepared = bound
                .prepare_natural_facet(
                    field,
                    &geometry,
                    (
                        &facet,
                        EntityIncidence {
                            entity: MeshEntity::new(1, 0),
                            local_ordinal: side,
                            orientation: OrientationCode::new(0),
                        },
                        &[side],
                    ),
                    &QuadratureRule::point(),
                    |_, _| Ok(vec![0.]),
                )
                .unwrap();
            if side == 0 {
                close(&prepared.residual(&[1., 2.]).unwrap(), &[0., 0.]);
            } else {
                assert_eq!(
                    prepared.residual(&[1., 2.]).unwrap_err().code(),
                    eqiora_core::diagnostic::codes::EXPRESSION_DOMAIN_VIOLATION
                );
            }
        }
    }
}
