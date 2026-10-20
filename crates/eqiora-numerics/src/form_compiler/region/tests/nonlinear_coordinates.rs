//! Independent real-coordinate Jacobians of physical complex constitutive laws.
use super::*;
use crate::form_compiler::region::coordinates::Real;
use crate::parameter_point::ParameterPoint;
use eqiora_ir::ScalarSymbolCoordinate;
use eqiora_meshing::QuadratureRule;
use eqiora_schema::kernel::SymbolRef;
use num_complex::Complex64 as C;

#[test]
fn mixed_nonlinear_cells_preserve_physical_scales_and_parameter_products() {
    let source = "model Mixed() {
        domain body=box(0,2,0,3); parameter p:1=2;
        state u:complex<1> on body; state v:1 on body;
        relation constitutive on body {
            1[s]*derivative(u)+(1+p*math.real(u))*u+v*math.conj(u)=math.complex(6,8);
        }
        relation magnitude on body {1[s]*derivative(v)+v+math.abs2(u)=7;}
    }";
    let (transaction, model, symbols) = compile("nonlinear-coordinates.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let form = CompiledRegionForm::<C>::derive(&program, symbols.get("body").unwrap(), 2).unwrap();
    let parameter = symbols.get("p").unwrap().downcast().unwrap();
    let coordinate = ScalarSymbolCoordinate::for_value(
        SymbolRef::Parameter(parameter),
        program.typed_value(parameter.erase()).unwrap().value_type(),
    )
    .unwrap()
    .remove(0);
    let reference = ReferenceCell::hypercube(2).unwrap();
    let geometry =
        AffineGeometryMap::new(reference, 2, vec![1., 1.5], vec![1., 0., 0., 1.5]).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
    for p in [0., 2.] {
        let form = form
            .bind_parameter_point(&ParameterPoint::bind(&program, &[parameter], &[p]).unwrap())
            .unwrap();
        let (mut fields, rows) = form.si_bindings(|_| Ok(Space::cell_constant())).unwrap();
        let u = symbols.get("u").unwrap();
        for field in &mut fields {
            field.scale = DynQuantity::new(
                if field.field == u { 2. } else { 4. },
                DimExponents::DIMENSIONLESS,
            );
        }
        let bound = form
            .bind(
                reference,
                &fields,
                &rows,
                Some(&RegionTimeBinding {
                    step: DynQuantity::new(2., dim([0, 0, 1, 0, 0, 0, 0])),
                    states: Vec::new(),
                }),
            )
            .unwrap();
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
        let (cell, tangent) = bound
            .prepare_projected_cell_actions::<Real>(&geometry, &geometry, None, &quadrature)
            .unwrap();
        assert!(tangent.is_none());
        let previous =
            BTreeMap::from([(u, vec![-2., 3.]), (symbols.get("v").unwrap(), vec![1.25])]);
        assert!(cell.evaluate(&previous).is_err());
        assert!(cell.residual(&BTreeMap::new(), &[0.; 3]).is_err());
        assert!(
            cell.residual(
                &BTreeMap::from([(u, vec![-2.]), (symbols.get("v").unwrap(), vec![1.25])]),
                &[0.; 3]
            )
            .is_err()
        );
        let (partial, _) = bound
            .parameter_partial(&coordinate)
            .unwrap()
            .prepare_projected_cell_actions::<Real>(&geometry, &geometry, None, &quadrature)
            .unwrap();
        for (a, b, v) in [(2., -1., 0.5), (-1., 2., 3.)] {
            // Physical values are (a+ib, v); algebraic unknowns carry inverse Field scales.
            let physical = [a, b, v];
            let scales = [2., 2., 4.];
            let point = coordinates
                .iter()
                .map(|&i| physical[i] / scales[i])
                .collect::<Vec<_>>();
            let expected = [
                (1. + p * a + v) * a - 6. + (a + 2.) / 2.,
                (1. + p * a - v) * b - 8. + (b - 3.) / 2.,
                v + a * a + b * b - 7. + (v - 1.25) / 2.,
            ];
            let jacobian = [
                [1.5 + 2. * p * a + v, 0., a],
                [p * b, 1.5 + p * a - v, -b],
                [2. * a, 2. * b, 1.5],
            ];
            // Holding physical Fields fixed gives dF/dp=(a²,ab,0).
            let expected_partial = [a * a, a * b, 0.];
            let mixed = [[2. * a, 0., 0.], [b, a, 0.], [0., 0., 0.]];
            for (cell, history, residual, matrix) in [
                (&cell, &previous, expected, jacobian),
                (&partial, &previous, expected_partial, mixed),
            ] {
                let action = cell.linearize(history, &point).unwrap();
                let values = cell.residual(history, &point).unwrap();
                assert_eq!(action.residual, values);
                assert_eq!(action.residual.len(), 3);
                for (row, &r) in coordinates.iter().enumerate() {
                    // P0 integrates each constant expression over area six.
                    close(action.residual[row], 6. * residual[r]);
                    for (column, &c) in coordinates.iter().enumerate() {
                        close(
                            action.jacobian[row * 3 + column],
                            6. * matrix[r][c] * scales[c],
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn complex_natural_loads_keep_the_original_material_domain() {
    use eqiora_core::diagnostic::codes;
    use eqiora_meshing::{EntityIncidence, MeshEntity, OrientationCode};
    for profile in ["first_partials", "value_only"] {
        let source = format!(
            "property contract Response(input value:1):complex<1> {{derivatives {profile};}}
property release Material:Response {{
 analytic {{value=math.complex(1+value^2,value);source_unit:1=1;}}
 validity value>=0 and value<=1.99;outside reject;branch single;
 citation org.example.manufactured;license spdx.CC0_1_0;
}}
component Equation(support body:volume(ambient_dimension=1),property response:Response) {{
 variable u:complex<1> on body in smooth;
 relation balance on body {{-div(response(value=math.real(u))*grad(u))=0;}}
}}
model Main() {{domain body=box(0,1);instance physics:Equation(body=body,response=Material);}}"
        );
        let (transaction, model, symbols) = compile("complex-material-facet.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let form =
            CompiledRegionForm::<C>::derive(&program, symbols.get("body").unwrap(), 1).unwrap();
        let (mut fields, rows) = form.si_bindings(|_| Ok(p1())).unwrap();
        fields[0].scale = DynQuantity::new(2., DimExponents::DIMENSIONLESS);
        let reference = ReferenceCell::hypercube(1).unwrap();
        let bound = form.bind(reference, &fields, &rows, None).unwrap();
        let geometry = AffineGeometryMap::new(reference, 1, vec![0.5], vec![0.5]).unwrap();
        let point = [0.5, 1., 1., 0.5];
        // Physical u=1+x+i(2-x). Three-point Gauss samples have Re(u)<1.99,
        // but the right endpoint has Re(u)=2, outside the original property domain.
        let (cell, _) = bound
            .prepare_projected_cell_actions::<Real>(
                &geometry,
                &geometry,
                None,
                &QuadratureRule::tensor_product_gauss_legendre(1, 3).unwrap(),
            )
            .unwrap();
        assert!(cell.residual(&BTreeMap::new(), &point).is_ok());
        for side in 0..2 {
            let facet =
                AffineGeometryMap::new(ReferenceCell::point(), 1, vec![side as f64], vec![])
                    .unwrap();
            let incidence = EntityIncidence {
                entity: MeshEntity::new(1, 0),
                local_ordinal: side,
                orientation: OrientationCode::new(0),
            };
            let prepared = bound
                .prepare_natural_facet(
                    bound.fields()[0].field,
                    &geometry,
                    (&facet, incidence, &[side]),
                    &QuadratureRule::point(),
                    |_, _| {
                        // At the left endpoint K=2+i and du/dx=1-i; outward
                        // traction is -(2+i)(1-i)=-3+i. Right uses zero datum,
                        // which must not erase its constitutive domain obligation.
                        Ok(vec![if side == 0 {
                            C::new(-3., 1.)
                        } else {
                            C::new(0., 0.)
                        }])
                    },
                )
                .unwrap();
            let projected = prepared.project::<Real>().unwrap();
            if side == 0 {
                let expected = [3., -1., 0., 0.];
                assert_eq!(projected.residual(&point).unwrap(), expected);
                let action = projected.linearize(&point).unwrap();
                assert_eq!(action.residual, expected);
                // Replaced traction needs its original primal, never a derivative,
                // even when the property contract promises only values.
                assert_eq!(action.jacobian, [0.; 16]);
            } else {
                assert_eq!(
                    projected.residual(&point).unwrap_err().code(),
                    codes::EXPRESSION_DOMAIN_VIOLATION
                );
                assert_eq!(
                    projected.linearize(&point).err().unwrap().code(),
                    codes::EXPRESSION_DOMAIN_VIOLATION
                );
            }
        }
    }
}
