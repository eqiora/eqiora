use super::super::*;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_schema::kernel::ExprNode;

#[test]
fn model_and_form_tangential_pairings_match_independent_oriented_boundary_rows() {
    for dimensions in [2, 3] {
        for complex in [false, true] {
            let bounds = vec!["0,1"; dimensions].join(",");
            let scalar = if complex { "complex<1>" } else { "1" };
            let source = format!(
                r#"model M() {{
                domain body=box({bounds});
                domain other=box({bounds});
                domain face=boundary(body,axis=0,side=lower);
                domain opposite=boundary(body,axis=0,side=upper);
                domain foreign=boundary(other,axis=0,side=lower);
                variable u:vector<{scalar},{dimensions}> on body in h1;
                variable v:vector<{scalar},{dimensions}> on other;
                relation other_law on other {{v=v;}}
                relation law on body {{u=u;}}
                relation surface on face {{tangential_trace(u)=tangential_trace(u);}}
                form weak for law {{
                    test eta:1 for u;
                    integrate(face,inner(tangential_trace(eta),tangential_trace(u)))=0;
                }}
            }}"#
            );
            let compiled =
                eqiora_compiler::CompiledModel::compile_selected("boundary.eqi", &source, "M", &[])
                    .unwrap();
            let projection = compiled
                .authored_formulations()
                .next()
                .unwrap()
                .projection()
                .clone();
            let (transaction, model, symbols) = compiled.into_parts();
            let mut store = InMemoryGraphStore::new();
            store.commit(transaction).unwrap();
            let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
            let field = symbols.get("u").unwrap().ulid().to_string();
            let face = symbols.get("face").unwrap().ulid().to_string();
            let mut context = Context {
                name: "eta",
                field: &field,
                dimensions,
                remaining: 65536,
                symbols: symbol_types(&program),
                supports: field_supports(&program),
                domains: domain_supports(&program),
                integration_domain: Some(face.clone()),
            };
            let typed = crate::form_compiler::scalar::typed_relation(
                &program,
                symbols.get("surface").unwrap(),
            )
            .unwrap();
            let ExprNode::Sub(left, _) = typed
                .expression()
                .node(typed.expression().roots()[0])
                .unwrap()
            else {
                panic!("retained boundary residual");
            };
            // Independently n×u has the cyclic rows (ny*uz-nz*uy,
            // nz*ux-nx*uz, nx*uy-ny*ux); 2D retains the last scalar row.
            let row = |test, j, k| {
                let normal = |i| Polynomial::atom(Atom::Normal(face.clone(), i));
                let value = |i| {
                    Polynomial::symbol(
                        if test {
                            Atom::TraceTest(vec![i])
                        } else {
                            Atom::TraceField(field.clone(), vec![i])
                        },
                        complex,
                    )
                };
                normal(j)
                    .checked_mul(&value(k))
                    .unwrap()
                    .checked_add(
                        &normal(k)
                            .checked_mul(&value(j))
                            .unwrap()
                            .checked_neg()
                            .unwrap(),
                    )
                    .unwrap()
            };
            // Normal contraction is the component pairing sum_i n_i u_i,
            // with no conjugation and no tangential lift.
            let expected_normal = (0..dimensions).fold(
                Polynomial::constant(ExactRational::integer(0)),
                |sum, axis| {
                    sum.checked_add(
                        &Polynomial::atom(Atom::Normal(face.clone(), axis))
                            .checked_mul(&Polynomial::symbol(
                                Atom::TraceField(field.clone(), vec![axis]),
                                complex,
                            ))
                            .unwrap(),
                    )
                    .unwrap()
                },
            );
            let normal = E::NormalTrace {
                on_ulid: face.clone(),
                value: Box::new(E::Field {
                    ulid: field.clone(),
                }),
            };
            assert_eq!(context.scalar(&normal, 0), Some(expected_normal));
            let wrong_normal = E::NormalTrace {
                on_ulid: symbols.get("opposite").unwrap().ulid().to_string(),
                value: Box::new(E::Field {
                    ulid: field.clone(),
                }),
            };
            assert!(context.scalar(&wrong_normal, 0).is_none());
            let rows = if dimensions == 2 {
                vec![(0, 1)]
            } else {
                vec![(1, 2), (2, 0), (0, 1)]
            };
            let mut pairing = Polynomial::constant(ExactRational::integer(0));
            let tangent = E::TangentialTrace {
                on_ulid: face.clone(),
                value: Box::new(E::Field {
                    ulid: field.clone(),
                }),
            };
            for (axis, (j, k)) in rows.into_iter().enumerate() {
                let coordinate = if dimensions == 2 { vec![] } else { vec![axis] };
                let expected = row(false, j, k);
                assert_eq!(
                    context.source(&typed, *left, &coordinate, 0),
                    Some(expected.clone())
                );
                let actual = if dimensions == 2 {
                    context.scalar(&tangent, 0)
                } else {
                    context.vector(&tangent, axis, 0)
                };
                assert_eq!(actual, Some(expected.clone()));
                assert_ne!(actual, Some(expected.checked_neg().unwrap()));
                pairing = pairing
                    .checked_add(
                        &row(true, j, k)
                            .conjugate()
                            .unwrap()
                            .checked_mul(&expected)
                            .unwrap(),
                    )
                    .unwrap();
            }
            assert!(
                context
                    .tangential_component(
                        &E::Field {
                            ulid: symbols.get("v").unwrap().ulid().to_string(),
                        },
                        if dimensions == 2 { &[] } else { &[0] },
                        0
                    )
                    .is_none()
            );
            context.integration_domain = Some(symbols.get("opposite").unwrap().ulid().to_string());
            assert!(
                context
                    .source(&typed, *left, if dimensions == 2 { &[] } else { &[0] }, 0)
                    .is_none()
            );
            context.integration_domain = None;
            let integral = &projection.equations()[0].1;
            let expected = pairing
                .checked_mul(&Polynomial::atom(Atom::Measure(face.clone())))
                .unwrap();
            assert_eq!(context.integral(integral), Some(expected.clone()));
            let E::Integrate { integrand, .. } = integral else {
                panic!("boundary integral");
            };
            for name in ["foreign", "body"] {
                let forged = E::Integrate {
                    domain_ulid: symbols.get(name).unwrap().ulid().to_string(),
                    integrand: integrand.clone(),
                };
                assert!(context.integral(&forged).is_none(), "{name}");
                assert!(context.integration_domain.is_none());
            }
            let opposite = E::Integrate {
                domain_ulid: symbols.get("opposite").unwrap().ulid().to_string(),
                integrand: integrand.clone(),
            };
            assert_ne!(context.integral(&opposite), Some(expected));
            // Moving only the measure must not reinterpret the retained trace targets.
            assert!(context.integral(&opposite).is_none());
            assert!(
                context
                    .tangential_component(
                        &E::Field {
                            ulid: field.clone()
                        },
                        &[],
                        0
                    )
                    .is_none()
            );
        }
    }
}
