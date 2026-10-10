//! Unequal material transmission requires authored continuity and flux balance.
mod fields;
use super::*;

const SOURCE: &str = r#"
model Transmission() {
  domain left=box(0,1);
  domain right=box(1,3);
  domain lower=boundary(left,axis=0,side=lower);
  domain left_face=boundary(left,axis=0,side=upper);
  domain right_face=boundary(right,axis=0,side=lower);
  domain upper=boundary(right,axis=0,side=upper);
  domain contact=interface(left_face,right_face);
  parameter kl:1=2;
  parameter kr:1=3;
  variable ul:1 on left in smooth;
  variable ur:1 on right in smooth;
  relation left_balance on left { -div(kl*grad(ul))=0; }
  relation right_balance on right { -div(kr*grad(ur))=0; }
  relation left_value on lower { trace(ul)=0; }
  relation right_value on upper { trace(ur)=7; }
  relation transmission on contact {
    trace(ul,on=contact)=trace(ur,on=contact);
    normal(kl*grad(ul),on=contact)=normal(kr*grad(ur),on=contact);
  }
  observable left_flux:1/m=integral(normal(-kl*grad(ul),on=contact),measure(contact));
  observable right_flux:1/m=integral(normal(-kr*grad(ur),on=contact),measure(contact));
  observable scalar_value_jump:1=integral(trace(kl*ul,on=contact)-trace(kr*ur,on=contact),measure(contact));
  observable opposite_scalar_value_jump:1=integral(trace(kr*ur,on=contact)-trace(kl*ul,on=contact),measure(contact));
  observable weighted_scalar_value:1=integral(0.25*trace(kl*ul,on=contact)+0.75*trace(kr*ur,on=contact),measure(contact));
  observable opposite_weighted_scalar_value:1=integral(0.25*trace(kr*ur,on=contact)+0.75*trace(kl*ul,on=contact),measure(contact));
  observable exchanged_weighted_scalar_value:1=integral(0.75*trace(kr*ur,on=contact)+0.25*trace(kl*ul,on=contact),measure(contact));
  observable opposite_gradient_jump:1/m=integral(normal(grad(ur),on=contact)-normal(grad(ul),on=contact),measure(contact));
  observable gradient_jump:1/m=integral(normal(grad(ul),on=contact)-normal(grad(ur),on=contact),measure(contact));
  observable weighted_gradient:1/m=integral(0.25*normal(grad(ul),on=contact)+0.75*normal(grad(ur),on=contact),measure(contact));
}
"#;

#[test]
fn physical_interface_requires_both_authored_laws_and_exact_material_coefficients() {
    let trace = "trace(ul,on=contact)=trace(ur,on=contact);";
    let flux = "normal(kl*grad(ul),on=contact)=normal(kr*grad(ur),on=contact);";
    for (source, message) in [
        (
            SOURCE.replace(
                trace,
                "0.25*trace(ul,on=contact)+0.75*trace(ur,on=contact)=3;",
            ),
            "requires exactly two opposite trace terms",
        ),
        (
            SOURCE.replace(trace, ""),
            "requires explicit trace equality and flux balance",
        ),
        (
            SOURCE.replace(flux, ""),
            "requires explicit trace equality and flux balance",
        ),
        (
            SOURCE.replace(
                flux,
                "normal(kr*grad(ul),on=contact)=normal(kr*grad(ur),on=contact);",
            ),
            "matching both volume coefficients",
        ),
        (
            SOURCE.replace(trace, &format!("{trace}\n{trace}")),
            "repeats a trace equality",
        ),
        (
            SOURCE.replace(flux, &format!("{flux}\n{flux}")),
            "repeats a trace equality or flux balance",
        ),
    ] {
        let (transaction, model, _) = eqiora_compiler::compile("invalid-transmission.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let error =
            crate::scalar_conservation::recognize_scalar_conservation(&program).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
    }
}

#[test]
fn authored_physical_interface_runs_unequal_materials_and_replays_owned_fields() {
    for (authored, reversed, right_test) in [
        (false, false, false),
        (false, true, false),
        (true, false, false),
        (true, true, false),
        (true, false, true),
        (true, true, true),
    ] {
        let source = if reversed {
            SOURCE.replace(
                "interface(left_face,right_face)",
                "interface(right_face,left_face)",
            )
        } else {
            SOURCE.to_owned()
        };
        let source = if authored {
            let flux = "normal(kl*grad(ul),on=contact)=normal(kr*grad(ur),on=contact);";
            let mut source = source.replace(
                flux,
                &format!("}} relation flux_balance on contact {{ {flux}"),
            );
            let end = source.rfind('}').unwrap();
            source.insert_str(
                end,
                r#"
  form weak_continuity for transmission {
    test eta:1 for ul in h1;
    integrate(contact,trace(eta,on=contact)*(trace(ul,on=contact)-trace(ur,on=contact)))=0;
  }
"#,
            );
            if right_test {
                source.replace("test eta:1 for ul", "test eta:1 for ur")
            } else {
                source
            }
        } else {
            source
        };
        let compiled = eqiora_compiler::compile("transmission.eqi", &source)
            .unwrap()
            .remove(0);
        let projection = compiled
            .authored_formulations()
            .next()
            .map(|form| form.projection().clone());
        assert_eq!(projection.is_some(), authored);
        let (transaction, model, symbols) = compiled.into_parts();
        assert!(
            transaction.ops().iter().all(|op| !matches!(
                op,
                eqiora_graph::Op::DefineKernelNode {
                    node: eqiora_schema::kernel::KernelNode::Connection(_)
                }
            )),
            "a physical interface must not manufacture a Connection"
        );
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let model = ModelEnvelope::from_program(&program).unwrap();
        let graph = GeometryGraph::new();
        let interval = graph.interval([0.0, 3.0]).unwrap();
        let [lower, upper]: [_; 2] = interval.boundaries().try_into().unwrap();
        let geometry = graph
            .build(
                &interval,
                &BTreeMap::from([
                    ("body".to_owned(), vec![interval.region().into()]),
                    ("lower".to_owned(), vec![lower.into()]),
                    ("upper".to_owned(), vec![upper.into()]),
                ]),
            )
            .unwrap();
        let resolve = |projection: Option<&eqiora_compiler::AuthoredFormulationProjection>| {
            ResolvedCommonPlan::resolve(
                &model,
                cartesian_box_resources(&geometry, &[6]),
                CommonSpatialPolicy::Q1,
                CommonSolvePolicy::Linear(exact_reference_linear(
                    LinearSolver::BiConjugateGradientStabilized,
                    1e-10,
                    1e-12,
                    NonZeroUsize::new(1000).unwrap(),
                )),
                None,
                None,
                &ResolveOnlyBackend,
                projection,
            )
        };
        let resolved = resolve(projection.as_ref()).unwrap();
        if let Some(projection) = &projection {
            use eqiora_compiler::AuthoredFormExpressionV1 as E;
            let original = &projection.equations()[0].1;
            let E::Integrate {
                domain_ulid,
                integrand,
            } = original
            else {
                panic!("weak integral");
            };
            let E::Mul {
                left: test,
                right: residual,
            } = integrand.as_ref()
            else {
                panic!("test pairing");
            };
            let E::Sub { left, right } = residual.as_ref() else {
                panic!("continuity difference");
            };
            let encoded = String::from_utf8(projection.canonical_bytes().to_vec()).unwrap();
            for wrong in [
                E::Add {
                    left: left.clone(),
                    right: right.clone(),
                },
                *left.clone(),
                E::Sub {
                    left: left.clone(),
                    right: left.clone(),
                },
                E::Mul {
                    left: Box::new(E::Rational {
                        numerator: 1,
                        denominator: 1,
                        dimension: eqiora_core::DimExponents::from_integers([1, 0, 0, 0, 0, 0, 0])
                            .unwrap()
                            .exponents(),
                    }),
                    right: residual.clone(),
                },
            ] {
                let changed = E::Integrate {
                    domain_ulid: domain_ulid.clone(),
                    integrand: Box::new(E::Mul {
                        left: test.clone(),
                        right: Box::new(wrong),
                    }),
                };
                let bytes = encoded.replacen(
                    &serde_json::to_string(original).unwrap(),
                    &serde_json::to_string(&changed).unwrap(),
                    1,
                );
                let changed =
                    eqiora_compiler::AuthoredFormulationProjection::decode(bytes.as_bytes())
                        .unwrap();
                let error = resolve(Some(&changed)).unwrap_err();
                assert!(
                    error.message().contains("exact continuity equality"),
                    "{error:?}"
                );
            }
            let original_test = &projection.test_restrictions()[0];
            let mut restricted = original_test.clone();
            restricted.2.push(
                symbols
                    .get(if right_test {
                        "right_face"
                    } else {
                        "left_face"
                    })
                    .unwrap()
                    .ulid()
                    .to_string(),
            );
            let bytes = encoded.replacen(
                &serde_json::to_string(original_test).unwrap(),
                &serde_json::to_string(&restricted).unwrap(),
                1,
            );
            let changed =
                eqiora_compiler::AuthoredFormulationProjection::decode(bytes.as_bytes()).unwrap();
            assert!(
                resolve(Some(&changed))
                    .unwrap_err()
                    .message()
                    .contains("unrestricted H1 test")
            );
        }
        let resolved = replay_plan(resolved, &ResolveOnlyBackend);
        if let Some(projection) = &projection {
            let description = resolved.formulation().unwrap();
            assert_eq!(description.requested(), FormulationSelectionMode::Authored);
            assert_eq!(
                description.requested_source_identity(),
                Some(projection.source_identity())
            );
        }
        reject_changed_equality_root(&resolved);
        let result = resolved
            .as_linear()
            .unwrap()
            .run_result(&REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let bytes = result.to_bytes().unwrap();
        let recovered = crate::CommonResult::from_bytes(&bytes, &resolved).unwrap();
        assert_eq!(recovered.to_bytes().unwrap(), bytes);
        assert_eq!(recovered.field_count(), 2);
        // Resistance is 1/2 + 2/3 = 7/6, so physical +x flux is -6.
        // The interface value is 3; piecewise gradients are 3 and 2.
        // These affine solutions lie exactly in Q1 on the half-unit mesh.
        for (name, expected) in [
            ("ul", vec![0.0, 1.5, 3.0]),
            ("ur", vec![3.0, 4.0, 5.0, 6.0, 7.0]),
        ] {
            let id = symbols.get(name).unwrap().ulid().to_string();
            let index = (0..2)
                .find(|&index| recovered.field(index).unwrap().0 == id)
                .unwrap();
            let (association, values, shape) = recovered.field_block(index, 0).unwrap();
            assert_eq!(association, "vertex");
            assert_eq!(shape, &[expected.len()]);
            for (actual, expected) in values.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-9,
                    "{name}: {actual} != {expected}"
                );
            }
        }
        // The derived scalar Fields k*u have one-sided values 6 and 9,
        // despite continuity of u. Their jump is -3; exchanging operands gives
        // +3. Ordered weights 1/4 and 3/4 give 33/4, or 27/4 after exchanging
        // the values only. Exchanging both values and weights retains 33/4.
        // Scalar values do not change when only the common normal is reversed.
        // For the unequal vector fluxes grad(u), select gradient_jump for a
        // left-first interface and opposite_gradient_jump for right-first:
        // the outward flux sum is +1 in both cases, not an odd scalar jump.
        for (name, forward, oriented) in [
            ("left_flux", -6.0, true),
            ("right_flux", -6.0, true),
            ("gradient_jump", 1.0, true),
            ("opposite_gradient_jump", -1.0, true),
            ("weighted_gradient", 2.25, true),
            ("scalar_value_jump", -3.0, false),
            ("opposite_scalar_value_jump", 3.0, false),
            ("weighted_scalar_value", 8.25, false),
            ("opposite_weighted_scalar_value", 6.75, false),
            ("exchanged_weighted_scalar_value", 8.25, false),
        ] {
            let observable = symbols.get(name).unwrap().downcast().unwrap();
            let flux = recovered
                .observe(
                    &model,
                    observable,
                    &std::collections::HashMap::from([(
                        symbols.get("contact").unwrap().downcast().unwrap(),
                        eqiora_meshing::QuadratureRule::point(),
                    )]),
                )
                .unwrap()
                .value()
                .real_scalar_value()
                .unwrap()
                .value();
            let expected = if reversed && oriented {
                -forward
            } else {
                forward
            };
            assert!(
                (flux - expected).abs() < 1e-9,
                "{name}: {flux} != {expected}"
            );
        }
    }
}

fn reject_changed_equality_root(plan: &ResolvedCommonPlan) {
    use base64::Engine as _;
    let base64 = base64::engine::general_purpose::STANDARD;
    let wire = String::from_utf8(plan.to_bytes().unwrap()).unwrap();
    let original = plan.as_linear().unwrap().portable_realization();
    let bytes = original.to_bytes().unwrap();
    let encoded = base64.encode(&bytes);
    let text = String::from_utf8(bytes).unwrap();
    let source = original
        .transformations()
        .iter()
        .find_map(|node| match node {
            eqiora_realization::TransformationNode::ConformingTraceQuotient {
                source:
                    eqiora_realization::ConformingTraceSource::RelationEquality { root_index, .. },
                ..
            } => Some(*root_index),
            _ => None,
        })
        .unwrap();
    let changed = text.replace(
        &format!("\"root_index\":{source}"),
        &format!("\"root_index\":{}", u32::MAX),
    );
    let changed =
        eqiora_realization::PortableRealizationGraph::from_bytes(changed.as_bytes()).unwrap();
    // Change only the canonical nested graph, preserving the outer wire order.
    let changed = wire.replace(&encoded, &base64.encode(changed.to_bytes().unwrap()));
    assert_ne!(changed, wire);
    assert!(
        ResolvedCommonPlan::from_bytes(
            changed.as_bytes(),
            &ResolveOnlyBackend,
            eqiora_time::TimeBackendCapabilities::new(
                eqiora_time::TimeBackendIdentity::new("eqiora.test.time", "1"),
                &[eqiora_core::ScalarDomain::Real],
                &[eqiora_core::ScalarType::F64]
            ),
        )
        .is_err()
    );
}
