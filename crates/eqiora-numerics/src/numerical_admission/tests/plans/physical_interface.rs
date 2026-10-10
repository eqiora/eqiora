//! Unequal material transmission requires authored continuity and flux balance.
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
    for reversed in [false, true] {
        let source = if reversed {
            SOURCE.replace(
                "interface(left_face,right_face)",
                "interface(right_face,left_face)",
            )
        } else {
            SOURCE.to_owned()
        };
        let (transaction, model, symbols) = eqiora_compiler::compile("transmission.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
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
        let resolved = ResolvedCommonPlan::resolve(
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
            None,
        )
        .unwrap();
        let resolved = replay_plan(resolved, &ResolveOnlyBackend);
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
        for (name, forward) in [
            ("left_flux", -6.0),
            ("right_flux", -6.0),
            ("gradient_jump", 1.0),
            ("weighted_gradient", 2.25),
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
            let expected = if reversed { -forward } else { forward };
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
