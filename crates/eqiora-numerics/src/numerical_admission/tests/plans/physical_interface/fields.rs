use super::*;

#[test]
fn authored_forms_select_each_exact_continuity_relation_on_one_interface() {
    let flux = "normal(kl*grad(ul),on=contact)=normal(kr*grad(ur),on=contact);";
    let mut source = SOURCE.replace(
        flux,
        &format!("}} relation flux_balance on contact {{ {flux}"),
    );
    let end = source.rfind('}').unwrap();
    source.insert_str(
        end,
        r#"
        variable vl:1 on left in smooth;
        variable vr:1 on right in smooth;
        relation vl_balance on left { -div(kl*grad(vl))=0; }
        relation vr_balance on right { -div(kr*grad(vr))=0; }
        relation vl_value on lower { trace(vl)=0; }
        relation vr_value on upper { trace(vr)=7; }
        relation v_continuity on contact { trace(vl)=trace(vr); }
        relation v_flux on contact { normal(kl*grad(vl))=normal(kr*grad(vr)); }
        form weak_u for transmission {
            test eta:1 for ul in h1;
            integrate(contact,trace(eta,on=contact)*(trace(ul,on=contact)-trace(ur,on=contact)))=0;
        }

    "#,
    );
    for select_v in [false, true] {
        let source = if select_v {
            source.replace(
                r#"        form weak_u for transmission {
            test eta:1 for ul in h1;
            integrate(contact,trace(eta,on=contact)*(trace(ul,on=contact)-trace(ur,on=contact)))=0;
        }"#,
                r#"        form weak_v for v_continuity {
            test zeta:1 for vl in h1;
            integrate(contact,trace(zeta,on=contact)*(trace(vl,on=contact)-trace(vr,on=contact)))=0;
        }"#,
            )
        } else {
            source.clone()
        };
        let compiled = eqiora_compiler::compile("multiple-interface-forms.eqi", &source)
            .unwrap()
            .remove(0);
        let projections = compiled
            .authored_formulations()
            .map(|form| form.projection().clone())
            .collect::<Vec<_>>();
        assert_eq!(projections.len(), 1);
        let (transaction, model, _) = compiled.into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let model = ModelEnvelope::from_program(&program).unwrap();
        let graph = GeometryGraph::new();
        let interval = graph.interval([0.0, 3.0]).unwrap();
        let boundaries = interval.boundaries();
        let geometry = graph
            .build(
                &interval,
                &BTreeMap::from([
                    ("body".to_owned(), vec![interval.region().into()]),
                    ("lower".to_owned(), vec![boundaries[0].into()]),
                    ("upper".to_owned(), vec![boundaries[1].into()]),
                ]),
            )
            .unwrap();
        for projection in projections {
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
                Some(&projection),
            )
            .unwrap();
            assert_eq!(
                resolved.formulation().unwrap().requested(),
                FormulationSelectionMode::Authored
            );
            let crate::numerical_admission::native::RecognizedNativeModel::Linear(equations) =
                resolved.as_linear().unwrap().admission.recognized_model()
            else {
                panic!("linear equations")
            };
            let mut interfaces = equations.interfaces.clone();
            for _ in 0..2 {
                crate::form_compiler::interface::admit(&projection, &program, &interfaces).unwrap();
                interfaces.reverse();
            }
            let resolved = replay_plan(resolved, &ResolveOnlyBackend);
            let result = resolved
                .as_linear()
                .unwrap()
                .run_result(&REFERENCE_LINEAR_SOLVER)
                .unwrap();
            assert_eq!(result.field_count(), 4);
        }
    }
}
