use super::*;

fn source(initials: &str) -> String {
    let mut source = String::from("model StoredRegions() {");
    for index in 0..2 {
        source += &format!(
            "domain body{index}=box({index},{});
             domain left{index}=boundary(body{index},axis=0,side=lower);
             domain right{index}=boundary(body{index},axis=0,side=upper);
             state u{index}:1 on body{index} in h1;
             law balance{index} on body{index} {{
                 storage 1[s/m^2]*u{index}; flux -grad(u{index}); source 0[1/m^2];
             }}
             relation fixed_left{index} on left{index} {{ trace(u{index})=0; }}
             relation fixed_right{index} on right{index} {{ trace(u{index})=0; }}",
            index + 1,
        );
    }
    source + initials + "}"
}

fn forms(initials: &str) -> Vec<Result<CompiledLinearBlockForm<f64>, Diagnostic>> {
    let (transaction, model, symbols) = compile("stored-regions.eqi", &source(initials))
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    (0..2)
        .map(|index| {
            CompiledLinearBlockForm::derive(
                &program,
                symbols.get(&format!("body{index}")).unwrap(),
                1,
                &BTreeSet::new(),
            )
        })
        .collect()
}

#[test]
fn initial_equations_are_owned_by_each_exact_region() {
    for initial in [
        "initial { u0=2; u1=7; }",
        "initial { u1=7; } initial { u0=2; }",
    ] {
        for (form, expected) in forms(initial).into_iter().zip([2., 7.]) {
            let form = form.unwrap();
            assert_eq!(
                form.initial_values_at(&[0.5]).unwrap(),
                BTreeMap::from([(form.fields()[0].0, vec![expected])]),
            );
        }
    }
    let mut missing = forms("initial { u1=7; }").into_iter();
    // Unspecified source data remain absent until exact State construction.
    assert!(
        missing
            .next()
            .unwrap()
            .unwrap()
            .initial_values_at(&[0.5])
            .unwrap()
            .is_empty()
    );
    assert!(missing.next().unwrap().is_ok());
    let mut duplicate = forms("initial { u0=2; u0=3; u1=7; }").into_iter();
    assert!(
        duplicate
            .next()
            .unwrap()
            .unwrap_err()
            .message()
            .contains("unique on its support")
    );
    assert!(duplicate.next().unwrap().is_ok());
}

#[test]
fn vector_initial_data_retains_components_and_external_scalar_factor() {
    let source = "model VectorInitial() {
        domain body=box(0,1,0,1);
        variable g:m on body in smooth;
        variable a:1 on body in smooth;
        relation potential on body { g=(coordinate(0)^2+coordinate(1)^2)/2[m]; }
        relation factor on body { a=coordinate(0)/1[m]; }
        state u:vector<1,2> on body in smooth;
        initial { u=a*grad(g); }
        relation balance on body { 1[s/m^2]*derivative(u)=div(grad(u)); }
    }";
    let (transaction, model, symbols) = compile("vector-initial.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let domain = symbols.get("body").unwrap();
    let field = symbols.get("u").unwrap();
    let roles = EquationRoles::derive(&program, [domain]).unwrap();
    let coefficients = coefficients::<f64>(&program, 2, &roles).unwrap();
    let initial = super::super::temporal::initial_values(
        &program,
        domain,
        2,
        &BTreeMap::from([(field, Data::constant(2, 1.))]),
        &coefficients,
        true,
        None,
    )
    .unwrap();
    // At (1/2,1), a grad(g)=(1/4,1/2); grad(a*g) would be (7/8,1/2).
    assert_eq!(
        initial[&field].evaluate(&[0.5, 1.], &[]).unwrap(),
        [0.25, 0.5]
    );
}

#[test]
fn vector_storage_rejects_foreign_partial_and_nonpositive_capacities() {
    let source = "model VectorStorage() {
        domain body=box(0,1,0,1);
        domain left=boundary(body,axis=0,side=lower);
        domain right=boundary(body,axis=0,side=upper);
        domain bottom=boundary(body,axis=1,side=lower);
        domain top=boundary(body,axis=1,side=upper);
        state u:vector<1,2> on body in smooth;
        state v:vector<1,2> on body in smooth;
        initial { u=0; v=0; }
        relation first on body { 1[s/m^2]*derivative(u)=div(grad(u)); }
        relation second on body { 1[s/m^2]*derivative(v)=div(grad(v)); }
        relation fixed_left on left { trace(u)=0; trace(v)=0; }
        relation fixed_right on right { trace(u)=0; trace(v)=0; }
        relation fixed_bottom on bottom { trace(u)=0; trace(v)=0; }
        relation fixed_top on top { trace(u)=0; trace(v)=0; }
    }";
    let derive = |source: &str| {
        let (transaction, model, symbols) = compile("vector-storage-negative.eqi", source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        CompiledLinearBlockForm::<f64>::derive(
            &program,
            symbols.get("body").unwrap(),
            2,
            &BTreeSet::new(),
        )
    };
    assert!(derive(source).unwrap().is_transient());
    for (source, message) in [
        (
            source.replace("derivative(u)", "derivative(v)"),
            // A foreign rate and the local diffusion identify two principal
            // Fields, so exact equation-role derivation rejects before binding.
            "no unique supported principal trial",
        ),
        (
            source.replace("1[s/m^2]*derivative(v)=div(grad(v))", "-div(grad(v))=0"),
            "storage for every unknown Field",
        ),
        (
            source.replace("1[s/m^2]", "0[s/m^2]"),
            "strictly positive constant capacity",
        ),
        (
            source.replace("1[s/m^2]", "-1[s/m^2]"),
            "strictly positive constant capacity",
        ),
    ] {
        let diagnostic = derive(&source).unwrap_err();
        assert!(diagnostic.message().contains(message), "{diagnostic:?}");
    }
}
