use super::*;

fn coupled_source() -> String {
    let mut source = chain_source(&[0, 1, 2], &["u0", "u1", "u2"]).replace(
        "model Chain() {",
        "model Chain() { parameter reaction: 1/m^2 = 1;",
    );
    for index in 0..3 {
        source = source.replace(
            &format!("relation balance{index} on body{index} {{ -div(conductivity * grad(u{index})) = 0; }}"),
            &format!("variable v{index}: 1 on body{index} in smooth;
                relation balance{index} on body{index} {{
                    -div(conductivity*grad(u{index})) + reaction*(u{index}-v{index})
                    = -reaction*coordinate(0)/3[m];
                }}
                relation feedback{index} on body{index} {{
                    -div(conductivity*grad(v{index})) + reaction*(v{index}-u{index})
                    = reaction*coordinate(0)/3[m];
                }}
                relation vl{index} on lower{index} {{ trace(v{index}) = 2*coordinate(0)/3[m]; }}
                relation vr{index} on upper{index} {{ trace(v{index}) = 2*coordinate(0)/3[m]; }}"),
        );
    }
    source
}

fn compile_coupled(source: &str) -> (ModelEnvelope, BTreeMap<String, RawId>) {
    let (transaction, model, symbols) = eqiora_compiler::compile("coupled-chain.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let ids = (0..3)
        .flat_map(|index| [format!("u{index}"), format!("v{index}")])
        .map(|name| {
            let id = symbols.get(&name).unwrap();
            (name, id)
        })
        .collect();
    (ModelEnvelope::from_program(&program).unwrap(), ids)
}

#[test]
fn coupled_regions_keep_unconnected_field_boundary_laws_and_bidirectional_feedback() {
    let (model, ids) = compile_coupled(&coupled_source());
    let resolved = replay_plan(chain_plan(&model, 3).unwrap(), &ResolveOnlyBackend);
    let result = resolved
        .as_linear()
        .unwrap()
        .run_result(&REFERENCE_LINEAR_SOLVER)
        .unwrap();
    assert_eq!(result.field_count(), 6);
    for region in 0..3 {
        for (prefix, factor) in [("u", 1.0), ("v", 2.0)] {
            let id = ids[&format!("{prefix}{region}")].ulid().to_string();
            let index = (0..6)
                .find(|&index| result.field(index).unwrap().0 == id)
                .unwrap();
            let (_, values, shape) = result.field_block(index, 0).unwrap();
            assert_eq!(shape, &[3]);
            // Independent affine solution: u=x/3, v=2x/3. Both Laplacians
            // vanish; the two opposite reaction rows equal the authored loads.
            for (local, value) in values.iter().enumerate() {
                let expected = factor * (region as f64 + local as f64 / 2.0) / 3.0;
                assert!(
                    (value - expected).abs() < 1e-9,
                    "{prefix}{region}: {value} != {expected}"
                );
            }
        }
    }
    let missing = coupled_source().replace(
        "relation vl1 on lower1 { trace(v1) = 2*coordinate(0)/3[m]; }",
        "",
    );
    let (model, _) = compile_coupled(&missing);
    assert!(
        chain_plan(&model, 3)
            .unwrap_err()
            .message()
            .contains("complete boundary law coverage")
    );
}
