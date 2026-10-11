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

#[test]
fn congruence_scaling_preserves_interface_row_orientation() {
    use eqiora_realization::{
        AlgebraicBlockScale, PositivePhysicalScale, SymmetricCongruenceScaling,
    };
    let mut source = coupled_source();
    // Connect both fields so identity ordering cannot hide a negative test row.
    for index in 0..3 {
        for (side, label) in [("lower", "l"), ("upper", "r")] {
            if (index == 0 && side == "lower") || (index == 2 && side == "upper") {
                continue;
            }
            source = source.replace(
                &format!("relation v{label}{index} on {side}{index} {{ trace(v{index}) = 2*coordinate(0)/3[m]; }}"),
                &format!("instance v_{side}{index}: Interface(body=body{index}, face={side}{index}, value=v{index}, conductivity=conductivity);"),
            );
        }
    }
    let end = source.rfind('}').unwrap();
    source.insert_str(
        end,
        "connect v_upper0.edge, v_lower1.edge; connect v_upper1.edge, v_lower2.edge;\n",
    );
    for reversed in [false, true] {
        let source = if reversed {
            source.replace("reaction*(v1-u1)", "reaction*(v1+u1)")
        } else {
            source.clone()
        };
        let (model, ids) = compile_coupled(&source);
        let dimensional = chain_plan(&model, 3).unwrap();
        let scales = SymmetricCongruenceScaling::new(
            ids.values().map(|id| {
                AlgebraicBlockScale::new(
                    eqiora_solver::AlgebraicBlock::Field(id.downcast().unwrap()),
                    PositivePhysicalScale::new(DynQuantity::new(
                        2.0,
                        eqiora_core::DimExponents::DIMENSIONLESS,
                    ))
                    .unwrap(),
                )
            }),
            PositivePhysicalScale::new(DynQuantity::new(
                3.0,
                eqiora_core::DimExponents::from_integers([0, -1, 0, 0, 0, 0, 0]).unwrap(),
            ))
            .unwrap(),
        )
        .unwrap();
        let scaled = ResolvedCommonPlan::resolve(
            &model,
            chain_resources(3),
            CommonSpatialPolicy::Q1,
            dimensional.canonical_solve_request().unwrap(),
            Some(CommonScalingRequest::SymmetricCongruence(scales)),
            None,
            &ResolveOnlyBackend,
            None,
        );
        if reversed {
            assert!(
                scaled
                    .unwrap_err()
                    .message()
                    .contains("matching applied test orientations")
            );
            continue;
        }
        let scaled = replay_plan(scaled.unwrap(), &ResolveOnlyBackend);
        let result = scaled
            .as_linear()
            .unwrap()
            .run_result(&REFERENCE_LINEAR_SOLVER)
            .unwrap();
        for region in 0..3 {
            for (prefix, factor) in [("u", 1.0), ("v", 2.0)] {
                let id = ids[&format!("{prefix}{region}")].ulid().to_string();
                let index = (0..6)
                    .find(|&index| result.field(index).unwrap().0 == id)
                    .unwrap();
                let (_, values, _) = result.field_block(index, 0).unwrap();
                // u=x/3 and v=2x/3 solve both volume rows and flux continuity.
                for (node, actual) in values.iter().enumerate() {
                    let expected = factor * (region as f64 + node as f64 / 2.0) / 3.0;
                    assert!((actual - expected).abs() < 1e-9);
                }
            }
        }
    }
}
