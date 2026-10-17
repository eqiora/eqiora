//! Retained static coefficients must satisfy the original equations after decoding.
use super::*;
use sha2::{Digest, Sha256};

fn reseal_values(result: &crate::CommonResult, field_index: usize, values: &[f64]) -> Vec<u8> {
    let bytes = result.to_bytes().unwrap();
    let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let field = &wire["content"]["payload"]["fields"][field_index];
    let original = &field["blocks"][0]["values"];
    let old = format!("\"values\":{}", serde_json::to_string(original).unwrap());
    let new = format!("\"values\":{}", serde_json::to_string(values).unwrap());
    let mut text = String::from_utf8(bytes).unwrap();
    let marker = format!(
        "\"field_id\":{}",
        serde_json::to_string(&field["field_id"]).unwrap()
    );
    let start = text.find(&marker).unwrap();
    let offset = start + text[start..].find(&old).unwrap();
    text.replace_range(offset..offset + old.len(), &new);
    reseal_content(text, &wire)
}

fn reseal_content(text: String, wire: &serde_json::Value) -> Vec<u8> {
    // Preserve canonical member order. Content identity is public hashing, not
    // a signature: changing coefficients and honestly rehashing must still reach
    // numerical acceptance, rather than fail only the checksum comparison.
    let content = text
        .strip_suffix('}')
        .unwrap()
        .split_once("\"content\":")
        .unwrap()
        .1;
    let mut digest = Sha256::new();
    digest.update(wire["schema"].as_str().unwrap().as_bytes());
    digest.update([0]);
    digest.update(content.as_bytes());
    let identity = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    text.replacen(wire["identity"].as_str().unwrap(), &identity, 1)
        .into_bytes()
}

#[test]
fn static_result_replay_rejects_resealed_equation_and_trace_violations() {
    let geometry = cartesian_interval();
    for (spatial, complex) in [
        (CommonSpatialPolicy::Q1, false),
        (CommonSpatialPolicy::CellCenteredTpfa, false),
        (CommonSpatialPolicy::Q1, true),
    ] {
        let source = if complex {
            POISSON_INTERVAL
                .replace("variable potential: 1", "variable potential: complex<1>")
                .replace(
                    "-div(grad(potential)) - source_scale = 0",
                    "-div(grad(potential)) - math.complex(source_scale, 2*source_scale) = 0",
                )
        } else {
            POISSON_INTERVAL.to_owned()
        };
        let model = scalar_box_model(&geometry, &source, "PoissonInterval", &["left", "right"]);
        let plan = resolve_scalar_box(&model, cartesian_box_resources(&geometry, &[4]), spatial);
        let result = plan.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
        let (_, values, _) = result.field_block(0, 0).unwrap();
        let expected: &[f64] = if spatial == CommonSpatialPolicy::Q1 {
            &[0., 3. / 32., 1. / 8., 3. / 32., 0.]
        } else {
            &[1. / 16., 1. / 8., 1. / 8., 1. / 16.]
        };
        // Q1 reproduces the nodal samples of x(1-x)/2. The four-cell TPFA
        // stencil gives [1,2,2,1]/16, including its half-cell boundary distances.
        // Complex forcing 1+2i scales both real and imaginary solutions.
        let width = if complex { 2 } else { 1 };
        for (value, expected) in values.chunks_exact(width).zip(expected) {
            assert!((value[0] - expected).abs() < 1e-11);
            if complex {
                assert!((value[1] - 2. * expected).abs() < 1e-11);
            }
        }
        let positive = reseal_values(&result, 0, values);
        assert_eq!(positive, result.to_bytes().unwrap());
        assert_eq!(
            crate::CommonResult::from_bytes(&positive, result.plan()).unwrap(),
            result
        );
        let interior = if spatial == CommonSpatialPolicy::Q1 {
            2
        } else {
            1
        };
        for index in [interior * width, 0] {
            let mut changed = values.to_vec();
            changed[index] = 1.;
            let error = crate::CommonResult::from_bytes(
                &reseal_values(&result, 0, &changed),
                result.plan(),
            )
            .expect_err(
                "rehashed finite coefficients must satisfy their original equation and trace",
            );
            assert!(
                error.message().contains("residual") || error.message().contains("prescribed"),
                "{error:?}"
            );
        }
    }
}

#[test]
fn coordinate_result_replay_rejects_resealed_primary_and_reconstructed_fields() {
    use eqiora_compiler::{CompiledModel, StaticBindingValue};
    use eqiora_graph::{GraphStore, InMemoryGraphStore};
    use eqiora_schema::kernel::AxisBounds;
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let projection = "model Sample(support body:interval(m)) {
        variable u:1 on body; relation value on body {u=2;}
    }";
    let radial = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../verify/language/factor-integrals/models/radial-diffusion.eqi"
    ));
    for (source, component, support, expected) in [
        (projection, "Sample", "body", 2.),
        (radial, "Particle", "radius", 1.),
    ] {
        let compiled = CompiledModel::compile_selected(
            "replay.eqi",
            source,
            component,
            &[(
                support,
                StaticBindingValue::CoordinateInterval(
                    AxisBounds::new(DynQuantity::new(0., length), DynQuantity::new(1., length))
                        .unwrap(),
                ),
            )],
        )
        .unwrap();
        let domain = compiled.symbols().get(support).unwrap().downcast().unwrap();
        let (transaction, model_id, _) = compiled.into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model_id).unwrap();
        let model = ModelEnvelope::from_program(&program).unwrap();
        let mesh = AuthenticatedCommonMesh::coordinate_factors(&model, domain, &[1]).unwrap();
        let plan = resolve_scalar_box(&model, mesh, CommonSpatialPolicy::CellCentered);
        let result = plan.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
        // Projection is u=2. For the single radial cell with D=1,q=6,R=1,
        // volume=1/3 and outer conductance=2 give c=1; boundary fluxes 0,2
        // reconstruct the retained cell flux as 1. Both Fields happen to be 1,
        // so each mutation must target its exact Field, not just matching bytes.
        for field in 0..result.field_count() {
            let (_, values, _) = result.field_block(field, 0).unwrap();
            assert_eq!(values.len(), 1);
            assert!((values[0] - expected).abs() < 1e-12);
            let original = reseal_values(&result, field, values);
            assert_eq!(original, result.to_bytes().unwrap());
            assert_eq!(
                crate::CommonResult::from_bytes(&original, result.plan()).unwrap(),
                result
            );
            let error = crate::CommonResult::from_bytes(
                &reseal_values(&result, field, &[3.]),
                result.plan(),
            )
            .expect_err(
                "rehashed coefficients must satisfy their original equation and reconstruction",
            );
            assert!(
                error.message().contains("residual") || error.message().contains("reconstruction"),
                "{error:?}"
            );
        }
    }
}

#[test]
fn stationary_run_retains_and_rechecks_its_initial_state() {
    use base64::{Engine as _, prelude::BASE64_STANDARD};
    let geometry = cartesian_interval();
    let model = scalar_box_model(
        &geometry,
        POISSON_INTERVAL,
        "PoissonInterval",
        &["left", "right"],
    );
    let plan = resolve_scalar_box(
        &model,
        cartesian_box_resources(&geometry, &[4]),
        CommonSpatialPolicy::Q1,
    );
    let field = plan.fields().next().unwrap().0;
    let seed = |values| {
        plan.initial_state(
            None,
            vec![
                CommonInitialField::new(
                    model.digest().unwrap(),
                    field,
                    Some(
                        CommonInitialValues::new(eqiora_core::ValueShape::scalar(), values)
                            .unwrap(),
                    ),
                    None,
                )
                .unwrap(),
            ],
        )
        .unwrap()
    };
    // -u''=1, u(0)=u(1)=0 gives u=x(1-x)/2. Q1 has these exact
    // nodal values on four uniform elements; no solve output supplies this seed.
    let exact = vec![0., 3. / 32., 1. / 8., 3. / 32., 0.];
    let initial = seed(exact.clone());
    let zero = seed(vec![0.; 5]);
    assert_ne!(
        plan.run_identity(Some(&initial)).unwrap(),
        plan.run_identity(Some(&zero)).unwrap()
    );
    assert_ne!(
        plan.run_identity(Some(&zero)).unwrap(),
        plan.run_identity(None).unwrap()
    );
    let result = plan
        .run_result(Some(&initial), &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    assert_eq!(result.field_block(0, 0).unwrap().1, exact);
    assert_eq!(
        result
            .solve_evidence(None, None)
            .map(|solve| solve.completed_iterations()),
        Some(0)
    );
    assert_eq!(
        result
            .solve_evidence(None, None)
            .map(|solve| solve.initial_residual_norm()),
        Some(0.)
    );
    let Some(crate::CommonInitialState::Spatial(retained)) = result.initial_state() else {
        panic!("missing seed");
    };
    assert_eq!(retained, &initial);
    let bytes = result.to_bytes().unwrap();
    assert_eq!(
        crate::CommonResult::from_bytes(&bytes, result.plan()).unwrap(),
        result
    );
    let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let encoded =
        serde_json::to_string(&wire["content"]["payload"]["acceptance"]["initial_state_base64"])
            .unwrap();
    assert_ne!(encoded, "null");
    // Both replacement and omission preserve a legitimate State encoding and
    // an honestly recomputed public content hash. Only the seed/residual
    // relation should reject them; this makes no producer-authentication claim.
    for replacement in [
        serde_json::to_string(&BASE64_STANDARD.encode(zero.to_bytes().unwrap())).unwrap(),
        "null".to_owned(),
    ] {
        let text = String::from_utf8(bytes.clone())
            .unwrap()
            .replacen(&encoded, &replacement, 1);
        let error = crate::CommonResult::from_bytes(&reseal_content(text, &wire), result.plan())
            .unwrap_err();
        assert!(error.message().contains("initial residual"), "{error:?}");
    }
    let from_zero = plan
        .run_result(Some(&zero), &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    // b=(1/4,1/4,1/4) in the three free coordinates.
    assert_eq!(
        from_zero
            .solve_evidence(None, None)
            .map(|solve| solve.initial_residual_norm()),
        Some((3.0_f64 / 16.).sqrt())
    );
    assert_eq!(
        crate::CommonResult::from_bytes(&from_zero.to_bytes().unwrap(), from_zero.plan()).unwrap(),
        from_zero
    );
}
