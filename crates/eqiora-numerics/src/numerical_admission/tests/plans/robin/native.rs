//! Four ordinary solves export actual operators; scientific acceptance is external.
use super::*;
use crate::numerical_admission::tests::artifact_capture::{
    binary64_hex, digest, json_bytes, write,
};
use eqiora_meshing::MeshTopology;

#[test]
fn ordinary_robin_producer_exports_full_operator_coefficients_and_reactions() {
    let root = super::runtime::output_root();
    let source_head = super::runtime::source_identity(&root);
    let oracle = super::oracle::Oracle::preflight(&root);
    let executable_sha = digest(&std::fs::read(std::env::current_exe().unwrap()).unwrap());
    for (name, inputs) in CASES {
        let directory = root.join(name);
        std::fs::create_dir(&directory).unwrap();
        let authored = source(inputs);
        let source_sha = write(&directory.join("model.eqi"), authored.as_bytes());
        write(
            &directory.join("case-attempt.json"),
            &json_bytes(&serde_json::json!({
                "schema":1,"case":name,"inputs":inputs,"status":"attempt-started",
                "source_head":source_head,"authored_source_sha256":source_sha,
                "native_executable_sha256":executable_sha,
            })),
        );
        let (model, original) = compile_and_resolve(inputs, &REFERENCE_LINEAR_SOLVER);
        let model_bytes = model.canonical_json().unwrap();
        let plan_bytes = original.to_bytes().unwrap();
        let mesh_bytes = original
            .as_spatial()
            .unwrap()
            .admission
            .authenticated_mesh()
            .to_bytes()
            .unwrap();
        write(&directory.join("model.json"), &model_bytes);
        write(&directory.join("plan.json"), &plan_bytes);
        write(&directory.join("mesh.json"), &mesh_bytes);
        let plan = replay_plan(original, &REFERENCE_LINEAR_SOLVER);
        assert_eq!(plan.to_bytes().unwrap(), plan_bytes);
        let spatial = plan.as_spatial().unwrap();
        assert_eq!(spatial.admission.linear.workers, NonZeroUsize::MIN);
        let (field, ty) = spatial.fields().next().unwrap();
        assert_eq!(spatial.fields().count(), 1);
        assert_eq!(ty.scalar_domain(), eqiora_core::ScalarDomain::Real);
        assert_eq!(
            ty.dimension(),
            DimExponents::from_integers([0, 0, 0, 0, 1, 0, 0]).unwrap()
        );
        assert_eq!(
            spatial.field_space(field).unwrap(),
            Space::continuous_lagrange(std::num::NonZeroU16::MIN)
        );
        let NativeMeshResources::Cartesian { mesh, .. } = spatial.admission.resources() else {
            panic!("Cartesian resource");
        };
        let mesh = mesh.mesh();
        assert_eq!(mesh.entity_count(0), Some(9));
        assert_eq!(mesh.entity_count(2), Some(4));
        for (axis, length) in inputs[..2].iter().enumerate() {
            let length = f64::from(*length);
            assert_eq!(
                mesh.axis_coordinates(axis).unwrap(),
                [0., length / 2., length]
            );
        }
        // Capture the exact finalized occurrence returned by the ordinary public Run.
        let guard = crate::numerical_admission::scalar::live_tests::capture::CaptureGuard::start();
        let result = spatial.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
        let result_bytes = result.to_bytes().unwrap();
        write(&directory.join("result.json"), &result_bytes);
        let capture = guard.finish(); // Disarm before independent artifact/reaction replay.
        assert_eq!(capture.plan_identity, spatial.identity());
        assert_eq!(capture.plan_identity, result.plan().identity());
        assert_eq!(capture.quadrature.len(), 1);
        assert_eq!(capture.dual_scales.len(), 1);
        assert_eq!(capture.mapping.full_count(), 9);
        assert_eq!(capture.mapping.free_count(), 6);
        assert_eq!(capture.mapping.field_scale(field.erase()).unwrap(), 1.);
        for rule in &capture.quadrature {
            assert!(rule.polynomial_exactness().unwrap() >= 3);
        }
        for (owner, dual) in &capture.dual_scales {
            assert_eq!(*owner, field.erase());
            assert_eq!(dual.value(), 1.);
            assert_eq!(
                dual.dim(),
                DimExponents::from_integers([1, 1, -3, 0, 0, 0, 0]).unwrap()
            );
        }
        let mut keys = capture.mapping.keys().collect::<Vec<_>>();
        assert_eq!(keys.len(), 9);
        keys.sort_by(|a, b| {
            let x = mesh.vertex_coordinates(a.entity).unwrap();
            let y = mesh.vertex_coordinates(b.entity).unwrap();
            x[0].total_cmp(&y[0]).then(x[1].total_cmp(&y[1]))
        });
        let coordinates = keys
            .iter()
            .map(|key| {
                assert_eq!(key.field, field.erase());
                assert_eq!(key.slot, 0);
                assert_eq!(key.component, 0);
                assert_eq!(key.entity.dimension(), 0);
                mesh.vertex_coordinates(key.entity).unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            keys.iter()
                .map(|key| capture.mapping.global_dof(*key).unwrap())
                .collect::<BTreeSet<_>>()
                .len(),
            9
        );
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(
                capture.mapping.free_dof(*key).is_none(),
                coordinates[i][0] == 0.
            );
        }
        let full = &capture.full;
        assert_eq!(full.rhs().len(), 9);
        assert_eq!(full.matrix().rows(), 9);
        assert_eq!(full.matrix().columns(), 9);
        let matrix = keys
            .iter()
            .map(|a| {
                keys.iter()
                    .map(|b| {
                        binary64_hex(
                            full.matrix()
                                .entry(
                                    capture.mapping.global_dof(*a).unwrap(),
                                    capture.mapping.global_dof(*b).unwrap(),
                                )
                                .unwrap(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let load = keys
            .iter()
            .map(|key| binary64_hex(full.rhs()[capture.mapping.global_dof(*key).unwrap()]))
            .collect::<Vec<_>>();
        let captured_operator_sha = write(
            &directory.join("captured-operator.json"),
            &json_bytes(&serde_json::json!({
                "schema":1,"case":name,"captured_live_plan_identity":capture.plan_identity,
                "field_id":field.ulid().to_string(),"result_identity":result.identity(),
                "result_sha256":digest(&result_bytes),
                "row_dual_scales":capture.dual_scales.iter().map(|(owner,scale)|serde_json::json!({
                    "field_id":owner.ulid().to_string(),"value_binary64_hex":binary64_hex(scale.value()),
                    "dimension":scale.dim().exponents(),
                })).collect::<Vec<_>>(),
                "column_scale":{"field_id":field.ulid().to_string(),
                    "value_binary64_hex":binary64_hex(capture.mapping.field_scale(field.erase()).unwrap()),
                    "dimension":ty.dimension().exponents()},
                "order":coordinates.iter().map(|p|p.iter().map(ToString::to_string).collect::<Vec<_>>()).collect::<Vec<_>>(),
                "global_dofs":keys.iter().map(|k|capture.mapping.global_dof(*k).unwrap()).collect::<Vec<_>>(),
                "A_binary64_hex":&matrix,"b_binary64_hex":&load,
                "units":{"coordinates":"m","A":"W/(m*K)","b":"W/m"},"physical_scales":"identity-checked",
            })),
        );
        // The ordinary Result above owns the same captured live occurrence.
        assert_eq!(
            crate::CommonResult::from_bytes(&result_bytes, &plan)
                .unwrap()
                .to_bytes()
                .unwrap(),
            result_bytes
        );
        assert_eq!(result.field_count(), 1);
        assert_eq!(result.field(0).unwrap().0, field.ulid().to_string());
        assert_eq!(result.field_block_count(0), 1);
        let (association, values, shape) = result.field_block(0, 0).unwrap();
        assert_eq!(association, "vertex");
        // Q1 Cartesian logical shape is one nodal extent per axis.
        assert_eq!(shape, [3, 3]);
        assert_eq!(values.len(), 9);
        let entities = spatial.field_coefficient_entities(field).unwrap();
        assert_eq!(entities.len(), 9);
        let by_entity = entities
            .iter()
            .copied()
            .zip(values.iter().copied())
            .collect::<BTreeMap<_, _>>();
        let records = keys
            .iter()
            .zip(&coordinates)
            .map(|(key, point)| {
                serde_json::json!({
                    "coordinates_m":point.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    "value_binary64_hex":binary64_hex(by_entity[&key.entity]),
                })
            })
            .collect::<Vec<_>>();
        let recovered = spatial.linear_reactions(values).unwrap();
        let reaction_scale = recovered.physical_scales[&field.erase()];
        assert_eq!(reaction_scale.value(), 1.);
        assert_eq!(
            reaction_scale.dim(),
            DimExponents::from_integers([1, 1, -3, 0, 0, 0, 0]).unwrap()
        );
        let mut reactions = recovered
            .constrained_actions
            .iter()
            .map(|(key, value)| {
                assert_eq!(key.field, field.erase());
                assert_eq!(key.slot, 0);
                assert_eq!(key.component, 0);
                let point = mesh.vertex_coordinates(key.entity).unwrap();
                assert_eq!(point[0], 0.);
                (point, recovered.physical_value(*key, *value).unwrap())
            })
            .collect::<Vec<_>>();
        reactions.sort_by(|a, b| a.0[1].total_cmp(&b.0[1]));
        assert_eq!(reactions.len(), 3);
        let reaction_records = reactions
            .iter()
            .map(|(point, value)| {
                serde_json::json!({
                    "coordinates_m":point.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    "value_binary64_hex":binary64_hex(*value),
                })
            })
            .collect::<Vec<_>>();
        let payload = serde_json::json!({"schema":1,"case":name,
            "order":coordinates.iter().map(|p|p.iter().map(ToString::to_string).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "units":{"coordinates":"m","A":"W/(m*K)","b":"W/m","coefficients":"K","left_reactions":"W/m","integrated_heat":"W/m","pointwise_heat_flux":"W/m^2"},
            "A_binary64_hex":matrix,"b_binary64_hex":load,"coefficients":records,
            "left_reactions":reaction_records,"heat_convention":"physical_outward_heat=-weak_reaction"});
        let payload_sha = write(&directory.join("robin-input.json"), &json_bytes(&payload));
        let receipt = serde_json::json!({"schema":1,"status":"exported-not-yet-compared","ordinary_solves":1,
            "native_executable_sha256":executable_sha,"source_head":source_head,"producer_source_sha256":digest(include_bytes!("native.rs")),
            "input_sha256":payload_sha,"captured_operator_sha256":captured_operator_sha,"model_sha256":digest(&model_bytes),"plan_sha256":digest(&plan_bytes),
            "mesh_sha256":digest(&mesh_bytes),"result_sha256":digest(&result_bytes),"plan_identity":plan.identity(),
            "model_digest":spatial.model_digest(),"mesh_digest":spatial.mesh_digest(),"result_identity":result.identity(),
            "field_id":field.ulid().to_string(),"coefficient_entities":entities.iter().map(|e|[e.dimension(),e.index()]).collect::<Vec<_>>(),
            "operator_global_dofs":keys.iter().map(|k|capture.mapping.global_dof(*k).unwrap()).collect::<Vec<_>>(),
            "effective_quadrature":spatial.admission.assembly_quadrature(),"physical_scales":"identity-checked",
            "captured_live_plan_identity":capture.plan_identity,"ordinary_public_runs":1,
            "physical_outward_left_heat_binary64_hex":binary64_hex(-reactions.iter().map(|(_,v)|v).sum::<f64>())});
        write(
            &directory.join("producer-receipt.json"),
            &json_bytes(&receipt),
        );
        oracle.compare(name, &directory, &payload_sha);
    }
    println!(
        "robin-export: {} (all four independent scientific comparisons accepted)",
        root.display()
    );
}
