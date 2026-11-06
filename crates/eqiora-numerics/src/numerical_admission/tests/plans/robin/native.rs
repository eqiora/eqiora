//! Four ordinary solves export actual operators; scientific acceptance is external.
use super::*;
use crate::numerical_admission::tests::artifact_capture::{
    binary64_hex, digest, json_bytes, write,
};
use eqiora_meshing::MeshTopology;
use std::path::PathBuf;

fn output_root() -> PathBuf {
    let root = PathBuf::from(
        std::env::var_os("EQIORA_ROBIN_OUTPUT")
            .expect("explicit home-backed new Robin capture directory required"),
    );
    assert!(root.is_absolute() && root.starts_with(std::env::var_os("HOME").unwrap()));
    assert!(!root.exists(), "preserve earlier candidate attempts");
    std::fs::create_dir_all(root.parent().unwrap()).unwrap();
    std::fs::create_dir(&root).unwrap();
    root
}

#[test]
#[ignore = "explicit four-case scientific producer; source and protocol review required"]
fn ordinary_robin_producer_exports_full_operator_coefficients_and_reactions() {
    let root = output_root();
    let source_head =
        std::env::var("EQIORA_ROBIN_SOURCE_HEAD").expect("record the reviewed producer HEAD");
    assert_eq!(source_head.len(), 40);
    let executable_sha = digest(&std::fs::read(std::env::current_exe().unwrap()).unwrap());
    for (name, inputs) in CASES {
        let directory = root.join(name);
        std::fs::create_dir(&directory).unwrap();
        let (model, original) = compile_and_resolve(inputs, &REFERENCE_LINEAR_SOLVER);
        let plan_bytes = original.to_bytes().unwrap();
        let plan = replay_plan(original, &REFERENCE_LINEAR_SOLVER);
        assert_eq!(plan.to_bytes().unwrap(), plan_bytes);
        let spatial = plan.as_spatial().unwrap();
        assert_eq!(spatial.admission.linear.workers, NonZeroUsize::MIN);
        let (field, ty) = spatial.fields().next().unwrap();
        assert_eq!(spatial.fields().count(), 1);
        assert_eq!(ty.scalar_domain(), eqiora_core::ScalarDomain::Real);
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
        let RecognizedNativeModel::Spatial(equations) = spatial.admission.recognized_model() else {
            panic!("real scalar equations");
        };
        let system = equations
            .finalize_static(
                spatial.admission.linear.workers,
                spatial.linear().clone(),
                spatial.admission.resources(),
                &spatial.admission.discretizations,
                &spatial.admission.system_scaling,
                RegionAssemblyPolicy {
                    operator_properties: spatial.admission.operator_properties,
                    quadrature: spatial.admission.assembly_quadrature(),
                },
            )
            .unwrap();
        assert_eq!(system.mapping.full_count(), 9);
        assert_eq!(system.mapping.free_count(), 6);
        assert_eq!(system.mapping.field_scale(field.erase()).unwrap(), 1.);
        for (form, rule) in &system.assembly.input.forms {
            assert!(rule.polynomial_exactness().unwrap() >= 3);
            let dual = form.physical_dual_scale(field.erase()).unwrap();
            assert_eq!(dual.value(), 1.);
            assert_eq!(
                dual.dim(),
                DimExponents::from_integers([1, 1, -3, 0, 0, 0, 0]).unwrap()
            );
        }
        let mut keys = system.mapping.keys().collect::<Vec<_>>();
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
                .map(|key| system.mapping.global_dof(*key).unwrap())
                .collect::<BTreeSet<_>>()
                .len(),
            9
        );
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(
                system.mapping.free_dof(*key).is_none(),
                coordinates[i][0] == 0.
            );
        }
        let full = &system.assembly.full;
        assert_eq!(full.rhs().len(), 9);
        let matrix = keys
            .iter()
            .map(|a| {
                keys.iter()
                    .map(|b| {
                        binary64_hex(
                            full.matrix()
                                .entry(
                                    system.mapping.global_dof(*a).unwrap(),
                                    system.mapping.global_dof(*b).unwrap(),
                                )
                                .unwrap(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let load = keys
            .iter()
            .map(|key| binary64_hex(full.rhs()[system.mapping.global_dof(*key).unwrap()]))
            .collect::<Vec<_>>();
        // Exactly one ordinary solve per fixed case. Matrix/reaction capture reassembles only.
        let output = system
            .solve(
                LinearSolveRequest::<f64>::new(&REFERENCE_LINEAR_SOLVER, spatial.linear().clone()),
                None,
            )
            .unwrap();
        let result = crate::CommonResult::accept_linear(spatial.clone(), None, 0., output).unwrap();
        let result_bytes = result.to_bytes().unwrap();
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
        assert_eq!(shape, [9]);
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
        let source = source(inputs);
        let model_bytes = model.canonical_json().unwrap();
        let mesh_bytes = spatial.admission.authenticated_mesh().to_bytes().unwrap();
        write(&directory.join("model.eqi"), source.as_bytes());
        write(&directory.join("model.json"), &model_bytes);
        write(&directory.join("plan.json"), &plan_bytes);
        write(&directory.join("mesh.json"), &mesh_bytes);
        write(&directory.join("result.json"), &result_bytes);
        let payload_sha = write(&directory.join("robin-input.json"), &json_bytes(&payload));
        let receipt = serde_json::json!({"schema":1,"status":"exported-not-yet-compared","ordinary_solves":1,
            "native_executable_sha256":executable_sha,"source_head":source_head,"producer_source_sha256":digest(include_bytes!("native.rs")),
            "input_sha256":payload_sha,"model_sha256":digest(&model_bytes),"plan_sha256":digest(&plan_bytes),
            "mesh_sha256":digest(&mesh_bytes),"result_sha256":digest(&result_bytes),"plan_identity":plan.identity(),
            "model_digest":spatial.model_digest(),"mesh_digest":spatial.mesh_digest(),"result_identity":result.identity(),
            "field_id":field.ulid().to_string(),"coefficient_entities":entities.iter().map(|e|[e.dimension(),e.index()]).collect::<Vec<_>>(),
            "operator_global_dofs":keys.iter().map(|k|system.mapping.global_dof(*k).unwrap()).collect::<Vec<_>>(),
            "effective_quadrature":spatial.admission.assembly_quadrature(),"physical_scales":"identity-checked",
            "physical_outward_left_heat_binary64_hex":binary64_hex(-reactions.iter().map(|(_,v)|v).sum::<f64>())});
        write(
            &directory.join("producer-receipt.json"),
            &json_bytes(&receipt),
        );
    }
    println!(
        "robin-export: {} (independent scientific comparator still required)",
        root.display()
    );
}
