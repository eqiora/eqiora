//! Native ordinary producer; scientific acceptance belongs to polynomial.py.
use super::*;
use crate::numerical_admission::tests::artifact_capture::{
    binary64_hex, digest, json_bytes, write,
};
use eqiora_meshing::MeshTopology;
use std::path::PathBuf;

fn source(dimension: usize) -> String {
    let lengths = [2, 3, 4];
    let names = [
        "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper",
    ];
    let factor = |i| format!("(1+(coordinate({i})/{}[m])^2)", lengths[i]);
    let product_except = |excluded| {
        (0..dimension)
            .filter(|&i| i != excluded)
            .map(factor)
            .collect::<Vec<_>>()
            .join("*")
    };
    let exact = (0..dimension).map(factor).collect::<Vec<_>>().join("*");
    let forcing = (0..dimension)
        .map(|i| format!("2/{}[m^2]*{}", lengths[i] * lengths[i], product_except(i)))
        .collect::<Vec<_>>()
        .join("+");
    let supports = names[..2 * dimension]
        .iter()
        .map(|name| format!("support {name}: boundary(parent=body)"))
        .collect::<Vec<_>>()
        .join(",");
    let boundaries = names[..2 * dimension]
        .iter()
        .enumerate()
        .map(|(ordinal, name)| {
            if *name == "x_upper" {
                format!("relation essential on {name} {{ trace(u)={exact}; }}")
            } else if ordinal % 2 == 0 {
                // The exact derivative vanishes at a lower Cartesian face.
                format!("relation flux_{name} on {name} {{ normal(grad(u))=0[1/m]; }}")
            } else {
                let axis = ordinal / 2;
                // The upper outward normal is +e_i, and x_i=L_i.
                format!(
                    "relation flux_{name} on {name} {{ normal(grad(u))=2/{}[m]*{}; }}",
                    lengths[axis],
                    product_except(axis)
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "public component QuadraticPatch(support body: volume(ambient_dimension={dimension}),{supports}) {{\nvariable u: 1 on body in smooth;\nrelation balance on body {{ -div(grad(u))=-({forcing}); }}\n{boundaries}\n}}\n"
    )
}

fn geometry(dimension: usize) -> CanonicalGeometryV1 {
    let graph = GeometryGraph::new();
    let operation = if dimension == 2 {
        graph.rectangle([0., 2.], [0., 3.]).unwrap()
    } else {
        graph.cuboid([0., 2.], [0., 3.], [0., 4.]).unwrap()
    };
    let names = [
        "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper",
    ];
    let mut topology = BTreeMap::from([("body".to_owned(), vec![operation.region().into()])]);
    for (name, boundary) in names.iter().zip(operation.boundaries()) {
        topology.insert((*name).to_owned(), vec![boundary.into()]);
    }
    graph.build(&operation, &topology).unwrap()
}

fn coordinates(mesh: &CartesianMesh, entity: eqiora_meshing::MeshEntity) -> Vec<f64> {
    let vertices = mesh.entity_vertices(entity).unwrap();
    (0..mesh.topological_dimension())
        .map(|axis| {
            vertices
                .iter()
                .map(|vertex| mesh.vertex_coordinates(*vertex).unwrap()[axis])
                .sum::<f64>()
                / vertices.len() as f64
        })
        .collect()
}

fn output_root() -> PathBuf {
    let root = std::env::var_os("EQIORA_Q2_POLYNOMIAL_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = PathBuf::from(std::env::var_os("HOME").expect("home-backed test scratch"));
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            home.join(".cache/eqiora")
                .join(format!("q2-polynomial-{}-{nonce}", std::process::id()))
        });
    assert!(!root.exists(), "retain previous native polynomial attempts");
    std::fs::create_dir_all(root.parent().unwrap()).unwrap();
    std::fs::create_dir(&root).unwrap();
    root
}

fn compile_and_resolve(
    dimension: usize,
    authored: &str,
    backend: &dyn LinearSolverBackend,
) -> (ModelEnvelope, ResolvedCommonPlan) {
    let geometry = geometry(dimension);
    let names = [
        "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper",
    ];
    let body = geometry.entity_set("body").unwrap();
    let mut supports = vec![("body", body, None)];
    supports.extend(names[..2 * dimension].iter().map(|name| {
        (
            *name,
            geometry.entity_set(name).unwrap(),
            Some(("body", body)),
        )
    }));
    let model = compile_model(
        "quadratic-patch.eqi",
        authored,
        &geometry,
        "QuadraticPatch",
        &supports,
        &[],
    );
    let owner = cartesian_box_resources(&geometry, &vec![2; dimension]);
    let gauss = eqiora_realization::QuadraturePolicy::GaussLegendre {
        points_per_axis: NonZeroUsize::new(4).unwrap(),
    };
    let original = ResolvedCommonPlan::resolve(
        &model,
        owner,
        CommonSpatialPolicy::Q2,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-14,
            1e-14,
            NonZeroUsize::new(100000).unwrap(),
        )),
        None,
        None,
        backend,
        None,
        None,
        Some(gauss),
    )
    .unwrap();
    (model, original)
}

#[test]
fn polynomial_source_compiles_and_resolves_both_dimensions_without_solves() {
    for dimension in [2, 3] {
        let (_, plan) = compile_and_resolve(dimension, &source(dimension), &ResolveOnlyBackend);
        let reopened = replay_plan(plan, &ResolveOnlyBackend);
        let spatial = reopened.as_spatial().unwrap();
        assert_eq!(spatial.fields().len(), 1);
        let field = spatial.fields().next().unwrap().0;
        assert_eq!(
            spatial.field_coefficient_entities(field).unwrap().len(),
            5usize.pow(dimension as u32)
        );
        assert_eq!(
            spatial.admission.assembly_quadrature(),
            Some(eqiora_realization::QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(4).unwrap()
            })
        );
    }
}

#[test]
fn ordinary_quadratic_polynomial_producer_retains_all_nodes_and_physical_reactions() {
    let root = output_root();
    println!(
        "q2-polynomial-launch: {} (two ordinary solves declared; comparisons pending)",
        root.display()
    );
    let executable = std::env::current_exe().unwrap();
    let executable_sha256 = digest(&std::fs::read(&executable).unwrap());
    for dimension in [2, 3] {
        let directory = root.join(format!("d{dimension}"));
        std::fs::create_dir(&directory).unwrap();
        let authored = source(dimension);
        write(&directory.join("model.eqi"), authored.as_bytes());
        let (model, original) = compile_and_resolve(dimension, &authored, &REFERENCE_LINEAR_SOLVER);
        let gauss = eqiora_realization::QuadraturePolicy::GaussLegendre {
            points_per_axis: NonZeroUsize::new(4).unwrap(),
        };
        let plan_bytes = original.to_bytes().unwrap();
        let plan = replay_plan(original, &REFERENCE_LINEAR_SOLVER);
        assert_eq!(plan.to_bytes().unwrap(), plan_bytes);
        let spatial = plan.as_spatial().unwrap();
        assert_eq!(spatial.admission.assembly_quadrature(), Some(gauss));
        assert!(
            spatial
                .portable_realization()
                .domains()
                .iter()
                .all(|domain| domain.discretization().quadrature() == gauss)
        );
        let fields = spatial.fields().collect::<Vec<_>>();
        assert_eq!(fields.len(), 1);
        let (field, ty) = fields[0];
        assert_eq!(ty.scalar_domain(), eqiora_core::ScalarDomain::Real);
        assert_eq!(ty.dimension(), DimExponents::DIMENSIONLESS);
        assert_eq!(ty.shape(), &eqiora_core::ValueShape::scalar());
        let space = Space::continuous_lagrange(std::num::NonZeroU16::new(2).unwrap());
        assert_eq!(spatial.field_space(field).unwrap(), space);
        let NativeMeshResources::Cartesian { mesh, .. } = spatial.admission.resources() else {
            panic!("actual Cartesian resource required")
        };
        let mesh = mesh.mesh();
        assert_eq!(mesh.topological_dimension(), dimension);
        assert_eq!(mesh.entity_count(0), Some(3usize.pow(dimension as u32)));
        assert_eq!(
            mesh.entity_count(dimension),
            Some(2usize.pow(dimension as u32))
        );
        for axis in 0..dimension {
            let length = [2., 3., 4.][axis];
            assert_eq!(
                mesh.axis_coordinates(axis).unwrap(),
                [0., length / 2., length]
            );
        }
        assert_eq!(spatial.model_digest(), model.digest().unwrap().to_string());
        assert_eq!(spatial.admission.linear.workers, NonZeroUsize::MIN);
        let entities = spatial.field_coefficient_entities(field).unwrap();
        assert_eq!(entities.len(), 5usize.pow(dimension as u32));
        assert!(entities.windows(2).all(|pair| pair[0] < pair[1]));
        for stratum in 0..=dimension {
            assert_eq!(
                entities
                    .iter()
                    .filter(|entity| entity.dimension() == stratum)
                    .count(),
                mesh.entity_count(stratum).unwrap()
            );
        }
        let mesh_bytes = spatial.admission.authenticated_mesh().to_bytes().unwrap();
        let model_bytes = model.canonical_json().unwrap();
        write(&directory.join("model.json"), &model_bytes);
        write(&directory.join("plan.json"), &plan_bytes);
        write(&directory.join("mesh.json"), &mesh_bytes);
        // Exactly one ordinary solve per dimension. Replay/reactions only reassemble.
        let result = spatial.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
        let result_bytes = result.to_bytes().unwrap();
        write(&directory.join("result.json"), &result_bytes);
        let reopened = crate::CommonResult::from_bytes(&result_bytes, &plan).unwrap();
        assert_eq!(reopened.to_bytes().unwrap(), result_bytes);
        assert_eq!(result.field_count(), 1);
        let metadata = result.field(0).unwrap();
        assert_eq!(metadata.0, field.ulid().to_string());
        assert_eq!(metadata.1, ty);
        assert_eq!(metadata.2, space);
        assert_eq!(result.field_block_count(0), dimension + 1);
        let mut values = Vec::new();
        let mut associations = Vec::new();
        for stratum in 0..=dimension {
            let (association, block, shape) = result.field_block(0, stratum).unwrap();
            let expected_association = if stratum == dimension {
                "cell"
            } else {
                ["vertex", "edge", "face"][stratum]
            };
            assert_eq!(association, expected_association);
            let count = entities
                .iter()
                .filter(|entity| entity.dimension() == stratum)
                .count();
            assert_eq!(shape, [count]);
            assert_eq!(block.len(), count);
            assert_eq!(reopened.field_block(0, stratum).unwrap().1, block);
            values.extend_from_slice(block);
            associations.push(association);
        }
        assert_eq!(values.len(), entities.len());
        let records = entities.iter().zip(&values).map(|(entity, value)| serde_json::json!({
            "coordinates_m": coordinates(mesh, *entity).iter().map(ToString::to_string).collect::<Vec<_>>(),
            "value_binary64_hex": binary64_hex(*value)
        })).collect::<Vec<_>>();
        // Honest content rehashing must not hide a changed nonvertex coefficient.
        for stratum in 1..=dimension {
            let mut altered = result.field_block(0, stratum).unwrap().1.to_vec();
            altered[0] += 1.;
            let resealed =
                super::super::result_replay::reseal_block_values(&result, 0, stratum, &altered);
            assert!(
                crate::CommonResult::from_bytes(&resealed, &plan).is_err(),
                "tampered nonvertex block {stratum}"
            );
        }
        let cell = dimension;
        let truncated = &result.field_block(0, cell).unwrap().1[1..];
        let missing_cell =
            super::super::result_replay::reseal_block_values(&result, 0, cell, truncated);
        assert!(crate::CommonResult::from_bytes(&missing_cell, &plan).is_err());
        let upper_edge = entities
            .iter()
            .filter(|entity| entity.dimension() == 1)
            .position(|entity| coordinates(mesh, *entity)[0] == 2.)
            .unwrap();
        let mut altered_boundary = result.field_block(0, 1).unwrap().1.to_vec();
        altered_boundary[upper_edge] += 1.;
        let resealed =
            super::super::result_replay::reseal_block_values(&result, 0, 1, &altered_boundary);
        assert!(
            crate::CommonResult::from_bytes(&resealed, &plan).is_err(),
            "boundary midpoint must not be skipped on reopening"
        );
        let recovered = spatial.linear_reactions(&values).unwrap();
        let mut action_units = [0; 7];
        action_units[1] = dimension as i32 - 2;
        assert_eq!(
            recovered.physical_scales[&field.erase()].dim(),
            DimExponents::from_integers(action_units).unwrap()
        );
        let upper = recovered.constrained_actions.iter().map(|(key, value)| {
            assert_eq!(key.field, field.erase()); assert_eq!(key.slot, 0); assert_eq!(key.component, 0);
            let point = coordinates(mesh, key.entity);
            assert_eq!(point[0], 2.);
            serde_json::json!({"coordinates_m": point.iter().map(ToString::to_string).collect::<Vec<_>>(), "value_binary64_hex": binary64_hex(recovered.physical_value(*key, *value).unwrap())})
        }).collect::<Vec<_>>();
        assert_eq!(upper.len(), 5usize.pow((dimension - 1) as u32));
        let upper_entities = recovered
            .constrained_actions
            .keys()
            .map(|key| {
                [
                    key.entity.dimension(),
                    key.entity.index(),
                    key.slot,
                    key.component,
                ]
            })
            .collect::<Vec<_>>();
        let input = serde_json::json!({"schema":1,"lengths_m":&[2,3,4][..dimension],"N":2,"gauss_points":4,"records":records,"upper_reactions":upper});
        let input_sha = write(
            &directory.join("polynomial-input.json"),
            &json_bytes(&input),
        );
        let receipt = serde_json::json!({"schema":1,"status":"exported-not-yet-compared","polynomial_input_sha256":input_sha,
            "plan_sha256":digest(&plan_bytes),"plan_identity":plan.identity(),"model_sha256":digest(&model_bytes),"model_digest":spatial.model_digest(),
            "mesh_sha256":digest(&mesh_bytes),"mesh_digest":spatial.mesh_digest(),"result_sha256":digest(&result_bytes),"result_identity":result.identity(),
            "authored_source_sha256":digest(authored.as_bytes()),"producer_source_sha256":digest(include_bytes!("native_patch.rs")),
            "field_id":field.ulid().to_string(),"coefficient_associations":associations,"coefficient_entities":entities.iter().map(|entity|[entity.dimension(),entity.index()]).collect::<Vec<_>>(),
            "effective_quadrature":gauss,"upper_reaction_entities":upper_entities,"physical_reaction_scale_binary64_hex":binary64_hex(recovered.physical_scales[&field.erase()].value()),"physical_reaction_units_exponents":action_units,"native_executable_sha256":executable_sha256,"ordinary_solves":1,"plan_reopens":1,"result_reopens":1,"reaction_reconstructions":1,"rejected_tampered_reopens":dimension+2});
        write(
            &directory.join("producer-receipt.json"),
            &json_bytes(&receipt),
        );
    }
    println!(
        "q2-polynomial-export: {} (scientific comparator still required)",
        root.display()
    );
}
