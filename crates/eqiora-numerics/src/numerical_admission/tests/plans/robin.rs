//! Ordinary Robin admission and explicit independent-reference producer.
use super::*;
mod native;

// Inputs are the four fixed rational cases in the separately reviewed reference.
const CASES: [(&str, [u32; 6]); 4] = [
    ("insulated", [2, 1, 3, 0, 280, 320]),
    ("robin", [2, 1, 3, 2, 280, 320]),
    ("changed_h", [2, 1, 3, 5, 280, 320]),
    ("changed_geometry_material_cooling", [3, 2, 2, 4, 310, 280]),
];

fn source([_, _, k, h, tc, ambient]: [u32; 6]) -> String {
    format!(
        "public component Robin(
      support body:volume(ambient_dimension=2),
      support left:boundary(parent=body),support right:boundary(parent=body),
      support bottom:boundary(parent=body),support top:boundary(parent=body)
    ) {{
      parameter k:W/(m*K)={k}[W/(m*K)];
      parameter h:W/(m^2*K)={h}[W/(m^2*K)];
      parameter ambient:K={ambient}[K];
      variable temperature:K on body in smooth;
      law heat on body {{ flux -k*grad(temperature);source 0[W/m^3]; }}
      relation fixed on left {{trace(temperature)={tc}[K];}}
      relation transfer on right {{normal(k*grad(temperature))+h*trace(temperature)=h*ambient;}}
      relation insulated_bottom on bottom {{normal(k*grad(temperature))=0[W/m^2];}}
      relation insulated_top on top {{normal(k*grad(temperature))=0[W/m^2];}}
    }}"
    )
}

fn compile_and_resolve(
    inputs: [u32; 6],
    backend: &dyn LinearSolverBackend,
) -> (ModelEnvelope, ResolvedCommonPlan) {
    let graph = GeometryGraph::new();
    let rectangle = graph
        .rectangle([0., f64::from(inputs[0])], [0., f64::from(inputs[1])])
        .unwrap();
    let names = ["left", "right", "bottom", "top"];
    let mut topology = BTreeMap::from([("body".to_owned(), vec![rectangle.region().into()])]);
    for (name, handle) in names.iter().zip(rectangle.boundaries()) {
        topology.insert((*name).to_owned(), vec![handle.into()]);
    }
    let geometry = graph.build(&rectangle, &topology).unwrap();
    let body = geometry.entity_set("body").unwrap();
    let mut supports = vec![("body", body, None)];
    supports.extend(names.iter().map(|&name| {
        (
            name,
            geometry.entity_set(name).unwrap(),
            Some(("body", body)),
        )
    }));
    let model = compile_model(
        "robin.eqi",
        &source(inputs),
        &geometry,
        "Robin",
        &supports,
        &[],
    );
    let plan = ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[2, 2]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-14,
            1e-14,
            NonZeroUsize::new(10000).unwrap(),
        )),
        None,
        None,
        backend,
        None,
        None,
        Some(eqiora_realization::QuadraturePolicy::GaussLegendre {
            points_per_axis: NonZeroUsize::new(3).unwrap(),
        }),
    )
    .unwrap();
    (model, plan)
}

#[test]
fn four_robin_sources_resolve_and_replay_without_solves() {
    for (_, inputs) in CASES {
        let (_, plan) = compile_and_resolve(inputs, &ResolveOnlyBackend);
        let bytes = plan.to_bytes().unwrap();
        let replayed = replay_plan(plan, &ResolveOnlyBackend);
        assert_eq!(replayed.to_bytes().unwrap(), bytes);
        let spatial = replayed.as_spatial().unwrap();
        let (field, ty) = spatial.fields().next().unwrap();
        assert_eq!(
            ty.dimension(),
            DimExponents::from_integers([0, 0, 0, 0, 1, 0, 0]).unwrap()
        );
        assert_eq!(spatial.field_coefficient_entities(field).unwrap().len(), 9);
    }
}

#[test]
fn robin_remains_unsupported_by_conservative_tpfa() {
    let (model, plan) = compile_and_resolve(CASES[1].1, &ResolveOnlyBackend);
    let mesh = plan.authenticated_mesh().unwrap().clone();
    let outcome = ResolvedCommonPlan::resolve(
        &model,
        mesh,
        CommonSpatialPolicy::CellCenteredTpfa,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-14,
            1e-14,
            NonZeroUsize::new(10000).unwrap(),
        )),
        None,
        None,
        &ResolveOnlyBackend,
        None,
        None,
        None,
    );
    assert!(
        outcome.is_err(),
        "FEM Robin must not silently become a prescribed TPFA flux"
    );
}
