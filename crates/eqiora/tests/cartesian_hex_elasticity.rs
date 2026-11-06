//! Ordinary 3D Hex preparation. Numerical qualification is separately reviewed.
use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use eqiora::api::ModelDocument;
use eqiora::artifact::{
    CartesianMeshCellsV2, CartesianMeshDecoderLimits, CartesianMeshEnvelopeV1,
    GeometryMeshCorrespondenceEnvelopeV1, MeshProductionLineageEnvelopeV1,
};
use eqiora::compiler::StaticBindingValue;
use eqiora::geometry::{GeometryGraph, GeometryTopologyHandle};
use eqiora::meshing::{CartesianMesh, MeshEntity, MeshTopology};

const SOURCE: &str = include_str!("../../../verify/numerics/cartesian-hex-elasticity/model.eqi");
const SIDES: [&str; 6] = [
    "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper",
];

#[test]
fn ordinary_hex_typed_source_and_retained_mesh_prepare_without_solving() {
    let graph = GeometryGraph::new();
    let cuboid = graph.cuboid([0., 2.], [0., 3.], [0., 4.]).unwrap();
    let mut names = BTreeMap::from([("body".to_owned(), vec![cuboid.region().into()])]);
    for (name, boundary) in SIDES.into_iter().zip(cuboid.boundaries()) {
        names.insert(
            name.to_owned(),
            vec![GeometryTopologyHandle::from(boundary)],
        );
    }
    let geometry = graph.build(&cuboid, &names).unwrap();
    let pressure = eqiora::DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).unwrap();
    let values = [
        (
            "mu",
            eqiora::ValueLiteral::new(
                eqiora::ValueType::scalar(eqiora::ScalarDomain::Real, pressure).unwrap(),
                [(3., 0.)],
            )
            .unwrap(),
        ),
        (
            "lambda",
            eqiora::ValueLiteral::new(
                eqiora::ValueType::scalar(eqiora::ScalarDomain::Real, pressure).unwrap(),
                [(2., 0.)],
            )
            .unwrap(),
        ),
    ];
    let mut bindings = vec![(
        "body",
        StaticBindingValue::GeometrySupport {
            geometry: &geometry,
            selection: geometry.entity_set("body").unwrap(),
            parent: None,
        },
    )];
    for side in SIDES {
        bindings.push((
            side,
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set(side).unwrap(),
                parent: Some(geometry.entity_set("body").unwrap()),
            },
        ));
    }
    bindings.extend(
        values
            .iter()
            .map(|(name, value)| (*name, StaticBindingValue::Value(value))),
    );
    let document = ModelDocument::compile_selected(
        "cartesian-hex-elasticity.eqi",
        SOURCE,
        "CartesianHexElasticity",
        &bindings,
    )
    .unwrap();
    assert!(
        document
            .authored_formulation_projection()
            .unwrap()
            .is_some()
    );
    let cells = CartesianMeshCellsV2::new([2; 3]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_cartesian_box_v1(&geometry, cells.cells())
            .unwrap();
    MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
        &cells,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
        &cells,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let owner = eqiora_numerics::AuthenticatedCommonMesh::structured_cartesian(
        geometry.clone(),
        mesh.clone(),
        correspondence,
        production,
    )
    .unwrap();
    let model = eqiora::artifact::ModelEnvelope::from_program(document.program()).unwrap();
    let solver = eqiora_numerics::CommonSolvePolicy::Linear(
        eqiora_numerics::CommonLinearRequest::exact(
            eqiora::solver::SolverPlan::new(
                eqiora::solver::LinearSolver::ConjugateGradient,
                1e-14,
                1e-14,
                NonZeroUsize::new(10000).unwrap(),
            )
            .unwrap()
            .with_preconditioner(eqiora::solver::PreconditionerPolicy::Identity)
            .with_reduction(eqiora::solver::ReductionPolicy::Reproducible),
            eqiora::solver::REFERENCE_SOLVER_PROVIDER,
        )
        .unwrap(),
    );
    let plan = eqiora_numerics::ResolvedCommonPlan::resolve(
        &model,
        owner,
        eqiora_numerics::CommonSpatialPolicy::Q1,
        solver,
        None,
        None,
        &eqiora::solver::REFERENCE_LINEAR_SOLVER,
        None,
        document.authored_formulation_projection().unwrap(),
        Some(eqiora_realization::QuadraturePolicy::GaussLegendre {
            points_per_axis: NonZeroUsize::new(2).unwrap(),
        }),
    )
    .unwrap();
    let description = plan.formulation().unwrap();
    assert_eq!(
        description.requested(),
        eqiora_numerics::FormulationSelectionMode::Authored
    );
    assert_eq!(
        description.requested_source_identity(),
        Some(
            document
                .authored_formulation_projection()
                .unwrap()
                .unwrap()
                .source_identity()
        )
    );
    let bytes = plan.to_bytes().unwrap();
    let replay = eqiora_numerics::ResolvedCommonPlan::from_bytes(
        &bytes,
        &eqiora::solver::REFERENCE_LINEAR_SOLVER,
        &[],
    )
    .unwrap();
    assert_eq!(replay.to_bytes().unwrap(), bytes);
    assert_physical_hex_corners(mesh.mesh());
}

fn assert_physical_hex_corners(native: &CartesianMesh) {
    assert_eq!(native.entity_count(0), Some(27));
    assert_eq!(native.entity_count(3), Some(8));
    for cell in 0..8 {
        let vertices = native.entity_vertices(MeshEntity::new(3, cell)).unwrap();
        assert_eq!(vertices.len(), 8);
        let corners: Vec<_> = vertices
            .into_iter()
            .map(|v| native.vertex_coordinates(v).unwrap())
            .collect();
        let lower: Vec<_> = (0..3)
            .map(|a| corners.iter().map(|p| p[a]).min_by(f64::total_cmp).unwrap())
            .collect();
        for (corner, point) in corners.iter().enumerate() {
            for axis in 0..3 {
                assert_eq!(
                    point[axis],
                    lower[axis] + [1., 1.5, 2.][axis] * ((corner >> axis) & 1) as f64
                );
            }
        }
    }
}

#[test]
fn native_hex_corner_order_binds_nonunit_physical_axes_without_solving() {
    let native = CartesianMesh::uniform(&[[0., 2.], [0., 3.], [0., 4.]], &[2; 3]).unwrap();
    assert_physical_hex_corners(&native);
}

#[test]
fn invalid_axes_and_real_decoder_budget_reject_before_any_result() {
    for axes in [
        vec![vec![0., 0.], vec![0., 3.], vec![0., 4.]],
        vec![vec![2., 0.], vec![0., 3.], vec![0., 4.]],
        vec![vec![0., f64::INFINITY], vec![0., 3.], vec![0., 4.]],
        vec![vec![0., f64::NAN], vec![0., 3.], vec![0., 4.]],
    ] {
        assert!(CartesianMesh::from_axes(axes).is_err());
    }
    let mesh = CartesianMesh::uniform(&[[0., 2.], [0., 3.], [0., 4.]], &[2; 3]).unwrap();
    let artifact = CartesianMeshEnvelopeV1::from_mesh(&mesh).unwrap();
    let limits = CartesianMeshDecoderLimits {
        max_cartesian_entities: 1,
        ..Default::default()
    };
    assert!(
        CartesianMeshEnvelopeV1::from_json_with_limits(&artifact.canonical_json().unwrap(), limits)
            .is_err()
    );
}
